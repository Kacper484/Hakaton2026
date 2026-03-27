from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely.geometry import Polygon


@dataclass(frozen=True)
class FeatureArtifacts:
    """Container for engineered H3 feature data."""

    hex_grid: gpd.GeoDataFrame
    features: pd.DataFrame


def _collect_bounds(layers: Iterable[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    non_empty = [g for g in layers if not g.empty]
    if not non_empty:
        return gpd.GeoDataFrame(geometry=[])

    combined_geometry = non_empty[0].geometry.unary_union
    for gdf in non_empty[1:]:
        combined_geometry = combined_geometry.union(gdf.geometry.unary_union)

    return gpd.GeoDataFrame(geometry=[combined_geometry], crs=non_empty[0].crs)


def _select_grid_source_layers(layers: Dict[str, gpd.GeoDataFrame]) -> list[gpd.GeoDataFrame]:
    """Prefer core city layers to avoid excessive grid extent from broad regional datasets."""
    preferred_order = [
        "cycling_paths",
        "bike_racks",
        "bike_infrastructure",
        "streets",
        "buildings",
    ]

    selected: list[gpd.GeoDataFrame] = []
    for key in preferred_order:
        gdf = layers.get(key)
        if gdf is not None and not gdf.empty:
            selected.append(gdf)

    if selected:
        return selected

    return [g for g in layers.values() if not g.empty]


def build_h3_grid(layers: Dict[str, gpd.GeoDataFrame], resolution: int) -> gpd.GeoDataFrame:
    """Build H3 hex grid that covers all non-empty input layers."""
    source_layers = _select_grid_source_layers(layers)
    boundary = _collect_bounds(source_layers)
    if boundary.empty:
        return gpd.GeoDataFrame(columns=["h3_id", "geometry"], geometry="geometry", crs="EPSG:4326")

    boundary_wgs84 = boundary.to_crs("EPSG:4326")
    geom = boundary_wgs84.geometry.iloc[0]

    if geom.is_empty:
        return gpd.GeoDataFrame(columns=["h3_id", "geometry"], geometry="geometry", crs="EPSG:4326")

    # Core transport layers can be line-based, so convert to a polygonal footprint for H3 fill.
    if geom.geom_type not in {"Polygon", "MultiPolygon"}:
        geom = geom.convex_hull
        if geom.geom_type != "Polygon":
            geom = geom.envelope

    cell_ids: set[str] = set()

    if geom.geom_type == "Polygon":
        cells = h3.geo_to_cells(geom.__geo_interface__, resolution)
        cell_ids.update(cells)
    elif geom.geom_type == "MultiPolygon":
        for polygon in geom.geoms:
            cells = h3.geo_to_cells(polygon.__geo_interface__, resolution)
            cell_ids.update(cells)

    cell_ids = sorted(cell_ids)

    polygons: List[Polygon] = []
    for cell_id in cell_ids:
        boundary_latlng = h3.cell_to_boundary(cell_id)
        polygons.append(Polygon([(lng, lat) for lat, lng in boundary_latlng]))

    return gpd.GeoDataFrame({"h3_id": cell_ids, "geometry": polygons}, geometry="geometry", crs="EPSG:4326")


def _count_points_per_hex(hex_grid: gpd.GeoDataFrame, points: gpd.GeoDataFrame, col_name: str) -> pd.DataFrame:
    """Count points within each hex using spatial index for performance."""
    if points.empty or hex_grid.empty:
        return pd.DataFrame({"h3_id": hex_grid.get("h3_id", pd.Series(dtype=str)), col_name: 0})

    joined = gpd.sjoin(
        points.to_crs(hex_grid.crs),
        hex_grid[["h3_id", "geometry"]],
        how="inner",
        predicate="within",
    )
    counts = joined.groupby("h3_id").size().rename(col_name)
    return counts.reindex(hex_grid["h3_id"], fill_value=0).reset_index()


def _line_length_per_hex(hex_grid: gpd.GeoDataFrame, lines: gpd.GeoDataFrame, col_name: str) -> pd.DataFrame:
    if lines.empty or hex_grid.empty:
        return pd.DataFrame({"h3_id": hex_grid.get("h3_id", pd.Series(dtype=str)), col_name: 0.0})

    lines_metric = lines.to_crs("EPSG:2180")
    lines_with_len = lines_metric[["geometry"]].copy()
    lines_with_len[col_name] = lines_with_len.geometry.length

    centroid_points = gpd.GeoDataFrame(
        lines_with_len[[col_name]].copy(),
        geometry=lines_with_len.geometry.centroid,
        crs=lines_metric.crs,
    ).to_crs(hex_grid.crs)

    joined = gpd.sjoin(
        centroid_points,
        hex_grid[["h3_id", "geometry"]],
        how="inner",
        predicate="within",
    )
    if joined.empty:
        return pd.DataFrame({"h3_id": hex_grid["h3_id"], col_name: 0.0})

    lengths = joined.groupby("h3_id")[col_name].sum()
    return lengths.reindex(hex_grid["h3_id"], fill_value=0.0).reset_index()


def _area_share_per_hex(hex_grid: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame, col_name: str) -> pd.DataFrame:
    if polygons.empty or hex_grid.empty:
        return pd.DataFrame({"h3_id": hex_grid.get("h3_id", pd.Series(dtype=str)), col_name: 0.0})

    poly_metric = polygons.to_crs("EPSG:2180")
    poly_with_area = poly_metric[["geometry"]].copy()
    poly_with_area["_area_m2"] = poly_with_area.geometry.area

    centroid_points = gpd.GeoDataFrame(
        poly_with_area[["_area_m2"]].copy(),
        geometry=poly_with_area.geometry.centroid,
        crs=poly_metric.crs,
    ).to_crs(hex_grid.crs)

    joined = gpd.sjoin(
        centroid_points,
        hex_grid[["h3_id", "geometry"]],
        how="inner",
        predicate="within",
    )
    if joined.empty:
        return pd.DataFrame({"h3_id": hex_grid["h3_id"], col_name: 0.0})

    covered = joined.groupby("h3_id")["_area_m2"].sum()

    hex_metric = hex_grid.to_crs("EPSG:2180")
    hex_area = hex_metric.set_index("h3_id").geometry.area
    share = (covered / hex_area).fillna(0.0).clip(0.0, 1.0).rename(col_name)
    return share.reindex(hex_grid["h3_id"], fill_value=0.0).reset_index()


def build_feature_table(hex_grid: gpd.GeoDataFrame, layers: Dict[str, gpd.GeoDataFrame]) -> pd.DataFrame:
    """Create core model features per H3 cell."""
    base = pd.DataFrame({"h3_id": hex_grid.get("h3_id", pd.Series(dtype=str))})
    if hex_grid.empty:
        return base

    bike_racks = _count_points_per_hex(hex_grid, layers.get("bike_racks", gpd.GeoDataFrame()), "bike_racks_count")
    bike_infra = _count_points_per_hex(hex_grid, layers.get("bike_infrastructure", gpd.GeoDataFrame()), "bike_infra_points_count")
    buildings = _count_points_per_hex(hex_grid, layers.get("buildings", gpd.GeoDataFrame()), "buildings_count")
    noise = _area_share_per_hex(hex_grid, layers.get("noise", gpd.GeoDataFrame()), "noise_area_share")
    greenery = _area_share_per_hex(hex_grid, layers.get("greenery", gpd.GeoDataFrame()), "greenery_area_share")
    cycle_len = _line_length_per_hex(hex_grid, layers.get("cycling_paths", gpd.GeoDataFrame()), "cycling_path_length_m")
    streets_len = _line_length_per_hex(hex_grid, layers.get("streets", gpd.GeoDataFrame()), "street_length_m")

    merged = base
    for part in [bike_racks, bike_infra, buildings, noise, greenery, cycle_len, streets_len]:
        merged = merged.merge(part, on="h3_id", how="left")

    merged = merged.fillna(0.0)
    merged["bike_path_coverage_ratio"] = np.where(
        merged["street_length_m"] > 0,
        merged["cycling_path_length_m"] / merged["street_length_m"],
        0.0,
    )
    merged["bike_path_coverage_ratio"] = merged["bike_path_coverage_ratio"].clip(0.0, 1.0)
    return merged


def run_h3_feature_engineering(layers: Dict[str, gpd.GeoDataFrame], resolution: int) -> FeatureArtifacts:
    """Build H3 grid and engineered feature table."""
    hex_grid = build_h3_grid(layers, resolution=resolution)
    feature_table = build_feature_table(hex_grid, layers)
    return FeatureArtifacts(hex_grid=hex_grid, features=feature_table)
