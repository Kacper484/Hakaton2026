from __future__ import annotations

from typing import Dict

import geopandas as gpd


def ensure_crs(gdf: gpd.GeoDataFrame, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Ensure a GeoDataFrame has CRS and return a copied frame."""
    if gdf.empty:
        return gdf.copy()

    frame = gdf.copy()
    if frame.crs is None:
        frame = frame.set_crs(crs)
    return frame


def to_crs_safe(gdf: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    """Transform frame to target CRS unless empty."""
    if gdf.empty:
        return gdf.copy()
    return ensure_crs(gdf).to_crs(crs)


def normalize_layers(
    layers: Dict[str, gpd.GeoDataFrame],
    target_crs: str,
) -> Dict[str, gpd.GeoDataFrame]:
    """Normalize geometry validity and CRS for all layers."""
    normalized: Dict[str, gpd.GeoDataFrame] = {}
    for name, gdf in layers.items():
        frame = to_crs_safe(gdf, target_crs)
        if frame.empty:
            normalized[name] = frame
            continue

        frame = frame[~frame.geometry.isna()].copy()
        frame = frame[frame.geometry.is_valid].copy()
        normalized[name] = frame
    return normalized
