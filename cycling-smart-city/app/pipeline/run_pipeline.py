from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import geopandas as gpd

from app.pipeline.config import PipelineSettings
from app.pipeline.h3_features import run_h3_feature_engineering
from app.pipeline.scoring import DemandWeights, add_interpretable_demand_score
from app.pipeline.spatial_prep import normalize_layers
from sourcing_data import LocalGeoData, OpenStreetMapData


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _safe_osm_data() -> dict[str, gpd.GeoDataFrame]:
    try:
        osm = OpenStreetMapData()
        return {
            "buildings": osm.buildings_df,
            "streets": osm.streets_df,
        }
    except Exception as exc:
        logger.warning("OSM fetch failed, continuing without OSM-derived layers: %s", exc)
        return {
            "buildings": gpd.GeoDataFrame(),
            "streets": gpd.GeoDataFrame(),
        }


def run(settings: PipelineSettings | None = None) -> Path:
    settings = settings or PipelineSettings()
    local = LocalGeoData()

    layers = {
        "bike_racks": local.bike_racks_df,
        "bike_infrastructure": local.bike_infrastructure_df,
        "cycling_paths": local.cycling_paths_df,
        "noise": local.noise_map_df,
        "greenery": local.greenery_df,
        "buildings": gpd.GeoDataFrame(),
        "streets": gpd.GeoDataFrame(),
    }

    normalized = normalize_layers(layers, target_crs=settings.target_crs)
    artifacts = run_h3_feature_engineering(normalized, resolution=settings.h3_resolution)
    features_with_score = add_interpretable_demand_score(
        artifacts.features,
        weights=DemandWeights(
            buildings_density=settings.weight_buildings_density,
            bike_gap=settings.weight_bike_gap,
            noise_exposure=settings.weight_noise_exposure,
            greenery_gap=settings.weight_greenery_gap,
            bike_infra_gap=settings.weight_bike_infra_gap,
        ),
    )

    output_path = settings.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = artifacts.hex_grid.merge(features_with_score, on="h3_id", how="left")
    result.to_parquet(output_path, index=False)
    logger.info("Saved H3 feature table: %s", output_path)
    return output_path


if __name__ == "__main__":
    run()
