from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineSettings:
    """Configuration for the H3 feature engineering pipeline.
    
    H3 Resolution scale:
    - 7: ~5.16 km² per cell (too coarse for city planning)
    - 8: ~0.73 km² per cell (good compromise)
    - 9: ~0.10 km² per cell (highly detailed)
    """

    h3_resolution: int = 9
    target_crs: str = "EPSG:2180"
    output_path: Path = Path("app/model/h3_features.parquet")
    weight_buildings_density: float = 0.30
    weight_bike_gap: float = 0.30
    weight_noise_exposure: float = 0.20
    weight_greenery_gap: float = 0.10
    weight_bike_infra_gap: float = 0.10
