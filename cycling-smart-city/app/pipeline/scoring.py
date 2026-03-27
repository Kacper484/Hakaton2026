from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DemandWeights:
    """Weights for the baseline interpretable demand score."""

    buildings_density: float = 0.30
    bike_gap: float = 0.30
    noise_exposure: float = 0.20
    greenery_gap: float = 0.10
    bike_infra_gap: float = 0.10


def _min_max(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    min_value = float(series.min())
    max_value = float(series.max())
    if max_value <= min_value:
        return pd.Series(0.0, index=series.index)
    return (series - min_value) / (max_value - min_value)


def add_interpretable_demand_score(
    features: pd.DataFrame,
    weights: DemandWeights | None = None,
) -> pd.DataFrame:
    """Add transparent demand score based on normalized, human-readable components."""
    frame = features.copy()
    weights = weights or DemandWeights()

    required_columns = [
        "buildings_count",
        "noise_area_share",
        "greenery_area_share",
        "bike_path_coverage_ratio",
        "bike_infra_points_count",
    ]
    for column in required_columns:
        if column not in frame.columns:
            frame[column] = 0.0

    frame["n_buildings"] = _min_max(frame["buildings_count"])
    frame["n_noise"] = _min_max(frame["noise_area_share"])
    frame["n_greenery"] = _min_max(frame["greenery_area_share"])
    frame["n_bike_coverage"] = _min_max(frame["bike_path_coverage_ratio"])
    frame["n_bike_infra"] = _min_max(frame["bike_infra_points_count"])

    frame["component_buildings_density"] = frame["n_buildings"]
    frame["component_bike_gap"] = 1.0 - frame["n_bike_coverage"]
    frame["component_noise_exposure"] = frame["n_noise"]
    frame["component_greenery_gap"] = 1.0 - frame["n_greenery"]
    frame["component_bike_infra_gap"] = 1.0 - frame["n_bike_infra"]

    frame["demand_score"] = (
        weights.buildings_density * frame["component_buildings_density"]
        + weights.bike_gap * frame["component_bike_gap"]
        + weights.noise_exposure * frame["component_noise_exposure"]
        + weights.greenery_gap * frame["component_greenery_gap"]
        + weights.bike_infra_gap * frame["component_bike_infra_gap"]
    )

    frame["demand_priority"] = pd.cut(
        frame["demand_score"],
        bins=[-1.0, 0.33, 0.66, 1.0],
        labels=["low", "medium", "high"],
    )

    return frame.drop(columns=["n_buildings", "n_noise", "n_greenery", "n_bike_coverage", "n_bike_infra"])
