"""Train and evaluate unsupervised models using preprocessed H3 features."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.unsupervised import EnsembleAnalyzer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATA_PATH = Path("app/model/h3_features.parquet")
RESULTS_PATH = Path("app/model/model_comparison.json")


def run():
    """Load features, train models, save results."""
    if not DATA_PATH.exists():
        logger.error(f"Feature data not found: {DATA_PATH}")
        logger.info("Run pipeline first: python app/pipeline/run_pipeline.py")
        return

    logger.info(f"Loading features from {DATA_PATH}")
    gdf = pd.read_parquet(DATA_PATH)
    
    feature_cols = [
        "bike_racks_count",
        "bike_infra_points_count",
        "buildings_count",
        "noise_area_share",
        "greenery_area_share",
        "cycling_path_length_m",
        "street_length_m",
        "bike_path_coverage_ratio",
    ]
    
    X = gdf[feature_cols].values
    logger.info(f"Features shape: {X.shape}")

    analyzer = EnsembleAnalyzer()
    results = analyzer.fit_predict(X)

    comparison_df = analyzer.compare_results(results)
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 80)
    logger.info("\n" + str(comparison_df.to_string()))
    logger.info("=" * 80)

    results_summary = {
        model_name: {
            "metrics": result.metrics,
            "model_type": result.model_name,
        }
        for model_name, result in results.items()
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results_summary, f, indent=2)
    logger.info(f"Saved model comparison to: {RESULTS_PATH}")

    for model_name, result in results.items():
        labels_path = RESULTS_PATH.parent / f"model_{model_name.lower().replace(' ', '_')}_labels.npy"
        import numpy as np
        np.save(labels_path, result.labels)
        logger.info(f"Saved {model_name} labels to: {labels_path}")


if __name__ == "__main__":
    run()
