"""Phase 4: hyperparameter optimization and stability comparison for unsupervised models."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hdbscan import HDBSCAN


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path("app/model/h3_features.parquet")
OUT_PATH = Path("app/model/phase4_optimization.json")

FEATURE_COLS = [
    "bike_racks_count",
    "bike_infra_points_count",
    "buildings_count",
    "noise_area_share",
    "greenery_area_share",
    "cycling_path_length_m",
    "street_length_m",
    "bike_path_coverage_ratio",
]


def _safe_metric(metric_fn, X: np.ndarray, labels: np.ndarray) -> float | None:
    mask = labels >= 0
    n_clusters = len(set(labels[mask])) if mask.any() else 0
    if n_clusters < 2:
        return None
    return float(metric_fn(X[mask], labels[mask]))


def _evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> dict:
    mask = labels >= 0
    n_clusters = len(set(labels[mask])) if mask.any() else 0
    return {
        "n_clusters": int(n_clusters),
        "n_noise_points": int((labels == -1).sum()),
        "silhouette": _safe_metric(silhouette_score, X, labels),
        "davies_bouldin": _safe_metric(davies_bouldin_score, X, labels),
        "calinski_harabasz": _safe_metric(calinski_harabasz_score, X, labels),
    }


def _stability_score(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    if len(labels_a) != len(labels_b):
        return 0.0
    # Map noise to a separate label bucket to include outlier consistency.
    la = np.where(labels_a < 0, -1, labels_a)
    lb = np.where(labels_b < 0, -1, labels_b)
    return float(adjusted_rand_score(la, lb))


def run() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing feature file: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)
    X = df[FEATURE_COLS].fillna(0.0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    results: list[dict] = []

    hdbscan_configs = [10, 15, 25, 40]
    for mcs in hdbscan_configs:
        model = HDBSCAN(min_cluster_size=mcs, core_dist_n_jobs=-1)
        labels = model.fit_predict(Xs)
        metrics = _evaluate_clustering(Xs, labels)
        results.append({
            "model": f"HDBSCAN_mcs{mcs}",
            "params": {"min_cluster_size": mcs},
            **metrics,
        })

    kmeans_configs = [4, 5, 6, 8, 10]
    kmeans_runs: dict[int, list[np.ndarray]] = {}
    for k in kmeans_configs:
        labels_runs = []
        silhouettes = []
        dbis = []
        chis = []

        for seed in [7, 21, 42]:
            model = KMeans(n_clusters=k, random_state=seed, n_init=10)
            labels = model.fit_predict(Xs)
            labels_runs.append(labels)
            silhouettes.append(float(silhouette_score(Xs, labels)))
            dbis.append(float(davies_bouldin_score(Xs, labels)))
            chis.append(float(calinski_harabasz_score(Xs, labels)))

        kmeans_runs[k] = labels_runs
        stability = np.mean([
            _stability_score(labels_runs[0], labels_runs[1]),
            _stability_score(labels_runs[0], labels_runs[2]),
            _stability_score(labels_runs[1], labels_runs[2]),
        ])

        results.append({
            "model": f"KMeans_k{k}",
            "params": {"k": k},
            "n_clusters": k,
            "n_noise_points": 0,
            "silhouette": float(np.mean(silhouettes)),
            "davies_bouldin": float(np.mean(dbis)),
            "calinski_harabasz": float(np.mean(chis)),
            "stability_ari": float(stability),
        })

    if_configs = [0.05, 0.1, 0.15]
    for contamination in if_configs:
        model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        labels = model.fit_predict(Xs)
        results.append({
            "model": f"IsolationForest_c{contamination}",
            "params": {"contamination": contamination},
            "n_anomalies": int((labels == -1).sum()),
            "anomaly_fraction": float((labels == -1).mean()),
        })

    ranking_df = pd.DataFrame(results)

    candidates = ranking_df[ranking_df["silhouette"].notna()].copy()
    if not candidates.empty:
        candidates["rank_silhouette"] = candidates["silhouette"].rank(ascending=False)
        candidates["rank_dbi"] = candidates["davies_bouldin"].rank(ascending=True)
        candidates["rank_chi"] = candidates["calinski_harabasz"].rank(ascending=False)
        if "stability_ari" in candidates.columns:
            candidates["rank_stability"] = candidates["stability_ari"].fillna(0).rank(ascending=False)
        else:
            candidates["rank_stability"] = 0

        candidates["composite_rank"] = (
            0.45 * candidates["rank_silhouette"]
            + 0.25 * candidates["rank_dbi"]
            + 0.20 * candidates["rank_chi"]
            + 0.10 * candidates["rank_stability"]
        )
        best_model = candidates.sort_values("composite_rank", ascending=True).iloc[0]["model"]
    else:
        best_model = None

    payload = {
        "n_samples": int(len(df)),
        "feature_columns": FEATURE_COLS,
        "best_model": best_model,
        "results": results,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Saved phase 4 optimization report: %s", OUT_PATH)
    if best_model:
        logger.info("Best model (composite rank): %s", best_model)


if __name__ == "__main__":
    run()
