"""Unsupervised learning models for cycling infrastructure demand."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Result of unsupervised model."""

    model_name: str
    labels: np.ndarray
    scores: np.ndarray | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    model_object: Any = None


class DemandModel:
    """Base unsupervised model for demand analysis."""

    def fit_predict(self, X: np.ndarray) -> ModelResult:
        raise NotImplementedError

    def _compute_clustering_metrics(
        self, X: np.ndarray, labels: np.ndarray, model_name: str
    ) -> dict[str, float]:
        """Compute standard clustering quality metrics (for labeled data)."""
        metrics: dict[str, float] = {}
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters < 2 or n_clusters >= len(X) - 1:
            logger.warning(f"{model_name}: Invalid number of clusters ({n_clusters})")
            return {"n_clusters": n_clusters, "note": "insufficient_clusters"}

        mask = labels >= 0
        if mask.sum() < 2:
            return {"n_clusters": n_clusters, "note": "insufficient_labeled_samples"}

        try:
            metrics["silhouette"] = float(silhouette_score(X[mask], labels[mask]))
        except Exception as e:
            logger.warning(f"{model_name} silhouette error: {e}")

        try:
            metrics["davies_bouldin"] = float(davies_bouldin_score(X[mask], labels[mask]))
        except Exception as e:
            logger.warning(f"{model_name} davies_bouldin error: {e}")

        try:
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(X[mask], labels[mask]))
        except Exception as e:
            logger.warning(f"{model_name} calinski_harabasz error: {e}")

        metrics["n_clusters"] = n_clusters
        metrics["n_noise_points"] = int((labels == -1).sum())

        return metrics


class HDBSCAN_Model(DemandModel):
    """Hierarchical DBSCAN for robust clustering and anomaly detection."""

    def __init__(self, min_cluster_size: int = 10):
        self.min_cluster_size = min_cluster_size

    def fit_predict(self, X: np.ndarray) -> ModelResult:
        model = HDBSCAN(min_cluster_size=self.min_cluster_size, core_dist_n_jobs=-1)
        labels = model.fit_predict(X)
        
        model_name = f"HDBSCAN_mcs{self.min_cluster_size}"
        metrics = self._compute_clustering_metrics(X, labels, model_name)
        scores = model.probabilities_  # Soft assignment scores

        return ModelResult(
            model_name=model_name,
            labels=labels,
            scores=scores,
            metrics=metrics,
            model_object=model,
        )


class KMeans_Model(DemandModel):
    """K-Means clustering for stable segmentation."""

    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters

    def fit_predict(self, X: np.ndarray) -> ModelResult:
        model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X)

        model_name = f"KMeans_k{self.n_clusters}"
        metrics = self._compute_clustering_metrics(X, labels, model_name)
        distances = model.transform(X)  # Soft scores: distance to centroid
        scores = 1.0 / (1.0 + distances.min(axis=1))  # Convert to [0, 1]

        return ModelResult(
            model_name=model_name,
            labels=labels,
            scores=scores,
            metrics=metrics,
            model_object=model,
        )


class IsolationForest_Model(DemandModel):
    """Isolation Forest for anomaly/outlier detection."""

    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination

    def fit_predict(self, X: np.ndarray) -> ModelResult:
        model = IsolationForest(contamination=self.contamination, random_state=42, n_jobs=-1)
        labels = model.fit_predict(X)  # -1 for anomaly, 1 for normal
        
        scores = model.score_samples(X)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        metrics = {
            "n_anomalies": int((labels == -1).sum()),
            "anomaly_fraction": float((labels == -1).mean()),
        }

        return ModelResult(
            model_name=f"IsolationForest_c{self.contamination}",
            labels=labels,
            scores=scores,
            metrics=metrics,
            model_object=model,
        )


class EnsembleAnalyzer:
    """Run multiple models and compare results."""

    def __init__(self):
        self.models = [
            HDBSCAN_Model(min_cluster_size=15),
            KMeans_Model(n_clusters=5),
            KMeans_Model(n_clusters=8),
            IsolationForest_Model(contamination=0.1),
        ]
        self.scaler = StandardScaler()

    def fit_predict(self, X: np.ndarray) -> dict[str, ModelResult]:
        """Fit all models and return results."""
        X_scaled = self.scaler.fit_transform(X)
        results = {}

        for model in self.models:
            try:
                result = model.fit_predict(X_scaled)
                results[result.model_name] = result
                logger.info(f"✓ {result.model_name}: metrics={result.metrics}")
            except Exception as e:
                logger.error(f"✗ {model.__class__.__name__} failed: {e}")

        return results

    def compare_results(self, results: dict[str, ModelResult]) -> pd.DataFrame:
        """Create comparison table of model metrics."""
        comparison_rows = []
        for model_name, result in results.items():
            row = {
                "model": model_name,
                **result.metrics,
            }
            comparison_rows.append(row)

        return pd.DataFrame(comparison_rows).fillna("N/A")
