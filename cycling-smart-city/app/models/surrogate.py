"""Surrogate decision tree for model interpretability."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree

logger = logging.getLogger(__name__)


def train_surrogate_tree(X: np.ndarray, clustering_labels: np.ndarray, max_depth: int = 5) -> tuple[DecisionTreeClassifier, dict]:
    """Train a simple decision tree as a surrogate model for clustering results."""
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42, min_samples_leaf=5)
    
    mask = clustering_labels >= 0
    X_labeled = X[mask]
    y_labeled = clustering_labels[mask]

    tree.fit(X_labeled, y_labeled)

    score = tree.score(X_labeled, y_labeled)
    logger.info(f"Surrogate tree accuracy on labeled data: {score:.3f}")

    feature_importance = pd.Series(
        tree.feature_importances_,
        index=[
            "bike_racks", "bike_infra_points", "buildings", 
            "noise", "greenery", "cycling_paths", "streets", "bike_coverage"
        ]
    ).sort_values(ascending=False)

    logger.info("Feature Importance:")
    logger.info("\n" + str(feature_importance))

    return tree, {
        "accuracy": float(score),
        "feature_importance": feature_importance.to_dict(),
        "n_nodes": tree.tree_.node_count,
        "max_depth": tree.get_depth(),
    }


def get_decision_rules(tree: DecisionTreeClassifier, feature_names: list[str], class_names: list[str]) -> str:
    """Extract human-readable decision rules from tree."""
    import io
    
    buf = io.StringIO()
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        out_file=buf,
    )
    return buf.getvalue()
