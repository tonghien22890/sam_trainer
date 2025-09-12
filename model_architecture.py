from __future__ import annotations

"""Phase 4 simplified models: scikit-learn classifiers.

This module exposes a simple factory for DecisionTree/RandomForest classifiers.
"""

from typing import Literal

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


EstimatorType = Literal["decision_tree", "random_forest"]


def make_estimator(kind: EstimatorType = "random_forest", **kwargs):
    if kind == "decision_tree":
        return DecisionTreeClassifier(**kwargs)
    if kind == "random_forest":
        return RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, **kwargs)
    raise ValueError(f"Unknown estimator kind: {kind}")


