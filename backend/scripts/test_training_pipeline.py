"""
Test script for Phase A — Advanced Retraining Enhancements.

Runs the training pipeline modules directly on a synthetic dataset
(no database or MinIO required).

Usage:
    cd backend
    python scripts/test_training_pipeline.py
"""

import json
import sys
import os
import logging

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from app.training.feature_selection import select_features
from app.training.hyperparameter_search import run_hyperparameter_search
from app.training.threshold_optimizer import find_best_threshold

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger("test_pipeline")


def main():
    print("=" * 70)
    print("  Phase A — Advanced Retraining Pipeline Test")
    print("=" * 70)

    # ── 1. Generate synthetic dataset ──
    X_raw, y = make_classification(
        n_samples=500,
        n_features=30,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )
    feature_names = [f"feat_{i}" for i in range(X_raw.shape[1])]
    X = pd.DataFrame(X_raw, columns=feature_names)
    y = pd.Series(y, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nDataset: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # ── 2. Feature Selection ──
    fs_config = {"enabled": True, "method": "model_importance", "top_k": 15}
    print(f"\n{'─' * 50}")
    print("Step 1: Feature Selection")
    print(f"  Config: {fs_config}")

    X_train_sel, selected_features = select_features(X_train, y_train, fs_config)
    X_test_sel = X_test[selected_features]

    print(f"  Selected {len(selected_features)} features: {selected_features}")

    # ── 3. Hyperparameter Search ──
    hp_config = {"enabled": True, "trials": 10}
    print(f"\n{'─' * 50}")
    print("Step 2: Hyperparameter Search (Optuna)")
    print(f"  Config: {hp_config}")

    best_params = run_hyperparameter_search(
        "RandomForest", X_train_sel, y_train, hp_config
    )
    print(f"  Best params: {best_params}")

    # ── 4. Train final model with best params ──
    print(f"\n{'─' * 50}")
    print("Step 3: Training Final Model")

    clf = RandomForestClassifier(**best_params, random_state=42)
    clf.fit(X_train_sel, y_train)

    y_pred = clf.predict(X_test_sel)
    y_probs = clf.predict_proba(X_test_sel)

    acc = round(float(accuracy_score(y_test, y_pred)), 4)
    f1 = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
    print(f"  Accuracy: {acc}")
    print(f"  F1 Score: {f1}")

    # ── 5. Threshold Optimization ──
    print(f"\n{'─' * 50}")
    print("Step 4: Threshold Optimization")

    threshold_result = find_best_threshold(y_test, y_probs, metric="f1")
    print(f"  Best threshold: {threshold_result['best_threshold']}")
    print(f"  Best score ({threshold_result['metric']}): {threshold_result['best_score']}")

    # ── 6. Final metrics output ──
    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        "optimal_threshold": threshold_result["best_threshold"],
        "threshold_score": threshold_result["best_score"],
        "selected_features": selected_features,
        "best_params": best_params,
    }

    print(f"\n{'=' * 70}")
    print("  FINAL METRICS OUTPUT")
    print("=" * 70)
    print(json.dumps(metrics, indent=2, default=str))
    print()

    # ── 7. Test disabled configs (backward compat) ──
    print("─" * 50)
    print("Backward Compatibility Check (all disabled):")

    X_train_noop, feats_noop = select_features(X_train, y_train, {"enabled": False})
    assert X_train_noop.shape == X_train.shape, "No-op feature selection should return all features"
    print("  ✓ Feature selection no-op OK")

    params_noop = run_hyperparameter_search("RandomForest", X_train, y_train, {"enabled": False})
    assert "n_estimators" in params_noop, "Disabled search should return defaults"
    print("  ✓ Hyperparameter search no-op OK")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()
