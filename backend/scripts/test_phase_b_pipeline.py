"""
Test script for Phase B — Model Selection, Rebalancing, Ensemble.

Runs the training pipeline modules on a synthetic imbalanced dataset
(no database or MinIO required).

Usage:
    cd backend
    python scripts/test_phase_b_pipeline.py
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from app.training.feature_selection import select_features
from app.training.rebalancing import rebalance_data
from app.training.model_selector import train_candidate_models
from app.training.ensemble import build_ensemble, WeightedEnsemble
from app.training.threshold_optimizer import find_best_threshold

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger("test_phase_b")


def main():
    print("=" * 70)
    print("  Phase B — Model Selection, Rebalancing, Ensemble Test")
    print("=" * 70)

    # ── 1. Generate synthetic IMBALANCED dataset ──
    X_raw, y = make_classification(
        n_samples=600,
        n_features=25,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        weights=[0.85, 0.15],   # 85/15 imbalance
        random_state=42,
        flip_y=0.01,
    )
    feature_names = [f"feat_{i}" for i in range(X_raw.shape[1])]
    X = pd.DataFrame(X_raw, columns=feature_names)
    y = pd.Series(y, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    class_counts = y_train.value_counts().to_dict()
    print(f"\nDataset: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
    print(f"Class distribution (train): {class_counts}")

    # ── 2. Feature Selection (Phase A) ──
    fs_config = {"enabled": True, "method": "model_importance", "top_k": 15}
    print(f"\n{'─' * 50}")
    print("Step 1: Feature Selection")

    X_train_sel, selected_features = select_features(X_train, y_train, fs_config)
    X_test_sel = X_test[selected_features]
    print(f"  Selected {len(selected_features)} features")

    # ── 3. Rebalancing (Phase B — NEW) ──
    rebal_config = {"enabled": True, "imbalance_threshold": 0.4}
    print(f"\n{'─' * 50}")
    print("Step 2: Data Rebalancing")
    print(f"  Config: {rebal_config}")

    X_train_bal, y_train_bal, rebalance_meta = rebalance_data(
        X_train_sel, y_train, rebal_config
    )
    print(f"  Imbalance ratio: {rebalance_meta['imbalance_ratio']}")
    print(f"  Method used: {rebalance_meta['method_used']}")
    print(f"  Rebalanced: {rebalance_meta['rebalanced']}")
    if rebalance_meta.get("resampled_size"):
        print(f"  Samples: {rebalance_meta['original_size']} → {rebalance_meta['resampled_size']}")

    # ── 4. Automated Model Selection (Phase B — NEW) ──
    ms_config = {"enabled": True, "metric": "f1_score", "random_seed": 42}
    print(f"\n{'─' * 50}")
    print("Step 3: Automated Model Selection")
    print(f"  Config: {ms_config}")

    best_model, best_model_name, best_metrics, leaderboard = train_candidate_models(
        X_train_bal, y_train_bal, X_test_sel, y_test, ms_config
    )
    print(f"\n  Leaderboard:")
    for entry in leaderboard:
        name = entry.get("model", "?")
        score = entry.get("f1_score", "N/A")
        print(f"    {name:25s}  f1={score}")
    print(f"\n  ✅ Best model: {best_model_name}")
    print(f"     Metrics: {best_metrics}")

    # ── 5. Threshold Optimization (Phase A) ──
    print(f"\n{'─' * 50}")
    print("Step 4: Threshold Optimization")

    y_probs = best_model.predict_proba(X_test_sel)
    threshold_result = find_best_threshold(y_test, y_probs, metric="f1")
    print(f"  Best threshold: {threshold_result['best_threshold']}")
    print(f"  Best score (f1): {threshold_result['best_score']}")

    # ── 6. Ensemble (Phase B — NEW) ──
    print(f"\n{'─' * 50}")
    print("Step 5: Ensemble")

    # Train a simple 'old model' to test ensemble
    old_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=0)
    old_model.fit(X_train_bal, y_train_bal)

    ens_config = {"enabled": True, "weight_new": 0.7}
    ensemble_model, ensemble_meta = build_ensemble(old_model, best_model, ens_config)
    print(f"  Ensemble used: {ensemble_meta['ensemble_used']}")
    print(f"  Ensemble type: {type(ensemble_model).__name__}")

    # Evaluate ensemble
    y_ens_pred = ensemble_model.predict(X_test_sel)
    ens_acc = round(float(accuracy_score(y_test, y_ens_pred)), 4)
    ens_f1 = round(float(f1_score(y_test, y_ens_pred, average="weighted", zero_division=0)), 4)
    print(f"  Ensemble accuracy: {ens_acc}")
    print(f"  Ensemble f1_score: {ens_f1}")

    # ── 7. Final metrics output ──
    metrics = {
        "accuracy": best_metrics.get("accuracy"),
        "f1_score": best_metrics.get("f1_score"),
        "optimal_threshold": threshold_result["best_threshold"],
        "selected_features": selected_features,
        "selected_model": best_model_name,
        "leaderboard": leaderboard,
        "imbalance_ratio": rebalance_meta.get("imbalance_ratio"),
        "rebalance_method": rebalance_meta.get("method_used"),
        "ensemble_used": ensemble_meta.get("ensemble_used"),
    }

    print(f"\n{'=' * 70}")
    print("  FINAL METRICS OUTPUT")
    print("=" * 70)
    print(json.dumps(metrics, indent=2, default=str))
    print()

    # ── 8. Backward compatibility checks ──
    print("─" * 50)
    print("Backward Compatibility Checks:")

    # Rebalancing disabled
    X_noop, y_noop, meta_noop = rebalance_data(X_train, y_train, {"enabled": False})
    assert X_noop.shape == X_train.shape, "Disabled rebalancing should return original X"
    print("  ✓ Rebalancing disabled → no-op OK")

    # Model selection disabled — should still work with single model
    from app.training.hyperparameter_search import run_hyperparameter_search
    params = run_hyperparameter_search("RandomForest", X_train, y_train, {"enabled": False})
    assert "n_estimators" in params
    print("  ✓ Hyperparameter search disabled → defaults OK")

    # Ensemble disabled
    model_only, meta_only = build_ensemble(old_model, best_model, {"enabled": False})
    assert not isinstance(model_only, WeightedEnsemble)
    assert meta_only["ensemble_used"] is False
    print("  ✓ Ensemble disabled → returns new model only OK")

    # Verify all expected fields present
    for field in ("selected_model", "leaderboard", "imbalance_ratio", "ensemble_used"):
        assert field in metrics, f"Missing field: {field}"
    print("  ✓ All expected metrics fields present OK")

    print("\n✅ All Phase B tests passed!")


if __name__ == "__main__":
    main()
