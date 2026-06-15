# -*- coding: utf-8 -*-
"""
Algorithm Verification Script -- tests all supported algorithms end-to-end.

Tests:
  1. _resolve_model_type() alias resolution for every known alias
  2. _build_model() instantiation for every algorithm (classifier + regressor)
  3. Fit + predict on synthetic data for all classifiers
  4. Fit + predict on synthetic data for all regressors
  5. Model selector (automated multi-algorithm training)
  6. Hyperparameter search objective validation
"""

import sys
import time
import traceback
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# ── Insert project root into path ──
sys.path.insert(0, ".")

# ── Imports (isolated from DB layer) ──
# We import the pieces we need directly to avoid the database connection issue.

from app.training.model_selector import (
    _ALGORITHMS as SELECTOR_ALGORITHMS,
    _DEFAULT_HYPERPARAMS as SELECTOR_HYPERPARAMS,
    train_candidate_models,
    _evaluate,
)
from app.training.hyperparameter_search import (
    _OBJECTIVES as HP_OBJECTIVES,
    _DEFAULT_PARAMS as HP_DEFAULTS,
)

# ── Import training_runner components ──
# We need to work around the DB import chain. Import the module-level objects
# after the module loads.
import importlib
import app.services.training_runner as tr_module

_ALGORITHM_REGISTRY = tr_module._ALGORITHM_REGISTRY
_MODEL_CLASS_ALIASES = tr_module._MODEL_CLASS_ALIASES
_resolve_model_type = tr_module._resolve_model_type
_build_model = tr_module._build_model

# ══════════════════════════════════════════════════════════════════
# Test Utilities
# ══════════════════════════════════════════════════════════════════

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

results = {"pass": 0, "fail": 0, "warn": 0}


def log_result(test_name: str, passed: bool, detail: str = "", warn: bool = False):
    if warn:
        results["warn"] += 1
        print(f"  {WARN}  {test_name}: {detail}")
    elif passed:
        results["pass"] += 1
        print(f"  {PASS}  {test_name}  {detail}")
    else:
        results["fail"] += 1
        print(f"  {FAIL}  {test_name}  {detail}")


# ══════════════════════════════════════════════════════════════════
# Generate synthetic datasets
# ══════════════════════════════════════════════════════════════════

print("=" * 70)
print("ALGORITHM VERIFICATION SUITE")
print("=" * 70)

print("\n[*] Generating synthetic datasets...")
X_cls, y_cls = make_classification(
    n_samples=500, n_features=10, n_informative=6,
    n_classes=2, random_state=42
)
X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

X_reg, y_reg = make_regression(
    n_samples=500, n_features=10, n_informative=6, random_state=42
)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
print(f"   Classification: {X_cls_train.shape[0]} train, {X_cls_test.shape[0]} test, {X_cls.shape[1]} features")
print(f"   Regression:     {X_reg_train.shape[0]} train, {X_reg_test.shape[0]} test, {X_reg.shape[1]} features")

# ══════════════════════════════════════════════════════════════════
# Test 1: Algorithm Registry Completeness
# ══════════════════════════════════════════════════════════════════

print(f"\n{'─' * 70}")
print("TEST 1: Algorithm Registry Completeness")
print(f"{'─' * 70}")

print(f"\n  Registry contains {len(_ALGORITHM_REGISTRY)} algorithms:")
for key, entry in _ALGORITHM_REGISTRY.items():
    cls_name = entry["classifier"].__name__ if entry["classifier"] else "—"
    reg_name = entry["regressor"].__name__ if entry["regressor"] else "—"
    print(f"    {key:25s}  clf={cls_name:30s}  reg={reg_name}")
    log_result(
        f"Registry[{key}]",
        entry["classifier"] is not None or entry["regressor"] is not None,
        f"has at least one estimator class",
    )

# ══════════════════════════════════════════════════════════════════
# Test 2: Alias Resolution
# ══════════════════════════════════════════════════════════════════

print(f"\n{'─' * 70}")
print("TEST 2: Alias Resolution (_resolve_model_type)")
print(f"{'─' * 70}")

# Test a representative sample of aliases
test_aliases = {
    "RandomForestClassifier": "RandomForest",
    "randomforest": "RandomForest",
    "GradientBoostingClassifier": "GradientBoosting",
    "gradient_boosting": "GradientBoosting",
    "LogisticRegression": "LogisticRegression",
    "logistic": "LogisticRegression",
    "DecisionTreeClassifier": "DecisionTree",
    "decision_tree": "DecisionTree",
    "SVC": "SVC",
    "svm": "SVC",
    "KNeighborsClassifier": "KNN",
    "knn": "KNN",
    "AdaBoostClassifier": "AdaBoost",
    "adaboost": "AdaBoost",
    "ExtraTreesClassifier": "ExtraTrees",
    "extra_trees": "ExtraTrees",
    "XGBClassifier": "XGBoost",
    "xgboost": "XGBoost",
    "LGBMClassifier": "LightGBM",
    "lightgbm": "LightGBM",
    "lgbm": "LightGBM",
    "Ridge": "Ridge",
    "Lasso": "Lasso",
    "LinearRegression": "LinearRegression",
    "UnknownModel": "RandomForest",  # Should fall back
}

for alias, expected in test_aliases.items():
    resolved = _resolve_model_type(alias)
    log_result(
        f"resolve('{alias}')",
        resolved == expected,
        f"→ {resolved}" + ("" if resolved == expected else f" (expected {expected})"),
    )

# ══════════════════════════════════════════════════════════════════
# Test 3: Classifier Training (all algorithms)
# ══════════════════════════════════════════════════════════════════

print(f"\n{'─' * 70}")
print("TEST 3: Classifier Training — fit + predict on synthetic data")
print(f"{'─' * 70}")

classifier_keys = [
    k for k, v in _ALGORITHM_REGISTRY.items() if v["classifier"] is not None
]

for key in classifier_keys:
    try:
        t0 = time.time()
        model = _build_model(key, {}, random_seed=42, is_regressor=False)
        model.fit(X_cls_train, y_cls_train)
        y_pred = model.predict(X_cls_test)
        elapsed = time.time() - t0

        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_cls_test, y_pred)
        f1 = f1_score(y_cls_test, y_pred, average="weighted", zero_division=0)

        log_result(
            f"Classifier[{key:20s}]",
            acc > 0.0 and len(y_pred) == len(y_cls_test),
            f"acc={acc:.4f}  f1={f1:.4f}  time={elapsed:.3f}s  type={type(model).__name__}",
        )
    except Exception as e:
        log_result(f"Classifier[{key:20s}]", False, f"ERROR: {e}")
        traceback.print_exc()

# ══════════════════════════════════════════════════════════════════
# Test 4: Regressor Training (all algorithms)
# ══════════════════════════════════════════════════════════════════

print(f"\n{'─' * 70}")
print("TEST 4: Regressor Training — fit + predict on synthetic data")
print(f"{'─' * 70}")

regressor_keys = [
    k for k, v in _ALGORITHM_REGISTRY.items() if v["regressor"] is not None
]

for key in regressor_keys:
    try:
        t0 = time.time()
        model = _build_model(key, {}, random_seed=42, is_regressor=True)
        model.fit(X_reg_train, y_reg_train)
        y_pred = model.predict(X_reg_test)
        elapsed = time.time() - t0

        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(y_reg_test, y_pred)
        mse = mean_squared_error(y_reg_test, y_pred)

        log_result(
            f"Regressor[{key:20s}]",
            len(y_pred) == len(y_reg_test),
            f"r2={r2:.4f}  mse={mse:.1f}  time={elapsed:.3f}s  type={type(model).__name__}",
        )
    except Exception as e:
        log_result(f"Regressor[{key:20s}]", False, f"ERROR: {e}")
        traceback.print_exc()

# ══════════════════════════════════════════════════════════════════
# Test 5: Model Selector (automated multi-algorithm training)
# ══════════════════════════════════════════════════════════════════

print(f"\n{'─' * 70}")
print("TEST 5: Model Selector — automated multi-algorithm training")
print(f"{'─' * 70}")

try:
    t0 = time.time()
    best_model, best_name, best_metrics, leaderboard = train_candidate_models(
        X_cls_train, y_cls_train, X_cls_test, y_cls_test,
        config={"enabled": True, "metric": "f1_score", "random_seed": 42},
    )
    elapsed = time.time() - t0

    print(f"\n  Leaderboard ({len(leaderboard)} models, {elapsed:.2f}s total):")
    print(f"  {'Rank':<6} {'Model':<25} {'Accuracy':<12} {'F1':<12} {'Status'}")
    print(f"  {'─' * 65}")
    for rank, entry in enumerate(leaderboard, 1):
        if "error" in entry:
            print(f"  {rank:<6} {entry['model']:<25} {'—':<12} {'—':<12} ERROR: {entry['error'][:40]}")
        else:
            print(f"  {rank:<6} {entry['model']:<25} {entry.get('accuracy', 0):<12.4f} {entry.get('f1_score', 0):<12.4f}")

    log_result(
        "Model selection",
        best_model is not None and best_name != "",
        f"best={best_name}  f1={best_metrics.get('f1_score', 0):.4f}",
    )
    log_result(
        "Leaderboard completeness",
        len(leaderboard) == len(SELECTOR_ALGORITHMS),
        f"{len(leaderboard)}/{len(SELECTOR_ALGORITHMS)} models evaluated",
    )
except Exception as e:
    log_result("Model Selector", False, f"ERROR: {e}")
    traceback.print_exc()

# ══════════════════════════════════════════════════════════════════
# Test 6: Hyperparameter Search Objectives (dry run — 1 trial each)
# ══════════════════════════════════════════════════════════════════

print(f"\n{'─' * 70}")
print("TEST 6: Hyperparameter Search — 1-trial smoke test per algorithm")
print(f"{'─' * 70}")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

for model_type, objective_fn in HP_OBJECTIVES.items():
    try:
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial, fn=objective_fn: fn(trial, X_cls_train, y_cls_train),
            n_trials=1,
            show_progress_bar=False,
        )
        score = study.best_value
        params = study.best_params
        log_result(
            f"HP_Search[{model_type:20s}]",
            score > 0.0,
            f"score={score:.4f}  params={params}",
        )
    except Exception as e:
        log_result(f"HP_Search[{model_type:20s}]", False, f"ERROR: {e}")

# ══════════════════════════════════════════════════════════════════
# Test 7: Edge cases — classifier fallbacks
# ══════════════════════════════════════════════════════════════════

print(f"\n{'─' * 70}")
print("TEST 7: Edge Cases — fallback behaviour")
print(f"{'─' * 70}")

# LinearRegression used as classifier → should fall back to RandomForestClassifier
model = _build_model("LinearRegression", {}, random_seed=42, is_regressor=False)
log_result(
    "LinearRegression as classifier",
    "RandomForest" in type(model).__name__,
    f"→ {type(model).__name__} (expected RandomForestClassifier fallback)",
)

# LogisticRegression used as regressor → should fall back to RandomForestRegressor
model = _build_model("LogisticRegression", {}, random_seed=42, is_regressor=True)
log_result(
    "LogisticRegression as regressor",
    "RandomForest" in type(model).__name__,
    f"→ {type(model).__name__} (expected RandomForestRegressor fallback)",
)

# Unknown model type → should fall back to RandomForest
model = _build_model("NonexistentModel", {}, random_seed=42, is_regressor=False)
log_result(
    "Unknown model fallback",
    "RandomForest" in type(model).__name__,
    f"→ {type(model).__name__} (expected RandomForestClassifier fallback)",
)

# ══════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════

print(f"\n{'═' * 70}")
print("SUMMARY")
print(f"{'═' * 70}")
total = results["pass"] + results["fail"] + results["warn"]
print(f"  Total tests: {total}")
print(f"  {PASS}: {results['pass']}")
print(f"  {FAIL}: {results['fail']}")
print(f"  {WARN}: {results['warn']}")

if results["fail"] == 0:
    print(f"\n  >>> ALL TESTS PASSED -- all algorithms are working correctly!")
else:
    print(f"\n  ⚠  {results['fail']} test(s) failed — review the output above.")
    sys.exit(1)
