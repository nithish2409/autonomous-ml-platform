"""
Training Runner — lightweight in-process model training.
Fetches dataset from MinIO, trains sklearn model, saves artifact back.
No Docker required.

Phase A: Feature selection, hyperparameter search, threshold optimization.
Phase B: Data rebalancing, automated model selection, weighted ensemble.
All enhancements are config-driven with safe defaults.
"""

import io
import os
import uuid
import logging
import tempfile
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

# Optional external frameworks — graceful fallback if not installed
try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func as sa_func

from app.core.storage import MinioClient
from app.models.model_registry import ModelRegistry
from app.models.model_version import ModelVersion
from app.models.dataset import Dataset
from app.models.training_job import TrainingJob

# Phase A modules
from app.training.feature_selection import select_features
from app.training.hyperparameter_search import run_hyperparameter_search
from app.training.threshold_optimizer import find_best_threshold

# Phase B modules
from app.training.rebalancing import rebalance_data
from app.training.model_selector import train_candidate_models
from app.training.ensemble import build_ensemble

logger = logging.getLogger("training_runner")

# ── Unified algorithm registry ───────────────────────────────────
# Maps user-facing key → (ClassifierClass, RegressorClass, default_hyperparams)
# ClassifierClass or RegressorClass can be None if only one task type is supported.

_ALGORITHM_REGISTRY: dict[str, dict[str, Any]] = {
    # sklearn — ensemble
    "RandomForest": {
        "classifier": RandomForestClassifier,
        "regressor": RandomForestRegressor,
        "params": {"n_estimators": 100, "max_depth": 10},
        "param_keys": {"n_estimators", "max_depth"},
    },
    "GradientBoosting": {
        "classifier": GradientBoostingClassifier,
        "regressor": GradientBoostingRegressor,
        "params": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
        "param_keys": {"n_estimators", "max_depth", "learning_rate"},
    },
    "AdaBoost": {
        "classifier": AdaBoostClassifier,
        "regressor": AdaBoostRegressor,
        "params": {"n_estimators": 100, "learning_rate": 1.0},
        "param_keys": {"n_estimators", "learning_rate"},
    },
    "ExtraTrees": {
        "classifier": ExtraTreesClassifier,
        "regressor": ExtraTreesRegressor,
        "params": {"n_estimators": 100, "max_depth": 10},
        "param_keys": {"n_estimators", "max_depth"},
    },
    # sklearn — linear
    "LogisticRegression": {
        "classifier": LogisticRegression,
        "regressor": None,
        "params": {"max_iter": 500},
        "param_keys": {"C", "max_iter"},
    },
    "LinearRegression": {
        "classifier": None,
        "regressor": LinearRegression,
        "params": {},
        "param_keys": set(),
    },
    "Ridge": {
        "classifier": None,
        "regressor": Ridge,
        "params": {"alpha": 1.0},
        "param_keys": {"alpha"},
    },
    "Lasso": {
        "classifier": None,
        "regressor": Lasso,
        "params": {"alpha": 1.0},
        "param_keys": {"alpha"},
    },
    # sklearn — tree
    "DecisionTree": {
        "classifier": DecisionTreeClassifier,
        "regressor": DecisionTreeRegressor,
        "params": {"max_depth": 10},
        "param_keys": {"max_depth"},
    },
    # sklearn — SVM
    "SVC": {
        "classifier": SVC,
        "regressor": SVR,
        "params": {},
        "param_keys": {"C", "kernel"},
    },
    # sklearn — neighbors
    "KNN": {
        "classifier": KNeighborsClassifier,
        "regressor": KNeighborsRegressor,
        "params": {"n_neighbors": 5},
        "param_keys": {"n_neighbors"},
    },
}

# Register XGBoost if installed
if _HAS_XGBOOST:
    _ALGORITHM_REGISTRY["XGBoost"] = {
        "classifier": XGBClassifier,
        "regressor": XGBRegressor,
        "params": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1,
                   "use_label_encoder": False, "eval_metric": "logloss", "verbosity": 0},
        "param_keys": {"n_estimators", "max_depth", "learning_rate"},
    }

# Register LightGBM if installed
if _HAS_LIGHTGBM:
    _ALGORITHM_REGISTRY["LightGBM"] = {
        "classifier": LGBMClassifier,
        "regressor": LGBMRegressor,
        "params": {"n_estimators": 100, "max_depth": -1, "learning_rate": 0.1, "verbosity": -1},
        "param_keys": {"n_estimators", "max_depth", "learning_rate", "num_leaves"},
    }

# ── Default training config (all enhancements disabled) ──────────────
_DEFAULT_TRAINING_CONFIG: dict[str, Any] = {
    # Phase A
    "feature_selection": {
        "enabled": False,
        "method": "model_importance",
        "top_k": 20,
    },
    "hyperparameter_search": {
        "enabled": False,
        "trials": 20,
    },
    "threshold_optimization": {
        "enabled": False,
        "metric": "f1",
    },
    # Phase B
    "rebalancing": {
        "enabled": False,
        "imbalance_threshold": 0.4,
    },
    "model_selection": {
        "enabled": False,
        "metric": "f1_score",
    },
    "ensemble": {
        "enabled": False,
        "weight_new": 0.7,
    },
}


def _merge_config(user_config: dict | None) -> dict[str, Any]:
    """Merge user-provided config with safe defaults."""
    base = {k: dict(v) for k, v in _DEFAULT_TRAINING_CONFIG.items()}
    if not user_config:
        return base
    for section in base:
        if section in user_config:
            base[section].update(user_config[section])
    return base


# ── Alias map: user-facing model_class string → registry key ─────
_MODEL_CLASS_ALIASES: dict[str, str] = {
    # Exact keys (case-insensitive matching is done in _resolve_model_type)
    "randomforest": "RandomForest",
    "randomforestclassifier": "RandomForest",
    "randomforestregressor": "RandomForest",
    "gradientboosting": "GradientBoosting",
    "gradientboostingclassifier": "GradientBoosting",
    "gradientboostingregressor": "GradientBoosting",
    "adaboost": "AdaBoost",
    "adaboostclassifier": "AdaBoost",
    "adaboostregressor": "AdaBoost",
    "extratrees": "ExtraTrees",
    "extratreesclassifier": "ExtraTrees",
    "extratreesregressor": "ExtraTrees",
    "logisticregression": "LogisticRegression",
    "logistic": "LogisticRegression",
    "linearregression": "LinearRegression",
    "linear": "LinearRegression",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "decisiontree": "DecisionTree",
    "decisiontreeclassifier": "DecisionTree",
    "decisiontreeregressor": "DecisionTree",
    "svc": "SVC",
    "svr": "SVC",
    "svm": "SVC",
    "knn": "KNN",
    "kneighborsclassifier": "KNN",
    "kneighborsregressor": "KNN",
    "xgboost": "XGBoost",
    "xgb": "XGBoost",
    "xgbclassifier": "XGBoost",
    "xgbregressor": "XGBoost",
    "lightgbm": "LightGBM",
    "lgbm": "LightGBM",
    "lgbmclassifier": "LightGBM",
    "lgbmregressor": "LightGBM",
}


def _resolve_model_type(model_class: str) -> str:
    """Map the registry model_class string to a canonical algorithm key."""
    mc = (model_class or "").lower().replace(" ", "").replace("_", "")
    resolved = _MODEL_CLASS_ALIASES.get(mc, "RandomForest")

    # Verify the resolved key exists in the registry (may be missing if
    # xgboost/lightgbm not installed)
    if resolved not in _ALGORITHM_REGISTRY:
        logger.warning(
            "Algorithm '%s' not available (missing dependency?) — falling back to RandomForest",
            resolved,
        )
        return "RandomForest"
    return resolved


def _build_model(model_type: str, params: dict, random_seed: int, is_regressor: bool):
    """Instantiate a model from the algorithm registry."""
    entry = _ALGORITHM_REGISTRY.get(model_type)
    if entry is None:
        logger.warning("Unknown model type '%s' — falling back to RandomForest", model_type)
        entry = _ALGORITHM_REGISTRY["RandomForest"]

    # Pick classifier or regressor class
    if is_regressor:
        cls = entry.get("regressor")
        if cls is None:
            # Regression-only fallback (e.g. LogisticRegression has no regressor)
            logger.warning("%s has no regressor variant — falling back to RandomForestRegressor", model_type)
            entry = _ALGORITHM_REGISTRY["RandomForest"]
            cls = entry.get("regressor")
    else:
        cls = entry.get("classifier")
        if cls is None:
            # Classification-only fallback (e.g. LinearRegression has no classifier)
            logger.warning("%s has no classifier variant — falling back to RandomForestClassifier", model_type)
            entry = _ALGORITHM_REGISTRY["RandomForest"]
            cls = entry.get("classifier")

    # Merge defaults with search params, filtering to allowed keys
    allowed_keys = entry.get("param_keys", set())
    defaults = dict(entry.get("params", {}))
    filtered_params = {k: v for k, v in params.items() if k in allowed_keys}
    defaults.update(filtered_params)

    # Inject random_state if the estimator accepts it
    import inspect
    sig = inspect.signature(cls)
    if "random_state" in sig.parameters:
        defaults["random_state"] = random_seed

    return cls(**defaults)


class TrainingRunner:
    """Lightweight in-process model training (sklearn / XGBoost)."""

    def __init__(self):
        self.storage = MinioClient()

    async def _augment_with_inference_logs(
        self,
        df: pd.DataFrame,
        model_id: str,
        db: AsyncSession,
        max_logs: int = 200,
    ) -> pd.DataFrame:
        """
        Augment training data with pseudo-labeled inference logs.

        Queries recent inference logs, extracts input features and model
        predictions, and appends them as additional training rows.  This
        allows retrained models to learn from the current (potentially
        drifted) inference distribution.
        """
        from app.models.inference_log import InferenceLog

        result = await db.execute(
            select(InferenceLog)
            .where(InferenceLog.model_id == model_id)
            .order_by(InferenceLog.created_at.desc())
            .limit(max_logs)
        )
        logs = result.scalars().all()

        if not logs:
            logger.info("No inference logs found for augmentation (model=%s)", model_id)
            return df

        target_col = df.columns[-1]  # last column is the target
        feature_cols = [c for c in df.columns if c != target_col]

        augmented_rows = []
        for log in logs:
            features = log.input_summary or {}
            pred = log.prediction_summary or {}

            # Check if we have the true label inside the input features (ground-truth feedback)
            if target_col in features:
                label = features.get(target_col)
            else:
                # Extract the predicted label as pseudo-label
                pred_val = pred.get("prediction", [None])
                if isinstance(pred_val, list) and len(pred_val) > 0:
                    label = pred_val[0]
                else:
                    label = pred_val

            if label is None:
                continue

            row = {col: features.get(col) for col in feature_cols}
            row[target_col] = label

            # Skip rows with missing feature values
            if any(v is None for v in row.values()):
                continue

            augmented_rows.append(row)

        if not augmented_rows:
            logger.info("No usable inference logs for augmentation (model=%s)", model_id)
            return df

        aug_df = pd.DataFrame(augmented_rows)
        combined = pd.concat([df, aug_df], ignore_index=True)

        logger.info(
            "Augmented training data: %d original + %d inference logs = %d total (model=%s)",
            len(df), len(augmented_rows), len(combined), model_id,
        )
        return combined

    async def train_candidate(
        self,
        model_id: str,
        db: AsyncSession,
        split_ratio: float = 0.2,
        random_seed: int = 42,
        training_config: dict[str, Any] | None = None,
    ) -> dict:
        """
        Train a candidate model with Phase A + Phase B enhancements:

        1.  Look up model + dataset
        2.  Fetch CSV from MinIO
        3.  Feature selection       (Phase A)
        4.  Rebalancing             (Phase B)
        5.  Hyperparameter search   (Phase A)
        6.  Model selection         (Phase B)
        7.  Threshold optimization  (Phase A)
        8.  Ensemble                (Phase B, optional)
        9.  Upload artifact to MinIO
        10. Return artifact_path + enriched metrics
        """
        cfg = _merge_config(training_config)

        # ── Lookup model ──
        result = await db.execute(
            select(ModelRegistry).where(ModelRegistry.id == model_id)
        )
        model = result.scalars().first()
        if model is None:
            raise ValueError(f"Model {model_id} not found")

        # ── Lookup dataset ──
        result = await db.execute(
            select(Dataset).where(Dataset.id == model.dataset_id)
        )
        dataset = result.scalars().first()
        if dataset is None:
            raise ValueError(f"Dataset {model.dataset_id} not found")
        if not dataset.minio_path:
            raise ValueError(f"Dataset {model.dataset_id} has no minio_path")

        # ── Compute next version ──
        result = await db.execute(
            select(sa_func.count(ModelVersion.id)).where(
                ModelVersion.model_id == model_id
            )
        )
        version_count = result.scalar() or 0
        version = f"v{version_count + 1}"
        previous_version = model.current_version

        # ── Create training job (pending → running) ──
        job_id = str(uuid.uuid4())
        job = TrainingJob(
            id=job_id,
            model_id=model_id,
            config={
                "framework": model.framework,
                "model_class": model.model_class,
                "dataset_path": dataset.minio_path,
                "version": version,
                "runner": "in_process",
                "split_ratio": split_ratio,
                "random_seed": random_seed,
                "training_config": cfg,
            },
            status="running",
        )
        db.add(job)
        await db.commit()

        try:
            # ── Fetch dataset from MinIO ──
            response = self.storage.client.get_object(
                self.storage.bucket, dataset.minio_path
            )
            df = pd.read_csv(io.BytesIO(response.read()))
            response.close()
            response.release_conn()

            if df.empty:
                raise ValueError("Dataset is empty")

            # ── Augment with inference logs (pseudo-labeling) ──
            df = await self._augment_with_inference_logs(df, model_id, db)

            # ── Prepare data ──
            X = df.iloc[:, :-1].select_dtypes(include="number")
            y = df.iloc[:, -1]

            if X.empty:
                raise ValueError("No numeric features found in dataset")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=split_ratio, random_state=random_seed
            )

            # ══════════════════════════════════════════════════════════
            # Phase A — Step 1: Feature Selection
            # ══════════════════════════════════════════════════════════
            X_train, selected_features = select_features(
                X_train, y_train, cfg["feature_selection"]
            )
            X_test = X_test[selected_features]

            # ══════════════════════════════════════════════════════════
            # Phase B — Step 2: Data Rebalancing
            # ══════════════════════════════════════════════════════════
            X_train, y_train, rebalance_meta = rebalance_data(
                X_train, y_train, cfg["rebalancing"]
            )

            # ── Resolve model type ──
            model_class = model.model_class or ""
            is_regressor = "regressor" in model_class.lower()
            model_type = _resolve_model_type(model_class)

            # ══════════════════════════════════════════════════════════
            # Phase A — Step 3: Hyperparameter Search
            # ══════════════════════════════════════════════════════════
            if not is_regressor:
                best_params = run_hyperparameter_search(
                    model_type, X_train, y_train, cfg["hyperparameter_search"]
                )
            else:
                best_params = {"n_estimators": 100, "max_depth": 8}

            # ══════════════════════════════════════════════════════════
            # Phase B — Steps 4–5: Model Selection OR single model
            # ══════════════════════════════════════════════════════════
            model_selection_cfg = cfg["model_selection"]
            leaderboard: list[dict[str, Any]] = []
            best_model_name = model_type

            if model_selection_cfg.get("enabled") and not is_regressor:
                # Automated model selection — trains multiple algorithms
                clf, best_model_name, _, leaderboard = train_candidate_models(
                    X_train, y_train, X_test, y_test,
                    {**model_selection_cfg, "random_seed": random_seed},
                )
            else:
                # Single model path (original behavior)
                clf = _build_model(model_type, best_params, random_seed, is_regressor)
                clf.fit(X_train, y_train)

            # ══════════════════════════════════════════════════════════
            # Phase A — Step 6: Threshold Optimization + Metrics
            # ══════════════════════════════════════════════════════════
            y_pred = clf.predict(X_test)

            if is_regressor:
                metrics: dict[str, Any] = {
                    "mse": round(float(mean_squared_error(y_test, y_pred)), 4),
                    "r2": round(float(r2_score(y_test, y_pred)), 4),
                }
            else:
                metrics = {
                    "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                    "f1_score": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
                }

                # Threshold optimization (classifiers only)
                if cfg["threshold_optimization"]["enabled"] and hasattr(clf, "predict_proba"):
                    y_probs = clf.predict_proba(X_test)
                    threshold_result = find_best_threshold(
                        y_test, y_probs,
                        metric=cfg["threshold_optimization"].get("metric", "f1"),
                    )
                    metrics["optimal_threshold"] = threshold_result["best_threshold"]
                    metrics["threshold_score"] = threshold_result["best_score"]
                    metrics["threshold_metric"] = threshold_result["metric"]

            # ══════════════════════════════════════════════════════════
            # Phase B — Step 7: Ensemble (optional)
            # ══════════════════════════════════════════════════════════
            ensemble_meta: dict[str, Any] = {"ensemble_used": False}
            if cfg["ensemble"].get("enabled") and not is_regressor:
                old_model = None
                if previous_version:
                    try:
                        version_result = await db.execute(
                            select(ModelVersion).where(
                                ModelVersion.model_id == model_id,
                                ModelVersion.version_number == previous_version,
                            )
                        )
                        current_version_entry = version_result.scalars().first()
                        if current_version_entry and current_version_entry.artifact_path:
                            # Download from MinIO
                            response = self.storage.client.get_object(
                                self.storage.bucket, current_version_entry.artifact_path
                            )
                            old_model = joblib.load(io.BytesIO(response.read()))
                            response.close()
                            response.release_conn()
                            logger.info("Loaded old model from MinIO for ensemble building.")
                    except Exception as e:
                        logger.warning("Failed to load old model for ensemble: %s", e)

                clf, ensemble_meta = build_ensemble(old_model, clf, cfg["ensemble"])

            # ══════════════════════════════════════════════════════════
            # Enrich metrics with Phase A + Phase B info
            # ══════════════════════════════════════════════════════════
            metrics["selected_features"] = selected_features
            metrics["best_params"] = best_params
            # Phase B fields
            metrics["selected_model"] = best_model_name
            metrics["leaderboard"] = leaderboard
            metrics["imbalance_ratio"] = rebalance_meta.get("imbalance_ratio")
            metrics["rebalance_method"] = rebalance_meta.get("method_used")
            metrics["ensemble_used"] = ensemble_meta.get("ensemble_used", False)

            # ── Save artifact to MinIO ──
            with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
                joblib.dump(clf, tmp.name)
                tmp_path = tmp.name

            artifact_path = f"models/{model_id}/{version}/model.joblib"
            self.storage.client.fput_object(self.storage.bucket, artifact_path, tmp_path)
            os.remove(tmp_path)

            # ── Update training job ──
            job.status = "completed"
            job.result_metrics = metrics
            await db.commit()

            logger.info(
                "Candidate trained: model=%s version=%s metrics=%s",
                model_id, version, metrics,
            )

            return {
                "job_id": job_id,
                "model_id": str(model_id),
                "version": version,
                "previous_version": previous_version,
                "artifact_path": artifact_path,
                "metrics": metrics,
                "status": "completed",
                "framework": model.framework,
                "model_class": model.model_class,
            }

        except Exception as e:
            job.status = "failed"
            job.result_metrics = {"error": str(e)}
            await db.commit()
            logger.error("Candidate training failed for model %s: %s", model_id, e)
            raise
