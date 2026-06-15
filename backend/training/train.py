"""
Standalone training script — runs inside an isolated Docker container.

Env vars required:
    MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET,
    DATASET_PATH, MODEL_ID, VERSION, FRAMEWORK, MODEL_CLASS

Optional env vars:
    TARGET_COLUMN     – name of target column (default: last numeric column)
    HYPERPARAMETERS   – JSON string of hyperparameters for the model
    SPLIT_RATIO       – test split ratio, e.g. "0.2" (default: 0.2)
    RANDOM_SEED       – random seed (default: 42)
"""

import io
import json
import os
import sys
import time
import tempfile

import joblib
import numpy as np
import pandas as pd
from minio import Minio


# ── Helpers ──────────────────────────────────────────────────────

def get_env(key: str) -> str:
    val = os.getenv(key)
    if val is None:
        raise RuntimeError(f"Missing required env var: {key}")
    return val


def get_env_optional(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)


# ── MinIO I/O ────────────────────────────────────────────────────

def download_dataset(client: Minio, bucket: str, dataset_path: str) -> pd.DataFrame:
    """Download CSV from MinIO and return as DataFrame."""
    response = client.get_object(bucket, dataset_path)
    df = pd.read_csv(io.BytesIO(response.read()))
    response.close()
    response.release_conn()
    return df


def upload_artifact(client: Minio, bucket: str, model, model_id: str, version: str) -> str:
    """Save model with joblib, upload to MinIO, return artifact path."""
    artifact_path = f"models/{model_id}/{version}.bin"

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        joblib.dump(model, tmp.name)
        tmp_path = tmp.name

    client.fput_object(bucket, artifact_path, tmp_path)
    os.remove(tmp_path)

    return artifact_path


# ── Model Factory ────────────────────────────────────────────────

def create_model(framework: str, model_class: str, hyperparameters: dict | None = None):
    """Instantiate a model from framework + class name + optional hyperparameters."""
    hp = hyperparameters or {}

    if framework == "sklearn":
        import sklearn.ensemble
        import sklearn.linear_model
        import sklearn.tree
        import sklearn.svm
        import sklearn.neighbors

        for module in [
            sklearn.ensemble,
            sklearn.linear_model,
            sklearn.tree,
            sklearn.svm,
            sklearn.neighbors,
        ]:
            cls = getattr(module, model_class, None)
            if cls is not None:
                return cls(**hp)
        raise ValueError(f"Unknown sklearn class: {model_class}")

    elif framework == "xgboost":
        import xgboost

        # Support 'XGBoostClassifier' or 'XGBoostRegressor' variants gracefully
        if model_class.startswith("XGBoost"):
            model_class = model_class.replace("XGBoost", "XGB")

        cls = getattr(xgboost, model_class, None)
        if cls is None:
            raise ValueError(f"Unknown xgboost class: {model_class}")
        return cls(**hp)

    elif framework == "lightgbm":
        import lightgbm

        cls = getattr(lightgbm, model_class, None)
        if cls is None:
            raise ValueError(f"Unknown lightgbm class: {model_class}")
        return cls(**hp)

    else:
        raise ValueError(f"Unsupported framework: {framework}")


# ── Metrics Computation ─────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob, task_type: str) -> dict:
    """Compute comprehensive metrics based on task type."""
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        mean_squared_error,
    )

    metrics = {}

    if task_type == "classification":
        metrics["accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4)

        # Handle multi-class with weighted average
        avg = "binary" if len(set(y_true)) == 2 else "weighted"
        metrics["precision"] = round(float(precision_score(y_true, y_pred, average=avg, zero_division=0)), 4)
        metrics["recall"] = round(float(recall_score(y_true, y_pred, average=avg, zero_division=0)), 4)
        metrics["f1_score"] = round(float(f1_score(y_true, y_pred, average=avg, zero_division=0)), 4)

        # ROC AUC (only if probabilities are available)
        if y_prob is not None:
            try:
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob[:, 1])), 4)
                elif y_prob.ndim == 2 and y_prob.shape[1] > 2:
                    metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")), 4)
                else:
                    metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
            except (ValueError, TypeError):
                metrics["roc_auc"] = None
    else:
        # Regression
        mse = float(mean_squared_error(y_true, y_pred))
        metrics["mse"] = round(mse, 4)
        metrics["rmse"] = round(float(np.sqrt(mse)), 4)

    return metrics


# ── Training Logic ───────────────────────────────────────────────

def train_model(
    df: pd.DataFrame,
    framework: str,
    model_class: str,
    target_column: str | None = None,
    hyperparameters: dict | None = None,
    split_ratio: float = 0.2,
    random_seed: int = 42,
) -> tuple:
    """Train a model and return (model, metrics_dict)."""
    from sklearn.model_selection import train_test_split

    numeric_df = df.select_dtypes(include="number").dropna()

    if numeric_df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns (features + target)")

    # Pick target column
    if target_column and target_column in numeric_df.columns:
        y = numeric_df[target_column]
        X = numeric_df.drop(columns=[target_column])
    else:
        X = numeric_df.iloc[:, :-1]
        y = numeric_df.iloc[:, -1]
        target_column = numeric_df.columns[-1]

    # Auto-detect task type
    n_unique = y.nunique()
    is_continuous = n_unique > 20 or (n_unique / len(y) > 0.5)
    task_type = "regression" if is_continuous else "classification"

    # Swap Classifier ↔ Regressor if mismatch
    actual_class = model_class
    if is_continuous and "Classifier" in model_class:
        actual_class = model_class.replace("Classifier", "Regressor")
    elif not is_continuous and "Regressor" in model_class:
        actual_class = model_class.replace("Regressor", "Classifier")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=random_seed,
    )

    # Create and train model
    model = create_model(framework, actual_class, hyperparameters)

    t0 = time.time()
    model.fit(X_train, y_train)
    training_time = round(time.time() - t0, 3)

    # Predictions
    y_pred = model.predict(X_test)

    # Probabilities (for ROC AUC)
    y_prob = None
    if task_type == "classification" and hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
        except Exception:
            y_prob = None

    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, y_prob, task_type)
    metrics["training_time"] = training_time
    metrics["n_samples"] = int(len(X))
    metrics["n_train_samples"] = int(len(X_train))
    metrics["n_test_samples"] = int(len(X_test))
    metrics["n_features"] = int(X.shape[1])
    metrics["target_column"] = target_column
    metrics["model_class_used"] = actual_class
    metrics["task_type"] = task_type
    metrics["hyperparameters"] = hyperparameters if hyperparameters else {}

    return model, metrics


# ── Main ─────────────────────────────────────────────────────────

def main():
    # Read config from env
    endpoint = get_env("MINIO_ENDPOINT")
    access_key = get_env("MINIO_ACCESS_KEY")
    secret_key = get_env("MINIO_SECRET_KEY")
    bucket = get_env("MINIO_BUCKET")
    dataset_path = get_env("DATASET_PATH")
    model_id = get_env("MODEL_ID")
    version = get_env("VERSION")
    framework = get_env("FRAMEWORK")
    model_class = get_env("MODEL_CLASS")

    # Optional config
    target_column = get_env_optional("TARGET_COLUMN")
    hp_json = get_env_optional("HYPERPARAMETERS", "{}")
    split_ratio = float(get_env_optional("SPLIT_RATIO", "0.2"))
    random_seed = int(get_env_optional("RANDOM_SEED", "42"))

    hyperparameters = {}
    if hp_json:
        try:
            hyperparameters = json.loads(hp_json)
        except json.JSONDecodeError:
            hyperparameters = {}

    client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)

    # 1. Download dataset
    print(f"[TRAIN] Downloading dataset: {dataset_path}", flush=True)
    df = download_dataset(client, bucket, dataset_path)
    print(f"[TRAIN] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns", flush=True)

    # 2. Train model
    print(f"[TRAIN] Training {framework}/{model_class}...", flush=True)
    model, metrics = train_model(
        df,
        framework,
        model_class,
        target_column=target_column,
        hyperparameters=hyperparameters,
        split_ratio=split_ratio,
        random_seed=random_seed,
    )
    print(f"[TRAIN] Training complete. Metrics: {json.dumps(metrics)}", flush=True)

    # 3. Upload artifact
    print(f"[TRAIN] Uploading artifact...", flush=True)
    artifact_path = upload_artifact(client, bucket, model, model_id, version)
    print(f"[TRAIN] Artifact uploaded: {artifact_path}", flush=True)

    # 4. Print JSON result to stdout (parsed by orchestrator)
    result = {
        "status": "completed",
        "artifact_path": artifact_path,
        "metrics": metrics,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_result = {"status": "failed", "error": str(e)}
        print(json.dumps(error_result))
        sys.exit(1)
