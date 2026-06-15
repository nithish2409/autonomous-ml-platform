"""
Prometheus metrics definitions for the Autonomous ML Platform.
"""

from prometheus_client import Counter, Gauge, Histogram

# ── Inference ────────────────────────────────────────────────────
inference_latency_seconds = Histogram(
    "inference_latency_seconds",
    "End-to-end inference latency in seconds",
    labelnames=["model_id"],
)

inference_requests_total = Counter(
    "inference_requests_total",
    "Total inference requests by model",
    labelnames=["model_id"],
)

inference_errors_total = Counter(
    "inference_errors_total",
    "Total inference errors by model",
    labelnames=["model_id"],
)

# ── Training ─────────────────────────────────────────────────────
training_jobs_total = Counter(
    "training_jobs_total",
    "Total training jobs by outcome",
    labelnames=["status"],
)

# ── Monitoring ───────────────────────────────────────────────────
drift_score_gauge = Gauge(
    "drift_score_gauge",
    "Latest drift score per model",
    labelnames=["model_id"],
)

# ── Automation ───────────────────────────────────────────────────
automation_actions_total = Counter(
    "automation_actions_total",
    "Automation executor actions by type and result",
    labelnames=["action", "status"],
)

# ── Model Registry ───────────────────────────────────────────────
model_version_changes_total = Counter(
    "model_version_changes_total",
    "Model version changes (retrain / rollback)",
    labelnames=["model_id", "change_type"],
)

# ── Policy & Guardrails ──────────────────────────────────────────
policy_decisions_total = Counter(
    "policy_decisions_total",
    "Policy evaluation outcomes (approved / requires_human / blocked)",
    labelnames=["outcome"],
)

policy_violations_total = Counter(
    "policy_violations_total",
    "Policy guardrail violations by rule name",
    labelnames=["rule"],
)
