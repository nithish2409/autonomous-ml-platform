"""Policy evaluator — checks a decision against the current policy config."""


from typing import Any


def evaluate_decision(decision: dict[str, Any], policies: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a decision against policy rules.

    Args:
        decision: dict with keys like confidence, severity, estimated_cost,
                  resource_profile, is_production, etc.
        policies: the full policy config dict (auto_approval, guardrails, escalation).

    Returns:
        {
            "approved": bool,
            "requires_human": bool,
            "blocked": bool,
            "reason": str,
        }
    """
    auto = policies.get("auto_approval", {})
    guard = policies.get("guardrails", {})

    confidence = _to_float(decision.get("confidence", 0))
    # Normalise to 0-100 scale
    if confidence <= 1:
        confidence *= 100

    severity = (decision.get("severity") or "low").lower()
    cost = _to_float(decision.get("estimated_cost", 0))
    is_production = decision.get("is_production", False)

    # ── 1. Freeze window check ───────────────────────────────
    if guard.get("freeze_window", False):
        return _result(
            blocked=True,
            reason="Blocked: deployment freeze window is active. No automated actions permitted.",
        )

    # ── 2. Guardrail violations ──────────────────────────────
    max_gpu = guard.get("max_gpu_per_job", 8)
    requested_gpu = _to_int(decision.get("gpu_count", 0))
    if requested_gpu > max_gpu:
        return _result(
            blocked=True,
            reason=f"Blocked: requested GPU count ({requested_gpu}) exceeds guardrail limit ({max_gpu}).",
        )

    max_daily = _to_float(guard.get("max_daily_cost", 9999))
    daily_spent = _to_float(decision.get("daily_cost_so_far", 0))
    if daily_spent + cost > max_daily:
        return _result(
            blocked=True,
            reason=f"Blocked: projected daily cost (${daily_spent + cost:.2f}) exceeds guardrail limit (${max_daily:.2f}).",
        )

    max_retrains = guard.get("max_retrains_24h", 50)
    retrains_today = _to_int(decision.get("retrains_today", 0))
    if retrains_today >= max_retrains:
        return _result(
            blocked=True,
            reason=f"Blocked: retrain count ({retrains_today}) has reached 24h limit ({max_retrains}).",
        )

    # ── 3. Severity not in allowed list ──────────────────────
    allowed = [s.lower() for s in auto.get("allowed_severity", ["low", "medium"])]
    if severity not in allowed:
        return _result(
            requires_human=True,
            reason=f"Requires human review: severity '{severity}' is not in the allowed auto-approval list {allowed}.",
        )

    # ── 4. Production block ──────────────────────────────────
    if is_production and auto.get("block_production", True):
        return _result(
            requires_human=True,
            reason="Requires human review: production model changes require manual approval per policy.",
        )

    # ── 5. Confidence + cost auto-approval ───────────────────
    min_conf = _to_float(auto.get("min_confidence", 85))
    max_cost = _to_float(auto.get("max_cost", 500))

    if confidence >= min_conf and cost <= max_cost:
        return _result(
            approved=True,
            reason=f"Auto-approved: confidence ({confidence:.1f}%) >= threshold ({min_conf}%), cost (${cost:.2f}) <= limit (${max_cost:.2f}).",
        )

    # ── 6. Critical severity ─────────────────────────────────
    if severity in ("critical", "high"):
        return _result(
            requires_human=True,
            reason=f"Requires human review: severity is '{severity}'. Confidence {confidence:.1f}%, cost ${cost:.2f}.",
        )

    # ── 7. Default fallback ──────────────────────────────────
    reasons = []
    if confidence < min_conf:
        reasons.append(f"confidence ({confidence:.1f}%) below threshold ({min_conf}%)")
    if cost > max_cost:
        reasons.append(f"cost (${cost:.2f}) exceeds limit (${max_cost:.2f})")
    reason_str = "; ".join(reasons) if reasons else "policy conditions not met"

    return _result(
        requires_human=True,
        reason=f"Requires human review: {reason_str}.",
    )


# ── Internal helpers ─────────────────────────────────────────

def _result(approved=False, requires_human=False, blocked=False, reason=""):
    return {
        "approved": approved,
        "requires_human": requires_human,
        "blocked": blocked,
        "reason": reason,
    }


def _to_float(v):
    try:
        return float(v) if v is not None else 0.0
    except (ValueError, TypeError):
        return 0.0


def _to_int(v):
    try:
        return int(v) if v is not None else 0
    except (ValueError, TypeError):
        return 0
