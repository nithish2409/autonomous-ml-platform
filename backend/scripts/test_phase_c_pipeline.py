"""
Test script for Phase C — Autonomous Evolution Layer.

Simulates three drift scenarios and verifies the full decision chain:
  Drift → Strategy → Config → Resource Profile

Usage:
    cd backend
    python scripts/test_phase_c_pipeline.py
"""

import json
import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.training.drift_analyzer import analyze_drift
from app.training.strategy_engine import choose_strategy
from app.training.config_generator import generate_training_config
from app.training.early_stopping import should_stop_early
from app.training.resource_manager import adjust_training_resources

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger("test_phase_c")


def run_scenario(name: str, drift_score: float, perf_delta: float, dataset_size: int):
    """Simulate one decision-chain scenario."""
    print(f"\n{'─' * 60}")
    print(f"  Scenario: {name}")
    print(f"  drift_score={drift_score}  perf_delta={perf_delta}  rows={dataset_size}")
    print(f"{'─' * 60}")

    # 1. Drift analysis
    drift = analyze_drift(drift_score, perf_delta)
    print(f"  1. Drift type: {drift['drift_type']}  severity: {drift['severity']}")

    # 2. Strategy selection
    strategy = choose_strategy("retrain", drift)
    print(f"  2. Strategy:   {strategy}")

    # 3. Config generation
    config = generate_training_config(strategy)
    if config is None:
        print(f"  3. Config:     None (skip retraining)")
    else:
        enabled = [k for k, v in config.items() if isinstance(v, dict) and v.get("enabled")]
        print(f"  3. Config:     enabled modules = {enabled}")

    # 4. Resource allocation
    resources = adjust_training_resources(dataset_size)
    print(f"  4. Resources:  profile={resources['profile']}  parallelism={resources['parallelism']}")

    return {
        "scenario": name,
        "drift": drift,
        "strategy": strategy,
        "config": config,
        "resources": resources,
    }


def main():
    print("=" * 60)
    print("  Phase C — Autonomous Evolution Layer Test")
    print("=" * 60)

    # ── Case 1: Low drift → incremental (minimal retraining) ──
    r1 = run_scenario(
        name="Low Drift — Incremental",
        drift_score=0.15,
        perf_delta=0.01,
        dataset_size=10_000,
    )
    assert r1["drift"]["drift_type"] == "no_drift"
    assert r1["strategy"] == "incremental"
    assert r1["config"] is not None  # incremental returns a config (all disabled)
    print("  ✓ Assertions passed")

    # ── Case 2: Feature drift → quick_tune ──
    r2 = run_scenario(
        name="Feature Drift — Quick Tune",
        drift_score=0.45,
        perf_delta=-0.02,
        dataset_size=80_000,
    )
    assert r2["drift"]["drift_type"] == "feature_drift"
    assert r2["strategy"] == "quick_tune"
    assert r2["config"] is not None
    assert r2["config"]["hyperparameter_search"]["enabled"] is True
    assert r2["config"]["hyperparameter_search"]["trials"] == 10
    print("  ✓ Assertions passed")

    # ── Case 3: Concept drift → full_search ──
    r3 = run_scenario(
        name="Concept Drift — Full Search",
        drift_score=0.8,
        perf_delta=-0.15,
        dataset_size=300_000,
    )
    assert r3["drift"]["drift_type"] == "concept_drift"
    assert r3["strategy"] == "full_search"
    assert r3["config"]["feature_selection"]["enabled"] is True
    assert r3["config"]["model_selection"]["enabled"] is True
    assert r3["config"]["ensemble"]["enabled"] is True
    assert r3["config"]["hyperparameter_search"]["trials"] == 25
    assert r3["resources"]["parallelism"] == 4
    print("  ✓ Assertions passed")

    # ── Case 4: Non-retrain action → no_action ──
    print(f"\n{'─' * 60}")
    print("  Scenario: Non-retrain action → no_action")
    print(f"{'─' * 60}")
    strategy = choose_strategy("rollback", {"drift_type": "concept_drift"})
    config = generate_training_config(strategy)
    assert strategy == "no_action"
    assert config is None
    print("  ✓ rollback → no_action → None config")

    # ── Case 5: Early stopping ──
    print(f"\n{'─' * 60}")
    print("  Scenario: Early Stopping Check")
    print(f"{'─' * 60}")

    # No improvement for 3 consecutive runs
    history_plateau = [
        {"f1_score": 0.85},
        {"f1_score": 0.855},
        {"f1_score": 0.856},
        {"f1_score": 0.857},
    ]
    stop = should_stop_early(history_plateau, patience=2)
    assert stop is True
    print("  ✓ Plateau detected → stop=True")

    # Improving history
    history_good = [
        {"f1_score": 0.80},
        {"f1_score": 0.85},
        {"f1_score": 0.90},
    ]
    stop2 = should_stop_early(history_good, patience=2)
    assert stop2 is False
    print("  ✓ Improving trend → stop=False")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("  DECISION CHAIN SUMMARY")
    print("=" * 60)
    summary = []
    for r in [r1, r2, r3]:
        summary.append({
            "scenario": r["scenario"],
            "drift_type": r["drift"]["drift_type"],
            "severity": r["drift"]["severity"],
            "strategy": r["strategy"],
            "enabled_modules": (
                [k for k, v in r["config"].items() if isinstance(v, dict) and v.get("enabled")]
                if r["config"] else []
            ),
            "resource_profile": r["resources"]["profile"],
        })
    print(json.dumps(summary, indent=2))

    print(f"\n✅ All Phase C tests passed!")


if __name__ == "__main__":
    main()
