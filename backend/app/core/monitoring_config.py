"""Configurable thresholds for the monitoring worker."""

# Drift detection
DRIFT_THRESHOLD = 0.2          # Drift score above this triggers an alert
DRIFT_CRITICAL = 0.5           # Drift score above this triggers a critical alert

# Performance degradation
PERFORMANCE_DROP_THRESHOLD = 0.1   # Absolute drop in score that triggers alert

# System metrics
LATENCY_THRESHOLD_MS = 500     # Average latency above this is flagged

# Scheduler
MONITORING_INTERVAL_MINUTES = 5

# Inference window
INFERENCE_WINDOW_MINUTES = 5   # Look at inferences from the last N minutes
