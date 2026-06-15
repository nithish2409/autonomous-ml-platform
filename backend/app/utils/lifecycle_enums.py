"""String enums for model lifecycle states and action types."""

from enum import Enum


class ModelStatus(str, Enum):
    """Lifecycle states for a model in the registry."""

    TRAINING = "training"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"

    # Legacy compatibility with existing code that uses "active"/"inactive"
    ACTIVE = "active"
    INACTIVE = "inactive"


class LifecycleAction(str, Enum):
    """Actions logged to automation_logs for lifecycle events."""

    MODEL_REGISTERED = "MODEL_REGISTERED"
    VERSION_ADDED = "VERSION_ADDED"
    PROMOTED = "PROMOTED"
    ROLLBACK = "ROLLBACK"
    ARCHIVED = "ARCHIVED"
