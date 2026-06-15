"""
Structured JSON logging — every log line becomes machine-parseable.
"""

import logging
import json
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


def configure_logging() -> None:
    """Replace default log handlers with JSON formatter across all loggers."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    # Silence noisy SQLAlchemy echo (we set echo=True on engine)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
