"""Structured (key=value) logging setup shared by all entry points.

Usage:
    logger.info("delivery summary", extra={"fields": {"delivered": 1000}})

renders as:
    2026-11-24T10:30:00Z INFO app.producer delivery summary delivered=1000
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _format_value(value: Any) -> str:
    text = str(value)
    return f'"{text}"' if " " in text or not text else text


def _utc_time(timestamp: float | None) -> time.struct_time:
    return time.gmtime(timestamp)


class KeyValueFormatter(logging.Formatter):
    """Formatter that appends `extra={"fields": {...}}` as `key=value` pairs."""

    def __init__(self, fmt: str | None = None, datefmt: str | None = None) -> None:
        """Create a formatter that emits UTC timestamps, matching event timestamps."""
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.converter = _utc_time

    def format(self, record: logging.LogRecord) -> str:
        """Format the record and append any structured fields."""
        base = super().format(record)
        fields: dict[str, Any] = getattr(record, "fields", {}) or {}
        if not fields:
            return base
        pairs = " ".join(f"{key}={_format_value(value)}" for key, value in fields.items())
        return f"{base} {pairs}"


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger to write structured lines to stderr.

    Safe to call more than once: existing handlers are replaced, not duplicated.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(KeyValueFormatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level.upper())


def log_kafka_client_error(error: Any) -> None:
    """Log librdkafka client-level errors (`error_cb`), e.g. an unreachable broker.

    librdkafka retries these internally, so they are logged as warnings; the
    affected messages are still reported through delivery callbacks or
    per-message errors.
    """
    logging.getLogger("app.kafka").warning(
        "kafka client error", extra={"fields": {"error": error.str()}}
    )
