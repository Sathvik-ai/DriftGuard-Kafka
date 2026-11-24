"""Application settings loaded from environment variables.

All tunable values live here so that the producer, consumer and scripts share
one source of truth. Defaults match `.env.example` and the Docker Compose
broker, so the demo works without creating a `.env` file.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_BOOTSTRAP_SERVERS = "localhost:9092"
DEFAULT_TOPIC = "research.telemetry.raw"
DEFAULT_ALERTS_TOPIC = "research.telemetry.alerts"
DEFAULT_DRIFT_TOPIC = "research.telemetry.drift"
DEFAULT_CONSUMER_GROUP = "driftguard-m1-consumer"
DEFAULT_CLIENT_ID = "driftguard-m1"
DEFAULT_MACHINE_COUNT = 5
DEFAULT_LOG_LEVEL = "INFO"

_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class ConfigError(ValueError):
    """Raised when an environment variable holds an invalid value."""


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration shared by producer and consumer."""

    bootstrap_servers: str
    topic: str
    consumer_group: str
    client_id: str
    machine_count: int
    log_level: str
    alerts_topic: str = DEFAULT_ALERTS_TOPIC
    drift_topic: str = DEFAULT_DRIFT_TOPIC


def _read_str(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name, default).strip()
    if not value:
        raise ConfigError(f"{name} must not be empty")
    return value


def _read_positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw_value = env.get(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ConfigError(f"{name} must be an integer, got {raw_value!r}") from error
    if value <= 0:
        raise ConfigError(f"{name} must be greater than zero, got {value}")
    return value


def _read_log_level(env: Mapping[str, str]) -> str:
    level = _read_str(env, "LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    if level not in _VALID_LOG_LEVELS:
        raise ConfigError(f"LOG_LEVEL must be one of {sorted(_VALID_LOG_LEVELS)}, got {level!r}")
    return level


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build `Settings` from environment variables.

    Args:
        env: Mapping to read from. Defaults to `os.environ`; tests pass a dict.

    Raises:
        ConfigError: If any variable is present but invalid.
    """
    source: Mapping[str, str] = os.environ if env is None else env
    return Settings(
        bootstrap_servers=_read_str(source, "KAFKA_BOOTSTRAP_SERVERS", DEFAULT_BOOTSTRAP_SERVERS),
        topic=_read_str(source, "KAFKA_TOPIC", DEFAULT_TOPIC),
        consumer_group=_read_str(source, "KAFKA_CONSUMER_GROUP", DEFAULT_CONSUMER_GROUP),
        client_id=_read_str(source, "KAFKA_CLIENT_ID", DEFAULT_CLIENT_ID),
        machine_count=_read_positive_int(source, "SIMULATOR_MACHINE_COUNT", DEFAULT_MACHINE_COUNT),
        log_level=_read_log_level(source),
        alerts_topic=_read_str(source, "KAFKA_ALERTS_TOPIC", DEFAULT_ALERTS_TOPIC),
        drift_topic=_read_str(source, "KAFKA_DRIFT_TOPIC", DEFAULT_DRIFT_TOPIC),
    )


def positive_int(text: str) -> int:
    """Parse an integer greater than zero (argparse `type=`)."""
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
    return value


def positive_float(text: str) -> float:
    """Parse a number greater than zero (argparse `type=`)."""
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
    return value


def non_negative_float(text: str) -> float:
    """Parse a number greater than or equal to zero (argparse `type=`)."""
    value = float(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value
