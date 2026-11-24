"""Tests for configuration loading, CLI validators and logging setup."""

from __future__ import annotations

import argparse
import logging

import pytest

from app.config import (
    DEFAULT_BOOTSTRAP_SERVERS,
    DEFAULT_TOPIC,
    ConfigError,
    load_settings,
    non_negative_float,
    positive_float,
    positive_int,
)
from app.logging_config import KeyValueFormatter, configure_logging


class TestLoadSettings:
    def test_defaults_are_used_when_env_is_empty(self) -> None:
        settings = load_settings({})
        assert settings.bootstrap_servers == DEFAULT_BOOTSTRAP_SERVERS
        assert settings.topic == DEFAULT_TOPIC
        assert settings.machine_count == 5
        assert settings.log_level == "INFO"

    def test_environment_values_override_defaults(self) -> None:
        settings = load_settings(
            {
                "KAFKA_BOOTSTRAP_SERVERS": "broker:29092",
                "KAFKA_TOPIC": "other.topic",
                "KAFKA_CONSUMER_GROUP": "group-a",
                "KAFKA_CLIENT_ID": "client-a",
                "SIMULATOR_MACHINE_COUNT": "12",
                "LOG_LEVEL": "debug",
            }
        )
        assert settings.bootstrap_servers == "broker:29092"
        assert settings.topic == "other.topic"
        assert settings.consumer_group == "group-a"
        assert settings.client_id == "client-a"
        assert settings.machine_count == 12
        assert settings.log_level == "DEBUG"

    @pytest.mark.parametrize("value", ["abc", "0", "-3", "1.5"])
    def test_invalid_machine_count_is_rejected(self, value: str) -> None:
        with pytest.raises(ConfigError, match="SIMULATOR_MACHINE_COUNT"):
            load_settings({"SIMULATOR_MACHINE_COUNT": value})

    def test_empty_string_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="KAFKA_TOPIC"):
            load_settings({"KAFKA_TOPIC": "   "})

    def test_unknown_log_level_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="LOG_LEVEL"):
            load_settings({"LOG_LEVEL": "VERBOSE"})


class TestCliValidators:
    def test_valid_values_are_parsed(self) -> None:
        assert positive_int("5") == 5
        assert positive_float("0.5") == 0.5
        assert non_negative_float("0") == 0.0

    @pytest.mark.parametrize(
        ("validator", "text"),
        [(positive_int, "0"), (positive_float, "0"), (non_negative_float, "-1")],
    )
    def test_out_of_range_values_are_rejected(self, validator: object, text: str) -> None:
        assert callable(validator)
        with pytest.raises(argparse.ArgumentTypeError):
            validator(text)


class TestLogging:
    def test_structured_fields_are_rendered_as_key_value(self) -> None:
        formatter = KeyValueFormatter(fmt="%(message)s")
        record = logging.LogRecord("test", logging.INFO, __file__, 1, "summary", None, None)
        record.fields = {"delivered": 1000, "topic": "a b"}
        assert formatter.format(record) == 'summary delivered=1000 topic="a b"'

    def test_record_without_fields_is_unchanged(self) -> None:
        formatter = KeyValueFormatter(fmt="%(message)s")
        record = logging.LogRecord("test", logging.INFO, __file__, 1, "plain", None, None)
        assert formatter.format(record) == "plain"

    def test_configure_logging_does_not_duplicate_handlers(self) -> None:
        configure_logging("DEBUG")
        configure_logging("INFO")
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 1
        assert root_logger.level == logging.INFO
