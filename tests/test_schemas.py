"""Tests for the telemetry event schema and validation rules."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from app.schemas import (
    REQUIRED_FIELDS,
    EventLabel,
    SchemaValidationError,
    TelemetryEvent,
    format_utc_timestamp,
    parse_event_json,
)

VALID_EVENT_ID = "3f2b8c1e-9a4d-4e7b-8c2a-1d5e6f7a8b9c"


def valid_event_dict(**overrides: Any) -> dict[str, Any]:
    event: dict[str, Any] = {
        "event_id": VALID_EVENT_ID,
        "machine_id": "machine_01",
        "timestamp": "2026-11-24T10:30:00Z",
        "temperature_c": 45.5,
        "vibration_mm_s": 2.1,
        "pressure_bar": 5.0,
        "label": "normal",
    }
    event.update(overrides)
    return event


class TestValidEvents:
    def test_valid_event_is_created(self) -> None:
        event = TelemetryEvent.from_dict(valid_event_dict())
        assert event.machine_id == "machine_01"
        assert event.temperature_c == 45.5
        assert event.label is EventLabel.NORMAL

    def test_round_trip_through_json(self) -> None:
        original = TelemetryEvent.from_dict(valid_event_dict())
        restored = parse_event_json(original.to_json_bytes())
        assert restored == original

    def test_serialised_json_matches_documented_schema(self) -> None:
        payload = json.loads(TelemetryEvent.from_dict(valid_event_dict()).to_json_bytes())
        assert set(payload) == REQUIRED_FIELDS
        assert payload["label"] == "normal"

    def test_integer_sensor_values_are_accepted(self) -> None:
        event = TelemetryEvent.from_dict(valid_event_dict(pressure_bar=5))
        assert event.pressure_bar == 5

    def test_sensor_range_boundaries_are_inclusive(self) -> None:
        TelemetryEvent.from_dict(valid_event_dict(temperature_c=-40.0, vibration_mm_s=0.0))
        TelemetryEvent.from_dict(valid_event_dict(temperature_c=150.0, pressure_bar=20.0))

    def test_event_is_immutable(self) -> None:
        event = TelemetryEvent.from_dict(valid_event_dict())
        with pytest.raises(AttributeError):
            event.temperature_c = 99.0  # type: ignore[misc]


class TestRequiredFields:
    @pytest.mark.parametrize("missing_field", sorted(REQUIRED_FIELDS))
    def test_missing_field_is_rejected(self, missing_field: str) -> None:
        data = valid_event_dict()
        del data[missing_field]
        with pytest.raises(SchemaValidationError, match="missing required fields"):
            TelemetryEvent.from_dict(data)

    def test_unexpected_field_is_rejected(self) -> None:
        with pytest.raises(SchemaValidationError, match="unexpected fields"):
            TelemetryEvent.from_dict(valid_event_dict(humidity=40.0))

    @pytest.mark.parametrize("payload", [[], "text", 42, None])
    def test_non_object_is_rejected(self, payload: object) -> None:
        with pytest.raises(SchemaValidationError, match="JSON object"):
            TelemetryEvent.from_dict(payload)


class TestInvalidIdentifiers:
    @pytest.mark.parametrize("event_id", ["not-a-uuid", "", 123, None])
    def test_invalid_event_id_is_rejected(self, event_id: object) -> None:
        with pytest.raises(SchemaValidationError, match="event_id"):
            TelemetryEvent.from_dict(valid_event_dict(event_id=event_id))

    def test_non_canonical_uuid_is_rejected(self) -> None:
        with pytest.raises(SchemaValidationError, match="canonical"):
            TelemetryEvent.from_dict(valid_event_dict(event_id=VALID_EVENT_ID.upper()))

    @pytest.mark.parametrize("machine_id", ["machine_1", "Machine_01", "pump_01", "", 7])
    def test_invalid_machine_id_is_rejected(self, machine_id: object) -> None:
        with pytest.raises(SchemaValidationError, match="machine_id"):
            TelemetryEvent.from_dict(valid_event_dict(machine_id=machine_id))

    def test_invalid_label_is_rejected(self) -> None:
        with pytest.raises(SchemaValidationError, match="label"):
            TelemetryEvent.from_dict(valid_event_dict(label="anomaly"))


class TestTimestamps:
    @pytest.mark.parametrize(
        "timestamp",
        ["2026-11-24T10:30:00Z", "2026-11-24T10:30:00.123Z", "2026-11-24T10:30:00.123456Z"],
    )
    def test_valid_utc_timestamps_are_accepted(self, timestamp: str) -> None:
        assert (
            TelemetryEvent.from_dict(valid_event_dict(timestamp=timestamp)).timestamp == timestamp
        )

    @pytest.mark.parametrize(
        "timestamp",
        [
            "2026-11-24 10:30:00Z",  # missing 'T'
            "2026-11-24T10:30:00",  # no timezone
            "2026-11-24T10:30:00+05:30",  # not UTC 'Z'
            "24/11/2026",
            1732444200,
        ],
    )
    def test_malformed_timestamps_are_rejected(self, timestamp: object) -> None:
        with pytest.raises(SchemaValidationError, match="ISO 8601"):
            TelemetryEvent.from_dict(valid_event_dict(timestamp=timestamp))

    def test_impossible_date_is_rejected(self) -> None:
        with pytest.raises(SchemaValidationError, match="not a real date"):
            TelemetryEvent.from_dict(valid_event_dict(timestamp="2026-02-30T10:30:00Z"))

    def test_format_utc_timestamp_converts_to_utc(self) -> None:
        ist = timezone(timedelta(hours=5, minutes=30))
        moment = datetime(2026, 11, 24, 16, 0, 0, 123456, tzinfo=ist)
        assert format_utc_timestamp(moment) == "2026-11-24T10:30:00.123Z"

    def test_format_utc_timestamp_rejects_naive_datetime(self) -> None:
        with pytest.raises(SchemaValidationError, match="timezone-aware"):
            format_utc_timestamp(datetime(2026, 11, 24, 10, 30))  # noqa: DTZ001


class TestSensorValues:
    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("temperature_c", -40.1),
            ("temperature_c", 150.1),
            ("vibration_mm_s", -0.1),
            ("vibration_mm_s", 50.1),
            ("pressure_bar", -1.0),
            ("pressure_bar", 20.5),
        ],
    )
    def test_out_of_range_values_are_rejected(self, field_name: str, value: float) -> None:
        with pytest.raises(SchemaValidationError, match="outside plausible range"):
            TelemetryEvent.from_dict(valid_event_dict(**{field_name: value}))

    @pytest.mark.parametrize("value", ["45.5", None, True, [45.5]])
    def test_non_numeric_values_are_rejected(self, value: object) -> None:
        with pytest.raises(SchemaValidationError, match="must be a number"):
            TelemetryEvent.from_dict(valid_event_dict(temperature_c=value))

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_values_are_rejected(self, value: float) -> None:
        with pytest.raises(SchemaValidationError, match="finite"):
            TelemetryEvent.from_dict(valid_event_dict(vibration_mm_s=value))


class TestParseEventJson:
    @pytest.mark.parametrize("payload", [None, b"", ""])
    def test_empty_payload_is_rejected(self, payload: bytes | str | None) -> None:
        with pytest.raises(SchemaValidationError, match="empty"):
            parse_event_json(payload)

    def test_invalid_json_is_rejected(self) -> None:
        with pytest.raises(SchemaValidationError, match="not valid JSON"):
            parse_event_json(b"{not json")

    def test_invalid_utf8_is_rejected(self) -> None:
        with pytest.raises(SchemaValidationError, match="UTF-8"):
            parse_event_json(b"\xff\xfe\xfa")

    def test_valid_json_string_is_parsed(self) -> None:
        event = parse_event_json(json.dumps(valid_event_dict()))
        assert event.event_id == VALID_EVENT_ID
