"""Telemetry event schema: the data contract between producer and consumer.

A `TelemetryEvent` validates itself on construction, so an invalid event can
never exist in memory. The consumer uses `parse_event_json` to turn raw Kafka
bytes into a validated event or a `SchemaValidationError`.

Event generation lives in `app.simulator`; this module only defines and
validates the structure.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

MACHINE_ID_PATTERN = re.compile(r"^machine_\d{2,}$")
DEVICE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
# ISO 8601 UTC with mandatory "Z" suffix and optional fractional seconds.
TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$")


class SchemaValidationError(ValueError):
    """Raised when data does not satisfy the telemetry event schema."""


class SourceType(str, Enum):
    """Origin of a telemetry event on `research.telemetry.raw`.

    Distinguishes online virtual IoT streams (M10-IoT) from the local Python
    synthetic simulator (M1–M3) and offline benchmark replays.
    """

    SIMULATOR = "simulator"
    VIRTUAL_IOT_WOKWI = "virtual_iot_wokwi"
    BENCHMARK_REPLAY = "benchmark_replay"


class EventLabel(str, Enum):
    """Ground-truth label for telemetry events.

    M1 only produces `NORMAL`. M2 introduces synthetic `SPIKE` and `STUCK`
    fault labels (stored separately from the detector). Gradual operating-regime
    drift retains `NORMAL` ground truth because drift is a shift in normal
    behaviour rather than a point/sensor fault.
    """

    NORMAL = "normal"
    SPIKE = "spike"
    STUCK = "stuck"


@dataclass(frozen=True)
class SensorRange:
    """Inclusive physical plausibility bounds for one sensor reading."""

    minimum: float
    maximum: float

    def contains(self, value: float) -> bool:
        """Return True when `value` lies within the inclusive bounds."""
        return self.minimum <= value <= self.maximum


# Plausibility bounds for an industrial machine; the simulator stays inside them.
TEMPERATURE_RANGE_C = SensorRange(minimum=-40.0, maximum=150.0)
VIBRATION_RANGE_MM_S = SensorRange(minimum=0.0, maximum=50.0)
PRESSURE_RANGE_BAR = SensorRange(minimum=0.0, maximum=20.0)

_SENSOR_RANGES: dict[str, SensorRange] = {
    "temperature_c": TEMPERATURE_RANGE_C,
    "vibration_mm_s": VIBRATION_RANGE_MM_S,
    "pressure_bar": PRESSURE_RANGE_BAR,
}

REQUIRED_FIELDS: frozenset[str] = frozenset(
    {"event_id", "machine_id", "timestamp", *_SENSOR_RANGES, "label"}
)
OPTIONAL_FIELDS: frozenset[str] = frozenset({"source_type", "device_id", "sequence_number"})
ALLOWED_FIELDS: frozenset[str] = REQUIRED_FIELDS | OPTIONAL_FIELDS


def format_utc_timestamp(moment: datetime) -> str:
    """Format an aware datetime as ISO 8601 UTC with millisecond precision.

    Example: ``2026-11-24T10:30:00.123Z``.

    Raises:
        SchemaValidationError: If `moment` has no timezone information.
    """
    if moment.tzinfo is None:
        raise SchemaValidationError("timestamp datetime must be timezone-aware")
    utc_moment = moment.astimezone(timezone.utc)
    return utc_moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{utc_moment.microsecond // 1000:03d}Z"


def _validate_event_id(event_id: object) -> None:
    if not isinstance(event_id, str):
        raise SchemaValidationError("event_id must be a string")
    try:
        canonical = str(uuid.UUID(event_id))
    except ValueError as error:
        raise SchemaValidationError(f"event_id is not a valid UUID: {event_id!r}") from error
    if canonical != event_id:
        raise SchemaValidationError(
            f"event_id must be a canonical lowercase UUID, got {event_id!r}"
        )


def _validate_machine_id(machine_id: object) -> None:
    if not isinstance(machine_id, str) or not MACHINE_ID_PATTERN.match(machine_id):
        raise SchemaValidationError(f"machine_id must look like 'machine_01', got {machine_id!r}")


def _validate_timestamp(timestamp: object) -> None:
    if not isinstance(timestamp, str) or not TIMESTAMP_PATTERN.match(timestamp):
        raise SchemaValidationError(
            f"timestamp must be ISO 8601 UTC ending in 'Z', got {timestamp!r}"
        )
    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise SchemaValidationError(f"timestamp is not a real date: {timestamp!r}") from error


def _validate_sensor_value(field_name: str, value: object) -> None:
    # bool is a subclass of int in Python, so reject it explicitly.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{field_name} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise SchemaValidationError(f"{field_name} must be finite, got {value!r}")
    sensor_range = _SENSOR_RANGES[field_name]
    if not sensor_range.contains(float(value)):
        raise SchemaValidationError(
            f"{field_name}={value} outside plausible range "
            f"[{sensor_range.minimum}, {sensor_range.maximum}]"
        )


def _validate_label(label: object) -> None:
    allowed = [member.value for member in EventLabel]
    if not isinstance(label, EventLabel) and label not in allowed:
        raise SchemaValidationError(f"label must be one of {allowed}, got {label!r}")


def _validate_source_type(source_type: object) -> None:
    allowed = [member.value for member in SourceType]
    if not isinstance(source_type, SourceType) and source_type not in allowed:
        raise SchemaValidationError(f"source_type must be one of {allowed}, got {source_type!r}")


def _validate_device_id(device_id: object) -> None:
    if device_id is None:
        return
    if not isinstance(device_id, str) or not DEVICE_ID_PATTERN.match(device_id):
        raise SchemaValidationError(
            f"device_id must be 1-64 alphanumeric/._- chars, got {device_id!r}"
        )


def _validate_sequence_number(sequence_number: object) -> None:
    if sequence_number is None:
        return
    if isinstance(sequence_number, bool) or not isinstance(sequence_number, int):
        raise SchemaValidationError(f"sequence_number must be an integer, got {sequence_number!r}")
    if sequence_number < 0:
        raise SchemaValidationError(f"sequence_number must be non-negative, got {sequence_number}")


@dataclass(frozen=True)
class TelemetryEvent:
    """One sensor reading from one simulated or virtual IoT machine.

    Construction validates every field and raises `SchemaValidationError`
    on the first problem found.
    """

    event_id: str
    machine_id: str
    timestamp: str
    temperature_c: float
    vibration_mm_s: float
    pressure_bar: float
    label: EventLabel = EventLabel.NORMAL
    source_type: SourceType = SourceType.SIMULATOR
    device_id: str | None = None
    sequence_number: int | None = None

    def __post_init__(self) -> None:
        """Validate all fields; called automatically by the dataclass constructor."""
        _validate_event_id(self.event_id)
        _validate_machine_id(self.machine_id)
        _validate_timestamp(self.timestamp)
        for field_name in _SENSOR_RANGES:
            _validate_sensor_value(field_name, getattr(self, field_name))
        _validate_label(self.label)
        _validate_source_type(self.source_type)
        _validate_device_id(self.device_id)
        _validate_sequence_number(self.sequence_number)
        # Normalise plain strings into enums for consistent types.
        object.__setattr__(self, "label", EventLabel(self.label))
        object.__setattr__(self, "source_type", SourceType(self.source_type))

    @classmethod
    def from_dict(cls, data: object) -> TelemetryEvent:
        """Build a validated event from a decoded JSON object.

        Unknown fields outside `ALLOWED_FIELDS` are rejected so schema drift
        between producer and consumer is detected immediately.
        """
        if not isinstance(data, dict):
            raise SchemaValidationError(f"event must be a JSON object, got {type(data).__name__}")
        missing = REQUIRED_FIELDS - data.keys()
        if missing:
            raise SchemaValidationError(f"missing required fields: {sorted(missing)}")
        unexpected = data.keys() - ALLOWED_FIELDS
        if unexpected:
            raise SchemaValidationError(f"unexpected fields: {sorted(unexpected)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        payload: dict[str, Any] = {
            "event_id": self.event_id,
            "machine_id": self.machine_id,
            "timestamp": self.timestamp,
            "temperature_c": self.temperature_c,
            "vibration_mm_s": self.vibration_mm_s,
            "pressure_bar": self.pressure_bar,
            "label": self.label.value,
        }
        if self.source_type is not SourceType.SIMULATOR:
            payload["source_type"] = self.source_type.value
        if self.device_id is not None:
            payload["device_id"] = self.device_id
        if self.sequence_number is not None:
            payload["sequence_number"] = self.sequence_number
        return payload

    def to_json_bytes(self) -> bytes:
        """Serialise the event as compact UTF-8 JSON for Kafka."""
        return json.dumps(self.to_dict(), separators=(",", ":")).encode("utf-8")


def parse_event_json(payload: bytes | str | None) -> TelemetryEvent:
    """Decode and validate a raw Kafka message value.

    Raises:
        SchemaValidationError: For empty payloads, invalid UTF-8, invalid
            JSON or any schema violation.
    """
    if payload is None or len(payload) == 0:
        raise SchemaValidationError("message value is empty")
    try:
        decoded: Any = json.loads(payload)
    except UnicodeDecodeError as error:
        raise SchemaValidationError(f"message value is not valid UTF-8: {error}") from error
    except json.JSONDecodeError as error:
        raise SchemaValidationError(f"message value is not valid JSON: {error}") from error
    return TelemetryEvent.from_dict(decoded)


@dataclass(frozen=True)
class GroundTruthRecord:
    """Ground-truth annotation stored separately from the Kafka detector stream."""

    event_id: str
    machine_id: str
    scenario: str
    ground_truth_label: EventLabel
    is_anomaly: bool
    stream_index: int = 0
    drift_active: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary for ground-truth JSONL logs."""
        return {
            "event_id": self.event_id,
            "machine_id": self.machine_id,
            "scenario": self.scenario,
            "ground_truth_label": self.ground_truth_label.value,
            "is_anomaly": self.is_anomaly,
            "stream_index": self.stream_index,
            "drift_active": self.drift_active,
        }
