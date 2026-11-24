"""Tests for the telemetry simulator."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from app.schemas import (
    PRESSURE_RANGE_BAR,
    TEMPERATURE_RANGE_C,
    VIBRATION_RANGE_MM_S,
    EventLabel,
    TelemetryEvent,
    parse_event_json,
)
from app.simulator import TelemetrySimulator, build_machine_ids, utc_now

FIXED_TIME = datetime(2026, 11, 24, 10, 30, 0, tzinfo=timezone.utc)


def fixed_clock() -> datetime:
    return FIXED_TIME


class TestMachineIds:
    def test_ids_are_zero_padded(self) -> None:
        assert build_machine_ids(3) == ["machine_01", "machine_02", "machine_03"]

    def test_padding_grows_for_large_fleets(self) -> None:
        ids = build_machine_ids(120)
        assert ids[0] == "machine_001"
        assert ids[-1] == "machine_120"

    @pytest.mark.parametrize("count", [0, -1])
    def test_non_positive_count_is_rejected(self, count: int) -> None:
        with pytest.raises(ValueError, match="positive"):
            build_machine_ids(count)


class TestEventGeneration:
    def test_generated_event_is_valid(self) -> None:
        event = TelemetrySimulator(machine_count=3, seed=1).generate_event()
        assert isinstance(event, TelemetryEvent)
        assert event.label is EventLabel.NORMAL

    def test_generated_event_survives_json_round_trip(self) -> None:
        event = TelemetrySimulator(machine_count=3, seed=1).generate_event()
        assert parse_event_json(event.to_json_bytes()) == event

    def test_event_ids_are_unique_uuids(self) -> None:
        events = list(TelemetrySimulator(machine_count=5, seed=7).generate_events(2000))
        event_ids = [event.event_id for event in events]
        assert len(set(event_ids)) == len(event_ids)
        assert all(uuid.UUID(event_id).version == 4 for event_id in event_ids)

    def test_event_ids_differ_across_runs_with_same_seed(self) -> None:
        first = TelemetrySimulator(machine_count=2, seed=42).generate_event()
        second = TelemetrySimulator(machine_count=2, seed=42).generate_event()
        assert first.event_id != second.event_id

    def test_same_seed_reproduces_sensor_values(self) -> None:
        def readings(seed: int) -> list[tuple[str, float, float, float]]:
            simulator = TelemetrySimulator(machine_count=4, seed=seed, clock=fixed_clock)
            return [
                (e.machine_id, e.temperature_c, e.vibration_mm_s, e.pressure_bar)
                for e in simulator.generate_events(50)
            ]

        assert readings(123) == readings(123)
        assert readings(123) != readings(456)

    def test_values_stay_within_schema_ranges(self) -> None:
        for event in TelemetrySimulator(machine_count=10, seed=3).generate_events(5000):
            assert TEMPERATURE_RANGE_C.contains(event.temperature_c)
            assert VIBRATION_RANGE_MM_S.contains(event.vibration_mm_s)
            assert PRESSURE_RANGE_BAR.contains(event.pressure_bar)

    def test_values_are_realistic_for_normal_operation(self) -> None:
        events = list(TelemetrySimulator(machine_count=5, seed=9).generate_events(1000))
        mean_temperature = sum(e.temperature_c for e in events) / len(events)
        assert 35.0 < mean_temperature < 65.0
        assert all(e.vibration_mm_s < 6.0 for e in events)

    def test_all_machines_are_used(self) -> None:
        simulator = TelemetrySimulator(machine_count=5, seed=11)
        used = {event.machine_id for event in simulator.generate_events(500)}
        assert used == set(simulator.machine_ids)

    def test_timestamp_comes_from_injected_clock(self) -> None:
        event = TelemetrySimulator(machine_count=1, clock=fixed_clock).generate_event()
        assert event.timestamp == "2026-11-24T10:30:00.000Z"

    def test_generate_events_yields_exact_count(self) -> None:
        simulator = TelemetrySimulator(machine_count=2, seed=0)
        assert len(list(simulator.generate_events(25))) == 25
        assert list(simulator.generate_events(0)) == []

    def test_negative_count_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            list(TelemetrySimulator(machine_count=1).generate_events(-1))


def test_utc_now_is_timezone_aware() -> None:
    assert utc_now().tzinfo is timezone.utc
