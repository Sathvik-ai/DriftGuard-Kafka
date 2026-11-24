"""Simulated industrial machine telemetry (no physical IoT devices needed).

Each machine gets a stable operating profile (its own "normal" baseline) and
every reading adds small Gaussian noise around that baseline. Values are
clamped to the plausibility ranges defined in `app.schemas`, so every
generated event is valid by construction.

Reproducibility: `seed` controls machine profiles and sensor noise. Event IDs
always come from `uuid.uuid4()` so they stay unique across runs even when the
same seed is reused.
"""

from __future__ import annotations

import random
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from app.schemas import (
    PRESSURE_RANGE_BAR,
    TEMPERATURE_RANGE_C,
    VIBRATION_RANGE_MM_S,
    EventLabel,
    GroundTruthRecord,
    SensorRange,
    TelemetryEvent,
    format_utc_timestamp,
)

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def build_machine_ids(machine_count: int) -> list[str]:
    """Return zero-padded machine IDs, e.g. ``['machine_01', 'machine_02']``.

    Raises:
        ValueError: If `machine_count` is not positive.
    """
    if machine_count <= 0:
        raise ValueError(f"machine_count must be positive, got {machine_count}")
    width = max(2, len(str(machine_count)))
    return [f"machine_{index:0{width}d}" for index in range(1, machine_count + 1)]


class Scenario(str, Enum):
    """Telemetry generation scenarios supported in M1/M2."""

    NORMAL = "normal"
    ANOMALIES = "anomalies"
    DRIFT = "drift"


@dataclass(frozen=True)
class MachineProfile:
    """Normal operating baseline and noise level for one machine."""

    machine_id: str
    temperature_mean_c: float
    vibration_mean_mm_s: float
    pressure_mean_bar: float
    temperature_noise_c: float = 1.5
    vibration_noise_mm_s: float = 0.3
    pressure_noise_bar: float = 0.2


def _clamp(value: float, sensor_range: SensorRange) -> float:
    return min(max(value, sensor_range.minimum), sensor_range.maximum)


class TelemetrySimulator:
    """Generates validated `TelemetryEvent` objects for a fleet of machines."""

    def __init__(
        self,
        machine_count: int,
        seed: int | None = None,
        clock: Clock = utc_now,
        scenario: Scenario = Scenario.NORMAL,
        anomaly_rate: float = 0.12,
    ) -> None:
        """Create a simulator.

        Args:
            machine_count: Number of simulated machines in the fleet.
            seed: Optional seed for reproducible sensor values.
            clock: Function returning the current aware datetime (injectable for tests).
            scenario: Telemetry scenario (`normal`, `anomalies`, or `drift`).
            anomaly_rate: Injection probability for `Scenario.ANOMALIES`.
        """
        if not 0.0 <= anomaly_rate <= 1.0:
            raise ValueError(f"anomaly_rate must be in [0, 1], got {anomaly_rate}")
        # Non-cryptographic PRNG is intended: it drives simulated sensor noise
        # and must be seedable for reproducible experiments.
        self._rng = random.Random(seed)  # noqa: S311
        self._clock = clock
        self._scenario = scenario
        self._anomaly_rate = anomaly_rate
        self._profiles: list[MachineProfile] = [
            self._create_profile(machine_id) for machine_id in build_machine_ids(machine_count)
        ]
        # Per-machine frozen sensor state for stuck-sensor injection:
        # machine_id -> (remaining_stuck_count, (temp, vib, press))
        self._stuck_state: dict[str, tuple[int, tuple[float, float, float]]] = {}

    @property
    def machine_ids(self) -> list[str]:
        """IDs of all simulated machines."""
        return [profile.machine_id for profile in self._profiles]

    def _create_profile(self, machine_id: str) -> MachineProfile:
        return MachineProfile(
            machine_id=machine_id,
            temperature_mean_c=self._rng.uniform(40.0, 60.0),
            vibration_mean_mm_s=self._rng.uniform(1.5, 3.5),
            pressure_mean_bar=self._rng.uniform(4.0, 6.0),
        )

    def _sample(self, mean: float, noise: float, sensor_range: SensorRange) -> float:
        return round(_clamp(self._rng.gauss(mean, noise), sensor_range), 2)

    def _sample_normal_readings(self, profile: MachineProfile) -> tuple[float, float, float]:
        return (
            self._sample(
                profile.temperature_mean_c, profile.temperature_noise_c, TEMPERATURE_RANGE_C
            ),
            self._sample(
                profile.vibration_mean_mm_s, profile.vibration_noise_mm_s, VIBRATION_RANGE_MM_S
            ),
            self._sample(profile.pressure_mean_bar, profile.pressure_noise_bar, PRESSURE_RANGE_BAR),
        )

    def _apply_spike(self, temp: float, vib: float, press: float) -> tuple[float, float, float]:
        sensor_choice = self._rng.choice(("temperature", "vibration", "pressure"))
        if sensor_choice == "temperature":
            temp = round(_clamp(temp + self._rng.uniform(30.0, 50.0), TEMPERATURE_RANGE_C), 2)
        elif sensor_choice == "vibration":
            vib = round(_clamp(vib + self._rng.uniform(5.0, 10.0), VIBRATION_RANGE_MM_S), 2)
        else:
            press = round(_clamp(press + self._rng.uniform(3.5, 6.5), PRESSURE_RANGE_BAR), 2)
        return temp, vib, press

    def _build_event_and_truth(
        self,
        profile: MachineProfile,
        temp: float,
        vib: float,
        press: float,
        true_label: EventLabel,
    ) -> tuple[TelemetryEvent, GroundTruthRecord]:
        event_id = str(uuid.uuid4())
        # Wire event always carries placeholder NORMAL so ground truth is never
        # exposed to the Kafka detector consumer.
        event = TelemetryEvent(
            event_id=event_id,
            machine_id=profile.machine_id,
            timestamp=format_utc_timestamp(self._clock()),
            temperature_c=temp,
            vibration_mm_s=vib,
            pressure_bar=press,
            label=EventLabel.NORMAL,
        )
        truth = GroundTruthRecord(
            event_id=event_id,
            machine_id=profile.machine_id,
            scenario=self._scenario.value,
            ground_truth_label=true_label,
            is_anomaly=true_label is not EventLabel.NORMAL,
        )
        return event, truth

    def generate_event(self) -> TelemetryEvent:
        """Generate one normal reading from a randomly chosen machine."""
        profile = self._rng.choice(self._profiles)
        temp, vib, press = self._sample_normal_readings(profile)
        event, _ = self._build_event_and_truth(profile, temp, vib, press, EventLabel.NORMAL)
        return event

    def generate_events(self, count: int) -> Iterator[TelemetryEvent]:
        """Lazily yield `count` normal events (M1 API).

        Raises:
            ValueError: If `count` is negative.
        """
        if count < 0:
            raise ValueError(f"count must not be negative, got {count}")
        for _ in range(count):
            yield self.generate_event()

    def generate_scenario_events(
        self, count: int
    ) -> Iterator[tuple[TelemetryEvent, GroundTruthRecord]]:
        """Yield `(wire_event, ground_truth)` pairs for the configured scenario.

        Raises:
            ValueError: If `count` is negative.
        """
        if count < 0:
            raise ValueError(f"count must not be negative, got {count}")
        warmup = int(count * 0.3)
        drift_span = max(1, count - warmup)

        for index in range(count):
            profile = self._rng.choice(self._profiles)
            if self._scenario is Scenario.NORMAL:
                temp, vib, press = self._sample_normal_readings(profile)
                yield self._build_event_and_truth(profile, temp, vib, press, EventLabel.NORMAL)
            elif self._scenario is Scenario.ANOMALIES:
                yield self._next_anomaly_scenario_event(profile)
            else:
                progress = max(0.0, (index - warmup) / drift_span)
                temp, vib, press = self._sample_normal_readings(profile)
                temp = round(_clamp(temp + 32.0 * progress, TEMPERATURE_RANGE_C), 2)
                vib = round(_clamp(vib + 4.5 * progress, VIBRATION_RANGE_MM_S), 2)
                press = round(_clamp(press + 3.5 * progress, PRESSURE_RANGE_BAR), 2)
                # Gradual concept drift shifts the normal operating pattern;
                # ground truth remains NORMAL (not an injected fault/anomaly).
                yield self._build_event_and_truth(profile, temp, vib, press, EventLabel.NORMAL)

    def _next_anomaly_scenario_event(
        self, profile: MachineProfile
    ) -> tuple[TelemetryEvent, GroundTruthRecord]:
        stuck = self._stuck_state.get(profile.machine_id)
        if stuck is not None:
            remaining, frozen = stuck
            if remaining > 1:
                self._stuck_state[profile.machine_id] = (remaining - 1, frozen)
            else:
                del self._stuck_state[profile.machine_id]
            return self._build_event_and_truth(profile, *frozen, EventLabel.STUCK)

        temp, vib, press = self._sample_normal_readings(profile)
        if self._rng.random() < self._anomaly_rate:
            if self._rng.random() < 0.5:
                spiked = self._apply_spike(temp, vib, press)
                return self._build_event_and_truth(profile, *spiked, EventLabel.SPIKE)
            # Freeze this reading for the next 5 events of this machine.
            self._stuck_state[profile.machine_id] = (5, (temp, vib, press))
        return self._build_event_and_truth(profile, temp, vib, press, EventLabel.NORMAL)
