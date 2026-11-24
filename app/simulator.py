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

from app.schemas import (
    PRESSURE_RANGE_BAR,
    TEMPERATURE_RANGE_C,
    VIBRATION_RANGE_MM_S,
    EventLabel,
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
    ) -> None:
        """Create a simulator.

        Args:
            machine_count: Number of simulated machines in the fleet.
            seed: Optional seed for reproducible sensor values.
            clock: Function returning the current aware datetime (injectable for tests).
        """
        # Non-cryptographic PRNG is intended: it drives simulated sensor noise
        # and must be seedable for reproducible experiments.
        self._rng = random.Random(seed)  # noqa: S311
        self._clock = clock
        self._profiles: list[MachineProfile] = [
            self._create_profile(machine_id) for machine_id in build_machine_ids(machine_count)
        ]

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

    def generate_event(self) -> TelemetryEvent:
        """Generate one normal reading from a randomly chosen machine."""
        profile = self._rng.choice(self._profiles)
        return TelemetryEvent(
            event_id=str(uuid.uuid4()),
            machine_id=profile.machine_id,
            timestamp=format_utc_timestamp(self._clock()),
            temperature_c=self._sample(
                profile.temperature_mean_c, profile.temperature_noise_c, TEMPERATURE_RANGE_C
            ),
            vibration_mm_s=self._sample(
                profile.vibration_mean_mm_s, profile.vibration_noise_mm_s, VIBRATION_RANGE_MM_S
            ),
            pressure_bar=self._sample(
                profile.pressure_mean_bar, profile.pressure_noise_bar, PRESSURE_RANGE_BAR
            ),
            label=EventLabel.NORMAL,
        )

    def generate_events(self, count: int) -> Iterator[TelemetryEvent]:
        """Lazily yield `count` events.

        Raises:
            ValueError: If `count` is negative.
        """
        if count < 0:
            raise ValueError(f"count must not be negative, got {count}")
        for _ in range(count):
            yield self.generate_event()
