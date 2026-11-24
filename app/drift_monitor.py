"""ADWIN-based concept drift monitor for DriftGuard-Kafka M3.

Consumes raw telemetry from `research.telemetry.raw`, strips any placeholder
`label` via `SensorReading.from_telemetry` so ground truth never reaches ADWIN,
maintains independent `river.drift.ADWIN` monitors per `(machine_id, feature)`,
and publishes structured drift events to `research.telemetry.drift`.

Static anomaly thresholds (`app.detector`) remain unchanged; no adaptive model
retuning or threshold updates are performed in M3.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Any

from confluent_kafka import Consumer, KafkaException, Message, Producer, TopicPartition
from river.drift import ADWIN

from app.config import Settings, load_settings, positive_float, positive_int
from app.consumer import build_consumer_config
from app.detector import SensorReading, ThresholdDetector
from app.logging_config import configure_logging
from app.producer import DeliveryTracker, build_producer_config
from app.run_summary import write_summary
from app.schemas import (
    SchemaValidationError,
    format_utc_timestamp,
    parse_event_json,
)

logger = logging.getLogger(__name__)

_POLL_TIMEOUT_S = 1.0
MONITORED_FEATURES: tuple[str, str, str] = (
    "temperature_c",
    "vibration_mm_s",
    "pressure_bar",
)


@dataclass(frozen=True)
class AdwinConfig:
    """Hyperparameters for per-(machine, feature) River ADWIN monitors."""

    delta: float = 0.002
    clock: int = 1
    max_buckets: int = 5
    min_window_length: int = 5
    grace_period: int = 10

    def __post_init__(self) -> None:
        """Validate ADWIN configuration parameters."""
        if not 0.0 < self.delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {self.delta}")
        if self.clock <= 0:
            raise ValueError(f"clock must be > 0, got {self.clock}")
        if self.min_window_length <= 0:
            raise ValueError(f"min_window_length must be > 0, got {self.min_window_length}")
        if self.grace_period <= 0:
            raise ValueError(f"grace_period must be > 0, got {self.grace_period}")


@dataclass(frozen=True)
class DriftEvent:
    """Structured drift detection event emitted when ADWIN detects a shift."""

    drift_id: str
    event_id: str
    machine_id: str
    feature: str
    feature_value: float
    adwin_estimation: float
    adwin_width: int
    delta: float
    event_timestamp: str
    detected_utc: str
    stream_position: int
    machine_position: int
    detector: str = "river_adwin_v1"

    def to_dict(self) -> dict[str, Any]:
        """Serialise drift event to a JSON-compatible dictionary."""
        return {
            "drift_id": self.drift_id,
            "event_id": self.event_id,
            "machine_id": self.machine_id,
            "feature": self.feature,
            "feature_value": self.feature_value,
            "adwin_estimation": self.adwin_estimation,
            "adwin_width": self.adwin_width,
            "delta": self.delta,
            "event_timestamp": self.event_timestamp,
            "detected_utc": self.detected_utc,
            "stream_position": self.stream_position,
            "machine_position": self.machine_position,
            "detector": self.detector,
        }


class MachineAdwinMonitor:
    """Maintains independent River ADWIN instances per `(machine_id, feature)`."""

    def __init__(self, config: AdwinConfig | None = None) -> None:
        """Initialise empty monitor registry with `config`."""
        self._config = config or AdwinConfig()
        self._detectors: dict[tuple[str, str], ADWIN] = {}
        self._machine_counts: Counter[str] = Counter()
        self._stream_count: int = 0

    @property
    def config(self) -> AdwinConfig:
        """Active ADWIN configuration."""
        return self._config

    @property
    def monitor_keys(self) -> list[tuple[str, str]]:
        """Sorted `(machine_id, feature)` keys currently instantiated."""
        return sorted(self._detectors)

    def _get_detector(self, machine_id: str, feature: str) -> ADWIN:
        key = (machine_id, feature)
        detector = self._detectors.get(key)
        if detector is None:
            detector = ADWIN(
                delta=self._config.delta,
                clock=self._config.clock,
                max_buckets=self._config.max_buckets,
                min_window_length=self._config.min_window_length,
                grace_period=self._config.grace_period,
            )
            self._detectors[key] = detector
        return detector

    def update(self, reading: SensorReading) -> tuple[int, int, list[DriftEvent]]:
        """Feed one unlabelled `SensorReading` into its 3 per-machine ADWIN monitors.

        Returns:
            `(stream_position, machine_position, triggered_drift_events)`
            Positions are 0-based sequential indices.
        """
        stream_pos = self._stream_count
        self._stream_count += 1
        machine_pos = self._machine_counts[reading.machine_id]
        self._machine_counts[reading.machine_id] += 1

        detected_events: list[DriftEvent] = []
        detected_utc = format_utc_timestamp(datetime.now(timezone.utc))

        for feature in MONITORED_FEATURES:
            value = float(getattr(reading, feature))
            detector = self._get_detector(reading.machine_id, feature)
            detector.update(value)
            if detector.drift_detected:
                detected_events.append(
                    DriftEvent(
                        drift_id=str(uuid.uuid4()),
                        event_id=reading.event_id,
                        machine_id=reading.machine_id,
                        feature=feature,
                        feature_value=value,
                        adwin_estimation=round(float(detector.estimation), 4),
                        adwin_width=int(detector.width),
                        delta=self._config.delta,
                        event_timestamp=reading.timestamp,
                        detected_utc=detected_utc,
                        stream_position=stream_pos,
                        machine_position=machine_pos,
                    )
                )
        return stream_pos, machine_pos, detected_events


@dataclass
class DriftMonitorStats:
    """Runtime counters for an ADWIN drift monitoring run."""

    received: int = 0
    valid: int = 0
    malformed: int = 0
    kafka_errors: int = 0
    drift_events_triggered: int = 0
    static_alerts_observed: int = 0
    by_feature: Counter[str] = field(default_factory=Counter)
    by_machine: Counter[str] = field(default_factory=Counter)

    def to_dict(
        self,
        drift_tracker: DeliveryTracker,
        duration_s: float,
        stop_reason: str,
        delta: float,
    ) -> dict[str, Any]:
        """Return JSON-serialisable summary of the drift monitor run."""
        return {
            "received": self.received,
            "valid": self.valid,
            "malformed": self.malformed,
            "kafka_errors": self.kafka_errors,
            "drift_events_triggered": self.drift_events_triggered,
            "drift_events_delivered": drift_tracker.delivered,
            "drift_events_failed": drift_tracker.failed,
            "static_alerts_observed": self.static_alerts_observed,
            "by_feature": dict(self.by_feature),
            "by_machine": dict(self.by_machine),
            "adwin_delta": delta,
            "duration_s": duration_s,
            "stop_reason": stop_reason,
        }


def _publish_drift_event(
    producer: Producer,
    drift_topic: str,
    drift_event: DriftEvent,
    tracker: DeliveryTracker,
) -> None:
    payload = json.dumps(drift_event.to_dict(), separators=(",", ":")).encode("utf-8")
    while True:
        try:
            producer.produce(
                topic=drift_topic,
                key=drift_event.machine_id.encode("utf-8"),
                value=payload,
                on_delivery=tracker.on_delivery,
            )
            producer.poll(0)
            return
        except BufferError:
            producer.poll(0.2)


def process_drift_stream(
    consumer: Consumer,
    producer: Producer,
    drift_topic: str,
    monitor: MachineAdwinMonitor,
    static_detector: ThresholdDetector,
    stats: DriftMonitorStats,
    drift_tracker: DeliveryTracker,
    stop_event: threading.Event,
    limit: int | None,
    idle_timeout_s: float | None,
    on_drift: Callable[[dict[str, Any]], None] | None = None,
    on_observation: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """Poll telemetry, update per-(machine, feature) ADWIN monitors, and emit drift events."""
    last_activity = time.monotonic()
    while not stop_event.is_set():
        if limit is not None and stats.received >= limit:
            return "limit_reached"
        message: Message | None = consumer.poll(_POLL_TIMEOUT_S)
        if message is None:
            if idle_timeout_s is not None and time.monotonic() - last_activity >= idle_timeout_s:
                return "idle_timeout"
            continue
        error = message.error()
        if error is not None:
            stats.kafka_errors += 1
            if error.fatal():
                raise KafkaException(error)
            logger.error("kafka message error", extra={"fields": {"error": error.str()}})
            continue

        observed_mono_ns = time.monotonic_ns()
        last_activity = time.monotonic()
        stats.received += 1
        try:
            event = parse_event_json(message.value())
        except SchemaValidationError as schema_err:
            stats.malformed += 1
            logger.warning("malformed telemetry", extra={"fields": {"reason": str(schema_err)}})
            consumer.store_offsets(message=message)
            continue

        stats.valid += 1
        reading = SensorReading.from_telemetry(event)
        # Static detector runs side-by-side UNCHANGED (no threshold adaptation in M3)
        static_res = static_detector.evaluate(reading)
        if static_res.is_anomaly:
            stats.static_alerts_observed += 1

        stream_pos, machine_pos, drift_events = monitor.update(reading)
        for drift_ev in drift_events:
            stats.drift_events_triggered += 1
            stats.by_feature[drift_ev.feature] += 1
            stats.by_machine[drift_ev.machine_id] += 1
            _publish_drift_event(producer, drift_topic, drift_ev, drift_tracker)
            if on_drift is not None:
                rec = drift_ev.to_dict()
                rec["detected_monotonic_ns"] = observed_mono_ns
                on_drift(rec)

        if on_observation is not None:
            on_observation(
                {
                    "event_id": reading.event_id,
                    "machine_id": reading.machine_id,
                    "timestamp": reading.timestamp,
                    "stream_position": stream_pos,
                    "machine_position": machine_pos,
                    "temperature_c": reading.temperature_c,
                    "vibration_mm_s": reading.vibration_mm_s,
                    "pressure_bar": reading.pressure_bar,
                    "static_alert": static_res.is_anomaly,
                    "drift_features": [d.feature for d in drift_events],
                    "observed_monotonic_ns": observed_mono_ns,
                }
            )
        consumer.store_offsets(message=message)
    return "shutdown_signal"


def verify_drift_events_on_topic(
    settings: Settings, expected_events: int, timeout_s: float = 15.0
) -> dict[str, Any]:
    """Consume drift events from `settings.drift_topic` to verify end-to-end delivery."""
    if expected_events == 0:
        return {"received": 0, "valid": 0, "malformed": 0}
    consumer = Consumer(build_consumer_config(settings))
    received = 0
    valid = 0
    malformed = 0
    required = {"drift_id", "event_id", "machine_id", "feature", "stream_position"}
    deadline = time.monotonic() + timeout_s
    try:
        consumer.subscribe([settings.drift_topic])
        while received < expected_events and time.monotonic() < deadline:
            msg = consumer.poll(0.5)
            if msg is None or msg.error() is not None:
                continue
            received += 1
            try:
                raw = msg.value()
                data = json.loads(raw) if raw is not None else None
                if isinstance(data, dict) and required.issubset(data.keys()):
                    valid += 1
                else:
                    malformed += 1
            except (UnicodeDecodeError, json.JSONDecodeError):
                malformed += 1
            consumer.store_offsets(message=msg)
    finally:
        consumer.close()
    return {"received": received, "valid": valid, "malformed": malformed}


def _install_shutdown_handlers(stop_event: threading.Event) -> None:
    def request_stop(signum: int, _frame: FrameType | None) -> None:
        logger.info("shutdown requested", extra={"fields": {"signal": signal.Signals(signum).name}})
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


def _log_assignment(_consumer: Consumer, partitions: list[TopicPartition]) -> None:
    logger.info(
        "partitions assigned", extra={"fields": {"partitions": [tp.partition for tp in partitions]}}
    )


def run_drift_monitor(
    settings: Settings,
    limit: int | None,
    idle_timeout_s: float | None,
    adwin_config: AdwinConfig,
    drift_file: Path | None = None,
    observations_file: Path | None = None,
) -> dict[str, Any]:
    """Run the Kafka ADWIN drift monitor consumer and return its summary."""
    stop_event = threading.Event()
    _install_shutdown_handlers(stop_event)

    consumer = Consumer(build_consumer_config(settings))
    producer = Producer(build_producer_config(settings))
    monitor = MachineAdwinMonitor(adwin_config)
    static_detector = ThresholdDetector()
    stats = DriftMonitorStats()
    drift_tracker = DeliveryTracker()

    logger.info(
        "drift monitor starting",
        extra={
            "fields": {
                "bootstrap": settings.bootstrap_servers,
                "raw_topic": settings.topic,
                "drift_topic": settings.drift_topic,
                "group": settings.consumer_group,
                "limit": limit,
                "delta": adwin_config.delta,
            }
        },
    )

    drift_handle = None
    on_drift: Callable[[dict[str, Any]], None] | None = None
    if drift_file is not None:
        drift_file.parent.mkdir(parents=True, exist_ok=True)
        drift_handle = drift_file.open("w", encoding="utf-8")

        def _write_drift(record: dict[str, Any]) -> None:
            drift_handle.write(json.dumps(record, separators=(",", ":")) + "\n")

        on_drift = _write_drift

    obs_handle = None
    on_obs: Callable[[dict[str, Any]], None] | None = None
    if observations_file is not None:
        observations_file.parent.mkdir(parents=True, exist_ok=True)
        obs_handle = observations_file.open("w", encoding="utf-8")

        def _write_obs(record: dict[str, Any]) -> None:
            obs_handle.write(json.dumps(record, separators=(",", ":")) + "\n")

        on_obs = _write_obs

    started = time.monotonic()
    try:
        consumer.subscribe([settings.topic], on_assign=_log_assignment)
        stop_reason = process_drift_stream(
            consumer,
            producer,
            settings.drift_topic,
            monitor,
            static_detector,
            stats,
            drift_tracker,
            stop_event,
            limit,
            idle_timeout_s,
            on_drift=on_drift,
            on_observation=on_obs,
        )
        remaining = producer.flush(30.0)
        if remaining > 0:
            logger.error("drift events flush timed out", extra={"fields": {"remaining": remaining}})
    finally:
        if drift_handle is not None:
            drift_handle.close()
        if obs_handle is not None:
            obs_handle.close()
        consumer.close()

    return stats.to_dict(
        drift_tracker, round(time.monotonic() - started, 3), stop_reason, adwin_config.delta
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the M3 ADWIN drift monitor."""
    parser = argparse.ArgumentParser(description="ADWIN-based concept drift monitor consumer.")
    parser.add_argument("--limit", type=positive_int, default=None)
    parser.add_argument("--idle-timeout-s", type=positive_float, default=None)
    parser.add_argument("--group", default=None, help="Override KAFKA_CONSUMER_GROUP")
    parser.add_argument("--delta", type=positive_float, default=0.002, help="ADWIN delta")
    parser.add_argument("--clock", type=positive_int, default=1, help="ADWIN check period")
    parser.add_argument("--summary-file", type=Path, default=None)
    parser.add_argument("--drift-file", type=Path, default=None)
    parser.add_argument("--observations-file", type=Path, default=None)
    parser.add_argument("--verify-drift", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the M3 ADWIN drift monitor."""
    args = parse_args(argv)
    settings = load_settings()
    if args.group:
        settings = replace(settings, consumer_group=args.group)
    configure_logging(settings.log_level)

    if args.verify_drift is not None:
        verified = verify_drift_events_on_topic(settings, args.verify_drift)
        if args.summary_file is not None:
            write_summary(args.summary_file, verified)
        ok = verified["received"] == args.verify_drift and verified["malformed"] == 0
        return 0 if ok else 1

    adwin_cfg = AdwinConfig(delta=args.delta, clock=args.clock)
    summary = run_drift_monitor(
        settings,
        args.limit,
        args.idle_timeout_s,
        adwin_cfg,
        drift_file=args.drift_file,
        observations_file=args.observations_file,
    )
    ok = (
        summary["malformed"] == 0
        and summary["kafka_errors"] == 0
        and summary["drift_events_failed"] == 0
        and summary["drift_events_delivered"] == summary["drift_events_triggered"]
        and (args.limit is None or summary["received"] >= args.limit)
    )
    logger.log(
        logging.INFO if ok else logging.ERROR,
        "drift monitor summary",
        extra={"fields": summary},
    )
    if args.summary_file is not None:
        write_summary(args.summary_file, summary)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
