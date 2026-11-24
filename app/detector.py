"""Threshold-based anomaly detection consumer for DriftGuard-Kafka M2.

Consumes raw telemetry from `research.telemetry.raw`, strips any placeholder
`label` field so ground-truth labels are never exposed to the detector, applies
static threshold + stuck-sensor rules, and publishes alerts to
`research.telemetry.alerts`.
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

from app.config import Settings, load_settings, positive_float, positive_int
from app.consumer import build_consumer_config
from app.logging_config import configure_logging
from app.producer import DeliveryTracker, build_producer_config
from app.run_summary import write_summary
from app.schemas import (
    SchemaValidationError,
    TelemetryEvent,
    format_utc_timestamp,
    parse_event_json,
)

logger = logging.getLogger(__name__)

_POLL_TIMEOUT_S = 1.0


@dataclass(frozen=True)
class ThresholdConfig:
    """Static operating thresholds and stuck-sensor window for baseline detection."""

    temp_min_c: float = 30.0
    temp_max_c: float = 70.0
    vib_min_mm_s: float = 0.5
    vib_max_mm_s: float = 5.0
    press_min_bar: float = 2.5
    press_max_bar: float = 7.5
    stuck_window: int = 4


@dataclass(frozen=True)
class SensorReading:
    """Unlabelled sensor reading exposed to the detector (never includes `label`)."""

    event_id: str
    machine_id: str
    timestamp: str
    temperature_c: float
    vibration_mm_s: float
    pressure_bar: float

    @classmethod
    def from_telemetry(cls, event: TelemetryEvent) -> SensorReading:
        """Strip `label` when constructing a detector input."""
        return cls(
            event_id=event.event_id,
            machine_id=event.machine_id,
            timestamp=event.timestamp,
            temperature_c=event.temperature_c,
            vibration_mm_s=event.vibration_mm_s,
            pressure_bar=event.pressure_bar,
        )


@dataclass(frozen=True)
class DetectionResult:
    """Outcome of running the baseline detector on one `SensorReading`."""

    is_anomaly: bool
    reason: str
    triggered_rules: tuple[str, ...]


class ThresholdDetector:
    """Static threshold + stuck-sensor baseline anomaly detector."""

    def __init__(self, config: ThresholdConfig | None = None) -> None:
        """Initialise detector with static thresholds and per-machine stuck state."""
        self._config = config or ThresholdConfig()
        self._last_reading: dict[str, tuple[tuple[float, float, float], int]] = {}

    def evaluate(self, reading: SensorReading) -> DetectionResult:
        """Evaluate one unlabelled `SensorReading`."""
        cfg = self._config
        rules: list[str] = []

        if reading.temperature_c < cfg.temp_min_c:
            rules.append(f"temperature_c<{cfg.temp_min_c}")
        elif reading.temperature_c > cfg.temp_max_c:
            rules.append(f"temperature_c>{cfg.temp_max_c}")

        if reading.vibration_mm_s < cfg.vib_min_mm_s:
            rules.append(f"vibration_mm_s<{cfg.vib_min_mm_s}")
        elif reading.vibration_mm_s > cfg.vib_max_mm_s:
            rules.append(f"vibration_mm_s>{cfg.vib_max_mm_s}")

        if reading.pressure_bar < cfg.press_min_bar:
            rules.append(f"pressure_bar<{cfg.press_min_bar}")
        elif reading.pressure_bar > cfg.press_max_bar:
            rules.append(f"pressure_bar>{cfg.press_max_bar}")

        stuck = self._update_stuck_streak(reading)
        if stuck:
            rules.append(f"stuck_sensor>={cfg.stuck_window}")

        if not rules:
            return DetectionResult(is_anomaly=False, reason="normal", triggered_rules=())

        has_threshold = any(not rule.startswith("stuck_sensor") for rule in rules)
        if has_threshold and stuck:
            reason = "threshold_and_stuck"
        elif stuck:
            reason = "stuck_sensor"
        else:
            reason = "threshold_breach"
        return DetectionResult(is_anomaly=True, reason=reason, triggered_rules=tuple(rules))

    def _update_stuck_streak(self, reading: SensorReading) -> bool:
        sample = (reading.temperature_c, reading.vibration_mm_s, reading.pressure_bar)
        prev = self._last_reading.get(reading.machine_id)
        streak = prev[1] + 1 if prev is not None and prev[0] == sample else 1
        self._last_reading[reading.machine_id] = (sample, streak)
        return streak >= self._config.stuck_window


@dataclass
class DetectorStats:
    """Counters for a detector consumer run."""

    received: int = 0
    valid: int = 0
    malformed: int = 0
    kafka_errors: int = 0
    alerts_triggered: int = 0
    by_reason: Counter[str] = field(default_factory=Counter)

    def to_dict(
        self,
        alert_tracker: DeliveryTracker,
        duration_s: float,
        stop_reason: str,
    ) -> dict[str, Any]:
        """Return JSON-serialisable detector summary."""
        return {
            "received": self.received,
            "valid": self.valid,
            "malformed": self.malformed,
            "kafka_errors": self.kafka_errors,
            "alerts_triggered": self.alerts_triggered,
            "alerts_delivered": alert_tracker.delivered,
            "alerts_failed": alert_tracker.failed,
            "by_reason": dict(self.by_reason),
            "duration_s": duration_s,
            "stop_reason": stop_reason,
        }


def build_alert_payload(reading: SensorReading, result: DetectionResult) -> dict[str, Any]:
    """Construct the JSON payload published to the alerts topic."""
    return {
        "alert_id": str(uuid.uuid4()),
        "event_id": reading.event_id,
        "machine_id": reading.machine_id,
        "timestamp": reading.timestamp,
        "detected_utc": format_utc_timestamp(datetime.now(timezone.utc)),
        "detector": "static_threshold_v1",
        "reason": result.reason,
        "triggered_rules": list(result.triggered_rules),
        "temperature_c": reading.temperature_c,
        "vibration_mm_s": reading.vibration_mm_s,
        "pressure_bar": reading.pressure_bar,
    }


def _publish_alert(
    producer: Producer,
    alerts_topic: str,
    reading: SensorReading,
    result: DetectionResult,
    tracker: DeliveryTracker,
) -> None:
    payload = json.dumps(build_alert_payload(reading, result), separators=(",", ":")).encode(
        "utf-8"
    )
    while True:
        try:
            producer.produce(
                topic=alerts_topic,
                key=reading.machine_id.encode("utf-8"),
                value=payload,
                on_delivery=tracker.on_delivery,
            )
            producer.poll(0)
            return
        except BufferError:
            producer.poll(0.2)


def process_messages(
    consumer: Consumer,
    producer: Producer,
    alerts_topic: str,
    detector: ThresholdDetector,
    stats: DetectorStats,
    alert_tracker: DeliveryTracker,
    stop_event: threading.Event,
    limit: int | None,
    idle_timeout_s: float | None,
    on_detection: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """Poll raw telemetry, detect anomalies, publish alerts, and store offsets."""
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

        detected_mono_ns = time.monotonic_ns()
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
        result = detector.evaluate(reading)
        stats.by_reason[result.reason] += 1

        if result.is_anomaly:
            stats.alerts_triggered += 1
            _publish_alert(producer, alerts_topic, reading, result, alert_tracker)

        if on_detection is not None:
            on_detection(
                {
                    "event_id": reading.event_id,
                    "machine_id": reading.machine_id,
                    "predicted_anomaly": result.is_anomaly,
                    "reason": result.reason,
                    "triggered_rules": list(result.triggered_rules),
                    "detected_monotonic_ns": detected_mono_ns,
                    "detected_utc": format_utc_timestamp(datetime.now(timezone.utc)),
                }
            )
        consumer.store_offsets(message=message)
    return "shutdown_signal"


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


def run_detector(
    settings: Settings,
    limit: int | None,
    idle_timeout_s: float | None,
    detections_file: Path | None = None,
) -> dict[str, Any]:
    """Run the threshold detector pipeline and return its summary."""
    stop_event = threading.Event()
    _install_shutdown_handlers(stop_event)

    consumer = Consumer(build_consumer_config(settings))
    producer = Producer(build_producer_config(settings))
    detector = ThresholdDetector()
    stats = DetectorStats()
    alert_tracker = DeliveryTracker()

    logger.info(
        "detector starting",
        extra={
            "fields": {
                "bootstrap": settings.bootstrap_servers,
                "raw_topic": settings.topic,
                "alerts_topic": settings.alerts_topic,
                "group": settings.consumer_group,
                "limit": limit,
            }
        },
    )

    detections_handle = None
    on_detection: Callable[[dict[str, Any]], None] | None = None
    if detections_file is not None:
        detections_file.parent.mkdir(parents=True, exist_ok=True)
        detections_handle = detections_file.open("w", encoding="utf-8")

        def _write_detection(record: dict[str, Any]) -> None:
            detections_handle.write(json.dumps(record, separators=(",", ":")) + "\n")

        on_detection = _write_detection

    started = time.monotonic()
    try:
        consumer.subscribe([settings.topic], on_assign=_log_assignment)
        stop_reason = process_messages(
            consumer,
            producer,
            settings.alerts_topic,
            detector,
            stats,
            alert_tracker,
            stop_event,
            limit,
            idle_timeout_s,
            on_detection=on_detection,
        )
        remaining = producer.flush(30.0)
        if remaining > 0:
            logger.error("alerts flush timed out", extra={"fields": {"remaining": remaining}})
    finally:
        if detections_handle is not None:
            detections_handle.close()
        consumer.close()

    return stats.to_dict(alert_tracker, round(time.monotonic() - started, 3), stop_reason)


def verify_alerts_on_topic(
    settings: Settings, expected_alerts: int, timeout_s: float = 15.0
) -> dict[str, Any]:
    """Consume alerts from `settings.alerts_topic` to verify end-to-end alert delivery."""
    if expected_alerts == 0:
        return {"received": 0, "valid": 0, "malformed": 0}
    consumer = Consumer(build_consumer_config(settings))
    received = 0
    valid = 0
    malformed = 0
    deadline = time.monotonic() + timeout_s
    try:
        consumer.subscribe([settings.alerts_topic])
        while received < expected_alerts and time.monotonic() < deadline:
            msg = consumer.poll(0.5)
            if msg is None or msg.error() is not None:
                continue
            received += 1
            try:
                raw = msg.value()
                data = json.loads(raw) if raw is not None else None
                if isinstance(data, dict) and "alert_id" in data and "event_id" in data:
                    valid += 1
                else:
                    malformed += 1
            except (UnicodeDecodeError, json.JSONDecodeError):
                malformed += 1
            consumer.store_offsets(message=msg)
    finally:
        consumer.close()
    return {"received": received, "valid": valid, "malformed": malformed}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse detector CLI arguments."""
    parser = argparse.ArgumentParser(description="Threshold-based anomaly detection consumer.")
    parser.add_argument("--limit", type=positive_int, default=None)
    parser.add_argument("--idle-timeout-s", type=positive_float, default=None)
    parser.add_argument("--group", default=None, help="Override KAFKA_CONSUMER_GROUP")
    parser.add_argument("--summary-file", type=Path, default=None)
    parser.add_argument("--detections-file", type=Path, default=None)
    parser.add_argument("--verify-alerts", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the M2 baseline detector."""
    args = parse_args(argv)
    settings = load_settings()
    if args.group:
        settings = replace(settings, consumer_group=args.group)
    configure_logging(settings.log_level)

    if args.verify_alerts is not None:
        alert_summary = verify_alerts_on_topic(settings, args.verify_alerts)
        if args.summary_file is not None:
            write_summary(args.summary_file, alert_summary)
        ok = alert_summary["received"] == args.verify_alerts and alert_summary["malformed"] == 0
        return 0 if ok else 1

    summary = run_detector(
        settings, args.limit, args.idle_timeout_s, detections_file=args.detections_file
    )
    ok = (
        summary["malformed"] == 0
        and summary["kafka_errors"] == 0
        and summary["alerts_failed"] == 0
        and summary["alerts_delivered"] == summary["alerts_triggered"]
        and (args.limit is None or summary["received"] >= args.limit)
    )
    logger.log(
        logging.INFO if ok else logging.ERROR,
        "detector summary",
        extra={"fields": summary},
    )
    if args.summary_file is not None:
        write_summary(args.summary_file, summary)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
