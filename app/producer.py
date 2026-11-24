"""Kafka producer: publishes simulated telemetry to `research.telemetry.raw`.

Example:
    python -m app.producer --count 1000 --interval-ms 2

How delivery confirmation works:
    `producer.produce()` only places a message in a local in-memory queue.
    A background librdkafka thread sends batches to the broker. When the
    broker acknowledges (or the send ultimately fails), the delivery callback
    `DeliveryTracker.on_delivery` is invoked from inside `poll()` / `flush()`.
    Therefore only callback results count as "delivered" -- never the
    number of `produce()` calls.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from confluent_kafka import KafkaError, Message, Producer

from app.config import Settings, load_settings, non_negative_float, positive_int
from app.logging_config import configure_logging, log_kafka_client_error
from app.run_summary import write_summary
from app.schemas import GroundTruthRecord, TelemetryEvent
from app.simulator import Scenario, TelemetrySimulator

logger = logging.getLogger(__name__)

_QUEUE_FULL_BACKOFF_S = 0.5


@dataclass
class DeliveryTracker:
    """Counts broker delivery confirmations reported via callbacks."""

    delivered: int = 0
    failed: int = 0
    per_partition: Counter[int] = field(default_factory=Counter)
    failure_reasons: Counter[str] = field(default_factory=Counter)

    def on_delivery(self, error: KafkaError | None, message: Message) -> None:
        """Delivery callback passed to `Producer.produce(on_delivery=...)`."""
        if error is not None:
            self.failed += 1
            self.failure_reasons[error.name()] += 1
            logger.error(
                "delivery failed",
                extra={"fields": {"error": error.str(), "key": _decode_key(message.key())}},
            )
            return
        self.delivered += 1
        partition = message.partition()
        if partition is not None:
            self.per_partition[partition] += 1

    @property
    def confirmed(self) -> int:
        """Messages with a final outcome (delivered or failed)."""
        return self.delivered + self.failed


@dataclass(frozen=True)
class ProducerRunSummary:
    """Final, callback-derived result of a producer run."""

    requested: int
    attempted: int
    delivered: int
    failed: int
    not_confirmed: int
    per_partition: dict[str, int]
    failure_reasons: dict[str, int]
    duration_s: float
    interrupted: bool

    @property
    def succeeded(self) -> bool:
        """True only if every requested event was acknowledged by the broker."""
        return (
            not self.interrupted
            and self.failed == 0
            and self.not_confirmed == 0
            and self.delivered == self.requested
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation (written to --summary-file)."""
        return {
            "requested": self.requested,
            "attempted": self.attempted,
            "delivered": self.delivered,
            "failed": self.failed,
            "not_confirmed": self.not_confirmed,
            "per_partition": self.per_partition,
            "failure_reasons": self.failure_reasons,
            "duration_s": self.duration_s,
            "interrupted": self.interrupted,
            "succeeded": self.succeeded,
        }


def _decode_key(key: bytes | str | None) -> str:
    if key is None:
        return "<none>"
    return key.decode("utf-8", errors="replace") if isinstance(key, bytes) else key


def build_producer_config(settings: Settings) -> dict[str, Any]:
    """Return librdkafka producer configuration.

    * acks=all: the broker confirms only after the write is committed to all
      in-sync replicas (with one broker that is just the leader's log).
    * enable.idempotence: retries cannot create duplicate copies in the topic.
      This is *not* end-to-end exactly-once processing.
    * linger.ms: wait up to 5 ms to batch messages for efficiency.
    * delivery.timeout.ms: give up (and report failure) after 30 s.
    """
    return {
        "bootstrap.servers": settings.bootstrap_servers,
        "client.id": f"{settings.client_id}-producer",
        "acks": "all",
        "enable.idempotence": True,
        "linger.ms": 5,
        "delivery.timeout.ms": 30_000,
        "error_cb": log_kafka_client_error,
    }


def _produce_event(
    producer: Producer,
    topic: str,
    event: TelemetryEvent,
    tracker: DeliveryTracker,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Enqueue one event, waiting for queue space if the local buffer is full."""
    if on_record is None:
        callback: Callable[[KafkaError | None, Message], None] = tracker.on_delivery
    else:
        enqueued_ns = time.monotonic_ns()

        def callback(error: KafkaError | None, message: Message) -> None:
            acked_ns = time.monotonic_ns()
            tracker.on_delivery(error, message)
            on_record(
                {
                    "event_id": event.event_id,
                    "machine_id": event.machine_id,
                    "produced_timestamp": event.timestamp,
                    "enqueued_monotonic_ns": enqueued_ns,
                    "acked_monotonic_ns": acked_ns,
                    "delivery_status": "delivered" if error is None else "failed",
                    "partition": message.partition(),
                    "offset": message.offset() if error is None else None,
                    "error": error.name() if error is not None else None,
                }
            )

    while True:
        try:
            producer.produce(
                topic=topic,
                key=event.machine_id.encode("utf-8"),
                value=event.to_json_bytes(),
                on_delivery=callback,
            )
            return
        except BufferError:
            logger.debug("local producer queue full; waiting for deliveries")
            producer.poll(_QUEUE_FULL_BACKOFF_S)


def publish_events(
    producer: Producer,
    topic: str,
    simulator: TelemetrySimulator,
    count: int,
    interval_ms: float,
    tracker: DeliveryTracker,
    on_record: Callable[[dict[str, Any]], None] | None = None,
    on_ground_truth: Callable[[GroundTruthRecord], None] | None = None,
) -> tuple[int, bool]:
    """Publish `count` events. Returns (attempted, interrupted)."""
    interval_s = interval_ms / 1000.0
    attempted = 0
    try:
        if on_ground_truth is None:
            stream: Any = ((event, None) for event in simulator.generate_events(count))
        else:
            stream = simulator.generate_scenario_events(count)
        for event, truth in stream:
            _produce_event(producer, topic, event, tracker, on_record=on_record)
            if truth is not None and on_ground_truth is not None:
                on_ground_truth(truth)
            attempted += 1
            producer.poll(0)  # Serve delivery callbacks without blocking.
            if interval_s > 0:
                time.sleep(interval_s)
    except KeyboardInterrupt:
        logger.warning(
            "interrupted; flushing already-queued events",
            extra={"fields": {"attempted": attempted}},
        )
        return attempted, True
    return attempted, False


def run_producer(
    settings: Settings,
    count: int,
    interval_ms: float,
    seed: int | None,
    flush_timeout_s: float,
    records_file: Path | None = None,
    scenario: Scenario = Scenario.NORMAL,
    anomaly_rate: float = 0.12,
    ground_truth_file: Path | None = None,
) -> ProducerRunSummary:
    """Create a producer, publish events, flush, and summarise delivery results."""
    producer = Producer(build_producer_config(settings))
    simulator = TelemetrySimulator(
        machine_count=settings.machine_count,
        seed=seed,
        scenario=scenario,
        anomaly_rate=anomaly_rate,
    )
    tracker = DeliveryTracker()
    logger.info(
        "producer starting",
        extra={
            "fields": {
                "bootstrap": settings.bootstrap_servers,
                "topic": settings.topic,
                "count": count,
                "interval_ms": interval_ms,
                "machines": settings.machine_count,
                "scenario": scenario.value,
            }
        },
    )

    records_handle = None
    on_record: Callable[[dict[str, Any]], None] | None = None
    if records_file is not None:
        records_file.parent.mkdir(parents=True, exist_ok=True)
        records_handle = records_file.open("w", encoding="utf-8")

        def _write_record(record: dict[str, Any]) -> None:
            records_handle.write(json.dumps(record, separators=(",", ":")) + "\n")

        on_record = _write_record

    truth_handle = None
    on_ground_truth: Callable[[GroundTruthRecord], None] | None = None
    if ground_truth_file is not None:
        ground_truth_file.parent.mkdir(parents=True, exist_ok=True)
        truth_handle = ground_truth_file.open("w", encoding="utf-8")

        def _write_truth(truth: GroundTruthRecord) -> None:
            truth_handle.write(json.dumps(truth.to_dict(), separators=(",", ":")) + "\n")

        on_ground_truth = _write_truth

    started = time.monotonic()
    try:
        attempted, interrupted = publish_events(
            producer,
            settings.topic,
            simulator,
            count,
            interval_ms,
            tracker,
            on_record=on_record,
            on_ground_truth=on_ground_truth,
        )
        remaining = producer.flush(flush_timeout_s)
        if remaining > 0:
            logger.error(
                "flush timed out with messages still queued",
                extra={"fields": {"remaining": remaining}},
            )
    finally:
        if records_handle is not None:
            records_handle.close()
        if truth_handle is not None:
            truth_handle.close()

    return ProducerRunSummary(
        requested=count,
        attempted=attempted,
        delivered=tracker.delivered,
        failed=tracker.failed,
        not_confirmed=attempted - tracker.confirmed,
        per_partition={str(part): num for part, num in sorted(tracker.per_partition.items())},
        failure_reasons=dict(tracker.failure_reasons),
        duration_s=round(time.monotonic() - started, 3),
        interrupted=interrupted,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse producer command-line arguments."""
    parser = argparse.ArgumentParser(description="Publish simulated telemetry to Kafka.")
    parser.add_argument(
        "--count",
        type=positive_int,
        default=1000,
        help="Number of events to publish (default: 1000)",
    )
    parser.add_argument(
        "--interval-ms",
        type=non_negative_float,
        default=0.0,
        help="Pause between events in milliseconds (default: 0)",
    )
    parser.add_argument(
        "--machines", type=positive_int, default=None, help="Override SIMULATOR_MACHINE_COUNT"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for reproducible sensor values"
    )
    parser.add_argument(
        "--flush-timeout-s",
        type=non_negative_float,
        default=30.0,
        help="Max seconds to wait for outstanding deliveries (default: 30)",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Optional path to write the JSON run summary",
    )
    parser.add_argument(
        "--records-file",
        type=Path,
        default=None,
        help="Optional path to write per-event JSONL delivery records",
    )
    parser.add_argument(
        "--scenario",
        type=Scenario,
        choices=list(Scenario),
        default=Scenario.NORMAL,
        help="Telemetry scenario: normal, anomalies, or drift (default: normal)",
    )
    parser.add_argument(
        "--anomaly-rate",
        type=non_negative_float,
        default=0.12,
        help="Anomaly injection probability for --scenario anomalies (default: 0.12)",
    )
    parser.add_argument(
        "--ground-truth-file",
        type=Path,
        default=None,
        help="Optional path to write separate ground-truth JSONL records",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns 0 only if every event was confirmed delivered."""
    args = parse_args(argv)
    settings = load_settings()
    if args.machines is not None:
        settings = replace(settings, machine_count=args.machines)
    configure_logging(settings.log_level)

    summary = run_producer(
        settings,
        args.count,
        args.interval_ms,
        args.seed,
        args.flush_timeout_s,
        records_file=args.records_file,
        scenario=args.scenario,
        anomaly_rate=args.anomaly_rate,
        ground_truth_file=args.ground_truth_file,
    )
    log_level = logging.INFO if summary.succeeded else logging.ERROR
    logger.log(log_level, "producer delivery summary", extra={"fields": summary.to_dict()})
    if args.summary_file is not None:
        write_summary(args.summary_file, summary.to_dict())
    return 0 if summary.succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
