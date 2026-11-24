"""Kafka consumer: reads and validates telemetry from `research.telemetry.raw`.

Example:
    python -m app.consumer --limit 1000

Processing guarantee: at-least-once. The offset of a message is *stored*
only after the message has been processed, and stored offsets are committed
periodically and on shutdown. If the consumer crashes between processing and
committing, the same messages are delivered again after restart -- which is
exactly why duplicate event IDs are tracked. This is NOT exactly-once.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import FrameType
from typing import Any

from confluent_kafka import Consumer, KafkaError, KafkaException, Message, TopicPartition

from app.config import Settings, load_settings, positive_float, positive_int
from app.logging_config import configure_logging, log_kafka_client_error
from app.run_summary import write_summary
from app.schemas import (
    SchemaValidationError,
    TelemetryEvent,
    format_utc_timestamp,
    parse_event_json,
)

logger = logging.getLogger(__name__)

_POLL_TIMEOUT_S = 1.0


class MessageOutcome(str, Enum):
    """Classification of one received message."""

    VALID = "valid"
    DUPLICATE = "duplicate"
    MALFORMED = "malformed"


class StopReason(str, Enum):
    """Why the consume loop ended."""

    LIMIT_REACHED = "limit_reached"
    IDLE_TIMEOUT = "idle_timeout"
    SHUTDOWN_SIGNAL = "shutdown_signal"


@dataclass
class ConsumerStats:
    """Counters for one consumer run.

    * received:   messages with a payload read from Kafka (valid or not).
    * valid:      messages that passed schema validation (includes duplicates).
    * duplicates: valid messages whose event_id was already seen in this run.
    * malformed:  messages that failed JSON parsing or schema validation.
    """

    received: int = 0
    valid: int = 0
    malformed: int = 0
    duplicates: int = 0
    kafka_errors: int = 0
    per_partition: Counter[int] = field(default_factory=Counter)
    seen_event_ids: set[str] = field(default_factory=set)
    last_offsets: dict[int, int] = field(default_factory=dict)

    def record_with_event(
        self, payload: bytes | None, partition: int
    ) -> tuple[MessageOutcome, TelemetryEvent | None]:
        """Validate one payload, update counters, and return (outcome, event)."""
        self.received += 1
        self.per_partition[partition] += 1
        try:
            event = parse_event_json(payload)
        except SchemaValidationError as error:
            self.malformed += 1
            logger.warning(
                "malformed message",
                extra={"fields": {"partition": partition, "reason": str(error)}},
            )
            return MessageOutcome.MALFORMED, None

        self.valid += 1
        if event.event_id in self.seen_event_ids:
            self.duplicates += 1
            logger.warning("duplicate event_id", extra={"fields": {"event_id": event.event_id}})
            return MessageOutcome.DUPLICATE, event
        self.seen_event_ids.add(event.event_id)
        return MessageOutcome.VALID, event

    def record(self, payload: bytes | None, partition: int) -> MessageOutcome:
        """Validate one payload and update counters. Returns its classification."""
        outcome, _ = self.record_with_event(payload, partition)
        return outcome

    def to_dict(
        self,
        stop_reason: StopReason,
        duration_s: float,
        consumer_lag: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """JSON-serialisable summary (written to --summary-file)."""
        return {
            "received": self.received,
            "valid": self.valid,
            "malformed": self.malformed,
            "duplicates": self.duplicates,
            "unique_event_ids": len(self.seen_event_ids),
            "kafka_errors": self.kafka_errors,
            "per_partition": {str(part): num for part, num in sorted(self.per_partition.items())},
            "consumer_lag": consumer_lag,
            "stop_reason": stop_reason.value,
            "duration_s": duration_s,
        }


def _log_assignment(_consumer: Consumer, partitions: list[TopicPartition]) -> None:
    logger.info(
        "partitions assigned", extra={"fields": {"partitions": [tp.partition for tp in partitions]}}
    )


def build_consumer_config(settings: Settings) -> dict[str, Any]:
    """Return librdkafka consumer configuration.

    * group.id: consumers sharing a group split the topic's partitions.
    * auto.offset.reset=earliest: a *new* group starts at the oldest message;
      an existing group resumes from its committed offset.
    * enable.auto.offset.store=False: we store each offset manually *after*
      processing, so an unprocessed message is never marked as done.
    * enable.auto.commit=True: stored offsets are committed every 5 s and on close.
    """
    return {
        "bootstrap.servers": settings.bootstrap_servers,
        "group.id": settings.consumer_group,
        "client.id": f"{settings.client_id}-consumer",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
        "enable.auto.offset.store": False,
        "error_cb": log_kafka_client_error,
    }


def _handle_kafka_error(error: KafkaError, stats: ConsumerStats) -> None:
    """Count and log a per-message Kafka error; raise if it is fatal."""
    stats.kafka_errors += 1
    if error.fatal():
        raise KafkaException(error)
    logger.error(
        "kafka message error", extra={"fields": {"code": error.name(), "error": error.str()}}
    )


def consume_events(
    consumer: Consumer,
    stats: ConsumerStats,
    stop_event: threading.Event,
    limit: int | None,
    idle_timeout_s: float | None,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> StopReason:
    """Poll messages until the limit, an idle timeout or a shutdown signal."""
    last_activity = time.monotonic()
    while not stop_event.is_set():
        if limit is not None and stats.received >= limit:
            return StopReason.LIMIT_REACHED
        message: Message | None = consumer.poll(_POLL_TIMEOUT_S)
        if message is None:
            if idle_timeout_s is not None and time.monotonic() - last_activity >= idle_timeout_s:
                logger.info(
                    "no messages received within idle timeout",
                    extra={"fields": {"idle_timeout_s": idle_timeout_s}},
                )
                return StopReason.IDLE_TIMEOUT
            continue
        error = message.error()
        if error is not None:
            _handle_kafka_error(error, stats)
            continue
        consumed_mono_ns = time.monotonic_ns()
        last_activity = time.monotonic()
        raw_partition = message.partition()
        partition = raw_partition if raw_partition is not None else -1
        outcome, event = stats.record_with_event(message.value(), partition)
        offset = message.offset() if hasattr(message, "offset") else None
        if partition >= 0 and isinstance(offset, int) and offset >= 0:
            stats.last_offsets[partition] = offset
        if on_record is not None:
            on_record(
                {
                    "event_id": event.event_id if event is not None else None,
                    "machine_id": event.machine_id if event is not None else None,
                    "produced_timestamp": event.timestamp if event is not None else None,
                    "consumed_timestamp": format_utc_timestamp(datetime.now(timezone.utc)),
                    "consumed_monotonic_ns": consumed_mono_ns,
                    "partition": partition,
                    "offset": offset,
                    "consume_status": outcome.value,
                }
            )
        consumer.store_offsets(message=message)
    return StopReason.SHUTDOWN_SIGNAL


def _measure_consumer_lag(
    consumer: Consumer, topic: str, last_offsets: dict[int, int]
) -> dict[str, int] | None:
    """Measure per-partition lag using broker high-watermark offsets."""
    if not last_offsets:
        return None
    lag_by_partition: dict[str, int] = {}
    for partition, last_offset in sorted(last_offsets.items()):
        try:
            _low, high = consumer.get_watermark_offsets(
                TopicPartition(topic, partition), timeout=1.0, cached=False
            )
        except KafkaException as error:
            logger.debug(
                "could not query watermark offsets",
                extra={"fields": {"partition": partition, "error": str(error)}},
            )
            return None
        lag_by_partition[str(partition)] = max(0, high - 1 - last_offset)
    return lag_by_partition


def _install_shutdown_handlers(stop_event: threading.Event) -> None:
    """Make Ctrl+C (SIGINT) and SIGTERM request a graceful stop."""

    def request_stop(signum: int, _frame: FrameType | None) -> None:
        logger.info("shutdown requested", extra={"fields": {"signal": signal.Signals(signum).name}})
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


def run_consumer(
    settings: Settings,
    limit: int | None,
    idle_timeout_s: float | None,
    records_file: Path | None = None,
) -> dict[str, Any]:
    """Subscribe, consume until a stop condition, close cleanly and summarise."""
    stop_event = threading.Event()
    _install_shutdown_handlers(stop_event)
    consumer = Consumer(build_consumer_config(settings))
    stats = ConsumerStats()
    logger.info(
        "consumer starting",
        extra={
            "fields": {
                "bootstrap": settings.bootstrap_servers,
                "topic": settings.topic,
                "group": settings.consumer_group,
                "limit": limit,
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

    consumer_lag: dict[str, int] | None = None
    started = time.monotonic()
    try:
        consumer.subscribe([settings.topic], on_assign=_log_assignment)
        stop_reason = consume_events(
            consumer, stats, stop_event, limit, idle_timeout_s, on_record=on_record
        )
        consumer_lag = _measure_consumer_lag(consumer, settings.topic, stats.last_offsets)
    finally:
        if records_handle is not None:
            records_handle.close()
        consumer.close()  # Commits stored offsets and leaves the group cleanly.
    return stats.to_dict(stop_reason, round(time.monotonic() - started, 3), consumer_lag)


def is_successful(summary: dict[str, Any], limit: int | None) -> bool:
    """Return True when no data/Kafka problems occurred and any limit was met."""
    clean = (
        summary["malformed"] == 0 and summary["duplicates"] == 0 and summary["kafka_errors"] == 0
    )
    return clean and (limit is None or summary["received"] >= limit)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse consumer command-line arguments."""
    parser = argparse.ArgumentParser(description="Consume and validate telemetry from Kafka.")
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Stop after this many messages (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "--idle-timeout-s",
        type=positive_float,
        default=None,
        help="Stop if no message arrives for this many seconds",
    )
    parser.add_argument("--group", default=None, help="Override KAFKA_CONSUMER_GROUP")
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
        help="Optional path to write per-event JSONL receipt records",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns 0 when the run was clean and any limit was reached."""
    args = parse_args(argv)
    settings = load_settings()
    if args.group:
        settings = replace(settings, consumer_group=args.group)
    configure_logging(settings.log_level)

    summary = run_consumer(
        settings, args.limit, args.idle_timeout_s, records_file=args.records_file
    )
    succeeded = is_successful(summary, args.limit)
    logger.log(
        logging.INFO if succeeded else logging.ERROR,
        "consumer receipt summary",
        extra={"fields": summary},
    )
    if args.summary_file is not None:
        write_summary(args.summary_file, summary)
    return 0 if succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
