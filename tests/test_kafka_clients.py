"""Unit tests for producer and consumer logic using in-memory fakes.

These tests exercise delivery accounting, message classification and the
poll loops without a broker. Real broker behaviour is covered by the
integration demo (`make demo`), which is not part of coverage measurement.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from app.config import load_settings
from app.consumer import (
    ConsumerStats,
    MessageOutcome,
    StopReason,
    _measure_consumer_lag,
    build_consumer_config,
    consume_events,
    is_successful,
)
from app.consumer import (
    parse_args as parse_consumer_args,
)
from app.producer import (
    DeliveryTracker,
    ProducerRunSummary,
    build_producer_config,
    publish_events,
)
from app.producer import (
    parse_args as parse_producer_args,
)
from app.simulator import TelemetrySimulator

DeliveryCallback = Callable[[Any, Any], None]


@dataclass
class FakeKafkaError:
    code_name: str = "_MSG_TIMED_OUT"
    is_fatal: bool = False

    def name(self) -> str:
        return self.code_name

    def str(self) -> str:
        return f"{self.code_name}: simulated"

    def fatal(self) -> bool:
        return self.is_fatal


@dataclass
class FakeMessage:
    payload: bytes | None = None
    partition_id: int | None = 0
    message_key: bytes | None = b"machine_01"
    kafka_error: FakeKafkaError | None = None

    def value(self) -> bytes | None:
        return self.payload

    def key(self) -> bytes | None:
        return self.message_key

    def partition(self) -> int | None:
        return self.partition_id

    def offset(self) -> int:
        return 0

    def error(self) -> FakeKafkaError | None:
        return self.kafka_error


@dataclass
class FakeProducer:
    """Records produced messages and confirms them on poll()."""

    buffer_errors_before_success: int = 0
    produced: list[dict[str, Any]] = field(default_factory=list)
    pending: list[tuple[DeliveryCallback, FakeMessage]] = field(default_factory=list)

    def produce(self, topic: str, key: bytes, value: bytes, on_delivery: DeliveryCallback) -> None:
        if self.buffer_errors_before_success > 0:
            self.buffer_errors_before_success -= 1
            raise BufferError("queue full")
        self.produced.append({"topic": topic, "key": key, "value": value})
        partition = len(self.produced) % 3
        self.pending.append((on_delivery, FakeMessage(value, partition, key)))

    def poll(self, _timeout: float) -> int:
        served = len(self.pending)
        for callback, message in self.pending:
            callback(None, message)
        self.pending.clear()
        return served


@dataclass
class FakeConsumer:
    messages: list[FakeMessage | None]
    stored: list[FakeMessage] = field(default_factory=list)

    def poll(self, _timeout: float) -> FakeMessage | None:
        return self.messages.pop(0) if self.messages else None

    def store_offsets(self, message: FakeMessage) -> None:
        self.stored.append(message)


def make_payload(seed: int = 0) -> bytes:
    return TelemetrySimulator(machine_count=2, seed=seed).generate_event().to_json_bytes()


class TestDeliveryTracker:
    def test_successful_delivery_is_counted_per_partition(self) -> None:
        tracker = DeliveryTracker()
        tracker.on_delivery(None, FakeMessage(partition_id=2))  # type: ignore[arg-type]
        tracker.on_delivery(None, FakeMessage(partition_id=2))  # type: ignore[arg-type]
        assert tracker.delivered == 2
        assert tracker.per_partition[2] == 2
        assert tracker.failed == 0

    def test_failed_delivery_is_counted_with_reason(self) -> None:
        tracker = DeliveryTracker()
        tracker.on_delivery(
            FakeKafkaError("_MSG_TIMED_OUT"),  # type: ignore[arg-type]
            FakeMessage(message_key=None),  # type: ignore[arg-type]
        )
        assert tracker.failed == 1
        assert tracker.delivered == 0
        assert tracker.failure_reasons["_MSG_TIMED_OUT"] == 1
        assert tracker.confirmed == 1


class TestPublishEvents:
    def test_every_event_is_produced_with_machine_id_key(self) -> None:
        producer = FakeProducer()
        tracker = DeliveryTracker()
        attempted, interrupted = publish_events(
            producer,  # type: ignore[arg-type]
            "topic.a",
            TelemetrySimulator(3, seed=1),
            20,
            0,
            tracker,
        )
        assert (attempted, interrupted) == (20, False)
        assert tracker.delivered == 20
        assert all(msg["key"].startswith(b"machine_") for msg in producer.produced)
        assert all(msg["topic"] == "topic.a" for msg in producer.produced)

    def test_full_local_queue_is_retried_not_dropped(self) -> None:
        producer = FakeProducer(buffer_errors_before_success=2)
        tracker = DeliveryTracker()
        attempted, _ = publish_events(
            producer,  # type: ignore[arg-type]
            "topic.a",
            TelemetrySimulator(1, seed=1),
            3,
            0,
            tracker,
        )
        assert attempted == 3
        assert len(producer.produced) == 3

    def test_interval_is_respected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr("app.producer.time.sleep", sleeps.append)
        publish_events(
            FakeProducer(),  # type: ignore[arg-type]
            "t",
            TelemetrySimulator(1),
            4,
            2.0,
            DeliveryTracker(),
        )
        assert sleeps == [0.002] * 4


class TestProducerRunSummary:
    @staticmethod
    def summary(**overrides: Any) -> ProducerRunSummary:
        values: dict[str, Any] = {
            "requested": 10,
            "attempted": 10,
            "delivered": 10,
            "failed": 0,
            "not_confirmed": 0,
            "per_partition": {"0": 10},
            "failure_reasons": {},
            "duration_s": 0.1,
            "interrupted": False,
        }
        values.update(overrides)
        return ProducerRunSummary(**values)

    def test_full_delivery_succeeds(self) -> None:
        assert self.summary().succeeded
        assert self.summary().to_dict()["succeeded"] is True

    @pytest.mark.parametrize(
        "overrides",
        [
            {"failed": 1, "delivered": 9},
            {"not_confirmed": 1, "delivered": 9},
            {"interrupted": True},
            {"delivered": 9},
        ],
    )
    def test_any_shortfall_fails(self, overrides: dict[str, Any]) -> None:
        assert not self.summary(**overrides).succeeded


class TestConsumerStats:
    def test_valid_message_is_counted(self) -> None:
        stats = ConsumerStats()
        assert stats.record(make_payload(), partition=1) is MessageOutcome.VALID
        assert (stats.received, stats.valid, stats.malformed) == (1, 1, 0)
        assert stats.per_partition[1] == 1

    def test_duplicate_event_id_is_detected(self) -> None:
        stats = ConsumerStats()
        payload = make_payload()
        stats.record(payload, partition=0)
        assert stats.record(payload, partition=0) is MessageOutcome.DUPLICATE
        assert stats.duplicates == 1
        assert len(stats.seen_event_ids) == 1

    @pytest.mark.parametrize("payload", [b"{broken", b"", None, b'{"event_id": "x"}'])
    def test_malformed_message_is_counted(self, payload: bytes | None) -> None:
        stats = ConsumerStats()
        assert stats.record(payload, partition=0) is MessageOutcome.MALFORMED
        assert stats.malformed == 1
        assert stats.valid == 0

    def test_summary_contains_all_counters(self) -> None:
        stats = ConsumerStats()
        stats.record(make_payload(), partition=2)
        summary = stats.to_dict(StopReason.LIMIT_REACHED, 1.5)
        assert summary["unique_event_ids"] == 1
        assert summary["per_partition"] == {"2": 1}
        assert summary["stop_reason"] == "limit_reached"


class TestConsumeEvents:
    def test_stops_at_limit_and_stores_offsets(self) -> None:
        messages: list[FakeMessage | None] = [FakeMessage(make_payload(i)) for i in range(5)]
        consumer = FakeConsumer(messages)
        stats = ConsumerStats()
        reason = consume_events(consumer, stats, threading.Event(), 3, None)  # type: ignore[arg-type]
        assert reason is StopReason.LIMIT_REACHED
        assert stats.received == 3
        assert len(consumer.stored) == 3

    def test_stops_after_idle_timeout(self) -> None:
        consumer = FakeConsumer([None])
        reason = consume_events(consumer, ConsumerStats(), threading.Event(), None, 0.001)  # type: ignore[arg-type]
        assert reason is StopReason.IDLE_TIMEOUT

    def test_stops_on_shutdown_signal(self) -> None:
        stop_event = threading.Event()
        stop_event.set()
        reason = consume_events(FakeConsumer([]), ConsumerStats(), stop_event, None, None)  # type: ignore[arg-type]
        assert reason is StopReason.SHUTDOWN_SIGNAL

    def test_non_fatal_kafka_error_is_counted_and_skipped(self) -> None:
        consumer = FakeConsumer(
            [FakeMessage(kafka_error=FakeKafkaError("_TRANSPORT")), FakeMessage(make_payload())]
        )
        stats = ConsumerStats()
        consume_events(consumer, stats, threading.Event(), 1, None)  # type: ignore[arg-type]
        assert stats.kafka_errors == 1
        assert stats.received == 1

    def test_fatal_kafka_error_is_raised(self) -> None:
        from confluent_kafka import KafkaException

        consumer = FakeConsumer([FakeMessage(kafka_error=FakeKafkaError("_FATAL", True))])
        with pytest.raises(KafkaException):
            consume_events(consumer, ConsumerStats(), threading.Event(), 1, None)  # type: ignore[arg-type]


class TestSuccessCriteria:
    BASE: dict[str, Any] = {"received": 10, "malformed": 0, "duplicates": 0, "kafka_errors": 0}

    def test_clean_run_meeting_limit_succeeds(self) -> None:
        assert is_successful(self.BASE, limit=10)
        assert is_successful(self.BASE, limit=None)

    @pytest.mark.parametrize("key", ["malformed", "duplicates", "kafka_errors"])
    def test_problems_fail_the_run(self, key: str) -> None:
        assert not is_successful({**self.BASE, key: 1}, limit=None)

    def test_shortfall_against_limit_fails(self) -> None:
        assert not is_successful(self.BASE, limit=11)


def test_client_configs_use_settings() -> None:
    settings = load_settings({"KAFKA_BOOTSTRAP_SERVERS": "host:1", "KAFKA_CONSUMER_GROUP": "g"})
    producer_config = build_producer_config(settings)
    consumer_config = build_consumer_config(settings)
    assert producer_config["bootstrap.servers"] == "host:1"
    assert producer_config["acks"] == "all"
    assert producer_config["enable.idempotence"] is True
    assert consumer_config["group.id"] == "g"
    assert consumer_config["enable.auto.offset.store"] is False


def test_publish_and_consume_emit_records_when_enabled() -> None:
    prod_records: list[dict[str, Any]] = []
    publish_events(
        FakeProducer(),  # type: ignore[arg-type]
        "topic.a",
        TelemetrySimulator(2, seed=1),
        2,
        0,
        DeliveryTracker(),
        on_record=prod_records.append,
    )
    assert len(prod_records) == 2
    assert prod_records[0]["delivery_status"] == "delivered"
    assert isinstance(prod_records[0]["enqueued_monotonic_ns"], int)

    cons_records: list[dict[str, Any]] = []
    consumer = FakeConsumer([FakeMessage(make_payload(1), partition_id=1)])
    consume_events(
        consumer,  # type: ignore[arg-type]
        ConsumerStats(),
        threading.Event(),
        1,
        None,
        on_record=cons_records.append,
    )
    assert len(cons_records) == 1
    assert cons_records[0]["consume_status"] == "valid"
    assert isinstance(cons_records[0]["consumed_monotonic_ns"], int)


def test_measure_consumer_lag_and_cli_args() -> None:
    from confluent_kafka import KafkaException

    class WatermarkConsumer:
        def __init__(self, fail: bool = False) -> None:
            self.fail = fail

        def get_watermark_offsets(
            self, _tp: Any, timeout: float = 1.0, cached: bool = False
        ) -> tuple[int, int]:
            if self.fail:
                raise KafkaException("unavailable")
            return (0, 10)

    assert _measure_consumer_lag(WatermarkConsumer(), "t", {}) is None  # type: ignore[arg-type]
    assert _measure_consumer_lag(WatermarkConsumer(), "t", {0: 9, 1: 7}) == {  # type: ignore[arg-type]
        "0": 0,
        "1": 2,
    }
    assert _measure_consumer_lag(WatermarkConsumer(fail=True), "t", {0: 5}) is None  # type: ignore[arg-type]

    p_args = parse_producer_args(["--count", "50", "--interval-ms", "1"])
    assert (p_args.count, p_args.interval_ms) == (50, 1.0)
    c_args = parse_consumer_args(["--limit", "50", "--group", "g1"])
    assert (c_args.limit, c_args.group) == (50, "g1")
