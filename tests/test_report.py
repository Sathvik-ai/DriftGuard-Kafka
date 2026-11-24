"""Unit tests for streaming metrics and report generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from app.report import (
    build_report,
    compute_latency_stats,
    compute_percentile,
    correlate_records,
    load_jsonl_records,
    main,
    render_markdown_report,
    write_report_bundle,
)
from app.run_summary import SummaryError, write_summary


def _producer_summary(**overrides: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "requested": 2,
        "attempted": 2,
        "delivered": 2,
        "failed": 0,
        "not_confirmed": 0,
        "per_partition": {"0": 1, "1": 1},
        "duration_s": 0.5,
    }
    summary.update(overrides)
    return summary


def _consumer_summary(**overrides: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "received": 2,
        "valid": 2,
        "malformed": 0,
        "duplicates": 0,
        "unique_event_ids": 2,
        "kafka_errors": 0,
        "per_partition": {"0": 1, "1": 1},
        "consumer_lag": {"0": 0, "1": 0},
        "duration_s": 1.0,
    }
    summary.update(overrides)
    return summary


def _sample_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    producer_records = [
        {
            "event_id": "id-1",
            "machine_id": "machine_01",
            "produced_timestamp": "2026-11-24T10:30:00.000Z",
            "enqueued_monotonic_ns": 1_000_000_000,
            "acked_monotonic_ns": 1_002_000_000,
            "delivery_status": "delivered",
            "partition": 0,
            "offset": 0,
        },
        {
            "event_id": "id-2",
            "machine_id": "machine_02",
            "produced_timestamp": "2026-11-24T10:30:00.010Z",
            "enqueued_monotonic_ns": 1_010_000_000,
            "acked_monotonic_ns": 1_015_000_000,
            "delivery_status": "delivered",
            "partition": 1,
            "offset": 0,
        },
    ]
    consumer_records = [
        {
            "event_id": "id-1",
            "machine_id": "machine_01",
            "produced_timestamp": "2026-11-24T10:30:00.000Z",
            "consumed_timestamp": "2026-11-24T10:30:00.004Z",
            "consumed_monotonic_ns": 1_004_000_000,
            "partition": 0,
            "offset": 0,
            "consume_status": "valid",
        },
        {
            "event_id": "id-2",
            "machine_id": "machine_02",
            "produced_timestamp": "2026-11-24T10:30:00.010Z",
            "consumed_timestamp": "2026-11-24T10:30:00.020Z",
            "consumed_monotonic_ns": 1_020_000_000,
            "partition": 1,
            "offset": 0,
            "consume_status": "valid",
        },
    ]
    return producer_records, consumer_records


class TestLatencyStats:
    def test_empty_latencies_return_none_metrics(self) -> None:
        stats = compute_latency_stats([])
        assert stats["count"] == 0
        assert stats["p50_ms"] is None
        assert stats["mean_ms"] is None

    def test_single_value_percentile(self) -> None:
        assert compute_percentile([4.25], 99.0) == 4.25

    def test_percentiles_and_mean_are_computed(self) -> None:
        stats = compute_latency_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats == {
            "count": 5,
            "min_ms": 1.0,
            "mean_ms": 3.0,
            "p50_ms": 3.0,
            "p95_ms": 4.8,
            "p99_ms": 4.96,
            "max_ms": 5.0,
        }


class TestCorrelationAndReport:
    def test_correlates_matched_and_unmatched_events(self) -> None:
        prod, cons = _sample_records()
        prod.append({"event_id": "id-only-prod", "machine_id": "machine_03"})
        cons.append({"event_id": "id-only-cons", "machine_id": "machine_04"})

        event_rows, latency_rows, counts = correlate_records(prod, cons)
        assert counts == {
            "matched_event_ids": 2,
            "unmatched_produced": 1,
            "unmatched_consumed": 1,
        }
        assert [row["latency_ms"] for row in latency_rows] == [4.0, 10.0]
        assert len(event_rows) == 4

    def test_build_report_distinguishes_unavailable_lag_from_zero(self) -> None:
        prod, cons = _sample_records()
        report, _, _ = build_report(
            run_id="run-1",
            expected_count=2,
            topic="research.telemetry.raw",
            bootstrap_servers="localhost:9092",
            producer_summary=_producer_summary(),
            consumer_summary=_consumer_summary(consumer_lag=None),
            producer_records=prod,
            consumer_records=cons,
        )
        assert report["consumption"]["consumer_lag_total"] is None
        assert report["consumption"]["consumer_lag_per_partition"] is None
        markdown = render_markdown_report(report)
        assert "| Consumer Lag (Total) | unavailable |" in markdown

    def test_write_report_bundle_creates_all_four_files(self, tmp_path: Path) -> None:
        prod, cons = _sample_records()
        report, event_rows, latency_rows = build_report(
            run_id="run-2",
            expected_count=2,
            topic="research.telemetry.raw",
            bootstrap_servers="localhost:9092",
            producer_summary=_producer_summary(),
            consumer_summary=_consumer_summary(),
            producer_records=prod,
            consumer_records=cons,
        )
        out_dir = tmp_path / "reports" / "runs" / "run-2"
        write_report_bundle(out_dir, report, event_rows, latency_rows)

        assert (out_dir / "report.json").exists()
        assert (out_dir / "report.md").exists()
        assert (out_dir / "events.csv").exists()
        assert (out_dir / "latency.csv").exists()

        parsed_json = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
        assert parsed_json["latency_ms"]["p50_ms"] == 7.0
        assert parsed_json["delivery"]["producer_throughput_events_per_sec"] == 4.0

        with (out_dir / "latency.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2
        assert rows[0]["event_id"] == "id-1"


class TestJsonlAndCli:
    def test_load_jsonl_rejects_missing_or_invalid(self, tmp_path: Path) -> None:
        with pytest.raises(SummaryError, match="not found"):
            load_jsonl_records(tmp_path / "missing.jsonl")

        bad = tmp_path / "bad.jsonl"
        bad.write_text("{broken\n", encoding="utf-8")
        with pytest.raises(SummaryError, match="invalid JSON"):
            load_jsonl_records(bad)

        non_obj = tmp_path / "arr.jsonl"
        non_obj.write_text("[1, 2]\n", encoding="utf-8")
        with pytest.raises(SummaryError, match="JSON object"):
            load_jsonl_records(non_obj)

    def test_cli_generates_bundle_and_handles_errors(self, tmp_path: Path) -> None:
        prod_sum = tmp_path / "p.json"
        cons_sum = tmp_path / "c.json"
        prod_rec = tmp_path / "p.jsonl"
        cons_rec = tmp_path / "c.jsonl"
        out_dir = tmp_path / "out"

        write_summary(prod_sum, _producer_summary())
        write_summary(cons_sum, _consumer_summary())
        prod, cons = _sample_records()
        prod_rec.write_text("\n".join(json.dumps(r) for r in prod) + "\n", encoding="utf-8")
        cons_rec.write_text("\n".join(json.dumps(r) for r in cons) + "\n", encoding="utf-8")

        exit_code = main(
            [
                "--run-id",
                "cli-run",
                "--expected",
                "2",
                "--topic",
                "research.telemetry.raw",
                "--bootstrap-servers",
                "localhost:9092",
                "--producer-summary",
                str(prod_sum),
                "--consumer-summary",
                str(cons_sum),
                "--producer-records",
                str(prod_rec),
                "--consumer-records",
                str(cons_rec),
                "--output-dir",
                str(out_dir),
            ]
        )
        assert exit_code == 0
        assert (out_dir / "report.md").exists()

        missing_code = main(
            [
                "--run-id",
                "cli-run",
                "--expected",
                "2",
                "--topic",
                "t",
                "--bootstrap-servers",
                "localhost:9092",
                "--producer-summary",
                str(tmp_path / "missing.json"),
                "--consumer-summary",
                str(cons_sum),
                "--producer-records",
                str(prod_rec),
                "--consumer-records",
                str(cons_rec),
                "--output-dir",
                str(out_dir),
            ]
        )
        assert missing_code == 2
