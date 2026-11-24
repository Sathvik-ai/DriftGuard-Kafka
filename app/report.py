"""Automated streaming experiment report generator for DriftGuard-Kafka M1.

Correlates producer delivery records and consumer receipt records by `event_id`
and produces:
  - report.json  (machine-readable metrics, metadata and boundaries)
  - report.md    (human-readable experiment summary)
  - events.csv   (per-event delivery & consumption summary)
  - latency.csv  (per-event end-to-end latency observations in ms)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from app.run_summary import SummaryError, load_summary
from app.schemas import format_utc_timestamp

MEASUREMENT_BOUNDARIES = [
    "End-to-end latency is measured as (consumed_monotonic_ns - enqueued_monotonic_ns) / 1e6 ms. "
    "This is valid because both processes run on the same host kernel and share a monotonic clock.",
    "Producer enqueue-to-consume latency includes librdkafka batching (linger.ms=5), "
    "broker append, and consumer poll interval.",
    "Consumer lag is queried from broker high-watermark offsets at consumer shutdown; if the "
    "broker is unreachable or watermark metadata is unavailable, lag is reported as "
    "null/unavailable rather than 0.",
    "Single-broker KRaft deployment with replication_factor=1 provides no distributed fault "
    "tolerance; results reflect local loopback / Docker bridge performance.",
]


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Load newline-delimited JSON objects from `path`."""
    if not path.exists():
        raise SummaryError(f"records file not found: {path}")
    records: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise SummaryError(f"invalid JSON in {path} line {line_no}: {error}") from error
        if not isinstance(item, dict):
            raise SummaryError(f"expected JSON object in {path} line {line_no}")
        records.append(item)
    return records


def compute_percentile(sorted_values: Sequence[float], percentile: float) -> float | None:
    """Compute linear-interpolated percentile (0..100) on pre-sorted values."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return round(sorted_values[0], 3)
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return round(sorted_values[lower], 3)
    fraction = rank - lower
    interpolated = sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])
    return round(interpolated, 3)


def compute_latency_stats(latencies_ms: Sequence[float]) -> dict[str, float | int | None]:
    """Return latency summary statistics, or None values if no samples exist."""
    if not latencies_ms:
        return {
            "count": 0,
            "min_ms": None,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
        }
    ordered = sorted(latencies_ms)
    return {
        "count": len(ordered),
        "min_ms": round(ordered[0], 3),
        "mean_ms": round(sum(ordered) / len(ordered), 3),
        "p50_ms": compute_percentile(ordered, 50.0),
        "p95_ms": compute_percentile(ordered, 95.0),
        "p99_ms": compute_percentile(ordered, 99.0),
        "max_ms": round(ordered[-1], 3),
    }


def _safe_rate(count: int, duration_s: float | None) -> float | None:
    if duration_s is None or duration_s <= 0:
        return None
    return round(count / duration_s, 2)


def _detect_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _detect_confluent_version() -> str | None:
    try:
        return version("confluent-kafka")
    except PackageNotFoundError:
        return None


def correlate_records(
    producer_records: Sequence[Mapping[str, Any]],
    consumer_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Correlate producer and consumer records by `event_id`.

    Returns:
        (event_rows, latency_rows, correlation_counts)
    """
    produced_by_id: dict[str, Mapping[str, Any]] = {}
    for record in producer_records:
        event_id = record.get("event_id")
        if isinstance(event_id, str) and event_id not in produced_by_id:
            produced_by_id[event_id] = record

    consumed_by_id: dict[str, Mapping[str, Any]] = {}
    for record in consumer_records:
        event_id = record.get("event_id")
        if isinstance(event_id, str) and event_id not in consumed_by_id:
            consumed_by_id[event_id] = record

    all_ids = sorted(set(produced_by_id) | set(consumed_by_id))
    event_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    matched = 0
    unmatched_produced = 0
    unmatched_consumed = 0

    for event_id in all_ids:
        prod = produced_by_id.get(event_id)
        cons = consumed_by_id.get(event_id)
        latency_ms: float | None = None

        if prod is not None and cons is not None:
            matched += 1
            enqueued_ns = prod.get("enqueued_monotonic_ns")
            consumed_ns = cons.get("consumed_monotonic_ns")
            if (
                isinstance(enqueued_ns, int)
                and isinstance(consumed_ns, int)
                and consumed_ns >= enqueued_ns
            ):
                latency_ms = round((consumed_ns - enqueued_ns) / 1_000_000.0, 3)
                latency_rows.append(
                    {
                        "event_id": event_id,
                        "machine_id": prod.get("machine_id") or cons.get("machine_id"),
                        "partition": cons.get("partition"),
                        "offset": cons.get("offset"),
                        "latency_ms": latency_ms,
                    }
                )
        elif prod is not None:
            unmatched_produced += 1
        else:
            unmatched_consumed += 1

        event_rows.append(
            {
                "event_id": event_id,
                "machine_id": (prod or cons or {}).get("machine_id"),
                "partition": (cons or prod or {}).get("partition"),
                "offset": (cons or prod or {}).get("offset"),
                "produced_timestamp": (prod or cons or {}).get("produced_timestamp"),
                "consumed_timestamp": cons.get("consumed_timestamp") if cons else None,
                "delivery_status": prod.get("delivery_status") if prod else "not_recorded",
                "consume_status": cons.get("consume_status") if cons else "not_consumed",
                "latency_ms": latency_ms,
            }
        )

    counts = {
        "matched_event_ids": matched,
        "unmatched_produced": unmatched_produced,
        "unmatched_consumed": unmatched_consumed,
    }
    return event_rows, latency_rows, counts


def build_report(
    run_id: str,
    expected_count: int,
    topic: str,
    bootstrap_servers: str,
    producer_summary: Mapping[str, Any],
    consumer_summary: Mapping[str, Any],
    producer_records: Sequence[Mapping[str, Any]],
    consumer_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build structured report dictionary plus CSV row lists."""
    event_rows, latency_rows, correlation = correlate_records(producer_records, consumer_records)
    latencies = [float(row["latency_ms"]) for row in latency_rows]
    latency_stats = compute_latency_stats(latencies)

    producer_duration_s = producer_summary.get("duration_s")
    consumer_duration_s = consumer_summary.get("duration_s")
    delivered = int(producer_summary["delivered"])
    received = int(consumer_summary["received"])

    raw_lag = consumer_summary.get("consumer_lag")
    total_lag: int | None = None
    if isinstance(raw_lag, Mapping):
        total_lag = sum(int(value) for value in raw_lag.values())

    report: dict[str, Any] = {
        "run_id": run_id,
        "generated_utc": format_utc_timestamp(datetime.now(timezone.utc)),
        "reproducibility": {
            "topic": topic,
            "bootstrap_servers": bootstrap_servers,
            "expected_events": expected_count,
            "python_version": platform.python_version(),
            "confluent_kafka_version": _detect_confluent_version(),
            "platform": platform.platform(),
            "machine_arch": platform.machine(),
            "git_commit": _detect_git_commit(),
        },
        "delivery": {
            "requested": int(producer_summary["requested"]),
            "attempted": int(producer_summary["attempted"]),
            "delivered": delivered,
            "failed": int(producer_summary["failed"]),
            "not_confirmed": int(producer_summary["not_confirmed"]),
            "producer_duration_s": producer_duration_s,
            "producer_throughput_events_per_sec": _safe_rate(delivered, producer_duration_s),
        },
        "consumption": {
            "received": received,
            "valid": int(consumer_summary["valid"]),
            "malformed": int(consumer_summary["malformed"]),
            "duplicates": int(consumer_summary["duplicates"]),
            "unique_event_ids": int(consumer_summary["unique_event_ids"]),
            "kafka_errors": int(consumer_summary["kafka_errors"]),
            "consumer_duration_s": consumer_duration_s,
            "consumer_throughput_events_per_sec": _safe_rate(received, consumer_duration_s),
            "consumer_lag_per_partition": raw_lag if isinstance(raw_lag, Mapping) else None,
            "consumer_lag_total": total_lag,
        },
        "correlation": correlation,
        "latency_ms": latency_stats,
        "per_partition": {
            "producer": dict(producer_summary["per_partition"]),
            "consumer": dict(consumer_summary["per_partition"]),
        },
        "measurement_boundaries": MEASUREMENT_BOUNDARIES,
    }
    return report, event_rows, latency_rows


def _fmt_metric(value: object, unit: str = "") -> str:
    if value is None:
        return "unavailable"
    return f"{value}{unit}"


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render a human-readable Markdown report from `report.json` data."""
    repro = report["reproducibility"]
    delivery = report["delivery"]
    consumption = report["consumption"]
    correlation = report["correlation"]
    latency = report["latency_ms"]
    prod_parts = report["per_partition"]["producer"]
    cons_parts = report["per_partition"]["consumer"]
    all_parts = sorted(set(prod_parts) | set(cons_parts))

    lines = [
        f"# DriftGuard-Kafka M1 Streaming Report — `{report['run_id']}`",
        "",
        f"Generated: `{report['generated_utc']}`",
        "",
        "## 1. Summary Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Expected Events | {repro['expected_events']} |",
        f"| Produced (Delivered / Failed / Unconfirmed) | {delivery['delivered']} / "
        f"{delivery['failed']} / {delivery['not_confirmed']} |",
        f"| Consumed (Received / Valid / Malformed / Duplicates) | {consumption['received']} / "
        f"{consumption['valid']} / {consumption['malformed']} / {consumption['duplicates']} |",
        f"| Unique Event IDs | {consumption['unique_event_ids']} |",
        f"| Correlated Event IDs | {correlation['matched_event_ids']} |",
        f"| Producer Duration | {_fmt_metric(delivery['producer_duration_s'], ' s')} |",
        f"| Consumer Duration | {_fmt_metric(consumption['consumer_duration_s'], ' s')} |",
        f"| Producer Throughput | "
        f"{_fmt_metric(delivery['producer_throughput_events_per_sec'], ' events/s')} |",
        f"| Consumer Throughput | "
        f"{_fmt_metric(consumption['consumer_throughput_events_per_sec'], ' events/s')} |",
        f"| Consumer Lag (Total) | {_fmt_metric(consumption['consumer_lag_total'])} |",
        "",
        "## 2. End-to-End Latency (ms)",
        "",
        "| Samples | Min | Mean | p50 | p95 | p99 | Max |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {latency['count']} | {_fmt_metric(latency['min_ms'])} | "
        f"{_fmt_metric(latency['mean_ms'])} | {_fmt_metric(latency['p50_ms'])} | "
        f"{_fmt_metric(latency['p95_ms'])} | {_fmt_metric(latency['p99_ms'])} | "
        f"{_fmt_metric(latency['max_ms'])} |",
        "",
        "## 3. Per-Partition Distribution",
        "",
        "| Partition | Producer Delivered | Consumer Received | Consumer Lag |",
        "|---:|---:|---:|---:|",
    ]
    lag_map = consumption.get("consumer_lag_per_partition") or {}
    for part in all_parts:
        lag_val = lag_map.get(part) if isinstance(lag_map, Mapping) else None
        lines.append(
            f"| {part} | {prod_parts.get(part, 0)} | {cons_parts.get(part, 0)} | "
            f"{_fmt_metric(lag_val)} |"
        )

    lines.extend(
        [
            "",
            "## 4. Reproducibility Metadata",
            "",
            f"- **Run ID**: `{report['run_id']}`",
            f"- **Topic**: `{repro['topic']}` (`{repro['bootstrap_servers']}`)",
            f"- **Python**: `{repro['python_version']}`",
            f"- **confluent-kafka**: `{_fmt_metric(repro['confluent_kafka_version'])}`",
            f"- **Platform**: `{repro['platform']}` (`{repro['machine_arch']}`)",
            f"- **Git Commit**: `{_fmt_metric(repro['git_commit'])}`",
            "",
            "## 5. Measurement Boundaries & Limitations",
            "",
        ]
    )
    for boundary in report["measurement_boundaries"]:
        lines.append(f"- {boundary}")
    lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report_bundle(
    output_dir: Path,
    report: Mapping[str, Any],
    event_rows: Sequence[Mapping[str, Any]],
    latency_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write report.json, report.md, events.csv and latency.csv to `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(render_markdown_report(report), encoding="utf-8")
    _write_csv(
        output_dir / "events.csv",
        [
            "event_id",
            "machine_id",
            "partition",
            "offset",
            "produced_timestamp",
            "consumed_timestamp",
            "delivery_status",
            "consume_status",
            "latency_ms",
        ],
        event_rows,
    )
    _write_csv(
        output_dir / "latency.csv",
        ["event_id", "machine_id", "partition", "offset", "latency_ms"],
        latency_rows,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate streaming experiment report.")
    parser.add_argument("--run-id", required=True, help="Unique identifier for the demo run")
    parser.add_argument("--expected", type=int, required=True, help="Expected event count")
    parser.add_argument("--topic", required=True, help="Kafka topic name")
    parser.add_argument("--bootstrap-servers", required=True, help="Kafka bootstrap servers")
    parser.add_argument("--producer-summary", type=Path, required=True)
    parser.add_argument("--consumer-summary", type=Path, required=True)
    parser.add_argument("--producer-records", type=Path, required=True)
    parser.add_argument("--consumer-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for generating a streaming report bundle."""
    args = _parse_args(argv)
    try:
        report, event_rows, latency_rows = build_report(
            run_id=args.run_id,
            expected_count=args.expected,
            topic=args.topic,
            bootstrap_servers=args.bootstrap_servers,
            producer_summary=load_summary(args.producer_summary),
            consumer_summary=load_summary(args.consumer_summary),
            producer_records=load_jsonl_records(args.producer_records),
            consumer_records=load_jsonl_records(args.consumer_records),
        )
    except SummaryError as error:
        print(f"REPORT GENERATION FAILED: {error}", file=sys.stderr)
        return 2
    write_report_bundle(args.output_dir, report, event_rows, latency_rows)
    print(f"Streaming report generated in {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
