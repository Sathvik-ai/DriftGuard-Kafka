"""M2 evaluation report generator: compares detector outputs against ground truth.

Reuses `app.report` and `app.run_summary` utilities to produce:
  - report.json      (confusion matrix, precision, recall, F1, FPR, latency, throughput)
  - report.md        (readable evaluation summary across scenarios)
  - evaluations.csv  (per-event ground truth vs. prediction table)
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.report import (
    _detect_confluent_version,
    _detect_git_commit,
    _safe_rate,
    compute_latency_stats,
    load_jsonl_records,
)
from app.run_summary import SummaryError, load_summary
from app.schemas import EventLabel, format_utc_timestamp


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def compute_classification_metrics(
    tp: int, fp: int, tn: int, fn: int
) -> dict[str, int | float | None]:
    """Compute confusion matrix and binary classification metrics.

    - Precision = TP / (TP + FP), or None when TP + FP == 0.
    - Recall    = TP / (TP + FN), or None when TP + FN == 0.
    - F1        = 2*TP / (2*TP + FP + FN), or None only when 2*TP + FP + FN == 0.
    - FPR       = FP / (FP + TN), or None when FP + TN == 0.
    """
    if min(tp, fp, tn, fn) < 0:
        raise ValueError("confusion matrix counts must be non-negative")
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)
    fpr = _safe_ratio(fp, fp + tn)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
    }


def evaluate_scenario(
    scenario: str,
    producer_summary: Mapping[str, Any],
    detector_summary: Mapping[str, Any],
    producer_records: Sequence[Mapping[str, Any]],
    ground_truth_records: Sequence[Mapping[str, Any]],
    detection_records: Sequence[Mapping[str, Any]],
    alerts_verified: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one scenario run against its separate ground-truth records."""
    produced_by_id = {
        str(r["event_id"]): r for r in producer_records if isinstance(r.get("event_id"), str)
    }
    truth_by_id = {
        str(r["event_id"]): r for r in ground_truth_records if isinstance(r.get("event_id"), str)
    }
    det_by_id = {
        str(r["event_id"]): r for r in detection_records if isinstance(r.get("event_id"), str)
    }

    tp = fp = tn = fn = 0
    unmatched = 0
    by_label: dict[str, dict[str, int]] = {}
    by_reason: Counter[str] = Counter()
    latencies_ms: list[float] = []
    eval_rows: list[dict[str, Any]] = []

    for event_id in sorted(truth_by_id):
        truth = truth_by_id[event_id]
        det = det_by_id.get(event_id)
        prod = produced_by_id.get(event_id)
        label = str(truth["ground_truth_label"])
        is_anomaly = bool(truth["is_anomaly"])
        expected_is_anomaly = label != EventLabel.NORMAL.value
        if is_anomaly != expected_is_anomaly:
            raise SummaryError(
                f"inconsistent ground truth for {event_id}: "
                f"label={label!r} but is_anomaly={is_anomaly}"
            )

        bucket = by_label.setdefault(
            label,
            {
                "is_anomaly": is_anomaly,
                "total": 0,
                "flagged": 0,
                "unflagged": 0,
                "detected": 0,
                "missed": 0,
                "false_positives": 0,
                "true_negatives": 0,
            },
        )
        bucket["total"] += 1

        if det is None:
            unmatched += 1
            continue

        predicted = bool(det["predicted_anomaly"])
        reason = str(det["reason"])
        by_reason[reason] += 1
        if predicted:
            bucket["flagged"] += 1
        else:
            bucket["unflagged"] += 1

        if is_anomaly and predicted:
            tp += 1
            outcome = "TP"
            bucket["detected"] += 1
        elif not is_anomaly and predicted:
            fp += 1
            outcome = "FP"
            bucket["false_positives"] += 1
        elif not is_anomaly and not predicted:
            tn += 1
            outcome = "TN"
            bucket["true_negatives"] += 1
        else:
            fn += 1
            outcome = "FN"
            bucket["missed"] += 1

        latency_ms: float | None = None
        if prod is not None:
            enq = prod.get("enqueued_monotonic_ns")
            det_ns = det.get("detected_monotonic_ns")
            if isinstance(enq, int) and isinstance(det_ns, int) and det_ns >= enq:
                latency_ms = round((det_ns - enq) / 1_000_000.0, 3)
                latencies_ms.append(latency_ms)

        eval_rows.append(
            {
                "scenario": scenario,
                "event_id": event_id,
                "machine_id": truth["machine_id"],
                "ground_truth_label": label,
                "is_anomaly": is_anomaly,
                "predicted_anomaly": predicted,
                "outcome": outcome,
                "reason": reason,
                "latency_ms": latency_ms,
            }
        )

    delivered = int(producer_summary["delivered"])
    received = int(detector_summary["received"])
    prod_dur = producer_summary.get("duration_s")
    det_dur = detector_summary.get("duration_s")

    result = {
        "scenario": scenario,
        "events": {
            "delivered": delivered,
            "received": received,
            "matched": tp + fp + tn + fn,
            "unmatched": unmatched,
            "malformed": int(detector_summary["malformed"]),
        },
        "metrics": compute_classification_metrics(tp, fp, tn, fn),
        "anomaly_breakdown": by_label,
        "detector_reasons": dict(by_reason),
        "alerts": {
            "triggered": int(detector_summary["alerts_triggered"]),
            "delivered": int(detector_summary["alerts_delivered"]),
            "failed": int(detector_summary["alerts_failed"]),
            "verified_on_topic": alerts_verified,
        },
        "throughput": {
            "producer_duration_s": prod_dur,
            "producer_events_per_sec": _safe_rate(delivered, prod_dur),
            "detector_duration_s": det_dur,
            "detector_events_per_sec": _safe_rate(received, det_dur),
        },
        "latency_ms": compute_latency_stats(latencies_ms),
    }
    return result, eval_rows


def build_m2_report(
    run_id: str,
    scenario_results: Sequence[Mapping[str, Any]],
    seed: int | None,
) -> dict[str, Any]:
    """Aggregate scenario evaluations into the top-level M2 report."""
    tp = sum(int(s["metrics"]["tp"]) for s in scenario_results)
    fp = sum(int(s["metrics"]["fp"]) for s in scenario_results)
    tn = sum(int(s["metrics"]["tn"]) for s in scenario_results)
    fn = sum(int(s["metrics"]["fn"]) for s in scenario_results)

    return {
        "run_id": run_id,
        "generated_utc": format_utc_timestamp(datetime.now(timezone.utc)),
        "reproducibility": {
            "seed": seed,
            "python_version": platform.python_version(),
            "confluent_kafka_version": _detect_confluent_version(),
            "platform": platform.platform(),
            "machine_arch": platform.machine(),
            "git_commit": _detect_git_commit(),
        },
        "overall_metrics": compute_classification_metrics(tp, fp, tn, fn),
        "scenarios": list(scenario_results),
        "measurement_boundaries": [
            "Ground-truth labels are stored separately from Kafka payloads and never exposed "
            "to the detector.",
            "Gradual concept drift shifts normal operating baselines and retains 'normal' "
            "ground-truth labels (is_anomaly=false); alerts fired during pure drift are "
            "counted as false positives (FP).",
            "Precision, recall, F1, and FPR are reported as null/unavailable when their "
            "denominators are zero (e.g. recall in normal or pure-drift scenarios).",
            "Detection latency is measured as "
            "(detected_monotonic_ns - enqueued_monotonic_ns) / 1e6 ms "
            "on the same host monotonic clock.",
        ],
    }


def _fmt(val: object, suffix: str = "") -> str:
    return "unavailable" if val is None else f"{val}{suffix}"


def render_m2_markdown(report: Mapping[str, Any]) -> str:
    """Render readable M2 evaluation report in Markdown."""
    overall = report["overall_metrics"]
    repro = report["reproducibility"]
    lines = [
        f"# DriftGuard-Kafka M2 Evaluation Report — `{report['run_id']}`",
        "",
        f"Generated: `{report['generated_utc']}` | Seed: `{_fmt(repro['seed'])}`",
        "",
        "## 1. Overall Detection Summary",
        "",
        "| TP | FP | TN | FN | Precision | Recall | F1 | FPR |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {overall['tp']} | {overall['fp']} | {overall['tn']} | {overall['fn']} | "
        f"{_fmt(overall['precision'])} | {_fmt(overall['recall'])} | "
        f"{_fmt(overall['f1'])} | {_fmt(overall['false_positive_rate'])} |",
        "",
        "## 2. Per-Scenario Results",
        "",
        "| Scenario | Events | TP/FP/TN/FN | Precision | Recall | F1 | FPR | "
        "Alerts (Trig/Del/Ver) | Detector Throughput | p50 / p95 / p99 ms |",
        "|---|---:|---|---:|---:|---:|---:|---|---:|---|",
    ]

    for sc in report["scenarios"]:
        m = sc["metrics"]
        ev = sc["events"]
        al = sc["alerts"]
        tp_info = sc["throughput"]
        lat = sc["latency_ms"]
        cm_str = f"{m['tp']}/{m['fp']}/{m['tn']}/{m['fn']}"
        al_str = f"{al['triggered']}/{al['delivered']}/{_fmt(al['verified_on_topic'])}"
        lat_str = f"{_fmt(lat['p50_ms'])} / {_fmt(lat['p95_ms'])} / {_fmt(lat['p99_ms'])}"
        lines.append(
            f"| {sc['scenario']} | {ev['matched']} | {cm_str} | "
            f"{_fmt(m['precision'])} | {_fmt(m['recall'])} | {_fmt(m['f1'])} | "
            f"{_fmt(m['false_positive_rate'])} | {al_str} | "
            f"{_fmt(tp_info['detector_events_per_sec'], ' ev/s')} | {lat_str} |"
        )

    lines.extend(["", "## 3. Ground-Truth Label & Anomaly Breakdown", ""])
    lines.append(
        "| Scenario | Label | Ground Truth | Total | Flagged (Alert) | Unflagged | Detail |"
    )
    lines.append("|---|---|---|---:|---:|---:|---|")
    for sc in report["scenarios"]:
        for label, counts in sorted(sc["anomaly_breakdown"].items()):
            gt_kind = "anomaly" if counts["is_anomaly"] else "normal"
            if counts["is_anomaly"]:
                detail = f"TP={counts['detected']}, FN={counts['missed']}"
            else:
                detail = f"TN={counts['true_negatives']}, FP={counts['false_positives']}"
            lines.append(
                f"| {sc['scenario']} | {label} | {gt_kind} | {counts['total']} | "
                f"{counts['flagged']} | {counts['unflagged']} | {detail} |"
            )

    lines.extend(["", "## 4. Measurement Boundaries", ""])
    for note in report["measurement_boundaries"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def write_m2_bundle(
    output_dir: Path, report: Mapping[str, Any], eval_rows: Sequence[Mapping[str, Any]]
) -> None:
    """Write report.json, report.md, and evaluations.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(render_m2_markdown(report), encoding="utf-8")
    with (output_dir / "evaluations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "event_id",
                "machine_id",
                "ground_truth_label",
                "is_anomaly",
                "predicted_anomaly",
                "outcome",
                "reason",
                "latency_ms",
            ],
        )
        writer.writeheader()
        for row in eval_rows:
            writer.writerow(row)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate M2 anomaly evaluation report.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="Directory with scenario subdirs"
    )
    parser.add_argument("--scenarios", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for M2 report generation."""
    args = _parse_args(argv)
    scenario_results: list[dict[str, Any]] = []
    all_eval_rows: list[dict[str, Any]] = []

    try:
        for scenario in args.scenarios:
            sc_dir = args.run_dir / scenario
            alerts_ver_path = sc_dir / "alerts_verified.json"
            alerts_verified: int | None = None
            if alerts_ver_path.exists():
                alerts_verified = int(load_summary(alerts_ver_path)["received"])

            sc_result, sc_rows = evaluate_scenario(
                scenario=scenario,
                producer_summary=load_summary(sc_dir / "producer_summary.json"),
                detector_summary=load_summary(sc_dir / "detector_summary.json"),
                producer_records=load_jsonl_records(sc_dir / "producer_records.jsonl"),
                ground_truth_records=load_jsonl_records(sc_dir / "ground_truth.jsonl"),
                detection_records=load_jsonl_records(sc_dir / "detections.jsonl"),
                alerts_verified=alerts_verified,
            )
            scenario_results.append(sc_result)
            all_eval_rows.extend(sc_rows)
    except SummaryError as error:
        print(f"M2 REPORT FAILED: {error}", file=sys.stderr)
        return 2

    report = build_m2_report(args.run_id, scenario_results, args.seed)
    write_m2_bundle(args.output_dir, report, all_eval_rows)
    print(f"M2 evaluation report generated in {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
