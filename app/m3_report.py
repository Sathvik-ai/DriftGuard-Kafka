"""M3 evaluation report generator: evaluates River ADWIN concept drift monitoring.

Evaluates ADWIN drift detections against separate ground-truth regime metadata
across stable normal, gradual drift, and abrupt drift scenarios, generating:
  - report.json              (machine-readable M3 metrics & metadata)
  - report.md                (human-readable research summary)
  - channel_evaluations.csv  (per-(scenario, machine, feature) evaluation)
  - dashboard.html           (interactive streaming points & drift visualiser)
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from app.drift_monitor import MONITORED_FEATURES
from app.m3_dashboard import write_m3_dashboard
from app.report import (
    _detect_confluent_version,
    _detect_git_commit,
    _safe_rate,
    compute_latency_stats,
    compute_percentile,
    load_jsonl_records,
)
from app.run_summary import SummaryError, load_summary
from app.schemas import EventLabel, format_utc_timestamp


def _detect_river_version() -> str | None:
    try:
        return version("river")
    except PackageNotFoundError:
        return None


def _compute_int_stats(values: Sequence[int]) -> dict[str, float | int | None]:
    """Compute summary statistics over integer sample/event delays."""
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    ordered = [float(v) for v in sorted(values)]
    return {
        "count": len(ordered),
        "min": int(ordered[0]),
        "mean": round(sum(ordered) / len(ordered), 2),
        "p50": compute_percentile(ordered, 50.0),
        "p95": compute_percentile(ordered, 95.0),
        "max": int(ordered[-1]),
    }


def evaluate_m3_scenario(
    scenario: str,
    producer_summary: Mapping[str, Any],
    monitor_summary: Mapping[str, Any],
    producer_records: Sequence[Mapping[str, Any]],
    ground_truth_records: Sequence[Mapping[str, Any]],
    drift_records: Sequence[Mapping[str, Any]],
    observation_records: Sequence[Mapping[str, Any]],
    drift_verified: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate one M3 scenario against separate ground-truth regime metadata.

    Returns:
        `(scenario_report, channel_rows, enriched_stream_points)`
    """
    produced_by_id = {
        str(r["event_id"]): r for r in producer_records if isinstance(r.get("event_id"), str)
    }
    truth_by_id = {
        str(r["event_id"]): r for r in ground_truth_records if isinstance(r.get("event_id"), str)
    }

    # Validate that drifted observations retain normal ground-truth labels
    for event_id, truth in truth_by_id.items():
        if bool(truth.get("drift_active")) and (
            truth.get("ground_truth_label") != EventLabel.NORMAL.value
            or bool(truth.get("is_anomaly"))
        ):
            raise SummaryError(
                f"drifted observation {event_id} must retain normal ground-truth label"
            )

    # Track per-machine onset positions from consumer-ordered observations
    onset_by_machine: dict[
        str, tuple[int, int]
    ] = {}  # machine_id -> (machine_pos, prod_stream_idx)
    last_prod_idx_by_machine: dict[str, int] = {}
    machines_seen: set[str] = set()
    stable_obs_count = 0
    latencies_ms: list[float] = []
    stream_points: list[dict[str, Any]] = []

    for obs in observation_records:
        event_id = str(obs["event_id"])
        obs_truth = truth_by_id.get(event_id)
        if obs_truth is None:
            raise SummaryError(f"missing ground truth for observed event {event_id}")
        machine_id = str(obs["machine_id"])
        machines_seen.add(machine_id)
        machine_pos = int(obs["machine_position"])
        prod_stream_idx = int(obs_truth.get("stream_index", obs["stream_position"]))
        prev_prod_idx = last_prod_idx_by_machine.get(machine_id)
        if prev_prod_idx is not None and prod_stream_idx <= prev_prod_idx:
            raise SummaryError(
                f"out-of-order per-machine observation on {machine_id}: "
                f"{prod_stream_idx} after {prev_prod_idx}"
            )
        last_prod_idx_by_machine[machine_id] = prod_stream_idx
        drift_active = bool(obs_truth.get("drift_active", False))

        if not drift_active:
            stable_obs_count += 1
        elif machine_id not in onset_by_machine:
            onset_by_machine[machine_id] = (machine_pos, prod_stream_idx)

        prod = produced_by_id.get(event_id)
        if prod is not None:
            enq = prod.get("enqueued_monotonic_ns")
            obs_ns = obs.get("observed_monotonic_ns")
            if isinstance(enq, int) and isinstance(obs_ns, int) and obs_ns >= enq:
                latencies_ms.append(round((obs_ns - enq) / 1_000_000.0, 3))

        stream_points.append(
            {
                "event_id": event_id,
                "machine_id": machine_id,
                "stream_position": int(obs["stream_position"]),
                "machine_position": machine_pos,
                "temperature_c": float(obs["temperature_c"]),
                "vibration_mm_s": float(obs["vibration_mm_s"]),
                "pressure_bar": float(obs["pressure_bar"]),
                "ground_truth_label": str(obs_truth["ground_truth_label"]),
                "drift_active": drift_active,
                "static_alert": bool(obs.get("static_alert", False)),
                "drift_features": list(obs.get("drift_features", [])),
            }
        )

    # Group ADWIN detections by (machine_id, feature)
    drifts_by_channel: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for d_rec in drift_records:
        key = (str(d_rec["machine_id"]), str(d_rec["feature"]))
        drifts_by_channel[key].append(d_rec)

    expected_channels = 0
    detected_channels = 0
    repeated_episode_alerts = 0
    false_alarms_stable = 0
    delays_machine: list[int] = []
    delays_stream: list[int] = []
    channel_rows: list[dict[str, Any]] = []
    per_feature: dict[str, dict[str, Any]] = {
        feat: {
            "expected": 0,
            "detected": 0,
            "missed": 0,
            "repeated_episode_alerts": 0,
            "total_alerts": 0,
            "false_alarms_stable": 0,
            "delays": [],
        }
        for feat in MONITORED_FEATURES
    }

    for machine_id in sorted(machines_seen):
        onset = onset_by_machine.get(machine_id)
        for feature in MONITORED_FEATURES:
            ch_events = sorted(
                drifts_by_channel.get((machine_id, feature), []),
                key=lambda item: int(item["machine_position"]),
            )
            ch_false_alarms = 0
            ch_repeated = 0
            first_post_onset: Mapping[str, Any] | None = None

            for ev in ch_events:
                ev_id = str(ev["event_id"])
                ev_truth = truth_by_id.get(ev_id)
                if ev_truth is None:
                    raise SummaryError(f"missing ground truth for drift event {ev_id}")
                is_active = bool(ev_truth.get("drift_active", False))
                if not is_active:
                    ch_false_alarms += 1
                elif first_post_onset is None:
                    first_post_onset = ev
                else:
                    ch_repeated += 1

            false_alarms_stable += ch_false_alarms
            repeated_episode_alerts += ch_repeated
            per_feature[feature]["false_alarms_stable"] += ch_false_alarms
            per_feature[feature]["repeated_episode_alerts"] += ch_repeated
            per_feature[feature]["total_alerts"] += len(ch_events)

            if onset is None:
                channel_rows.append(
                    {
                        "scenario": scenario,
                        "machine_id": machine_id,
                        "feature": feature,
                        "expected_drift": False,
                        "onset_machine_pos": None,
                        "first_detect_machine_pos": None,
                        "delay_machine_samples": None,
                        "delay_stream_events": None,
                        "status": "stable_false_alarm" if ch_false_alarms > 0 else "stable_clean",
                        "repeated_episode_alerts": 0,
                        "total_alerts": len(ch_events),
                        "false_alarms_stable": ch_false_alarms,
                    }
                )
            else:
                expected_channels += 1
                per_feature[feature]["expected"] += 1
                onset_m_pos, onset_s_idx = onset
                if first_post_onset is not None:
                    detected_channels += 1
                    per_feature[feature]["detected"] += 1
                    det_m_pos = int(first_post_onset["machine_position"])
                    det_truth = truth_by_id[str(first_post_onset["event_id"])]
                    det_s_idx = int(
                        det_truth.get("stream_index", first_post_onset["stream_position"])
                    )
                    m_delay = max(0, det_m_pos - onset_m_pos)
                    s_delay = max(0, det_s_idx - onset_s_idx)
                    delays_machine.append(m_delay)
                    delays_stream.append(s_delay)
                    per_feature[feature]["delays"].append(m_delay)
                    channel_rows.append(
                        {
                            "scenario": scenario,
                            "machine_id": machine_id,
                            "feature": feature,
                            "expected_drift": True,
                            "onset_machine_pos": onset_m_pos,
                            "first_detect_machine_pos": det_m_pos,
                            "delay_machine_samples": m_delay,
                            "delay_stream_events": s_delay,
                            "status": "detected",
                            "repeated_episode_alerts": ch_repeated,
                            "total_alerts": len(ch_events),
                            "false_alarms_stable": ch_false_alarms,
                        }
                    )
                else:
                    per_feature[feature]["missed"] += 1
                    channel_rows.append(
                        {
                            "scenario": scenario,
                            "machine_id": machine_id,
                            "feature": feature,
                            "expected_drift": True,
                            "onset_machine_pos": onset_m_pos,
                            "first_detect_machine_pos": None,
                            "delay_machine_samples": None,
                            "delay_stream_events": None,
                            "status": "missed",
                            "repeated_episode_alerts": 0,
                            "total_alerts": len(ch_events),
                            "false_alarms_stable": ch_false_alarms,
                        }
                    )

    missed_channels = expected_channels - detected_channels
    total_adwin_alerts = detected_channels + repeated_episode_alerts + false_alarms_stable
    detection_rate = (
        round(detected_channels / expected_channels, 4) if expected_channels > 0 else None
    )
    stable_channel_obs = stable_obs_count * len(MONITORED_FEATURES)
    false_alarm_rate = (
        round(false_alarms_stable / stable_channel_obs, 6) if stable_channel_obs > 0 else None
    )

    feature_summary = {
        feat: {
            "expected": info["expected"],
            "detected": info["detected"],
            "missed": info["missed"],
            "first_episode_detections": info["detected"],
            "repeated_episode_alerts": info["repeated_episode_alerts"],
            "total_alerts": info["total_alerts"],
            "false_alarms_stable": info["false_alarms_stable"],
            "mean_delay_machine_samples": (
                round(sum(info["delays"]) / len(info["delays"]), 2) if info["delays"] else None
            ),
        }
        for feat, info in per_feature.items()
    }

    delivered = int(producer_summary["delivered"])
    received = int(monitor_summary["received"])
    prod_dur = producer_summary.get("duration_s")
    mon_dur = monitor_summary.get("duration_s")

    report = {
        "scenario": scenario,
        "events": {
            "delivered": delivered,
            "received": received,
            "malformed": int(monitor_summary["malformed"]),
            "stable_observations": stable_obs_count,
            "static_alerts_observed": int(monitor_summary["static_alerts_observed"]),
        },
        "drift_metrics": {
            "expected_drift_channels": expected_channels,
            "detected_drift_channels": detected_channels,
            "first_episode_detections": detected_channels,
            "repeated_episode_alerts": repeated_episode_alerts,
            "total_adwin_alerts": total_adwin_alerts,
            "missed_drift_channels": missed_channels,
            "detection_rate": detection_rate,
            "false_alarms_stable": false_alarms_stable,
            "false_alarm_rate_per_channel_obs": false_alarm_rate,
        },
        "delay_machine_samples": _compute_int_stats(delays_machine),
        "delay_stream_events": _compute_int_stats(delays_stream),
        "per_feature": feature_summary,
        "drift_topic": {
            "triggered": int(monitor_summary["drift_events_triggered"]),
            "delivered": int(monitor_summary["drift_events_delivered"]),
            "failed": int(monitor_summary["drift_events_failed"]),
            "verified_on_topic": drift_verified,
        },
        "throughput": {
            "producer_duration_s": prod_dur,
            "producer_events_per_sec": _safe_rate(delivered, prod_dur),
            "monitor_duration_s": mon_dur,
            "monitor_events_per_sec": _safe_rate(received, mon_dur),
        },
        "latency_ms": compute_latency_stats(latencies_ms),
    }
    return report, channel_rows, stream_points


def build_m3_report(
    run_id: str,
    scenario_results: Sequence[Mapping[str, Any]],
    seed: int | None,
    adwin_delta: float,
) -> dict[str, Any]:
    """Aggregate M3 scenario evaluations into the top-level report."""
    expected = sum(int(s["drift_metrics"]["expected_drift_channels"]) for s in scenario_results)
    detected = sum(int(s["drift_metrics"]["detected_drift_channels"]) for s in scenario_results)
    repeated = sum(int(s["drift_metrics"]["repeated_episode_alerts"]) for s in scenario_results)
    missed = sum(int(s["drift_metrics"]["missed_drift_channels"]) for s in scenario_results)
    false_alarms = sum(int(s["drift_metrics"]["false_alarms_stable"]) for s in scenario_results)
    trig = sum(int(s["drift_topic"]["triggered"]) for s in scenario_results)
    ver_list = [
        int(s["drift_topic"]["verified_on_topic"])
        for s in scenario_results
        if s["drift_topic"]["verified_on_topic"] is not None
    ]

    weighted_delay_num = sum(
        float(s["delay_machine_samples"]["mean"]) * int(s["delay_machine_samples"]["count"])
        for s in scenario_results
        if s["delay_machine_samples"]["mean"] is not None
    )
    delay_cnt = sum(int(s["delay_machine_samples"]["count"]) for s in scenario_results)

    return {
        "run_id": run_id,
        "generated_utc": format_utc_timestamp(datetime.now(timezone.utc)),
        "adwin_delta": adwin_delta,
        "reproducibility": {
            "seed": seed,
            "python_version": platform.python_version(),
            "confluent_kafka_version": _detect_confluent_version(),
            "river_version": _detect_river_version(),
            "platform": platform.platform(),
            "machine_arch": platform.machine(),
            "git_commit": _detect_git_commit(),
        },
        "overall_metrics": {
            "expected_drift_channels": expected,
            "detected_drift_channels": detected,
            "first_episode_detections": detected,
            "repeated_episode_alerts": repeated,
            "missed_drift_channels": missed,
            "detection_rate": round(detected / expected, 4) if expected > 0 else None,
            "false_alarms_stable": false_alarms,
            "mean_delay_machine_samples": (
                round(weighted_delay_num / delay_cnt, 2) if delay_cnt > 0 else None
            ),
            "total_drift_events_triggered": trig,
            "total_drift_events_verified": sum(ver_list)
            if len(ver_list) == len(scenario_results)
            else None,
        },
        "scenarios": list(scenario_results),
        "measurement_boundaries": [
            "Each (machine_id, feature) pair maintains an independent River ADWIN monitor "
            "consuming unlabelled telemetry only; ground truth never reaches ADWIN.",
            "Drifted observations retain normal ground-truth labels (is_anomaly=false).",
            "Detection delay is measured in per-machine samples (and stream events) from the "
            "first post-onset observation of that machine to ADWIN's first post-onset detection.",
            "Static M2 anomaly thresholds remain unchanged; no adaptive threshold retuning is "
            "performed in M3.",
        ],
    }


def _fmt(val: object, suffix: str = "") -> str:
    return "unavailable" if val is None else f"{val}{suffix}"


def render_m3_markdown(report: Mapping[str, Any]) -> str:
    """Render human-readable Markdown report for M3."""
    ov = report["overall_metrics"]
    repro = report["reproducibility"]
    lines = [
        f"# DriftGuard-Kafka M3 ADWIN Drift Report — `{report['run_id']}`",
        "",
        f"Generated: `{report['generated_utc']}` | Seed: `{_fmt(repro['seed'])}` | "
        f"ADWIN delta: `{report['adwin_delta']}` | River: `{_fmt(repro['river_version'])}`",
        "",
        "## 1. Overall Drift Monitoring Summary",
        "",
        "| Expected Channels | Detected | Missed | Detection Rate | Stable False Alarms | "
        "Mean Delay (Machine Samples) | Drift Events (Trig / Verified) |",
        "|---:|---:|---:|---:|---:|---:|---|",
        f"| {ov['expected_drift_channels']} | {ov['detected_drift_channels']} | "
        f"{ov['missed_drift_channels']} | {_fmt(ov['detection_rate'])} | "
        f"{ov['false_alarms_stable']} | {_fmt(ov['mean_delay_machine_samples'])} | "
        f"{ov['total_drift_events_triggered']} / {_fmt(ov['total_drift_events_verified'])} |",
        "",
        "## 2. Per-Scenario Results",
        "",
        "| Scenario | Events | Expected/Detected/Missed | Detection Rate | "
        "First / Repeated / Stable FA | Delay Mean / p50 / p95 (samples) | "
        "Unchanged Static Alerts | Drift Topic (Trig/Del/Ver) | Throughput |",
        "|---|---:|---|---:|---|---:|---:|---|---:|",
    ]
    for sc in report["scenarios"]:
        dm = sc["drift_metrics"]
        dly = sc["delay_machine_samples"]
        dt = sc["drift_topic"]
        ev = sc["events"]
        tp_info = sc["throughput"]
        ch_str = (
            f"{dm['expected_drift_channels']}/{dm['detected_drift_channels']}/"
            f"{dm['missed_drift_channels']}"
        )
        ep_str = (
            f"{dm['first_episode_detections']} / {dm['repeated_episode_alerts']} / "
            f"{dm['false_alarms_stable']}"
        )
        dly_str = f"{_fmt(dly['mean'])} / {_fmt(dly['p50'])} / {_fmt(dly['p95'])}"
        dt_str = f"{dt['triggered']}/{dt['delivered']}/{_fmt(dt['verified_on_topic'])}"
        lines.append(
            f"| {sc['scenario']} | {ev['received']} | {ch_str} | {_fmt(dm['detection_rate'])} | "
            f"{ep_str} | {dly_str} | {ev['static_alerts_observed']} | "
            f"{dt_str} | {_fmt(tp_info['monitor_events_per_sec'], ' ev/s')} |"
        )

    lines.extend(
        [
            "",
            "## 3. Per-Feature Breakdown",
            "",
            "| Scenario | Feature | Expected | First Detected | Missed | "
            "Repeated Alerts | Total Alerts | Stable False Alarms | Mean Delay (Machine Samples) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for sc in report["scenarios"]:
        for feat, finfo in sc["per_feature"].items():
            lines.append(
                f"| {sc['scenario']} | {feat} | {finfo['expected']} | {finfo['detected']} | "
                f"{finfo['missed']} | {finfo['repeated_episode_alerts']} | "
                f"{finfo['total_alerts']} | {finfo['false_alarms_stable']} | "
                f"{_fmt(finfo['mean_delay_machine_samples'])} |"
            )

    lines.extend(["", "## 4. Measurement Boundaries", ""])
    for note in report["measurement_boundaries"]:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## 5. Interactive Dashboard",
            "",
            "- Open `dashboard.html` in this directory to inspect per-machine streaming telemetry "
            "points, static threshold lines, drift onset boundaries, and ADWIN detections.",
            "",
        ]
    )
    return "\n".join(lines)


def write_m3_bundle(
    output_dir: Path,
    report: Mapping[str, Any],
    channel_rows: Sequence[Mapping[str, Any]],
    scenario_streams: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    """Write report.json, report.md, channel_evaluations.csv, and dashboard.html."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(render_m3_markdown(report), encoding="utf-8")
    with (output_dir / "channel_evaluations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "machine_id",
                "feature",
                "expected_drift",
                "onset_machine_pos",
                "first_detect_machine_pos",
                "delay_machine_samples",
                "delay_stream_events",
                "status",
                "repeated_episode_alerts",
                "total_alerts",
                "false_alarms_stable",
            ],
        )
        writer.writeheader()
        for row in channel_rows:
            writer.writerow(row)
    write_m3_dashboard(output_dir / "dashboard.html", report, scenario_streams)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate M3 ADWIN drift evaluation report.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--delta", type=float, default=0.002)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for M3 report and dashboard generation."""
    args = _parse_args(argv)
    scenario_results: list[dict[str, Any]] = []
    all_channel_rows: list[dict[str, Any]] = []
    scenario_streams: dict[str, list[dict[str, Any]]] = {}

    try:
        for scenario in args.scenarios:
            sc_dir = args.run_dir / scenario
            ver_path = sc_dir / "drift_verified.json"
            drift_verified: int | None = None
            if ver_path.exists():
                drift_verified = int(load_summary(ver_path)["received"])

            drift_path = sc_dir / "drift_events.jsonl"
            drift_recs = load_jsonl_records(drift_path) if drift_path.exists() else []

            sc_report, ch_rows, points = evaluate_m3_scenario(
                scenario=scenario,
                producer_summary=load_summary(sc_dir / "producer_summary.json"),
                monitor_summary=load_summary(sc_dir / "monitor_summary.json"),
                producer_records=load_jsonl_records(sc_dir / "producer_records.jsonl"),
                ground_truth_records=load_jsonl_records(sc_dir / "ground_truth.jsonl"),
                drift_records=drift_recs,
                observation_records=load_jsonl_records(sc_dir / "observations.jsonl"),
                drift_verified=drift_verified,
            )
            scenario_results.append(sc_report)
            all_channel_rows.extend(ch_rows)
            scenario_streams[scenario] = points
    except SummaryError as error:
        print(f"M3 REPORT FAILED: {error}", file=sys.stderr)
        return 2

    report = build_m3_report(args.run_id, scenario_results, args.seed, args.delta)
    write_m3_bundle(args.output_dir, report, all_channel_rows, scenario_streams)
    print(f"M3 drift report and dashboard generated in {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
