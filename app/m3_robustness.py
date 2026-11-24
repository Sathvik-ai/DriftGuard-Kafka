"""Multi-seed robustness evaluation for DriftGuard-Kafka M3 ADWIN monitoring.

Executes `normal`, `drift` (gradual), and `abrupt_drift` across >= 10 reproducible
seeds (either end-to-end through Apache Kafka or in-process for unit tests),
recording ADWIN parameters, static threshold configuration, and scenario metadata
for every run, and reporting mean, standard deviation, and range for:
  - Detection rate
  - Per-machine sample detection delay
  - Stable-period false alarms
  - Missed drift channels
  - First-episode detections vs. repeated ADWIN window re-cuts
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from confluent_kafka import Consumer, Producer, TopicPartition

from app.config import Settings, load_settings, positive_float, positive_int
from app.consumer import build_consumer_config
from app.detector import SensorReading, ThresholdConfig, ThresholdDetector
from app.drift_monitor import (
    AdwinConfig,
    DriftMonitorStats,
    MachineAdwinMonitor,
    process_drift_stream,
)
from app.logging_config import configure_logging
from app.m3_report import _detect_river_version, evaluate_m3_scenario
from app.producer import DeliveryTracker, build_producer_config, publish_events
from app.report import _detect_confluent_version, _detect_git_commit
from app.schemas import GroundTruthRecord, format_utc_timestamp
from app.simulator import Scenario, TelemetrySimulator

DEFAULT_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51)
ROBUSTNESS_SCENARIOS: tuple[Scenario, ...] = (
    Scenario.NORMAL,
    Scenario.DRIFT,
    Scenario.ABRUPT_DRIFT,
)


def describe_scenario_config(
    scenario: Scenario, event_count: int, machine_count: int, seed: int
) -> dict[str, Any]:
    """Return explicit reproducible configuration metadata for one scenario run."""
    if scenario is Scenario.NORMAL:
        warmup = event_count
        profile = "stationary Gaussian baseline"
    elif scenario is Scenario.DRIFT:
        warmup = int(event_count * 0.3) + 1
        profile = "linear ramp (+32.0C temp, +4.5 mm/s vib, +3.5 bar press after 30% warmup)"
    else:
        warmup = event_count // 2
        profile = "abrupt step (+20.0C temp, +3.0 mm/s vib, +2.2 bar press at 50% stream)"
    return {
        "scenario": scenario.value,
        "seed": seed,
        "event_count": event_count,
        "machine_count": machine_count,
        "stable_warmup_events": warmup,
        "drift_profile": profile,
    }


def compute_series_dispersion(values: Sequence[float | int]) -> dict[str, float | int | None]:
    """Return count, mean, sample std (ddof=1, or 0.0 for n=1), min, and max."""
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    floats = [float(v) for v in values]
    std_val = statistics.stdev(floats) if len(floats) >= 2 else 0.0
    return {
        "count": len(floats),
        "mean": round(statistics.fmean(floats), 4),
        "std": round(std_val, 4),
        "min": round(min(floats), 4),
        "max": round(max(floats), 4),
    }


def run_single_inprocess(
    scenario: Scenario,
    seed: int,
    event_count: int,
    machine_count: int,
    adwin_config: AdwinConfig,
) -> dict[str, Any]:
    """Execute one `(scenario, seed)` run in-memory (used by fast unit tests)."""
    sim = TelemetrySimulator(machine_count=machine_count, seed=seed, scenario=scenario)
    monitor = MachineAdwinMonitor(adwin_config)
    static_detector = ThresholdDetector()

    prod_recs: list[dict[str, Any]] = []
    gt_recs: list[dict[str, Any]] = []
    drift_recs: list[dict[str, Any]] = []
    obs_recs: list[dict[str, Any]] = []
    static_alerts = 0

    t0 = time.monotonic()
    for ev, gt in sim.generate_scenario_events(event_count):
        mono_ns = time.monotonic_ns()
        prod_recs.append({"event_id": ev.event_id, "enqueued_monotonic_ns": mono_ns})
        gt_recs.append(gt.to_dict())

        reading = SensorReading.from_telemetry(ev)
        s_res = static_detector.evaluate(reading)
        if s_res.is_anomaly:
            static_alerts += 1

        s_pos, m_pos, d_events = monitor.update(reading)
        for d_ev in d_events:
            rec = d_ev.to_dict()
            rec["detected_monotonic_ns"] = mono_ns
            drift_recs.append(rec)

        obs_recs.append(
            {
                "event_id": reading.event_id,
                "machine_id": reading.machine_id,
                "timestamp": reading.timestamp,
                "stream_position": s_pos,
                "machine_position": m_pos,
                "temperature_c": reading.temperature_c,
                "vibration_mm_s": reading.vibration_mm_s,
                "pressure_bar": reading.pressure_bar,
                "static_alert": s_res.is_anomaly,
                "drift_features": [d.feature for d in d_events],
                "observed_monotonic_ns": mono_ns,
            }
        )
    elapsed = round(time.monotonic() - t0, 4)
    sc_report, _, _ = evaluate_m3_scenario(
        scenario=scenario.value,
        producer_summary={"delivered": event_count, "duration_s": elapsed},
        monitor_summary={
            "received": event_count,
            "malformed": 0,
            "static_alerts_observed": static_alerts,
            "drift_events_triggered": len(drift_recs),
            "drift_events_delivered": len(drift_recs),
            "drift_events_failed": 0,
            "duration_s": elapsed,
        },
        producer_records=prod_recs,
        ground_truth_records=gt_recs,
        drift_records=drift_recs,
        observation_records=obs_recs,
        drift_verified=len(drift_recs),
    )
    sc_report["scenario_config"] = describe_scenario_config(
        scenario, event_count, machine_count, seed
    )
    sc_report["adwin_config"] = asdict(adwin_config)
    return sc_report


def _high_watermarks(consumer: Consumer, topic: str, partitions: int = 3) -> list[TopicPartition]:
    tps: list[TopicPartition] = []
    for part in range(partitions):
        _low, high = consumer.get_watermark_offsets(
            TopicPartition(topic, part), timeout=5.0, cached=False
        )
        tps.append(TopicPartition(topic, part, high))
    return tps


def _count_verified_from_offsets(
    consumer: Consumer, start_tps: list[TopicPartition], expected: int, timeout_s: float = 10.0
) -> int:
    if expected == 0:
        return 0
    consumer.assign(start_tps)
    verified = 0
    required = {"drift_id", "event_id", "machine_id", "feature", "stream_position"}
    deadline = time.monotonic() + timeout_s
    while verified < expected and time.monotonic() < deadline:
        msg = consumer.poll(0.25)
        if msg is None or msg.error() is not None:
            continue
        raw = msg.value()
        data = json.loads(raw) if raw is not None else None
        if isinstance(data, dict) and required.issubset(data.keys()):
            verified += 1
    return verified


def run_single_kafka(
    settings: Settings,
    scenario: Scenario,
    seed: int,
    event_count: int,
    adwin_config: AdwinConfig,
) -> dict[str, Any]:
    """Execute one `(scenario, seed)` run end-to-end through the real Kafka broker."""
    run_tag = f"{scenario.value}-s{seed}-{int(time.time() * 1000)}"
    mon_settings = Settings(**{**asdict(settings), "consumer_group": f"dg-m3-rob-{run_tag}"})
    ver_settings = Settings(**{**asdict(settings), "consumer_group": f"dg-m3-ver-{run_tag}"})

    raw_consumer = Consumer(build_consumer_config(mon_settings))
    ver_consumer = Consumer(build_consumer_config(ver_settings))
    telemetry_producer = Producer(build_producer_config(settings))
    drift_producer = Producer(build_producer_config(settings))

    prod_recs: list[dict[str, Any]] = []
    gt_recs: list[dict[str, Any]] = []
    drift_recs: list[dict[str, Any]] = []
    obs_recs: list[dict[str, Any]] = []

    def _on_truth(gt: GroundTruthRecord) -> None:
        gt_recs.append(gt.to_dict())

    try:
        raw_start = _high_watermarks(raw_consumer, settings.topic)
        drift_start = _high_watermarks(ver_consumer, settings.drift_topic)
        raw_consumer.assign(raw_start)

        sim = TelemetrySimulator(machine_count=settings.machine_count, seed=seed, scenario=scenario)
        prod_tracker = DeliveryTracker()
        t_prod = time.monotonic()
        publish_events(
            telemetry_producer,
            settings.topic,
            sim,
            event_count,
            0.0,
            prod_tracker,
            on_record=prod_recs.append,
            on_ground_truth=_on_truth,
        )
        telemetry_producer.flush(15.0)
        prod_dur = round(time.monotonic() - t_prod, 4)

        monitor = MachineAdwinMonitor(adwin_config)
        stats = DriftMonitorStats()
        drift_tracker = DeliveryTracker()
        t_mon = time.monotonic()
        stop_reason = process_drift_stream(
            raw_consumer,
            drift_producer,
            settings.drift_topic,
            monitor,
            ThresholdDetector(),
            stats,
            drift_tracker,
            threading.Event(),
            limit=event_count,
            idle_timeout_s=15.0,
            on_drift=drift_recs.append,
            on_observation=obs_recs.append,
        )
        drift_producer.flush(15.0)
        mon_dur = round(time.monotonic() - t_mon, 4)

        verified = _count_verified_from_offsets(ver_consumer, drift_start, drift_tracker.delivered)
    finally:
        raw_consumer.close()
        ver_consumer.close()

    sc_report, _, _ = evaluate_m3_scenario(
        scenario=scenario.value,
        producer_summary={"delivered": prod_tracker.delivered, "duration_s": prod_dur},
        monitor_summary=stats.to_dict(drift_tracker, mon_dur, stop_reason, adwin_config.delta),
        producer_records=prod_recs,
        ground_truth_records=gt_recs,
        drift_records=drift_recs,
        observation_records=obs_recs,
        drift_verified=verified,
    )
    sc_report["scenario_config"] = describe_scenario_config(
        scenario, event_count, settings.machine_count, seed
    )
    sc_report["adwin_config"] = asdict(adwin_config)
    return sc_report


def aggregate_robustness_runs(
    run_id: str,
    seeds: Sequence[int],
    event_count: int,
    machine_count: int,
    adwin_config: AdwinConfig,
    threshold_config: ThresholdConfig,
    mode: str,
    runs_by_scenario: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Aggregate multi-seed scenario results into consolidated JSON + CSV rows."""
    scenario_summaries: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []

    for sc_name, sc_runs in runs_by_scenario.items():
        det_rates = [
            float(r["drift_metrics"]["detection_rate"])
            for r in sc_runs
            if r["drift_metrics"]["detection_rate"] is not None
        ]
        mean_delays = [
            float(r["delay_machine_samples"]["mean"])
            for r in sc_runs
            if r["delay_machine_samples"]["mean"] is not None
        ]
        false_alarms = [int(r["drift_metrics"]["false_alarms_stable"]) for r in sc_runs]
        missed = [int(r["drift_metrics"]["missed_drift_channels"]) for r in sc_runs]
        first_dets = [int(r["drift_metrics"]["first_episode_detections"]) for r in sc_runs]
        repeated = [int(r["drift_metrics"]["repeated_episode_alerts"]) for r in sc_runs]
        total_alerts = [int(r["drift_metrics"]["total_adwin_alerts"]) for r in sc_runs]
        static_fps = [int(r["events"]["static_alerts_observed"]) for r in sc_runs]

        scenario_summaries[sc_name] = {
            "seed_count": len(sc_runs),
            "detection_rate": compute_series_dispersion(det_rates),
            "delay_machine_samples": compute_series_dispersion(mean_delays),
            "false_alarms_stable": compute_series_dispersion(false_alarms),
            "missed_drift_channels": compute_series_dispersion(missed),
            "first_episode_detections": compute_series_dispersion(first_dets),
            "repeated_episode_alerts": compute_series_dispersion(repeated),
            "total_adwin_alerts": compute_series_dispersion(total_alerts),
            "unchanged_static_alerts": compute_series_dispersion(static_fps),
            "runs": list(sc_runs),
        }

        for r in sc_runs:
            dm = r["drift_metrics"]
            dly = r["delay_machine_samples"]
            dt = r["drift_topic"]
            cfg = r["scenario_config"]
            csv_rows.append(
                {
                    "scenario": sc_name,
                    "seed": cfg["seed"],
                    "event_count": cfg["event_count"],
                    "machine_count": cfg["machine_count"],
                    "adwin_delta": r["adwin_config"]["delta"],
                    "adwin_clock": r["adwin_config"]["clock"],
                    "expected_channels": dm["expected_drift_channels"],
                    "first_episode_detections": dm["first_episode_detections"],
                    "repeated_episode_alerts": dm["repeated_episode_alerts"],
                    "total_adwin_alerts": dm["total_adwin_alerts"],
                    "missed_drift_channels": dm["missed_drift_channels"],
                    "detection_rate": dm["detection_rate"],
                    "false_alarms_stable": dm["false_alarms_stable"],
                    "delay_mean_samples": dly["mean"],
                    "delay_min_samples": dly["min"],
                    "delay_max_samples": dly["max"],
                    "unchanged_static_alerts": r["events"]["static_alerts_observed"],
                    "kafka_verified": dt["verified_on_topic"],
                }
            )

    report = {
        "run_id": run_id,
        "generated_utc": format_utc_timestamp(datetime.now(timezone.utc)),
        "execution_mode": mode,
        "seeds": list(seeds),
        "events_per_run": event_count,
        "machine_count": machine_count,
        "adwin_parameters": asdict(adwin_config),
        "static_threshold_parameters": asdict(threshold_config),
        "reproducibility": {
            "python_version": platform.python_version(),
            "confluent_kafka_version": _detect_confluent_version(),
            "river_version": _detect_river_version(),
            "platform": platform.platform(),
            "machine_arch": platform.machine(),
            "git_commit": _detect_git_commit(),
        },
        "scenarios": scenario_summaries,
        "audit_notes": [
            "First detection per (machine, feature) drift episode (`first_episode_detections`) "
            "is tracked separately from subsequent window re-cuts within the same continuous "
            "ramp (`repeated_episode_alerts`). Total ADWIN alerts = first + repeated + stable_FA.",
            "Gradual drift (`Scenario.DRIFT`) ramps temperature by +32C (~21 sigma), causing "
            "ADWIN to cut once early in the ramp and re-cut a second time late in the ramp on "
            "steep temperature channels; these are genuine sequential cuts, not duplicates.",
            "Detection delay is measured strictly in per-machine samples from each machine's "
            "actual first post-onset observation, with per-machine FIFO order verified.",
            "ADWIN consumes unlabelled `SensorReading` objects only; ground truth never reaches "
            "the monitor.",
            "Static M2 anomaly thresholds remain completely unchanged across all runs.",
        ],
    }
    return report, csv_rows


def _fmt_disp(disp: Mapping[str, Any]) -> str:
    if disp["mean"] is None:
        return "unavailable"
    return f"{disp['mean']} ± {disp['std']} [{disp['min']}, {disp['max']}]"


def render_robustness_markdown(report: Mapping[str, Any]) -> str:
    """Render consolidated Markdown report for multi-seed M3 robustness."""
    adw = report["adwin_parameters"]
    sth = report["static_threshold_parameters"]
    seeds = report["seeds"]
    lines = [
        f"# DriftGuard-Kafka M3 Multi-Seed Robustness Report — `{report['run_id']}`",
        "",
        f"Generated: `{report['generated_utc']}` | Mode: `{report['execution_mode']}` | "
        f"Seeds (`n={len(seeds)}`): `{seeds}` | Events/run: `{report['events_per_run']}` | "
        f"Machines: `{report['machine_count']}`",
        "",
        "## 1. Fixed Model & Threshold Configuration",
        "",
        f"- **ADWIN Parameters**: `delta={adw['delta']}`, `clock={adw['clock']}`, "
        f"`max_buckets={adw['max_buckets']}`, `min_window_length={adw['min_window_length']}`, "
        f"`grace_period={adw['grace_period']}`",
        f"- **Unchanged Static M2 Thresholds**: `temp=[{sth['temp_min_c']}, {sth['temp_max_c']}]`, "
        f"`vib=[{sth['vib_min_mm_s']}, {sth['vib_max_mm_s']}]`, "
        f"`press=[{sth['press_min_bar']}, {sth['press_max_bar']}]`, "
        f"`stuck_window={sth['stuck_window']}`",
        "",
        "## 2. Multi-Seed Summary (`mean ± std [min, max]`)",
        "",
        "| Scenario | Seeds | Detection Rate | Delay (Machine Samples) | Stable False Alarms | "
        "Missed Channels | First Detections | Repeated Ramp Alerts | Total ADWIN Alerts | "
        "Unchanged Static Alerts |",
        "|---|---:|---|---|---|---|---|---|---|---|",
    ]
    for sc_name, sc_sum in report["scenarios"].items():
        lines.append(
            f"| `{sc_name}` | {sc_sum['seed_count']} | "
            f"{_fmt_disp(sc_sum['detection_rate'])} | "
            f"{_fmt_disp(sc_sum['delay_machine_samples'])} | "
            f"{_fmt_disp(sc_sum['false_alarms_stable'])} | "
            f"{_fmt_disp(sc_sum['missed_drift_channels'])} | "
            f"{_fmt_disp(sc_sum['first_episode_detections'])} | "
            f"{_fmt_disp(sc_sum['repeated_episode_alerts'])} | "
            f"{_fmt_disp(sc_sum['total_adwin_alerts'])} | "
            f"{_fmt_disp(sc_sum['unchanged_static_alerts'])} |"
        )

    lines.extend(["", "## 3. Audit Findings & Measurement Notes", ""])
    for note in report["audit_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def write_robustness_bundle(
    output_dir: Path,
    report: Mapping[str, Any],
    csv_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write robustness_report.json, robustness_report.md, and seed_runs.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "robustness_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "robustness_report.md").write_text(
        render_robustness_markdown(report), encoding="utf-8"
    )
    if csv_rows:
        with (output_dir / "seed_runs.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for M3 multi-seed robustness evaluation."""
    parser = argparse.ArgumentParser(description="Run multi-seed M3 ADWIN robustness benchmark.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--count", type=positive_int, default=500)
    parser.add_argument("--delta", type=positive_float, default=0.002)
    parser.add_argument("--clock", type=positive_int, default=1)
    parser.add_argument("--mode", choices=("kafka", "inprocess"), default="kafka")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for multi-seed M3 robustness experiments."""
    args = parse_args(argv)
    settings = load_settings()
    configure_logging(settings.log_level)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    adwin_cfg = AdwinConfig(delta=args.delta, clock=args.clock)

    runs_by_scenario: dict[str, list[dict[str, Any]]] = {
        sc.value: [] for sc in ROBUSTNESS_SCENARIOS
    }
    for scenario in ROBUSTNESS_SCENARIOS:
        for seed in args.seeds:
            if args.mode == "kafka":
                sc_res = run_single_kafka(settings, scenario, seed, args.count, adwin_cfg)
            else:
                sc_res = run_single_inprocess(
                    scenario, seed, args.count, settings.machine_count, adwin_cfg
                )
            runs_by_scenario[scenario.value].append(sc_res)

    report, csv_rows = aggregate_robustness_runs(
        run_id=run_id,
        seeds=args.seeds,
        event_count=args.count,
        machine_count=settings.machine_count,
        adwin_config=adwin_cfg,
        threshold_config=ThresholdConfig(),
        mode=args.mode,
        runs_by_scenario=runs_by_scenario,
    )
    write_robustness_bundle(args.output_dir, report, csv_rows)
    print(render_robustness_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
