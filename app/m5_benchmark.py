"""Comparative M5 benchmark runner: LSTM Autoencoder vs. M4 baselines.

Reuses the M4 dataset loaders (`app.datasets`), classification & ranking metrics
(`app.m2_report`, `app.m4_benchmark`), and baseline detectors (`app.m4_models`)
to evaluate the sequence-to-sequence PyTorch `LSTMAutoencoderDetector` on:
  - **SKAB**: multivariate sensor windows (`channel_mode="multivariate"`).
  - **TSB-AD-M**: independent univariate series (`channel_mode="independent_univariate"`),
    honouring Research Rule #10 (never falsely representing heterogeneous TSB
    channels as coupled multivariate sensor vectors).

Measures Precision, Recall, F1, FPR, PR-AUC, VUS-PR, episode detection delay,
inference latency, throughput, training duration, and peak memory, writing:
  - `m5_report.json`
  - `m5_report.md`
  - `comparison_summary.csv`
  - `per_experiment_metrics.csv`
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import platform
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.datasets import (
    ChronologicalSplit,
    DatasetFormatError,
    DatasetNotAvailableError,
    DatasetSuiteBundle,
    load_skab_suite,
    load_tsb_ad_m_suite,
)
from app.m2_report import compute_classification_metrics
from app.m4_benchmark import _mean_optional, compute_ranking_metrics
from app.m4_models import BenchmarkDetector, build_default_detectors
from app.m5_lstm_autoencoder import (
    LSTMAutoencoderConfig,
    LSTMAutoencoderDetector,
    select_compute_device,
)
from app.report import _detect_git_commit
from app.schemas import format_utc_timestamp

IntArray = NDArray[np.int64]

M5_MEASUREMENT_BOUNDARIES: list[str] = [
    "Normalization (`TrainingOnlyScaler`) is fit strictly on normal training observations "
    "(`x_train_normal`) only; validation and test splits never influence scaler statistics.",
    "The LSTM Autoencoder is trained strictly on contiguous normal windows generated within "
    "each individual experiment/series boundary; windows never cross machine, series, or "
    "train/validation/test boundaries.",
    "Decision thresholds are calibrated strictly at `calibration_quantile` on normal validation "
    "reconstruction error scores (`x_val_normal`), never using test labels.",
    "SKAB is modeled in `multivariate` mode (`input_dim=8` coupled testbed sensors), whereas "
    "TSB series are modeled in `independent_univariate` mode (`input_dim=1` per channel) so "
    "heterogeneous or command-indicator TSB columns are never falsely represented as coupled "
    "multivariate vectors.",
    "Episode detection delay is measured in samples (`first_alert_idx - episode_start_idx`) "
    "across contiguous true anomaly episodes `[start, end]`; missed episodes are counted "
    "separately and never fabricated as zero delay.",
    "All M1–M4 pipelines and static/ADWIN components remain untouched; model checkpoints and "
    "benchmark reports are excluded from Git.",
]


def _pkg_version(dist_name: str) -> str | None:
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return None


def extract_anomaly_episodes(y_true: IntArray) -> list[tuple[int, int]]:
    """Return contiguous `(start_idx, end_idx)` inclusive anomaly intervals."""
    if y_true.size == 0:
        return []
    padded = np.pad((y_true > 0).astype(np.int64), (1, 1), constant_values=0)
    diffs = np.diff(padded)
    starts = np.flatnonzero(diffs == 1)
    ends = np.flatnonzero(diffs == -1) - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def compute_detection_delay(
    y_true: IntArray,
    predictions: IntArray,
) -> dict[str, int | float | None]:
    """Compute episode-level detection delay (in samples) and episode recall."""
    episodes = extract_anomaly_episodes(y_true)
    if not episodes:
        return {
            "total_episodes": 0,
            "detected_episodes": 0,
            "missed_episodes": 0,
            "episode_detection_rate": None,
            "mean_detection_delay_samples": None,
        }

    delays: list[int] = []
    missed = 0
    for start, end in episodes:
        rel_hits = np.flatnonzero(predictions[start : end + 1] > 0)
        if rel_hits.size > 0:
            delays.append(int(rel_hits[0]))
        else:
            missed += 1

    detected = len(delays)
    total = len(episodes)
    mean_delay = round(sum(delays) / detected, 4) if detected > 0 else None
    return {
        "total_episodes": total,
        "detected_episodes": detected,
        "missed_episodes": missed,
        "episode_detection_rate": round(detected / total, 4),
        "mean_detection_delay_samples": mean_delay,
    }


def evaluate_m5_detector_on_split(
    detector: BenchmarkDetector,
    split: ChronologicalSplit,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    """Fit/calibrate `detector` on `split` and measure full M5 metric suite."""
    cal = detector.fit_and_calibrate(
        split.x_train_normal,
        split.x_val_normal,
        split.feature_names,
    )
    if checkpoint_dir is not None and isinstance(detector, LSTMAutoencoderDetector):
        safe_id = split.experiment_id.replace("/", "__").replace(".csv", "")
        detector.save_checkpoint(checkpoint_dir / split.dataset_suite / f"{safe_id}.pt")

    inf = detector.score_and_predict(split.x_test, split.feature_names)
    y_true = split.y_test_anomaly
    preds = inf.predictions

    tp = int(np.sum((y_true == 1) & (preds == 1)))
    fp = int(np.sum((y_true == 0) & (preds == 1)))
    tn = int(np.sum((y_true == 0) & (preds == 0)))
    fn = int(np.sum((y_true == 1) & (preds == 0)))

    cls_metrics = compute_classification_metrics(tp, fp, tn, fn)
    rank_metrics = compute_ranking_metrics(
        y_true=y_true,
        scores=inf.scores,
        predictions=preds,
        sliding_window=split.sliding_window,
    )
    delay_metrics = compute_detection_delay(y_true=y_true, predictions=preds)

    return {
        "dataset_suite": split.dataset_suite,
        "experiment_id": split.experiment_id,
        "domain": split.domain,
        "is_anomaly_free": split.is_anomaly_free,
        "model": detector.name,
        "channel_mode": inf.extra_diagnostics.get("channel_mode", "multivariate_baseline"),
        "selected_device": inf.extra_diagnostics.get("selected_device", "cpu"),
        "n_features": len(split.feature_names),
        "train_normal_samples": cal.train_samples,
        "val_normal_samples": cal.val_samples,
        "test_samples": inf.test_samples,
        "test_anomalies": int(np.sum(y_true)),
        "calibrated_threshold": cal.calibrated_threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": cls_metrics["precision"],
        "recall": cls_metrics["recall"],
        "f1": cls_metrics["f1"],
        "false_positive_rate": cls_metrics["false_positive_rate"],
        "pr_auc": rank_metrics["pr_auc"],
        "vus_pr": rank_metrics["vus_pr"],
        "vus_roc": rank_metrics["vus_roc"],
        "total_episodes": delay_metrics["total_episodes"],
        "detected_episodes": delay_metrics["detected_episodes"],
        "missed_episodes": delay_metrics["missed_episodes"],
        "mean_detection_delay_samples": delay_metrics["mean_detection_delay_samples"],
        "training_duration_s": cal.fit_wall_time_s,
        "inference_wall_time_s": inf.inference_wall_time_s,
        "mean_latency_ms": inf.mean_latency_ms,
        "throughput_events_per_sec": inf.throughput_events_per_sec,
        "cpu_time_s": inf.cpu_time_s,
        "peak_tracemalloc_kib": inf.peak_tracemalloc_kib,
        "rss_memory_mib": inf.rss_memory_mib,
        "top_error_feature": inf.extra_diagnostics.get("top_error_feature"),
        "epochs_trained": inf.extra_diagnostics.get("epochs_trained"),
    }


def aggregate_m5_rows(
    dataset_suite: str,
    evaluation_scope: str,
    model_name: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-experiment M5 rows into suite-level comparison metrics."""
    tp = sum(int(r["tp"]) for r in rows)
    fp = sum(int(r["fp"]) for r in rows)
    tn = sum(int(r["tn"]) for r in rows)
    fn = sum(int(r["fn"]) for r in rows)
    micro = compute_classification_metrics(tp, fp, tn, fn)

    total_test = sum(int(r["test_samples"]) for r in rows)
    total_train_s = sum(float(r["training_duration_s"]) for r in rows)
    total_inf_s = sum(float(r["inference_wall_time_s"]) for r in rows)
    throughput = round(total_test / total_inf_s, 2) if total_inf_s > 0 else None
    mean_lat_ms = round((total_inf_s * 1000.0) / total_test, 6) if total_test > 0 else None

    tot_ep = sum(int(r["total_episodes"]) for r in rows)
    det_ep = sum(int(r["detected_episodes"]) for r in rows)
    miss_ep = sum(int(r["missed_episodes"]) for r in rows)
    ep_rate = round(det_ep / tot_ep, 4) if tot_ep > 0 else None

    return {
        "dataset_suite": dataset_suite,
        "evaluation_scope": evaluation_scope,
        "model": model_name,
        "experiments": len(rows),
        "test_samples": total_test,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        "micro_f1": micro["f1"],
        "micro_fpr": micro["false_positive_rate"],
        "macro_precision": _mean_optional([r["precision"] for r in rows]),
        "macro_recall": _mean_optional([r["recall"] for r in rows]),
        "macro_f1": _mean_optional([r["f1"] for r in rows]),
        "macro_fpr": _mean_optional([r["false_positive_rate"] for r in rows]),
        "mean_pr_auc": _mean_optional([r["pr_auc"] for r in rows]),
        "mean_vus_pr": _mean_optional([r["vus_pr"] for r in rows]),
        "total_episodes": tot_ep,
        "detected_episodes": det_ep,
        "missed_episodes": miss_ep,
        "episode_detection_rate": ep_rate,
        "mean_detection_delay_samples": _mean_optional(
            [r["mean_detection_delay_samples"] for r in rows]
        ),
        "total_training_duration_s": round(total_train_s, 4),
        "total_inference_wall_s": round(total_inf_s, 4),
        "mean_latency_ms": mean_lat_ms,
        "throughput_events_per_sec": throughput,
        "max_peak_tracemalloc_kib": round(
            max((float(r["peak_tracemalloc_kib"]) for r in rows), default=0.0), 3
        ),
        "max_rss_mib": round(max((float(r["rss_memory_mib"]) for r in rows), default=0.0), 3),
    }


def run_m5_suite_comparison(
    bundle: DatasetSuiteBundle,
    base_lstm_cfg: LSTMAutoencoderConfig,
    skab_mode: str = "multivariate",
    tsb_mode: str = "independent_univariate",
    include_m4_baselines: bool = True,
    max_experiments: int | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Evaluate LSTM Autoencoder alongside M4 baselines on `bundle`."""
    splits = bundle.splits[:max_experiments] if max_experiments else bundle.splits
    mode = skab_mode if bundle.suite_name == "SKAB" else tsb_mode
    suite_cfg = dataclasses.replace(
        base_lstm_cfg,
        channel_mode=mode,  # type: ignore[arg-type]
    )

    per_exp_rows: list[dict[str, Any]] = []
    model_params: dict[str, Any] = {}

    for split in splits:
        detectors: list[BenchmarkDetector] = []
        if include_m4_baselines:
            detectors.extend(
                build_default_detectors(
                    seed=suite_cfg.seed,
                    calibration_quantile=suite_cfg.calibration_quantile,
                )
            )
        detectors.append(LSTMAutoencoderDetector(suite_cfg))

        for det in detectors:
            row = evaluate_m5_detector_on_split(det, split, checkpoint_dir=checkpoint_dir)
            per_exp_rows.append(row)
            key = f"{det.name}:{bundle.suite_name}" if det.name == "lstm_autoencoder" else det.name
            if key not in model_params:
                model_params[key] = det.get_parameters()

    model_order = [
        "static_threshold",
        "isolation_forest",
        "half_space_trees",
        "lstm_autoencoder",
    ]
    summary_rows: list[dict[str, Any]] = []

    if bundle.suite_name == "SKAB":
        fault_by_m: dict[str, list[dict[str, Any]]] = defaultdict(list)
        af_by_m: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in per_exp_rows:
            target = af_by_m if r["is_anomaly_free"] else fault_by_m
            target[str(r["model"])].append(r)
        for m_name in model_order:
            if fault_by_m[m_name]:
                summary_rows.append(
                    aggregate_m5_rows("SKAB", "fault_experiments_only", m_name, fault_by_m[m_name])
                )
        for m_name in model_order:
            if af_by_m[m_name]:
                summary_rows.append(
                    aggregate_m5_rows("SKAB", "anomaly_free_baseline", m_name, af_by_m[m_name])
                )
    else:
        by_m: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in per_exp_rows:
            by_m[str(r["model"])].append(r)
        for m_name in model_order:
            if by_m[m_name]:
                summary_rows.append(
                    aggregate_m5_rows(
                        bundle.suite_name, "independent_univariate_protocol", m_name, by_m[m_name]
                    )
                )

    return summary_rows, per_exp_rows, model_params


def _fmt(val: object, suffix: str = "") -> str:
    return "unavailable" if val is None else f"{val}{suffix}"


def render_m5_markdown(report: Mapping[str, Any]) -> str:
    """Render readable M5 LSTM Autoencoder vs. M4 baselines Markdown report."""
    repro = report["reproducibility"]
    dev = repro["device_info"]
    cfg = report["lstm_configuration"]
    lines = [
        f"# DriftGuard-Kafka M5 LSTM Autoencoder Benchmark Report — `{report['run_id']}`",
        "",
        f"Generated: `{report['generated_utc']}` | Seed: `{repro['seed']}` | "
        f"Device: `{dev['selected']}` (Apple Silicon: `{dev['is_apple_silicon']}`, "
        f"MPS Available: `{dev['mps_available']}`)",
        "",
        "## 1. Comparative Model Performance (M5 LSTM Autoencoder vs. M4 Baselines)",
        "",
        "| Suite | Scope | Model | Exps | Test Rows | Micro P / R / F1 | Micro FPR | "
        "Macro F1 | Mean PR-AUC | Mean VUS-PR | Mean Delay (samples) | Train s | "
        "Latency | Throughput | Peak Mem |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in report["comparison_summary"]:
        prf = f"{_fmt(s['micro_precision'])} / {_fmt(s['micro_recall'])} / {_fmt(s['micro_f1'])}"
        lines.append(
            f"| {s['dataset_suite']} | `{s['evaluation_scope']}` | `{s['model']}` | "
            f"{s['experiments']} | {s['test_samples']} | {prf} | {_fmt(s['micro_fpr'])} | "
            f"{_fmt(s['macro_f1'])} | {_fmt(s['mean_pr_auc'])} | {_fmt(s['mean_vus_pr'])} | "
            f"{_fmt(s['mean_detection_delay_samples'])} | "
            f"{_fmt(s['total_training_duration_s'], ' s')} | "
            f"{_fmt(s['mean_latency_ms'], ' ms')} | "
            f"{_fmt(s['throughput_events_per_sec'], ' ev/s')} | "
            f"{_fmt(s['max_peak_tracemalloc_kib'], ' KiB')} |"
        )

    lines.extend(
        [
            "",
            "## 2. Seq2Seq LSTM Autoencoder Architecture & Configuration",
            "",
            f"- **Window Size (`W`)**: `{cfg['window_size']}` | **Hidden Dim**: "
            f"`{cfg['hidden_dim']}` | **Latent Bottleneck (`z`)**: `{cfg['latent_dim']}` | "
            f"**Layers**: `{cfg['num_layers']}` | **Dropout**: `{cfg['dropout']}`",
            f"- **Optimization**: Adam (`lr={cfg['learning_rate']}`, "
            f"`batch_size={cfg['batch_size']}`, `epochs={cfg['epochs']}`, "
            f"`early_stopping_patience={cfg['early_stopping_patience']}`)",
            f"- **Threshold Calibration**: Normal validation quantile "
            f"`{cfg['calibration_quantile']}` (`score_reduction={cfg['score_reduction']}`)",
            f"- **Series Modes**: SKAB=`{report['series_modes']['SKAB']}`, "
            f"TSB-AD-M=`{report['series_modes']['TSB-AD-M']}`",
            "",
            "## 3. Dataset Splits & Coverage",
            "",
        ]
    )
    for suite_name, prov in report["dataset_provenance"].items():
        lines.append(f"### {suite_name}")
        for k, v in prov.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.append("")

    lines.extend(["## 4. Research Rules & Measurement Boundaries", ""])
    for boundary in report["measurement_boundaries"]:
        lines.append(f"- {boundary}")
    lines.append("")
    return "\n".join(lines)


M5_SUMMARY_CSV_FIELDS: list[str] = [
    "dataset_suite",
    "evaluation_scope",
    "model",
    "experiments",
    "test_samples",
    "tp",
    "fp",
    "tn",
    "fn",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "micro_fpr",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "macro_fpr",
    "mean_pr_auc",
    "mean_vus_pr",
    "total_episodes",
    "detected_episodes",
    "missed_episodes",
    "episode_detection_rate",
    "mean_detection_delay_samples",
    "total_training_duration_s",
    "total_inference_wall_s",
    "mean_latency_ms",
    "throughput_events_per_sec",
    "max_peak_tracemalloc_kib",
    "max_rss_mib",
]

M5_PER_EXP_CSV_FIELDS: list[str] = [
    "dataset_suite",
    "experiment_id",
    "domain",
    "is_anomaly_free",
    "model",
    "channel_mode",
    "selected_device",
    "n_features",
    "train_normal_samples",
    "val_normal_samples",
    "test_samples",
    "test_anomalies",
    "calibrated_threshold",
    "tp",
    "fp",
    "tn",
    "fn",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "pr_auc",
    "vus_pr",
    "vus_roc",
    "total_episodes",
    "detected_episodes",
    "missed_episodes",
    "mean_detection_delay_samples",
    "training_duration_s",
    "inference_wall_time_s",
    "mean_latency_ms",
    "throughput_events_per_sec",
    "cpu_time_s",
    "peak_tracemalloc_kib",
    "rss_memory_mib",
    "top_error_feature",
    "epochs_trained",
]


def write_m5_bundle(
    output_dir: Path,
    report: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    per_exp_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write M5 JSON, Markdown, and CSV report bundle to `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "m5_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "m5_report.md").write_text(render_m5_markdown(report), encoding="utf-8")

    with (output_dir / "comparison_summary.csv").open("w", encoding="utf-8", newline="") as f_sum:
        writer = csv.DictWriter(f_sum, fieldnames=M5_SUMMARY_CSV_FIELDS)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    with (output_dir / "per_experiment_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f_exp:
        writer = csv.DictWriter(f_exp, fieldnames=M5_PER_EXP_CSV_FIELDS)
        writer.writeheader()
        for row in per_exp_rows:
            writer.writerow(row)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for M5 LSTM Autoencoder training and evaluation."""
    parser = argparse.ArgumentParser(
        description="Run DriftGuard-Kafka M5 LSTM Autoencoder training & comparative benchmark."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/m5_lstm_ae.json"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto, mps, cpu, or cuda")
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--calibration-quantile", type=float, default=None)
    parser.add_argument("--skab-dir", type=Path, default=Path("data/external/skab"))
    parser.add_argument("--tsb-dir", type=Path, default=Path("data/external/tsb_ad_m"))
    parser.add_argument("--tsb-max-per-domain", type=int, default=None)
    parser.add_argument("--max-skab-files", type=int, default=None)
    parser.add_argument("--save-models-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for M5 LSTM Autoencoder training and evaluation."""
    args = parse_args(argv)
    raw_cfg: dict[str, Any] = {}
    if args.config is not None and args.config.exists():
        loaded = json.loads(args.config.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            raw_cfg.update(loaded)

    cli_overrides = {
        "seed": args.seed,
        "device": args.device,
        "window_size": args.window_size,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "calibration_quantile": args.calibration_quantile,
    }
    for key, val in cli_overrides.items():
        if val is not None:
            raw_cfg[key] = val

    skab_mode = str(raw_cfg.pop("skab_channel_mode", "multivariate"))
    tsb_mode = str(raw_cfg.pop("tsb_channel_mode", "independent_univariate"))
    cfg_tsb_max = int(raw_cfg.pop("tsb_max_per_domain", 1))
    tsb_max_raw = args.tsb_max_per_domain if args.tsb_max_per_domain is not None else cfg_tsb_max

    allowed = set(LSTMAutoencoderConfig.__dataclass_fields__)
    lstm_cfg = LSTMAutoencoderConfig(**{k: v for k, v in raw_cfg.items() if k in allowed})

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (Path("reports/m5") / run_id)

    try:
        skab_bundle = load_skab_suite(args.skab_dir)
        tsb_max = None if tsb_max_raw <= 0 else tsb_max_raw
        tsb_bundle = load_tsb_ad_m_suite(args.tsb_dir, max_per_domain=tsb_max)
    except (DatasetNotAvailableError, DatasetFormatError, ValueError) as exc:
        print(f"M5 BENCHMARK FAILED: {exc}", file=sys.stderr)
        return 2

    all_summary: list[dict[str, Any]] = []
    all_per_exp: list[dict[str, Any]] = []
    merged_params: dict[str, Any] = {}

    for bundle, cap in ((skab_bundle, args.max_skab_files), (tsb_bundle, None)):
        s_rows, e_rows, params = run_m5_suite_comparison(
            bundle=bundle,
            base_lstm_cfg=lstm_cfg,
            skab_mode=skab_mode,
            tsb_mode=tsb_mode,
            include_m4_baselines=True,
            max_experiments=cap,
            checkpoint_dir=args.save_models_dir,
        )
        all_summary.extend(s_rows)
        all_per_exp.extend(e_rows)
        merged_params.update(params)

    dev_info = select_compute_device(lstm_cfg.device)
    report: dict[str, Any] = {
        "run_id": run_id,
        "generated_utc": format_utc_timestamp(datetime.now(timezone.utc)),
        "reproducibility": {
            "seed": lstm_cfg.seed,
            "device_info": dev_info.to_dict(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine_arch": platform.machine(),
            "git_commit": _detect_git_commit(),
            "package_versions": {
                "torch": _pkg_version("torch"),
                "scikit-learn": _pkg_version("scikit-learn"),
                "river": _pkg_version("river"),
                "tsb-ad": _pkg_version("tsb-ad"),
                "pandas": _pkg_version("pandas"),
                "numpy": _pkg_version("numpy"),
                "psutil": _pkg_version("psutil"),
            },
        },
        "lstm_configuration": dataclasses.asdict(lstm_cfg),
        "series_modes": {"SKAB": skab_mode, "TSB-AD-M": tsb_mode},
        "dataset_provenance": {
            "SKAB": {
                **skab_bundle.provenance,
                "evaluated_files": (
                    min(len(skab_bundle.splits), args.max_skab_files)
                    if args.max_skab_files
                    else len(skab_bundle.splits)
                ),
            },
            "TSB-AD-M": tsb_bundle.provenance,
        },
        "model_parameters": merged_params,
        "comparison_summary": all_summary,
        "per_experiment_count": len(all_per_exp),
        "measurement_boundaries": M5_MEASUREMENT_BOUNDARIES,
    }

    write_m5_bundle(output_dir, report, all_summary, all_per_exp)
    print(f"M5 LSTM Autoencoder report generated in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
