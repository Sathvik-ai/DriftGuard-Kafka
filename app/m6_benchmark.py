"""Comparative M6 benchmark runner: TCN Autoencoder vs. M5 LSTM-AE & M4 baselines.

Reuses the exact M5 chronological split protocol (`app.datasets`) and metric suite
(`app.m2_report`, `app.m4_benchmark`, `app.m5_benchmark`) to compare:
  1. `static_threshold` (M4 baseline)
  2. `isolation_forest` (M4 Scikit-learn ensemble)
  3. `half_space_trees` (M4 River streaming trees)
  4. `lstm_autoencoder` (M5 Seq2Seq LSTM Autoencoder)
  5. `tcn_autoencoder` (M6 Causal Dilated Residual TCN Autoencoder)

Supports both full in-process execution of all 5 models and split-verified reuse
of an existing `reports/m5/<run_id>` bundle (`--reuse-m5-dir`) so M5 results are
preserved verbatim without modification or replacement.
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
from pathlib import Path
from typing import Any

from app.datasets import (
    ChronologicalSplit,
    DatasetFormatError,
    DatasetNotAvailableError,
    DatasetSuiteBundle,
    load_skab_suite,
    load_tsb_ad_m_suite,
)
from app.m4_models import BenchmarkDetector, build_default_detectors
from app.m5_benchmark import (
    M5_PER_EXP_CSV_FIELDS,
    M5_SUMMARY_CSV_FIELDS,
    _pkg_version,
    aggregate_m5_rows,
    evaluate_m5_detector_on_split,
)
from app.m5_lstm_autoencoder import (
    LSTMAutoencoderConfig,
    LSTMAutoencoderDetector,
    select_compute_device,
)
from app.m6_tcn_autoencoder import TCNAutoencoderConfig, TCNAutoencoderDetector
from app.report import _detect_git_commit
from app.schemas import format_utc_timestamp

M6_MEASUREMENT_BOUNDARIES: list[str] = [
    "Normalization (`TrainingOnlyScaler`) is fit strictly on normal training observations "
    "(`x_train_normal`) only; validation and test observations never influence scaling.",
    "TCN Autoencoder uses strictly causal left-padded convolutions `(kernel_size - 1) * dilation` "
    "with exponential dilations and residual blocks, trained exclusively on normal windows.",
    "Windows are generated strictly inside individual experiment/series and split boundaries; "
    "no window ever crosses machine, experiment, series, or train/val/test boundaries.",
    "Decision thresholds are calibrated strictly at `calibration_quantile` on normal validation "
    "reconstruction error scores (`x_val_normal`), never using test labels.",
    "SKAB is modeled in `multivariate` mode (`input_dim=8`), whereas TSB series are modeled in "
    "`independent_univariate` mode (`input_dim=1`) so heterogeneous TSB columns are never "
    "falsely represented as coupled multivariate data.",
    "Existing M1–M5 benchmark reports and checkpoints are never modified or overwritten; M6 "
    "writes exclusively to `reports/m6/<run_id>/` and `models/m6/<run_id>/`.",
]

MODEL_ORDER_M6: tuple[str, ...] = (
    "static_threshold",
    "isolation_forest",
    "half_space_trees",
    "lstm_autoencoder",
    "tcn_autoencoder",
)


def _coerce_csv_row(raw: Mapping[str, str]) -> dict[str, Any]:
    int_fields = {
        "n_features",
        "train_normal_samples",
        "val_normal_samples",
        "test_samples",
        "test_anomalies",
        "tp",
        "fp",
        "tn",
        "fn",
        "total_episodes",
        "detected_episodes",
        "missed_episodes",
    }
    float_fields = {
        "calibrated_threshold",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "pr_auc",
        "vus_pr",
        "vus_roc",
        "mean_detection_delay_samples",
        "training_duration_s",
        "inference_wall_time_s",
        "mean_latency_ms",
        "throughput_events_per_sec",
        "cpu_time_s",
        "peak_tracemalloc_kib",
        "rss_memory_mib",
    }
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if v == "" or v is None:
            out[k] = None
        elif k == "is_anomaly_free":
            out[k] = v.lower() == "true"
        elif k == "epochs_trained":
            out[k] = int(float(v))
        elif k in int_fields:
            out[k] = int(v)
        elif k in float_fields:
            out[k] = float(v)
        else:
            out[k] = v
    return out


def load_verified_m5_rows(
    m5_dir: Path,
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    """Load read-only M4+M5 per-experiment rows and parameters from `m5_dir`."""
    csv_path = m5_dir / "per_experiment_metrics.csv"
    json_path = m5_dir / "m5_report.json"
    if not csv_path.exists() or not json_path.exists():
        raise FileNotFoundError(f"M5 report bundle incomplete in {m5_dir}")

    indexed: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with csv_path.open(encoding="utf-8") as handle:
        for raw_row in csv.DictReader(handle):
            parsed = _coerce_csv_row(raw_row)
            key = (str(parsed["dataset_suite"]), str(parsed["experiment_id"]))
            indexed[key].append(parsed)

    m5_json = json.loads(json_path.read_text(encoding="utf-8"))
    params = dict(m5_json.get("model_parameters", {}))
    return indexed, params


def _verify_and_extract_prior_rows(
    split: ChronologicalSplit,
    indexed_m5: Mapping[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]] | None:
    key = (split.dataset_suite, split.experiment_id)
    candidates = indexed_m5.get(key)
    if not candidates:
        return None
    expected_models = {
        "static_threshold",
        "isolation_forest",
        "half_space_trees",
        "lstm_autoencoder",
    }
    found_models = {str(r["model"]) for r in candidates}
    if not expected_models.issubset(found_models):
        return None

    matched: list[dict[str, Any]] = []
    for row in candidates:
        if str(row["model"]) not in expected_models:
            continue
        if (
            int(row["train_normal_samples"]) != int(split.x_train_normal.shape[0])
            or int(row["val_normal_samples"]) != int(split.x_val_normal.shape[0])
            or int(row["test_samples"]) != int(split.x_test.shape[0])
            or int(row["test_anomalies"]) != int(split.y_test_anomaly.sum())
        ):
            return None
        matched.append(dict(row))
    return matched if len(matched) == 4 else None


def run_m6_suite_comparison(
    bundle: DatasetSuiteBundle,
    base_tcn_cfg: TCNAutoencoderConfig,
    base_lstm_cfg: LSTMAutoencoderConfig,
    skab_mode: str = "multivariate",
    tsb_mode: str = "independent_univariate",
    indexed_m5: Mapping[tuple[str, str], list[dict[str, Any]]] | None = None,
    prior_params: Mapping[str, Any] | None = None,
    max_experiments: int | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], int]:
    """Evaluate M6 TCN Autoencoder alongside M5 LSTM-AE and M4 baselines."""
    splits = bundle.splits[:max_experiments] if max_experiments else bundle.splits
    mode = skab_mode if bundle.suite_name == "SKAB" else tsb_mode
    tcn_cfg = dataclasses.replace(
        base_tcn_cfg,
        channel_mode=mode,  # type: ignore[arg-type]
    )
    lstm_cfg = dataclasses.replace(
        base_lstm_cfg,
        channel_mode=mode,  # type: ignore[arg-type]
    )

    per_exp_rows: list[dict[str, Any]] = []
    model_params: dict[str, Any] = dict(prior_params) if prior_params else {}
    reused_splits = 0

    for split in splits:
        prior_rows = (
            _verify_and_extract_prior_rows(split, indexed_m5) if indexed_m5 is not None else None
        )
        if prior_rows is not None:
            per_exp_rows.extend(prior_rows)
            reused_splits += 1
        else:
            detectors: list[BenchmarkDetector] = list(
                build_default_detectors(
                    seed=tcn_cfg.seed,
                    calibration_quantile=tcn_cfg.calibration_quantile,
                )
            )
            detectors.append(LSTMAutoencoderDetector(lstm_cfg))
            for det in detectors:
                row = evaluate_m5_detector_on_split(det, split, checkpoint_dir=None)
                per_exp_rows.append(row)
                p_key = (
                    f"{det.name}:{bundle.suite_name}"
                    if det.name == "lstm_autoencoder"
                    else det.name
                )
                if p_key not in model_params:
                    model_params[p_key] = det.get_parameters()

        tcn_det = TCNAutoencoderDetector(tcn_cfg)
        if checkpoint_dir is not None:
            cal = tcn_det.fit_and_calibrate(
                split.x_train_normal,
                split.x_val_normal,
                split.feature_names,
            )
            safe_id = split.experiment_id.replace("/", "__").replace(".csv", "")
            tcn_det.save_checkpoint(checkpoint_dir / split.dataset_suite / f"{safe_id}.pt")
            # Evaluate test directly using already-calibrated TCN detector
            tcn_row = _score_calibrated_detector(tcn_det, cal, split)
        else:
            tcn_row = evaluate_m5_detector_on_split(tcn_det, split, checkpoint_dir=None)

        per_exp_rows.append(tcn_row)
        tcn_key = f"tcn_autoencoder:{bundle.suite_name}"
        if tcn_key not in model_params:
            model_params[tcn_key] = tcn_det.get_parameters()

    summary_rows: list[dict[str, Any]] = []
    if bundle.suite_name == "SKAB":
        fault_by_m: dict[str, list[dict[str, Any]]] = defaultdict(list)
        af_by_m: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in per_exp_rows:
            target = af_by_m if r["is_anomaly_free"] else fault_by_m
            target[str(r["model"])].append(r)
        for m_name in MODEL_ORDER_M6:
            if fault_by_m[m_name]:
                summary_rows.append(
                    aggregate_m5_rows("SKAB", "fault_experiments_only", m_name, fault_by_m[m_name])
                )
        for m_name in MODEL_ORDER_M6:
            if af_by_m[m_name]:
                summary_rows.append(
                    aggregate_m5_rows("SKAB", "anomaly_free_baseline", m_name, af_by_m[m_name])
                )
    else:
        by_m: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in per_exp_rows:
            by_m[str(r["model"])].append(r)
        for m_name in MODEL_ORDER_M6:
            if by_m[m_name]:
                summary_rows.append(
                    aggregate_m5_rows(
                        bundle.suite_name, "independent_univariate_protocol", m_name, by_m[m_name]
                    )
                )

    return summary_rows, per_exp_rows, model_params, reused_splits


def _score_calibrated_detector(
    detector: TCNAutoencoderDetector,
    cal: Any,
    split: ChronologicalSplit,
) -> dict[str, Any]:
    import numpy as np

    from app.m2_report import compute_classification_metrics
    from app.m4_benchmark import compute_ranking_metrics
    from app.m5_benchmark import compute_detection_delay

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
        "channel_mode": inf.extra_diagnostics.get("channel_mode", "multivariate"),
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


def _fmt(val: object, suffix: str = "") -> str:
    return "unavailable" if val is None else f"{val}{suffix}"


def render_m6_markdown(report: Mapping[str, Any]) -> str:
    """Render readable M6 TCN Autoencoder vs. M5 LSTM-AE & M4 baselines report."""
    repro = report["reproducibility"]
    dev = repro["device_info"]
    cfg = report["tcn_configuration"]
    lines = [
        f"# DriftGuard-Kafka M6 TCN Autoencoder Benchmark Report — `{report['run_id']}`",
        "",
        f"Generated: `{report['generated_utc']}` | Seed: `{repro['seed']}` | "
        f"Device: `{dev['selected']}` (Apple Silicon: `{dev['is_apple_silicon']}`, "
        f"MPS Available: `{dev['mps_available']}`) | "
        f"Prior Baseline Source: `{repro['baseline_source']}`",
        "",
        "## 1. Comparative Model Performance (M6 TCN-AE vs. M5 LSTM-AE vs. M4 Baselines)",
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
            "## 2. Causal Dilated Residual TCN Autoencoder Architecture",
            "",
            f"- **Window Size (`W`)**: `{cfg['window_size']}` | **Hidden Channels**: "
            f"`{cfg['hidden_channels']}` | **Latent Bottleneck (`z`)**: `{cfg['latent_dim']}` | "
            f"**Kernel Size**: `{cfg['kernel_size']}` | **Dilations**: `{cfg['dilations']}` "
            f"(Causal Receptive Field = `{cfg['receptive_field']}` steps)",
            f"- **Optimization**: Adam (`lr={cfg['learning_rate']}`, "
            f"`batch_size={cfg['batch_size']}`, `epochs={cfg['epochs']}`, "
            f"`early_stopping_patience={cfg['early_stopping_patience']}`, "
            f"`dropout={cfg['dropout']}`)",
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


def write_m6_bundle(
    output_dir: Path,
    report: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    per_exp_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write M6 JSON, Markdown, and CSV comparison bundle to `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "m6_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "m6_report.md").write_text(render_m6_markdown(report), encoding="utf-8")

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
    """Parse CLI arguments for M6 TCN Autoencoder training and evaluation."""
    parser = argparse.ArgumentParser(
        description="Run DriftGuard-Kafka M6 TCN Autoencoder vs M5 LSTM-AE & M4 benchmark."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/m6_tcn_ae.json"))
    parser.add_argument("--lstm-config", type=Path, default=Path("configs/m5_lstm_ae.json"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--hidden-channels", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--kernel-size", type=int, default=None)
    parser.add_argument("--dilations", type=int, nargs="+", default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--calibration-quantile", type=float, default=None)
    parser.add_argument("--skab-dir", type=Path, default=Path("data/external/skab"))
    parser.add_argument("--tsb-dir", type=Path, default=Path("data/external/tsb_ad_m"))
    parser.add_argument("--tsb-max-per-domain", type=int, default=None)
    parser.add_argument("--max-skab-files", type=int, default=None)
    parser.add_argument(
        "--reuse-m5-dir",
        type=Path,
        default=None,
        help="Optional read-only M5 report dir to reuse verified M4+M5 baseline rows",
    )
    parser.add_argument("--save-models-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for M6 TCN Autoencoder training and evaluation."""
    args = parse_args(argv)
    raw_cfg: dict[str, Any] = {}
    if args.config is not None and args.config.exists():
        loaded = json.loads(args.config.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            raw_cfg.update(loaded)

    cli_overrides: dict[str, Any] = {
        "seed": args.seed,
        "device": args.device,
        "window_size": args.window_size,
        "hidden_channels": args.hidden_channels,
        "latent_dim": args.latent_dim,
        "kernel_size": args.kernel_size,
        "dilations": tuple(args.dilations) if args.dilations is not None else None,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "calibration_quantile": args.calibration_quantile,
    }
    for key, val in cli_overrides.items():
        if val is not None:
            raw_cfg[key] = val

    if "dilations" in raw_cfg and isinstance(raw_cfg["dilations"], list):
        raw_cfg["dilations"] = tuple(int(x) for x in raw_cfg["dilations"])

    skab_mode = str(raw_cfg.pop("skab_channel_mode", "multivariate"))
    tsb_mode = str(raw_cfg.pop("tsb_channel_mode", "independent_univariate"))
    cfg_tsb_max = int(raw_cfg.pop("tsb_max_per_domain", 1))
    tsb_max_raw = args.tsb_max_per_domain if args.tsb_max_per_domain is not None else cfg_tsb_max

    allowed_tcn = set(TCNAutoencoderConfig.__dataclass_fields__)
    tcn_cfg = TCNAutoencoderConfig(**{k: v for k, v in raw_cfg.items() if k in allowed_tcn})

    if args.lstm_config is not None and args.lstm_config.exists():
        lstm_cfg = LSTMAutoencoderConfig.from_json_file(args.lstm_config)
        lstm_cfg = dataclasses.replace(
            lstm_cfg,
            seed=tcn_cfg.seed,
            device=tcn_cfg.device,
            calibration_quantile=tcn_cfg.calibration_quantile,
        )
    else:
        lstm_cfg = LSTMAutoencoderConfig(
            window_size=tcn_cfg.window_size,
            epochs=tcn_cfg.epochs,
            batch_size=tcn_cfg.batch_size,
            seed=tcn_cfg.seed,
            device=tcn_cfg.device,
            calibration_quantile=tcn_cfg.calibration_quantile,
        )

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (Path("reports/m6") / run_id)

    try:
        skab_bundle = load_skab_suite(args.skab_dir)
        tsb_max = None if tsb_max_raw <= 0 else tsb_max_raw
        tsb_bundle = load_tsb_ad_m_suite(args.tsb_dir, max_per_domain=tsb_max)
        indexed_m5: dict[tuple[str, str], list[dict[str, Any]]] | None = None
        prior_params: dict[str, Any] = {}
        if args.reuse_m5_dir is not None:
            indexed_m5, prior_params = load_verified_m5_rows(args.reuse_m5_dir)
    except (DatasetNotAvailableError, DatasetFormatError, FileNotFoundError, ValueError) as exc:
        print(f"M6 BENCHMARK FAILED: {exc}", file=sys.stderr)
        return 2

    all_summary: list[dict[str, Any]] = []
    all_per_exp: list[dict[str, Any]] = []
    merged_params: dict[str, Any] = dict(prior_params)
    total_reused = 0
    total_splits = 0

    for bundle, cap in ((skab_bundle, args.max_skab_files), (tsb_bundle, None)):
        s_rows, e_rows, params, reused = run_m6_suite_comparison(
            bundle=bundle,
            base_tcn_cfg=tcn_cfg,
            base_lstm_cfg=lstm_cfg,
            skab_mode=skab_mode,
            tsb_mode=tsb_mode,
            indexed_m5=indexed_m5,
            prior_params=merged_params,
            max_experiments=cap,
            checkpoint_dir=args.save_models_dir,
        )
        all_summary.extend(s_rows)
        all_per_exp.extend(e_rows)
        merged_params.update(params)
        total_reused += reused
        total_splits += len(bundle.splits[:cap] if cap else bundle.splits)

    if args.reuse_m5_dir is not None and total_reused == total_splits:
        baseline_src = f"reused_verified_m5_bundle:{args.reuse_m5_dir.name}"
    elif total_reused > 0:
        baseline_src = f"hybrid_reused_{total_reused}_of_{total_splits}_from:{args.reuse_m5_dir}"
    else:
        baseline_src = "executed_in_process"

    dev_info = select_compute_device(tcn_cfg.device)
    tcn_cfg_dict = dataclasses.asdict(tcn_cfg)
    tcn_cfg_dict["dilations"] = list(tcn_cfg.dilations)
    tcn_cfg_dict["receptive_field"] = tcn_cfg.receptive_field

    report: dict[str, Any] = {
        "run_id": run_id,
        "generated_utc": format_utc_timestamp(datetime.now(timezone.utc)),
        "reproducibility": {
            "seed": tcn_cfg.seed,
            "device_info": dev_info.to_dict(),
            "baseline_source": baseline_src,
            "reused_m5_splits": total_reused,
            "total_splits": total_splits,
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
        "tcn_configuration": tcn_cfg_dict,
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
        "measurement_boundaries": M6_MEASUREMENT_BOUNDARIES,
    }

    write_m6_bundle(output_dir, report, all_summary, all_per_exp)
    print(f"M6 TCN Autoencoder report generated in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
