"""Comparative M7 benchmark runner: TranAD vs. M6 TCN, M5 LSTM, and M4 baselines.

Evaluates the two-phase self-conditioning `TranADDetector` (`app.m7_tranad`)
across SKAB (multivariate) and TSB-AD-M (justified independent univariate mode)
and compares side-by-side against:
  1. `static_threshold` (M4)
  2. `isolation_forest` (M4)
  3. `half_space_trees` (M4)
  4. `lstm_autoencoder` (M5)
  5. `tcn_autoencoder` (M6)
  6. `tranad` (M7)

Supports read-only split-verified reuse of an existing `reports/m6/<run_id>`
bundle (`--reuse-m6-dir`) so M1–M6 results remain untouched and preserved.
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
from app.m6_benchmark import _coerce_csv_row
from app.m6_tcn_autoencoder import TCNAutoencoderConfig, TCNAutoencoderDetector
from app.m7_tranad import TRANAD_ARCHITECTURAL_NOTES, TranADConfig, TranADDetector
from app.report import _detect_git_commit
from app.schemas import format_utc_timestamp

MODEL_ORDER_M7: tuple[str, ...] = (
    "static_threshold",
    "isolation_forest",
    "half_space_trees",
    "lstm_autoencoder",
    "tcn_autoencoder",
    "tranad",
)

M7_MEASUREMENT_BOUNDARIES: list[str] = [
    "Normalization (`TrainingOnlyMinMaxScaler` mapping normal `[min, max]` to `[0, 1]` for "
    "TranAD's `Sigmoid` head) is fit strictly on normal training data (`x_train_normal`).",
    "TranAD is trained strictly on normal windows generated within individual machine/series "
    "boundaries; no window ever crosses experiment, series, or train/val/test boundaries.",
    "Decision thresholds are calibrated strictly at `calibration_quantile` on normal validation "
    "scores (`x_val_normal`), never using test labels or SPOT/POT test distribution fitting.",
    "SKAB is modeled in `multivariate` mode (`n_feats=8, d_model=16, nhead=8`), whereas TSB "
    "series are modeled in `independent_univariate` mode (`n_feats=1, d_model=2, nhead=1`) so "
    "heterogeneous/command-flag TSB columns are never falsely coupled as multivariate data.",
    "Existing M1–M6 reports and checkpoints are preserved untouched; M7 writes exclusively to "
    "`reports/m7/<run_id>/` and `models/m7/<run_id>/`.",
]


def load_verified_prior_rows(
    prior_dir: Path,
    report_filename: str = "m6_report.json",
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    """Load read-only prior per-experiment rows and model parameters."""
    csv_path = prior_dir / "per_experiment_metrics.csv"
    json_path = prior_dir / report_filename
    if not csv_path.exists() or not json_path.exists():
        raise FileNotFoundError(f"Prior report bundle incomplete in {prior_dir}")

    indexed: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with csv_path.open(encoding="utf-8") as handle:
        for raw_row in csv.DictReader(handle):
            parsed = _coerce_csv_row(raw_row)
            key = (str(parsed["dataset_suite"]), str(parsed["experiment_id"]))
            indexed[key].append(parsed)

    rep_json = json.loads(json_path.read_text(encoding="utf-8"))
    return indexed, dict(rep_json.get("model_parameters", {}))


def _verify_prior_m6_rows(
    split: ChronologicalSplit,
    indexed_prior: Mapping[tuple[str, str], list[dict[str, Any]]],
    required_models: set[str],
) -> list[dict[str, Any]] | None:
    candidates = indexed_prior.get((split.dataset_suite, split.experiment_id))
    if not candidates:
        return None
    found = {str(r["model"]) for r in candidates}
    if not required_models.issubset(found):
        return None

    matched: list[dict[str, Any]] = []
    for row in candidates:
        if str(row["model"]) not in required_models:
            continue
        if (
            int(row["train_normal_samples"]) != int(split.x_train_normal.shape[0])
            or int(row["val_normal_samples"]) != int(split.x_val_normal.shape[0])
            or int(row["test_samples"]) != int(split.x_test.shape[0])
            or int(row["test_anomalies"]) != int(split.y_test_anomaly.sum())
        ):
            return None
        matched.append(dict(row))
    return matched if len(matched) == len(required_models) else None


def _evaluate_and_optionally_save(
    detector: TranADDetector,
    split: ChronologicalSplit,
    checkpoint_dir: Path | None,
) -> dict[str, Any]:
    import numpy as np

    from app.m2_report import compute_classification_metrics
    from app.m4_benchmark import compute_ranking_metrics
    from app.m5_benchmark import compute_detection_delay

    cal = detector.fit_and_calibrate(
        split.x_train_normal,
        split.x_val_normal,
        split.feature_names,
    )
    if checkpoint_dir is not None:
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


def run_m7_suite_comparison(
    bundle: DatasetSuiteBundle,
    base_tranad_cfg: TranADConfig,
    skab_mode: str = "multivariate",
    tsb_mode: str = "independent_univariate",
    indexed_m6: Mapping[tuple[str, str], list[dict[str, Any]]] | None = None,
    prior_params: Mapping[str, Any] | None = None,
    max_experiments: int | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], int]:
    """Evaluate M7 TranAD alongside M4, M5 LSTM-AE, and M6 TCN-AE."""
    splits = bundle.splits[:max_experiments] if max_experiments else bundle.splits
    mode = skab_mode if bundle.suite_name == "SKAB" else tsb_mode
    tranad_cfg = dataclasses.replace(
        base_tranad_cfg,
        channel_mode=mode,  # type: ignore[arg-type]
    )

    required_prior = {
        "static_threshold",
        "isolation_forest",
        "half_space_trees",
        "lstm_autoencoder",
        "tcn_autoencoder",
    }
    per_exp_rows: list[dict[str, Any]] = []
    model_params: dict[str, Any] = dict(prior_params) if prior_params else {}
    reused_splits = 0

    for split in splits:
        prior_rows = (
            _verify_prior_m6_rows(split, indexed_m6, required_prior)
            if indexed_m6 is not None
            else None
        )
        if prior_rows is not None:
            per_exp_rows.extend(prior_rows)
            reused_splits += 1
        else:
            lstm_cfg = LSTMAutoencoderConfig(
                window_size=max(4, tranad_cfg.window_size),
                hidden_dim=16,
                latent_dim=8,
                num_layers=1,
                dropout=0.0,
                epochs=max(3, min(6, tranad_cfg.epochs)),
                batch_size=tranad_cfg.batch_size,
                seed=tranad_cfg.seed,
                device=tranad_cfg.device,
                calibration_quantile=tranad_cfg.calibration_quantile,
                channel_mode=mode,  # type: ignore[arg-type]
            )
            tcn_cfg = TCNAutoencoderConfig(
                window_size=max(4, tranad_cfg.window_size),
                hidden_channels=16,
                latent_dim=6,
                dilations=(1, 2),
                dropout=0.0,
                epochs=max(3, min(6, tranad_cfg.epochs)),
                batch_size=tranad_cfg.batch_size,
                seed=tranad_cfg.seed,
                device=tranad_cfg.device,
                calibration_quantile=tranad_cfg.calibration_quantile,
                channel_mode=mode,  # type: ignore[arg-type]
            )
            detectors: list[BenchmarkDetector] = list(
                build_default_detectors(
                    seed=tranad_cfg.seed,
                    calibration_quantile=tranad_cfg.calibration_quantile,
                )
            )
            detectors.extend([LSTMAutoencoderDetector(lstm_cfg), TCNAutoencoderDetector(tcn_cfg)])
            for det in detectors:
                row = evaluate_m5_detector_on_split(det, split, checkpoint_dir=None)
                per_exp_rows.append(row)
                p_key = (
                    f"{det.name}:{bundle.suite_name}"
                    if det.name in {"lstm_autoencoder", "tcn_autoencoder"}
                    else det.name
                )
                if p_key not in model_params:
                    model_params[p_key] = det.get_parameters()

        tranad_det = TranADDetector(tranad_cfg)
        tranad_row = _evaluate_and_optionally_save(tranad_det, split, checkpoint_dir)
        per_exp_rows.append(tranad_row)
        t_key = f"tranad:{bundle.suite_name}"
        if t_key not in model_params:
            model_params[t_key] = tranad_det.get_parameters()

    summary_rows: list[dict[str, Any]] = []
    if bundle.suite_name == "SKAB":
        fault_by_m: dict[str, list[dict[str, Any]]] = defaultdict(list)
        af_by_m: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in per_exp_rows:
            target = af_by_m if r["is_anomaly_free"] else fault_by_m
            target[str(r["model"])].append(r)
        for m_name in MODEL_ORDER_M7:
            if fault_by_m[m_name]:
                summary_rows.append(
                    aggregate_m5_rows("SKAB", "fault_experiments_only", m_name, fault_by_m[m_name])
                )
        for m_name in MODEL_ORDER_M7:
            if af_by_m[m_name]:
                summary_rows.append(
                    aggregate_m5_rows("SKAB", "anomaly_free_baseline", m_name, af_by_m[m_name])
                )
    else:
        by_m: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in per_exp_rows:
            by_m[str(r["model"])].append(r)
        for m_name in MODEL_ORDER_M7:
            if by_m[m_name]:
                summary_rows.append(
                    aggregate_m5_rows(
                        bundle.suite_name, "independent_univariate_protocol", m_name, by_m[m_name]
                    )
                )

    return summary_rows, per_exp_rows, model_params, reused_splits


def _fmt(val: object, suffix: str = "") -> str:
    return "unavailable" if val is None else f"{val}{suffix}"


def render_m7_markdown(report: Mapping[str, Any]) -> str:
    """Render readable M7 TranAD vs. M6 TCN, M5 LSTM, and M4 baselines report."""
    repro = report["reproducibility"]
    dev = repro["device_info"]
    cfg = report["tranad_configuration"]
    lines = [
        f"# DriftGuard-Kafka M7 TranAD Benchmark Report — `{report['run_id']}`",
        "",
        f"Generated: `{report['generated_utc']}` | Seed: `{repro['seed']}` | "
        f"Device: `{dev['selected']}` (Apple Silicon: `{dev['is_apple_silicon']}`, "
        f"MPS Available: `{dev['mps_available']}`) | "
        f"Prior Baseline Source: `{repro['baseline_source']}`",
        "",
        "## 1. Comparative Model Performance (M7 TranAD vs. M6 TCN, M5 LSTM & M4 Baselines)",
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
            "## 2. Two-Phase Self-Conditioning TranAD Architecture & Deviations",
            "",
            f"- **Context Window (`W`)**: `{cfg['window_size']}` | **Feedforward Dim**: "
            f"`{cfg['dim_feedforward']}` | **Encoder/Decoder Layers**: "
            f"`{cfg['num_encoder_layers']}` / `{cfg['num_decoder_layers']}` | "
            f"**Dropout**: `{cfg['dropout']}` | **Score Mode**: `{cfg['score_mode']}`",
            f"- **Optimization**: AdamW (`lr={cfg['learning_rate']}`, "
            f"`weight_decay={cfg['weight_decay']}`, "
            f"`StepLR={cfg['lr_step_size']}/{cfg['lr_gamma']}`, "
            f"`batch_size={cfg['batch_size']}`, `epochs={cfg['epochs']}`)",
            f"- **Series Modes**: SKAB=`{report['series_modes']['SKAB']}`, "
            f"TSB-AD-M=`{report['series_modes']['TSB-AD-M']}`",
            "",
        ]
    )
    for note in report["architectural_notes"]:
        lines.append(f"- {note}")

    lines.extend(["", "## 3. Dataset Splits & Coverage", ""])
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


def write_m7_bundle(
    output_dir: Path,
    report: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    per_exp_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write M7 JSON, Markdown, and CSV comparison bundle to `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "m7_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "m7_report.md").write_text(render_m7_markdown(report), encoding="utf-8")

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
    """Parse CLI arguments for M7 TranAD training and evaluation."""
    parser = argparse.ArgumentParser(
        description="Run DriftGuard-Kafka M7 TranAD vs M6 TCN, M5 LSTM & M4 benchmark."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/m7_tranad.json"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--dim-feedforward", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--calibration-quantile", type=float, default=None)
    parser.add_argument("--skab-dir", type=Path, default=Path("data/external/skab"))
    parser.add_argument("--tsb-dir", type=Path, default=Path("data/external/tsb_ad_m"))
    parser.add_argument("--tsb-max-per-domain", type=int, default=None)
    parser.add_argument("--max-skab-files", type=int, default=None)
    parser.add_argument("--reuse-m6-dir", type=Path, default=None)
    parser.add_argument("--save-models-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for M7 TranAD training and comparative benchmarking."""
    args = parse_args(argv)
    raw_cfg: dict[str, Any] = {}
    if args.config is not None and args.config.exists():
        loaded = json.loads(args.config.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            raw_cfg.update(loaded)

    overrides: dict[str, Any] = {
        "seed": args.seed,
        "device": args.device,
        "window_size": args.window_size,
        "dim_feedforward": args.dim_feedforward,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "calibration_quantile": args.calibration_quantile,
    }
    for key, val in overrides.items():
        if val is not None:
            raw_cfg[key] = val

    skab_mode = str(raw_cfg.pop("skab_channel_mode", "multivariate"))
    tsb_mode = str(raw_cfg.pop("tsb_channel_mode", "independent_univariate"))
    cfg_tsb_max = int(raw_cfg.pop("tsb_max_per_domain", 1))
    tsb_max_raw = args.tsb_max_per_domain if args.tsb_max_per_domain is not None else cfg_tsb_max

    allowed = set(TranADConfig.__dataclass_fields__)
    tranad_cfg = TranADConfig(**{k: v for k, v in raw_cfg.items() if k in allowed})

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (Path("reports/m7") / run_id)

    try:
        skab_bundle = load_skab_suite(args.skab_dir)
        tsb_max = None if tsb_max_raw <= 0 else tsb_max_raw
        tsb_bundle = load_tsb_ad_m_suite(args.tsb_dir, max_per_domain=tsb_max)
        indexed_m6: dict[tuple[str, str], list[dict[str, Any]]] | None = None
        prior_params: dict[str, Any] = {}
        if args.reuse_m6_dir is not None:
            indexed_m6, prior_params = load_verified_prior_rows(args.reuse_m6_dir, "m6_report.json")
    except (DatasetNotAvailableError, DatasetFormatError, FileNotFoundError, ValueError) as exc:
        print(f"M7 BENCHMARK FAILED: {exc}", file=sys.stderr)
        return 2

    all_summary: list[dict[str, Any]] = []
    all_per_exp: list[dict[str, Any]] = []
    merged_params: dict[str, Any] = dict(prior_params)
    total_reused = 0
    total_splits = 0

    for bundle, cap in ((skab_bundle, args.max_skab_files), (tsb_bundle, None)):
        s_rows, e_rows, params, reused = run_m7_suite_comparison(
            bundle=bundle,
            base_tranad_cfg=tranad_cfg,
            skab_mode=skab_mode,
            tsb_mode=tsb_mode,
            indexed_m6=indexed_m6,
            prior_params=merged_params,
            max_experiments=cap,
            checkpoint_dir=args.save_models_dir,
        )
        all_summary.extend(s_rows)
        all_per_exp.extend(e_rows)
        merged_params.update(params)
        total_reused += reused
        total_splits += len(bundle.splits[:cap] if cap else bundle.splits)

    if args.reuse_m6_dir is not None and total_reused == total_splits:
        baseline_src = f"reused_verified_m6_bundle:{args.reuse_m6_dir.name}"
    elif total_reused > 0:
        baseline_src = f"hybrid_reused_{total_reused}_of_{total_splits}_from:{args.reuse_m6_dir}"
    else:
        baseline_src = "executed_in_process"

    dev_info = select_compute_device(tranad_cfg.device)
    report: dict[str, Any] = {
        "run_id": run_id,
        "generated_utc": format_utc_timestamp(datetime.now(timezone.utc)),
        "reproducibility": {
            "seed": tranad_cfg.seed,
            "device_info": dev_info.to_dict(),
            "baseline_source": baseline_src,
            "reused_prior_splits": total_reused,
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
        "tranad_configuration": dataclasses.asdict(tranad_cfg),
        "architectural_notes": TRANAD_ARCHITECTURAL_NOTES,
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
        "measurement_boundaries": M7_MEASUREMENT_BOUNDARIES,
    }

    write_m7_bundle(output_dir, report, all_summary, all_per_exp)
    print(f"M7 TranAD report generated in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
