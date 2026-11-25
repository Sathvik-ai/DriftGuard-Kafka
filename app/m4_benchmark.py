"""Comparative ML anomaly detection benchmark runner for DriftGuard-Kafka M4.

Evaluates three unsupervised anomaly detectors on **SKAB** and **TSB-AD-M**:
  1. `static_threshold` (`StaticThresholdBaseline`)
  2. `isolation_forest` (`IsolationForestDetector`)
  3. `half_space_trees` (`HalfSpaceTreesDetector`)

Generates reproducible `comparison_summary.csv`, `per_experiment_metrics.csv`,
`benchmark_report.json`, and `benchmark_report.md` under `reports/m4/<run_id>/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import average_precision_score
from TSB_AD.evaluation.metrics import get_metrics as tsb_get_metrics

from app.datasets import (
    ChronologicalSplit,
    DatasetFormatError,
    DatasetNotAvailableError,
    DatasetSuiteBundle,
    ensure_skab_downloaded,
    ensure_tsb_ad_m_downloaded,
    load_skab_suite,
    load_tsb_ad_m_suite,
)
from app.m2_report import compute_classification_metrics
from app.m4_models import BenchmarkDetector, build_default_detectors
from app.report import _detect_git_commit
from app.schemas import format_utc_timestamp

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

M4_MEASUREMENT_BOUNDARIES: list[str] = [
    "All models are trained strictly on normal observations (`y == 0`) from the chronological "
    "training prefix; anomalous points occurring inside raw training prefixes (e.g. SKAB "
    "`other/2.csv` or TSB-AD-M series with `1st < tr`) are excluded before fitting.",
    "Decision thresholds are calibrated exclusively at `calibration_quantile` on held-out "
    "chronological normal validation scores (`x_val_normal`), never using test labels.",
    "Time-series rows are partitioned chronologically or at the experiment level; random "
    "shuffling is never performed.",
    "Half-Space Trees scales features online via `river.preprocessing.MinMaxScaler`, completes "
    "at least one full reference mass-pivot window on normal training data before validation, "
    "and strictly executes `score_one` before `learn_one` on every validation and test sample.",
    "VUS-PR and VUS-ROC are computed using the official `TSB_AD.evaluation.metrics.get_metrics` "
    "implementation (`tsb-ad==1.5`) when test splits contain both normal and anomalous samples; "
    "on purely normal series (`anomaly-free.csv`), Recall, PR-AUC, and VUS-PR are reported as "
    "`null`/`unavailable` rather than `0`.",
    "SKAB `anomaly` and `changepoint` labels are preserved separately; binary outlier metrics "
    "are evaluated against `anomaly`, while `changepoint` counts and 60-sample post-changepoint "
    "alert hits are tracked independently.",
    "ADWIN remains completely independent of M4 detectors; no adaptive retraining or dynamic "
    "threshold retuning is performed.",
]


def _pkg_version(dist_name: str) -> str | None:
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return None


def _clean_float(val: object, digits: int = 4) -> float | None:
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, float, np.floating, np.integer)):
        fval = float(val)
        if math.isfinite(fval):
            return round(fval, digits)
    return None


def _mean_optional(values: Sequence[float | int | None], digits: int = 4) -> float | None:
    valid = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not valid:
        return None
    return round(sum(valid) / len(valid), digits)


def compute_ranking_metrics(
    y_true: IntArray,
    scores: FloatArray,
    predictions: IntArray,
    sliding_window: int,
) -> dict[str, float | None]:
    """Compute PR-AUC and official TSB-AD VUS-PR / VUS-ROC when labels have both classes."""
    n_total = int(y_true.shape[0])
    n_pos = int(np.sum(y_true))
    if n_total == 0 or n_pos == 0 or n_pos == n_total:
        return {"pr_auc": None, "vus_pr": None, "vus_roc": None}

    pr_auc = _clean_float(average_precision_score(y_true, scores), digits=4)
    eff_window = max(2, min(int(sliding_window), max(2, n_total // 2)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        tsb_out = tsb_get_metrics(
            np.asarray(scores, dtype=np.float64),
            np.asarray(y_true, dtype=np.int64),
            slidingWindow=eff_window,
            pred=np.asarray(predictions, dtype=np.int64),
        )
    return {
        "pr_auc": pr_auc,
        "vus_pr": _clean_float(tsb_out.get("VUS-PR"), digits=4),
        "vus_roc": _clean_float(tsb_out.get("VUS-ROC"), digits=4),
    }


def _evaluate_changepoint_hits(
    y_changepoint: IntArray | None,
    predictions: IntArray,
    right_window: int = 60,
) -> tuple[int | None, int | None]:
    """Count SKAB changepoints and alerts within `[cp, cp + right_window]` separately."""
    if y_changepoint is None:
        return None, None
    cp_indices = np.flatnonzero(y_changepoint > 0)
    total_cps = int(cp_indices.shape[0])
    flagged_cps = 0
    n_len = int(predictions.shape[0])
    for cp_idx in cp_indices:
        end_idx = min(n_len, int(cp_idx) + right_window + 1)
        if np.any(predictions[int(cp_idx) : end_idx] > 0):
            flagged_cps += 1
    return total_cps, flagged_cps


def evaluate_detector_on_split(
    detector: BenchmarkDetector,
    split: ChronologicalSplit,
) -> dict[str, Any]:
    """Fit/calibrate `detector` on normal prefix of `split` and evaluate on `x_test`."""
    cal = detector.fit_and_calibrate(
        split.x_train_normal,
        split.x_val_normal,
        split.feature_names,
    )
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
    cp_total, cp_flagged = _evaluate_changepoint_hits(split.y_test_changepoint, preds)

    return {
        "dataset_suite": split.dataset_suite,
        "experiment_id": split.experiment_id,
        "domain": split.domain,
        "is_anomaly_free": split.is_anomaly_free,
        "model": detector.name,
        "n_features": len(split.feature_names),
        "train_boundary_index": split.train_boundary_index,
        "train_normal_samples": cal.train_samples,
        "val_normal_samples": cal.val_samples,
        "dropped_train_anomalies": split.dropped_train_anomalies,
        "test_samples": inf.test_samples,
        "test_anomalies": int(np.sum(y_true)),
        "test_changepoints": cp_total,
        "changepoints_flagged_60s": cp_flagged,
        "sliding_window": split.sliding_window,
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
        "fit_wall_time_s": cal.fit_wall_time_s,
        "inference_wall_time_s": inf.inference_wall_time_s,
        "mean_latency_ms": inf.mean_latency_ms,
        "throughput_events_per_sec": inf.throughput_events_per_sec,
        "cpu_time_s": inf.cpu_time_s,
        "cpu_utilization_pct": inf.cpu_utilization_pct,
        "peak_tracemalloc_kib": inf.peak_tracemalloc_kib,
        "rss_memory_mib": inf.rss_memory_mib,
    }


def aggregate_experiment_rows(
    dataset_suite: str,
    evaluation_scope: str,
    model_name: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-experiment evaluations into micro + macro summary statistics."""
    tp = sum(int(r["tp"]) for r in rows)
    fp = sum(int(r["fp"]) for r in rows)
    tn = sum(int(r["tn"]) for r in rows)
    fn = sum(int(r["fn"]) for r in rows)
    micro = compute_classification_metrics(tp, fp, tn, fn)

    total_test = sum(int(r["test_samples"]) for r in rows)
    total_wall_s = sum(float(r["inference_wall_time_s"]) for r in rows)
    total_cpu_s = sum(float(r["cpu_time_s"]) for r in rows)
    overall_throughput = round(total_test / total_wall_s, 2) if total_wall_s > 0 else None
    overall_latency_ms = round((total_wall_s * 1000.0) / total_test, 6) if total_test > 0 else None
    overall_cpu_pct = round(100.0 * (total_cpu_s / total_wall_s), 2) if total_wall_s > 0 else None

    cp_totals = [int(r["test_changepoints"]) for r in rows if r["test_changepoints"] is not None]
    cp_hits = [
        int(r["changepoints_flagged_60s"])
        for r in rows
        if r["changepoints_flagged_60s"] is not None
    ]

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
        "mean_vus_roc": _mean_optional([r["vus_roc"] for r in rows]),
        "changepoints_total": sum(cp_totals) if cp_totals else None,
        "changepoints_flagged_60s": sum(cp_hits) if cp_hits else None,
        "mean_latency_ms": overall_latency_ms,
        "throughput_events_per_sec": overall_throughput,
        "total_inference_wall_s": round(total_wall_s, 4),
        "total_cpu_s": round(total_cpu_s, 4),
        "cpu_utilization_pct": overall_cpu_pct,
        "max_peak_tracemalloc_kib": round(
            max((float(r["peak_tracemalloc_kib"]) for r in rows), default=0.0), 3
        ),
        "max_rss_mib": round(max((float(r["rss_memory_mib"]) for r in rows), default=0.0), 3),
    }


def run_suite_benchmark(
    bundle: DatasetSuiteBundle,
    seed: int = 42,
    calibration_quantile: float = 0.99,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run all three detectors across every split in `bundle`."""
    per_exp_rows: list[dict[str, Any]] = []
    model_params: dict[str, Any] = {}

    for split in bundle.splits:
        detectors = build_default_detectors(seed=seed, calibration_quantile=calibration_quantile)
        for det in detectors:
            row = evaluate_detector_on_split(det, split)
            per_exp_rows.append(row)
            if det.name not in model_params:
                model_params[det.name] = det.get_parameters()

    model_order = ["static_threshold", "isolation_forest", "half_space_trees"]
    summary_rows: list[dict[str, Any]] = []

    if bundle.suite_name == "SKAB":
        fault_rows_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        all_rows_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        af_rows_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in per_exp_rows:
            m_name = str(row["model"])
            all_rows_by_model[m_name].append(row)
            if row["is_anomaly_free"]:
                af_rows_by_model[m_name].append(row)
            else:
                fault_rows_by_model[m_name].append(row)

        for m_name in model_order:
            if fault_rows_by_model[m_name]:
                summary_rows.append(
                    aggregate_experiment_rows(
                        "SKAB", "fault_experiments_only", m_name, fault_rows_by_model[m_name]
                    )
                )
        for m_name in model_order:
            if all_rows_by_model[m_name]:
                summary_rows.append(
                    aggregate_experiment_rows(
                        "SKAB", "all_files_with_anomaly_free", m_name, all_rows_by_model[m_name]
                    )
                )
        for m_name in model_order:
            if af_rows_by_model[m_name]:
                summary_rows.append(
                    aggregate_experiment_rows(
                        "SKAB", "anomaly_free_baseline", m_name, af_rows_by_model[m_name]
                    )
                )
    else:
        by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in per_exp_rows:
            by_model[str(row["model"])].append(row)
        for m_name in model_order:
            if by_model[m_name]:
                summary_rows.append(
                    aggregate_experiment_rows(
                        bundle.suite_name, "official_tr_split", m_name, by_model[m_name]
                    )
                )

    return summary_rows, per_exp_rows, model_params


def build_m4_report(
    run_id: str,
    seed: int,
    calibration_quantile: float,
    suites: Sequence[DatasetSuiteBundle],
    summary_rows: Sequence[Mapping[str, Any]],
    per_exp_rows: Sequence[Mapping[str, Any]],
    model_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble full machine-readable M4 benchmark report."""
    domain_breakdown: dict[str, dict[str, dict[str, float | None]]] = defaultdict(dict)
    tsb_rows = [r for r in per_exp_rows if r["dataset_suite"] == "TSB-AD-M"]
    if tsb_rows:
        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for r in tsb_rows:
            grouped[(str(r["domain"]), str(r["model"]))].append(r)
        for (dom, mod), drows in sorted(grouped.items()):
            agg = aggregate_experiment_rows("TSB-AD-M", f"domain:{dom}", mod, drows)
            domain_breakdown[dom][mod] = {
                "experiments": agg["experiments"],
                "micro_f1": agg["micro_f1"],
                "macro_f1": agg["macro_f1"],
                "mean_pr_auc": agg["mean_pr_auc"],
                "mean_vus_pr": agg["mean_vus_pr"],
            }

    return {
        "run_id": run_id,
        "generated_utc": format_utc_timestamp(datetime.now(timezone.utc)),
        "reproducibility": {
            "seed": seed,
            "calibration_quantile": calibration_quantile,
            "adwin_integration": "independent_no_adaptive_retraining",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine_arch": platform.machine(),
            "git_commit": _detect_git_commit(),
            "package_versions": {
                "scikit-learn": _pkg_version("scikit-learn"),
                "river": _pkg_version("river"),
                "tsb-ad": _pkg_version("tsb-ad"),
                "pandas": _pkg_version("pandas"),
                "numpy": _pkg_version("numpy"),
                "scipy": _pkg_version("scipy"),
                "statsmodels": _pkg_version("statsmodels"),
                "psutil": _pkg_version("psutil"),
            },
        },
        "dataset_provenance": {b.suite_name: b.provenance for b in suites},
        "model_parameters": dict(model_parameters),
        "comparison_summary": list(summary_rows),
        "tsb_ad_m_domain_breakdown": dict(domain_breakdown),
        "per_experiment_count": len(per_exp_rows),
        "measurement_boundaries": M4_MEASUREMENT_BOUNDARIES,
    }


def _fmt(val: object, suffix: str = "") -> str:
    return "unavailable" if val is None else f"{val}{suffix}"


def render_m4_markdown(report: Mapping[str, Any]) -> str:
    """Render readable Markdown comparison report for DriftGuard-Kafka M4."""
    repro = report["reproducibility"]
    pkgs = repro["package_versions"]
    lines = [
        f"# DriftGuard-Kafka M4 Comparative ML Benchmark Report — `{report['run_id']}`",
        "",
        f"Generated: `{report['generated_utc']}` | Seed: `{repro['seed']}` | "
        f"Validation Threshold Quantile: `{repro['calibration_quantile']}`",
        "",
        "## 1. Comparative Model Performance Summary",
        "",
        "| Suite | Scope | Model | Exps | Test Rows | Micro P / R / F1 | Micro FPR | "
        "Macro F1 | Mean PR-AUC | Mean VUS-PR | Mean Latency | Throughput | CPU s | Peak Mem |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for s in report["comparison_summary"]:
        prf = f"{_fmt(s['micro_precision'])} / {_fmt(s['micro_recall'])} / {_fmt(s['micro_f1'])}"
        lines.append(
            f"| {s['dataset_suite']} | `{s['evaluation_scope']}` | `{s['model']}` | "
            f"{s['experiments']} | {s['test_samples']} | {prf} | {_fmt(s['micro_fpr'])} | "
            f"{_fmt(s['macro_f1'])} | {_fmt(s['mean_pr_auc'])} | {_fmt(s['mean_vus_pr'])} | "
            f"{_fmt(s['mean_latency_ms'], ' ms')} | "
            f"{_fmt(s['throughput_events_per_sec'], ' ev/s')} | "
            f"{_fmt(s['total_cpu_s'], ' s')} | "
            f"{_fmt(s['max_peak_tracemalloc_kib'], ' KiB')} |"
        )

    skab_fault = [
        s
        for s in report["comparison_summary"]
        if s["dataset_suite"] == "SKAB" and s["evaluation_scope"] == "fault_experiments_only"
    ]
    if skab_fault:
        lines.extend(
            [
                "",
                "## 2. SKAB Separate `changepoint` Preservation Summary",
                "",
                "| Model | True Test Changepoints | Flagged Within 60-Step Transition Window |",
                "|---|---:|---:|",
            ]
        )
        for s in skab_fault:
            lines.append(
                f"| `{s['model']}` | {_fmt(s['changepoints_total'])} | "
                f"{_fmt(s['changepoints_flagged_60s'])} |"
            )

    dom_map = report.get("tsb_ad_m_domain_breakdown") or {}
    if dom_map:
        lines.extend(
            [
                "",
                "## 3. TSB-AD-M Per-Domain VUS-PR & F1 Breakdown",
                "",
                "| Domain | Exps | Static Threshold (VUS-PR / F1) | "
                "Isolation Forest (VUS-PR / F1) | Half-Space Trees (VUS-PR / F1) |",
                "|---|---:|---|---|---|",
            ]
        )
        for dom, mods in sorted(dom_map.items()):
            exps = next(iter(mods.values()))["experiments"]
            st = mods.get("static_threshold", {})
            ifo = mods.get("isolation_forest", {})
            hst = mods.get("half_space_trees", {})
            lines.append(
                f"| `{dom}` | {exps} | "
                f"{_fmt(st.get('mean_vus_pr'))} / {_fmt(st.get('micro_f1'))} | "
                f"{_fmt(ifo.get('mean_vus_pr'))} / {_fmt(ifo.get('micro_f1'))} | "
                f"{_fmt(hst.get('mean_vus_pr'))} / {_fmt(hst.get('micro_f1'))} |"
            )

    lines.extend(["", "## 4. Dataset Provenance & Coverage", ""])
    for suite_name, prov in report["dataset_provenance"].items():
        lines.append(f"### {suite_name}")
        for key, val in prov.items():
            lines.append(f"- **{key}**: `{val}`")
        lines.append("")

    lines.extend(["## 5. Model Parameters & Environment", ""])
    for mod_name, params in report["model_parameters"].items():
        lines.append(f"- **`{mod_name}`**: `{json.dumps(params, sort_keys=True)}`")
    lines.append(
        f"- **Packages**: `scikit-learn={pkgs['scikit-learn']}`, `river={pkgs['river']}`, "
        f"`tsb-ad={pkgs['tsb-ad']}`, `pandas={pkgs['pandas']}`, `psutil={pkgs['psutil']}`"
    )
    lines.append(
        f"- **Python / Platform / Git**: `{repro['python_version']}` / "
        f"`{repro['platform']}` / `{_fmt(repro['git_commit'])}`"
    )

    lines.extend(["", "## 6. Research Protocol & Measurement Boundaries", ""])
    for note in report["measurement_boundaries"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


SUMMARY_CSV_FIELDS: list[str] = [
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
    "mean_vus_roc",
    "changepoints_total",
    "changepoints_flagged_60s",
    "mean_latency_ms",
    "throughput_events_per_sec",
    "total_inference_wall_s",
    "total_cpu_s",
    "cpu_utilization_pct",
    "max_peak_tracemalloc_kib",
    "max_rss_mib",
]

PER_EXPERIMENT_CSV_FIELDS: list[str] = [
    "dataset_suite",
    "experiment_id",
    "domain",
    "is_anomaly_free",
    "model",
    "n_features",
    "train_boundary_index",
    "train_normal_samples",
    "val_normal_samples",
    "dropped_train_anomalies",
    "test_samples",
    "test_anomalies",
    "test_changepoints",
    "changepoints_flagged_60s",
    "sliding_window",
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
    "fit_wall_time_s",
    "inference_wall_time_s",
    "mean_latency_ms",
    "throughput_events_per_sec",
    "cpu_time_s",
    "cpu_utilization_pct",
    "peak_tracemalloc_kib",
    "rss_memory_mib",
]


def write_m4_bundle(
    output_dir: Path,
    report: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    per_exp_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write reproducible JSON, Markdown, and CSV artefacts to `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "benchmark_report.md").write_text(render_m4_markdown(report), encoding="utf-8")

    with (output_dir / "comparison_summary.csv").open("w", encoding="utf-8", newline="") as f_sum:
        writer = csv.DictWriter(f_sum, fieldnames=SUMMARY_CSV_FIELDS)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    with (output_dir / "per_experiment_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f_exp:
        writer = csv.DictWriter(f_exp, fieldnames=PER_EXPERIMENT_CSV_FIELDS)
        writer.writeheader()
        for row in per_exp_rows:
            writer.writerow(row)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for the M4 comparative benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run DriftGuard-Kafka M4 comparative ML benchmark on SKAB and TSB-AD-M."
    )
    parser.add_argument("--run-id", default=None, help="Run identifier (defaults to UTC timestamp)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for models")
    parser.add_argument(
        "--calibration-quantile",
        type=float,
        default=0.99,
        help="Validation normal score quantile for threshold calibration",
    )
    parser.add_argument(
        "--skab-dir",
        type=Path,
        default=Path("data/external/skab"),
        help="Directory containing SKAB dataset",
    )
    parser.add_argument(
        "--tsb-dir",
        type=Path,
        default=Path("data/external/tsb_ad_m"),
        help="Directory containing TSB-AD-M dataset",
    )
    parser.add_argument(
        "--tsb-max-per-domain",
        type=int,
        default=2,
        help="Max canonical Eva series per TSB-AD-M domain (0 = all 180 Eva files)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output report directory (default: reports/m4/<run_id>)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing external datasets into data/external/ over HTTPS",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for DriftGuard-Kafka M4 benchmarking."""
    args = parse_args(argv)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (Path("reports/m4") / run_id)

    try:
        if args.download:
            ensure_skab_downloaded(args.skab_dir)
            ensure_tsb_ad_m_downloaded(args.tsb_dir)

        skab_bundle = load_skab_suite(args.skab_dir)
        tsb_max = None if args.tsb_max_per_domain <= 0 else args.tsb_max_per_domain
        tsb_bundle = load_tsb_ad_m_suite(args.tsb_dir, max_per_domain=tsb_max)
    except (DatasetNotAvailableError, DatasetFormatError, ValueError) as exc:
        print(f"M4 BENCHMARK FAILED: {exc}", file=sys.stderr)
        return 2

    all_summary: list[dict[str, Any]] = []
    all_per_exp: list[dict[str, Any]] = []
    merged_params: dict[str, Any] = {}

    for bundle in (skab_bundle, tsb_bundle):
        s_rows, e_rows, params = run_suite_benchmark(
            bundle,
            seed=args.seed,
            calibration_quantile=args.calibration_quantile,
        )
        all_summary.extend(s_rows)
        all_per_exp.extend(e_rows)
        merged_params.update(params)

    report = build_m4_report(
        run_id=run_id,
        seed=args.seed,
        calibration_quantile=args.calibration_quantile,
        suites=[skab_bundle, tsb_bundle],
        summary_rows=all_summary,
        per_exp_rows=all_per_exp,
        model_parameters=merged_params,
    )
    write_m4_bundle(output_dir, report, all_summary, all_per_exp)
    print(f"M4 comparative benchmark report generated in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
