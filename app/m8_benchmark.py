"""Comparative M8 benchmark runner: Chronos-2 & TimesFM 3.0 vs. M4–M7 models.

Evaluates zero-shot probabilistic forecasting foundation models:
  - **Amazon Chronos-2** (`amazon/chronos-2`, 119.5M params, Apache-2.0):
    `chronos2_zero_shot` (normalized residual) & `chronos2_interval` (`[0.1, 0.9]` violation)
  - **Google TimesFM 3.0** (`google/timesfm-3.0-pytorch`, 330.7M params,
    Code: Apache-2.0 / Weights: `timesfm-non-commercial-license-v1.0`):
    `timesfm3_zero_shot` (normalized residual) & `timesfm3_interval` (`[0.1, 0.9]` violation)

Compares side-by-side on identical splits against:
  - `static_threshold`, `isolation_forest`, `half_space_trees` (M4)
  - `lstm_autoencoder` (M5)
  - `tcn_autoencoder` (M6)
  - `tranad` (M7)

Measures Precision, Recall, F1, FPR, PR-AUC, VUS-PR, episode detection delay,
forecasting error (`MAE`, `RMSE`, `Normalized MAE`, `80% Interval Coverage`),
inference latency, throughput, and memory, writing `m8_report.json`,
`m8_report.md`, `comparison_summary.csv`, and `per_experiment_metrics.csv`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
import tracemalloc
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
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
from app.m5_benchmark import compute_detection_delay, evaluate_m5_detector_on_split
from app.m5_lstm_autoencoder import (
    DeviceInfo,
    LSTMAutoencoderConfig,
    LSTMAutoencoderDetector,
    select_compute_device,
)
from app.m6_tcn_autoencoder import TCNAutoencoderConfig, TCNAutoencoderDetector
from app.m7_benchmark import load_verified_prior_rows
from app.m7_tranad import TranADConfig, TranADDetector
from app.m8_chronos2 import (
    CHRONOS2_DEFAULT_MODEL_ID,
    CHRONOS2_LICENSE,
    Chronos2Detector,
    is_chronos2_installed,
)
from app.m8_timesfm import (
    TIMESFM3_DEFAULT_MODEL_ID,
    TIMESFM3_LICENSE,
    TimesFM3Detector,
    is_timesfm3_installed,
)
from app.m8_tsfm_base import (
    ForecastBackend,
    ForecastBatchOutput,
    TSFMDetectorConfig,
    ZeroShotForecastDetector,
)
from app.report import _detect_git_commit
from app.schemas import format_utc_timestamp

MODEL_ORDER_M8: tuple[str, ...] = (
    "static_threshold",
    "isolation_forest",
    "half_space_trees",
    "lstm_autoencoder",
    "tcn_autoencoder",
    "tranad",
    "chronos2_zero_shot",
    "chronos2_interval",
    "timesfm3_zero_shot",
    "timesfm3_interval",
)

M8_MEASUREMENT_BOUNDARIES: list[str] = [
    "Both Chronos-2 and TimesFM 3.0 are evaluated strictly zero-shot without gradient "
    "fine-tuning; residual normalization scales (`sigma_d`) are fit strictly on `x_train_normal`.",
    "To forecast block `[t_0, t_0 + H)`, only historical observations strictly before `t_0` "
    "(`<= t_0 - 1`) enter the context window; future test observations never leak into context.",
    "Anomaly thresholds for both `normalized_residual` and `interval_violation` (`[q_0.1, q_0.9]`) "
    "are calibrated strictly on normal validation observations (`x_val_normal`).",
    "SKAB (`8` coupled testbed channels) uses native multivariate forecasting on both Chronos-2 "
    "and TimesFM 3.0; TSB series use independent univariate forecasting per active channel.",
    "TimesFM 3.0 pretrained weights are governed by `timesfm-non-commercial-license-v1.0` "
    "(non-commercial research only), whereas Chronos-2 code and weights are `Apache-2.0`.",
    "All M1–M7 reports and model checkpoints remain untouched; M8 writes exclusively to "
    "`reports/m8/<run_id>/`.",
]


class DeterministicPersistenceBackend:
    """Lightweight causal persistence + empirical spread backend for CI smoke tests."""

    def __init__(self, model_id: str = "mock/persistence-tsfm") -> None:
        """Initialise zero-download deterministic backend."""
        self._model_id = model_id
        self._dev = select_compute_device("cpu")

    @property
    def model_id(self) -> str:
        """Return mock backend identifier."""
        return self._model_id

    @property
    def license_name(self) -> str:
        """Return Apache-2.0."""
        return "Apache-2.0 (CI Mock Backend)"

    @property
    def supports_multivariate(self) -> bool:
        """Return True."""
        return True

    @property
    def device_info(self) -> DeviceInfo:
        """Return CPU device info."""
        return self._dev

    def forecast_multivariate_blocks(
        self,
        contexts_bdl: NDArray[np.float32],
        horizon: int,
        batch_size: int,
    ) -> ForecastBatchOutput:
        """Predict causal rolling mean +/- `1.28 * std` (`[0.1, 0.9]` normal quantiles)."""
        del batch_size
        tail = contexts_bdl[:, :, -min(16, contexts_bdl.shape[2]) :]
        mu = np.mean(tail, axis=2, keepdims=True)
        sd = np.maximum(np.std(tail, axis=2, keepdims=True), 1e-3)
        point = np.repeat(mu, horizon, axis=2).astype(np.float32)
        ql = (point - 1.2816 * sd).astype(np.float32)
        qh = (point + 1.2816 * sd).astype(np.float32)
        return ForecastBatchOutput(point_forecast=point, q_low=ql, q_high=qh)


def _pkg_version(dist_name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist_name)
    except PackageNotFoundError:
        return None


def _rss_mib() -> float:
    return round(float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0), 3)


def evaluate_tsfm_dual_scores_on_split(
    detector: ZeroShotForecastDetector,
    residual_name: str,
    interval_name: str,
    split: ChronologicalSplit,
) -> list[dict[str, Any]]:
    """Run one zero-shot forecast pass on `split` and score both residual & interval modes."""
    cfg = detector.config
    # 1. Fit training-only scaler & normal context tail + score normal validation set
    fit_t0 = time.perf_counter()
    detector._feature_names = split.feature_names
    detector._fit_normal(split.x_train_normal, split.feature_names)
    if detector._train_tail is None:
        raise RuntimeError("training context tail missing after _fit_normal")
    val_details = detector.forecast_and_score(
        detector._train_tail, split.x_val_normal, split.feature_names
    )
    thr_res = float(np.quantile(val_details.residual_scores, cfg.calibration_quantile))
    thr_iv = float(np.quantile(val_details.interval_violation_scores, cfg.calibration_quantile))
    cal_wall_s = round(time.perf_counter() - fit_t0, 6)

    pre_test_tail = np.vstack([detector._train_tail, split.x_val_normal])[-cfg.context_length :]

    # 2. Untouched test evaluation (single zero-shot forecast pass!)
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    else:
        tracemalloc.reset_peak()

    cpu_t0 = time.process_time()
    wall_ns0 = time.perf_counter_ns()
    try:
        test_details = detector.forecast_and_score(pre_test_tail, split.x_test, split.feature_names)
    finally:
        wall_ns = max(1, time.perf_counter_ns() - wall_ns0)
        cpu_s = max(0.0, time.process_time() - cpu_t0)
        _, peak_bytes = tracemalloc.get_traced_memory()
        if not was_tracing:
            tracemalloc.stop()

    wall_s = wall_ns / 1_000_000_000.0
    n_test = int(split.x_test.shape[0])
    mean_lat_ms = round((wall_ns / n_test) / 1_000_000.0, 6)
    throughput = round(n_test / wall_s, 2) if wall_s > 0 else None
    peak_kib = round(peak_bytes / 1024.0, 3)
    rss_now = _rss_mib()
    backend = detector._resolve_backend()

    y_true = split.y_test_anomaly
    rows: list[dict[str, Any]] = []
    for m_name, scores, thr in (
        (residual_name, test_details.residual_scores, thr_res),
        (interval_name, test_details.interval_violation_scores, thr_iv),
    ):
        preds = (scores > thr).astype(np.int64)
        tp = int(np.sum((y_true == 1) & (preds == 1)))
        fp = int(np.sum((y_true == 0) & (preds == 1)))
        tn = int(np.sum((y_true == 0) & (preds == 0)))
        fn = int(np.sum((y_true == 1) & (preds == 0)))
        cls_m = compute_classification_metrics(tp, fp, tn, fn)
        rank_m = compute_ranking_metrics(y_true, scores, preds, split.sliding_window)
        delay_m = compute_detection_delay(y_true, preds)

        rows.append(
            {
                "dataset_suite": split.dataset_suite,
                "experiment_id": split.experiment_id,
                "domain": split.domain,
                "is_anomaly_free": split.is_anomaly_free,
                "model": m_name,
                "channel_mode": cfg.channel_mode,
                "selected_device": backend.device_info.selected,
                "n_features": len(split.feature_names),
                "train_normal_samples": int(split.x_train_normal.shape[0]),
                "val_normal_samples": int(split.x_val_normal.shape[0]),
                "test_samples": n_test,
                "test_anomalies": int(np.sum(y_true)),
                "calibrated_threshold": round(thr, 6),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": cls_m["precision"],
                "recall": cls_m["recall"],
                "f1": cls_m["f1"],
                "false_positive_rate": cls_m["false_positive_rate"],
                "pr_auc": rank_m["pr_auc"],
                "vus_pr": rank_m["vus_pr"],
                "vus_roc": rank_m["vus_roc"],
                "total_episodes": delay_m["total_episodes"],
                "detected_episodes": delay_m["detected_episodes"],
                "missed_episodes": delay_m["missed_episodes"],
                "mean_detection_delay_samples": delay_m["mean_detection_delay_samples"],
                "training_duration_s": cal_wall_s,
                "inference_wall_time_s": round(wall_s, 6),
                "mean_latency_ms": mean_lat_ms,
                "throughput_events_per_sec": throughput,
                "cpu_time_s": round(cpu_s, 6),
                "peak_tracemalloc_kib": peak_kib,
                "rss_memory_mib": rss_now,
                "forecast_mae": test_details.forecast_mae,
                "forecast_rmse": test_details.forecast_rmse,
                "normalized_mae": test_details.normalized_mae,
                "interval_coverage_80pct": test_details.interval_coverage_80pct,
            }
        )
    return rows


def _enrich_prior_row(row: Mapping[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    for k in ("forecast_mae", "forecast_rmse", "normalized_mae", "interval_coverage_80pct"):
        enriched.setdefault(k, None)
    return enriched


def aggregate_m8_rows(
    dataset_suite: str,
    evaluation_scope: str,
    model_name: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-experiment M8 rows including forecasting accuracy metrics."""
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
        "episode_detection_rate": round(det_ep / tot_ep, 4) if tot_ep > 0 else None,
        "mean_detection_delay_samples": _mean_optional(
            [r["mean_detection_delay_samples"] for r in rows]
        ),
        "mean_forecast_mae": _mean_optional([r.get("forecast_mae") for r in rows], digits=5),
        "mean_forecast_rmse": _mean_optional([r.get("forecast_rmse") for r in rows], digits=5),
        "mean_normalized_mae": _mean_optional([r.get("normalized_mae") for r in rows], digits=5),
        "mean_interval_coverage_80pct": _mean_optional(
            [r.get("interval_coverage_80pct") for r in rows], digits=4
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


def _verify_prior_m7_rows(
    split: ChronologicalSplit,
    indexed_m7: Mapping[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]] | None:
    required = {
        "static_threshold",
        "isolation_forest",
        "half_space_trees",
        "lstm_autoencoder",
        "tcn_autoencoder",
        "tranad",
    }
    candidates = indexed_m7.get((split.dataset_suite, split.experiment_id))
    if not candidates:
        return None
    if not required.issubset({str(r["model"]) for r in candidates}):
        return None
    matched: list[dict[str, Any]] = []
    for row in candidates:
        if str(row["model"]) not in required:
            continue
        if (
            int(row["train_normal_samples"]) != int(split.x_train_normal.shape[0])
            or int(row["val_normal_samples"]) != int(split.x_val_normal.shape[0])
            or int(row["test_samples"]) != int(split.x_test.shape[0])
            or int(row["test_anomalies"]) != int(split.y_test_anomaly.sum())
        ):
            return None
        matched.append(_enrich_prior_row(row))
    return matched if len(matched) == len(required) else None


def run_m8_suite_comparison(
    bundle: DatasetSuiteBundle,
    chronos_cfg: TSFMDetectorConfig,
    timesfm_cfg: TSFMDetectorConfig,
    seed: int = 42,
    indexed_m7: Mapping[tuple[str, str], list[dict[str, Any]]] | None = None,
    prior_params: Mapping[str, Any] | None = None,
    chronos_backend: ForecastBackend | None = None,
    timesfm_backend: ForecastBackend | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], int]:
    """Evaluate Chronos-2 & TimesFM 3.0 alongside M4–M7 models on `bundle`."""
    per_exp_rows: list[dict[str, Any]] = []
    model_params: dict[str, Any] = dict(prior_params) if prior_params else {}
    reused_splits = 0

    # 1. Collect M4–M7 baseline rows (either verified reuse or live execution)
    for split in bundle.splits:
        prior = _verify_prior_m7_rows(split, indexed_m7) if indexed_m7 is not None else None
        if prior is not None:
            per_exp_rows.extend(prior)
            reused_splits += 1
        else:
            q_cal = chronos_cfg.calibration_quantile
            mode = chronos_cfg.channel_mode
            detectors: list[BenchmarkDetector] = list(
                build_default_detectors(seed=seed, calibration_quantile=q_cal)
            )
            detectors.extend(
                [
                    LSTMAutoencoderDetector(
                        LSTMAutoencoderConfig(
                            window_size=8,
                            hidden_dim=12,
                            latent_dim=6,
                            num_layers=1,
                            dropout=0.0,
                            epochs=3,
                            seed=seed,
                            device="cpu",
                            calibration_quantile=q_cal,
                            channel_mode=mode,
                        )
                    ),
                    TCNAutoencoderDetector(
                        TCNAutoencoderConfig(
                            window_size=8,
                            hidden_channels=12,
                            latent_dim=4,
                            dilations=(1, 2),
                            dropout=0.0,
                            epochs=3,
                            seed=seed,
                            device="cpu",
                            calibration_quantile=q_cal,
                            channel_mode=mode,
                        )
                    ),
                    TranADDetector(
                        TranADConfig(
                            window_size=8,
                            dim_feedforward=12,
                            epochs=3,
                            dropout=0.0,
                            seed=seed,
                            device="cpu",
                            calibration_quantile=q_cal,
                            channel_mode=mode,
                        )
                    ),
                ]
            )
            for det in detectors:
                r = evaluate_m5_detector_on_split(det, split, checkpoint_dir=None)
                per_exp_rows.append(_enrich_prior_row(r))
                if det.name not in model_params:
                    model_params[det.name] = det.get_parameters()

    # 2. Run Amazon Chronos-2 across all splits
    c_det = Chronos2Detector(chronos_cfg, backend=chronos_backend)
    for split in bundle.splits:
        per_exp_rows.extend(
            evaluate_tsfm_dual_scores_on_split(
                c_det, "chronos2_zero_shot", "chronos2_interval", split
            )
        )
    model_params[f"chronos2:{bundle.suite_name}"] = c_det.get_parameters()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # 3. Run Google TimesFM 3.0 across all splits
    t_det = TimesFM3Detector(timesfm_cfg, backend=timesfm_backend)
    for split in bundle.splits:
        per_exp_rows.extend(
            evaluate_tsfm_dual_scores_on_split(
                t_det, "timesfm3_zero_shot", "timesfm3_interval", split
            )
        )
    model_params[f"timesfm3:{bundle.suite_name}"] = t_det.get_parameters()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    summary_rows: list[dict[str, Any]] = []
    if bundle.suite_name == "SKAB":
        fault_by_m: dict[str, list[dict[str, Any]]] = defaultdict(list)
        af_by_m: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in per_exp_rows:
            target = af_by_m if r["is_anomaly_free"] else fault_by_m
            target[str(r["model"])].append(r)
        for m_name in MODEL_ORDER_M8:
            if fault_by_m[m_name]:
                summary_rows.append(
                    aggregate_m8_rows("SKAB", "fault_experiments_only", m_name, fault_by_m[m_name])
                )
        for m_name in MODEL_ORDER_M8:
            if af_by_m[m_name]:
                summary_rows.append(
                    aggregate_m8_rows("SKAB", "anomaly_free_baseline", m_name, af_by_m[m_name])
                )
    else:
        by_m: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in per_exp_rows:
            by_m[str(r["model"])].append(r)
        for m_name in MODEL_ORDER_M8:
            if by_m[m_name]:
                summary_rows.append(
                    aggregate_m8_rows(
                        bundle.suite_name, "independent_univariate_protocol", m_name, by_m[m_name]
                    )
                )

    return summary_rows, per_exp_rows, model_params, reused_splits


M8_SUMMARY_CSV_FIELDS: list[str] = [
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
    "mean_forecast_mae",
    "mean_forecast_rmse",
    "mean_normalized_mae",
    "mean_interval_coverage_80pct",
    "total_training_duration_s",
    "total_inference_wall_s",
    "mean_latency_ms",
    "throughput_events_per_sec",
    "max_peak_tracemalloc_kib",
    "max_rss_mib",
]

M8_PER_EXP_CSV_FIELDS: list[str] = [
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
    "forecast_mae",
    "forecast_rmse",
    "normalized_mae",
    "interval_coverage_80pct",
]


def _fmt(val: object, suffix: str = "") -> str:
    return "unavailable" if val is None else f"{val}{suffix}"


def render_m8_markdown(report: Mapping[str, Any]) -> str:
    """Render readable M8 Time-Series Foundation Model comparative report."""
    repro = report["reproducibility"]
    dev = repro["device_info"]
    compat = report["foundation_model_compatibility"]
    lines = [
        f"# DriftGuard-Kafka M8 Foundation Model Benchmark Report — `{report['run_id']}`",
        "",
        f"Generated: `{report['generated_utc']}` | Seed: `{repro['seed']}` | "
        f"Device: `{dev['selected']}` (Apple Silicon: `{dev['is_apple_silicon']}`, "
        f"MPS Available: `{dev['mps_available']}`) | "
        f"Prior Baseline Source: `{repro['baseline_source']}`",
        "",
        "## 1. Comparative Anomaly Detection Summary (M8 TSFMs vs. M4–M7 Models)",
        "",
        "| Suite | Scope | Model | Exps | Test Rows | Micro P / R / F1 | Micro FPR | "
        "Macro F1 | Mean PR-AUC | Mean VUS-PR | Mean Delay | Cal/Train s | "
        "Latency | Throughput | Max RSS |",
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
            f"{_fmt(s['max_rss_mib'], ' MiB')} |"
        )

    tsfm_rows = [
        s
        for s in report["comparison_summary"]
        if str(s["model"]).startswith(("chronos2_", "timesfm3_"))
    ]
    if tsfm_rows:
        lines.extend(
            [
                "",
                "## 2. Zero-Shot Forecasting Accuracy & Quantile Interval Coverage",
                "",
                "| Suite | Scope | Detector Variant | Mean Forecast MAE | Mean Forecast RMSE | "
                "Mean Normalized MAE | 80% Interval Coverage (`[0.1, 0.9]`) |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for s in tsfm_rows:
            lines.append(
                f"| {s['dataset_suite']} | `{s['evaluation_scope']}` | `{s['model']}` | "
                f"{_fmt(s['mean_forecast_mae'])} | {_fmt(s['mean_forecast_rmse'])} | "
                f"{_fmt(s['mean_normalized_mae'])} | "
                f"{_fmt(s['mean_interval_coverage_80pct'])} |"
            )

    lines.extend(
        [
            "",
            "## 3. Foundation Model Compatibility, Licensing & Resource Comparison",
            "",
            "| Model | Checkpoint | Params | Weights Size | License | Multivariate API | "
            "Quantiles | Apple Silicon MPS |",
            "|---|---|---:|---:|---|---|---|---|",
        ]
    )
    for m_info in compat:
        lines.append(
            f"| **{m_info['name']}** | `{m_info['checkpoint']}` | `{m_info['parameters']}` | "
            f"`{m_info['download_size']}` | `{m_info['license']}` | "
            f"`{m_info['multivariate_support']}` | `{m_info['quantiles']}` | "
            f"`{m_info['mps_compatible']}` |"
        )

    lines.extend(["", "## 4. Dataset Splits & Coverage", ""])
    for suite_name, prov in report["dataset_provenance"].items():
        lines.append(f"### {suite_name}")
        for k, v in prov.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.append("")

    lines.extend(["## 5. Research Rules & Measurement Boundaries", ""])
    for boundary in report["measurement_boundaries"]:
        lines.append(f"- {boundary}")
    lines.append("")
    return "\n".join(lines)


def write_m8_bundle(
    output_dir: Path,
    report: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    per_exp_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write M8 JSON, Markdown, and CSV artefacts to `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "m8_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "m8_report.md").write_text(render_m8_markdown(report), encoding="utf-8")

    with (output_dir / "comparison_summary.csv").open("w", encoding="utf-8", newline="") as f_sum:
        writer = csv.DictWriter(f_sum, fieldnames=M8_SUMMARY_CSV_FIELDS)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    with (output_dir / "per_experiment_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f_exp:
        writer = csv.DictWriter(f_exp, fieldnames=M8_PER_EXP_CSV_FIELDS)
        writer.writeheader()
        for row in per_exp_rows:
            writer.writerow({k: row.get(k) for k in M8_PER_EXP_CSV_FIELDS})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for M8 TSFM anomaly detection benchmarking."""
    parser = argparse.ArgumentParser(
        description="Run DriftGuard-Kafka M8 Chronos-2 & TimesFM 3.0 benchmark."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/m8_tsfm.json"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--prediction-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--calibration-quantile", type=float, default=None)
    parser.add_argument("--max-tsb-series-rows", type=int, default=None)
    parser.add_argument("--skab-dir", type=Path, default=Path("data/external/skab"))
    parser.add_argument("--tsb-dir", type=Path, default=Path("data/external/tsb_ad_m"))
    parser.add_argument("--tsb-max-per-domain", type=int, default=None)
    parser.add_argument("--max-skab-files", type=int, default=None)
    parser.add_argument("--reuse-m7-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--mock-backends",
        action="store_true",
        help="Use lightweight deterministic mock backends for offline CI smoke tests",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for M8 foundation model benchmarking."""
    args = parse_args(argv)
    raw_cfg: dict[str, Any] = {}
    if args.config is not None and args.config.exists():
        loaded = json.loads(args.config.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            raw_cfg.update(loaded)

    seed = int(args.seed if args.seed is not None else raw_cfg.get("seed", 42))
    device = str(args.device if args.device is not None else raw_cfg.get("device", "auto"))
    ctx_len = int(
        args.context_length
        if args.context_length is not None
        else raw_cfg.get("context_length", 128)
    )
    pred_len = int(
        args.prediction_length
        if args.prediction_length is not None
        else raw_cfg.get("prediction_length", 64)
    )
    bsz = int(args.batch_size if args.batch_size is not None else raw_cfg.get("batch_size", 64))
    q_cal = float(
        args.calibration_quantile
        if args.calibration_quantile is not None
        else raw_cfg.get("calibration_quantile", 0.99)
    )
    max_tsb_rows = int(
        args.max_tsb_series_rows
        if args.max_tsb_series_rows is not None
        else raw_cfg.get("max_tsb_series_rows", 50000)
    )
    max_uv_ch = int(raw_cfg.get("max_active_univariate_channels", 16))
    cache_dir = str(
        args.cache_dir
        if args.cache_dir is not None
        else raw_cfg.get("cache_dir", "models/hf_cache")
    )
    chronos_id = str(raw_cfg.get("chronos2_model_id", CHRONOS2_DEFAULT_MODEL_ID))
    timesfm_id = str(raw_cfg.get("timesfm3_model_id", TIMESFM3_DEFAULT_MODEL_ID))
    skab_mode = str(raw_cfg.get("skab_channel_mode", "multivariate"))
    tsb_mode = str(raw_cfg.get("tsb_channel_mode", "independent_univariate"))
    tsb_max_raw = int(
        args.tsb_max_per_domain
        if args.tsb_max_per_domain is not None
        else raw_cfg.get("tsb_max_per_domain", 1)
    )

    if not args.mock_backends and (not is_chronos2_installed() or not is_timesfm3_installed()):
        print(
            "M8 BENCHMARK FAILED: chronos-forecasting>=2.0 and timesfm>=3.0 are required "
            "(install via `uv pip install -r requirements-m8.txt` or pass --mock-backends).",
            file=sys.stderr,
        )
        return 2

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (Path("reports/m8") / run_id)

    try:
        skab_bundle = load_skab_suite(args.skab_dir)
        if args.max_skab_files:
            skab_bundle = DatasetSuiteBundle(
                suite_name="SKAB",
                splits=skab_bundle.splits[: args.max_skab_files],
                provenance={
                    **skab_bundle.provenance,
                    "evaluated_files": len(skab_bundle.splits[: args.max_skab_files]),
                },
            )
        tsb_max = None if tsb_max_raw <= 0 else tsb_max_raw
        tsb_raw_bundle = load_tsb_ad_m_suite(args.tsb_dir, max_per_domain=tsb_max)
        if max_tsb_rows > 0:
            filtered_tsb = [
                s
                for s in tsb_raw_bundle.splits
                if (
                    s.x_train_normal.shape[0]
                    + s.x_val_normal.shape[0]
                    + s.x_test.shape[0]
                    + s.dropped_train_anomalies
                )
                <= max_tsb_rows
            ]
            if not filtered_tsb:
                filtered_tsb = tsb_raw_bundle.splits[:1]
            tsb_bundle = DatasetSuiteBundle(
                suite_name="TSB-AD-M",
                splits=filtered_tsb,
                provenance={
                    **tsb_raw_bundle.provenance,
                    "evaluated_files": len(filtered_tsb),
                    "evaluated_domains": sorted({s.domain for s in filtered_tsb}),
                    "domains_covered_count": len({s.domain for s in filtered_tsb}),
                    "max_tsb_series_rows_filter": max_tsb_rows,
                    "evaluated_test_rows": sum(int(s.x_test.shape[0]) for s in filtered_tsb),
                    "evaluated_test_anomalies": sum(
                        int(s.y_test_anomaly.sum()) for s in filtered_tsb
                    ),
                },
            )
        else:
            tsb_bundle = tsb_raw_bundle

        indexed_m7: dict[tuple[str, str], list[dict[str, Any]]] | None = None
        prior_params: dict[str, Any] = {}
        if args.reuse_m7_dir is not None:
            indexed_m7, prior_params = load_verified_prior_rows(args.reuse_m7_dir, "m7_report.json")
    except (DatasetNotAvailableError, DatasetFormatError, FileNotFoundError, ValueError) as exc:
        print(f"M8 BENCHMARK FAILED: {exc}", file=sys.stderr)
        return 2

    mock_c: ForecastBackend | None = (
        DeterministicPersistenceBackend("mock/chronos-2") if args.mock_backends else None
    )
    mock_t: ForecastBackend | None = (
        DeterministicPersistenceBackend("mock/timesfm-3.0") if args.mock_backends else None
    )

    all_summary: list[dict[str, Any]] = []
    all_per_exp: list[dict[str, Any]] = []
    merged_params: dict[str, Any] = dict(prior_params)
    total_reused = 0
    total_splits = 0

    for bundle, mode in ((skab_bundle, skab_mode), (tsb_bundle, tsb_mode)):
        c_cfg = TSFMDetectorConfig(
            model_id=chronos_id,
            context_length=ctx_len,
            prediction_length=pred_len,
            batch_size=bsz,
            channel_mode=mode,  # type: ignore[arg-type]
            calibration_quantile=q_cal,
            max_active_univariate_channels=max_uv_ch,
            device=device,
            cache_dir=cache_dir,
            local_files_only=args.local_files_only,
        )
        t_cfg = TSFMDetectorConfig(
            model_id=timesfm_id,
            context_length=ctx_len,
            prediction_length=pred_len,
            batch_size=bsz,
            channel_mode=mode,  # type: ignore[arg-type]
            calibration_quantile=q_cal,
            max_active_univariate_channels=max_uv_ch,
            device=device,
            cache_dir=cache_dir,
            local_files_only=args.local_files_only,
        )
        s_rows, e_rows, params, reused = run_m8_suite_comparison(
            bundle=bundle,
            chronos_cfg=c_cfg,
            timesfm_cfg=t_cfg,
            seed=seed,
            indexed_m7=indexed_m7,
            prior_params=merged_params,
            chronos_backend=mock_c,
            timesfm_backend=mock_t,
        )
        all_summary.extend(s_rows)
        all_per_exp.extend(e_rows)
        merged_params.update(params)
        total_reused += reused
        total_splits += len(bundle.splits)

    if args.reuse_m7_dir is not None and total_reused == total_splits:
        baseline_src = f"reused_verified_m7_bundle:{args.reuse_m7_dir.name}"
    elif total_reused > 0:
        baseline_src = f"hybrid_reused_{total_reused}_of_{total_splits}_from:{args.reuse_m7_dir}"
    else:
        baseline_src = "executed_in_process"

    dev_info = select_compute_device(device)
    compat_table = [
        {
            "name": "Amazon Chronos-2",
            "checkpoint": chronos_id,
            "package": f"chronos-forecasting=={_pkg_version('chronos-forecasting') or 'optional'}",
            "parameters": "119,477,664 (~120M)",
            "download_size": "455.8 MiB (safetensors)",
            "license": CHRONOS2_LICENSE,
            "multivariate_support": "Native group attention (3D batch x variates x context)",
            "quantiles": "[0.1, 0.5, 0.9] (configurable quantile_levels)",
            "mps_compatible": "Verified on Apple Silicon mps + cpu fallback",
        },
        {
            "name": "Google TimesFM 3.0",
            "checkpoint": timesfm_id,
            "package": f"timesfm=={_pkg_version('timesfm') or 'optional'}",
            "parameters": "330,710,976 (~331M)",
            "download_size": "1.23 GiB (safetensors)",
            "license": TIMESFM3_LICENSE,
            "multivariate_support": "Native variate attention (2D variates x context)",
            "quantiles": "9 deciles [0.1, 0.2, ..., 0.9]",
            "mps_compatible": "Verified on Apple Silicon mps + cpu fallback",
        },
    ]

    report: dict[str, Any] = {
        "run_id": run_id,
        "generated_utc": format_utc_timestamp(datetime.now(timezone.utc)),
        "reproducibility": {
            "seed": seed,
            "device_info": dev_info.to_dict(),
            "baseline_source": baseline_src,
            "mock_backends_used": bool(args.mock_backends),
            "reused_prior_splits": total_reused,
            "total_splits": total_splits,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine_arch": platform.machine(),
            "git_commit": _detect_git_commit(),
            "package_versions": {
                "torch": _pkg_version("torch"),
                "chronos-forecasting": _pkg_version("chronos-forecasting"),
                "timesfm": _pkg_version("timesfm"),
                "scikit-learn": _pkg_version("scikit-learn"),
                "river": _pkg_version("river"),
                "tsb-ad": _pkg_version("tsb-ad"),
            },
        },
        "tsfm_configuration": {
            "context_length": ctx_len,
            "prediction_length": pred_len,
            "batch_size": bsz,
            "calibration_quantile": q_cal,
            "max_active_univariate_channels": max_uv_ch,
            "max_tsb_series_rows": max_tsb_rows,
            "skab_channel_mode": skab_mode,
            "tsb_channel_mode": tsb_mode,
        },
        "foundation_model_compatibility": compat_table,
        "dataset_provenance": {
            "SKAB": skab_bundle.provenance,
            "TSB-AD-M": tsb_bundle.provenance,
        },
        "model_parameters": merged_params,
        "comparison_summary": all_summary,
        "per_experiment_count": len(all_per_exp),
        "measurement_boundaries": M8_MEASUREMENT_BOUNDARIES,
    }

    write_m8_bundle(output_dir, report, all_summary, all_per_exp)
    print(f"M8 Foundation Model report generated in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
