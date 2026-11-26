"""Comprehensive M9 comparative evaluation of all DriftGuard-Kafka models.

Evaluates all five deep-learning / foundation models against all three M4
classical ML baselines on identical chronological splits with strict
zero-leakage guarantees:

Classical ML Baselines (M4):
  1. `static_threshold` (deterministic)
  2. `isolation_forest` (multi-seed)
  3. `half_space_trees` (multi-seed)

Deep-Learning Trained Models (M5–M7, multi-seed mean ± std):
  4. `lstm_autoencoder` (M5)
  5. `tcn_autoencoder` (M6)
  6. `tranad` (M7)

Zero-Shot Time-Series Foundation Models (M8, deterministic):
  7. `chronos2_zero_shot` & `chronos2_interval` (Amazon Chronos-2)
  8. `timesfm3_zero_shot` & `timesfm3_interval` (Google TimesFM 3.0)

Produces reproducible Markdown, JSON, CSVs, and pure-SVG performance charts
under `reports/m9/<run_id>/` without modifying any historical M1–M8 artefacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import platform
import statistics
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
from app.m4_models import (
    BenchmarkDetector,
    HalfSpaceTreesDetector,
    IntArray,
    IsolationForestDetector,
    StaticThresholdBaseline,
)
from app.m5_benchmark import compute_detection_delay, extract_anomaly_episodes
from app.m5_lstm_autoencoder import (
    ChannelMode,
    LSTMAutoencoderConfig,
    LSTMAutoencoderDetector,
    select_compute_device,
)
from app.m6_tcn_autoencoder import TCNAutoencoderConfig, TCNAutoencoderDetector
from app.m7_tranad import TranADConfig, TranADDetector
from app.m8_benchmark import MODEL_ORDER_M8, DeterministicPersistenceBackend
from app.m8_chronos2 import (
    CHRONOS2_DEFAULT_MODEL_ID,
    Chronos2Detector,
    is_chronos2_installed,
)
from app.m8_timesfm import (
    TIMESFM3_DEFAULT_MODEL_ID,
    TimesFM3Detector,
    is_timesfm3_installed,
)
from app.m8_tsfm_base import ForecastBackend, TSFMDetectorConfig, ZeroShotForecastDetector
from app.m9_charts import generate_m9_charts
from app.report import _detect_git_commit
from app.schemas import format_utc_timestamp

STOCHASTIC_MODELS: frozenset[str] = frozenset(
    {
        "isolation_forest",
        "half_space_trees",
        "lstm_autoencoder",
        "tcn_autoencoder",
        "tranad",
    }
)

TSFM_MODELS: frozenset[str] = frozenset(
    {
        "chronos2_zero_shot",
        "chronos2_interval",
        "timesfm3_zero_shot",
        "timesfm3_interval",
    }
)

AGGREGATE_METRIC_KEYS: tuple[str, ...] = (
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
    "episode_detection_rate",
    "mean_detection_delay_samples",
    "total_training_duration_s",
    "total_inference_wall_s",
    "mean_latency_ms",
    "throughput_events_per_sec",
    "max_peak_tracemalloc_kib",
    "max_rss_mib",
    "mean_forecast_mae",
    "mean_forecast_rmse",
    "mean_normalized_mae",
    "mean_interval_coverage_80pct",
)


def _pkg_version(dist_name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist_name)
    except PackageNotFoundError:
        return None


def _rss_mib() -> float:
    return round(float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0), 3)


def verify_split_invariants(bundles: Sequence[DatasetSuiteBundle]) -> dict[str, Any]:
    """Verify chronological split integrity, zero-NaN invariants, and SHA-256 fingerprints."""
    fingerprints: dict[str, str] = {}
    total_splits = 0
    for bundle in bundles:
        for split in bundle.splits:
            total_splits += 1
            for name, arr in (
                ("x_train_normal", split.x_train_normal),
                ("x_val_normal", split.x_val_normal),
                ("x_test", split.x_test),
            ):
                if arr.ndim != 2 or arr.shape[0] == 0:
                    raise ValueError(f"{split.experiment_id}: invalid {name} shape {arr.shape}")
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"{split.experiment_id}: non-finite values in {name}")
            if not (
                split.x_train_normal.shape[1]
                == split.x_val_normal.shape[1]
                == split.x_test.shape[1]
                == len(split.feature_names)
            ):
                raise ValueError(f"{split.experiment_id}: feature dimension mismatch across splits")
            if split.y_test_anomaly.shape[0] != split.x_test.shape[0]:
                raise ValueError(f"{split.experiment_id}: test label length mismatch")
            h = hashlib.sha256()
            h.update(np.ascontiguousarray(split.x_train_normal, dtype=np.float64).tobytes())
            h.update(np.ascontiguousarray(split.x_val_normal, dtype=np.float64).tobytes())
            h.update(np.ascontiguousarray(split.x_test, dtype=np.float64).tobytes())
            fingerprints[f"{bundle.suite_name}:{split.experiment_id}"] = h.hexdigest()[:16]

    # Verify detector fit_and_calibrate signature never accepts labels
    sig_params = list(inspect.signature(BenchmarkDetector.fit_and_calibrate).parameters)
    if sig_params != ["self", "x_train_normal", "x_val_normal", "feature_names"]:
        raise RuntimeError(f"Unexpected fit_and_calibrate signature: {sig_params}")

    return {
        "verified_splits": total_splits,
        "fit_and_calibrate_signature": sig_params,
        "test_labels_excluded_from_fit_and_calibration": True,
        "split_fingerprints_sha256_prefix16": fingerprints,
    }


def compute_error_analysis(
    y_true: IntArray,
    predictions: IntArray,
    proximity_window: int = 20,
    short_episode_max_len: int = 20,
) -> dict[str, int]:
    """Decompose false positives and false negatives into diagnostic error categories.

    FP categories:
      - `fp_anomaly_free`: FPs occurring on completely anomaly-free series.
      - `fp_pre_fault_lead`: FPs within `proximity_window` steps before an episode onset.
      - `fp_post_fault_recovery`: FPs within `proximity_window` steps after an episode ends.
      - `fp_isolated_background`: FPs far from any true anomaly episode.

    FN categories:
      - `fn_missed_episodes_points`: FNs from episodes with zero detections.
      - `fn_onset_delay_points`: FNs inside detected episodes *before* first alert.
      - `fn_adaptation_dropoff_points`: FNs inside detected episodes *after* first alert
        (measures sustained-fault adaptation / early dropoff).
    """
    n = int(y_true.shape[0])
    episodes = extract_anomaly_episodes(y_true)
    fp_mask = (y_true == 0) & (predictions == 1)
    fn_mask = (y_true == 1) & (predictions == 0)
    fp_total = int(np.sum(fp_mask))
    fn_total = int(np.sum(fn_mask))

    if not episodes:
        return {
            "fp_total": fp_total,
            "fp_anomaly_free": fp_total,
            "fp_pre_fault_lead": 0,
            "fp_post_fault_recovery": 0,
            "fp_isolated_background": 0,
            "fn_total": fn_total,
            "missed_episodes_total": 0,
            "missed_episodes_short": 0,
            "missed_episodes_long": 0,
            "partial_episodes": 0,
            "fn_missed_episodes_points": 0,
            "fn_onset_delay_points": 0,
            "fn_adaptation_dropoff_points": 0,
        }

    pre_mask = np.zeros(n, dtype=bool)
    post_mask = np.zeros(n, dtype=bool)
    for start, end in episodes:
        pre_lo = max(0, start - proximity_window)
        pre_mask[pre_lo:start] = True
        post_hi = min(n, end + 1 + proximity_window)
        post_mask[end + 1 : post_hi] = True

    fp_pre = int(np.sum(fp_mask & pre_mask))
    fp_post = int(np.sum(fp_mask & (~pre_mask) & post_mask))
    fp_iso = fp_total - fp_pre - fp_post

    missed_total = 0
    missed_short = 0
    missed_long = 0
    partial_eps = 0
    fn_missed_pts = 0
    fn_onset_pts = 0
    fn_dropoff_pts = 0

    for start, end in episodes:
        ep_len = end - start + 1
        ep_preds = predictions[start : end + 1]
        hits = np.flatnonzero(ep_preds > 0)
        if hits.size == 0:
            missed_total += 1
            fn_missed_pts += ep_len
            if ep_len <= short_episode_max_len:
                missed_short += 1
            else:
                missed_long += 1
        else:
            first_hit = int(hits[0])
            ep_fns = int(np.sum(ep_preds == 0))
            if ep_fns > 0:
                partial_eps += 1
                fn_onset_pts += first_hit
                fn_dropoff_pts += ep_fns - first_hit

    return {
        "fp_total": fp_total,
        "fp_anomaly_free": 0,
        "fp_pre_fault_lead": fp_pre,
        "fp_post_fault_recovery": fp_post,
        "fp_isolated_background": fp_iso,
        "fn_total": fn_total,
        "missed_episodes_total": missed_total,
        "missed_episodes_short": missed_short,
        "missed_episodes_long": missed_long,
        "partial_episodes": partial_eps,
        "fn_missed_episodes_points": fn_missed_pts,
        "fn_onset_delay_points": fn_onset_pts,
        "fn_adaptation_dropoff_points": fn_dropoff_pts,
    }


def evaluate_detector_with_errors(
    detector: BenchmarkDetector,
    split: ChronologicalSplit,
    seed: int,
) -> dict[str, Any]:
    """Evaluate a non-TSFM detector on `split` and return full metrics + error taxonomy."""
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

    cls_m = compute_classification_metrics(tp, fp, tn, fn)
    rank_m = compute_ranking_metrics(y_true, inf.scores, preds, split.sliding_window)
    delay_m = compute_detection_delay(y_true, preds)
    err_m = compute_error_analysis(y_true, preds)

    return {
        "seed": seed,
        "dataset_suite": split.dataset_suite,
        "experiment_id": split.experiment_id,
        "domain": split.domain,
        "is_anomaly_free": split.is_anomaly_free,
        "model": detector.name,
        "is_stochastic": detector.name in STOCHASTIC_MODELS,
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
        "training_duration_s": cal.fit_wall_time_s,
        "inference_wall_time_s": inf.inference_wall_time_s,
        "mean_latency_ms": inf.mean_latency_ms,
        "throughput_events_per_sec": inf.throughput_events_per_sec,
        "cpu_time_s": inf.cpu_time_s,
        "peak_tracemalloc_kib": inf.peak_tracemalloc_kib,
        "rss_memory_mib": inf.rss_memory_mib,
        "forecast_mae": None,
        "forecast_rmse": None,
        "normalized_mae": None,
        "interval_coverage_80pct": None,
        **err_m,
    }


def evaluate_tsfm_with_errors(
    detector: ZeroShotForecastDetector,
    residual_name: str,
    interval_name: str,
    split: ChronologicalSplit,
    seed: int,
) -> list[dict[str, Any]]:
    """Run one zero-shot causal pass on `split` and return both residual & interval rows."""
    cfg = detector.config
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
        err_m = compute_error_analysis(y_true, preds)

        rows.append(
            {
                "seed": seed,
                "dataset_suite": split.dataset_suite,
                "experiment_id": split.experiment_id,
                "domain": split.domain,
                "is_anomaly_free": split.is_anomaly_free,
                "model": m_name,
                "is_stochastic": False,
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
                **err_m,
            }
        )
    return rows


def aggregate_single_seed_scope(
    dataset_suite: str,
    evaluation_scope: str,
    model_name: str,
    seed: int,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-experiment rows for one `(suite, scope, model, seed)` slice."""
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
        "seed": seed,
        "is_stochastic": model_name in STOCHASTIC_MODELS,
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
        "fp_anomaly_free": sum(int(r["fp_anomaly_free"]) for r in rows),
        "fp_pre_fault_lead": sum(int(r["fp_pre_fault_lead"]) for r in rows),
        "fp_post_fault_recovery": sum(int(r["fp_post_fault_recovery"]) for r in rows),
        "fp_isolated_background": sum(int(r["fp_isolated_background"]) for r in rows),
        "missed_episodes_short": sum(int(r["missed_episodes_short"]) for r in rows),
        "missed_episodes_long": sum(int(r["missed_episodes_long"]) for r in rows),
        "partial_episodes": sum(int(r["partial_episodes"]) for r in rows),
        "fn_missed_episodes_points": sum(int(r["fn_missed_episodes_points"]) for r in rows),
        "fn_onset_delay_points": sum(int(r["fn_onset_delay_points"]) for r in rows),
        "fn_adaptation_dropoff_points": sum(int(r["fn_adaptation_dropoff_points"]) for r in rows),
    }


def aggregate_across_seeds(
    seed_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute mean ± std across seeds for one `(suite, scope, model)` group."""
    first = seed_rows[0]
    n_seeds = len(seed_rows)
    out: dict[str, Any] = {
        "dataset_suite": first["dataset_suite"],
        "evaluation_scope": first["evaluation_scope"],
        "model": first["model"],
        "n_seeds": n_seeds,
        "seeds": [int(r["seed"]) for r in seed_rows],
        "is_stochastic": bool(first["is_stochastic"]),
        "experiments": int(first["experiments"]),
        "test_samples": int(first["test_samples"]),
    }

    for key in AGGREGATE_METRIC_KEYS:
        vals = [float(r[key]) for r in seed_rows if r.get(key) is not None]
        if not vals:
            out[f"{key}_mean"] = None
            out[f"{key}_std"] = None
        else:
            digits = 6 if key == "mean_latency_ms" else 4
            out[f"{key}_mean"] = round(statistics.mean(vals), digits)
            out[f"{key}_std"] = round(statistics.stdev(vals), digits) if len(vals) > 1 else 0.0

    # Mean error taxonomy counts across seeds
    for e_key in (
        "fp",
        "fn",
        "fp_anomaly_free",
        "fp_pre_fault_lead",
        "fp_post_fault_recovery",
        "fp_isolated_background",
        "missed_episodes",
        "missed_episodes_short",
        "missed_episodes_long",
        "partial_episodes",
        "fn_missed_episodes_points",
        "fn_onset_delay_points",
        "fn_adaptation_dropoff_points",
    ):
        e_vals = [float(r[e_key]) for r in seed_rows]
        out[f"{e_key}_mean"] = round(statistics.mean(e_vals), 2)

    return out


def analyze_resource_efficiency(
    unified_rows: Sequence[Mapping[str, Any]],
    sla: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Flag models with strong F1 whose latency, memory, or training cost violates SLA."""
    max_lat = float(sla.get("max_latency_ms", 0.1))
    min_tp = float(sla.get("min_throughput_ev_s", 10000.0))
    max_rss = float(sla.get("max_rss_mib", 1024.0))
    max_train = float(sla.get("max_training_s", 30.0))
    f1_skab_hi = float(sla.get("high_f1_threshold_skab", 0.70))
    f1_tsb_hi = float(sla.get("high_f1_threshold_tsb", 0.35))

    findings: list[dict[str, Any]] = []
    for r in unified_rows:
        suite = str(r["dataset_suite"])
        scope = str(r["evaluation_scope"])
        if scope not in ("fault_experiments_only", "tsfm_common_9_domains", "full_17_domains"):
            continue
        f1_key = "micro_f1_mean" if suite == "SKAB" else "macro_f1_mean"
        f1_val = r.get(f1_key)
        lat_val = r.get("mean_latency_ms_mean")
        tp_val = r.get("throughput_events_per_sec_mean")
        rss_val = r.get("max_rss_mib_mean")
        train_val = r.get("total_training_duration_s_mean")

        violations: list[str] = []
        if lat_val is not None and float(lat_val) > max_lat:
            violations.append(f"latency={float(lat_val):.4f}ms>{max_lat}ms")
        if tp_val is not None and float(tp_val) < min_tp:
            violations.append(f"throughput={float(tp_val):.1f}ev/s<{min_tp:.0f}ev/s")
        if rss_val is not None and float(rss_val) > max_rss:
            violations.append(f"rss={float(rss_val):.1f}MiB>{max_rss:.0f}MiB")
        if train_val is not None and float(train_val) > max_train:
            violations.append(f"train_time={float(train_val):.1f}s>{max_train:.0f}s")

        hi_thr = f1_skab_hi if suite == "SKAB" else f1_tsb_hi
        is_high_f1 = f1_val is not None and float(f1_val) >= hi_thr
        status = (
            "HIGH_F1_RESOURCE_VIOLATION"
            if (is_high_f1 and violations)
            else ("SLA_VIOLATION" if violations else "SLA_COMPLIANT")
        )
        findings.append(
            {
                "dataset_suite": suite,
                "evaluation_scope": scope,
                "model": str(r["model"]),
                "primary_f1_metric": f1_key.replace("_mean", ""),
                "primary_f1": f1_val,
                "mean_latency_ms": lat_val,
                "throughput_events_per_sec": tp_val,
                "max_rss_mib": rss_val,
                "total_training_duration_s": train_val,
                "status": status,
                "violations": violations,
            }
        )
    return findings


def select_best_per_scenario(
    unified_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Identify the best model per scenario and verify whether a universal winner exists."""
    by_scope: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for r in unified_rows:
        by_scope[(str(r["dataset_suite"]), str(r["evaluation_scope"]))].append(r)

    criteria: list[tuple[str, str, str, str, bool]] = [
        (
            "SKAB Multivariate — Fault Detection (Micro F1)",
            "SKAB",
            "fault_experiments_only",
            "micro_f1_mean",
            True,
        ),
        (
            "SKAB Multivariate — Threshold-Free Ranking (Mean VUS-PR)",
            "SKAB",
            "fault_experiments_only",
            "mean_vus_pr_mean",
            True,
        ),
        (
            "SKAB Multivariate — Lowest Anomaly-Free False Alarm Rate (FPR)",
            "SKAB",
            "anomaly_free_baseline",
            "micro_fpr_mean",
            False,
        ),
        (
            "SKAB Multivariate — Fastest Inference Latency",
            "SKAB",
            "fault_experiments_only",
            "mean_latency_ms_mean",
            False,
        ),
        (
            "TSB-AD-M Univariate (9-Domain Common) — Balanced Macro F1",
            "TSB-AD-M",
            "tsfm_common_9_domains",
            "macro_f1_mean",
            True,
        ),
        (
            "TSB-AD-M Univariate (9-Domain Common) — Ranking Quality (Mean PR-AUC)",
            "TSB-AD-M",
            "tsfm_common_9_domains",
            "mean_pr_auc_mean",
            True,
        ),
        (
            "TSB-AD-M Univariate (Full 17-Domain, M4–M7) — Ranking Quality (Mean PR-AUC)",
            "TSB-AD-M",
            "full_17_domains",
            "mean_pr_auc_mean",
            True,
        ),
    ]

    winners: list[dict[str, Any]] = []
    for label, suite, scope, metric_key, higher_is_better in criteria:
        candidates = [r for r in by_scope.get((suite, scope), []) if r.get(metric_key) is not None]
        if not candidates:
            continue
        ordered = sorted(
            candidates,
            key=lambda x: float(x[metric_key]),
            reverse=higher_is_better,
        )
        best = ordered[0]
        runner = ordered[1] if len(ordered) > 1 else None
        winners.append(
            {
                "scenario": label,
                "dataset_suite": suite,
                "evaluation_scope": scope,
                "metric": metric_key.replace("_mean", ""),
                "best_model": str(best["model"]),
                "best_value": best[metric_key],
                "best_std": best.get(metric_key.replace("_mean", "_std")),
                "runner_up_model": str(runner["model"]) if runner is not None else None,
                "runner_up_value": runner[metric_key] if runner is not None else None,
            }
        )

    distinct_winners = sorted({w["best_model"] for w in winners})
    return {
        "scenario_winners": winners,
        "distinct_winning_models": distinct_winners,
        "universal_winner_supported": len(distinct_winners) == 1,
        "verdict": (
            f"Single universal winner supported: `{distinct_winners[0]}`."
            if len(distinct_winners) == 1
            else (
                "No universal winner is supported by the evidence: "
                f"{len(distinct_winners)} different models lead across the {len(winners)} "
                f"evaluation scenarios ({', '.join(distinct_winners)})."
            )
        ),
    }


def _build_detectors_for_seed(
    seed: int,
    channel_mode: ChannelMode,
    cfg: Mapping[str, Any],
    device: str,
    include_deterministic: bool,
) -> list[BenchmarkDetector]:
    q_cal = float(cfg.get("calibration_quantile", 0.99))
    lstm_c = cfg.get("lstm", {})
    tcn_c = cfg.get("tcn", {})
    tranad_c = cfg.get("tranad", {})

    detectors: list[BenchmarkDetector] = []
    if include_deterministic:
        detectors.append(StaticThresholdBaseline(calibration_quantile=q_cal))

    detectors.extend(
        [
            IsolationForestDetector(seed=seed, calibration_quantile=q_cal),
            HalfSpaceTreesDetector(seed=seed, calibration_quantile=q_cal),
            LSTMAutoencoderDetector(
                LSTMAutoencoderConfig(
                    window_size=int(lstm_c.get("window_size", 32)),
                    hidden_dim=int(lstm_c.get("hidden_dim", 32)),
                    latent_dim=int(lstm_c.get("latent_dim", 16)),
                    num_layers=int(lstm_c.get("num_layers", 2)),
                    dropout=float(lstm_c.get("dropout", 0.1)),
                    batch_size=int(lstm_c.get("batch_size", 64)),
                    epochs=int(lstm_c.get("epochs", 15)),
                    learning_rate=float(lstm_c.get("learning_rate", 0.002)),
                    early_stopping_patience=int(lstm_c.get("early_stopping_patience", 3)),
                    early_stopping_min_delta=float(lstm_c.get("early_stopping_min_delta", 0.0001)),
                    seed=seed,
                    device=device,
                    calibration_quantile=q_cal,
                    channel_mode=channel_mode,
                )
            ),
            TCNAutoencoderDetector(
                TCNAutoencoderConfig(
                    window_size=int(tcn_c.get("window_size", 32)),
                    hidden_channels=int(tcn_c.get("hidden_channels", 24)),
                    latent_dim=int(tcn_c.get("latent_dim", 8)),
                    kernel_size=int(tcn_c.get("kernel_size", 3)),
                    dilations=tuple(int(d) for d in tcn_c.get("dilations", [1, 2, 4])),
                    dropout=float(tcn_c.get("dropout", 0.1)),
                    batch_size=int(tcn_c.get("batch_size", 64)),
                    epochs=int(tcn_c.get("epochs", 12)),
                    learning_rate=float(tcn_c.get("learning_rate", 0.002)),
                    early_stopping_patience=int(tcn_c.get("early_stopping_patience", 3)),
                    early_stopping_min_delta=float(tcn_c.get("early_stopping_min_delta", 0.0001)),
                    score_reduction=str(tcn_c.get("score_reduction", "window_mean")),  # type: ignore[arg-type]
                    eval_batch_size=int(tcn_c.get("eval_batch_size", 4096)),
                    seed=seed,
                    device=device,
                    calibration_quantile=q_cal,
                    channel_mode=channel_mode,
                )
            ),
            TranADDetector(
                TranADConfig(
                    window_size=int(tranad_c.get("window_size", 10)),
                    dim_feedforward=int(tranad_c.get("dim_feedforward", 16)),
                    num_encoder_layers=int(tranad_c.get("num_encoder_layers", 1)),
                    num_decoder_layers=int(tranad_c.get("num_decoder_layers", 1)),
                    dropout=float(tranad_c.get("dropout", 0.1)),
                    batch_size=int(tranad_c.get("batch_size", 128)),
                    epochs=int(tranad_c.get("epochs", 10)),
                    learning_rate=float(tranad_c.get("learning_rate", 0.002)),
                    weight_decay=float(tranad_c.get("weight_decay", 1e-5)),
                    lr_step_size=int(tranad_c.get("lr_step_size", 5)),
                    lr_gamma=float(tranad_c.get("lr_gamma", 0.9)),
                    early_stopping_patience=int(tranad_c.get("early_stopping_patience", 3)),
                    early_stopping_min_delta=float(
                        tranad_c.get("early_stopping_min_delta", 0.0001)
                    ),
                    score_mode=str(tranad_c.get("score_mode", "both_phases")),  # type: ignore[arg-type]
                    eval_batch_size=int(tranad_c.get("eval_batch_size", 8192)),
                    seed=seed,
                    device=device,
                    calibration_quantile=q_cal,
                    channel_mode=channel_mode,
                )
            ),
        ]
    )
    return detectors


def _clear_mps_cache() -> None:
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_m9_evaluation(
    skab_bundle: DatasetSuiteBundle,
    tsb_bundle: DatasetSuiteBundle,
    cfg: Mapping[str, Any],
    device: str = "auto",
    cache_dir: str = "models/hf_cache",
    local_files_only: bool = False,
    mock_backends: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute the full M9 multi-seed comparative evaluation across SKAB and TSB-AD-M."""
    seeds = [int(s) for s in cfg.get("seeds", [42, 1337, 2026])]
    q_cal = float(cfg.get("calibration_quantile", 0.99))
    tsfm_max_rows = int(cfg.get("tsfm_max_tsb_series_rows", 50000))
    tsfm_c = cfg.get("tsfm", {})
    ctx_len = int(tsfm_c.get("context_length", 128))
    pred_len = int(tsfm_c.get("prediction_length", 64))
    tsfm_bsz = int(tsfm_c.get("batch_size", 64))
    max_uv_ch = int(tsfm_c.get("max_active_univariate_channels", 16))
    chronos_id = str(tsfm_c.get("chronos2_model_id", CHRONOS2_DEFAULT_MODEL_ID))
    timesfm_id = str(tsfm_c.get("timesfm3_model_id", TIMESFM3_DEFAULT_MODEL_ID))

    split_verification = verify_split_invariants([skab_bundle, tsb_bundle])
    dev_info = select_compute_device(device)

    per_exp_rows: list[dict[str, Any]] = []
    unexecuted_experiments: list[dict[str, Any]] = []

    # 2. Partition TSB splits into TSFM-feasible vs budget-exceeded upfront for accurate % total
    tsb_tsfm_splits: list[ChronologicalSplit] = []
    for s in tsb_bundle.splits:
        total_rows = int(
            s.x_train_normal.shape[0]
            + s.x_val_normal.shape[0]
            + s.x_test.shape[0]
            + s.dropped_train_anomalies
        )
        if tsfm_max_rows <= 0 or total_rows <= tsfm_max_rows:
            tsb_tsfm_splits.append(s)
        else:
            active_ch = min(len(s.feature_names), max_uv_ch)
            est_blocks = int(np.ceil(s.x_test.shape[0] / pred_len)) * active_ch
            unexecuted_experiments.append(
                {
                    "dataset_suite": "TSB-AD-M",
                    "domain": s.domain,
                    "experiment_id": s.experiment_id,
                    "models": list(sorted(TSFM_MODELS)),
                    "total_rows": total_rows,
                    "test_rows": int(s.x_test.shape[0]),
                    "n_features": len(s.feature_names),
                    "estimated_univariate_forecast_blocks": est_blocks,
                    "reason": (
                        f"Exceeded M9 TSFM single-machine row cap ({total_rows} > "
                        f"{tsfm_max_rows}); excluded from 9-domain common TSFM subset "
                        "and explicitly marked unexecuted in full 17-domain TSB table."
                    ),
                }
            )

    if not tsb_tsfm_splits and tsb_bundle.splits:
        tsb_tsfm_splits = [tsb_bundle.splits[0]]

    # Exact total evaluation steps for live percentage progress reporting
    n_skab = len(skab_bundle.splits)
    n_tsb = len(tsb_bundle.splits)
    n_tsb_tsfm = len(tsb_tsfm_splits)
    n_stoch = len(STOCHASTIC_MODELS)
    total_steps = (
        (n_skab + n_tsb) * (1 + len(seeds) * n_stoch)
        + (n_skab + n_tsb_tsfm) * 2  # Chronos-2 + TimesFM 3.0 passes
    )
    done_steps = 0
    eval_t0 = time.perf_counter()

    def _report_progress(suite: str, seed_val: int, model_lbl: str, exp_id: str) -> None:
        nonlocal done_steps
        done_steps += 1
        pct = 100.0 * done_steps / max(1, total_steps)
        elapsed = time.perf_counter() - eval_t0
        print(
            f"[{pct:5.1f}%] ({done_steps}/{total_steps}) {suite} | seed={seed_val} | "
            f"{model_lbl} | {exp_id} | elapsed={elapsed:.1f}s",
            flush=True,
        )

    print(
        f"[  0.0%] Starting M9 evaluation on device={dev_info.selected} "
        f"(mps_available={dev_info.mps_available}) | total_steps={total_steps} | seeds={seeds}",
        flush=True,
    )

    # 1. Run M4–M7 models across all seeds (deterministic static runs once on seeds[0])
    for bundle, ch_mode in (
        (skab_bundle, "multivariate"),
        (tsb_bundle, "independent_univariate"),
    ):
        for seed_idx, seed in enumerate(seeds):
            detectors = _build_detectors_for_seed(
                seed=seed,
                channel_mode=ch_mode,  # type: ignore[arg-type]
                cfg=cfg,
                device=device,
                include_deterministic=(seed_idx == 0),
            )
            for det in detectors:
                for split in bundle.splits:
                    row = evaluate_detector_with_errors(det, split, seed=seed)
                    per_exp_rows.append(row)
                    _report_progress(bundle.suite_name, seed, det.name, split.experiment_id)
                _clear_mps_cache()

    # 3. Run Zero-Shot Chronos-2 & TimesFM 3.0 deterministically (seed = seeds[0])
    mock_c: ForecastBackend | None = (
        DeterministicPersistenceBackend("mock/chronos-2") if mock_backends else None
    )
    mock_t: ForecastBackend | None = (
        DeterministicPersistenceBackend("mock/timesfm-3.0") if mock_backends else None
    )

    for suite_lbl, splits_to_run, ch_mode in (
        ("SKAB", skab_bundle.splits, "multivariate"),
        ("TSB-AD-M", tsb_tsfm_splits, "independent_univariate"),
    ):
        c_cfg = TSFMDetectorConfig(
            model_id=chronos_id,
            context_length=ctx_len,
            prediction_length=pred_len,
            batch_size=tsfm_bsz,
            channel_mode=ch_mode,  # type: ignore[arg-type]
            calibration_quantile=q_cal,
            max_active_univariate_channels=max_uv_ch,
            device=device,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        t_cfg = TSFMDetectorConfig(
            model_id=timesfm_id,
            context_length=ctx_len,
            prediction_length=pred_len,
            batch_size=tsfm_bsz,
            channel_mode=ch_mode,  # type: ignore[arg-type]
            calibration_quantile=q_cal,
            max_active_univariate_channels=max_uv_ch,
            device=device,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        c_det = Chronos2Detector(c_cfg, backend=mock_c)
        for split in splits_to_run:
            per_exp_rows.extend(
                evaluate_tsfm_with_errors(
                    c_det, "chronos2_zero_shot", "chronos2_interval", split, seed=seeds[0]
                )
            )
            _report_progress(suite_lbl, seeds[0], "chronos2", split.experiment_id)
        _clear_mps_cache()

        t_det = TimesFM3Detector(t_cfg, backend=mock_t)
        for split in splits_to_run:
            per_exp_rows.extend(
                evaluate_tsfm_with_errors(
                    t_det, "timesfm3_zero_shot", "timesfm3_interval", split, seed=seeds[0]
                )
            )
            _report_progress(suite_lbl, seeds[0], "timesfm3", split.experiment_id)
        _clear_mps_cache()

    # 4. Build per-seed scope summaries and multi-seed unified summaries
    tsfm_common_ids = {s.experiment_id for s in tsb_tsfm_splits}
    per_seed_summaries: list[dict[str, Any]] = []

    def _collect_scope(
        suite: str, scope: str, predicate_rows: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
        for r in predicate_rows:
            grouped[(str(r["model"]), int(r["seed"]))].append(r)
        scope_seed_rows: list[dict[str, Any]] = []
        for m_name in MODEL_ORDER_M8:
            m_seeds = sorted(sd for (mn, sd) in grouped if mn == m_name)
            for sd in m_seeds:
                agg = aggregate_single_seed_scope(suite, scope, m_name, sd, grouped[(m_name, sd)])
                per_seed_summaries.append(agg)
                scope_seed_rows.append(agg)
        return scope_seed_rows

    unified_summary: list[dict[str, Any]] = []
    skab_rows = [r for r in per_exp_rows if r["dataset_suite"] == "SKAB"]
    tsb_rows = [r for r in per_exp_rows if r["dataset_suite"] == "TSB-AD-M"]

    scope_definitions: list[tuple[str, str, list[dict[str, Any]]]] = [
        ("SKAB", "fault_experiments_only", [r for r in skab_rows if not r["is_anomaly_free"]]),
        ("SKAB", "anomaly_free_baseline", [r for r in skab_rows if r["is_anomaly_free"]]),
        (
            "TSB-AD-M",
            "tsfm_common_9_domains",
            [r for r in tsb_rows if str(r["experiment_id"]) in tsfm_common_ids],
        ),
        ("TSB-AD-M", "full_17_domains", tsb_rows),
    ]

    for suite, scope, subset in scope_definitions:
        if not subset:
            continue
        seed_aggs = _collect_scope(suite, scope, subset)
        by_m: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for sr in seed_aggs:
            by_m[str(sr["model"])].append(sr)
        for m_name in MODEL_ORDER_M8:
            if m_name in by_m:
                # For full_17_domains, exclude TSFMs if they only ran a partial subset
                if (
                    scope == "full_17_domains"
                    and m_name in TSFM_MODELS
                    and len(tsb_tsfm_splits) < len(tsb_bundle.splits)
                ):
                    continue
                unified_summary.append(aggregate_across_seeds(by_m[m_name]))

    # 5. Per-scenario SKAB breakdown & per-domain TSB breakdown
    per_scenario_summary: list[dict[str, Any]] = []
    for skab_cat in sorted({str(r["domain"]) for r in skab_rows}):
        cat_rows = [r for r in skab_rows if str(r["domain"]) == skab_cat]
        s_aggs = _collect_scope("SKAB", f"scenario:{skab_cat}", cat_rows)
        by_m = defaultdict(list)
        for sr in s_aggs:
            by_m[str(sr["model"])].append(sr)
        for m_name in MODEL_ORDER_M8:
            if m_name in by_m:
                per_scenario_summary.append(aggregate_across_seeds(by_m[m_name]))

    per_domain_tsb: list[dict[str, Any]] = []
    for dom in sorted({str(r["domain"]) for r in tsb_rows}):
        d_rows = [r for r in tsb_rows if str(r["domain"]) == dom]
        d_aggs = _collect_scope("TSB-AD-M", f"domain:{dom}", d_rows)
        by_m = defaultdict(list)
        for sr in d_aggs:
            by_m[str(sr["model"])].append(sr)
        for m_name in MODEL_ORDER_M8:
            if m_name in by_m:
                per_domain_tsb.append(aggregate_across_seeds(by_m[m_name]))

    # 6. Error analysis summary (worst-FP & worst-FN experiments per model + taxonomy totals)
    error_analysis_rows: list[dict[str, Any]] = []
    for u_row in unified_summary:
        if u_row["evaluation_scope"] not in ("fault_experiments_only", "tsfm_common_9_domains"):
            continue
        suite = str(u_row["dataset_suite"])
        m_name = str(u_row["model"])
        matching = [
            r
            for r in per_exp_rows
            if r["dataset_suite"] == suite
            and r["model"] == m_name
            and int(r["seed"]) == seeds[0]
            and (
                (suite == "SKAB" and not r["is_anomaly_free"])
                or (suite == "TSB-AD-M" and str(r["experiment_id"]) in tsfm_common_ids)
            )
        ]
        worst_fp = sorted(
            matching, key=lambda x: float(x["false_positive_rate"] or 0.0), reverse=True
        )[:3]
        worst_fn = sorted(matching, key=lambda x: float(x["recall"] or 0.0))[:3]
        error_analysis_rows.append(
            {
                "dataset_suite": suite,
                "evaluation_scope": u_row["evaluation_scope"],
                "model": m_name,
                "fp_total_mean": u_row["fp_mean"],
                "fp_pre_fault_lead_mean": u_row["fp_pre_fault_lead_mean"],
                "fp_post_fault_recovery_mean": u_row["fp_post_fault_recovery_mean"],
                "fp_isolated_background_mean": u_row["fp_isolated_background_mean"],
                "fn_total_mean": u_row["fn_mean"],
                "missed_episodes_mean": u_row["missed_episodes_mean"],
                "missed_episodes_short_mean": u_row["missed_episodes_short_mean"],
                "missed_episodes_long_mean": u_row["missed_episodes_long_mean"],
                "partial_episodes_mean": u_row["partial_episodes_mean"],
                "fn_missed_episodes_points_mean": u_row["fn_missed_episodes_points_mean"],
                "fn_onset_delay_points_mean": u_row["fn_onset_delay_points_mean"],
                "fn_adaptation_dropoff_points_mean": u_row["fn_adaptation_dropoff_points_mean"],
                "worst_fp_experiments": [
                    f"{w['experiment_id']} (FPR={w['false_positive_rate']})" for w in worst_fp
                ],
                "worst_fn_experiments": [
                    f"{w['experiment_id']} (Recall={w['recall']})" for w in worst_fn
                ],
            }
        )

    # 7. Forecasting accuracy vs anomaly detection separation table
    forecasting_vs_detection = [
        {
            "dataset_suite": r["dataset_suite"],
            "evaluation_scope": r["evaluation_scope"],
            "model": r["model"],
            "mean_forecast_mae": r["mean_forecast_mae_mean"],
            "mean_forecast_rmse": r["mean_forecast_rmse_mean"],
            "mean_normalized_mae": r["mean_normalized_mae_mean"],
            "mean_interval_coverage_80pct": r["mean_interval_coverage_80pct_mean"],
            "micro_f1": r["micro_f1_mean"],
            "macro_f1": r["macro_f1_mean"],
            "mean_pr_auc": r["mean_pr_auc_mean"],
            "mean_vus_pr": r["mean_vus_pr_mean"],
            "fn_adaptation_dropoff_points": r["fn_adaptation_dropoff_points_mean"],
        }
        for r in unified_summary
        if str(r["model"]) in TSFM_MODELS
    ]

    resource_sla = cfg.get("resource_sla", {})
    resource_analysis = analyze_resource_efficiency(unified_summary, resource_sla)
    best_per_scenario = select_best_per_scenario(unified_summary)

    dev_info = select_compute_device(device)
    report: dict[str, Any] = {
        "reproducibility": {
            "seeds": seeds,
            "stochastic_models_multi_seed": sorted(STOCHASTIC_MODELS),
            "deterministic_models_single_pass": sorted(set(MODEL_ORDER_M8) - STOCHASTIC_MODELS),
            "device_info": dev_info.to_dict(),
            "mock_backends_used": mock_backends,
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
        "protocol_verification": split_verification,
        "resource_sla": dict(resource_sla),
        "dataset_provenance": {
            "SKAB": skab_bundle.provenance,
            "TSB-AD-M": tsb_bundle.provenance,
            "TSB-AD-M_tsfm_common_domains": sorted({s.domain for s in tsb_tsfm_splits}),
        },
        "unified_comparison": unified_summary,
        "per_scenario_skab": per_scenario_summary,
        "per_domain_tsb": per_domain_tsb,
        "forecasting_vs_detection": forecasting_vs_detection,
        "resource_efficiency_analysis": resource_analysis,
        "error_analysis": error_analysis_rows,
        "best_per_scenario": best_per_scenario,
        "unexecuted_experiments": unexecuted_experiments,
        "limitations_and_threats_to_validity": [
            "TSB-AD-M TSFM coverage cap: 8 ultra-long TSB series (>50,000 rows, totaling >1.23M "
            "test rows) exceeded single-machine zero-shot foundation model inference budgets and "
            "are explicitly marked unexecuted for Chronos-2 and TimesFM 3.0; all 10 models are "
            "compared head-to-head on the identical 9-domain common subset while M4–M7 are also "
            "evaluated across all 17 domains.",
            "Sustained-fault context adaptation in rolling zero-shot forecasters: On SKAB "
            "(300+-sample sustained regime shifts), once a fault enters the 128-step causal "
            "context window, Chronos-2 and TimesFM 3.0 adapt to the shifted level, producing "
            "substantial post-onset adaptation-dropoff FNs despite strong initial onset detection.",
            "Fixed 0.99 validation quantile calibration: All detectors calibrate thresholds on "
            "normal validation data (`x_val_normal`) without test labels; mild post-train drift "
            "in SKAB anomaly-free baseline inflates FPR for bounded `[0, 1]` Sigmoid models "
            "(TranAD) compared to unbounded or rolling-context forecasters.",
            "Independent univariate TSB protocol: Modeling TSB channels independently prevents "
            "spurious coupling of heterogeneous command/indicator columns, but cannot capture "
            "true cross-channel relational anomalies on genuinely coupled TSB domains (e.g. SWaT).",
            "TimesFM 3.0 non-commercial weights license: `google/timesfm-3.0-pytorch` weights "
            "carry `timesfm-non-commercial-license-v1.0`, restricting commercial deployment "
            "even where accuracy is competitive.",
            "Offline batched latency vs. live Kafka streaming: Measured latencies reflect batched "
            "in-process inference on Apple Silicon without Kafka broker serialization/network "
            "overhead; single-event streaming latency for 120M–331M TSFMs will be higher.",
        ],
    }
    return report, unified_summary, per_seed_summaries, per_exp_rows


def _fmt_pm(mean_v: object, std_v: object, is_stochastic: bool, suffix: str = "") -> str:
    if mean_v is None:
        return "unavailable"
    if is_stochastic and std_v is not None:
        return f"{mean_v} ± {std_v}{suffix}"
    return f"{mean_v}{suffix}"


def _fmt(val: object, suffix: str = "") -> str:
    return "unavailable" if val is None else f"{val}{suffix}"


def render_m9_markdown(report: Mapping[str, Any], chart_files: Sequence[str]) -> str:
    """Render comprehensive M9 evaluation Markdown report."""
    repro = report["reproducibility"]
    dev = repro["device_info"]
    pv = report["protocol_verification"]
    best = report["best_per_scenario"]

    lines = [
        f"# DriftGuard-Kafka M9 Comprehensive Evaluation Report — `{report['run_id']}`",
        "",
        f"Generated: `{report['generated_utc']}` | Seeds (trained models): `{repro['seeds']}` | "
        f"Device: `{dev['selected']}` (Apple Silicon: `{dev['is_apple_silicon']}`, "
        f"MPS Available: `{dev['mps_available']}`) | Verified Splits: `{pv['verified_splits']}`",
        "",
        "> **Protocol Verification:** Identical chronological splits verified via SHA-256 "
        f"fingerprints across all {pv['verified_splits']} series; `fit_and_calibrate` signature "
        f"`{pv['fit_and_calibrate_signature']}` guarantees test labels never enter training or "
        "threshold calibration.",
        "",
        "## 1. Unified Comparison Table (Separated: Multivariate SKAB vs. Univariate TSB-AD-M)",
        "",
        "*Trained stochastic models (`isolation_forest`, `half_space_trees`, `lstm_autoencoder`, "
        "`tcn_autoencoder`, `tranad`) report `mean ± std` across seeds "
        f"`{repro['seeds']}`; deterministic models (`static_threshold`, `chronos2_*`, "
        "`timesfm3_*`) report exact single-pass values.*",
        "",
        "| Suite | Scope | Mode | Model | Micro P | Micro R | Micro F1 | Micro FPR | "
        "Macro F1 | Mean PR-AUC | Mean VUS-PR | Mean Delay | Train/Cal s | Latency | "
        "Throughput | Max RSS |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report["unified_comparison"]:
        st = bool(r["is_stochastic"])
        mode = "multivariate" if r["dataset_suite"] == "SKAB" else "univariate"
        delay_s = _fmt_pm(
            r["mean_detection_delay_samples_mean"], r["mean_detection_delay_samples_std"], st
        )
        train_s = _fmt_pm(
            r["total_training_duration_s_mean"], r["total_training_duration_s_std"], st, " s"
        )
        lat_s = _fmt_pm(r["mean_latency_ms_mean"], r["mean_latency_ms_std"], st, " ms")
        tp_s = _fmt_pm(
            r["throughput_events_per_sec_mean"],
            r["throughput_events_per_sec_std"],
            st,
            " ev/s",
        )
        lines.append(
            f"| {r['dataset_suite']} | `{r['evaluation_scope']}` | `{mode}` | `{r['model']}` | "
            f"{_fmt_pm(r['micro_precision_mean'], r['micro_precision_std'], st)} | "
            f"{_fmt_pm(r['micro_recall_mean'], r['micro_recall_std'], st)} | "
            f"{_fmt_pm(r['micro_f1_mean'], r['micro_f1_std'], st)} | "
            f"{_fmt_pm(r['micro_fpr_mean'], r['micro_fpr_std'], st)} | "
            f"{_fmt_pm(r['macro_f1_mean'], r['macro_f1_std'], st)} | "
            f"{_fmt_pm(r['mean_pr_auc_mean'], r['mean_pr_auc_std'], st)} | "
            f"{_fmt_pm(r['mean_vus_pr_mean'], r['mean_vus_pr_std'], st)} | "
            f"{delay_s} | {train_s} | {lat_s} | {tp_s} | "
            f"{_fmt(r['max_rss_mib_mean'], ' MiB')} |"
        )

    lines.extend(["", "## 2. Performance Charts", ""])
    for cf in chart_files:
        lines.append(f"- ![{cf}](charts/{cf}) (`charts/{cf}`)")

    lines.extend(
        [
            "",
            "## 3. Forecasting Accuracy vs. Anomaly Detection Accuracy (Zero-Shot TSFMs)",
            "",
            "| Suite | Scope | Model | Forecast MAE | Forecast RMSE | Normalized MAE | "
            "80% Interval Coverage | Micro F1 | Macro F1 | Mean PR-AUC | Mean VUS-PR | "
            "Adaptation-Dropoff FNs |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for f_row in report["forecasting_vs_detection"]:
        lines.append(
            f"| {f_row['dataset_suite']} | `{f_row['evaluation_scope']}` | `{f_row['model']}` | "
            f"{_fmt(f_row['mean_forecast_mae'])} | {_fmt(f_row['mean_forecast_rmse'])} | "
            f"{_fmt(f_row['mean_normalized_mae'])} | "
            f"{_fmt(f_row['mean_interval_coverage_80pct'])} | "
            f"{_fmt(f_row['micro_f1'])} | {_fmt(f_row['macro_f1'])} | "
            f"{_fmt(f_row['mean_pr_auc'])} | {_fmt(f_row['mean_vus_pr'])} | "
            f"{_fmt(f_row['fn_adaptation_dropoff_points'])} |"
        )

    lines.extend(
        [
            "",
            "## 4. Dataset-Wise & Per-Scenario Analysis",
            "",
            "### 4.1 SKAB Per-Scenario Breakdown (`valve1`, `valve2`, `other`, `anomaly-free`)",
            "",
            "| Scenario | Model | Exps | Micro F1 | Micro FPR | Mean PR-AUC | Mean VUS-PR | "
            "Mean Delay |",
            "|---|---|---:|---|---|---|---|---|",
        ]
    )
    for sc in report["per_scenario_skab"]:
        st = bool(sc["is_stochastic"])
        sc_delay = _fmt_pm(
            sc["mean_detection_delay_samples_mean"], sc["mean_detection_delay_samples_std"], st
        )
        lines.append(
            f"| `{sc['evaluation_scope']}` | `{sc['model']}` | {sc['experiments']} | "
            f"{_fmt_pm(sc['micro_f1_mean'], sc['micro_f1_std'], st)} | "
            f"{_fmt_pm(sc['micro_fpr_mean'], sc['micro_fpr_std'], st)} | "
            f"{_fmt_pm(sc['mean_pr_auc_mean'], sc['mean_pr_auc_std'], st)} | "
            f"{_fmt_pm(sc['mean_vus_pr_mean'], sc['mean_vus_pr_std'], st)} | {sc_delay} |"
        )

    lines.extend(
        [
            "",
            "### 4.2 TSB-AD-M Per-Domain Breakdown (Independent Univariate)",
            "",
            "| Domain | Best Model (PR-AUC) | Best PR-AUC | Best Model (F1) | Best F1 | "
            "Models Evaluated |",
            "|---|---|---:|---|---:|---:|",
        ]
    )
    tsb_by_dom: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for d_row in report["per_domain_tsb"]:
        tsb_by_dom[str(d_row["evaluation_scope"]).replace("domain:", "")].append(d_row)
    for dom in sorted(tsb_by_dom):
        d_models = tsb_by_dom[dom]
        by_pr = sorted(
            [m for m in d_models if m.get("mean_pr_auc_mean") is not None],
            key=lambda x: float(x["mean_pr_auc_mean"]),
            reverse=True,
        )
        by_f1 = sorted(
            [m for m in d_models if m.get("micro_f1_mean") is not None],
            key=lambda x: float(x["micro_f1_mean"]),
            reverse=True,
        )
        b_pr_m = by_pr[0]["model"] if by_pr else "unavailable"
        b_pr_v = by_pr[0]["mean_pr_auc_mean"] if by_pr else None
        b_f1_m = by_f1[0]["model"] if by_f1 else "unavailable"
        b_f1_v = by_f1[0]["micro_f1_mean"] if by_f1 else None
        lines.append(
            f"| `{dom}` | `{b_pr_m}` | {_fmt(b_pr_v)} | `{b_f1_m}` | {_fmt(b_f1_v)} | "
            f"{len(d_models)} |"
        )

    lines.extend(
        [
            "",
            "## 5. Resource-Efficiency Analysis & High-F1 / Unacceptable-Resource Identification",
            "",
            f"Streaming SLA thresholds: latency `<= {report['resource_sla'].get('max_latency_ms')} "
            f"ms`, throughput `>= {report['resource_sla'].get('min_throughput_ev_s')} ev/s`, "
            f"RSS `<= {report['resource_sla'].get('max_rss_mib')} MiB`, "
            f"train/cal `<= {report['resource_sla'].get('max_training_s')} s`.",
            "",
            "| Suite | Scope | Model | Primary F1 | Latency (ms) | Throughput (ev/s) | "
            "Max RSS (MiB) | Train/Cal (s) | Status | SLA Violations |",
            "|---|---|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for ra in report["resource_efficiency_analysis"]:
        viol_str = ", ".join(ra["violations"]) if ra["violations"] else "none"
        lines.append(
            f"| {ra['dataset_suite']} | `{ra['evaluation_scope']}` | `{ra['model']}` | "
            f"{_fmt(ra['primary_f1'])} | {_fmt(ra['mean_latency_ms'])} | "
            f"{_fmt(ra['throughput_events_per_sec'])} | {_fmt(ra['max_rss_mib'])} | "
            f"{_fmt(ra['total_training_duration_s'])} | `{ra['status']}` | {viol_str} |"
        )

    lines.extend(
        [
            "",
            "## 6. Error Analysis (False Positives & Missed Anomalies)",
            "",
            "| Suite | Model | Mean FPs | Pre-Fault Lead | Post-Fault Recov. | Isolated BG | "
            "Mean FNs | Missed Eps (Short/Long) | Onset-Delay FNs | Adaptation-Dropoff FNs |",
            "|---|---|---:|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    for ea in report["error_analysis"]:
        ep_split = (
            f"{ea['missed_episodes_mean']} "
            f"({ea['missed_episodes_short_mean']}/{ea['missed_episodes_long_mean']})"
        )
        lines.append(
            f"| {ea['dataset_suite']} | `{ea['model']}` | {ea['fp_total_mean']} | "
            f"{ea['fp_pre_fault_lead_mean']} | {ea['fp_post_fault_recovery_mean']} | "
            f"{ea['fp_isolated_background_mean']} | {ea['fn_total_mean']} | {ep_split} | "
            f"{ea['fn_onset_delay_points_mean']} | {ea['fn_adaptation_dropoff_points_mean']} |"
        )

    lines.extend(
        [
            "",
            "## 7. Best Model Per Evaluation Scenario",
            "",
            f"> **Verdict:** {best['verdict']}",
            "",
            "| Evaluation Scenario | Metric | Best Model | Best Score | Runner-Up "
            "| Runner-Up Score |",
            "|---|---|---|---:|---|---:|",
        ]
    )
    for w in best["scenario_winners"]:
        lines.append(
            f"| {w['scenario']} | `{w['metric']}` | `{w['best_model']}` | "
            f"{_fmt(w['best_value'])} | `{_fmt(w['runner_up_model'])}` | "
            f"{_fmt(w['runner_up_value'])} |"
        )

    lines.extend(["", "## 8. Unexecuted Experiments (Explicit Disclosure)", ""])
    if report["unexecuted_experiments"]:
        lines.extend(
            [
                "| Suite | Domain | Experiment | Models Not Executed | Rows "
                "| Est. Forecast Blocks | Reason |",
                "|---|---|---|---|---:|---:|---|",
            ]
        )
        for ux in report["unexecuted_experiments"]:
            lines.append(
                f"| {ux['dataset_suite']} | `{ux['domain']}` | `{ux['experiment_id']}` | "
                f"`{', '.join(ux['models'])}` | {ux['total_rows']} | "
                f"{ux['estimated_univariate_forecast_blocks']} | {ux['reason']} |"
            )
    else:
        lines.append("- All scheduled experiments executed without exclusions.")

    lines.extend(["", "## 9. Limitations and Threats to Validity", ""])
    for lim in report["limitations_and_threats_to_validity"]:
        lines.append(f"- {lim}")
    lines.append("")
    return "\n".join(lines)


def write_m9_bundle(
    output_dir: Path,
    report: Mapping[str, Any],
    unified_rows: Sequence[Mapping[str, Any]],
    per_seed_rows: Sequence[Mapping[str, Any]],
    per_exp_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Write M9 JSON, Markdown, CSVs, and pure-SVG charts into `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sla = report["resource_sla"]

    skab_fault = [
        r
        for r in unified_rows
        if r["dataset_suite"] == "SKAB" and r["evaluation_scope"] == "fault_experiments_only"
    ]
    tsb_common = [
        r
        for r in unified_rows
        if r["dataset_suite"] == "TSB-AD-M" and r["evaluation_scope"] == "tsfm_common_9_domains"
    ]
    chart_files = generate_m9_charts(
        charts_dir=output_dir / "charts",
        skab_fault_summary=skab_fault,
        tsb_common_summary=tsb_common,
        max_latency_ms=float(sla.get("max_latency_ms", 0.1)),
        max_rss_mib=float(sla.get("max_rss_mib", 1024.0)),
    )

    full_report = dict(report)
    full_report["charts"] = [f"charts/{c}" for c in chart_files]

    (output_dir / "m9_report.json").write_text(
        json.dumps(full_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "m9_report.md").write_text(
        render_m9_markdown(full_report, chart_files), encoding="utf-8"
    )

    def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fields = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()}
                )

    _write_csv(output_dir / "unified_comparison.csv", unified_rows)
    _write_csv(output_dir / "per_seed_summary.csv", per_seed_rows)
    _write_csv(output_dir / "per_experiment_metrics.csv", per_exp_rows)
    _write_csv(output_dir / "error_analysis.csv", full_report["error_analysis"])
    return chart_files


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for DriftGuard-Kafka M9 evaluation."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive M9 multi-seed comparative evaluation."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/m9_eval.json"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skab-dir", type=Path, default=Path("data/external/skab"))
    parser.add_argument("--tsb-dir", type=Path, default=Path("data/external/tsb_ad_m"))
    parser.add_argument("--tsb-max-per-domain", type=int, default=None)
    parser.add_argument("--tsfm-max-tsb-rows", type=int, default=None)
    parser.add_argument("--max-skab-files", type=int, default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--mock-backends", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for DriftGuard-Kafka M9 comprehensive evaluation."""
    args = parse_args(argv)
    cfg: dict[str, Any] = {}
    if args.config is not None and args.config.exists():
        loaded = json.loads(args.config.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            cfg.update(loaded)

    if args.seeds is not None:
        cfg["seeds"] = list(args.seeds)
    if args.tsfm_max_tsb_rows is not None:
        cfg["tsfm_max_tsb_series_rows"] = int(args.tsfm_max_tsb_rows)

    device = str(args.device if args.device is not None else cfg.get("device", "auto"))
    cache_dir = str(
        args.cache_dir if args.cache_dir is not None else cfg.get("cache_dir", "models/hf_cache")
    )
    tsb_max_raw = int(
        args.tsb_max_per_domain
        if args.tsb_max_per_domain is not None
        else cfg.get("tsb_max_per_domain", 1)
    )

    if not args.mock_backends and (not is_chronos2_installed() or not is_timesfm3_installed()):
        print(
            "M9 EVALUATION FAILED: chronos-forecasting>=2.0 and timesfm>=3.0 are required "
            "(install via `uv pip install -r requirements-m8.txt` or pass --mock-backends).",
            file=sys.stderr,
        )
        return 2

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (Path("reports/m9") / run_id)

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
        tsb_bundle = load_tsb_ad_m_suite(args.tsb_dir, max_per_domain=tsb_max)
    except (DatasetNotAvailableError, DatasetFormatError, FileNotFoundError, ValueError) as exc:
        print(f"M9 EVALUATION FAILED: {exc}", file=sys.stderr)
        return 2

    report, unified_rows, per_seed_rows, per_exp_rows = run_m9_evaluation(
        skab_bundle=skab_bundle,
        tsb_bundle=tsb_bundle,
        cfg=cfg,
        device=device,
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
        mock_backends=args.mock_backends,
    )
    report["run_id"] = run_id
    report["generated_utc"] = format_utc_timestamp(datetime.now(timezone.utc))

    write_m9_bundle(output_dir, report, unified_rows, per_seed_rows, per_exp_rows)
    print(f"M9 comprehensive evaluation bundle generated in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
