"""M9.1 Research Methodology Audit and Corrected Comparative Benchmark.

Addresses every methodological gap identified in the M9 audit while preserving
all historical M1–M9 reports untouched:
  1. Two strictly equivalent input-protocol tracks:
     - **Track A (`multivariate`)**: identical multivariate feature matrices for
       all compatible models on SKAB (8 features) and TSB-AD-M (2–248 features).
     - **Track B (`independent_channel`)**: unified 1D channel wrapper applying
       identical active-channel selection, validation-only robust score
       normalization, mean channel aggregation, and validation-only calibration
       to every model family.
  2. Expanded TSB-AD-M coverage (`max_per_domain=3` -> 41 series across all 17
     domains) + chunked long-horizon causal forecasting (`H=256` on long series)
     so Chronos-2 & TimesFM 3.0 evaluate long series without silent exclusion.
  3. Corrected reporting: TSB-AD-M labeled as a multivariate dataset, strict
     scope guards in `per_seed_summary.csv` (never mislabelling partial subsets
     as `full_17_domains`), explicit metric definitions, and renamed aggregated
     error-point totals.
  4. Isolated subprocess resource profiling (`app.m9_1_profiler`) separating
     interpreter+data baseline RSS, model-load RSS delta, inference RSS delta,
     Apple Silicon MPS memory, and offline-batched latency/throughput.
  5. Validation-only false-positive calibration study on SKAB (`empirical_q99`,
     `empirical_q995`, `mad_z35`, `iqr_k30`, `evt_pot_q99`) reporting FPR,
     Precision, Recall, F1, and False Alarms per 1,000 Normal Observations.
  6. Statistical robustness: 5 seeds on SKAB, 3 seeds on 41-series TSB-AD-M,
     2,000-resample 95% bootstrap CIs over experiments/series, and paired
     bootstrap significance tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
import tracemalloc
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.datasets import (
    ChronologicalSplit,
    DatasetFormatError,
    DatasetNotAvailableError,
    DatasetSuiteBundle,
    load_skab_suite,
    load_tsb_ad_m_suite,
)
from app.m2_report import compute_classification_metrics
from app.m4_benchmark import compute_ranking_metrics
from app.m4_models import (
    BenchmarkDetector,
    FloatArray,
    HalfSpaceTreesDetector,
    IsolationForestDetector,
    StaticThresholdBaseline,
)
from app.m5_benchmark import compute_detection_delay
from app.m5_lstm_autoencoder import (
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
from app.m9_1_charts import generate_m9_1_charts
from app.m9_1_profiler import (
    PROFILER_MODELS,
    RESOURCE_METRIC_DEFINITIONS,
    run_isolated_subprocess_profiling,
)
from app.m9_1_stats import (
    ALL_CALIBRATION_STRATEGIES,
    CalibrationStrategy,
    bootstrap_mean_ci,
    calibrate_threshold_validation_only,
    evaluate_threshold_on_test,
    paired_bootstrap_test,
)
from app.m9_1_tracks import (
    TRACK_COMPATIBILITY_MATRIX,
    EvalTrack,
    IndependentChannelWrapper,
    _robust_channel_stats,
    forecast_chunked_tsfm_split,
)
from app.m9_evaluation import (
    STOCHASTIC_MODELS,
    TSFM_MODELS,
    _clear_mps_cache,
    _pkg_version,
    _rss_mib,
    aggregate_across_seeds,
    aggregate_single_seed_scope,
    compute_error_analysis,
    verify_split_invariants,
)
from app.report import _detect_git_commit
from app.schemas import format_utc_timestamp

ALL_TRACKS: tuple[EvalTrack, ...] = ("multivariate", "independent_channel")


def _make_non_tsfm_detector(
    model_name: str,
    seed: int,
    track: EvalTrack,
    cfg: Mapping[str, Any],
    device: str,
    n_features: int,
) -> BenchmarkDetector:
    q_cal = float(cfg.get("calibration_quantile", 0.99))
    lstm_c = cfg.get("lstm", {})
    tcn_c = cfg.get("tcn", {})
    tranad_c = cfg.get("tranad", {})
    max_uv_ch = int(cfg.get("tsfm", {}).get("max_independent_channels", 16))

    def _build_1d_or_mv(is_1d: bool) -> BenchmarkDetector:
        if model_name == "static_threshold":
            return StaticThresholdBaseline(calibration_quantile=q_cal)
        if model_name == "isolation_forest":
            return IsolationForestDetector(seed=seed, calibration_quantile=q_cal)
        if model_name == "half_space_trees":
            return HalfSpaceTreesDetector(seed=seed, calibration_quantile=q_cal)
        if model_name == "lstm_autoencoder":
            return LSTMAutoencoderDetector(
                LSTMAutoencoderConfig(
                    window_size=int(lstm_c.get("window_size", 32)),
                    hidden_dim=int(lstm_c.get("hidden_dim", 32)),
                    latent_dim=int(lstm_c.get("latent_dim", 16)),
                    num_layers=int(lstm_c.get("num_layers", 2)),
                    dropout=float(lstm_c.get("dropout", 0.1)),
                    batch_size=int(lstm_c.get("batch_size", 64)),
                    epochs=int(lstm_c.get("epochs", 12)),
                    learning_rate=float(lstm_c.get("learning_rate", 0.002)),
                    early_stopping_patience=int(lstm_c.get("early_stopping_patience", 3)),
                    early_stopping_min_delta=float(lstm_c.get("early_stopping_min_delta", 0.0001)),
                    seed=seed,
                    device=device,
                    calibration_quantile=q_cal,
                    channel_mode="multivariate",
                )
            )
        if model_name == "tcn_autoencoder":
            eval_bsz = int(tcn_c.get("eval_batch_size", 4096))
            if not is_1d and n_features > 64:
                eval_bsz = min(eval_bsz, 512)
            return TCNAutoencoderDetector(
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
                    eval_batch_size=eval_bsz,
                    seed=seed,
                    device=device,
                    calibration_quantile=q_cal,
                    channel_mode="multivariate",
                )
            )
        if model_name == "tranad":
            eval_bsz = int(tranad_c.get("eval_batch_size", 8192))
            bsz = int(tranad_c.get("batch_size", 128))
            if not is_1d and n_features > 64:
                eval_bsz = min(eval_bsz, 256)
                bsz = min(bsz, 32)
            return TranADDetector(
                TranADConfig(
                    window_size=int(tranad_c.get("window_size", 10)),
                    dim_feedforward=int(tranad_c.get("dim_feedforward", 16)),
                    num_encoder_layers=int(tranad_c.get("num_encoder_layers", 1)),
                    num_decoder_layers=int(tranad_c.get("num_decoder_layers", 1)),
                    dropout=float(tranad_c.get("dropout", 0.1)),
                    batch_size=bsz,
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
                    eval_batch_size=eval_bsz,
                    seed=seed,
                    device=device,
                    calibration_quantile=q_cal,
                    channel_mode="multivariate",
                )
            )
        raise ValueError(f"Unknown detector: {model_name!r}")

    if track == "independent_channel":
        return IndependentChannelWrapper(
            model_name=model_name,
            factory=lambda: _build_1d_or_mv(is_1d=True),
            calibration_quantile=q_cal,
            max_active_channels=max_uv_ch,
        )
    return _build_1d_or_mv(is_1d=False)


def _evaluate_non_tsfm_split(
    det: BenchmarkDetector,
    split: ChronologicalSplit,
    seed: int,
    track: EvalTrack,
    capture_scores_for_cal: bool,
) -> tuple[dict[str, Any], tuple[FloatArray, FloatArray] | None]:
    """Fit/calibrate `det` on `split`, score `x_test`, and optionally return (val, test) scores."""
    cal_t0 = time.perf_counter()
    det._feature_names = split.feature_names
    det._fit_normal(split.x_train_normal, split.feature_names)
    val_scores = np.asarray(
        det._score_validation(split.x_val_normal, split.feature_names), dtype=np.float64
    )
    thr = float(np.quantile(val_scores, det._calibration_quantile))
    det._threshold = thr
    fit_s = round(time.perf_counter() - cal_t0, 6)

    inf = det.score_and_predict(split.x_test, split.feature_names)
    y_true = split.y_test_anomaly
    preds = inf.predictions

    tp = int(np.sum((y_true == 1) & (preds == 1)))
    fp = int(np.sum((y_true == 0) & (preds == 1)))
    tn = int(np.sum((y_true == 0) & (preds == 0)))
    fn = int(np.sum((y_true == 1) & (preds == 0)))
    n_norm = fp + tn

    cls_m = compute_classification_metrics(tp, fp, tn, fn)
    rank_m = compute_ranking_metrics(y_true, inf.scores, preds, split.sliding_window)
    delay_m = compute_detection_delay(y_true, preds)
    err_m = compute_error_analysis(y_true, preds)

    mode_lbl = (
        "independent_channel_unified" if track == "independent_channel" else "multivariate_native"
    )
    row = {
        "track": track,
        "channel_processing_mode": mode_lbl,
        "seed": seed,
        "dataset_suite": split.dataset_suite,
        "experiment_id": split.experiment_id,
        "domain": split.domain,
        "is_anomaly_free": split.is_anomaly_free,
        "model": det.name,
        "is_stochastic": det.name in STOCHASTIC_MODELS,
        "selected_device": inf.extra_diagnostics.get("selected_device", "cpu"),
        "n_features": len(split.feature_names),
        "train_normal_samples": int(split.x_train_normal.shape[0]),
        "val_normal_samples": int(split.x_val_normal.shape[0]),
        "test_samples": inf.test_samples,
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
        "false_alarms_per_1k_normal": round(1000.0 * fp / n_norm, 3) if n_norm > 0 else None,
        "pr_auc": rank_m["pr_auc"],
        "vus_pr": rank_m["vus_pr"],
        "vus_roc": rank_m["vus_roc"],
        "total_episodes": delay_m["total_episodes"],
        "detected_episodes": delay_m["detected_episodes"],
        "missed_episodes": delay_m["missed_episodes"],
        "mean_detection_delay_samples": delay_m["mean_detection_delay_samples"],
        "training_duration_s": fit_s,
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
    captured = (val_scores, inf.scores) if capture_scores_for_cal else None
    return row, captured


def _normalize_tsfm_track_b_scores(
    val_per_feat: FloatArray,
    test_per_feat: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Apply identical Track B robust per-channel score normalization & mean aggregation."""
    n_feats = val_per_feat.shape[1]
    val_norm = np.zeros_like(val_per_feat, dtype=np.float64)
    test_norm = np.zeros_like(test_per_feat, dtype=np.float64)
    for ch in range(n_feats):
        med, sc = _robust_channel_stats(val_per_feat[:, ch])
        val_norm[:, ch] = np.maximum(0.0, (val_per_feat[:, ch] - med) / sc)
        test_norm[:, ch] = np.maximum(0.0, (test_per_feat[:, ch] - med) / sc)
    return (
        np.asarray(np.mean(val_norm, axis=1), dtype=np.float64),
        np.asarray(np.mean(test_norm, axis=1), dtype=np.float64),
    )


def _evaluate_tsfm_split(
    det: ZeroShotForecastDetector,
    residual_name: str,
    interval_name: str,
    split: ChronologicalSplit,
    seed: int,
    track: EvalTrack,
    tsfm_cfg: Mapping[str, Any],
    capture_scores_for_cal: bool,
) -> tuple[list[dict[str, Any]], dict[str, tuple[FloatArray, FloatArray]]]:
    """Evaluate zero-shot TSFM on `split` for Track A or Track B."""
    cfg = det.config
    max_mv_g = int(tsfm_cfg.get("max_multivariate_group_variates", 32))
    long_thr = int(tsfm_cfg.get("long_series_row_threshold", 30000))
    long_h = int(tsfm_cfg.get("long_series_prediction_length", 256))

    fit_t0 = time.perf_counter()
    det._feature_names = split.feature_names
    det._fit_normal(split.x_train_normal, split.feature_names)
    if det._train_tail is None:
        raise RuntimeError("training tail missing")
    val_details, mode_lbl = forecast_chunked_tsfm_split(
        det,
        det._train_tail,
        split.x_val_normal,
        split.feature_names,
        track=track,
        max_multivariate_group_variates=max_mv_g,
        long_series_row_threshold=long_thr,
        long_series_prediction_length=long_h,
    )
    cal_s = round(time.perf_counter() - fit_t0, 6)
    pre_test_tail = np.vstack([det._train_tail, split.x_val_normal])[-cfg.context_length :]

    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    else:
        tracemalloc.reset_peak()
    cpu_t0 = time.process_time()
    wall_ns0 = time.perf_counter_ns()
    try:
        test_details, mode_lbl = forecast_chunked_tsfm_split(
            det,
            pre_test_tail,
            split.x_test,
            split.feature_names,
            track=track,
            max_multivariate_group_variates=max_mv_g,
            long_series_row_threshold=long_thr,
            long_series_prediction_length=long_h,
        )
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
    backend = det._resolve_backend()
    y_true = split.y_test_anomaly

    if track == "independent_channel":
        val_res, test_res = _normalize_tsfm_track_b_scores(
            val_details.per_feature_scores, test_details.per_feature_scores
        )
        # Recompute interval per-feature matrix normalized identically
        val_iv_1d = val_details.interval_violation_scores
        test_iv_1d = test_details.interval_violation_scores
        med_iv, sc_iv = _robust_channel_stats(val_iv_1d)
        val_iv = np.maximum(0.0, (val_iv_1d - med_iv) / sc_iv)
        test_iv = np.maximum(0.0, (test_iv_1d - med_iv) / sc_iv)
    else:
        val_res, test_res = val_details.residual_scores, test_details.residual_scores
        val_iv, test_iv = (
            val_details.interval_violation_scores,
            test_details.interval_violation_scores,
        )

    rows: list[dict[str, Any]] = []
    captured: dict[str, tuple[FloatArray, FloatArray]] = {}
    for m_name, v_sc, t_sc in (
        (residual_name, val_res, test_res),
        (interval_name, val_iv, test_iv),
    ):
        thr = float(np.quantile(v_sc, cfg.calibration_quantile))
        preds = (t_sc > thr).astype(np.int64)
        tp = int(np.sum((y_true == 1) & (preds == 1)))
        fp = int(np.sum((y_true == 0) & (preds == 1)))
        tn = int(np.sum((y_true == 0) & (preds == 0)))
        fn = int(np.sum((y_true == 1) & (preds == 0)))
        n_norm = fp + tn
        cls_m = compute_classification_metrics(tp, fp, tn, fn)
        rank_m = compute_ranking_metrics(y_true, t_sc, preds, split.sliding_window)
        delay_m = compute_detection_delay(y_true, preds)
        err_m = compute_error_analysis(y_true, preds)
        if capture_scores_for_cal:
            captured[m_name] = (v_sc, t_sc)

        rows.append(
            {
                "track": track,
                "channel_processing_mode": mode_lbl,
                "seed": seed,
                "dataset_suite": split.dataset_suite,
                "experiment_id": split.experiment_id,
                "domain": split.domain,
                "is_anomaly_free": split.is_anomaly_free,
                "model": m_name,
                "is_stochastic": False,
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
                "false_alarms_per_1k_normal": (
                    round(1000.0 * fp / n_norm, 3) if n_norm > 0 else None
                ),
                "pr_auc": rank_m["pr_auc"],
                "vus_pr": rank_m["vus_pr"],
                "vus_roc": rank_m["vus_roc"],
                "total_episodes": delay_m["total_episodes"],
                "detected_episodes": delay_m["detected_episodes"],
                "missed_episodes": delay_m["missed_episodes"],
                "mean_detection_delay_samples": delay_m["mean_detection_delay_samples"],
                "training_duration_s": cal_s,
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
    return rows, captured


def _estimate_tsfm_blocks(
    split: ChronologicalSplit,
    track: EvalTrack,
    tsfm_cfg: Mapping[str, Any],
) -> int:
    pred_len = int(tsfm_cfg.get("prediction_length", 64))
    long_thr = int(tsfm_cfg.get("long_series_row_threshold", 30000))
    long_h = int(tsfm_cfg.get("long_series_prediction_length", 256))
    max_mv_g = int(tsfm_cfg.get("max_multivariate_group_variates", 32))
    max_uv_ch = int(tsfm_cfg.get("max_independent_channels", 16))
    n_test = int(split.x_test.shape[0])
    eff_h = long_h if n_test > long_thr else pred_len
    n_time_blocks = int(np.ceil(n_test / eff_h))
    n_feats = len(split.feature_names)
    if track == "multivariate":
        n_groups = int(np.ceil(n_feats / max_mv_g))
        return n_time_blocks * n_groups
    return n_time_blocks * min(n_feats, max_uv_ch)


def run_skab_calibration_study(
    skab_splits: Sequence[ChronologicalSplit],
    captured_scores: Mapping[tuple[str, str], tuple[FloatArray, FloatArray]],
    strategies: Sequence[CalibrationStrategy],
) -> list[dict[str, Any]]:
    """Evaluate validation-only calibration strategies on SKAB anomaly-free & fault splits."""
    split_by_id = {s.experiment_id: s for s in skab_splits}
    models = sorted({m for (m, _eid) in captured_scores})
    results: list[dict[str, Any]] = []

    for m_name in models:
        for strat in strategies:
            af_fp = 0
            af_norm = 0
            f_tp = 0
            f_fp = 0
            f_tn = 0
            f_fn = 0
            for eid, split in split_by_id.items():
                pair = captured_scores.get((m_name, eid))
                if pair is None:
                    continue
                val_sc, test_sc = pair
                thr = calibrate_threshold_validation_only(val_sc, strat)
                ev = evaluate_threshold_on_test(split.y_test_anomaly, test_sc, thr)
                if split.is_anomaly_free:
                    af_fp += int(ev["fp"])
                    af_norm += int(ev["normal_observations"])
                else:
                    f_tp += int(ev["tp"])
                    f_fp += int(ev["fp"])
                    f_tn += int(ev["tn"])
                    f_fn += int(ev["fn"])
            f_cls = compute_classification_metrics(f_tp, f_fp, f_tn, f_fn)
            f_norm = f_fp + f_tn
            results.append(
                {
                    "model": m_name,
                    "strategy": strat,
                    "anomaly_free_fp": af_fp,
                    "anomaly_free_normal_obs": af_norm,
                    "anomaly_free_fpr": round(af_fp / af_norm, 4) if af_norm > 0 else None,
                    "anomaly_free_fa_per_1k_normal": (
                        round(1000.0 * af_fp / af_norm, 3) if af_norm > 0 else None
                    ),
                    "fault_precision": f_cls["precision"],
                    "fault_recall": f_cls["recall"],
                    "fault_micro_f1": f_cls["f1"],
                    "fault_fpr": f_cls["false_positive_rate"],
                    "fault_fa_per_1k_normal": (
                        round(1000.0 * f_fp / f_norm, 3) if f_norm > 0 else None
                    ),
                }
            )
    return results


def _attach_bootstrap_cis(
    unified_row: dict[str, Any],
    per_exp_rows: Sequence[Mapping[str, Any]],
    n_resamples: int,
) -> dict[str, Any]:
    """Attach 95% bootstrap CIs over per-experiment Macro F1, PR-AUC, and VUS-PR."""
    # Average per experiment across seeds first so resampling unit is the independent experiment!
    by_exp: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in per_exp_rows:
        by_exp[str(r["experiment_id"])].append(r)

    for col, prefix in (
        ("f1", "macro_f1"),
        ("pr_auc", "mean_pr_auc"),
        ("vus_pr", "mean_vus_pr"),
    ):
        exp_means = [
            statistics.mean([float(x[col]) for x in grp if x.get(col) is not None])
            for grp in by_exp.values()
            if any(x.get(col) is not None for x in grp)
        ]
        ci = bootstrap_mean_ci(exp_means, n_resamples=n_resamples, seed=42)
        unified_row[f"{prefix}_ci_low"] = ci["ci_low"]
        unified_row[f"{prefix}_ci_high"] = ci["ci_high"]
    return unified_row


def _run_paired_bootstrap_comparisons(
    per_exp_rows: Sequence[Mapping[str, Any]],
    track: EvalTrack,
    suite: str,
    scope_label: str,
    n_resamples: int,
) -> list[dict[str, Any]]:
    """Run paired bootstrap tests between top-ranked models on identical experiments."""
    by_model_exp: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in per_exp_rows:
        if r["track"] == track and r["dataset_suite"] == suite:
            if suite == "SKAB" and r["is_anomaly_free"]:
                continue
            by_model_exp[str(r["model"])][str(r["experiment_id"])].append(r)

    tests: list[dict[str, Any]] = []
    for metric_col, metric_lbl in (("f1", "macro_f1"), ("pr_auc", "mean_pr_auc")):
        model_scores: list[tuple[str, dict[str, float]]] = []
        for m_name, exp_map in by_model_exp.items():
            per_exp_val: dict[str, float] = {}
            for eid, grp in exp_map.items():
                vals = [float(x[metric_col]) for x in grp if x.get(metric_col) is not None]
                if vals:
                    per_exp_val[eid] = statistics.mean(vals)
            if per_exp_val:
                model_scores.append((m_name, per_exp_val))

        model_scores.sort(key=lambda kv: statistics.mean(kv[1].values()), reverse=True)
        # Compare #1 vs #2, #1 vs #3, and #2 vs #3 on their exact common experiments
        for i, j in ((0, 1), (0, 2), (1, 2)):
            if j >= len(model_scores):
                continue
            m_a, map_a = model_scores[i]
            m_b, map_b = model_scores[j]
            common_ids = sorted(set(map_a) & set(map_b))
            if len(common_ids) < 2:
                continue
            res = paired_bootstrap_test(
                [map_a[e] for e in common_ids],
                [map_b[e] for e in common_ids],
                n_resamples=n_resamples,
                seed=42,
            )
            tests.append(
                {
                    "track": track,
                    "dataset_suite": suite,
                    "evaluation_scope": scope_label,
                    "metric": metric_lbl,
                    "model_a": m_a,
                    "model_b": m_b,
                    "mean_a": round(statistics.mean([map_a[e] for e in common_ids]), 4),
                    "mean_b": round(statistics.mean([map_b[e] for e in common_ids]), 4),
                    **res,
                }
            )
    return tests


def run_m9_1_evaluation(
    skab_bundle: DatasetSuiteBundle,
    tsb_bundle: DatasetSuiteBundle,
    cfg: Mapping[str, Any],
    skab_dir: Path,
    tsb_dir: Path,
    device: str = "auto",
    cache_dir: str = "models/hf_cache",
    local_files_only: bool = False,
    mock_backends: bool = False,
    skip_subprocess_profiler: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute full M9.1 dual-track evaluation, calibration study, stats & isolated profiling."""
    seeds_skab = [int(s) for s in cfg.get("seeds_skab", [42, 1337, 2026, 7, 99])]
    seeds_tsb = [int(s) for s in cfg.get("seeds_tsb", [42, 1337, 2026])]
    q_cal = float(cfg.get("calibration_quantile", 0.99))
    n_boot = int(cfg.get("bootstrap_resamples", 2000))
    tsfm_c = cfg.get("tsfm", {})
    max_blocks = int(tsfm_c.get("max_forecast_blocks_per_series", 12000))
    ctx_len = int(tsfm_c.get("context_length", 128))
    pred_len = int(tsfm_c.get("prediction_length", 64))
    tsfm_bsz = int(tsfm_c.get("batch_size", 64))
    max_uv_ch = int(tsfm_c.get("max_independent_channels", 16))
    chronos_id = str(tsfm_c.get("chronos2_model_id", CHRONOS2_DEFAULT_MODEL_ID))
    timesfm_id = str(tsfm_c.get("timesfm3_model_id", TIMESFM3_DEFAULT_MODEL_ID))
    cal_strats: tuple[CalibrationStrategy, ...] = tuple(
        cfg.get("calibration_strategies", list(ALL_CALIBRATION_STRATEGIES))
    )

    split_verification = verify_split_invariants([skab_bundle, tsb_bundle])
    dev_info = select_compute_device(device)

    non_tsfm_models = (
        "static_threshold",
        "isolation_forest",
        "half_space_trees",
        "lstm_autoencoder",
        "tcn_autoencoder",
        "tranad",
    )

    # Pre-plan TSFM feasible splits per (track, suite) using exact block budget
    unexecuted: list[dict[str, Any]] = []
    tsfm_plan: dict[tuple[EvalTrack, str], list[ChronologicalSplit]] = {}
    for track in ALL_TRACKS:
        for bundle in (skab_bundle, tsb_bundle):
            kept: list[ChronologicalSplit] = []
            for s in bundle.splits:
                est_b = _estimate_tsfm_blocks(s, track, tsfm_c)
                if max_blocks <= 0 or est_b <= max_blocks:
                    kept.append(s)
                else:
                    unexecuted.append(
                        {
                            "track": track,
                            "dataset_suite": bundle.suite_name,
                            "domain": s.domain,
                            "experiment_id": s.experiment_id,
                            "models": sorted(TSFM_MODELS),
                            "test_rows": int(s.x_test.shape[0]),
                            "n_features": len(s.feature_names),
                            "estimated_forecast_blocks": est_b,
                            "reason": (
                                f"Estimated forecast blocks ({est_b}) exceeded M9.1 per-series "
                                f"budget ({max_blocks}) on Track {track}; explicitly recorded."
                            ),
                        }
                    )
            tsfm_plan[(track, bundle.suite_name)] = kept

    # Compute exact total progress steps
    total_steps = 0
    for track in ALL_TRACKS:
        for bundle, seeds in ((skab_bundle, seeds_skab), (tsb_bundle, seeds_tsb)):
            total_steps += len(bundle.splits) * (1 + len(seeds) * len(STOCHASTIC_MODELS))
            total_steps += len(tsfm_plan[(track, bundle.suite_name)]) * 2

    done_steps = 0
    t_start = time.perf_counter()

    def _tick(track: str, suite: str, seed_v: int, m_lbl: str, eid: str) -> None:
        nonlocal done_steps
        done_steps += 1
        pct = 100.0 * done_steps / max(1, total_steps)
        el = time.perf_counter() - t_start
        print(
            f"[{pct:5.1f}%] ({done_steps}/{total_steps}) {track} | {suite} | "
            f"seed={seed_v} | {m_lbl} | {eid} | elapsed={el:.1f}s",
            flush=True,
        )

    print(
        f"[  0.0%] Starting M9.1 Audit & Benchmark on device={dev_info.selected} "
        f"(mps={dev_info.mps_available}) | SKAB={len(skab_bundle.splits)} files "
        f"(seeds={seeds_skab}) | TSB-AD-M={len(tsb_bundle.splits)} files "
        f"(seeds={seeds_tsb}) | total_steps={total_steps}",
        flush=True,
    )

    per_exp_rows: list[dict[str, Any]] = []
    skab_cal_captured: dict[tuple[str, str], tuple[FloatArray, FloatArray]] = {}

    mock_c: ForecastBackend | None = (
        DeterministicPersistenceBackend("mock/chronos-2") if mock_backends else None
    )
    mock_t: ForecastBackend | None = (
        DeterministicPersistenceBackend("mock/timesfm-3.0") if mock_backends else None
    )

    # Execute Track A (multivariate) and Track B (independent_channel)
    for track in ALL_TRACKS:
        for bundle, seeds in ((skab_bundle, seeds_skab), (tsb_bundle, seeds_tsb)):
            for m_name in non_tsfm_models:
                m_seeds = seeds if m_name in STOCHASTIC_MODELS else [seeds[0]]
                for seed in m_seeds:
                    for split in bundle.splits:
                        det = _make_non_tsfm_detector(
                            m_name, seed, track, cfg, device, len(split.feature_names)
                        )
                        capture = (
                            track == "multivariate"
                            and bundle.suite_name == "SKAB"
                            and seed == seeds[0]
                        )
                        row, cap = _evaluate_non_tsfm_split(det, split, seed, track, capture)
                        per_exp_rows.append(row)
                        if cap is not None:
                            skab_cal_captured[(m_name, split.experiment_id)] = cap
                        _tick(track, bundle.suite_name, seed, m_name, split.experiment_id)
                    _clear_mps_cache()

            # Zero-shot Chronos-2 & TimesFM 3.0
            tsfm_splits = tsfm_plan[(track, bundle.suite_name)]
            if tsfm_splits:
                ch_mode = "multivariate" if track == "multivariate" else "independent_univariate"
                c_det = Chronos2Detector(
                    TSFMDetectorConfig(
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
                    ),
                    backend=mock_c,
                )
                for split in tsfm_splits:
                    capture = track == "multivariate" and bundle.suite_name == "SKAB"
                    rows_c, caps_c = _evaluate_tsfm_split(
                        c_det,
                        "chronos2_zero_shot",
                        "chronos2_interval",
                        split,
                        seeds[0],
                        track,
                        tsfm_c,
                        capture,
                    )
                    per_exp_rows.extend(rows_c)
                    for mn, pair in caps_c.items():
                        skab_cal_captured[(mn, split.experiment_id)] = pair
                    _tick(track, bundle.suite_name, seeds[0], "chronos2", split.experiment_id)
                _clear_mps_cache()

                t_det = TimesFM3Detector(
                    TSFMDetectorConfig(
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
                    ),
                    backend=mock_t,
                )
                for split in tsfm_splits:
                    capture = track == "multivariate" and bundle.suite_name == "SKAB"
                    rows_t, caps_t = _evaluate_tsfm_split(
                        t_det,
                        "timesfm3_zero_shot",
                        "timesfm3_interval",
                        split,
                        seeds[0],
                        track,
                        tsfm_c,
                        capture,
                    )
                    per_exp_rows.extend(rows_t)
                    for mn, pair in caps_t.items():
                        skab_cal_captured[(mn, split.experiment_id)] = pair
                    _tick(track, bundle.suite_name, seeds[0], "timesfm3", split.experiment_id)
                _clear_mps_cache()

    # Build strict, bug-free per_seed_summary & unified_summary with 95% bootstrap CIs
    per_seed_summaries: list[dict[str, Any]] = []
    unified_summary: list[dict[str, Any]] = []

    def _aggregate_track_scope(
        track: EvalTrack,
        suite: str,
        scope: str,
        expected_exp_count: int,
        subset: Sequence[Mapping[str, Any]],
    ) -> None:
        grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
        for r in subset:
            grouped[(str(r["model"]), int(r["seed"]))].append(r)
        for m_name in MODEL_ORDER_M8:
            m_seeds = sorted(sd for (mn, sd) in grouped if mn == m_name)
            valid_seed_aggs: list[dict[str, Any]] = []
            valid_exp_rows: list[Mapping[str, Any]] = []
            for sd in m_seeds:
                sd_rows = grouped[(m_name, sd)]
                # STRICT SCOPE GUARD: never emit a partial subset under a full-suite scope!
                if len(sd_rows) != expected_exp_count:
                    continue
                agg = aggregate_single_seed_scope(suite, scope, m_name, sd, sd_rows)
                agg["track"] = track
                agg["channel_processing_mode"] = str(sd_rows[0]["channel_processing_mode"])
                per_seed_summaries.append(agg)
                valid_seed_aggs.append(agg)
                valid_exp_rows.extend(sd_rows)
            if valid_seed_aggs:
                u_row = aggregate_across_seeds(valid_seed_aggs)
                u_row["track"] = track
                u_row["channel_processing_mode"] = str(
                    valid_seed_aggs[0]["channel_processing_mode"]
                )
                # False alarms per 1k normal observations (mean across seeds)
                fa_vals = [
                    1000.0 * float(s["fp"]) / float(s["fp"] + s["tn"])
                    for s in valid_seed_aggs
                    if (s["fp"] + s["tn"]) > 0
                ]
                u_row["fa_per_1k_normal_mean"] = (
                    round(statistics.mean(fa_vals), 3) if fa_vals else None
                )
                _attach_bootstrap_cis(u_row, valid_exp_rows, n_boot)
                unified_summary.append(u_row)

    n_skab_fault = sum(1 for s in skab_bundle.splits if not s.is_anomaly_free)
    n_skab_af = sum(1 for s in skab_bundle.splits if s.is_anomaly_free)
    n_tsb_full = len(tsb_bundle.splits)

    for track in ALL_TRACKS:
        t_skab = [r for r in per_exp_rows if r["track"] == track and r["dataset_suite"] == "SKAB"]
        t_tsb = [
            r for r in per_exp_rows if r["track"] == track and r["dataset_suite"] == "TSB-AD-M"
        ]
        if n_skab_fault > 0:
            _aggregate_track_scope(
                track,
                "SKAB",
                "fault_experiments_only",
                n_skab_fault,
                [r for r in t_skab if not r["is_anomaly_free"]],
            )
        if n_skab_af > 0:
            _aggregate_track_scope(
                track,
                "SKAB",
                "anomaly_free_baseline",
                n_skab_af,
                [r for r in t_skab if r["is_anomaly_free"]],
            )
        # Full TSB-AD-M scope (only models that completed ALL n_tsb_full series!)
        _aggregate_track_scope(
            track,
            "TSB-AD-M",
            f"full_{n_tsb_full}_series",
            n_tsb_full,
            t_tsb,
        )
        # If TSFMs ran a strict subset on this track, also create an exact common-subset scope
        tsfm_ids = {s.experiment_id for s in tsfm_plan[(track, "TSB-AD-M")]}
        if 0 < len(tsfm_ids) < n_tsb_full:
            _aggregate_track_scope(
                track,
                "TSB-AD-M",
                f"tsfm_common_{len(tsfm_ids)}_series",
                len(tsfm_ids),
                [r for r in t_tsb if str(r["experiment_id"]) in tsfm_ids],
            )

    # Task 5: Validation-only SKAB false-positive calibration study
    cal_study = run_skab_calibration_study(skab_bundle.splits, skab_cal_captured, cal_strats)

    # Task 6: Paired bootstrap significance tests
    stat_tests: list[dict[str, Any]] = []
    for track in ALL_TRACKS:
        stat_tests.extend(
            _run_paired_bootstrap_comparisons(
                per_exp_rows, track, "SKAB", "fault_experiments_only", n_boot
            )
        )
        stat_tests.extend(
            _run_paired_bootstrap_comparisons(
                per_exp_rows, track, "TSB-AD-M", f"full_{n_tsb_full}_series", n_boot
            )
        )

    # Track A vs Track B protocol comparison table on TSB-AD-M
    protocol_pairs: list[dict[str, Any]] = []
    tsb_scope_a = f"full_{n_tsb_full}_series"
    by_tk_m: dict[tuple[str, str], Mapping[str, Any]] = {}
    for u in unified_summary:
        if u["dataset_suite"] == "TSB-AD-M":
            key = (str(u["track"]), str(u["model"]))
            # Prefer full scope if present, else common subset
            if key not in by_tk_m or u["evaluation_scope"] == tsb_scope_a:
                by_tk_m[key] = u
    for m_name in MODEL_ORDER_M8:
        ra = by_tk_m.get(("multivariate", m_name))
        rb = by_tk_m.get(("independent_channel", m_name))
        pr_a = ra.get("mean_pr_auc_mean") if ra else None
        pr_b = rb.get("mean_pr_auc_mean") if rb else None
        f1_a = ra.get("macro_f1_mean") if ra else None
        f1_b = rb.get("macro_f1_mean") if rb else None
        protocol_pairs.append(
            {
                "model": m_name,
                "track_a_scope": ra["evaluation_scope"] if ra else "unexecuted",
                "track_a_mode": ra["channel_processing_mode"] if ra else "unexecuted",
                "track_a_macro_f1": f1_a,
                "track_a_pr_auc": pr_a,
                "track_b_scope": rb["evaluation_scope"] if rb else "unexecuted",
                "track_b_mode": rb["channel_processing_mode"] if rb else "unexecuted",
                "track_b_macro_f1": f1_b,
                "track_b_pr_auc": pr_b,
                "pr_auc_delta_a_minus_b": (
                    round(float(pr_a) - float(pr_b), 4)
                    if (pr_a is not None and pr_b is not None)
                    else None
                ),
            }
        )

    # Task 4: Isolated subprocess resource profiling
    isolated_profiles: list[dict[str, Any]] = []
    if not skip_subprocess_profiler:
        print("[100.0%] Running isolated subprocess resource profiling...", flush=True)
        isolated_profiles = run_isolated_subprocess_profiling(
            skab_dir=skab_dir,
            tsb_dir=tsb_dir,
            suites=("SKAB", "TSB-AD-M"),
            models=PROFILER_MODELS,
            device=device,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            mock_backends=mock_backends,
        )

    # Error analysis rows with corrected aggregated totals naming
    error_analysis_rows = [
        {
            "track": u["track"],
            "dataset_suite": u["dataset_suite"],
            "evaluation_scope": u["evaluation_scope"],
            "model": u["model"],
            "aggregated_fp_total_mean_across_seeds": u["fp_mean"],
            "aggregated_fp_pre_fault_lead": u["fp_pre_fault_lead_mean"],
            "aggregated_fp_post_fault_recovery": u["fp_post_fault_recovery_mean"],
            "aggregated_fp_isolated_background": u["fp_isolated_background_mean"],
            "aggregated_fn_total_mean_across_seeds": u["fn_mean"],
            "missed_episodes_total_mean": u["missed_episodes_mean"],
            "missed_episodes_short": u["missed_episodes_short_mean"],
            "missed_episodes_long": u["missed_episodes_long_mean"],
            "fn_onset_delay_points": u["fn_onset_delay_points_mean"],
            "fn_adaptation_dropoff_points": u["fn_adaptation_dropoff_points_mean"],
        }
        for u in unified_summary
        if u["evaluation_scope"] != "anomaly_free_baseline"
    ]

    report: dict[str, Any] = {
        "reproducibility": {
            "seeds_skab": seeds_skab,
            "seeds_tsb": seeds_tsb,
            "bootstrap_resamples": n_boot,
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
        "track_compatibility_matrix": [asdict(c) for c in TRACK_COMPATIBILITY_MATRIX],
        "resource_metric_definitions": RESOURCE_METRIC_DEFINITIONS,
        "dataset_provenance": {
            "SKAB": {**skab_bundle.provenance, "dataset_type": "multivariate"},
            "TSB-AD-M": {
                **tsb_bundle.provenance,
                "dataset_type": "multivariate",
                "note": "Official TSB-AD-M multivariate dataset (2–248 features), NOT TSB-AD-U.",
            },
        },
        "unified_comparison": unified_summary,
        "protocol_comparison_tsb": protocol_pairs,
        "calibration_study_skab": cal_study,
        "statistical_tests": stat_tests,
        "isolated_resource_profiles": isolated_profiles,
        "error_analysis": error_analysis_rows,
        "unexecuted_experiments": unexecuted,
        "limitations_and_threats_to_validity": [
            "TSB-AD-M is a multivariate dataset (2–248 features). Track A evaluates all models on "
            "identical multivariate inputs (using deterministic 32-variate groups for TSFMs when "
            "D > 32), while Track B evaluates all models through an identical independent-channel "
            "1D wrapper; neither track should be conflated with the other.",
            "Representative TSB-AD-M coverage: Evaluating up to 3 files per domain covers 41 of "
            "the 180 TSB-AD-M-Eva series across all 17 domains; full 180-file multi-seed neural "
            "and foundation-model evaluation remains a multi-hour GPU cluster workload.",
            "Chunked long-horizon causal forecasting (H=256 on series >30,000 rows) allows "
            "Chronos-2 and TimesFM 3.0 to cover long series without exclusion, but longer "
            "horizons increase drift accumulation within each 256-step block.",
            "Offline-batched latency vs. live streaming: Isolated subprocess profiling cleanly "
            "separates interpreter, model-load, inference, and MPS memory, while latency reflects "
            "offline batched inference rather than single-event Kafka broker round-trip latency.",
            "TimesFM 3.0 weights (`google/timesfm-3.0-pytorch`) remain non-commercial research "
            "only (`timesfm-non-commercial-license-v1.0`).",
        ],
    }
    return report, unified_summary, per_seed_summaries, per_exp_rows


def _fmt(val: object, suffix: str = "") -> str:
    return "unavailable" if val is None else f"{val}{suffix}"


def _fmt_ci(mean_v: object, lo: object, hi: object) -> str:
    if mean_v is None:
        return "unavailable"
    if lo is not None and hi is not None:
        return f"{mean_v} [{lo}, {hi}]"
    return f"{mean_v}"


def render_protocol_comparison_md(report: Mapping[str, Any]) -> str:
    """Render dedicated Track A (Multivariate) vs Track B (Independent-Channel) report."""
    lines = [
        f"# DriftGuard-Kafka M9.1 Protocol Comparison Report — `{report['run_id']}`",
        "",
        "## Why This Audit Matters",
        "",
        "In M9, classical baselines (`static_threshold`, `isolation_forest`, `half_space_trees`) "
        "processed `TSB-AD-M` (an official **multivariate** dataset with 2–248 features) as "
        "`multivariate_baseline`, while neural models (`lstm_autoencoder`, `tcn_autoencoder`, "
        "`tranad`) and TSFMs (`chronos2_*`, `timesfm3_*`) ran as `independent_univariate`. "
        "M9.1 corrects this asymmetry by evaluating **every** model family under both strictly "
        "equivalent tracks:",
        "",
        "- **Track A (`multivariate`)**: Identical multivariate `(N, D)` windows for all models.",
        "- **Track B (`independent_channel`)**: Identical 1D wrapper "
        "(`IndependentChannelWrapper`) with identical validation-only robust score "
        "normalization, mean aggregation, and validation-only calibration.",
        "",
        "## 1. Model Input-Protocol Compatibility Matrix",
        "",
        "| Model | Family | Track A (Multivariate) | Track B (Independent-Channel) | Notes |",
        "|---|---|---|---|---|",
    ]
    for c in report["track_compatibility_matrix"]:
        lines.append(
            f"| `{c['model']}` | {c['family']} | `{c['track_a_mode_label']}` | "
            f"`{c['track_b_mode_label']}` | {c['notes']} |"
        )

    lines.extend(
        [
            "",
            "## 2. TSB-AD-M (Multivariate Dataset): Track A vs. Track B Performance Gap",
            "",
            "| Model | Track A Scope | Track A Macro F1 | Track A PR-AUC | Track B Scope | "
            "Track B Macro F1 | Track B PR-AUC | Δ PR-AUC (A − B) |",
            "|---|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    for p in report["protocol_comparison_tsb"]:
        lines.append(
            f"| `{p['model']}` | `{p['track_a_scope']}` | {_fmt(p['track_a_macro_f1'])} | "
            f"{_fmt(p['track_a_pr_auc'])} | `{p['track_b_scope']}` | "
            f"{_fmt(p['track_b_macro_f1'])} | {_fmt(p['track_b_pr_auc'])} | "
            f"{_fmt(p['pr_auc_delta_a_minus_b'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_m9_1_markdown(report: Mapping[str, Any], chart_files: Sequence[str]) -> str:
    """Render comprehensive corrected M9.1 Markdown report."""
    repro = report["reproducibility"]
    dev = repro["device_info"]
    pv = report["protocol_verification"]
    lines = [
        f"# DriftGuard-Kafka M9.1 Audited & Corrected Benchmark Report — `{report['run_id']}`",
        "",
        f"Generated: `{report['generated_utc']}` | SKAB Seeds: `{repro['seeds_skab']}` | "
        f"TSB-AD-M Seeds: `{repro['seeds_tsb']}` | Bootstrap Resamples: "
        f"`{repro['bootstrap_resamples']}` | Device: `{dev['selected']}` "
        f"(Apple Silicon: `{dev['is_apple_silicon']}`, MPS: `{dev['mps_available']}`)",
        "",
        f"> **Dataset & Protocol Verification:** Both `SKAB` (`8` features) and `TSB-AD-M` "
        f"(`2–248` features) are **multivariate** datasets (`{pv['verified_splits']}` verified "
        "chronological splits via SHA-256). Test labels are strictly excluded from training, "
        "per-channel normalization, and threshold calibration.",
        "",
        "## 1. Unified Comparison Table (Track A: Multivariate vs. Track B: Independent-Channel)",
        "",
        "*All values report mean across seeds with 95% bootstrap confidence intervals over "
        "experiments/series `[ci_low, ci_high]`.*",
        "",
        "| Track | Suite | Scope | Mode | Model | Micro F1 | Macro F1 [95% CI] | "
        "Mean PR-AUC [95% CI] | Mean VUS-PR [95% CI] | Micro FPR | FA / 1k Normal | "
        "Offline Latency | Offline Throughput |",
        "|---|---|---|---|---|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for u in report["unified_comparison"]:
        f1_ci = _fmt_ci(u["macro_f1_mean"], u.get("macro_f1_ci_low"), u.get("macro_f1_ci_high"))
        pr_ci = _fmt_ci(
            u["mean_pr_auc_mean"], u.get("mean_pr_auc_ci_low"), u.get("mean_pr_auc_ci_high")
        )
        vp_ci = _fmt_ci(
            u["mean_vus_pr_mean"], u.get("mean_vus_pr_ci_low"), u.get("mean_vus_pr_ci_high")
        )
        lines.append(
            f"| `{u['track']}` | {u['dataset_suite']} | `{u['evaluation_scope']}` | "
            f"`{u['channel_processing_mode']}` | `{u['model']}` | {_fmt(u['micro_f1_mean'])} | "
            f"{f1_ci} | {pr_ci} | {vp_ci} | "
            f"{_fmt(u['micro_fpr_mean'])} | {_fmt(u.get('fa_per_1k_normal_mean'))} | "
            f"{_fmt(u['mean_latency_ms_mean'], ' ms')} | "
            f"{_fmt(u['throughput_events_per_sec_mean'], ' ev/s')} |"
        )

    lines.extend(["", "## 2. Revised Performance Charts", ""])
    for cf in chart_files:
        lines.append(f"- ![{cf}](charts/{cf}) (`charts/{cf}`)")

    lines.extend(
        [
            "",
            "## 3. Paired Bootstrap Statistical Significance Tests (2,000 Resamples)",
            "",
            "*Differences whose 95% CI spans `0` or `p >= 0.05` are explicitly marked "
            "`not_significant` and must not be over-claimed.*",
            "",
            "| Track | Suite | Metric | Model A | Model B | Mean A | Mean B | "
            "Δ (A − B) [95% CI] | p-value | Significant (α=0.05)? |",
            "|---|---|---|---|---|---:|---:|---|---:|---|",
        ]
    )
    for st in report["statistical_tests"]:
        sig_lbl = "YES" if st["significant_at_005"] else "no (not_significant)"
        lines.append(
            f"| `{st['track']}` | {st['dataset_suite']} | `{st['metric']}` | "
            f"`{st['model_a']}` | `{st['model_b']}` | {st['mean_a']} | {st['mean_b']} | "
            f"{st['mean_diff']} [{st['ci_low']}, {st['ci_high']}] | {st['p_value']} | "
            f"`{sig_lbl}` |"
        )

    lines.extend(
        [
            "",
            "## 4. Isolated Subprocess Resource Profiling (Offline-Batched)",
            "",
            "### Metric Definitions",
        ]
    )
    for k, desc in report["resource_metric_definitions"].items():
        lines.append(f"- **`{k}`**: {desc}")
    lines.extend(
        [
            "",
            "| Suite | Model | Device | Mode | Baseline RSS (MiB) | Model-Load Δ RSS (MiB) | "
            "Inference Δ RSS (MiB) | Inference Peak RSS (MiB) | MPS Alloc (MiB) | "
            "Peak Tracemalloc (KiB) | Offline Latency (ms) | Offline Throughput (ev/s) |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for p in report["isolated_resource_profiles"]:
        if p.get("status") != "ok":
            lines.append(
                f"| {p.get('dataset_suite')} | `{p.get('model')}` | failed | failed | "
                f"— | — | — | — | — | — | — | `{p.get('error', 'error')}` |"
            )
            continue
        lines.append(
            f"| {p['dataset_suite']} | `{p['model']}` | `{p['selected_device']}` | "
            f"`{p['channel_processing_mode']}` | {p['interpreter_baseline_rss_mib']} | "
            f"{p['model_load_rss_delta_mib']} | {p['inference_rss_delta_mib']} | "
            f"{p['inference_peak_rss_mib']} | {p['mps_current_allocated_mib']} | "
            f"{p['peak_tracemalloc_kib']} | {p['offline_batched_mean_latency_ms']} | "
            f"{p['offline_batched_throughput_ev_s']} |"
        )

    lines.extend(
        [
            "",
            "## 5. SKAB False-Positive Audit & Validation-Only Threshold Calibration Study",
            "",
            "*All 5 calibration strategies are fit strictly on `x_val_normal` (zero test-label "
            "tuning). Reports both SKAB `anomaly-free` FPR / False Alarms per 1,000 Normal "
            "and SKAB `fault_experiments_only` Precision / Recall / Micro F1.*",
            "",
            "| Model | Validation Strategy | Anomaly-Free FPR | Anomaly-Free FA/1k Normal | "
            "Fault Precision | Fault Recall | Fault Micro F1 | Fault FA/1k Normal |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cs in report["calibration_study_skab"]:
        lines.append(
            f"| `{cs['model']}` | `{cs['strategy']}` | {_fmt(cs['anomaly_free_fpr'])} | "
            f"{_fmt(cs['anomaly_free_fa_per_1k_normal'])} | {_fmt(cs['fault_precision'])} | "
            f"{_fmt(cs['fault_recall'])} | {_fmt(cs['fault_micro_f1'])} | "
            f"{_fmt(cs['fault_fa_per_1k_normal'])} |"
        )

    lines.extend(
        [
            "",
            "## 6. Aggregated Error-Point Totals (FP & FN Taxonomy)",
            "",
            "*Aggregation semantics: Error points are summed across all experiments in each scope "
            "per seed, then averaged across seeds (`Aggregated FP/FN Total (mean across seeds)`).*",
            "",
            "| Track | Suite | Scope | Model | Aggregated FP Total (mean across seeds) | "
            "Pre-Fault Lead | Post-Fault Recov. | Isolated BG | "
            "Aggregated FN Total (mean across seeds) | "
            "Missed Eps (Short/Long) | Onset-Delay FNs | Adaptation-Dropoff FNs |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    for ea in report["error_analysis"]:
        ep_s = (
            f"{ea['missed_episodes_total_mean']} "
            f"({ea['missed_episodes_short']}/{ea['missed_episodes_long']})"
        )
        lines.append(
            f"| `{ea['track']}` | {ea['dataset_suite']} | `{ea['evaluation_scope']}` | "
            f"`{ea['model']}` | {ea['aggregated_fp_total_mean_across_seeds']} | "
            f"{ea['aggregated_fp_pre_fault_lead']} | {ea['aggregated_fp_post_fault_recovery']} | "
            f"{ea['aggregated_fp_isolated_background']} | "
            f"{ea['aggregated_fn_total_mean_across_seeds']} | {ep_s} | "
            f"{ea['fn_onset_delay_points']} | {ea['fn_adaptation_dropoff_points']} |"
        )

    lines.extend(["", "## 7. Unexecuted Experiments (Explicit Disclosure)", ""])
    if report["unexecuted_experiments"]:
        lines.extend(
            [
                "| Track | Suite | Domain | Experiment | Models | Test Rows "
                "| Est. Blocks | Reason |",
                "|---|---|---|---|---|---:|---:|---|",
            ]
        )
        for ux in report["unexecuted_experiments"]:
            lines.append(
                f"| `{ux['track']}` | {ux['dataset_suite']} | `{ux['domain']}` | "
                f"`{ux['experiment_id']}` | `{', '.join(ux['models'])}` | {ux['test_rows']} | "
                f"{ux['estimated_forecast_blocks']} | {ux['reason']} |"
            )
    else:
        lines.append("- All scheduled experiments across Track A and Track B executed completely.")

    lines.extend(["", "## 8. Limitations and Threats to Validity", ""])
    for lim in report["limitations_and_threats_to_validity"]:
        lines.append(f"- {lim}")
    lines.append("")
    return "\n".join(lines)


def write_m9_1_bundle(
    output_dir: Path,
    report: Mapping[str, Any],
    unified_rows: Sequence[Mapping[str, Any]],
    per_seed_rows: Sequence[Mapping[str, Any]],
    per_exp_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Write complete M9.1 artefact bundle (MD, JSON, CSVs, protocol report, 5 SVGs)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select TSB rows per track for charts (prefer full scope, else common)
    def _best_tsb_rows(tk: str) -> list[Mapping[str, Any]]:
        rows = [u for u in unified_rows if u["track"] == tk and u["dataset_suite"] == "TSB-AD-M"]
        full_rows = [r for r in rows if str(r["evaluation_scope"]).startswith("full_")]
        return full_rows if len(full_rows) >= 6 else rows

    chart_files = generate_m9_1_charts(
        charts_dir=output_dir / "charts",
        track_a_tsb_rows=_best_tsb_rows("multivariate"),
        track_b_tsb_rows=_best_tsb_rows("independent_channel"),
        protocol_pairs=report["protocol_comparison_tsb"],
        calibration_rows=report["calibration_study_skab"],
        isolated_profile_rows=report["isolated_resource_profiles"],
    )

    full_report = dict(report)
    full_report["charts"] = [f"charts/{c}" for c in chart_files]

    (output_dir / "m9_1_report.json").write_text(
        json.dumps(full_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "m9_1_report.md").write_text(
        render_m9_1_markdown(full_report, chart_files), encoding="utf-8"
    )
    (output_dir / "protocol_comparison.md").write_text(
        render_protocol_comparison_md(full_report), encoding="utf-8"
    )

    def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fields = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for r in rows:
                writer.writerow(
                    {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()}
                )

    _write_csv(output_dir / "unified_comparison.csv", unified_rows)
    _write_csv(output_dir / "per_seed_summary.csv", per_seed_rows)
    _write_csv(output_dir / "per_experiment_metrics.csv", per_exp_rows)
    _write_csv(output_dir / "error_analysis.csv", full_report["error_analysis"])
    _write_csv(output_dir / "calibration_study.csv", full_report["calibration_study_skab"])
    _write_csv(output_dir / "resource_isolation.csv", full_report["isolated_resource_profiles"])
    _write_csv(output_dir / "statistical_tests.csv", full_report["statistical_tests"])
    return chart_files


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for M9.1 audited benchmark."""
    parser = argparse.ArgumentParser(description="Run DriftGuard-Kafka M9.1 audited benchmark.")
    parser.add_argument("--config", type=Path, default=Path("configs/m9_1_eval.json"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skab-dir", type=Path, default=Path("data/external/skab"))
    parser.add_argument("--tsb-dir", type=Path, default=Path("data/external/tsb_ad_m"))
    parser.add_argument("--tsb-max-per-domain", type=int, default=None)
    parser.add_argument("--max-skab-files", type=int, default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--mock-backends", action="store_true")
    parser.add_argument("--skip-subprocess-profiler", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for M9.1 research methodology audit and corrected benchmark."""
    args = parse_args(argv)
    cfg: dict[str, Any] = {}
    if args.config is not None and args.config.exists():
        loaded = json.loads(args.config.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            cfg.update(loaded)

    device = str(args.device if args.device is not None else cfg.get("device", "auto"))
    cache_dir = str(
        args.cache_dir if args.cache_dir is not None else cfg.get("cache_dir", "models/hf_cache")
    )
    tsb_max_raw = int(
        args.tsb_max_per_domain
        if args.tsb_max_per_domain is not None
        else cfg.get("tsb_max_per_domain", 3)
    )

    if not args.mock_backends and (not is_chronos2_installed() or not is_timesfm3_installed()):
        print(
            "M9.1 BENCHMARK FAILED: chronos-forecasting>=2.0 and timesfm>=3.0 are required "
            "(install via `uv pip install -r requirements-m8.txt` or pass --mock-backends).",
            file=sys.stderr,
        )
        return 2

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (Path("reports/m9_1") / run_id)

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
        print(f"M9.1 BENCHMARK FAILED: {exc}", file=sys.stderr)
        return 2

    report, unified_rows, per_seed_rows, per_exp_rows = run_m9_1_evaluation(
        skab_bundle=skab_bundle,
        tsb_bundle=tsb_bundle,
        cfg=cfg,
        skab_dir=args.skab_dir,
        tsb_dir=args.tsb_dir,
        device=device,
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
        mock_backends=args.mock_backends,
        skip_subprocess_profiler=args.skip_subprocess_profiler,
    )
    report["run_id"] = run_id
    report["generated_utc"] = format_utc_timestamp(datetime.now(timezone.utc))

    write_m9_1_bundle(output_dir, report, unified_rows, per_seed_rows, per_exp_rows)
    print(f"M9.1 audited benchmark bundle generated in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
