"""Unit and end-to-end smoke tests for DriftGuard-Kafka M9.1 audited benchmark."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from app.m4_models import IsolationForestDetector, StaticThresholdBaseline
from app.m8_benchmark import DeterministicPersistenceBackend
from app.m8_chronos2 import Chronos2Detector
from app.m8_tsfm_base import TSFMDetectorConfig
from app.m9_1_benchmark import main as m9_1_main
from app.m9_1_profiler import run_isolated_subprocess_profiling
from app.m9_1_stats import (
    ALL_CALIBRATION_STRATEGIES,
    bootstrap_mean_ci,
    calibrate_threshold_validation_only,
    evaluate_threshold_on_test,
    paired_bootstrap_test,
)
from app.m9_1_tracks import IndependentChannelWrapper, forecast_chunked_tsfm_split
from tests.test_m4_benchmark import _write_synthetic_skab_tree, _write_synthetic_tsb_tree


def test_independent_channel_wrapper_and_chunked_tsfm_tracks() -> None:
    rng = np.random.default_rng(123)
    x_norm = np.column_stack(
        [
            10.0 + 0.05 * rng.normal(size=180),
            20.0 + 0.05 * rng.normal(size=180),
            30.0 + 0.05 * rng.normal(size=180),
            40.0 + 0.05 * rng.normal(size=180),
            50.0 + 0.05 * rng.normal(size=180),
            np.full(180, 7.0),  # constant inactive channel
        ]
    )
    x_train, x_val = x_norm[:100], x_norm[100:140]
    x_test = x_norm[140:].copy()
    x_test[15:25, 0] += 9.0
    feats = ("c0", "c1", "c2", "c3", "c4", "const")

    # Track B wrapper works identically for both StaticThreshold and IsolationForest
    for name, factory in (
        ("static_threshold", lambda: StaticThresholdBaseline(calibration_quantile=0.95)),
        ("isolation_forest", lambda: IsolationForestDetector(seed=42, calibration_quantile=0.95)),
    ):
        wrap = IndependentChannelWrapper(
            model_name=name,
            factory=factory,
            calibration_quantile=0.95,
            max_active_channels=4,
        )
        wrap.fit_and_calibrate(x_train, x_val, feats)
        inf = wrap.score_and_predict(x_test, feats)
        assert inf.extra_diagnostics["channel_mode"] == "independent_channel_unified"
        assert int(np.sum(inf.predictions[15:25])) >= 8

    # Chunked multivariate TSFM groups (D=6 > group=4 -> multivariate_chunked_groups_4)
    det = Chronos2Detector(
        TSFMDetectorConfig(model_id="mock/c2", context_length=16, prediction_length=8),
        backend=DeterministicPersistenceBackend("mock/c2"),
    )
    det._feature_names = feats
    det._fit_normal(x_train, feats)
    assert det._train_tail is not None
    details_mv, mode_mv = forecast_chunked_tsfm_split(
        det,
        det._train_tail,
        x_test,
        feats,
        track="multivariate",
        max_multivariate_group_variates=4,
        long_series_row_threshold=20,
        long_series_prediction_length=16,
    )
    assert mode_mv == "multivariate_chunked_groups_4"
    assert details_mv.residual_scores.shape == (40,)

    details_ic, mode_ic = forecast_chunked_tsfm_split(
        det,
        det._train_tail,
        x_test,
        feats,
        track="independent_channel",
    )
    assert mode_ic == "independent_channel_unified"
    assert details_ic.residual_scores.shape == (40,)


def test_validation_only_calibration_and_bootstrap_significance() -> None:
    rng = np.random.default_rng(55)
    val_scores = np.abs(rng.normal(loc=0.5, scale=0.1, size=200))
    test_scores = np.abs(rng.normal(loc=0.5, scale=0.1, size=200))
    y_true = np.zeros(200, dtype=np.int64)
    y_true[100:120] = 1
    test_scores[100:120] += 3.5

    for strat in ALL_CALIBRATION_STRATEGIES:
        thr = calibrate_threshold_validation_only(val_scores, strat)
        ev = evaluate_threshold_on_test(y_true, test_scores, thr)
        assert ev["recall"] >= 0.9
        assert ev["false_alarms_per_1k_normal"] is not None
        assert ev["false_alarms_per_1k_normal"] < 150.0

    # Bootstrap CI and paired test: 0.0006 noisy difference is NOT significant
    a_close = [0.380, 0.350, 0.410, 0.370, 0.390, 0.360, 0.400, 0.375, 0.385]
    b_close = [0.381, 0.348, 0.412, 0.369, 0.388, 0.362, 0.398, 0.376, 0.381]
    ci = bootstrap_mean_ci(a_close, n_resamples=500, seed=42)
    assert ci["mean"] is not None and ci["ci_low"] is not None and ci["ci_high"] is not None
    assert float(ci["ci_low"]) <= float(ci["mean"]) <= float(ci["ci_high"])

    res_ns = paired_bootstrap_test(a_close, b_close, n_resamples=500, seed=42)
    assert res_ns["significant_at_005"] is False

    # Clear systematic gain (+0.20 on every series) IS significant
    b_much_lower = [v - 0.20 for v in a_close]
    res_sig = paired_bootstrap_test(a_close, b_much_lower, n_resamples=500, seed=42)
    assert res_sig["significant_at_005"] is True
    assert res_sig["p_value"] < 0.05


def test_isolated_subprocess_profiler_and_end_to_end_m9_1_smoke(tmp_path: Path) -> None:
    skab_root = tmp_path / "skab"
    tsb_root = tmp_path / "tsb"
    _write_synthetic_skab_tree(skab_root)
    _write_synthetic_tsb_tree(tsb_root)

    # 1. Isolated subprocess profiler on 2 models (classical + mock TSFM)
    prof = run_isolated_subprocess_profiling(
        skab_dir=skab_root,
        tsb_dir=tsb_root,
        suites=("SKAB",),
        models=("static_threshold", "chronos2_zero_shot"),
        device="cpu",
        local_files_only=True,
        mock_backends=True,
    )
    assert len(prof) == 2
    for row in prof:
        assert row["status"] == "ok"
        assert row["interpreter_baseline_rss_mib"] > 0.0
        assert row["offline_batched_mean_latency_ms"] > 0.0

    # 2. Full M9.1 dual-track smoke test
    cfg_path = tmp_path / "m9_1_smoke.json"
    cfg_path.write_text(
        json.dumps(
            {
                "seeds_skab": [42, 1337],
                "seeds_tsb": [42, 1337],
                "device": "cpu",
                "calibration_quantile": 0.98,
                "calibration_strategies": ["empirical_q99", "mad_z35", "iqr_k30"],
                "tsb_max_per_domain": 1,
                "bootstrap_resamples": 200,
                "lstm": {
                    "window_size": 8,
                    "hidden_dim": 8,
                    "latent_dim": 4,
                    "num_layers": 1,
                    "dropout": 0.0,
                    "epochs": 2,
                    "batch_size": 32,
                },
                "tcn": {
                    "window_size": 8,
                    "hidden_channels": 8,
                    "latent_dim": 4,
                    "dilations": [1, 2],
                    "dropout": 0.0,
                    "epochs": 2,
                    "batch_size": 32,
                },
                "tranad": {
                    "window_size": 8,
                    "dim_feedforward": 8,
                    "dropout": 0.0,
                    "epochs": 2,
                    "batch_size": 32,
                },
                "tsfm": {
                    "context_length": 16,
                    "prediction_length": 8,
                    "long_series_row_threshold": 50000,
                    "long_series_prediction_length": 16,
                    "max_multivariate_group_variates": 32,
                    "max_independent_channels": 4,
                    "max_forecast_blocks_per_series": 5000,
                    "batch_size": 16,
                },
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "m9_1_out"
    rc = m9_1_main(
        [
            "--config",
            str(cfg_path),
            "--run-id",
            "m9_1_smoke",
            "--skab-dir",
            str(skab_root),
            "--tsb-dir",
            str(tsb_root),
            "--mock-backends",
            "--skip-subprocess-profiler",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    for fname in (
        "m9_1_report.json",
        "m9_1_report.md",
        "protocol_comparison.md",
        "unified_comparison.csv",
        "per_seed_summary.csv",
        "per_experiment_metrics.csv",
        "error_analysis.csv",
        "calibration_study.csv",
        "statistical_tests.csv",
    ):
        fpath = out_dir / fname
        assert fpath.exists() and fpath.stat().st_size > 0, f"Missing/empty {fname}"

    for ch in (
        "track_a_multivariate.svg",
        "track_b_independent_channel.svg",
        "protocol_gap_comparison.svg",
        "calibration_fpr_tradeoff.svg",
        "isolated_resource_profile.svg",
    ):
        assert (out_dir / "charts" / ch).exists()

    rep = json.loads((out_dir / "m9_1_report.json").read_text(encoding="utf-8"))
    assert rep["dataset_provenance"]["TSB-AD-M"]["dataset_type"] == "multivariate"
    tracks_present = {r["track"] for r in rep["unified_comparison"]}
    assert tracks_present == {"multivariate", "independent_channel"}

    # Verify strict per_seed_summary scope guard (never mislabel partial subsets!)
    with (out_dir / "per_seed_summary.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["evaluation_scope"] == "full_2_series":
                assert int(row["experiments"]) == 2
