"""Unit and end-to-end smoke tests for DriftGuard-Kafka M9 evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from app.m9_evaluation import (
    aggregate_across_seeds,
    analyze_resource_efficiency,
    compute_error_analysis,
    select_best_per_scenario,
)
from app.m9_evaluation import (
    main as m9_main,
)
from tests.test_m4_benchmark import _write_synthetic_skab_tree, _write_synthetic_tsb_tree


def test_error_analysis_fp_and_fn_taxonomy() -> None:
    # 120 samples with two anomaly episodes: [30..45] (len 16, short) and [70..100] (len 31, long)
    y_true = np.zeros(120, dtype=np.int64)
    y_true[30:46] = 1
    y_true[70:101] = 1

    preds = np.zeros(120, dtype=np.int64)
    # Isolated background FP at index 2 (far from episode at 30)
    preds[2] = 1
    # Pre-fault lead FP at index 25 (5 steps before 30)
    preds[25] = 1
    # Post-fault recovery FP at index 50 (5 steps after 45)
    preds[50] = 1
    # Episode 1 [30..45] completely missed (short episode: 16 FNs)
    # Episode 2 [70..100]: detected at 73..80 (3 onset-delay FNs: 70,71,72; 20 dropoff FNs: 81..100)
    preds[73:81] = 1

    err = compute_error_analysis(y_true, preds, proximity_window=10, short_episode_max_len=20)
    assert err["fp_total"] == 3
    assert err["fp_anomaly_free"] == 0
    assert err["fp_pre_fault_lead"] == 1
    assert err["fp_post_fault_recovery"] == 1
    assert err["fp_isolated_background"] == 1
    assert err["fn_total"] == 16 + 3 + 20
    assert err["missed_episodes_total"] == 1
    assert err["missed_episodes_short"] == 1
    assert err["missed_episodes_long"] == 0
    assert err["partial_episodes"] == 1
    assert err["fn_missed_episodes_points"] == 16
    assert err["fn_onset_delay_points"] == 3
    assert err["fn_adaptation_dropoff_points"] == 20

    # Anomaly-free series classifies all FPs under fp_anomaly_free
    y_af = np.zeros(40, dtype=np.int64)
    p_af = np.zeros(40, dtype=np.int64)
    p_af[[5, 10]] = 1
    err_af = compute_error_analysis(y_af, p_af)
    assert err_af["fp_total"] == 2
    assert err_af["fp_anomaly_free"] == 2


def test_multiseed_aggregation_sla_flagging_and_scenario_winners() -> None:
    seed_rows = [
        {
            "dataset_suite": "SKAB",
            "evaluation_scope": "fault_experiments_only",
            "model": "tranad",
            "seed": 42,
            "is_stochastic": True,
            "experiments": 34,
            "test_samples": 1000,
            "micro_precision": 0.70,
            "micro_recall": 0.86,
            "micro_f1": 0.77,
            "micro_fpr": 0.40,
            "macro_precision": 0.70,
            "macro_recall": 0.86,
            "macro_f1": 0.76,
            "macro_fpr": 0.40,
            "mean_pr_auc": 0.79,
            "mean_vus_pr": 0.81,
            "episode_detection_rate": 1.0,
            "mean_detection_delay_samples": 7.0,
            "total_training_duration_s": 4.0,
            "total_inference_wall_s": 0.2,
            "mean_latency_ms": 0.2,
            "throughput_events_per_sec": 5000.0,
            "max_peak_tracemalloc_kib": 900.0,
            "max_rss_mib": 1400.0,
            "mean_forecast_mae": None,
            "mean_forecast_rmse": None,
            "mean_normalized_mae": None,
            "mean_interval_coverage_80pct": None,
            "fp": 100,
            "fn": 50,
            "fp_anomaly_free": 0,
            "fp_pre_fault_lead": 10,
            "fp_post_fault_recovery": 10,
            "fp_isolated_background": 80,
            "missed_episodes": 0,
            "missed_episodes_short": 0,
            "missed_episodes_long": 0,
            "partial_episodes": 2,
            "fn_missed_episodes_points": 0,
            "fn_onset_delay_points": 15,
            "fn_adaptation_dropoff_points": 35,
        },
        {
            "dataset_suite": "SKAB",
            "evaluation_scope": "fault_experiments_only",
            "model": "tranad",
            "seed": 1337,
            "is_stochastic": True,
            "experiments": 34,
            "test_samples": 1000,
            "micro_precision": 0.72,
            "micro_recall": 0.88,
            "micro_f1": 0.79,
            "micro_fpr": 0.38,
            "macro_precision": 0.72,
            "macro_recall": 0.88,
            "macro_f1": 0.78,
            "macro_fpr": 0.38,
            "mean_pr_auc": 0.81,
            "mean_vus_pr": 0.83,
            "episode_detection_rate": 1.0,
            "mean_detection_delay_samples": 9.0,
            "total_training_duration_s": 4.4,
            "total_inference_wall_s": 0.2,
            "mean_latency_ms": 0.2,
            "throughput_events_per_sec": 5000.0,
            "max_peak_tracemalloc_kib": 920.0,
            "max_rss_mib": 1420.0,
            "mean_forecast_mae": None,
            "mean_forecast_rmse": None,
            "mean_normalized_mae": None,
            "mean_interval_coverage_80pct": None,
            "fp": 90,
            "fn": 40,
            "fp_anomaly_free": 0,
            "fp_pre_fault_lead": 8,
            "fp_post_fault_recovery": 12,
            "fp_isolated_background": 70,
            "missed_episodes": 0,
            "missed_episodes_short": 0,
            "missed_episodes_long": 0,
            "partial_episodes": 2,
            "fn_missed_episodes_points": 0,
            "fn_onset_delay_points": 11,
            "fn_adaptation_dropoff_points": 29,
        },
    ]
    agg = aggregate_across_seeds(seed_rows)
    assert agg["n_seeds"] == 2
    assert agg["micro_f1_mean"] == 0.78
    assert agg["micro_f1_std"] > 0.0

    # High F1 (0.78 >= 0.70) with latency 0.2ms > 0.1ms and RSS > 1024 MiB -> flagged!
    sla_res = analyze_resource_efficiency(
        [agg],
        {
            "max_latency_ms": 0.1,
            "min_throughput_ev_s": 10000.0,
            "max_rss_mib": 1024.0,
            "max_training_s": 30.0,
            "high_f1_threshold_skab": 0.70,
        },
    )
    assert sla_res[0]["status"] == "HIGH_F1_RESOURCE_VIOLATION"
    assert len(sla_res[0]["violations"]) >= 2

    # Different winners across scenarios -> universal_winner_supported is False
    fast_row = dict(agg)
    fast_row["model"] = "static_threshold"
    fast_row["micro_f1_mean"] = 0.75
    fast_row["mean_latency_ms_mean"] = 0.002
    best = select_best_per_scenario([agg, fast_row])
    assert best["universal_winner_supported"] is False
    assert set(best["distinct_winning_models"]) == {"tranad", "static_threshold"}


def test_end_to_end_m9_evaluation_smoke_and_charts(tmp_path: Path) -> None:
    skab_root = tmp_path / "skab"
    tsb_root = tmp_path / "tsb"
    _write_synthetic_skab_tree(skab_root)
    _write_synthetic_tsb_tree(tsb_root)

    cfg_path = tmp_path / "m9_smoke.json"
    cfg_path.write_text(
        json.dumps(
            {
                "seeds": [42, 1337],
                "device": "cpu",
                "calibration_quantile": 0.98,
                "tsb_max_per_domain": 1,
                "tsfm_max_tsb_series_rows": 50000,
                "resource_sla": {
                    "max_latency_ms": 0.1,
                    "min_throughput_ev_s": 10000.0,
                    "max_rss_mib": 1024.0,
                    "max_training_s": 30.0,
                    "high_f1_threshold_skab": 0.70,
                    "high_f1_threshold_tsb": 0.35,
                },
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
                    "batch_size": 16,
                    "max_active_univariate_channels": 4,
                },
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "m9_out"
    rc = m9_main(
        [
            "--config",
            str(cfg_path),
            "--run-id",
            "m9_smoke",
            "--skab-dir",
            str(skab_root),
            "--tsb-dir",
            str(tsb_root),
            "--mock-backends",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    for fname in (
        "m9_report.json",
        "m9_report.md",
        "unified_comparison.csv",
        "per_seed_summary.csv",
        "per_experiment_metrics.csv",
        "error_analysis.csv",
    ):
        fpath = out_dir / fname
        assert fpath.exists() and fpath.stat().st_size > 0

    for chart_name in (
        "skab_multivariate_metrics.svg",
        "tsb_univariate_metrics.svg",
        "f1_vs_latency_pareto.svg",
        "resource_efficiency.svg",
    ):
        cpath = out_dir / "charts" / chart_name
        assert cpath.exists() and "<svg" in cpath.read_text(encoding="utf-8")

    rep = json.loads((out_dir / "m9_report.json").read_text(encoding="utf-8"))
    assert rep["protocol_verification"]["verified_splits"] == 5
    assert rep["protocol_verification"]["test_labels_excluded_from_fit_and_calibration"] is True
    assert len(rep["forecasting_vs_detection"]) >= 8
    assert len(rep["error_analysis"]) == 20  # 10 models * 2 suites

    # Verify multi-seed trained models have n_seeds == 2 vs deterministic n_seeds == 1
    by_m = {
        r["model"]: r
        for r in rep["unified_comparison"]
        if r["evaluation_scope"] == "fault_experiments_only"
    }
    assert by_m["tranad"]["n_seeds"] == 2
    assert by_m["lstm_autoencoder"]["n_seeds"] == 2
    assert by_m["tcn_autoencoder"]["n_seeds"] == 2
    assert by_m["isolation_forest"]["n_seeds"] == 2
    assert by_m["half_space_trees"]["n_seeds"] == 2
    assert by_m["static_threshold"]["n_seeds"] == 1
    assert by_m["chronos2_zero_shot"]["n_seeds"] == 1
    assert by_m["timesfm3_zero_shot"]["n_seeds"] == 1

    with (out_dir / "per_experiment_metrics.csv").open(encoding="utf-8") as handle:
        # 5 splits * (5 stochastic models * 2 seeds + 5 deterministic models * 1 seed) = 75 rows
        assert len(list(csv.DictReader(handle))) == 75
