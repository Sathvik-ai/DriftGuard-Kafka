"""Unit and smoke tests for DriftGuard-Kafka M5 LSTM Autoencoder."""

from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from app.m5_benchmark import (
    compute_detection_delay,
    extract_anomaly_episodes,
)
from app.m5_benchmark import (
    main as m5_main,
)
from app.m5_lstm_autoencoder import (
    LSTMAutoencoder,
    LSTMAutoencoderConfig,
    LSTMAutoencoderDetector,
    TrainingOnlyScaler,
    generate_causal_windows,
    generate_training_windows,
    select_compute_device,
)
from tests.test_m4_benchmark import _write_synthetic_skab_tree, _write_synthetic_tsb_tree


def test_device_detection_and_cpu_fallback() -> None:
    cpu_info = select_compute_device("cpu")
    assert cpu_info.selected == "cpu"
    assert cpu_info.fallback_reason is None

    auto_info = select_compute_device("auto")
    assert auto_info.selected in {"mps", "cuda", "cpu"}
    if not auto_info.mps_available:
        mps_fallback = select_compute_device("mps")
        assert mps_fallback.selected == "cpu"
        assert mps_fallback.fallback_reason is not None

    with pytest.raises(ValueError):
        select_compute_device("tpu")


def test_window_generation_preserves_series_boundaries_and_causality() -> None:
    series = np.arange(30, dtype=np.float64).reshape(10, 3)
    train_win = generate_training_windows(series, window_size=4, stride=1)
    assert train_win.shape == (7, 4, 3)
    assert np.allclose(train_win[0], series[:4])
    assert np.allclose(train_win[-1], series[6:10])

    # Short series (< window_size) left-pads strictly from series[0] without crossing splits.
    short_win = generate_training_windows(series[:2], window_size=5, stride=1)
    assert short_win.shape == (1, 5, 3)
    assert np.allclose(short_win[0, :3], np.repeat(series[:1], 3, axis=0))
    assert np.allclose(short_win[0, 3:], series[:2])

    causal_win = generate_causal_windows(series, window_size=4)
    assert causal_win.shape == (10, 4, 3)
    # At t=0, causal window contains only series[0] repeated 4 times (zero future lookahead).
    assert np.allclose(causal_win[0], np.repeat(series[:1], 4, axis=0))
    # At t=5, causal window is strictly series[2:6].
    assert np.allclose(causal_win[5], series[2:6])


def test_lstm_autoencoder_shapes_and_training_only_scaler() -> None:
    model = LSTMAutoencoder(input_dim=8, hidden_dim=16, latent_dim=6, num_layers=2, dropout=0.1)
    x = torch.randn(5, 12, 8)
    recon, z = model(x)
    assert z.shape == (5, 6)
    assert recon.shape == (5, 12, 8)

    # Constant zero-variance channel should receive safe scale=1.0 without NaNs.
    train_data = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]], dtype=np.float64)
    scaler = TrainingOnlyScaler().fit(train_data)
    assert list(scaler.active_channel_mask) == [True, False]
    assert scaler.scale_[1] == 1.0
    transformed = scaler.transform(np.array([[3.0, 9.0]], dtype=np.float64))
    assert np.isfinite(transformed).all()
    assert transformed[0, 0] == pytest.approx(0.0)
    assert transformed[0, 1] == pytest.approx(4.0)


def test_deterministic_training_scoring_thresholding_and_modes(tmp_path: Path) -> None:
    rng = np.random.default_rng(99)
    t = np.linspace(0, 12.0 * np.pi, 140)
    x_normal = np.column_stack(
        [
            np.sin(t) + 0.03 * rng.normal(size=140),
            np.cos(t) + 0.03 * rng.normal(size=140),
            0.5 * np.sin(2 * t) + 0.03 * rng.normal(size=140),
        ]
    )
    x_train = x_normal[:90]
    x_val = x_normal[90:115]
    x_test = x_normal[115:].copy()
    x_test[10:18] += 6.5  # strong multivariate anomaly block
    feats = ("s0", "s1", "s2")

    # Verify structural label exclusion on fit_and_calibrate / score_and_predict.
    cfg_mv = LSTMAutoencoderConfig(
        window_size=8,
        hidden_dim=16,
        latent_dim=8,
        num_layers=1,
        dropout=0.0,
        batch_size=16,
        epochs=12,
        learning_rate=0.005,
        early_stopping_patience=3,
        calibration_quantile=0.95,
        channel_mode="multivariate",
        device="cpu",
        seed=42,
    )
    det1 = LSTMAutoencoderDetector(cfg_mv)
    assert list(inspect.signature(det1.fit_and_calibrate).parameters) == [
        "x_train_normal",
        "x_val_normal",
        "feature_names",
    ]
    cal1 = det1.fit_and_calibrate(x_train, x_val, feats)
    inf1 = det1.score_and_predict(x_test, feats)

    # Re-running with identical seed produces identical threshold and scores.
    det2 = LSTMAutoencoderDetector(cfg_mv)
    cal2 = det2.fit_and_calibrate(x_train, x_val, feats)
    inf2 = det2.score_and_predict(x_test, feats)
    assert cal1.calibrated_threshold == pytest.approx(cal2.calibrated_threshold, rel=1e-5)
    assert np.allclose(inf1.scores, inf2.scores, rtol=1e-5, atol=1e-6)

    # Per-feature and overall scores reflect injected anomaly window.
    per_feat = det1.last_per_feature_scores
    assert per_feat is not None and per_feat.shape == x_test.shape
    assert np.mean(inf1.scores[10:18]) > 5.0 * np.mean(inf1.scores[:10])
    assert int(np.sum(inf1.predictions[10:18])) >= 6

    ckpt_path = tmp_path / "model.pt"
    det1.save_checkpoint(ckpt_path)
    assert ckpt_path.exists() and ckpt_path.stat().st_size > 0

    # Independent univariate mode (used for TSB series) models each channel as (B, W, 1).
    cfg_uv = LSTMAutoencoderConfig(
        window_size=8,
        hidden_dim=12,
        latent_dim=6,
        num_layers=1,
        dropout=0.0,
        batch_size=16,
        epochs=10,
        learning_rate=0.005,
        calibration_quantile=0.95,
        channel_mode="independent_univariate",
        score_reduction="window_mean",
        device="cpu",
        seed=42,
    )
    det_uv = LSTMAutoencoderDetector(cfg_uv)
    det_uv.fit_and_calibrate(x_train, x_val, feats)
    assert det_uv.training_history["input_dim"] == 1
    inf_uv = det_uv.score_and_predict(x_test, feats)
    assert inf_uv.scores.shape == (x_test.shape[0],)
    assert np.mean(inf_uv.scores[10:18]) > np.mean(inf_uv.scores[:8])


def test_episode_detection_delay_and_end_to_end_m5_cli(tmp_path: Path) -> None:
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0], dtype=np.int64)
    preds = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int64)
    assert extract_anomaly_episodes(y_true) == [(2, 4), (6, 7)]
    delay = compute_detection_delay(y_true, preds)
    assert delay["total_episodes"] == 2
    assert delay["detected_episodes"] == 1
    assert delay["missed_episodes"] == 1
    assert delay["mean_detection_delay_samples"] == 1.0

    skab_root = tmp_path / "skab"
    tsb_root = tmp_path / "tsb"
    _write_synthetic_skab_tree(skab_root)
    _write_synthetic_tsb_tree(tsb_root)

    cfg_path = tmp_path / "m5_smoke.json"
    cfg_path.write_text(
        json.dumps(
            {
                "seed": 42,
                "device": "cpu",
                "window_size": 8,
                "hidden_dim": 12,
                "latent_dim": 6,
                "num_layers": 1,
                "dropout": 0.0,
                "batch_size": 32,
                "epochs": 4,
                "learning_rate": 0.005,
                "early_stopping_patience": 2,
                "calibration_quantile": 0.98,
                "skab_channel_mode": "multivariate",
                "tsb_channel_mode": "independent_univariate",
                "tsb_max_per_domain": 1,
            }
        ),
        encoding="utf-8",
    )
    loaded_cfg = LSTMAutoencoderConfig.from_json_file(cfg_path)
    assert loaded_cfg.window_size == 8

    out_dir = tmp_path / "reports_m5"
    models_dir = tmp_path / "models_m5"
    rc = m5_main(
        [
            "--config",
            str(cfg_path),
            "--run-id",
            "m5_unit_smoke",
            "--skab-dir",
            str(skab_root),
            "--tsb-dir",
            str(tsb_root),
            "--save-models-dir",
            str(models_dir),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    for fname in (
        "m5_report.json",
        "m5_report.md",
        "comparison_summary.csv",
        "per_experiment_metrics.csv",
    ):
        fpath = out_dir / fname
        assert fpath.exists() and fpath.stat().st_size > 0

    report = json.loads((out_dir / "m5_report.json").read_text(encoding="utf-8"))
    assert report["series_modes"] == {
        "SKAB": "multivariate",
        "TSB-AD-M": "independent_univariate",
    }
    models_evaluated = {row["model"] for row in report["comparison_summary"]}
    assert models_evaluated == {
        "static_threshold",
        "isolation_forest",
        "half_space_trees",
        "lstm_autoencoder",
    }
    with (out_dir / "per_experiment_metrics.csv").open(encoding="utf-8") as handle:
        exp_rows = list(csv.DictReader(handle))
    # (3 SKAB + 2 stratified TSB-AD-M) * 4 models = 20 rows
    assert len(exp_rows) == 20
    assert any(models_dir.rglob("*.pt"))
