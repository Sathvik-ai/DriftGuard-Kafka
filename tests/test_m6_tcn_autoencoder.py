"""Unit and smoke tests for DriftGuard-Kafka M6 TCN Autoencoder."""

from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from app.m5_benchmark import main as m5_main
from app.m6_benchmark import main as m6_main
from app.m6_tcn_autoencoder import (
    CausalConv1d,
    TCNAutoencoder,
    TCNAutoencoderConfig,
    TCNAutoencoderDetector,
    TCNResidualBlock,
)
from tests.test_m4_benchmark import _write_synthetic_skab_tree, _write_synthetic_tsb_tree


def test_causal_conv_and_tcn_encoder_strictly_prevent_future_leakage() -> None:
    conv = CausalConv1d(in_channels=3, out_channels=6, kernel_size=3, dilation=2)
    x = torch.randn(2, 3, 12, requires_grad=True)
    y = conv(x)
    assert y.shape == (2, 6, 12)
    # Output at step t=4 must have zero gradient wrt all future inputs s > 4!
    (grad_x,) = torch.autograd.grad(y[:, :, 4].sum(), x)
    assert torch.all(grad_x[:, :, 5:] == 0.0)

    # Verify full multi-block dilated encoder stack also obeys strict causality at step t=7.
    model = TCNAutoencoder(
        input_dim=4,
        window_size=16,
        hidden_channels=12,
        latent_dim=6,
        kernel_size=3,
        dilations=(1, 2, 4),
        dropout=0.0,
    )
    model.eval()
    seq = torch.randn(2, 16, 4, requires_grad=True)
    enc_feat = model.encode_causal_features(seq)
    assert enc_feat.shape == (2, 12, 16)
    (grad_seq,) = torch.autograd.grad(enc_feat[:, :, 7].sum(), seq)
    assert torch.all(grad_seq[:, 8:, :] == 0.0)


def test_tcn_residual_blocks_receptive_field_and_undercomplete_bottleneck() -> None:
    cfg = TCNAutoencoderConfig(window_size=32, kernel_size=3, dilations=(1, 2, 4, 8), latent_dim=16)
    # 1 + 2 * (3 - 1) * (1 + 2 + 4 + 8) = 61 >= 32
    assert cfg.receptive_field == 61

    # Identity shortcut (in == out) and 1x1 projection shortcut (in != out)
    blk_proj = TCNResidualBlock(in_channels=2, out_channels=8, kernel_size=3, dilation=2)
    blk_id = TCNResidualBlock(in_channels=8, out_channels=8, kernel_size=3, dilation=4)
    h = blk_id(blk_proj(torch.randn(3, 2, 20)))
    assert h.shape == (3, 8, 20)

    # For 1D univariate window_size=10 (total values = 10), latent_dim=16 shrinks to 5
    # so the bottleneck stays strictly undercomplete.
    uv_model = TCNAutoencoder(input_dim=1, window_size=10, hidden_channels=8, latent_dim=16)
    assert uv_model.latent_dim == 5
    recon, z = uv_model(torch.randn(4, 10, 1))
    assert z.shape == (4, 5)
    assert recon.shape == (4, 10, 1)


def test_tcn_detector_training_modes_determinism_and_thresholding(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    t = np.linspace(0, 14.0 * np.pi, 150)
    x_normal = np.column_stack(
        [
            np.sin(t) + 0.03 * rng.normal(size=150),
            np.cos(t) + 0.03 * rng.normal(size=150),
            0.5 * np.cos(2 * t) + 0.03 * rng.normal(size=150),
        ]
    )
    x_train = x_normal[:95]
    x_val = x_normal[95:120]
    x_test = x_normal[120:].copy()
    x_test[8:18] += 7.0
    feats = ("c0", "c1", "c2")

    cfg_mv = TCNAutoencoderConfig(
        window_size=12,
        hidden_channels=16,
        latent_dim=6,
        kernel_size=3,
        dilations=(1, 2, 4),
        dropout=0.0,
        batch_size=16,
        epochs=12,
        learning_rate=0.004,
        calibration_quantile=0.95,
        channel_mode="multivariate",
        device="cpu",
        seed=42,
    )
    det1 = TCNAutoencoderDetector(cfg_mv)
    assert list(inspect.signature(det1.fit_and_calibrate).parameters) == [
        "x_train_normal",
        "x_val_normal",
        "feature_names",
    ]
    cal1 = det1.fit_and_calibrate(x_train, x_val, feats)
    inf1 = det1.score_and_predict(x_test, feats)

    det2 = TCNAutoencoderDetector(cfg_mv)
    cal2 = det2.fit_and_calibrate(x_train, x_val, feats)
    inf2 = det2.score_and_predict(x_test, feats)
    assert cal1.calibrated_threshold == pytest.approx(cal2.calibrated_threshold, rel=1e-5)
    assert np.allclose(inf1.scores, inf2.scores, rtol=1e-5, atol=1e-6)

    per_feat = det1.last_per_feature_scores
    assert per_feat is not None and per_feat.shape == x_test.shape
    assert np.mean(inf1.scores[8:18]) > 5.0 * np.mean(inf1.scores[:8])
    assert int(np.sum(inf1.predictions[8:18])) >= 7

    ckpt = tmp_path / "tcn.pt"
    det1.save_checkpoint(ckpt)
    assert ckpt.exists() and ckpt.stat().st_size > 0

    # Vectorized independent_univariate mode
    cfg_uv = TCNAutoencoderConfig(
        window_size=12,
        hidden_channels=12,
        latent_dim=4,
        kernel_size=3,
        dilations=(1, 2),
        dropout=0.0,
        batch_size=16,
        epochs=10,
        learning_rate=0.004,
        calibration_quantile=0.95,
        channel_mode="independent_univariate",
        score_reduction="window_mean",
        device="cpu",
        seed=42,
    )
    det_uv = TCNAutoencoderDetector(cfg_uv)
    det_uv.fit_and_calibrate(x_train, x_val, feats)
    assert det_uv.training_history["input_dim"] == 1
    inf_uv = det_uv.score_and_predict(x_test, feats)
    assert inf_uv.scores.shape == (x_test.shape[0],)
    assert np.mean(inf_uv.scores[8:18]) > np.mean(inf_uv.scores[:6])


def test_end_to_end_m6_benchmark_in_process_and_m5_reuse_preservation(tmp_path: Path) -> None:
    skab_root = tmp_path / "skab"
    tsb_root = tmp_path / "tsb"
    _write_synthetic_skab_tree(skab_root)
    _write_synthetic_tsb_tree(tsb_root)

    m5_out = tmp_path / "reports_m5" / "m5_ref"
    rc5 = m5_main(
        [
            "--run-id",
            "m5_ref",
            "--device",
            "cpu",
            "--window-size",
            "8",
            "--hidden-dim",
            "12",
            "--latent-dim",
            "6",
            "--num-layers",
            "1",
            "--dropout",
            "0.0",
            "--epochs",
            "3",
            "--tsb-max-per-domain",
            "1",
            "--skab-dir",
            str(skab_root),
            "--tsb-dir",
            str(tsb_root),
            "--output-dir",
            str(m5_out),
        ]
    )
    assert rc5 == 0
    m5_snapshot_before = (m5_out / "m5_report.json").read_bytes()

    cfg_path = tmp_path / "m6_smoke.json"
    cfg_path.write_text(
        json.dumps(
            {
                "seed": 42,
                "device": "cpu",
                "window_size": 8,
                "hidden_channels": 12,
                "latent_dim": 4,
                "kernel_size": 3,
                "dilations": [1, 2],
                "dropout": 0.0,
                "batch_size": 32,
                "epochs": 4,
                "learning_rate": 0.005,
                "calibration_quantile": 0.98,
                "skab_channel_mode": "multivariate",
                "tsb_channel_mode": "independent_univariate",
                "tsb_max_per_domain": 1,
            }
        ),
        encoding="utf-8",
    )
    parsed_cfg = TCNAutoencoderConfig.from_json_file(cfg_path)
    assert parsed_cfg.dilations == (1, 2)

    # 1. Test split-verified reuse of M5 bundle without modifying M5 files!
    m6_reuse_out = tmp_path / "reports_m6_reuse"
    models_m6 = tmp_path / "models_m6"
    rc6_reuse = m6_main(
        [
            "--config",
            str(cfg_path),
            "--run-id",
            "m6_reuse_test",
            "--skab-dir",
            str(skab_root),
            "--tsb-dir",
            str(tsb_root),
            "--reuse-m5-dir",
            str(m5_out),
            "--save-models-dir",
            str(models_m6),
            "--output-dir",
            str(m6_reuse_out),
        ]
    )
    assert rc6_reuse == 0
    # Verify M5 report was NOT modified byte-for-byte!
    assert (m5_out / "m5_report.json").read_bytes() == m5_snapshot_before

    rep_reuse = json.loads((m6_reuse_out / "m6_report.json").read_text(encoding="utf-8"))
    assert rep_reuse["reproducibility"]["baseline_source"] == "reused_verified_m5_bundle:m5_ref"
    assert {r["model"] for r in rep_reuse["comparison_summary"]} == {
        "static_threshold",
        "isolation_forest",
        "half_space_trees",
        "lstm_autoencoder",
        "tcn_autoencoder",
    }
    with (m6_reuse_out / "per_experiment_metrics.csv").open(encoding="utf-8") as handle:
        # (3 SKAB + 2 TSB) * 5 models = 25 rows
        assert len(list(csv.DictReader(handle))) == 25

    # 2. Test standalone in-process 5-model execution (when --reuse-m5-dir is omitted).
    m6_inproc_out = tmp_path / "reports_m6_inproc"
    rc6_inproc = m6_main(
        [
            "--config",
            str(cfg_path),
            "--run-id",
            "m6_inproc_test",
            "--skab-dir",
            str(skab_root),
            "--tsb-dir",
            str(tsb_root),
            "--max-skab-files",
            "1",
            "--output-dir",
            str(m6_inproc_out),
        ]
    )
    assert rc6_inproc == 0
    rep_inproc = json.loads((m6_inproc_out / "m6_report.json").read_text(encoding="utf-8"))
    assert rep_inproc["reproducibility"]["baseline_source"] == "executed_in_process"
