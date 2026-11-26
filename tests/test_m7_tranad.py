"""Unit and smoke tests for DriftGuard-Kafka M7 TranAD."""

from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from app.m6_benchmark import main as m6_main
from app.m7_benchmark import main as m7_main
from app.m7_tranad import (
    TrainingOnlyMinMaxScaler,
    TranADConfig,
    TranADDetector,
    TranADNetwork,
)
from tests.test_m4_benchmark import _write_synthetic_skab_tree, _write_synthetic_tsb_tree


def test_tranad_two_phase_self_conditioning_and_focus_signal() -> None:
    net = TranADNetwork(n_feats=4, window_size=10, dim_feedforward=16, dropout=0.0)
    net.eval()
    src = torch.rand(10, 5, 4)
    elem = src[-1:, :, :]

    x1, x2, focus_c2 = net.forward_with_focus(src, elem)
    assert x1.shape == (1, 5, 4)
    assert x2.shape == (1, 5, 4)
    assert focus_c2.shape == (10, 5, 4)
    # Focus matrix c2 must equal exact squared deviation (x1 - src) ** 2!
    assert torch.allclose(focus_c2, (x1 - src).pow(2), atol=1e-6)
    # Sigmoid head constrains reconstructions to (0, 1).
    assert bool((x1 >= 0.0).all() and (x1 <= 1.0).all())
    assert bool((x2 >= 0.0).all() and (x2 <= 1.0).all())
    # Perturbing the focus signal must change Phase 2 output x2 (proving active conditioning).
    tgt_alt, mem_alt = net.encode(src, focus_c2 + 5.0, elem)
    x2_alt = net._run_decoder(net.decoder2_layers, tgt_alt, mem_alt)
    assert not torch.allclose(x2, x2_alt)


def test_training_only_minmax_scaler_and_feature_attribution(tmp_path: Path) -> None:
    train_mat = np.array([[10.0, 3.0], [15.0, 3.0], [20.0, 3.0]], dtype=np.float64)
    scaler = TrainingOnlyMinMaxScaler().fit(train_mat)
    assert list(scaler.active_channel_mask) == [True, False]
    scaled_train = scaler.transform(train_mat)
    assert np.allclose(scaled_train[:, 0], [0.0, 0.5, 1.0])
    assert np.allclose(scaled_train[:, 1], [0.5, 0.5, 0.5])

    rng = np.random.default_rng(23)
    t = np.linspace(0, 10.0 * np.pi, 140)
    x_normal = np.column_stack(
        [
            10.0 + np.sin(t) + 0.03 * rng.normal(size=140),
            20.0 + np.cos(t) + 0.03 * rng.normal(size=140),
            5.0 + 0.5 * np.sin(2 * t) + 0.03 * rng.normal(size=140),
        ]
    )
    x_train = x_normal[:90]
    x_val = x_normal[90:115]
    x_test = x_normal[115:].copy()
    # Inject anomaly specifically on feature index 1 ("pressure")
    x_test[8:18, 1] += 12.0
    feats = ("temp", "pressure", "vib")

    cfg = TranADConfig(
        window_size=8,
        dim_feedforward=16,
        epochs=10,
        batch_size=16,
        learning_rate=0.004,
        dropout=0.0,
        calibration_quantile=0.95,
        score_mode="both_phases",
        channel_mode="multivariate",
        device="cpu",
        seed=42,
    )
    det1 = TranADDetector(cfg)
    assert list(inspect.signature(det1.fit_and_calibrate).parameters) == [
        "x_train_normal",
        "x_val_normal",
        "feature_names",
    ]
    cal1 = det1.fit_and_calibrate(x_train, x_val, feats)
    inf1 = det1.score_and_predict(x_test, feats)

    det2 = TranADDetector(cfg)
    cal2 = det2.fit_and_calibrate(x_train, x_val, feats)
    inf2 = det2.score_and_predict(x_test, feats)
    assert cal1.calibrated_threshold == pytest.approx(cal2.calibrated_threshold, rel=1e-5)
    assert np.allclose(inf1.scores, inf2.scores, rtol=1e-5, atol=1e-6)

    # Feature-level attribution must identify "pressure" as top anomalous feature!
    assert inf1.extra_diagnostics["top_error_feature"] == "pressure"
    out = det1.last_score_output
    assert out is not None
    assert out.per_feature_scores.shape == x_test.shape
    assert np.mean(out.per_feature_scores[8:18, 1]) > 10.0 * np.mean(
        out.per_feature_scores[8:18, 0]
    )
    assert int(np.sum(inf1.predictions[8:18])) >= 8

    ckpt = tmp_path / "tranad.pt"
    det1.save_checkpoint(ckpt)
    assert ckpt.exists() and ckpt.stat().st_size > 0

    # Independent univariate mode (n_feats=1, d_model=2, nhead=1)
    cfg_uv = TranADConfig(
        window_size=8,
        dim_feedforward=12,
        epochs=8,
        batch_size=16,
        learning_rate=0.004,
        dropout=0.0,
        calibration_quantile=0.95,
        score_mode="phase2",
        channel_mode="independent_univariate",
        device="cpu",
        seed=42,
    )
    det_uv = TranADDetector(cfg_uv)
    det_uv.fit_and_calibrate(x_train, x_val, feats)
    assert det_uv.training_history["input_dim"] == 1
    inf_uv = det_uv.score_and_predict(x_test, feats)
    assert np.mean(inf_uv.scores[8:18]) > np.mean(inf_uv.scores[:6])


def test_end_to_end_m7_benchmark_and_m6_preservation(tmp_path: Path) -> None:
    skab_root = tmp_path / "skab"
    tsb_root = tmp_path / "tsb"
    _write_synthetic_skab_tree(skab_root)
    _write_synthetic_tsb_tree(tsb_root)

    m6_out = tmp_path / "reports_m6" / "m6_ref"
    rc6 = m6_main(
        [
            "--run-id",
            "m6_ref",
            "--device",
            "cpu",
            "--window-size",
            "8",
            "--hidden-channels",
            "8",
            "--latent-dim",
            "4",
            "--dilations",
            "1",
            "2",
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
            str(m6_out),
        ]
    )
    assert rc6 == 0
    m6_bytes_before = (m6_out / "m6_report.json").read_bytes()

    cfg_path = tmp_path / "m7_smoke.json"
    cfg_path.write_text(
        json.dumps(
            {
                "seed": 42,
                "device": "cpu",
                "window_size": 8,
                "dim_feedforward": 12,
                "epochs": 4,
                "batch_size": 32,
                "learning_rate": 0.004,
                "dropout": 0.0,
                "calibration_quantile": 0.98,
                "skab_channel_mode": "multivariate",
                "tsb_channel_mode": "independent_univariate",
                "tsb_max_per_domain": 1,
            }
        ),
        encoding="utf-8",
    )
    assert TranADConfig.from_json_file(cfg_path).window_size == 8

    m7_out = tmp_path / "reports_m7"
    rc7 = m7_main(
        [
            "--config",
            str(cfg_path),
            "--run-id",
            "m7_smoke_test",
            "--skab-dir",
            str(skab_root),
            "--tsb-dir",
            str(tsb_root),
            "--reuse-m6-dir",
            str(m6_out),
            "--output-dir",
            str(m7_out),
        ]
    )
    assert rc7 == 0
    assert (m6_out / "m6_report.json").read_bytes() == m6_bytes_before

    rep = json.loads((m7_out / "m7_report.json").read_text(encoding="utf-8"))
    assert rep["reproducibility"]["baseline_source"] == "reused_verified_m6_bundle:m6_ref"
    assert {r["model"] for r in rep["comparison_summary"]} == {
        "static_threshold",
        "isolation_forest",
        "half_space_trees",
        "lstm_autoencoder",
        "tcn_autoencoder",
        "tranad",
    }
    with (m7_out / "per_experiment_metrics.csv").open(encoding="utf-8") as handle:
        # (3 SKAB + 2 TSB) * 6 models = 30 rows
        assert len(list(csv.DictReader(handle))) == 30
