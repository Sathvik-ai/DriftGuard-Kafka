"""Unit and smoke tests for DriftGuard-Kafka M8 TSFM adapters (Chronos-2 & TimesFM 3.0)."""

from __future__ import annotations

import csv
import inspect
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from app.m7_benchmark import main as m7_main
from app.m8_benchmark import DeterministicPersistenceBackend
from app.m8_benchmark import main as m8_main
from app.m8_chronos2 import (
    _PIPELINE_CACHE,
    Chronos2Backend,
    Chronos2Detector,
    is_chronos2_installed,
)
from app.m8_timesfm import (
    _FORECASTER_CACHE,
    TimesFM3Backend,
    TimesFM3Detector,
    is_timesfm3_installed,
)
from app.m8_tsfm_base import (
    TSFMDetectorConfig,
    build_causal_context_blocks,
    resolve_hf_cache_dir,
)
from tests.test_m4_benchmark import _write_synthetic_skab_tree, _write_synthetic_tsb_tree


def test_causal_context_blocks_strictly_prevent_future_leakage() -> None:
    rng = np.random.default_rng(101)
    history = rng.normal(size=(40, 3))
    target_a = rng.normal(size=(50, 3))
    target_b = target_a.copy()
    # Mutate future target values from index 16 onward by +9999.0!
    target_b[16:, :] += 9999.0

    ctx_a, spans_a = build_causal_context_blocks(
        history, target_a, context_length=24, prediction_length=16
    )
    ctx_b, spans_b = build_causal_context_blocks(
        history, target_b, context_length=24, prediction_length=16
    )
    assert spans_a == spans_b == [(0, 16), (16, 32), (32, 48), (48, 50)]
    # Block 0 predicts [0, 16) -> uses only history[-24:]
    assert np.array_equal(ctx_a[0], ctx_b[0])
    # Block 1 predicts [16, 32) -> uses history[-8:] + target[:16], so mutating target[16:]
    # MUST NOT alter ctx_b[1] by even a single bit!
    assert np.array_equal(ctx_a[1], ctx_b[1])
    # Block 2 predicts [32, 48) -> sees target[16:32] in past context, so it reflects target_b!
    assert not np.array_equal(ctx_a[2], ctx_b[2])


def test_chronos2_and_timesfm3_adapters_residual_and_interval_scoring() -> None:
    rng = np.random.default_rng(77)
    x_normal = np.column_stack(
        [
            10.0 + 0.05 * rng.normal(size=180),
            20.0 + 0.05 * rng.normal(size=180),
            np.full(180, 5.0),  # constant zero-variance channel
        ]
    )
    x_train = x_normal[:100]
    x_val = x_normal[100:140]
    x_test = x_normal[140:].copy()
    x_test[12:22, 0] += 8.5  # strong anomaly violating both residual & [0.1, 0.9] interval
    feats = ("ch0", "ch1", "const_ch")

    mock_backend = DeterministicPersistenceBackend("mock/tsfm")

    for det_cls, expected_res_name, expected_iv_name in (
        (Chronos2Detector, "chronos2_zero_shot", "chronos2_interval"),
        (TimesFM3Detector, "timesfm3_zero_shot", "timesfm3_interval"),
    ):
        cfg_res = TSFMDetectorConfig(
            model_id="mock/model",
            context_length=24,
            prediction_length=8,
            batch_size=16,
            score_mode="normalized_residual",
            channel_mode="multivariate",
            calibration_quantile=0.95,
        )
        det_res = det_cls(cfg_res, backend=mock_backend)
        assert det_res.name == expected_res_name
        assert list(inspect.signature(det_res.fit_and_calibrate).parameters) == [
            "x_train_normal",
            "x_val_normal",
            "feature_names",
        ]
        det_res.fit_and_calibrate(x_train, x_val, feats)
        inf_res = det_res.score_and_predict(x_test, feats)
        assert int(np.sum(inf_res.predictions[12:22])) >= 8
        assert inf_res.extra_diagnostics["top_error_feature"] == "ch0"
        assert inf_res.extra_diagnostics["forecast_mae"] >= 0.0

        cfg_iv = TSFMDetectorConfig(
            model_id="mock/model",
            context_length=24,
            prediction_length=8,
            batch_size=16,
            score_mode="interval_violation",
            channel_mode="independent_univariate",
            calibration_quantile=0.95,
        )
        det_iv = det_cls(cfg_iv, backend=mock_backend)
        assert det_iv.name == expected_iv_name
        det_iv.fit_and_calibrate(x_train, x_val, feats)
        inf_iv = det_iv.score_and_predict(x_test, feats)
        assert int(np.sum(inf_iv.predictions[12:22])) >= 8
        details = det_iv.last_score_details
        assert details is not None
        assert 0.0 <= details.interval_coverage_80pct <= 1.0

    with pytest.raises(ValueError):
        TSFMDetectorConfig(model_id="x", context_length=4)


def test_chronos2_and_timesfm3_backend_wrappers_and_cache_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. Verify resolve_hf_cache_dir handles both root and hub/ layouts
    hub_sub = tmp_path / "hf_root" / "hub" / "models--amazon--chronos-2"
    hub_sub.mkdir(parents=True)
    direct_sub = tmp_path / "hf_root" / "models--google--timesfm-3.0-pytorch"
    direct_sub.mkdir(parents=True)
    assert resolve_hf_cache_dir(str(tmp_path / "hf_root"), "amazon/chronos-2").endswith("/hub")
    assert resolve_hf_cache_dir(str(tmp_path / "hf_root"), "google/timesfm-3.0-pytorch") == str(
        (tmp_path / "hf_root").resolve()
    )

    # 2. Fake chronos module to unit-test Chronos2Backend without weight download
    class _FakeChronosPipe:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> _FakeChronosPipe:
            del model_id, kwargs
            return cls()

        def predict_quantiles(
            self,
            inputs: np.ndarray,
            prediction_length: int,
            quantile_levels: list[float],
            batch_size: int,
        ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
            del batch_size
            b, d, _ = inputs.shape
            q_list = [
                torch.zeros((d, prediction_length, len(quantile_levels)), dtype=torch.float32)
                for _ in range(b)
            ]
            m_list = [torch.zeros((d, prediction_length), dtype=torch.float32) for _ in range(b)]
            return q_list, m_list

    fake_chronos = types.ModuleType("chronos")
    fake_chronos.Chronos2Pipeline = _FakeChronosPipe  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "chronos", fake_chronos)
    _PIPELINE_CACHE.clear()
    assert is_chronos2_installed() is True

    cb = Chronos2Backend(
        model_id="fake/chronos-2",
        device="cpu",
        cache_dir=str(tmp_path / "c_cache"),
        local_files_only=True,
    )
    assert cb.model_id == "fake/chronos-2"
    assert cb.supports_multivariate is True
    assert "Apache-2.0" in cb.license_name
    out_c = cb.forecast_multivariate_blocks(
        np.ones((2, 3, 16), dtype=np.float32), horizon=4, batch_size=8
    )
    assert out_c.point_forecast.shape == (2, 3, 4)
    assert out_c.q_low.shape == (2, 3, 4)
    assert out_c.q_high.shape == (2, 3, 4)

    # 3. Fake timesfm module to unit-test TimesFM3Backend (both 1D and 2D paths)
    class _FakeOut:
        def __init__(self, forecast: np.ndarray, quantiles: np.ndarray) -> None:
            self.forecast = forecast
            self.quantiles = quantiles

    class _FakeTimesFM3:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> _FakeTimesFM3:
            del model_id, kwargs
            return cls()

        def predict(self, ctx: np.ndarray, horizon: int, return_quantiles: bool = True) -> _FakeOut:
            del ctx, return_quantiles
            return _FakeOut(
                np.zeros((horizon,), dtype=np.float32),
                np.zeros((horizon, 9), dtype=np.float32),
            )

        def predict_batch(
            self,
            contexts: list[np.ndarray],
            horizon: int,
            return_quantiles: bool = True,
            use_znorm: bool = True,
            use_symmetric_averaging: bool = False,
        ) -> list[_FakeOut]:
            del return_quantiles, use_znorm, use_symmetric_averaging
            res: list[_FakeOut] = []
            for c in contexts:
                if c.ndim == 1:
                    res.append(
                        _FakeOut(
                            np.zeros((horizon,), dtype=np.float32),
                            np.zeros((horizon, 9), dtype=np.float32),
                        )
                    )
                else:
                    d = c.shape[0]
                    res.append(
                        _FakeOut(
                            np.zeros((d, horizon), dtype=np.float32),
                            np.zeros((d, horizon, 9), dtype=np.float32),
                        )
                    )
            return res

    fake_tfm = types.ModuleType("timesfm")
    fake_tfm.TimesFM3Forecaster = _FakeTimesFM3  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "timesfm", fake_tfm)
    _FORECASTER_CACHE.clear()
    assert is_timesfm3_installed() is True

    tb = TimesFM3Backend(
        model_id="fake/timesfm-3.0",
        device="cpu",
        per_core_batch_size=8,
        cache_dir=str(tmp_path / "t_cache"),
        local_files_only=True,
    )
    assert tb.model_id == "fake/timesfm-3.0"
    assert tb.supports_multivariate is True
    assert "timesfm-non-commercial-license-v1.0" in tb.license_name
    out_uv = tb.forecast_multivariate_blocks(
        np.ones((2, 1, 16), dtype=np.float32), horizon=5, batch_size=8
    )
    assert out_uv.point_forecast.shape == (2, 1, 5)
    out_mv = tb.forecast_multivariate_blocks(
        np.ones((2, 4, 16), dtype=np.float32), horizon=5, batch_size=8
    )
    assert out_mv.point_forecast.shape == (2, 4, 5)


def test_end_to_end_m8_benchmark_and_m7_preservation(tmp_path: Path) -> None:
    skab_root = tmp_path / "skab"
    tsb_root = tmp_path / "tsb"
    _write_synthetic_skab_tree(skab_root)
    _write_synthetic_tsb_tree(tsb_root)

    m7_out = tmp_path / "reports_m7" / "m7_ref"
    rc7 = m7_main(
        [
            "--run-id",
            "m7_ref",
            "--device",
            "cpu",
            "--window-size",
            "8",
            "--dim-feedforward",
            "8",
            "--epochs",
            "3",
            "--dropout",
            "0.0",
            "--tsb-max-per-domain",
            "1",
            "--skab-dir",
            str(skab_root),
            "--tsb-dir",
            str(tsb_root),
            "--output-dir",
            str(m7_out),
        ]
    )
    assert rc7 == 0
    m7_bytes_before = (m7_out / "m7_report.json").read_bytes()

    m8_out = tmp_path / "reports_m8"
    rc8 = m8_main(
        [
            "--run-id",
            "m8_smoke",
            "--device",
            "cpu",
            "--context-length",
            "24",
            "--prediction-length",
            "12",
            "--batch-size",
            "16",
            "--calibration-quantile",
            "0.98",
            "--tsb-max-per-domain",
            "1",
            "--skab-dir",
            str(skab_root),
            "--tsb-dir",
            str(tsb_root),
            "--reuse-m7-dir",
            str(m7_out),
            "--mock-backends",
            "--output-dir",
            str(m8_out),
        ]
    )
    assert rc8 == 0
    assert (m7_out / "m7_report.json").read_bytes() == m7_bytes_before

    for fname in (
        "m8_report.json",
        "m8_report.md",
        "comparison_summary.csv",
        "per_experiment_metrics.csv",
    ):
        fpath = m8_out / fname
        assert fpath.exists() and fpath.stat().st_size > 0

    rep = json.loads((m8_out / "m8_report.json").read_text(encoding="utf-8"))
    assert rep["reproducibility"]["baseline_source"] == "reused_verified_m7_bundle:m7_ref"
    assert len(rep["foundation_model_compatibility"]) == 2
    assert {r["model"] for r in rep["comparison_summary"]} == {
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
    }
    with (m8_out / "per_experiment_metrics.csv").open(encoding="utf-8") as handle:
        # (3 SKAB + 2 TSB) * 10 detector entries = 50 rows
        assert len(list(csv.DictReader(handle))) == 50
