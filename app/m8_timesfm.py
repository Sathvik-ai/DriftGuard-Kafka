"""Google TimesFM 3.0 zero-shot time-series foundation model adapter (M8).

Official reference:
  - Repository: https://github.com/google-research/timesfm
  - Checkpoint: `google/timesfm-3.0-pytorch` (~330.7M parameters, ~1.23 GiB safetensors)
  - Code license: `Apache-2.0`
  - Weights license: `timesfm-non-commercial-license-v1.0` (non-commercial research
    use only; commercial/production use requires Google Cloud BigQuery ML / Vertex AI)

Capabilities leveraged:
  - Zero-shot native multivariate `(D, context_length)` and univariate `(context_length,)`
    forecasting via `timesfm.TimesFM3Forecaster.predict_batch(..., return_quantiles=True)`.
  - Continuous decile quantiles (`0.1` through `0.9`), supporting both normalized
    forecast residual scoring and `[0.1, 0.9]` prediction interval violation scoring.
  - Apple Silicon `mps` acceleration with automatic `cpu` fallback.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.m5_lstm_autoencoder import DeviceInfo, select_compute_device
from app.m8_tsfm_base import (
    ForecastBackend,
    ForecastBatchOutput,
    TSFMDetectorConfig,
    ZeroShotForecastDetector,
    resolve_hf_cache_dir,
)

TIMESFM3_DEFAULT_MODEL_ID = "google/timesfm-3.0-pytorch"
TIMESFM3_LICENSE = "Code: Apache-2.0 | Weights: timesfm-non-commercial-license-v1.0"

_FORECASTER_CACHE: dict[tuple[str, str, int, str, bool], Any] = {}


def is_timesfm3_installed() -> bool:
    """Return True if `timesfm>=3.0` (`TimesFM3Forecaster`) is importable."""
    try:
        from timesfm import TimesFM3Forecaster  # noqa: F401

        return True
    except ImportError:
        return False


class TimesFM3Backend:
    """Adapter around official `timesfm.TimesFM3Forecaster` (`timesfm==3.0.2`)."""

    def __init__(
        self,
        model_id: str = TIMESFM3_DEFAULT_MODEL_ID,
        device: str = "auto",
        per_core_batch_size: int = 64,
        cache_dir: str = "models/hf_cache",
        local_files_only: bool = False,
    ) -> None:
        """Load or reuse cached `TimesFM3Forecaster` with MPS/CPU probe fallback."""
        self._model_id = model_id
        self._device_info = select_compute_device(device)
        self._per_core_batch_size = per_core_batch_size
        self._cache_dir = resolve_hf_cache_dir(cache_dir, model_id)
        self._local_files_only = local_files_only
        self._forecaster = self._load_forecaster()

    def _load_forecaster(self) -> Any:
        import timesfm

        dev = self._device_info.selected
        key = (
            self._model_id,
            dev,
            self._per_core_batch_size,
            self._cache_dir,
            self._local_files_only,
        )
        if key in _FORECASTER_CACHE:
            return _FORECASTER_CACHE[key]

        os.environ.setdefault("HF_HOME", self._cache_dir)
        if self._local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        Path(self._cache_dir).mkdir(parents=True, exist_ok=True)
        try:
            forecaster = timesfm.TimesFM3Forecaster.from_pretrained(
                self._model_id,
                device=dev,
                per_core_batch_size=self._per_core_batch_size,
                cache_dir=self._cache_dir,
                local_files_only=self._local_files_only,
            )
            # Probe a tiny 1D forecast to confirm all ops work on `dev` (e.g. MPS).
            probe_ctx = np.linspace(0.0, 1.0, 32, dtype=np.float32)
            _ = forecaster.predict(probe_ctx, horizon=4, return_quantiles=True)
        except Exception:
            if dev != "cpu":
                self._device_info = select_compute_device("cpu")
                cpu_key = (
                    self._model_id,
                    "cpu",
                    self._per_core_batch_size,
                    self._cache_dir,
                    self._local_files_only,
                )
                if cpu_key in _FORECASTER_CACHE:
                    return _FORECASTER_CACHE[cpu_key]
                forecaster = timesfm.TimesFM3Forecaster.from_pretrained(
                    self._model_id,
                    device="cpu",
                    per_core_batch_size=self._per_core_batch_size,
                    cache_dir=self._cache_dir,
                    local_files_only=self._local_files_only,
                )
                _FORECASTER_CACHE[cpu_key] = forecaster
                return forecaster
            raise

        _FORECASTER_CACHE[key] = forecaster
        return forecaster

    @property
    def model_id(self) -> str:
        """Return Hugging Face checkpoint ID."""
        return self._model_id

    @property
    def license_name(self) -> str:
        """Return code + non-commercial weights license string."""
        return TIMESFM3_LICENSE

    @property
    def supports_multivariate(self) -> bool:
        """Return True (`TimesFM 3.0` natively supports variate attention)."""
        return True

    @property
    def device_info(self) -> DeviceInfo:
        """Return compute accelerator diagnostics."""
        return self._device_info

    def forecast_multivariate_blocks(
        self,
        contexts_bdl: NDArray[np.float32],
        horizon: int,
        batch_size: int,
    ) -> ForecastBatchOutput:
        """Forecast `horizon` steps and `[0.1, 0.9]` deciles via `TimesFM3Forecaster`."""
        del batch_size  # controlled by `per_core_batch_size` on `TimesFM3Forecaster`
        n_blocks, n_vars, _ctx_len = contexts_bdl.shape
        if n_vars == 1:
            ctx_list = [
                np.ascontiguousarray(contexts_bdl[i, 0, :], dtype=np.float32)
                for i in range(n_blocks)
            ]
            outputs = list(
                self._forecaster.predict_batch(
                    contexts=ctx_list,
                    horizon=horizon,
                    return_quantiles=True,
                    use_znorm=True,
                    use_symmetric_averaging=False,
                )
            )
            point = np.stack(
                [np.asarray(o.forecast, dtype=np.float32)[None, :] for o in outputs],
                axis=0,
            )
            q_low = np.stack(
                [np.asarray(o.quantiles[:, 0], dtype=np.float32)[None, :] for o in outputs],
                axis=0,
            )
            q_high = np.stack(
                [np.asarray(o.quantiles[:, -1], dtype=np.float32)[None, :] for o in outputs],
                axis=0,
            )
            return ForecastBatchOutput(point_forecast=point, q_low=q_low, q_high=q_high)

        ctx_mv = [np.ascontiguousarray(contexts_bdl[i], dtype=np.float32) for i in range(n_blocks)]
        outputs_mv = list(
            self._forecaster.predict_batch(
                contexts=ctx_mv,
                horizon=horizon,
                return_quantiles=True,
                use_znorm=True,
                use_symmetric_averaging=False,
            )
        )
        point_mv = np.stack([np.asarray(o.forecast, dtype=np.float32) for o in outputs_mv], axis=0)
        ql_mv = np.stack(
            [np.asarray(o.quantiles[:, :, 0], dtype=np.float32) for o in outputs_mv],
            axis=0,
        )
        qh_mv = np.stack(
            [np.asarray(o.quantiles[:, :, -1], dtype=np.float32) for o in outputs_mv],
            axis=0,
        )
        return ForecastBatchOutput(point_forecast=point_mv, q_low=ql_mv, q_high=qh_mv)


class TimesFM3Detector(ZeroShotForecastDetector):
    """Zero-shot Google TimesFM 3.0 anomaly detector."""

    def __init__(
        self,
        config: TSFMDetectorConfig | None = None,
        backend: ForecastBackend | None = None,
    ) -> None:
        """Initialise TimesFM 3.0 detector with `TSFMDetectorConfig`."""
        cfg = config or TSFMDetectorConfig(model_id=TIMESFM3_DEFAULT_MODEL_ID)
        super().__init__(config=cfg, backend=backend)
        self._resolved_backend: ForecastBackend | None = backend

    @property
    def name(self) -> str:
        """Return detector name distinguishing residual vs interval scoring."""
        if self._cfg.score_mode == "interval_violation":
            return "timesfm3_interval"
        return "timesfm3_zero_shot"

    def _resolve_backend(self) -> ForecastBackend:
        if self._resolved_backend is None:
            self._resolved_backend = TimesFM3Backend(
                model_id=self._cfg.model_id,
                device=self._cfg.device,
                per_core_batch_size=self._cfg.batch_size,
                cache_dir=self._cfg.cache_dir,
                local_files_only=self._cfg.local_files_only,
            )
        return self._resolved_backend
