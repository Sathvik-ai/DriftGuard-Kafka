"""Amazon Chronos-2 zero-shot time-series foundation model adapter (M8).

Official reference:
  - Repository: https://github.com/amazon-science/chronos-forecasting
  - Checkpoint: `amazon/chronos-2` (~119.5M parameters, ~456 MiB safetensors)
  - License: `Apache-2.0`

Capabilities leveraged:
  - Zero-shot multivariate `(B, D, context_length)` and univariate `(B, 1, context_length)`
    forecasting via `Chronos2Pipeline.predict_quantiles`.
  - Native probabilistic quantile intervals `[0.1, 0.5, 0.9]`, supporting both
    normalized forecast residual scoring and prediction interval violation scoring.
  - Apple Silicon `mps` acceleration with automatic `cpu` fallback.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from app.m5_lstm_autoencoder import DeviceInfo, select_compute_device
from app.m8_tsfm_base import (
    ForecastBackend,
    ForecastBatchOutput,
    TSFMDetectorConfig,
    ZeroShotForecastDetector,
    resolve_hf_cache_dir,
)

CHRONOS2_DEFAULT_MODEL_ID = "amazon/chronos-2"
CHRONOS2_LICENSE = "Apache-2.0"

_PIPELINE_CACHE: dict[tuple[str, str, str, bool], Any] = {}


def is_chronos2_installed() -> bool:
    """Return True if `chronos-forecasting>=2.0` (`Chronos2Pipeline`) is importable."""
    try:
        from chronos import Chronos2Pipeline  # noqa: F401

        return True
    except ImportError:
        return False


class Chronos2Backend:
    """Adapter around official `chronos.Chronos2Pipeline` for zero-shot forecasting."""

    def __init__(
        self,
        model_id: str = CHRONOS2_DEFAULT_MODEL_ID,
        device: str = "auto",
        cache_dir: str = "models/hf_cache",
        local_files_only: bool = False,
    ) -> None:
        """Load or reuse cached `Chronos2Pipeline` on `mps` / `cuda` / `cpu`."""
        self._model_id = model_id
        self._device_info = select_compute_device(device)
        self._cache_dir = resolve_hf_cache_dir(cache_dir, model_id)
        self._local_files_only = local_files_only
        self._pipeline = self._load_pipeline()

    def _load_pipeline(self) -> Any:
        from chronos import Chronos2Pipeline

        key = (
            self._model_id,
            self._device_info.selected,
            self._cache_dir,
            self._local_files_only,
        )
        if key in _PIPELINE_CACHE:
            return _PIPELINE_CACHE[key]

        os.environ.setdefault("HF_HOME", self._cache_dir)
        if self._local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        Path(self._cache_dir).mkdir(parents=True, exist_ok=True)
        dev = self._device_info.selected
        try:
            pipe = Chronos2Pipeline.from_pretrained(
                self._model_id,
                device_map=dev,
                dtype=torch.float32,
                cache_dir=self._cache_dir,
                local_files_only=self._local_files_only,
            )
        except Exception:
            if dev != "cpu":
                self._device_info = select_compute_device("cpu")
                pipe = Chronos2Pipeline.from_pretrained(
                    self._model_id,
                    device_map="cpu",
                    dtype=torch.float32,
                    cache_dir=self._cache_dir,
                    local_files_only=self._local_files_only,
                )
            else:
                raise
        _PIPELINE_CACHE[key] = pipe
        return pipe

    @property
    def model_id(self) -> str:
        """Return Hugging Face model identifier."""
        return self._model_id

    @property
    def license_name(self) -> str:
        """Return upstream license."""
        return CHRONOS2_LICENSE

    @property
    def supports_multivariate(self) -> bool:
        """Return True (`Chronos-2` natively models multivariate group attention)."""
        return True

    @property
    def device_info(self) -> DeviceInfo:
        """Return compute accelerator metadata."""
        return self._device_info

    def forecast_multivariate_blocks(
        self,
        contexts_bdl: NDArray[np.float32],
        horizon: int,
        batch_size: int,
    ) -> ForecastBatchOutput:
        """Forecast `horizon` steps and `[0.1, 0.5, 0.9]` quantiles via `Chronos2Pipeline`."""
        q_list, mean_list = self._pipeline.predict_quantiles(
            inputs=np.ascontiguousarray(contexts_bdl, dtype=np.float32),
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            batch_size=batch_size,
        )
        # Each item in q_list has shape (D, horizon, 3); mean_list has shape (D, horizon).
        point = np.stack([m.detach().cpu().numpy().astype(np.float32) for m in mean_list], axis=0)
        q_arr = np.stack([q.detach().cpu().numpy().astype(np.float32) for q in q_list], axis=0)
        return ForecastBatchOutput(
            point_forecast=point,
            q_low=q_arr[:, :, :, 0],
            q_high=q_arr[:, :, :, 2],
        )


class Chronos2Detector(ZeroShotForecastDetector):
    """Zero-shot Amazon Chronos-2 anomaly detector."""

    def __init__(
        self,
        config: TSFMDetectorConfig | None = None,
        backend: ForecastBackend | None = None,
    ) -> None:
        """Initialise Chronos-2 detector with `TSFMDetectorConfig`."""
        cfg = config or TSFMDetectorConfig(model_id=CHRONOS2_DEFAULT_MODEL_ID)
        super().__init__(config=cfg, backend=backend)
        self._resolved_backend: ForecastBackend | None = backend

    @property
    def name(self) -> str:
        """Return detector name distinguishing residual vs interval scoring."""
        if self._cfg.score_mode == "interval_violation":
            return "chronos2_interval"
        return "chronos2_zero_shot"

    def _resolve_backend(self) -> ForecastBackend:
        if self._resolved_backend is None:
            self._resolved_backend = Chronos2Backend(
                model_id=self._cfg.model_id,
                device=self._cfg.device,
                cache_dir=self._cfg.cache_dir,
                local_files_only=self._cfg.local_files_only,
            )
        return self._resolved_backend
