"""Shared zero-shot forecasting anomaly detection base for DriftGuard-Kafka M8.

Provides a strictly causal, leakage-free bridge turning zero-shot probabilistic
time-series foundation models (Amazon Chronos-2 and Google TimesFM 3.0) into
anomaly detectors compatible with `BenchmarkDetector`:

1. **Zero-shot**: Model weights are never updated or fine-tuned on benchmark data.
2. **Causal blocked rolling forecasts**: To forecast target steps `[t_0, t_0 + H)`,
   only observations strictly prior to `t_0` (`<= t_0 - 1`) enter the context
   window—future observations never leak into the forecast context.
3. **Training-only residual normalization**: Per-feature standard deviations
   `sigma_d` are fit strictly on `x_train_normal` (`TrainingOnlyScaler`).
4. **Dual anomaly score modes**:
   - `normalized_residual`: `((y - y_hat) / sigma_d) ** 2`
   - `interval_violation`: Normalized violation outside `[q_low, q_high]`
     (`0.1`–`0.9` prediction interval) with smooth interior residual tie-breaking.
5. **Validation-only threshold calibration**: Calibrated strictly on normal
   validation observations (`x_val_normal`); test data stays untouched until
   evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from app.m4_models import BenchmarkDetector, FloatArray, _validate_2d_features
from app.m5_lstm_autoencoder import ChannelMode, DeviceInfo, TrainingOnlyScaler

TSFMScoreMode = Literal["normalized_residual", "interval_violation"]


def resolve_hf_cache_dir(cache_dir: str, model_id: str) -> str:
    """Resolve Hugging Face cache root whether weights sit under `cache_dir` or `cache_dir/hub`."""
    base = Path(cache_dir).resolve()
    repo_folder = f"models--{model_id.replace('/', '--')}"
    if (base / repo_folder).exists():
        return str(base)
    if (base / "hub" / repo_folder).exists():
        return str((base / "hub").resolve())
    return str(base)


@dataclass(frozen=True)
class ForecastBatchOutput:
    """Point and `[q_low, q_high]` interval forecasts of shape `(B, D, H)`."""

    point_forecast: NDArray[np.float32]
    q_low: NDArray[np.float32]
    q_high: NDArray[np.float32]


class ForecastBackend(Protocol):
    """Protocol implemented by concrete zero-shot foundation model backends."""

    @property
    def model_id(self) -> str:
        """Return Hugging Face repository or backend identifier."""

    @property
    def license_name(self) -> str:
        """Return upstream license identifier."""

    @property
    def supports_multivariate(self) -> bool:
        """Return True if backend natively forecasts multivariate `(D, C)` inputs."""

    @property
    def device_info(self) -> DeviceInfo:
        """Return hardware accelerator diagnostics."""

    def forecast_multivariate_blocks(
        self,
        contexts_bdl: NDArray[np.float32],
        horizon: int,
        batch_size: int,
    ) -> ForecastBatchOutput:
        """Forecast `horizon` steps from contexts `(B, D, context_len)`."""


@dataclass(frozen=True)
class TSFMDetectorConfig:
    """Configuration shared by zero-shot TSFM anomaly detectors."""

    model_id: str
    context_length: int = 128
    prediction_length: int = 32
    batch_size: int = 64
    score_mode: TSFMScoreMode = "normalized_residual"
    channel_mode: ChannelMode = "multivariate"
    calibration_quantile: float = 0.99
    max_active_univariate_channels: int = 32
    device: str = "auto"
    cache_dir: str = "models/hf_cache"
    local_files_only: bool = False

    def __post_init__(self) -> None:
        """Validate TSFM detector configuration."""
        if self.context_length < 8:
            raise ValueError("context_length must be >= 8")
        if self.prediction_length < 1:
            raise ValueError("prediction_length must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.score_mode not in ("normalized_residual", "interval_violation"):
            raise ValueError(f"unsupported score_mode: {self.score_mode!r}")
        if self.channel_mode not in ("multivariate", "independent_univariate"):
            raise ValueError(f"unsupported channel_mode: {self.channel_mode!r}")
        if not 0.0 < self.calibration_quantile < 1.0:
            raise ValueError("calibration_quantile must be strictly in (0, 1)")


@dataclass(frozen=True)
class TSFMScoreDetails:
    """Forecast outputs, forecasting error metrics, and dual anomaly scores."""

    overall_scores: FloatArray
    per_feature_scores: FloatArray
    residual_scores: FloatArray
    interval_violation_scores: FloatArray
    point_forecasts: FloatArray
    q_low: FloatArray
    q_high: FloatArray
    forecast_mae: float
    forecast_rmse: float
    normalized_mae: float
    interval_coverage_80pct: float
    feature_names: tuple[str, ...]


def build_causal_context_blocks(
    history_prefix: FloatArray,
    target_series: FloatArray,
    context_length: int,
    prediction_length: int,
) -> tuple[NDArray[np.float32], list[tuple[int, int]]]:
    """Construct strictly causal `(n_blocks, D, context_length)` context tensors.

    For block `[t_0, t_1)` in `target_series`, the context slices strictly up to
    `t_0` (`history_prefix` + `target_series[:t_0]`), guaranteeing zero future
    observations from `[t_0, N)` ever leak into the context window.
    """
    hist = _validate_2d_features(history_prefix, "history_prefix")
    tgt = _validate_2d_features(target_series, "target_series")
    if hist.shape[1] != tgt.shape[1]:
        raise ValueError("history_prefix and target_series feature dimensions must match")

    n_target, n_feats = tgt.shape
    prefix_tail = (
        hist[-context_length:]
        if hist.shape[0] >= context_length
        else np.vstack([np.repeat(hist[:1], context_length - hist.shape[0], axis=0), hist])
    )
    combined = np.vstack([prefix_tail, tgt]).astype(np.float32, copy=False)
    offset = int(prefix_tail.shape[0])

    contexts: list[NDArray[np.float32]] = []
    spans: list[tuple[int, int]] = []
    for t0 in range(0, n_target, prediction_length):
        t1 = min(n_target, t0 + prediction_length)
        # Slice [offset + t0 - context_length : offset + t0] -> ends at target[t0 - 1]!
        ctx_td = combined[offset + t0 - context_length : offset + t0]
        contexts.append(np.ascontiguousarray(ctx_td.T, dtype=np.float32))
        spans.append((t0, t1))

    return np.stack(contexts, axis=0), spans


class ZeroShotForecastDetector(BenchmarkDetector, ABC):
    """Abstract base detector converting zero-shot TSFM forecasts into anomaly scores."""

    def __init__(
        self,
        config: TSFMDetectorConfig,
        backend: ForecastBackend | None = None,
    ) -> None:
        """Initialise detector with configuration and optional injected backend."""
        super().__init__(calibration_quantile=config.calibration_quantile)
        self._cfg = config
        self._injected_backend = backend
        self._scaler: TrainingOnlyScaler | None = None
        self._train_tail: FloatArray | None = None
        self._pre_test_tail: FloatArray | None = None
        self._last_details: TSFMScoreDetails | None = None

    @property
    def config(self) -> TSFMDetectorConfig:
        """Return immutable configuration."""
        return self._cfg

    @property
    def last_score_details(self) -> TSFMScoreDetails | None:
        """Return forecast arrays, error metrics, and scores from last test run."""
        return self._last_details

    @abstractmethod
    def _resolve_backend(self) -> ForecastBackend:
        """Return active `ForecastBackend` instance."""

    def get_parameters(self) -> dict[str, Any]:
        """Return JSON-serialisable metadata and zero-shot hyperparameters."""
        backend = self._resolve_backend()
        params = asdict(self._cfg)
        params["zero_shot"] = True
        params["fine_tuned"] = False
        params["license"] = backend.license_name
        params["supports_multivariate"] = backend.supports_multivariate
        params["selected_device"] = backend.device_info.selected
        params["mps_available"] = backend.device_info.mps_available
        return params

    def _fit_normal(self, x_train_normal: FloatArray, feature_names: tuple[str, ...]) -> None:
        del feature_names
        # Zero-shot: no gradient training; record normal training scale & historical tail only!
        self._scaler = TrainingOnlyScaler().fit(x_train_normal)
        ctx_len = self._cfg.context_length
        self._train_tail = np.array(x_train_normal[-ctx_len:], dtype=np.float64, copy=True)
        self._pre_test_tail = self._train_tail.copy()

    def forecast_and_score(
        self,
        history_prefix: FloatArray,
        target_matrix: FloatArray,
        feature_names: Sequence[str] | None = None,
    ) -> TSFMScoreDetails:
        """Run zero-shot causal forecasts on `target_matrix` and compute anomaly scores."""
        if self._scaler is None:
            raise RuntimeError(f"{self.name} must call fit_and_calibrate before scoring")
        tgt = _validate_2d_features(target_matrix, "target_matrix")
        names = tuple(feature_names) if feature_names is not None else self._feature_names
        if len(names) != tgt.shape[1]:
            raise ValueError("feature_names length mismatch")

        backend = self._resolve_backend()
        cfg = self._cfg
        n_rows, n_feats = tgt.shape
        scale = self._scaler.scale_
        mean = self._scaler.mean_

        point_pred = np.empty((n_rows, n_feats), dtype=np.float64)
        ql_pred = np.empty((n_rows, n_feats), dtype=np.float64)
        qh_pred = np.empty((n_rows, n_feats), dtype=np.float64)

        use_mv = cfg.channel_mode == "multivariate" and backend.supports_multivariate
        if use_mv:
            contexts_bdl, spans = build_causal_context_blocks(
                history_prefix, tgt, cfg.context_length, cfg.prediction_length
            )
            fb_out = backend.forecast_multivariate_blocks(
                contexts_bdl=contexts_bdl,
                horizon=cfg.prediction_length,
                batch_size=cfg.batch_size,
            )
            for idx, (t0, t1) in enumerate(spans):
                step_len = t1 - t0
                point_pred[t0:t1, :] = fb_out.point_forecast[idx, :, :step_len].T
                ql_pred[t0:t1, :] = fb_out.q_low[idx, :, :step_len].T
                qh_pred[t0:t1, :] = fb_out.q_high[idx, :, :step_len].T
        else:
            # Independent univariate forecasting per active channel; constant inactive
            # channels predict their exact training constant `mean[ch]`.
            active_mask = self._scaler.active_channel_mask
            inactive_idx = np.flatnonzero(~active_mask)
            for ch_i in inactive_idx:
                c_val = float(mean[int(ch_i)])
                point_pred[:, int(ch_i)] = c_val
                ql_pred[:, int(ch_i)] = c_val
                qh_pred[:, int(ch_i)] = c_val

            active_idx = np.flatnonzero(active_mask)
            if active_idx.size > cfg.max_active_univariate_channels:
                # Deterministic variance-ranked cap for 200+-column datasets (e.g. OPPORTUNITY)
                variances = np.square(scale[active_idx])
                top_order = np.argsort(variances)[::-1][: cfg.max_active_univariate_channels]
                skipped_idx = active_idx[
                    np.argsort(variances)[::-1][cfg.max_active_univariate_channels :]
                ]
                for ch_s in skipped_idx:
                    # Persistence causal baseline from preceding block boundary for capped channels
                    c_val = float(mean[int(ch_s)])
                    point_pred[:, int(ch_s)] = c_val
                    ql_pred[:, int(ch_s)] = c_val - 1.28 * float(scale[int(ch_s)])
                    qh_pred[:, int(ch_s)] = c_val + 1.28 * float(scale[int(ch_s)])
                active_idx = np.sort(active_idx[top_order])

            if active_idx.size > 0:
                sub_hist = history_prefix[:, active_idx]
                sub_tgt = tgt[:, active_idx]
                contexts_bdl, spans = build_causal_context_blocks(
                    sub_hist, sub_tgt, cfg.context_length, cfg.prediction_length
                )
                n_blocks, n_act, ctx_l = contexts_bdl.shape
                # Fold (n_blocks, n_act, ctx_l) -> (n_blocks * n_act, 1, ctx_l)
                folded_ctx = contexts_bdl.reshape(n_blocks * n_act, 1, ctx_l)
                fb_uv = backend.forecast_multivariate_blocks(
                    contexts_bdl=folded_ctx,
                    horizon=cfg.prediction_length,
                    batch_size=cfg.batch_size,
                )
                pt_b = fb_uv.point_forecast.reshape(n_blocks, n_act, cfg.prediction_length)
                ql_b = fb_uv.q_low.reshape(n_blocks, n_act, cfg.prediction_length)
                qh_b = fb_uv.q_high.reshape(n_blocks, n_act, cfg.prediction_length)
                for idx, (t0, t1) in enumerate(spans):
                    step_len = t1 - t0
                    point_pred[t0:t1, active_idx] = pt_b[idx, :, :step_len].T
                    ql_pred[t0:t1, active_idx] = ql_b[idx, :, :step_len].T
                    qh_pred[t0:t1, active_idx] = qh_b[idx, :, :step_len].T

        abs_err = np.abs(tgt - point_pred)
        sq_err = np.square(tgt - point_pred)
        norm_abs_err = abs_err / scale[None, :]
        per_feat_residual = np.square(norm_abs_err)

        outside = np.maximum(0.0, np.maximum(ql_pred - tgt, tgt - qh_pred))
        norm_outside = outside / scale[None, :]
        per_feat_interval = np.square(norm_outside) + 1e-3 * norm_abs_err

        res_overall = np.asarray(np.mean(per_feat_residual, axis=1), dtype=np.float64)
        iv_overall = np.asarray(np.mean(per_feat_interval, axis=1), dtype=np.float64)

        if cfg.score_mode == "interval_violation":
            chosen_per_feat = per_feat_interval
            chosen_overall = iv_overall
        else:
            chosen_per_feat = per_feat_residual
            chosen_overall = res_overall

        inside_80 = (tgt >= ql_pred) & (tgt <= qh_pred)
        return TSFMScoreDetails(
            overall_scores=chosen_overall,
            per_feature_scores=chosen_per_feat,
            residual_scores=res_overall,
            interval_violation_scores=iv_overall,
            point_forecasts=point_pred,
            q_low=ql_pred,
            q_high=qh_pred,
            forecast_mae=round(float(np.mean(abs_err)), 6),
            forecast_rmse=round(float(np.sqrt(np.mean(sq_err))), 6),
            normalized_mae=round(float(np.mean(norm_abs_err)), 6),
            interval_coverage_80pct=round(float(np.mean(inside_80)), 4),
            feature_names=names,
        )

    def _score_validation(
        self, x_val_normal: FloatArray, feature_names: tuple[str, ...]
    ) -> FloatArray:
        if self._train_tail is None:
            raise RuntimeError("training context tail missing")
        details = self.forecast_and_score(self._train_tail, x_val_normal, feature_names)
        combined_normal = np.vstack([self._train_tail, x_val_normal])
        self._pre_test_tail = np.array(
            combined_normal[-self._cfg.context_length :], dtype=np.float64, copy=True
        )
        return details.overall_scores

    def _score_test(
        self, x_test: FloatArray, feature_names: tuple[str, ...]
    ) -> tuple[FloatArray, dict[str, Any]]:
        if self._pre_test_tail is None:
            raise RuntimeError("pre-test context tail missing")
        details = self.forecast_and_score(self._pre_test_tail, x_test, feature_names)
        self._last_details = details
        feat_means = np.mean(details.per_feature_scores, axis=0)
        top_idx = int(np.argmax(feat_means))
        backend = self._resolve_backend()
        diagnostics: dict[str, Any] = {
            "model_id": backend.model_id,
            "license": backend.license_name,
            "channel_mode": self._cfg.channel_mode,
            "score_mode": self._cfg.score_mode,
            "selected_device": backend.device_info.selected,
            "forecast_mae": details.forecast_mae,
            "forecast_rmse": details.forecast_rmse,
            "normalized_mae": details.normalized_mae,
            "interval_coverage_80pct": details.interval_coverage_80pct,
            "top_error_feature": feature_names[top_idx],
        }
        return details.overall_scores, diagnostics
