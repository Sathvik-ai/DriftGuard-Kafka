"""Fair input protocol tracks (Track A: Multivariate, Track B: Independent-Channel) for M9.1.

Resolves the M9 input-protocol asymmetry on TSB-AD-M (which is an official multivariate
dataset with 2–248 features) by establishing two strictly equivalent evaluation tracks:

- **Track A (`multivariate`)**:
  Every compatible detector receives the exact same multivariate `(N, D)` feature matrix
  and multivariate temporal windows. For TSFMs (`Chronos-2`, `TimesFM 3.0`), series with
  `D <= max_multivariate_group_variates` run native full-multivariate variate attention
  (`multivariate_native`), while high-dimensional series (`D > 32`, e.g., `MSL=55`,
  `SWaT=66`, `OPPORTUNITY=248`) run deterministic variate-group chunks
  (`multivariate_chunked_groups`) with adaptive long-horizon blocks (`H=256` on long
  series) so long series are evaluated without silent exclusion.

- **Track B (`independent_channel`)**:
  Every eligible model (`static_threshold`, `isolation_forest`, `half_space_trees`,
  `lstm_autoencoder`, `tcn_autoencoder`, `tranad`, `chronos2_*`, `timesfm3_*`) is
  wrapped with `IndependentChannelWrapper`, which applies the **identical**:
    1. Active-channel selection (`std(x_train_normal[:, d]) > 1e-8`);
    2. Per-channel 1D model fitting strictly on `x_train_normal[:, d:d+1]`;
    3. Per-channel score normalization (`(s - med_val) / scale_val`, fit strictly on
       `x_val_normal[:, d:d+1]`);
    4. Uniform mean aggregation across channels (`(N, D) -> (N,)`);
    5. Validation-only threshold calibration on aggregated `x_val_normal` scores.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from app.m4_models import BenchmarkDetector, FloatArray, _validate_2d_features
from app.m5_lstm_autoencoder import TrainingOnlyScaler
from app.m8_tsfm_base import (
    ForecastBackend,
    TSFMScoreDetails,
    ZeroShotForecastDetector,
    build_causal_context_blocks,
)

EvalTrack = Literal["multivariate", "independent_channel"]


@dataclass(frozen=True)
class ModelTrackCapability:
    """Documents exact input-protocol compatibility for each model across Tracks A & B."""

    model: str
    family: str
    track_a_multivariate: bool
    track_a_mode_label: str
    track_b_independent_channel: bool
    track_b_mode_label: str
    notes: str


TRACK_COMPATIBILITY_MATRIX: tuple[ModelTrackCapability, ...] = (
    ModelTrackCapability(
        model="static_threshold",
        family="M4 Classical",
        track_a_multivariate=True,
        track_a_mode_label="multivariate_native",
        track_b_independent_channel=True,
        track_b_mode_label="independent_channel_unified",
        notes=(
            "Native multivariate z-bound + stuck-sensor check in Track A; "
            "unified 1D wrapper in Track B."
        ),
    ),
    ModelTrackCapability(
        model="isolation_forest",
        family="M4 Classical",
        track_a_multivariate=True,
        track_a_mode_label="multivariate_native",
        track_b_independent_channel=True,
        track_b_mode_label="independent_channel_unified",
        notes=(
            "Joint D-dimensional tree splits in Track A; "
            "per-channel 1D forests with identical score normalization in Track B."
        ),
    ),
    ModelTrackCapability(
        model="half_space_trees",
        family="M4 Classical",
        track_a_multivariate=True,
        track_a_mode_label="multivariate_native",
        track_b_independent_channel=True,
        track_b_mode_label="independent_channel_unified",
        notes="Joint D-dimensional streaming HST in Track A; per-channel 1D HST in Track B.",
    ),
    ModelTrackCapability(
        model="lstm_autoencoder",
        family="M5 Deep Learning",
        track_a_multivariate=True,
        track_a_mode_label="multivariate_native",
        track_b_independent_channel=True,
        track_b_mode_label="independent_channel_unified",
        notes=(
            "Joint (B, W, D) LSTM reconstruction in Track A; "
            "identical per-channel 1D LSTM wrapper in Track B."
        ),
    ),
    ModelTrackCapability(
        model="tcn_autoencoder",
        family="M6 Deep Learning",
        track_a_multivariate=True,
        track_a_mode_label="multivariate_native",
        track_b_independent_channel=True,
        track_b_mode_label="independent_channel_unified",
        notes=(
            "Joint (B, W, D) causal dilated TCN in Track A; "
            "identical per-channel 1D TCN wrapper in Track B."
        ),
    ),
    ModelTrackCapability(
        model="tranad",
        family="M7 Deep Learning",
        track_a_multivariate=True,
        track_a_mode_label="multivariate_native",
        track_b_independent_channel=True,
        track_b_mode_label="independent_channel_unified",
        notes=(
            "Two-phase self-conditioning Transformer (d_model=2D, nhead=D) in Track A; "
            "per-channel 1D TranAD in Track B."
        ),
    ),
    ModelTrackCapability(
        model="chronos2_zero_shot",
        family="M8 Foundation",
        track_a_multivariate=True,
        track_a_mode_label="multivariate_native_or_chunked_groups",
        track_b_independent_channel=True,
        track_b_mode_label="independent_channel_unified",
        notes=(
            "Native group-attention multivariate (<=32 vars) or 32-var groups (>32 vars) "
            "in Track A; 1D channel wrapper in Track B."
        ),
    ),
    ModelTrackCapability(
        model="chronos2_interval",
        family="M8 Foundation",
        track_a_multivariate=True,
        track_a_mode_label="multivariate_native_or_chunked_groups",
        track_b_independent_channel=True,
        track_b_mode_label="independent_channel_unified",
        notes="[0.1, 0.9] quantile interval violation from the exact same Chronos-2 forward pass.",
    ),
    ModelTrackCapability(
        model="timesfm3_zero_shot",
        family="M8 Foundation",
        track_a_multivariate=True,
        track_a_mode_label="multivariate_native_or_chunked_groups",
        track_b_independent_channel=True,
        track_b_mode_label="independent_channel_unified",
        notes=(
            "Native variate-attention multivariate (<=32 vars) or 32-var groups (>32 vars) "
            "in Track A; 1D channel wrapper in Track B."
        ),
    ),
    ModelTrackCapability(
        model="timesfm3_interval",
        family="M8 Foundation",
        track_a_multivariate=True,
        track_a_mode_label="multivariate_native_or_chunked_groups",
        track_b_independent_channel=True,
        track_b_mode_label="independent_channel_unified",
        notes="[0.1, 0.9] decile interval violation from the exact same TimesFM 3.0 forward pass.",
    ),
)


def _robust_channel_stats(val_scores_1d: FloatArray) -> tuple[float, float]:
    """Compute validation-only robust location & scale for fair cross-channel aggregation."""
    med = float(np.median(val_scores_1d))
    q75 = float(np.quantile(val_scores_1d, 0.75))
    q25 = float(np.quantile(val_scores_1d, 0.25))
    iqr_scale = (q75 - q25) / 1.3489795
    std_scale = float(np.std(val_scores_1d))
    scale = max(iqr_scale, std_scale, 1e-6)
    return med, scale


class IndependentChannelWrapper(BenchmarkDetector):
    """Unified Track B wrapper applying identical 1D protocol to any `BenchmarkDetector`.

    Every wrapped model receives the exact same:
      - Active-channel detection (`std(x_train_normal[:, d]) > 1e-8`)
      - Deterministic variance-ranked channel cap (`max_active_channels`, if exceeded)
      - Per-channel robust score normalization fit strictly on `x_val_normal`
      - Mean aggregation across all `D` channels
      - Validation-only quantile threshold calibration
    """

    def __init__(
        self,
        model_name: str,
        factory: Callable[[], BenchmarkDetector],
        calibration_quantile: float = 0.99,
        max_active_channels: int = 16,
    ) -> None:
        """Configure Track B wrapper around a 1D detector `factory`."""
        super().__init__(calibration_quantile=calibration_quantile)
        self._model_name = model_name
        self._factory = factory
        self._max_active_channels = max_active_channels
        self._scaler: TrainingOnlyScaler | None = None
        self._channel_detectors: dict[int, BenchmarkDetector] = {}
        self._channel_norm: dict[int, tuple[float, float]] = {}
        self._active_indices: tuple[int, ...] = ()
        self._capped_indices: tuple[int, ...] = ()
        self._inactive_indices: tuple[int, ...] = ()

    @property
    def name(self) -> str:
        """Return canonical model name."""
        return self._model_name

    def get_parameters(self) -> dict[str, Any]:
        """Return Track B wrapper parameters plus inner detector parameters."""
        sample = self._factory()
        return {
            "track": "independent_channel",
            "channel_processing_mode": "independent_channel_unified",
            "max_active_channels": self._max_active_channels,
            "score_normalization": "val_median_robust_scale",
            "channel_aggregation": "uniform_mean",
            "calibration_quantile": self._calibration_quantile,
            "inner_parameters": sample.get_parameters(),
        }

    def _fit_normal(self, x_train_normal: FloatArray, feature_names: tuple[str, ...]) -> None:
        self._scaler = TrainingOnlyScaler().fit(x_train_normal)
        active_mask = self._scaler.active_channel_mask
        inactive = np.flatnonzero(~active_mask)
        active = np.flatnonzero(active_mask)

        capped: NDArray[np.intp] = np.array([], dtype=np.intp)
        if self._max_active_channels > 0 and active.size > self._max_active_channels:
            variances = np.square(self._scaler.scale_[active])
            order = np.argsort(variances)[::-1]
            kept = np.sort(active[order[: self._max_active_channels]])
            capped = np.sort(active[order[self._max_active_channels :]])
            active = kept

        self._active_indices = tuple(int(i) for i in active)
        self._capped_indices = tuple(int(i) for i in capped)
        self._inactive_indices = tuple(int(i) for i in inactive)
        self._channel_detectors = {}
        self._channel_norm = {}

        for ch in self._active_indices:
            det = self._factory()
            det._feature_names = (feature_names[ch],)
            det._fit_normal(x_train_normal[:, ch : ch + 1], (feature_names[ch],))
            self._channel_detectors[ch] = det

    def _score_matrix_channels(
        self,
        matrix: FloatArray,
        feature_names: tuple[str, ...],
        fit_norm: bool,
    ) -> FloatArray:
        if self._scaler is None:
            raise RuntimeError("IndependentChannelWrapper must be fit before scoring")
        n_rows, n_feats = matrix.shape
        norm_scores = np.zeros((n_rows, n_feats), dtype=np.float64)

        for ch in self._active_indices:
            det = self._channel_detectors[ch]
            ch_arr = matrix[:, ch : ch + 1]
            ch_name = (feature_names[ch],)
            if fit_norm:
                raw = np.asarray(det._score_validation(ch_arr, ch_name), dtype=np.float64)
                med, sc = _robust_channel_stats(raw)
                self._channel_norm[ch] = (med, sc)
            else:
                raw_t, _ = det._score_test(ch_arr, ch_name)
                raw = np.asarray(raw_t, dtype=np.float64)
                med, sc = self._channel_norm[ch]
            norm_scores[:, ch] = np.maximum(0.0, (raw - med) / sc)

        # Capped or inactive channels use identical training-only standardized squared deviation
        mean = self._scaler.mean_
        scale = self._scaler.scale_
        for ch in (*self._capped_indices, *self._inactive_indices):
            dev = np.square((matrix[:, ch] - mean[ch]) / scale[ch])
            if fit_norm:
                med, sc = _robust_channel_stats(dev)
                self._channel_norm[ch] = (med, sc)
            else:
                med, sc = self._channel_norm[ch]
            norm_scores[:, ch] = np.maximum(0.0, (dev - med) / sc)

        return np.asarray(np.mean(norm_scores, axis=1), dtype=np.float64)

    def _score_validation(
        self, x_val_normal: FloatArray, feature_names: tuple[str, ...]
    ) -> FloatArray:
        return self._score_matrix_channels(x_val_normal, feature_names, fit_norm=True)

    def _score_test(
        self, x_test: FloatArray, feature_names: tuple[str, ...]
    ) -> tuple[FloatArray, dict[str, Any]]:
        scores = self._score_matrix_channels(x_test, feature_names, fit_norm=False)
        diag: dict[str, Any] = {
            "channel_mode": "independent_channel_unified",
            "active_channels_modeled": len(self._active_indices),
            "capped_channels": len(self._capped_indices),
            "inactive_channels": len(self._inactive_indices),
            "selected_device": "cpu",
        }
        if self._channel_detectors:
            first_det = next(iter(self._channel_detectors.values()))
            p = first_det.get_parameters()
            if "selected_device" in p:
                diag["selected_device"] = p["selected_device"]
        return scores, diag


def forecast_chunked_tsfm_split(
    detector: ZeroShotForecastDetector,
    history_prefix: FloatArray,
    target_matrix: FloatArray,
    feature_names: Sequence[str],
    track: EvalTrack,
    max_multivariate_group_variates: int = 32,
    long_series_row_threshold: int = 30000,
    long_series_prediction_length: int = 256,
) -> tuple[TSFMScoreDetails, str]:
    """Run causal chunked TSFM forecasting for Track A (multivariate) or Track B (independent).

    - On long series (`target_matrix.shape[0] > long_series_row_threshold`), uses
      `long_series_prediction_length` (e.g. `256`) so long TSB-AD-M series execute
      cleanly with 4× fewer forward passes while keeping strictly causal contexts
      (`<= t_0 - 1`).
    - On Track A (`multivariate`), series with `D <= max_multivariate_group_variates`
      run `multivariate_native`; series with `D > max_multivariate_group_variates`
      partition variates into deterministic contiguous chunks of size
      `max_multivariate_group_variates` (`multivariate_chunked_groups`).
    - On Track B (`independent_channel`), runs independent 1D channel forecasting with
      identical per-channel validation score normalization applied by the caller.
    """
    if detector._scaler is None:
        raise RuntimeError(f"{detector.name} must call _fit_normal before forecasting")
    tgt = _validate_2d_features(target_matrix, "target_matrix")
    names = tuple(feature_names)
    n_rows, n_feats = tgt.shape
    cfg = detector.config
    backend: ForecastBackend = detector._resolve_backend()

    eff_horizon = (
        long_series_prediction_length
        if n_rows > long_series_row_threshold
        else cfg.prediction_length
    )

    scale = detector._scaler.scale_
    mean = detector._scaler.mean_
    point_pred = np.empty((n_rows, n_feats), dtype=np.float64)
    ql_pred = np.empty((n_rows, n_feats), dtype=np.float64)
    qh_pred = np.empty((n_rows, n_feats), dtype=np.float64)

    if track == "multivariate" and backend.supports_multivariate:
        if n_feats <= max_multivariate_group_variates:
            mode_label = "multivariate_native"
            contexts_bdl, spans = build_causal_context_blocks(
                history_prefix, tgt, cfg.context_length, eff_horizon
            )
            fb_out = backend.forecast_multivariate_blocks(
                contexts_bdl=contexts_bdl,
                horizon=eff_horizon,
                batch_size=cfg.batch_size,
            )
            for idx, (t0, t1) in enumerate(spans):
                step_len = t1 - t0
                point_pred[t0:t1, :] = fb_out.point_forecast[idx, :, :step_len].T
                ql_pred[t0:t1, :] = fb_out.q_low[idx, :, :step_len].T
                qh_pred[t0:t1, :] = fb_out.q_high[idx, :, :step_len].T
        else:
            mode_label = f"multivariate_chunked_groups_{max_multivariate_group_variates}"
            for g0 in range(0, n_feats, max_multivariate_group_variates):
                g1 = min(n_feats, g0 + max_multivariate_group_variates)
                contexts_bdl, spans = build_causal_context_blocks(
                    history_prefix[:, g0:g1],
                    tgt[:, g0:g1],
                    cfg.context_length,
                    eff_horizon,
                )
                fb_out = backend.forecast_multivariate_blocks(
                    contexts_bdl=contexts_bdl,
                    horizon=eff_horizon,
                    batch_size=cfg.batch_size,
                )
                for idx, (t0, t1) in enumerate(spans):
                    step_len = t1 - t0
                    point_pred[t0:t1, g0:g1] = fb_out.point_forecast[idx, :, :step_len].T
                    ql_pred[t0:t1, g0:g1] = fb_out.q_low[idx, :, :step_len].T
                    qh_pred[t0:t1, g0:g1] = fb_out.q_high[idx, :, :step_len].T
    else:
        mode_label = "independent_channel_unified"
        active_mask = detector._scaler.active_channel_mask
        for ch_i in np.flatnonzero(~active_mask):
            c_val = float(mean[int(ch_i)])
            point_pred[:, int(ch_i)] = c_val
            ql_pred[:, int(ch_i)] = c_val
            qh_pred[:, int(ch_i)] = c_val

        active_idx = np.flatnonzero(active_mask)
        if active_idx.size > cfg.max_active_univariate_channels:
            variances = np.square(scale[active_idx])
            order = np.argsort(variances)[::-1]
            kept = np.sort(active_idx[order[: cfg.max_active_univariate_channels]])
            skipped = active_idx[order[cfg.max_active_univariate_channels :]]
            for ch_s in skipped:
                c_val = float(mean[int(ch_s)])
                point_pred[:, int(ch_s)] = c_val
                ql_pred[:, int(ch_s)] = c_val - 1.2816 * float(scale[int(ch_s)])
                qh_pred[:, int(ch_s)] = c_val + 1.2816 * float(scale[int(ch_s)])
            active_idx = kept

        if active_idx.size > 0:
            contexts_bdl, spans = build_causal_context_blocks(
                history_prefix[:, active_idx],
                tgt[:, active_idx],
                cfg.context_length,
                eff_horizon,
            )
            n_blocks, n_act, ctx_l = contexts_bdl.shape
            folded_ctx = contexts_bdl.reshape(n_blocks * n_act, 1, ctx_l)
            fb_uv = backend.forecast_multivariate_blocks(
                contexts_bdl=folded_ctx,
                horizon=eff_horizon,
                batch_size=cfg.batch_size,
            )
            pt_b = fb_uv.point_forecast.reshape(n_blocks, n_act, eff_horizon)
            ql_b = fb_uv.q_low.reshape(n_blocks, n_act, eff_horizon)
            qh_b = fb_uv.q_high.reshape(n_blocks, n_act, eff_horizon)
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
    inside_80 = (tgt >= ql_pred) & (tgt <= qh_pred)

    details = TSFMScoreDetails(
        overall_scores=res_overall,
        per_feature_scores=per_feat_residual,
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
    return details, mode_label
