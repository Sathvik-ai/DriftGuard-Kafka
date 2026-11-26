"""Sequence-to-sequence PyTorch LSTM Autoencoder for DriftGuard-Kafka M5.

Implements a leakage-free reconstruction-based time-series anomaly detector:
- **Encoder**: Multi-layer `nn.LSTM` compressing temporal windows `(B, W, D)` into
  a low-dimensional bottleneck representation `z` of shape `(B, latent_dim)`.
- **Decoder**: Bridge projection + multi-layer `nn.LSTM` + linear head reconstructing
  the full input sensor window `(B, W, D)`.
- **Dual series modes**:
  1. `multivariate`: Jointly models coupled multi-sensor windows (e.g. SKAB 8D).
  2. `independent_univariate`: Models each channel strictly as an independent 1D
     series `(B, W, 1)` without assuming cross-channel physical covariance (used
     for TSB series per Research Rule #10).
- **Leakage-free guarantees**:
  - Normalization (`TrainingOnlyScaler`) is fit strictly on normal training data.
  - Sliding windows never cross series, machine, or train/val/test boundaries.
  - Early stopping uses normal-only windows; thresholds are calibrated strictly
    on normal validation reconstruction errors (`x_val_normal`).
  - Produces both per-feature `(N, D)` and overall `(N,)` causal anomaly scores.
"""

from __future__ import annotations

import copy
import json
import platform
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from app.m4_models import BenchmarkDetector, FloatArray, _validate_2d_features

ChannelMode = Literal["multivariate", "independent_univariate"]
ScoreReduction = Literal["last_step", "window_mean"]


@dataclass(frozen=True)
class DeviceInfo:
    """Hardware and PyTorch accelerator diagnostics for macOS / Linux."""

    requested: str
    selected: str
    is_apple_silicon: bool
    mps_built: bool
    mps_available: bool
    cuda_available: bool
    fallback_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serialisable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class LSTMAutoencoderConfig:
    """Hyperparameters and execution configuration for `LSTMAutoencoderDetector`."""

    window_size: int = 32
    hidden_dim: int = 32
    latent_dim: int = 16
    num_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 64
    epochs: int = 15
    learning_rate: float = 0.002
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    calibration_quantile: float = 0.99
    channel_mode: ChannelMode = "multivariate"
    score_reduction: ScoreReduction = "last_step"
    max_train_windows: int = 4096
    eval_batch_size: int = 512
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        """Validate configuration bounds."""
        if self.window_size < 2:
            raise ValueError("window_size must be >= 2")
        if self.hidden_dim < 1 or self.latent_dim < 1:
            raise ValueError("hidden_dim and latent_dim must be >= 1")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")
        if self.batch_size < 1 or self.eval_batch_size < 1:
            raise ValueError("batch_size and eval_batch_size must be >= 1")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be >= 1")
        if not 0.0 < self.calibration_quantile < 1.0:
            raise ValueError("calibration_quantile must be strictly in (0, 1)")
        if self.channel_mode not in ("multivariate", "independent_univariate"):
            raise ValueError(f"unsupported channel_mode: {self.channel_mode!r}")
        if self.score_reduction not in ("last_step", "window_mean"):
            raise ValueError(f"unsupported score_reduction: {self.score_reduction!r}")

    @classmethod
    def from_json_file(cls, path: Path) -> LSTMAutoencoderConfig:
        """Load configuration fields from a JSON config file."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"expected JSON object in {path}")
        allowed = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in raw.items() if k in allowed}
        return cls(**filtered)


@dataclass(frozen=True)
class LSTMScoreOutput:
    """Detailed reconstruction anomaly scores for a single continuous series."""

    overall_scores: FloatArray
    per_feature_scores: FloatArray
    feature_names: tuple[str, ...]


def select_compute_device(requested: str = "auto") -> DeviceInfo:
    """Detect Apple Silicon MPS / CUDA availability with safe CPU fallback."""
    req = requested.strip().lower()
    if req not in {"auto", "mps", "cpu", "cuda"}:
        raise ValueError(f"unsupported device {requested!r}; expected auto, mps, cpu, or cuda")

    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    mps_built = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built())
    mps_avail = bool(mps_built and torch.backends.mps.is_available())
    cuda_avail = bool(torch.cuda.is_available())

    def _probe(dev_name: str) -> str | None:
        try:
            probe = torch.zeros((2, 2), device=torch.device(dev_name))
            _ = float(probe.sum().item())
            return None
        except Exception as exc:  # noqa: BLE001 - safe hardware fallback
            return f"{type(exc).__name__}: {exc}"

    if req == "cpu":
        return DeviceInfo(req, "cpu", is_apple_silicon, mps_built, mps_avail, cuda_avail, None)

    if req in {"auto", "mps"} and mps_avail:
        err = _probe("mps")
        if err is None:
            return DeviceInfo(req, "mps", is_apple_silicon, mps_built, mps_avail, cuda_avail, None)
        return DeviceInfo(
            req,
            "cpu",
            is_apple_silicon,
            mps_built,
            mps_avail,
            cuda_avail,
            f"mps_probe_failed:{err}",
        )

    if req == "mps" and not mps_avail:
        return DeviceInfo(
            req,
            "cpu",
            is_apple_silicon,
            mps_built,
            mps_avail,
            cuda_avail,
            "mps_unavailable_falling_back_to_cpu",
        )

    if req in {"auto", "cuda"} and cuda_avail:
        err = _probe("cuda")
        if err is None:
            return DeviceInfo(req, "cuda", is_apple_silicon, mps_built, mps_avail, cuda_avail, None)
        return DeviceInfo(
            req,
            "cpu",
            is_apple_silicon,
            mps_built,
            mps_avail,
            cuda_avail,
            f"cuda_probe_failed:{err}",
        )

    if req == "cuda" and not cuda_avail:
        return DeviceInfo(
            req,
            "cpu",
            is_apple_silicon,
            mps_built,
            mps_avail,
            cuda_avail,
            "cuda_unavailable_falling_back_to_cpu",
        )

    reason = "mps_unavailable_in_environment" if is_apple_silicon and not mps_avail else None
    return DeviceInfo(req, "cpu", is_apple_silicon, mps_built, mps_avail, cuda_avail, reason)


def set_deterministic_seed(seed: int) -> None:
    """Configure deterministic RNG state across NumPy and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class TrainingOnlyScaler:
    """Per-feature standardization fit strictly on normal training observations."""

    def __init__(self) -> None:
        """Initialise unfitted scaler."""
        self._mean: FloatArray | None = None
        self._std: FloatArray | None = None
        self._active_mask: NDArray[np.bool_] | None = None

    @property
    def mean_(self) -> FloatArray:
        """Return fitted per-feature mean vector."""
        if self._mean is None:
            raise RuntimeError("TrainingOnlyScaler has not been fitted")
        return np.array(self._mean, dtype=np.float64, copy=True)

    @property
    def scale_(self) -> FloatArray:
        """Return fitted per-feature scale vector (1.0 for zero-variance channels)."""
        if self._std is None:
            raise RuntimeError("TrainingOnlyScaler has not been fitted")
        return np.array(self._std, dtype=np.float64, copy=True)

    @property
    def active_channel_mask(self) -> NDArray[np.bool_]:
        """Return boolean mask of channels with non-zero training variance."""
        if self._active_mask is None:
            raise RuntimeError("TrainingOnlyScaler has not been fitted")
        return np.array(self._active_mask, dtype=np.bool_, copy=True)

    def fit(self, x_train_normal: FloatArray) -> TrainingOnlyScaler:
        """Fit per-feature mean and standard deviation strictly on `x_train_normal`."""
        arr = _validate_2d_features(x_train_normal, "x_train_normal")
        mean = np.asarray(np.mean(arr, axis=0, dtype=np.float64), dtype=np.float64)
        raw_std = np.asarray(np.std(arr, axis=0, dtype=np.float64), dtype=np.float64)
        active = np.asarray(raw_std > 1e-8, dtype=np.bool_)
        safe_std = np.asarray(np.where(active, raw_std, 1.0), dtype=np.float64)
        self._mean = mean
        self._std = safe_std
        self._active_mask = active
        return self

    def transform(self, matrix: FloatArray) -> FloatArray:
        """Standardize `matrix` using stored training-only statistics."""
        if self._mean is None or self._std is None:
            raise RuntimeError("TrainingOnlyScaler must be fitted before transform")
        arr = _validate_2d_features(matrix, "matrix")
        if arr.shape[1] != self._mean.shape[0]:
            raise ValueError("feature count mismatch in TrainingOnlyScaler.transform")
        return np.asarray((arr - self._mean) / self._std, dtype=np.float64)


def generate_training_windows(
    series: FloatArray,
    window_size: int,
    stride: int = 1,
) -> NDArray[np.float32]:
    """Build contiguous `(N_win, W, D)` windows strictly inside one series split.

    Never crosses machine or series boundaries. If `len(series) < window_size`,
    causal left-padding with `series[0]` produces a single `(1, W, D)` window.
    """
    arr = _validate_2d_features(series, "series")
    if window_size < 2 or stride < 1:
        raise ValueError("window_size >= 2 and stride >= 1 are required")
    n_rows, n_feats = arr.shape
    arr32 = arr.astype(np.float32, copy=False)
    if n_rows < window_size:
        pad_len = window_size - n_rows
        padded = np.vstack([np.repeat(arr32[:1], pad_len, axis=0), arr32])
        return padded.reshape(1, window_size, n_feats)

    # sliding_window_view along axis 0 yields shape (N - W + 1, D, W); swap to (N_win, W, D).
    raw_view = sliding_window_view(arr32, window_shape=window_size, axis=0)
    win = np.swapaxes(raw_view[::stride], 1, 2)
    return np.ascontiguousarray(win, dtype=np.float32)


def generate_causal_windows(series: FloatArray, window_size: int) -> NDArray[np.float32]:
    """Generate `(N, W, D)` causal windows ending at each timestamp `t = 0..N-1`.

    Timestamp `t` uses only `series[max(0, t - W + 1) : t + 1]`, left-padded at
    `t < W - 1` with `series[0]`, guaranteeing zero lookahead into `s > t`.
    """
    arr = _validate_2d_features(series, "series")
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    arr32 = arr.astype(np.float32, copy=False)
    padded = np.vstack([np.repeat(arr32[:1], window_size - 1, axis=0), arr32])
    raw_view = sliding_window_view(padded, window_shape=window_size, axis=0)
    return np.ascontiguousarray(np.swapaxes(raw_view, 1, 2), dtype=np.float32)


def iter_causal_window_batches(
    series: FloatArray,
    window_size: int,
    batch_size: int,
) -> Iterator[tuple[int, int, NDArray[np.float32]]]:
    """Yield `(start_idx, end_idx, causal_windows_batch)` with bounded RAM."""
    arr = _validate_2d_features(series, "series")
    n_rows = int(arr.shape[0])
    arr32 = arr.astype(np.float32, copy=False)
    padded = np.vstack([np.repeat(arr32[:1], window_size - 1, axis=0), arr32])

    for start in range(0, n_rows, batch_size):
        end = min(n_rows, start + batch_size)
        # Slice padded[start : end + window_size - 1] -> exact causal windows for [start:end]
        chunk = padded[start : end + window_size - 1]
        raw_view = sliding_window_view(chunk, window_shape=window_size, axis=0)
        batch_win = np.ascontiguousarray(np.swapaxes(raw_view, 1, 2), dtype=np.float32)
        yield start, end, batch_win


class LSTMAutoencoder(nn.Module):
    """Sequence-to-sequence LSTM Autoencoder for sensor window reconstruction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        latent_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Construct encoder LSTM, latent bottleneck, and decoder LSTM."""
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        eff_dropout = dropout if num_layers > 1 else 0.0

        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=eff_dropout,
        )
        self.encoder_bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),
        )
        self.decoder_bridge = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=eff_dropout,
        )
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input window `(B, W, D)` into latent vector `(B, latent_dim)`."""
        _enc_seq, (h_n, _c_n) = self.encoder_lstm(x)
        top_hidden = h_n[-1]
        latent: torch.Tensor = self.encoder_bottleneck(top_hidden)
        return latent

    def decode(self, z: torch.Tensor, window_size: int) -> torch.Tensor:
        """Decode latent vector `(B, latent_dim)` back into `(B, W, D)`."""
        bridged = self.decoder_bridge(z)
        repeated = bridged.unsqueeze(1).expand(-1, window_size, -1).contiguous()
        dec_seq, _ = self.decoder_lstm(repeated)
        recon: torch.Tensor = self.output_proj(dec_seq)
        return recon

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct `x` `(B, W, D)` and return `(reconstructed, latent_z)`."""
        window_size = int(x.shape[1])
        z = self.encode(x)
        recon = self.decode(z, window_size=window_size)
        return recon, z


class LSTMAutoencoderDetector(BenchmarkDetector):
    """Leakage-free LSTM Autoencoder detector implementing `BenchmarkDetector`."""

    def __init__(self, config: LSTMAutoencoderConfig | None = None) -> None:
        """Initialise detector with `LSTMAutoencoderConfig`."""
        cfg = config or LSTMAutoencoderConfig()
        super().__init__(calibration_quantile=cfg.calibration_quantile)
        self._cfg = cfg
        self._device_info = select_compute_device(cfg.device)
        self._scaler: TrainingOnlyScaler | None = None
        self._model: LSTMAutoencoder | None = None
        self._training_history: dict[str, Any] = {}
        self._last_per_feature_scores: FloatArray | None = None

    @property
    def name(self) -> str:
        """Return canonical detector name."""
        return "lstm_autoencoder"

    @property
    def config(self) -> LSTMAutoencoderConfig:
        """Return immutable detector configuration."""
        return self._cfg

    @property
    def device_info(self) -> DeviceInfo:
        """Return hardware/MPS selection metadata."""
        return self._device_info

    @property
    def training_history(self) -> dict[str, Any]:
        """Return epoch losses and early-stopping diagnostics from last fit."""
        return dict(self._training_history)

    @property
    def last_per_feature_scores(self) -> FloatArray | None:
        """Return `(N, D)` per-feature anomaly scores from the most recent evaluation."""
        if self._last_per_feature_scores is None:
            return None
        return np.array(self._last_per_feature_scores, dtype=np.float64, copy=True)

    def get_parameters(self) -> dict[str, Any]:
        """Return JSON-serialisable hyperparameter and device dictionary."""
        params = asdict(self._cfg)
        params["selected_device"] = self._device_info.selected
        params["mps_available"] = self._device_info.mps_available
        params["is_apple_silicon"] = self._device_info.is_apple_silicon
        if self._training_history:
            params["epochs_trained"] = self._training_history.get("epochs_trained")
            params["early_stopped"] = self._training_history.get("early_stopped")
            params["best_monitor_loss"] = self._training_history.get("best_monitor_loss")
        return params

    def save_checkpoint(self, path: Path) -> None:
        """Save trained weights, training-only scaler, and calibrated threshold."""
        if self._model is None or self._scaler is None or self._threshold is None:
            raise RuntimeError("cannot save checkpoint before fit_and_calibrate")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self._cfg),
            "feature_names": list(self._feature_names),
            "threshold": self._threshold,
            "scaler_mean": self._scaler.mean_,
            "scaler_scale": self._scaler.scale_,
            "state_dict": self._model.state_dict(),
            "training_history": self._training_history,
        }
        torch.save(payload, path)

    def _build_normal_windows_for_training(self, x_scaled: FloatArray) -> NDArray[np.float32]:
        cfg = self._cfg
        n_rows, n_feats = x_scaled.shape
        if cfg.channel_mode == "multivariate":
            raw_count = max(1, n_rows - cfg.window_size + 1)
            stride = max(1, raw_count // cfg.max_train_windows)
            return generate_training_windows(x_scaled, cfg.window_size, stride=stride)

        # independent_univariate mode: build strictly independent 1D windows per channel
        # without coupling channels into multivariate vectors.
        active_mask = (
            self._scaler.active_channel_mask
            if self._scaler is not None
            else np.ones(n_feats, dtype=bool)
        )
        active_indices = np.flatnonzero(active_mask)
        if active_indices.size == 0:
            active_indices = np.array([0], dtype=np.int64)

        raw_per_ch = max(1, n_rows - cfg.window_size + 1)
        total_raw = raw_per_ch * int(active_indices.size)
        stride = max(1, total_raw // cfg.max_train_windows)

        ch_windows: list[NDArray[np.float32]] = []
        for ch_idx in active_indices:
            col = x_scaled[:, int(ch_idx) : int(ch_idx) + 1]
            ch_windows.append(generate_training_windows(col, cfg.window_size, stride=stride))
        combined = np.concatenate(ch_windows, axis=0)
        if combined.shape[0] > cfg.max_train_windows:
            idx = np.linspace(0, combined.shape[0] - 1, cfg.max_train_windows, dtype=np.int64)
            combined = combined[idx]
        return combined

    def _fit_normal(self, x_train_normal: FloatArray, feature_names: tuple[str, ...]) -> None:
        del feature_names
        cfg = self._cfg
        set_deterministic_seed(cfg.seed)

        scaler = TrainingOnlyScaler().fit(x_train_normal)
        self._scaler = scaler
        x_scaled = scaler.transform(x_train_normal)

        all_windows = self._build_normal_windows_for_training(x_scaled)
        n_total = int(all_windows.shape[0])
        # Chronological split inside normal training windows for early stopping:
        if n_total >= 8:
            split_idx = max(4, int(round(0.85 * n_total)))
            split_idx = min(split_idx, n_total - 2)
            fit_win = all_windows[:split_idx]
            mon_win = all_windows[split_idx:]
        else:
            fit_win = all_windows
            mon_win = all_windows

        input_dim = int(all_windows.shape[2])
        dev = torch.device(self._device_info.selected)
        model = LSTMAutoencoder(
            input_dim=input_dim,
            hidden_dim=cfg.hidden_dim,
            latent_dim=cfg.latent_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        ).to(dev)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        criterion = nn.MSELoss()
        rng = np.random.default_rng(cfg.seed)

        best_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        patience_left = cfg.early_stopping_patience
        train_losses: list[float] = []
        monitor_losses: list[float] = []
        early_stopped = False

        mon_tensor = torch.from_numpy(mon_win).to(dev)
        n_fit = int(fit_win.shape[0])

        for _epoch in range(cfg.epochs):
            model.train()
            perm = rng.permutation(n_fit)
            batch_losses: list[float] = []
            for start in range(0, n_fit, cfg.batch_size):
                batch_np = fit_win[perm[start : start + cfg.batch_size]]
                batch_t = torch.from_numpy(batch_np).to(dev)
                optimizer.zero_grad(set_to_none=True)
                recon, _ = model(batch_t)
                loss = criterion(recon, batch_t)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu().item()))

            epoch_train_loss = float(sum(batch_losses) / max(1, len(batch_losses)))
            train_losses.append(round(epoch_train_loss, 6))

            model.eval()
            with torch.inference_mode():
                mon_recon, _ = model(mon_tensor)
                mon_loss = float(criterion(mon_recon, mon_tensor).cpu().item())
            monitor_losses.append(round(mon_loss, 6))

            if mon_loss < best_loss - cfg.early_stopping_min_delta:
                best_loss = mon_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_left = cfg.early_stopping_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    early_stopped = True
                    break

        model.load_state_dict(best_state)
        model.eval()
        self._model = model
        self._training_history = {
            "epochs_configured": cfg.epochs,
            "epochs_trained": len(train_losses),
            "early_stopped": early_stopped,
            "best_monitor_loss": round(best_loss, 6),
            "final_train_loss": train_losses[-1] if train_losses else None,
            "train_losses": train_losses,
            "monitor_losses": monitor_losses,
            "training_windows": n_fit,
            "monitor_windows": int(mon_win.shape[0]),
            "input_dim": input_dim,
            "channel_mode": cfg.channel_mode,
        }

    def _reduce_window_sq_error(self, sq_err: torch.Tensor) -> torch.Tensor:
        if self._cfg.score_reduction == "last_step":
            return sq_err[:, -1, :]
        return torch.mean(sq_err, dim=1)

    def score_details(
        self,
        matrix: FloatArray,
        feature_names: Sequence[str] | None = None,
    ) -> LSTMScoreOutput:
        """Compute causal per-feature `(N, D)` and overall `(N,)` anomaly scores."""
        if self._scaler is None or self._model is None:
            raise RuntimeError("LSTMAutoencoderDetector must be fitted before scoring")
        arr = _validate_2d_features(matrix, "matrix")
        names = tuple(feature_names) if feature_names is not None else self._feature_names
        if len(names) != arr.shape[1]:
            raise ValueError("feature_names length must match matrix columns")

        x_scaled = self._scaler.transform(arr)
        n_rows, n_feats = x_scaled.shape
        per_feature = np.empty((n_rows, n_feats), dtype=np.float64)
        dev = torch.device(self._device_info.selected)
        self._model.eval()

        with torch.inference_mode():
            if self._cfg.channel_mode == "multivariate":
                for start, end, win_batch in iter_causal_window_batches(
                    x_scaled, self._cfg.window_size, self._cfg.eval_batch_size
                ):
                    batch_t = torch.from_numpy(win_batch).to(dev)
                    recon_t, _ = self._model(batch_t)
                    sq_err = (batch_t - recon_t).pow(2)
                    step_err = self._reduce_window_sq_error(sq_err)
                    per_feature[start:end, :] = step_err.cpu().numpy().astype(np.float64)
            else:
                for ch_idx in range(n_feats):
                    col = x_scaled[:, ch_idx : ch_idx + 1]
                    for start, end, win_batch in iter_causal_window_batches(
                        col, self._cfg.window_size, self._cfg.eval_batch_size
                    ):
                        batch_t = torch.from_numpy(win_batch).to(dev)
                        recon_t, _ = self._model(batch_t)
                        sq_err = (batch_t - recon_t).pow(2)
                        step_err = self._reduce_window_sq_error(sq_err)
                        per_feature[start:end, ch_idx] = (
                            step_err[:, 0].cpu().numpy().astype(np.float64)
                        )

        overall = np.asarray(np.mean(per_feature, axis=1), dtype=np.float64)
        return LSTMScoreOutput(
            overall_scores=overall,
            per_feature_scores=per_feature,
            feature_names=names,
        )

    def _score_validation(
        self, x_val_normal: FloatArray, feature_names: tuple[str, ...]
    ) -> FloatArray:
        return self.score_details(x_val_normal, feature_names).overall_scores

    def _score_test(
        self, x_test: FloatArray, feature_names: tuple[str, ...]
    ) -> tuple[FloatArray, dict[str, Any]]:
        out = self.score_details(x_test, feature_names)
        self._last_per_feature_scores = out.per_feature_scores
        feat_means = np.mean(out.per_feature_scores, axis=0)
        top_idx = int(np.argmax(feat_means))
        diagnostics: dict[str, Any] = {
            "channel_mode": self._cfg.channel_mode,
            "score_reduction": self._cfg.score_reduction,
            "selected_device": self._device_info.selected,
            "epochs_trained": self._training_history.get("epochs_trained"),
            "early_stopped": self._training_history.get("early_stopped"),
            "top_error_feature": feature_names[top_idx],
            "per_feature_mean_error": {
                fname: round(float(val), 6)
                for fname, val in zip(feature_names, feat_means, strict=True)
            },
        }
        return out.overall_scores, diagnostics


if __name__ == "__main__":
    import sys

    from app.m5_benchmark import main as _m5_main

    sys.exit(_m5_main())
