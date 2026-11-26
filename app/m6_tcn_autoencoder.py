"""Temporal Convolutional Network (TCN) Autoencoder for DriftGuard-Kafka M6.

Implements a sequence autoencoder built from stacked causal, dilated 1D
convolutional residual blocks (`CausalConv1d` + `TCNResidualBlock`):
- **Causal convolutions**: Left-only padding `(kernel_size - 1) * dilation`
  ensures feature maps at step `t` never access future steps `s > t`.
- **Dilated multi-scale receptive field**: Exponential dilations `(1, 2, 4, 8)`
  capture both short-term high-frequency fluctuations and long-term temporal
  trends across sensor windows.
- **Residual blocks**: Two causal dilated convolutions with GELU activations,
  dropout, and `1x1` projection shortcuts stabilize deep temporal training.
- **Undercomplete temporal bottleneck**: Compresses multi-scale encoder states
  into a low-dimensional latent vector `z` before causal TCN decoding so trivial
  identity pass-through cannot occur on either multivariate (SKAB 8D) or
  independent univariate (TSB 1D) series.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from app.m4_models import BenchmarkDetector, FloatArray, _validate_2d_features
from app.m5_lstm_autoencoder import (
    ChannelMode,
    DeviceInfo,
    LSTMScoreOutput,
    ScoreReduction,
    TrainingOnlyScaler,
    generate_training_windows,
    iter_causal_window_batches,
    select_compute_device,
    set_deterministic_seed,
)


@dataclass(frozen=True)
class TCNAutoencoderConfig:
    """Architecture and training configuration for `TCNAutoencoderDetector`."""

    window_size: int = 32
    hidden_channels: int = 24
    latent_dim: int = 8
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4)
    dropout: float = 0.1
    batch_size: int = 64
    epochs: int = 12
    learning_rate: float = 0.002
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    calibration_quantile: float = 0.99
    channel_mode: ChannelMode = "multivariate"
    score_reduction: ScoreReduction = "window_mean"
    max_train_windows: int = 4096
    eval_batch_size: int = 4096
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        """Validate TCN configuration invariants."""
        if self.window_size < 2:
            raise ValueError("window_size must be >= 2")
        if self.hidden_channels < 1 or self.latent_dim < 1:
            raise ValueError("hidden_channels and latent_dim must be >= 1")
        if self.kernel_size < 2:
            raise ValueError("kernel_size must be >= 2 for causal temporal receptive field")
        if not self.dilations or any(d < 1 for d in self.dilations):
            raise ValueError("dilations must be a non-empty tuple of positive integers")
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

    @property
    def receptive_field(self) -> int:
        """Return effective 1D causal receptive field across residual TCN stack."""
        return 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)

    @classmethod
    def from_json_file(cls, path: Path) -> TCNAutoencoderConfig:
        """Load TCN configuration from a JSON file."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"expected JSON object in {path}")
        allowed = set(cls.__dataclass_fields__)
        filtered: dict[str, Any] = {k: v for k, v in raw.items() if k in allowed}
        if "dilations" in filtered and isinstance(filtered["dilations"], list):
            filtered["dilations"] = tuple(int(x) for x in filtered["dilations"])
        return cls(**filtered)


class CausalConv1d(nn.Module):
    """1D convolution with strict left-only causal padding `(kernel_size - 1) * dilation`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        """Initialise causal 1D convolution layer."""
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply left causal pad followed by dilated 1D convolution on `(B, C, W)`."""
        padded = functional.pad(x, (self.left_padding, 0))
        out: torch.Tensor = self.conv(padded)
        return out


class TCNResidualBlock(nn.Module):
    """Two-layer causal dilated TCN residual block with `1x1` channel shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ) -> None:
        """Construct causal dilated conv pair and residual shortcut."""
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)
        self.shortcut: nn.Module = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.out_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute residual causal TCN block output `(B, out_channels, W)`."""
        h = self.drop1(self.act1(self.conv1(x)))
        h = self.drop2(self.act2(self.conv2(h)))
        res: torch.Tensor = self.out_act(h + self.shortcut(x))
        return res


class TCNAutoencoder(nn.Module):
    """Causal dilated TCN sequence autoencoder reconstructing `(B, W, D)` windows."""

    def __init__(
        self,
        input_dim: int,
        window_size: int = 32,
        hidden_channels: int = 32,
        latent_dim: int = 8,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8),
        dropout: float = 0.1,
    ) -> None:
        """Build multi-scale causal TCN encoder, bottleneck, and causal TCN decoder."""
        super().__init__()
        if input_dim < 1 or window_size < 2:
            raise ValueError("input_dim >= 1 and window_size >= 2 are required")
        self.input_dim = input_dim
        self.window_size = window_size
        self.hidden_channels = hidden_channels
        # Guarantee undercomplete bottleneck (< window_size * input_dim when possible).
        max_undercomplete = max(2, (window_size * input_dim) // 2)
        self.latent_dim = min(latent_dim, max_undercomplete)

        enc_blocks: list[nn.Module] = []
        in_ch = input_dim
        for dil in dilations:
            enc_blocks.append(
                TCNResidualBlock(
                    in_channels=in_ch,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=int(dil),
                    dropout=dropout,
                )
            )
            in_ch = hidden_channels
        self.encoder_tcn = nn.Sequential(*enc_blocks)

        self.encoder_bottleneck = nn.Sequential(
            nn.Linear(hidden_channels, self.latent_dim),
            nn.Tanh(),
        )
        self.decoder_bridge = nn.Linear(self.latent_dim, hidden_channels * window_size)

        dec_blocks: list[nn.Module] = []
        for dil in dilations:
            dec_blocks.append(
                TCNResidualBlock(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=int(dil),
                    dropout=dropout,
                )
            )
        self.decoder_tcn = nn.Sequential(*dec_blocks)
        self.output_head = nn.Conv1d(hidden_channels, input_dim, kernel_size=1)

    def encode_causal_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run strictly causal encoder TCN stack returning `(B, hidden_channels, W)`."""
        x_ch_first = x.transpose(1, 2).contiguous()
        feat: torch.Tensor = self.encoder_tcn(x_ch_first)
        return feat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compress input window `(B, W, D)` into bottleneck vector `(B, latent_dim)`."""
        enc_feat = self.encode_causal_features(x)
        # Final causal step summarizes full multi-scale history across 0..W-1.
        last_step_feat = enc_feat[:, :, -1]
        z: torch.Tensor = self.encoder_bottleneck(last_step_feat)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector `(B, latent_dim)` into reconstructed `(B, W, D)`."""
        bsz = int(z.shape[0])
        seeded = self.decoder_bridge(z).view(bsz, self.hidden_channels, self.window_size)
        dec_feat = self.decoder_tcn(seeded)
        out_ch_first = self.output_head(dec_feat)
        recon: torch.Tensor = out_ch_first.transpose(1, 2).contiguous()
        return recon

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct `(B, W, D)` sensor window and return `(reconstructed, z)`."""
        if int(x.shape[1]) != self.window_size:
            raise ValueError(f"expected window_size={self.window_size}, got {int(x.shape[1])}")
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


class TCNAutoencoderDetector(BenchmarkDetector):
    """Leakage-free TCN Autoencoder detector implementing `BenchmarkDetector`."""

    def __init__(self, config: TCNAutoencoderConfig | None = None) -> None:
        """Initialise detector with `TCNAutoencoderConfig`."""
        cfg = config or TCNAutoencoderConfig()
        super().__init__(calibration_quantile=cfg.calibration_quantile)
        self._cfg = cfg
        self._device_info = select_compute_device(cfg.device)
        self._scaler: TrainingOnlyScaler | None = None
        self._model: TCNAutoencoder | None = None
        self._training_history: dict[str, Any] = {}
        self._last_per_feature_scores: FloatArray | None = None

    @property
    def name(self) -> str:
        """Return canonical detector identifier."""
        return "tcn_autoencoder"

    @property
    def config(self) -> TCNAutoencoderConfig:
        """Return immutable TCN configuration."""
        return self._cfg

    @property
    def device_info(self) -> DeviceInfo:
        """Return compute accelerator diagnostics."""
        return self._device_info

    @property
    def training_history(self) -> dict[str, Any]:
        """Return training and early-stopping diagnostics."""
        return dict(self._training_history)

    @property
    def last_per_feature_scores(self) -> FloatArray | None:
        """Return `(N, D)` per-feature reconstruction error scores from last test call."""
        if self._last_per_feature_scores is None:
            return None
        return np.array(self._last_per_feature_scores, dtype=np.float64, copy=True)

    def get_parameters(self) -> dict[str, Any]:
        """Return JSON-serialisable hyperparameter dictionary."""
        params = asdict(self._cfg)
        params["dilations"] = list(self._cfg.dilations)
        params["receptive_field"] = self._cfg.receptive_field
        params["selected_device"] = self._device_info.selected
        params["mps_available"] = self._device_info.mps_available
        params["is_apple_silicon"] = self._device_info.is_apple_silicon
        if self._training_history:
            params["epochs_trained"] = self._training_history.get("epochs_trained")
            params["early_stopped"] = self._training_history.get("early_stopped")
            params["effective_latent_dim"] = self._training_history.get("effective_latent_dim")
        return params

    def save_checkpoint(self, path: Path) -> None:
        """Save trained TCN weights, scaler statistics, and calibrated threshold."""
        if self._model is None or self._scaler is None or self._threshold is None:
            raise RuntimeError("cannot save checkpoint before fit_and_calibrate")
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg_dict = asdict(self._cfg)
        cfg_dict["dilations"] = list(self._cfg.dilations)
        torch.save(
            {
                "config": cfg_dict,
                "feature_names": list(self._feature_names),
                "threshold": self._threshold,
                "scaler_mean": self._scaler.mean_,
                "scaler_scale": self._scaler.scale_,
                "state_dict": self._model.state_dict(),
                "training_history": self._training_history,
            },
            path,
        )

    def _build_normal_windows_for_training(self, x_scaled: FloatArray) -> NDArray[np.float32]:
        cfg = self._cfg
        n_rows, n_feats = x_scaled.shape
        if cfg.channel_mode == "multivariate":
            raw_count = max(1, n_rows - cfg.window_size + 1)
            stride = max(1, raw_count // cfg.max_train_windows)
            return generate_training_windows(x_scaled, cfg.window_size, stride=stride)

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

        ch_windows = [
            generate_training_windows(
                x_scaled[:, int(ch_idx) : int(ch_idx) + 1], cfg.window_size, stride=stride
            )
            for ch_idx in active_indices
        ]
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
        if n_total >= 8:
            split_idx = min(max(4, int(round(0.85 * n_total))), n_total - 2)
            fit_win = all_windows[:split_idx]
            mon_win = all_windows[split_idx:]
        else:
            fit_win = all_windows
            mon_win = all_windows

        input_dim = int(all_windows.shape[2])
        dev = torch.device(self._device_info.selected)
        model = TCNAutoencoder(
            input_dim=input_dim,
            window_size=cfg.window_size,
            hidden_channels=cfg.hidden_channels,
            latent_dim=cfg.latent_dim,
            kernel_size=cfg.kernel_size,
            dilations=cfg.dilations,
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

            epoch_loss = float(sum(batch_losses) / max(1, len(batch_losses)))
            train_losses.append(round(epoch_loss, 6))

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
            "effective_latent_dim": model.latent_dim,
            "receptive_field": cfg.receptive_field,
            "channel_mode": cfg.channel_mode,
        }

    def _reduce_window_sq_error(self, sq_err: torch.Tensor) -> torch.Tensor:
        if self._cfg.score_reduction == "last_step":
            return sq_err[:, -1, :]
        return torch.mean(sq_err, dim=1)

    def _score_overlapping_windows(
        self,
        x_scaled: FloatArray,
        dev: torch.device,
    ) -> FloatArray:
        """Reconstruct 50%-overlapping contiguous windows inside `x_scaled` in large batches."""
        if self._model is None or self._scaler is None:
            raise RuntimeError("TCNAutoencoderDetector is not fitted")
        win_size = self._cfg.window_size
        stride = max(1, win_size // 2)
        n_rows, n_feats = x_scaled.shape
        per_feature = np.zeros((n_rows, n_feats), dtype=np.float64)

        if self._cfg.channel_mode == "multivariate":
            active_idx = np.arange(n_feats, dtype=np.int64)
        else:
            active_mask = self._scaler.active_channel_mask
            inactive_mask = ~active_mask
            if np.any(inactive_mask):
                # Inactive zero-variance channels have normal mean=0, scale=1 -> squared deviation.
                per_feature[:, inactive_mask] = np.square(x_scaled[:, inactive_mask])
            active_idx = np.flatnonzero(active_mask)
            if active_idx.size == 0:
                return per_feature

        sub = np.ascontiguousarray(x_scaled[:, active_idx], dtype=np.float32)
        n_act = int(active_idx.size)

        if n_rows < win_size:
            pad_len = win_size - n_rows
            padded = np.vstack([np.repeat(sub[:1], pad_len, axis=0), sub])
            win_all = padded.reshape(1, win_size, n_act)
            starts = np.array([0], dtype=np.int64)
        else:
            raw_view = sliding_window_view(sub, window_shape=win_size, axis=0)
            base_starts = list(range(0, n_rows - win_size + 1, stride))
            if base_starts[-1] != n_rows - win_size:
                base_starts.append(n_rows - win_size)
            starts = np.asarray(base_starts, dtype=np.int64)
            win_all = np.ascontiguousarray(np.swapaxes(raw_view[starts], 1, 2), dtype=np.float32)

        n_win = int(win_all.shape[0])
        sq_err_all = np.empty((n_win, win_size, n_act), dtype=np.float32)

        if self._cfg.channel_mode == "multivariate":
            step_b = self._cfg.eval_batch_size
            for b0 in range(0, n_win, step_b):
                b1 = min(n_win, b0 + step_b)
                bt = torch.from_numpy(win_all[b0:b1]).to(dev)
                recon_t, _ = self._model(bt)
                sq_err_all[b0:b1] = (bt - recon_t).pow(2).cpu().numpy()
        else:
            # Fold independent channels (n_win, W, n_act) -> (n_win * n_act, W, 1)
            folded = np.transpose(win_all, (0, 2, 1)).reshape(n_win * n_act, win_size, 1)
            folded_err = np.empty_like(folded)
            step_b = self._cfg.eval_batch_size
            for b0 in range(0, int(folded.shape[0]), step_b):
                b1 = min(int(folded.shape[0]), b0 + step_b)
                bt = torch.from_numpy(folded[b0:b1]).to(dev)
                recon_t, _ = self._model(bt)
                folded_err[b0:b1] = (bt - recon_t).pow(2).cpu().numpy()
            sq_err_all = np.transpose(folded_err.reshape(n_win, n_act, win_size), (0, 2, 1))

        if n_rows < win_size:
            pad_len = win_size - n_rows
            per_feature[:, active_idx] = sq_err_all[0, pad_len:, :].astype(np.float64)
            return per_feature

        err_sum = np.zeros((n_rows, n_act), dtype=np.float64)
        err_cnt = np.zeros((n_rows, 1), dtype=np.float64)
        for w_i, s_idx in enumerate(starts):
            s_int = int(s_idx)
            err_sum[s_int : s_int + win_size] += sq_err_all[w_i].astype(np.float64)
            err_cnt[s_int : s_int + win_size] += 1.0
        per_feature[:, active_idx] = err_sum / np.maximum(err_cnt, 1.0)
        return per_feature

    def score_details(
        self,
        matrix: FloatArray,
        feature_names: Sequence[str] | None = None,
    ) -> LSTMScoreOutput:
        """Compute per-feature `(N, D)` and overall `(N,)` TCN anomaly scores."""
        if self._scaler is None or self._model is None:
            raise RuntimeError("TCNAutoencoderDetector must be fitted before scoring")
        arr = _validate_2d_features(matrix, "matrix")
        names = tuple(feature_names) if feature_names is not None else self._feature_names
        if len(names) != arr.shape[1]:
            raise ValueError("feature_names length must match matrix columns")

        x_scaled = self._scaler.transform(arr)
        n_rows, n_feats = x_scaled.shape
        dev = torch.device(self._device_info.selected)
        self._model.eval()

        with torch.inference_mode():
            if self._cfg.score_reduction == "window_mean":
                per_feature = self._score_overlapping_windows(x_scaled, dev)
            elif self._cfg.channel_mode == "multivariate":
                per_feature = np.empty((n_rows, n_feats), dtype=np.float64)
                for start, end, win_batch in iter_causal_window_batches(
                    x_scaled, self._cfg.window_size, self._cfg.eval_batch_size
                ):
                    batch_t = torch.from_numpy(win_batch).to(dev)
                    recon_t, _ = self._model(batch_t)
                    sq_err = (batch_t - recon_t).pow(2)
                    step_err = self._reduce_window_sq_error(sq_err)
                    per_feature[start:end, :] = step_err.cpu().numpy().astype(np.float64)
            else:
                per_feature = np.empty((n_rows, n_feats), dtype=np.float64)
                ch_chunk = max(1, self._cfg.eval_batch_size // max(1, n_feats))
                for start, end, win_batch in iter_causal_window_batches(
                    x_scaled, self._cfg.window_size, ch_chunk
                ):
                    b_len = end - start
                    folded = (
                        np.transpose(win_batch, (0, 2, 1))
                        .reshape(b_len * n_feats, self._cfg.window_size, 1)
                        .copy()
                    )
                    batch_t = torch.from_numpy(folded).to(dev)
                    recon_t, _ = self._model(batch_t)
                    sq_err = (batch_t - recon_t).pow(2)
                    step_err = self._reduce_window_sq_error(sq_err)
                    per_feature[start:end, :] = (
                        step_err[:, 0].cpu().numpy().reshape(b_len, n_feats).astype(np.float64)
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
            "receptive_field": self._cfg.receptive_field,
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

    from app.m6_benchmark import main as _m6_main

    sys.exit(_m6_main())
