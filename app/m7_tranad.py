"""TranAD Transformer-based Anomaly Detection for DriftGuard-Kafka M7.

Implements the two-phase self-conditioning Transformer architecture of:
  Tuli, Casale, and Jennings, "TranAD: Deep Transformer Networks for Anomaly
  Detection in Multivariate Time Series Data", PVLDB 15(6), 2022.
  Official repository: https://github.com/imperial-qore/TranAD

Architecture faithful to `imperial-qore/TranAD` (`src/models.py` `class TranAD`
and `main.py` `backprop`):
1. **Context Window & Target**: Causal context window `W` of shape `(W, B, D)`
   paired with latest observation `elem = W[-1:, :, :]` of shape `(1, B, D)`.
2. **Concatenated Conditioning Embedding**: Input `src` `(W, B, D)` is
   concatenated with focus score `c` `(W, B, D)` -> `(W, B, 2D)`, scaled by
   `sqrt(D)`, passed through combined `sin + cos` `TranADPositionalEncoding`,
   and encoded via `TranADEncoderLayer` (`d_model=2D, nhead=D, LeakyReLU`).
3. **Phase 1 (Initial Reconstruction)**: Focus signal initialized to zeros
   `c_1 = 0`; Decoder 1 + `Linear(2D, D) + Sigmoid` produces `x_1` `(1, B, D)`.
4. **Phase 2 (Self-Conditioned Reconstruction)**: Phase 1 squared deviation
   `c_2 = (x_1 - src) ** 2` `(W, B, D)` conditions a second encoder pass and
   Decoder 2 + `Linear(2D, D) + Sigmoid` to produce `x_2` `(1, B, D)`.
5. **Evolving Two-Phase Objective**: Trained on normal windows using the
   official `imperial-qore/TranAD` epoch-evolving loss
   `(1 / n) * MSE(x_1, elem) + (1 - 1 / n) * MSE(x_2, elem)` (`n = epoch + 1`)
   with `AdamW` + `StepLR`.

Documented deviations from original paper/repo (required for leakage-free M7):
- Uses `float32` tensors instead of `DoubleTensor` (`float64`) for Apple Silicon
  `mps` GPU compatibility.
- Replaces SPOT/POT (`pot_eval` in `main.py`, which inspected test distributions
  and rolled test arrays by 1 step) with strict `calibration_quantile` threshold
  calibration on held-out chronological normal validation data (`x_val_normal`).
- Fits `TrainingOnlyMinMaxScaler` (`[0, 1]` for `Sigmoid` head) strictly on
  `x_train_normal` only; no MAML cross-dataset meta-initialization is used so
  each machine/experiment remains strictly isolated without cross-series leakage.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from app.m4_models import BenchmarkDetector, FloatArray, _validate_2d_features
from app.m5_lstm_autoencoder import (
    ChannelMode,
    DeviceInfo,
    generate_training_windows,
    iter_causal_window_batches,
    select_compute_device,
    set_deterministic_seed,
)

TranADScoreMode = Literal["phase2", "both_phases"]

TRANAD_ARCHITECTURAL_NOTES: list[str] = [
    "Implements the exact two-phase self-conditioning `TranAD` module from "
    "`imperial-qore/TranAD/src/models.py`: concatenated `(src, c)` of width `2*D`, "
    "`sin+cos` positional encoding, `nhead=D` multi-head attention with `LeakyReLU` "
    "feedforward blocks, dual Transformer decoders (`decoder1`, `decoder2`), and a "
    "shared `Linear(2*D, D) + Sigmoid` projection.",
    "Phase 1 evaluates with zero focus `c_1 = 0` to produce `x_1`; Phase 2 computes "
    "the focus matrix `c_2 = (x_1 - src) ** 2` across the causal window and "
    "re-encodes `(src, c_2)` before `decoder2` produces `x_2`.",
    "Deviation 1 (Precision / Hardware): Uses `torch.float32` rather than "
    "`torch.DoubleTensor` (`float64`) so Apple Silicon `mps` acceleration is supported.",
    "Deviation 2 (Leakage-free Thresholding): Replaces `pot_eval` + `torch.roll` "
    "from `imperial-qore/TranAD/main.py` with strict quantile threshold calibration "
    "on held-out normal validation observations (`x_val_normal`) only.",
    "Deviation 3 (Strict Per-Series Isolation): Omits MAML cross-task meta-updates "
    "and follows the official `main.py` single-series epoch-evolving objective "
    "`(1/n)*MSE(x_1, elem) + (1 - 1/n)*MSE(x_2, elem)` so series/machine boundaries "
    "remain strictly isolated.",
]


@dataclass(frozen=True)
class TranADConfig:
    """Configuration for `TranADDetector`."""

    window_size: int = 10
    dim_feedforward: int = 16
    num_encoder_layers: int = 1
    num_decoder_layers: int = 1
    dropout: float = 0.1
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 0.002
    weight_decay: float = 1e-5
    lr_step_size: int = 5
    lr_gamma: float = 0.9
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    calibration_quantile: float = 0.99
    score_mode: TranADScoreMode = "both_phases"
    channel_mode: ChannelMode = "multivariate"
    max_train_windows: int = 4096
    eval_batch_size: int = 8192
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        """Validate TranAD hyperparameters."""
        if self.window_size < 2:
            raise ValueError("window_size must be >= 2")
        if self.dim_feedforward < 1:
            raise ValueError("dim_feedforward must be >= 1")
        if self.num_encoder_layers < 1 or self.num_decoder_layers < 1:
            raise ValueError("num_encoder_layers and num_decoder_layers must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")
        if self.batch_size < 1 or self.eval_batch_size < 1:
            raise ValueError("batch_size and eval_batch_size must be >= 1")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be > 0 and weight_decay >= 0")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be >= 1")
        if not 0.0 < self.calibration_quantile < 1.0:
            raise ValueError("calibration_quantile must be strictly in (0, 1)")
        if self.score_mode not in ("phase2", "both_phases"):
            raise ValueError(f"unsupported score_mode: {self.score_mode!r}")
        if self.channel_mode not in ("multivariate", "independent_univariate"):
            raise ValueError(f"unsupported channel_mode: {self.channel_mode!r}")

    @classmethod
    def from_json_file(cls, path: Path) -> TranADConfig:
        """Load configuration from a JSON file."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"expected JSON object in {path}")
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in allowed})


@dataclass(frozen=True)
class TranADScoreOutput:
    """Detailed two-phase anomaly scores and per-feature attributions."""

    overall_scores: FloatArray
    per_feature_scores: FloatArray
    phase1_per_feature_scores: FloatArray
    phase2_per_feature_scores: FloatArray
    feature_names: tuple[str, ...]


class TrainingOnlyMinMaxScaler:
    """Min-max `[0, 1]` scaler fit strictly on normal training data for Sigmoid head."""

    def __init__(self) -> None:
        """Initialise unfitted min-max scaler."""
        self._min: FloatArray | None = None
        self._span: FloatArray | None = None
        self._active_mask: NDArray[np.bool_] | None = None

    @property
    def min_(self) -> FloatArray:
        """Return fitted per-feature minimum."""
        if self._min is None:
            raise RuntimeError("TrainingOnlyMinMaxScaler has not been fitted")
        return np.array(self._min, dtype=np.float64, copy=True)

    @property
    def span_(self) -> FloatArray:
        """Return fitted per-feature span (`1.0` for constant training features)."""
        if self._span is None:
            raise RuntimeError("TrainingOnlyMinMaxScaler has not been fitted")
        return np.array(self._span, dtype=np.float64, copy=True)

    @property
    def active_channel_mask(self) -> NDArray[np.bool_]:
        """Return boolean mask of channels with non-zero training range."""
        if self._active_mask is None:
            raise RuntimeError("TrainingOnlyMinMaxScaler has not been fitted")
        return np.array(self._active_mask, dtype=np.bool_, copy=True)

    def fit(self, x_train_normal: FloatArray) -> TrainingOnlyMinMaxScaler:
        """Fit per-feature `[min, max]` strictly on `x_train_normal`."""
        arr = _validate_2d_features(x_train_normal, "x_train_normal")
        vmin = np.asarray(np.min(arr, axis=0), dtype=np.float64)
        vmax = np.asarray(np.max(arr, axis=0), dtype=np.float64)
        raw_span = np.asarray(vmax - vmin, dtype=np.float64)
        active = np.asarray(raw_span > 1e-8, dtype=np.bool_)
        safe_span = np.asarray(np.where(active, raw_span, 1.0), dtype=np.float64)
        self._min = vmin
        self._span = safe_span
        self._active_mask = active
        return self

    def transform(self, matrix: FloatArray) -> FloatArray:
        """Map active channels to `[0, 1]` on normal range (constant channels -> `0.5`)."""
        if self._min is None or self._span is None or self._active_mask is None:
            raise RuntimeError("TrainingOnlyMinMaxScaler must be fitted before transform")
        arr = _validate_2d_features(matrix, "matrix")
        if arr.shape[1] != self._min.shape[0]:
            raise ValueError("feature dimension mismatch in TrainingOnlyMinMaxScaler")
        scaled = (arr - self._min) / self._span
        inactive = ~self._active_mask
        if np.any(inactive):
            scaled[:, inactive] = (arr[:, inactive] - self._min[inactive]) + 0.5
        return np.asarray(scaled, dtype=np.float64)


class TranADPositionalEncoding(nn.Module):
    """Combined `sin + cos` positional encoding from `imperial-qore/TranAD`."""

    pe: torch.Tensor

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        """Precompute `sin + cos` positional table of shape `(max_len, 1, d_model)`."""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = pe + torch.sin(position * div_term) + torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x: torch.Tensor, pos: int = 0) -> torch.Tensor:
        """Add positional encoding to sequence-first tensor `(W, B, d_model)`."""
        out: torch.Tensor = self.dropout(x + self.pe[pos : pos + int(x.size(0)), :])
        return out


class TranADEncoderLayer(nn.Module):
    """Attention + LeakyReLU feedforward encoder layer from `imperial-qore/TranAD`."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 16,
        dropout: float = 0.1,
    ) -> None:
        """Construct self-attention and LeakyReLU FFN sublayers."""
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Encode `(W, B, d_model)` sequence."""
        src2, _ = self.self_attn(src, src, src, need_weights=False)
        src = src + self.dropout1(src2)
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        out: torch.Tensor = src + self.dropout2(ff)
        return out


class TranADDecoderLayer(nn.Module):
    """Self-attn + cross-attn + LeakyReLU decoder layer from `imperial-qore/TranAD`."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 16,
        dropout: float = 0.1,
    ) -> None:
        """Construct target self-attention, encoder cross-attention, and FFN."""
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Decode `(1, B, d_model)` target conditioned on `(W, B, d_model)` memory."""
        tgt2, _ = self.self_attn(tgt, tgt, tgt, need_weights=False)
        tgt = tgt + self.dropout1(tgt2)
        tgt2, _ = self.multihead_attn(tgt, memory, memory, need_weights=False)
        tgt = tgt + self.dropout2(tgt2)
        ff = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        out: torch.Tensor = tgt + self.dropout3(ff)
        return out


class TranADNetwork(nn.Module):
    """Two-phase self-conditioning TranAD network (`imperial-qore/TranAD`)."""

    def __init__(
        self,
        n_feats: int,
        window_size: int = 10,
        dim_feedforward: int = 16,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        """Construct shared encoder, dual decoders (`D1`, `D2`), and Sigmoid head."""
        super().__init__()
        if n_feats < 1 or window_size < 2:
            raise ValueError("n_feats >= 1 and window_size >= 2 are required")
        self.n_feats = n_feats
        self.n_window = window_size
        d_model = 2 * n_feats
        nhead = n_feats

        self.pos_encoder = TranADPositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_len=max(5000, window_size + 16),
        )
        self.encoder_layers = nn.ModuleList(
            [
                TranADEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder1_layers = nn.ModuleList(
            [
                TranADDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.decoder2_layers = nn.ModuleList(
            [
                TranADDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.fcn = nn.Sequential(
            nn.Linear(d_model, n_feats),
            nn.Sigmoid(),
        )

    def encode(
        self,
        src: torch.Tensor,
        focus_c: torch.Tensor,
        tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate `(src, focus_c)` along feature axis and run encoder stack."""
        combined = torch.cat((src, focus_c), dim=2) * math.sqrt(self.n_feats)
        memory = self.pos_encoder(combined)
        for layer in self.encoder_layers:
            memory = layer(memory)
        tgt_rep = tgt.repeat(1, 1, 2)
        return tgt_rep, memory

    def _run_decoder(
        self,
        layers: nn.ModuleList,
        tgt_rep: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        out = tgt_rep
        for layer in layers:
            out = layer(out, memory)
        proj: torch.Tensor = self.fcn(out)
        return proj

    def forward_with_focus(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute Phase 1 & Phase 2 returning `(x1, x2, focus_score_c2)`."""
        # Phase 1: zero focus score matrix
        c1 = torch.zeros_like(src)
        tgt1, mem1 = self.encode(src, c1, tgt)
        x1 = self._run_decoder(self.decoder1_layers, tgt1, mem1)

        # Phase 2: self-conditioning focus signal from Phase 1 reconstruction error
        focus_c2 = (x1 - src).pow(2)
        tgt2, mem2 = self.encode(src, focus_c2, tgt)
        x2 = self._run_decoder(self.decoder2_layers, tgt2, mem2)
        return x1, x2, focus_c2

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard two-phase forward returning `(x1, x2)` each `(1, B, D)`."""
        x1, x2, _ = self.forward_with_focus(src, tgt)
        return x1, x2


class TranADDetector(BenchmarkDetector):
    """Leakage-free TranAD detector implementing `BenchmarkDetector`."""

    def __init__(self, config: TranADConfig | None = None) -> None:
        """Initialise detector with `TranADConfig`."""
        cfg = config or TranADConfig()
        super().__init__(calibration_quantile=cfg.calibration_quantile)
        self._cfg = cfg
        self._device_info = select_compute_device(cfg.device)
        self._scaler: TrainingOnlyMinMaxScaler | None = None
        self._model: TranADNetwork | None = None
        self._training_history: dict[str, Any] = {}
        self._last_score_output: TranADScoreOutput | None = None

    @property
    def name(self) -> str:
        """Return canonical model identifier."""
        return "tranad"

    @property
    def config(self) -> TranADConfig:
        """Return immutable configuration."""
        return self._cfg

    @property
    def device_info(self) -> DeviceInfo:
        """Return accelerator diagnostics."""
        return self._device_info

    @property
    def training_history(self) -> dict[str, Any]:
        """Return epoch losses and early-stopping diagnostics."""
        return dict(self._training_history)

    @property
    def last_score_output(self) -> TranADScoreOutput | None:
        """Return full two-phase + per-feature attribution from most recent test call."""
        return self._last_score_output

    def get_parameters(self) -> dict[str, Any]:
        """Return JSON-serialisable hyperparameter dictionary."""
        params = asdict(self._cfg)
        params["selected_device"] = self._device_info.selected
        params["mps_available"] = self._device_info.mps_available
        params["is_apple_silicon"] = self._device_info.is_apple_silicon
        if self._training_history:
            params["epochs_trained"] = self._training_history.get("epochs_trained")
            params["early_stopped"] = self._training_history.get("early_stopped")
        return params

    def save_checkpoint(self, path: Path) -> None:
        """Save trained TranAD weights, scaler bounds, and calibrated threshold."""
        if self._model is None or self._scaler is None or self._threshold is None:
            raise RuntimeError("cannot save checkpoint before fit_and_calibrate")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self._cfg),
                "feature_names": list(self._feature_names),
                "threshold": self._threshold,
                "scaler_min": self._scaler.min_,
                "scaler_span": self._scaler.span_,
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

    def _select_working_device(self, input_dim: int) -> torch.device:
        """Probe TranAD `MultiheadAttention` on selected device with CPU fallback."""
        cfg = self._cfg
        dev = torch.device(self._device_info.selected)
        if dev.type == "cpu":
            return dev
        try:
            probe = TranADNetwork(
                n_feats=input_dim,
                window_size=cfg.window_size,
                dim_feedforward=cfg.dim_feedforward,
                num_encoder_layers=1,
                num_decoder_layers=1,
                dropout=0.0,
            ).to(dev)
            x_p = torch.zeros((cfg.window_size, 2, input_dim), device=dev)
            e_p = x_p[-1:, :, :]
            o1, o2 = probe(x_p, e_p)
            (o1.mean() + o2.mean()).backward()
            return dev
        except Exception:  # noqa: BLE001 - safe fallback to CPU if MPS attention op unsupported
            return torch.device("cpu")

    def _fit_normal(self, x_train_normal: FloatArray, feature_names: tuple[str, ...]) -> None:
        del feature_names
        cfg = self._cfg
        set_deterministic_seed(cfg.seed)

        scaler = TrainingOnlyMinMaxScaler().fit(x_train_normal)
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
        dev = self._select_working_device(input_dim)
        model = TranADNetwork(
            n_feats=input_dim,
            window_size=cfg.window_size,
            dim_feedforward=cfg.dim_feedforward,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            dropout=cfg.dropout,
        ).to(dev)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma
        )
        criterion = nn.MSELoss()
        rng = np.random.default_rng(cfg.seed)

        best_loss = float("inf")
        best_state = copy.deepcopy(model.state_dict())
        patience_left = cfg.early_stopping_patience
        train_losses: list[float] = []
        monitor_losses: list[float] = []
        early_stopped = False

        # Pre-transpose monitor tensor to (W, N_mon, D)
        mon_t = torch.from_numpy(np.transpose(mon_win, (1, 0, 2)).copy()).to(dev)
        mon_elem = mon_t[-1:, :, :]
        n_fit = int(fit_win.shape[0])

        for epoch_idx in range(cfg.epochs):
            n_step = float(epoch_idx + 1)
            w_phase1 = 1.0 / n_step
            w_phase2 = 1.0 - w_phase1

            model.train()
            perm = rng.permutation(n_fit)
            batch_losses: list[float] = []
            for start in range(0, n_fit, cfg.batch_size):
                batch_np = fit_win[perm[start : start + cfg.batch_size]]
                # Permute (B, W, D) -> (W, B, D) matching imperial-qore/TranAD
                win_t = torch.from_numpy(np.transpose(batch_np, (1, 0, 2)).copy()).to(dev)
                elem_t = win_t[-1:, :, :]
                optimizer.zero_grad(set_to_none=True)
                x1, x2 = model(win_t, elem_t)
                loss = w_phase1 * criterion(x1, elem_t) + w_phase2 * criterion(x2, elem_t)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu().item()))

            scheduler.step()
            train_losses.append(round(float(sum(batch_losses) / max(1, len(batch_losses))), 6))

            model.eval()
            with torch.inference_mode():
                m_x1, m_x2 = model(mon_t, mon_elem)
                # Evaluate converged two-phase reconstruction fidelity on normal monitor set
                mon_loss = float(
                    (0.5 * criterion(m_x1, mon_elem) + 0.5 * criterion(m_x2, mon_elem)).cpu().item()
                )
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
            "executed_device": str(dev),
            "channel_mode": cfg.channel_mode,
        }

    def score_details(
        self,
        matrix: FloatArray,
        feature_names: Sequence[str] | None = None,
    ) -> TranADScoreOutput:
        """Compute Phase-1, Phase-2, per-feature `(N, D)`, and overall `(N,)` scores."""
        if self._scaler is None or self._model is None:
            raise RuntimeError("TranADDetector must be fitted before scoring")
        arr = _validate_2d_features(matrix, "matrix")
        names = tuple(feature_names) if feature_names is not None else self._feature_names
        if len(names) != arr.shape[1]:
            raise ValueError("feature_names length must match matrix columns")

        x_scaled = self._scaler.transform(arr)
        n_rows, n_feats = x_scaled.shape
        p1_feat = np.zeros((n_rows, n_feats), dtype=np.float64)
        p2_feat = np.zeros((n_rows, n_feats), dtype=np.float64)

        dev = next(self._model.parameters()).device
        win_size = self._cfg.window_size
        self._model.eval()

        with torch.inference_mode():
            if self._cfg.channel_mode == "multivariate":
                for start, end, win_batch in iter_causal_window_batches(
                    x_scaled, win_size, self._cfg.eval_batch_size
                ):
                    win_t = torch.from_numpy(np.transpose(win_batch, (1, 0, 2)).copy()).to(dev)
                    elem_t = win_t[-1:, :, :]
                    x1, x2 = self._model(win_t, elem_t)
                    p1_feat[start:end, :] = (x1 - elem_t).pow(2)[0].cpu().numpy().astype(np.float64)
                    p2_feat[start:end, :] = (x2 - elem_t).pow(2)[0].cpu().numpy().astype(np.float64)
            else:
                active_mask = self._scaler.active_channel_mask
                inactive_mask = ~active_mask
                if np.any(inactive_mask):
                    # Inactive constant channels were mapped to 0.5 at training constant.
                    dev_sq = np.square(x_scaled[:, inactive_mask] - 0.5)
                    p1_feat[:, inactive_mask] = dev_sq
                    p2_feat[:, inactive_mask] = dev_sq

                active_idx = np.flatnonzero(active_mask)
                if active_idx.size > 0:
                    sub = np.ascontiguousarray(x_scaled[:, active_idx], dtype=np.float32)
                    n_act = int(active_idx.size)
                    padded = np.vstack([np.repeat(sub[:1], win_size - 1, axis=0), sub])
                    raw_view = sliding_window_view(padded, window_shape=win_size, axis=0)
                    row_step = max(1, self._cfg.eval_batch_size // n_act)
                    for start in range(0, n_rows, row_step):
                        end = min(n_rows, start + row_step)
                        b_len = end - start
                        # raw_view[start:end] has shape (b_len, n_act, W)
                        # Reshape to (b_len * n_act, W) -> transpose to (W, b_len * n_act, 1)
                        chunk = raw_view[start:end].reshape(b_len * n_act, win_size)
                        win_t = torch.from_numpy(
                            np.ascontiguousarray(chunk.T[:, :, None], dtype=np.float32)
                        ).to(dev)
                        elem_t = win_t[-1:, :, :]
                        x1, x2 = self._model(win_t, elem_t)
                        e1 = (x1 - elem_t).pow(2)[0, :, 0].cpu().numpy().reshape(b_len, n_act)
                        e2 = (x2 - elem_t).pow(2)[0, :, 0].cpu().numpy().reshape(b_len, n_act)
                        p1_feat[start:end, active_idx] = e1.astype(np.float64)
                        p2_feat[start:end, active_idx] = e2.astype(np.float64)

        if self._cfg.score_mode == "both_phases":
            per_feature = 0.5 * p1_feat + 0.5 * p2_feat
        else:
            per_feature = p2_feat.copy()

        overall = np.asarray(np.mean(per_feature, axis=1), dtype=np.float64)
        return TranADScoreOutput(
            overall_scores=overall,
            per_feature_scores=per_feature,
            phase1_per_feature_scores=p1_feat,
            phase2_per_feature_scores=p2_feat,
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
        self._last_score_output = out
        feat_means = np.mean(out.per_feature_scores, axis=0)
        p1_mean = float(np.mean(out.phase1_per_feature_scores))
        p2_mean = float(np.mean(out.phase2_per_feature_scores))
        top_idx = int(np.argmax(feat_means))
        diagnostics: dict[str, Any] = {
            "channel_mode": self._cfg.channel_mode,
            "score_mode": self._cfg.score_mode,
            "selected_device": self._training_history.get(
                "executed_device", self._device_info.selected
            ),
            "epochs_trained": self._training_history.get("epochs_trained"),
            "early_stopped": self._training_history.get("early_stopped"),
            "phase1_mean_error": round(p1_mean, 6),
            "phase2_mean_error": round(p2_mean, 6),
            "top_error_feature": feature_names[top_idx],
            "per_feature_mean_error": {
                fname: round(float(val), 6)
                for fname, val in zip(feature_names, feat_means, strict=True)
            },
        }
        return out.overall_scores, diagnostics


if __name__ == "__main__":
    import sys

    from app.m7_benchmark import main as _m7_main

    sys.exit(_m7_main())
