"""M10.1 Live Streaming Multi-Model ML / DL / TSFM & ADWIN Inference Worker.

Consumes verified telemetry from `research.telemetry.raw` under an independent
Kafka consumer group (`driftguard-m10-1-inference`), strips ground-truth labels
via `SensorReading.from_telemetry`, executes all M2–M8 detectors strictly aligned
to the 3-channel IoT telemetry schema `("temperature_c", "vibration_mm_s",
"pressure_bar")`, and publishes per-event score summaries to
`research.telemetry.model_scores` plus ADWIN concept-drift alerts to
`research.telemetry.drift`.

Research & Production Integrity Guarantees:
  1. Never passes ground-truth labels (`event.label`) into any detector.
  2. Never loads 8-feature SKAB or 1-feature TSB checkpoints onto the 3-channel
     live IoT stream (`validate_checkpoint_schema` raises `SchemaMismatchError`).
  3. Fits/calibrates 3-channel classical + deep models (`LSTM-AE`, `TCN-AE`,
     `TranAD`) and calibrates zero-shot TSFM thresholds (`Chronos-2`, `TimesFM 3.0`)
     strictly on normal-only training + normal-only validation series.
  4. Runs PyTorch DL & TSFM inference on Apple Silicon `mps` when available
     (with explicit `cpu` fallback and zero fabrication of GPU execution) inside
     `torch.inference_mode()`.
  5. Schedules heavy zero-shot TSFMs (`Chronos-2`, `TimesFM 3.0`) every
     `tsfm_every_n_events` ticks over strict causal history buffers so high-rate
     live ingestion remains non-blocking while retaining verified real scores.
  6. Separately reports `process_rss_mib`, `mps_allocated_mib`, `mps_driver_mib`,
     and `system_available_mib` without conflating allocator pools.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import pickle
import platform
import signal
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from confluent_kafka import Consumer, KafkaError, KafkaException, Producer

from app.config import ConfigError, Settings, load_settings
from app.detector import SensorReading
from app.drift_monitor import (
    MONITORED_FEATURES,
    AdwinConfig,
    MachineAdwinMonitor,
    _publish_drift_event,
)
from app.logging_config import configure_logging
from app.m4_models import (
    BenchmarkDetector,
    FloatArray,
    HalfSpaceTreesDetector,
    IsolationForestDetector,
    StaticThresholdBaseline,
)
from app.m5_lstm_autoencoder import (
    DeviceInfo,
    LSTMAutoencoderConfig,
    LSTMAutoencoderDetector,
    select_compute_device,
)
from app.m6_tcn_autoencoder import TCNAutoencoderConfig, TCNAutoencoderDetector
from app.m7_tranad import TranADConfig, TranADDetector
from app.m8_chronos2 import Chronos2Detector, is_chronos2_installed
from app.m8_timesfm import TimesFM3Detector, is_timesfm3_installed
from app.m8_tsfm_base import ForecastBackend, TSFMDetectorConfig, ZeroShotForecastDetector
from app.producer import DeliveryTracker, build_producer_config
from app.schemas import SchemaValidationError, format_utc_timestamp, parse_event_json

logger = logging.getLogger(__name__)

LIVE_FEATURE_NAMES: tuple[str, ...] = MONITORED_FEATURES
DEFAULT_INFERENCE_GROUP = "driftguard-m10-1-inference"
DEFAULT_M10_1_MODELS_DIR = Path("models/m10_1")


class SchemaMismatchError(ValueError):
    """Raised when a model checkpoint schema disagrees with live stream channels."""


def validate_checkpoint_schema(
    expected_features: Sequence[str],
    checkpoint_payload: Mapping[str, Any],
) -> None:
    """Reject checkpoints whose schema or dimensionality mismatches `expected_features`."""
    expected = list(expected_features)
    raw_features = checkpoint_payload.get("feature_names")
    if not isinstance(raw_features, (list, tuple)) or list(raw_features) != expected:
        raise SchemaMismatchError(
            f"Checkpoint feature schema {raw_features!r} does not match expected {expected!r}"
        )

    cfg = checkpoint_payload.get("config")
    if isinstance(cfg, Mapping) and cfg.get("channel_mode") == "independent_univariate":
        # In independent_univariate mode, the network itself operates on 1 channel at a time,
        # while feature_names lists all stream features calibrated together.
        return
    scaler_mean = checkpoint_payload.get("scaler_mean")
    scaler_min = checkpoint_payload.get("scaler_min")
    scale_vec = scaler_mean if scaler_mean is not None else scaler_min
    if scale_vec is not None and len(scale_vec) != len(expected):
        raise SchemaMismatchError(
            f"Checkpoint scaler dimensionality {len(scale_vec)} != expected {len(expected)}"
        )


def audit_legacy_checkpoints(
    models_root: Path = Path("models"),
    expected_features: Sequence[str] = LIVE_FEATURE_NAMES,
) -> list[dict[str, Any]]:
    """Inspect existing M5-M7 checkpoints and document exact compatibility status."""
    findings: list[dict[str, Any]] = []
    for milestone in ("m5", "m6", "m7"):
        root = models_root / milestone
        if not root.exists():
            continue
        for ckpt_path in sorted(root.rglob("*.pt")):
            record: dict[str, Any] = {
                "path": str(ckpt_path),
                "milestone": milestone,
                "compatible_with_3ch_iot": False,
                "reason": "uninspected",
                "feature_count": None,
            }
            try:
                payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if isinstance(payload, Mapping):
                    feats = payload.get("feature_names")
                    if isinstance(feats, (list, tuple)):
                        record["feature_count"] = len(feats)
                        record["feature_names_sample"] = list(feats[:5])
                    validate_checkpoint_schema(expected_features, payload)
                    record["compatible_with_3ch_iot"] = True
                    record["reason"] = "schema_match"
                else:
                    record["reason"] = "non_mapping_checkpoint"
            except SchemaMismatchError as exc:
                record["reason"] = str(exc)
            except Exception as exc:
                record["reason"] = f"inspection_error:{exc}"
            findings.append(record)
    return findings


def generate_normal_iot_splits(
    seed_train: int = 42,
    seed_val: int = 43,
    n_train: int = 240,
    n_val: int = 160,
) -> tuple[FloatArray, FloatArray]:
    """Generate deterministic normal-only 3-channel baseline splits matching Wokwi `normal`.

    Strictly zero anomalies or concept shifts are included in either split.
    """
    if n_train < 80 or n_val < 80:
        raise ValueError("n_train and n_val must both be >= 80 for causal context calibration")

    def _sample(seed: int, n_rows: int) -> FloatArray:
        rng = np.random.default_rng(seed)
        t = np.arange(n_rows, dtype=np.float64)
        # Stationary operating distribution aligned with `iot/wokwi_esp32/sketch.ino` normal mode
        temp = 46.5 + 0.25 * np.sin(2.0 * np.pi * t / 45.0) + rng.uniform(-0.60, 0.60, n_rows)
        vib = 2.10 + 0.05 * np.cos(2.0 * np.pi * t / 30.0) + rng.uniform(-0.18, 0.18, n_rows)
        press = 5.00 + 0.04 * np.sin(2.0 * np.pi * t / 60.0) + rng.uniform(-0.12, 0.12, n_rows)
        return np.column_stack([temp, vib, press]).astype(np.float64)

    return _sample(seed_train, n_train), _sample(seed_val, n_val)


def collect_hardware_diagnostics(device_request: str = "auto") -> dict[str, Any]:
    """Collect verified OS, Apple Silicon, PyTorch, and MPS hardware status."""
    dev_info: DeviceInfo = select_compute_device(device_request)
    return {
        "os_platform": platform.platform(),
        "machine_arch": platform.machine(),
        "torch_version": torch.__version__,
        "mps_built": dev_info.mps_built,
        "mps_available": dev_info.mps_available,
        "is_apple_silicon": dev_info.is_apple_silicon,
        "cuda_available": dev_info.cuda_available,
        "selected_dl_device": dev_info.selected,
        "fallback_reason": dev_info.fallback_reason,
    }


def collect_memory_metrics_mib(use_mps: bool | None = None) -> dict[str, float | None]:
    """Collect strictly separated process RSS, MPS allocator, and system available memory."""
    proc = psutil.Process(os.getpid())
    rss_mib = round(float(proc.memory_info().rss) / (1024.0 * 1024.0), 3)
    avail_mib = round(float(psutil.virtual_memory().available) / (1024.0 * 1024.0), 3)

    mps_ready = (
        torch.backends.mps.is_available()
        if use_mps is None
        else (use_mps and torch.backends.mps.is_available())
    )
    mps_alloc_mib: float | None = None
    mps_driver_mib: float | None = None
    if mps_ready:
        try:
            mps_alloc_mib = round(
                float(torch.mps.current_allocated_memory()) / (1024.0 * 1024.0), 3
            )
            mps_driver_mib = round(
                float(torch.mps.driver_allocated_memory()) / (1024.0 * 1024.0), 3
            )
        except Exception:
            mps_alloc_mib = None
            mps_driver_mib = None

    return {
        "process_rss_mib": rss_mib,
        "mps_allocated_mib": mps_alloc_mib,
        "mps_driver_mib": mps_driver_mib,
        "system_available_mib": avail_mib,
    }


@dataclass
class StreamingInferenceConfig:
    """Runtime configuration for M10.1 live multi-model streaming inference."""

    device: str = "auto"
    models_dir: Path = DEFAULT_M10_1_MODELS_DIR
    hf_cache_dir: str = "models/hf_cache"
    local_files_only: bool = True
    enable_tsfms: bool = True
    tsfm_every_n_events: int = 10
    tsfm_context_length: int = 64
    tsfm_prediction_length: int = 1
    ae_window_size: int = 16
    tranad_window_size: int = 10
    calibration_quantile: float = 0.99
    seed: int = 42
    adwin_delta: float = 0.002
    consumer_group: str = DEFAULT_INFERENCE_GROUP
    auto_offset_reset: str = "latest"
    force_retrain_3ch: bool = False
    empty_mps_cache_every_n: int = 10


@dataclass
class _ModelSlotState:
    model_id: str
    family: str
    channel_mode: str
    device: str
    status: str
    unavailable_reason: str | None = None
    last_score: float = 0.0
    last_threshold: float = 1.0
    last_anomaly: bool = False
    last_latency_ms: float = 0.0
    last_interval_violation: float | None = None
    eval_count: int = 0
    anomaly_count: int = 0
    latencies_ms: list[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.latencies_ms is None:
            self.latencies_ms = []


class LiveModelSuite:
    """Container managing all validated 3-channel classical, DL, TSFM, and ADWIN models."""

    def __init__(
        self,
        config: StreamingInferenceConfig,
        x_train_normal: FloatArray,
        x_val_normal: FloatArray,
        chronos_backend: ForecastBackend | None = None,
        timesfm_backend: ForecastBackend | None = None,
    ) -> None:
        """Fit/load and calibrate all 3-channel models strictly from normal splits."""
        self.config = config
        self.x_train_normal = x_train_normal
        self.x_val_normal = x_val_normal
        self.hardware = collect_hardware_diagnostics(config.device)
        self.legacy_audit = audit_legacy_checkpoints(Path("models"), LIVE_FEATURE_NAMES)

        self.classical_models: dict[str, BenchmarkDetector] = {}
        self.dl_models: dict[str, BenchmarkDetector] = {}
        self.tsfm_models: dict[str, ZeroShotForecastDetector] = {}
        self.slots: dict[str, _ModelSlotState] = {}
        self.manifest: dict[str, Any] = {
            "feature_schema": list(LIVE_FEATURE_NAMES),
            "train_normal_rows": int(x_train_normal.shape[0]),
            "val_normal_rows": int(x_val_normal.shape[0]),
            "calibration_quantile": config.calibration_quantile,
            "seed": config.seed,
            "hardware": self.hardware,
            "legacy_checkpoint_audit": self.legacy_audit,
            "models": {},
        }
        self._init_all_models(chronos_backend=chronos_backend, timesfm_backend=timesfm_backend)

    def _register_slot(
        self,
        model_id: str,
        family: str,
        channel_mode: str,
        device: str,
        threshold: float,
        status: str = "active",
        unavailable_reason: str | None = None,
    ) -> None:
        self.slots[model_id] = _ModelSlotState(
            model_id=model_id,
            family=family,
            channel_mode=channel_mode,
            device=device,
            status=status,
            unavailable_reason=unavailable_reason,
            last_threshold=float(threshold),
        )
        self.manifest["models"][model_id] = {
            "family": family,
            "channel_mode": channel_mode,
            "device": device,
            "status": status,
            "unavailable_reason": unavailable_reason,
            "calibrated_threshold": round(float(threshold), 6),
            "feature_schema": list(LIVE_FEATURE_NAMES),
        }

    def _init_all_models(
        self,
        chronos_backend: ForecastBackend | None,
        timesfm_backend: ForecastBackend | None,
    ) -> None:
        cfg = self.config
        cfg.models_dir.mkdir(parents=True, exist_ok=True)
        q = cfg.calibration_quantile

        # 1. Classical ML Baselines (M2 + M4)
        st = StaticThresholdBaseline(calibration_quantile=q)
        st_cal = st.fit_and_calibrate(self.x_train_normal, self.x_val_normal, LIVE_FEATURE_NAMES)
        self.classical_models["static_threshold"] = st
        self._register_slot(
            "static_threshold", "classical", "multivariate", "cpu", st_cal.calibrated_threshold
        )

        iso = IsolationForestDetector(n_estimators=100, seed=cfg.seed, calibration_quantile=q)
        iso_cal = iso.fit_and_calibrate(self.x_train_normal, self.x_val_normal, LIVE_FEATURE_NAMES)
        self.classical_models["isolation_forest"] = iso
        self._register_slot(
            "isolation_forest", "classical", "multivariate", "cpu", iso_cal.calibrated_threshold
        )

        hst = HalfSpaceTreesDetector(
            n_trees=25, height=8, window_size=64, seed=cfg.seed, calibration_quantile=q
        )
        hst_cal = hst.fit_and_calibrate(self.x_train_normal, self.x_val_normal, LIVE_FEATURE_NAMES)
        self.classical_models["half_space_trees"] = hst
        self._register_slot(
            "half_space_trees", "classical", "multivariate", "cpu", hst_cal.calibrated_threshold
        )

        # Save classical baseline metadata artifact
        with (cfg.models_dir / "classical_3ch.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "feature_names": list(LIVE_FEATURE_NAMES),
                    "static_threshold": st_cal.calibrated_threshold,
                    "isolation_forest": iso,
                    "half_space_trees_threshold": hst_cal.calibrated_threshold,
                },
                handle,
            )

        # 2. Deep Learning Models (LSTM-AE, TCN-AE, TranAD) trained strictly on 3ch normal data
        lstm = LSTMAutoencoderDetector(
            LSTMAutoencoderConfig(
                window_size=cfg.ae_window_size,
                hidden_dim=32,
                latent_dim=16,
                num_layers=2,
                batch_size=32,
                epochs=15,
                calibration_quantile=q,
                channel_mode="multivariate",
                seed=cfg.seed,
                device=cfg.device,
            )
        )
        lstm_cal = lstm.fit_and_calibrate(
            self.x_train_normal, self.x_val_normal, LIVE_FEATURE_NAMES
        )
        lstm_ckpt = cfg.models_dir / "lstm_autoencoder_3ch.pt"
        lstm.save_checkpoint(lstm_ckpt)
        validate_checkpoint_schema(
            LIVE_FEATURE_NAMES,
            torch.load(lstm_ckpt, map_location="cpu", weights_only=False),
        )
        self.dl_models["lstm_autoencoder"] = lstm
        self._register_slot(
            "lstm_autoencoder",
            "deep_learning",
            "multivariate",
            str(lstm.get_parameters().get("selected_device", "cpu")),
            lstm_cal.calibrated_threshold,
        )

        tcn = TCNAutoencoderDetector(
            TCNAutoencoderConfig(
                window_size=cfg.ae_window_size,
                hidden_channels=24,
                latent_dim=8,
                kernel_size=3,
                dilations=(1, 2, 4),
                batch_size=32,
                epochs=12,
                calibration_quantile=q,
                channel_mode="multivariate",
                seed=cfg.seed,
                device=cfg.device,
            )
        )
        tcn_cal = tcn.fit_and_calibrate(self.x_train_normal, self.x_val_normal, LIVE_FEATURE_NAMES)
        tcn_ckpt = cfg.models_dir / "tcn_autoencoder_3ch.pt"
        tcn.save_checkpoint(tcn_ckpt)
        validate_checkpoint_schema(
            LIVE_FEATURE_NAMES,
            torch.load(tcn_ckpt, map_location="cpu", weights_only=False),
        )
        self.dl_models["tcn_autoencoder"] = tcn
        self._register_slot(
            "tcn_autoencoder",
            "deep_learning",
            "multivariate",
            str(tcn.get_parameters().get("selected_device", "cpu")),
            tcn_cal.calibrated_threshold,
        )

        tranad = TranADDetector(
            TranADConfig(
                window_size=cfg.tranad_window_size,
                dim_feedforward=16,
                batch_size=32,
                epochs=10,
                calibration_quantile=q,
                channel_mode="multivariate",
                seed=cfg.seed,
                device=cfg.device,
            )
        )
        tranad_cal = tranad.fit_and_calibrate(
            self.x_train_normal, self.x_val_normal, LIVE_FEATURE_NAMES
        )
        tranad_ckpt = cfg.models_dir / "tranad_3ch.pt"
        tranad.save_checkpoint(tranad_ckpt)
        validate_checkpoint_schema(
            LIVE_FEATURE_NAMES,
            torch.load(tranad_ckpt, map_location="cpu", weights_only=False),
        )
        self.dl_models["tranad"] = tranad
        self._register_slot(
            "tranad",
            "deep_learning",
            "multivariate",
            str(tranad.get_parameters().get("selected_device", "cpu")),
            tranad_cal.calibrated_threshold,
        )

        # 3. Zero-Shot Foundation Models (M8 Chronos-2 + TimesFM 3.0) calibrated on x_val_normal
        if cfg.local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self._init_tsfm(
            model_id="chronos2_zero_shot",
            enabled=cfg.enable_tsfms or chronos_backend is not None,
            installed=(chronos_backend is not None) or is_chronos2_installed(),
            channel_mode="multivariate",
            ctor=lambda: Chronos2Detector(
                TSFMDetectorConfig(
                    model_id="amazon/chronos-2",
                    context_length=cfg.tsfm_context_length,
                    prediction_length=cfg.tsfm_prediction_length,
                    batch_size=32,
                    channel_mode="multivariate",
                    calibration_quantile=q,
                    device=cfg.device,
                    cache_dir=cfg.hf_cache_dir,
                    local_files_only=cfg.local_files_only,
                ),
                backend=chronos_backend,
            ),
        )
        self._init_tsfm(
            model_id="timesfm3_zero_shot",
            enabled=cfg.enable_tsfms or timesfm_backend is not None,
            installed=(timesfm_backend is not None) or is_timesfm3_installed(),
            channel_mode="independent_univariate",
            ctor=lambda: TimesFM3Detector(
                TSFMDetectorConfig(
                    model_id="google/timesfm-3.0-pytorch",
                    context_length=cfg.tsfm_context_length,
                    prediction_length=cfg.tsfm_prediction_length,
                    batch_size=32,
                    channel_mode="independent_univariate",
                    calibration_quantile=q,
                    device=cfg.device,
                    cache_dir=cfg.hf_cache_dir,
                    local_files_only=cfg.local_files_only,
                ),
                backend=timesfm_backend,
            ),
        )

        manifest_path = cfg.models_dir / "live_3ch_manifest.json"
        manifest_path.write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _init_tsfm(
        self,
        *,
        model_id: str,
        enabled: bool,
        installed: bool,
        channel_mode: str,
        ctor: Any,
    ) -> None:
        if not enabled:
            self._register_slot(
                model_id,
                "tsfm",
                channel_mode,
                "unavailable",
                1.0,
                status="unavailable",
                unavailable_reason="disabled_by_config",
            )
            return
        if not installed:
            self._register_slot(
                model_id,
                "tsfm",
                channel_mode,
                "unavailable",
                1.0,
                status="unavailable",
                unavailable_reason="package_not_installed",
            )
            return
        try:
            det: ZeroShotForecastDetector = ctor()
            cal = det.fit_and_calibrate(self.x_train_normal, self.x_val_normal, LIVE_FEATURE_NAMES)
            dev = str(det.get_parameters().get("selected_device", "cpu"))
            self.tsfm_models[model_id] = det
            self._register_slot(
                model_id, "tsfm", channel_mode, dev, cal.calibrated_threshold, status="active"
            )
        except Exception as exc:
            logger.warning(
                "zero-shot TSFM unavailable on this runtime",
                extra={"fields": {"model_id": model_id, "error": str(exc)}},
            )
            self._register_slot(
                model_id,
                "tsfm",
                channel_mode,
                "unavailable",
                1.0,
                status="unavailable",
                unavailable_reason=f"init_failed:{exc}",
            )


class StreamingInferenceEngine:
    """Per-event streaming inference & ADWIN engine producing Kafka model_scores & drift."""

    def __init__(
        self,
        suite: LiveModelSuite,
        kafka_settings: Settings,
        kafka_producer: Any | None = None,
    ) -> None:
        """Initialise rolling buffers, ADWIN monitor, and Kafka producer."""
        self.suite = suite
        self.cfg = suite.config
        self._kafka_settings = kafka_settings
        self._producer: Any = (
            kafka_producer
            if kafka_producer is not None
            else Producer(build_producer_config(kafka_settings))
        )
        self._delivery = DeliveryTracker()
        self._adwin = MachineAdwinMonitor(
            AdwinConfig(
                delta=self.cfg.adwin_delta,
                clock=16,
                max_buckets=5,
                min_window_length=10,
                grace_period=10,
            )
        )
        self._buffers: dict[str, deque[list[float]]] = {}
        self._max_buf = max(64, self.cfg.tsfm_context_length + 1, self.cfg.ae_window_size + 4)
        self.events_processed: int = 0
        self.drift_events_emitted: int = 0
        self.all_event_latencies_ms: list[float] = []
        self._last_memory_snapshot: dict[str, float | None] = collect_memory_metrics_mib()

    def _get_machine_buffer(self, machine_id: str) -> deque[list[float]]:
        buf = self._buffers.get(machine_id)
        if buf is None:
            buf = deque(maxlen=self._max_buf)
            self._buffers[machine_id] = buf
        return buf

    def process_telemetry_bytes(self, raw_bytes: bytes) -> dict[str, Any] | None:
        """Decode one Kafka telemetry message, run inference + ADWIN, and publish outputs."""
        try:
            event = parse_event_json(raw_bytes)
        except SchemaValidationError as exc:
            logger.warning("skipping malformed telemetry in inference worker: %s", exc)
            return None

        # Structurally strip ground-truth label before inference!
        reading = SensorReading.from_telemetry(event)
        vec = [reading.temperature_c, reading.vibration_mm_s, reading.pressure_bar]
        x_t = np.asarray([vec], dtype=np.float64)

        buf = self._get_machine_buffer(reading.machine_id)
        # Strict causal history prior to `x_t` (for zero-shot TSFM forecasting)
        if len(buf) >= 2:
            history_arr = np.asarray(list(buf), dtype=np.float64)
        else:
            history_arr = self.suite.x_train_normal[-self.cfg.tsfm_context_length :]
        buf.append(vec)
        ctx_and_curr = np.asarray(list(buf), dtype=np.float64)

        self.events_processed += 1
        t_start = time.perf_counter()
        models_out: dict[str, Any] = {}

        # 1. Classical models (M2, M4)
        for mid, det in self.suite.classical_models.items():
            t0 = time.perf_counter()
            if mid == "half_space_trees":
                # Stream sequentially one row at a time so learn_one executes exactly once/event
                res = det.score_and_predict(x_t, LIVE_FEATURE_NAMES)
            else:
                res = det.score_and_predict(
                    ctx_and_curr[-self.cfg.ae_window_size :], LIVE_FEATURE_NAMES
                )
            lat_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            score = float(res.scores[-1])
            thr = float(res.threshold)
            anom = bool(res.predictions[-1])
            slot = self.suite.slots[mid]
            slot.last_score = score
            slot.last_threshold = thr
            slot.last_anomaly = anom
            slot.last_latency_ms = lat_ms
            slot.eval_count += 1
            if anom:
                slot.anomaly_count += 1
            slot.latencies_ms.append(lat_ms)
            models_out[mid] = self._format_slot_output(slot, evaluated_this_event=True)

        # 2. Deep learning autoencoders / TranAD on Apple Silicon MPS / CPU under inference_mode
        dl_slice = ctx_and_curr[-self.cfg.ae_window_size :]
        with torch.inference_mode():
            for mid, det in self.suite.dl_models.items():
                t0 = time.perf_counter()
                res = det.score_and_predict(dl_slice, LIVE_FEATURE_NAMES)
                lat_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                score = float(res.scores[-1])
                thr = float(res.threshold)
                anom = bool(res.predictions[-1])
                slot = self.suite.slots[mid]
                slot.last_score = score
                slot.last_threshold = thr
                slot.last_anomaly = anom
                slot.last_latency_ms = lat_ms
                slot.eval_count += 1
                if anom:
                    slot.anomaly_count += 1
                slot.latencies_ms.append(lat_ms)
                models_out[mid] = self._format_slot_output(slot, evaluated_this_event=True)

            # 3. Scheduled zero-shot TSFMs (Chronos-2 + TimesFM 3.0) every N events
            should_run_tsfm = self.events_processed == 1 or (
                self.events_processed % max(1, self.cfg.tsfm_every_n_events) == 0
            )
            for mid in ("chronos2_zero_shot", "timesfm3_zero_shot"):
                slot = self.suite.slots[mid]
                if slot.status == "unavailable":
                    models_out[mid] = self._format_slot_output(slot, evaluated_this_event=False)
                    continue
                det_tsfm = self.suite.tsfm_models[mid]
                if should_run_tsfm:
                    t0 = time.perf_counter()
                    details = det_tsfm.forecast_and_score(
                        history_prefix=history_arr[-self.cfg.tsfm_context_length :],
                        target_matrix=x_t,
                        feature_names=LIVE_FEATURE_NAMES,
                    )
                    lat_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                    score = float(details.overall_scores[-1])
                    thr = float(det_tsfm._threshold or slot.last_threshold)
                    anom = bool(score >= thr)
                    slot.last_score = score
                    slot.last_threshold = thr
                    slot.last_anomaly = anom
                    slot.last_latency_ms = lat_ms
                    slot.last_interval_violation = float(details.interval_violation_scores[-1])
                    slot.eval_count += 1
                    if anom:
                        slot.anomaly_count += 1
                    slot.latencies_ms.append(lat_ms)
                    models_out[mid] = self._format_slot_output(slot, evaluated_this_event=True)
                else:
                    models_out[mid] = self._format_slot_output(
                        slot,
                        evaluated_this_event=False,
                        status_override="cached_skip",
                    )

        if self.suite.hardware["mps_available"] and (
            self.events_processed % max(1, self.cfg.empty_mps_cache_every_n) == 0
        ):
            try:
                torch.mps.empty_cache()
            except Exception:
                logger.debug("mps cache clear failed", exc_info=True)

        # 4. ADWIN Drift Detection (independent of ML detectors)
        _, _, drift_events = self._adwin.update(reading)
        drift_features: list[str] = []
        for de in drift_events:
            drift_features.append(de.feature)
            self.drift_events_emitted += 1
            _publish_drift_event(
                self._producer,
                self._kafka_settings.drift_topic,
                de,
                self._delivery,
            )

        total_lat_ms = round((time.perf_counter() - t_start) * 1000.0, 3)
        self.all_event_latencies_ms.append(total_lat_ms)
        mem = collect_memory_metrics_mib(use_mps=bool(self.suite.hardware["mps_available"]))
        self._last_memory_snapshot = mem

        score_payload: dict[str, Any] = {
            "event_id": event.event_id,
            "machine_id": event.machine_id,
            "device_id": event.device_id,
            "sequence_number": event.sequence_number,
            "timestamp": event.timestamp,
            "processed_utc": format_utc_timestamp(datetime.now(timezone.utc)),
            "source_type": event.source_type.value,
            "feature_schema": list(LIVE_FEATURE_NAMES),
            "telemetry": {
                "temperature_c": reading.temperature_c,
                "vibration_mm_s": reading.vibration_mm_s,
                "pressure_bar": reading.pressure_bar,
            },
            "hardware": self.suite.hardware,
            "memory_mib": mem,
            "total_inference_latency_ms": total_lat_ms,
            "models": models_out,
            "adwin": {
                "drift_detected": bool(drift_features),
                "drift_features": drift_features,
                "drift_events_total": self.drift_events_emitted,
            },
        }
        self._producer.produce(
            topic=self._kafka_settings.model_scores_topic,
            key=event.machine_id.encode("utf-8"),
            value=json.dumps(score_payload).encode("utf-8"),
            on_delivery=self._delivery.on_delivery,
        )
        self._producer.poll(0)
        return score_payload

    @staticmethod
    def _format_slot_output(
        slot: _ModelSlotState,
        *,
        evaluated_this_event: bool,
        status_override: str | None = None,
    ) -> dict[str, Any]:
        thr = max(slot.last_threshold, 1e-9)
        return {
            "family": slot.family,
            "channel_mode": slot.channel_mode,
            "device": slot.device,
            "status": status_override or slot.status,
            "unavailable_reason": slot.unavailable_reason,
            "evaluated_this_event": evaluated_this_event,
            "score": round(slot.last_score, 6),
            "threshold": round(slot.last_threshold, 6),
            "normalized_ratio": round(slot.last_score / thr, 4),
            "anomaly": slot.last_anomaly,
            "latency_ms": slot.last_latency_ms,
            "interval_violation_score": (
                round(slot.last_interval_violation, 6)
                if slot.last_interval_violation is not None
                else None
            ),
        }

    def build_runtime_summary(self) -> dict[str, Any]:
        """Return JSON summary of streaming latency, throughput, models, and memory."""
        models_summary: dict[str, Any] = {}
        for mid, slot in self.slots_snapshot().items():
            lats = np.asarray(slot.latencies_ms, dtype=np.float64)
            models_summary[mid] = {
                "family": slot.family,
                "channel_mode": slot.channel_mode,
                "device": slot.device,
                "status": slot.status,
                "unavailable_reason": slot.unavailable_reason,
                "threshold": round(slot.last_threshold, 6),
                "evaluations": slot.eval_count,
                "anomalies_flagged": slot.anomaly_count,
                "mean_latency_ms": round(float(np.mean(lats)), 3) if lats.size else None,
                "p95_latency_ms": round(float(np.percentile(lats, 95)), 3) if lats.size else None,
                "max_latency_ms": round(float(np.max(lats)), 3) if lats.size else None,
            }
        tot = np.asarray(self.all_event_latencies_ms, dtype=np.float64)
        return {
            "generated_utc": format_utc_timestamp(datetime.now(timezone.utc)),
            "events_processed": self.events_processed,
            "model_scores_delivered": self._delivery.delivered,
            "drift_events_emitted": self.drift_events_emitted,
            "hardware": copy.deepcopy(self.suite.hardware),
            "memory_mib": copy.deepcopy(self._last_memory_snapshot),
            "total_event_latency_ms": {
                "mean": round(float(np.mean(tot)), 3) if tot.size else None,
                "p95": round(float(np.percentile(tot, 95)), 3) if tot.size else None,
                "max": round(float(np.max(tot)), 3) if tot.size else None,
            },
            "models": models_summary,
            "legacy_checkpoint_audit": self.suite.legacy_audit,
        }

    def slots_snapshot(self) -> dict[str, _ModelSlotState]:
        """Return slot map."""
        return self.suite.slots

    def flush(self, timeout_s: float = 10.0) -> int:
        """Flush pending Kafka messages."""
        return int(self._producer.flush(timeout_s))


def run_streaming_inference_loop(
    engine: StreamingInferenceEngine,
    kafka_settings: Settings,
    *,
    duration_s: float | None = None,
    max_events: int | None = None,
    summary_output_path: Path | None = None,
    external_stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Consume `research.telemetry.raw` continuously and run streaming inference."""
    stop_event = external_stop_event if external_stop_event is not None else threading.Event()
    consumer = Consumer(
        {
            "bootstrap.servers": kafka_settings.bootstrap_servers,
            "group.id": engine.cfg.consumer_group,
            "client.id": f"{kafka_settings.client_id}-m10-1-inference",
            "auto.offset.reset": engine.cfg.auto_offset_reset,
            "enable.auto.commit": True,
        }
    )
    consumer.subscribe([kafka_settings.topic])

    def _sig(_signum: int, _frame: Any) -> None:
        stop_event.set()

    prev_int = signal.getsignal(signal.SIGINT)
    prev_term = signal.getsignal(signal.SIGTERM)
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, _sig)
        signal.signal(signal.SIGTERM, _sig)

    t0 = time.monotonic()
    try:
        while not stop_event.is_set():
            if duration_s is not None and (time.monotonic() - t0) >= duration_s:
                break
            if max_events is not None and engine.events_processed >= max_events:
                break
            msg = consumer.poll(0.25)
            if msg is None:
                continue
            err = msg.error()
            if err is not None:
                if err.code() != KafkaError._PARTITION_EOF:
                    raise KafkaException(err)
                continue
            raw = msg.value()
            if raw is not None:
                engine.process_telemetry_bytes(raw)
    finally:
        consumer.close()
        engine.flush(10.0)
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, prev_int)
            signal.signal(signal.SIGTERM, prev_term)

    summary = engine.build_runtime_summary()
    if summary_output_path is not None:
        summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_output_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the M10.1 streaming inference worker."""
    parser = argparse.ArgumentParser(
        description="DriftGuard-Kafka M10.1 Multi-Model Streaming Inference Worker."
    )
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    parser.add_argument("--tsfm-every-n-events", type=int, default=10)
    parser.add_argument("--disable-tsfms", action="store_true")
    parser.add_argument("--consumer-group", default=DEFAULT_INFERENCE_GROUP)
    parser.add_argument("--auto-offset-reset", default="latest", choices=("latest", "earliest"))
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for running the M10.1 streaming inference worker."""
    args = parse_args(argv)
    kafka_settings = load_settings()
    configure_logging(kafka_settings.log_level)

    inf_cfg = StreamingInferenceConfig(
        device=args.device,
        tsfm_every_n_events=args.tsfm_every_n_events,
        enable_tsfms=not args.disable_tsfms,
        consumer_group=args.consumer_group,
        auto_offset_reset=args.auto_offset_reset,
    )
    x_train, x_val = generate_normal_iot_splits()
    suite = LiveModelSuite(inf_cfg, x_train, x_val)
    engine = StreamingInferenceEngine(suite, kafka_settings)
    try:
        run_streaming_inference_loop(
            engine,
            kafka_settings,
            duration_s=args.duration_s,
            max_events=args.max_events,
            summary_output_path=args.summary_out,
        )
    except ConfigError as exc:
        logger.error("configuration error: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
