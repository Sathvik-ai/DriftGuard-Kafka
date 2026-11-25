"""Comparative unsupervised anomaly detection models for DriftGuard-Kafka M4.

Implements three anomaly detectors under a strict, label-free training and
validation calibration contract:

1. `StaticThresholdBaseline`: Multivariate static operating-bound + stuck-sensor
   detector preserving the M2 `ThresholdDetector` design across arbitrary sensor
   dimensionality.
2. `IsolationForestDetector`: Scikit-learn `IsolationForest` with `StandardScaler`
   fit strictly on normal training observations.
3. `HalfSpaceTreesDetector`: River streaming `HalfSpaceTrees` with online
   `MinMaxScaler`, explicit reference-window warm-up on normal training data,
   and strict `score_one` -> `learn_one` ordering on validation and test streams.

Ground-truth labels are structurally excluded from all `fit_and_calibrate` and
`score_and_predict` signatures. ADWIN remains independent; no adaptive model
retraining is performed in M4.
"""

from __future__ import annotations

import os
import time
import tracemalloc
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import psutil
from numpy.typing import NDArray
from river import anomaly as river_anomaly
from river import preprocessing as river_preprocessing
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class CalibrationSummary:
    """Metadata recorded when fitting on normal train and calibrating on normal val."""

    model_name: str
    train_samples: int
    val_samples: int
    n_features: int
    calibration_quantile: float
    calibrated_threshold: float
    fit_wall_time_s: float
    fit_cpu_time_s: float
    parameters: dict[str, Any]


@dataclass(frozen=True)
class InferenceResult:
    """Anomaly scores, binary predictions, and measured resource consumption."""

    scores: FloatArray
    predictions: IntArray
    threshold: float
    test_samples: int
    inference_wall_time_s: float
    mean_latency_ms: float
    throughput_events_per_sec: float | None
    cpu_time_s: float
    cpu_utilization_pct: float | None
    peak_tracemalloc_kib: float
    rss_memory_mib: float
    extra_diagnostics: dict[str, Any]


def _validate_2d_features(matrix: FloatArray, name: str) -> FloatArray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D float array, got shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def _measure_rss_mib() -> float:
    proc = psutil.Process(os.getpid())
    return round(float(proc.memory_info().rss) / (1024.0 * 1024.0), 3)


class BenchmarkDetector(ABC):
    """Abstract base class for leakage-free unsupervised benchmark detectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return canonical model identifier."""

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """Return JSON-serialisable hyperparameter dictionary."""

    @abstractmethod
    def _fit_normal(self, x_train_normal: FloatArray, feature_names: tuple[str, ...]) -> None:
        """Fit internal state strictly on normal training observations."""

    @abstractmethod
    def _score_validation(
        self, x_val_normal: FloatArray, feature_names: tuple[str, ...]
    ) -> FloatArray:
        """Produce continuous anomaly scores on normal validation observations."""

    @abstractmethod
    def _score_test(
        self, x_test: FloatArray, feature_names: tuple[str, ...]
    ) -> tuple[FloatArray, dict[str, Any]]:
        """Produce continuous anomaly scores on test observations."""

    def __init__(self, calibration_quantile: float = 0.99) -> None:
        """Initialise base detector with validation threshold quantile."""
        if not 0.0 < calibration_quantile < 1.0:
            raise ValueError("calibration_quantile must be strictly in (0, 1)")
        self._calibration_quantile = calibration_quantile
        self._threshold: float | None = None
        self._feature_names: tuple[str, ...] = ()

    @property
    def calibrated_threshold(self) -> float:
        """Return threshold calibrated from normal validation scores."""
        if self._threshold is None:
            raise RuntimeError(f"{self.name} has not been fitted and calibrated yet")
        return self._threshold

    def fit_and_calibrate(
        self,
        x_train_normal: FloatArray,
        x_val_normal: FloatArray,
        feature_names: Sequence[str],
    ) -> CalibrationSummary:
        """Fit on `x_train_normal` and calibrate threshold on `x_val_normal`."""
        train_arr = _validate_2d_features(x_train_normal, "x_train_normal")
        val_arr = _validate_2d_features(x_val_normal, "x_val_normal")
        if train_arr.shape[1] != val_arr.shape[1]:
            raise ValueError("x_train_normal and x_val_normal feature dimensions do not match")
        if len(feature_names) != train_arr.shape[1]:
            raise ValueError("feature_names length must match number of columns")

        self._feature_names = tuple(feature_names)
        wall_start_ns = time.perf_counter_ns()
        cpu_start_s = time.process_time()

        self._fit_normal(train_arr, self._feature_names)
        val_scores = np.asarray(
            self._score_validation(val_arr, self._feature_names), dtype=np.float64
        )
        if val_scores.shape != (val_arr.shape[0],):
            raise RuntimeError("validation score array length mismatch")

        self._threshold = float(np.quantile(val_scores, self._calibration_quantile))
        fit_wall_s = (time.perf_counter_ns() - wall_start_ns) / 1_000_000_000.0
        fit_cpu_s = time.process_time() - cpu_start_s

        return CalibrationSummary(
            model_name=self.name,
            train_samples=int(train_arr.shape[0]),
            val_samples=int(val_arr.shape[0]),
            n_features=int(train_arr.shape[1]),
            calibration_quantile=self._calibration_quantile,
            calibrated_threshold=round(self._threshold, 6),
            fit_wall_time_s=round(fit_wall_s, 6),
            fit_cpu_time_s=round(fit_cpu_s, 6),
            parameters=self.get_parameters(),
        )

    def score_and_predict(
        self,
        x_test: FloatArray,
        feature_names: Sequence[str] | None = None,
    ) -> InferenceResult:
        """Score `x_test`, apply calibrated threshold, and record resource metrics."""
        if self._threshold is None:
            raise RuntimeError(f"{self.name} must be calibrated before calling score_and_predict")
        test_arr = _validate_2d_features(x_test, "x_test")
        names = tuple(feature_names) if feature_names is not None else self._feature_names
        if len(names) != test_arr.shape[1]:
            raise ValueError("feature_names length does not match x_test columns")

        was_tracing = tracemalloc.is_tracing()
        if not was_tracing:
            tracemalloc.start()
        else:
            tracemalloc.reset_peak()

        cpu_start_s = time.process_time()
        wall_start_ns = time.perf_counter_ns()
        try:
            raw_scores, diagnostics = self._score_test(test_arr, names)
            scores = np.asarray(raw_scores, dtype=np.float64)
            predictions = (scores > self._threshold).astype(np.int64)
        finally:
            wall_ns = max(1, time.perf_counter_ns() - wall_start_ns)
            cpu_s = max(0.0, time.process_time() - cpu_start_s)
            _, peak_bytes = tracemalloc.get_traced_memory()
            if not was_tracing:
                tracemalloc.stop()

        wall_s = wall_ns / 1_000_000_000.0
        n_samples = int(test_arr.shape[0])
        mean_latency_ms = (wall_ns / n_samples) / 1_000_000.0
        throughput = round(n_samples / wall_s, 2) if wall_s > 0 else None
        cpu_pct = round(100.0 * (cpu_s / wall_s), 2) if wall_s > 0 else None

        return InferenceResult(
            scores=scores,
            predictions=predictions,
            threshold=self._threshold,
            test_samples=n_samples,
            inference_wall_time_s=round(wall_s, 6),
            mean_latency_ms=round(mean_latency_ms, 6),
            throughput_events_per_sec=throughput,
            cpu_time_s=round(cpu_s, 6),
            cpu_utilization_pct=cpu_pct,
            peak_tracemalloc_kib=round(peak_bytes / 1024.0, 3),
            rss_memory_mib=_measure_rss_mib(),
            extra_diagnostics=diagnostics,
        )


class StaticThresholdBaseline(BenchmarkDetector):
    """Multivariate static threshold + stuck-sensor baseline detector.

    Preserves the M2 `ThresholdDetector` design (static per-channel operating
    bounds plus a consecutive identical-reading `stuck_window` check) while
    supporting arbitrary multivariate sensor channels. Bounds are fixed from
    `x_train_normal` only and never adapt during validation or testing.
    """

    def __init__(
        self,
        z_bound: float = 3.0,
        stuck_window: int = 4,
        calibration_quantile: float = 0.99,
    ) -> None:
        """Configure static z-bound multiplier, stuck window, and val quantile."""
        super().__init__(calibration_quantile=calibration_quantile)
        if z_bound <= 0:
            raise ValueError("z_bound must be positive")
        if stuck_window < 2:
            raise ValueError("stuck_window must be >= 2")
        self._z_bound = z_bound
        self._stuck_window = stuck_window
        self._center: FloatArray | None = None
        self._half_width: FloatArray | None = None

    @property
    def name(self) -> str:
        """Return canonical model identifier."""
        return "static_threshold"

    def get_parameters(self) -> dict[str, Any]:
        """Return static threshold baseline hyperparameters."""
        return {
            "z_bound": self._z_bound,
            "stuck_window": self._stuck_window,
            "calibration_quantile": self._calibration_quantile,
            "adaptive": False,
        }

    def _fit_normal(self, x_train_normal: FloatArray, feature_names: tuple[str, ...]) -> None:
        del feature_names
        mean = np.mean(x_train_normal, axis=0)
        std = np.std(x_train_normal, axis=0)
        safe_std = np.where(std > 1e-9, std, 1.0)
        low = mean - self._z_bound * safe_std
        high = mean + self._z_bound * safe_std
        self._center = 0.5 * (low + high)
        self._half_width = np.maximum(0.5 * (high - low), 1e-9)

    def _compute_scores(self, matrix: FloatArray) -> tuple[FloatArray, int]:
        if self._center is None or self._half_width is None:
            raise RuntimeError("StaticThresholdBaseline is not fitted")
        # Normalized static bound ratio: > 1.0 means outside [low, high] static operating band.
        norm_dev = np.abs(matrix - self._center) / self._half_width
        max_channel_ratio: FloatArray = np.max(norm_dev, axis=1).astype(np.float64)

        stuck_hits = 0
        streak = 1
        scores = max_channel_ratio.copy()
        for idx in range(1, matrix.shape[0]):
            if np.array_equal(matrix[idx], matrix[idx - 1]):
                streak += 1
            else:
                streak = 1
            if streak >= self._stuck_window:
                stuck_hits += 1
                # Ensure stuck-sensor streaks exceed normal bound ratios cleanly.
                scores[idx] = max(float(scores[idx]), 2.0)
        return scores, stuck_hits

    def _score_validation(
        self, x_val_normal: FloatArray, feature_names: tuple[str, ...]
    ) -> FloatArray:
        del feature_names
        scores, _ = self._compute_scores(x_val_normal)
        return scores

    def _score_test(
        self, x_test: FloatArray, feature_names: tuple[str, ...]
    ) -> tuple[FloatArray, dict[str, Any]]:
        del feature_names
        scores, stuck_hits = self._compute_scores(x_test)
        return scores, {"stuck_sensor_triggers": stuck_hits}


class IsolationForestDetector(BenchmarkDetector):
    """Scikit-learn `IsolationForest` unsupervised anomaly detector."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: str | int | float = "auto",
        seed: int = 42,
        calibration_quantile: float = 0.99,
    ) -> None:
        """Configure Isolation Forest ensemble parameters and random seed."""
        super().__init__(calibration_quantile=calibration_quantile)
        self._n_estimators = n_estimators
        self._max_samples = max_samples
        self._seed = seed
        self._scaler: StandardScaler | None = None
        self._model: IsolationForest | None = None

    @property
    def name(self) -> str:
        """Return canonical model identifier."""
        return "isolation_forest"

    def get_parameters(self) -> dict[str, Any]:
        """Return Isolation Forest hyperparameters."""
        return {
            "n_estimators": self._n_estimators,
            "max_samples": self._max_samples,
            "random_state": self._seed,
            "n_jobs": 1,
            "scaler": "StandardScaler",
            "calibration_quantile": self._calibration_quantile,
        }

    def _fit_normal(self, x_train_normal: FloatArray, feature_names: tuple[str, ...]) -> None:
        del feature_names
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_train_normal)
        model = IsolationForest(
            n_estimators=self._n_estimators,
            max_samples=self._max_samples,
            random_state=self._seed,
            n_jobs=1,
        )
        model.fit(x_scaled)
        self._scaler = scaler
        self._model = model

    def _score_validation(
        self, x_val_normal: FloatArray, feature_names: tuple[str, ...]
    ) -> FloatArray:
        del feature_names
        if self._scaler is None or self._model is None:
            raise RuntimeError("IsolationForestDetector is not fitted")
        x_scaled = self._scaler.transform(x_val_normal)
        return np.asarray(-self._model.score_samples(x_scaled), dtype=np.float64)

    def _score_test(
        self, x_test: FloatArray, feature_names: tuple[str, ...]
    ) -> tuple[FloatArray, dict[str, Any]]:
        del feature_names
        if self._scaler is None or self._model is None:
            raise RuntimeError("IsolationForestDetector is not fitted")
        x_scaled = self._scaler.transform(x_test)
        scores = np.asarray(-self._model.score_samples(x_scaled), dtype=np.float64)
        return scores, {}


class HalfSpaceTreesDetector(BenchmarkDetector):
    """River online `HalfSpaceTrees` detector with warm-up and score-before-learn.

    Research guarantees:
    1. Feature scaling: `river.preprocessing.MinMaxScaler` maps normal features
       into `[0, 1]` matching `HalfSpaceTrees`'s unit-hypercube partition assumption.
    2. Warm-up: `effective_window_size = min(window_size, len(x_train_normal))`
       guarantees at least one full mass-pivot window completes on normal training
       observations before validation scoring begins (`_first_window == False`).
    3. Score-before-learn: on every validation and test observation, features are
       scaled with `scaler.transform_one(x)`, scored with `hst.score_one(x_scaled)`,
       and only afterwards passed to `hst.learn_one(x_scaled)` and
       `scaler.learn_one(x)`.
    """

    def __init__(
        self,
        n_trees: int = 25,
        height: int = 6,
        window_size: int = 100,
        seed: int = 42,
        calibration_quantile: float = 0.99,
    ) -> None:
        """Configure River Half-Space Trees hyperparameters and seed."""
        super().__init__(calibration_quantile=calibration_quantile)
        self._n_trees = n_trees
        self._height = height
        self._requested_window_size = window_size
        self._effective_window_size = window_size
        self._seed = seed
        self._scaler: Any = None
        self._hst: Any = None
        self._operation_trace: list[tuple[str, int]] = []
        self._record_trace: bool = False

    @property
    def name(self) -> str:
        """Return canonical model identifier."""
        return "half_space_trees"

    @property
    def operation_trace(self) -> list[tuple[str, int]]:
        """Return recorded `('score'|'learn', sample_idx)` trace when enabled."""
        return list(self._operation_trace)

    def enable_trace(self, enabled: bool = True) -> None:
        """Enable or disable per-sample `score` / `learn` call-order tracing."""
        self._record_trace = enabled
        if not enabled:
            self._operation_trace.clear()

    def get_parameters(self) -> dict[str, Any]:
        """Return Half-Space Trees hyperparameters including effective warm-up window."""
        return {
            "n_trees": self._n_trees,
            "height": self._height,
            "requested_window_size": self._requested_window_size,
            "effective_window_size": self._effective_window_size,
            "seed": self._seed,
            "scaler": "river.preprocessing.MinMaxScaler",
            "evaluation_order": "transform_one -> score_one -> learn_one -> scaler.learn_one",
            "calibration_quantile": self._calibration_quantile,
        }

    def _fit_normal(self, x_train_normal: FloatArray, feature_names: tuple[str, ...]) -> None:
        n_train = int(x_train_normal.shape[0])
        self._effective_window_size = max(2, min(self._requested_window_size, n_train))
        scaler: Any = river_preprocessing.MinMaxScaler()
        hst: Any = river_anomaly.HalfSpaceTrees(
            n_trees=self._n_trees,
            height=self._height,
            window_size=self._effective_window_size,
            seed=self._seed,
        )

        # Step 1: Fit initial normal feature range in MinMaxScaler on normal training prefix.
        for row in x_train_normal:
            x_dict = {feat: float(val) for feat, val in zip(feature_names, row, strict=True)}
            scaler.learn_one(x_dict)

        # Step 2: Warm up Half-Space Trees across normal training observations so at least
        # one full reference window pivots (`hst._first_window` becomes False).
        for row in x_train_normal:
            x_dict = {feat: float(val) for feat, val in zip(feature_names, row, strict=True)}
            x_scaled = scaler.transform_one(x_dict)
            hst.learn_one(x_scaled)

        self._scaler = scaler
        self._hst = hst
        self._operation_trace.clear()

    def _stream_score_then_learn(
        self, matrix: FloatArray, feature_names: tuple[str, ...]
    ) -> tuple[FloatArray, int]:
        if self._scaler is None or self._hst is None:
            raise RuntimeError("HalfSpaceTreesDetector is not fitted")
        scaler = self._scaler
        hst = self._hst
        n_rows = int(matrix.shape[0])
        scores = np.empty(n_rows, dtype=np.float64)

        for idx, row in enumerate(matrix):
            x_dict = {feat: float(val) for feat, val in zip(feature_names, row, strict=True)}
            x_scaled = scaler.transform_one(x_dict)
            if self._record_trace:
                self._operation_trace.append(("score", idx))
            scores[idx] = float(hst.score_one(x_scaled))
            if self._record_trace:
                self._operation_trace.append(("learn", idx))
            hst.learn_one(x_scaled)
            scaler.learn_one(x_dict)

        return scores, n_rows

    def _score_validation(
        self, x_val_normal: FloatArray, feature_names: tuple[str, ...]
    ) -> FloatArray:
        scores, _ = self._stream_score_then_learn(x_val_normal, feature_names)
        return scores

    def _score_test(
        self, x_test: FloatArray, feature_names: tuple[str, ...]
    ) -> tuple[FloatArray, dict[str, Any]]:
        warmup_complete = bool(self._hst is not None and not self._hst._first_window)
        scores, streamed = self._stream_score_then_learn(x_test, feature_names)
        return scores, {
            "warmup_complete_before_test": warmup_complete,
            "effective_window_size": self._effective_window_size,
            "streamed_samples": streamed,
            "score_before_learn": True,
        }


def build_default_detectors(
    seed: int = 42,
    calibration_quantile: float = 0.99,
) -> list[BenchmarkDetector]:
    """Instantiate the three M4 benchmark detectors with reproducible seed."""
    return [
        StaticThresholdBaseline(
            z_bound=3.0,
            stuck_window=4,
            calibration_quantile=calibration_quantile,
        ),
        IsolationForestDetector(
            n_estimators=100,
            max_samples="auto",
            seed=seed,
            calibration_quantile=calibration_quantile,
        ),
        HalfSpaceTreesDetector(
            n_trees=25,
            height=6,
            window_size=100,
            seed=seed,
            calibration_quantile=calibration_quantile,
        ),
    ]
