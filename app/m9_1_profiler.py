"""Isolated subprocess resource profiler for DriftGuard-Kafka M9.1 (Task 4).

Measures each model inside a fresh, isolated Python subprocess so interpreter
overhead, dataset memory, model-load memory, inference peak memory, and Apple
Silicon MPS GPU allocations are cleanly separated and never contaminated by
previously loaded models in the same process.

Metric definitions (all explicitly documented in M9.1 reports):
  - `interpreter_baseline_rss_mib`: Process RSS (MiB) after Python interpreter
    startup + loading the single profiling split into NumPy, BEFORE importing
    or instantiating the model.
  - `model_load_rss_delta_mib`: `rss_after_load - interpreter_baseline_rss_mib`
    (captures model library import + weight loading into RAM).
  - `inference_peak_rss_mib`: Process RSS (MiB) immediately after scoring `x_test`.
  - `inference_rss_delta_mib`: `max(0, inference_peak_rss_mib - rss_after_load)`.
  - `mps_current_allocated_mib`: `torch.mps.current_allocated_memory()` (MiB)
    on Apple Silicon MPS (`0.0` on CPU).
  - `mps_driver_allocated_mib`: `torch.mps.driver_allocated_memory()` (MiB)
    on Apple Silicon MPS (`0.0` on CPU).
  - `peak_tracemalloc_kib`: Peak Python heap allocation (KiB) during `x_test`
    scoring via `tracemalloc`.
  - `offline_batched_mean_latency_ms`: `(inference_wall_ns / n_test) / 1e6`
    measured in offline batched mode (NOT single-event Kafka broker latency).
  - `offline_batched_throughput_ev_s`: `n_test / inference_wall_s` in offline
    batched mode.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from app.datasets import ChronologicalSplit, load_skab_suite, load_tsb_ad_m_suite

PROFILER_MODELS: tuple[str, ...] = (
    "static_threshold",
    "isolation_forest",
    "half_space_trees",
    "lstm_autoencoder",
    "tcn_autoencoder",
    "tranad",
    "chronos2_zero_shot",
    "timesfm3_zero_shot",
)

RESOURCE_METRIC_DEFINITIONS: dict[str, str] = {
    "interpreter_baseline_rss_mib": (
        "Fresh Python subprocess RSS (MiB) after loading the profiling split into NumPy, "
        "measured strictly BEFORE importing or instantiating the detector."
    ),
    "model_load_rss_delta_mib": (
        "Incremental RSS (MiB) added by importing the model backend and loading weights "
        "(rss_after_load - interpreter_baseline_rss_mib)."
    ),
    "inference_peak_rss_mib": (
        "Total subprocess RSS (MiB) immediately following offline-batched test inference."
    ),
    "inference_rss_delta_mib": (
        "Incremental RSS (MiB) added during test inference beyond model load "
        "(max(0, inference_peak_rss_mib - rss_after_load))."
    ),
    "mps_current_allocated_mib": (
        "PyTorch Apple Silicon MPS active tensor memory (MiB) via "
        "`torch.mps.current_allocated_memory()` (0.0 on CPU)."
    ),
    "mps_driver_allocated_mib": (
        "PyTorch Apple Silicon MPS driver-reserved memory (MiB) via "
        "`torch.mps.driver_allocated_memory()` (0.0 on CPU)."
    ),
    "peak_tracemalloc_kib": (
        "Peak Python heap allocation (KiB) captured by `tracemalloc` strictly during "
        "the test inference window."
    ),
    "offline_batched_mean_latency_ms": (
        "Mean wall-clock time per test observation (ms) during offline batched scoring "
        "(`inference_wall_ms / n_test`). Excludes Kafka broker/network serialization."
    ),
    "offline_batched_throughput_ev_s": (
        "Offline batched scoring throughput (`n_test / inference_wall_s`)."
    ),
}


def _rss_mib() -> float:
    return round(float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 * 1024.0), 3)


def _mps_mem_mib() -> tuple[float, float]:
    try:
        import torch

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            cur = float(torch.mps.current_allocated_memory()) / (1024.0 * 1024.0)
            drv = float(torch.mps.driver_allocated_memory()) / (1024.0 * 1024.0)
            return round(cur, 3), round(drv, 3)
    except Exception:
        return 0.0, 0.0
    return 0.0, 0.0


def _load_target_split(
    suite: str,
    skab_dir: Path,
    tsb_dir: Path,
) -> ChronologicalSplit:
    if suite == "SKAB":
        bundle = load_skab_suite(skab_dir)
        # First fault experiment (deterministic)
        for s in bundle.splits:
            if not s.is_anomaly_free:
                return s
        return bundle.splits[0]
    bundle = load_tsb_ad_m_suite(tsb_dir, max_per_domain=1)
    return bundle.splits[0]


def profile_single_model_in_process(
    model_name: str,
    split: ChronologicalSplit,
    device: str = "auto",
    cache_dir: str = "models/hf_cache",
    local_files_only: bool = True,
    mock_backends: bool = False,
) -> dict[str, Any]:
    """Measure baseline RSS, model-load delta, inference peak RSS, MPS, and latency."""
    baseline_rss = _rss_mib()
    load_t0 = time.perf_counter()
    selected_dev = "cpu"

    if model_name in ("chronos2_zero_shot", "timesfm3_zero_shot"):
        from app.m8_benchmark import DeterministicPersistenceBackend
        from app.m8_chronos2 import CHRONOS2_DEFAULT_MODEL_ID, Chronos2Detector
        from app.m8_timesfm import TIMESFM3_DEFAULT_MODEL_ID, TimesFM3Detector
        from app.m8_tsfm_base import ForecastBackend, TSFMDetectorConfig, ZeroShotForecastDetector
        from app.m9_1_tracks import forecast_chunked_tsfm_split

        backend: ForecastBackend | None = (
            DeterministicPersistenceBackend(f"mock/{model_name}") if mock_backends else None
        )
        tsfm_det: ZeroShotForecastDetector
        if model_name == "chronos2_zero_shot":
            cfg = TSFMDetectorConfig(
                model_id=CHRONOS2_DEFAULT_MODEL_ID,
                context_length=128,
                prediction_length=64,
                batch_size=64,
                device=device,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            tsfm_det = Chronos2Detector(cfg, backend=backend)
        else:
            cfg = TSFMDetectorConfig(
                model_id=TIMESFM3_DEFAULT_MODEL_ID,
                context_length=128,
                prediction_length=64,
                batch_size=64,
                device=device,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            tsfm_det = TimesFM3Detector(cfg, backend=backend)

        resolved = tsfm_det._resolve_backend()
        selected_dev = resolved.device_info.selected
        load_s = round(time.perf_counter() - load_t0, 4)
        after_load_rss = _rss_mib()

        tsfm_det._feature_names = split.feature_names
        tsfm_det._fit_normal(split.x_train_normal, split.feature_names)
        if tsfm_det._train_tail is None:
            raise RuntimeError("training tail missing")
        pre_test = np.vstack([tsfm_det._train_tail, split.x_val_normal])[-cfg.context_length :]

        tracemalloc.start()
        t0_ns = time.perf_counter_ns()
        _details, mode_lbl = forecast_chunked_tsfm_split(
            tsfm_det,
            pre_test,
            split.x_test,
            split.feature_names,
            track="multivariate",
        )
        wall_ns = max(1, time.perf_counter_ns() - t0_ns)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        inf_rss = _rss_mib()
        mps_cur, mps_drv = _mps_mem_mib()
        n_test = int(split.x_test.shape[0])
        wall_s = wall_ns / 1_000_000_000.0
        return {
            "model": model_name,
            "dataset_suite": split.dataset_suite,
            "experiment_id": split.experiment_id,
            "channel_processing_mode": mode_lbl,
            "selected_device": selected_dev,
            "test_samples": n_test,
            "n_features": len(split.feature_names),
            "model_load_time_s": load_s,
            "interpreter_baseline_rss_mib": baseline_rss,
            "model_load_rss_delta_mib": round(max(0.0, after_load_rss - baseline_rss), 3),
            "inference_peak_rss_mib": inf_rss,
            "inference_rss_delta_mib": round(max(0.0, inf_rss - after_load_rss), 3),
            "mps_current_allocated_mib": mps_cur,
            "mps_driver_allocated_mib": mps_drv,
            "peak_tracemalloc_kib": round(peak_bytes / 1024.0, 3),
            "offline_batched_mean_latency_ms": round((wall_ns / n_test) / 1_000_000.0, 6),
            "offline_batched_throughput_ev_s": round(n_test / wall_s, 2),
        }

    # Non-TSFM models
    from app.m4_models import (
        BenchmarkDetector,
        HalfSpaceTreesDetector,
        IsolationForestDetector,
        StaticThresholdBaseline,
    )

    det: BenchmarkDetector
    if model_name == "static_threshold":
        det = StaticThresholdBaseline(calibration_quantile=0.99)
    elif model_name == "isolation_forest":
        det = IsolationForestDetector(seed=42, calibration_quantile=0.99)
    elif model_name == "half_space_trees":
        det = HalfSpaceTreesDetector(seed=42, calibration_quantile=0.99)
    elif model_name == "lstm_autoencoder":
        from app.m5_lstm_autoencoder import LSTMAutoencoderConfig, LSTMAutoencoderDetector

        det = LSTMAutoencoderDetector(
            LSTMAutoencoderConfig(
                window_size=32,
                hidden_dim=32,
                latent_dim=16,
                num_layers=2,
                epochs=5,
                seed=42,
                device=device,
                channel_mode="multivariate",
            )
        )
    elif model_name == "tcn_autoencoder":
        from app.m6_tcn_autoencoder import TCNAutoencoderConfig, TCNAutoencoderDetector

        det = TCNAutoencoderDetector(
            TCNAutoencoderConfig(
                window_size=32,
                hidden_channels=24,
                latent_dim=8,
                dilations=(1, 2, 4),
                epochs=5,
                seed=42,
                device=device,
                channel_mode="multivariate",
            )
        )
    elif model_name == "tranad":
        from app.m7_tranad import TranADConfig, TranADDetector

        det = TranADDetector(
            TranADConfig(
                window_size=10,
                dim_feedforward=16,
                epochs=5,
                seed=42,
                device=device,
                channel_mode="multivariate",
            )
        )
    else:
        raise ValueError(f"Unknown profiler model: {model_name!r}")

    load_s = round(time.perf_counter() - load_t0, 4)
    after_load_rss = _rss_mib()

    det.fit_and_calibrate(split.x_train_normal, split.x_val_normal, split.feature_names)
    after_fit_rss = _rss_mib()
    inf = det.score_and_predict(split.x_test, split.feature_names)
    inf_rss = _rss_mib()
    mps_cur, mps_drv = _mps_mem_mib()
    selected_dev = str(inf.extra_diagnostics.get("selected_device", "cpu"))

    return {
        "model": model_name,
        "dataset_suite": split.dataset_suite,
        "experiment_id": split.experiment_id,
        "channel_processing_mode": "multivariate_native",
        "selected_device": selected_dev,
        "test_samples": inf.test_samples,
        "n_features": len(split.feature_names),
        "model_load_time_s": load_s,
        "interpreter_baseline_rss_mib": baseline_rss,
        "model_load_rss_delta_mib": round(max(0.0, after_fit_rss - baseline_rss), 3),
        "inference_peak_rss_mib": inf_rss,
        "inference_rss_delta_mib": round(max(0.0, inf_rss - after_load_rss), 3),
        "mps_current_allocated_mib": mps_cur,
        "mps_driver_allocated_mib": mps_drv,
        "peak_tracemalloc_kib": inf.peak_tracemalloc_kib,
        "offline_batched_mean_latency_ms": inf.mean_latency_ms,
        "offline_batched_throughput_ev_s": inf.throughput_events_per_sec,
    }


def run_isolated_subprocess_profiling(
    skab_dir: Path,
    tsb_dir: Path,
    suites: Sequence[str] = ("SKAB", "TSB-AD-M"),
    models: Sequence[str] = PROFILER_MODELS,
    device: str = "auto",
    cache_dir: str = "models/hf_cache",
    local_files_only: bool = True,
    mock_backends: bool = False,
) -> list[dict[str, Any]]:
    """Spawn one fresh isolated Python subprocess per `(suite, model)` pair."""
    results: list[dict[str, Any]] = []
    env = dict(os.environ)
    if local_files_only:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

    for suite in suites:
        for m_name in models:
            cmd = [
                sys.executable,
                "-m",
                "app.m9_1_profiler",
                "--worker",
                "--model",
                m_name,
                "--suite",
                suite,
                "--skab-dir",
                str(skab_dir),
                "--tsb-dir",
                str(tsb_dir),
                "--device",
                device,
                "--cache-dir",
                cache_dir,
            ]
            if local_files_only:
                cmd.append("--local-files-only")
            if mock_backends:
                cmd.append("--mock-backends")
            proc = subprocess.run(  # noqa: S603 - fixed sys.executable + internal worker module
                cmd,
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            if proc.returncode != 0:
                results.append(
                    {
                        "model": m_name,
                        "dataset_suite": suite,
                        "status": "failed",
                        "error": (proc.stderr or proc.stdout).strip()[-400:],
                    }
                )
                continue
            # Parse last JSON line emitted by worker
            json_lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
            if not json_lines:
                results.append(
                    {
                        "model": m_name,
                        "dataset_suite": suite,
                        "status": "failed",
                        "error": "No JSON output from profiler worker",
                    }
                )
                continue
            parsed: Mapping[str, Any] = json.loads(json_lines[-1])
            row = dict(parsed)
            row["status"] = "ok"
            results.append(row)
    return results


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M9.1 isolated subprocess resource profiler.")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", required=True)
    parser.add_argument("--suite", choices=["SKAB", "TSB-AD-M"], required=True)
    parser.add_argument("--skab-dir", type=Path, default=Path("data/external/skab"))
    parser.add_argument("--tsb-dir", type=Path, default=Path("data/external/tsb_ad_m"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", default="models/hf_cache")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--mock-backends", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Worker entry point emitting a single JSON resource profile line to stdout."""
    args = _parse_args(argv)
    split = _load_target_split(args.suite, args.skab_dir, args.tsb_dir)
    res = profile_single_model_in_process(
        model_name=args.model,
        split=split,
        device=args.device,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        mock_backends=args.mock_backends,
    )
    print(json.dumps(res, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
