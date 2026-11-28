"""Validation-only calibration strategies and bootstrap uncertainty analysis for M9.1.

Provides:
1. Five strictly validation-only threshold calibration strategies (`x_val_normal` only;
   zero test-label access):
   - `empirical_q99`: Empirical 99th percentile of `val_scores`.
   - `empirical_q995`: Empirical 99.5th percentile of `val_scores`.
   - `mad_z35`: `median(val) + 3.5 * 1.4826 * MAD(val)`.
   - `iqr_k30`: `Q3(val) + 3.0 * IQR(val)`.
   - `evt_pot_q99`: Validation-only Peaks-Over-Threshold exponential tail above `Q90(val)`
     extrapolated to tail probability `0.01` (`q=0.99`, zero test distribution inspection).
2. False-positive audit helpers including `false_alarms_per_1k_normal`.
3. Non-parametric bootstrap 95% confidence intervals over experiments/series and
   paired bootstrap hypothesis tests so small differences (e.g. `0.0006` Macro F1)
   are never over-claimed as statistically meaningful.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from app.m2_report import compute_classification_metrics
from app.m4_models import FloatArray, IntArray

CalibrationStrategy = Literal[
    "empirical_q99",
    "empirical_q995",
    "mad_z35",
    "iqr_k30",
    "evt_pot_q99",
]

ALL_CALIBRATION_STRATEGIES: tuple[CalibrationStrategy, ...] = (
    "empirical_q99",
    "empirical_q995",
    "mad_z35",
    "iqr_k30",
    "evt_pot_q99",
)


def calibrate_threshold_validation_only(
    val_scores: FloatArray,
    strategy: CalibrationStrategy,
) -> float:
    """Compute anomaly threshold strictly from normal validation scores `val_scores`."""
    arr = np.asarray(val_scores, dtype=np.float64)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("val_scores must be non-empty and finite")

    if strategy == "empirical_q99":
        return float(np.quantile(arr, 0.99))
    if strategy == "empirical_q995":
        return float(np.quantile(arr, 0.995))
    if strategy == "mad_z35":
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        scale = max(1.4826 * mad, float(np.std(arr)), 1e-9)
        return float(med + 3.5 * scale)
    if strategy == "iqr_k30":
        q25 = float(np.quantile(arr, 0.25))
        q75 = float(np.quantile(arr, 0.75))
        iqr = max(q75 - q25, float(np.std(arr)), 1e-9)
        return float(q75 + 3.0 * iqr)
    if strategy == "evt_pot_q99":
        # Validation-only Peaks-Over-Threshold exponential tail above u = Q90(val)
        u = float(np.quantile(arr, 0.90))
        exceedances = arr[arr > u] - u
        if exceedances.size < 3:
            return float(np.quantile(arr, 0.995))
        beta = max(float(np.mean(exceedances)), 1e-9)
        p_tail = float(exceedances.size) / float(arr.size)
        # Solve P(X > thr) = p_tail * exp(-(thr - u) / beta) = 0.01 (capped for stability)
        ratio = max(1.0, p_tail / 0.01)
        return float(u + beta * math.log(ratio))

    raise ValueError(f"Unsupported calibration strategy: {strategy!r}")


def evaluate_threshold_on_test(
    y_true: IntArray,
    test_scores: FloatArray,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate a pre-calibrated threshold on `y_true` and compute FA per 1,000 normal."""
    preds = (np.asarray(test_scores, dtype=np.float64) > threshold).astype(np.int64)
    tp = int(np.sum((y_true == 1) & (preds == 1)))
    fp = int(np.sum((y_true == 0) & (preds == 1)))
    tn = int(np.sum((y_true == 0) & (preds == 0)))
    fn = int(np.sum((y_true == 1) & (preds == 0)))
    cls_m = compute_classification_metrics(tp, fp, tn, fn)
    n_normal = fp + tn
    fa_per_1k = round(1000.0 * fp / n_normal, 3) if n_normal > 0 else None
    return {
        "threshold": round(threshold, 6),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "normal_observations": n_normal,
        "precision": cls_m["precision"],
        "recall": cls_m["recall"],
        "f1": cls_m["f1"],
        "fpr": cls_m["false_positive_rate"],
        "false_alarms_per_1k_normal": fa_per_1k,
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float | int | None]:
    """Compute non-parametric percentile bootstrap 95% CI over experiment/series scores."""
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None, "std_err": None}
    mean_v = float(np.mean(arr))
    if n == 1:
        return {
            "n": 1,
            "mean": round(mean_v, 4),
            "ci_low": round(mean_v, 4),
            "ci_high": round(mean_v, 4),
            "std_err": 0.0,
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = np.mean(arr[idx], axis=1)
    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    se = float(np.std(boot_means, ddof=1))
    return {
        "n": n,
        "mean": round(mean_v, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "std_err": round(se, 4),
    }


def paired_bootstrap_test(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired bootstrap test of `mean(A - B)` across identical experiments/series.

    Returns mean difference, 95% CI of the difference, two-sided p-value, and
    whether the difference is statistically significant at `alpha` (`p < alpha`
    and 95% CI does not cross 0).
    """
    a = np.asarray([float(v) for v in values_a], dtype=np.float64)
    b = np.asarray([float(v) for v in values_b], dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("Paired bootstrap inputs must be non-empty and identical length")
    diffs = a - b
    n = int(diffs.size)
    obs_diff = float(np.mean(diffs))
    if n == 1 or float(np.std(diffs)) < 1e-12:
        is_nonzero = abs(obs_diff) >= 1e-12
        return {
            "n_paired": n,
            "mean_diff": round(obs_diff, 4),
            "ci_low": round(obs_diff, 4),
            "ci_high": round(obs_diff, 4),
            "p_value": 0.0 if is_nonzero else 1.0,
            "significant_at_005": bool(n >= 2 and is_nonzero),
        }

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_diffs = np.mean(diffs[idx], axis=1)
    lo = float(np.quantile(boot_diffs, alpha / 2.0))
    hi = float(np.quantile(boot_diffs, 1.0 - alpha / 2.0))

    # Centered bootstrap null distribution for two-sided p-value
    centered = boot_diffs - obs_diff
    p_val = float(np.mean(np.abs(centered) >= abs(obs_diff)))
    sig = bool(p_val < alpha and (lo > 0.0 or hi < 0.0))
    return {
        "n_paired": n,
        "mean_diff": round(obs_diff, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "p_value": round(p_val, 4),
        "significant_at_005": sig,
    }
