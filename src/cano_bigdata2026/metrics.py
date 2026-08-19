"""Event-level point prediction and interval metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from . import contracts as C


def _physical_field(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 3 and values.shape[0] == C.N_OUTPUT_CHANNELS:
        return values.reshape(C.N_LEADS, len(C.OUTPUT_VARIABLES), *values.shape[1:])
    if values.ndim == 4 and values.shape[:2] == (
        C.N_LEADS,
        len(C.OUTPUT_VARIABLES),
    ):
        return values
    raise ValueError("field must be [72,H,W] or [24,3,H,W]")


def _nse(prediction: np.ndarray, truth: np.ndarray) -> float | None:
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    if denominator <= 0.0:
        return None
    return float(1.0 - np.sum((prediction - truth) ** 2) / denominator)


def event_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    valid_mask: np.ndarray,
    *,
    wet_threshold_m: float = C.WET_THRESHOLD_M,
    csi_thresholds_m: Iterable[float] = C.CSI_THRESHOLDS_M,
) -> dict[str, object]:
    pred = _physical_field(prediction)
    target = _physical_field(truth)
    mask = np.asarray(valid_mask, dtype=bool)
    if pred.shape != target.shape or pred.shape[2:] != mask.shape:
        raise ValueError("prediction, truth, and mask shapes differ")
    pred_h = pred[:, 0, mask].reshape(-1)
    truth_h = target[:, 0, mask].reshape(-1)
    if not np.isfinite(pred_h).all() or not np.isfinite(truth_h).all():
        raise ValueError("valid H values must be finite")
    wet = truth_h >= float(wet_threshold_m)
    rmse = float(np.sqrt(np.mean((pred_h - truth_h) ** 2)))
    wet_rmse = (
        float(np.sqrt(np.mean((pred_h[wet] - truth_h[wet]) ** 2)))
        if np.any(wet)
        else None
    )
    peak_errors = [
        abs(float(np.max(pred[lead, 0, mask])) - float(np.max(target[lead, 0, mask])))
        for lead in range(C.N_LEADS)
    ]
    csi: dict[str, float | None] = {}
    for threshold in csi_thresholds_m:
        predicted_positive = pred_h >= threshold
        truth_positive = truth_h >= threshold
        intersection = int(np.count_nonzero(predicted_positive & truth_positive))
        union = int(np.count_nonzero(predicted_positive | truth_positive))
        csi[f"{float(threshold):g}"] = float(intersection / union) if union else None
    return {
        "h_rmse_m": rmse,
        "h_wet_rmse_m": wet_rmse,
        "nse": _nse(pred_h, truth_h),
        "wet_nse": _nse(pred_h[wet], truth_h[wet]) if np.any(wet) else None,
        "peak_depth_abs_error_m": float(np.mean(peak_errors)),
        "csi": csi,
        "n_valid_cell_time": int(pred_h.size),
        "n_truth_wet": int(np.count_nonzero(wet)),
    }


def interval_metrics(
    lower: np.ndarray,
    upper: np.ndarray,
    truth: np.ndarray,
    *,
    nominal_coverage: float,
) -> dict[str, float]:
    lo = np.asarray(lower, dtype=np.float64).reshape(-1)
    hi = np.asarray(upper, dtype=np.float64).reshape(-1)
    target = np.asarray(truth, dtype=np.float64).reshape(-1)
    if lo.shape != hi.shape or lo.shape != target.shape or np.any(lo > hi):
        raise ValueError("invalid interval arrays")
    alpha = 1.0 - float(nominal_coverage)
    if not 0.0 < alpha < 1.0:
        raise ValueError("nominal coverage must be in (0,1)")
    covered = (target >= lo) & (target <= hi)
    score = hi - lo
    score = score + 2.0 / alpha * (lo - target) * (target < lo)
    score = score + 2.0 / alpha * (target - hi) * (target > hi)
    coverage = float(np.mean(covered))
    return {
        "coverage": coverage,
        "absolute_coverage_error": abs(coverage - nominal_coverage),
        "mean_interval_width": float(np.mean(hi - lo)),
        "winkler_score": float(np.mean(score)),
    }


def aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        raise ValueError("records are empty")
    scalar_keys = (
        "h_rmse_m",
        "h_wet_rmse_m",
        "nse",
        "wet_nse",
        "peak_depth_abs_error_m",
    )
    result: dict[str, object] = {"n_events": len(records)}
    for key in scalar_keys:
        values = [float(row[key]) for row in records if row.get(key) is not None]
        result[key] = float(np.mean(values)) if values else None
    thresholds = records[0]["csi"]
    assert isinstance(thresholds, dict)
    result["csi"] = {
        key: float(
            np.mean(
                [
                    float(row["csi"][key])
                    for row in records
                    if isinstance(row["csi"], dict) and row["csi"].get(key) is not None
                ]
            )
        )
        for key in thresholds
    }
    return result


__all__ = ["event_metrics", "interval_metrics", "aggregate"]
