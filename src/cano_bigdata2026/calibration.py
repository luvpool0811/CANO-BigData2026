"""Node-normalized, target-aligned empirical conformal calibration.

The node scale is fitted on development events, the dimensionless residual
quantiles are fitted on separate calibration events, and evaluation events are
used only after both objects have been fixed. The operational population uses
the prediction and a physical threshold; it never uses the unknown target to
select cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from . import contracts as C


DEFAULT_ALPHA_LEVELS = (0.5, 0.2, 0.1)


def _event_field(array: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim != 4 or values.shape[1] != 24:
        raise ValueError(f"{label} must be [events,24,H,W]")
    return values


def fit_node_scale(
    predictions_h: np.ndarray,
    truths_h: np.ndarray,
    valid_mask: np.ndarray,
    *,
    floor_m: float = 1.0e-3,
) -> np.ndarray:
    """Fit a lead-by-node RMS residual scale on development events only."""

    prediction = _event_field(predictions_h, label="predictions_h")
    truth = _event_field(truths_h, label="truths_h")
    mask = np.asarray(valid_mask, dtype=bool)
    if prediction.shape != truth.shape or prediction.shape[2:] != mask.shape:
        raise ValueError("prediction, truth, and valid-mask shapes differ")
    if prediction.shape[0] < 1 or floor_m <= 0:
        raise ValueError("at least one event and a positive floor are required")
    if not np.isfinite(prediction[:, :, mask]).all() or not np.isfinite(
        truth[:, :, mask]
    ).all():
        raise ValueError("development values must be finite on valid cells")
    scale = np.full(prediction.shape[1:], np.nan, dtype=np.float32)
    rms = np.sqrt(np.mean((truth[:, :, mask] - prediction[:, :, mask]) ** 2, axis=0))
    scale[:, mask] = np.maximum(rms, float(floor_m)).astype(np.float32)
    return scale


def _single_h_field(array: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != 24:
        raise ValueError(f"{label} must be [24,H,W]")
    return values


def fit_node_scale_events(
    events: Iterable[tuple[np.ndarray, np.ndarray]],
    valid_mask: np.ndarray,
    *,
    floor_m: float = 1.0e-3,
) -> np.ndarray:
    """Fit the development scale without stacking all events in memory."""

    mask = np.asarray(valid_mask, dtype=bool)
    if mask.ndim != 2 or not np.any(mask) or floor_m <= 0:
        raise ValueError("a nonempty 2-D mask and positive floor are required")
    sum_squared: np.ndarray | None = None
    count = 0
    for prediction_h, truth_h in events:
        prediction = _single_h_field(prediction_h, label="prediction_h")
        truth = _single_h_field(truth_h, label="truth_h")
        if prediction.shape != truth.shape or prediction.shape[1:] != mask.shape:
            raise ValueError("development event shapes differ")
        if not np.isfinite(prediction[:, mask]).all() or not np.isfinite(
            truth[:, mask]
        ).all():
            raise ValueError("development values must be finite on valid cells")
        if sum_squared is None:
            sum_squared = np.zeros(prediction.shape, dtype=np.float64)
        residual = truth - prediction
        sum_squared[:, mask] += residual[:, mask] ** 2
        count += 1
    if count == 0 or sum_squared is None:
        raise ValueError("at least one development event is required")
    scale = np.full(sum_squared.shape, np.nan, dtype=np.float32)
    scale[:, mask] = np.maximum(
        np.sqrt(sum_squared[:, mask] / count), float(floor_m)
    ).astype(np.float32)
    return scale


def operational_population(
    prediction_h: np.ndarray,
    valid_mask: np.ndarray,
    *,
    threshold_m: float,
) -> np.ndarray:
    prediction = np.asarray(prediction_h, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if prediction.ndim != 3 or prediction.shape[0] != 24:
        raise ValueError("prediction_h must be [24,H,W]")
    if prediction.shape[1:] != mask.shape:
        raise ValueError("prediction and valid-mask shapes differ")
    return (prediction >= float(threshold_m)) & np.broadcast_to(mask, prediction.shape)


def _event_balanced_quantile(
    sorted_scores: list[np.ndarray], alpha: float
) -> tuple[float, float]:
    events = [values for values in sorted_scores if values.size]
    if not events:
        return float("inf"), float("nan")

    def risk(q: float) -> float:
        return float(
            np.mean(
                [
                    (values.size - np.searchsorted(values, q, side="right"))
                    / values.size
                    for values in events
                ]
            )
        )

    maximum = max(float(values[-1]) for values in events)
    lower, upper = 0.0, maximum
    for _ in range(64):
        midpoint = 0.5 * (lower + upper)
        if risk(midpoint) <= alpha:
            upper = midpoint
        else:
            lower = midpoint
    candidates = []
    for values in events:
        index = np.searchsorted(values, upper, side="left")
        if index < values.size:
            candidates.append(float(values[index]))
    q = min(candidates) if candidates else maximum
    return q, risk(q)


@dataclass(frozen=True)
class TargetAlignedCalibrator:
    threshold_m: float
    alpha_to_q: dict[float, float]
    alpha_to_empirical_risk: dict[float, float]
    n_calibration_events: int

    def interval(
        self,
        prediction_h: np.ndarray,
        node_scale_h: np.ndarray,
        *,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        prediction = np.asarray(prediction_h, dtype=np.float64)
        scale = np.asarray(node_scale_h, dtype=np.float64)
        if prediction.shape != scale.shape or prediction.ndim != 3:
            raise ValueError("prediction and scale must share [24,H,W]")
        q = self.alpha_to_q[float(alpha)]
        halfwidth = scale * q
        return prediction - halfwidth, prediction + halfwidth


def fit_target_aligned_calibrator(
    predictions_h: np.ndarray,
    truths_h: np.ndarray,
    valid_mask: np.ndarray,
    node_scale_h: np.ndarray,
    *,
    threshold_m: float = C.OPERATIONAL_TARGET_THRESHOLD_M,
    alpha_levels: Iterable[float] = DEFAULT_ALPHA_LEVELS,
) -> TargetAlignedCalibrator:
    """Fit event-balanced normalized residual quantiles on calibration events."""

    prediction = _event_field(predictions_h, label="predictions_h")
    truth = _event_field(truths_h, label="truths_h")
    mask = np.asarray(valid_mask, dtype=bool)
    scale = np.asarray(node_scale_h, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.shape[2:] != mask.shape:
        raise ValueError("prediction, truth, and valid-mask shapes differ")
    if scale.shape != prediction.shape[1:]:
        raise ValueError("node_scale_h must be [24,H,W]")
    if not np.isfinite(scale[:, mask]).all() or np.any(scale[:, mask] <= 0):
        raise ValueError("node scale must be finite and positive on valid cells")
    return fit_target_aligned_calibrator_events(
        zip(prediction, truth, strict=True),
        mask,
        scale,
        threshold_m=threshold_m,
        alpha_levels=alpha_levels,
    )


def fit_target_aligned_calibrator_events(
    events: Iterable[tuple[np.ndarray, np.ndarray]],
    valid_mask: np.ndarray,
    node_scale_h: np.ndarray,
    *,
    threshold_m: float = C.OPERATIONAL_TARGET_THRESHOLD_M,
    alpha_levels: Iterable[float] = DEFAULT_ALPHA_LEVELS,
) -> TargetAlignedCalibrator:
    """Fit event-balanced residual thresholds from a streaming iterator."""

    mask = np.asarray(valid_mask, dtype=bool)
    scale = np.asarray(node_scale_h, dtype=np.float64)
    if mask.ndim != 2 or not np.any(mask) or scale.shape != (24, *mask.shape):
        raise ValueError("node scale must be [24,H,W] on a nonempty 2-D mask")
    if not np.isfinite(scale[:, mask]).all() or np.any(scale[:, mask] <= 0):
        raise ValueError("node scale must be finite and positive on valid cells")
    scores: list[np.ndarray] = []
    for event_prediction, event_truth in events:
        prediction = _single_h_field(event_prediction, label="prediction_h")
        truth = _single_h_field(event_truth, label="truth_h")
        if prediction.shape != truth.shape or prediction.shape[1:] != mask.shape:
            raise ValueError("calibration event shapes differ")
        selected = operational_population(
            prediction, mask, threshold_m=threshold_m
        )
        residual = np.abs(truth - prediction)
        values = residual[selected] / scale[selected]
        if not np.isfinite(values).all():
            raise ValueError("calibration scores must be finite")
        scores.append(np.sort(values))
    if not scores:
        raise ValueError("at least one calibration event is required")
    levels = tuple(float(alpha) for alpha in alpha_levels)
    if len(set(levels)) != len(levels) or any(not 0 < alpha < 1 for alpha in levels):
        raise ValueError("alpha levels must be unique and lie in (0,1)")
    fitted = {alpha: _event_balanced_quantile(scores, alpha) for alpha in levels}
    return TargetAlignedCalibrator(
        threshold_m=float(threshold_m),
        alpha_to_q={alpha: value[0] for alpha, value in fitted.items()},
        alpha_to_empirical_risk={alpha: value[1] for alpha, value in fitted.items()},
        n_calibration_events=len(scores),
    )


def evaluate_target_aligned_event(
    prediction_h: np.ndarray,
    truth_h: np.ndarray,
    valid_mask: np.ndarray,
    node_scale_h: np.ndarray,
    calibrator: TargetAlignedCalibrator,
) -> dict[str, object]:
    """Produce one event-level calibration record and a multi-alpha WIS."""

    prediction = np.asarray(prediction_h, dtype=np.float64)
    truth = np.asarray(truth_h, dtype=np.float64)
    scale = np.asarray(node_scale_h, dtype=np.float64)
    selected = operational_population(
        prediction, valid_mask, threshold_m=calibrator.threshold_m
    )
    if truth.shape != prediction.shape or scale.shape != prediction.shape:
        raise ValueError("prediction, truth, and scale shapes differ")
    if not np.any(selected):
        return {"empty": True, "target_ace": None, "target_wis": None}
    residual = np.abs(truth[selected] - prediction[selected])
    interval_records: dict[str, dict[str, float]] = {}
    weighted = 0.5 * float(np.mean(residual))
    for alpha in calibrator.alpha_to_q:
        lower, upper = calibrator.interval(prediction, scale, alpha=alpha)
        lo, hi, target = lower[selected], upper[selected], truth[selected]
        covered = (target >= lo) & (target <= hi)
        winkler = hi - lo
        winkler += 2.0 / alpha * (lo - target) * (target < lo)
        winkler += 2.0 / alpha * (target - hi) * (target > hi)
        coverage = float(np.mean(covered))
        interval_records[f"{alpha:g}"] = {
            "coverage": coverage,
            "ace": abs(coverage - (1.0 - alpha)),
            "mean_halfwidth_m": float(np.mean((hi - lo) * 0.5)),
            "winkler_score_m": float(np.mean(winkler)),
        }
        weighted += alpha / 2.0 * float(np.mean(winkler))
    wis = weighted / (len(calibrator.alpha_to_q) + 0.5)
    primary_alpha = 0.1
    if primary_alpha not in calibrator.alpha_to_q:
        primary_alpha = min(calibrator.alpha_to_q)
    return {
        "empty": False,
        "n_operational_cells": int(np.count_nonzero(selected)),
        "target_ace": interval_records[f"{primary_alpha:g}"]["ace"],
        "target_wis": float(wis),
        "intervals": interval_records,
    }


__all__ = [
    "DEFAULT_ALPHA_LEVELS",
    "TargetAlignedCalibrator",
    "fit_node_scale",
    "fit_node_scale_events",
    "operational_population",
    "fit_target_aligned_calibrator",
    "fit_target_aligned_calibrator_events",
    "evaluate_target_aligned_event",
]
