"""Event-level evidence records for the operational-target protocol."""

from __future__ import annotations

from typing import Any

import numpy as np

from .calibration import (
    TargetAlignedCalibrator,
    evaluate_target_aligned_event,
)
from .metrics import event_metrics


def evaluate_event(
    *,
    event_id: str,
    prediction_huv: np.ndarray,
    truth_huv: np.ndarray,
    valid_mask: np.ndarray,
    node_scale_h: np.ndarray | None = None,
    calibrator: TargetAlignedCalibrator | None = None,
) -> dict[str, Any]:
    """Evaluate one event without pooling its cells with another event."""

    point = event_metrics(prediction_huv, truth_huv, valid_mask)
    record: dict[str, Any] = {
        "event_id": str(event_id),
        "point_prediction": point,
    }
    if (node_scale_h is None) != (calibrator is None):
        raise ValueError("node_scale_h and calibrator must be supplied together")
    if calibrator is not None:
        prediction = np.asarray(prediction_huv).reshape(24, 3, *valid_mask.shape)
        truth = np.asarray(truth_huv).reshape(24, 3, *valid_mask.shape)
        record["target_aligned_calibration"] = evaluate_target_aligned_event(
            prediction[:, 0],
            truth[:, 0],
            valid_mask,
            np.asarray(node_scale_h),
            calibrator,
        )
    return record


__all__ = ["evaluate_event"]
