from __future__ import annotations

import numpy as np

from cano_bigdata2026.calibration import (
    evaluate_target_aligned_event,
    fit_node_scale,
    fit_target_aligned_calibrator,
    operational_population,
)
from cano_bigdata2026.evidence import evaluate_event


def _fields(seed: int, events: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.0, 0.8, size=(events, 24, 3, 4))
    prediction = truth + rng.normal(0.0, 0.03, size=truth.shape)
    mask = np.ones((3, 4), dtype=bool)
    return prediction, truth, mask


def test_target_aligned_fit_and_event_evidence() -> None:
    development_prediction, development_truth, mask = _fields(1)
    scale = fit_node_scale(development_prediction, development_truth, mask)
    assert scale.shape == (24, 3, 4)
    assert np.all(scale[:, mask] >= 1e-3)
    calibration_prediction, calibration_truth, _ = _fields(2)
    fitted = fit_target_aligned_calibrator(
        calibration_prediction,
        calibration_truth,
        mask,
        scale,
        threshold_m=0.3,
    )
    assert set(fitted.alpha_to_q) == {0.5, 0.2, 0.1}
    assert all(value >= 0 for value in fitted.alpha_to_q.values())
    assert all(
        fitted.alpha_to_empirical_risk[alpha] <= alpha + 1e-12
        for alpha in fitted.alpha_to_q
    )
    result = evaluate_target_aligned_event(
        calibration_prediction[0], calibration_truth[0], mask, scale, fitted
    )
    assert result["empty"] is False
    assert 0 <= result["target_ace"] <= 1
    assert result["target_wis"] >= 0

    prediction_huv = np.zeros((72, 3, 4), dtype=np.float64)
    truth_huv = np.zeros_like(prediction_huv)
    prediction_huv.reshape(24, 3, 3, 4)[:, 0] = calibration_prediction[0]
    truth_huv.reshape(24, 3, 3, 4)[:, 0] = calibration_truth[0]
    evidence = evaluate_event(
        event_id="event-01",
        prediction_huv=prediction_huv,
        truth_huv=truth_huv,
        valid_mask=mask,
        node_scale_h=scale,
        calibrator=fitted,
    )
    assert evidence["event_id"] == "event-01"
    assert "target_aligned_calibration" in evidence


def test_population_uses_prediction_not_truth() -> None:
    prediction = np.zeros((24, 2, 2), dtype=np.float64)
    prediction[:, 0, 0] = 0.4
    mask = np.ones((2, 2), dtype=bool)
    selected = operational_population(prediction, mask, threshold_m=0.3)
    assert np.array_equal(
        selected,
        operational_population(prediction, mask, threshold_m=0.3),
    )
