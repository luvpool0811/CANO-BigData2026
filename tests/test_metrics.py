from __future__ import annotations

import numpy as np

from cano_bigdata2026 import contracts as C
from cano_bigdata2026.metrics import event_metrics, interval_metrics


def test_event_metrics_perfect_prediction() -> None:
    truth = np.zeros((72, 3, 4), dtype=np.float32)
    for lead in range(24):
        truth[lead * 3] = np.linspace(0.0, 1.0, 12).reshape(3, 4)
    mask = np.ones((3, 4), dtype=bool)
    result = event_metrics(truth, truth, mask)
    assert result["h_rmse_m"] == 0.0
    assert result["h_wet_rmse_m"] == 0.0
    assert result["nse"] == 1.0
    assert result["wet_nse"] == 1.0
    assert result["peak_depth_abs_error_m"] == 0.0
    assert all(value == 1.0 for value in result["csi"].values())


def test_interval_metrics() -> None:
    truth = np.asarray([0.0, 1.0, 2.0])
    result = interval_metrics(
        truth - 0.2, truth + 0.2, truth, nominal_coverage=0.9
    )
    assert result["coverage"] == 1.0
    assert np.isclose(result["absolute_coverage_error"], 0.1)
    assert np.isclose(result["mean_interval_width"], 0.4)
    assert np.isclose(result["winkler_score"], 0.4)


def test_truth_wet_and_operational_thresholds_are_distinct() -> None:
    assert C.TRUTH_WET_THRESHOLD_M == 0.01
    assert C.OPERATIONAL_TARGET_THRESHOLD_M == 0.30
    assert C.WET_THRESHOLD_M == C.TRUTH_WET_THRESHOLD_M

    truth = np.zeros((72, 1, 3), dtype=np.float32)
    truth.reshape(24, 3, 1, 3)[:, 0, 0] = np.asarray([0.0, 0.02, 0.4])
    result = event_metrics(truth, truth, np.ones((1, 3), dtype=bool))
    assert result["n_truth_wet"] == 48
