"""Public CANO reproducibility package for the IEEE BigData 2026 study."""

from .contracts import N_INPUT_CHANNELS, N_OUTPUT_CHANNELS, N_LEADS
from .calibration import (
    EventBalancedCRCThreshold,
    TargetAlignedCalibrator,
    crc_maximum_empirical_event_risk,
    fit_event_balanced_crc,
    fit_node_scale_events,
    fit_target_aligned_calibrator,
    fit_target_aligned_calibrator_events,
)
from .models import CANO, build_model, real_parameter_count

__version__ = "1.1.0"

__all__ = [
    "CANO",
    "EventBalancedCRCThreshold",
    "TargetAlignedCalibrator",
    "N_INPUT_CHANNELS",
    "N_OUTPUT_CHANNELS",
    "N_LEADS",
    "build_model",
    "crc_maximum_empirical_event_risk",
    "fit_event_balanced_crc",
    "fit_node_scale_events",
    "fit_target_aligned_calibrator",
    "fit_target_aligned_calibrator_events",
    "real_parameter_count",
]
