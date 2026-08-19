"""Common input, output, and metric definitions used by every model."""

from __future__ import annotations

N_LEADS = 24
OUTPUT_VARIABLES = ("H", "U", "V")
N_OUTPUT_CHANNELS = N_LEADS * len(OUTPUT_VARIABLES)

INPUT_STATE_VARIABLES = ("H0", "U0", "V0")
INPUT_STATIC_VARIABLES = ("DEM", "ROUGHNESS")
INPUT_COORDINATES = ("GRID_Y", "GRID_X")
N_RAIN_CHANNELS = N_LEADS
N_INPUT_CHANNELS = (
    len(INPUT_STATE_VARIABLES)
    + len(INPUT_STATIC_VARIABLES)
    + len(INPUT_COORDINATES)
    + N_RAIN_CHANNELS
)

CH_H0, CH_U0, CH_V0 = 0, 1, 2
CH_DEM, CH_ROUGHNESS = 3, 4
CH_GRID_Y, CH_GRID_X = 5, 6
CH_RAIN_START = 7

CSI_THRESHOLDS_M = (0.01, 0.10, 0.30, 0.50)
WET_THRESHOLD_M = 0.30


def output_channel(lead: int, variable: int) -> int:
    if not 0 <= int(lead) < N_LEADS:
        raise ValueError("lead must be in [0, 23]")
    if not 0 <= int(variable) < len(OUTPUT_VARIABLES):
        raise ValueError("variable must be H=0, U=1, or V=2")
    return int(lead) * len(OUTPUT_VARIABLES) + int(variable)


__all__ = [name for name in globals() if name.isupper()] + ["output_channel"]
