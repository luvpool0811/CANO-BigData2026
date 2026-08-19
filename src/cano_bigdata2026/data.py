"""Small, explicit NPZ interface for public training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from . import contracts as C


@dataclass(frozen=True)
class Normalization:
    mean: dict[str, float]
    std: dict[str, float]

    @classmethod
    def load(cls, path: str | Path) -> "Normalization":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        mean = {key: float(value) for key, value in payload["mean"].items()}
        std = {key: float(value) for key, value in payload["std"].items()}
        for variable in C.OUTPUT_VARIABLES:
            if variable not in mean or variable not in std or std[variable] <= 0:
                raise ValueError(f"invalid normalization for {variable}")
        return cls(mean, std)

    def denormalize_output(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim not in (3, 4):
            raise ValueError("output must be [72,H,W] or [B,72,H,W]")
        squeeze = tensor.ndim == 3
        output = tensor.unsqueeze(0).clone() if squeeze else tensor.clone()
        if output.shape[1] != C.N_OUTPUT_CHANNELS:
            raise ValueError("output has the wrong channel count")
        for lead in range(C.N_LEADS):
            for index, variable in enumerate(C.OUTPUT_VARIABLES):
                channel = C.output_channel(lead, index)
                output[:, channel] = (
                    output[:, channel] * self.std[variable] + self.mean[variable]
                )
        return output[0] if squeeze else output


class EventNPZDataset(Dataset):
    """Directory dataset with one compressed NPZ file per rainfall event.

    Each file contains ``input`` [31,H,W], ``target`` [72,H,W], ``mask``
    [H,W], and an optional scalar string ``event_id``.  Input and target arrays
    are normalized with train-only statistics stored in ``normalization.json``.
    """

    def __init__(self, files: Sequence[str | Path]):
        self.files = tuple(Path(path) for path in files)
        if not self.files:
            raise ValueError("no event files were provided")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.files[index]
        with np.load(path, allow_pickle=False) as payload:
            inputs = np.asarray(payload["input"], dtype=np.float32)
            target = np.asarray(payload["target"], dtype=np.float32)
            mask = np.asarray(payload["mask"], dtype=bool)
            event_id = (
                str(payload["event_id"].item())
                if "event_id" in payload
                else path.stem
            )
        if inputs.ndim != 3 or inputs.shape[0] != C.N_INPUT_CHANNELS:
            raise ValueError(f"invalid input shape in {path}: {inputs.shape}")
        if target.shape != (C.N_OUTPUT_CHANNELS, *inputs.shape[1:]):
            raise ValueError(f"invalid target shape in {path}: {target.shape}")
        if mask.shape != inputs.shape[1:] or not np.any(mask):
            raise ValueError(f"invalid mask in {path}: {mask.shape}")
        if not np.isfinite(inputs[:, mask]).all() or not np.isfinite(target[:, mask]).all():
            raise ValueError(f"non-finite valid values in {path}")
        return {
            "input": torch.from_numpy(inputs),
            "target": torch.from_numpy(target),
            "mask": torch.from_numpy(mask),
            "event_id": event_id,
        }


def event_files(root: str | Path, split: str) -> list[Path]:
    directory = Path(root) / split
    files = sorted(directory.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no NPZ events found in {directory}")
    return files


__all__ = ["Normalization", "EventNPZDataset", "event_files"]
