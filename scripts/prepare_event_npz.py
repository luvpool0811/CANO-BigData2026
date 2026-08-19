#!/usr/bin/env python3
"""Validate common arrays and package one event as a compressed NPZ file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = np.asarray(np.load(args.input, allow_pickle=False), dtype=np.float32)
    target = np.asarray(np.load(args.target, allow_pickle=False), dtype=np.float32)
    mask = np.asarray(np.load(args.mask, allow_pickle=False), dtype=bool)
    if inputs.ndim != 3 or inputs.shape[0] != 31:
        raise ValueError(f"input must be [31,H,W], found {inputs.shape}")
    if target.shape != (72, *inputs.shape[1:]):
        raise ValueError(f"target must be [72,H,W], found {target.shape}")
    if mask.shape != inputs.shape[1:] or not np.any(mask):
        raise ValueError(f"mask must be nonempty [H,W], found {mask.shape}")
    if not np.isfinite(inputs[:, mask]).all() or not np.isfinite(target[:, mask]).all():
        raise ValueError("valid input and target values must be finite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        input=inputs,
        target=target,
        mask=mask,
        event_id=np.asarray(args.event_id),
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
