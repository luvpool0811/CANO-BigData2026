#!/usr/bin/env python3
"""Evaluate a CANO or baseline checkpoint."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cano_bigdata2026.cli import evaluate_main


if __name__ == "__main__":
    raise SystemExit(evaluate_main())
