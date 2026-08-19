#!/usr/bin/env python3
"""Rebuild public tables and figures from the checked-in CSV files."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cano_bigdata2026.results import main


if __name__ == "__main__":
    raise SystemExit(main())
