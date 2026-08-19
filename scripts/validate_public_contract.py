#!/usr/bin/env python3
"""Validate the public experiment, HPO, split, and claim contracts."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cano_bigdata2026.public_contract import main


if __name__ == "__main__":
    raise SystemExit(main())
