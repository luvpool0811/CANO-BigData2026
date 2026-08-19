#!/usr/bin/env python3
"""Regenerate the central operational-reliability tables and figure."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cano_bigdata2026.operational_evidence import main


if __name__ == "__main__":
    raise SystemExit(main())
