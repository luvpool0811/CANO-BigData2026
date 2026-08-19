#!/usr/bin/env python3
"""Train CANO or an adapted UrbanFloodCast baseline."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cano_bigdata2026.cli import train_main


if __name__ == "__main__":
    raise SystemExit(train_main())
