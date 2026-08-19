#!/usr/bin/env python3
"""Run the self-contained CANO quickstart from a source checkout."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cano_bigdata2026.cli import quickstart_main


if __name__ == "__main__":
    raise SystemExit(quickstart_main())
