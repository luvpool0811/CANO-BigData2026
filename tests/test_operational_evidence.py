from __future__ import annotations

import csv
from pathlib import Path

import pytest

from cano_bigdata2026.operational_evidence import (
    reproduce,
    validate_operational_inputs,
)


def _copy_csvs(source: Path, target: Path) -> None:
    target.mkdir(parents=True)
    for name in (
        "operational_contrasts.csv",
        "target_calibration.csv",
        "claim_evidence.csv",
        "baseline_fairness.csv",
    ):
        (target / name).write_bytes((source / name).read_bytes())


def test_operational_disclosures_regenerate(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "generated"
    result = reproduce(results_dir=root / "results/paper", output_dir=output)
    assert result == {
        "status": "PASS",
        "operational_rows": 6,
        "target_calibration_rows": 4,
        "claim_rows": 11,
        "fairness_rows": 4,
        "statistics_recomputed": False,
        "output_dir": str(output),
    }
    for name in (
        "operational_contrasts.md",
        "target_calibration.md",
        "claim_evidence.md",
        "baseline_fairness.md",
        "operational_reliability_effects.png",
    ):
        assert (output / name).stat().st_size > 100


def test_population_identity_tamper_is_rejected(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "results/paper"
    copied = tmp_path / "paper"
    _copy_csvs(source, copied)
    path = copied / "operational_contrasts.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames
    assert fields is not None
    rows[0]["population_effect"] = "0.09"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="full minus selected"):
        validate_operational_inputs(copied)


def test_evaluation_data_cannot_enter_selection(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    copied = tmp_path / "paper"
    _copy_csvs(root / "results/paper", copied)
    path = copied / "baseline_fairness.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames
    assert fields is not None
    rows[1]["test_used_for_selection"] = "true"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="evaluation data"):
        validate_operational_inputs(copied)
