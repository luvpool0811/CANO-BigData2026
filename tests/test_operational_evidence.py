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
        "operational_evaluation_specification.csv",
        "target_calibration.csv",
        "crc_calibration.csv",
        "deployment_budget_effects.csv",
        "warning_rule_migration.csv",
        "claim_evidence.csv",
        "baseline_fairness.csv",
        "hpo_candidates.csv",
        "reproducibility_scope.csv",
    ):
        (target / name).write_bytes((source / name).read_bytes())


def test_operational_disclosures_regenerate(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "generated"
    result = reproduce(results_dir=root / "results/paper", output_dir=output)
    assert result == {
        "status": "PASS",
        "operational_rows": 6,
        "operational_specification_rows": 1,
        "target_calibration_rows": 4,
        "crc_calibration_rows": 2,
        "deployment_budget_rows": 9,
        "warning_rule_rows": 36,
        "warning_near_tie_rows": 8,
        "claim_rows": 14,
        "fairness_rows": 4,
        "hpo_candidate_rows": 24,
        "reproducibility_scope_rows": 9,
        "statistics_recomputed": False,
        "output_dir": str(output),
    }
    for name in (
        "operational_contrasts.md",
        "operational_evaluation_specification.md",
        "target_calibration.md",
        "crc_calibration.md",
        "deployment_budget_effects.md",
        "warning_rule_migration.md",
        "claim_evidence.md",
        "baseline_fairness.md",
        "hpo_candidates.md",
        "reproducibility_scope.md",
        "operational_reliability_effects.png",
        "deployment_boundaries.png",
        "deployment_boundaries.pdf",
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


def test_crc_correction_tamper_is_rejected(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    copied = tmp_path / "paper"
    _copy_csvs(root / "results/paper", copied)
    path = copied / "crc_calibration.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames
    assert fields is not None
    rows[0]["maximum_empirical_event_risk"] = "0.04"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="finite-sample correction"):
        validate_operational_inputs(copied)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("calibrator_refit_each_replicate", "false"),
        ("common_method_indices", "false"),
        ("empty_event_policy", "treat empty events as zero loss"),
    ),
)
def test_crc_resampling_contract_tamper_is_rejected(
    tmp_path: Path, field: str, value: str
) -> None:
    root = Path(__file__).resolve().parents[1]
    copied = tmp_path / "paper"
    _copy_csvs(root / "results/paper", copied)
    path = copied / "crc_calibration.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames
    assert fields is not None
    rows[0][field] = value
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="CRC resampling and refit contract"):
        validate_operational_inputs(copied)


def test_deployment_kappa_tamper_is_rejected(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    copied = tmp_path / "paper"
    _copy_csvs(root / "results/paper", copied)
    path = copied / "deployment_budget_effects.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames
    assert fields is not None
    rows[0]["kappa"] = "1.25"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="budget/prevalence"):
        validate_operational_inputs(copied)


def test_warning_strategy_code_tamper_is_rejected(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    copied = tmp_path / "paper"
    _copy_csvs(root / "results/paper", copied)
    path = copied / "warning_rule_migration.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames
    assert fields is not None
    rows[0]["winner_code"] = "Gl"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="code does not match"):
        validate_operational_inputs(copied)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("loss_name", "unspecified warning loss", "loss definition"),
        ("near_tie", "true", "near-tie marker"),
    ),
)
def test_warning_loss_contract_tamper_is_rejected(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    root = Path(__file__).resolve().parents[1]
    copied = tmp_path / "paper"
    _copy_csvs(root / "results/paper", copied)
    path = copied / "warning_rule_migration.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames
    assert fields is not None
    rows[0][field] = value
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match=message):
        validate_operational_inputs(copied)


def test_rq1_specification_must_match_reported_contrast(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    copied = tmp_path / "paper"
    _copy_csvs(root / "results/paper", copied)
    path = copied / "operational_evaluation_specification.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames
    assert fields is not None
    rows[0]["point_predictor"] = "unspecified predictor"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="point_predictor"):
        validate_operational_inputs(copied)


def test_hpo_nonminimum_candidate_is_rejected(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    copied = tmp_path / "paper"
    _copy_csvs(root / "results/paper", copied)
    path = copied / "hpo_candidates.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames
    assert fields is not None
    for row in rows:
        if row["system"] == "CANO":
            row["selected"] = "true" if row["candidate_index"] == "4" else "false"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="does not minimize"):
        validate_operational_inputs(copied)
