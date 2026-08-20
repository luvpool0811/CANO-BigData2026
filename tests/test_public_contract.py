from __future__ import annotations

from pathlib import Path
import csv
import shutil
import zipfile

import numpy as np
import pytest

from cano_bigdata2026.public_contract import (
    create_provider_preflight_receipt,
    validate_prepared_split_ids,
    validate_provider_archive_membership,
    validate_static_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_public_contract() -> None:
    result = validate_static_contract(ROOT)
    assert result == {
        "status": "PASS",
        "checkpoint_selection_metric": "development_event_macro_physical_h_rmse",
        "public_role_rows": 125,
        "provider_identity_rows": 125,
        "operational_specification_rows": 1,
        "hpo_candidate_rows": 24,
        "reproducibility_scope_rows": 9,
        "table_ii_event_rows_recomputed": 48,
        "table_ii_metrics_recomputed": 7,
        "raw_field_arrays_opened": 0,
    }


def _membership() -> list[dict[str, str]]:
    with (ROOT / "configs/evaluation/berlin_i_role_membership.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        return list(csv.DictReader(stream))


def _metadata_event(path: Path, event_id: str, row: dict[str, str]) -> None:
    np.savez_compressed(
        path,
        event_id=np.asarray(event_id),
        provider_event_name=np.asarray(row["provider_event_name"]),
        provider_relative_path=np.asarray(row["provider_relative_path"]),
    )


def test_prepared_split_identity_check_reads_metadata_only(tmp_path: Path) -> None:
    membership = _membership()
    specs = (("train", 85), ("validation", 15), ("calibration", 13), ("test", 12))
    for directory, count in specs:
        (tmp_path / directory).mkdir()
        selected = [row for row in membership if row["directory"] == directory]
        assert len(selected) == count
        for index, row in enumerate(selected):
            _metadata_event(
                tmp_path / directory / f"event-{index:03d}.npz",
                f"{directory}-{index:03d}",
                row,
            )
    result = validate_prepared_split_ids(ROOT, tmp_path)
    assert result["unique_event_ids"] == 125
    assert result["field_arrays_opened"] == 0
    assert result["identity_metadata_only"] is True
    assert result["provider_identities_matched"] == 125
    assert len(result["prepared_identity_sha256"]) == 64

    receipt_path = tmp_path / "provider-preflight.json"
    receipt = create_provider_preflight_receipt(ROOT, tmp_path, receipt_path)
    assert receipt["schema_id"] == "cano_provider_preflight_receipt_v1"
    assert receipt["prepared_identity_sha256"] == result["prepared_identity_sha256"]
    assert receipt_path.is_file()

    duplicate = tmp_path / "test/event-000.npz"
    evaluation = next(row for row in membership if row["directory"] == "test")
    _metadata_event(duplicate, "train-000", evaluation)
    with pytest.raises(ValueError, match="overlaps prepared roles"):
        validate_prepared_split_ids(ROOT, tmp_path)


def test_prepared_provider_identity_tamper_is_rejected(tmp_path: Path) -> None:
    membership = _membership()
    for directory in ("train", "validation", "calibration", "test"):
        (tmp_path / directory).mkdir()
        for index, row in enumerate(
            item for item in membership if item["directory"] == directory
        ):
            _metadata_event(
                tmp_path / directory / f"event-{index:03d}.npz",
                f"{directory}-{index:03d}",
                row,
            )
    path = tmp_path / "test/event-000.npz"
    row = next(item for item in membership if item["directory"] == "train")
    _metadata_event(path, "test-000", row)
    with pytest.raises(ValueError, match="not assigned to role evaluation"):
        validate_prepared_split_ids(ROOT, tmp_path)


def test_provider_zip_directory_identity_matches_without_payload_reads(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "UrbanFloodCast_Dataset.zip"
    rows = _membership()
    with zipfile.ZipFile(archive, "w") as bundle:
        for row in rows:
            bundle.writestr(row["provider_relative_path"] + "rain.txt", "")
    result = validate_provider_archive_membership(ROOT, archive)
    assert result["provider_event_directories"] == 125
    assert result["payload_members_opened"] == 0

    missing = tmp_path / "missing.zip"
    with zipfile.ZipFile(missing, "w") as bundle:
        for row in rows[1:]:
            bundle.writestr(row["provider_relative_path"] + "rain.txt", "")
    with pytest.raises(ValueError, match="missing=1"):
        validate_provider_archive_membership(ROOT, missing)


def test_static_contract_rejects_event_summary_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "configs", root / "configs")
    shutil.copytree(ROOT / "results/paper", root / "results/paper")
    path = root / "results/paper/event_level_results.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["target_wis"] = str(float(rows[0]["target_wis"]) + 0.1)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="event mean .* does not match"):
        validate_static_contract(root)
