from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cano_bigdata2026.public_contract import (
    validate_prepared_split_ids,
    validate_static_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_public_contract() -> None:
    result = validate_static_contract(ROOT)
    assert result == {
        "status": "PASS",
        "checkpoint_selection_metric": "development_event_macro_physical_h_rmse",
        "public_role_rows": 125,
        "hpo_candidate_rows": 24,
        "reproducibility_scope_rows": 7,
        "raw_field_arrays_opened": 0,
    }


def _metadata_event(path: Path, event_id: str) -> None:
    np.savez_compressed(path, event_id=np.asarray(event_id))


def test_prepared_split_identity_check_reads_metadata_only(tmp_path: Path) -> None:
    specs = (("train", 85), ("validation", 15), ("calibration", 13), ("test", 12))
    for directory, count in specs:
        (tmp_path / directory).mkdir()
        for index in range(count):
            _metadata_event(
                tmp_path / directory / f"event-{index:03d}.npz",
                f"{directory}-{index:03d}",
            )
    result = validate_prepared_split_ids(ROOT, tmp_path)
    assert result["unique_event_ids"] == 125
    assert result["field_arrays_opened"] == 0
    assert result["event_id_metadata_only"] is True

    duplicate = tmp_path / "test/event-000.npz"
    _metadata_event(duplicate, "train-000")
    with pytest.raises(ValueError, match="overlaps prepared roles"):
        validate_prepared_split_ids(ROOT, tmp_path)
