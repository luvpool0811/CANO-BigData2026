from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
import pytest

from cano_bigdata2026.results import reproduce


ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_event_level_means_match_main_results() -> None:
    main = {
        row["system"].split(" (")[0]: row
        for row in _read(ROOT / "results/paper/main_results.csv")
        if row["comparison"] == "standard"
    }
    events = _read(ROOT / "results/paper/event_level_results.csv")
    for system, summary in main.items():
        selected = [row for row in events if row["system"] == system]
        assert len(selected) == 12
        for key in (
            "h_rmse_m",
            "nse",
            "wet_rmse_m",
            "wet_nse",
            "peak_depth_abs_error_m",
            "target_ace",
            "target_wis",
        ):
            observed = np.mean([float(row[key]) for row in selected])
            assert np.isclose(observed, float(summary[key]), rtol=0, atol=1e-12)


def test_reproduce_public_results(tmp_path: Path) -> None:
    result = reproduce(
        results_dir=ROOT / "results/paper", output_dir=tmp_path / "generated"
    )
    assert result["status"] == "PASS"
    assert result["rows"] == 6
    assert result["event_rows"] == 48
    assert result["statistics_recomputed"] is True
    assert result["standard_systems_recomputed"] == 4
    assert result["event_rows_recomputed"] == 48
    assert len(result["metrics_recomputed"]) == 7
    table = (tmp_path / "generated/main_results.md").read_text(encoding="utf-8")
    assert "## Standard-objective comparison" in table
    assert "## CANO training-objective ablation" in table
    assert table.count("CANO (standard objective)") == 2
    assert table.count("CANO (selection-matched control)") == 1
    assert table.count("CANO (peak-aware objective)") == 1
    for name in (
        "point_prediction_metrics.png",
        "uncertainty_metrics.png",
        "cano_objective_ablation.png",
        "event_level_h_rmse.png",
    ):
        assert (tmp_path / "generated" / name).stat().st_size > 1_000


def test_reproduce_rejects_event_summary_mismatch(tmp_path: Path) -> None:
    results_dir = tmp_path / "paper"
    shutil.copytree(ROOT / "results/paper", results_dir)
    path = results_dir / "event_level_results.csv"
    rows = _read(path)
    rows[0]["h_rmse_m"] = str(float(rows[0]["h_rmse_m"]) + 0.1)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="event mean .* does not match"):
        reproduce(results_dir=results_dir, output_dir=tmp_path / "generated")
