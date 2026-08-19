from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from cano_bigdata2026.inference import (
    compute_primary_inference,
    reduction_rows,
    reproduce_inference,
)


ROOT = Path(__file__).resolve().parents[1]


def test_primary_inference_matches_reported_contrasts() -> None:
    rows = compute_primary_inference(ROOT / "results/paper/event_level_results.csv")
    assert len(rows) == 6
    expected = {
        ("h_rmse_m", "DNO-3"): (-0.32307376519397346, -0.37500115581927557, -0.270106113482621),
        ("h_rmse_m", "FNO3D"): (-0.3366050101164526, -0.38555396238358214, -0.2877880234785748),
        ("h_rmse_m", "U-Net3D"): (-0.4877957509988199, -0.5679247068206124, -0.40389665219007753),
        ("target_wis", "DNO-3"): (-0.44973334796986586, -0.5500989951759441, -0.3467121141946786),
        ("target_wis", "FNO3D"): (-0.2754007443579052, -0.39875751255643305, -0.1538457716183805),
        ("target_wis", "U-Net3D"): (-0.6605856833058785, -0.7618320317452032, -0.563339483123211),
    }
    for row in rows:
        point, lower, upper = expected[(str(row["endpoint"]), str(row["comparator"]))]
        assert np.isclose(float(row["relative_effect"]), point, rtol=0, atol=1e-15)
        assert np.isclose(float(row["lower_95"]), lower, rtol=0, atol=1e-15)
        assert np.isclose(float(row["upper_95"]), upper, rtol=0, atol=1e-15)
        assert float(row["holm_adjusted_p_two_sided"]) == 0.00146484375
        assert row["directional_support"] is True


def test_reproduce_inference_outputs(tmp_path: Path) -> None:
    output_csv = tmp_path / "paired.csv"
    output_md = tmp_path / "paired.md"
    output_pdf = tmp_path / "forest.pdf"
    output_png = tmp_path / "forest.png"
    result = reproduce_inference(
        event_csv=ROOT / "results/paper/event_level_results.csv",
        output_csv=output_csv,
        output_markdown=output_md,
        output_figure_pdf=output_pdf,
        output_figure_png=output_png,
    )
    assert result["status"] == "PASS"
    assert result["contrasts"] == 6
    assert result["all_supported"] is True
    with output_csv.open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 6
    text = output_md.read_text(encoding="utf-8")
    assert "5,000-replicate paired-event bootstrap" in text
    assert "Holm-adjusted" in text
    assert output_pdf.stat().st_size > 1_000
    assert output_png.stat().st_size > 10_000


def test_reduction_transform_reverses_interval_bounds() -> None:
    rows = reduction_rows(
        compute_primary_inference(ROOT / "results/paper/event_level_results.csv")
    )
    first = rows[0]
    assert first["comparator"] == "DNO-3"
    assert np.isclose(first["reduction_pct"], 32.307376519397346)
    assert np.isclose(first["reduction_lower_95_pct"], 27.0106113482621)
    assert np.isclose(first["reduction_upper_95_pct"], 37.50011558192756)
