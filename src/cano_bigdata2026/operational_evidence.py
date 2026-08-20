"""Regenerate the paper's central operational-reliability disclosure artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPECTED_ROWS = {
    "operational_contrasts.csv": 6,
    "operational_evaluation_specification.csv": 1,
    "target_calibration.csv": 4,
    "claim_evidence.csv": 11,
    "baseline_fairness.csv": 4,
    "hpo_candidates.csv": 24,
    "reproducibility_scope.csv": 7,
}


def _validate_rq1_specification(
    payload: Mapping[str, list[dict[str, str]]],
) -> None:
    """Bind the reader-facing RQ1 contract to the reported contrast."""

    spec = payload["operational_evaluation_specification.csv"][0]
    contrast = next(
        row
        for row in payload["operational_contrasts.csv"]
        if row["benchmark"] == "UrbanFloodCast" and row["setting"] == "Berlin I"
    )
    claim = next(
        row
        for row in payload["claim_evidence.csv"]
        if row["claim_id"] == "RQ1-UFC-BerlinI"
    )
    exact_text = {
        "claim_id": "RQ1-UFC-BerlinI",
        "point_predictor": "protocol-aligned DNO",
        "predictor_initialization": "seed 42",
        "reference_population": "all valid cell-times across 24 forecast leads",
        "operational_population": (
            "prediction-selected cell-times with predicted H >= 0.30 m"
        ),
        "selection_information": (
            "prediction only; target residuals and interval endpoints do not "
            "define membership"
        ),
        "interval_rule": "two-sided global absolute-residual band [mu-q, mu+q]",
        "nonconformity_score": "absolute H residual |y-mu| in physical metres",
        "calibration_weighting": "pooled valid cell-times with one global q",
        "reporting_unit": "rainfall event",
        "coverage_aggregation": (
            "equal event-macro mean after within-event cell-time coverage"
        ),
        "estimand": "full-field minus prediction-selected coverage",
    }
    for key, expected in exact_text.items():
        if spec[key] != expected:
            raise ValueError(f"RQ1 specification changed: {key}")
    numeric_links = {
        "full_coverage": "full_coverage",
        "selected_coverage": "selected_coverage",
        "population_effect": "population_effect",
        "population_ci_lower": "population_ci_lower",
        "population_ci_upper": "population_ci_upper",
    }
    for spec_key, contrast_key in numeric_links.items():
        if abs(_finite(spec, spec_key) - _finite(contrast, contrast_key)) > 1e-15:
            raise ValueError(f"RQ1 specification differs from contrast: {spec_key}")
    if (
        _finite(spec, "operational_threshold_m") != 0.30
        or _finite(spec, "miscoverage_alpha") != 0.10
        or int(spec["bootstrap_replicates"]) != int(contrast["bootstrap_replicates"])
        or int(spec["bootstrap_seed"]) != 20260731
        or spec["calibrator_refit_in_bootstrap"].lower() != "false"
        or contrast["calibrator_refit"].lower() != "no"
        or spec["resampling"]
        != "paired rainfall-event bootstrap on the 12 within-event contrasts"
        or "paired event bootstrap" not in claim["resampling"]
        or claim["independent_unit"] != "rainfall event"
        or "fixed fitted calibrator" not in claim["calibrator_treatment"]
    ):
        raise ValueError("RQ1 resampling or calibration contract changed")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_ROWS[path.name]:
        raise ValueError(f"{path.name} has {len(rows)} rows; expected {EXPECTED_ROWS[path.name]}")
    if not rows or any(value is None for row in rows for value in row.values()):
        raise ValueError(f"{path.name} contains an incomplete record")
    return rows


def _finite(row: Mapping[str, str], key: str, *, optional: bool = False) -> float | None:
    raw = row[key].strip()
    if optional and not raw:
        return None
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def validate_operational_inputs(results_dir: Path) -> dict[str, list[dict[str, str]]]:
    payload = {name: _rows(results_dir / name) for name in EXPECTED_ROWS}
    contrasts = payload["operational_contrasts.csv"]
    for row in contrasts:
        n_units = int(row["n_units"])
        if n_units < 2:
            raise ValueError("each operational contrast requires at least two units")
        for prefix in ("population", "axis"):
            point = _finite(row, f"{prefix}_effect")
            lower = _finite(row, f"{prefix}_ci_lower")
            upper = _finite(row, f"{prefix}_ci_upper")
            if not (lower <= point <= upper):
                raise ValueError(f"{prefix} confidence interval does not contain its point")
        full = _finite(row, "full_coverage", optional=True)
        selected = _finite(row, "selected_coverage", optional=True)
        if (full is None) != (selected is None):
            raise ValueError("full and selected coverage must be jointly present or absent")
        if full is not None and abs((full - selected) - float(row["population_effect"])) > 1e-12:
            raise ValueError("population effect does not equal full minus selected coverage")
    settings = [row["setting"] for row in payload["target_calibration.csv"]]
    if settings != ["California", "Tennessee", "Berlin I", "Berlin II"]:
        raise ValueError("target-calibration setting order changed")
    claims = payload["claim_evidence.csv"]
    if len({row["claim_id"] for row in claims}) != len(claims):
        raise ValueError("claim identifiers must be unique")
    systems = [row["system"] for row in payload["baseline_fairness.csv"]]
    if systems != ["CANO", "DNO-3", "FNO3D", "U-Net3D"]:
        raise ValueError("baseline fairness system order changed")
    if any(row["test_used_for_selection"].lower() != "false" for row in payload["baseline_fairness.csv"]):
        raise ValueError("evaluation data may not enter setting or checkpoint selection")
    if any(
        row["checkpoint_selection"]
        != "minimum development event-macro physical-H RMSE"
        for row in payload["baseline_fairness.csv"]
    ):
        raise ValueError("checkpoint selection must match the public training code")
    if any(
        not row["shared_budget_contract"].strip()
        or not row["per_update_supervision"].strip()
        for row in payload["baseline_fairness.csv"]
    ):
        raise ValueError("training budget and per-update supervision must be explicit")
    hpo_rows = payload["hpo_candidates.csv"]
    expected_systems = ("CANO", "DNO-3", "FNO3D", "U-Net3D")
    for system in expected_systems:
        candidates = [row for row in hpo_rows if row["system"] == system]
        if [int(row["candidate_index"]) for row in candidates] != list(range(6)):
            raise ValueError(f"{system} must disclose candidates 0 through 5")
        scores = [_finite(row, "best_development_event_macro_h_rmse_m") for row in candidates]
        selected = [row for row in candidates if row["selected"].lower() == "true"]
        if len(selected) != 1:
            raise ValueError(f"{system} must disclose exactly one selected candidate")
        winner = min(range(6), key=lambda index: (scores[index], index))
        if int(selected[0]["candidate_index"]) != winner:
            raise ValueError(f"{system} selection does not minimize development H-RMSE")
        if any(int(row["selection_seed"]) != 42 for row in candidates):
            raise ValueError(f"{system} HPO selection seed changed")
        for row in candidates:
            if not isinstance(json.loads(row["model_config"]), dict):
                raise ValueError("HPO model_config must contain a JSON object")
    scope_rows = payload["reproducibility_scope.csv"]
    allowed_levels = {
        "event-summary recomputation",
        "paired-event recomputation",
        "aggregate regeneration",
        "summary regeneration",
        "code path only",
    }
    if any(row["reproduction_level"] not in allowed_levels for row in scope_rows):
        raise ValueError("reproducibility scope contains an unknown level")
    if any(
        row["statistics_recomputed"].lower() not in {"true", "false"}
        or row["requires_provider_data_or_checkpoints"].lower() not in {"true", "false"}
        for row in scope_rows
    ):
        raise ValueError("reproducibility scope booleans are invalid")
    _validate_rq1_specification(payload)
    return payload


def _markdown(rows: Sequence[Mapping[str, str]], columns: Sequence[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(str(row[key]).replace("|", "\\|") for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join((header, rule, *body)) + "\n"


def _effect_plot(rows: Sequence[Mapping[str, str]], output: Path) -> None:
    labels = [row["setting"] for row in rows]
    y = list(reversed(range(len(rows))))
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    panels = (
        ("population", r"$C_{all}-C_{selected}$"),
        ("axis", r"$C_{magnitude}-C_{lead}$"),
    )
    for axis, (prefix, title) in zip(axes, panels, strict=True):
        points = [float(row[f"{prefix}_effect"]) for row in rows]
        lower = [float(row[f"{prefix}_ci_lower"]) for row in rows]
        upper = [float(row[f"{prefix}_ci_upper"]) for row in rows]
        axis.errorbar(
            points,
            y,
            xerr=[
                [point - lo for point, lo in zip(points, lower, strict=True)],
                [hi - point for point, hi in zip(points, upper, strict=True)],
            ],
            fmt="o",
            color="#2F5597",
            capsize=3,
            linewidth=1.3,
        )
        axis.axvline(0.0, color="#555555", linestyle="--", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Coverage contrast")
        axis.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(y, labels)
    figure.suptitle("Operational population and calibration-axis effects")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def reproduce(*, results_dir: Path, output_dir: Path) -> dict[str, object]:
    payload = validate_operational_inputs(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "operational_evaluation_specification.md": (
            "operational_evaluation_specification.csv",
            (
                ("claim_id", "Claim"),
                ("point_predictor", "Predictor"),
                ("prediction_archive", "Prediction archive"),
                ("operational_population", "Operational population"),
                ("interval_rule", "Interval rule"),
                ("calibration_role", "Calibration role"),
                ("evaluation_role", "Evaluation role"),
                ("reporting_unit", "Unit"),
                ("resampling", "Uncertainty"),
                ("claim_boundary", "Boundary"),
            ),
        ),
        "operational_contrasts.md": (
            "operational_contrasts.csv",
            (("setting", "Setting"), ("evidence_role", "Role"), ("n_units", "n"), ("population_effect", "Population effect"), ("axis_effect", "Axis effect"), ("resampling", "Resampling")),
        ),
        "target_calibration.md": (
            "target_calibration.csv",
            (("setting", "Setting"), ("calibration_units", "Calibration units"), ("domain_coverage", "Domain coverage"), ("target_coverage", "Target coverage"), ("delta_ace", "Delta ACE"), ("relative_winkler", "Relative Winkler")),
        ),
        "claim_evidence.md": (
            "claim_evidence.csv",
            (("claim_id", "Claim"), ("estimand", "Estimand"), ("n", "n"), ("independent_unit", "Unit"), ("resampling", "Resampling"), ("calibrator_treatment", "Calibrator"), ("multiplicity", "Multiplicity"), ("prespecification", "Status")),
        ),
        "baseline_fairness.md": (
            "baseline_fairness.csv",
            (("system", "System"), ("candidate_settings", "Candidates"), ("setting_selection_rule", "Setting rule"), ("seeds", "Seeds"), ("total_completed_epochs", "Epochs"), ("total_optimizer_steps", "Optimizer steps"), ("reported_training_hours", "Training hours")),
        ),
        "hpo_candidates.md": (
            "hpo_candidates.csv",
            (("system", "System"), ("candidate_index", "Candidate"), ("model_config", "Configuration"), ("best_development_event_macro_h_rmse_m", "Best development H-RMSE (m)"), ("best_epoch", "Epoch"), ("selected", "Selected")),
        ),
        "reproducibility_scope.md": (
            "reproducibility_scope.csv",
            (("claim_group", "Claim group"), ("public_evidence", "Public evidence"), ("reproduction_level", "Level"), ("statistics_recomputed", "Statistics recomputed"), ("requires_provider_data_or_checkpoints", "External inputs required"), ("scope", "Scope")),
        ),
    }
    for filename, (source, columns) in tables.items():
        (output_dir / filename).write_text(
            _markdown(payload[source], columns), encoding="utf-8"
        )
    _effect_plot(
        payload["operational_contrasts.csv"],
        output_dir / "operational_reliability_effects.png",
    )
    return {
        "status": "PASS",
        "operational_rows": len(payload["operational_contrasts.csv"]),
        "operational_specification_rows": len(
            payload["operational_evaluation_specification.csv"]
        ),
        "target_calibration_rows": len(payload["target_calibration.csv"]),
        "claim_rows": len(payload["claim_evidence.csv"]),
        "fairness_rows": len(payload["baseline_fairness.csv"]),
        "hpo_candidate_rows": len(payload["hpo_candidates.csv"]),
        "reproducibility_scope_rows": len(payload["reproducibility_scope.csv"]),
        "statistics_recomputed": False,
        "output_dir": str(output_dir),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--results-dir", type=Path, default=root / "results/paper")
    parser.add_argument("--output-dir", type=Path, default=root / "results/generated")
    args = parser.parse_args(argv)
    print(reproduce(results_dir=args.results_dir, output_dir=args.output_dir))
    return 0


__all__ = ["reproduce", "validate_operational_inputs", "main"]
