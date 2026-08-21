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
from matplotlib.colors import ListedColormap


EXPECTED_ROWS = {
    "operational_contrasts.csv": 6,
    "operational_evaluation_specification.csv": 1,
    "target_calibration.csv": 4,
    "crc_calibration.csv": 2,
    "deployment_budget_effects.csv": 9,
    "warning_rule_migration.csv": 36,
    "claim_evidence.csv": 14,
    "baseline_fairness.csv": 4,
    "hpo_candidates.csv": 24,
    "reproducibility_scope.csv": 9,
}

WARNING_CODES = {
    "global": "Gl",
    "lead": "Ld",
    "magnitude": "Mg",
    "local_quantile": "Nl",
    "node_local": "Nl",
    "normalized_cp": "Nn",
    "node_normalized": "Nn",
    "severity_norm": "Ts",
    "train_severity": "Ts",
}

WARNING_LOSS_NAME = "prevalence-weighted event-macro loss L_prev=(r*FN+FP)/N"
WARNING_NEAR_TIE_GAP = 1.0e-4
CRC_RESAMPLING = "event bootstrap with replacement"
CRC_EMPTY_EVENT_POLICY = (
    "exclude symmetrically; estimand conditional on nonempty "
    "prediction-selected events"
)

REPRODUCIBILITY_SCOPE_CONTRACT = {
    "standard_point_metrics": {
        "public_evidence": "event_level_results.csv",
        "reproduction_level": "event-summary recomputation",
        "statistics_recomputed": "true",
        "requires_provider_data_or_checkpoints": "false",
        "scope": (
            "seven non-CSI metrics (H-RMSE, NSE, wet RMSE, wet NSE, peak "
            "error, ACE, and WIS) are recomputed for four standard systems "
            "from 48 public event rows"
        ),
    },
    "standard_CSI": {
        "public_evidence": "main_results.csv",
        "reproduction_level": "aggregate regeneration",
        "statistics_recomputed": "false",
        "requires_provider_data_or_checkpoints": "false",
        "scope": (
            "four CSI thresholds are validated and regenerated from reported "
            "aggregate rows; event-level CSI is not publicly recomputed"
        ),
    },
    "paired_baseline_inference": {
        "public_evidence": "event_level_results.csv",
        "reproduction_level": "paired-event recomputation",
        "statistics_recomputed": "true",
        "requires_provider_data_or_checkpoints": "false",
        "scope": (
            "six relative effects, bootstrap intervals, exact sign-flip tests, "
            "and Holm adjustments are recomputed from paired public event rows"
        ),
    },
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
    crc_rows = payload["crc_calibration.csv"]
    if [row["setting"] for row in crc_rows] != ["Berlin I", "Berlin II"]:
        raise ValueError("CRC setting order changed")
    for row in crc_rows:
        n = int(row["calibration_events"])
        alpha = _finite(row, "miscoverage_alpha")
        limit = _finite(row, "maximum_empirical_event_risk")
        expected_limit = (alpha * (n + 1) - 1.0) / n
        if abs(limit - expected_limit) > 1e-15:
            raise ValueError("CRC finite-sample correction changed")
        point = _finite(row, "delta_ace")
        lower = _finite(row, "delta_ace_ci_lower")
        upper = _finite(row, "delta_ace_ci_upper")
        if not lower <= point <= upper:
            raise ValueError("CRC confidence interval does not contain its point")
        if (
            row["statistics_recomputed"].lower() != "false"
            or int(row["bootstrap_replicates"]) != 2000
            or int(row["bootstrap_seed"]) != 20260801
            or row["independent_unit"] != "rainfall event"
            or row["calibration_resampling"] != CRC_RESAMPLING
            or row["evaluation_resampling"] != CRC_RESAMPLING
            or row["calibrator_refit_each_replicate"].lower() != "true"
            or row["common_method_indices"].lower() != "true"
            or row["empty_event_policy"] != CRC_EMPTY_EVENT_POLICY
        ):
            raise ValueError("public CRC resampling and refit contract changed")

    budget_rows = payload["deployment_budget_effects.csv"]
    if [row["record_id"] for row in budget_rows] != [
        "ufc_berlin_i_b05",
        "ufc_berlin_i_b10",
        "ufc_berlin_i_b20",
        "ufc_berlin_i_b30",
        "ufc_berlin_i_b50",
        "ufc_berlin_ii_b05",
        "ufc_berlin_ii_mid",
        "ufb_california_mid",
        "ufb_tennessee_mid",
    ]:
        raise ValueError("deployment-budget record order changed")
    for row in budget_rows:
        budget = _finite(row, "budget_fraction")
        prevalence = _finite(row, "prevalence")
        kappa = _finite(row, "kappa")
        point = _finite(row, "effect")
        lower = _finite(row, "ci_lower")
        upper = _finite(row, "ci_upper")
        if prevalence <= 0.0 or abs(kappa - budget / prevalence) > 1e-12:
            raise ValueError("deployment kappa does not equal budget/prevalence")
        if not lower <= point <= upper:
            raise ValueError("deployment confidence interval does not contain its point")
        if int(row["n_units"]) < 2 or row["statistics_recomputed"].lower() != "false":
            raise ValueError("deployment summary contract changed")

    warning_rows = payload["warning_rule_migration.csv"]
    expected_grids = {
        ("UFB", "California"): ({0.3, 0.5, 1.0}, {2, 5, 10, 20, 50, 100}),
        ("UFC", "Berlin I"): ({0.1, 0.3, 0.5}, {2, 5, 10, 20, 50, 100}),
    }
    for identity, (thresholds, costs) in expected_grids.items():
        selected = [
            row
            for row in warning_rows
            if (row["benchmark"], row["setting"]) == identity
        ]
        observed = {
            (_finite(row, "H_m"), int(row["miss_to_false_alarm_ratio"]))
            for row in selected
        }
        expected = {(threshold, cost) for threshold in thresholds for cost in costs}
        if observed != expected or len(selected) != len(expected):
            raise ValueError(f"warning-rule grid changed for {identity}")
    for row in warning_rows:
        strategy = row["winner_strategy"]
        if WARNING_CODES.get(strategy) != row["winner_code"]:
            raise ValueError("warning-rule code does not match its strategy")
        gap = _finite(row, "runner_up_loss_gap")
        if _finite(row, "winner_loss") < 0.0 or gap < 0.0:
            raise ValueError("warning loss and runner-up gap must be nonnegative")
        if row["loss_name"] != WARNING_LOSS_NAME:
            raise ValueError("warning-rule loss definition changed")
        expected_near_tie = gap <= WARNING_NEAR_TIE_GAP
        if (row["near_tie"].lower() == "true") != expected_near_tie:
            raise ValueError("warning-rule near-tie marker does not match its gap")
        if row["interpretation"] != "descriptive test-event ranking":
            raise ValueError("warning-rule ranking lost its descriptive boundary")
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
        "archived-summary regeneration",
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
    scope_by_claim = {row["claim_group"]: row for row in scope_rows}
    if len(scope_by_claim) != len(scope_rows):
        raise ValueError("reproducibility scope claim groups must be unique")
    for claim_group, expected in REPRODUCIBILITY_SCOPE_CONTRACT.items():
        if scope_by_claim.get(claim_group) != {
            "claim_group": claim_group,
            **expected,
        }:
            raise ValueError(
                f"reproducibility scope changed for {claim_group}"
            )
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


def _deployment_plot(
    budget_rows: Sequence[Mapping[str, str]],
    warning_rows: Sequence[Mapping[str, str]],
    output_dir: Path,
) -> None:
    """Regenerate the paper's two-panel RQ3/RQ4 summary from public rows."""

    figure = plt.figure(figsize=(11.8, 5.4))
    grid = figure.add_gridspec(2, 2, width_ratios=(1.45, 1.0), hspace=0.48)
    budget_axis = figure.add_subplot(grid[:, 0])
    styles = {
        ("UFC", "Berlin I"): ("#2F5597", "o", "UFC Berlin I"),
        ("UFC", "Berlin II"): ("#C54A45", "s", "UFC Berlin II"),
        ("UFB", "California"): ("#4C8B3B", "^", "UFB California"),
        ("UFB", "Tennessee"): ("#D9822B", "D", "UFB Tennessee"),
    }
    for identity, (color, marker, label) in styles.items():
        rows = [
            row
            for row in budget_rows
            if (row["benchmark"], row["setting"]) == identity
        ]
        x = [float(row["kappa"]) for row in rows]
        y = [float(row["effect"]) for row in rows]
        lower = [float(row["ci_lower"]) for row in rows]
        upper = [float(row["ci_upper"]) for row in rows]
        budget_axis.errorbar(
            x,
            y,
            yerr=(
                [point - lo for point, lo in zip(y, lower, strict=True)],
                [hi - point for point, hi in zip(y, upper, strict=True)],
            ),
            color=color,
            marker=marker,
            linestyle="-" if identity == ("UFC", "Berlin I") else "none",
            capsize=3,
            linewidth=1.4,
            label=label,
        )
    budget_axis.axhline(0.0, color="#555555", linestyle="--", linewidth=0.8)
    budget_axis.set_xlabel(r"Normalized budget $\kappa=b/\pi$")
    budget_axis.set_ylabel(r"Capture difference $\Delta$")
    budget_axis.set_title("(a) Forcing value over normalized budget")
    budget_axis.grid(alpha=0.22)
    budget_axis.legend(fontsize=8, loc="upper right")

    code_order = ("Gl", "Ld", "Mg", "Nl", "Nn", "Ts")
    palette = ("#D9D9D9", "#C9D7F4", "#BFE7ED", "#C7F2C7", "#F6D9A7", "#F4B8B8")
    code_index = {code: index for index, code in enumerate(code_order)}
    cmap = ListedColormap(palette)
    heatmap_specs = (
        (("UFB", "California"), (0.3, 0.5, 1.0), "UFB California (development)"),
        (("UFC", "Berlin I"), (0.1, 0.3, 0.5), "UFC Berlin I (external evaluation)"),
    )
    costs = (2, 5, 10, 20, 50, 100)
    for row_index, (identity, thresholds, title) in enumerate(heatmap_specs):
        axis = figure.add_subplot(grid[row_index, 1])
        selected = {
            (float(row["H_m"]), int(row["miss_to_false_alarm_ratio"])): row
            for row in warning_rows
            if (row["benchmark"], row["setting"]) == identity
        }
        matrix = [
            [code_index[selected[(threshold, cost)]["winner_code"]] for cost in costs]
            for threshold in thresholds
        ]
        axis.imshow(matrix, cmap=cmap, vmin=-0.5, vmax=len(code_order) - 0.5, aspect="auto")
        for y_index, threshold in enumerate(thresholds):
            for x_index, cost in enumerate(costs):
                row = selected[(threshold, cost)]
                axis.text(
                    x_index,
                    y_index,
                    row["winner_code"] + ("*" if row["near_tie"].lower() == "true" else ""),
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        axis.set_xticks(range(len(costs)), costs)
        axis.set_yticks(range(len(thresholds)), thresholds)
        axis.set_ylabel(r"$H$ (m)")
        axis.set_title(title, fontsize=9, fontweight="bold")
        if row_index == 1:
            axis.set_xlabel(r"Miss-to-false-alarm cost ratio $r$")
    legend = "  ".join(f"{code}: {name}" for code, name in zip(
        code_order,
        ("global", "lead", "magnitude", "node-local", "node-normalized", "train-severity"),
        strict=True,
    ))
    figure.text(0.70, 0.018, legend, ha="center", fontsize=7)
    figure.text(
        0.70,
        -0.012,
        r"$*$ winner--runner-up loss gap $\leq 10^{-4}$; "
        r"$L_{\mathrm{prev},e}=(r\,\mathrm{FN}_e+\mathrm{FP}_e)/N_e$",
        ha="center",
        fontsize=7,
    )
    figure.suptitle("Deployment boundaries induced by budget and warning loss", fontsize=12)
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in (("png", 300), ("pdf", 300)):
        metadata = (
            {"CreationDate": None, "ModDate": None}
            if suffix == "pdf"
            else None
        )
        figure.savefig(
            output_dir / f"deployment_boundaries.{suffix}",
            dpi=dpi,
            bbox_inches="tight",
            metadata=metadata,
        )
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
        "crc_calibration.md": (
            "crc_calibration.csv",
            (("setting", "Setting"), ("calibration_events", "Calibration events"), ("maximum_empirical_event_risk", "Maximum empirical event risk"), ("domain_coverage", "Domain coverage"), ("target_coverage", "Target coverage"), ("delta_ace", "Delta ACE"), ("delta_ace_ci_lower", "CI lower"), ("delta_ace_ci_upper", "CI upper"), ("calibrator_refit_each_replicate", "Refit each replicate"), ("empty_event_policy", "Empty-event policy"), ("interpretation", "Interpretation")),
        ),
        "deployment_budget_effects.md": (
            "deployment_budget_effects.csv",
            (("setting", "Setting"), ("predictor_contrast", "Predictor contrast"), ("budget_fraction", "Budget"), ("prevalence", "Prevalence"), ("kappa", "Kappa"), ("effect", "Effect"), ("ci_lower", "CI lower"), ("ci_upper", "CI upper")),
        ),
        "warning_rule_migration.md": (
            "warning_rule_migration.csv",
            (("setting", "Setting"), ("H_m", "H (m)"), ("miss_to_false_alarm_ratio", "Cost ratio"), ("loss_name", "Loss"), ("winner_strategy", "Observed lowest-loss strategy"), ("winner_loss", "Winner loss"), ("runner_up_loss_gap", "Runner-up gap"), ("near_tie", "Near tie"), ("interpretation", "Boundary")),
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
    _deployment_plot(
        payload["deployment_budget_effects.csv"],
        payload["warning_rule_migration.csv"],
        output_dir,
    )
    return {
        "status": "PASS",
        "operational_rows": len(payload["operational_contrasts.csv"]),
        "operational_specification_rows": len(
            payload["operational_evaluation_specification.csv"]
        ),
        "target_calibration_rows": len(payload["target_calibration.csv"]),
        "crc_calibration_rows": len(payload["crc_calibration.csv"]),
        "deployment_budget_rows": len(payload["deployment_budget_effects.csv"]),
        "warning_rule_rows": len(payload["warning_rule_migration.csv"]),
        "warning_near_tie_rows": sum(
            row["near_tie"].lower() == "true"
            for row in payload["warning_rule_migration.csv"]
        ),
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
