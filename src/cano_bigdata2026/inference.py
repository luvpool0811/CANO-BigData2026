"""Reproduce the six prespecified paired comparisons from public event records."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260731
FAMILYWISE_ALPHA = 0.05
PRIMARY_SYSTEM = "CANO"
COMPARATORS = ("DNO-3", "FNO3D", "U-Net3D")
ENDPOINTS = ("h_rmse_m", "target_wis")


def exact_sign_flip_pvalue(event_differences: np.ndarray) -> float:
    values = np.asarray(event_differences, dtype=np.float64).reshape(-1)
    if not 2 <= values.size <= 20 or not np.isfinite(values).all():
        raise ValueError("exact sign flip requires 2--20 finite event differences")
    patterns = np.arange(1 << values.size, dtype=np.uint64)[:, None]
    bits = (patterns >> np.arange(values.size, dtype=np.uint64)) & 1
    signs = np.where(bits == 1, 1.0, -1.0)
    null = np.mean(signs * values[None, :], axis=1)
    observed = abs(float(np.mean(values)))
    return float(np.mean(np.abs(null) >= observed - 1e-15))


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    raw = np.asarray(p_values, dtype=np.float64)
    if (
        raw.ndim != 1
        or raw.size == 0
        or not np.isfinite(raw).all()
        or np.any((raw < 0.0) | (raw > 1.0))
    ):
        raise ValueError("p-values must be a nonempty finite vector in [0,1]")
    order = np.argsort(raw, kind="stable")
    ranked = np.maximum.accumulate((raw.size - np.arange(raw.size)) * raw[order])
    adjusted = np.empty_like(raw)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def paired_relative_bootstrap(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    left = np.asarray(candidate, dtype=np.float64).reshape(-1)
    right = np.asarray(reference, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size < 2:
        raise ValueError("candidate and reference require paired event vectors")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("paired event vectors must be finite")
    if np.any(right <= 0.0):
        raise ValueError("relative contrast requires positive reference values")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, left.size, size=(int(replicates), left.size))
    bootstrap = np.mean(left[indices], axis=1) / np.mean(right[indices], axis=1) - 1.0
    lower, upper = np.percentile(bootstrap, [2.5, 97.5])
    return {
        "n_events": int(left.size),
        "replicates": int(replicates),
        "seed": int(seed),
        "relative_effect": float(np.mean(left) / np.mean(right) - 1.0),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "exact_sign_flip_p_two_sided": exact_sign_flip_pvalue(left - right),
    }


def _event_vectors(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected_systems = (PRIMARY_SYSTEM, *COMPARATORS)
    grouped: dict[str, list[dict[str, str]]] = {
        system: [row for row in rows if row.get("system") == system]
        for system in expected_systems
    }
    if any(len(grouped[system]) != 12 for system in expected_systems):
        raise ValueError("the public primary comparison requires 12 events per system")
    event_order = [row["event"] for row in grouped[PRIMARY_SYSTEM]]
    if len(set(event_order)) != 12:
        raise ValueError("CANO event identifiers are not unique")
    vectors: dict[str, dict[str, np.ndarray]] = {}
    for system in expected_systems:
        if [row["event"] for row in grouped[system]] != event_order:
            raise ValueError("paired systems do not share the same event order")
        vectors[system] = {
            endpoint: np.asarray(
                [float(row[endpoint]) for row in grouped[system]], dtype=np.float64
            )
            for endpoint in ENDPOINTS
        }
    return vectors


def compute_primary_inference(event_csv: str | Path) -> list[dict[str, object]]:
    vectors = _event_vectors(Path(event_csv))
    rows: list[dict[str, object]] = []
    for endpoint in ENDPOINTS:
        family: list[dict[str, object]] = []
        for comparator in COMPARATORS:
            result = paired_relative_bootstrap(
                vectors[PRIMARY_SYSTEM][endpoint], vectors[comparator][endpoint]
            )
            family.append(
                {
                    "candidate": PRIMARY_SYSTEM,
                    "comparator": comparator,
                    "endpoint": endpoint,
                    **result,
                }
            )
        adjusted = holm_adjust(
            [float(row["exact_sign_flip_p_two_sided"]) for row in family]
        )
        for row, value in zip(family, adjusted, strict=True):
            row["holm_adjusted_p_two_sided"] = float(value)
            row["directional_support"] = bool(
                float(row["upper_95"]) < 0.0 and float(value) < FAMILYWISE_ALPHA
            )
        rows.extend(family)
    return rows


def reduction_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Convert negative relative effects into positive reduction percentages."""

    expected = {
        (endpoint, comparator)
        for endpoint in ENDPOINTS
        for comparator in COMPARATORS
    }
    observed = {(str(row["endpoint"]), str(row["comparator"])) for row in rows}
    if len(rows) != 6 or observed != expected:
        raise ValueError("the forest plot requires the six prespecified comparisons")
    converted: list[dict[str, object]] = []
    for endpoint in ENDPOINTS:
        for comparator in COMPARATORS:
            row = next(
                item
                for item in rows
                if item["endpoint"] == endpoint and item["comparator"] == comparator
            )
            point = -100.0 * float(row["relative_effect"])
            lower = -100.0 * float(row["upper_95"])
            upper = -100.0 * float(row["lower_95"])
            if not (0.0 < lower <= point <= upper <= 80.0):
                raise ValueError(
                    "reduction confidence interval is outside the plot contract"
                )
            converted.append(
                {
                    **row,
                    "reduction_pct": point,
                    "reduction_lower_95_pct": lower,
                    "reduction_upper_95_pct": upper,
                }
            )
    return converted


def _forest_plot(rows: Sequence[dict[str, object]], pdf: Path, png: Path) -> None:
    converted = reduction_rows(rows)
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    figure, axis = plt.subplots(figsize=(3.45, 2.75))
    positions = (6.0, 5.0, 4.0, 2.4, 1.4, 0.4)
    colors = {"h_rmse_m": "#2166AC", "target_wis": "#D55E00"}
    markers = {"h_rmse_m": "o", "target_wis": "s"}
    for row, position in zip(converted, positions, strict=True):
        point = float(row["reduction_pct"])
        lower = float(row["reduction_lower_95_pct"])
        upper = float(row["reduction_upper_95_pct"])
        endpoint = str(row["endpoint"])
        axis.errorbar(
            point,
            position,
            xerr=[[point - lower], [upper - point]],
            fmt=markers[endpoint],
            color=colors[endpoint],
            markerfacecolor="white" if endpoint == "target_wis" else colors[endpoint],
            markeredgewidth=1.0,
            markersize=4.8,
            capsize=2.2,
            linewidth=1.2,
            zorder=3,
        )
        axis.text(
            1.03,
            position,
            f"{point:.2f} [{lower:.2f}, {upper:.2f}]",
            transform=axis.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=6.2,
        )
    axis.axvline(0.0, color="#555555", linestyle="--", linewidth=0.8)
    axis.set_xlim(0.0, 80.0)
    axis.set_ylim(-0.25, 6.75)
    axis.set_xticks((0, 20, 40, 60, 80))
    axis.set_yticks(positions, [str(row["comparator"]) for row in converted])
    axis.set_xlabel("Relative reduction (%)", fontsize=7.5)
    axis.tick_params(axis="both", labelsize=7)
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.text(
        -0.28,
        6.55,
        "H RMSE",
        transform=axis.get_yaxis_transform(),
        fontsize=7.2,
        fontweight="bold",
    )
    axis.text(
        -0.28,
        2.95,
        "Target WIS",
        transform=axis.get_yaxis_transform(),
        fontsize=7.2,
        fontweight="bold",
    )
    axis.text(
        1.03,
        6.55,
        "Reduction [95% CI]",
        transform=axis.get_yaxis_transform(),
        fontsize=6.2,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.27, right=0.65, bottom=0.18, top=0.94)
    for path in (pdf, png):
        path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = (
        "candidate",
        "comparator",
        "endpoint",
        "n_events",
        "replicates",
        "seed",
        "relative_effect",
        "lower_95",
        "upper_95",
        "exact_sign_flip_p_two_sided",
        "holm_adjusted_p_two_sided",
        "directional_support",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Prespecified paired comparisons",
        "",
        "Relative effects are CANO/reference - 1; negative values favor CANO.",
        "Confidence intervals use a 5,000-replicate paired-event bootstrap",
        "with seed 20260731. Two-sided exact paired sign-flip p-values are",
        "Holm-adjusted separately within the three H-RMSE and three Target-WIS comparisons.",
        "",
        "| Endpoint | Comparator | Relative effect | 95% CI | Raw p | Holm p | Support |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    labels = {"h_rmse_m": "H RMSE", "target_wis": "Target WIS"}
    for row in rows:
        lines.append(
            "| {endpoint} | {comparator} | {effect:.2%} | [{lower:.2%}, {upper:.2%}] | "
            "{raw:.8f} | {adjusted:.8f} | {support} |".format(
                endpoint=labels[str(row["endpoint"])],
                comparator=row["comparator"],
                effect=float(row["relative_effect"]),
                lower=float(row["lower_95"]),
                upper=float(row["upper_95"]),
                raw=float(row["exact_sign_flip_p_two_sided"]),
                adjusted=float(row["holm_adjusted_p_two_sided"]),
                support="yes" if row["directional_support"] else "no",
            )
        )
    lines.append("")
    return "\n".join(lines)


def reproduce_inference(
    *,
    event_csv: Path,
    output_csv: Path,
    output_markdown: Path,
    output_figure_pdf: Path | None = None,
    output_figure_png: Path | None = None,
) -> dict[str, object]:
    rows = compute_primary_inference(event_csv)
    _write_csv(rows, output_csv)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(_markdown(rows), encoding="utf-8")
    if (output_figure_pdf is None) != (output_figure_png is None):
        raise ValueError("both figure output paths must be supplied together")
    if output_figure_pdf is not None and output_figure_png is not None:
        _forest_plot(rows, output_figure_pdf, output_figure_png)
    return {
        "status": "PASS",
        "contrasts": len(rows),
        "all_supported": all(bool(row["directional_support"]) for row in rows),
        "output_csv": str(output_csv),
        "output_markdown": str(output_markdown),
        "output_figure_pdf": str(output_figure_pdf) if output_figure_pdf else None,
        "output_figure_png": str(output_figure_png) if output_figure_png else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--event-csv", type=Path, default=root / "results/paper/event_level_results.csv"
    )
    parser.add_argument(
        "--output-csv", type=Path, default=root / "results/generated/paired_inference.csv"
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=root / "results/generated/paired_inference.md",
    )
    parser.add_argument(
        "--output-figure-pdf",
        type=Path,
        default=root / "results/generated/controlled_baseline_effects.pdf",
    )
    parser.add_argument(
        "--output-figure-png",
        type=Path,
        default=root / "results/generated/controlled_baseline_effects.png",
    )
    args = parser.parse_args(argv)
    result = reproduce_inference(
        event_csv=args.event_csv,
        output_csv=args.output_csv,
        output_markdown=args.output_markdown,
        output_figure_pdf=args.output_figure_pdf,
        output_figure_png=args.output_figure_png,
    )
    print(result)
    return 0


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "compute_primary_inference",
    "exact_sign_flip_pvalue",
    "holm_adjust",
    "paired_relative_bootstrap",
    "reduction_rows",
    "reproduce_inference",
    "main",
]
