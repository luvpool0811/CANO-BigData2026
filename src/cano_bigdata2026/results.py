"""Rebuild the public tables and figures from the reported CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_LABELS = {
    "h_rmse_m": "H RMSE (m)",
    "nse": "NSE",
    "wet_nse": "Wet-domain NSE",
    "peak_depth_abs_error_m": "Peak-depth error (m)",
    "target_ace": "Target ACE",
    "target_wis": "Target WIS",
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _display_label(row: dict[str, str]) -> str:
    label = row["system"]
    if row.get("comparison") == "standard":
        return label.replace(" (standard objective)", "")
    return {
        "CANO (selection-matched control)": "Selection-matched control",
        "CANO (peak-aware objective)": "Peak-aware objective",
    }.get(label, label)


def _bar_panels(
    rows: list[dict[str, str]],
    metrics: Sequence[str],
    output: Path,
    *,
    title: str,
) -> None:
    figure, axes = plt.subplots(1, len(metrics), figsize=(4.1 * len(metrics), 3.8))
    if len(metrics) == 1:
        axes = [axes]
    labels = [_display_label(row) for row in rows]
    colors = [
        "#2F5597" if row["system"].startswith("CANO") else "#9AA0A6"
        for row in rows
    ]
    for axis, metric in zip(axes, metrics):
        values = [_float(row, metric) for row in rows]
        bars = axis.bar(range(len(rows)), values, color=colors)
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xticks(range(len(rows)), labels, rotation=28, ha="right")
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _event_plot(rows: list[dict[str, str]], output: Path) -> None:
    systems = list(dict.fromkeys(row["system"] for row in rows))
    figure, axis = plt.subplots(figsize=(8.2, 4.1))
    for system in systems:
        selected = [row for row in rows if row["system"] == system]
        axis.plot(
            range(1, len(selected) + 1),
            [_float(row, "h_rmse_m") for row in selected],
            marker="o",
            linewidth=1.5,
            markersize=3.5,
            label=system,
        )
    axis.set_xlabel("Evaluation event")
    axis.set_ylabel("H RMSE (m)")
    axis.set_xticks(range(1, 13))
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _markdown(rows: list[dict[str, str]]) -> str:
    header = (
        "| System | Parameters (M) | H RMSE | NSE | Wet RMSE | Wet NSE | "
        "Peak error | CSI .01 | CSI .10 | CSI .30 | CSI .50 | Target ACE | Target WIS |"
    )
    rule = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [
        "# Reported results",
        "",
        "Event-macro averages over the 12 BerlinI evaluation events.",
        "",
        header,
        rule,
    ]
    for row in rows:
        lines.append(
            "| {system} | {parameters_m:.3f} | {h_rmse_m:.4f} | {nse:.4f} | "
            "{wet_rmse_m:.4f} | {wet_nse:.4f} | {peak_depth_abs_error_m:.4f} | "
            "{csi_001:.4f} | {csi_010:.4f} | {csi_030:.4f} | {csi_050:.4f} | "
            "{target_ace:.4f} | {target_wis:.4f} |".format(
                system=row["system"],
                **{
                    key: float(value)
                    for key, value in row.items()
                    if key not in {"system", "comparison"}
                },
            )
        )
    lines.extend(
        [
            "",
            "NSE and wet-domain NSE are core point-prediction metrics. Target ACE "
            "and Target WIS summarize target-aligned uncertainty calibration.",
            "",
        ]
    )
    return "\n".join(lines)


def reproduce(*, results_dir: Path, output_dir: Path) -> dict[str, object]:
    main_rows = _rows(results_dir / "main_results.csv")
    event_rows = _rows(results_dir / "event_level_results.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "main_results.md").write_text(
        _markdown(main_rows), encoding="utf-8"
    )
    standard = [row for row in main_rows if row["comparison"] == "standard"]
    ablation = [row for row in main_rows if row["comparison"] == "cano_ablation"]
    _bar_panels(
        standard,
        ("h_rmse_m", "nse", "wet_nse"),
        output_dir / "point_prediction_metrics.png",
        title="Standard-objective point-prediction performance",
    )
    _bar_panels(
        standard,
        ("target_ace", "target_wis"),
        output_dir / "uncertainty_metrics.png",
        title="Target-aligned uncertainty metrics",
    )
    _bar_panels(
        ablation,
        ("peak_depth_abs_error_m", "nse", "wet_nse"),
        output_dir / "cano_objective_ablation.png",
        title="CANO training-objective ablation",
    )
    _event_plot(event_rows, output_dir / "event_level_h_rmse.png")
    return {
        "status": "PASS",
        "rows": len(main_rows),
        "event_rows": len(event_rows),
        "output_dir": str(output_dir),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--results-dir", type=Path, default=project_root / "results/paper"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=project_root / "results/generated"
    )
    args = parser.parse_args(argv)
    payload = reproduce(results_dir=args.results_dir, output_dir=args.output_dir)
    print(payload)
    return 0


__all__ = ["reproduce", "main"]
