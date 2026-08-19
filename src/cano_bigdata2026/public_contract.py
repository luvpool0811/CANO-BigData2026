"""Fail-closed checks for the public experiment and disclosure contract."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from .operational_evidence import validate_operational_inputs
from .training import CHECKPOINT_SELECTION_METRIC
from .workflow import RoleContract, validate_role_contracts


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(value is None for row in rows for value in row.values()):
        raise ValueError(f"{path.name} is empty or incomplete")
    return rows


def _role_configuration(root: Path) -> tuple[dict[str, object], dict[str, RoleContract]]:
    path = root / "configs/evaluation/berlin_i_roles.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Berlin I role configuration must be a mapping")
    return config, validate_role_contracts(config)


def _validate_public_role_membership(
    root: Path,
    config: Mapping[str, object],
    contracts: Mapping[str, RoleContract],
) -> None:
    filename = config.get("public_membership_file")
    if filename != "berlin_i_role_membership.csv":
        raise ValueError("Berlin I public membership file changed")
    rows = _csv_rows(root / "configs/evaluation" / str(filename))
    observed: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        observed.setdefault(row["role"], []).append(row)
    if set(observed) != set(contracts):
        raise ValueError("public role membership does not contain exactly four roles")
    for role, contract in contracts.items():
        selected = observed[role]
        aliases = tuple(row["public_event_alias"] for row in selected)
        ordinals = [int(row["ordinal"]) for row in selected]
        directories = {row["directory"] for row in selected}
        if (
            aliases != contract.public_event_aliases
            or ordinals != list(range(1, contract.count + 1))
            or directories != {contract.directory}
        ):
            raise ValueError(f"public role membership changed for {role}")


def _validate_training_configs(root: Path) -> None:
    paths = (
        root / "configs/cano/standard.yaml",
        root / "configs/baselines/dno3.yaml",
        root / "configs/baselines/fno3d.yaml",
        root / "configs/baselines/unet3d.yaml",
    )
    for path in paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        metric = config.get("training", {}).get("checkpoint_selection_metric")
        if metric != CHECKPOINT_SELECTION_METRIC:
            raise ValueError(f"{path.name} changed the checkpoint selection metric")


def _validate_wis_scope(root: Path) -> None:
    rows = _csv_rows(root / "results/paper/claim_evidence.csv")
    row = next(item for item in rows if item["claim_id"] == "CANO-baseline-WIS")
    required = "each model's own prediction-selected population"
    if required not in row["evidence_boundary"]:
        raise ValueError("WIS evidence must disclose model-specific selected populations")


def validate_static_contract(root: Path) -> dict[str, object]:
    """Validate checked-in experiment, HPO, split, and claim disclosures."""

    root = root.resolve()
    role_config, contracts = _role_configuration(root)
    _validate_public_role_membership(root, role_config, contracts)
    _validate_training_configs(root)
    _validate_wis_scope(root)
    payload = validate_operational_inputs(root / "results/paper")
    return {
        "status": "PASS",
        "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
        "public_role_rows": sum(contract.count for contract in contracts.values()),
        "hpo_candidate_rows": len(payload["hpo_candidates.csv"]),
        "reproducibility_scope_rows": len(payload["reproducibility_scope.csv"]),
        "raw_field_arrays_opened": 0,
    }


def validate_prepared_split_ids(root: Path, data_root: Path) -> dict[str, object]:
    """Check prepared split identity disjointness without reading field arrays."""

    _, contracts = _role_configuration(root.resolve())
    owner: dict[str, str] = {}
    counts: dict[str, int] = {}
    for role, contract in contracts.items():
        paths = sorted((data_root / contract.directory).glob("*.npz"))
        if len(paths) != contract.count:
            raise ValueError(
                f"{role} contains {len(paths)} files; expected {contract.count}"
            )
        for path in paths:
            with np.load(path, allow_pickle=False) as payload:
                event_id = (
                    str(payload["event_id"].item())
                    if "event_id" in payload
                    else path.stem
                )
            previous = owner.get(event_id)
            if previous is not None:
                if previous == role:
                    raise ValueError(f"duplicate event identity in prepared role {role}")
                raise ValueError(
                    f"event identity overlaps prepared roles {previous} and {role}"
                )
            owner[event_id] = role
        counts[role] = len(paths)
    if len(owner) != sum(counts.values()):
        raise ValueError("prepared event identities are not globally unique")
    return {
        "status": "PASS",
        "role_counts": counts,
        "unique_event_ids": len(owner),
        "field_arrays_opened": 0,
        "event_id_metadata_only": True,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args(argv)
    result = validate_static_contract(args.root)
    if args.data_root is not None:
        result["prepared_split_check"] = validate_prepared_split_ids(
            args.root, args.data_root
        )
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = ["main", "validate_prepared_split_ids", "validate_static_contract"]
