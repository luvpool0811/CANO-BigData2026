"""Fail-closed checks for the public experiment and disclosure contract."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping
import zipfile

import numpy as np
import yaml

from .operational_evidence import validate_operational_inputs
from .results import validate_event_level_means
from .training import CHECKPOINT_SELECTION_METRIC
from .workflow import RoleContract, validate_role_contracts


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(value is None for row in rows for value in row.values()):
        raise ValueError(f"{path.name} is empty or incomplete")
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_digest(records: list[dict[str, str]]) -> str:
    payload = json.dumps(
        records, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
) -> list[dict[str, str]]:
    filename = config.get("public_membership_file")
    if filename != "berlin_i_role_membership.csv":
        raise ValueError("Berlin I public membership file changed")
    rows = _csv_rows(root / "configs/evaluation" / str(filename))
    provider_contract = {
        "provider_release_doi": "https://doi.org/10.5281/zenodo.15700880",
        "provider_archive_root": "UrbanFloodCast_Dataset",
        "provider_domain": "BerlinI",
        "provider_primary_view": "Seen regions and unseen rainfall events",
        "provider_identity_fields": [
            "provider_split",
            "provider_event_name",
            "provider_relative_path",
        ],
    }
    if any(config.get(key) != value for key, value in provider_contract.items()):
        raise ValueError("Berlin I provider release identity contract changed")
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
        expected_split = {
            "training": "Train",
            "development": "Train",
            "calibration": "Valid",
            "evaluation": "Test",
        }[role]
        for row in selected:
            expected_path = (
                f"{provider_contract['provider_archive_root']}/"
                f"{provider_contract['provider_domain']}/"
                f"{provider_contract['provider_primary_view']}/"
                f"{expected_split}/{row['provider_event_name']}/"
            )
            if (
                row["provider_split"] != expected_split
                or row["provider_relative_path"] != expected_path
            ):
                raise ValueError(f"provider identity changed for {role}")
    provider_paths = [row["provider_relative_path"] for row in rows]
    if len(provider_paths) != len(set(provider_paths)):
        raise ValueError("provider event directories must be globally unique")
    return rows


def _provider_membership(root: Path) -> list[dict[str, str]]:
    config, contracts = _role_configuration(root.resolve())
    return _validate_public_role_membership(root.resolve(), config, contracts)


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
    provider_rows = _validate_public_role_membership(root, role_config, contracts)
    _validate_training_configs(root)
    _validate_wis_scope(root)
    payload = validate_operational_inputs(root / "results/paper")
    recomputation = validate_event_level_means(
        _csv_rows(root / "results/paper/main_results.csv"),
        _csv_rows(root / "results/paper/event_level_results.csv"),
    )
    return {
        "status": "PASS",
        "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
        "public_role_rows": sum(contract.count for contract in contracts.values()),
        "provider_identity_rows": len(provider_rows),
        "operational_specification_rows": len(
            payload["operational_evaluation_specification.csv"]
        ),
        "hpo_candidate_rows": len(payload["hpo_candidates.csv"]),
        "reproducibility_scope_rows": len(payload["reproducibility_scope.csv"]),
        "table_ii_event_rows_recomputed": recomputation["event_rows_recomputed"],
        "table_ii_metrics_recomputed": len(recomputation["metrics_recomputed"]),
        "raw_field_arrays_opened": 0,
    }


def validate_prepared_split_ids(root: Path, data_root: Path) -> dict[str, object]:
    """Check exact provider split membership without reading field arrays."""

    _, contracts = _role_configuration(root.resolve())
    membership = _provider_membership(root.resolve())
    expected_by_directory: dict[str, dict[str, str]] = {}
    for row in membership:
        expected_by_directory.setdefault(row["directory"], {})[
            row["provider_relative_path"]
        ] = row["provider_event_name"]
    owner: dict[str, str] = {}
    observed_provider_paths: set[str] = set()
    counts: dict[str, int] = {}
    identity_records: list[dict[str, str]] = []
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
                if "provider_relative_path" not in payload or "provider_event_name" not in payload:
                    raise ValueError(
                        f"{path.name} lacks provider identity metadata; recreate it "
                        "with prepare_event_npz.py"
                    )
                provider_path = str(payload["provider_relative_path"].item())
                provider_name = str(payload["provider_event_name"].item())
            expected_name = expected_by_directory[contract.directory].get(provider_path)
            if expected_name is None or provider_name != expected_name:
                raise ValueError(
                    f"prepared provider identity is not assigned to role {role}"
                )
            if provider_path in observed_provider_paths:
                raise ValueError("provider identity is duplicated across prepared roles")
            observed_provider_paths.add(provider_path)
            previous = owner.get(event_id)
            if previous is not None:
                if previous == role:
                    raise ValueError(f"duplicate event identity in prepared role {role}")
                raise ValueError(
                    f"event identity overlaps prepared roles {previous} and {role}"
                )
            owner[event_id] = role
            identity_records.append(
                {
                    "role": role,
                    "filename": path.name,
                    "event_id": event_id,
                    "provider_event_name": provider_name,
                    "provider_relative_path": provider_path,
                }
            )
        counts[role] = len(paths)
    if len(owner) != sum(counts.values()):
        raise ValueError("prepared event identities are not globally unique")
    expected_provider_paths = {
        row["provider_relative_path"] for row in membership
    }
    if observed_provider_paths != expected_provider_paths:
        raise ValueError("prepared provider identities do not match the public ledger")
    return {
        "status": "PASS",
        "role_counts": counts,
        "unique_event_ids": len(owner),
        "provider_identities_matched": len(observed_provider_paths),
        "prepared_identity_sha256": _identity_digest(identity_records),
        "field_arrays_opened": 0,
        "identity_metadata_only": True,
    }


def create_provider_preflight_receipt(
    root: Path, data_root: Path, output_path: Path
) -> dict[str, object]:
    """Create the provider-bound receipt required by the evidence pipeline."""

    root = root.resolve()
    data_root = data_root.resolve()
    result = validate_prepared_split_ids(root, data_root)
    role_config_path = root / "configs/evaluation/berlin_i_roles.yaml"
    membership_path = root / "configs/evaluation/berlin_i_role_membership.csv"
    receipt: dict[str, object] = {
        "schema_id": "cano_provider_preflight_receipt_v1",
        "status": "PASS",
        "data_root_resolved": str(data_root),
        "role_config_sha256": _sha256_file(role_config_path),
        "provider_membership_sha256": _sha256_file(membership_path),
        **result,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def validate_provider_archive_membership(root: Path, archive: Path) -> dict[str, object]:
    """Match the public ledger to provider ZIP directory identities only."""

    expected = {row["provider_relative_path"] for row in _provider_membership(root)}
    prefix = "UrbanFloodCast_Dataset/BerlinI/Seen regions and unseen rainfall events/"
    observed: set[str] = set()
    with zipfile.ZipFile(archive, "r") as bundle:
        members = bundle.namelist()
    for member in members:
        if not member.startswith(prefix):
            continue
        relative = member[len(prefix):].split("/")
        if len(relative) < 3 or relative[0] not in {"Train", "Valid", "Test"}:
            continue
        observed.add(f"{prefix}{relative[0]}/{relative[1]}/")
    if observed != expected:
        missing = len(expected - observed)
        extra = len(observed - expected)
        raise ValueError(
            f"provider archive membership differs from public ledger: "
            f"missing={missing}, extra={extra}"
        )
    return {
        "status": "PASS",
        "provider_event_directories": len(observed),
        "zip_directory_entries_scanned": len(members),
        "payload_members_opened": 0,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--provider-archive", type=Path)
    parser.add_argument("--write-receipt", type=Path)
    args = parser.parse_args(argv)
    result = validate_static_contract(args.root)
    if args.data_root is not None:
        result["prepared_split_check"] = validate_prepared_split_ids(
            args.root, args.data_root
        )
    if args.write_receipt is not None:
        if args.data_root is None:
            parser.error("--write-receipt requires --data-root")
        result["provider_preflight_receipt"] = create_provider_preflight_receipt(
            args.root, args.data_root, args.write_receipt
        )
    if args.provider_archive is not None:
        result["provider_archive_check"] = validate_provider_archive_membership(
            args.root.resolve(), args.provider_archive
        )
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = [
    "main",
    "create_provider_preflight_receipt",
    "validate_prepared_split_ids",
    "validate_provider_archive_membership",
    "validate_static_contract",
]
