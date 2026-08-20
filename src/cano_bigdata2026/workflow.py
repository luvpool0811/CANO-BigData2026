"""Three-seed ensemble, calibration, and event-level evaluation workflow."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from itertools import chain
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import yaml

from . import contracts as C
from .calibration import (
    fit_node_scale_events,
    fit_target_aligned_calibrator_events,
)
from .data import EventNPZDataset, Normalization, event_files
from .evidence import evaluate_event
from .metrics import aggregate
from .models import CANO, build_model


@dataclass(frozen=True)
class PhysicalEvent:
    event_id: str
    prediction: np.ndarray
    truth: np.ndarray
    mask: np.ndarray

    @property
    def prediction_h(self) -> np.ndarray:
        return self.prediction.reshape(24, 3, *self.mask.shape)[:, 0]

    @property
    def truth_h(self) -> np.ndarray:
        return self.truth.reshape(24, 3, *self.mask.shape)[:, 0]


@dataclass(frozen=True)
class RoleContract:
    directory: str
    count: int
    public_event_aliases: tuple[str, ...]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping")
    return payload


def _load_models(
    checkpoints: Sequence[Path],
    *,
    device: torch.device,
    upstream_source: Path | None,
) -> tuple[list[torch.nn.Module], list[int], str, dict[str, Any]]:
    if len(checkpoints) != 3:
        raise ValueError("the reported ensemble requires exactly three checkpoints")
    models: list[torch.nn.Module] = []
    seeds: list[int] = []
    identity: tuple[str, str] | None = None
    for path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model_name = str(payload["model_name"])
        model_config = dict(payload["model_config"])
        current = (model_name, json.dumps(model_config, sort_keys=True))
        if identity is None:
            identity = current
        elif current != identity:
            raise ValueError("ensemble checkpoints do not share one model configuration")
        seed = int(payload["seed"])
        if seed in seeds:
            raise ValueError("ensemble checkpoint seeds must be unique")
        model = build_model(
            model_name, model_config, upstream_source=upstream_source
        )
        model.load_state_dict(payload["state_dict"], strict=True)
        models.append(model.to(device).eval())
        seeds.append(seed)
    if identity is None:
        raise ValueError("no checkpoint identity was established")
    return models, seeds, identity[0], json.loads(identity[1])


def _predict_role(
    models: Sequence[torch.nn.Module],
    *,
    data_root: Path,
    split: str,
    normalization: Normalization,
    device: torch.device,
    query_chunk_size: int,
) -> Iterator[PhysicalEvent]:
    dataset = EventNPZDataset(event_files(data_root, split))
    with torch.no_grad():
        for sample in dataset:
            inputs = sample["input"].unsqueeze(0).to(device)
            predictions: list[torch.Tensor] = []
            for model in models:
                if isinstance(model, CANO):
                    output = model(inputs, query_chunk_size=query_chunk_size)[0]
                else:
                    output = model(inputs)[0]
                predictions.append(normalization.denormalize_output(output.cpu()))
            prediction = torch.mean(torch.stack(predictions), dim=0).numpy()
            truth = normalization.denormalize_output(sample["target"]).numpy()
            yield PhysicalEvent(
                event_id=str(sample["event_id"]),
                prediction=np.asarray(prediction, dtype=np.float32),
                truth=np.asarray(truth, dtype=np.float32),
                mask=np.asarray(sample["mask"].numpy(), dtype=bool),
            )


def _public_aliases(row: Mapping[str, Any], role: str, count: int) -> tuple[str, ...]:
    explicit = row.get("public_event_aliases")
    if explicit is not None:
        if not isinstance(explicit, Sequence) or isinstance(explicit, (str, bytes)):
            raise ValueError(f"public event aliases for {role} must be a sequence")
        aliases = tuple(str(value).strip() for value in explicit)
    else:
        prefix = str(row.get("public_event_alias_prefix", "")).strip()
        if not prefix:
            raise ValueError(f"role contract for {role} lacks public event aliases")
        digits = max(2, len(str(count)))
        aliases = tuple(f"{prefix} {index:0{digits}d}" for index in range(1, count + 1))
    if len(aliases) != count or any(not value for value in aliases):
        raise ValueError(f"public event aliases for {role} do not match its count")
    if len(set(aliases)) != count:
        raise ValueError(f"public event aliases for {role} are not unique")
    return aliases


def _role_contract(config: Mapping[str, Any], role: str) -> RoleContract:
    roles = config.get("roles")
    row = roles.get(role) if isinstance(roles, Mapping) else None
    if not isinstance(row, Mapping):
        raise ValueError(f"role contract is missing {role}")
    directory = str(row.get("directory", ""))
    count = int(row.get("count", -1))
    if not directory or count < 1:
        raise ValueError(f"role contract for {role} is invalid")
    return RoleContract(
        directory=directory,
        count=count,
        public_event_aliases=_public_aliases(row, role, count),
    )


def validate_role_contracts(config: Mapping[str, Any]) -> dict[str, RoleContract]:
    """Validate public role identities before any dataset file is opened."""

    names = ("training", "development", "calibration", "evaluation")
    contracts = {name: _role_contract(config, name) for name in names}
    directories = [contract.directory for contract in contracts.values()]
    if len(set(directories)) != len(directories):
        raise ValueError("role directories must be pairwise distinct")
    ownership: dict[str, str] = {}
    for role, contract in contracts.items():
        for alias in contract.public_event_aliases:
            previous = ownership.setdefault(alias, role)
            if previous != role:
                raise ValueError(
                    f"public event alias {alias!r} overlaps roles {previous} and {role}"
                )
    return contracts


def _event_id_digest(event_id: str, *, namespace: str) -> str:
    return hashlib.sha256(f"{namespace}|{event_id}".encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepared_identity_fingerprint(
    data_root: Path, role_contracts: Mapping[str, RoleContract]
) -> dict[str, Any]:
    records: list[dict[str, str]] = []
    event_owners: dict[str, str] = {}
    provider_paths: set[str] = set()
    counts: dict[str, int] = {}
    for role, contract in role_contracts.items():
        paths = sorted((data_root / contract.directory).glob("*.npz"))
        if len(paths) != contract.count:
            raise ValueError(
                f"provider preflight binding found {len(paths)} {role} files; "
                f"expected {contract.count}"
            )
        for path in paths:
            with np.load(path, allow_pickle=False) as payload:
                required = (
                    "event_id",
                    "provider_event_name",
                    "provider_relative_path",
                )
                if any(key not in payload for key in required):
                    raise ValueError(
                        f"{path.name} lacks provider identity metadata required "
                        "by the preflight receipt"
                    )
                event_id = str(payload["event_id"].item())
                provider_name = str(payload["provider_event_name"].item())
                provider_path = str(payload["provider_relative_path"].item())
            previous = event_owners.setdefault(event_id, role)
            if previous != role:
                raise ValueError(
                    f"event identity overlaps provider roles {previous} and {role}"
                )
            if provider_path in provider_paths:
                raise ValueError("provider identity is duplicated across roles")
            provider_paths.add(provider_path)
            records.append(
                {
                    "role": role,
                    "filename": path.name,
                    "event_id": event_id,
                    "provider_event_name": provider_name,
                    "provider_relative_path": provider_path,
                }
            )
        counts[role] = len(paths)
    canonical = json.dumps(
        records, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "role_counts": counts,
        "unique_event_ids": len(event_owners),
        "provider_identities_matched": len(provider_paths),
        "prepared_identity_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _verify_provider_preflight_receipt(
    *,
    receipt_path: Path,
    data_root: Path,
    role_config_path: Path,
    role_config: Mapping[str, Any],
    role_contracts: Mapping[str, RoleContract],
) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    membership_name = role_config.get("public_membership_file")
    membership_path = role_config_path.parent / str(membership_name)
    expected_counts = {
        role: contract.count for role, contract in role_contracts.items()
    }
    try:
        current = prepared_identity_fingerprint(data_root.resolve(), role_contracts)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(
            "provider preflight receipt does not bind the current data root, "
            "role contract, and provider identities"
        ) from error
    if (
        receipt.get("schema_id") != "cano_provider_preflight_receipt_v1"
        or receipt.get("status") != "PASS"
        or Path(str(receipt.get("data_root_resolved", ""))).resolve()
        != data_root.resolve()
        or receipt.get("role_config_sha256") != _sha256_file(role_config_path)
        or not membership_path.is_file()
        or receipt.get("provider_membership_sha256")
        != _sha256_file(membership_path)
        or receipt.get("role_counts") != expected_counts
        or receipt.get("field_arrays_opened") != 0
        or receipt.get("identity_metadata_only") is not True
        or any(receipt.get(key) != value for key, value in current.items())
    ):
        raise ValueError(
            "provider preflight receipt does not bind the current data root, "
            "role contract, and provider identities"
        )
    return {
        **current,
        "provider_preflight_receipt_sha256": _sha256_file(receipt_path),
        "role_config_sha256": _sha256_file(role_config_path),
        "provider_membership_sha256": _sha256_file(membership_path),
    }


def run_evidence_pipeline(
    *,
    checkpoints: Sequence[Path],
    data_root: Path,
    normalization_path: Path,
    role_config_path: Path,
    calibration_config_path: Path,
    provider_preflight_receipt_path: Path,
    output_path: Path,
    device: str,
    upstream_source: Path | None = None,
    query_chunk_size: int = 32768,
) -> dict[str, Any]:
    role_config = _load_yaml(role_config_path)
    role_contracts = validate_role_contracts(role_config)
    provider_binding = _verify_provider_preflight_receipt(
        receipt_path=provider_preflight_receipt_path,
        data_root=data_root,
        role_config_path=role_config_path,
        role_config=role_config,
        role_contracts=role_contracts,
    )
    target_device = torch.device(device)
    models, seeds, model_name, model_config = _load_models(
        checkpoints, device=target_device, upstream_source=upstream_source
    )
    calibration_config = _load_yaml(calibration_config_path)
    expected_seeds = role_config.get("ensemble_seeds")
    if expected_seeds is not None and sorted(
        int(value) for value in expected_seeds
    ) != sorted(seeds):
        raise ValueError("checkpoint seeds differ from the role contract")
    normalization = Normalization.load(normalization_path)
    development = role_contracts["development"]
    calibration = role_contracts["calibration"]
    evaluation = role_contracts["evaluation"]
    node_config = calibration_config.get("node_scale", {})
    residual_config = calibration_config.get("residual_calibration", {})
    evaluation_config = calibration_config.get("evaluation", {})
    if (
        node_config.get("fit_split") != "development"
        or residual_config.get("fit_split") != "calibration"
        or evaluation_config.get("event_macro_aggregation") is not True
        or evaluation_config.get("target_used_for_population_selection") is not False
    ):
        raise ValueError("calibration role and evaluation contracts are invalid")
    floor_m = float(node_config.get("floor_m", 0.001))
    operational_threshold = float(
        residual_config.get(
            "operational_threshold_m", C.OPERATIONAL_TARGET_THRESHOLD_M
        )
    )
    alpha_levels = tuple(float(value) for value in residual_config["alpha_levels"])

    first_mask: np.ndarray | None = None
    observed_owner: dict[str, str] = {}
    role_id_digests: dict[str, list[str]] = {
        role: [] for role in ("development", "calibration", "evaluation")
    }
    digest_namespace = str(
        role_config.get(
            "event_id_digest_namespace",
            "CANO-BigData2026/role-identity/v1",
        )
    )

    def role_events(role: str, contract: RoleContract) -> Iterator[PhysicalEvent]:
        nonlocal first_mask
        count = 0
        for event in _predict_role(
            models,
            data_root=data_root,
            split=contract.directory,
            normalization=normalization,
            device=target_device,
            query_chunk_size=query_chunk_size,
        ):
            if first_mask is None:
                first_mask = event.mask.copy()
            elif not np.array_equal(first_mask, event.mask):
                raise ValueError("all role events must use the same valid-cell mask")
            previous = observed_owner.get(event.event_id)
            if previous is not None:
                raise ValueError(
                    f"event identity is duplicated in role {role}"
                    if previous == role
                    else f"event identity overlaps roles {previous} and {role}"
                )
            observed_owner[event.event_id] = role
            if count >= contract.count:
                raise ValueError(
                    f"role {role} contains more than {contract.count} events"
                )
            role_id_digests[role].append(
                _event_id_digest(event.event_id, namespace=digest_namespace)
            )
            alias = contract.public_event_aliases[count]
            count += 1
            yield PhysicalEvent(
                event_id=alias,
                prediction=event.prediction,
                truth=event.truth,
                mask=event.mask,
            )
        if count != contract.count:
            raise ValueError(
                f"role {role} has {count} events; expected {contract.count}"
            )

    development_pairs = (
        (event.prediction_h, event.truth_h)
        for event in role_events("development", development)
    )
    try:
        first_development = next(development_pairs)
    except StopIteration as error:
        raise ValueError("development role is empty") from error
    if first_mask is None:
        raise ValueError("development role did not establish a valid mask")
    node_scale = fit_node_scale_events(
        chain((first_development,), development_pairs), first_mask, floor_m=floor_m
    )

    calibration_events = role_events("calibration", calibration)
    calibrator = fit_target_aligned_calibrator_events(
        ((event.prediction_h, event.truth_h) for event in calibration_events),
        first_mask,
        node_scale,
        threshold_m=operational_threshold,
        alpha_levels=alpha_levels,
    )

    event_records: list[dict[str, Any]] = []
    for event in role_events("evaluation", evaluation):
        event_records.append(
            evaluate_event(
                event_id=event.event_id,
                prediction_huv=event.prediction,
                truth_huv=event.truth,
                valid_mask=event.mask,
                node_scale_h=node_scale,
                calibrator=calibrator,
            )
        )
    point = aggregate([row["point_prediction"] for row in event_records])
    target_rows = [row["target_aligned_calibration"] for row in event_records]
    if any(bool(row["empty"]) for row in target_rows):
        raise ValueError("an evaluation event has an empty operational population")
    target_macro = {
        "target_ace": float(np.mean([float(row["target_ace"]) for row in target_rows])),
        "target_wis": float(np.mean([float(row["target_wis"]) for row in target_rows])),
    }
    payload: dict[str, Any] = {
        "status": "PASS",
        "estimator": "three-seed physical-field ensemble",
        "model_name": model_name,
        "model_config": model_config,
        "checkpoint_count": len(models),
        "checkpoint_seeds": sorted(seeds),
        "truth_wet_threshold_m": C.TRUTH_WET_THRESHOLD_M,
        "operational_target_threshold_m": operational_threshold,
        "role_counts": {
            role: contract.count for role, contract in role_contracts.items()
        },
        "role_identity": {
            "digest_algorithm": "sha256",
            "digest_namespace": digest_namespace,
            "pairwise_disjoint_observed_roles": True,
            "event_id_digests": role_id_digests,
            "public_event_aliases": {
                role: list(contract.public_event_aliases)
                for role, contract in role_contracts.items()
            },
        },
        "input_bindings": {
            **provider_binding,
            "normalization_sha256": _sha256_file(normalization_path),
            "calibration_config_sha256": _sha256_file(calibration_config_path),
            "checkpoint_sha256": [_sha256_file(path) for path in checkpoints],
        },
        "evaluation_target_reads": evaluation.count,
        "point_prediction_event_macro": point,
        "target_aligned_event_macro": target_macro,
        "events": event_records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--checkpoints", type=Path, nargs=3, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument(
        "--provider-preflight-receipt", type=Path, required=True
    )
    parser.add_argument(
        "--role-config",
        type=Path,
        default=root / "configs/evaluation/berlin_i_roles.yaml",
    )
    parser.add_argument(
        "--calibration-config",
        type=Path,
        default=root / "configs/calibration/target_aligned.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--upstream-source", type=Path)
    parser.add_argument("--query-chunk-size", type=int, default=32768)
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    payload = run_evidence_pipeline(
        checkpoints=args.checkpoints,
        data_root=args.data_root,
        normalization_path=args.normalization,
        role_config_path=args.role_config,
        calibration_config_path=args.calibration_config,
        provider_preflight_receipt_path=args.provider_preflight_receipt,
        output_path=args.output,
        device=args.device,
        upstream_source=args.upstream_source,
        query_chunk_size=args.query_chunk_size,
    )
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))
    return 0


__all__ = [
    "PhysicalEvent",
    "RoleContract",
    "prepared_identity_fingerprint",
    "run_evidence_pipeline",
    "validate_role_contracts",
    "main",
]
