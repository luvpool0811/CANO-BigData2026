from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from cano_bigdata2026.models import build_model
from cano_bigdata2026.workflow import (
    evaluation_data_identity_fingerprint,
    run_evidence_pipeline,
    validate_role_contracts,
)


MODEL_CONFIG = {
    "latent_dim": 8,
    "branch_depth": 1,
    "forcing_hidden_dim": 8,
    "decoder_hidden_dim": 16,
    "n_freqs_xy": 2,
    "n_freqs_z": 1,
    "n_freqs_t": 1,
    "dropout": 0.0,
}


def _event(path: Path, seed: int, *, role: str) -> None:
    generator = np.random.default_rng(seed)
    inputs = generator.normal(size=(31, 5, 7)).astype(np.float32)
    target = generator.uniform(0.1, 0.8, size=(72, 5, 7)).astype(np.float32)
    mask = np.ones((5, 7), dtype=bool)
    np.savez_compressed(
        path,
        input=inputs,
        target=target,
        mask=mask,
        event_id=np.asarray(path.stem),
        provider_event_name=np.asarray(f"Provider {role}"),
        provider_relative_path=np.asarray(f"Provider/{role}/{path.stem}/"),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_three_seed_evidence_pipeline_smoke(tmp_path: Path) -> None:
    data = tmp_path / "data"
    for index, split in enumerate(("train", "validation", "calibration", "test")):
        (data / split).mkdir(parents=True)
        _event(data / split / f"{split}-01.npz", index + 10, role=split)
    normalization = data / "normalization.json"
    normalization.write_text(
        json.dumps(
            {
                "mean": {"H": 0.0, "U": 0.0, "V": 0.0},
                "std": {"H": 1.0, "U": 1.0, "V": 1.0},
            }
        ),
        encoding="utf-8",
    )
    checkpoints: list[Path] = []
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        model = build_model("cano", MODEL_CONFIG)
        path = tmp_path / f"seed-{seed}.pt"
        torch.save(
            {
                "model_name": "cano",
                "model_config": MODEL_CONFIG,
                "state_dict": model.state_dict(),
                "seed": seed,
            },
            path,
        )
        checkpoints.append(path)
    roles = tmp_path / "roles.yaml"
    roles.write_text(
        "ensemble_seeds: [1, 2, 3]\n"
        "event_id_digest_namespace: synthetic-test/v1\n"
        "public_membership_file: membership.csv\n"
        "roles:\n"
        "  training: {directory: train, count: 1, public_event_alias_prefix: Train}\n"
        "  development: {directory: validation, count: 1, public_event_alias_prefix: Development}\n"
        "  calibration: {directory: calibration, count: 1, public_event_alias_prefix: Calibration}\n"
        "  evaluation: {directory: test, count: 1, public_event_alias_prefix: Evaluation}\n",
        encoding="utf-8",
    )
    membership = tmp_path / "membership.csv"
    membership.write_text("synthetic membership\n", encoding="utf-8")
    calibration = tmp_path / "calibration.yaml"
    calibration.write_text(
        "node_scale: {fit_split: development, floor_m: 0.001}\n"
        "residual_calibration:\n"
        "  fit_split: calibration\n"
        "  operational_threshold_m: -100.0\n"
        "  alpha_levels: [0.5, 0.2, 0.1]\n"
        "evaluation:\n"
        "  event_macro_aggregation: true\n"
        "  target_used_for_population_selection: false\n",
        encoding="utf-8",
    )
    output = tmp_path / "evidence.json"
    role_contracts = validate_role_contracts(yaml.safe_load(roles.read_text()))
    identity = evaluation_data_identity_fingerprint(data, role_contracts)
    record = tmp_path / "data-integrity-verification.json"
    record.write_text(
        json.dumps(
            {
                "schema_id": "cano_evaluation_data_integrity_record_v1",
                "status": "PASS",
                "data_root_resolved": str(data.resolve()),
                "role_config_sha256": _sha256(roles),
                "provider_membership_sha256": _sha256(membership),
                "field_arrays_opened": 0,
                "identity_metadata_only": True,
                **identity,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    payload = run_evidence_pipeline(
        checkpoints=checkpoints,
        data_root=data,
        normalization_path=normalization,
        role_config_path=roles,
        calibration_config_path=calibration,
        data_integrity_record_path=record,
        output_path=output,
        device="cpu",
        query_chunk_size=16,
    )
    assert payload["status"] == "PASS"
    assert payload["model_name"] == "cano"
    assert payload["checkpoint_count"] == 3
    assert payload["checkpoint_seeds"] == [1, 2, 3]
    assert payload["truth_wet_threshold_m"] == 0.01
    assert payload["evaluation_target_reads"] == 1
    assert payload["role_counts"] == {
        "training": 1,
        "development": 1,
        "calibration": 1,
        "evaluation": 1,
    }
    assert payload["role_identity"]["pairwise_disjoint_observed_roles"] is True
    assert payload["input_bindings"]["evaluation_data_identity_sha256"] == identity[
        "evaluation_data_identity_sha256"
    ]
    assert payload["input_bindings"]["data_integrity_verification_record_sha256"] == _sha256(
        record
    )
    assert len(payload["role_identity"]["event_id_digests"]["evaluation"]) == 1
    assert payload["events"][0]["event_id"] == "Evaluation 01"
    assert payload["point_prediction_event_macro"]["n_events"] == 1
    assert len(payload["events"]) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "PASS"

    calibration_event = data / "calibration/calibration-01.npz"
    with np.load(calibration_event, allow_pickle=False) as archive:
        duplicate = {
            key: np.asarray(archive[key]).copy()
            for key in (
                "input",
                "target",
                "mask",
                "provider_event_name",
                "provider_relative_path",
            )
        }
    np.savez_compressed(
        calibration_event,
        **duplicate,
        event_id=np.asarray("validation-01"),
    )
    with pytest.raises(ValueError, match="data-integrity verification record does not bind"):
        run_evidence_pipeline(
            checkpoints=checkpoints,
            data_root=data,
            normalization_path=normalization,
            role_config_path=roles,
            calibration_config_path=calibration,
            data_integrity_record_path=record,
            output_path=tmp_path / "overlap.json",
            device="cpu",
            query_chunk_size=16,
        )



def test_public_role_aliases_must_be_pairwise_disjoint() -> None:
    config = {
        "roles": {
            "training": {
                "directory": "train",
                "count": 1,
                "public_event_aliases": ["same"],
            },
            "development": {
                "directory": "validation",
                "count": 1,
                "public_event_aliases": ["same"],
            },
            "calibration": {
                "directory": "calibration",
                "count": 1,
                "public_event_alias_prefix": "Calibration",
            },
            "evaluation": {
                "directory": "test",
                "count": 1,
                "public_event_alias_prefix": "Evaluation",
            },
        }
    }
    with pytest.raises(ValueError, match="overlaps roles"):
        validate_role_contracts(config)


def test_checked_in_role_contract_is_complete_and_disjoint() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs/evaluation/berlin_i_roles.yaml").read_text(
            encoding="utf-8"
        )
    )
    contracts = validate_role_contracts(config)
    assert {role: row.count for role, row in contracts.items()} == {
        "training": 85,
        "development": 15,
        "calibration": 13,
        "evaluation": 12,
    }
    aliases = [
        alias
        for contract in contracts.values()
        for alias in contract.public_event_aliases
    ]
    assert len(aliases) == len(set(aliases)) == 125
