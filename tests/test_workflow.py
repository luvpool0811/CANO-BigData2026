from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from cano_bigdata2026.models import build_model
from cano_bigdata2026.workflow import run_evidence_pipeline


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


def _event(path: Path, seed: int) -> None:
    generator = np.random.default_rng(seed)
    inputs = generator.normal(size=(31, 5, 7)).astype(np.float32)
    target = generator.uniform(0.1, 0.8, size=(72, 5, 7)).astype(np.float32)
    mask = np.ones((5, 7), dtype=bool)
    np.savez_compressed(
        path, input=inputs, target=target, mask=mask, event_id=np.asarray(path.stem)
    )


def test_three_seed_evidence_pipeline_smoke(tmp_path: Path) -> None:
    data = tmp_path / "data"
    for index, split in enumerate(("validation", "calibration", "test")):
        (data / split).mkdir(parents=True)
        _event(data / split / f"{split}-01.npz", index + 10)
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
        "roles:\n"
        "  development: {directory: validation, count: 1}\n"
        "  calibration: {directory: calibration, count: 1}\n"
        "  evaluation: {directory: test, count: 1}\n",
        encoding="utf-8",
    )
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
    payload = run_evidence_pipeline(
        checkpoints=checkpoints,
        data_root=data,
        normalization_path=normalization,
        role_config_path=roles,
        calibration_config_path=calibration,
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
    assert payload["point_prediction_event_macro"]["n_events"] == 1
    assert len(payload["events"]) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "PASS"
