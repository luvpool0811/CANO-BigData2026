"""Three-seed ensemble, calibration, and event-level evaluation workflow."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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


def _role_contract(config: Mapping[str, Any], role: str) -> tuple[str, int]:
    roles = config.get("roles")
    row = roles.get(role) if isinstance(roles, Mapping) else None
    if not isinstance(row, Mapping):
        raise ValueError(f"role contract is missing {role}")
    directory = str(row.get("directory", ""))
    count = int(row.get("count", -1))
    if not directory or count < 1:
        raise ValueError(f"role contract for {role} is invalid")
    return directory, count


def run_evidence_pipeline(
    *,
    checkpoints: Sequence[Path],
    data_root: Path,
    normalization_path: Path,
    role_config_path: Path,
    calibration_config_path: Path,
    output_path: Path,
    device: str,
    upstream_source: Path | None = None,
    query_chunk_size: int = 32768,
) -> dict[str, Any]:
    target_device = torch.device(device)
    models, seeds, model_name, model_config = _load_models(
        checkpoints, device=target_device, upstream_source=upstream_source
    )
    role_config = _load_yaml(role_config_path)
    calibration_config = _load_yaml(calibration_config_path)
    expected_seeds = role_config.get("ensemble_seeds")
    if expected_seeds is not None and sorted(
        int(value) for value in expected_seeds
    ) != sorted(seeds):
        raise ValueError("checkpoint seeds differ from the role contract")
    normalization = Normalization.load(normalization_path)
    development_split, development_count = _role_contract(role_config, "development")
    calibration_split, calibration_count = _role_contract(role_config, "calibration")
    evaluation_split, evaluation_count = _role_contract(role_config, "evaluation")
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

    def role_events(split: str, expected_count: int) -> Iterator[PhysicalEvent]:
        nonlocal first_mask
        count = 0
        for event in _predict_role(
            models,
            data_root=data_root,
            split=split,
            normalization=normalization,
            device=target_device,
            query_chunk_size=query_chunk_size,
        ):
            if first_mask is None:
                first_mask = event.mask.copy()
            elif not np.array_equal(first_mask, event.mask):
                raise ValueError("all role events must use the same valid-cell mask")
            count += 1
            yield event
        if count != expected_count:
            raise ValueError(
                f"role {split} has {count} events; expected {expected_count}"
            )

    development_pairs = (
        (event.prediction_h, event.truth_h)
        for event in role_events(development_split, development_count)
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

    calibration_events = role_events(calibration_split, calibration_count)
    calibrator = fit_target_aligned_calibrator_events(
        ((event.prediction_h, event.truth_h) for event in calibration_events),
        first_mask,
        node_scale,
        threshold_m=operational_threshold,
        alpha_levels=alpha_levels,
    )

    event_records: list[dict[str, Any]] = []
    for event in role_events(evaluation_split, evaluation_count):
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
            "development": development_count,
            "calibration": calibration_count,
            "evaluation": evaluation_count,
        },
        "evaluation_target_reads": evaluation_count,
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
        output_path=args.output,
        device=args.device,
        upstream_source=args.upstream_source,
        query_chunk_size=args.query_chunk_size,
    )
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))
    return 0


__all__ = ["PhysicalEvent", "run_evidence_pipeline", "main"]
