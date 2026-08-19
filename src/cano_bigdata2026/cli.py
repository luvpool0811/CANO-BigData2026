"""Command-line entry points for the public CANO repository."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from . import contracts as C
from .data import EventNPZDataset, Normalization, event_files
from .metrics import aggregate, event_metrics
from .models import CANO, build_model, real_parameter_count
from .training import set_seed, train


def _load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("configuration must contain a model mapping")
    return payload


def _device(value: str) -> str:
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return value


def _train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CANO or a public baseline")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--upstream-source", type=Path)
    return parser


def train_main(argv: Sequence[str] | None = None) -> int:
    args = _train_parser().parse_args(argv)
    result = train(
        config=_load_yaml(args.config),
        data_root=args.data_root,
        output_dir=args.output_dir,
        seed=args.seed,
        device=_device(args.device),
        upstream_source=args.upstream_source,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "best_validation_loss": result.best_validation_loss,
                "epochs_completed": result.epochs_completed,
                "parameter_count": result.parameter_count,
                "checkpoint": str(result.checkpoint),
            },
            sort_keys=True,
        )
    )
    return 0


def _evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--upstream-source", type=Path)
    parser.add_argument("--query-chunk-size", type=int, default=32768)
    return parser


def evaluate_main(argv: Sequence[str] | None = None) -> int:
    args = _evaluate_parser().parse_args(argv)
    target_device = torch.device(_device(args.device))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(
        checkpoint["model_name"],
        checkpoint["model_config"],
        upstream_source=args.upstream_source,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(target_device).eval()
    normalization = Normalization.load(args.normalization)
    dataset = EventNPZDataset(event_files(args.data_root, args.split))
    records: list[dict[str, object]] = []
    with torch.no_grad():
        for sample in dataset:
            inputs = sample["input"].unsqueeze(0).to(target_device)
            if isinstance(model, CANO):
                prediction = model(inputs, query_chunk_size=args.query_chunk_size)[0]
            else:
                prediction = model(inputs)[0]
            prediction = normalization.denormalize_output(prediction.cpu())
            truth = normalization.denormalize_output(sample["target"])
            row = event_metrics(
                prediction.numpy(), truth.numpy(), sample["mask"].numpy()
            )
            row["event_id"] = sample["event_id"]
            records.append(row)
    payload = {
        "model": checkpoint["model_name"],
        "seed": int(checkpoint["seed"]),
        "split": args.split,
        "aggregate": aggregate(records),
        "events": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "output": str(args.output)}, sort_keys=True))
    return 0


def _quickstart_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small synthetic CANO smoke test")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs/demo/quickstart.yaml",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("outputs/quickstart.json"))
    return parser


def quickstart_main(argv: Sequence[str] | None = None) -> int:
    args = _quickstart_parser().parse_args(argv)
    config = _load_yaml(args.config)
    device = torch.device(_device(args.device))
    seed = int(config["training"].get("seed", 42))
    set_seed(seed)
    model_settings = dict(config["model"])
    name = str(model_settings.pop("name"))
    if name != "cano":
        raise ValueError("the self-contained quickstart uses CANO")
    model = build_model(name, model_settings).to(device)
    height = int(config["data"]["height"])
    width = int(config["data"]["width"])
    query_count = int(config["training"]["queries_per_event"])
    steps = int(config["training"]["steps"])
    inputs = torch.randn(1, C.N_INPUT_CHANNELS, height, width, device=device)
    query = torch.rand(1, query_count, 2, device=device) * 2.0 - 1.0
    with torch.no_grad():
        reference_model = build_model(name, model_settings).to(device).eval()
        target = reference_model.forward_queries(inputs, query, lead=5).detach()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"])
    )
    losses: list[float] = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model.forward_queries(inputs, query, lead=5)
        loss = torch.mean((prediction - target) ** 2)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    payload = {
        "status": "PASS" if np.isfinite(losses).all() else "FAIL",
        "device": str(device),
        "steps": steps,
        "parameter_count": real_parameter_count(model),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "output_shape": [1, query_count, 3],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


__all__ = ["train_main", "evaluate_main", "quickstart_main"]
