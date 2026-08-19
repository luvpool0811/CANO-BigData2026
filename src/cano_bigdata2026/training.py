"""Small, transparent training helpers for CANO and the baseline adapters."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import contracts as C
from .data import EventNPZDataset, event_files
from .models import CANO, build_model, real_parameter_count


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _coordinates_from_indices(
    rows: torch.Tensor,
    columns: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    x = columns.to(torch.float32) * (2.0 / max(width - 1, 1)) - 1.0
    y = rows.to(torch.float32) * (2.0 / max(height - 1, 1)) - 1.0
    return torch.stack([x, y], dim=-1)


def _cano_query_loss(
    model: CANO,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    *,
    queries_per_event: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if inputs.shape[0] != 1:
        raise ValueError("CANO query training currently uses micro-batch size 1")
    _, _, height, width = inputs.shape
    valid = torch.nonzero(masks[0], as_tuple=False)
    if valid.numel() == 0:
        raise ValueError("event contains no valid cells")
    count = min(int(queries_per_event), valid.shape[0])
    choice = torch.randperm(valid.shape[0], generator=generator)[:count]
    selected = valid[choice]
    lead = int(torch.randint(C.N_LEADS, (1,), generator=generator).item())
    coordinates = _coordinates_from_indices(
        selected[:, 0], selected[:, 1], height=height, width=width
    ).to(inputs.device).unsqueeze(0)
    prediction = model.forward_queries(inputs, coordinates, lead)
    physical = targets.reshape(1, C.N_LEADS, 3, height, width)
    truth = physical[0, lead, :, selected[:, 0], selected[:, 1]].transpose(0, 1)
    return torch.mean((prediction[0] - truth) ** 2)


def _dense_loss(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    prediction = model(inputs)
    valid = masks.unsqueeze(1).expand_as(targets)
    return torch.mean((prediction[valid] - targets[valid]) ** 2)


@dataclass(frozen=True)
class TrainResult:
    best_validation_loss: float
    epochs_completed: int
    parameter_count: int
    checkpoint: Path


def train(
    *,
    config: Mapping[str, Any],
    data_root: str | Path,
    output_dir: str | Path,
    seed: int,
    device: str,
    upstream_source: str | Path | None = None,
) -> TrainResult:
    """Train one seed with early stopping and write a portable checkpoint."""

    set_seed(seed)
    target_device = torch.device(device)
    model_settings = dict(config["model"])
    model_name = str(model_settings.pop("name"))
    model = build_model(
        model_name, model_settings, upstream_source=upstream_source
    ).to(target_device)
    settings = dict(config["training"])
    train_dataset = EventNPZDataset(event_files(data_root, "train"))
    validation_dataset = EventNPZDataset(event_files(data_root, "validation"))
    batch_size = int(settings.get("micro_batch_size", 1))
    if model_name == "cano" and batch_size != 1:
        raise ValueError("CANO query training requires micro_batch_size: 1")
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings.get("weight_decay", 0.0)),
    )
    accumulation = int(settings.get("gradient_accumulation_steps", 1))
    gradient_clip = float(settings.get("gradient_clip_norm", 0.0))
    query_count = int(settings.get("queries_per_event", 4096))
    max_epochs = int(settings["max_epochs"])
    patience = int(settings["patience"])
    sample_generator = torch.Generator().manual_seed(seed + 1009)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best.pt"
    best = float("inf")
    stale = 0
    epochs_completed = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            inputs = batch["input"].to(target_device)
            targets = batch["target"].to(target_device)
            masks = batch["mask"].to(target_device)
            if model_name == "cano":
                loss = _cano_query_loss(
                    model,
                    inputs,
                    targets,
                    masks,
                    queries_per_event=query_count,
                    generator=sample_generator,
                )
            else:
                loss = _dense_loss(model, inputs, targets, masks)
            (loss / accumulation).backward()
            if step % accumulation == 0 or step == len(train_loader):
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        model.eval()
        validation_losses: list[float] = []
        with torch.no_grad():
            for batch in validation_loader:
                inputs = batch["input"].to(target_device)
                targets = batch["target"].to(target_device)
                masks = batch["mask"].to(target_device)
                if model_name == "cano":
                    value = _cano_query_loss(
                        model,
                        inputs,
                        targets,
                        masks,
                        queries_per_event=query_count,
                        generator=sample_generator,
                    )
                else:
                    value = _dense_loss(model, inputs, targets, masks)
                validation_losses.append(float(value.cpu()))
        score = float(np.mean(validation_losses))
        epochs_completed = epoch + 1
        if score < best:
            best = score
            stale = 0
            torch.save(
                {
                    "model_name": model_name,
                    "model_config": model_settings,
                    "state_dict": model.state_dict(),
                    "seed": int(seed),
                    "epoch": int(epoch),
                    "validation_loss": score,
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= patience:
                break

    summary = {
        "model": model_name,
        "seed": int(seed),
        "parameter_count": real_parameter_count(model),
        "best_validation_loss": best,
        "epochs_completed": epochs_completed,
        "checkpoint": checkpoint_path.name,
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return TrainResult(
        best_validation_loss=best,
        epochs_completed=epochs_completed,
        parameter_count=int(summary["parameter_count"]),
        checkpoint=checkpoint_path,
    )


__all__ = ["TrainResult", "set_seed", "train"]
