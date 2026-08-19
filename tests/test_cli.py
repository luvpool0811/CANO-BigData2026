from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cano_bigdata2026.cli import evaluate_main, quickstart_main
from cano_bigdata2026.training import train


ROOT = Path(__file__).resolve().parents[1]


def test_quickstart_cpu(tmp_path: Path) -> None:
    output = tmp_path / "quickstart.json"
    assert (
        quickstart_main(
            [
                "--config",
                str(ROOT / "configs/demo/quickstart.yaml"),
                "--device",
                "cpu",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["output_shape"] == [1, 64, 3]
    assert np.isfinite(payload["initial_loss"])
    assert np.isfinite(payload["final_loss"])


def _event(path: Path, seed: int) -> None:
    generator = np.random.default_rng(seed)
    inputs = generator.normal(size=(31, 5, 7)).astype(np.float32)
    target = generator.normal(size=(72, 5, 7)).astype(np.float32)
    mask = np.ones((5, 7), dtype=bool)
    np.savez_compressed(
        path, input=inputs, target=target, mask=mask, event_id=np.asarray(path.stem)
    )


def test_train_and_evaluate_smoke(tmp_path: Path) -> None:
    data = tmp_path / "data"
    for split, count in (("train", 2), ("validation", 1), ("test", 1)):
        (data / split).mkdir(parents=True)
        for index in range(count):
            _event(data / split / f"{split}-{index}.npz", index + len(split))
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
    config = {
        "model": {
            "name": "cano",
            "latent_dim": 8,
            "branch_depth": 1,
            "forcing_hidden_dim": 8,
            "decoder_hidden_dim": 16,
            "n_freqs_xy": 2,
            "n_freqs_z": 1,
            "n_freqs_t": 1,
            "dropout": 0.0,
        },
        "training": {
            "max_epochs": 1,
            "patience": 1,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "queries_per_event": 8,
        },
    }
    output = tmp_path / "training"
    trained = train(
        config=config,
        data_root=data,
        output_dir=output,
        seed=42,
        device="cpu",
    )
    assert trained.checkpoint.is_file()
    metrics = tmp_path / "metrics.json"
    assert (
        evaluate_main(
            [
                "--checkpoint",
                str(trained.checkpoint),
                "--data-root",
                str(data),
                "--split",
                "test",
                "--normalization",
                str(normalization),
                "--output",
                str(metrics),
                "--device",
                "cpu",
                "--query-chunk-size",
                "16",
            ]
        )
        == 0
    )
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert payload["model"] == "cano"
    assert payload["aggregate"]["n_events"] == 1
