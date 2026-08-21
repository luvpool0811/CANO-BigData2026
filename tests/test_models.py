from __future__ import annotations

import torch

from cano_bigdata2026.models import build_model, real_parameter_count


OFFICIAL = {
    "latent_dim": 64,
    "branch_depth": 4,
    "forcing_hidden_dim": 64,
    "decoder_hidden_dim": 192,
    "n_freqs_xy": 8,
    "n_freqs_z": 4,
    "n_freqs_t": 4,
    "dropout": 0.0,
}


def test_official_cano_parameter_count() -> None:
    model = build_model("cano", OFFICIAL)
    assert real_parameter_count(model) == 276_821


def test_cano_coordinate_query_shape_and_gradient() -> None:
    model = build_model(
        "cano",
        {
            "latent_dim": 16,
            "branch_depth": 1,
            "forcing_hidden_dim": 16,
            "decoder_hidden_dim": 32,
            "n_freqs_xy": 3,
            "n_freqs_z": 2,
            "n_freqs_t": 2,
            "dropout": 0.0,
        },
    )
    inputs = torch.randn(2, 31, 8, 10)
    queries = torch.rand(2, 13, 2) * 2 - 1
    output = model.forward_queries(inputs, queries, lead=7)
    assert output.shape == (2, 13, 3)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert model.decoder.log_variance_head.weight.grad is None
    assert model.decoder.exceedance_head.weight.grad is None
    assert model.decoder.h_head.weight.grad is not None


def test_cano_dense_query_lattice_matches_align_corners_contract() -> None:
    model = build_model("cano", OFFICIAL)
    inputs = torch.zeros(2, 31, 3, 4)
    lattice = model._dense_query_lattice(inputs)
    assert lattice.shape == (2, 12, 2)
    assert torch.equal(lattice[0], lattice[1])
    assert torch.equal(lattice[0, 0], torch.tensor([-1.0, -1.0]))
    assert torch.equal(lattice[0, -1], torch.tensor([1.0, 1.0]))
