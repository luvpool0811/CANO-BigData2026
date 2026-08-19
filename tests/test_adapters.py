from __future__ import annotations

import torch
import torch.nn as nn
import pytest

import cano_bigdata2026.urbanfloodcast_adapters as adapters
from cano_bigdata2026.urbanfloodcast_adapters import (
    UrbanFloodCastAdapter,
    common_input_to_upstream,
    upstream_output_to_common,
)


class IdentityShapeCore(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[..., :3].squeeze(-2)


def test_common_input_mapping() -> None:
    common = torch.randn(2, 31, 6, 7)
    upstream = common_input_to_upstream(common, include_roughness=True)
    assert upstream.shape == (2, 6, 7, 24, 1, 6)
    assert torch.equal(upstream[:, :, :, :, 0, 0], common[:, 0].unsqueeze(-1).expand(-1, -1, -1, 24))
    assert torch.equal(upstream[:, :, :, :, 0, 3], common[:, 7:].permute(0, 2, 3, 1))


def test_output_mapping_and_wrapper() -> None:
    common = torch.randn(1, 31, 6, 7)
    adapter = UrbanFloodCastAdapter(
        IdentityShapeCore(), name="dummy", include_roughness=True
    )
    output = adapter(common)
    assert output.shape == (1, 72, 6, 7)
    upstream = torch.randn(2, 6, 7, 24, 3)
    roundtrip = upstream_output_to_common(upstream)
    assert roundtrip.shape == (2, 72, 6, 7)
    assert torch.equal(roundtrip.reshape(2, 24, 3, 6, 7), upstream.permute(0, 3, 4, 1, 2))


class FakeFNO(nn.Module):
    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        self.modes3 = 4


def test_fno3d_effective_modes_are_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    classes = type("Classes", (), {"fno3d": FakeFNO})()
    monkeypatch.setattr(adapters, "load_upstream_classes", lambda _: classes)
    with pytest.raises(ValueError, match="fixes modes3 to 4"):
        adapters.build_upstream_model(
            "fno3d",
            "/unused",
            {
                "modes1": 12,
                "modes2": 12,
                "modes3": 8,
                "width": 24,
                "include_roughness": True,
            },
        )
    model = adapters.build_upstream_model(
        "fno3d",
        "/unused",
        {
            "modes1": 12,
            "modes2": 12,
            "modes3": 4,
            "width": 24,
            "include_roughness": True,
        },
    )
    assert model.core.modes3 == 4
