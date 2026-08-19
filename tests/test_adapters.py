from __future__ import annotations

import torch
import torch.nn as nn

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

