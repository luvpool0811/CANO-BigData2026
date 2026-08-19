"""CANO model and model factory used by the public training scripts.

The standard configuration in ``configs/cano/standard.yaml`` is the exact
architecture used for the reported 0.277M-parameter CANO system.  The model
keeps rainfall forcing and arbitrary coordinate queries explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import contracts as C


class FourierFeatures1D(nn.Module):
    def __init__(self, n_freqs: int, base: float = 2.0):
        super().__init__()
        self.n_freqs = int(n_freqs)
        frequencies = base ** torch.arange(n_freqs, dtype=torch.float32) * math.pi
        self.register_buffer("frequencies", frequencies)

    @property
    def out_dim(self) -> int:
        return 2 * self.n_freqs

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        projected = value.unsqueeze(-1) * self.frequencies
        return torch.cat([projected.sin(), projected.cos()], dim=-1)


class FourierFeatures2D(nn.Module):
    def __init__(self, n_freqs: int, base: float = 2.0):
        super().__init__()
        self.n_freqs = int(n_freqs)
        frequencies = base ** torch.arange(n_freqs, dtype=torch.float32) * math.pi
        self.register_buffer("frequencies", frequencies)

    @property
    def out_dim(self) -> int:
        return 4 * self.n_freqs

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        projected = coordinates.unsqueeze(-1) * self.frequencies
        return torch.cat([projected.sin(), projected.cos()], dim=-1).flatten(-2)


class RainfallForcingEncoder(nn.Module):
    """Causal rainfall hydrograph encoder."""

    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(1, hidden_dim, batch_first=True)
        self.projection = nn.Linear(hidden_dim, latent_dim)

    def forward(self, rainfall: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.gru(rainfall)
        return self.projection(sequence)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        groups = 4 if channels % 4 == 0 else 1
        self.network = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.gelu(tensor + self.network(tensor))


class CoordinateDecoder(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_dim: int,
        n_freqs_xy: int,
        n_freqs_z: int,
        n_freqs_t: int,
        dropout: float,
    ):
        super().__init__()
        self.xy_features = FourierFeatures2D(n_freqs_xy)
        self.z_features = FourierFeatures1D(n_freqs_z)
        self.tau_features = FourierFeatures1D(1)
        self.time_features = FourierFeatures1D(n_freqs_t)
        input_dim = (
            latent_dim
            + self.xy_features.out_dim
            + self.z_features.out_dim
            + self.tau_features.out_dim
            + self.time_features.out_dim
        )
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.GELU(),
            nn.LayerNorm(input_dim),
        )
        self.h_head = nn.Linear(input_dim, 1)
        self.log_variance_head = nn.Linear(input_dim, 1)
        self.exceedance_head = nn.Linear(input_dim, 1)
        self.u_head = nn.Linear(input_dim, 1)
        self.v_head = nn.Linear(input_dim, 1)
        nn.init.zeros_(self.log_variance_head.weight)
        nn.init.constant_(self.log_variance_head.bias, -2.0)

    def forward(
        self,
        latent: torch.Tensor,
        coordinates_01: torch.Tensor,
        elevation_01: torch.Tensor,
        time_01: torch.Tensor,
    ) -> torch.Tensor:
        tau = torch.zeros_like(time_01)
        features = torch.cat(
            [
                latent,
                self.xy_features(coordinates_01),
                self.z_features(elevation_01),
                self.tau_features(tau),
                self.time_features(time_01),
            ],
            dim=-1,
        )
        encoded = self.network(features)
        h = self.h_head(encoded).squeeze(-1)
        u = self.u_head(encoded).squeeze(-1)
        v = self.v_head(encoded).squeeze(-1)
        return torch.stack([h, u, v], dim=-1)


@dataclass(frozen=True)
class EncodedEvent:
    local_grid: torch.Tensor
    forcing_sequence: torch.Tensor
    elevation_grid: torch.Tensor


class CANO(nn.Module):
    """Coverage-Aware Neural Operator for the 31-channel UFC task."""

    def __init__(
        self,
        *,
        latent_dim: int = 64,
        branch_depth: int = 4,
        forcing_hidden_dim: int = 64,
        decoder_hidden_dim: int = 192,
        n_freqs_xy: int = 8,
        n_freqs_z: int = 4,
        n_freqs_t: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.branch_stem = nn.Conv2d(C.CH_RAIN_START, latent_dim, 3, padding=1)
        dilations = (1, 2, 4, 8)
        self.branch_blocks = nn.Sequential(
            *[
                ResidualConvBlock(latent_dim, dilations[index % 4])
                for index in range(branch_depth)
            ]
        )
        self.forcing_encoder = RainfallForcingEncoder(
            latent_dim=latent_dim, hidden_dim=forcing_hidden_dim
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )
        self.decoder = CoordinateDecoder(
            latent_dim=latent_dim,
            hidden_dim=decoder_hidden_dim,
            n_freqs_xy=n_freqs_xy,
            n_freqs_z=n_freqs_z,
            n_freqs_t=n_freqs_t,
            dropout=dropout,
        )

    def encode(self, inputs: torch.Tensor) -> EncodedEvent:
        if inputs.ndim != 4 or inputs.shape[1] != C.N_INPUT_CHANNELS:
            raise ValueError(
                f"expected [B,{C.N_INPUT_CHANNELS},H,W], got {tuple(inputs.shape)}"
            )
        local = self.branch_blocks(self.branch_stem(inputs[:, : C.CH_RAIN_START]))
        rainfall = inputs[:, C.CH_RAIN_START :].mean(dim=(-2, -1)).unsqueeze(-1)
        forcing = self.forcing_encoder(rainfall)
        elevation = inputs[:, C.CH_DEM : C.CH_DEM + 1]
        return EncodedEvent(local, forcing, elevation)

    @staticmethod
    def _sample(grid: torch.Tensor, coordinates_xy: torch.Tensor) -> torch.Tensor:
        sampled = F.grid_sample(
            grid,
            coordinates_xy.unsqueeze(2),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(-1).transpose(1, 2)

    def decode_queries(
        self,
        encoded: EncodedEvent,
        coordinates_xy: torch.Tensor,
        lead: int,
    ) -> torch.Tensor:
        if not 0 <= int(lead) < C.N_LEADS:
            raise ValueError("lead must be in [0, 23]")
        local = self._sample(encoded.local_grid, coordinates_xy)
        forcing = encoded.forcing_sequence[:, int(lead)].unsqueeze(1)
        forcing = forcing.expand(-1, coordinates_xy.shape[1], -1)
        latent = self.fusion(torch.cat([local, forcing], dim=-1))
        elevation = torch.sigmoid(
            self._sample(encoded.elevation_grid, coordinates_xy).squeeze(-1)
        )
        coordinates_01 = (coordinates_xy + 1.0) * 0.5
        time_01 = torch.full(
            coordinates_xy.shape[:2],
            (int(lead) + 1) / C.N_LEADS,
            dtype=coordinates_xy.dtype,
            device=coordinates_xy.device,
        )
        return self.decoder(latent, coordinates_01, elevation, time_01)

    def forward_queries(
        self, inputs: torch.Tensor, coordinates_xy: torch.Tensor, lead: int
    ) -> torch.Tensor:
        return self.decode_queries(self.encode(inputs), coordinates_xy, lead)

    def forward(self, inputs: torch.Tensor, query_chunk_size: int = 32768) -> torch.Tensor:
        encoded = self.encode(inputs)
        batch, _, height, width = inputs.shape
        yy = torch.linspace(-1.0, 1.0, height, device=inputs.device, dtype=inputs.dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=inputs.device, dtype=inputs.dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        coordinates = torch.stack([gx, gy], dim=-1).reshape(1, -1, 2)
        coordinates = coordinates.expand(batch, -1, -1)
        output = inputs.new_empty((batch, C.N_OUTPUT_CHANNELS, height * width))
        for lead in range(C.N_LEADS):
            for start in range(0, height * width, query_chunk_size):
                stop = min(start + query_chunk_size, height * width)
                values = self.decode_queries(encoded, coordinates[:, start:stop], lead)
                for variable in range(3):
                    output[:, C.output_channel(lead, variable), start:stop] = values[
                        ..., variable
                    ]
        return output.reshape(batch, C.N_OUTPUT_CHANNELS, height, width)


def real_parameter_count(model: nn.Module) -> int:
    return int(
        sum(
            parameter.numel() * (2 if parameter.is_complex() else 1)
            for parameter in model.parameters()
        )
    )


def build_model(
    name: str,
    config: Mapping[str, Any],
    *,
    upstream_source: str | Path | None = None,
) -> nn.Module:
    options = dict(config)
    options.pop("name", None)
    if name == "cano":
        return CANO(**options)
    if name in {"dno3", "fno3d", "unet3d"}:
        if upstream_source is None:
            raise ValueError(f"{name} requires --upstream-source")
        from .urbanfloodcast_adapters import build_upstream_model

        return build_upstream_model(name, upstream_source, options)
    raise ValueError(f"unknown model: {name}")


__all__ = ["CANO", "build_model", "real_parameter_count"]
