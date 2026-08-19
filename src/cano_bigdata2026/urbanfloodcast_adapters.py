"""Adapters for the authors' public UrbanFloodCast DNO/FNO3D/U-Net3D code.

The upstream source is not redistributed here.  Clone
https://github.com/HydroPML/UrbanFloodCast and check out the commit below.  The
adapter imports the released classes without editing them and maps the common
31-channel raster contract to their one-shot space-time contract.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import subprocess
import sys
import types
from typing import Any, Iterator, Mapping

import numpy as np
import torch
import torch.nn as nn

from . import contracts as C


UPSTREAM_REPOSITORY = "https://github.com/HydroPML/UrbanFloodCast"
UPSTREAM_COMMIT = "f08846a1d0ed5a82d9241d2229df8ec8997ebfd5"
UPSTREAM_LEADS = 24
UPSTREAM_VARIABLES = 3
UNET_POOLING_FACTOR = 16
UNET_BERLIN_I_REMAINDER = (6, 7)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_upstream_checkout(source_root: str | Path) -> Path:
    root = Path(source_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    commit = _git(root, "rev-parse", "HEAD")
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"UrbanFloodCast commit must be {UPSTREAM_COMMIT}; found {commit}"
        )
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("UrbanFloodCast checkout contains tracked modifications")
    required = (
        "DNO/models/DNO.py",
        "DNO/models/FNO.py",
        "DNO/models/Unet.py",
        "DNO/models/integral_operators.py",
        "DNO/models/utilities3.py",
        "DNO/models/Adam.py",
        "DNO/utils25.py",
    )
    for relative in required:
        candidate = (root / relative).resolve(strict=True)
        if root not in candidate.parents or not candidate.is_file():
            raise RuntimeError(f"missing or unsafe upstream file: {relative}")
    return root


@contextmanager
def _preserve_rng() -> Iterator[None]:
    torch_state = torch.random.get_rng_state()
    numpy_state = np.random.get_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        torch.random.set_rng_state(torch_state)
        np.random.set_state(numpy_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _package(name: str, path: Path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    return module


def _execute(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class UpstreamClasses:
    dno: type[nn.Module]
    fno3d: type[nn.Module]
    unet3d: type[nn.Module]


_CLASS_CACHE: dict[str, UpstreamClasses] = {}


def load_upstream_classes(source_root: str | Path) -> UpstreamClasses:
    root = verify_upstream_checkout(source_root)
    key = str(root)
    if key in _CLASS_CACHE:
        return _CLASS_CACHE[key]
    namespace = f"_cano_ufc_{UPSTREAM_COMMIT[:12]}"
    dno_root = root / "DNO"
    models_root = dno_root / "models"
    prior_utils = sys.modules.get("utils")
    prior_dont_write = sys.dont_write_bytecode
    created: list[str] = []
    try:
        sys.modules[namespace] = _package(namespace, dno_root)
        sys.modules[f"{namespace}.models"] = _package(
            f"{namespace}.models", models_root
        )
        created.extend([namespace, f"{namespace}.models"])
        sys.dont_write_bytecode = True
        with _preserve_rng():
            utils = _execute(f"{namespace}.utils25", dno_root / "utils25.py")
            created.append(f"{namespace}.utils25")
            sys.modules["utils"] = utils
            for module_name in ("integral_operators", "utilities3", "Adam"):
                qualified = f"{namespace}.models.{module_name}"
                _execute(qualified, models_root / f"{module_name}.py")
                created.append(qualified)
            dno = _execute(f"{namespace}.models.DNO", models_root / "DNO.py")
            fno = _execute(f"{namespace}.models.FNO", models_root / "FNO.py")
            unet = _execute(f"{namespace}.models.Unet", models_root / "Unet.py")
            created.extend(
                [
                    f"{namespace}.models.DNO",
                    f"{namespace}.models.FNO",
                    f"{namespace}.models.Unet",
                ]
            )
    except BaseException:
        for name in reversed(created):
            sys.modules.pop(name, None)
        raise
    finally:
        sys.dont_write_bytecode = prior_dont_write
        if prior_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = prior_utils
    classes = UpstreamClasses(dno.DNO, fno.FNO3d, unet.UNet3d)
    _CLASS_CACHE[key] = classes
    return classes


def common_input_to_upstream(
    common_input: torch.Tensor, *, include_roughness: bool = True
) -> torch.Tensor:
    if common_input.ndim != 4 or common_input.shape[1] != C.N_INPUT_CHANNELS:
        raise ValueError(
            f"expected [B,{C.N_INPUT_CHANNELS},H,W], got {tuple(common_input.shape)}"
        )
    batch, _, height, width = common_input.shape
    state = common_input[:, C.CH_H0 : C.CH_V0 + 1]
    state = state.permute(0, 2, 3, 1).unsqueeze(3).expand(
        batch, height, width, UPSTREAM_LEADS, 3
    )
    rainfall = common_input[:, C.CH_RAIN_START :]
    rainfall = rainfall.permute(0, 2, 3, 1).unsqueeze(-1)
    elevation = common_input[:, C.CH_DEM : C.CH_DEM + 1]
    elevation = elevation.permute(0, 2, 3, 1).unsqueeze(3).expand(
        batch, height, width, UPSTREAM_LEADS, 1
    )
    fields = [state, rainfall, elevation]
    if include_roughness:
        roughness = common_input[:, C.CH_ROUGHNESS : C.CH_ROUGHNESS + 1]
        roughness = roughness.permute(0, 2, 3, 1).unsqueeze(3).expand(
            batch, height, width, UPSTREAM_LEADS, 1
        )
        fields.append(roughness)
    return torch.cat(fields, dim=-1).unsqueeze(-2).contiguous()


def upstream_output_to_common(output: torch.Tensor) -> torch.Tensor:
    if output.ndim != 5 or output.shape[3:] != (
        UPSTREAM_LEADS,
        UPSTREAM_VARIABLES,
    ):
        raise ValueError(f"expected [B,H,W,24,3], got {tuple(output.shape)}")
    batch, height, width, _, _ = output.shape
    return output.permute(0, 3, 4, 1, 2).reshape(
        batch, C.N_OUTPUT_CHANNELS, height, width
    ).contiguous()


class UrbanFloodCastAdapter(nn.Module):
    def __init__(
        self,
        core: nn.Module,
        *,
        name: str,
        include_roughness: bool,
        required_remainder: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.core = core
        self.name = name
        self.include_roughness = include_roughness
        self.required_remainder = required_remainder

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.required_remainder is not None:
            height, width = inputs.shape[-2:]
            observed = (height % UNET_POOLING_FACTOR, width % UNET_POOLING_FACTOR)
            if observed != self.required_remainder:
                raise ValueError(
                    f"{self.name} requires spatial remainder {self.required_remainder}; "
                    f"found {observed}"
                )
        upstream = common_input_to_upstream(
            inputs, include_roughness=self.include_roughness
        )
        return upstream_output_to_common(self.core(upstream))


def build_upstream_model(
    name: str,
    source_root: str | Path,
    config: Mapping[str, Any],
) -> UrbanFloodCastAdapter:
    classes = load_upstream_classes(source_root)
    options = dict(config)
    include_roughness = bool(options.pop("include_roughness", True))
    channels = 6 if include_roughness else 5
    if name == "dno3":
        core = classes.dno(
            num_channels=channels,
            width=int(options.pop("width", 11)),
            initial_step=1,
            pad=0,
            factor=int(options.pop("factor", 1)),
            pad_both=False,
        )
        remainder = None
    elif name == "fno3d":
        core = classes.fno3d(
            num_channels=channels,
            modes1=int(options.pop("modes1", 12)),
            modes2=int(options.pop("modes2", 12)),
            modes3=int(options.pop("modes3", 8)),
            width=int(options.pop("width", 24)),
            initial_step=1,
            time=True,
            time_pad=False,
        )
        remainder = None
    elif name == "unet3d":
        core = classes.unet3d(
            in_channels=channels,
            out_channels=3,
            init_features=int(options.pop("init_features", 16)),
            grid_type="cartesian",
            time=True,
            time_pad=False,
        )
        remainder = UNET_BERLIN_I_REMAINDER
    else:
        raise ValueError(f"unknown upstream model: {name}")
    if options:
        raise ValueError(f"unused model settings for {name}: {sorted(options)}")
    return UrbanFloodCastAdapter(
        core,
        name=name,
        include_roughness=include_roughness,
        required_remainder=remainder,
    )


def real_parameter_count(model: nn.Module) -> int:
    return int(
        sum(
            parameter.numel() * (2 if parameter.is_complex() else 1)
            for parameter in model.parameters()
        )
    )


__all__ = [
    "UPSTREAM_REPOSITORY",
    "UPSTREAM_COMMIT",
    "UrbanFloodCastAdapter",
    "verify_upstream_checkout",
    "common_input_to_upstream",
    "upstream_output_to_common",
    "build_upstream_model",
    "real_parameter_count",
]
