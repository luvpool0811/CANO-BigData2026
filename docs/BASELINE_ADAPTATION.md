# Fair adaptation of the external baselines

The external comparison uses the DNO, FNO3D, and U-Net3D architecture classes
published by the UrbanFloodCast authors. Their source files are not copied or
modified here. `urbanfloodcast_adapters.py` performs only the interface work
needed for a controlled comparison.

## Source checkout

```bash
git clone https://github.com/HydroPML/UrbanFloodCast.git external/UrbanFloodCast
git -C external/UrbanFloodCast checkout f08846a1d0ed5a82d9241d2229df8ec8997ebfd5
```

The loader refuses a different revision or tracked local modifications. This
protects against accidentally testing a different implementation while keeping
the third-party code outside this repository.

## Shared experimental contract

All four models receive the same information:

- 31 input channels: initial H/U/V, DEM, roughness, grid Y/X, and 24 rainfall
  channels;
- 24 forecast lead times;
- three output variables per lead (H/U/V);
- identical train, validation, and evaluation event partitions;
- seeds 7, 31, and 42; and
- the same optimizer-level settings in the published YAML files.

The upstream architectures use a one-shot space-time tensor. The adapter maps
the common tensor `[B,31,H,W]` to `[B,H,W,24,1,6]`, invokes the original model,
and maps `[B,H,W,24,3]` back to `[B,72,H,W]`.

## Reported architecture settings

| Model | Settings | Trainable real-valued parameters |
|---|---|---:|
| CANO | latent 64, four branch blocks, decoder width 192 | 276,821 |
| DNO-3 | width 11, factor 1 | 10,814,514 |
| FNO3D | effective modes 12/12/4, width 24 | 10,626,963 |
| U-Net3D | initial features 16 | 5,649,875 |

These settings are encoded in `configs/`. The adapter is our integration code;
it is not a redistribution of, or a substitute license for, the upstream
implementation.

The pinned upstream `FNO3d` class accepts a `modes3` constructor argument but
sets its effective temporal mode count to `4`. The public configuration records
that effective value, and the adapter rejects any conflicting value instead of
silently reporting `12/12/8`.

## Setting selection and training exposure

The complete disclosure is checked in as
`results/paper/baseline_fairness.csv` and regenerated as
`results/generated/baseline_fairness.md`. Each system had six candidate
settings evaluated on development data only. The selected external
configuration was then trained from scratch for seeds 7, 31, and 42 for 100
epochs per seed (12,900 optimizer steps per system). Candidate checkpoint
weights were not imported, and held-out evaluation events were not used for
setting or checkpoint selection.

Measured wall time is included for transparency, but the systems do not share
identical kernels or memory behavior, so it is not interpreted as a
hardware-normalized efficiency comparison. The table evaluates complete
systems and does not identify an architecture-only causal effect.
