# Dataset preparation

## Original source

The paper uses Berlin I from **Urban Flood Modeling and Forecasting with Deep
Neural Operator**:

- dataset: https://doi.org/10.5281/zenodo.15700880
- authors' code: https://github.com/HydroPML/UrbanFloodCast
- associated article: https://doi.org/10.1016/j.jhydrol.2025.133705

Download the files directly from the provider and retain their original split.
This repository does not redistribute the data and does not automate access to
the evaluation split.

## Public NPZ interface

The release scripts expect one compressed NPZ file per event:

```text
prepared/berlin-i/
├── train/*.npz
├── validation/*.npz
├── calibration/*.npz
├── test/*.npz
└── normalization.json
```

The paper's role contract is recorded in
`configs/evaluation/berlin_i_roles.yaml`: 85 training events, 15 development
events, 13 calibration events, and 12 held-out evaluation events. Keep these
roles disjoint. The checked-in contract expands 125 role-specific neutral
aliases and rejects an alias or directory assigned to more than one role. The
expanded rows are published in
`configs/evaluation/berlin_i_role_membership.csv`. The
evidence pipeline additionally rejects repeated observed IDs and emits only
public aliases plus namespaced SHA-256 digests. It does not redistribute
provider files or identifiers.

After preparing the NPZ directories, verify the full identity contract with:

```bash
python scripts/validate_public_contract.py \
  --data-root /path/to/prepared/berlin-i
```

The optional check opens only the scalar `event_id` metadata member, validates
the 85/15/13/12 counts, and rejects duplicates within or across all four roles.
It does not read `input`, `target`, or `mask` arrays and does not print provider
event IDs.

Each NPZ file contains:

| key | dtype | shape | meaning |
|---|---|---|---|
| `input` | float32 | `[31,H,W]` | H0, U0, V0, DEM, roughness, grid Y/X, and 24 rainfall channels |
| `target` | float32 | `[72,H,W]` | 24 lead times × H/U/V, interleaved by lead |
| `mask` | bool | `[H,W]` | valid spatial cells |
| `event_id` | string | scalar | event identifier |

The arrays supplied to training are normalized. `normalization.json` stores the
training-only output statistics used to convert predictions and targets back to
physical units during evaluation:

```json
{
  "mean": {"H": 0.0, "U": 0.0, "V": 0.0},
  "std": {"H": 1.0, "U": 1.0, "V": 1.0}
}
```

Replace the example values with statistics computed exclusively from the
training split. Do not recompute them from validation or evaluation events.
The development split fits the node scale, while the calibration split fits
normalized residual quantiles. Neither role may be replaced by the held-out
evaluation split.

## Converting existing arrays

If the provider files have already been converted to the common input and
target tensors, use `scripts/prepare_event_npz.py`:

```bash
python scripts/prepare_event_npz.py \
  --input arrays/input_event_001.npy \
  --target arrays/target_event_001.npy \
  --mask arrays/valid_mask.npy \
  --event-id event_001 \
  --output prepared/berlin-i/train/event_001.npz
```

The utility validates shapes and finite values but does not invent split or
normalization choices. The UrbanFloodCast repository also provides TIFF-to-PT
conversion tools for its native release format.
