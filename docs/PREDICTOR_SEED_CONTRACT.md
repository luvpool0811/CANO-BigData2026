# Predictor and random-seed contract

The manuscript describes ensembles and initialization-robustness analyses at
the scientific-method level. Exact random-seed values and predictor identities
are centralized here so that they remain reproducible without interrupting the
paper's narrative.

## Operational analyses

| Evidence block | Predictor | Primary training seed | Additional seeds | Role |
|---|---|---:|---|---|
| UFB California/Tennessee | UFB coordinate-query predictor | 42 within each development fold | none | five-fold development evidence |
| UFC Berlin I RQ1--RQ4 | Protocol-aligned DNO | 42 | 2026, 202607 | seed 42 defines the prespecified operational archive; the two additional independently trained predictors form the prospectively locked initialization-robustness extension |
| WB2 | fixed Pangu-Weather forecasts | not applicable | not applicable | cross-domain evaluation; no model training is performed in this repository |

The UFC operational predictor is a study-owned DNO implementation trained from
scratch to enforce the physical-input, provider-role split, train-only
normalization, checkpoint-selection, and forcing-ablation contracts. Its
forcing-aware and forcing-agnostic variants have the same architecture,
capacity, optimizer, data roles, and training seed; only the rainfall-forcing
information changes.

## Controlled predictor comparison (Table II)

CANO, DNO-3, FNO3D, and U-Net3D use training seeds **7, 31, and 42**. For each
system, the three physical-unit H fields are averaged cellwise before
development-only node-scale fitting, calibration-only target-aligned fitting,
and event-level evaluation. The complete contract is encoded in the model YAML
files and `results/paper/baseline_fairness.csv`.

DNO-3 is not the UFC operational DNO. DNO-3 uses the DNO architecture class
from the pinned UrbanFloodCast public revision and is retrained through the
common Table II interface. The protocol-aligned UFC DNO is the fixed predictor
used for the benchmark-native operational questions. Their results answer
different questions and are not pooled.

## Selection and resampling seeds

- Table II hyperparameter selection compares all candidates with training seed
  **42** on development events only; the selected configuration is then trained
  with seeds **7, 31, and 42**.
- The six CANO--baseline paired-event bootstrap intervals use random seed
  **20260731** and 5,000 resamples.
- The central UFC RQ1 paired-event bootstrap uses random seed **20260731** and
  2,000 resamples, conditional on the fixed primary predictor and calibration
  archive.
- Synthetic quick-start examples default to seed **42**; these are smoke tests,
  not paper-scale measurements.

Random seeds control software initialization and resampling only. The
independent statistical units are rainfall events (or forecast issuance days
for WB2), never neural-network initializations or individual grid cells.
