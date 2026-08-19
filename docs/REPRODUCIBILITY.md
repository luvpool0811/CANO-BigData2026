# Reproducibility guide

## Levels of reproduction

1. **Immediate smoke test (CPU, under a minute)**

   ```bash
   python scripts/quickstart.py --device cpu
   ```

2. **Rebuild reported tables and plots (CPU, seconds)**

   ```bash
   python scripts/reproduce_results.py
   ```

3. **Train one CANO seed (GPU, prepared UrbanFloodCast data required)**

   ```bash
   python scripts/train.py --config configs/cano/standard.yaml \
     --data-root /path/to/prepared/berlin-i \
     --output-dir outputs/cano/seed-42 --seed 42 --device cuda
   ```

4. **Train an external model under the common contract**

   Prepare the specified UrbanFloodCast checkout and pass it with
   `--upstream-source`. See `BASELINE_ADAPTATION.md`.

## What the checked-in results represent

`results/paper/main_results.csv` contains the event-macro values reported in
the paper. `event_level_results.csv` exposes 12 anonymized event records for
the standard-objective comparison. The reproduction script reads only these
small CSV files. It does not claim to retrain models or recalculate metrics from
bulk arrays.

## Operational-target calibration

`fit_node_scale` computes a lead-by-node RMS residual scale from development
events. `fit_target_aligned_calibrator` then fits event-balanced, normalized
residual thresholds on a separate calibration split. The operational cell
population is selected from the prediction and a 0.30 m threshold; the unknown
target does not define membership. `evaluate_event` preserves one evidence
record per physical event before event-macro aggregation.

## Determinism

Training seeds are explicit in every command and configuration. CUDA kernels
can still exhibit platform-dependent numerical differences. Record the GPU,
CUDA, PyTorch, and driver versions with any independently reproduced result.

## Scope

The public comparison supports the reported BerlinI event set and named model
configurations. Broader spatial generalization and the peak-aware CANO objective
should be evaluated on independent events and additional urban domains.
