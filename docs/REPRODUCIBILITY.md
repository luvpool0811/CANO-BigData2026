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

3. **Reproduce the six paired-event comparisons (CPU, seconds)**

   ```bash
   python scripts/reproduce_inference.py
   ```

4. **Rebuild central operational/statistical disclosures (CPU, seconds)**

   ```bash
   python scripts/reproduce_operational_evidence.py
   ```

5. **Train one CANO seed (GPU, prepared UrbanFloodCast data required)**

   ```bash
   python scripts/train.py --config configs/cano/standard.yaml \
     --data-root /path/to/prepared/berlin-i \
     --output-dir outputs/cano/seed-42 --seed 42 --device cuda
   ```

6. **Train an external model under the common contract**

   Prepare the specified UrbanFloodCast checkout and pass it with
   `--upstream-source`. See `BASELINE_ADAPTATION.md`.

## What the checked-in results represent

`results/paper/main_results.csv` contains the event-macro values reported in
the paper. `event_level_results.csv` exposes 12 anonymized event records for
the standard-objective comparison. The reproduction script reads only these
small CSV files. It does not claim to retrain models or recalculate metrics from
bulk arrays.

`operational_contrasts.csv` and `target_calibration.csv` are the compact source
tables for the paper's central Fig. 2/Table I evidence. They contain only
reporting-unit summaries and no field arrays. `claim_evidence.csv` and
`baseline_fairness.csv` expose the statistical and training-design disclosures
that are easy to miss in the page-limited manuscript.

`operational_evaluation_specification.csv` closes the central Berlin I RQ1
contract in one machine-readable row. It identifies the fixed seed-42
forcing-aware DNO prediction archive, full and prediction-selected
populations, global absolute-residual interval construction, provider
Valid/Test roles, event-macro estimand, paired bootstrap, and claim boundary.
The validator binds its numerical fields to `operational_contrasts.csv` and its
resampling/calibrator fields to `claim_evidence.csv`.

`reproduce_inference.py` uses the checked-in event rows to reproduce the six
prespecified CANO-versus-Original comparisons. It reports the paired relative
effect, a 5,000-resample event bootstrap interval (seed 20260731), the exact
two-sided sign-flip p-value, and Holm adjustment separately within the three
H-RMSE and three target-WIS families.

## Claim-level reproduction matrix

The machine-readable matrix is `results/paper/reproducibility_scope.csv` and
its rendered form is `results/generated/reproducibility_scope.md`.

| Level | Meaning |
|---|---|
| Event-summary recomputation | Metric means are recomputed from public event rows. |
| Paired-event recomputation | Effects, intervals, tests, and multiplicity adjustment are recomputed from public event rows. |
| Summary regeneration | A checked reporting-unit summary is validated and rendered; its bootstrap is not rerun. |
| Aggregate regeneration | A reported aggregate row is validated and rendered. |
| Code path only | Provider data or frozen checkpoints are required for paper-scale execution. |

The six paired CANO--baseline contrasts are statistic recomputations. The
central UFB/UFC/WB2 operational contrasts and target-calibration profiles are
summary regenerations. The repository does not imply that every paper
statistic is recomputed from unavailable field arrays.

## Checkpoint and HPO selection

Training optimizes each architecture's native normalized MSE. Checkpoints are
selected separately using `development_event_macro_physical_h_rmse`: every
development event, all 24 leads, and every valid cell are evaluated in physical
metres before event H-RMSE values are averaged. The metric and score are stored
in `best.pt` and `training_summary.json`.

`results/paper/hpo_candidates.csv` publishes all six seed-42 development-only
candidates per system, including configuration, best physical H-RMSE, best
epoch, parameter count, and selected flag. The selected row minimizes the score
with lower candidate index as the tie-break.

## Operational-target calibration

`fit_node_scale` computes a lead-by-node RMS residual scale from development
events. `fit_target_aligned_calibrator` then fits event-balanced, normalized
residual thresholds on a separate calibration split. The operational cell
population is selected from the prediction and a 0.30 m threshold; the unknown
target does not define membership. This operational threshold is distinct from
the model-independent truth-wet mask, which uses $H\geq0.01$ m for wet RMSE and
wet NSE. `evaluate_event` preserves one evidence record per physical event
before event-macro aggregation.

## End-to-end evidence workflow

`scripts/run_evidence_pipeline.py` connects exactly three frozen seed
checkpoints to physical-field ensemble averaging, development-only node-scale
fitting, calibration-only target-aligned fitting, and one event record per
held-out evaluation event. The split counts and seed contract are declared in
`configs/evaluation/berlin_i_roles.yaml`. The unit suite runs this complete
chain on small synthetic fields; a paper-scale run requires the provider data
and trained checkpoints. It validates pairwise-disjoint public aliases before
data access, rejects observed ID reuse within/across development, calibration,
and evaluation roles, and reports public aliases plus namespaced event-ID
digests.

`configs/evaluation/berlin_i_role_membership.csv` expands the 125-row public
role contract and binds every alias to the provider split, event name, and
archive-relative directory. Run `scripts/validate_public_contract.py
--data-root ...` to check exact provider membership, the prepared 85/15/13/12
counts, and event-ID disjointness across all four roles. It accesses only
scalar identity metadata. With `--provider-archive ...`, it independently
matches all 125 directories from the ZIP central directory without opening a
payload member.

## Determinism

Training seeds are explicit in every command and configuration. CUDA kernels
can still exhibit platform-dependent numerical differences. Record the GPU,
CUDA, PyTorch, and driver versions with any independently reproduced result.

## Scope

The public comparison supports the reported BerlinI event set and named model
configurations. The peak-aware aggregate/config scope is public, but the compact
training CLI reproduces the standard objective only. Broader spatial
generalization and the peak-aware CANO objective should be evaluated on
independent events and additional urban domains.
