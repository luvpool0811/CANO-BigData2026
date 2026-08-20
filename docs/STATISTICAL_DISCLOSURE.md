# Statistical and fairness disclosure

The public package separates the paper's primary operational-reliability
evidence from its supporting predictor comparison and its non-confirmatory
objective follow-up.

## Claim-to-evidence map

`results/paper/claim_evidence.csv` records, for each claim, the estimand,
independent unit, sample size, resampling method, calibrator treatment,
multiplicity family, prespecification status, and evidence boundary. Rebuild the
readable table with:

```bash
python scripts/reproduce_operational_evidence.py
```

For the six CANO--baseline comparisons, the exact sign-flip test assumes that
the paired event differences are independent across events and sign-symmetric
under the null. Holm adjustment is applied separately to the three H-RMSE
comparisons and the three target-aligned WIS comparisons. The WIS bootstrap
holds the development-fitted node scales and calibration-event quantiles fixed;
its intervals are therefore conditional on those fixed archives.

Target WIS compares complete operational systems. Each model defines its own
population from its own prediction and is evaluated with its own development-
fitted node scale and calibration quantile. It is therefore neither a common-
population contrast nor an architecture-only causal comparison.

For finite-sample CRC, the estimand is event-mean loss conditional on a
nonempty prediction-selected population. Empty events are recorded separately
and excluded symmetrically from calibration and evaluation. Each stored paired
bootstrap contract resamples calibration and evaluation events separately with
replacement, applies common indices across matched methods, and refits every
CRC and empirical calibrator before evaluating the resampled events. The public
package validates and renders this archived aggregate reporting summary; it does
not rerun the field-array bootstrap.

The Fig. 3(b) winner map minimizes the event-macro prevalence-weighted warning
loss $L_{\mathrm{prev},e}=(r\,\mathrm{FN}_e+\mathrm{FP}_e)/N_e$. A near tie is
a display annotation for a winner--runner-up gap at most $10^{-4}$; it is not
an equivalence test or an inferential policy comparison.

## Baseline setting and training exposure

`results/paper/baseline_fairness.csv` reports the candidate count, development-
only setting rule, selected configuration, seeds, event exposure, completed
epochs, optimizer steps, and measured training wall time for the reported
systems. The three external models reuse only the winning development-selected
configuration from six candidates per architecture; their weights are trained
from scratch for the controlled comparison. Evaluation events are not used for
setting or checkpoint selection.

The shared budget is six candidates, seeds 7/31/42, 100 epochs per seed, and
12,900 optimizer updates per reported system. This is an optimizer-level
contract, not an assertion of identical label exposure per update. CANO samples
one lead and up to 4,096 query coordinates per event/update; the grid baselines
use their native dense 24-lead H/U/V field supervision. The full candidate
ledger and development scores are in `results/paper/hpo_candidates.csv`.

The comparison is between complete systems under a common interface. It does
not isolate architecture as a causal factor, and the wall times are descriptive
measurements rather than a hardware-normalized efficiency claim.

## Objective-follow-up scope

`configs/cano/peak_aware_followup.yaml` discloses the reported CANO follow-up
objective and the exact public scope. The checked-in aggregate rows and plot are
reproducible, but the compact public training CLI implements the standard
objective only. The peak-aware result was motivated and evaluated on the same
12-event set, so it is a non-confirmatory follow-up rather than independent
evidence.
