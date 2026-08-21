| Claim group | Public evidence | Level | Statistics recomputed | External inputs required | Scope |
|---|---|---|---|---|---|
| standard_point_metrics | event_level_results.csv | event-summary recomputation | true | false | seven non-CSI metrics (H-RMSE, NSE, wet RMSE, wet NSE, peak error, ACE, and WIS) are recomputed for four standard systems from 48 public event rows |
| standard_CSI | main_results.csv | aggregate regeneration | false | false | four CSI thresholds are validated and regenerated from reported aggregate rows; event-level CSI is not publicly recomputed |
| paired_baseline_inference | event_level_results.csv | paired-event recomputation | true | false | six relative effects, bootstrap intervals, exact sign-flip tests, and Holm adjustments are recomputed from paired public event rows |
| operational_contrasts | operational_contrasts.csv | archived-summary regeneration | false | false | Fig. 2-style population and calibration-axis effects are regenerated from archived aggregate reporting summaries |
| target_calibration | target_calibration.csv | archived-summary regeneration | false | false | Table I-style calibration profiles are regenerated from archived aggregate reporting summaries |
| crc_finite_sample | crc_calibration.csv | archived-summary regeneration | false | false | Table I Panel B CRC sensitivity is regenerated from two archived aggregate reporting summaries; field-array bootstrap and calibrator refits are not rerun |
| deployment_boundaries | deployment_budget_effects.csv and warning_rule_migration.csv | archived-summary regeneration | false | false | Fig. 3 budget effects and warning-rule rankings are regenerated from archived aggregate reporting summaries |
| paper_scale_training | training and workflow code | code path only | false | true | provider data and three trained checkpoints are required |
| peak_aware_followup | main_results.csv and peak_aware_followup.yaml | aggregate regeneration | false | true | the compact public trainer does not reproduce the post-evaluation objective run |
