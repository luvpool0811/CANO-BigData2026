# Prespecified paired comparisons

Relative effects are CANO/reference - 1; negative values favor CANO.
Confidence intervals use a 5,000-replicate paired-event bootstrap
with seed 20260731. Two-sided exact paired sign-flip p-values are
Holm-adjusted separately within the three H-RMSE and three Target-WIS comparisons.

| Endpoint | Comparator | Relative effect | 95% CI | Raw p | Holm p | Support |
|---|---|---:|---:|---:|---:|---|
| H RMSE | DNO-3 | -32.31% | [-37.50%, -27.01%] | 0.00048828 | 0.00146484 | yes |
| H RMSE | FNO3D | -33.66% | [-38.56%, -28.78%] | 0.00048828 | 0.00146484 | yes |
| H RMSE | U-Net3D | -48.78% | [-56.79%, -40.39%] | 0.00048828 | 0.00146484 | yes |
| Target WIS | DNO-3 | -44.97% | [-55.01%, -34.67%] | 0.00048828 | 0.00146484 | yes |
| Target WIS | FNO3D | -27.54% | [-39.88%, -15.38%] | 0.00146484 | 0.00146484 | yes |
| Target WIS | U-Net3D | -66.06% | [-76.18%, -56.33%] | 0.00048828 | 0.00146484 | yes |
