# Level-B BEC Validation — Guo et al. 2021

## What this is

A phenomenological quartic-curvature test against real, independently
measured Bose-Einstein condensate collective-mode dispersion data.

**This is explicitly a Level-B / negative-result control module. It is
not a Λ-model validation.**

## Data

- Source: Guo, Y., Kroeze, R. M., Marsh, B. P., Gopalakrishnan, S.,
  Keeling, J. & Lev, B. L., "An optical lattice with sound," Nature 599,
  211 (2021). arXiv:2104.13922.
- Dataset: Harvard Dataverse, DOI:10.7910/DVN/LGT5O6, CC0 1.0 (public
  domain).
- File used: `Fig4_dispersion.tab` (above-threshold Goldstone/phonon
  dispersion of density-wave polaritons in a BEC-confocal-cavity system,
  pump strength η²/η²_th = 1.25). Reproduced here as `Fig4_dispersion.csv`.

**Physical caveat.** This is a cavity-mediated density-wave polariton
mode, not a free homogeneous-BEC phonon. It has its own characteristic
momentum scale ζ = 1/ξ (ξ ≈ 5 μm, set by the confocal cavity's mode
structure), and the source paper's own theory predicts flattening toward
a Debye-like frequency at large k_⊥ — a mechanism structurally different
from a Λk⁴ correction. Any curvature found here cannot be attributed to
the Λ-model without further justification. This module tests only
whether the data show detectable curvature of the tested functional form,
nothing more.

## Method

Six (k, ω, σ_ω) points, used exactly as archived — no re-binning,
re-weighting, or exclusion. Two nested models fit directly to ω(k) (not
ω², to avoid distorting the reported Gaussian uncertainties under a
nonlinear transform):

```
M0: ω = a k                     (pure linear baseline)
M1: ω = a k + b k³               (equivalent, to leading order, to
                                   ω² = A k² + B k⁴)
```

See `fit_quartic.py` for the exact implementation (weighted χ², AIC, BIC,
F-test, k=0 sensitivity check).

## Results (summary — full numbers in results.csv and provenance.md)

| Quantity | M0 (ω=ak) | M1 (ω=ak+bk³) |
|---|---:|---:|
| a | 202.79 ± 7.85 | 232.22 ± 20.04 |
| b | — | (−3.54 ± 2.22) × 10⁵ |
| χ² (dof) | 29.13 (5) | 26.59 (4) |
| reduced χ² | 5.83 | 6.65 |
| AIC | 31.13 | 30.59 |
| BIC | 30.93 | 30.17 |

- **b significance: −1.60σ** (below conventional 2σ threshold)
- **ΔAIC (M1−M0) = −0.55, ΔBIC (M1−M0) = −0.76** (both below the
  conventional |Δ|>2 "notable preference" threshold)
- **F-test: F=0.38, p≈0.57** (does not reject M0)
- **Result robust to removal of k=0** (identical qualitative conclusion)
- **χ²_ν(M0) ≈ 5.8** — the reported uncertainties do not fully account
  for the point-to-point scatter; standardized residuals are non-monotonic
  in k, consistent with scatter rather than coherent curvature

## Bottom line

**No statistically significant quartic curvature is detected in this
dataset.** This is a genuine null result on real, independent data,
recorded here on the same footing as positive results elsewhere in
Paper 3 — not omitted because it does not support the Λ-model.

**This is a small-sample (N=6) exploratory test.** Neither a positive
nor a negative outcome from six points should be over-interpreted; the
result is reported as exactly what it is — a completed Level-B check,
not evidence for or against the Λ-model.

## Files in this module

- `README.md` — this file
- `Fig4_dispersion.csv` — the six (k, ω, σ_ω) data points, as archived
- `fit_quartic.py` — the analysis script (reproducible, no external
  dependencies beyond numpy/scipy)
- `results.csv` — full numerical output of both fits
- `provenance.md` — search context (why no Level-A dataset was used),
  full interpretation, and explicit claim boundaries

## Relationship to the rest of Paper 3

This module is independent of, and does not affect, the giant-vortex
q=1/q=2 experimental validation (`PAPER3_ADDENDUM_BLIND_2D.md`) or the
main validation status (`PAPER3_VALIDATION_STATUS.md`). It is retained
as a separate, explicitly-scoped negative-result control.
