# Level-A BEC Experimental Consistency Check — Ozeri et al. 2002

## What this is

An experimental consistency check of the Λ-model's BEC-sector prediction
(Λ = ξ²/4 in the Paper-3 healing-length convention) against real,
directly-measured atomic-BEC excitation-energy data.

**This is explicitly a consistency check, not a confirmation or
detection.** See the claim boundary at the end of this README.

## Data source

- Ozeri, R., Steinhauer, J., Katz, N. & Davidson, N., "Direct observation
  of the phonon energy in a Bose-Einstein condensate by tomographic
  imaging," Phys. Rev. Lett. 88, 220401 (2002). arXiv:cond-mat/0112496.
- Real atomic ⁸⁷Rb BEC (~10⁵ atoms), excitation energy measured directly
  from time-of-flight images via computerized tomography, at four
  different momenta (Fig. 5, "released-phonon cloud only" measurements).
- Data are **digitized** from the published figure (see Provenance below)
  — not machine-readable raw data (none is publicly available; see
  `provenance.md` for the search record).

## Physical caveat

The measured BEC is **trapped and inhomogeneous**; the authors analyze
it under the local-density approximation (LDA), where the measured
excitation energy is a density-weighted average over the condensate
(Eq. 1 of the source paper), not a strict homogeneous-system dispersion
relation. This is a real, if secondary, caveat relative to an idealized
homogeneous-BEC test — but it is a far closer physical match to the
Λ-model's intended BEC domain than the cavity-mediated system used in
the companion Level-B module (`../level_b_guo2021/`).

## Convention audit (see `convention_audit.md` for full derivation)

Paper 3 (`example_BEC.py`) and Ozeri (2002) define the BEC healing length
differently (ξ_P = ħ/(mc_s) vs ξ_O = √(ħ²/2mgn)). After conversion:

```
Λ_Paper3 = ξ_P²/4 = ξ_O²/2 = Λ_pred (same physical value, both conventions)
```

This is verified by an independent re-derivation directly from the
Ozeri paper's own Bogoliubov formula (not assumed), and enforced in code
by a runtime assertion in `fit_bogoliubov_lambda.py`.

## Method

Unified model (x ≡ k·ξ_O, the dimensionless variable Ozeri's Fig. 5 uses):

```
E/h = A · x · sqrt(1 + β·x²)

β = 0    → M0:   pure linear (free-phonon) dispersion
β = 0.5  → M_B:  Bogoliubov dispersion = Λ-model prediction (pre-registered,
                  fixed BEFORE fitting — not chosen after seeing the data)
β free   → M_Λf: exploratory 2-parameter fit
```

Two analyses:
- **Primary**: all N=4 digitized points
- **Sensitivity**: N=3, excluding k·ξ_O=0.324, which the *original authors*
  flag as affected by a systematic low-k recoil effect (not a post-hoc
  exclusion chosen by this analysis)

## Results (summary — full numbers in `results.csv`)

### Primary (N=4)

| Model | reduced χ² | AIC | β_fit |
|---|---:|---:|---:|
| M0 (linear) | 11.93 | 37.78 | — |
| M_B (Bogoliubov, fixed) | 7.87 | 25.61 | 0.5 (fixed) |
| M_Λf (free) | 5.31 | 14.63 | 5.56 ± 3.89 |

**Inconclusive.** All models fit poorly; the apparent preference for
curvature is dominated by the k·ξ_O=0.324 point (standardized residual
under M0: −5.26σ), which the source paper itself attributes to a
systematic effect.

### Sensitivity (N=3, k·ξ_O=0.324 excluded)

| Model | reduced χ² | AIC | β_fit |
|---|---:|---:|---:|
| M0 (linear) | 1.97 | 5.94 | — |
| M_B (Bogoliubov, fixed) | **0.24** | **2.47** | 0.5 (fixed) |
| M_Λf (free) | 0.02 | 4.02 | 0.90 ± 0.71 |

- β_fit = 0.90 ± 0.71 vs. β_pred = 0.5: **0.57σ difference** — data do
  not show a measurable deviation from the pre-registered prediction.
- AIC/BIC **prefer the fixed-prediction model M_B** over the free-β fit
  (ΔAIC(M_Λf − M_B) = +1.55), i.e. the extra free parameter is not
  justified by these three points.

## Bottom line — explicit claim boundary

> **Primary (N=4) analysis is inconclusive**, because the apparent
> curvature preference is dominated by a single point for which the
> original authors report a documented systematic effect.
>
> **Sensitivity (N=3) analysis finds the data consistent with the
> pre-registered Λ-model/Bogoliubov prediction** (β_fit differs from
> β_pred by <0.6σ), and model selection favors the fixed prediction over
> a free fit. Given the very small sample (N=3, 1 degree of freedom for
> the free-β model), **this is an experimental consistency check, not a
> confirmation or detection of Λ = ξ²/4.**

This should NOT be described, here or elsewhere in Paper 3, as:
- confirmation of Λ = ξ²/4 as a universal or even BEC-sector-established
  constant;
- a statistically significant detection of quartic dispersion;
- a result independent of the LDA/trapped-BEC and digitization caveats
  documented in `provenance.md`.

It MAY be described as:
- an experimental consistency check, using real, independently-measured
  atomic-BEC data, that does not exclude the Λ-model's specific,
  pre-registered BEC-sector prediction.

## Relationship to the Level-B module

This module (Level-A) is retained **alongside**, not instead of,
`../level_b_guo2021/` (Level-B). They test different things on different
systems:

- **Level-B (Guo 2021)**: phenomenological quartic-curvature control on
  cavity-mediated collective modes — a system the Λ-model was never
  claimed to describe. Result: null (no curvature detected).
- **Level-A (Ozeri 2002)**: physics-specific consistency check on real
  atomic-BEC excitation energies, against the pre-registered Bogoliubov/Λ
  prediction. Result: not excluded (sensitivity analysis), inconclusive
  (primary analysis).

## Files in this module

- `README.md` — this file
- `digitized_fig5.csv` — the four (k·ξ_O, E/h, σ) points, with provenance
  notes
- `convention_audit.md` — full ξ_P vs ξ_O derivation and equivalence proof
- `fit_bogoliubov_lambda.py` — the analysis script (reproducible; includes
  a runtime assertion enforcing the convention audit)
- `results.csv` — full numerical output (primary + sensitivity, all
  three models)
- `provenance.md` — digitization methodology (pixel calibration, blob
  detection, cross-check against the text-quoted value), search record
  for a genuine machine-readable Level-A dataset, and full claim-boundary
  discussion
