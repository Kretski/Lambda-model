# Level-A BEC Experimental Consistency Check — Ozeri et al. 2002

## What this is

An experimental consistency check of the Λ-model's BEC-sector prediction(Λ = ξ²/4 in the Paper-3 healing-length convention) against real,directly-measured atomic-BEC excitation-energy data.

**This is explicitly a consistency check, not a confirmation ordetection.** See the claim boundary at the end of this README.

## Data source

* Ozeri, R., Steinhauer, J., Katz, N. & Davidson, N., "Direct observationof the phonon energy in a Bose-Einstein condensate by tomographicimaging," Phys. Rev. Lett. 88, 220401 (2002). arXiv:cond-mat/0112496.
* Real atomic ⁸⁷Rb BEC (~10⁵ atoms), excitation energy measured directlyfrom time-of-flight images via computerized tomography, at fourdifferent momenta (Fig. 5, "released-phonon cloud only" measurements).
* Data are **digitized** from the published figure (see Provenance below)— not machine-readable raw data (none is publicly available; see`provenance.md` for the search record).

## Physical caveat

The measured BEC is **trapped and inhomogeneous**; the authors analyzeit under the local-density approximation (LDA), where the measuredexcitation energy is a density-weighted average over the condensate(Eq. 1 of the source paper), not a strict homogeneous-system dispersionrelation. This is a real, if secondary, caveat relative to an idealizedhomogeneous-BEC test — but it is a far closer physical match to theΛ-model's intended BEC domain than the cavity-mediated system used inthe companion Level-B module (`../level_b_guo2021/`).

## Convention audit (see `convention_audit.md` for full derivation)

Paper 3 (`example_BEC.py`) and Ozeri (2002) define the BEC healing lengthdifferently (ξ_P = ħ/(mc_s) vs ξ_O = √(ħ²/2mgn)). After conversion:

    Λ_Paper3 = ξ_P²/4 = ξ_O²/2 = Λ_pred (same physical value, both conventions)

This is verified by an independent re-derivation directly from theOzeri paper's own Bogoliubov formula (not assumed), and enforced in codeby a runtime assertion in `fit_bogoliubov_lambda.py`.

## Method

Unified model (x ≡ k·ξ_O, the dimensionless variable Ozeri's Fig. 5 uses):

    E/h = A · x · sqrt(1 + β·x²)
    
    β = 0    → M0:   pure linear (free-phonon) dispersion
    β = 0.5  → M_B:  Bogoliubov dispersion = Λ-model prediction (pre-registered,
                      fixed BEFORE fitting — not chosen after seeing the data)
    β free   → M_Λf: exploratory 2-parameter fit

Two analyses:

* **Primary**: all N=4 digitized points
* **Sensitivity**: N=3, excluding k·ξ_O=0.324, which the *original authors*flag as affected by a systematic low-k recoil effect (not a post-hocexclusion chosen by this analysis)

## Results (summary — full numbers in `results.csv`)

### Primary (N=4)

| Model | reduced χ² | AIC | β_fit |
| --- | --- | --- | --- |
| M0 (linear) | 11.93 | 37.78 | —   |
| M_B (Bogoliubov, fixed) | 7.87 | 25.61 | 0.5 (fixed) |
| M_Λf (free) | 5.31 | 14.63 | 5.56 ± 3.89 |

**Inconclusive.** All models fit poorly; the apparent preference forcurvature is dominated by the k·ξ_O=0.324 point (standardized residualunder M0: −5.26σ), which the source paper itself attributes to asystematic effect.

### Sensitivity (N=3, k·ξ_O=0.324 excluded)

| Model | reduced χ² | AIC | β_fit |
| --- | --- | --- | --- |
| M0 (linear) | 1.97 | 5.94 | —   |
| M_B (Bogoliubov, fixed) | **0.24** | **2.47** | 0.5 (fixed) |
| M_Λf (free) | 0.02 | 4.02 | 0.90 ± 0.71 |

* β_fit = 0.90 ± 0.71 vs. β_pred = 0.5: **0.57σ difference** — data donot show a measurable deviation from the pre-registered prediction.
* AIC/BIC **prefer the fixed-prediction model M_B** over the free-β fit(ΔAIC(M_Λf − M_B) = +1.55), i.e. the extra free parameter is notjustified by these three points.

## Bottom line — explicit claim boundary

> **Primary (N=4) analysis is inconclusive**, because the apparentcurvature preference is dominated by a single point for which theoriginal authors report a documented systematic effect.
> 
> **Sensitivity (N=3) analysis finds the data consistent with thepre-registered Λ-model/Bogoliubov prediction** (β_fit differs fromβ_pred by <0.6σ), and model selection favors the fixed prediction overa free fit. Given the very small sample (N=3, 1 degree of freedom forthe free-β model), **this is an experimental consistency check, not aconfirmation or detection of Λ = ξ²/4.**

This should NOT be described, here or elsewhere in Paper 3, as:

* confirmation of Λ = ξ²/4 as a universal or even BEC-sector-establishedconstant;
* a statistically significant detection of quartic dispersion;
* a result independent of the LDA/trapped-BEC and digitization caveatsdocumented in `provenance.md`.

It MAY be described as:

* an experimental consistency check, using real, independently-measuredatomic-BEC data, that does not exclude the Λ-model's specific,pre-registered BEC-sector prediction.

## Relationship to the Level-B module

This module (Level-A) is retained **alongside**, not instead of,`../level_b_guo2021/` (Level-B). They test different things on differentsystems:

* **Level-B (Guo 2021)**: phenomenological quartic-curvature control oncavity-mediated collective modes — a system the Λ-model was neverclaimed to describe. Result: null (no curvature detected).
* **Level-A (Ozeri 2002)**: physics-specific consistency check on realatomic-BEC excitation energies, against the pre-registered Bogoliubov/Λprediction. Result: not excluded (sensitivity analysis), inconclusive(primary analysis).

## Files in this module

* `README.md` — this file
* `digitized_fig5.csv` — the four (k·ξ_O, E/h, σ) points, with provenancenotes
* `convention_audit.md` — full ξ_P vs ξ_O derivation and equivalence proof
* `fit_bogoliubov_lambda.py` — the analysis script (reproducible; includesa runtime assertion enforcing the convention audit)
* `results.csv` — full numerical output (primary + sensitivity, allthree models)
* `provenance.md` — digitization methodology (pixel calibration, blobdetection, cross-check against the text-quoted value), search recordfor a genuine machine-readable Level-A dataset, and full claim-boundarydiscussion
