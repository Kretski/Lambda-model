# Provenance — Level-B BEC Validation (Guo et al. 2021)

## Search context: why no Level-A dataset was used

Before this Level-B test was carried out, a targeted search was made for a
genuine "Level-A" dataset: real atomic BEC + independently measured
excitation frequency ω(k) + broad wavevector coverage (multiple k, not a
handful of fixed points) + reported uncertainties + machine-readable data.

### Datasets checked and rejected

1. **Lopes, Eigen et al., "Quasiparticle energy in a strongly interacting
   homogeneous Bose-Einstein condensate," PRL 118, 210401 (2017).**
   Cambridge University Repository, DOI:10.17863/CAM.9210, CC BY 4.0.
   Inspected directly (raw Bragg resonance files, `Fig3a`/`Fig3b`/`Fig4a`/
   `Fig4b` tables). Measures frequency shift as a function of scattering
   length (interaction strength) at only three fixed recoil momenta
   (q1=1.1, q2=1.7, q3=2.0 k_rec). Not a k-sweep at fixed interaction
   strength. Rejected as unsuitable for a dispersion-curve fit.

2. **Lopes, Eigen, Navon et al., "Quantum Depletion of a Homogeneous
   Bose-Einstein Condensate," PRL 119, 190404 (2017).** Cambridge
   University Repository, DOI:10.17863/CAM.13808, CC BY 4.0. Inspected
   directly. Measures diffracted fraction (Rabi oscillations) as a
   function of √n·a³ (interaction parameter), not ω(k) at all. Rejected.

3. **Ozeri, Steinhauer, Katz & Davidson, "Direct Observation of the Phonon
   Energy in a Bose-Einstein Condensate by Tomographic Imaging," PRL 88,
   220401 (2002).** A genuine ω(k) measurement via time-of-flight
   tomography, in good agreement with the Bogoliubov spectrum. Predates
   open-data norms (2002); no public repository, arXiv supplement, or raw
   numerical table found. Full text itself is paywalled.

4. **Shammass, Rinott, Berkovitz, Schley & Steinhauer, "Phonon Dispersion
   Relation of an Atomic Bose-Einstein Condensate," PRL 109, 195301
   (2012), arXiv:1207.3440.** Conceptually the closest match to the
   desired Level-A test: short Bragg pulses, in-situ standing-wave
   oscillations directly giving ω(k) over a range spanning the 3D-to-1D
   crossover. Searched directly on the Steinhauer Atomic Physics
   Laboratory (Technion) publications page and "Phonons and Correlations"
   research page — the paper is listed but no supplementary data,
   repository link, or raw table is provided (paper predates 2012-era
   open-data mandates). No public dataset located.

### Level-B candidate selected

**Guo, Kroeze, Marsh, Gopalakrishnan, Keeling & Lev, "An optical lattice
with sound," Nature 599, 211 (2021), arXiv:2104.13922.** Harvard
Dataverse, DOI:10.7910/DVN/LGT5O6, CC0 1.0 (public domain), confirmed as
the dataset explicitly cited in the paper's Data Availability statement.
41 files inspected; `Fig4_dispersion.tab` (3 variables, matching k, ω, σ)
identified as the genuine multi-k dispersion measurement with reported
uncertainties. Selected as the Level-B dataset precisely because it *is*
real, independent, multi-k, and has uncertainties — with the explicit
caveat (below) that its underlying physics differs from the Λ-model's
intended domain.

## Physical caveat (why this is Level-B, not Level-A-equivalent)

The measured quantity in `Fig4_dispersion.tab` is the above-threshold
Goldstone (phonon) mode of a density-wave polariton condensate formed by a
BEC coupled to a confocal optical cavity under a double-pumping scheme
(Guo et al., main text and Supplementary Information §VIII). This is not
a free homogeneous-BEC phonon. Key structural differences:

- The dispersion has its own characteristic momentum scale
  ζ ≡ 1/ξ ≈ 0.02 k_r, set by the cavity-mediated interaction range
  ξ ≈ 5 μm (a photon-mediated, not contact, interaction).
- The full theoretical dispersion (Guo et al. Eq. S57-S60, Fig. S3) is
  *not* a simple ω² = A k² + B k⁴ polynomial; it involves the full
  cavity-response kernel χ(ν,k) and is predicted to flatten toward a
  Debye-like frequency (~5 kHz) at large k_⊥, a saturation effect from
  the finite interaction range — mechanistically distinct from a
  quartic Λ-correction to vacuum/free-particle dispersion.
- At the small-k regime tested here (k_⊥/k_r ≤ 0.0106, well below
  ζ ≈ 0.02 k_r), the theoretical prediction is that the dispersion is
  linear, ω(k) = v_s|k| — i.e., the source paper's own theory predicts
  M0, not M1, in this regime, with any deviation attributable to the
  cavity length scale rather than a universal quartic term.

**Consequence:** a significant b in M1 (had one been found) could not
honestly have been reported as evidence for the Λ-model's Λk⁴ term
without a separate derivation connecting the two mechanisms — exactly
the kind of unjustified cross-mechanism claim this project's methodology
is designed to avoid (cf. the Λ_NH vs Λ_GW dimensional-incompatibility
discussion in `PAPER3_VALIDATION_STATUS.md`). Conversely, the null result
obtained here is a clean statement about this specific dataset and
nothing more — it neither supports nor undermines the Λ-model, because
the model being tested (M1) was never claimed to be the correct
description of this system in the first place.

## Full statistical results

See `results.csv` for the complete machine-readable output. Summary:

### Primary fit (N=6, all points)

| | M0: ω=ak | M1: ω=ak+bk³ |
|---|---:|---:|
| a | 202.7854 ± 7.8452 | 232.2196 ± 20.0410 |
| b | — | (−3.5365 ± 2.2157) × 10⁵ |
| χ² | 29.1333 | 26.5858 |
| dof | 5 | 4 |
| reduced χ² | 5.8267 | 6.6465 |
| AIC | 31.1333 | 30.5858 |
| BIC | 30.9250 | 30.1694 |

- ΔAIC (M1−M0) = −0.5474
- ΔBIC (M1−M0) = −0.7557
- F-test: F=0.3833, p=0.5694
- b significance: −1.596σ

### Sensitivity check (N=5, k=0 excluded)

| | M0: ω=ak | M1: ω=ak+bk³ |
|---|---:|---:|
| a | 202.7854 ± 7.8452 | 232.2196 ± 20.0410 |
| b | — | (−3.5365 ± 2.2157) × 10⁵ |
| χ² | 29.0782 | 26.5307 |
| dof | 4 | 3 |
| reduced χ² | 7.2695 | 8.8436 |
| AIC | 31.0782 | 30.5307 |
| BIC | 30.6876 | 29.7496 |

- ΔAIC (M1−M0) = −0.5474
- ΔBIC (M1−M0) = −0.9380
- F-test: F=0.2881, p=0.6287
- b significance: −1.596σ (unchanged)

Point parameter estimates (a, b) are identical with and without the k=0
point because that point carries very little statistical weight (its
central value is close to the M0 prediction of 0 at k=0, and its
uncertainty is comparable to the others); the fit is driven by the same
five nonzero-k points in both cases.

## Interpretation — explicit claim boundaries

**What this result establishes:**

- No statistically significant curvature (of the specific ω=ak+bk³ form
  tested) is detected in this real, independently-measured dataset.
- The result is stable under a reasonable sensitivity check.
- The reported uncertainties in the source dataset do not fully account
  for the point-to-point scatter observed (reduced χ² ≈ 5.8–8.8 for the
  baseline model, well above 1), which limits the statistical power of
  this test in either direction and should be kept in mind if this
  dataset is revisited.

**What this result does NOT establish:**

- It is not a test of the Λ-model, because the tested system's own theory
  does not predict a Λk⁴-type correction in this regime (see Physical
  caveat above) — a null result here is not evidence against Λ, and a
  positive result would not have been evidence for it.
- It is not a general statement about quartic dispersion corrections in
  nature — this is one dataset, N=6, from one specific engineered
  cavity-QED system.
- It is not a substitute for a genuine homogeneous-BEC Level-A test,
  which remains unavailable (see Search context above).

## Reproducibility

Run `python3 fit_quartic.py` in this directory (requires numpy, scipy).
The script reads the six data points directly (hard-coded from
`Fig4_dispersion.csv`, reproduced exactly as archived) and regenerates
`results.csv` and the console output shown above. No external data
fetching, network access, or manual steps are required.

## Relationship to other Paper 3 documents

This module is a self-contained negative-result control. It supplements,
and is independent of:

- `paper3/PAPER3_VALIDATION_STATUS.md` — main validation status (this
  module is not currently referenced there as a required result; it may
  be linked as an optional cross-reference)
- `paper3/gw/matched_filter/_archive/qnm_branch_work/PAPER3_ADDENDUM_BLIND_2D.md`
  — the giant-vortex q=1/q2 experimental validation (a positive result,
  on an entirely different physical system, with a different and much
  stronger evidentiary status — see that document for details)

No claim in this module should be combined with, or used to qualify,
the claims made in either of those documents.
