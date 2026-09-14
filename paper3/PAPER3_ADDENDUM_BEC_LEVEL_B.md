# Paper 3 — BEC Collective-Mode Level-B Addendum

## Purpose

This addendum documents a phenomenological quartic-curvature test performed
against real, independently-measured Bose-Einstein condensate collective-mode
data, as an exploratory check of whether real dispersion data from an
unrelated physical system show any detectable curvature of the form probed
by the Λ-model's quartic dispersion term.

This is explicitly a **Level-B control test**, not a Λ-model validation. It
is recorded to document that a real independent dataset was checked and did
not show a supporting signal, rather than omitting a negative result from
the record.

## Search context

Prior to this test, a targeted search was carried out for a genuine
homogeneous-BEC, multi-k, Bragg-spectroscopy dispersion dataset with
reported uncertainties (i.e., a "Level-A" dataset: real atomic BEC +
independently measured excitation frequency + broad wavevector coverage +
uncertainties + machine-readable data). Two Cambridge University repository
datasets (Lopes/Eigen et al., PRL 118.210401 and PRL 119.190404) were
obtained and inspected; neither measures ω(k) across a range of k — both
measure frequency shift or diffracted fraction as a function of interaction
strength at a small number of fixed momenta, and were judged unsuitable.
The Ozeri/Steinhauer/Davidson (2002) and Shammass et al. (2012, PRL
109.195301) papers — which do report genuine multi-k phonon dispersion
measurements — predate open-data norms; no public repository or raw
numerical table could be located for either despite a direct search of the
Steinhauer Atomic Physics Laboratory (Technion) publication and research
pages. No Level-A dataset was found.

## Dataset used (Level-B)

Data are taken from the publicly archived supporting data (Harvard
Dataverse, DOI:10.7910/DVN/LGT5O6, CC0) for Guo et al., "An optical lattice
with sound," Nature 599, 211 (2021) (arXiv:2104.13922). Specifically, the
file `Fig4_dispersion.tab`/`.csv`, which contains the above-threshold
Goldstone (phonon) mode dispersion of density-wave polaritons in a
BEC-confocal-cavity system, at pump strength η²/η²_th = 1.25.

**Important physical caveat.** This is not a free, homogeneous-BEC phonon
dispersion. The measured collective mode is a cavity-mediated density-wave
polariton with its own characteristic momentum scale ζ = 1/ξ (ξ ≈ 5 μm, set
by the cavity's mode structure), and the theory in the source paper predicts
flattening toward a Debye-like frequency at large k_⊥ due to the finite
range of the cavity-mediated interaction — a structurally different
mechanism from a Λk⁴ dispersion correction. Any curvature detected in this
dataset cannot be attributed to the Λ-model's quartic term without
additional justification; this test is therefore a purely phenomenological
curvature check, not a probe of Λ specifically.

## Method

Six (k, ω, σ_ω) points were used exactly as archived, with no re-binning,
re-weighting, or exclusion. Two nested models were fit directly to ω(k)
(not to ω²(k), to avoid distorting the reported Gaussian uncertainties
under a nonlinear transform):

```
M0: ω = a k                     (pure linear, Goldstone-mode baseline)
M1: ω = a k + b k³               (equivalent, to leading order, to
                                   ω² = A k² + B k⁴)
```

Weighted χ², reduced χ², AIC, and BIC were computed for both models,
along with an F-test for the nested-model comparison and the significance
of b. The k=0 point was included in the primary fit as a genuine
measurement; a sensitivity check excluding it was run separately.

## Results

| Quantity | M0 (ω=ak) | M1 (ω=ak+bk³) |
|---|---:|---:|
| a | 202.79 ± 7.85 | 232.22 ± 20.04 |
| b | — | (−3.54 ± 2.22) × 10⁵ |
| χ² (dof) | 29.13 (5) | 26.59 (4) |
| reduced χ² | 5.83 | 6.65 |
| AIC | 31.13 | 30.59 |
| BIC | 30.93 | 30.17 |

- **b significance: −1.60σ** — below the conventional 2σ threshold.
- **ΔAIC (M1−M0) = −0.55, ΔBIC (M1−M0) = −0.76** — both below the
  conventional |Δ|>2 threshold for a "notable" preference; the two models
  are statistically indistinguishable by this criterion.
- **F-test: F=0.38, p=0.57** — does not reject M0 in favor of M1.
- **Sensitivity check (k=0 excluded):** identical conclusion (b
  significance −1.60σ, ΔAIC=−0.55, ΔBIC=−0.94); the result is not driven
  by the k=0 point.
- **M0 reduced χ² ≈ 5.8** is notably larger than 1, indicating the
  point-to-point scatter in this six-point dataset exceeds what the
  reported σ_ω values alone would predict for a clean linear dispersion.
  Standardized residuals do not show a smooth, monotonic trend with k
  (+0.24, +3.68, +3.04, −2.50, −0.11, −0.31), which is inconsistent with a
  coherent curvature signal and more consistent with measurement scatter.

## Interpretation

No statistically significant preference for the quartic-augmented model
M1 over the linear baseline M0 is found in this dataset. This is a genuine
null result, obtained by pre-registering the two models and the test
statistics before inspecting whether the outcome would be convenient for
the Λ-model.

This result does **not** constitute:

- a Λ-model validation or falsification (the underlying physical system —
  a cavity-mediated DW polariton — is not the system the Λ-model's quartic
  term is intended to describe, as noted in the caveat above);
- evidence against quartic dispersion corrections in general (the dataset
  is small, N=6, and the reduced χ² of the baseline model itself indicates
  under-estimated scatter, limiting the statistical power of this test in
  either direction);
- a substitute for a genuine Level-A homogeneous-BEC test, which remains
  unavailable due to the absence of public raw data from the studies best
  suited to it (Shammass et al. 2012; Ozeri et al. 2002).

## Status

**Status:** Level-B phenomenological test, completed — null result.

**Claim level:** no evidence for or against quartic curvature in this
specific real dataset; explicitly not a Λ-model test given the mismatched
underlying physics.

**Next steps (not committed to):** a related dataset, `Fig3e_Energy.tab` /
`Fig3e_Err.tab` (below-threshold roton dispersion of the same system, five
pump-strength curves × six k points each, from the same Harvard Dataverse
repository), could be considered as a secondary robustness check. This is
deferred pending clarification of whether the five curves share common
systematic/calibration uncertainties — treating the 30 points as fully
independent without establishing this could artificially inflate
statistical significance. This is not treated as a required next step.

## Provenance note

This addendum is recorded specifically to document that a real, independent
dataset was located, tested under a pre-specified protocol, and did not
yield a result supporting the Λ-model's quartic dispersion term. This
negative result is retained in the repository on the same footing as
positive results (cf. `PAPER3_ADDENDUM_BLIND_2D.md`), consistent with the
project's stated practice of reporting negative and incomplete results
rather than omitting them.
