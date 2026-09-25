# Provenance — Level-A Ozeri et al. 2002 BEC Consistency Check

## Search context: why this dataset, and what was tried first

The Λ-model's BEC-sector prediction (Λ = ξ²/4) is analytically established
by coefficient-matching against the Bogoliubov dispersion relation (see
`convention_audit.md`). Testing this against a *genuine, direct*
homogeneous-BEC excitation-energy measurement was the original goal.
Before settling on Ozeri et al. (2002) as digitized here, several
candidate datasets were checked and found unsuitable (see the
companion Level-B module's `provenance.md`,
`../level_b_guo2021/provenance.md`, for the fuller search record,
including two Cambridge University repository datasets that measure
interaction-strength-dependent shifts rather than a k-sweep, and a
targeted search of the Steinhauer Atomic Physics Laboratory (Technion)
publication pages, which located no public repository or raw table for
either Ozeri et al. 2002 or Shammass et al. 2012 (PRL 109.195301) — both
predate open-data norms).

Ozeri et al. (2002) was ultimately selected because:
1. it is a genuine, direct atomic-BEC excitation-energy measurement
   (not an interaction-strength scan, not a cavity-mediated collective
   mode);
2. the full text is openly available on arXiv (cond-mat/0112496);
3. it reports multiple momenta (four points, Fig. 5) with error bars;
4. one specific numerical value is explicitly quoted in the text
   (kξ=0.96: 1840±100 Hz), providing an independent cross-check for
   the digitization pipeline (see below).

No machine-readable numerical table is available for this paper; the
four data points used here were obtained by digitizing the published
figure, as described next.

## Digitization methodology

### Source figure

Fig. 5 of arXiv:cond-mat/0112496 ("Excitation energies measured from the
released-phonon cloud only"), filled circles (measured from the slope of
energy vs. number of excitations — the paper's primary, most precise
method; open circles, a secondary single-image method, were not used).

### Procedure

1. The PDF page containing Fig. 5 was rasterized at 300 DPI
   (`pdftoppm -png -r 300`).
2. Axis calibration: pixel positions of the tick marks at kξ =
   {0.0, 0.5, 1.0} and E/h = {0, 1, 2} kHz were located programmatically
   by thresholding for dark pixels immediately adjacent to the axis
   lines and averaging over each tick's pixel span. This gives an exact
   linear pixel-to-data conversion for both axes (no manual placement).
3. Data-point localization: the four filled-circle markers were
   isolated from the intersecting solid (Bogoliubov) and dotted
   (hydrodynamic-simulation) curve lines by binary morphological erosion
   (7×7 structuring element), which removes thin line segments (curves,
   error-bar whiskers) while preserving the larger, solid disk markers.
   This left exactly four connected components, whose centroids were
   taken as the data-point centers.
4. Error bars: for each point, the vertical extent of dark pixels in a
   narrow column strip (±2 px) around the marker's x-position was
   measured in a tight window around the marker center (±25 px, chosen
   to capture the small whisker caps visible immediately above/below
   each marker without extending into the nearby curve lines — verified
   visually per-point; see below). Half the top-to-bottom span was taken
   as σ.

### Independent cross-check

The source text explicitly quotes, for the kξ=0.96 point: "we get an
energy of 1840±100 Hz per excitation in the released-phonon cloud."
The digitization pipeline independently returned **E/h = 1841 ± 101
Hz** for this point — agreement to within 1 Hz on both the central
value and the uncertainty, without this number being used anywhere in
the calibration. This is treated as strong validation of the pixel
calibration and centroid/whisker-detection procedure.

### Visual verification

Each of the four points was additionally inspected visually (cropped,
upscaled figure regions) to confirm that:
- the isolated component is genuinely the filled-circle marker with its
  two small whisker caps, not a segment of the solid or dotted curve
  (which pass close to, and in three of four cases directly through,
  the markers);
- the initial wide-window whisker search for the lowest point
  (kξ=0.324) had incorrectly included the nearby solid Bogoliubov curve
  as if it were the upper whisker cap; this was caught by visual
  zoom-in and corrected by narrowing the search window to ±25 px
  (see script history / conversation record for this correction).

### Digitization uncertainty vs. reported uncertainty

Pixel scale: 218.75 px/kHz (E-axis), 758.5 px per unit kξ (x-axis) at
300 DPI. Conservatively assuming ±1.5 px uncertainty in centroid
localization:

```
sigma_digit(E) ≈ 0.0069 kHz
```

compared to the whisker-derived (reported) uncertainties of
0.064–0.101 kHz across the four points — i.e. σ_digit/σ_reported ≈
4–11%. Combined in quadrature, digitization uncertainty changes
σ_total by <1% relative to σ_reported alone. **σ_reported dominates**;
the values in `digitized_fig5.csv` use σ_total (reported ⊕ digitization,
in quadrature), but this is numerically almost identical to σ_reported.

## Convention audit

See `convention_audit.md` for the full, independently-rederived proof
that Λ_Paper3 (= ξ_P²/4) and the Ozeri-native quartic coefficient
(= ξ_O²/2) refer to the same physical Λ, related by ξ_P = √2·ξ_O. This
was verified directly from the Ozeri paper's own Bogoliubov formula
(not assumed from the Paper-3 side alone), and is enforced by a runtime
assertion in `fit_bogoliubov_lambda.py`.

## LDA / trapped-BEC caveat

The Ozeri BEC is confined in a magnetic trap (elongated, cylindrically
symmetric, radial/axial trapping frequencies 220 Hz / 25 Hz); ξ⁻¹ = 4.3
μm⁻¹ is quoted as an average over the whole condensate. The excitation
energy is analyzed under the local-density approximation (Eq. 1 of the
source paper): a density-weighted average of the local (homogeneous-BEC)
excitation energy over the trap. This means the measured E(k) is not,
strictly, the dispersion relation of an idealized infinite homogeneous
BEC — it is an LDA-averaged quantity for this specific trapped system.
The source paper's own Bogoliubov theory curve (shown as the solid line
in Fig. 5, "calculated in the LDA for the measured E_BEC with no fit
parameters") already accounts for this; the fixed-β=0.5 model (M_B) used
here compares against the same underlying Bogoliubov physics, so this
caveat affects both the M_B fixed-prediction fit and the free-β fit
symmetrically — it is not expected to bias the β_fit-vs-β_pred
comparison, but it does mean this test is not a check against a
theoretically idealized homogeneous dispersion relation in the
strictest sense.

## Full statistical results

See `results.csv` for complete machine-readable output. Summary tables
are given in `README.md`. Key numbers, reproduced here for the record:

**Primary (N=4):**
- M0: A=1.6126±0.0723, χ²=35.78 (dof=3), reduced χ²=11.93
- M_B (β=0.5 fixed): A=1.4341±00.0635, χ²=23.61 (dof=3), reduced χ²=7.87
- M_Λfree: A=0.7985±0.2167, β_fit=5.5619±3.8851 (deviation from
  β_pred=0.5: +1.303σ), χ²=10.63 (dof=2)

**Sensitivity (N=3, kξ=0.324 excluded):**
- M0: A=1.7717±0.0776, χ²=3.94 (dof=2), reduced χ²=1.97
- M_B (β=0.5 fixed): A=1.5400±0.0672, χ²=0.4735 (dof=2), reduced
  χ²=0.2368
- M_Λfree: A=1.4066±0.2154, β_fit=0.9005±0.7062 (deviation from
  β_pred=0.5: +0.567σ), χ²=0.0216 (dof=1)
- ΔAIC(M_Λfree − M_B) = +1.5481, ΔBIC(M_Λfree − M_B) = +0.6467 (fixed
  prediction preferred over free fit)

## Interpretation — explicit claim boundaries

**What this result establishes:**

- The digitization pipeline is independently validated against a
  text-quoted published number (agreement to <1 Hz).
- The convention audit connecting Paper-3's Λ=ξ_P²/4 to Ozeri's native
  variables (Λ=ξ_O²/2) is independently re-derived and verified by code
  assertion.
- In the sensitivity analysis (N=3, excluding the author-flagged point),
  the free-fit curvature parameter is statistically consistent with the
  pre-registered Bogoliubov/Λ-model prediction (<0.6σ difference), and
  model selection (AIC/BIC) favors the fixed prediction over allowing
  the curvature to float.

**What this result does NOT establish:**

- It does not confirm or detect Λ = ξ²/4 as a validated physical
  constant — the sensitivity-analysis sample is far too small (N=3, 1
  degree of freedom for the free fit) for any such claim.
- The primary (N=4) analysis is inconclusive, dominated by a single
  point with a documented systematic effect; it must always be reported
  alongside the sensitivity analysis, not omitted, to avoid the
  appearance of post-hoc point selection.
- It is not independent of the LDA/trapped-BEC caveat above, nor of the
  digitization procedure (though the latter's uncertainty is shown to be
  small relative to the reported measurement uncertainties).
- It does not by itself establish Λ as a universal constant across
  domains (cf. the honesty statement in the main repository README and
  `PAPER3_VALIDATION_STATUS.md`).

## Reproducibility

Run `python3 fit_bogoliubov_lambda.py` in this directory (requires
numpy, scipy). The script reads `digitized_fig5.csv`, performs the
convention-audit assertion, runs both the primary and sensitivity
analyses, and writes `results.csv`. No network access, manual
digitization steps, or external tools are required at run time — the
digitization itself (pixel calibration, blob/whisker detection) was a
one-time procedure applied to the source figure, documented above for
reproducibility, with the resulting four points frozen in
`digitized_fig5.csv`.

## Relationship to other Paper 3 documents

This module supplements, and should be read alongside:

- `../level_b_guo2021/` — the companion Level-B phenomenological control
  (different physical system, different question, null result)
- `paper3/PAPER3_VALIDATION_STATUS.md` — main validation status
- `paper3/gw/matched_filter/_archive/qnm_branch_work/PAPER3_ADDENDUM_BLIND_2D.md`
  — the giant-vortex q=1/q2 experimental validation (a different,
  independently stronger real-data result, on a different physical
  system entirely)

No claim in this module should be combined with, or used to strengthen,
claims made in any of those other documents without explicit,
independent justification.
