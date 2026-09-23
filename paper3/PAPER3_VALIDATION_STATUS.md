# Paper 3 — Validation Status

*Last updated: 2026-09-23. Every claim in this document was re-run and checked
against actual test output before being recorded here, not taken from earlier
analysis or chat sessions. Where a previous version of this document was wrong,
the correction is noted rather than silently overwritten.*

## Status summary

| Category                                        | Status | Verified via                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ----------------------------------------------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mathematical formulation                        | ✅      | `waveform.py`, `likelihood.py`                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Synthetic Λ recovery (code check)               | ✅      | `dispersion/tests/test_model_discrimination.py::test_lambda_model_recovery` — calls the repository's `fit_lambda` on 300 noisy realisations; unbiased, and the reported standard error is calibrated (pull width ≈1, ~95% of ±2σ intervals contain the true value)                                                                                                                                                                                 |
| Λ=0 null control (code check)                   | ✅      | `test_lambda_zero_control` — 300 noisy Λ=0 realisations through `fit_lambda`; false 3σ rate below 2%. Separately, `[5] Λ=0 NULL CONTROL` in the matched-filter suite gives Λ_fit=−1.1×10⁻⁵                                                                                                                                                                                                                                                         |
| Model discrimination, quartic vs Λ (code check) | ⚠️     | `test_pure_quartic_is_not_lambda_model` — a strong quartic term (ω = ck + 0.3k⁴) gives reduced χ² > 10 under the Λ-model fit. `test_weak_quartic_is_indistinguishable` documents the limit: for a weak term (0.02k⁴) the two forms are practically degenerate (reduced χ² ≈ 1.25). A good Λ-model fit alone therefore does not show that data follow the Λ-model                                                                                   |
| Synthetic matched-filter pipeline (code check)  | ⚠️     | `gw/matched_filter/test_matched_filter_pipeline.py`, 6/6 tests pass. **Not yet re-audited** for tests that pass by construction (see correction note below)                                                                                                                                                                                                                                                                                        |
| GW150914 GPS/data extraction                    | ✅      | `stage6E3H-R_v2.py` (highpass+Tukey conditioning, alias-safe Λ grid, injection-recovery correlation 0.81)                                                                                                                                                                                                                                                                                                                                          |
| GW150914 propagation Λ test                     | ✅      | `gw150914_lambda_result_wide.json`: Λ_on=−0.72, μ_off=+0.219, σ_off=2.206, z=−0.43σ, n=60 off-source null                                                                                                                                                                                                                                                                                                                                          |
| **Significant non-zero Λ detection**            | ❌      | None of the tests in this document finds a statistically significant deviation from Λ=0                                                                                                                                                                                                                                                                                                                                                            |
| Near-horizon (Paper 2) photon-ring solver       | ⏳      | `A1v3_zamo_v04.py`, `A2_chromatic_shadow_generalized.py` — n-family generalization, regression-tested against Paper 2's published C_pro/C_ret to 4 significant figures. **Not applied to real observational data**                                                                                                                                                                                                                                 |
| Photon-ring/shadow observational test           | ⏳      | The EHT radio band (230 GHz) gives an uninformative bound: the near-horizon Λ_n correction is suppressed by ~150 orders of magnitude at that photon energy. A gamma-ray-band channel would be needed                                                                                                                                                                                                                                               |
| Giant-vortex QNMs, q=1                          | ✅      | Independent reimplementation of the giant-vortex QNM calculation (Švančara et al., EXP B), using the experiment's dispersion relation and the flow parameters of Table SI (arXiv:2308.10773), not fitted to the resonance frequencies. Mean relative deviation 0.021% (max 0.036%) over 18 azimuthal numbers, m = −21 … −4                                                                                                                         |
| Giant-vortex QNMs, q=2 (Root-B branch)          | ✅      | 0.011–0.060% for m ≤ −10, growing smoothly to 1.01% at m = −4. Continuation covers m = −14…−11 and −6…−4; the other points come from the discovery grid. The small-\|m\| trend is a property of the branch, not of how it was located. Roots cross-checked by an experiment-blind 2D search (m = −14) and an independent Newton solver (14/14; same physical residual function). See `PAPER3_ADDENDUM_BLIND_2D.md`, `STAGE6_5B4_3_ROOTB_FREEZE.md` |
| Giant-vortex q=2 near m = −15                   | ⚠️     | Continuation stops at m ≈ −14.95 at two step resolutions (Stage 5I-33). A procedural limit, not a demonstrated branch termination. See `STAGE6_5B4_3_supplement_5I33_boundary.md`                                                                                                                                                                                                                                                                  |
| Giant-vortex QNMs, q=3/q=4                      | ⏳      | No corresponding branch identified. Three lines of inquiry closed (5I-27, discovery-grid candidates, deep-pole branch 5I-32). Closed unless a new candidate branch or method appears                                                                                                                                                                                                                                                               |
| Giant-vortex statistical test                   | 🕒     | Planned. No χ² or significance is reported, because measured frequency uncertainties are not available; they have been requested from the experimental group. The provenance of the Table SI flow parameters (independent flow measurement vs. fit involving the spectra) is also being confirmed, since it determines whether "not fitted" is the right description                                                                               |

## Correction note (2026-09-23)

An earlier version of this document listed "10/10" synthetic and regression tests
as passed. Several of those tests could not fail:

- `test_lambda_dimensions_are_not_inferred_from_beta4` compared two hand-written
  unit strings;
- `test_quartic_scaling` and the fiber tests fitted synthetic arrays generated as
  ω ∝ k⁴, so they recovered their own input;
- the earlier discrimination and null tests used estimators defined inside the
  test file rather than the repository's `fit_lambda`, and compared against the
  exact generating model or noiseless data.

These tests and the rows based on them ("Quartic structural scaling",
"Photonic/fiber negative control") have been removed. The replacement tests call
`fit_lambda`, use noise, and were checked by deliberately breaking `fit_lambda`
(wrong error bar, bias, factor-2 convention); every such break makes at least one
test fail. The fiber module in `dispersion/` currently runs only on synthetic
demonstration arrays and produces no physical result.

## Critical scope clarification

Three separate realizations of the model appear in this work, each with its own
parameter:

- **Λ_NH** (Paper 2, near-horizon Hamiltonian, H=½[g^μν k_μk_ν+Λk_loc⁴]):
  dimension **[L²]**.
- **Λ_GW** (propagation dispersion, ΔΨ(f)=−4π³ΛK(z)f³/c³): dimension **[L³/T]**.
- **The analog-system coefficients** (giant vortex, BEC): properties of those
  media, tested in their own units.

Λ_NH = Λ_GW/c is dimensionally consistent ([L³/T]/[L/T]=[L²]), and the flat limit
of the Paper 2 Hamiltonian has the form ω²=c²k²(1+Λk²). **This is a
dimensional-consistency observation, not a physical derivation.** No equation
connects the near-horizon mechanism (k_loc⁴ correction near a Kerr photon ring)
to the FLRW propagation mechanism (K(z)f³ phase) that would single out this
conversion over any other dimensionally valid one. Applying it to the GWTC-4.0
bound gives Λ_NH ≲ 7.4×10⁻¹² m² in the graviton sector, about 41 orders of
magnitude weaker than the photon-sector Fermi-LAT bound (Λ ≲ 10⁻⁵³ m²). The gap
is expected from the different energy scales involved, but it is not evidence
that the sectors are related.

**The GW tests constrain Λ_GW only.** They do not test Paper 2's photon-ring or
QNM predictions, and they say nothing about the laboratory results. No derivation
connects the analog-system coefficients to Λ_NH or Λ_GW.

## External cross-check (LVK GWTC-4.0)

The Λ_GW phase form is mathematically identical to the standard LVK α=4
modified-dispersion-relation test (Mirshekari–Yunes–Will parametrization,
E�=p²c²+A_αp^αc^α). Direct comparison of the two phase formulas gives an exact
conversion, verified numerically to match D₄=c·K(z)/(1+z)³ to machine precision:

```
Λ_GW = h²c³/(4π²) · A₄  =  ħ²c³ · A₄
```

Applied to the published GWTC-4.0 combined bound (83 events, arXiv:2603.19020,
Table 5): **A₄ ∈ [−0.62, +0.19]×10³ eV⁻² (90% CI)** →
**Λ_GW ∈ [−7.2×10⁻³, +2.2×10⁻³] m³/s (90% CI)**, consistent with Λ=0.

This is a translation of a published, peer-reviewed LVK result, not an
independent reanalysis. The same flat-space translation was published
independently in arXiv:2607.17431 (see `horava_lifshitz_crosscheck.md`).

## What this status does NOT claim

- ❌ No experimental detection of non-zero Λ (Λ_NH, Λ_GW, or any analog coefficient).
- ❌ The GW150914 test says nothing about Paper 2's near-horizon predictions.
- ❌ The synthetic test suites (7 dispersion tests; the 6 matched-filter pipeline
  tests have not yet been re-audited) are code checks, not physical validation.
- ❌ The giant-vortex comparison does not establish a universal non-zero Λ, a new
  fundamental interaction, or that the same parameter governs astrophysical black
  holes. It uses the experiment's own dispersion relation and parameters, so it is
  an independent reproduction of that spectral calculation, not a test of new
  physics.

## What this status DOES support

- ✅ The dispersion fitting routine is unbiased, reports calibrated standard errors,
  and keeps Λ=0 data at the nominal false-positive rate — checked with tests that
  fail when the routine is deliberately broken.
- ✅ Λ=0 (pure GR) is not excluded by any test performed to date, in the
  propagation channel (single event and LVK catalogue) or elsewhere.
- ✅ The near-horizon, propagation and analog-system realizations are kept
  explicitly separate, so claims do not leak from one to another.
- ✅ An independent reimplementation of the giant-vortex QNM calculation reproduces
  the measured q=1 and q=2 resonances without fitting the individual frequencies,
  including a documented, smooth deviation of q=2 toward small |m|. This is the
  strongest real-data result in this work.

## Permanent boundaries of this work

The items below are not open action items. They are structural limits of the
evidence presented here: even a fully successful outcome of every remaining step
would not resolve them, because nothing in the project's logical structure
connects these pieces. They are recorded so that future updates do not silently
drift past them.

* **No single Λ is established across realizations.** Λ_NH (near-horizon),
  Λ_GW (propagation) and the analog-system coefficients tested against the
  giant-vortex and BEC data are separate constructions, each tested in its own
  context. No derivation connecting any two of them exists in this work. The
  relation Λ_NH = Λ_GW/c is a dimensional-consistency observation, not such a
  derivation.

* **Universality of Λ is neither claimed nor implied.** Every result here —
  laboratory, GW, theoretical cross-check — is scoped to the system it was
  obtained from.

* **Λ_GW = 0 (pure GR) remains fully consistent with the current astrophysical
  constraint.** The GWTC-4.0-derived bound is a constraint, not a detection. No
  re-analysis of the same posterior can turn it into a detection; only new,
  independent observational data could.

* **The Hořava–Lifshitz cross-check is a theoretical cross-check, not an
  independent observational validation, permanently.** Both it and this project's
  GWTC-4.0 analysis draw on the same published LVK posterior.

* **q=3/q=4 higher-overtone correspondence remains an open question, not a pending
  detection.** If no convincing branch correspondence is found, the correct final
  status is "unresolved" — not that the model has failed, and not that the
  overtones should be forced into the validated set.

**Summary.** This work shows that one specific Λ-deformed dispersion architecture
is mathematically self-consistent and numerically recoverable, and that an
independent reimplementation of the giant-vortex QNM calculation reproduces the
measured q=1/q=2 spectrum. It does not show, and does not claim to show, a single
physical constant governing multiple domains. Confirmed, null, inconclusive and
unresolved results are reported on equal footing.

## Next steps (not yet done)

1. Obtain measured frequency uncertainties and the provenance of the Table SI flow
   parameters from the experimental group, then perform a proper statistical
   comparison for q=1/q=2.
2. Re-audit `gw/matched_filter/test_matched_filter_pipeline.py` for tests that
   pass by construction, as was done for the dispersion tests.
3. Convert the remaining files that use the older 1 + 2Λk² convention
   (`paper3/STATUS.md`, `dispersion/README.md`,
   `fiber_event_horizon_structural_test.py`, `test_matched_filter_pipeline.py`,
   `gwosc_zero_crossing_injection_recovery.py`), checking each for changes in
   numerical results.
4. Apply the near-horizon photon-ring solver to observational data only if a
   suitable channel becomes available (a plasma-corrected multi-frequency shadow
   measurement, or a gamma-ray-band observation); the radio band is uninformative.
