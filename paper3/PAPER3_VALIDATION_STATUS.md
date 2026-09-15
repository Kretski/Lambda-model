# Paper 3 — Validation Status

*Last updated: 2026-09-15. Every claim in this document was independently
re-run and verified against actual test output before being recorded here
— not taken on faith from prior analysis or chat sessions.*

## Status summary

| Category | Status | Verified via |
|---|---|---|
| Mathematical formulation | ✅ | `waveform.py`, `likelihood.py` |
| Synthetic Λ recovery | ✅ | `dispersion/tests/test_model_discrimination.py::test_lambda_model_recovery` — Λ_true=0.05 → Λ_fit=0.04999181 (rel. error 1.6e-4) |
| Λ=0 null control | ✅ | `test_lambda_zero_control` (dispersion suite) AND independently `[5] Λ=0 NULL CONTROL` (matched-filter suite) — Λ_fit=-1.1e-5 |
| Model discrimination (quartic ≠ Λ) | ✅ | `test_pure_quartic_is_not_lambda_model` — pure quartic data gives poor Λ-model RSS (3.1e-4) vs. perfect quartic-model RSS (0.0) |
| Quartic structural scaling | ✅ | `test_quartic_scaling` — measured exponent 4.00000000 |
| Photonic/fiber negative control | ✅ | Fiber dataset scaling ≈4 (quartic), explicitly **not** identified with Λ (different physical dimension/definition) |
| Synthetic matched-filter pipeline | ✅ | `gw/matched_filter/test_matched_filter_pipeline.py`, 6/6 tests passed |
| GW150914 GPS/data extraction | ✅ | `stage6E3H-R_v2.py` (highpass+tukey conditioning, alias-safe Λ grid, validated injection-recovery correlation=0.81) |
| GW150914 propagation Λ test | ✅ | `gw150914_lambda_result_wide.json`: Λ_on=-0.72, μ_off=+0.219, σ_off=2.206, z=-0.43σ, n=60 off-source null |
| **Significant non-zero Λ detection** | ❌ | None of the above tests found a statistically significant deviation from Λ=0 |
| Near-horizon (Paper 2) photon-ring test | ⏳ | Solver exists and is validated (`A1v3_zamo_v04.py`, `A2_chromatic_shadow_generalized.py` — n-family generalization, regression-tested against Paper 2's published C_pro/C_ret to 4 significant figures), but **not yet applied to real observational data** |
| QNM / analog-gravity test | ✅ (q=1, q=2) / ⏳ (q=3, q=4) | Švančara et al. giant-quantum-vortex experiment (EXP B): using flow parameters taken directly from the experiment's published Table SI (not fitted to resonance frequencies), the model reproduces the fundamental (q=1, mean rel. error 0.021% across 18 azimuthal numbers) and first-overtone (q=2 Root-B branch, 0.011–0.060% at validated points) resonance branches. Independently confirmed by a blind 2D complex-plane discovery scan. q=3/q=4 correspondence remains unresolved — see `PAPER3_ADDENDUM_BLIND_2D.md` for full data, methodology, and claim-boundary discussion |
| Photon-ring/shadow observational test | ⏳ | EHT radio-band (230 GHz) shown to give an uninformative bound (photon energy E~2e-76 in dimensionless units — near-horizon Λ_n correction suppressed by ~150 orders of magnitude); a gamma-ray-band channel would be needed for a meaningful near-horizon constraint |

## Critical scope clarification

**Two structurally different "Λ" parameters exist in this line of work, confirmed dimensionally incompatible:**

- **Λ_NH** (Paper 2, near-horizon Hamiltonian, `H=½[g^μν k_μk_ν+Λk_loc⁴]`): dimension **[L²]** (confirmed in Paper 2 text: "Λ>0 has dimensions of length squared"; consistent with Paper 1's `Λ≲10⁻⁵³ m²` LIV bound).
- **Λ_GW** (Paper 3, propagation dispersion, `ΔΨ(f)=-4π³ΛK(z)f³/c³`): dimension **[L³/T]** (derived by direct dimensional analysis of the phase formula).

No explicit derivation connecting Λ_NH → Λ_GW currently exists in either paper. **The GW150914 propagation test above constrains Λ_GW only — it does not constrain, confirm, or test the near-horizon Λ_NH or any of Paper 2's photon-ring/QNM predictions.**

The giant-vortex QNM test above is a separate, third realization of the
model (a flat-space/analog-system quartic dispersion, not the near-horizon
Λ_NH Hamiltonian nor the FLRW propagation Λ_GW), tested in the laboratory
system's own units. No derivation connecting it to Λ_NH or Λ_GW currently
exists either; see `PAPER3_ADDENDUM_BLIND_2D.md` for the explicit claim
boundary on this point.

**Update (convention_check.py, committed separately):** a *dimensional*
bridge has been proposed, Λ_NH = Λ_GW/c (verified: [L³/T]/[L/T]=[L²],
consistent). Applying it to the GWTC-4.0-derived Λ_GW bound gives a
graviton-sector bound Λ_NH ≲ 7.4×10⁻¹² m². **This is dimensional
consistency, not a physical derivation** — no equation has been shown
connecting the near-horizon Hamiltonian mechanism (k_loc⁴ correction
near a Kerr photon-ring) to the FLRW propagation mechanism (K(z)f³
phase) that would justify this specific factor of 1/c over any other
dimensionally-valid conversion. Treat as an untested hypothesis, not a
closed result. Note also the resulting bound (~7×10⁻¹² m², graviton
sector) is ~41 orders of magnitude weaker than the Fermi-LAT
photon-sector bound (Λ≲10⁻⁵³ m², Paper 1) — expected under a
universality assumption given the vastly different photon (GeV) vs.
graviton (~10⁻¹² eV) energy scales involved, but this gap is not
itself evidence the two sectors are related.

## External cross-check (LVK GWTC-4.0)

Λ_GW's phase form is mathematically identical to the standard LVK α=4 modified-dispersion-relation (MDR) test (Mirshekari-Yunes-Will parametrization, `E²=p²c²+A_αp^αc^α`), routinely applied by the LVK collaboration to every GWTC catalog. Direct comparison of the two phase formulas gives an exact conversion (verified numerically to match D₄=c·K(z)/(1+z)³ to machine precision):

```
Λ_GW = h²c³/(4π²) · A₄
```

Applying this to the published GWTC-4.0 combined bound (83 cumulative events, arXiv:2603.19020, Table 5): **A₄ ∈ [-0.62, +0.19]×10³ eV⁻² (90% CI)** → **Λ_GW ∈ [-7.2×10⁻³, +2.2×10⁻³] m³/s (90% CI)**, consistent with Λ=0.

This is a far more statistically powerful bound than the single-event GW150914 test above, and comes directly from a published, peer-reviewed LVK result — not from an independent reanalysis.

## What this status does NOT claim

- ❌ Does not claim experimental detection of non-zero Λ (either Λ_NH or Λ_GW).
- ❌ Does not claim the GW150914 test says anything about Paper 2's near-horizon predictions (dimensionally distinct parameter).
- ❌ Does not claim the synthetic/regression test suites (10/10 combined) constitute physical validation — they are explicitly labeled by their own authors as methodological integrity checks only.
- ❌ Does not claim the giant-vortex QNM result establishes a universal non-zero Λ, a new fundamental interaction, or that the same Λ parameter governs astrophysical black-hole physics — see `PAPER3_ADDENDUM_BLIND_2D.md`.

## What this status DOES support

- ✅ The mathematical/numerical machinery (dispersion fitting, matched-filter recovery, photon-ring solver) is internally consistent and has been independently re-verified, not just asserted.
- ✅ Λ=0 (pure GR) is not excluded by any test performed to date — in the propagation channel (single-event and LVK-catalog-derived), and in the synthetic/regression suites.
- ✅ A clear, dimensionally-honest separation exists between the near-horizon, propagation, and analog-system realizations of the model, preventing accidental cross-contamination of claims between them.
- ✅ The analog-system (giant-vortex) realization of the model's dispersion relation is experimentally validated against real laboratory data for its fundamental and first-overtone resonance branches, without fitting the individual resonance frequencies as free parameters — the strongest real-data, non-frequency-fitted result obtained to date across all three realizations of the model.
- ## Permanent boundaries of this work

The items below are not open action items awaiting future closure. Theyare structural limits of the evidence architecture presented here: even afully successful outcome of every remaining planned step (Section "Nextsteps") would not resolve them, because the project's logical structuredoes not connect these pieces. They are recorded here so that futureupdates to this document do not silently drift past them.

* **No single Λ is established across realizations.** $\Lambda_{\rm NH}$(near-horizon, Paper 2), $\Lambda_{\rm GW}$ (propagation, Paper 3), andthe analog-system quartic coefficient tested against the giant-vortexand BEC laboratory data are three separate mathematical constructions,each internally consistent and separately tested in its own context. Noderivation connecting any two of them exists in this work. Thedimensional bridge $\Lambda_{\rm NH}=\Lambda_{\rm GW}/c$ noted elsewherein this document is a dimensional-consistency observation, not such aderivation (see "Critical scope clarification" above). Establishing aphysical connection between these realizations is outside the scope ofwhat has been attempted here, not a step that remains to be finished.
  
* **Universality of Λ is neither claimed nor implied.** Every result inthis document — laboratory, GW, theoretical cross-check — is scoped tothe specific system it was obtained from. No result here should be readas evidence that a single Λ parameter governs multiple physicaldomains.
  
* **$\Lambda_{\rm GW}=0$ (pure GR) remains fully consistent with thecurrent astrophysical constraint.** The GWTC-4.0-derived bound$\Lambda_{\rm GW}\in[-7.2\times10^{-3},+2.2\times10^{-3}]$ m³/s is aconstraint, not a detection. No amount of re-analysis of the sameGWTC-4.0 posterior changes this into a detection; only new, independentobservational data could do so.
  
* **The Hořava–Lifshitz cross-check is a theoretical cross-check, not anindependent observational validation, permanently.** Both it and thisproject's own GWTC-4.0 analysis draw on the identical published LVKposterior. This is a structural feature of using the same dataset, nota temporary limitation awaiting a new run.
  
* **q=3/q=4 higher-overtone correspondence is retained as an openprediction, not a pending detection.** If future continuation or awider blind spectral search does not produce a convincing branchcorrespondence, the correct and final status is that these overtonesremain unresolved — not that the model has failed, and not that theyshould be forced into the validated set. A null or inconclusive outcomehere is reported on the same footing as the positive q=1/q=2 result,consistent with the treatment of the Guo et al. negative controlelsewhere in this document.
  

**Summary framing.** This work demonstrates that one specificΛ-deformed wave-dispersion architecture is mathematically self-consistent,numerically recoverable, and supported by several independentempirical/theoretical consistency checks — with the laboratory q=1/q=2resonance structure as its strongest real result. It does not demonstrate,and does not claim to demonstrate, a single universal physical constantgoverning multiple domains. Confirmed, null, inconclusive, and unresolvedresults are reported on equal footing throughout.

## Next steps (not yet done)

1. Apply the validated near-horizon photon-ring solver to real EHT/VLBI data once a plasma-corrected ("shadow-only") multi-frequency measurement is publicly available (not yet published as of this writing — active `CHARM`-framework development, per arXiv 2606.30753).
2. Resolve the q=3/q=4 higher-overtone correspondence for the giant-vortex QNM test, via further branch continuation or a wider blind spectral search — see `PAPER3_ADDENDUM_BLIND_2D.md` for current status.
3. Consider a gamma-ray-band (not radio-band) observational channel for a meaningful near-horizon Λ_n constraint, given the EHT radio-band suppression finding above.
