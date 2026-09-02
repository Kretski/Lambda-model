# Paper 3 — Validation Status

*Last updated: 2026-09-02. Every claim in this document was independently
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
| QNM / analog-gravity test | ⏳ | In contact with Švančara et al. (giant quantum vortex experiment); awaiting fitted flow parameters (C, Ω, h₀) and uncertainties |
| Photon-ring/shadow observational test | ⏳ | EHT radio-band (230 GHz) shown to give an uninformative bound (photon energy E~2e-76 in dimensionless units — near-horizon Λ_n correction suppressed by ~150 orders of magnitude); a gamma-ray-band channel would be needed for a meaningful near-horizon constraint |

## Critical scope clarification

**Two structurally different "Λ" parameters exist in this line of work, confirmed dimensionally incompatible:**

- **Λ_NH** (Paper 2, near-horizon Hamiltonian, `H=½[g^μν k_μk_ν+Λk_loc⁴]`): dimension **[L²]** (confirmed in Paper 2 text: "Λ>0 has dimensions of length squared"; consistent with Paper 1's `Λ≲10⁻⁵³ m²` LIV bound).
- **Λ_GW** (Paper 3, propagation dispersion, `ΔΨ(f)=-4π³ΛK(z)f³/c³`): dimension **[L³/T]** (derived by direct dimensional analysis of the phase formula).

No explicit derivation connecting Λ_NH → Λ_GW currently exists in either paper. **The GW150914 propagation test above constrains Λ_GW only — it does not constrain, confirm, or test the near-horizon Λ_NH or any of Paper 2's photon-ring/QNM predictions.**

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

## What this status DOES support

- ✅ The mathematical/numerical machinery (dispersion fitting, matched-filter recovery, photon-ring solver) is internally consistent and has been independently re-verified, not just asserted.
- ✅ Λ=0 (pure GR) is not excluded by any test performed to date — in the propagation channel (single-event and LVK-catalog-derived), and in the synthetic/regression suites.
- ✅ A clear, dimensionally-honest separation exists between the near-horizon and propagation realizations of the model, preventing accidental cross-contamination of claims between them.

## Next steps (not yet done)

1. Apply the validated near-horizon photon-ring solver to real EHT/VLBI data once a plasma-corrected ("shadow-only") multi-frequency measurement is publicly available (not yet published as of this writing — active `CHARM`-framework development, per arXiv 2606.30753).
2. Complete the Švančara et al. residual analysis once fitted flow parameters and uncertainties are received.
3. Consider a gamma-ray-band (not radio-band) observational channel for a meaningful near-horizon Λ_n constraint, given the EHT radio-band suppression finding above.
