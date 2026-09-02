# Λ-model — A Quartic Dispersion Framework

**Author:** Dimitar Kretski
**ORCID:** [0000-0001-5108-2243](https://orcid.org/0000-0001-5108-2243)
**Affiliation:** Center for Hydro- and Aerodynamics, Bulgarian Academy of Sciences, Varna, Bulgaria

---

## What this repository is for

This repository implements a one-parameter dispersion relation,

$$
\omega(k) = c\,k\,\sqrt{1 + \Lambda k^2},
$$

together with a validated numerical PDE solver for its wave equation, and
a fitting tool that tests whether a real, measured $(k,\omega)$ dataset
is consistent with this form.

**What a researcher can actually do with it, in five minutes:**

1. **Test your own dispersion data.** Run
   `lambda_experimental_validator.py --omega your_omega.csv --k your_k.csv`
   on any $(k,\omega)$ pairs you have — no domain assumptions required.
   You get a fitted $\Lambda$, its standard error, $R^2$, and an explicit
   significance test against $\Lambda=0$.
2. **Simulate propagation under this dispersion.** The 2D spectral PDE
   solver (`wave_equation_2D_solver.py`) is validated to machine precision
   in space and $O(1/N)$ in time (see Numerical status below) — use it
   directly if you need to propagate a field under a given $\Lambda$.
   The full leapfrog time-integration path (not just the spectral-exact
   shortcut) is separately validated for spatial convergence — see
   `paper3_h_convergence_test.py` and the status table.
3. **If you work with BEC:** the mapping $\Lambda=\xi^2/4$ is physically
   established (exact match to the Bogoliubov dispersion relation, not an
   analogy). Plug in your healing length and sound speed and test it
   against your own Bragg-spectroscopy data directly.
4. **If you work with Dirac materials or photonic dispersion:** the
   repository is explicit about which mappings are *not* currently
   supported, so you don't waste time chasing a claim that doesn't hold up
   (see Status table below). The fitting infrastructure still works on
   your data — you supply the physical mapping.
5. **Extend it.** Adding a new domain mapping is one function
   (`lambda_from_<domain>(...)`) plus a CLI entry; the fitting, error
   reporting, and significance-testing machinery is already there and
   does not need to be reimplemented.

## Status of results — read before citing

| Component                                   | Status                                                                                                                                    |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Numerical PDE solver (spatial, spectral-exact shortcut) | ✅ Exact (each Fourier mode assigned its analytic $\omega(k)$; no discretization in this path)                                |
| Numerical PDE solver (spatial, **full leapfrog integration**) | ✅ Convergence order **2.0 confirmed** between N=64→128 (genuinely discretized Laplacian², not a shortcut). ⚠️ **N<64 unreliable** (19% error at N=32). ⚠️ **N>128 untested** (CFL cost scales as N⁴); do not assume convergence beyond this without re-testing. |
| Numerical PDE solver (temporal)             | ✅ Validated $O(1/N)$ convergence                                                                                                          |
| Discrete biharmonic operator $D_n^4$        | ✅ Validated $O(h^2)$ convergence                                                                                                          |
| $\Lambda$-recovery from noisy data          | ✅ Validated, sub-percent accuracy (analytic $\omega(k)$ input; see above for recovery from actual simulated/integrated fields)           |
| Fermi-LAT bound                             | ✅ Audited: $\Lambda < 1.4421\times10^{-53}\,\mathrm{m^2}$ (corrected from a $3/2$ normalization error)                                    |
| EHT sensitivity ceiling                     | ⚠️ Validated as a *sensitivity estimate*, not a fit to real M87\* data                                                                    |
| GW forecast                                 | ⚠️ Illustrative projection, not a fitted LIGO/Virgo/KAGRA constraint. Superseded by the matched-filter pipeline below.                    |
| GW matched-filter pipeline (Stages 1–4)     | ✅ Validated architecture — Stage 4 (coherent H1+L1) passes off-source null trials and injection/recovery on real strain. A similar unconditioned-strain root cause (no highpass/Tukey window before PSD estimation) was independently found and fixed in the Stage 6+ pipeline below; whether it fully explains this specific PN-order systematic has not been directly re-tested on the Stage 3/4 code path. See `paper3/gw/matched_filter/STAGE4_FINAL_STATUS.md` for the original diagnosis. |
| GW propagation-dispersion pipeline (Stage 6+, `stage6E3H-R_v2.py`) | ✅ **Physically-conditioned pipeline.** Adds explicit highpass (0.9×f_low) + Tukey windowing before PSD/matched-filter (root cause of the earlier PN-order artifact — unconditioned strain gave noise-only matched-filter scores ~450–500, i.e. dominated by unfiltered low-frequency content, not the injected signal). Also fixes a Λ-grid phase-aliasing bug (grid must stay within ±half the aliasing period, ≈±0.57 for this band/redshift; the original ±20 grid wrapped ~35 times). Injection-recovery correlation on the corrected pipeline: 0.81 (n=50, real O1 noise). **GW150914 result:** Λ_on=−0.72, off-source null μ=+0.22, σ=2.21 (n=60 real off-source epochs), **z=−0.43σ — consistent with GR, no significant deviation.** This supersedes the Stage 3/4 exploratory point estimates above. Full status: `paper3/PAPER3_VALIDATION_STATUS.md`. |
| Near-horizon photon-ring solver, generalized to arbitrary dispersion power *n* | ✅ `A1v3_zamo_v04.py` / `A2_chromatic_shadow_generalized.py` generalize the Paper 2 quartic Hamiltonian (H=½[g^μν k_μk_ν+Λk_loc⁴], the *n*=1 case) to H_n=½[g^μν k_μk_ν+Λ_n k_loc^(2n+2)]. Regression-tested: *n*=1 reproduces Paper 2's published photon-ring coefficients (C_pro=0.6230, C_ret=11.495) to 4 significant figures. Derived and numerically confirmed: δb ∝ Λ_n·E^(2n) for each *n* (1–4 tested, exact match to predicted exponent). **Not yet applied to real observational data** — EHT radio-band (230 GHz) shown to give an uninformative bound for Sgr A* (photon energy E≈2×10⁻⁷⁶ in this framework's units, suppressing the Λ_n correction by ~150 orders of magnitude); a gamma-ray-band channel would be needed for a meaningful near-horizon constraint. |
| External cross-check: LVK GWTC-4.0 α=4 modified-dispersion-relation bound | ✅ The propagation phase ΔΨ(f,Λ)=−4π³ΛK(z)f³/c³ is mathematically identical to the standard LVK α=4 MDR parametrization (Mirshekari–Yunes–Will, E²=p²c²+A_αp^αc^α), routinely tested in every GWTC catalog. Exact conversion derived and verified numerically (D₄=c·K(z)/(1+z)³ to machine precision): Λ=h²c³/(4π²)·A₄. Applying the published GWTC-4.0 combined bound (83 events, arXiv:2603.19020, Table 5) gives Λ∈[−7.2×10⁻³, +2.2×10⁻³] m³/s (90% CI) — consistent with Λ=0, and far more statistically powerful than the single-event test above. |
| **Critical scope note** | Λ_NH (near-horizon, Paper 2, dimension **[L²]**) and Λ_GW (propagation, Paper 3, dimension **[L³/T]**) are **dimensionally incompatible** — confirmed by direct analysis of both phase formulas. No derivation connecting them currently exists. The GW150914/LVK results above constrain Λ_GW only; they say nothing about Λ_NH or Paper 2's photon-ring/QNM predictions. See `paper3/PAPER3_VALIDATION_STATUS.md` for the full separation. |
| BEC mapping ($\Lambda=\xi^2/4$)             | ✅ Physically established (exact Bogoliubov coefficient match)                                                                             |
| Dirac mapping ($\Lambda=(1-\eta^2)v_F^2/4$) | ❌ **Speculative, not supported** by the cited literature (Fu 2009 describes an anisotropic $k^6$ effect, not this isotropic $k^4$ form)   |
| Photonic mapping ($\Lambda=\beta_4^k/c^2$)  | ⚠️ Dimensionally correct and properly derived, but $\beta_4^k$ is **not yet connected** to any standard measurable dispersion coefficient |

**Any $\Lambda$ value extracted from simulated (not purely analytic) data
must state the grid resolution used.** Results from N<64 spatial grids
should not be trusted; see `paper3_h_convergence_test.py`.

See `paper3/paper3_final.tex` for the full derivations and the honesty
statement on what is proved, audited, forecast, or retracted.

## Repository structure

```
Lambda-model/
├── README.md
├── LICENSE
├── requirements.txt
│
├── paper3/
│   ├── wave_equation_2D_solver.py       # Core PDE solver, omega(k)=ck*sqrt(1+Lambda k^2)
│   ├── paper3_Dn4_test.py               # Validates the discrete quartic operator D_n^4
│   ├── paper3_grid_convergence.py       # Temporal convergence + Lambda-recovery tests (analytic omega input)
│   ├── paper3_h_convergence_test.py     # Spatial h->h/2->h/4 convergence, REAL leapfrog PDE integration
│   ├── lambda_experimental_validator.py # Fits Lambda from (k, omega) data; domain mappings
│   ├── paper3_final.tex / .pdf          # Full writeup, including the honesty audit
│   ├── dispersion/                      # Extended validation module (fiber structural test, real-data pipeline)
│   ├── gw/                              # Gravitational-wave dispersion tests
│   │   ├── gwosc_chirp_dispersion_test.py       # legacy/invalidated — Hilbert extraction, 51.9σ bias
│   │   ├── gwosc_injection_recovery_test.py     # diagnostic — proved pipeline bias (Λ=0→Λ_fit=-1.91)
│   │   ├── gwosc_extraction_ablation.py         # diagnostic — localized bias to Hilbert-transform math
│   │   ├── gwosc_frequency_extraction_comparison.py  # diagnostic — tested 5 extraction methods
│   │   ├── gwosc_zero_crossing_injection_recovery_v2.py  # diagnostic — zero-crossing also unstable
│   │   └── matched_filter/              # VALIDATED phase-domain approach (Stages 1-4)
│   │       ├── waveform.py              # GR + Lambda phase-domain waveform model
│   │       ├── likelihood.py            # matched-filter likelihood, grid search
│   │       ├── synthetic_injection.py   # frequency-domain injection generator
│   │       ├── recovery_test.py         # Stage 1 — synthetic noise — PASS
│   │       ├── stage2_real_noise_recovery.py       # Stage 2 — real PSD, Gaussian noise — PASS
│   │       ├── stage3_real_strain_validation.py    # Stage 3 — real H1 strain — PASS (A+B,C); GW150914 exploratory
│   │       ├── stage4_coherent_h1l1_validation.py  # Stage 4 — coherent H1+L1 — PASS (4C,4D); PN-systematic found
│   │       ├── STAGE3_FINAL_STATUS.md   # Full Stage 3 audit trail
│   │       ├── STAGE4_FINAL_STATUS.md   # Full Stage 4 audit trail
│   │       ├── stage6E3H-R_v2.py        # Stage 6+ — highpass/Tukey conditioning + alias-safe Λ grid (fixes the Stage 4 PN-systematic root cause)
│   │       ├── gw150914_lambda_onsource_test.py  # GW150914 propagation-Λ test: z=-0.43σ vs off-source null
│   │       ├── A1v3_zamo_v04.py         # Near-horizon photon-ring solver (Paper 2 quartic Hamiltonian)
│   │       ├── A2_chromatic_shadow_generalized.py  # Photon-ring solver generalized to dispersion power n=1..4
│   │       ├── gr_vs_lambda_search_auc.py  # GR-only vs Λ-search matched-filter AUC comparison (look-elsewhere cost)
│   │       ├── triaxis_analyzer_v4.py / triaxis_analyzer_v5.py  # Template-free H1/L1 cross-correlation detector (separate Zenodo deposit)
│   │       └── injection_recovery_lambda.py  # Λ-deformed signal injection through the TriAxis detector
│   └── figures/                         # Generated plots
│
│   PAPER3_VALIDATION_STATUS.md          # Independently re-verified status of every test suite above,
│                                         # including the Λ_NH vs Λ_GW dimensional scope separation
│
└── examples/
    ├── example_BEC.py                   # Lambda = xi^2/4 — established mapping
    ├── example_dirac.py                 # SPECULATIVE — raises a runtime warning
    └── example_photonic.py              # Corrected derivation; not yet measurable
```

## Quick start

```bash
pip install -r requirements.txt

# Validate the numerical solver against the exact dispersion relation
python paper3/wave_equation_2D_solver.py

# Check the discrete biharmonic operator and grid/temporal convergence
python paper3/paper3_Dn4_test.py
python paper3/paper3_grid_convergence.py

# Spatial convergence test using REAL PDE time-integration (not the
# spectral-exact shortcut) -- confirms Lambda extracted from an actual
# simulated field converges as h -> h/2 -> h/4
python paper3/paper3_h_convergence_test.py

# Test whether YOUR data is consistent with the Lambda-model
python paper3/lambda_experimental_validator.py --omega data_omega.csv --k data_k.csv

# Gravitational-wave matched-filter pipeline (Stages 1-4, phase-domain
# approach validated after legacy time-domain methods were found biased)
python paper3/gw/matched_filter/recovery_test.py                    # Stage 1: synthetic
python paper3/gw/matched_filter/stage2_real_noise_recovery.py       # Stage 2: real PSD
python paper3/gw/matched_filter/stage3_real_strain_validation.py --h1 <H1.hdf5> --event GW150914
python paper3/gw/matched_filter/stage4_coherent_h1l1_validation.py --h1 <H1.hdf5> --l1 <L1.hdf5> --event GW150914

# Stage 6+: physically-conditioned propagation-Λ test (fixes the Stage 4
# PN-order artifact; see PAPER3_VALIDATION_STATUS.md)
python paper3/gw/matched_filter/gw150914_lambda_onsource_test.py \
    --data <H1.hdf5> --n-null 60 --out gw150914_lambda_result.json

# Near-horizon photon-ring solver, generalized dispersion power n=1..4
# (n=1 regression-tests against Paper 2's published coefficients)
python paper3/gw/matched_filter/A2_chromatic_shadow_generalized.py --n 1 --n 2 --n 3 --n 4

# Worked examples (each prints its own validity status)
python examples/example_BEC.py         # established mapping
python examples/example_dirac.py       # speculative — reads its own warning
python examples/example_photonic.py    # corrected, not yet measurable
```

## What researchers in each field can do

| Group                                      | What you can do                                                                                                                                                                                                                                                                                                                                 |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Anyone with $(k,\omega)$ data**          | Run `lambda_experimental_validator.py` directly — no domain assumptions needed, just numbers.                                                                                                                                                                                                                                                   |
| **Theorists**                              | Inspect the covariant derivation in `paper3_final.tex`, verify the wave equation, propose new metric mappings.                                                                                                                                                                                                                                  |
| **BEC experimentalists**                   | Supply your healing length $\xi$ and sound speed $c_s$; test $\Lambda=\xi^2/4$ against your measured dispersion. This mapping is physically established, not speculative.                                                                                                                                                                       |
| **Dirac-material / ARPES groups**          | The fitting tool works on your data, but do **not** treat $\Lambda=(1-\eta^2)v_F^2/4$ as a prediction of this framework — see the audit in `paper3_final.tex` §5.2 and the runtime warning in `lambda_from_dirac()`. If you can derive a literature-supported quartic (not $k^6$) correction for your material, we would like to hear about it. |
| **Photonic-crystal / fiber-optics groups** | $\Lambda=\beta_4^k/c^2$ is now correctly derived from the model, but $\beta_4^k$ is not yet connected to a standard telecom $\beta_4$ or any other directly-measured quantity — see `paper3_final.tex` §5.3 for what's missing. Contributions on this specific gap are welcome.                                                                 |
| **Numerical relativity / computational physics reviewers** | Run `paper3_h_convergence_test.py` to independently verify the spatial convergence claim. The test integrates the actual fourth-order wave PDE via explicit leapfrog with a genuinely discretized biharmonic operator (not the spectral-exact shortcut used elsewhere in the repo) — this is the appropriate test for scrutinizing whether $\Lambda$ is a numerical artifact of grid spacing. |

## Honesty statement

This is a **falsifiable framework**, not a confirmed law of nature. A
single value of $\Lambda$ that is *simultaneously* consistent with
Fermi-LAT, EHT, GW, and BEC data would be evidence of a shared underlying
mechanism. Inconsistent values would falsify the universal-$\Lambda$
hypothesis while leaving each domain-specific dispersion law intact.

This repository also documents its own audit trail: two of the three
originally-proposed condensed-matter mappings (Dirac materials,
photonic crystals) did not survive scrutiny in their original form and
are reported here as negative/incomplete results rather than removed
silently. The spatial-convergence claim underwent the same process: an
earlier version of this repository asserted grid convergence based only
on a spectral-exact shortcut and a temporal-resolution test, neither of
which could have detected a genuine spatial-discretization problem had
one existed. `paper3_h_convergence_test.py` closes that specific gap by
testing the real leapfrog-integrated PDE instead. We consider negative
and corrected results as important to publish as positive ones.

## Paper 3 — Dispersion Validation

Paper 3 is currently treated as a constrained numerical and
experimental-validation study rather than as a claim that every
dispersive platform realizes the same Lambda model.

The central model is
    omega^2 = c^2 k^2 (1 + 2 Lambda k^2)

with a linear low-k branch and a quartic correction.

### Current validation status

| Component                           | Status                                    |
| ------------------------------------ | ----------------------------------------- |
| BEC Lambda mapping                  | Primary physical mapping                  |
| Lambda dispersion fitting           | Implemented                               |
| Synthetic Lambda recovery           | PASS                                      |
| Spatial (h) convergence, real PDE   | PASS, N=64-128 range only (see above)     |
| Dimensional audit                   | Implemented                               |
| Fiber structural test               | PASS for supplied demonstration regime    |
| Fiber -> Lambda identification      | REFUSED                                   |
| Dirac independent Lambda validation | Not established                           |
| Full 2D QNM solver                  | Diagnosed but not yet physically complete |

### Fiber result

The supplied fiber event-horizon demonstration data give
    p = 3.99870888

for
    Omega ~ |Delta k|^p.

The local log-log slope is approximately 4 across the tested range.

This is consistent with a pure quartic regime and is structurally
different from the Lambda-model low-k structure.

Therefore the repository does not interpret this result as an
independent measurement of Lambda.

### Dimensional safeguard

The experimental fiber quantity beta4 has units
    s^4 / m.

Consequently,
    beta4 / c^2

has units
    s^6 / m^3,

not m^2.

The validator therefore refuses the identification
    Lambda = beta4 / (4 c^2)

for this quantity.

This refusal is intentional and is part of the validation methodology.

### Reproducibility

The Paper 3 dispersion tools can be run directly from the repository:
    python paper3/dispersion/lambda_experimental_validator.py

and
    python paper3/dispersion/fiber_event_horizon_structural_test.py

For the spatial-convergence audit specifically:
    python paper3/paper3_h_convergence_test.py

When no experimental input is supplied, the Lambda validator runs a
synthetic self-test. Synthetic results must not be interpreted as
experimental evidence.

The fiber validator currently uses a demonstration dataset and clearly
labels it as such.

Real experimental data should replace the demonstration arrays before
making experimental claims.

## Citing this work

If you use this framework, please cite:

- Kretski, D. (2026). *A Hamiltonian Oscillator Extension of Wave Propagation in Schwarzschild Spacetime*. Zenodo. https://doi.org/10.5281/zenodo.22018715 (Paper 1, submitted to CQG, ref. CQG-117140)
- Kretski, D. (2026). *A Hamiltonian Dispersion Framework for Kerr Photon Rings, Frequency-Dependent Shadow Sensitivity, Superradiance, and Eikonal Quasinormal Modes*. Zenodo. https://doi.org/10.5281/zenodo.22051427 (Paper 2)
- Kretski, D. (2026). *A Universal Quartic Dispersion Framework: Numerical Validation, Multi-Messenger Time-of-Flight Bounds, and Condensed-Matter Analog Mappings*. (Paper 3, `paper3/paper3_final.tex` in this repository)

## License

MIT License (see `LICENSE`). Free to use, modify, and redistribute with attribution.

## Contact

Dimitar Kretski — Center for Hydro- and Aerodynamics, BAS, Varna, Bulgaria
ORCID: [0000-0001-5108-2243](https://orcid.org/0000-0001-5108-2243)
