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

## New here? Start with the one-line summary

- **If you have (k,ω) dispersion data**: run `lambda_experimental_validator.py` on it — see Quick start.
- **If you want the current honest bottom line**: `paper3/PAPER3_VALIDATION_STATUS.md` is the single most
  up-to-date status document — start there, not the full table below.
- **If you're reviewing the GW/LIGO work specifically**: the GWTC-4.0 cross-check result
  (Λ ∈ [−7.24×10⁻³, +2.22×10⁻³] m³/s = [−2.41×10⁻¹¹, +7.40×10⁻¹²] m², consistent with GR)
  is the most statistically robust result in the repo. **Note that it constrains the
  graviton sector only** — it does not bound Paper 2's photon-ring or shadow predictions.
  See the Critical scope note in the status table.

## Status of results — read before citing

| Component                                   | Status                                                                                                                                    |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Numerical PDE solver (spatial, spectral-exact shortcut) | ✅ Exact (each Fourier mode assigned its analytic $\omega(k)$; no discretization in this path)                                |
| Numerical PDE solver (spatial, **full leapfrog integration**) | ✅ Convergence order **2.0 confirmed** between N=64→128 (genuinely discretized Laplacian², not a shortcut). ⚠️ **N<64 unreliable** (19% error at N=32). ⚠️ **N>128 untested** (CFL cost scales as N⁴); do not assume convergence beyond this without re-testing. |
| Numerical PDE solver (temporal)             | ✅ Validated $O(1/N)$ convergence                                                                                                          |
| Discrete biharmonic operator $D_n^4$        | ✅ Validated $O(h^2)$ convergence                                                                                                          |
| $\Lambda$-recovery from noisy data          | ✅ Validated, sub-percent accuracy (analytic $\omega(k)$ input; see above for recovery from actual simulated/integrated fields)           |
| Fermi-LAT bound (photon sector)             | ✅ Audited: $\Lambda < 1.4421\times10^{-53}\,\mathrm{m^2}$ (corrected from a $3/2$ normalization error)                                    |
| EHT sensitivity ceiling                     | ⚠️ Validated as a *sensitivity estimate*, not a fit to real M87\* data                                                                    |
| GW forecast                                 | ⚠️ Illustrative projection, not a fitted LIGO/Virgo/KAGRA constraint. Superseded by the matched-filter pipeline below.                    |
| GW matched-filter pipeline (Stages 1–4)     | ✅ Validated architecture — Stage 4 (coherent H1+L1) passes off-source null trials and injection/recovery on real strain. A similar unconditioned-strain root cause (no highpass/Tukey window before PSD estimation) was independently found and fixed in the Stage 6+ pipeline below; whether it fully explains this specific PN-order systematic has not been directly re-tested on the Stage 3/4 code path. See `paper3/gw/matched_filter/STAGE4_FINAL_STATUS.md` for the original diagnosis. |
| GW propagation-dispersion pipeline (Stage 6+, `stage6E3H-R_v2.py`) | ✅ **Physically-conditioned pipeline.** Adds explicit highpass (0.9×f_low) + Tukey windowing before PSD/matched-filter (root cause of the earlier PN-order artifact — unconditioned strain gave noise-only matched-filter scores ~450–500, i.e. dominated by unfiltered low-frequency content, not the injected signal). Also fixes a Λ-grid phase-aliasing bug (grid must stay within ±half the aliasing period, ≈±0.57 for this band/redshift; the original ±20 grid wrapped ~35 times). Injection-recovery correlation on the corrected pipeline: 0.81 (n=50, real O1 noise). **GW150914 result:** Λ_on=−0.72, off-source null μ=+0.22, σ=2.21 (n=60 real off-source epochs), **z=−0.43σ — consistent with GR, no significant deviation.** This supersedes the Stage 3/4 exploratory point estimates above. Full status: `paper3/PAPER3_VALIDATION_STATUS.md`. |
| Near-horizon photon-ring solver, generalized to arbitrary dispersion power *n* | ✅ `A1v3_zamo_v04.py` / `A2_chromatic_shadow_generalized.py` generalize the Paper 2 quartic Hamiltonian (H=½[g^μν k_μk_ν+Λk_loc⁴], the *n*=1 case) to H_n=½[g^μν k_μk_ν+Λ_n k_loc^(2n+2)]. Regression-tested: *n*=1 reproduces Paper 2's published photon-ring coefficients (C_pro=0.6230, C_ret=11.495) to 4 significant figures. Derived and numerically confirmed: δb ∝ Λ_n·E^(2n) for each *n* (1–4 tested, exact match to predicted exponent). **Not yet applied to real observational data** — EHT radio-band (230 GHz) shown to give an uninformative bound for Sgr A* (photon energy E≈2×10⁻⁷⁶ in this framework's units, suppressing the Λ_n correction by ~150 orders of magnitude); a gamma-ray-band channel would be needed for a meaningful near-horizon constraint. |
| External cross-check: LVK GWTC-4.0 α=4 modified-dispersion-relation bound | ✅ The propagation phase ΔΨ(f,Λ)=−4π³ΛK(z)f³/c³ is mathematically identical to the standard LVK α=4 MDR parametrization (Mirshekari–Yunes–Will, E²=p²c²+A_αp^αc^α). **Verified line-by-line against the GWTC-4.0 arXiv source** (arXiv:2603.19020, `paperII__tests_of_propagation.tex` line 25): ratio 1.0000 at every redshift tested, fitted (1+z) exponent 0.000 — no residual redshift systematic, so the bound conversion needs no correction factor. Distance measures agree to machine precision (D₄ = c·K(z)/(1+z)³). Exact correspondence: **Λ = ħ²c³A₄** (equivalently h²c³A₄/4π²). Reproducible via `paper3/gw/matched_filter/convention_check.py`. Applying the published GWTC-4.0 combined bound — A₄ ∈ [−620, +190] eV⁻² at 90% credibility, two-sided (Q_GR = 82.5%), from 83 events (43 from GWTC-3.0 + 40 new from O4a; FAR ≤ 10⁻³/yr; BBH only, BNS/NSBH excluded from the MDR analysis) — gives **Λ ∈ [−7.24×10⁻³, +2.22×10⁻³] m³/s = [−2.41×10⁻¹¹, +7.40×10⁻¹²] m²**, consistent with Λ=0 and far more statistically powerful than the single-event test above. If the model requires Λ>0 as assumed in Paper 2, only the positive arm applies: Λ ≲ 7.40×10⁻¹² m². |
| Prior art on the flat-space chain | ℹ️ The flat-space translation ω²=k²+αk⁴ → A₄ → GWTC-4.0 bound was published independently in **arXiv:2607.17431** (Araújo Filho, Silva, Heidari, Zhu, Lobo & Bezerra, *Gravitational wave propagation in Hořava–Lifshitz gravity*, July 2026), which also derives the source-side generation phase and chirp corrections. That work is entirely flat-space: a search of its source contains no Kerr, photon-ring, shadow, superradiance, QNM or ZAMO content, so Paper 2's curved-background sector is unaffected. Note also that α=4 is identified in the GWTC-4.0 text itself as the case corresponding to Hořava–Lifshitz and extra-dimensional theories. The contribution claimed here is the Hamiltonian formulation on a curved background and the explicit sector separation below — **not** the flat-space GW bound. |
| **Critical scope note — sectors, not dimensions** | Λ_NH (near-horizon, Paper 2, dimension **[L²]**) and Λ_GW (propagation, Paper 3, dimension **[L³/T]**) are related by a single factor of *c*: **Λ_NH = Λ_GW / c**. This follows because Paper 2's Hamiltonian H=½[g^μν k_μk_ν+Λ(k_loc^ZAMO)⁴] reduces in the flat limit (α→1, k_loc→k) to ω²=c²k²(1+Λk²), whose k² coefficient carries units of length squared. An earlier version of this README stated the two were dimensionally incompatible with no known connection; **that note is superseded.** However, the two remain **physically distinct parameters**: Paper 2's Λ acts on photon trajectories (photon ring, shadow chromaticity), while the GW bound constrains the **graviton** sector. Lorentz-violating dispersion in the photon and graviton sectors are independent parameters, and identifying them requires a universality assumption this framework does not postulate. The photon-sector bound is Λ < 1.4421×10⁻⁵³ m² (Fermi-LAT, Paper 1) — 41 orders of magnitude tighter than the graviton-sector bound, a gap fully accounted for by the E²·D scaling of the quartic term (GeV photons versus ~10⁻¹² eV gravitons). **The GW results constrain Λ_GW only; they say nothing about Paper 2's photon-ring or QNM predictions.** See `paper3/PAPER3_VALIDATION_STATUS.md`. |
| Path-dependence test (`paper3/gw/path_test/`) | ❌ **Not feasible with current data — negative result, reported rather than removed.** A test for whether a GW anomaly statistic correlates with line-of-sight path properties (Galactic dark-matter and baryonic column densities, angular distance to solar-system bodies), keeping the anomaly statistic and the path variable strictly separate and controlling for SNR. The regression machinery is validated (null p<0.05 rate 0.058; unbiased slope recovery; false-positive rate 0.98 *without* the SNR covariate versus 0.093 with it). The test nonetheless cannot be run: across 174 events the reliability λ = 1 − ⟨σ_x²⟩/Var(x) is **0 for all three path axes** — sky-localisation uncertainty (tens to hundreds of deg²) explains the entire between-event scatter in smooth Galactic variables. The best sub-sample (baryon column, n=122) gives expected σ_b ≈ 0.53 against a ≈0.3 usefulness threshold. Closed at the design stage, before any anomaly statistic was computed. The code and the sensitivity table are kept so the calculation need not be repeated. |
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
│   │   ├── matched_filter/              # VALIDATED phase-domain approach (Stages 1-4)
│   │   │   ├── waveform.py              # GR + Lambda phase-domain waveform model (canonical
│   │   │   │                            #   lambda_phase_correction lives here)
│   │   │   ├── lalsim_waveform.py       # LALSuite waveform path; imports the canonical phase correction
│   │   │   ├── convention_check.py      # Validates lambda_phase_correction against the GWTC-4.0 MDR
│   │   │   │                            #   formula; documents the Λ<->A_4 conversion and sector caveat
│   │   │   ├── likelihood.py            # matched-filter likelihood, grid search
│   │   │   ├── synthetic_injection.py   # frequency-domain injection generator
│   │   │   ├── recovery_test.py         # Stage 1 — synthetic noise — PASS
│   │   │   ├── stage2_real_noise_recovery.py       # Stage 2 — real PSD, Gaussian noise — PASS
│   │   │   ├── stage3_real_strain_validation.py    # Stage 3 — real H1 strain — PASS (A+B,C); GW150914 exploratory
│   │   │   ├── stage4_coherent_h1l1_validation.py  # Stage 4 — coherent H1+L1 — PASS (4C,4D); PN-systematic found
│   │   │   ├── STAGE3_FINAL_STATUS.md   # Full Stage 3 audit trail
│   │   │   ├── STAGE4_FINAL_STATUS.md   # Full Stage 4 audit trail
│   │   │   ├── stage6E3H-R_v2.py        # Stage 6+ — highpass/Tukey conditioning + alias-safe Λ grid
│   │   │   ├── gw150914_lambda_onsource_test.py  # GW150914 propagation-Λ test: z=-0.43σ vs off-source null
│   │   │   ├── A1v3_zamo_v04.py         # Near-horizon photon-ring solver (Paper 2 quartic Hamiltonian)
│   │   │   ├── A2_chromatic_shadow_generalized.py  # Photon-ring solver generalized to dispersion power n=1..4
│   │   │   ├── gr_vs_lambda_search_auc.py  # GR-only vs Λ-search matched-filter AUC comparison
│   │   │   ├── triaxis_analyzer_v4.py / triaxis_analyzer_v5.py  # Template-free H1/L1 cross-correlation detector
│   │   │   └── injection_recovery_lambda.py  # Λ-deformed signal injection through the TriAxis detector
│   │   └── path_test/                   # NEGATIVE RESULT — path-dependence test, not feasible (see status table)
│   │       ├── README.md                # Why the test cannot be run: sensitivity table and reliability λ
│   │       ├── fetch_catalog.py         # GWOSC catalogue -> events.csv (version merge, FAR cut)
│   │       ├── fetch_skymaps.py         # Per-catalogue skymap archives -> per-event FITS
│   │       ├── galactic_column.py       # NFW dark-matter and exponential-disc column densities
│   │       ├── sky_path_vars.py         # Skymap posterior -> marginalised path variables
│   │       ├── path_stats.py            # Weighted regression, permutation null, upper bounds
│   │       ├── selftest.py              # Validation of the regression machinery
│   │       └── run_test.py              # Joins anomaly and path CSVs, reports all three axes
│   └── figures/                         # Generated plots
│
│   PAPER3_VALIDATION_STATUS.md          # Independently re-verified status of every test suite above,
│                                         # including the Λ_NH vs Λ_GW sector separation
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

# Cross-check the Lambda phase correction against the published GWTC-4.0
# modified-dispersion formula (expects ratio 1.0000 at all redshifts)
python paper3/gw/matched_filter/convention_check.py

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
| **GW / tests-of-GR groups**                | The Λ↔A₄ correspondence is verified against the GWTC-4.0 source formula and reproducible in one command (`convention_check.py`). If you use a different (1+z) convention or a different α, that script will report the mismatch as a fitted power of (1+z) rather than failing silently.                                                        |
| **Numerical relativity / computational physics reviewers** | Run `paper3_h_convergence_test.py` to independently verify the spatial convergence claim. The test integrates the actual fourth-order wave PDE via explicit leapfrog with a genuinely discretized biharmonic operator (not the spectral-exact shortcut used elsewhere in the repo) — this is the appropriate test for scrutinizing whether $\Lambda$ is a numerical artifact of grid spacing. |

## Honesty statement

This is a **falsifiable framework**, not a confirmed law of nature. A
single value of $\Lambda$ that is *simultaneously* consistent with
Fermi-LAT, EHT, GW, and BEC data would be evidence of a shared underlying
mechanism. Inconsistent values would falsify the universal-$\Lambda$
hypothesis while leaving each domain-specific dispersion law intact.
Note that this universality is a hypothesis under test, not an
assumption of the framework: the photon-sector and graviton-sector
bounds quoted above are independent constraints and are not currently
required to agree.

This repository also documents its own audit trail: two of the three
originally-proposed condensed-matter mappings (Dirac materials,
photonic crystals) did not survive scrutiny in their original form and
are reported here as negative/incomplete results rather than removed
silently. The spatial-convergence claim underwent the same process: an
earlier version of this repository asserted grid convergence based only
on a spectral-exact shortcut and a temporal-resolution test, neither of
which could have detected a genuine spatial-discretization problem had
one existed. `paper3_h_convergence_test.py` closes that specific gap by
testing the real leapfrog-integrated PDE instead. The path-dependence
test in `paper3/gw/path_test/` is a third such case: validated code, a
correctly posed question, and a demonstration that the measurement has
no leverage with current sky localisations. We consider negative
and corrected results as important to publish as positive ones.

## Paper 3 — Dispersion Validation

Paper 3 is currently treated as a constrained numerical and
experimental-validation study rather than as a claim that every
dispersive platform realizes the same Lambda model.

The central model is

    omega^2 = c^2 k^2 (1 + Lambda k^2)

with a linear low-k branch and a quartic correction. This is the same
convention used by `wave_equation_2D_solver.py`, by
`lambda_phase_correction` in `waveform.py`, and by the flat limit of the
Paper 2 Hamiltonian. (An earlier version of this README wrote the
right-hand side as `1 + 2 Lambda k^2` in this section only; that factor
of 2 was inconsistent with the rest of the repository and has been
removed.)

### Current validation status

| Component                           | Status                                    |
| ------------------------------------ | ----------------------------------------- |
| BEC Lambda mapping                  | Primary physical mapping                  |
| Lambda dispersion fitting           | Implemented                               |
| Synthetic Lambda recovery           | PASS                                      |
| Spatial (h) convergence, real PDE   | PASS, N=64-128 range only (see above)     |
| Dimensional audit                   | Implemented                               |
| Lambda_NH <-> Lambda_GW relation    | RESOLVED (factor of c); sectors still distinct |
| GWTC-4.0 conversion cross-check     | PASS (ratio 1.0000, no (1+z) residual)    |
| Fiber structural test               | PASS for supplied demonstration regime    |
| Fiber -> Lambda identification      | REFUSED                                   |
| Dirac independent Lambda validation | Not established                           |
| Path-dependence test                | NOT FEASIBLE (no leverage; see status table) |
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

For the GWTC-4.0 convention cross-check:

    python paper3/gw/matched_filter/convention_check.py

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

Work that should be cited alongside the GW propagation results here:

- Abac, A. G. *et al.* (LIGO Scientific, Virgo & KAGRA Collaborations) (2026). *GWTC-4.0: Tests of General Relativity. II. Parameterized Tests*. arXiv:2603.19020 — source of the A₄ bound used above.
- Araújo Filho, A. A., Silva, J. L. A., Heidari, N., Zhu, J., Lobo, I. P. & Bezerra, V. B. (2026). *Gravitational wave propagation in Hořava–Lifshitz gravity*. arXiv:2607.17431 — independent derivation of the flat-space ω²=k²+αk⁴ → A₄ translation.
- Mirshekari, S., Yunes, N. & Will, C. M. (2012). *Constraining generic Lorentz violation and the speed of the graviton with gravitational waves*. Phys. Rev. D 85, 024041.

## License

MIT License (see `LICENSE`). Free to use, modify, and redistribute with attribution.

## Contact

Dimitar Kretski — Center for Hydro- and Aerodynamics, BAS, Varna, Bulgaria
ORCID: [0000-0001-5108-2243](https://orcid.org/0000-0001-5108-2243)
