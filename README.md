# Λ-model — A Quartic Dispersion Framework

**Author:** Dimitar Kretski**ORCID:** [0000-0001-5108-2243](https://orcid.org/0000-0001-5108-2243)**Affiliation:** Center for Hydro- and Aerodynamics, Bulgarian Academy of Sciences, Varna, Bulgaria

* * *

## What this repository is for

This repository implements a one-parameter dispersion relation,

$$
\omega(k) = c\,k\,\sqrt{1 + \Lambda k^2},
$$

together with a validated numerical PDE solver for its wave equation, anda fitting tool that tests whether a real, measured $(k,\omega)$ datasetis consistent with this form.

**What a researcher can actually do with it, in five minutes:**

1. **Test your own dispersion data.** Run`lambda_experimental_validator.py --omega your_omega.csv --k your_k.csv`on any $(k,\omega)$ pairs you have — no domain assumptions required.You get a fitted $\Lambda$, its standard error, $R^2$, and an explicitsignificance test against $\Lambda=0$.
2. **Simulate propagation under this dispersion.** The 2D spectral PDEsolver (`wave_equation_2D_solver.py`) is validated to machine precisionin space and $O(1/N)$ in time (see Numerical status below) — use itdirectly if you need to propagate a field under a given $\Lambda$.The full leapfrog time-integration path (not just the spectral-exactshortcut) is separately validated for spatial convergence — see`paper3_h_convergence_test.py` and the status table.
3. **If you work with BEC:** the mapping $\Lambda=\xi^2/4$ is physicallyestablished (exact match to the Bogoliubov dispersion relation, not ananalogy). Plug in your healing length and sound speed and test itagainst your own Bragg-spectroscopy data directly. Two independentexperimental checks against real published BEC data are in`paper3/bec_validation/` — see the status table.
4. **If you work with Dirac materials or photonic dispersion:** therepository is explicit about which mappings are *not* currentlysupported, so you don't waste time chasing a claim that doesn't hold up(see Status table below). The fitting infrastructure still works onyour data — you supply the physical mapping.
5. **Extend it.** Adding a new domain mapping is one function(`lambda_from_<domain>(...)`) plus a CLI entry; the fitting, errorreporting, and significance-testing machinery is already there anddoes not need to be reimplemented.

## New here? Start with the one-line summary

* **If you have (k,ω) dispersion data**: run `lambda_experimental_validator.py` on it — see Quick start.
* **If you want the current honest bottom line**: `paper3/PAPER3_VALIDATION_STATUS.md` is the single mostup-to-date status document — start there, not the full table below.
* **If you're reviewing the GW/LIGO work specifically**: the GWTC-4.0 cross-check result(Λ ∈ [−7.24×10⁻³, +2.22×10⁻³] m³/s = [−2.41×10⁻¹¹, +7.40×10⁻¹²] m², consistent with GR)is the most statistically robust result in the repo. **Note that it constrains thegraviton sector only** — it does not bound Paper 2's photon-ring or shadow predictions.See the Critical scope note in the status table.
* **If you're reviewing the analog-gravity / laboratory work specifically**: the giant-vortexq=1/q=2 parameter-free match against real Švančara et al. resonance data(`paper3/PAPER3_ADDENDUM_BLIND_2D.md`, `paper3/STAGE6_5B4_3_ROOTB_FREEZE.md`) is thestrongest real-data, non-fitted result in this repository — stronger evidentiary statusthan any of the astrophysical bounds above, though scoped to this one laboratory systemonly (see Critical scope note).

## Status of results — read before citing

| Component | Status |
| --- | --- |
| Numerical PDE solver (spatial, spectral-exact shortcut) | ✅ Exact (each Fourier mode assigned its analytic $\omega(k)$; no discretization in this path) |
| Numerical PDE solver (spatial, **full leapfrog integration**) | ✅ Convergence order **2.0 confirmed** between N=64→128 (genuinely discretized Laplacian², not a shortcut). ⚠️ **N<64 unreliable** (19% error at N=32). ⚠️ **N>128 untested** (CFL cost scales as N⁴); do not assume convergence beyond this without re-testing. |
| Numerical PDE solver (temporal) | ✅ Validated $O(1/N)$ convergence |
| Discrete biharmonic operator $D_n^4$ | ✅ Validated $O(h^2)$ convergence |
| $\Lambda$-recovery from noisy data | ✅ Validated, sub-percent accuracy (analytic $\omega(k)$ input; see above for recovery from actual simulated/integrated fields) |
| Fermi-LAT bound (photon sector) | ✅ Audited: $\Lambda < 1.4421\times10^{-53}\,\mathrm{m^2}$ (corrected from a $3/2$ normalization error) |
| EHT sensitivity ceiling | ⚠️ Validated as a *sensitivity estimate*, not a fit to real M87\* data |
| GW forecast | ⚠️ Illustrative projection, not a fitted LIGO/Virgo/KAGRA constraint. Superseded by the matched-filter pipeline below. |
| GW matched-filter pipeline (Stages 1–4) | ✅ Validated architecture — Stage 4 (coherent H1+L1) passes off-source null trials and injection/recovery on real strain. A similar unconditioned-strain root cause (no highpass/Tukey window before PSD estimation) was independently found and fixed in the Stage 6+ pipeline below; whether it fully explains this specific PN-order systematic has not been directly re-tested on the Stage 3/4 code path. See `paper3/gw/matched_filter/STAGE4_FINAL_STATUS.md` for the original diagnosis. |
| GW propagation-dispersion pipeline (Stage 6+, `stage6E3H-R_v2.py`) | ✅ **Physically-conditioned pipeline.** Adds explicit highpass (0.9×f_low) + Tukey windowing before PSD/matched-filter (root cause of the earlier PN-order artifact — unconditioned strain gave noise-only matched-filter scores ~450–500, i.e. dominated by unfiltered low-frequency content, not the injected signal). Also fixes a Λ-grid phase-aliasing bug (grid must stay within ±half the aliasing period, ≈±0.57 for this band/redshift; the original ±20 grid wrapped ~35 times). Injection-recovery correlation on the corrected pipeline: 0.81 (n=50, real O1 noise). **GW150914 result:** Λ_on=−0.72, off-source null μ=+0.22, σ=2.21 (n=60 real off-source epochs), **z=−0.43σ — consistent with GR, no significant deviation.** This supersedes the Stage 3/4 exploratory point estimates above. Full status: `paper3/PAPER3_VALIDATION_STATUS.md`. |
| Near-horizon photon-ring solver, generalized to arbitrary dispersion power *n* | ✅ `A1v3_zamo_v04.py` / `A2_chromatic_shadow_generalized.py` generalize the Paper 2 quartic Hamiltonian (H=½[g^μν k_μk_ν+Λk_loc⁴], the *n*=1 case) to H_n=½[g^μν k_μk_ν+Λ_n k_loc^(2n+2)]. Regression-tested: *n*=1 reproduces Paper 2's published photon-ring coefficients (C_pro=0.6230, C_ret=11.495) to 4 significant figures. Derived and numerically confirmed: δb ∝ Λ_n·E^(2n) for each *n* (1–4 tested, exact match to predicted exponent). **Not yet applied to real observational data** — EHT radio-band (230 GHz) shown to give an uninformative bound for Sgr A* (photon energy E≈2×10⁻⁷⁶ in this framework's units, suppressing the Λ_n correction by ~150 orders of magnitude); a gamma-ray-band channel would be needed for a meaningful near-horizon constraint. |
| External cross-check: LVK GWTC-4.0 α=4 modified-dispersion-relation bound | ✅ The propagation phase ΔΨ(f,Λ)=−4π³ΛK(z)f³/c³ is mathematically identical to the standard LVK α=4 MDR parametrization (Mirshekari–Yunes–Will, E²=p²c²+A_αp^αc^α). **Verified line-by-line against the GWTC-4.0 arXiv source** (arXiv:2603.19020, `paperII__tests_of_propagation.tex` line 25): ratio 1.0000 at every redshift tested, fitted (1+z) exponent 0.000 — no residual redshift systematic, so the bound conversion needs no correction factor. Distance measures agree to machine precision (D₄ = c·K(z)/(1+z)³). Exact correspondence: **Λ = ħ²c³A₄** (equivalently h²c³A₄/4π²). Reproducible via `paper3/gw/matched_filter/convention_check.py`. Applying the published GWTC-4.0 combined bound — A₄ ∈ [−620, +190] eV⁻² at 90% credibility, two-sided (Q_GR = 82.5%), from 83 events (43 from GWTC-3.0 + 40 new from O4a; FAR ≤ 10⁻³/yr; BBH only, BNS/NSBH excluded from the MDR analysis) — gives **Λ ∈ [−7.24×10⁻³, +2.22×10⁻³] m³/s = [−2.41×10⁻¹¹, +7.40×10⁻¹²] m²**, consistent with Λ=0 and far more statistically powerful than the single-event test above. If the model requires Λ>0 as assumed in Paper 2, only the positive arm applies: Λ ≲ 7.40×10⁻¹² m². |
| **Giant-vortex analog QNM: experimental validation (q=1, q=2) and branch-tracking** | ✅ **q=1 and q=2 parameter-free experimental validation — the strongest real-data result in this repository.** Flow parameters (C, Ω, γ, h₀, g, r_B) taken directly from the published Table SI of Švančara et al. (giant quantum vortex, superfluid-helium analog-gravity experiment) — **not fitted to the resonance frequencies** — predict complex QNM poles compared directly against the experiment's four counter-rotating resonances (q=1..4), m=−21 to −4. **q=1:** mean relative error 0.021% (max 0.036%) across all 18 tested azimuthal numbers. **q=2 (Root-B branch):** 0.011–0.060% where continuation-validated (m=−19 to −11), degrading smoothly to ~1% at m=−4 — independently confirmed to be a genuine property of the branch, not a numerical artifact (see Root-B freeze below). Independently reproduced by (i) a blind, experiment-blind 2D complex-plane discovery scan, and (ii) a custom Newton solver built without `scipy.optimize.root`/`minimize`, no continuation seeding, no experimental data — 14/14 independent local solves converged to the frozen roots, max residual 3.2×10⁻¹¹, max deviation 5.5×10⁻¹⁰ Hz. **q=3/q=4 remain unresolved** — neither continuation from q=2 nor the blind discovery scan locates an unambiguous experimental counterpart; one additional strongly-damped mathematical pole is numerically validated but experimentally unassigned. This is reported as an open prediction, not a detection failure. Full data, methodology, and explicit claim boundaries: `paper3/PAPER3_ADDENDUM_BLIND_2D.md` (q=1/q=2 tables, blind-discovery cross-check) and `paper3/STAGE6_5B4_3_ROOTB_FREEZE.md` (continuation + independent-solver + experimental-comparison freeze). **Scope:** this validates the Λ-model's dispersion relation in this specific laboratory analog system only — it does not by itself establish a Kerr black-hole mapping, astrophysical relevance, or universality of Λ across sectors (see Critical scope note). |
| **BEC experimental validation (Level A / Level B)** | ⚠️/❌ **Two independent real-data checks beyond the theoretical BEC mapping below — both inconclusive-to-null, reported for completeness.** *Level A* (`paper3/bec_validation/level_a_ozeri2002/`): digitized Ozeri, Steinhauer, Katz & Davidson (PRL 88, 220401, 2002) homogeneous-BEC phonon energies. Convention-audited (ξ_Paper3=√2·ξ_Ozeri, Λ=ξ²/4=ξ_Ozeri²/2, verified by independent re-derivation from the paper's own Bogoliubov formula). Primary fit (N=4) is inconclusive, dominated by an author-flagged systematic outlier at kξ=0.32; sensitivity fit excluding it (N=3) is consistent with the model's Bogoliubov-equivalent prediction (β_fit=0.90±0.71 vs β_pred=0.5, 0.57σ) and AIC/BIC mildly prefer the fixed prediction over a free fit — but N=3 (1 degree of freedom) is statistically weak and this is **not** reported as a confirmation. *Level B* (`paper3/bec_validation/level_b_guo2021/`): real cavity-QED collective-mode dispersion (Guo et al., Nature 599, 211, 2021; Harvard Dataverse DOI:10.7910/DVN/LGT5O6, CC0) used as a phenomenological negative control (the measured system is a cavity-mediated polariton, not a free BEC phonon, so this is explicitly **not** a Λ-model test) — clean null result, no significant quartic curvature detected (Δb/σ_b≈−1.6, ΔAIC≈ΔBIC≈−0.6, robust to exclusion of k=0). Full provenance, digitization method (with independent cross-check against the source paper's own quoted values), and claim boundaries in each module's `README.md`/`provenance.md`. |
| Prior art on the flat-space chain | ℹ️ The flat-space translation ω²=k²+αk⁴ → A₄ → GWTC-4.0 bound was published independently in **arXiv:2607.17431** (Araújo Filho, Silva, Heidari, Zhu, Lobo & Bezerra, *Gravitational wave propagation in Hořava–Lifshitz gravity*, July 2026), which also derives the source-side generation phase and chirp corrections. That work is entirely flat-space: a search of its source contains no Kerr, photon-ring, shadow, superradiance, QNM or ZAMO content, so Paper 2's curved-background sector is unaffected. Note also that α=4 is identified in the GWTC-4.0 text itself as the case corresponding to Hořava–Lifshitz and extra-dimensional theories. The contribution claimed here is the Hamiltonian formulation on a curved background and the explicit sector separation below — **not** the flat-space GW bound. A fuller audit (α↔A₄ identification verified against the paper's own HTML source, independent SI-unit re-derivation, propagation-phase structural check, and an explicit list of what is and is not used from the source paper given an independent reliability caveat on its luminosity/chirp section) is in `paper3/horava_lifshitz_crosscheck.md`. |
| **Critical scope note — sectors, not dimensions** | Λ_NH (near-horizon, Paper 2, dimension **[L²]**) and Λ_GW (propagation, Paper 3, dimension **[L³/T]**) are related by a single factor of *c*: **Λ_NH = Λ_GW / c**. This follows because Paper 2's Hamiltonian H=½[g^μν k_μk_ν+Λ(k_loc^ZAMO)⁴] reduces in the flat limit (α→1, k_loc→k) to ω²=c²k²(1+Λk²), whose k² coefficient carries units of length squared. An earlier version of this README stated the two were dimensionally incompatible with no known connection; **that note is superseded.** However, the two remain **physically distinct parameters**: Paper 2's Λ acts on photon trajectories (photon ring, shadow chromaticity), while the GW bound constrains the **graviton** sector, and the giant-vortex analog-system Λ above is a third, separately-realized flat-space quartic coefficient with no derivation connecting it to either. Lorentz-violating dispersion in the photon, graviton, and analog-system sectors are independent parameters, and identifying any two of them requires a universality assumption this framework does not postulate. The photon-sector bound is Λ < 1.4421×10⁻⁵³ m² (Fermi-LAT, Paper 1) — 41 orders of magnitude tighter than the graviton-sector bound, a gap fully accounted for by the E²·D scaling of the quartic term (GeV photons versus ~10⁻¹² eV gravitons). **The GW results constrain Λ_GW only; they say nothing about Paper 2's photon-ring or QNM predictions, and nothing about the giant-vortex or BEC laboratory results.** See `paper3/PAPER3_VALIDATION_STATUS.md`, "Permanent boundaries." |
| Path-dependence test (`paper3/gw/path_test/`) | ❌ **Not feasible with current data — negative result, reported rather than removed.** A test for whether a GW anomaly statistic correlates with line-of-sight path properties (Galactic dark-matter and baryonic column densities, angular distance to solar-system bodies), keeping the anomaly statistic and the path variable strictly separate and controlling for SNR. The regression machinery is validated (null p<0.05 rate 0.058; unbiased slope recovery; false-positive rate 0.98 *without* the SNR covariate versus 0.093 with it). The test nonetheless cannot be run: across 174 events the reliability λ = 1 − ⟨σ_x²⟩/Var(x) is **0 for all three path axes** — sky-localisation uncertainty (tens to hundreds of deg²) explains the entire between-event scatter in smooth Galactic variables. The best sub-sample (baryon column, n=122) gives expected σ_b ≈ 0.53 against a ≈0.3 usefulness threshold. Closed at the design stage, before any anomaly statistic was computed. The code and the sensitivity table are kept so the calculation need not be repeated. |
| BEC mapping ($\Lambda=\xi^2/4$) | ✅ Physically established (exact Bogoliubov coefficient match). See "BEC experimental validation" above for real-data checks against this mapping. |
| Dirac mapping ($\Lambda=(1-\eta^2)v_F^2/4$) | ❌ **Speculative, not supported** by the cited literature (Fu 2009 describes an anisotropic $k^6$ effect, not this isotropic $k^4$ form) |
| Photonic mapping ($\Lambda=\beta_4^k/c^2$) | ⚠️ Dimensionally correct and properly derived, but $\beta_4^k$ is **not yet connected** to any standard measurable dispersion coefficient |

**Any $\Lambda$ value extracted from simulated (not purely analytic) datamust state the grid resolution used.** Results from N<64 spatial gridsshould not be trusted; see `paper3_h_convergence_test.py`.

See `paper3/paper3_final.tex` for the full derivations and the honestystatement on what is proved, audited, forecast, or retracted.

## Repository structure

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
    │   ├── bec_validation/                  # Real-data BEC experimental checks (Level A/B, see status table)
    │   │   ├── level_a_ozeri2002/           # Homogeneous-BEC phonon dispersion vs Ozeri et al. 2002 (digitized)
    │   │   │   ├── README.md
    │   │   │   ├── digitized_fig5.csv
    │   │   │   ├── convention_audit.md      # xi_Paper3 <-> xi_Ozeri healing-length convention derivation
    │   │   │   ├── fit_bogoliubov_lambda.py
    │   │   │   ├── results.csv
    │   │   │   └── provenance.md
    │   │   └── level_b_guo2021/             # Cavity-QED negative control vs Guo et al. 2021 (Harvard Dataverse)
    │   │       ├── README.md
    │   │       ├── Fig4_dispersion.csv
    │   │       ├── fit_quartic.py
    │   │       ├── results.csv
    │   │       └── provenance.md
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
    │   │   ├── _archive/qnm_branch_work/    # Giant-vortex analog QNM branch-tracking + experimental validation (Stage 5I-5K, 6.5B4)
    │   │   │   ├── stage5I_1_2_dispersion_radial_p.py   # Base dispersion/radial-p solver + light-ring finder
    │   │   │   ├── stage5I_3_resonance_v2.py            # QNM resonance search (real + complex refinement)
    │   │   │   ├── stage5I_4_full_m_scan_v6_FIXED.py    # Full 18-m QNM scan, NM->hybr validated pipeline
    │   │   │   ├── generalized_branch_resolution.py     # Assigns model roots to experimental q=1/q=2 branches
    │   │   │   ├── forensic_continuation_comparison.py  # Independent-reimplementation cross-check of p(r) continuation
    │   │   │   ├── stage5I_24_coverage_sensitivity_finding.py  # Found "Root B" -- 2nd genuine root at m=-11..-14
    │   │   │   ├── stage5I_25_expanded_discovery_scan.py       # Wide-grid full-m scan confirming Root B, no m=-15 pole either width
    │   │   │   ├── stage5I_26_singleton_im_expanded.py         # Expanded Im-bound retry, resolved boundary-vs-genuine-absence for singleton m
    │   │   │   ├── stage5I_27_q3_continuation_search.py        # Experiment-blind q=3 continuation search from Root-B (unresolved)
    │   │   │   ├── stage5I_28_blind_2d_discovery.py             # Blind 2D complex-plane root discovery at m=-14 (independent q=1/q=2 cross-check)
    │   │   │   ├── stage5I_29_rootB_continuation_toward_m11.py # Root-B continuation m=-14 -> -11 (fills integer-m gap)
    │   │   │   ├── stage5I_30_rootB_continuation_toward_m4.py  # Root-B continuation m=-14 -> -4 (extended range)
    │   │   │   ├── stage5I_31_rootB_resume_from_m5p6.py        # Resume continuation from a checkpoint (session-interruption recovery)
    │   │   │   ├── stage6_5B4_2_INDEPENDENT_root_reproduction.py  # Custom Newton solver, independent of scipy/continuation/experiment — 14/14 PASS
    │   │   │   ├── stage5J_branch_continuation.py        # Root A continuation + pre-registered fold/breakdown classifier
    │   │   │   ├── stage5J_rootB_continuation.py         # Root B continuation, same classifier (PRIMARY working branch)
    │   │   │   ├── stage5K_1_observable_extraction.py    # Root A: f_R(m), tau(m), Q(m) (diagnostic)
    │   │   │   ├── stage5K_1_observable_extraction_rootB.py  # Root B: f_R(m), tau(m), Q(m) (primary)
    │   │   │   ├── stage5K_3_ringdown_injection_recovery.py / _rootB.py  # Noiseless recovery sanity check, both roots
    │   │   │   └── stage5K_5_noise_robustness.py / _rootB.py            # SNR-sweep recovery robustness, both roots
    │   │   ├── STAGE5J_5K_STATUS.md         # Frozen status writeup -- Root A vs Root B, branch-tracking detail
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
    │                                         # including the Λ_NH vs Λ_GW sector separation and the
    │                                         # "Permanent boundaries" of what this work does not claim
    │   PAPER3_ADDENDUM_BLIND_2D.md          # Giant-vortex q=1/q=2 parameter-free experimental validation:
    │                                         # full m-by-m tables, blind-2D-discovery cross-check, claim boundaries
    │   PAPER3_ADDENDUM_BEC_LEVEL_B.md       # Guo et al. 2021 negative-control summary (see bec_validation/level_b_guo2021/)
    │   STAGE6_5B4_3_ROOTB_FREEZE.md         # Root-B branch freeze: continuation + independent-solver +
    │                                         # experimental-comparison, combined into one evidentiary record
    │   horava_lifshitz_crosscheck.md        # External theoretical cross-check audit (arXiv:2607.17431):
    │                                         # verified α<->A4 identification, independent SI-unit and
    │                                         # propagation-phase checks, explicit exclusions
    │
    └── examples/
        ├── example_BEC.py                   # Lambda = xi^2/4 — established mapping
        ├── example_dirac.py                 # SPECULATIVE — raises a runtime warning
        └── example_photonic.py              # Corrected derivation; not yet measurable

## Quick start

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
    
    # Giant-vortex analog QNM: q=1/q=2 experimental validation + branch-tracking.
    # q=1/q=2 are parameter-free-validated against real data (see status table
    # and PAPER3_ADDENDUM_BLIND_2D.md); q=3/q=4 remain open/unresolved.
    # Root B is the primary working branch; Root A is a retained diagnostic.
    cd paper3/gw/matched_filter/_archive/qnm_branch_work
    python stage5J_rootB_continuation.py               # Root B (primary) continuation + classification
    python stage5K_1_observable_extraction_rootB.py     # Root B: f_R(m), tau(m), Q(m)
    python stage5K_3_ringdown_injection_recovery_rootB.py  # Root B: noiseless recovery sanity check
    python stage5K_5_noise_robustness_rootB.py          # Root B: SNR-sweep recovery robustness
    python stage5I_28_blind_2d_discovery.py             # Independent, experiment-blind cross-check at m=-14
    python stage6_5B4_2_INDEPENDENT_root_reproduction.py  # Independent Newton-solver reproduction of frozen Root-B
    # Root A (diagnostic/secondary) — same scripts without the _rootB suffix
    
    # BEC experimental validation (real published data, see status table)
    python paper3/bec_validation/level_a_ozeri2002/fit_bogoliubov_lambda.py  # vs Ozeri et al. 2002
    python paper3/bec_validation/level_b_guo2021/fit_quartic.py             # vs Guo et al. 2021 (negative control)
    
    # Worked examples (each prints its own validity status)
    python examples/example_BEC.py         # established mapping
    python examples/example_dirac.py       # speculative — reads its own warning
    python examples/example_photonic.py    # corrected, not yet measurable

## What researchers in each field can do

| Group | What you can do |
| --- | --- |
| **Anyone with $(k,\omega)$ data** | Run `lambda_experimental_validator.py` directly — no domain assumptions needed, just numbers. |
| **Theorists** | Inspect the covariant derivation in `paper3_final.tex`, verify the wave equation, propose new metric mappings. |
| **BEC experimentalists** | Supply your healing length $\xi$ and sound speed $c_s$; test $\Lambda=\xi^2/4$ against your measured dispersion. This mapping is physically established, not speculative. See `paper3/bec_validation/` for two worked examples against real published data (one inconclusive-but-consistent, one a clean negative control) and the convention audit needed to compare your own $\xi$ definition correctly. |
| **Dirac-material / ARPES groups** | The fitting tool works on your data, but do **not** treat $\Lambda=(1-\eta^2)v_F^2/4$ as a prediction of this framework — see the audit in `paper3_final.tex` §5.2 and the runtime warning in `lambda_from_dirac()`. If you can derive a literature-supported quartic (not $k^6$) correction for your material, we would like to hear about it. |
| **Photonic-crystal / fiber-optics groups** | $\Lambda=\beta_4^k/c^2$ is now correctly derived from the model, but $\beta_4^k$ is not yet connected to a standard telecom $\beta_4$ or any other directly-measured quantity — see `paper3_final.tex` §5.3 for what's missing. Contributions on this specific gap are welcome. |
| **Analog-gravity / cold-atom & superfluid groups** | The giant-vortex q=1/q=2 result (`PAPER3_ADDENDUM_BLIND_2D.md`) is a template for testing this dispersion relation against any resonance/QNM spectrum from your own system, with explicit method (parameter-free, blind-discovery cross-checked, independent-solver reproduced) and explicit claim-boundary language you're welcome to reuse or challenge. |
| **GW / tests-of-GR groups** | The Λ↔A₄ correspondence is verified against the GWTC-4.0 source formula and reproducible in one command (`convention_check.py`). If you use a different (1+z) convention or a different α, that script will report the mismatch as a fitted power of (1+z) rather than failing silently. |
| **Numerical relativity / computational physics reviewers** | Run `paper3_h_convergence_test.py` to independently verify the spatial convergence claim. The test integrates the actual fourth-order wave PDE via explicit leapfrog with a genuinely discretized biharmonic operator (not the spectral-exact shortcut used elsewhere in the repo) — this is the appropriate test for scrutinizing whether $\Lambda$ is a numerical artifact of grid spacing. |

## Honesty statement

This is a **falsifiable framework**, not a confirmed law of nature. Asingle value of $\Lambda$ that is *simultaneously* consistent withFermi-LAT, EHT, GW, giant-vortex, and BEC data would be evidence of a shared underlyingmechanism. Inconsistent values would falsify the universal-$\Lambda$hypothesis while leaving each domain-specific dispersion law intact.Note that this universality is a hypothesis under test, not anassumption of the framework: the photon-sector, graviton-sector, and laboratory-analog-systembounds quoted above are independent constraints and are not currentlyrequired to agree — and no attempt to unify them is currently claimed as established;see the "Permanent boundaries" section of `PAPER3_VALIDATION_STATUS.md`.

This repository also documents its own audit trail: two of the threeoriginally-proposed condensed-matter mappings (Dirac materials,photonic crystals) did not survive scrutiny in their original form andare reported here as negative/incomplete results rather than removedsilently. The spatial-convergence claim underwent the same process: anearlier version of this repository asserted grid convergence based onlyon a spectral-exact shortcut and a temporal-resolution test, neither ofwhich could have detected a genuine spatial-discretization problem hadone existed. `paper3_h_convergence_test.py` closes that specific gap bytesting the real leapfrog-integrated PDE instead. The path-dependencetest in `paper3/gw/path_test/` is a third such case: validated code, acorrectly posed question, and a demonstration that the measurement hasno leverage with current sky localisations. The BEC Level-A/B experimentalchecks continue this practice: one is reported as statistically weak-but-consistentrather than a confirmation, and the other as a clean null result rather than beingomitted for not supporting the model. We consider negativeand corrected results as important to publish as positive ones.

## Paper 3 — Dispersion Validation

Paper 3 is currently treated as a constrained numerical andexperimental-validation study rather than as a claim that everydispersive platform realizes the same Lambda model.

The central model is omega^2 = c^2 k^2 (1 + Lambda k^2)

with a linear low-k branch and a quartic correction. This is the sameconvention used by `wave_equation_2D_solver.py`, by`lambda_phase_correction` in `waveform.py`, and by the flat limit of thePaper 2 Hamiltonian. (An earlier version of this README wrote theright-hand side as `1 + 2 Lambda k^2` in this section only; that factorof 2 was inconsistent with the rest of the repository and has beenremoved.)

### Current validation status

| Component | Status |
| --- | --- |
| BEC Lambda mapping (theoretical) | Primary physical mapping — established |
| BEC Lambda mapping (experimental, real data) | Level A: inconclusive/weak-but-consistent (N=3-4); Level B: clean null (negative control, different system) |
| Lambda dispersion fitting | Implemented |
| Synthetic Lambda recovery | PASS |
| Spatial (h) convergence, real PDE | PASS, N=64-128 range only (see above) |
| Dimensional audit | Implemented |
| Lambda_NH <-> Lambda_GW relation | RESOLVED (factor of c); sectors still distinct |
| GWTC-4.0 conversion cross-check | PASS (ratio 1.0000, no (1+z) residual) |
| Hořava-Lifshitz external theoretical cross-check | PASS (α=A₄ identification verified from source; independent SI-unit and propagation-phase re-derivation) — theoretical cross-check, not independent observational validation (same GWTC-4.0 dataset) |
| Giant-vortex q=1/q=2 experimental validation | PASS — parameter-free, real data, independently cross-checked (blind discovery + independent solver) |
| Giant-vortex q=3/q=4 | UNRESOLVED — open prediction, not established |
| Fiber structural test | PASS for supplied demonstration regime |
| Fiber -> Lambda identification | REFUSED |
| Dirac independent Lambda validation | Not established |
| Path-dependence test | NOT FEASIBLE (no leverage; see status table) |
| Full 2D QNM solver | Diagnosed but not yet physically complete |

### Fiber result

The supplied fiber event-horizon demonstration data give p = 3.99870888

for Omega ~ |Delta k|^p.

The local log-log slope is approximately 4 across the tested range.

This is consistent with a pure quartic regime and is structurallydifferent from the Lambda-model low-k structure.

Therefore the repository does not interpret this result as anindependent measurement of Lambda.

### Dimensional safeguard

The experimental fiber quantity beta4 has units s^4 / m.

Consequently, beta4 / c^2

has units s^6 / m^3,

not m^2.

The validator therefore refuses the identification Lambda = beta4 / (4 c^2)

for this quantity.

This refusal is intentional and is part of the validation methodology.

### Reproducibility

The Paper 3 dispersion tools can be run directly from the repository: python paper3/dispersion/lambda_experimental_validator.py

and python paper3/dispersion/fiber_event_horizon_structural_test.py

For the spatial-convergence audit specifically: python paper3/paper3_h_convergence_test.py

For the GWTC-4.0 convention cross-check: python paper3/gw/matched_filter/convention_check.py

For the giant-vortex q=1/q=2 experimental validation and independent cross-checks: see the Quick start commands above and `paper3/PAPER3_ADDENDUM_BLIND_2D.md` / `paper3/STAGE6_5B4_3_ROOTB_FREEZE.md`.

For the BEC experimental checks: python paper3/bec_validation/level_a_ozeri2002/fit_bogoliubov_lambda.py and python paper3/bec_validation/level_b_guo2021/fit_quartic.py

When no experimental input is supplied, the Lambda validator runs asynthetic self-test. Synthetic results must not be interpreted asexperimental evidence.

The fiber validator currently uses a demonstration dataset and clearlylabels it as such.

Real experimental data should replace the demonstration arrays beforemaking experimental claims.

## Citing this work

If you use this framework, please cite:

* Kretski, D. (2026). *A Hamiltonian Oscillator Extension of Wave Propagation in Schwarzschild Spacetime*. Zenodo. https://doi.org/10.5281/zenodo.22018715 (Paper 1, submitted to CQG, ref. CQG-117140)
* Kretski, D. (2026). *A Hamiltonian Dispersion Framework for Kerr Photon Rings, Frequency-Dependent Shadow Sensitivity, Superradiance, and Eikonal Quasinormal Modes*. Zenodo. https://doi.org/10.5281/zenodo.22051427 (Paper 2)
* Kretski, D. (2026). *A Universal Quartic Dispersion Framework: Numerical Validation, Multi-Messenger Time-of-Flight Bounds, and Condensed-Matter Analog Mappings*. (Paper 3, `paper3/paper3_final.tex` in this repository)

Work that should be cited alongside the GW propagation results here:

* Abac, A. G. *et al.* (LIGO Scientific, Virgo & KAGRA Collaborations) (2026). *GWTC-4.0: Tests of General Relativity. II. Parameterized Tests*. arXiv:2603.19020 — source of the A₄ bound used above.
* Araújo Filho, A. A., Silva, J. L. A., Heidari, N., Zhu, J., Lobo, I. P. & Bezerra, V. B. (2026). *Gravitational wave propagation in Hořava–Lifshitz gravity*. arXiv:2607.17431 — independent derivation of the flat-space ω²=k²+αk⁴ → A₄ translation.
* Mirshekari, S., Yunes, N. & Will, C. M. (2012). *Constraining generic Lorentz violation and the speed of the graviton with gravitational waves*. Phys. Rev. D 85, 024041.

Work that should be cited alongside the giant-vortex and BEC experimental results here:

* Smaniotto, F., Solidoro, V., Patrick, R., Švančara, P. *et al.* *Black-hole spectroscopy from a giant quantum vortex*. arXiv:2502.11209 — source of the giant-vortex EXP B q=1..4 resonance data used in `PAPER3_ADDENDUM_BLIND_2D.md` and `STAGE6_5B4_3_ROOTB_FREEZE.md`.
* Švančara, P. *et al.* *Rotating Curved Spacetime Signatures from a Giant Quantum Vortex*. arXiv:2308.10773 — the analog-gravity experimental setup and flow-parameter Table SI used for the parameter-free prediction.
* Ozeri, R., Steinhauer, J., Katz, N. & Davidson, N. (2002). *Direct Observation of the Phonon Energy in a Bose-Einstein Condensate by Tomographic Imaging*. Phys. Rev. Lett. 88, 220401 — source of the digitized Level-A BEC data in `paper3/bec_validation/level_a_ozeri2002/`.
* Guo, Y., Kroeze, R. M., Marsh, B. P., Gopalakrishnan, S., Keeling, J. & Lev, B. L. (2021). *An optical lattice with sound*. Nature 599, 211 — source of the Level-B cavity-QED negative-control data in `paper3/bec_validation/level_b_guo2021/` (data: Harvard Dataverse, DOI:10.7910/DVN/LGT5O6).

## License

MIT License (see `LICENSE`). Free to use, modify, and redistribute with attribution.

## Contact

Dimitar Kretski — Center for Hydro- and Aerodynamics, BAS, Varna, BulgariaORCID: [0000-0001-5108-2243](https://orcid.org/0000-0001-5108-2243)
