# Λ-model — A Quartic Dispersion Framework

**Author:** Dimitar Kretski**ORCID:** [0000-0001-5108-2243](https://orcid.org/0000-0001-5108-2243)**Affiliation:** Center for Hydro- and Aerodynamics, Institute of Metal Science, Equipment and Technologies "Acad. A. Balevski", Bulgarian Academy of Sciences, Varna, Bulgaria

* * *

## What this repository is for

This repository implements a one-parameter dispersion relation,

$$
\omega(k) = c\,k\,\sqrt{1 + \Lambda k^2}, \qquad\text{equivalently}\qquad \omega^2 = c^2k^2\left(1+\Lambda k^2\right),
$$

together with a validated numerical PDE solver for its wave equation, and afitting tool that tests whether a real, measured $(k,\omega)$ dataset isconsistent with this form. This is the only convention used in therepository (see "Convention note" in the Paper 3 section below).

**What a researcher can actually do with it, in five minutes:**

1. **Test your own dispersion data.** Run`lambda_experimental_validator.py --omega your_omega.csv --k your_k.csv`on any $(k,\omega)$ pairs you have — no domain assumptions required. Youget a fitted $\Lambda$, its standard error, $R^2$, and an explicitsignificance test against $\Lambda=0$.
2. **Simulate propagation under this dispersion.** The 2D spectral PDEsolver (`wave_equation_2D_solver.py`) is validated to machine precision inspace and $O(1/N)$ in time (see the status table) — use it directly if youneed to propagate a field under a given $\Lambda$. The full leapfrogtime-integration path (not just the spectral-exact shortcut) isseparately validated for spatial convergence — see`paper3_h_convergence_test.py`.
3. **If you work with BEC:** the mapping $\Lambda=\xi^2/4$, with the healinglength defined as $\xi=\hbar/(m c_s)$, is an exact match to theBogoliubov dispersion relation, not an analogy. If your $\xi$ uses theother common convention $\xi=\hbar/(\sqrt{2}\,m c_s)$, the same mappingreads $\Lambda=\xi^2/2$ — see the convention audit in`paper3/bec_validation/level_a_ozeri2002/`. Two experimental checksagainst real published BEC data are in `paper3/bec_validation/`.
4. **If you work with Dirac materials or photonic dispersion:** therepository is explicit about which mappings are *not* currentlysupported, so you don't waste time chasing a claim that doesn't hold up(see the status table). The fitting infrastructure still works on yourdata — you supply the physical mapping.
5. **Extend it.** Adding a new domain mapping is one function(`lambda_from_<domain>(...)`) plus a CLI entry; the fitting, errorreporting and significance-testing machinery is already there and doesnot need to be reimplemented.

## New here? Start with the one-line summary

* **If you have (k,ω) dispersion data:** run `lambda_experimental_validator.py`on it — see Quick start.
* **If you want the current honest bottom line:**`paper3/PAPER3_VALIDATION_STATUS.md` is the single most up-to-date andauthoritative status document — start there, not the full table below.Where this README and that document differ, that document wins.
* **If you're reviewing the GW/LIGO work specifically:** the GWTC-4.0cross-check (Λ ∈ [−7.24×10⁻³, +2.22×10⁻³] m³/s, consistent with GR) is themost statistically robust astrophysical result in the repository. Itconstrains the **graviton-propagation sector only** — it does not boundPaper 2's photon-ring or shadow predictions, and it says nothing about thelaboratory results.
* **If you're reviewing the analog-gravity / laboratory workspecifically:** the giant-vortex q=1/q=2 parameter-free comparison withreal Švančara et al. resonance data (`paper3/PAPER3_ADDENDUM_BLIND_2D.md`,`paper3/STAGE6_5B4_3_ROOTB_FREEZE.md`) is the strongest real-data,non-fitted result in this repository, scoped to that one laboratorysystem only.

## Status of results — read before citing

| Component | Status |
| --- | --- |
| Numerical PDE solver (spatial, spectral-exact shortcut) | ✅ Exact (each Fourier mode assigned its analytic $\omega(k)$; no discretization in this path) |
| Numerical PDE solver (spatial, **full leapfrog integration**) | ✅ Convergence order **2.0 confirmed** between N=64→128 (genuinely discretized Laplacian², not a shortcut). ⚠️ **N<64 unreliable** (19% error at N=32). ⚠️ **N>128 untested** (CFL cost scales as N⁴); do not assume convergence beyond this without re-testing. |
| Numerical PDE solver (temporal) | ✅ Validated $O(1/N)$ convergence |
| Discrete biharmonic operator $D_n^4$ | ✅ Validated $O(h^2)$ convergence |
| $\Lambda$-recovery from noisy data | ✅ Validated, sub-percent accuracy (analytic $\omega(k)$ input, synthetic only) |
| Fermi-LAT bound (photon sector) | ✅ Audited: $\Lambda < 1.4421\times10^{-53}\,\mathrm{m^2}$ (corrected from a $3/2$ normalization error) |
| GW forecast | ⚠️ Illustrative projection, not a fitted constraint. Superseded by the matched-filter pipeline and the GWTC-4.0 cross-check below. |
| GW matched-filter pipeline (Stages 1–4) | ✅ Validated architecture — Stage 4 (coherent H1+L1) passes off-source null trials and injection/recovery on real strain. A similar unconditioned-strain root cause (no highpass/Tukey window before PSD estimation) was independently found and fixed in the Stage 6+ pipeline below; whether it fully explains the Stage 3/4 PN-order systematic has not been directly re-tested on that code path. See `paper3/gw/matched_filter/STAGE4_FINAL_STATUS.md`. |
| GW propagation-dispersion pipeline (Stage 6+, `stage6E3H-R_v2.py`) | ✅ **Physically conditioned pipeline.** Adds explicit highpass (0.9×f_low) + Tukey windowing before PSD/matched-filter estimation (root cause of the earlier PN-order artifact) and fixes a Λ-grid phase-aliasing bug (the grid must stay within ±half the aliasing period, ≈±0.57 for this band/redshift; the original ±20 grid wrapped ~35 times). Injection-recovery correlation: 0.81 (n=50, real O1 noise). **GW150914:** Λ_on=−0.72, off-source null μ=+0.22, σ=2.21 (n=60 real off-source epochs), **z=−0.43σ — consistent with GR.** Supersedes the Stage 3/4 exploratory point estimates. |
| Near-horizon photon-ring solver, generalized to dispersion power *n* | ✅ Solver generalizes the Paper 2 quartic Hamiltonian (H=½[g^μν k_μk_ν+Λk_loc⁴], the *n*=1 case) to H_n=½[g^μν k_μk_ν+Λ_n k_loc^(2n+2)]. Regression-tested: *n*=1 reproduces Paper 2's published photon-ring coefficients (C_pro=0.6230, C_ret=11.495) to 4 significant figures; δb ∝ Λ_n·E^(2n) confirmed for *n*=1–4. **Not applied to real observational data.** The EHT radio band (230 GHz) gives an uninformative bound (the Λ_n correction is suppressed by ~150 orders of magnitude at that photon energy); a gamma-ray-band channel would be needed. |
| External cross-check: LVK GWTC-4.0 α=4 modified-dispersion-relation bound | ✅ The propagation phase ΔΨ(f,Λ)=−4π³ΛK(z)f³/c³ is mathematically identical to the standard LVK α=4 MDR parametrization (Mirshekari–Yunes–Will, E²=p²c²+A_αp^αc^α), verified against the GWTC-4.0 source (arXiv:2603.19020): ratio 1.0000 at every redshift tested, no residual (1+z) dependence, distance measures agreeing to machine precision (D₄=c·K(z)/(1+z)³). Correspondence: **Λ_GW = ħ²c³A₄**. Reproducible via `paper3/gw/matched_filter/convention_check.py`. The published combined bound, A₄ ∈ [−620, +190] eV⁻² (90% credibility, 83 BBH events), gives **Λ_GW ∈ [−7.24×10⁻³, +2.22×10⁻³] m³/s**, consistent with Λ=0. This is a translation of a published LVK result, not an independent reanalysis. |
| **Giant-vortex analog QNMs, q=1 (fundamental)** | ✅ **Parameter-free comparison with real data.** Flow parameters (C, Ω, γ, h₀, g, r_B) are taken from the published Table SI of Švančara et al. (arXiv:2308.10773) — **not fitted to the resonance frequencies** — and the predicted QNM poles are compared post hoc with the measured EXP B resonances (Smaniotto et al., arXiv:2502.11209), m=−21 to −4. Mean relative error 0.021% (max 0.036%) across all 18 azimuthal numbers. |
| **Giant-vortex analog QNMs, q=2 (first overtone, Root-B branch)** | ✅ Relative error 0.011–0.060% for m=−19 to −10, degrading smoothly and monotonically to 1.01% at m=−4. Continuation from the m=−14 anchor covers m=−14…−11 and m=−6…−4 (with documented gaps in the permanent record); m=−19…−16 and m=−10…−7 come from the discovery grid. The small-\\|m\\| degradation was tested and **is a genuine property of the branch, not a branch-selection artifact** (continued and discovery-grid values coincide at m=−6, −5, −4). Cross-checked by (i) an experiment-blind 2D complex-plane discovery scan at m=−14 and (ii) a custom damped-Newton solver written without scipy root/minimize, no continuation seeding and no experimental data: 14/14 solves converged to the frozen roots (max residual 3.2×10⁻¹¹). The Newton check reuses the same physical residual function, so it tests solver independence, not the physics. See `paper3/STAGE6_5B4_3_ROOTB_FREEZE.md`. |
| Giant-vortex q=2 near m=−15 | ⚠️ Continuation stops at m≈−14.95 at two step resolutions (Stage 5I-33); no validated pole at m=−15. A procedural continuation limit, **not** a demonstrated branch termination. |
| Giant-vortex q=3/q=4 | ⏳ **Unresolved.** Three lines of inquiry closed without a match: continuation from Root-B (5I-27), the discovery-grid candidates, and the strongly damped deep-pole branch (5I-32; best 3–6% at m=−14, not within 1% anywhere in the tested range). Reported as an open question, not a detection failure. |
| Giant-vortex statistical hypothesis test | 🕒 **Planned.** No χ² or significance value is reported, because measured uncertainties of the resonance frequencies are not yet available. They have been requested from the experimental group. |
| **BEC experimental checks (Level A / Level B)** | ⚠️/❌ Two real-data checks beyond the theoretical mapping, both inconclusive-to-null and reported for completeness. *Level A* (`paper3/bec_validation/level_a_ozeri2002/`): digitized Ozeri, Steinhauer, Katz & Davidson (PRL 88, 220401, 2002) homogeneous-BEC phonon energies, convention-audited (ξ_Paper3=√2·ξ_Ozeri). The primary fit (N=4) is inconclusive, dominated by an author-flagged systematic outlier at kξ=0.32; excluding it (N=3), the data are consistent with the Bogoliubov-equivalent prediction (β_fit=0.90±0.71 vs β_pred=0.5, 0.57σ) — statistically weak, **not** a confirmation. *Level B* (`paper3/bec_validation/level_b_guo2021/`): cavity-QED collective-mode dispersion (Guo et al., Nature 599, 211, 2021; Harvard Dataverse DOI:10.7910/DVN/LGT5O6, CC0) used as a phenomenological negative control (a cavity-mediated polariton, not a free BEC phonon, so **not** a Λ-model test) — clean null result (Δb/σ_b≈−1.6, ΔAIC≈ΔBIC≈−0.6). |
| BEC mapping ($\Lambda=\hbar^2/4m^2c_s^2$) | ✅ Exact Bogoliubov coefficient match; equals $\xi^2/4$ for $\xi=\hbar/(mc_s)$. |
| Prior art on the flat-space chain | ℹ️ The flat-space translation ω²=k²+αk⁴ → A₄ → GWTC-4.0 bound was published independently in **arXiv:2607.17431** (Araújo Filho et al., July 2026). That work contains no Kerr, photon-ring, shadow, superradiance, QNM or ZAMO content, so Paper 2's curved-background sector is unaffected. The contribution claimed here is the Hamiltonian formulation on a curved background and the explicit sector separation — **not** the flat-space GW bound. Audit: `paper3/horava_lifshitz_crosscheck.md`. |
| **Critical scope note — sectors, not a universal constant** | Λ_NH (near-horizon, Paper 2, **[L²]**) and Λ_GW (propagation, **[L³/T]**) satisfy Λ_NH = Λ_GW/c dimensionally, and the flat limit of Paper 2's Hamiltonian has the same ω²=c²k²(1+Λk²) form. This is a **dimensional-consistency observation, not a physical derivation**: no equation connects the near-horizon mechanism to the FLRW propagation mechanism. The photon sector, the graviton sector and the analog-system coefficients (giant vortex, BEC) are **independent parameters**; identifying any two of them would require a universality assumption this framework does not make. The GW results constrain Λ_GW only; they say nothing about Paper 2's photon-ring or QNM predictions, or about the laboratory results. See `paper3/PAPER3_VALIDATION_STATUS.md`, "Permanent boundaries." |
| Path-dependence test (`paper3/gw/path_test/`) | ❌ **Not feasible with current data — negative result, reported rather than removed.** Tests whether a GW anomaly statistic correlates with line-of-sight path properties (Galactic dark-matter and baryonic columns, angular distance to solar-system bodies), with the anomaly statistic and path variable kept separate and SNR controlled. The regression machinery is validated (null p<0.05 rate 0.058; unbiased slope recovery; false-positive rate 0.98 *without* the SNR covariate vs 0.093 with it). Across 174 events the reliability λ = 1 − ⟨σ_x²⟩/Var(x) is **0 for all three path axes** — sky-localisation uncertainty explains the entire between-event scatter. Closed at the design stage, before any anomaly statistic was computed. |
| Dirac mapping ($\Lambda=(1-\eta^2)v_F^2/4$) | ❌ **Speculative, not supported** by the cited literature (Fu 2009 describes an anisotropic $k^6$ effect, not this isotropic $k^4$ form) |
| Photonic mapping ($\Lambda=\beta_4^k/c^2$) | ⚠️ Dimensionally correct and properly derived, but $\beta_4^k$ is **not yet connected** to any standard measurable dispersion coefficient |
| Fiber event-horizon data | ✅ Quartic scaling p ≈ 3.999 found; identification with Λ **refused** (dimensional mismatch, see below) |

**Any $\Lambda$ value extracted from simulated (not purely analytic) datamust state the grid resolution used.** Results from N<64 spatial grids shouldnot be trusted; see `paper3_h_convergence_test.py`.

`paper3/paper3_final.tex` is an earlier draft (August 2026) containing thefull derivations. Its title and framing predate the sector separation above;for the current scope, `PAPER3_VALIDATION_STATUS.md` takes precedence.

## Repository structure

    Lambda-model/
    ├── README.md
    ├── CONTRIBUTING.md
    ├── LICENSE
    ├── requirements.txt
    ├── examples/
    │   ├── example_BEC.py                   # Lambda = xi^2/4, xi = hbar/(m c_s) — established mapping
    │   ├── example_dirac.py                 # SPECULATIVE — raises a runtime warning
    │   └── example_photonic.py              # Corrected derivation; not yet measurable
    ├── fiber_structural_validation/         # Quartic-scaling test; Lambda identification refused
    │
    └── paper3/
        ├── PAPER3_VALIDATION_STATUS.md      # AUTHORITATIVE status, incl. "Permanent boundaries"
        ├── PAPER3_ADDENDUM_BLIND_2D.md      # Giant-vortex q=1/q=2 tables, blind 2D discovery,
        │                                    #   q=3/q=4 closure test (5I-32), claim boundaries
        ├── STAGE6_5B4_3_ROOTB_FREEZE.md     # Root-B freeze: continuation + independent solver +
        │                                    #   post-hoc experimental comparison
        ├── STAGE6_5B4_3_supplement_5I33_boundary.md  # Root-B continuation limit near m=-15
        ├── PAPER3_ADDENDUM_BEC_LEVEL_B.md   # Guo et al. 2021 negative-control summary
        ├── horava_lifshitz_crosscheck.md    # Audit of arXiv:2607.17431 (theoretical cross-check)
        │
        ├── wave_equation_2D_solver.py       # Core PDE solver, omega^2 = c^2 k^2 (1 + Lambda k^2)
        ├── paper3_Dn4_test.py               # Discrete quartic operator D_n^4
        ├── paper3_grid_convergence.py       # Temporal convergence + Lambda recovery (analytic input)
        ├── paper3_h_convergence_test.py     # Spatial convergence, REAL leapfrog PDE integration
        ├── lambda_experimental_validator.py # Fits Lambda from (k, omega) data; domain mappings
        ├── paper3_final.tex / .pdf          # Earlier full draft (August 2026)
        │
        ├── gwosc_chirp_dispersion_test.py               # legacy/invalidated — Hilbert extraction, 51.9σ bias
        ├── gwosc_injection_recovery_test.py             # diagnostic — proved pipeline bias (Λ=0 → Λ_fit=-1.91)
        ├── gwosc_extraction_ablation.py                 # diagnostic — bias localized to Hilbert-transform math
        ├── gwosc_frequency_extraction_comparison.py     # diagnostic — 5 extraction methods compared
        ├── gwosc_zero_crossing_injection_recovery.py    # diagnostic — zero-crossing also unstable
        │
        ├── dispersion/                      # Extended validation module (fiber structural test,
        │                                    #   real-data pipeline, test suite)
        ├── bec_validation/
        │   ├── level_a_ozeri2002/           # Ozeri et al. 2002 (digitized) + convention audit
        │   └── level_b_guo2021/             # Guo et al. 2021 negative control (Harvard Dataverse)
        │
        └── gw/
            ├── path_test/                   # NEGATIVE RESULT — path-dependence test not feasible
            │                                #   (skymaps are downloaded by fetch_skymaps.py and are
            │                                #    not stored in the repository)
            └── matched_filter/
                ├── waveform.py              # GR + Lambda phase-domain waveform (canonical
                │                            #   lambda_phase_correction)
                ├── lalsim_waveform.py       # LALSuite path; imports the canonical phase correction
                ├── convention_check.py      # Lambda <-> A_4 conversion vs the GWTC-4.0 MDR formula
                ├── likelihood.py            # Matched-filter likelihood, grid search
                ├── recovery_test.py         # Stage 1 — synthetic noise
                ├── stage2_real_noise_recovery.py       # Stage 2 — real PSD
                ├── stage3_real_strain_validation.py    # Stage 3 — real H1 strain
                ├── stage4_coherent_h1l1_validation.py  # Stage 4 — coherent H1+L1
                ├── STAGE3_FINAL_STATUS.md / STAGE4_FINAL_STATUS.md
                ├── stage6E3H-R_v2.py        # Stage 6+ — conditioning + alias-safe Lambda grid
                ├── gw150914_lambda_onsource_test.py    # GW150914: z = -0.43σ vs off-source null
                ├── A2_chromatic_shadow_generalized.py  # Photon-ring solver, dispersion power n = 1..4
                ├── triaxis_analyzer_v4.py / triaxis_analyzer_v5.py  # Template-free H1/L1 detector
                ├── injection_recovery_lambda.py        # Lambda-deformed injections through TriAxis
                ├── qnms.csv                 # EXP B resonance frequencies (Švančara et al.)
                ├── STAGE5J_5K_STATUS.md     # Root A vs Root B branch-tracking detail
                └── _archive/qnm_branch_work/          # Giant-vortex QNM work (Stages 5I–5K, 6.5B4)
                    ├── stage5I_1_2_dispersion_radial_p.py        # Dispersion / radial-p solver, light ring
                    ├── stage5I_3_resonance_v2.py                 # Resonance condition (physical residual)
                    ├── stage5I_4_full_m_scan_v6_FIXED.py         # Full 18-m discovery scan
                    ├── stage5I_24_coverage_sensitivity_finding.py  # Found Root B
                    ├── stage5I_25_expanded_discovery_scan.py
                    ├── stage5I_26_singleton_im_expanded.py
                    ├── stage5I_27_q3_continuation_search.py      # q=3 search from Root B (unresolved)
                    ├── stage5I_28_q3_blind_2d_discovery.py       # Experiment-blind 2D discovery at m=-14
                    ├── stage5I_29_rootB_continuation_toward_m11.py
                    ├── stage5I_30_rootB_continuation_toward_m4.py
                    ├── stage5I_31_rootB_resume_from_m5p6.py
                    ├── stage5I_32*_deep_branch_*.py              # q=3/q=4 closure test on deep-pole branch
                    ├── stage5I_33_rootB_fine_window_m15.py       # Root-B limit near m=-15
                    ├── stage6_5B4_2_INDEPENDENT_root_reproduction.py  # Custom Newton solver, 14/14
                    ├── stage5J_rootB_continuation.py             # Root B continuation + classifier
                    └── stage5K_*_rootB.py                        # Observables, injection, noise sweep

## Quick start

    pip install -r requirements.txt
    
    # Validate the numerical solver against the exact dispersion relation
    python paper3/wave_equation_2D_solver.py
    
    # Discrete biharmonic operator and grid/temporal convergence
    python paper3/paper3_Dn4_test.py
    python paper3/paper3_grid_convergence.py
    
    # Spatial convergence using REAL PDE time integration
    python paper3/paper3_h_convergence_test.py
    
    # Test whether YOUR data is consistent with the Lambda-model
    python paper3/lambda_experimental_validator.py --omega data_omega.csv --k data_k.csv
    
    # GW matched-filter pipeline (Stages 1-4)
    python paper3/gw/matched_filter/recovery_test.py
    python paper3/gw/matched_filter/stage2_real_noise_recovery.py
    python paper3/gw/matched_filter/stage3_real_strain_validation.py --h1 <H1.hdf5> --event GW150914
    python paper3/gw/matched_filter/stage4_coherent_h1l1_validation.py --h1 <H1.hdf5> --l1 <L1.hdf5> --event GW150914
    
    # Stage 6+: conditioned propagation-Lambda test on GW150914
    python paper3/gw/matched_filter/gw150914_lambda_onsource_test.py \
        --data <H1.hdf5> --n-null 60 --out gw150914_lambda_result.json
    
    # Lambda phase correction vs the published GWTC-4.0 formula (expects ratio 1.0000)
    python paper3/gw/matched_filter/convention_check.py
    
    # Photon-ring solver, dispersion power n=1..4 (n=1 regression-tests Paper 2)
    python paper3/gw/matched_filter/A2_chromatic_shadow_generalized.py --n 1 --n 2 --n 3 --n 4
    
    # Giant-vortex QNMs (q=1/q=2 comparison; Root B is the primary branch)
    cd paper3/gw/matched_filter/_archive/qnm_branch_work
    python stage5I_28_q3_blind_2d_discovery.py                 # experiment-blind 2D discovery, m=-14
    python stage5I_29_rootB_continuation_toward_m11.py         # Root-B continuation m=-14 -> -11
    python stage6_5B4_2_INDEPENDENT_root_reproduction.py       # independent Newton reproduction
    python stage5J_rootB_continuation.py                       # Root-B continuation + classification
    python stage5K_1_observable_extraction_rootB.py            # f_R(m), tau(m), Q(m)
    
    # BEC experimental checks
    python paper3/bec_validation/level_a_ozeri2002/fit_bogoliubov_lambda.py
    python paper3/bec_validation/level_b_guo2021/fit_quartic.py
    
    # Worked examples (each prints its own validity status)
    python examples/example_BEC.py
    python examples/example_dirac.py
    python examples/example_photonic.py

## What researchers in each field can do

| Group | What you can do |
| --- | --- |
| **Anyone with $(k,\omega)$ data** | Run `lambda_experimental_validator.py` directly — no domain assumptions needed, just numbers. |
| **Theorists** | Inspect the derivations in `paper3_final.tex`, verify the wave equation, propose new metric mappings. |
| **BEC experimentalists** | Supply your healing length $\xi$ and sound speed $c_s$; test $\Lambda=\hbar^2/(4m^2c_s^2)$ against your measured dispersion. State which $\xi$ convention you use. See `paper3/bec_validation/` for two worked examples and the convention audit. |
| **Dirac-material / ARPES groups** | The fitting tool works on your data, but do **not** treat $\Lambda=(1-\eta^2)v_F^2/4$ as a prediction of this framework — see `paper3_final.tex` §5.2 and the runtime warning in `lambda_from_dirac()`. A literature-supported quartic (not $k^6$) correction for your material would be welcome. |
| **Photonic-crystal / fiber-optics groups** | $\Lambda=\beta_4^k/c^2$ is correctly derived from the model, but $\beta_4^k$ is not yet connected to a standard telecom $\beta_4$ or any other directly measured quantity — see `paper3_final.tex` §5.3. Contributions on this gap are welcome. |
| **Analog-gravity / cold-atom & superfluid groups** | The giant-vortex q=1/q=2 comparison (`PAPER3_ADDENDUM_BLIND_2D.md`) is a template for testing a dispersion relation against a resonance/QNM spectrum, with explicit method and claim-boundary language you are welcome to reuse or challenge. Frequency uncertainties from your own system would allow a proper statistical test. |
| **GW / tests-of-GR groups** | The Λ↔A₄ correspondence is reproducible in one command (`convention_check.py`). A different (1+z) convention or α is reported as a fitted power of (1+z) rather than failing silently. |
| **Numerical-relativity / computational reviewers** | Run `paper3_h_convergence_test.py` to verify the spatial convergence claim on the actual fourth-order PDE with a genuinely discretized biharmonic operator. |

## Honesty statement

This is a **falsifiable framework**, not a confirmed law of nature. Eachresult in this repository is scoped to the system it was obtained from. Thephoton-sector, graviton-sector and laboratory-analog coefficients areindependent parameters that differ by tens of orders of magnitude, and nosingle universal value of Λ is claimed or implied (see "Permanent boundaries"in `PAPER3_VALIDATION_STATUS.md`). The laboratory results validate thedispersive spectral machinery in those specific systems; they are not testsof gravitational dispersion.

The repository also documents its own audit trail. Two of the threeoriginally proposed condensed-matter mappings (Dirac materials, photoniccrystals) did not survive scrutiny in their original form and are reportedas negative or incomplete results rather than removed silently. Thespatial-convergence claim went through the same process: an earlier versionasserted grid convergence based only on a spectral-exact shortcut and atemporal-resolution test, neither of which could have detected a genuinespatial-discretization problem; `paper3_h_convergence_test.py` closes thatgap. The path-dependence test is a third such case: validated code, acorrectly posed question, and a demonstration that the measurement has noleverage with current sky localisations. The BEC Level A/B checks follow thesame practice: one is reported as statistically weak-but-consistent ratherthan as a confirmation, the other as a clean null result. The q=2 comparisonreports its smooth degradation toward small |m| as a genuine property of thebranch rather than hiding it. We consider negative and corrected results asimportant to publish as positive ones.

## Paper 3 — Dispersion Validation

Paper 3 is treated as a constrained numerical and experimental-validationstudy, not as a claim that every dispersive platform realizes the same Λ.

### Convention note

The central model is omega^2 = c^2 k^2 (1 + Lambda k^2)

with a linear low-k branch and a quartic correction. This is the conventionused by `wave_equation_2D_solver.py`, by `lambda_phase_correction` in`waveform.py`, and by the flat limit of the Paper 2 Hamiltonian. Some olderfiles (`paper3/STATUS.md`, `paper3/dispersion/README.md`,`fiber_event_horizon_structural_test.py`, `test_matched_filter_pipeline.py`,`gwosc_zero_crossing_injection_recovery.py`) still write `1 + 2 Lambda k^2`,which defines a Λ smaller by a factor of 2. Any Λ value taken from thosefiles must be converted before comparison.

### Current validation status

| Component | Status |
| --- | --- |
| BEC Λ mapping (theoretical) | Established (exact Bogoliubov match) |
| BEC Λ mapping (experimental, real data) | Level A: weak-but-consistent (N=3–4); Level B: clean null (negative control, different system) |
| Λ dispersion fitting | Implemented |
| Synthetic Λ recovery | PASS |
| Spatial (h) convergence, real PDE | PASS, N=64–128 only |
| Dimensional audit | Implemented |
| Λ_NH ↔ Λ_GW relation | Dimensional consistency only (Λ_NH = Λ_GW/c); **no physical derivation** |
| GWTC-4.0 conversion cross-check | PASS (ratio 1.0000, no (1+z) residual) |
| Hořava–Lifshitz external cross-check | PASS — theoretical cross-check, not independent observational validation (same GWTC-4.0 dataset) |
| Giant-vortex q=1 | PASS — parameter-free, real data, 0.021% mean error |
| Giant-vortex q=2 (Root B) | PASS — parameter-free, 0.01–0.06% for m≤−10, degrading smoothly to 1.0% at m=−4 (genuine branch property) |
| Giant-vortex q=3/q=4 | UNRESOLVED — open question |
| Giant-vortex hypothesis test | PLANNED — awaiting measured frequency uncertainties |
| Fiber structural test | PASS for supplied demonstration regime |
| Fiber → Λ identification | REFUSED |
| Dirac independent Λ validation | Not established |
| Path-dependence test | NOT FEASIBLE (no leverage) |
| Full 2D QNM solver | Diagnosed but not yet physically complete |

### Fiber result

The supplied fiber event-horizon demonstration data give p = 3.99870888 forΩ ~ |Δk|^p. The local log-log slope is approximately 4 across the testedrange. This is consistent with a pure quartic regime and is structurallydifferent from the Λ-model low-k structure, so the repository does notinterpret it as an independent measurement of Λ.

### Dimensional safeguard

The experimental fiber quantity β₄ has units s⁴/m. Consequently β₄/c² hasunits s⁶/m³, not m². The validator therefore refuses the identificationΛ = β₄/(4c²) for this quantity. This refusal is intentional and is part ofthe validation methodology.

### Reproducibility

    python paper3/dispersion/lambda_experimental_validator.py
    python paper3/dispersion/fiber_event_horizon_structural_test.py
    python paper3/paper3_h_convergence_test.py
    python paper3/gw/matched_filter/convention_check.py

For the giant-vortex and BEC results, see the Quick start commands above.

When no experimental input is supplied, the Λ validator runs a syntheticself-test. Synthetic results must not be interpreted as experimentalevidence. The fiber validator currently uses a demonstration dataset andlabels it as such; real experimental data should replace the demonstrationarrays before making experimental claims.

## Citing this work

If you use this framework, please cite:

* Kretski, D. (2026). *A Hamiltonian Oscillator Extension of Wave Propagationin Schwarzschild Spacetime*. Zenodo preprint.https://doi.org/10.5281/zenodo.22018715 (Paper 1)
* Kretski, D. (2026). *A Hamiltonian Dispersion Framework for Kerr PhotonRings, Frequency-Dependent Shadow Sensitivity, Superradiance, and EikonalQuasinormal Modes*. Zenodo. https://doi.org/10.5281/zenodo.22051427(Paper 2)
* Kretski, D. (2026). *Parameter-Free Spectral Validation of a QuarticDispersion Model in an Analog-Gravity System* [FULL TITLE]. Submitted toPhysical Review D. (Paper 3)

Work that should be cited alongside the GW propagation results:

* Abac, A. G. *et al.* (LIGO Scientific, Virgo & KAGRA Collaborations)(2026). *GWTC-4.0: Tests of General Relativity. II. Parameterized Tests*.arXiv:2603.19020 — source of the A₄ bound.
* Araújo Filho, A. A., Silva, J. L. A., Heidari, N., Zhu, J., Lobo, I. P. &Bezerra, V. B. (2026). *Gravitational wave propagation in Hořava–Lifshitzgravity*. arXiv:2607.17431 — independent derivation of the flat-spaceω²=k²+αk⁴ → A₄ translation.
* Mirshekari, S., Yunes, N. & Will, C. M. (2012). *Constraining genericLorentz violation and the speed of the graviton with gravitational waves*.Phys. Rev. D 85, 024041.

Work that should be cited alongside the giant-vortex and BEC results:

* Smaniotto, F., Solidoro, V., Patrick, R., Švančara, P. *et al.**Black-hole spectroscopy from a giant quantum vortex*. arXiv:2502.11209 —source of the EXP B q=1..4 resonance data. Resonance frequencies kindlyprovided by P. Švančara (CNRS Institut Néel).
* Švančara, P. *et al.* *Rotating curved spacetime signatures from a giantquantum vortex*. arXiv:2308.10773 — experimental setup and flow-parameterTable SI used for the parameter-free prediction.
* Ozeri, R., Steinhauer, J., Katz, N. & Davidson, N. (2002). *Directobservation of the phonon energy in a Bose-Einstein condensate bytomographic imaging*. Phys. Rev. Lett. 88, 220401.
* Guo, Y., Kroeze, R. M., Marsh, B. P., Gopalakrishnan, S., Keeling, J. &Lev, B. L. (2021). *An optical lattice with sound*. Nature 599, 211 (data:Harvard Dataverse, DOI:10.7910/DVN/LGT5O6).

## License

MIT License (see `LICENSE`). Free to use, modify and redistribute withattribution. Third-party data included in the repository remain subject totheir original terms.

## Contact

Dimitar Kretski — Center for Hydro- and Aerodynamics, BAS, Varna, BulgariaORCID: [0000-0001-5108-2243](https://orcid.org/0000-0001-5108-2243)
