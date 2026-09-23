# Λ-model

**Author:** Dimitar Kretski**ORCID:** [0000-0001-5108-2243](https://orcid.org/0000-0001-5108-2243)**Affiliation:** Center for Hydro- and Aerodynamics, Institute of Metal Science, Equipment and Technologies "Acad. A. Balevski", Bulgarian Academy of Sciences, Varna, Bulgaria

This repository contains code and analyses for the one-parameter dispersion relation ω² = c² k² (1 + Λ k²)

applied separately to a laboratory analog system, to gravitational-wave propagation, and to near-horizon photon orbits. No result here detects a non-zero Λ, and no single value of Λ is assumed to apply across these systems.

The authoritative, detailed status is [`paper3/PAPER3_VALIDATION_STATUS.md`](paper3/PAPER3_VALIDATION_STATUS.md). Where this page and that document differ, that document is correct.

## Results

**1. Giant quantum vortex (laboratory analog system).** An independent reimplementation of the quasinormal-mode calculation for the superfluid-helium giant vortex of Švančara et al., using their dispersion relation and the flow parameters from their Table SI (arXiv:2308.10773), compared with the measured EXP B resonances (arXiv:2502.11209). The resonance frequencies were not fitted.

* q=1: mean relative deviation 0.021% (max 0.036%) over m = −21 … −4.
* q=2: 0.01–0.06% for m ≤ −10, growing smoothly to 1.0% at m = −4. This trend is a property of the branch, not of the numerical method.
* q=3 and q=4: no corresponding branch identified. The search is closed unless a new method or candidate appears.
* No statistical test is reported yet: measured frequency uncertainties are not available.

Details: [`paper3/PAPER3_ADDENDUM_BLIND_2D.md`](paper3/PAPER3_ADDENDUM_BLIND_2D.md), [`paper3/STAGE6_5B4_3_ROOTB_FREEZE.md`](paper3/STAGE6_5B4_3_ROOTB_FREEZE.md).

**2. Gravitational-wave propagation.** The propagation phase used here is identical to the LVK α=4 modified-dispersion parametrization, with Λ_GW = ħ²c³A₄. The published GWTC-4.0 bound (arXiv:2603.19020) translates to Λ_GW ∈ [−7.2×10⁻³, +2.2×10⁻³] m³/s (90%), consistent with Λ = 0. The same translation was published independently in arXiv:2607.17431. A separate analysis of GW150914 gives z = −0.43σ against an off-source null, also consistent with Λ = 0.

**3. Photon sector.** Fermi-LAT time-of-flight data give Λ < 1.44×10⁻⁵³ m². The near-horizon photon-ring solver is regression-tested against Paper 2 but has not been applied to observational data; at EHT radio frequencies the effect is too small to be measured.

## Negative and inconclusive results

* **BEC data** (Ozeri et al. 2002, digitized): consistent with the Bogoliubov prediction, but with N = 3–4 points this is not a confirmation. A cavity-QED control dataset (Guo et al. 2021) gives a clean null.
* **Path-dependence test for GW events:** not feasible with current sky localisations (reliability 0 on all three axes, 174 events). Closed before any anomaly statistic was computed.
* **Dirac-material mapping:** not supported by the cited literature.
* **Photonic and fiber mappings:** not connected to any measured quantity. The fiber module (`paper3/dispersion/`) currently runs only on synthetic demonstration arrays (omega ~ k^4 plus noise), so it produces no physical result.

## What is not shown

* The three systems above use separate parameters. The relation Λ_NH = Λ_GW/c is a dimensional observation, not a derivation, and the laboratory coefficients are properties of those media.
* The laboratory results test the numerical methods against real spectra. They are not tests of gravitational dispersion, and no mapping from the vortex to a Kerr black hole has been derived.

## Code verification

These checks confirm that the code is correct. They are not physical results.

* Spectral PDE solver: exact in space; O(1/N) in time; second-order spatial convergence of the full leapfrog integration for N = 64–128 (N < 64 unreliable).
* Recovery of a known Λ from synthetic data and a Λ = 0 control: [`paper3/dispersion/tests/`](paper3/dispersion/tests/).
* GW matched-filter pipeline: injection/recovery and off-source null tests on real LIGO noise ([`paper3/gw/matched_filter/`](paper3/gw/matched_filter/)).
* QNM roots: reproduced by an experiment-blind 2D search and by an independent Newton solver (14/14; same physical residual function).

## Convention

All current code uses ω² = c²k²(1 + Λk²). Some older files (`paper3/STATUS.md`, `paper3/dispersion/README.md`, `fiber_event_horizon_structural_test.py`, `gwosc_zero_crossing_injection_recovery.py`) write 1 + 2Λk², which defines a Λ smaller by a factor of 2. For BEC, Λ = ħ²/(4m²c_s²), which equals ξ²/4 for ξ = ħ/(m c_s).

## Where things are

| Topic | Location |
| --- | --- |
| Status and claim boundaries | `paper3/PAPER3_VALIDATION_STATUS.md` |
| Giant-vortex QNM code | `paper3/gw/matched_filter/_archive/qnm_branch_work/` |
| GW analyses | `paper3/gw/matched_filter/` |
| GW path-dependence test | `paper3/gw/path_test/` |
| BEC checks | `paper3/bec_validation/` |
| Fitting tool for your own (k, ω) data | `paper3/lambda_experimental_validator.py` |
| Solver and convergence tests | `paper3/wave_equation_2D_solver.py`, `paper3/paper3_h_convergence_test.py` |
| Earlier full draft (August 2026) | `paper3/paper3_final.tex` |

Quick start: pip install -r requirements.txt python paper3/lambda_experimental_validator.py --omega data_omega.csv --k data_k.csv python paper3/gw/matched_filter/convention_check.py

Further commands are listed in each subdirectory.

## Citing

* Kretski, D. (2026). *A Hamiltonian Oscillator Extension of Wave Propagation in Schwarzschild Spacetime*. Zenodo preprint. https://doi.org/10.5281/zenodo.22018715
* Kretski, D. (2026). *A Hamiltonian Dispersion Framework for Kerr Photon Rings, Frequency-Dependent Shadow Sensitivity, Superradiance, and Eikonal Quasinormal Modes*. Zenodo. https://doi.org/10.5281/zenodo.22051427
* Kretski, D. (2026). *[FULL TITLE]*. Submitted to Physical Review D.

Please also cite the work this repository depends on:

* Smaniotto, F., Solidoro, V., Patrick, R., Švančara, P. *et al.* *Black-hole spectroscopy from a giant quantum vortex*. arXiv:2502.11209. Resonance data kindly provided by P. Švančara (CNRS Institut Néel).
* Švančara, P. *et al.* *Rotating curved spacetime signatures from a giant quantum vortex*. arXiv:2308.10773.
* Abac, A. G. *et al.* (LVK) (2026). *GWTC-4.0: Tests of General Relativity. II. Parameterized Tests*. arXiv:2603.19020.
* Araújo Filho, A. A. *et al.* (2026). *Gravitational wave propagation in Hořava–Lifshitz gravity*. arXiv:2607.17431.
* Mirshekari, S., Yunes, N. & Will, C. M. (2012). Phys. Rev. D 85, 024041.
* Ozeri, R. *et al.* (2002). Phys. Rev. Lett. 88, 220401.
* Guo, Y. *et al.* (2021). Nature 599, 211. Data: Harvard Dataverse, DOI:10.7910/DVN/LGT5O6.

## License

MIT (see `LICENSE`). Third-party data remain subject to their original terms.
