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
| GW forecast                                 | ⚠️ Illustrative projection, not a fitted LIGO/Virgo/KAGRA constraint                                                                      |
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
│   └── figures/                         # Generated plots
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
