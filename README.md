# Λ-model

**Author:** Dimitar Kretski**ORCID:** [0000-0001-5108-2243](https://orcid.org/0000-0001-5108-2243)**Affiliation:** Center for Hydro- and Aerodynamics, Institute of Metal Science, Equipment and Technologies "Acad. A. Balevski", Bulgarian Academy of Sciences, Varna, Bulgaria

This repository contains code and analyses for the one-parameter dispersion relation

$$
\boxed{\omega^2=c^2k^2(1+\Lambda k^2)}
$$

applied separately to:

1. a laboratory analog system,
2. gravitational-wave propagation,
3. near-horizon photon orbits.

The same mathematical form is therefore used in several physical contexts, but **the corresponding parameters are not assumed to be the same physical constant**.

No result in this repository detects a non-zero fundamental $\Lambda$, and no universal value of $\Lambda$ is assumed across the systems.

The **universal-$\Lambda$ hypothesis** may be considered as a separate theoretical hypothesis, but no derivation establishing such universality is currently provided.

The authoritative, detailed status is:

[`paper3/PAPER3_VALIDATION_STATUS.md`](paper3/PAPER3_VALIDATION_STATUS.md)

Where this README and that document differ, the status document is authoritative.

* * *

# Results

## 1. Giant quantum vortex — laboratory analog system

An independent reimplementation of the quasinormal-mode calculation for the superfluid-helium giant vortex of Švančara et al., using their dispersion relation and the flow parameters from their Table SI (arXiv:2308.10773), compared with the measured EXP B resonances reported in arXiv:2502.11209.

The resonance frequencies were **not fitted**.

### Results

* **q = 1:** mean relative deviation 0.021% (maximum 0.036%) over $m=-21,\ldots,-4$.
* **q = 2:** 0.01–0.06% for $m\leq-10$, growing smoothly to approximately 1.0% at $m=-4$.
* The q=2 trend is a property of the physical branch and not evidence of a numerical fitting artifact.
* **q = 3 and q = 4:** no corresponding branch was identified.
* The q=3/q=4 search is closed unless a new method or candidate branch appears.
* No statistical significance test is reported because measured frequency uncertainties are not available.

Details:

* [`paper3/PAPER3_ADDENDUM_BLIND_2D.md`](paper3/PAPER3_ADDENDUM_BLIND_2D.md)
* [`paper3/STAGE6_5B4_3_ROOTB_FREEZE.md`](paper3/STAGE6_5B4_3_ROOTB_FREEZE.md)

### Interpretation

These calculations validate the numerical treatment against an experimentally measured analog spectrum.

They **do not establish a gravitational value of $\Lambda$** and do not establish a mapping between the giant-vortex system and Kerr spacetime.

* * *

## 2. Gravitational-wave propagation

The GW analysis uses the same mathematical dispersion relation

$$
\omega^2=c^2k^2(1+\Lambda_{\rm GW}k^2),
$$

with

$$
\boxed{\Lambda_{\rm GW}=A_4(\hbar c)^2=\hbar^2c^2A_4}
$$

where $A_4$ is the $\alpha=4$ coefficient used in the LVK modified-dispersion parametrization.

Thus

$$
[\Lambda_{\rm GW}]={\rm m^2}.
$$

### Published GWTC-4.0 constraint

The combined GWTC-4.0 result reported by LVK gives, for $\alpha=4$,

$$
\boxed{A_4\in[-620,+190]\ {\rm eV}^{-2}}
$$

at 90% credibility.

Using

$$
\Lambda_{\rm GW}=A_4(\hbar c)^2,
$$

this corresponds to

$$
\boxed{\Lambda_{\rm GW}\in[-2.4\times10^{-11},\,+7.4\times10^{-12}]\ {\rm m^2}}
$$

at 90% credibility.

This interval contains

$$
\Lambda_{\rm GW}=0.
$$

This is a **translation of the published LVK posterior**, not an independent constraint obtained by this repository.

Reference:

Abac, A. G. et al. (LVK), *GWTC-4.0: Tests of General Relativity. II. Parameterized Tests*, arXiv:2603.19020.

### GW phase convention

A convention distinction is important.

The dispersion relation alone does not identify which velocity prescription is used when constructing the accumulated propagation phase. Two prescriptions have appeared in the literature:

* the earlier **particle-velocity** prescription;
* the **group-velocity** prescription.

For GWTC-4.0, LVK use the **group-velocity prescription**, motivated by its agreement with the WKB treatment (Ezquiaga et al. 2022).

For $\alpha=4$, the corresponding phase correction in the present $\Lambda_{\rm GW}$ normalization is

$$
\boxed{\Delta\Psi(f)=-\frac{4\pi^3}{c^3}\Lambda_{\rm GW}I_4(z)f^3}
$$

where $I_4(z)$ denotes the cosmological propagation integral in the chosen convention.

This is the phase convention used for the present GW model.

Earlier LVK analyses used the particle-velocity prescription. For general $\alpha$, the two prescriptions differ by the factor

$$
1-\alpha.
$$

Therefore for $\alpha=4$,

$$
1-\alpha=-3,
$$

giving both a factor-of-three change in magnitude and a sign reversal between the two conventions.

### Relation to LALSimulation

The repository contains a convention check showing that the relevant LALSimulation implementation reproduces the older particle-velocity form, including its characteristic factor $1/3$.

That agreement is a **software/convention verification**. It does not establish that the older prescription is the physically preferred one.

Consequently:

> **The built-in LALSimulation dispersive phase must not be used as the physical injection model for the present $\Lambda_{\rm GW}$ analysis.**

For injections testing the present model, the propagation phase is added explicitly using the group-velocity/WKB expression above.

Details:

[`paper3/gw/matched_filter/convention_check.py`](paper3/gw/matched_filter/convention_check.py)

### Role of the repository GW scanner

The published LVK constraint is expected to be substantially more sensitive than the independent scanner developed here.

The purpose of the repository scanner is therefore **not to replace the LVK catalogue analysis**.

Its role is to provide an independent methodological cross-check based on:

* an independently implemented matched-filter statistic,
* explicit $\Lambda$-dependent phase templates,
* controlled injections,
* a $\Lambda=0$ null,
* off-source real-noise controls,
* and waveform/model robustness tests.

The scanner should therefore be interpreted as an **independent validation tool**, not as a claim of a stronger constraint than LVK.

A separate, pre-registered residual-based test of the same $\alpha=4$ channel is being prepared in [residual-first-framework](https://github.com/Kretski/residual-first-framework) (Module 1).

Details:

[`paper3/gw/matched_filter/`](paper3/gw/matched_filter/)

* * *

## 3. Photon sector

The photon sector is treated separately from the GW sector.

LHAASO observations of GRB 221009A constrain quadratic photon dispersion. In the corresponding $n=2$ parameterization this can be mapped formally onto

$$
\omega^2=c^2k^2(1+\Lambda_\gamma k^2),
$$

with a sector-specific parameter of the form

$$
\Lambda_\gamma=-S\left(\frac{\hbar c}{E_{\rm QG,2}}\right)^2,
$$

where $S$ denotes the sign convention used for the corresponding superluminal or subluminal branch.

For the LHAASO analysis of GRB 221009A, the reported maximum-likelihood 95% limits translate approximately to

$$
\Lambda_\gamma < 7.5\times10^{-56}\ {\rm m^2}
$$

for the superluminal branch and

$$
|\Lambda_\gamma|<2.7\times10^{-56}\ {\rm m^2}
$$

for the subluminal branch.

The weakest of the paper's three spectral models gives approximately

$$
\Lambda_\gamma<2.0\times10^{-55}\ {\rm m^2}.
$$

Reference:

Yang, R.-Z., Bi, X.-J. & Yin, P.-F., JCAP 04 (2024) 060, arXiv:2312.09079.

These limits assume no intrinsic energy-dependent emission delay and vary with the adopted light-curve model.

### Important interpretation

The identification $\Lambda_\gamma=\Lambda_{\rm GW}$ is **not assumed**, and no physical identification between them has been derived.

The photon result therefore constrains a **photon-sector dispersion coefficient**. It is not, by itself, a measurement of the gravitational-wave $\Lambda_{\rm GW}$.

The near-horizon photon-ring solver is regression-tested against Paper 2 but has not been applied to observational data.

* * *

# Negative and inconclusive results

## BEC data

Steinhauer et al. (2002), using 13 measured points extracted from the vector EPS in the arXiv source, were tested against the local-density Bogoliubov spectrum.

Data and analysis:

[`paper3/bec_validation/steinhauer2002/`](paper3/bec_validation/steinhauer2002/)

The fit requires no additional $k^4$ term beyond standard Bogoliubov theory.

The fitted additional coefficient is constrained approximately by

$$
-3\%\lesssim g_{\rm extra}\lesssim+6\%
$$

at 95%.

This is a test of the Bogoliubov description of the condensate.

It is **not a detection of the $\Lambda$-model**.

The quartic term already present in the Bogoliubov spectrum is the standard quantum-pressure contribution,

$$
\Lambda_{\rm BEC}=\frac{\hbar^2}{4m^2c_s^2}=\frac{\xi^2}{4}
$$

under the present convention

$$
\omega^2=c_s^2k^2(1+\Lambda_{\rm BEC}k^2).
$$

Ozeri et al. (2002), based on a very small digitized sample (approximately 3–4 points), are consistent with the same standard description but are not sufficient for an independent confirmation.

A cavity-QED control dataset from Guo et al. (2021) gives a clean null.

### Why BEC is not treated as a universal-$\Lambda$ measurement

The BEC quartic coefficient is determined by the microscopic properties of the condensate.

No derivation has been established that identifies

$$
\Lambda_{\rm BEC}
$$

with either

$$
\Lambda_{\rm GW}
$$

or

$$
\Lambda_\gamma.
$$

The BEC result therefore remains an **analog/validation result**, not evidence for a universal fundamental $\Lambda$.

* * *

## GW path-dependence test

A path-dependence test for GW events was considered but found infeasible with the current sky localisations.

Reliability was zero on all three tested axes for the 174-event sample.

The test was therefore closed **before any anomaly statistic was computed**.

* * *

## Dirac-material mapping

The proposed Dirac-material mapping is not supported by the cited literature.

No physical result is claimed from this mapping.

* * *

## Photonic and fiber mappings

No connection to a measured physical quantity has been established.

The fiber module

[`paper3/dispersion/`](paper3/dispersion/)

currently operates only on synthetic demonstration arrays of approximately $k^4$-type dispersion plus noise.

It therefore produces no physical experimental result.

* * *

# What is not shown

The following points are deliberately not claimed.

### 1. No universal value of $\Lambda$

The laboratory, GW, photon, and near-horizon calculations currently use **sector-specific parameters**.

No physical derivation has established

$$
\Lambda_{\rm BEC}=\Lambda_{\rm GW}=\Lambda_\gamma=\Lambda_{\rm NH}.
$$

A universal-$\Lambda$ model remains a theoretical hypothesis requiring a derivation.

### 2. No mapping from the giant vortex to Kerr spacetime

The laboratory vortex calculations validate the numerical treatment against an analog system.

They do not establish a quantitative Kerr mapping.

In particular, no equation-level derivation has been obtained that maps the laboratory dispersion coefficient onto a Kerr gravitational parameter.

### 3. No derivation of $\Lambda_{\rm NH}=\Lambda_{\rm GW}/c$

Older versions of the project used the dimensional relation

$$
\Lambda_{\rm NH}=\frac{\Lambda_{\rm GW}}{c}.
$$

This is retained only as a **dimensional observation**.

It is not a physical derivation and must not be interpreted as evidence that the near-horizon and GW parameters are identical.

### 4. No non-zero $\Lambda$ detection

None of the analyses in this repository establishes a statistically significant non-zero fundamental $\Lambda$.

The principal results are:

* numerical validation,
* null tests,
* consistency tests,
* independent reproduction of published constraints,
* and identification of model/convention boundaries.

* * *

# Code verification

These checks confirm numerical or software correctness.

They are **not physical detections**.

## Spectral PDE solver

* Exact spatial discretization within the implemented spectral representation.
* $O(1/N)$ time convergence in the relevant solver configuration.
* Second-order spatial convergence of the full leapfrog integration for $N=64$–128.
* $N<64$ is unreliable for the convergence study.

Relevant files:

* `paper3/wave_equation_2D_solver.py`
* `paper3/paper3_h_convergence_test.py`

## Known-$\Lambda$ recovery

Synthetic injection/recovery tests verify recovery of a known dispersion coefficient and a $\Lambda=0$ control.

Tests:

[`paper3/dispersion/tests/`](paper3/dispersion/tests/)

These tests establish estimator behavior under controlled synthetic conditions.

They do not establish a physical non-zero $\Lambda$.

## GW matched-filter pipeline

The GW pipeline has been tested using:

* synthetic injection/recovery,
* $\Lambda=0$ controls,
* real LIGO noise,
* and off-source null segments.

Files:

[`paper3/gw/matched_filter/`](paper3/gw/matched_filter/)

The GW scanner is an independent methodological implementation and is not intended to supersede the published LVK catalogue constraint.

## Dispersion-convention check

The repository contains a dedicated test of the relation between the implemented dispersion convention and the LALSimulation dispersive phase.

File:

[`paper3/gw/matched_filter/convention_check.py`](paper3/gw/matched_filter/convention_check.py)

The check reproduces the older LALSimulation/particle-velocity convention.

The present physical GW injection model instead uses the GWTC-4.0 group-velocity/WKB prescription.

## QNM roots

QNM roots were reproduced by:

* an experiment-blind 2D search,
* and an independent Newton solver.

The two implementations agree for 14/14 tested roots using the same physical residual function.

* * *

# Convention

All current code is intended to use

$$
\boxed{\omega^2=c^2k^2(1+\Lambda k^2)}
$$

with

$$
[\Lambda]={\rm m^2}.
$$

This is the **authoritative normalization** for the current repository.

## Legacy $1+2\Lambda k^2$ convention

Some older files use

$$
\omega^2=c^2k^2(1+2\Lambda k^2).
$$

These files define a parameter that is smaller by a factor of two relative to the current convention.

Known legacy files include:

* `paper3/STATUS.md`
* `paper3/dispersion/README.md`
* `fiber_event_horizon_structural_test.py`
* `gwosc_zero_crossing_injection_recovery.py`

When comparing results from these files with current results, the factor-of-two normalization must be accounted for explicitly.

## BEC normalization

For the Bogoliubov spectrum,

$$
\boxed{\Lambda_{\rm BEC}=\frac{\hbar^2}{4m^2c_s^2}=\frac{\xi^2}{4}}
$$

under the present $1+\Lambda k^2$ convention, with

$$
\xi=\frac{\hbar}{mc_s}.
$$

## GW normalization

For

$$
E^2=p^2c^2+A_4p^4c^4
$$

and

$$
E=\hbar\omega,\qquad p=\hbar k,
$$

the corresponding coefficient in the present dispersion equation is

$$
\boxed{\Lambda_{\rm GW}=A_4\hbar^2c^2=A_4(\hbar c)^2}
$$

with units of ${\rm m^2}$.

An older bookkeeping convention in some project material used

$$
\hbar^2c^3A_4,
$$

which has units ${\rm m^3/s}$.

That quantity is **not the $\Lambda$ defined by the current master equation** and should not be mixed with the current $\Lambda_{\rm GW}$.

* * *

# Scientific interpretation

The project currently supports three distinct statements.

### Statement A — mathematical structure

Several physical systems exhibit or can be represented by dispersion relations containing a quartic-in-$k$ correction of the form

$$
\omega^2\sim c^2k^2(1+\Lambda k^2).
$$

### Statement B — system-specific tests

The coefficient can be estimated or constrained independently in different systems.

### Statement C — fundamental universality

There is currently **no derivation establishing that these system-specific coefficients represent one universal physical constant**.

Therefore the repository does not combine the BEC, photon, GW, laboratory-vortex, and near-horizon values into a single global estimate.

If a single universal $\Lambda$ applied to both gravitational waves and photons, the photon limit ($\sim10^{-55}\ {\rm m^2}$) would be about 44 orders of magnitude stronger than the gravitational-wave limit ($\sim10^{-11}\ {\rm m^2}$), and gravitational-wave data would add no constraint. The GW channel is informative only for a gravity-sector $\Lambda_{\rm GW}$ distinct from the photon sector.

A universal-$\Lambda$ interpretation remains a hypothesis that would require:

1. a common underlying theory,
2. an equation-level mapping between sectors,
3. consistent normalization,
4. consistent sign convention,
5. and simultaneous agreement with independent observations.

* * *

# Where things are

| Topic | Location |
| --- | --- |
| Status and claim boundaries | `paper3/PAPER3_VALIDATION_STATUS.md` |
| Giant-vortex QNM code | `paper3/gw/matched_filter/_archive/qnm_branch_work/` |
| GW analyses | `paper3/gw/matched_filter/` |
| GW convention check | `paper3/gw/matched_filter/convention_check.py` |
| GW path-dependence test | `paper3/gw/path_test/` |
| BEC checks | `paper3/bec_validation/` |
| Experimental $(k,\omega)$ fitting tool | `paper3/lambda_experimental_validator.py` |
| Spectral solver | `paper3/wave_equation_2D_solver.py` |
| Solver convergence tests | `paper3/paper3_h_convergence_test.py` |
| Earlier full draft | `paper3/paper3_final.tex` |

* * *

# Quick start

Install the required Python dependencies:

    pip install -r requirements.txt

Run the experimental dispersion validator:

    python paper3/lambda_experimental_validator.py \
        --omega data_omega.csv \
        --k data_k.csv

Run the GW convention check:

    python paper3/gw/matched_filter/convention_check.py

The GW matched-filter tests are run from `paper3/gw/matched_filter/`; consult that directory's documentation for the exact commands and required input data.

The experimental validator fits a single $\Lambda$ within the selected mathematical dispersion model. For real physical media, the preferred procedure is a residual test against the known medium-specific baseline, as implemented for the BEC validation.

* * *

# Data and interpretation policy

The repository distinguishes between:

### Published results

Results obtained from external published analyses and translated into the repository's normalization.

Example:

$$
A_4^{\rm LVK}\rightarrow\Lambda_{\rm GW}.
$$

These are not independent measurements by this repository.

### Independent numerical reproductions

Calculations that reproduce published or experimentally measured quantities using an independently implemented method.

### Synthetic validation

Tests where a known artificial $\Lambda$ is injected into synthetic data and recovered.

### Null tests

Tests designed to determine whether the analysis produces a non-zero result when the underlying signal contains no such modification.

### Physical claims

Only results surviving the relevant calibration, null, injection, and model-robustness tests may be interpreted as physical evidence.

* * *

# Current scientific status

At the present stage:

* The laboratory giant-vortex calculation reproduces the relevant measured resonance branches to the stated numerical accuracy.
* The BEC analysis finds no need for an additional quartic correction beyond standard Bogoliubov quantum pressure.
* The GWTC-4.0 LVK result provides a published constraint on the $\alpha=4$ modified-dispersion coefficient.
* In the present normalization, that published constraint corresponds to

$$
\boxed{\Lambda_{\rm GW}\in[-2.4\times10^{-11},+7.4\times10^{-12}]\ {\rm m^2}}
$$

at 90% credibility.

* The photon sector provides much tighter bounds on its own quadratic dispersion coefficient, but no physical identification with $\Lambda_{\rm GW}$ has been derived.
* The near-horizon photon-ring calculation has been numerically tested but has not been connected to an observational constraint.
* No analysis in this repository establishes a non-zero universal fundamental $\Lambda$.

The GW scanner is therefore best regarded as an **independent methodological cross-check of the $\alpha=4$ dispersion channel**, rather than as a replacement for the published LVK constraint.

* * *

# Citing

### This repository

Kretski, D. (2026). *Λ-model*. GitHub repository.

### Related work

* Kretski, D. (2026). *A Hamiltonian Oscillator Extension of Wave Propagation in Schwarzschild Spacetime*. Zenodo preprint.https://doi.org/10.5281/zenodo.22018715
  
* Kretski, D. (2026). *A Hamiltonian Dispersion Framework for Kerr Photon Rings, Frequency-Dependent Shadow Sensitivity, Superradiance, and Eikonal Quasinormal Modes*. Zenodo.https://doi.org/10.5281/zenodo.22051427
  
* Abac, A. G. et al. (LVK) (2026). *GWTC-4.0: Tests of General Relativity. II. Parameterized Tests*. arXiv:2603.19020.
  
* Araújo Filho, A. A. et al. (2026). *Gravitational wave propagation in Hořava–Lifshitz gravity*. arXiv:2607.17431.
  
* Ezquiaga, J. M., Hu, W., Lagos, M., Lin, M.-X. & Xu, F. (2022). *Modified gravitational wave propagation with higher modes and its degeneracies with lensing*. JCAP 08, 016 (group-velocity phase).
  
* Mirshekari, S., Yunes, N. & Will, C. M. (2012). *Constraining Lorentz-violating, modified dispersion relations with gravitational waves*. Phys. Rev. D 85, 024041.
  
* Smaniotto, F., Solidoro, V., Patrick, R., Švančara, P. et al. *Black-hole spectroscopy from a giant quantum vortex*. arXiv:2502.11209.
  
* Švančara, P. et al. *Rotating curved spacetime signatures from a giant quantum vortex*. arXiv:2308.10773.
  
* Steinhauer, J., Ozeri, R., Katz, N. & Davidson, N. (2002). Phys. Rev. Lett. 88, 120407. arXiv:cond-mat/0111438.
  
* Ozeri, R. et al. (2002). Phys. Rev. Lett. 88, 220401.
  
* Yang, R.-Z., Bi, X.-J. & Yin, P.-F. (2024). JCAP 04, 060. arXiv:2312.09079.
  
* Guo, Y. et al. (2021). Nature 599, 211. Data: Harvard Dataverse, DOI:10.7910/DVN/LGT5O6.
  

* * *

# License

MIT (see [`LICENSE`](LICENSE)).

Third-party data remain subject to their original terms.
