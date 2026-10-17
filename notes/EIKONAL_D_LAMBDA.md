# Eikonal D_Λ: co- and counter-rotating photon rings in Kerr (supplement to Paper 2, Remark 4)

*October 2026. Reproduced by `notes/eikonal_d_lambda.py` (mpmath + sympy). Eikonal estimate
only: it is not a quasinormal-mode spectrum from the wave equation and does not show that the
corresponding modes exist with well-defined horizon boundary conditions (see A2(a)). Papers 1–3
are not changed by this note.*

## Setup

Same Hamiltonian as Paper 2 (Eqs. 1 and 5): $H = \tfrac12\left[g^{\mu\nu}k_\mu k_\nu + \Lambda\,(k_{\rm loc}^2)^2\right]$ with
$k_{\rm loc}^2 = g^{\mu\nu}k_\mu k_\nu + (n^\mu k_\mu)^2$ for the ZAMO $n^\mu$. Kerr equatorial plane, $a = 0.9M$,
$G = c = M = 1$, signature $(-,+,+,+)$, $k_\mu = (-E, 0, 0, L)$, photon energy $E$ (in units of $1/M$),
$\Lambda$ in units of $M^2$ ($\hat\Lambda = \Lambda/M_{\rm geom}^2$). Circular null orbits: $H = 0$, $\partial_r H = 0$.
Prograde ring: $L > 0$; retrograde: $L < 0$. Orbital frequency, as defined in Paper 2:
$\Omega_{\rm ph} = \dot\phi/\dot t = (\partial H/\partial L)/(\partial H/\partial k_t)$.

## Results

**Validation.** The ring radii and the prograde impact parameter reproduce Paper 2, Table 1, and the
first-order impact-parameter coefficients reproduce Paper 2, Eq. 9 exactly:
$db/d\Lambda = -0.62296$ (prograde) and $+11.4953$ (retrograde).

**First-order orbital-frequency shifts** (two independent methods agree to 10 digits: numerical
root finding with central differences, and implicit differentiation with exact symbolic derivatives):

| Ring | $\Omega_{\rm ph}^{(0)}$ | $d\Omega_{\rm ph}/d\Lambda$ at $E=1$ | relative |
| --- | --- | --- | --- |
| prograde | $+0.351565$ | $+0.230991$ | $+0.657\,\hat\Lambda E^2$ |
| retrograde | $-0.146363$ | $-0.738760$ | $+5.047\,\hat\Lambda E^2$ ($|\Omega|$ grows) |

**Energy scaling.** $d\Omega_{\rm ph}/d\Lambda$ is exactly proportional to $E^2$ (checked at $E = 1, 2, 4$), as for
the impact parameter (Paper 2, Theorem 1).

**Eikonal double difference.** In the eikonal correspondence the photon energy is the mode frequency
itself, $E = \omega \simeq l|\Omega_{\rm ph}|$, so for the equatorial modes $m = \pm l$:

$$\delta\omega_{+l} = 0.02855\,\hat\Lambda\,l^3,\qquad \delta\omega_{-l} = 0.01583\,\hat\Lambda\,l^3,\qquad
D_\Lambda^{\rm eik} = \delta\omega_{+l} - \delta\omega_{-l} = 0.0127\,\hat\Lambda\,l^3$$

(geometric units; $\omega_{\rm phys} = \omega\,c^3/GM$). The general-relativistic splitting cancels in this
difference by construction.

## Points of disagreement with Paper 2 (to be checked against the Paper 2 code)

1. **Table 1, retrograde $b$ for $\Lambda > 0$.** The tabulated values move opposite to Paper 2's own
   coefficient $C_{\rm ret} = +11.495$ (e.g. $-6.946$ at $\Lambda = 10^{-2}$, whereas $C_{\rm ret}$ and the full
   solution give $\approx -6.72$). The radii in the same table agree to all digits.
2. **Table 3 and Eq. 20.** The quoted $\delta\Omega_{\rm ph}/\Omega_{\rm ph} \simeq -0.219\,\Lambda$ coincides numerically with
   $\delta b/b$ of the prograde ring ($-0.6230/2.8444$), not with the orbital frequency $\dot\phi/\dot t$ that the
   paper defines; the latter gives $+0.657\,\hat\Lambda E^2$.
3. **Energy dependence in the eikonal QNM estimate.** Paper 2 evaluates the correction at $E = 1$. Since
   it scales as $E^2$ and $E = \omega \simeq l\Omega$, the dispersive correction to an eikonal mode grows as $l^3$,
   not linearly in $l$; Eqs. 18–20 as written omit this.

## Size of the effect

For a $10\,M_\odot$ black hole and $l = 2$: with $\Lambda = 7.4\times10^{-12}\,{\rm m^2}$ (the GWTC-4.0 upper bound),
$\hat\Lambda \approx 3\times10^{-20}$ and $D_\Lambda \approx 10^{-17}$ Hz; with a Planck-scale estimate
$\Lambda \approx 2.6\times10^{-70}\,{\rm m^2}$, $D_\Lambda \approx 4\times10^{-76}$ Hz. For radio photons around EHT targets the
corresponding photon-ring effect is likewise negligible at natural values of $\Lambda$. The result is a
theoretical completion of Paper 2's eikonal framework, not an observable test.
