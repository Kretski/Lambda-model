# Status of the Lambda model as a gravitational-wave propagation theory

**Status note — October 2026.** A statement of the current theoretical status, not a claim
that the model has been disproved.

## 1. Scope

This note assesses whether the current framework provides a physically derived value, sign or
range for the dispersion parameter that would apply to gravitational-wave propagation. It is
based on:

- Paper 1: *A Hamiltonian Oscillator Extension of Wave Propagation in Schwarzschild Spacetime*
  (Zenodo, doi:10.5281/zenodo.22018716; version v0.4);
- Paper 2: *A Hamiltonian Dispersion Framework for Kerr Photon Rings, Frequency-Dependent Shadow
  Constraints, Superradiance, and Eikonal Quasinormal Modes* (draft v0.4, 22 August 2026).

It is kept separate from the methodological gravitational-wave analysis
(residual-first-framework, Module 1), whose results do not depend on this note.

## 2. What the framework establishes

Paper 1 introduces the quartic dispersion relation as a minimal ansatz (Eq. 3):

$$\omega^2 = c^2 k^2 \alpha^2(r)\,\left(1 + \Lambda k^2\right),$$

with $\Lambda$ in m². It then **derives** the corresponding characteristic equation from an action
for a **massless scalar field** with a quartic local-frame term (Section 4.1, Eqs. 7–12):

$$S = -\frac{1}{2}\int d^4x\,\sqrt{-g}\left[g^{\mu\nu}\partial_\mu\phi\,\partial_\nu\phi + \frac{\Lambda}{2}\left(h^{\mu\nu}\partial_\mu\phi\,\partial_\nu\phi\right)^2\right]
\;\Rightarrow\; g^{\mu\nu}k_\mu k_\nu + \Lambda k_{\rm loc}^4 = 0 .$$

Paper 2 adopts the Kerr generalisation (Eq. 1), with $\Lambda > 0$ by definition:

$$H = \frac{1}{2}\left[g^{\mu\nu}k_\mu k_\nu + \Lambda\left(k_{\rm loc}^{\rm ZAMO}\right)^4\right].$$

Within this Hamiltonian it derives photon-ring corrections, a frequency-dependent shadow, an
unchanged superradiant threshold, and eikonal quasinormal-mode estimates. The QNM estimates use
the photon-orbit correspondence as an effective approximation; a first-principles derivation
from the radial wave equation is deferred to Paper 3 (Paper 2, Section 7.1).

These are consequences of the adopted Hamiltonian. They do not establish that the same
Hamiltonian, with the same coefficient, governs electromagnetic or gravitational perturbations.

## 3. The missing link to gravitational waves

Neither paper derives a quartic dispersion term for **tensor** perturbations from a
gravitational-sector action. The identification

$$\Lambda_{\rm GW} = \Lambda$$

is therefore an additional universality assumption, not a result of the scalar-field derivation
or of the Kerr photon-ring calculation. The observational formula of Paper 1 (Eq. 26) is written
for photons, and the physical origin of $\Lambda$ is listed as open in both papers (Paper 1,
Section 9: "What is the gravitational analogue of the GP self-interaction that sets $\Lambda$?";
Paper 2, Section 8).

Paper 1 also leaves two propagation scenarios open (Section 5, Table 1):

- **Route A** (coordinate dispersion): correct GR limit at $\Lambda = 0$; a residual vacuum
  dispersion in flat space, which accumulates over cosmological distances.
- **Route B** (local dispersion): no vacuum dispersion in flat space, and, as Paper 1 notes,
  an incorrect GR limit at $\Lambda = 0$ unless the Killing structure or the local-to-coordinate
  transformation is modified.

Only Route A produces the propagation effect that gravitational-wave dispersion tests measure.
The choice between the routes is not made in the current papers.

## 4. Sign and magnitude

- **Sign.** $\Lambda > 0$ is imposed (Paper 1, Eqs. 17 and 19; Paper 2, Eq. 1). It is not derived
  from stability, causality, or an underlying gravitational theory.
- **Magnitude.** Paper 1 (Section 6) identifies $\Lambda = c_6/M_{\rm QG}^2$ (natural units,
  converted to SI with $(\hbar c)^2$), the coefficient of a dimension-6 operator. For $c_6 \sim 1$
  this gives about $2.6\times10^{-70}\ {\rm m^2}$ at the Planck scale and about
  $3.9\times10^{-64}\ {\rm m^2}$ at the GUT scale. Paper 1 itself states that the Planck-scale value is
  "unobservable at any foreseeable energy scale" (Section 8). This is a naturalness estimate,
  not a derived prediction for $\Lambda_{\rm GW}$.
- **Photon sector.** Paper 1 quotes an order-of-magnitude bound $\Lambda \lesssim 2\times10^{-53}\ {\rm m^2}$
  from Fermi-LAT (GRB 090902B), for photons.

## 5. Conditional comparison with gravitational-wave bounds

For the quartic case of the modified dispersion relation used by LVK,
$E^2 = p^2c^2 + A_4\,p^4c^4$, the flat-space relation is
$\omega^2 = c^2k^2\left[1 + A_4(\hbar c)^2 k^2\right]$, so

$$\Lambda_{\rm phen} = A_4\,(\hbar c)^2 = A_4 \times 3.894\times10^{-14}\ {\rm m^2\,eV^2}.$$

The GWTC-4.0 combined result (arXiv:2603.19020, 83 events, 90% credible interval,
group-velocity convention) is $A_4 \in [-620,\,+190]\ {\rm eV^{-2}}$, i.e.
$\Lambda_{\rm phen} \in [-2.4\times10^{-11},\,+7.4\times10^{-12}]\ {\rm m^2}$; for $\Lambda > 0$ the upper
end applies.

This comparison constrains the phenomenological coefficient. It constrains the Lambda model's
own parameter only if $\Lambda_{\rm GW} = \Lambda$ (Route A, universal coefficient) is justified.
Under that assumption the naturalness estimates of Section 4 lie 52–58 orders of magnitude below
the gravitational-wave bound, and the photon bound lies about 41 orders below it: a
gravitational-wave propagation effect would then be unobservable, and a detected one would
contradict the model's own estimate rather than support it.

## 6. What would make the model predictive for gravitational waves

1. An action, or an equivalent dynamical equation, for tensor perturbations containing the
   quartic local-frame correction.
2. An argument whether $\Lambda_{\rm GW}$ equals the scalar or photon coefficient, differs from it
   (and why), or is not defined by the same parameter.
3. The observable phase correction for gravitational waves, including propagation through the
   cosmological background and the local-to-observer frequency conversion.
4. The sign and magnitude from the theory, or an explicit statement that $\Lambda_{\rm GW}$ is a free
   phenomenological parameter.
5. A resolution of Route A versus Route B.

## 7. Consequences for data analysis

The residual analysis of Module 1 (residual-first-framework; Zenodo doi:10.5281/zenodo.23324211)
tests a specified phenomenological dispersion template of the class $\alpha \approx 2$–$4$. Until
the tensor-sector derivation exists, it is not a test of a predicted value of the Lambda model,
and new gravitational-wave data (for example GWTC-5.0) would only tighten a phenomenological
bound. The methodological results stand independently of this limitation.

## Conclusion

The current framework gives a mathematical structure for quartic dispersion in curved spacetime
and a scalar-field derivation of it. It does not yet give a derived, numerically predictive
$\Lambda_{\rm GW}$. The next milestone is theoretical: the derivation connecting the Hamiltonian
to the dynamics and observable phase of gravitational waves.
