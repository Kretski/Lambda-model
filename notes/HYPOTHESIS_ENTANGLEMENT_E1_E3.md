# Internal note — hypothesis: Λ from quantum entanglement (checks E1–E3)

*October 2026. Status: hypothesis under screening. Not part of Papers 1–3; no Paper 4 is
opened on its basis. This note records the checks a mechanism must pass before further work.*

## Hypothesis

The dispersion coefficient is not a free constant but emerges from quantum correlations,
for example in the simplest form

$$\Lambda = \ell_P^2\, S_{\rm ent}, \qquad \ell_P^2 = \hbar G/c^3 \approx 2.61\times10^{-70}\ {\rm m^2}.$$

Prior work on spacetime from entanglement (e.g. Jacobson 1995; Van Raamsdonk 2010; Swingle,
Annu. Rev. Condens. Matter Phys. 2018) and on $k^4$/$k^6$ dispersion in cosmology
(e.g. Mukohyama, JCAP 2009, scale-invariant perturbations in Hořava–Lifshitz gravity without
inflation) means that neither "spacetime emerges from entanglement" nor "quartic dispersion
changes early-universe dynamics" is new. A contribution would have to be the specific mechanism
that fixes Λ.

## E1 — dimensional and numerical screening (done)

Scope: this excludes the simple proportionality $\Lambda = \ell_P^2 S$ with a horizon entropy. It
does not exclude every model connecting entanglement and Λ.

With the Bekenstein–Hawking entropy $S = A/(4\ell_P^2)$, $A = 4\pi r_s^2$:

$$\Lambda = \ell_P^2 S = A/4 = \pi r_s^2 .$$

| Horizon | Radius | $\Lambda = \pi R^2$ |
| --- | --- | --- |
| black hole, $1\,M_\odot$ | $r_s = 2954$ m | $2.7\times10^{7}\ {\rm m^2}$ |
| Hubble horizon ($H_0 = 67.7$ km/s/Mpc) | $1.37\times10^{26}$ m | $5.9\times10^{52}\ {\rm m^2}$ |

**(a) Internal check, independent of universality.** If Λ is tied to the entropy of a black hole,
its effect on that black hole's own perturbations is scale-free: in the units of Paper 2,
$\hat\Lambda = \Lambda/M_{\rm geom}^2 = 4\pi \approx 12.6$ for every mass. For the $l=2$ ringdown
($\omega M_{\rm geom}/c \approx 0.374$), $\Lambda k^2 \approx 1.75$, and the linear eikonal correction
of Paper 2 (Eq. 20, $\delta\Omega_{\rm ph}/\Omega_{\rm ph}^{\rm GR} \simeq -0.219\,\hat\Lambda$) would give
$-2.75$, far outside its range of validity. The dispersion would change every black-hole ringdown
at order unity, while observed ringdowns are consistent with general relativity within tens of
percent. The simple horizon-entropy form is therefore excluded within the framework itself.

**(b) Comparison with propagation bounds (requires universality and Route A).** The solar-mass
value exceeds the GWTC-4.0 bound ($\Lambda_{\rm phen} \lesssim 7.4\times10^{-12}\ {\rm m^2}$) by about 19 orders
of magnitude and the photon estimate of Paper 1 ($\lesssim 2\times10^{-53}\ {\rm m^2}$) by about 60.

**Consequence.** A viable entanglement mechanism must give Λ close to $\ell_P^2$ without a large
entropy factor, or a functional form that is not a simple proportionality to a horizon entropy.
In the first case the entanglement input changes nothing measurable; in the second case E2 and
E3 must be passed with that specific form.

## E2 — origin of the operator (open)

Can a concrete quantum state and its microscopic dynamics produce the $k^4$ term in the effective
wave equation, without the term being put in by hand? Pass condition: the operator, its local
frame and its regime of validity follow from the stated dynamics.

## E3 — predictive power (open)

Does the mechanism fix the value or at least the sign of Λ, or a relation between the scalar,
photon and tensor coefficients, and does it give an observable that differs from the
preferred-frame theories (THEORY_STATUS_GW.md, Section 8)? Pass condition: at least one
quantitative prediction that is not an input.

## Decision rule

No further numerical work on this hypothesis unless a specific mechanism passes E2 and gives the
coefficient or its sign (E3). Papers 1–3 are not changed on its basis. The current theoretical
priority remains line A: the strong-field predictions from the wave equation.
