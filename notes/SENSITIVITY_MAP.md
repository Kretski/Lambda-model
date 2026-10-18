# Sensitivity map: where can Λ be seen, and where are the blind spots?

*October 2026. Literature bounds converted to the Lambda-model parameter. Markers: [source] =
published bound; [conversion] = done here; [assumption] = needed for the bound to apply to Λ.
Orders of magnitude only; conventions may differ by factors of order unity.*

## Conversion

The model's dispersion $\omega^2 = c^2k^2(1+\Lambda k^2)$ gives a group velocity $v \simeq c\,(1 + \tfrac32\Lambda k^2)$:
for $\Lambda > 0$ (the sign required by stability, `notes/A0_LINEAR_ACTION.md`) waves are
**superluminal**. In the usual quadratic (n = 2) parametrisation,
$v = c\,[1 - s\,\tfrac32 (E/E_{\rm QG,2})^2]$ with $s = -1$ for superluminal, or equivalently
$E^2 = p^2c^2 + p^4c^4/E_{\rm QG,2}^2$, this is [conversion]

$$\Lambda = \left(\frac{\hbar c}{E_{\rm QG,2}}\right)^2,\qquad \hbar c = 1.973\times10^{-16}\ {\rm GeV\,m}.$$

Only **superluminal** bounds apply. For gravitational waves, $\Lambda = A_4(\hbar c)^2$ (THEORY_STATUS_GW.md, §5).

## Map (superluminal, quadratic)

| Channel | Probe | Bound [source] | Λ upper limit [conversion] | Assumptions for it to bound Λ |
| --- | --- | --- | --- | --- |
| Gravitational waves | LVK, 83 events, time/phase of GW signals | $A_4 \le 190\ {\rm eV^{-2}}$ (GWTC-4.0, arXiv:2603.19020) | $7\times10^{-12}\ {\rm m^2}$ | $\Lambda_{\rm GW} = \Lambda$, Route A |
| Photons, time of flight | LHAASO, GRB 221009A, 0.2–13 TeV | $E_{\rm QG,2} > 4.6\times10^{-8}E_{\rm Pl}$ (Piran & Ofengeim, PRD 109, L081501, 2024); $> 7.2\times10^{11}$ GeV (Yang, Bi & Yin, arXiv:2312.09079); $> 2.4\times10^{12}$ GeV (DisCan, Chin. Phys. C) | $10^{-55}$ to $7\times10^{-57}\ {\rm m^2}$ | photon coefficient $= \Lambda$, Route A |
| Photons, decay $\gamma\to e^+e^-$ | PeV photons | $E_{\rm QG,2} > 0.8\times10^{14}$ GeV (review, arXiv:2109.09072) | $6\times10^{-60}\ {\rm m^2}$ | electron sector Lorentz invariant |
| Photons, splitting $\gamma\to N\gamma$ | HAWC; LHAASO (Crab, J2032+4102, up to 1.4 PeV) | $E_{\rm QG,2} > 1.0\times10^{15}$ GeV (HAWC); $> 2.0\times10^{16}$ GeV (LHAASO) (both as quoted in arXiv:2109.09072) | $4\times10^{-62}$; $1\times10^{-64}\ {\rm m^2}$ | splitting rate model |
| Neutrinos, vacuum pair emission | IceCube PeV cascades | $M_{\rm QG} \gtrsim 10^{-4} M_{\rm Pl}$ (Borriello et al., PRD 87, 116009, 2013) | $\sim3\times10^{-62}\ {\rm m^2}$ | neutrino coefficient $= \Lambda$ |
| Natural values (Paper 1, §6) | — | $\Lambda = c_6/M^2$, $c_6 \sim 1$ | GUT $\approx 4\times10^{-64}$; Planck $\approx 3\times10^{-70}\ {\rm m^2}$ | — |

## Reading the map

1. **If Λ is universal** (the same for photons, neutrinos and gravitational waves), the photon
   channels dominate by 50–53 orders of magnitude, and the strongest one (photon splitting,
   LHAASO) is already **below the GUT-scale natural value**: a universal Λ with $c_6\sim1$ at the GUT
   scale is excluded, a Planck-scale one is not reached by any channel.
2. **Blind spot 1 — gravity-only Λ.** If only gravitational perturbations disperse, the only probe is
   gravitational waves, which reach $\sim10^{-11}\ {\rm m^2}$, about 58 orders above the Planck-scale
   value. This is where the model is effectively untested, and where future detectors (Einstein
   Telescope, Cosmic Explorer) would help only by a few orders.
3. **Blind spot 2 — Route B.** A local effect without flat-space dispersion escapes every
   propagation channel above; it would show up only in strong-field observables, where the model
   currently has no well-posed wave prediction (A2(a)) and eikonal effects are negligible
   (EIKONAL_D_LAMBDA.md).
4. **Not a blind spot — Λ < 0.** Subluminal dispersion is excluded inside the model by stability;
   observationally it is also strongly constrained, but this map lists only the superluminal side.

## Caveats

Bounds from different methods rest on different assumptions (source emission time, background
light, decay or splitting rates, the electron sector) and are quoted with different conventions;
the entries are orders of magnitude, to be checked against each source before citation.
