# A2(a) — Schwarzschild: boundary behaviour of the candidate equation

*October 2026. Continues A1. Checks: `notes/a2a_checks.py` (sympy). No quasinormal frequencies
are computed. Papers 1–3 are unchanged.*

With $\phi = e^{-i\omega t}R(r)Y_{lm}$, $f = 1 - 2M/r$, $N = \sqrt f$ and the static (for $a = 0$, ZAMO)
slices, the candidate equation $\Box\phi - (\Lambda/N)D^2(ND^2\phi) = 0$ gives the fourth-order radial
equation

$$\frac{\omega^2}{f}R + \frac{(r^2 f R')'}{r^2} - \frac{L}{r^2}R - \frac{\Lambda}{\sqrt f}\,\mathcal D\!\left(\sqrt f\,\mathcal D R\right) = 0,
\qquad \mathcal D F = \frac{\sqrt f}{r^2}\left(r^2\sqrt f F'\right)' - \frac{L}{r^2}F,\quad L = l(l+1).$$

## 1. Large $r$: well posed

The flat dispersion $\omega^2 = k^2 + \Lambda k^4$ gives
$k^2 = \left(-1 \pm \sqrt{1 + 4\Lambda\omega^2}\right)/(2\Lambda)$: one propagating pair ($k^2 \approx \omega^2$) and one
evanescent pair ($k^2 \approx -1/\Lambda$). The physical condition is an outgoing propagating wave plus
the decaying evanescent solution: two conditions, as needed.

## 2. Near the horizon: the GR condition is lost

With $r = 2M + x$ and $R \sim x^s$, at leading order

- general-relativity terms $\sim x^{s-1}\left(2M\omega^2 + s^2/2M\right)$, giving $s = \pm 2iM\omega$ for $\Lambda = 0$
  (the infinitely oscillating ingoing and outgoing solutions);
- $\Lambda$ terms $\sim x^{s-2}\, s(s-1)(2s-1)^2/(16M^2)$, which are more singular and dominate as $x \to 0$.

The near-horizon exponents are therefore $s = 0,\ \tfrac12,\ \tfrac12$ (with a logarithm), $1$: all four
solutions are finite and non-oscillating. The GR behaviour survives only in an intermediate zone
$x_* \ll x \ll M$, with $x_* \sim \Lambda/(8M^3\omega^2)$.

## 3. Consequence

Close to the Killing horizon the dispersion makes high-wavenumber modes superluminal, and the
notion of an ingoing wave disappears from the local solutions. The Boyer–Lindquist (static/ZAMO)
foliation ends at the horizon ($N \to 0$) and says nothing about the interior. The construction
therefore gives **no physical criterion** for selecting two of the four finite near-horizon
solutions: **the quasinormal-mode problem is not well posed with this foliation.**

Imposing the GR ingoing condition in the intermediate zone would be an extra assumption, not a
consequence of the model. A well-posed problem requires a preferred foliation that is regular
across the horizon, as in Einstein-aether and khronometric theories (where the relevant causal
boundary is a universal horizon inside the Killing horizon). That is a change of the setup, to be
decided before any quasinormal-mode solver is written.

## Status

- Large-$r$ boundary conditions: consistent.
- Horizon boundary conditions with the Boyer–Lindquist/ZAMO foliation: not determined by the model.
- Next decision: either (i) reformulate with a horizon-regular foliation and re-derive A1–A2, or
  (ii) record that, as constructed, the Lambda model does not yet define quasinormal modes, which
  limits Paper 2's eikonal QNM estimates (Section 7) to an approximation without a wave-equation
  basis at the horizon.
