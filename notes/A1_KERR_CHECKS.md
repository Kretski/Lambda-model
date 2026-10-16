# A1 — Symbolic checks of the candidate scalar equation on Kerr

*October 2026. Continues A0 (notes/A0_LINEAR_ACTION.md). Checks reproduced by
`notes/a1_checks.py` (sympy). No quasinormal modes are computed here. Papers 1–3 are unchanged.*

Background: Kerr, Boyer–Lindquist coordinates, signature $(-,+,+,+)$, $G = c = 1$. Lapse
$N = \sqrt{\Sigma\Delta/A}$, ZAMO angular velocity $\Omega = 2Mar/A$, induced metric of the slices
$t = {\rm const}$: $\gamma = {\rm diag}(\Sigma/\Delta,\ \Sigma,\ A\sin^2\theta/\Sigma)$; the ZAMO four-velocity is the
unit normal of these slices. Candidate action (A0):

$$S_2 = -\frac12\int d^4x\,\sqrt{-g}\left[g^{\mu\nu}\partial_\mu\phi\,\partial_\nu\phi + \Lambda\,(D^2\phi)^2\right].$$

## Results

| Check | Result |
| --- | --- |
| 1. Eikonal symbol: $g^{\mu\nu}k_\mu k_\nu + (n^\mu k_\mu)^2 = \gamma^{ij}k_ik_j$ for the ZAMO | identity, simplifies to 0 exactly |
| 2. Euler–Lagrange equation of $S_2$ equals $\sqrt{-g}\,[\,\Box\phi - (\Lambda/N)\,D^2(N D^2\phi)\,]$ | agreement to $9\times10^{-16}$ (relative) at random points, $a = 0.7$, $\Lambda = 0.1$, test field with $m = 2$ |
| 3a. Separation for $a = 0$ | $D^2(R(r)Y_{lm})/Y_{lm}$ has no $\theta$ dependence: separable with spherical harmonics |
| 3b. Carter constant on the null shell, $\{Q, H\}$ | GR ($\Lambda = 0$, $a = 0.9$): $3\times10^{-19}$ (zero); $\Lambda = 0.1$, $a = 0.9$: $6.6\times10^{-4}$ (non-zero); $\Lambda = 0.1$, $a = 0$: zero |

Consequences:

1. **The candidate is a linear wave equation whose eikonal limit is exactly the Paper 2
   Hamiltonian** (check 1). The ray results of Paper 2 now sit on a linear field equation.
2. **The field equation, including the factor $1/N$, is confirmed** (check 2):
   $$\Box\phi - \frac{\Lambda}{N}\,D^2\!\left(N\,D^2\phi\right) = 0 .$$
3. **Separability.** Schwarzschild separates (spherical symmetry). For $a \neq 0$ the Carter constant
   is not conserved by the dispersive Hamiltonian (independent confirmation of Paper 2, Appendix A),
   so the wave equation is not expected to separate in the Carter/Teukolsky manner.

## Issues that A2 must address before any quasinormal-mode number

1. **Rotating case.** Without separation, Kerr modes need either a two-dimensional $(r,\theta)$
   eigenvalue problem or perturbation theory to first order in $\Lambda$ around the GR modes
   (which does not require separability).
2. **Order of the radial equation.** For $a = 0$ the radial equation is fourth order. At large $r$ the
   flat dispersion $\omega^2 = k^2 + \Lambda k^4$ has two propagating roots and two evanescent roots
   ($k \sim \pm i/\sqrt\Lambda$); the boundary condition must select the outgoing and the decaying one.
3. **The horizon.** The preferred foliation is the Boyer–Lindquist slicing, whose lapse vanishes at
   the horizon ($N \to 0$); the ZAMO normals become null there. With $\Lambda > 0$ high-wavenumber
   modes are superluminal, so the Killing horizon is not a causal boundary for them. The standard
   "ingoing at the horizon" condition is therefore not automatically justified, and the singular
   behaviour of $N$ enters the operator directly. In preferred-frame theories this question is
   tied to a foliation that is regular across the horizon (and to "universal horizons");
   whether the Boyer–Lindquist/ZAMO choice is admissible near the horizon is open.
4. **Stability on Kerr** is not implied by the flat-space argument of A0.
5. This is still a **scalar** equation, not a theory of gravitational perturbations.

## Next step (A2), proposed order

(a) Schwarzschild only: write the fourth-order radial equation, analyse its behaviour at large
$r$ and near the horizon, and decide whether consistent quasinormal boundary conditions exist
with the Boyer–Lindquist foliation. (b) Only if they do: first-order-in-$\Lambda$ frequency shifts,
checked against the eikonal estimate of Paper 2 at large $l$. (c) Kerr only after (a)–(b).
