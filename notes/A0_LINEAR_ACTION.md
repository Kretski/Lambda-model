# A0 — From the Paper 1 action to a linear quartic-dispersion operator

*October 2026. Status: audit of Paper 1 plus a candidate action for checking. Papers 1–3 are not
changed. The candidate is a proposal, not yet the established formulation of the Lambda model.
Symbolic checks: sympy (flat space), reproduced in `notes/a0_checks.py`.*

Conventions: signature $(-,+,+,+)$, $\hbar = c = 1$ unless stated, canonically normalised real
scalar $\phi$ with mass dimension 1 in four dimensions.

## 1. The Paper 1 term does not produce a linear dispersion

Paper 1 (Eq. 7) uses

$$\mathcal L_1 = -\frac12\left[g^{\mu\nu}\partial_\mu\phi\,\partial_\nu\phi + \frac{\Lambda}{2}\,Y^2\right],
\qquad Y = h^{\mu\nu}\partial_\mu\phi\,\partial_\nu\phi .$$

- **Linearisation.** $Y^2$ is quartic in $\phi$. Linearising the field equation about $\phi = 0$
  removes it entirely: in flat space with the static observer the linear equation is
  $-\ddot\psi + \nabla^2\psi = 0$, with no $\Lambda$ (checked symbolically). Small-amplitude waves
  therefore propagate exactly as in general relativity. A $\Lambda$ effect exists only around a
  background with $Y \neq 0$, where it is amplitude-dependent and still second order in derivatives.
- **Dimension.** $X = g^{\mu\nu}\partial\phi\partial\phi$ has mass dimension 4, so $Y^2$ has dimension 8 and
  its coefficient has dimension $M^{-4}$ (length$^4$). Paper 1 calls it a dimension-6 operator with
  $\Lambda = c_6/M_{\rm QG}^2$; that identification does not match this term.
- **Origin of the $k^4$ in Paper 1.** The step from Eq. 10 to Eqs. 11–12 replaces $\partial_\mu\phi$ by the
  wave vector $k_\mu$ with unit amplitude. This gives a characteristic condition of the nonlinear
  equation, not the dispersion relation of linear waves.

## 2. A linear candidate with the intended dispersion

Flat space, preferred frame $u = \partial_t$:

$$\mathcal L_2 = \frac12\dot\phi^2 - \frac12(\nabla\phi)^2 - \frac{\Lambda}{2}\left(\nabla^2\phi\right)^2 .$$

- **Field equation** (higher-derivative Euler–Lagrange): $\ddot\phi - \nabla^2\phi + \Lambda\nabla^4\phi = 0$.
- **Dispersion:** for $\phi \propto e^{i(kx - \omega t)}$,

$$\omega^2 = k^2 + \Lambda k^4 = k^2\left(1 + \Lambda k^2\right),$$

  the relation of Paper 1 (Eq. 3) in flat space, now for linear waves.
- **Dimension:** $(\nabla^2\phi)^2$ has dimension 6, so $\Lambda$ has dimension $M^{-2}$ (length$^2$), consistent
  with $\Lambda = c_6/M^2$.
- **Time derivatives** stay second order; only spatial derivatives are higher. There is no
  Ostrogradsky ghost, at the price of a preferred frame (as in Hořava–Lifshitz gravity).
- **Sign from stability.** The energy density is
  $\mathcal H = \frac12\dot\phi^2 + \frac12(\nabla\phi)^2 + \frac{\Lambda}{2}(\nabla^2\phi)^2$; per unit plane-wave amplitude its
  average is $k^2(1+\Lambda k^2)/2$. It is positive for $\Lambda \ge 0$. For $\Lambda < 0$, $\omega^2 < 0$ when
  $k^2 > 1/|\Lambda|$: short-wavelength instability. Within the regime where this effective
  description holds, **$\Lambda > 0$ follows from stability** rather than being imposed. (If the
  effective description is cut off below $k \sim |\Lambda|^{-1/2}$, the sign is not fixed by this argument.)
  This is a flat-space statement: it does not imply the stability of every solution on a Kerr
  background, which is part of A1–A2.

## 3. Curved background with the ZAMO foliation

In Kerr, the ZAMO four-velocity is the unit normal to the Boyer–Lindquist slices
$t = {\rm const}$ (lapse $N$, induced metric $\gamma_{ij}$, $\sqrt{-g} = N\sqrt\gamma$). The natural spatial
operator is the Laplacian of those slices, $D^2\phi = \gamma^{ij}D_iD_j\phi$. Candidate:

$$S_2 = -\frac12\int d^4x\,\sqrt{-g}\left[g^{\mu\nu}\partial_\mu\phi\,\partial_\nu\phi + \Lambda\,(D^2\phi)^2\right].$$

Varying, with integration by parts on each slice:

$$\frac{1}{\sqrt{-g}}\,\partial_\mu\!\left(\sqrt{-g}\,g^{\mu\nu}\partial_\nu\phi\right) - \frac{\Lambda}{N}\,D^2\!\left(N\,D^2\phi\right) = 0 .$$

**Eikonal limit.** With $\phi \sim e^{iS}$, $k_\mu = \partial_\mu S$, the leading symbol is
$g^{\mu\nu}k_\mu k_\nu + \Lambda\,(\gamma^{ij}k_ik_j)^2 = 0$. Because $\gamma^{ij}k_ik_j = g^{\mu\nu}k_\mu k_\nu + (n^\mu k_\mu)^2 = k_{\rm loc}^2$
for the normal (ZAMO) observer, this is exactly the Hamiltonian of Paper 2 (Eqs. 1 and 5). The
candidate therefore reproduces the ray dynamics of Paper 2 from a linear wave equation.
(Only the flat-space statements are checked symbolically in `a0_checks.py`. The curved field
equation above, including the factor $1/N$, and its eikonal limit on Kerr are derived by hand and
are to be verified symbolically in A1.)

## 4. What A0 establishes and what remains

Established (1 and 2 checked symbolically; 3 derived by hand):
1. The Paper 1 quartic term gives no linear dispersion; its coefficient has dimension 8, not 6.
2. A linear, dimension-6 operator with higher spatial derivatives gives $\omega^2 = k^2(1+\Lambda k^2)$,
   has no higher time derivatives, and is stable for $\Lambda > 0$.
3. On Kerr, the ZAMO-foliation version is expected to reproduce the Paper 2 Hamiltonian in the
   eikonal limit (derived by hand; symbolic verification in A1).

Open (for A1–A2):
1. Symbolic verification of the curved field equation and of its separability on Kerr.
2. Boundary conditions for quasinormal modes, recovery of the Kerr scalar spectrum at
   $\Lambda = 0$, and the first-order correction in $\Lambda$.
3. This remains a **scalar** equation. It is not yet a theory of gravitational perturbations
   (THEORY_STATUS_GW.md, Sections 3 and 8).
4. Whether Papers 1–3 need an erratum or a revision of the action and of the EFT identification
   is to be decided after A1; until then this note records the issue and the candidate fix.
