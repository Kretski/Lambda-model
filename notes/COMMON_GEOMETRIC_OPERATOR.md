# Common geometric operator: does a shared spatial structure make Λ universal?

*October 2026. Follows notes/UNIVERSALITY_AUDIT.md. A limited audit of one question, not a
treatment of the radiative stability of Lorentz invariance. Markers: [construction] built here;
[known] established in the literature; [conclusion] what follows.*

## Questions

1. Can the Λ correction be written, for all fields, with one operator built from the geometry of
   the preferred spatial slices?
2. If so, does that force one dispersion coefficient for all fields, before and after quantum
   corrections?

## 1. The common structure exists [construction]

With a unit timelike $u^\mu$ ($u^\mu u_\mu = -1$) normal to the preferred slices, the projector
$q_{\mu\nu} = g_{\mu\nu} + u_\mu u_\nu$ defines the induced metric and its covariant derivative $D_i$, and hence
the Laplacian $D^2 = q^{ij}D_iD_j$ of the slices, acting on scalars and (as the rough Laplacian) on
spatial vectors and tensors. In a local inertial frame aligned with $u^\mu$ it reduces to $\nabla^2$.
Lowest-order quadratic terms giving $\omega^2 = k^2(1 + \Lambda_A k^2)$ in flat space:

| Field $A$ | Term added to the Lagrangian | Flat-space dispersion |
| --- | --- | --- |
| scalar $\phi$ | $-\tfrac{\Lambda_\phi}{2}\,(D^2\phi)^2$ (A0) | $\omega^2 = k^2(1+\Lambda_\phi k^2)$ |
| photon, $u$-frame fields $E_i$, $B_i$ | $-\tfrac{\Lambda_\gamma}{2}\,D_iB_j\,D^iB^j$ | $\omega^2 = k^2(1+\Lambda_\gamma k^2)$, both polarisations (isotropic, no birefringence) |
| tensor modes $h_{ij}^{\rm TT}$ | $-\tfrac{M_{\rm Pl}^2}{8}\Lambda_g\,(D^2h_{ij})^2$ | $\omega^2 = k^2(1+\Lambda_g k^2)$ (the Hořava–Lifshitz form, THEORY_STATUS_GW.md §8) |

So one geometric object, $D^2$ of the preferred slices, can carry the correction for every field:
answer to question 1 is **yes**, given the preferred foliation.

## 2. It does not fix the coefficients [conclusion]

The symmetry that remains with a preferred foliation (diffeomorphisms preserving the foliation)
allows each of the three terms separately, each with its own coefficient $\Lambda_\phi$, $\Lambda_\gamma$, $\Lambda_g$;
nothing in the construction relates them. Equal coefficients would require an additional
principle, for example a **universal-coupling postulate**: all fields propagate with one common
operator $g^{\mu\nu}\partial_\mu\partial_\nu + \Lambda\,(D^2)^2$, in the way all fields couple to one metric in
general relativity. That is a postulate, not a consequence: the equivalence principle constrains
couplings to the metric, not the coefficients of additional higher-derivative operators.

Even with such a postulate, quantum corrections generically generate field-dependent terms,
including dimension-4 differences in the limiting speeds (Collins et al., Phys. Rev. Lett. 93,
191301 (2004)) [known]. Keeping the coefficients equal would need a symmetry that forbids the
differences; the model does not have one.

## Result

Of the three possible outcomes, this is the second: **a common geometric structure exists, but it
does not by itself make Λ universal.** Universality requires an explicit universal-coupling
postulate and, beyond tree level, a protection mechanism; neither is part of the Lambda model at
present. The coefficients $\Lambda_\phi$, $\Lambda_\gamma$, $\Lambda_g$ therefore remain independent phenomenological
parameters, and the photon bounds of SENSITIVITY_MAP.md constrain $\Lambda_\gamma$, not $\Lambda_g$.

Not claimed here: a solution of the radiative-stability problem, a protected universality, or
any new gravitational-wave prediction.
