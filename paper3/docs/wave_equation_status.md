# Paper 3 — Wave Equation Numerical Status

## Verified PASS

The following components have been independently tested:

- angular derivative operator;
- second angular derivative operator;
- inverse Kerr metric;
- Kerr metric identity;
- GR QNM frequencies against the independently published `qnm`
  implementation.

The tested GR QNM sector is:

```text
s = 0
l = m = 2
n = 0

Validated frequencies include:

a = 0.0 : 0.48364387 - 0.09675878 i
a = 0.3 : 0.53697917 - 0.09583854 i
a = 0.6 : 0.61736371 - 0.09124540 i
a = 0.9 : 0.78163829 - 0.06928931 i
Verified FAIL

The finite-domain perturbative solver does not exhibit grid or domain
convergence.

The instability remains after correcting:

inverse Kerr metric sign;
angular derivative transformation;
spin and horizon-radius parameter handling.
Architectural diagnosis

The present outer boundary is Dirichlet.

This is a hard-wall condition and therefore represents a finite cavity,
not the outgoing boundary condition required for a true quasinormal mode.

The current solver should therefore not be presented as a validated
full QNM solver.

Current status
Operators:                 VERIFIED
Kerr metric:              VERIFIED
GR frequency reference:   VERIFIED
Perturbative QNM solver:  NOT CONVERGED
Physical outgoing BC:     NOT IMPLEMENTED
Future work

A structurally sound QNM implementation requires an outgoing-wave
treatment, such as hyperboloidal compactification, or an independent
cross-check against a method such as Leaver's continued fraction in
the appropriate unperturbed sector.
Experimental dispersion validator

Location:

dispersion/lambda_experimental_validator.py

The validator tests reported experimental beta4 scaling.

Current result:

R² = 0.99876966
Delta AIC = 18.10139008

However, dimensional analysis gives:

[beta4] = s^4 / m

and:

[beta4 / c^2] = s^6 / m^3

Therefore the code deliberately refuses to identify beta4/(4c²)
with Lambda.

This is a successful validation guard, not an error.

Fiber structural test

Location:

dispersion/fiber_event_horizon_structural_test.py

For the supplied Delta-k/Omega regime:

p = 3.998708879978
mean local slope = 3.99521169

The supplied regime is therefore approximately quartic.

The Lambda model is not competitive with the supplied data.

This result is retained as a structural negative test.

Wave-equation solver

The wave-equation work is documented separately.

The following have been verified:

angular operators;
Kerr metric identity;
GR QNM reference frequencies.

The finite-domain perturbative solver itself does not converge.

The identified architectural issue is the hard Dirichlet outer boundary,
which is not the outgoing boundary condition of a true QNM problem.

Therefore the current implementation is not claimed to be a validated
full QNM solver.

Platform status
BEC

Validated mapping / primary platform.

Fiber

Structural negative test.

The supplied regime is approximately quartic and is not identified
with the Lambda dispersion model.

Experimental beta4 dataset

Scaling result only.

The dimensional audit prevents independent Lambda identification.

Dirac / tilted Dirac

Not independently validated.

No independent Lambda measurement is claimed.

Scientific policy

This repository follows a strict distinction between:

experimental scaling;
dimensional consistency;
parameter identification;
model compatibility;
numerical convergence.

A strong fit is not automatically treated as a physical measurement.

A failed numerical test is not hidden when it identifies a structural
limitation of the method.

The purpose of the repository is therefore not only to demonstrate
positive results, but also to make unsupported interpretations
computationally difficult.