# Paper 3 — Dispersion Validation

This directory contains the numerical validation tools associated with
the dispersion analysis of Paper 3.

The repository deliberately separates:

1. direct Lambda-model fitting,
2. dimensional consistency checks,
3. structural dispersion tests,
4. experimental scaling analysis.

## Current status

### Lambda-model validator

`lambda_experimental_validator.py`

Tests the model
    omega = c k sqrt(1 + 2 Lambda k^2)

against supplied `(k, omega)` data.

If no data are supplied, the program runs a synthetic self-test.

The synthetic test is NOT experimental evidence.

It verifies only that the fitting machinery can recover a known
non-zero Lambda from synthetic data.

### Fiber event-horizon test

`fiber_event_horizon_structural_test.py`

Tests whether supplied `(Delta k, Omega)` data have the structural form
    Omega ~ |Delta k|^p

and compares this with a linear-plus-quartic Lambda-type dispersion.

The current demonstration dataset gives
    p ≈ 3.999

and therefore behaves as a pure quartic regime.

This is structurally different from
    omega^2 = c^2 k^2 (1 + 2 Lambda k^2)

which contains a linear low-k branch with a quartic correction.

Therefore the supplied fiber dataset is NOT treated as an independent
measurement of Lambda.

## Scientific interpretation

The repository does not force unrelated experimental platforms into
the Lambda model.

A platform is considered a valid Lambda test only if:

1. its measured observable has the same relevant dispersion structure;
2. the mapping to Lambda is dimensionally valid;
3. Lambda is determined from independently measurable quantities;
4. the resulting prediction can be compared with an independent
   observable.

Failure of any of these conditions is reported explicitly.

## Current platform status

### BEC

Status: retained as the primary physical mapping.

The BEC/Bogoliubov dispersion has the required linear low-k branch with
a higher-order correction.

### Photonic fiber event horizon

Status: structural mismatch in the supplied regime.

The supplied demonstration data are consistent with
    Omega ~ Delta k^4

rather than a linear light-cone plus quartic correction.

This does NOT prove that every fiber event-horizon configuration has
purely quartic dispersion. It establishes only the result for the
supplied regime/data.

### Dirac / tilted materials

Status: not accepted as an independent Lambda validation without an
independent, dimensionally valid mapping and independently measured
parameters.

## Important distinction

A strong power-law fit does not automatically constitute a Lambda
measurement.

For example, a measured quantity can obey
    beta4 ~ n_g^p

with excellent R² while still failing the dimensional requirements
needed to identify beta4 with Lambda.

The validator therefore refuses such an identification rather than
silently applying an invalid conversion.

## Reproducibility

All demonstration calculations are deterministic.

Replace the demonstration arrays with real experimental measurements
before using the scripts for experimental claims.
