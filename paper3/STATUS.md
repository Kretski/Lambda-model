# Paper 3 — Verified Status

## Verified

### Lambda dispersion fitting

The repository contains an operational fitting implementation for

    omega = c k sqrt(1 + 2 Lambda k^2).

The synthetic self-test recovers a known injected Lambda:

    Lambda_true = 0.05

with fitted value approximately

    Lambda_fit = 0.04948363.

This demonstrates numerical recovery of the injected parameter.

It does NOT constitute experimental evidence.

---

## Verified

### Fiber structural analysis

For the supplied demonstration dataset:

    Omega ~ |Delta k|^p

gives

    p = 3.998708879978

with

    R^2 = 0.999997902443.

The mean local log-log slope is

    3.99521169.

The dataset is therefore consistent with a quartic regime.

---

## Verified

### Fiber / Lambda structural distinction

The Lambda model requires a low-k linear branch with a quartic
correction:

    omega^2 = c^2 k^2 (1 + 2 Lambda k^2).

The supplied fiber data instead behave approximately as

    Omega ~ Delta k^4.

The Lambda-model fit is therefore not competitive with the quartic
description for this supplied dataset.

---

## Verified

### Dimensional refusal

The measured fiber quantity beta4 is reported with dimensions

    s^4 / m.

Since

    [beta4/c^2] = s^6/m^3,

the expression

    beta4/(4c^2)

cannot represent Lambda with dimensions m^2.

The validator intentionally refuses this mapping.

---

## Not established

The following claims are NOT currently established:

- independent experimental measurement of Lambda from the fiber data;
- universal applicability of the Lambda model to fiber event horizons;
- independent Dirac-material measurement of Lambda;
- experimental confirmation of the Lambda model from the current
  demonstration datasets.

---

## Interpretation

The strongest currently defensible core is:

1. the Lambda dispersion model is numerically implementable;
2. synthetic data recover an injected Lambda;
3. the BEC mapping provides the primary physical realization;
4. the supplied fiber regime is structurally different;
5. dimensional checks prevent an invalid beta4 -> Lambda conversion.

This narrower interpretation is intentional.