# Convention Audit — Paper 3 vs Ozeri (2002) Healing-Length Definitions

## Purpose

Before applying the Λ-model's BEC mapping to Ozeri et al.'s data, the
healing-length convention used in each source must be reconciled. Using
mismatched conventions without checking would produce an apparent
"factor of two" discrepancy that is purely notational, not physical —
exactly the kind of error this project's methodology is designed to
catch (cf. the Λ_NH vs Λ_GW dimensional-incompatibility discussion in
`paper3/PAPER3_VALIDATION_STATUS.md`).

## Source 1: Paper 3 (`examples/example_BEC.py`)

Verbatim from the script's docstring and implementation:

```
omega(k) = c_s * k * sqrt(1 + (xi*k/2)^2)
c_s = sound speed = sqrt(g*n0/m)
xi  = healing length = hbar/(m*c_s)
```

with the stated identity `Lambda = xi^2 / 4`.

Define:
```
xi_P ≡ hbar / (m * c_s)
Lambda_P ≡ xi_P^2 / 4
```

## Source 2: Ozeri, Steinhauer, Katz & Davidson (2002), arXiv:cond-mat/0112496

Verbatim from the paper (Bogoliubov dispersion, as quoted in-text):

```
E_k = sqrt(E_r * (E_r + 2*g*n))
E_r = (hbar*k)^2 / (2m)
xi  = sqrt(hbar^2 / (2*m*g*n))      [healing length, as explicitly defined in the paper]
```

Define:
```
xi_O ≡ sqrt(hbar^2 / (2*m*g*n))
```

## Independent derivation (performed directly from the Ozeri formula, not assumed)

Starting from the Ozeri Bogoliubov form and dividing by ħ² (ω = E_k/ħ):

```
omega^2 = E_r^2/hbar^2 + 2*g*n*E_r/hbar^2
        = (hbar^2 k^4)/(4 m^2) + (g n k^2)/m
```

With c_s^2 ≡ g*n/m (same definition in both sources):

```
omega^2 = c_s^2 k^2 + (hbar^2 / (4 m^2)) k^4
        = c_s^2 k^2 * [1 + (hbar^2 / (4 m^2 c_s^2)) k^2]
        = c_s^2 k^2 * [1 + (xi_P * k / 2)^2]
```

using `xi_P = hbar/(m*c_s)` exactly as defined in Source 1. This is an
exact algebraic identity — it reproduces the Paper-3 dispersion form
starting *only* from the Ozeri paper's own Bogoliubov formula, with no
free parameters or assumptions beyond `c_s^2 = gn/m` (shared by both
sources).

## Relation between the two healing-length definitions

```
xi_P^2 = hbar^2 / (m^2 c_s^2) = hbar^2 / (m * g * n)
xi_O^2 = hbar^2 / (2 * m * g * n)

=>  xi_P^2 = 2 * xi_O^2
=>  xi_P = sqrt(2) * xi_O
```

## Resulting quartic-coefficient equivalence

```
Lambda_P = xi_P^2 / 4 = (2 * xi_O^2) / 4 = xi_O^2 / 2
```

**Boxed result:**

```
Lambda_Paper3 = xi_P^2 / 4  =  xi_O^2 / 2  =  Lambda_pred (in Ozeri's own
                                                variables)
```

This is a **convention conversion, not a physical discrepancy**. The two
sources predict the identical physical dispersion relation; they simply
parametrize the healing length with different numerical prefactors
(a common situation in the BEC literature — some authors define ξ via
the coefficient of the linear term, others via the coefficient inside
the square root, differing by a factor of √2).

## Summary table

| | Paper 3 (`example_BEC.py`) | Ozeri et al. (2002) |
|---|---:|---:|
| Healing length | ξ_P = ħ/(m·c_s) | ξ_O = √(ħ²/(2mgn)) |
| c_s² | gn/m | gn/m (same) |
| Relation | ξ_P = √2 · ξ_O | — |
| Quartic coefficient (native form) | Λ = ξ_P²/4 | Λ = ξ_O²/2 |
| Physical Λ | **same** | **same** |

## Dimensionless (kξ) form used in the fit

In terms of the dimensionless variable `x ≡ k·ξ_O` (the natural variable
in which Ozeri report their data, `kξ` on the Fig. 5 axis), the unified
model used in `fit_bogoliubov_lambda.py` is:

```
omega(x) = A * x * sqrt(1 + beta * x^2)
```

where `beta ≡ Λ/ξ_O²`. In this parametrization:

- `beta = 0`   → pure linear (free-phonon) dispersion, M0
- `beta = 1/2` → exactly the Bogoliubov dispersion / Λ-model prediction
                  with Λ = ξ_O²/2 = ξ_P²/4 (Paper-3 native value), M_B
- `beta` free  → exploratory fit, M_Lambda_free

**This equivalence (M_B at beta=0.5 ⟺ Paper-3's Λ=ξ_P²/4) is enforced in
code as an explicit runtime assertion** (see `BETA_PRED` derivation in
`fit_bogoliubov_lambda.py`), so that a future edit cannot silently
reintroduce a factor-of-two error without the assertion failing.

## Status

**Convention audit: PASSED.** No physical discrepancy found. Both the
Ozeri-native form (`ξ_O²/2`) and the Paper-3-native form (`ξ_P²/4`) refer
to the identical physical Λ, connected by `ξ_P = √2·ξ_O`. This document
and the code assertion together serve as the permanent record of this
check, so that an external reviewer encountering `ξ²/2` in one context and
`ξ²/4` in another does not mistake the difference for an error.
