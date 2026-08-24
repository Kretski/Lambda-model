#!/usr/bin/env python3
"""
PAPER 3 — FIBER EVENT-HORIZON STRUCTURAL TEST

Purpose
-------
Numerically compare two structurally different low-k models:

MODEL A — Lambda / Bogoliubov-like:
    omega^2 = c^2 k^2 (1 + 2 Lambda k^2)

MODEL B — pure quartic fiber-event-horizon:
    Omega = A * (Delta k)^4

The script does NOT assume beforehand which model wins.

It tests:
    1. quartic scaling
    2. presence/absence of linear low-k term
    3. local log-log slope
    4. model residuals
    5. whether a meaningful Lambda can be inferred
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ================================================================
# CONFIGURATION
# ================================================================

OUTPUT_DIR = Path(
    "fiber_structural_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# 1. SYNTHETIC / DERIVED FIBER TEST DATA
# ================================================================
#
# IMPORTANT:
#
# Replace these arrays with the REAL Delta-k / Omega values
# from the fiber derivation or experimental dataset.
#
# For the structural test we use a pure quartic reference
# with small numerical perturbations.
#
# This is NOT presented as experimental data.
# ================================================================

DK = np.array([
    0.05,
    0.07,
    0.10,
    0.14,
    0.20,
    0.28,
    0.40,
    0.55,
    0.75,
    1.00,
])

A_TRUE = 1.0

OMEGA = (
    A_TRUE *
    DK**4
)

# Small deterministic perturbation.
# This prevents the fit from being artificially perfect.
perturbation = np.array([
     0.010,
    -0.008,
     0.006,
    -0.004,
     0.005,
    -0.006,
     0.004,
    -0.005,
     0.003,
    -0.002,
])

OMEGA = (
    OMEGA *
    (1.0 + perturbation)
)


# ================================================================
# HEADER
# ================================================================

print("=" * 72)
print("PAPER 3 — FIBER EVENT-HORIZON STRUCTURAL TEST")
print("=" * 72)


# ================================================================
# 2. DATA
# ================================================================

DATA = pd.DataFrame({
    "Delta_k": DK,
    "Omega": OMEGA,
})

print("\n[1] INPUT DATA")
print("-" * 72)

print(DATA.to_string(index=False))


# ================================================================
# 3. LOG-LOG FIT
# ================================================================
#
# Omega = A * Delta_k^p
#
# log Omega = log A + p log Delta_k
#
# ================================================================

x = np.log(DK)
y = np.log(np.abs(OMEGA))

p, logA = np.polyfit(
    x,
    y,
    1
)

A = np.exp(logA)

OMEGA_POWER = (
    A * DK**p
)

residual_power = (
    OMEGA -
    OMEGA_POWER
)

relative_power_residual = (
    residual_power /
    OMEGA
)

print("\n[2] POWER-LAW STRUCTURE")
print("-" * 72)

print(
    f"Omega = A * |Delta k|^p"
)

print(
    f"A = {A:.12e}"
)

print(
    f"p = {p:.12f}"
)


# ================================================================
# 4. R²
# ================================================================

y_pred = np.log(
    np.abs(OMEGA_POWER)
)

ss_res = np.sum(
    (y - y_pred)**2
)

ss_tot = np.sum(
    (y - np.mean(y))**2
)

R2_power = (
    1.0 -
    ss_res / ss_tot
)

print(
    f"R² = {R2_power:.12f}"
)


# ================================================================
# 5. LOCAL LOG SLOPE
# ================================================================

local_slope = np.diff(y) / np.diff(x)

DATA["local_slope"] = np.nan

DATA.loc[
    1:,
    "local_slope"
] = local_slope

print("\n[3] LOCAL LOG-LOG SLOPE")
print("-" * 72)

print(
    DATA[
        [
            "Delta_k",
            "Omega",
            "local_slope"
        ]
    ].to_string(index=False)
)

mean_slope = np.mean(
    local_slope
)

print()
print(
    f"Mean local slope = {mean_slope:.8f}"
)

quartic_distance = abs(
    mean_slope - 4.0
)

print(
    f"Distance from quartic = "
    f"{quartic_distance:.8e}"
)


# ================================================================
# 6. TEST FOR LINEAR LOW-k TERM
# ================================================================
#
# General polynomial:
#
# Omega =
#     a1*k
#   + a2*k^2
#   + a3*k^3
#   + a4*k^4
#
# Fit in the low-k region.
#
# ================================================================

LOW_K_N = 5

k_low = DK[:LOW_K_N]
omega_low = OMEGA[:LOW_K_N]

M = np.column_stack([
    k_low,
    k_low**2,
    k_low**3,
    k_low**4,
])

coeffs, *_ = np.linalg.lstsq(
    M,
    omega_low,
    rcond=None
)

a1, a2, a3, a4 = coeffs

print("\n[4] LOW-k POLYNOMIAL")
print("-" * 72)

print(
    "Omega ≈ "
    "a1*k + a2*k² + a3*k³ + a4*k⁴"
)

print(
    f"a1 = {a1:.12e}"
)

print(
    f"a2 = {a2:.12e}"
)

print(
    f"a3 = {a3:.12e}"
)

print(
    f"a4 = {a4:.12e}"
)

if abs(a1) < 1e-6 * max(
    abs(a4),
    1.0
):
    print()
    print(
        "RESULT: no numerically significant "
        "linear term detected."
    )
else:
    print()
    print(
        "RESULT: linear term detected."
    )


# ================================================================
# 7. BOGOLIUBOV / LAMBDA MODEL
# ================================================================
#
# omega^2 = c^2 k^2 (1 + 2 Lambda k^2)
#
# => omega = c |k| sqrt(1 + 2 Lambda k^2)
#
# For structural comparison we normalize c=1.
#
# ================================================================

c = 1.0

# Search Lambda values.
lambda_grid = np.logspace(
    -8,
    8,
    2000
)

best_lambda = None
best_rss = np.inf
best_prediction = None

for Lambda in lambda_grid:

    omega_lambda = (
        c *
        DK *
        np.sqrt(
            1.0 +
            2.0 *
            Lambda *
            DK**2
        )
    )

    rss = np.sum(
        (OMEGA -
         omega_lambda)**2
    )

    if rss < best_rss:
        best_rss = rss
        best_lambda = Lambda
        best_prediction = omega_lambda


print("\n[5] LAMBDA-MODEL COMPARISON")
print("-" * 72)

print(
    "Model:"
)

print(
    "omega = c*k*sqrt(1 + 2*Lambda*k²)"
)

print(
    f"Best positive Lambda = "
    f"{best_lambda:.12e}"
)

print(
    f"RSS = {best_rss:.12e}"
)


# ================================================================
# 8. FREE LINEAR + QUARTIC MODEL
# ================================================================
#
# Omega = b1*k + b4*k^4
#
# This directly tests whether a linear term is needed.
#
# ================================================================

X = np.column_stack([
    DK,
    DK**4,
])

b1, b4 = np.linalg.lstsq(
    X,
    OMEGA,
    rcond=None
)[0]

OMEGA_LIN_QUARTIC = (
    b1 * DK +
    b4 * DK**4
)

rss_lq = np.sum(
    (OMEGA -
     OMEGA_LIN_QUARTIC)**2
)

print("\n[6] LINEAR + QUARTIC MODEL")
print("-" * 72)

print(
    "Omega = b1*k + b4*k^4"
)

print(
    f"b1 = {b1:.12e}"
)

print(
    f"b4 = {b4:.12e}"
)

print(
    f"RSS = {rss_lq:.12e}"
)


# ================================================================
# 9. MODEL COMPARISON
# ================================================================

rss_power = np.sum(
    residual_power**2
)

print("\n[7] MODEL COMPARISON")
print("-" * 72)

print(
    f"Pure power law RSS       = "
    f"{rss_power:.12e}"
)

print(
    f"Linear + quartic RSS     = "
    f"{rss_lq:.12e}"
)

print(
    f"Lambda-model RSS         = "
    f"{best_rss:.12e}"
)


# ================================================================
# 10. STRUCTURAL DECISION
# ================================================================

print("\n[8] STRUCTURAL TEST")
print("-" * 72)

quartic_pass = (
    abs(p - 4.0) < 0.10
)

linear_absent = (
    abs(b1) <
    0.01 * max(
        abs(b4),
        1.0
    )
)

lambda_structural_match = (
    best_rss <
    10.0 * rss_power
)

print(
    "Quartic scaling:",
    "PASS" if quartic_pass else "FAIL"
)

print(
    "Linear term absent:",
    "PASS" if linear_absent else "FAIL"
)

print(
    "Lambda-model competitive:",
    "YES" if lambda_structural_match else "NO"
)


# ================================================================
# 11. SAVE RESULTS
# ================================================================

DATA["omega_power"] = OMEGA_POWER
DATA["relative_power_residual"] = (
    relative_power_residual
)

DATA["omega_lambda"] = best_prediction
DATA["omega_linear_quartic"] = (
    OMEGA_LIN_QUARTIC
)

csv_path = (
    OUTPUT_DIR /
    "fiber_structural_results.csv"
)

DATA.to_csv(
    csv_path,
    index=False
)


# ================================================================
# 12. SVG — LOG-LOG
# ================================================================

svg_log = (
    OUTPUT_DIR /
    "fiber_loglog_structure.svg"
)

fig, ax = plt.subplots(
    figsize=(7, 5)
)

ax.loglog(
    DK,
    OMEGA,
    "o",
    label="fiber data"
)

dk_dense = np.logspace(
    np.log10(DK.min()),
    np.log10(DK.max()),
    500
)

ax.loglog(
    dk_dense,
    A * dk_dense**p,
    "-",
    label=f"power law p={p:.3f}"
)

ax.set_xlabel(
    r"$|\Delta k|$"
)

ax.set_ylabel(
    r"$|\Omega|$"
)

ax.set_title(
    "Fiber event-horizon dispersion structure"
)

ax.grid(
    True,
    which="both",
    alpha=0.25
)

ax.legend()

fig.tight_layout()

fig.savefig(
    svg_log,
    format="svg"
)

plt.close(fig)


# ================================================================
# 13. SVG — MODEL COMPARISON
# ================================================================

svg_compare = (
    OUTPUT_DIR /
    "fiber_model_comparison.svg"
)

fig, ax = plt.subplots(
    figsize=(7, 5)
)

ax.loglog(
    DK,
    np.abs(OMEGA),
    "o",
    label="fiber"
)

ax.loglog(
    DK,
    np.abs(OMEGA_POWER),
    "-",
    label="pure power law"
)

ax.loglog(
    DK,
    np.abs(best_prediction),
    "--",
    label="Lambda model"
)

ax.loglog(
    DK,
    np.abs(OMEGA_LIN_QUARTIC),
    ":",
    label="linear + quartic"
)

ax.set_xlabel(
    r"$|\Delta k|$"
)

ax.set_ylabel(
    r"$|\Omega|$"
)

ax.set_title(
    "Structural model comparison"
)

ax.grid(
    True,
    which="both",
    alpha=0.25
)

ax.legend()

fig.tight_layout()

fig.savefig(
    svg_compare,
    format="svg"
)

plt.close(fig)


# ================================================================
# FINAL
# ================================================================

print("\n[9] OUTPUT")
print("-" * 72)

print(
    f"CSV: {csv_path}"
)

print(
    f"SVG: {svg_log}"
)

print(
    f"SVG: {svg_compare}"
)

print("\n" + "=" * 72)
print("FINAL STRUCTURAL RESULT")
print("=" * 72)

print()
print(
    f"Power-law exponent p = {p:.8f}"
)

print(
    f"Mean local slope       = {mean_slope:.8f}"
)

print(
    f"Linear coefficient b1  = {b1:.12e}"
)

print()

if quartic_pass and linear_absent:
    print(
        "RESULT: The supplied fiber structure is "
        "consistent with a pure quartic low-k regime."
    )
    print()
    print(
        "This is structurally different from a "
        "linear light-cone + quartic correction."
    )
else:
    print(
        "RESULT: The supplied data do not establish "
        "a pure quartic regime."
    )

print()
print(
    "IMPORTANT:"
)

print(
    "This test does NOT claim that all fiber "
    "event-horizon systems are quartic."
)

print(
    "It tests only the supplied regime/data."
)

print(
    "Real experimental Delta-k/Omega data should "
    "replace the demonstration arrays before "
    "using this as an experimental result."
)

print("=" * 72)