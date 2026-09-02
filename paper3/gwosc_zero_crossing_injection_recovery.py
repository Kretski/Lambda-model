import numpy as np
import math
import matplotlib.pyplot as plt

# ================================================================
# GWOSC / PAPER 3
# ZERO-CROSSING INJECTION / RECOVERY TEST
#
# Purpose:
#   Verify that zero-crossing frequency extraction can recover
#   injected Lambda values before using real GW150914 data.
#
# IMPORTANT:
#   This is a methodological validation only.
# ================================================================

print("=" * 72)
print("ZERO-CROSSING INJECTION / RECOVERY TEST")
print("=" * 72)

# ================================================================
# PARAMETERS
# ================================================================

FS = 4096.0
DURATION = 2.0

CHIRP_MASS_MSUN = 28.716

# Test values. Dimensionless Lambda in the normalized model.
LAMBDA_TRUE_VALUES = np.array([
    0.0,
    0.01,
    0.05,
    0.10,
    0.50,
    1.00,
])

NOISE_LEVELS = [
    0.0,
    0.005,
    0.01,
]

RNG = np.random.default_rng(12345)

# ================================================================
# 1. 0PN CHIRP
# ================================================================

G = 6.67430e-11
C = 299792458.0
MSUN = 1.98847e30

MCHIRP = CHIRP_MASS_MSUN * MSUN


def chirp_frequency(t, tc=2.5):
    """
    Newtonian 0PN inspiral frequency.

    f(t) = (1 / 8pi) * (G Mc / c^3)^(-5/8)
           * (tc-t)^(-3/8)
    """

    tau = np.maximum(tc - t, 1e-4)

    return (
        1.0 / (8.0 * np.pi)
        * (G * MCHIRP / C**3) ** (-5.0 / 8.0)
        * tau ** (-3.0 / 8.0)
    )


# ================================================================
# 2. GENERATE PHASE
# ================================================================

def generate_phase(t):
    """
    Numerically integrate 2*pi*f(t).
    """

    f = chirp_frequency(t)

    phase = np.zeros_like(t)

    dt = t[1] - t[0]

    phase[1:] = np.cumsum(
        2.0 * np.pi * 0.5 * (f[1:] + f[:-1]) * dt
    )

    return phase


# ================================================================
# 3. DISPERSIVE PHASE MODEL
# ================================================================

def apply_lambda_dispersion(phase, t, lam):
    """
    Synthetic dispersion injection.

    We construct the observable instantaneous frequency as

        f_obs = f * sqrt(1 + 2 Lambda f^2)

    in normalized units.

    This is intentionally explicit: the same forward model is used
    for injection and for recovery.
    """

    dt = t[1] - t[0]

    f0 = np.gradient(phase, dt) / (2.0 * np.pi)

    f_obs = f0 * np.sqrt(
        np.maximum(1.0 + 2.0 * lam * f0**2, 1e-15)
    )

    phase_obs = np.zeros_like(t)

    phase_obs[1:] = np.cumsum(
        2.0 * np.pi * 0.5
        * (f_obs[1:] + f_obs[:-1])
        * dt
    )

    return phase_obs, f0, f_obs


# ================================================================
# 4. ZERO-CROSSING FREQUENCY EXTRACTION
# ================================================================

def zero_crossing_frequency(signal, fs):
    """
    Estimate instantaneous frequency from positive-going
    zero crossings.

    Returns:
        time_mid
        frequency
    """

    crossings = []

    for i in range(len(signal) - 1):

        y1 = signal[i]
        y2 = signal[i + 1]

        if y1 <= 0.0 and y2 > 0.0:

            if y2 != y1:
                frac = -y1 / (y2 - y1)
            else:
                frac = 0.0

            crossings.append(
                (i + frac) / fs
            )

    crossings = np.asarray(crossings)

    if len(crossings) < 3:
        return np.array([]), np.array([])

    periods = np.diff(crossings)

    frequency = 1.0 / periods

    time_mid = 0.5 * (
        crossings[:-1] + crossings[1:]
    )

    return time_mid, frequency


# ================================================================
# 5. FIT LAMBDA
# ================================================================

def fit_lambda(f_true, f_obs):

    # Avoid pathological points.
    mask = (
        np.isfinite(f_true)
        & np.isfinite(f_obs)
        & (f_true > 0)
        & (f_obs > 0)
    )

    f_true = f_true[mask]
    f_obs = f_obs[mask]

    if len(f_true) < 5:
        return np.nan, np.nan, 0

    # From:
    #
    # f_obs = f_true * sqrt(1 + 2 Lambda f_true^2)
    #
    # therefore:
    #
    # (f_obs/f_true)^2 - 1 = 2 Lambda f_true^2

    x = f_true**2

    y = (
        (f_obs / f_true)**2 - 1.0
    )

    # Fit through origin.
    denom = 2.0 * np.dot(x, x)

    if denom == 0:
        return np.nan, np.nan, 0

    lam = np.dot(x, y) / denom

    residual = y - 2.0 * lam * x

    dof = max(len(x) - 1, 1)

    sigma2 = np.sum(residual**2) / dof

    stderr = np.sqrt(
        sigma2 / (2.0 * np.dot(x, x))
    )

    return lam, stderr, len(x)


# ================================================================
# 6. SINGLE RUN
# ================================================================

def run_single(lam_true, noise_level):

    t = np.arange(
        0.0,
        DURATION,
        1.0 / FS
    )

    phase = generate_phase(t)

    phase_obs, f_true, f_disp = apply_lambda_dispersion(
        phase,
        t,
        lam_true
    )

    # Pure signal.
    signal = np.sin(phase_obs)

    # Optional additive noise.
    if noise_level > 0:
        signal = signal + RNG.normal(
            0.0,
            noise_level,
            len(signal)
        )

    tcross, f_cross = zero_crossing_frequency(
        signal,
        FS
    )

    if len(tcross) < 5:
        return np.nan, np.nan, 0

    # Interpolate theoretical nondispersive frequency
    # at the zero-crossing times.
    f_ref = np.interp(
        tcross,
        t,
        f_true
    )

    # Remove edge regions where interpolation and chirp
    # initialization can become problematic.
    valid = (
        (tcross > 0.1)
        & (tcross < DURATION - 0.1)
        & (f_ref > 0)
        & (f_cross > 0)
    )

    f_ref = f_ref[valid]
    f_cross = f_cross[valid]

    return fit_lambda(
        f_ref,
        f_cross
    )


# ================================================================
# 7. MAIN INJECTION GRID
# ================================================================

results = []

print()
print("[1] ZERO-CROSSING INJECTION GRID")
print("-" * 72)

for noise in NOISE_LEVELS:

    print()
    print(f"Noise level = {noise}")

    print(
        f"{'Lambda_true':>14}"
        f"{'Lambda_fit':>16}"
        f"{'Lambda_err':>16}"
        f"{'sigma':>14}"
        f"{'n':>8}"
    )

    print("-" * 72)

    for lam_true in LAMBDA_TRUE_VALUES:

        lam_fit, lam_err, n = run_single(
            lam_true,
            noise
        )

        if np.isfinite(lam_fit) and np.isfinite(lam_err):

            sigma = abs(
                lam_fit - lam_true
            ) / max(lam_err, 1e-30)

        else:
            sigma = np.nan

        results.append([
            noise,
            lam_true,
            lam_fit,
            lam_err,
            sigma,
            n
        ])

        print(
            f"{lam_true:14.4f}"
            f"{lam_fit:16.6f}"
            f"{lam_err:16.6f}"
            f"{sigma:14.2f}"
            f"{n:8d}"
        )


results = np.asarray(results)


# ================================================================
# 8. RECOVERY DIAGNOSTICS
# ================================================================

print()
print("=" * 72)
print("RECOVERY DIAGNOSTICS")
print("=" * 72)

for noise in NOISE_LEVELS:

    mask = results[:, 0] == noise

    true = results[mask, 1]
    fit = results[mask, 2]

    finite = np.isfinite(fit)

    true = true[finite]
    fit = fit[finite]

    if len(true) == 0:
        continue

    bias = fit - true

    rmse = np.sqrt(
        np.mean(bias**2)
    )

    max_abs = np.max(
        np.abs(bias)
    )

    print()
    print(f"Noise = {noise}")
    print(f"RMSE bias = {rmse:.6e}")
    print(f"Max abs bias = {max_abs:.6e}")


# ================================================================
# 9. PASS/FAIL CRITERIA
# ================================================================

print()
print("=" * 72)
print("VALIDATION CRITERIA")
print("=" * 72)

# Lambda=0 must be statistically compatible with zero.
zero_rows = results[
    (results[:, 0] == 0.0)
    & (results[:, 1] == 0.0)
]

zero_ok = True

for row in zero_rows:

    lam_fit = row[2]
    lam_err = row[3]

    if not np.isfinite(lam_fit):
        zero_ok = False
        continue

    sigma0 = abs(lam_fit) / max(
        lam_err,
        1e-30
    )

    print(
        f"Noise={row[0]:.3f}: "
        f"Lambda_fit={lam_fit:.6e}, "
        f"sigma={sigma0:.3f}"
    )

    if sigma0 > 3.0:
        zero_ok = False


# Nonzero injections should recover within 20%.
nonzero_ok = True

for row in results:

    noise = row[0]
    true = row[1]
    fit = row[2]

    if true <= 0:
        continue

    if not np.isfinite(fit):
        nonzero_ok = False
        continue

    rel_error = abs(
        fit - true
    ) / true

    if rel_error > 0.20:
        nonzero_ok = False


print()
print(
    "[PASS]" if zero_ok
    else "[FAIL]",
    "Lambda=0 recovery"
)

print(
    "[PASS]" if nonzero_ok
    else "[FAIL]",
    "Nonzero Lambda recovery (<20% relative error)"
)


# ================================================================
# 10. SAVE CSV
# ================================================================

out = "gwosc_results/zero_crossing_injection_recovery.csv"

import os
os.makedirs(
    "gwosc_results",
    exist_ok=True
)

np.savetxt(
    out,
    results,
    delimiter=",",
    header=(
        "noise,lambda_true,lambda_fit,"
        "lambda_err,sigma,n_points"
    ),
    comments=""
)

print()
print(f"CSV saved -> {out}")


# ================================================================
# 11. PLOT
# ================================================================

plt.figure(figsize=(9, 6))

for noise in NOISE_LEVELS:

    mask = results[:, 0] == noise

    true = results[mask, 1]
    fit = results[mask, 2]

    plt.plot(
        true,
        fit,
        "o-",
        label=f"noise={noise}"
    )

plt.plot(
    LAMBDA_TRUE_VALUES,
    LAMBDA_TRUE_VALUES,
    "--",
    label="ideal recovery"
)

plt.xlabel("Injected Lambda")
plt.ylabel("Recovered Lambda")
plt.title(
    "Zero-Crossing Lambda Injection / Recovery"
)

plt.grid(True, alpha=0.3)
plt.legend()

plot_file = (
    "gwosc_results/"
    "zero_crossing_injection_recovery.png"
)

plt.tight_layout()
plt.savefig(
    plot_file,
    dpi=180
)

print(f"Plot saved -> {plot_file}")


# ================================================================
# FINAL
# ================================================================

print()
print("=" * 72)
print("FINAL RESULT")
print("=" * 72)

if zero_ok and nonzero_ok:

    print()
    print(
        "PASS: zero-crossing extraction passes the "
        "synthetic injection/recovery test."
    )

    print()
    print(
        "The pipeline can now proceed to a controlled "
        "real-data test."
    )

else:

    print()
    print(
        "FAIL: zero-crossing extraction is not yet "
        "validated sufficiently."
    )

    print()
    print(
        "Do NOT interpret a real GW150914 Lambda fit "
        "as physical evidence yet."
    )

print("=" * 72)