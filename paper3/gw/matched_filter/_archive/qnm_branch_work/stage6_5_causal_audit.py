#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 6.5
CAUSAL / INDEPENDENT PHYSICAL CONSEQUENCE AUDIT
Lambda / Root-B resonance branch

PURPOSE
-------
Do NOT:
    - fit Kerr
    - fit GW150914
    - choose a frequency scale from GW data
    - identify m with Kerr m
    - identify C with Kerr a/M

Instead:

    1. Establish the baseline Root-B mechanism.
    2. Remove one physical ingredient at a time.
    3. Measure whether the resonant structure changes.
    4. Check controlled asymptotic regimes.
    5. Check numerical robustness.
    6. Produce candidate independent observables.

IMPORTANT
---------
This stage does NOT claim "new physics".

It asks:

    Does the model contain a causal, parameter-dependent
    resonant prediction that survives controlled tests?

The actual complex resonance solver is imported from
stage5I_3_resonance.py.

If that file is unavailable, the script stops rather than
silently replacing the validated solver with a different model.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from importlib import import_module

import numpy as np


# ============================================================
# PATH
# ============================================================

HERE = Path(__file__).resolve().parent

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


# ============================================================
# IMPORT VALIDATED ROOT-B MACHINERY
# ============================================================

try:
    import stage5I_3_resonance as RB
except Exception as exc:
    print()
    print("=" * 78)
    print("ERROR: could not import stage5I_3_resonance.py")
    print("=" * 78)
    print(exc)
    print()
    print("Put this script in the same directory as:")
    print("    stage5I_3_resonance.py")
    print()
    raise


# ============================================================
# OUTPUT
# ============================================================

OUTDIR = HERE / "stage6_5_causal_audit"
OUTDIR.mkdir(exist_ok=True)

CSV_OUT = OUTDIR / "stage6_5_causal_results.csv"
JSON_OUT = OUTDIR / "stage6_5_causal_results.json"


# ============================================================
# BASELINE PARAMETERS
# ============================================================

# Read directly from validated Root-B module whenever available.

BASE = {
    "C": float(getattr(RB, "C_METHODS")),
    "Omega": float(getattr(RB, "OMEGA")),
    "gamma": float(getattr(RB, "GAMMA")),
    "h0": float(getattr(RB, "H0")),
    "g": float(getattr(RB, "G_GRAV")),
    "r_B": float(getattr(RB, "R_B")),
}

M_DEFAULT = -12


# ============================================================
# BASIC DISPERSION
# ============================================================

def F_dispersion(k, gamma=None, h0=None, g=None):
    """
    F(k) = (g k + gamma k^3) tanh(h0 k)
    """

    if gamma is None:
        gamma = BASE["gamma"]

    if h0 is None:
        h0 = BASE["h0"]

    if g is None:
        g = BASE["g"]

    k = np.asarray(k, dtype=float)

    return (
        (g * k + gamma * k**3)
        * np.tanh(h0 * k)
    )


# ============================================================
# ROOT-B DOPPLER TERM
# ============================================================

def D_rootB(m, r, C, Omega):
    return (
        m * Omega
        + m * C / r**2
    )


# ============================================================
# k AT p=0
# ============================================================

def k_p0(m, r):
    return abs(m) / r


# ============================================================
# ROOT-B p=0 DISPERSION
# ============================================================

def omega_p0(
    m,
    r,
    C,
    Omega,
    gamma,
    h0,
    g,
):
    """
    omega_D^+(p=0)
    """

    k = k_p0(m, r)

    F = float(
        F_dispersion(
            k,
            gamma=gamma,
            h0=h0,
            g=g,
        )
    )

    if F < 0:
        return np.nan

    return (
        D_rootB(m, r, C, Omega)
        + np.sqrt(F)
    )


# ============================================================
# LIGHT RING
# ============================================================

def find_light_ring(
    m,
    C,
    Omega,
    gamma,
    h0,
    g,
    r_min=1.0e-3,
    r_max=None,
    n=20000,
):
    """
    Independent reimplementation of the p=0 light-ring search.

    This is used only as a diagnostic cross-check.

    We deliberately do not call the Root-B solver here so that
    this stage can detect obvious parameter-response changes
    independently.
    """

    if r_max is None:
        r_max = BASE["r_B"]

    r = np.linspace(
        r_min,
        r_max,
        n,
    )

    omega = np.array([
        omega_p0(
            m,
            rr,
            C,
            Omega,
            gamma,
            h0,
            g,
        )
        for rr in r
    ])

    finite = np.isfinite(omega)

    if not np.any(finite):
        return None, None

    idx = np.nanargmax(omega)

    if idx == 0 or idx == len(r) - 1:
        return None, None

    # Local quadratic interpolation around maximum.
    i0 = max(1, min(idx, len(r) - 2))

    x = r[i0 - 1:i0 + 2]
    y = omega[i0 - 1:i0 + 2]

    try:
        coeff = np.polyfit(x, y, 2)

        a, b, c = coeff

        if abs(a) > 0:
            r_vertex = -b / (2.0 * a)
        else:
            r_vertex = r[i0]

        if (
            r_vertex < x[0]
            or r_vertex > x[-1]
        ):
            r_vertex = r[i0]

    except Exception:
        r_vertex = r[i0]

    omega_vertex = omega_p0(
        m,
        r_vertex,
        C,
        Omega,
        gamma,
        h0,
        g,
    )

    return (
        float(r_vertex),
        float(omega_vertex),
    )


# ============================================================
# REGIME AUDIT
# ============================================================

def regime_metrics(m, r, gamma, h0, g):
    """
    Quantities controlling the validity of the long-wave /
    gravity-dominated approximation.

        epsilon_h = h0*k

        epsilon_cap = gamma*k^2/g

    Long-wave requires epsilon_h << 1.

    Gravity dominated requires epsilon_cap << 1.
    """

    k = abs(m) / r

    eps_h = h0 * k

    eps_cap = gamma * k**2 / g

    return {
        "k": float(k),
        "h0_k": float(eps_h),
        "capillary_ratio": float(eps_cap),
        "long_wave_controlled": bool(eps_h < 0.1),
        "gravity_controlled": bool(eps_cap < 0.1),
    }


# ============================================================
# CAUSAL PARAMETER VARIATIONS
# ============================================================

CASES = [
    {
        "name": "BASELINE",
        "C_factor": 1.0,
        "Omega_factor": 1.0,
        "gamma_factor": 1.0,
    },
    {
        "name": "C_ZERO",
        "C_factor": 0.0,
        "Omega_factor": 1.0,
        "gamma_factor": 1.0,
    },
    {
        "name": "OMEGA_ZERO",
        "C_factor": 1.0,
        "Omega_factor": 0.0,
        "gamma_factor": 1.0,
    },
    {
        "name": "GAMMA_ZERO",
        "C_factor": 1.0,
        "Omega_factor": 1.0,
        "gamma_factor": 0.0,
    },
    {
        "name": "C_HALF",
        "C_factor": 0.5,
        "Omega_factor": 1.0,
        "gamma_factor": 1.0,
    },
    {
        "name": "C_DOUBLE",
        "C_factor": 2.0,
        "Omega_factor": 1.0,
        "gamma_factor": 1.0,
    },
    {
        "name": "OMEGA_HALF",
        "C_factor": 1.0,
        "Omega_factor": 0.5,
        "gamma_factor": 1.0,
    },
    {
        "name": "OMEGA_DOUBLE",
        "C_factor": 1.0,
        "Omega_factor": 2.0,
        "gamma_factor": 1.0,
    },
    {
        "name": "GAMMA_HALF",
        "C_factor": 1.0,
        "Omega_factor": 1.0,
        "gamma_factor": 0.5,
    },
    {
        "name": "GAMMA_DOUBLE",
        "C_factor": 1.0,
        "Omega_factor": 1.0,
        "gamma_factor": 2.0,
    },
]


# ============================================================
# RUN PARAMETER AUDIT
# ============================================================

def run_parameter_audit(m=M_DEFAULT):

    print()
    print("=" * 78)
    print("STAGE 6.5 — CAUSAL / INDEPENDENT PHYSICAL CONSEQUENCE AUDIT")
    print("=" * 78)
    print()

    print("Baseline Root-B parameters:")
    for key, value in BASE.items():
        print(f"  {key:>8} = {value:.12e}")

    print()
    print(f"Mode m = {m}")
    print()

    rows = []

    for case in CASES:

        C = BASE["C"] * case["C_factor"]
        Omega = BASE["Omega"] * case["Omega_factor"]
        gamma = BASE["gamma"] * case["gamma_factor"]

        r_sp, omega_lr = find_light_ring(
            m=m,
            C=C,
            Omega=Omega,
            gamma=gamma,
            h0=BASE["h0"],
            g=BASE["g"],
        )

        if r_sp is None:
            print(
                f"{case['name']:>15}: "
                "NO INTERIOR LIGHT RING"
            )

            rows.append({
                "case": case["name"],
                "C": C,
                "Omega": Omega,
                "gamma": gamma,
                "r_sp_m": np.nan,
                "f_lr_hz": np.nan,
                "k_m_inv": np.nan,
                "h0_k": np.nan,
                "capillary_ratio": np.nan,
                "long_wave_controlled": False,
                "gravity_controlled": False,
                "status": "NO_LIGHT_RING",
            })

            continue

        f_lr = omega_lr / (2.0 * np.pi)

        regime = regime_metrics(
            m=m,
            r=r_sp,
            gamma=gamma,
            h0=BASE["h0"],
            g=BASE["g"],
        )

        print(
            f"{case['name']:>15}: "
            f"r_sp={r_sp*1e3:10.5f} mm   "
            f"f_lr={f_lr:12.7f} Hz   "
            f"h0*k={regime['h0_k']:.4e}   "
            f"cap={regime['capillary_ratio']:.4e}"
        )

        rows.append({
            "case": case["name"],
            "C": C,
            "Omega": Omega,
            "gamma": gamma,
            "r_sp_m": r_sp,
            "f_lr_hz": f_lr,
            "k_m_inv": regime["k"],
            "h0_k": regime["h0_k"],
            "capillary_ratio": regime["capillary_ratio"],
            "long_wave_controlled": regime[
                "long_wave_controlled"
            ],
            "gravity_controlled": regime[
                "gravity_controlled"
            ],
            "status": "OK",
        })

    return rows


# ============================================================
# LOCAL SENSITIVITY
# ============================================================

def finite_difference_sensitivity(
    m,
    parameter,
    relative_step=1e-3,
):
    """
    Derivative of the light-ring frequency with respect to
    one model parameter.

    No fitting is performed.

        d f_lr / d parameter

    is estimated locally around the baseline.
    """

    params0 = BASE.copy()

    p0 = params0[parameter]

    if p0 == 0:
        raise ValueError(
            f"Cannot use relative derivative for zero parameter: "
            f"{parameter}"
        )

    p_plus = p0 * (1.0 + relative_step)
    p_minus = p0 * (1.0 - relative_step)

    def get_f(value):

        params = params0.copy()
        params[parameter] = value

        r_sp, omega = find_light_ring(
            m=m,
            C=params["C"],
            Omega=params["Omega"],
            gamma=params["gamma"],
            h0=params["h0"],
            g=params["g"],
        )

        if r_sp is None:
            return np.nan

        return omega / (2.0 * np.pi)

    f_plus = get_f(p_plus)
    f_minus = get_f(p_minus)

    derivative = (
        f_plus - f_minus
    ) / (
        p_plus - p_minus
    )

    return {
        "parameter": parameter,
        "f_minus": f_minus,
        "f_plus": f_plus,
        "df_dp": derivative,
        "relative_step": relative_step,
    }


# ============================================================
# NUMERICAL CONVERGENCE
# ============================================================

def convergence_test(m=M_DEFAULT):

    print()
    print("-" * 78)
    print("LIGHT-RING NUMERICAL CONVERGENCE")
    print("-" * 78)

    results = []

    for n in [5000, 10000, 20000, 40000, 80000]:

        r_sp, omega = find_light_ring(
            m=m,
            C=BASE["C"],
            Omega=BASE["Omega"],
            gamma=BASE["gamma"],
            h0=BASE["h0"],
            g=BASE["g"],
            n=n,
        )

        if r_sp is None:
            raise RuntimeError(
                f"Light ring disappeared at n={n}"
            )

        f = omega / (2.0 * np.pi)

        results.append({
            "n": n,
            "r_sp_m": r_sp,
            "f_lr_hz": f,
        })

        print(
            f"  n={n:6d}   "
            f"r_sp={r_sp*1e3:.9f} mm   "
            f"f={f:.10f} Hz"
        )

    ref = results[-1]

    print()
    print("Differences relative to n=80000:")

    for row in results[:-1]:
        dr = abs(
            row["r_sp_m"]
            - ref["r_sp_m"]
        )

        df = abs(
            row["f_lr_hz"]
            - ref["f_lr_hz"]
        )

        print(
            f"  n={row['n']:6d}: "
            f"Delta r={dr:.3e} m   "
            f"Delta f={df:.3e} Hz"
        )

    return results


# ============================================================
# LONG-WAVE VALIDITY MAP
# ============================================================

def validity_map(m=M_DEFAULT):

    print()
    print("-" * 78)
    print("CONTROLLED-REGIME AUDIT")
    print("-" * 78)

    r_sp, omega = find_light_ring(
        m=m,
        C=BASE["C"],
        Omega=BASE["Omega"],
        gamma=BASE["gamma"],
        h0=BASE["h0"],
        g=BASE["g"],
    )

    if r_sp is None:
        return None

    print(
        f"Baseline light ring: "
        f"r_sp={r_sp*1e3:.6f} mm"
    )

    print()

    regime = regime_metrics(
        m,
        r_sp,
        BASE["gamma"],
        BASE["h0"],
        BASE["g"],
    )

    print(
        f"k               = "
        f"{regime['k']:.8e} 1/m"
    )

    print(
        f"h0*k            = "
        f"{regime['h0_k']:.8e}"
    )

    print(
        f"gamma*k^2/g     = "
        f"{regime['capillary_ratio']:.8e}"
    )

    print()

    if regime["long_wave_controlled"]:
        print("LONG-WAVE CONTROL: PASS")
    else:
        print(
            "LONG-WAVE CONTROL: FAIL "
            "(do not use reduced long-wave equation here)"
        )

    if regime["gravity_controlled"]:
        print("GRAVITY-DOMINATED CONTROL: PASS")
    else:
        print(
            "GRAVITY-DOMINATED CONTROL: FAIL"
        )

    return regime


# ============================================================
# WRITE CSV
# ============================================================

def write_csv(rows):

    if not rows:
        return

    keys = sorted({
        key
        for row in rows
        for key in row.keys()
    })

    with CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=keys,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# ============================================================
# MAIN
# ============================================================

def main():

    parameter_rows = run_parameter_audit()

    print()
    print("=" * 78)
    print("LOCAL PARAMETER SENSITIVITY")
    print("=" * 78)

    sensitivities = []

    for parameter in [
        "C",
        "Omega",
        "gamma",
    ]:

        result = finite_difference_sensitivity(
            M_DEFAULT,
            parameter,
        )

        sensitivities.append(result)

        print(
            f"  {parameter:>6}: "
            f"df/dp={result['df_dp']:.8e}"
        )

    convergence = convergence_test()

    validity = validity_map()

    # --------------------------------------------------------
    # Causal interpretation
    # --------------------------------------------------------

    baseline = next(
        row
        for row in parameter_rows
        if row["case"] == "BASELINE"
    )

    print()
    print("=" * 78)
    print("CAUSAL AUDIT VERDICT")
    print("=" * 78)
    print()

    c_zero = next(
        row
        for row in parameter_rows
        if row["case"] == "C_ZERO"
    )

    omega_zero = next(
        row
        for row in parameter_rows
        if row["case"] == "OMEGA_ZERO"
    )

    gamma_zero = next(
        row
        for row in parameter_rows
        if row["case"] == "GAMMA_ZERO"
    )

    print(
        "Baseline light ring:       ",
        baseline["status"],
    )

    print(
        "C -> 0 response:           ",
        c_zero["status"],
    )

    print(
        "Omega -> 0 response:      ",
        omega_zero["status"],
    )

    print(
        "gamma -> 0 response:      ",
        gamma_zero["status"],
    )

    print()

    for s in sensitivities:

        if np.isfinite(s["df_dp"]):
            if abs(s["df_dp"]) > 0:
                flag = "NONZERO RESPONSE"
            else:
                flag = "ZERO RESPONSE"
        else:
            flag = "UNDEFINED"

        print(
            f"{s['parameter']:>8}: {flag}"
        )

    print()

    if validity is not None:

        print(
            "Controlled long-wave regime at "
            "light ring:",
            validity["long_wave_controlled"],
        )

        print(
            "Controlled gravity regime at "
            "light ring:",
            validity["gravity_controlled"],
        )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "This stage establishes parameter dependence and "
        "numerical robustness."
    )
    print(
        "It does NOT establish a new fundamental interaction."
    )
    print(
        "The next step is an independent observable prediction."
    )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    payload = {
        "stage": "6.5",
        "title": (
            "Causal / Independent Physical "
            "Consequence Audit"
        ),
        "baseline": BASE,
        "mode": M_DEFAULT,
        "parameter_audit": parameter_rows,
        "sensitivities": sensitivities,
        "convergence": convergence,
        "validity": validity,
        "scientific_status": (
            "PARAMETER_DEPENDENCE_AUDIT_ONLY"
        ),
    }

    with JSON_OUT.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
            default=lambda x: (
                None
                if isinstance(x, float)
                and not np.isfinite(x)
                else x
            ),
        )

    write_csv(parameter_rows)

    print()
    print(
        f"CSV  -> {CSV_OUT}"
    )
    print(
        f"JSON -> {JSON_OUT}"
    )
    print()


if __name__ == "__main__":
    main()