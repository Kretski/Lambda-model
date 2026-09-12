#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage5I_4_v5_light_test.py

LIGHT ADAPTIVE POLE TEST – Stage 5I-4 v5

This is a light version of the pole search algorithm.
It uses reduced resolution and adaptive search to find complex poles
of the resonance factor.

Author: Dimitar Kretski
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, root

# ============================================================================
# TRY TO IMPORT STAGE 5I-3 v2
# ============================================================================

HERE = Path(__file__).resolve().parent

# Try multiple possible locations
possible_paths = [
    HERE / "stage5I_3_resonance_v2.py",
    HERE.parent / "stage5I_3_resonance_v2.py",
    HERE / ".." / "stage5I_3_resonance_v2.py",
]

stage5_module = None
stage5_path = None

for path in possible_paths:
    if path.exists():
        stage5_path = path
        break

if stage5_path is None:
    print("=" * 78)
    print("ERROR: stage5I_3_resonance_v2.py not found")
    print("=" * 78)
    print()
    print("Please ensure the following file is in the same directory:")
    print("  stage5I_3_resonance_v2.py")
    print()
    print("Or provide the correct path in the 'possible_paths' list.")
    sys.exit(1)

# Add the directory to sys.path
if str(stage5_path.parent) not in sys.path:
    sys.path.insert(0, str(stage5_path.parent))

try:
    from stage5I_3_resonance_v2 import (
        find_light_ring,
        find_last_scattering_point,
        resonance_factor,
    )
    print(f"✅ Imported stage5I_3_resonance_v2 from: {stage5_path}")
except ImportError as e:
    print("=" * 78)
    print("ERROR: Failed to import stage5I_3_resonance_v2")
    print("=" * 78)
    print()
    print(f"Error: {e}")
    print()
    print("Please ensure the file exists and contains the required functions:")
    print("  - find_light_ring")
    print("  - find_last_scattering_point")
    print("  - resonance_factor")
    sys.exit(1)

# ============================================================================
# LIGHT TEST SETTINGS
# ============================================================================

M = -12

F_MIN = 7.0
F_MAX = 10.5

# Much lighter than the original 5000-point search.
N_COARSE = 350

# Reduced radial work during the real scan.
# We temporarily override the radial integration resolution used by
# stage5I_3_v2 through the module variable.
RADIAL_N_COARSE = 64

# Reduced turning-point scan for the coarse phase.
TURNING_SCAN_COARSE = 400

# Only genuinely promising real-axis minima enter complex refinement.
MAX_REAL_RES_FOR_COMPLEX = 0.10

# Primary seed.
PRIMARY_IM_SEED = -5.0e-4

# Adaptive fallback seeds.
FALLBACK_IM_SEEDS = [
    -1.0e-3,
    -1.0e-2,
    -1.0e-1,
]

# Complex refinement window.
RE_HALF_WIDTH = 0.08

IM_MIN = -0.20
IM_MAX = 0.0

# Root acceptance.
ROOT_RES_TOL = 1.0e-8

# Expected regression value.
EXPECTED_RE = 8.3066307169
EXPECTED_IM = -0.0005664350

REGRESSION_RE_TOL = 1.0e-7
REGRESSION_IM_TOL = 1.0e-7

# ============================================================================
# HELPERS
# ============================================================================

def configure_light_mode():
    """
    Reduce the expensive radial integration resolution in the imported
    physics module.

    This changes numerical resolution only.
    It does NOT change the Hamiltonian, dispersion relation, reflection
    coefficient, resonance equation, or complex continuation physics.
    """

    try:
        import stage5I_3_resonance_v2 as physics
        physics.DEFAULT_RADIAL_N = RADIAL_N_COARSE
        return physics
    except ImportError:
        print("WARNING: Could not configure physics module")
        return None


def evaluate_real_res(
    f,
    m,
    r_sp,
    omega_lr,
):
    """
    Evaluate real-axis |Res|.
    """

    try:

        z = resonance_factor(
            omega=(
                2.0
                * np.pi
                * f
                + 0.0j
            ),
            m=m,
            r_sp=r_sp,
            omega_lr=omega_lr,
            complex_action=False,
        )

        if z is None:
            return np.inf

        value = abs(z)

        if not np.isfinite(value):
            return np.inf

        return float(value)

    except Exception as e:
        return np.inf


# ============================================================================
# COARSE REAL SCAN
# ============================================================================

def coarse_real_scan(
    m,
    f_min,
    f_max,
    r_sp,
    omega_lr,
):
    """
    Lightweight real-axis scan.

    No complex calculation is performed here.
    """

    print()
    print(
        f"[m={m}] coarse scan: "
        f"{N_COARSE} points | "
        f"radial_n={RADIAL_N_COARSE} | "
        f"turning_scan={TURNING_SCAN_COARSE}"
    )

    # Temporarily reduce turning-point search resolution.
    try:
        import stage5I_3_resonance_v2 as physics

        original_turning_scan = physics.find_turning_points

        def light_find_turning_points(
            omega,
            m,
            r_min=5.0e-3,
            r_max=physics.R_B,
            n_scan=TURNING_SCAN_COARSE,
        ):
            return original_turning_scan(
                omega,
                m,
                r_min=r_min,
                r_max=r_max,
                n_scan=TURNING_SCAN_COARSE,
            )

        physics.find_turning_points = light_find_turning_points

        f_grid = np.linspace(
            f_min,
            f_max,
            N_COARSE,
        )

        values = np.full(
            N_COARSE,
            np.inf,
            dtype=float,
        )

        progress_step = max(
            1,
            N_COARSE // 10,
        )

        for i, f in enumerate(f_grid):

            values[i] = evaluate_real_res(
                f,
                m,
                r_sp,
                omega_lr,
            )

            if (
                (i + 1) % progress_step == 0
                or i == N_COARSE - 1
            ):

                pct = (
                    100.0
                    * (i + 1)
                    / N_COARSE
                )

                print(
                    f"    {i + 1:4d}/{N_COARSE} "
                    f"({pct:5.1f}%)"
                )

        return f_grid, values

    except Exception as e:
        print(f"ERROR in coarse_real_scan: {e}")
        return None, None

    finally:
        try:
            import stage5I_3_resonance_v2 as physics
            physics.find_turning_points = original_turning_scan
        except:
            pass


# ============================================================================
# LOCAL MINIMA
# ============================================================================

def find_local_candidates(
    f_grid,
    values,
):
    """
    Find local minima of |Res|.
    """

    if f_grid is None or values is None:
        return []

    candidates = []

    for i in range(
        1,
        len(f_grid) - 1,
    ):

        if not np.isfinite(values[i]):
            continue

        if (
            values[i]
            <= values[i - 1]
            and
            values[i]
            <= values[i + 1]
        ):

            candidates.append(i)

    candidates.sort(
        key=lambda i: values[i]
    )

    return candidates


# ============================================================================
# REAL-AXIS REFINEMENT
# ============================================================================

def refine_real_candidate(
    f0,
    df,
    m,
    r_sp,
    omega_lr,
):
    """
    Higher-accuracy real-axis minimization.
    """

    lo = max(
        F_MIN,
        f0 - max(0.08, 3.0 * df),
    )

    hi = min(
        F_MAX,
        f0 + max(0.08, 3.0 * df),
    )

    def objective(f):

        return evaluate_real_res(
            f,
            m,
            r_sp,
            omega_lr,
        )

    try:
        result = minimize(
            objective,
            x0=np.array([f0]),
            method="Nelder-Mead",
            options={
                "xatol": 1.0e-10,
                "fatol": 1.0e-12,
                "maxiter": 200,
            },
        )

        f = float(
            result.x[0]
        )

        # Keep result inside the local search interval.
        f = min(
            max(f, lo),
            hi,
        )

        value = evaluate_real_res(
            f,
            m,
            r_sp,
            omega_lr,
        )

        return {
            "f_real": f,
            "abs_res": value,
            "success": bool(result.success),
        }
    except Exception:
        return {
            "f_real": f0,
            "abs_res": np.inf,
            "success": False,
        }


# ============================================================================
# COMPLEX OBJECTIVE
# ============================================================================

def complex_objective(
    x,
    f0,
    m,
):
    """
    Objective for complex Nelder-Mead refinement.
    """

    f_re = float(x[0])
    f_im = float(x[1])

    if not (
        f0 - 4.0 * RE_HALF_WIDTH
        <= f_re
        <= f0 + 4.0 * RE_HALF_WIDTH
    ):
        return 1.0e50

    if not (
        IM_MIN
        <= f_im
        <= IM_MAX
    ):
        return 1.0e50

    omega = (
        2.0
        * np.pi
        * (
            f_re
            + 1j * f_im
        )
    )

    try:

        z = resonance_factor(
            omega=omega,
            m=m,
            complex_action=True,
        )

        if z is None:
            return 1.0e50

        value = abs(z)

        if not np.isfinite(value):
            return 1.0e50

        return float(
            np.log1p(value)
        )

    except Exception:

        return 1.0e50


# ============================================================================
# NELDER-MEAD
# ============================================================================

def complex_nm(
    f0,
    im_seed,
    m,
):
    """
    One complex Nelder-Mead attempt.
    """

    try:
        result = minimize(
            complex_objective,
            x0=np.array([
                f0,
                im_seed,
            ]),
            args=(
                f0,
                m,
            ),
            method="Nelder-Mead",
            options={
                "xatol": 1.0e-9,
                "fatol": 1.0e-10,
                "maxiter": 700,
            },
        )

        if not np.isfinite(
            result.fun
        ):
            return None

        f_re = float(
            result.x[0]
        )

        f_im = float(
            result.x[1]
        )

        if not (
            f0 - 4.0 * RE_HALF_WIDTH
            <= f_re
            <= f0 + 4.0 * RE_HALF_WIDTH
        ):
            return None

        if not (
            IM_MIN
            <= f_im
            <= IM_MAX
        ):
            return None

        residual = (
            np.expm1(
                result.fun
            )
        )

        if not np.isfinite(residual):
            return None

        return {
            "f_re": f_re,
            "f_im": f_im,
            "abs_res": float(residual),
            "nm_success": bool(result.success),
        }
    except Exception:
        return None


# ============================================================================
# ROOT POLISH
# ============================================================================

def root_polish(
    candidate,
    m,
):
    """
    Solve Re(Res)=0 and Im(Res)=0 directly.
    """

    def equations(x):

        f_re = float(x[0])
        f_im = float(x[1])

        omega = (
            2.0
            * np.pi
            * (
                f_re
                + 1j * f_im
            )
        )

        try:

            z = resonance_factor(
                omega=omega,
                m=m,
                complex_action=True,
            )

            if z is None:

                return np.array([
                    1.0e6,
                    1.0e6,
                ])

            return np.array([
                z.real,
                z.imag,
            ])

        except Exception:

            return np.array([
                1.0e6,
                1.0e6,
            ])

    try:
        result = root(
            equations,
            np.array([
                candidate["f_re"],
                candidate["f_im"],
            ]),
            method="hybr",
            options={
                "xtol": 1.0e-10,
                "maxfev": 300,
            },
        )

        f_re = float(
            result.x[0]
        )

        f_im = float(
            result.x[1]
        )

        omega = (
            2.0
            * np.pi
            * (
                f_re
                + 1j * f_im
            )
        )

        try:

            z = resonance_factor(
                omega=omega,
                m=m,
                complex_action=True,
            )

        except Exception:

            z = None

        if z is None:

            return {
                "f_re": f_re,
                "f_im": f_im,
                "abs_res": np.inf,
                "success": False,
            }

        abs_res = abs(z)

        success = (
            bool(result.success)
            and np.isfinite(abs_res)
            and abs_res < ROOT_RES_TOL
        )

        return {
            "f_re": f_re,
            "f_im": f_im,
            "omega_re": 2.0 * np.pi * f_re,
            "omega_im": 2.0 * np.pi * f_im,
            "abs_res": float(abs_res),
            "success": success,
        }
    except Exception:
        return None


# ============================================================================
# ADAPTIVE COMPLEX SEARCH
# ============================================================================

def adaptive_complex_search(
    f0,
    m,
):
    """
    Fast path:
        one primary Im seed.

    Fallback:
        only if the primary seed does not produce a valid pole.
    """

    seeds = [
        PRIMARY_IM_SEED,
        *FALLBACK_IM_SEEDS,
    ]

    for index, seed in enumerate(seeds):

        if index == 0:

            print(
                f"        primary Im seed = "
                f"{seed:.6e} Hz"
            )

        else:

            print(
                f"        fallback Im seed = "
                f"{seed:.6e} Hz"
            )

        nm = complex_nm(
            f0,
            seed,
            m,
        )

        if nm is None:

            print(
                "            NM -> unavailable"
            )

            continue

        print(
            f"            NM -> "
            f"Re={nm['f_re']:.10f} "
            f"Im={nm['f_im']:.10f} "
            f"|Res|={nm['abs_res']:.4e}"
        )

        polished = root_polish(
            nm,
            m,
        )

        if polished is None:
            print("            ROOT -> failed")
            continue

        print(
            f"            ROOT -> "
            f"Re={polished['f_re']:.10f} "
            f"Im={polished['f_im']:.10f} "
            f"|Res|={polished['abs_res']:.4e} "
            f"success={polished['success']}"
        )

        if polished["success"]:

            return polished

    return None


# ============================================================================
# MAIN TEST
# ============================================================================

def main():

    print("=" * 78)
    print(
        "STAGE 5I-4 v5 — LIGHT ADAPTIVE POLE TEST"
    )
    print("=" * 78)

    print()
    print(
        "Physics source: "
        "stage5I_3_resonance_v2.py"
    )

    print(
        "No experimental frequencies are used as seeds."
    )

    print()
    print(
        "Strategy:"
    )
    print(
        "    coarse real scan"
    )
    print(
        "      -> local minima"
    )
    print(
        "      -> real refinement"
    )
    print(
        "      -> |Res| gate"
    )
    print(
        "      -> primary Im seed"
    )
    print(
        "      -> adaptive fallback only if needed"
    )
    print(
        "      -> root polish"
    )

    print()
    print("=" * 78)
    print(
        f"STAGE 5I-4 v5   m={M}"
    )
    print("=" * 78)

    physics = configure_light_mode()

    # ------------------------------------------------------------------
    # Light ring
    # ------------------------------------------------------------------

    try:
        r_sp, omega_lr = find_light_ring(M)
    except Exception as e:
        print(f"ERROR: find_light_ring failed: {e}")
        return

    if r_sp is None:

        print(
            "ERROR: no light ring found."
        )

        return

    f_lr = (
        omega_lr
        / (
            2.0
            * np.pi
        )
    )

    print(
        f"light ring: "
        f"r_sp={r_sp * 1.0e3:.9f} mm "
        f"f_lr={f_lr:.12f} Hz"
    )

    # ------------------------------------------------------------------
    # Coarse scan
    # ------------------------------------------------------------------

    f_grid, values = coarse_real_scan(
        M,
        F_MIN,
        F_MAX,
        r_sp,
        omega_lr,
    )

    if f_grid is None or values is None:
        print("ERROR: coarse scan failed")
        return

    finite = np.isfinite(
        values
    )

    if not np.any(finite):

        print()
        print(
            "ERROR: no finite real-axis samples."
        )

        return

    # ------------------------------------------------------------------
    # Candidates
    # ------------------------------------------------------------------

    indices = find_local_candidates(
        f_grid,
        values,
    )

    print()
    print(
        "Coarse candidates:"
    )

    if not indices:

        print(
            "    none"
        )

        return

    # Limit to strongest few.
    indices = indices[:5]

    df = (
        f_grid[1]
        - f_grid[0]
    )

    candidates = []

    for index in indices:

        print(
            f"    f={f_grid[index]:.10f} Hz "
            f"|Res|={values[index]:.6e}"
        )

        refined = refine_real_candidate(
            f_grid[index],
            df,
            M,
            r_sp,
            omega_lr,
        )

        candidates.append(
            refined
        )

    # ------------------------------------------------------------------
    # Sort / deduplicate
    # ------------------------------------------------------------------

    candidates.sort(
        key=lambda x:
            x["f_real"]
    )

    distinct = []

    for candidate in candidates:

        if not distinct:

            distinct.append(
                candidate
            )

            continue

        if abs(
            candidate["f_real"]
            - distinct[-1]["f_real"]
        ) > 0.5 * df:

            distinct.append(
                candidate
            )

        elif (
            candidate["abs_res"]
            < distinct[-1]["abs_res"]
        ):

            distinct[-1] = candidate

    print()
    print(
        "Real-axis refined candidates:"
    )

    for candidate in distinct:

        print(
            f"    f={candidate['f_real']:.10f} Hz "
            f"|Res|={candidate['abs_res']:.6e}"
        )

    # ------------------------------------------------------------------
    # Complex search
    # ------------------------------------------------------------------

    print()
    print(
        "Complex refinement:"
    )

    valid_poles = []

    for candidate in distinct:

        f0 = candidate["f_real"]
        real_res = candidate["abs_res"]

        print()
        print(
            f"    candidate f={f0:.10f} Hz "
            f"|Res|={real_res:.6e}"
        )

        # Cheap gate.
        if (
            real_res
            > MAX_REAL_RES_FOR_COMPLEX
        ):

            print(
                f"        SKIP: "
                f"|Res| > "
                f"{MAX_REAL_RES_FOR_COMPLEX}"
            )

            continue

        pole = adaptive_complex_search(
            f0,
            M,
        )

        if pole is None:

            print(
                "        NO VALID POLE"
            )

            continue

        valid_poles.append(
            pole
        )

    # ------------------------------------------------------------------
    # Final poles
    # ------------------------------------------------------------------

    valid_poles.sort(
        key=lambda x:
            x["f_re"]
    )

    print()
    print("=" * 78)
    print(
        "VALIDATED POLES"
    )
    print("=" * 78)

    if not valid_poles:

        print(
            "    none"
        )

    for q, pole in enumerate(
        valid_poles,
        start=1,
    ):

        print()
        print(
            f"q={q}"
        )

        print(
            f"    f_re = "
            f"{pole['f_re']:.12f} Hz"
        )

        print(
            f"    f_im = "
            f"{pole['f_im']:.12f} Hz"
        )

        print(
            f"    omega_re = "
            f"{pole['omega_re']:.12f} rad/s"
        )

        print(
            f"    omega_im = "
            f"{pole['omega_im']:.12f} rad/s"
        )

        print(
            f"    |Res| = "
            f"{pole['abs_res']:.8e}"
        )

        print(
            f"    root_success = "
            f"{pole['success']}"
        )

    # ------------------------------------------------------------------
    # Regression
    # ------------------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "REGRESSION CHECK"
    )
    print("=" * 78)

    if not valid_poles:

        print(
            "FAIL: no validated pole."
        )

        return

    best = min(
        valid_poles,
        key=lambda x:
            abs(
                x["f_re"]
                - EXPECTED_RE
            )
            +
            abs(
                x["f_im"]
                - EXPECTED_IM
            ),
    )

    delta_re = abs(
        best["f_re"]
        - EXPECTED_RE
    )

    delta_im = abs(
        best["f_im"]
        - EXPECTED_IM
    )

    print(
        f"Expected:"
    )

    print(
        f"    Re = "
        f"{EXPECTED_RE:.12f} Hz"
    )

    print(
        f"    Im = "
        f"{EXPECTED_IM:.12f} Hz"
    )

    print()

    print(
        f"Found:"
    )

    print(
        f"    Re = "
        f"{best['f_re']:.12f} Hz"
    )

    print(
        f"    Im = "
        f"{best['f_im']:.12f} Hz"
    )

    print()

    print(
        f"|Delta Re| = "
        f"{delta_re:.3e} Hz"
    )

    print(
        f"|Delta Im| = "
        f"{delta_im:.3e} Hz"
    )

    if (
        delta_re
        <= REGRESSION_RE_TOL
        and
        delta_im
        <= REGRESSION_IM_TOL
        and
        best["success"]
    ):

        print()
        print(
            "REGRESSION: PASS"
        )

    else:

        print()
        print(
            "REGRESSION: FAIL"
        )

    print()
    print("=" * 78)
    print(
        "STAGE 5I-4 v5 LIGHT TEST COMPLETE"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()