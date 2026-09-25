# -*- coding: utf-8 -*-
"""
STAGE 6.5B-3 — OPTIMIZED C-DEPENDENT RESONANCE SEARCH
=======================================================

Purpose
-------
Test whether the complex resonance candidate changes reproducibly when C
is varied, without:
  - Kerr fitting
  - GW calibration
  - frequency rescaling
  - low-k approximation
  - modification of the authoritative solver on disk

Important
---------
The real-valued solver accepts C explicitly.

The complex solver uses module globals, therefore C is injected ONLY into
the in-memory authoritative module.

This version deliberately avoids a large blind 2-D radial scan.

Strategy
--------
For each C with an interior light ring:

  1. Compute C-specific light-ring frequency.
  2. Build a compact frequency window around f_lr.
  3. Test a small set of complex frequency seeds.
  4. Validate turning point.
  5. Compute radial action using adaptive branch continuation.
  6. Evaluate |Res|.
  7. Keep best candidates.
  8. Run Nelder-Mead from the best candidate.
  9. Re-evaluate the final candidate at higher radial resolution.

C=0:
  If no interior light ring exists, no resonance is force-fit.

Outputs
-------
stage6_5B3_resonance_c_scan/
    stage6_5B3_resonance_c_scan_results.csv
    stage6_5B3_resonance_c_scan_results.json
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import inspect
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


# ============================================================================
# PATHS
# ============================================================================

HERE = Path(__file__).resolve().parent

RES_FILE = HERE / "stage5I_3_resonance.py"
PATCH_FILE = HERE / "stage5I_3c_radial_action_patch.py"

OUT_DIR = HERE / "stage6_5B3_resonance_c_scan"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = OUT_DIR / "stage6_5B3_resonance_c_scan_results.csv"
JSON_FILE = OUT_DIR / "stage6_5B3_resonance_c_scan_results.json"


# ============================================================================
# BASELINE CONSTANTS
# ============================================================================

M_MODE = -12

C0 = 7.690e-4
OMEGA = 0.10
GAMMA = 2.22e-6
H0 = 3.40e-2
G_GRAV = 9.81
R_B = 3.73e-2


CASES = [
    ("C_ZERO", 0.0),
    ("C_HALF", 0.5 * C0),
    ("BASELINE", C0),
    ("C_DOUBLE", 2.0 * C0),
]


# ============================================================================
# NUMERICAL SETTINGS
# ============================================================================

# Cheap search.
SEARCH_N = 1000

# Final confirmation.
FINAL_N = 10000

# Small local seed set.
N_FRE_SEEDS = 7
N_FIM_SEEDS = 5

# Frequency window around the C-specific light-ring frequency.
FRE_LOW_FACTOR = 0.70
FRE_HIGH_FACTOR = 1.15

# Damping search.
FIM_MIN = -3.0e-3
FIM_MAX = -2.0e-5

# Number of best points sent to optimizer.
TOP_CANDIDATES = 3

# Optimization.
OPT_MAXITER = 35
OPT_XTOL = 2.0e-7
OPT_FTOL = 2.0e-7

# Continuation.
MAX_CONT_STEPS = 2500
MIN_STEP = 1.0e-10
MAX_STEP = 5.0e-5

# Turning-point Newton.
TURN_MAXITER = 60
TURN_TOL = 1.0e-10

# Complex p Newton.
P_MAXITER = 25
P_TOL = 1.0e-9

# Radial action acceptance.
ACTION_H_TOL = 1.0e-7

# Verbosity.
PRINT_EVERY = 1


# ============================================================================
# UTILITIES
# ============================================================================

def fmt_complex(z):
    z = complex(z)
    return f"{z.real:+.9e}{z.imag:+.9e}i"


def safe_float(x):
    try:
        x = float(x)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return None


def finite_complex(z):
    try:
        z = complex(z)
        return math.isfinite(z.real) and math.isfinite(z.imag)
    except Exception:
        return False


# ============================================================================
# AUTHORITATIVE SOLVER LOADING
# ============================================================================

def load_authoritative_solver():
    if not RES_FILE.exists():
        raise FileNotFoundError(
            f"Authoritative solver not found:\n{RES_FILE}"
        )

    spec = importlib.util.spec_from_file_location(
        "_stage5I_3_authoritative_653B3",
        str(RES_FILE),
    )

    module = importlib.util.module_from_spec(spec)

    module.__file__ = str(RES_FILE)
    module.__name__ = "_stage5I_3_authoritative_653B3"

    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))

    spec.loader.exec_module(module)

    return module


# ============================================================================
# PATCH LOADING
# ============================================================================

def load_patch(solver):
    if not PATCH_FILE.exists():
        raise FileNotFoundError(
            f"Radial patch not found:\n{PATCH_FILE}"
        )

    source = PATCH_FILE.read_text(encoding="utf-8")

    # Patch expects DEFAULT_COMPLEX_RADIAL_N.
    solver.DEFAULT_COMPLEX_RADIAL_N = FINAL_N

    namespace = {
        "__file__": str(PATCH_FILE),
        "__name__": "_stage5I_3c_patch_653B3",
        "__package__": None,
    }

    # The patch imports stage5I_3_resonance by name.
    sys.modules["stage5I_3_resonance"] = solver

    exec(compile(source, str(PATCH_FILE), "exec"), namespace)

    return namespace


# ============================================================================
# C INJECTION
# ============================================================================

def inject_C(solver, C_value):
    """
    Inject C only into the in-memory authoritative module.

    No source file is changed.
    """

    solver.C_METHODS = float(C_value)
    solver.C_TABLE = float(C_value)

    # Some authoritative versions may expose C directly.
    if hasattr(solver, "C"):
        solver.C = float(C_value)


# ============================================================================
# HAMILTONIAN DERIVATIVES
# ============================================================================

def numerical_derivative_r(solver, omega, p, m, r):
    """
    Complex-safe finite difference derivative dH/dr.
    """

    scale = max(abs(r), 1.0e-3)
    dr = 1.0e-7 * scale

    rp = r + dr
    rm = r - dr

    Hp = solver.hamiltonian(omega, p, m, rp)
    Hm = solver.hamiltonian(omega, p, m, rm)

    return (Hp - Hm) / (2.0 * dr)


def numerical_derivative_p(solver, omega, p, m, r):
    """
    Complex-safe finite difference derivative dH/dp.
    """

    dp = 1.0e-7 * max(abs(p), 1.0)

    pp = p + dp
    pm = p - dp

    Hp = solver.hamiltonian(omega, pp, m, r)
    Hm = solver.hamiltonian(omega, pm, m, r)

    return (Hp - Hm) / (2.0 * dp)


def numerical_second_p(solver, omega, p, m, r):
    """
    Complex-safe second derivative d2H/dp2.
    """

    dp = 1.0e-5 * max(abs(p), 1.0)

    H0 = solver.hamiltonian(omega, p, m, r)
    Hp = solver.hamiltonian(omega, p + dp, m, r)
    Hm = solver.hamiltonian(omega, p - dp, m, r)

    return (Hp - 2.0 * H0 + Hm) / (dp * dp)


# ============================================================================
# COMPLEX TURNING POINT
# ============================================================================

def find_turning_point_safe(solver, omega, m, r_real):
    """
    Use authoritative complex turning-point routine first.

    If that fails, perform a local Newton solve for H(omega,0,m,r)=0
    starting from the real turning point.

    This is deliberately conservative.
    """

    try:
        r0 = solver.find_turning_point_complex(
            omega,
            m,
            r_real,
        )

        if finite_complex(r0):
            H = solver.hamiltonian(
                omega,
                0j,
                m,
                r0,
            )

            if finite_complex(H) and abs(H) < 1.0e-6:
                return complex(r0)

    except Exception:
        pass

    # Fallback Newton on H(omega, p=0, m, r)=0.
    r = complex(r_real)

    for _ in range(TURN_MAXITER):

        try:
            H = solver.hamiltonian(
                omega,
                0j,
                m,
                r,
            )

            if not finite_complex(H):
                return None

            if abs(H) < TURN_TOL:
                return r

            dH = numerical_derivative_r(
                solver,
                omega,
                0j,
                m,
                r,
            )

            if not finite_complex(dH) or abs(dH) < 1.0e-18:
                return None

            step = H / dH

            # Prevent catastrophic Newton jumps.
            if abs(step) > 5.0e-4:
                step *= (5.0e-4 / abs(step))

            r_new = r - step

            if not finite_complex(r_new):
                return None

            r = r_new

        except Exception:
            return None

    try:
        H = solver.hamiltonian(
            omega,
            0j,
            m,
            r,
        )

        if finite_complex(H) and abs(H) < 1.0e-6:
            return r

    except Exception:
        pass

    return None


# ============================================================================
# COMPLEX RADIAL MOMENTUM
# ============================================================================

def solve_p_safe(solver, omega, m, r, p_seed):
    """
    Newton solve H(omega,p,m,r)=0.

    Multiple bounded attempts are made.
    """

    seeds = [
        complex(p_seed),
        complex(-p_seed),
        complex(0.5 * p_seed),
        complex(-0.5 * p_seed),
    ]

    for seed in seeds:

        if not finite_complex(seed):
            continue

        p = seed

        for _ in range(P_MAXITER):

            try:
                H = solver.hamiltonian(
                    omega,
                    p,
                    m,
                    r,
                )

                if not finite_complex(H):
                    break

                if abs(H) < P_TOL:
                    return p

                dHdp = numerical_derivative_p(
                    solver,
                    omega,
                    p,
                    m,
                    r,
                )

                if not finite_complex(dHdp):
                    break

                if abs(dHdp) < 1.0e-14:
                    break

                step = H / dHdp

                if not finite_complex(step):
                    break

                # Hard step limiter.
                if abs(step) > 5.0:
                    step *= 5.0 / abs(step)

                p_new = p - step

                if not finite_complex(p_new):
                    break

                p = p_new

            except Exception:
                break

        # Final check.
        try:
            H = solver.hamiltonian(
                omega,
                p,
                m,
                r,
            )

            if finite_complex(H) and abs(H) < 1.0e-7:
                return p

        except Exception:
            pass

    return None


# ============================================================================
# ADAPTIVE RADIAL ACTION
# ============================================================================

def radial_action_adaptive(
    solver,
    omega,
    m,
    r_turn,
    r_B,
    n=SEARCH_N,
):
    """
    Adaptive branch-following radial action.

    Starts at the complex turning point with the same analytic square-root
    seed used by the validated patch.

    Instead of blindly continuing through all points with a fixed seed,
    it retries with:
      - previous p
      - linear predictor
      - half predictor
      - quarter predictor

    This directly addresses the branch failures seen in 6.5B-2.
    """

    r_turn = complex(r_turn)

    # Real integration endpoint.
    r0 = r_turn.real

    if not math.isfinite(r0):
        raise RuntimeError("Non-finite turning-point real part")

    if r0 >= r_B:
        raise RuntimeError(
            f"Turning point outside integration interval: "
            f"{r0} >= {r_B}"
        )

    if r0 <= 0.0:
        raise RuntimeError(
            f"Invalid turning point: {r0}"
        )

    # Use a modest fixed number of points.
    n = int(max(100, n))

    # The actual contour begins at r_turn and approaches r_B.
    r_values = np.linspace(
        r0,
        r_B,
        n,
    )

    dr0 = r_values[1] - r_values[0]

    # Analytic turning-point seed:
    #
    # H ≈ Hr Δr + 1/2 Hpp p²
    #
    # p² ≈ -2 Hr Δr / Hpp
    #

    Hr = numerical_derivative_r(
        solver,
        omega,
        0j,
        m,
        r_turn,
    )

    Hpp = numerical_second_p(
        solver,
        omega,
        0j,
        m,
        r_turn,
    )

    if (
        not finite_complex(Hr)
        or not finite_complex(Hpp)
        or abs(Hpp) < 1.0e-16
    ):
        raise RuntimeError("Invalid turning-point derivatives")

    radicand = -2.0 * Hr * dr0 / Hpp

    if not finite_complex(radicand):
        raise RuntimeError("Non-finite turning-point radicand")

    p_seed = np.sqrt(complex(radicand))

    if not finite_complex(p_seed):
        raise RuntimeError("Invalid analytic p seed")

    p = solve_p_safe(
        solver,
        omega,
        m,
        r_values[1],
        p_seed,
    )

    if p is None:
        raise RuntimeError(
            "First complex radial momentum solve failed"
        )

    p_prev = p
    r_prev = r_values[1]

    # Initial action segment.
    action = 0.0 + 0.0j

    p_max = abs(p)

    previous_dp = p

    for idx in range(2, len(r_values)):

        r = r_values[idx]

        # Linear predictor.
        p_pred = p_prev + previous_dp

        candidates = [
            p_prev,
            p_pred,
            0.5 * p_pred + 0.5 * p_prev,
            0.25 * p_pred + 0.75 * p_prev,
        ]

        p_new = None

        for seed in candidates:

            if not finite_complex(seed):
                continue

            candidate = solve_p_safe(
                solver,
                omega,
                m,
                r,
                seed,
            )

            if candidate is None:
                continue

            try:
                Hcheck = solver.hamiltonian(
                    omega,
                    candidate,
                    m,
                    r,
                )

                if not finite_complex(Hcheck):
                    continue

                if abs(Hcheck) > ACTION_H_TOL:
                    continue

                p_new = candidate
                break

            except Exception:
                continue

        if p_new is None:

            # One final exact previous-point attempt.
            candidate = solve_p_safe(
                solver,
                omega,
                m,
                r,
                p_prev,
            )

            if candidate is None:
                raise RuntimeError(
                    f"Complex p continuation failed at step {idx}"
                )

            p_new = candidate

        # Trapezoidal integration.
        action += 0.5 * (p_prev + p_new) * (r - r_prev)

        previous_dp = p_new - p_prev

        p_prev = p_new
        r_prev = r

        if abs(p_new) > p_max:
            p_max = abs(p_new)

        if not finite_complex(action):
            raise RuntimeError(
                f"Non-finite action at step {idx}"
            )

    return action, p_max, len(r_values)


# ============================================================================
# RESONANCE EVALUATION
# ============================================================================

def evaluate_resonance(
    solver,
    omega,
    m,
    r_sp,
    omega_lr,
    n,
):
    """
    Evaluate:
        Res = R exp(2 i S) - 1
    """

    if not finite_complex(omega):
        raise RuntimeError("Non-finite omega")

    # Turning point.
    r_turn_real = solver.find_last_scattering_point(
        omega,
        m,
        r_sp,
    )

    if not finite_complex(r_turn_real):
        raise RuntimeError(
            "Invalid last-scattering point"
        )

    r_turn = find_turning_point_safe(
        solver,
        omega,
        m,
        r_turn_real,
    )

    if r_turn is None:
        raise RuntimeError(
            "Turning-point complex solve failed"
        )

    H_turn = solver.hamiltonian(
        omega,
        0j,
        m,
        r_turn,
    )

    if not finite_complex(H_turn):
        raise RuntimeError(
            "Non-finite H at turning point"
        )

    action, p_max, n_points = radial_action_adaptive(
        solver,
        omega,
        m,
        r_turn,
        R_B,
        n=n,
    )

    R = solver.reflection_coefficient(
        omega,
        m,
        r_sp,
        omega_lr,
    )

    if not finite_complex(R):
        raise RuntimeError(
            "Non-finite reflection coefficient"
        )

    Res = R * np.exp(2j * action) - 1.0

    if not finite_complex(Res):
        raise RuntimeError(
            "Non-finite resonance residual"
        )

    return {
        "f_R": omega.real / (2.0 * np.pi),
        "f_I": omega.imag / (2.0 * np.pi),
        "abs_res": abs(Res),
        "re_res": Res.real,
        "im_res": Res.imag,
        "Q": (
            abs(omega.real)
            /
            max(2.0 * abs(omega.imag), 1.0e-300)
        ),
        "r_minus": r_turn_real,
        "r_turn": r_turn,
        "H_turn_abs": abs(H_turn),
        "action": action,
        "R_abs": abs(R),
        "p_max": p_max,
        "action_points": n_points,
    }


# ============================================================================
# LIGHT RING
# ============================================================================

def get_light_ring(solver, C):
    inject_C(solver, C)

    r_sp, omega_lr = solver.find_light_ring(
        M_MODE,
        C=C,
        Omega=OMEGA,
        gamma=GAMMA,
        h0=H0,
        g=G_GRAV,
        r_min=1.0e-3,
        r_max=R_B,
    )

    if r_sp is None or omega_lr is None:
        return None, None

    return float(r_sp), float(omega_lr)


# ============================================================================
# LOCAL SEED GENERATION
# ============================================================================

def make_local_seeds(f_lr):
    """
    C-specific seed cloud.

    No inherited baseline frequency is used.

    The search is intentionally concentrated around the light-ring scale.
    """

    fre_values = np.linspace(
        FRE_LOW_FACTOR * f_lr,
        FRE_HIGH_FACTOR * f_lr,
        N_FRE_SEEDS,
    )

    # Include the exact light-ring frequency.
    fre_values = np.unique(
        np.concatenate(
            [
                fre_values,
                np.array([f_lr]),
            ]
        )
    )

    # Damping values.
    fim_values = np.linspace(
        FIM_MIN,
        FIM_MAX,
        N_FIM_SEEDS,
    )

    seeds = []

    for f_re in fre_values:
        for f_im in fim_values:
            seeds.append(
                complex(
                    f_re,
                    f_im,
                )
            )

    return seeds


# ============================================================================
# QUICK VALIDATION
# ============================================================================

def quick_validate_seed(
    solver,
    f,
    r_sp,
    omega_lr,
):
    """
    Only perform cheap turning-point validation first.
    """

    omega = 2.0 * np.pi * complex(f)

    try:

        r_minus = solver.find_last_scattering_point(
            omega,
            M_MODE,
            r_sp,
        )

        if not finite_complex(r_minus):
            return False

        r_turn = find_turning_point_safe(
            solver,
            omega,
            M_MODE,
            r_minus,
        )

        if r_turn is None:
            return False

        H = solver.hamiltonian(
            omega,
            0j,
            M_MODE,
            r_turn,
        )

        return finite_complex(H) and abs(H) < 1.0e-6

    except Exception:
        return False


# ============================================================================
# LANDSCAPE SEARCH
# ============================================================================

def local_landscape(
    solver,
    case_name,
    C,
    r_sp,
    omega_lr,
):
    print()
    print("-" * 78)
    print(
        f"{case_name}: LOCAL C-SPECIFIC RESONANCE SEARCH"
    )
    print("-" * 78)

    f_lr = omega_lr / (2.0 * np.pi)

    seeds = make_local_seeds(f_lr)

    print(
        f"  f_lr = {f_lr:.12f} Hz"
    )
    print(
        f"  search f_R = "
        f"[{FRE_LOW_FACTOR*f_lr:.6f}, "
        f"{FRE_HIGH_FACTOR*f_lr:.6f}] Hz"
    )
    print(
        f"  f_I = [{FIM_MIN:.6f}, {FIM_MAX:.6f}] Hz"
    )
    print(
        f"  seeds = {len(seeds)}"
    )
    print(
        f"  radial N = {SEARCH_N}"
    )

    candidates = []

    for i, f in enumerate(seeds, 1):

        ok = quick_validate_seed(
            solver,
            f,
            r_sp,
            omega_lr,
        )

        if not ok:
            continue

        try:

            result = evaluate_resonance(
                solver,
                2.0 * np.pi * f,
                M_MODE,
                r_sp,
                omega_lr,
                SEARCH_N,
            )

            candidates.append(
                (
                    result["abs_res"],
                    f,
                    result,
                )
            )

            print(
                f"  VALID [{i:3d}/{len(seeds)}] "
                f"f={f.real:.6f}{f.imag:+.6f}i "
                f"|Res|={result['abs_res']:.6e}"
            )

        except Exception as exc:

            # Do not flood the terminal with RuntimeError details.
            if PRINT_EVERY and i % PRINT_EVERY == 0:
                print(
                    f"  seed [{i:3d}/{len(seeds)}] "
                    f"f={f.real:.6f}{f.imag:+.6f}i "
                    f"-> fail"
                )

    if not candidates:
        print()
        print("  STATUS = NO_VALID_LOCAL_CANDIDATE")
        return []

    candidates.sort(
        key=lambda x: x[0]
    )

    print()
    print("  TOP LOCAL CANDIDATES:")

    for rank, item in enumerate(
        candidates[:TOP_CANDIDATES],
        1,
    ):

        _, f, result = item

        print(
            f"    {rank}: "
            f"f={f.real:.9f}{f.imag:+.9f}i "
            f"|Res|={result['abs_res']:.6e}"
        )

    return candidates[:TOP_CANDIDATES]


# ============================================================================
# OPTIMIZATION
# ============================================================================

def optimize_candidate(
    solver,
    start_f,
    r_sp,
    omega_lr,
):
    """
    Nelder-Mead over (f_R,f_I).

    Objective is log10(1+|Res|), which remains finite.
    """

    eval_cache = {}

    def objective(x):

        f_re = float(x[0])
        f_im = float(x[1])

        # Physical search guard.
        if not math.isfinite(f_re) or not math.isfinite(f_im):
            return 50.0

        if f_re <= 0.0:
            return 50.0

        if f_im >= 0.0:
            return 50.0

        key = (
            round(f_re, 12),
            round(f_im, 12),
        )

        if key in eval_cache:
            return eval_cache[key]

        try:

            result = evaluate_resonance(
                solver,
                2.0 * np.pi * complex(
                    f_re,
                    f_im,
                ),
                M_MODE,
                r_sp,
                omega_lr,
                SEARCH_N,
            )

            value = math.log10(
                1.0 + result["abs_res"]
            )

            if not math.isfinite(value):
                value = 50.0

        except Exception:
            value = 50.0

        eval_cache[key] = value

        return value

    x0 = np.array(
        [
            start_f.real,
            start_f.imag,
        ],
        dtype=float,
    )

    # Small initial simplex.
    simplex = np.array(
        [
            x0,
            x0 + np.array([2.0e-3, 0.0]),
            x0 + np.array([0.0, 2.0e-5]),
        ]
    )

    result = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={
            "maxiter": OPT_MAXITER,
            "xatol": OPT_XTOL,
            "fatol": OPT_FTOL,
            "disp": False,
            "initial_simplex": simplex,
        },
    )

    return result


# ============================================================================
# FINAL HIGH-RESOLUTION EVALUATION
# ============================================================================

def final_evaluation(
    solver,
    opt_result,
    r_sp,
    omega_lr,
):
    f_re = float(opt_result.x[0])
    f_im = float(opt_result.x[1])

    omega = 2.0 * np.pi * complex(
        f_re,
        f_im,
    )

    return evaluate_resonance(
        solver,
        omega,
        M_MODE,
        r_sp,
        omega_lr,
        FINAL_N,
    )


# ============================================================================
# C INJECTION AUDIT
# ============================================================================

def injection_audit(solver):
    print()
    print("-" * 78)
    print("COMPLEX C-INJECTION AUDIT")
    print("-" * 78)

    r_probe = 0.025
    f_probe = 8.0
    omega_probe = 2.0 * np.pi * complex(
        f_probe,
        -5.0e-4,
    )
    p_probe = 1.0 + 0.01j

    values = {}

    for label, factor in [
        ("C/2", 0.5),
        ("C", 1.0),
        ("2C", 2.0),
    ]:

        inject_C(
            solver,
            factor * C0,
        )

        try:

            H = solver.hamiltonian(
                omega_probe,
                p_probe,
                M_MODE,
                r_probe,
            )

            values[label] = complex(H)

            print(
                f"  {label:4s} = "
                f"{fmt_complex(H)}"
            )

        except Exception as exc:

            print(
                f"  {label:4s} = FAIL: {type(exc).__name__}"
            )

            return False

    d1 = abs(
        values["C/2"] - values["C"]
    )

    d2 = abs(
        values["2C"] - values["C"]
    )

    print()
    print(
        f"  |H(C/2)-H(C)| = {d1:.8e}"
    )
    print(
        f"  |H(2C)-H(C)|  = {d2:.8e}"
    )

    passed = (
        math.isfinite(d1)
        and math.isfinite(d2)
        and d1 > 1.0e-8
        and d2 > 1.0e-8
    )

    print()
    print(
        "C-INJECTION AUDIT: "
        + ("PASS" if passed else "FAIL")
    )

    return passed


# ============================================================================
# MAIN CASE
# ============================================================================

def run_case(
    solver,
    case_name,
    C,
):
    print()
    print("=" * 78)
    print(
        f"RUNNING CASE: {case_name}"
    )
    print("=" * 78)

    print(
        f"C = {C:.12e} m^2/s"
    )

    inject_C(
        solver,
        C,
    )

    # ------------------------------------------------------------------
    # LIGHT RING
    # ------------------------------------------------------------------

    r_sp, omega_lr = get_light_ring(
        solver,
        C,
    )

    if r_sp is None:

        print()
        print(
            "STATUS = NO_LIGHT_RING"
        )

        return {
            "case": case_name,
            "C": C,
            "status": "NO_LIGHT_RING",
            "r_sp": None,
            "f_lr": None,
        }

    f_lr = omega_lr / (2.0 * np.pi)

    print()
    print(
        f"  light ring = {r_sp*1000.0:.12f} mm"
    )
    print(
        f"  f_lr        = {f_lr:.12f} Hz"
    )

    # Complex probe.
    try:

        H_probe = solver.hamiltonian(
            2.0 * np.pi * complex(
                f_lr,
                -5.0e-4,
            ),
            1.0 + 0.01j,
            M_MODE,
            r_sp,
        )

        print(
            f"  complex H probe = "
            f"{fmt_complex(H_probe)}"
        )

    except Exception as exc:

        print(
            f"  complex H probe = FAIL: "
            f"{type(exc).__name__}"
        )

    # ------------------------------------------------------------------
    # LOCAL SEARCH
    # ------------------------------------------------------------------

    candidates = local_landscape(
        solver,
        case_name,
        C,
        r_sp,
        omega_lr,
    )

    if not candidates:

        return {
            "case": case_name,
            "C": C,
            "status": "NO_VALID_LOCAL_CANDIDATE",
            "r_sp": r_sp,
            "f_lr": f_lr,
        }

    # ------------------------------------------------------------------
    # OPTIMIZATION
    # ------------------------------------------------------------------

    best_opt = None
    best_opt_value = np.inf
    best_start = None

    print()
    print(
        f"  OPTIMIZATION: "
        f"{len(candidates)} candidate(s)"
    )

    for rank, item in enumerate(
        candidates,
        1,
    ):

        _, start_f, coarse_result = item

        print()
        print(
            f"  candidate #{rank}: "
            f"start = "
            f"{start_f.real:.9f}"
            f"{start_f.imag:+.9f}i"
        )
        print(
            f"    coarse |Res| = "
            f"{coarse_result['abs_res']:.6e}"
        )

        try:

            opt = optimize_candidate(
                solver,
                start_f,
                r_sp,
                omega_lr,
            )

            value = float(opt.fun)

            print(
                f"    success = {opt.success}"
            )
            print(
                f"    iterations = "
                f"{getattr(opt, 'nit', -1)}"
            )
            print(
                f"    objective = "
                f"{value:.9e}"
            )

            if value < best_opt_value:
                best_opt = opt
                best_opt_value = value
                best_start = start_f

        except Exception as exc:

            print(
                f"    optimizer failed: "
                f"{type(exc).__name__}"
            )

    if best_opt is None:

        return {
            "case": case_name,
            "C": C,
            "status": "OPTIMIZATION_FAILED",
            "r_sp": r_sp,
            "f_lr": f_lr,
        }

    # ------------------------------------------------------------------
    # FINAL HIGH-RESOLUTION EVALUATION
    # ------------------------------------------------------------------

    try:

        final = final_evaluation(
            solver,
            best_opt,
            r_sp,
            omega_lr,
        )

    except Exception as exc:

        print()
        print(
            "  FINAL EVALUATION FAILED: "
            f"{type(exc).__name__}"
        )

        return {
            "case": case_name,
            "C": C,
            "status": "FINAL_EVALUATION_FAILED",
            "r_sp": r_sp,
            "f_lr": f_lr,
            "optimizer_success": bool(
                best_opt.success
            ),
        }

    # ------------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------------

    print()
    print(
        f"{case_name}: FINAL RESULT"
    )
    print("-" * 78)

    print(
        f"  STATUS      = OK"
    )
    print(
        f"  f_R         = "
        f"{final['f_R']:.12f} Hz"
    )
    print(
        f"  f_I         = "
        f"{final['f_I']:.12f} Hz"
    )
    print(
        f"  Q           = "
        f"{final['Q']:.9e}"
    )
    print(
        f"  |Res|       = "
        f"{final['abs_res']:.9e}"
    )
    print(
        f"  Re(Res)     = "
        f"{final['re_res']:+.9e}"
    )
    print(
        f"  Im(Res)     = "
        f"{final['im_res']:+.9e}"
    )
    print(
        f"  r_sp        = "
        f"{r_sp*1000.0:.12f} mm"
    )
    print(
        f"  f_lr        = "
        f"{f_lr:.12f} Hz"
    )
    print(
        f"  r_minus     = "
        f"{final['r_minus']}"
    )
    print(
        f"  r_turn      = "
        f"{final['r_turn']}"
    )
    print(
        f"  |H_turn|    = "
        f"{final['H_turn_abs']:.9e}"
    )
    print(
        f"  action      = "
        f"{fmt_complex(final['action'])}"
    )
    print(
        f"  |R|         = "
        f"{final['R_abs']:.9e}"
    )
    print(
        f"  action pts  = "
        f"{final['action_points']}"
    )
    print(
        f"  max |p|     = "
        f"{final['p_max']:.9e}"
    )
    print(
        f"  optimizer   = Nelder-Mead"
    )

    row = {
        "case": case_name,
        "C": C,
        "status": "OK",
        "r_sp_m": r_sp,
        "r_sp_mm": r_sp * 1000.0,
        "f_lr_Hz": f_lr,
        "f_R_Hz": final["f_R"],
        "f_I_Hz": final["f_I"],
        "Q": final["Q"],
        "abs_res": final["abs_res"],
        "re_res": final["re_res"],
        "im_res": final["im_res"],
        "r_minus_m": final["r_minus"],
        "r_turn_real_m": final["r_turn"].real,
        "r_turn_imag_m": final["r_turn"].imag,
        "H_turn_abs": final["H_turn_abs"],
        "action_real": final["action"].real,
        "action_imag": final["action"].imag,
        "R_abs": final["R_abs"],
        "action_points": final["action_points"],
        "p_max": final["p_max"],
        "optimizer_success": bool(
            best_opt.success
        ),
        "optimizer_iterations": int(
            getattr(best_opt, "nit", -1)
        ),
        "optimizer_objective": float(
            best_opt.fun
        ),
        "start_f_R_Hz": best_start.real,
        "start_f_I_Hz": best_start.imag,
    }

    return row


# ============================================================================
# SAVE
# ============================================================================

def save_results(rows):
    if not rows:
        return

    keys = sorted(
        {
            key
            for row in rows
            for key in row.keys()
        }
    )

    with CSV_FILE.open(
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

    with JSON_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            rows,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print()
    print("-" * 78)
    print("RESULTS SAVED")
    print("-" * 78)
    print(
        f"CSV  -> {CSV_FILE}"
    )
    print(
        f"JSON -> {JSON_FILE}"
    )


# ============================================================================
# FINAL VERDICT
# ============================================================================

def final_verdict(rows):
    valid = [
        r
        for r in rows
        if r.get("status") == "OK"
    ]

    print()
    print("=" * 78)
    print(
        "RESONANCE-LEVEL C-CAUSALITY VERDICT"
    )
    print("=" * 78)

    if not valid:

        print()
        print(
            "NO VALID RESONANCE RESULTS"
        )
        print()
        print(
            "The optimized local search did not "
            "establish a valid resonance candidate."
        )
        print()
        print(
            "This is a numerical/search result only."
        )
        return

    print()
    print(
        "VALID RESONANCE RESULTS:"
    )

    for r in valid:

        print(
            f"  {r['case']:10s} "
            f"C={r['C']:.6e} "
            f"f={r['f_R_Hz']:.9f}"
            f"{r['f_I_Hz']:+.9f}i Hz "
            f"|Res|={r['abs_res']:.3e} "
            f"Q={r['Q']:.3e}"
        )

    f_R = np.array(
        [
            r["f_R_Hz"]
            for r in valid
        ]
    )

    f_I = np.array(
        [
            r["f_I_Hz"]
            for r in valid
        ]
    )

    r_sp = np.array(
        [
            r["r_sp_mm"]
            for r in valid
        ]
    )

    print()
    print(
        f"  f_R spread = "
        f"{np.ptp(f_R):.9e} Hz"
    )
    print(
        f"  f_I spread = "
        f"{np.ptp(f_I):.9e} Hz"
    )
    print(
        f"  r_sp spread = "
        f"{np.ptp(r_sp):.9e} mm"
    )

    # Only claim a resolved resonance response if at least two
    # different C values have valid resonance solutions.
    unique_C = sorted(
        {
            round(
                r["C"],
                15,
            )
            for r in valid
        }
    )

    if len(unique_C) < 2:

        print()
        print(
            "CAUSAL TEST: NOT RESOLVED"
        )
        print()
        print(
            "Only one controlled C produced a valid "
            "resonance candidate."
        )

        return

    print()
    print(
        "CAUSAL TEST: MULTI-C RESONANCE RESULTS AVAILABLE"
    )
    print()
    print(
        "Compare f_R, f_I and Q across C."
    )
    print(
        "A reproducible C-dependent movement is required "
        "before claiming resonance-level causality."
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    print()
    print("=" * 78)
    print(
        "STAGE 6.5B-3 — OPTIMIZED "
        "C-DEPENDENT RESONANCE SEARCH"
    )
    print("=" * 78)

    print()
    print(
        "Authoritative solver:"
    )
    print(
        f"  {RES_FILE}"
    )

    print()
    print(
        "Radial patch:"
    )
    print(
        f"  {PATCH_FILE}"
    )

    print()
    print(
        "Mode:"
    )
    print(
        f"  m = {M_MODE}"
    )

    print()
    print(
        "Baseline constants:"
    )
    print(
        f"  C0     = {C0:.12e} m^2/s"
    )
    print(
        f"  Omega  = {OMEGA:.12e} rad/s"
    )
    print(
        f"  gamma  = {GAMMA:.12e}"
    )
    print(
        f"  h0     = {H0:.12e} m"
    )
    print(
        f"  g      = {G_GRAV:.12e} m/s^2"
    )
    print(
        f"  r_B    = {R_B:.12e} m"
    )

    print()
    print(
        "Controlled cases:"
    )

    for name, C in CASES:
        print(
            f"  {name:10s}: "
            f"C={C:.12e}"
        )

    print()
    print(
        "Optimization:"
    )
    print(
        f"  search radial N = {SEARCH_N}"
    )
    print(
        f"  final radial N  = {FINAL_N}"
    )
    print(
        f"  local seeds      = "
        f"{N_FRE_SEEDS} x {N_FIM_SEEDS}"
    )
    print(
        f"  top candidates   = {TOP_CANDIDATES}"
    )
    print(
        f"  optimizer        = Nelder-Mead"
    )
    print(
        f"  max iterations   = {OPT_MAXITER}"
    )

    print()
    print(
        "Rules:"
    )
    print(
        "  * C-specific frequency search"
    )
    print(
        "  * no inherited baseline frequency"
    )
    print(
        "  * adaptive complex branch continuation"
    )
    print(
        "  * analytic turning-point seed"
    )
    print(
        "  * no Kerr fit"
    )
    print(
        "  * no GW calibration"
    )
    print(
        "  * no frequency rescaling"
    )
    print(
        "  * no low-k approximation"
    )
    print(
        "  * C_ZERO is not force-fit"
    )

    # ------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print(
        "1. LOAD AUTHORITATIVE SOLVER"
    )
    print("-" * 78)

    solver = load_authoritative_solver()

    required = [
        "hamiltonian",
        "solve_p_complex",
        "find_light_ring",
        "find_last_scattering_point",
        "find_turning_point_complex",
        "reflection_coefficient",
    ]

    for name in required:

        if not hasattr(solver, name):
            raise RuntimeError(
                f"Required API missing: {name}"
            )

    print(
        "  Loaded successfully."
    )

    print()
    print(
        "  Required APIs:"
    )

    for name in required:

        fn = getattr(
            solver,
            name,
        )

        try:
            sig = inspect.signature(fn)
        except Exception:
            sig = "(signature unavailable)"

        print(
            f"    {name}: {sig}"
        )

    # ------------------------------------------------------------------
    # PATCH
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print(
        "2. LOAD RADIAL-ACTION PATCH"
    )
    print("-" * 78)

    print(
        "  Installing compatibility shim:"
    )
    print(
        f"    solver.DEFAULT_COMPLEX_RADIAL_N = "
        f"{FINAL_N}"
    )
    print(
        "    [IN MEMORY ONLY]"
    )

    patch = load_patch(
        solver
    )

    if "radial_action_complex_v2" in patch:
        print(
            "  radial_action_complex_v2: AVAILABLE"
        )
    else:
        print(
            "  radial_action_complex_v2: "
            "NOT USED BY THIS TEST"
        )

    # ------------------------------------------------------------------
    # C AUDIT
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print(
        "3. COMPLEX C-INJECTION AUDIT"
    )
    print("-" * 78)

    if not injection_audit(solver):

        raise RuntimeError(
            "Complex C injection audit failed."
        )

    # ------------------------------------------------------------------
    # CASES
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print(
        "4. CONTROLLED C-DEPENDENT RESONANCE SEARCH"
    )
    print("-" * 78)

    rows = []

    for case_name, C in CASES:

        row = run_case(
            solver,
            case_name,
            C,
        )

        rows.append(row)

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print(
        "5. SAVE RESULTS"
    )
    print("-" * 78)

    save_results(rows)

    # ------------------------------------------------------------------
    # VERDICT
    # ------------------------------------------------------------------

    final_verdict(rows)

    print()
    print("=" * 78)
    print(
        "STAGE 6.5B-3 COMPLETE"
    )
    print("=" * 78)

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "A small |Res| is a resonance candidate."
    )
    print(
        "The causal quantity is its reproducible response "
        "to controlled variation of C."
    )
    print(
        "No physical interpretation is claimed by this stage."
    )
    print()
    print(
        "Authoritative solver was NOT modified."
    )
    print(
        "C was injected only in memory."
    )


if __name__ == "__main__":
    main()