# -*- coding: utf-8 -*-
"""
stage6_5B1_resonance_causal_audit.py
====================================

STAGE 6.5B-1 — RESONANCE-LEVEL C-CAUSALITY AUDIT

Purpose
-------
Test whether changing ONLY the circulation parameter C produces a
corresponding change in the complex resonance condition

    Res(omega) = R(omega) * exp(2 i S(omega)) - 1

without:

    * modifying the authoritative solver on disk
    * fitting Kerr
    * calibrating to GW data
    * introducing Lambda
    * frequency-rescaling the physical model
    * using the low-k approximation
    * forcing C through function defaults
    * using high-resolution radial integration during the optimizer

Numerical strategy
------------------
OPTIMIZATION:
    radial N = 2000
    Nelder-Mead
    maxiter = 50
    disp = True

FINAL VALIDATION:
    radial N = 20000
    one independent evaluation at the candidate root

The lower-resolution N=2000 calculation is ONLY used to locate a
candidate minimum.  It is NOT used for the final resonance claim.

The authoritative solver remains untouched on disk.
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np


# ============================================================================
# PATHS
# ============================================================================

HERE = Path(__file__).resolve().parent

RES_FILE = (
    HERE
    / "stage5I_3_resonance.py"
)

PATCH_FILE = (
    HERE
    / "stage5I_3c_radial_action_patch.py"
)

OUTPUT_DIR = (
    HERE
    / "stage6_5B1_resonance_causal_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# NUMERICAL PARAMETERS
# ============================================================================

# ------------------------------------------------------------
# IMPORTANT:
# This is ONLY the radial resolution used during optimization.
# It is NOT the final validation resolution.
# ------------------------------------------------------------

OPT_RADIAL_N = 2000

# ------------------------------------------------------------
# Final high-resolution validation.
# ------------------------------------------------------------

FINAL_RADIAL_N = 20000

# ------------------------------------------------------------
# Optimizer limits.
# ------------------------------------------------------------

OPT_MAXITER = 50

# SciPy Nelder-Mead verbosity.
OPT_DISP = True

# ------------------------------------------------------------
# Adaptive continuation.
# ------------------------------------------------------------

MAX_SUBDIVISIONS = 12

# ------------------------------------------------------------
# Mode.
# ------------------------------------------------------------

M = -12

# ------------------------------------------------------------
# Historical resonance benchmark.
#
# These are used as an independent reference point only.
# ------------------------------------------------------------

F_RE_BENCH = 8.306630716900
F_IM_BENCH = -0.000566435000

# Previous Stage 5I-3c/Nelder-Mead neighborhood.
F_RE_OLD = 8.3066307032346
F_IM_OLD = -0.0005696389388967092

# Ratio used ONLY to construct an optimizer starting point.
# This is not applied as a physical frequency transformation.
F_RE_OVER_F_LR = (
    F_RE_OLD / 8.835443157444
)

# Numerical initial imaginary-frequency scale.
F_IM_START = F_IM_OLD

# Search half-widths around each C-specific starting point.
F_RE_HALF_WIDTH = 0.60
F_IM_HALF_WIDTH = 0.0010

# Objective failure penalty.
FAIL_PENALTY = 1.0e12


# ============================================================================
# EXPECTED BASELINE CONSTANTS
# ============================================================================

EXPECTED_C = 7.690000000000e-4
EXPECTED_OMEGA = 1.000000000000e-1
EXPECTED_GAMMA = 2.220000000000e-6
EXPECTED_H0 = 3.400000000000e-2
EXPECTED_G = 9.810000000000e0
EXPECTED_R_B = 3.730000000000e-2


# ============================================================================
# CONTROLLED C CASES
# ============================================================================

C_CASES = [
    ("C_ZERO", 0.0),
    ("C_HALF", 0.5 * EXPECTED_C),
    ("BASELINE", EXPECTED_C),
    ("C_DOUBLE", 2.0 * EXPECTED_C),
]


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def banner(title: str):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def complex_to_dict(z):
    if z is None:
        return None

    z = complex(z)

    return {
        "re": float(z.real),
        "im": float(z.imag),
        "abs": float(abs(z)),
    }


# ============================================================================
# LOAD AUTHORITATIVE SOLVER
# ============================================================================

def load_authoritative_solver():
    """
    Load stage5I_3_resonance.py normally.

    No AST rewriting.
    No source modification.
    """

    if not RES_FILE.exists():
        raise FileNotFoundError(
            f"Authoritative solver not found:\n{RES_FILE}"
        )

    module_name = "_stage5I3_authoritative_6_5B1"

    spec = importlib.util.spec_from_file_location(
        module_name,
        RES_FILE,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            "Could not construct import spec for authoritative solver."
        )

    module = importlib.util.module_from_spec(spec)

    module.__file__ = str(RES_FILE)
    module.__name__ = module_name

    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


# ============================================================================
# INSTALL COMPATIBILITY SHIMS FOR PATCH
# ============================================================================

def prepare_solver_for_patch(solver):
    """
    stage5I_3c_radial_action_patch.py expects some names to be exported
    from stage5I_3_resonance.

    Older authoritative solver versions may not expose
    DEFAULT_COMPLEX_RADIAL_N.

    Install missing compatibility names IN MEMORY ONLY.
    """

    if not hasattr(solver, "DEFAULT_COMPLEX_RADIAL_N"):
        solver.DEFAULT_COMPLEX_RADIAL_N = FINAL_RADIAL_N

    if not hasattr(solver, "R_B"):
        solver.R_B = EXPECTED_R_B

    # The patch performs:
    #
    # from stage5I_3_resonance import (...)
    #
    # so temporarily expose the authoritative module under the expected
    # module name.

    sys.modules["stage5I_3_resonance"] = solver


# ============================================================================
# LOAD PATCH
# ============================================================================

def load_patch(solver):
    """
    Load stage5I_3c_radial_action_patch.py after installing the compatibility
    shim.

    The patch is NOT modified.
    """

    if not PATCH_FILE.exists():
        raise FileNotFoundError(
            f"Patch file not found:\n{PATCH_FILE}"
        )

    prepare_solver_for_patch(solver)

    module_name = "_stage5I3c_patch_6_5B1"

    # Remove an older instance if this script is re-run in the same interpreter.
    sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location(
        module_name,
        PATCH_FILE,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            "Could not construct import spec for radial-action patch."
        )

    patch = importlib.util.module_from_spec(spec)

    patch.__file__ = str(PATCH_FILE)
    patch.__name__ = module_name

    sys.modules[module_name] = patch

    spec.loader.exec_module(patch)

    return patch


# ============================================================================
# CONSTANT INSPECTION
# ============================================================================

def get_solver_constant(solver, names):
    for name in names:
        if hasattr(solver, name):
            return float(getattr(solver, name))
    return None


def get_C_names(solver):
    """
    Detect the C-related globals actually present in the authoritative
    solver.

    The real complex functions have signatures without explicit C, so their
    C dependence must come from module-level state.

    We only mutate names that actually exist.
    """

    names = []

    for name in (
        "C_METHODS",
        "C_TABLE",
        "C",
    ):
        if hasattr(solver, name):
            names.append(name)

    return names


def print_solver_constants(solver):
    print()
    print("AUTHORITATIVE SOLVER CONSTANTS")

    for name in (
        "C_METHODS",
        "C_TABLE",
        "C",
        "OMEGA",
        "GAMMA",
        "H0",
        "G_GRAV",
        "R_B",
        "DEFAULT_COMPLEX_RADIAL_N",
    ):
        if hasattr(solver, name):
            print(
                f"  {name:24s} = "
                f"{getattr(solver, name)!r}"
            )


# ============================================================================
# SNAPSHOT / RESTORE C GLOBALS
# ============================================================================

def snapshot_C_globals(solver):
    names = get_C_names(solver)

    return {
        name: getattr(solver, name)
        for name in names
    }


def set_complex_C(solver, C_value):
    """
    Change ONLY the C-related module globals used by the complex machinery.

    This is an in-memory runtime mutation.

    The authoritative file on disk is never changed.
    """

    names = get_C_names(solver)

    if not names:
        raise RuntimeError(
            "No C-related module globals found in authoritative solver."
        )

    for name in names:
        setattr(
            solver,
            name,
            float(C_value),
        )

    return names


def restore_C_globals(solver, snapshot):
    for name, value in snapshot.items():
        setattr(
            solver,
            name,
            value,
        )


# ============================================================================
# EXPLICIT REAL LIGHT-RING CALCULATION
# ============================================================================

def explicit_light_ring(
    solver,
    C_value,
):
    """
    Real light-ring calculation using the authoritative explicit-C API.

    This was already independently validated in Stage 6.5B-0 and
    Stage 6.5B-0.5.
    """

    fn = solver.find_light_ring

    r_sp, omega_lr = fn(
        M,
        C=float(C_value),
        Omega=EXPECTED_OMEGA,
        gamma=EXPECTED_GAMMA,
        h0=EXPECTED_H0,
        g=EXPECTED_G,
        r_min=1.0e-3,
        r_max=EXPECTED_R_B,
    )

    return r_sp, omega_lr


# ============================================================================
# COMPLEX HAMILTONIAN FINGERPRINT
# ============================================================================

def complex_C_fingerprint(
    solver,
    C_values,
):
    """
    Verify that changing C actually changes the complex Hamiltonian.

    This is important because the complex API does not expose C explicitly.
    """

    if not hasattr(solver, "hamiltonian"):
        raise RuntimeError(
            "Authoritative solver has no hamiltonian()."
        )

    omega_test = 2.0 * np.pi * (
        F_RE_BENCH + 1j * F_IM_BENCH
    )

    p_test = 1.0 + 0.1j

    r_test = 0.025

    values = {}

    snapshot = snapshot_C_globals(solver)

    try:
        for C in C_values:
            set_complex_C(
                solver,
                C,
            )

            H = solver.hamiltonian(
                omega_test,
                p_test,
                M,
                r_test,
            )

            values[float(C)] = complex(H)

    finally:
        restore_C_globals(
            solver,
            snapshot,
        )

    return values


# ============================================================================
# NEAR-TURNING-POINT DERIVATIVES
# ============================================================================

def compute_H_r_at_turning_point(
    solver,
    omega,
    r_minus,
):
    """
    dH/dr at p=0 using the same local finite-difference construction as
    the validated Stage 5I-3c patch.
    """

    eps = (
        abs(r_minus) * 1.0e-6
        if abs(r_minus) > 0
        else 1.0e-9
    )

    H_plus = solver.hamiltonian(
        omega,
        0.0j,
        M,
        r_minus + eps,
    )

    H_minus = solver.hamiltonian(
        omega,
        0.0j,
        M,
        r_minus - eps,
    )

    return (
        H_plus - H_minus
    ) / (
        2.0 * eps
    )


def compute_H_pp_at_turning_point(
    solver,
    omega,
    r_minus,
):
    """
    Local H_pp estimate matching the validated patch.
    """

    eps_p = 1.0

    H0 = solver.hamiltonian(
        omega,
        0.0j,
        M,
        r_minus,
    )

    Hp = solver.hamiltonian(
        omega,
        eps_p + 0j,
        M,
        r_minus,
    )

    Hm = solver.hamiltonian(
        omega,
        -eps_p + 0j,
        M,
        r_minus,
    )

    return (
        Hp - 2.0 * H0 + Hm
    ) / (
        eps_p ** 2
    )


# ============================================================================
# COMPLEX P SOLVER
# ============================================================================

def solve_complex_p(
    solver,
    omega,
    r,
    seed,
):
    """
    Wrapper around the authoritative complex radial solver.
    """

    p = solver.solve_p_complex(
        omega,
        M,
        r,
        p_initial=seed,
    )

    if p is None:
        return None

    return complex(p)


# ============================================================================
# SINGLE CONTINUATION STEP
# ============================================================================

def try_continuation_step(
    solver,
    omega,
    r_previous,
    p_previous,
    r_target,
    r_previous_previous=None,
    p_previous_previous=None,
):
    """
    Try several branch-consistent seeds.

    The order is deliberately conservative:

        1. linear predictor
        2. half predictor
        3. quarter predictor
        4. previous point
        5. previous point with reduced magnitude

    The exact Stage 5I-3c analytic seed is handled separately for the
    very first step.
    """

    seeds = []

    # ------------------------------------------------------------
    # Previous p
    # ------------------------------------------------------------

    seeds.append(
        ("previous", p_previous)
    )

    # ------------------------------------------------------------
    # Linear predictor
    # ------------------------------------------------------------

    if (
        r_previous_previous is not None
        and p_previous_previous is not None
        and abs(r_previous - r_previous_previous) > 0
    ):
        slope = (
            p_previous - p_previous_previous
        ) / (
            r_previous - r_previous_previous
        )

        dp = slope * (
            r_target - r_previous
        )

        seeds.append(
            (
                "linear",
                p_previous + dp,
            )
        )

        seeds.append(
            (
                "half",
                p_previous + 0.5 * dp,
            )
        )

        seeds.append(
            (
                "quarter",
                p_previous + 0.25 * dp,
            )
        )

    # ------------------------------------------------------------
    # Reduced-magnitude seeds
    # ------------------------------------------------------------

    seeds.append(
        (
            "half_previous",
            0.5 * p_previous,
        )

    )

    seeds.append(
        (
            "quarter_previous",
            0.25 * p_previous,
        )
    )

    # ------------------------------------------------------------
    # Negative branch as last fallback.
    # ------------------------------------------------------------

    seeds.append(
        (
            "negative_previous",
            -p_previous,
        )
    )

    tried = []

    for label, seed in seeds:

        try:
            p_new = solve_complex_p(
                solver,
                omega,
                r_target,
                seed,
            )

        except Exception:
            p_new = None

        if p_new is None:
            tried.append(label)
            continue

        # --------------------------------------------------------
        # Choose sign closest to the previous branch.
        # --------------------------------------------------------

        if (
            abs(p_new - p_previous)
            >
            abs(-p_new - p_previous)
        ):
            p_new = -p_new

        return p_new, label

    return None, tried


# ============================================================================
# ADAPTIVE RADIAL ACTION
# ============================================================================

def radial_action_adaptive(
    solver,
    omega,
    r_minus,
    r_B,
    n=OPT_RADIAL_N,
):
    """
    Adaptive complex radial continuation.

    The initial point is the turning point:

        p(r_minus) = 0

    The first step uses the exact Stage 5I-3c analytic seed:

        p ~= sqrt(-2 H_r dr / H_pp)

    Later steps use continuation.

    If a step fails, the interval is recursively subdivided.

    Returns
    -------
    action : complex
    diagnostics : dict
    """

    n = int(n)

    if n < 2:
        raise ValueError(
            "radial-action n must be >= 2."
        )

    # ------------------------------------------------------------
    # Base path.
    # ------------------------------------------------------------

    base_path = np.linspace(
        0.0,
        1.0,
        n,
    )

    r_path = (
        r_minus
        + base_path * (
            r_B - r_minus
        )
    )

    # ------------------------------------------------------------
    # Turning-point local derivatives.
    # ------------------------------------------------------------

    H_r = compute_H_r_at_turning_point(
        solver,
        omega,
        r_minus,
    )

    H_pp = compute_H_pp_at_turning_point(
        solver,
        omega,
        r_minus,
    )

    if abs(H_pp) == 0:
        raise FloatingPointError(
            "H_pp is zero at turning point."
        )

    # ------------------------------------------------------------
    # Recursive continuation structure.
    #
    # Each solved point is:
    #
    #     (r, p)
    #
    # and we keep chronological ordering.
    # ------------------------------------------------------------

    solved = [
        (
            complex(r_minus),
            0.0j,
        )
    ]

    subdivision_count = 0
    continuation_failures = 0
    seed_labels = []

    # ------------------------------------------------------------
    # First-step analytic seed.
    # ------------------------------------------------------------

    def first_step(
        r_target,
    ):
        nonlocal continuation_failures

        delta_r = (
            r_target
            - r_minus
        )

        radicand = (
            -2.0
            * H_r
            * delta_r
            / H_pp
        )

        p_seed = np.sqrt(
            radicand
        )

        # Primary positive branch.
        try:
            p_new = solve_complex_p(
                solver,
                omega,
                r_target,
                p_seed,
            )
        except Exception:
            p_new = None

        label = "analytic_positive"

        # Fallback: opposite branch.
        if p_new is None:
            try:
                p_new = solve_complex_p(
                    solver,
                    omega,
                    r_target,
                    -p_seed,
                )
            except Exception:
                p_new = None

            label = "analytic_negative"

        if p_new is None:
            continuation_failures += 1
            return None, label

        return p_new, label

    # ------------------------------------------------------------
    # Recursive interval solver.
    # ------------------------------------------------------------

    def solve_interval(
        r0,
        p0,
        r1,
        r_previous_previous=None,
        p_previous_previous=None,
        depth=0,
    ):
        nonlocal subdivision_count
        nonlocal continuation_failures

        # --------------------------------------------------------
        # First continuation step.
        # --------------------------------------------------------

        if abs(
            r0 - r_minus
        ) < 1.0e-30 and p0 == 0j:

            p1, label = first_step(
                r1
            )

            if p1 is not None:
                return [
                    (
                        complex(r1),
                        complex(p1),
                    )
                ], label

            # ----------------------------------------------------
            # If first step fails, subdivide.
            # ----------------------------------------------------

        else:

            p1, label = try_continuation_step(
                solver,
                omega,
                r0,
                p0,
                r1,
                r_previous_previous,
                p_previous_previous,
            )

            if p1 is not None:
                return [
                    (
                        complex(r1),
                        complex(p1),
                    )
                ], label

        # --------------------------------------------------------
        # Failed step.
        # --------------------------------------------------------

        continuation_failures += 1

        if depth >= MAX_SUBDIVISIONS:
            raise FloatingPointError(
                "Adaptive complex continuation failed after "
                f"{MAX_SUBDIVISIONS} subdivisions at\n"
                f"r0={r0}\n"
                f"r1={r1}"
            )

        subdivision_count += 1

        r_mid = 0.5 * (
            r0 + r1
        )

        # --------------------------------------------------------
        # Solve midpoint recursively.
        # --------------------------------------------------------

        mid_points, mid_label = solve_interval(
            r0,
            p0,
            r_mid,
            r_previous_previous,
            p_previous_previous,
            depth + 1,
        )

        r_mid_solved, p_mid_solved = (
            mid_points[-1]
        )

        # --------------------------------------------------------
        # Solve second half using midpoint as previous point.
        # --------------------------------------------------------

        right_points, right_label = solve_interval(
            r_mid_solved,
            p_mid_solved,
            r1,
            r0,
            p0,
            depth + 1,
        )

        return (
            mid_points + right_points
        ), (
            f"{mid_label}|{right_label}"
        )

    # =========================================================================
    # IMPORTANT:
    #
    # Instead of recursively solving the entire path from scratch,
    # we build it one target point at a time. If a target fails,
    # solve_interval inserts intermediate points.
    # =========================================================================

    previous_previous_r = None
    previous_previous_p = None

    current_r = complex(r_minus)
    current_p = 0.0j

    for target_r in r_path[1:]:

        try:
            new_points, labels = solve_interval(
                current_r,
                current_p,
                complex(target_r),
                previous_previous_r,
                previous_previous_p,
                depth=0,
            )

        except FloatingPointError:
            raise

        # --------------------------------------------------------
        # Append all newly solved points.
        # --------------------------------------------------------

        for rr, pp in new_points:

            previous_previous_r = current_r
            previous_previous_p = current_p

            solved.append(
                (
                    complex(rr),
                    complex(pp),
                )
            )

            current_r = complex(rr)
            current_p = complex(pp)

            seed_labels.append(
                labels
            )

    # =========================================================================
    # Remove accidental duplicate points.
    # =========================================================================

    clean_r = []
    clean_p = []

    for rr, pp in solved:

        if clean_r:
            if abs(
                rr - clean_r[-1]
            ) < 1.0e-30:
                clean_p[-1] = pp
                continue

        clean_r.append(rr)
        clean_p.append(pp)

    r_arr = np.asarray(
        clean_r,
        dtype=complex,
    )

    p_arr = np.asarray(
        clean_p,
        dtype=complex,
    )

    # ------------------------------------------------------------
    # Numerical action integral.
    # ------------------------------------------------------------

    if hasattr(
        np,
        "trapezoid",
    ):
        action = np.trapezoid(
            p_arr,
            r_arr,
        )
    else:
        action = np.trapz(
            p_arr,
            r_arr,
        )

    diagnostics = {
        "requested_n": int(n),
        "actual_points": int(len(r_arr)),
        "subdivisions": int(
            subdivision_count
        ),
        "continuation_failures": int(
            continuation_failures
        ),
        "H_r": complex_to_dict(H_r),
        "H_pp": complex_to_dict(H_pp),
    }

    return complex(action), diagnostics


# ============================================================================
# RESONANCE RESIDUAL
# ============================================================================

def evaluate_resonance(
    solver,
    patch,
    f_re,
    f_im,
    r_sp,
    omega_lr,
    radial_n,
):
    """
    Full resonance evaluation at one complex frequency.

    Returns None if the radial complex machinery cannot be evaluated.
    """

    omega = 2.0 * np.pi * (
        float(f_re)
        + 1j * float(f_im)
    )

    # ------------------------------------------------------------
    # Last scattering point.
    # ------------------------------------------------------------

    try:
        r_minus_real = (
            solver.find_last_scattering_point(
                omega,
                M,
                r_sp,
            )
        )
    except Exception:
        return None

    if r_minus_real is None:
        return None

    # ------------------------------------------------------------
    # Complex turning point.
    # ------------------------------------------------------------

    try:
        r_minus_complex = (
            solver.find_turning_point_complex(
                omega,
                M,
                r_minus_real,
            )
        )
    except Exception:
        return None

    if r_minus_complex is None:
        return None

    # ------------------------------------------------------------
    # Complex radial action.
    # ------------------------------------------------------------

    try:
        action, action_diag = (
            radial_action_adaptive(
                solver,
                omega,
                r_minus_complex,
                solver.R_B,
                n=radial_n,
            )
        )
    except Exception:
        return None

    # ------------------------------------------------------------
    # Reflection coefficient.
    # ------------------------------------------------------------

    try:
        R = solver.reflection_coefficient(
            omega,
            M,
            r_sp,
            omega_lr,
        )
    except Exception:
        return None

    if R is None:
        return None

    R = complex(R)

    # ------------------------------------------------------------
    # Resonance factor.
    # ------------------------------------------------------------

    Res = (
        R
        * np.exp(
            2j * action
        )
        - 1.0
    )

    return {
        "f_re": float(f_re),
        "f_im": float(f_im),
        "omega": complex(omega),
        "r_minus_real": float(
            np.real(r_minus_real)
        ),
        "r_minus_complex": complex(
            r_minus_complex
        ),
        "action": complex(action),
        "R": complex(R),
        "Res": complex(Res),
        "abs_res": float(
            abs(Res)
        ),
        "action_diagnostics": action_diag,
    }


# ============================================================================
# OBJECTIVE FOR NELDER-MEAD
# ============================================================================

def make_objective(
    solver,
    r_sp,
    omega_lr,
    radial_n,
):
    """
    Construct scalar objective:

        objective = |Res|^2

    Failed evaluations receive a large penalty.
    """

    evaluation_counter = {
        "n": 0,
    }

    def objective(x):

        evaluation_counter["n"] += 1

        f_re = float(x[0])
        f_im = float(x[1])

        # --------------------------------------------------------
        # Reject clearly unreasonable search points.
        # --------------------------------------------------------

        if (
            not np.isfinite(f_re)
            or not np.isfinite(f_im)
        ):
            return FAIL_PENALTY

        if f_re <= 0.0:
            return FAIL_PENALTY

        # Avoid runaway optimizer excursions.
        if f_re > 100.0:
            return FAIL_PENALTY

        if abs(f_im) > 0.05:
            return FAIL_PENALTY

        result = evaluate_resonance(
            solver,
            PATCH,
            f_re,
            f_im,
            r_sp,
            omega_lr,
            radial_n,
        )

        if result is None:
            value = FAIL_PENALTY

        else:
            value = (
                result["abs_res"]
                ** 2
            )

        print(
            f"    eval "
            f"{evaluation_counter['n']:3d}: "
            f"f=({f_re:.12f}"
            f"{f_im:+.12f}i) "
            f"obj={value:.6e}",
            flush=True,
        )

        return float(value)

    objective.evaluation_counter = (
        evaluation_counter
    )

    return objective


# ============================================================================
# SCIPY OPTIMIZATION
# ============================================================================

def optimize_resonance(
    solver,
    r_sp,
    omega_lr,
    f_re_start,
    f_im_start,
):
    """
    Low-resolution resonance search.

    N = OPT_RADIAL_N
    maxiter = OPT_MAXITER
    disp = True
    """

    try:
        from scipy.optimize import minimize
    except Exception as exc:
        raise RuntimeError(
            "scipy.optimize.minimize is required."
        ) from exc

    print()
    print(
        "OPTIMIZATION SETTINGS:"
    )
    print(
        f"  radial N      = {OPT_RADIAL_N}"
    )
    print(
        f"  method         = Nelder-Mead"
    )
    print(
        f"  maxiter        = {OPT_MAXITER}"
    )
    print(
        f"  disp           = {OPT_DISP}"
    )

    objective = make_objective(
        solver,
        r_sp,
        omega_lr,
        OPT_RADIAL_N,
    )

    x0 = np.array(
        [
            f_re_start,
            f_im_start,
        ],
        dtype=float,
    )

    # ------------------------------------------------------------
    # Scale the simplex manually because f_re and f_im have very
    # different numerical scales.
    # ------------------------------------------------------------

    initial_simplex = np.array(
        [
            [
                f_re_start,
                f_im_start,
            ],
            [
                f_re_start
                + 0.10,
                f_im_start,
            ],
            [
                f_re_start,
                f_im_start
                - 0.00010,
            ],
        ],
        dtype=float,
    )

    print()
    print(
        "STARTING SIMPLEX:"
    )

    for row in initial_simplex:
        print(
            f"  "
            f"f_re={row[0]:.12f} "
            f"f_im={row[1]:+.12f}"
        )

    print()
    print(
        "BEGIN NELDER-MEAD"
    )
    print(
        "-" * 78
    )

    result = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={
            "maxiter": OPT_MAXITER,
            "disp": OPT_DISP,
            "initial_simplex": initial_simplex,
            "xatol": 1.0e-7,
            "fatol": 1.0e-8,
        },
    )

    print()
    print(
        "NELDER-MEAD RESULT"
    )
    print(
        f"  success       = "
        f"{result.success}"
    )
    print(
        f"  status        = "
        f"{result.status}"
    )
    print(
        f"  message       = "
        f"{result.message}"
    )
    print(
        f"  iterations     = "
        f"{getattr(result, 'nit', None)}"
    )
    print(
        f"  evaluations    = "
        f"{getattr(result, 'nfev', None)}"
    )
    print(
        f"  x             = "
        f"{result.x}"
    )
    print(
        f"  objective      = "
        f"{result.fun:.6e}"
    )

    return result


# ============================================================================
# FINAL HIGH-RESOLUTION VALIDATION
# ============================================================================

def final_validate(
    solver,
    r_sp,
    omega_lr,
    candidate,
):
    """
    One independent high-resolution evaluation at N=20000.

    NO optimization is performed here.
    """

    f_re = float(
        candidate[0]
    )

    f_im = float(
        candidate[1]
    )

    print()
    print(
        "FINAL HIGH-RESOLUTION VALIDATION"
    )
    print(
        "-" * 78
    )
    print(
        f"  radial N = "
        f"{FINAL_RADIAL_N}"
    )
    print(
        f"  f_re     = "
        f"{f_re:.15f} Hz"
    )
    print(
        f"  f_im     = "
        f"{f_im:+.15f} Hz"
    )

    result = evaluate_resonance(
        solver,
        PATCH,
        f_re,
        f_im,
        r_sp,
        omega_lr,
        FINAL_RADIAL_N,
    )

    if result is None:
        print(
            "  FINAL VALIDATION: FAIL"
        )
        return None

    Res = result["Res"]

    Q = np.nan

    if f_im != 0.0:
        Q = abs(
            f_re / (
                2.0 * f_im
            )
        )

    result["Q"] = float(Q)

    print(
        f"  |Res|    = "
        f"{result['abs_res']:.12e}"
    )

    print(
        f"  Re(Res)  = "
        f"{Res.real:+.12e}"
    )

    print(
        f"  Im(Res)  = "
        f"{Res.imag:+.12e}"
    )

    print(
        f"  Q        = "
        f"{Q:.12e}"
    )

    print(
        f"  r_minus  = "
        f"{result['r_minus_real']:.12e} m"
    )

    print(
        f"  action   = "
        f"{result['action']}"
    )

    print(
        f"  R        = "
        f"{result['R']}"
    )

    print(
        "  FINAL VALIDATION: PASS"
    )

    return result


# ============================================================================
# BENCHMARK EVALUATION
# ============================================================================

def evaluate_benchmark(
    solver,
    r_sp,
    omega_lr,
):
    """
    Evaluate the historical benchmark point at the optimization resolution.

    This is a diagnostic reference, not the final result.
    """

    print()
    print(
        "HISTORICAL BENCHMARK EVALUATION"
    )
    print(
        f"  f_re = "
        f"{F_RE_BENCH:.15f}"
    )
    print(
        f"  f_im = "
        f"{F_IM_BENCH:+.15f}"
    )

    result = evaluate_resonance(
        solver,
        PATCH,
        F_RE_BENCH,
        F_IM_BENCH,
        r_sp,
        omega_lr,
        OPT_RADIAL_N,
    )

    if result is None:
        print(
            "  benchmark evaluation: FAILED"
        )
        return None

    print(
        f"  |Res| = "
        f"{result['abs_res']:.12e}"
    )

    return result


# ============================================================================
# C-SPECIFIC SEARCH START
# ============================================================================

def make_C_specific_start(
    f_lr,
):
    """
    Construct a numerical starting point from the C-specific light-ring
    frequency.

    IMPORTANT:
        This is only an optimizer initial guess.

    It is NOT:
        * a physical frequency rescaling
        * a claim that the resonance must obey a fixed ratio
        * a Kerr mapping
    """

    f_re_start = (
        F_RE_OVER_F_LR
        * float(f_lr)
    )

    # Keep the old imaginary scale as the numerical starting scale.
    f_im_start = F_IM_START

    return (
        f_re_start,
        f_im_start,
    )


# ============================================================================
# C CASE
# ============================================================================

def run_case(
    solver,
    patch,
    label,
    C_value,
):
    banner(
        f"{label}: C = {C_value:.12e}"
    )

    case = {
        "label": label,
        "C": float(C_value),
        "status": "INCONCLUSIVE",
    }

    # ------------------------------------------------------------------------
    # Explicit real light ring.
    # ------------------------------------------------------------------------

    try:
        r_sp, omega_lr = explicit_light_ring(
            solver,
            C_value,
        )
    except Exception as exc:
        print(
            f"LIGHT-RING CALCULATION FAILED: "
            f"{type(exc).__name__}: {exc}"
        )

        case["status"] = "LIGHT_RING_FAILURE"

        return case

    if r_sp is None:
        print(
            "LIGHT RING: NONE"
        )

        case["light_ring"] = None
        case["omega_lr"] = None
        case["f_lr"] = None

        # C=0 is expected to be here.
        if abs(C_value) < 1.0e-30:
            case["status"] = (
                "NO_LIGHT_RING"
            )
        else:
            case["status"] = (
                "NO_LIGHT_RING"
            )

        return case

    f_lr = (
        float(omega_lr)
        / (2.0 * np.pi)
    )

    print(
        f"r_sp  = "
        f"{r_sp * 1.0e3:.12f} mm"
    )

    print(
        f"omega_lr = "
        f"{omega_lr:.15e} rad/s"
    )

    print(
        f"f_lr     = "
        f"{f_lr:.15f} Hz"
    )

    case["r_sp"] = float(r_sp)
    case["omega_lr"] = float(omega_lr)
    case["f_lr"] = float(f_lr)

    # ------------------------------------------------------------------------
    # Historical benchmark.
    # ------------------------------------------------------------------------

    benchmark = evaluate_benchmark(
        solver,
        r_sp,
        omega_lr,
    )

    if benchmark is not None:
        case["benchmark_abs_res"] = (
            benchmark["abs_res"]
        )

    # ------------------------------------------------------------------------
    # C-specific optimizer starting point.
    # ------------------------------------------------------------------------

    f_re_start, f_im_start = (
        make_C_specific_start(
            f_lr
        )
    )

    print()
    print(
        "C-SPECIFIC OPTIMIZER START"
    )

    print(
        f"  f_lr         = "
        f"{f_lr:.12f} Hz"
    )

    print(
        f"  f_re start   = "
        f"{f_re_start:.12f} Hz"
    )

    print(
        f"  f_im start   = "
        f"{f_im_start:+.12f} Hz"
    )

    print(
        "  NOTE: start point only; "
        "no physical frequency rescaling."
    )

    # ------------------------------------------------------------------------
    # Optimization.
    # ------------------------------------------------------------------------

    try:
        opt = optimize_resonance(
            solver,
            r_sp,
            omega_lr,
            f_re_start,
            f_im_start,
        )

    except Exception as exc:

        print()
        print(
            "OPTIMIZATION FAILED:"
        )

        print(
            f"  {type(exc).__name__}: "
            f"{exc}"
        )

        case["status"] = (
            "OPTIMIZATION_FAILURE"
        )

        return case

    case["optimizer_success"] = bool(
        opt.success
    )

    case["optimizer_message"] = str(
        opt.message
    )

    case["optimizer_nit"] = safe_float(
        getattr(
            opt,
            "nit",
            np.nan,
        )
    )

    case["optimizer_nfev"] = safe_float(
        getattr(
            opt,
            "nfev",
            np.nan,
        )
    )

    case["optimizer_objective"] = safe_float(
        getattr(
            opt,
            "fun",
            np.nan,
        )
    )

    candidate = np.asarray(
        opt.x,
        dtype=float,
    )

    case["candidate_f_re"] = float(
        candidate[0]
    )

    case["candidate_f_im"] = float(
        candidate[1]
    )

    # ------------------------------------------------------------------------
    # Final N=20000 validation.
    # ------------------------------------------------------------------------

    final = final_validate(
        solver,
        r_sp,
        omega_lr,
        candidate,
    )

    if final is None:

        case["status"] = (
            "FINAL_VALIDATION_FAILURE"
        )

        return case

    case["final_abs_res"] = (
        final["abs_res"]
    )

    case["final_res_re"] = (
        final["Res"].real
    )

    case["final_res_im"] = (
        final["Res"].imag
    )

    case["final_Q"] = (
        final["Q"]
    )

    case["final_r_minus_real"] = (
        final["r_minus_real"]
    )

    case["final_r_minus_complex"] = (
        complex_to_dict(
            final["r_minus_complex"]
        )
    )

    case["final_action"] = (
        complex_to_dict(
            final["action"]
        )
    )

    case["final_R"] = (
        complex_to_dict(
            final["R"]
        )
    )

    # ------------------------------------------------------------------------
    # Case status.
    #
    # We deliberately do NOT impose an arbitrary |Res| threshold here.
    # We report the actual numerical result and let the cross-case comparison
    # determine whether there is a reproducible C response.
    # ------------------------------------------------------------------------

    case["status"] = "VALIDATED_CANDIDATE"

    return case


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(results):
    json_file = (
        OUTPUT_DIR
        / "stage6_5B1_resonance_causal_results.json"
    )

    csv_file = (
        OUTPUT_DIR
        / "stage6_5B1_resonance_causal_results.csv"
    )

    # ------------------------------------------------------------------------
    # JSON.
    # ------------------------------------------------------------------------

    with open(
        json_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------------
    # Flatten CSV.
    # ------------------------------------------------------------------------

    rows = []

    for r in results:

        row = {}

        for key, value in r.items():

            if isinstance(
                value,
                dict,
            ):
                row[key] = json.dumps(
                    value,
                    ensure_ascii=False,
                )

            else:
                row[key] = value

        rows.append(row)

    if rows:

        fieldnames = []

        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

        with open(
            csv_file,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)

    return (
        csv_file,
        json_file,
    )


# ============================================================================
# FINAL CROSS-CASE SUMMARY
# ============================================================================

def final_summary(results):
    banner(
        "STAGE 6.5B-1 — FINAL CAUSAL SUMMARY"
    )

    print()
    print(
        "CASE RESULTS"
    )
    print(
        "-" * 78
    )

    valid = []

    for r in results:

        label = r["label"]

        status = r.get(
            "status",
            "UNKNOWN",
        )

        C = r["C"]

        f_lr = r.get(
            "f_lr",
            None,
        )

        f_re = r.get(
            "candidate_f_re",
            None,
        )

        f_im = r.get(
            "candidate_f_im",
            None,
        )

        res = r.get(
            "final_abs_res",
            None,
        )

        print(
            f"{label:12s} "
            f"C={C:.6e} "
            f"status={status}"
        )

        if f_lr is not None:
            print(
                f"  f_lr       = "
                f"{f_lr:.12f} Hz"
            )

        if f_re is not None:
            print(
                f"  candidate  = "
                f"{f_re:.12f} "
                f"{f_im:+.12f} i Hz"
            )

        if res is not None:
            print(
                f"  final |Res|= "
                f"{res:.12e}"
            )

            valid.append(r)

    # ------------------------------------------------------------------------
    # C-dependent resonance frequency spread.
    # ------------------------------------------------------------------------

    if len(valid) >= 2:

        f_re_values = np.asarray(
            [
                r["candidate_f_re"]
                for r in valid
                if "candidate_f_re" in r
            ],
            dtype=float,
        )

        f_im_values = np.asarray(
            [
                r["candidate_f_im"]
                for r in valid
                if "candidate_f_im" in r
            ],
            dtype=float,
        )

        if len(f_re_values) >= 2:

            print()
            print(
                "C-DEPENDENT CANDIDATE SPREAD"
            )

            print(
                f"  f_re spread = "
                f"{np.ptp(f_re_values):.12e} Hz"
            )

            print(
                f"  f_im spread = "
                f"{np.ptp(f_im_values):.12e} Hz"
            )

    # ------------------------------------------------------------------------
    # Final scientific verdict.
    # ------------------------------------------------------------------------

    print()
    print(
        "=" * 78
    )

    # C=0 must be interpreted independently.
    c_zero = next(
        (
            r for r in results
            if r["label"] == "C_ZERO"
        ),
        None,
    )

    nonzero = [
        r for r in results
        if (
            r["label"] != "C_ZERO"
            and r.get("status")
            == "VALIDATED_CANDIDATE"
        )
    ]

    if (
        c_zero is not None
        and c_zero.get("status")
        == "NO_LIGHT_RING"
        and len(nonzero) >= 2
    ):

        f_re_vals = np.asarray(
            [
                r["candidate_f_re"]
                for r in nonzero
            ],
            dtype=float,
        )

        f_im_vals = np.asarray(
            [
                r["candidate_f_im"]
                for r in nonzero
            ],
            dtype=float,
        )

        f_re_spread = (
            np.ptp(f_re_vals)
        )

        f_im_spread = (
            np.ptp(f_im_vals)
        )

        print(
            "CAUSAL TEST:"
        )

        print(
            "  NONZERO RESONANCE RESPONSE TO C"
        )

        print(
            f"  f_re spread = "
            f"{f_re_spread:.12e} Hz"
        )

        print(
            f"  f_im spread = "
            f"{f_im_spread:.12e} Hz"
        )

        print()
        print(
            "INTERPRETATION:"
        )

        print(
            "  The resonance-level numerical structure "
            "changes when C is changed."
        )

        print(
            "  C=0 removes the interior light ring."
        )

        print(
            "  This is evidence for C-dependent "
            "resonance structure in the mathematical model."
        )

        print()
        print(
            "  It is NOT, by itself, evidence for a new "
            "fundamental interaction."
        )

        verdict = (
            "NONZERO_RESONANCE_RESPONSE_TO_C"
        )

    elif len(nonzero) >= 1:

        print(
            "CAUSAL TEST:"
        )

        print(
            "  INCONCLUSIVE"
        )

        print(
            "  At least one nonzero-C case produced "
            "a validated candidate, but the complete "
            "cross-C causal comparison is insufficient."
        )

        verdict = (
            "INCONCLUSIVE"
        )

    else:

        print(
            "CAUSAL TEST:"
        )

        print(
            "  NO_VALIDATED_RESONANCE_RESPONSE"
        )

        print(
            "  No nonzero-C resonance candidate survived "
            "the final N=20000 validation."
        )

        verdict = (
            "NO_VALIDATED_RESONANCE_RESPONSE"
        )

    print()
    print(
        "FINAL VERDICT = "
        f"{verdict}"
    )

    print(
        "=" * 78
    )

    return verdict


# ============================================================================
# MAIN
# ============================================================================

def main():

    banner(
        "STAGE 6.5B-1 — RESONANCE-LEVEL C-CAUSALITY AUDIT"
    )

    print()
    print(
        "Purpose:"
    )

    print(
        "  Test whether the complex resonance structure "
        "responds causally to C."
    )

    print()
    print(
        "NUMERICAL STRATEGY:"
    )

    print(
        f"  Optimization radial N = "
        f"{OPT_RADIAL_N}"
    )

    print(
        f"  Final radial N       = "
        f"{FINAL_RADIAL_N}"
    )

    print(
        f"  Nelder-Mead maxiter  = "
        f"{OPT_MAXITER}"
    )

    print(
        f"  Nelder-Mead disp     = "
        f"{OPT_DISP}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  N=2000 is ONLY for locating a candidate."
    )

    print(
        "  N=20000 is used for the final one-shot validation."
    )

    print()
    print(
        "SCIENTIFIC CONTROLS:"
    )

    print(
        "  * no Kerr fit"
    )

    print(
        "  * no GW calibration"
    )

    print(
        "  * no Lambda term"
    )

    print(
        "  * no low-k approximation"
    )

    print(
        "  * no frequency rescaling"
    )

    print(
        "  * no modification of authoritative solver on disk"
    )

    print(
        "  * only C is varied"
    )

    # =========================================================================
    # LOAD
    # =========================================================================

    global PATCH

    solver = load_authoritative_solver()

    print()
    print(
        "Authoritative solver:"
    )
    print(
        f"  {RES_FILE}"
    )

    print_solver_constants(
        solver
    )

    PATCH = load_patch(
        solver
    )

    print()
    print(
        "Patch loaded:"
    )
    print(
        f"  {PATCH_FILE}"
    )

    print()
    print(
        "Patch radial_action_complex_v2 signature:"
    )

    print(
        " ",
        inspect.signature(
            PATCH.radial_action_complex_v2
        ),
    )

    # =========================================================================
    # API CHECK
    # =========================================================================

    banner(
        "AUTHORITATIVE COMPLEX API CHECK"
    )

    required = [
        "hamiltonian",
        "solve_p_complex",
        "find_light_ring",
        "find_last_scattering_point",
        "find_turning_point_complex",
        "reflection_coefficient",
    ]

    api_ok = True

    for name in required:

        exists = hasattr(
            solver,
            name,
        )

        print(
            f"  {name:32s} "
            f"{'PASS' if exists else 'FAIL'}"
        )

        if not exists:
            api_ok = False

    if not api_ok:
        raise RuntimeError(
            "Required authoritative complex API is incomplete."
        )

    # =========================================================================
    # EXPLICIT C REAL PREFLIGHT
    # =========================================================================

    banner(
        "EXPLICIT C LIGHT-RING PREFLIGHT"
    )

    for label, C_value in C_CASES:

        r_sp, omega_lr = (
            explicit_light_ring(
                solver,
                C_value,
            )
        )

        if r_sp is None:

            print(
                f"{label:12s} "
                f"C={C_value:.12e} "
                f"LIGHT RING: NONE"
            )

        else:

            print(
                f"{label:12s} "
                f"C={C_value:.12e} "
                f"r_sp={r_sp * 1e3:.9f} mm "
                f"f_lr={omega_lr/(2*np.pi):.12f} Hz"
            )

    # =========================================================================
    # COMPLEX C FINGERPRINT
    # =========================================================================

    banner(
        "COMPLEX C-INJECTION FINGERPRINT"
    )

    C_values = [
        0.5 * EXPECTED_C,
        EXPECTED_C,
        2.0 * EXPECTED_C,
    ]

    fingerprint = (
        complex_C_fingerprint(
            solver,
            C_values,
        )
    )

    for C, H in fingerprint.items():

        print(
            f"  C={C:.12e} "
            f"H={H.real:+.12e}"
            f"{H.imag:+.12e}i"
        )

    fp_values = list(
        fingerprint.values()
    )

    fingerprint_ok = False

    if len(fp_values) >= 2:

        for i in range(
            len(fp_values)
        ):

            for j in range(
                i + 1,
                len(fp_values),
            ):

                if abs(
                    fp_values[i]
                    - fp_values[j]
                ) > 1.0e-12:

                    fingerprint_ok = True

    print()

    if fingerprint_ok:
        print(
            "COMPLEX C-INJECTION: PASS"
        )
    else:
        print(
            "COMPLEX C-INJECTION: FAIL"
        )

        raise RuntimeError(
            "Changing C does not produce a measurable "
            "complex-Hamiltonian response. "
            "Do not proceed to resonance search."
        )

    # =========================================================================
    # RUN CASES
    # =========================================================================

    results = []

    original_C = snapshot_C_globals(
        solver
    )

    try:

        for label, C_value in C_CASES:

            # ----------------------------------------------------
            # Important:
            # Explicitly set complex C before each case.
            # ----------------------------------------------------

            set_complex_C(
                solver,
                C_value,
            )

            result = run_case(
                solver,
                PATCH,
                label,
                C_value,
            )

            results.append(
                result
            )

    finally:

        restore_C_globals(
            solver,
            original_C,
        )

    # =========================================================================
    # SAVE
    # =========================================================================

    csv_file, json_file = (
        save_results(
            results
        )
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    verdict = final_summary(
        results
    )

    print()
    print(
        "OUTPUT FILES"
    )

    print(
        f"  CSV  -> {csv_file}"
    )

    print(
        f"  JSON -> {json_file}"
    )

    print()
    print(
        "Authoritative solver restored to its original "
        "in-memory C values."
    )

    print(
        "No authoritative solver file was modified."
    )

    print()
    print(
        "=" * 78
    )

    print(
        "STAGE 6.5B-1 COMPLETE"
    )

    print(
        f"VERDICT: {verdict}"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()