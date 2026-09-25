#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 6.5B4-1A
MICRO C-CONTINUATION OF THE VALIDATED BASELINE RESONANCE

Purpose
-------
Test local continuity of the complex resonance branch around the validated
baseline C0 using very small C steps:

    downward: C0 -> 0.995 C0 -> 0.990 C0
    upward:   C0 -> 1.005 C0 -> 1.010 C0

This is NOT a blind resonance scan.

Rules
-----
1. Authoritative stage5I_3_resonance.py is read-only.
2. C is changed only in memory.
3. Real light ring is evaluated with explicit C injection.
4. Complex machinery uses the same in-memory C.
5. Each new point is seeded from the immediately preceding accepted root.
6. No Kerr mapping.
7. No GW calibration.
8. No frequency rescaling.
9. No long-wave approximation.
10. Do not force continuation toward C=0, because C=0 has no interior
    light ring.

Interpretation
--------------
Success means:
    a continuous local resonance branch can be numerically followed.

Failure means:
    this continuation procedure cannot establish the branch locally.

Failure is NOT interpreted as:
    "C has no resonance effect."

The clean causal observable is the reproducible local response

    df_R/dC
    df_I/dC

together with residual control and branch continuity.

Addition (this version)
------------------------
In addition to the continuation chain, this version independently
re-solves for the resonance root directly AT C0 (rather than only
trusting the supplied B4-0 baseline value). This provides:

  1. An independent cross-check of the B4-0 baseline.
  2. A genuinely root-solved point at C0 with residual comparable to
     the neighboring continuation points, instead of the looser
     supplied-baseline residual.
  3. Two clean derivative intervals that straddle C0 symmetrically
     (0.995 C0 -> C0) and (C0 -> 1.005 C0), instead of skipping over it.
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import root


# ============================================================================
# PATHS
# ============================================================================

HERE = Path(__file__).resolve().parent

RES_FILE = HERE / "stage5I_3_resonance.py"
PATCH_FILE = HERE / "stage5I_3c_radial_action_patch.py"

OUT_DIR = HERE / "stage6_5B4_1A_micro_continuation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = OUT_DIR / "stage6_5B4_1A_micro_results.csv"
JSON_FILE = OUT_DIR / "stage6_5B4_1A_micro_results.json"


# ============================================================================
# AUTHORITATIVE CONSTANTS
# ============================================================================

C0 = 7.690000000000e-04
OMEGA = 1.000000000000e-01
GAMMA = 2.220000000000e-06
H0 = 3.400000000000e-02
G_GRAV = 9.810000000000e+00
R_B = 3.730000000000e-02

M_MODE = -12

# Validated B4-0 / B3 baseline resonance.
BASELINE_F_RE = 8.306625532027001
BASELINE_F_IM = -0.000586962309000

# The authoritative complex machinery is configured to this value in memory.
RADIAL_N = 1000


# ============================================================================
# MICRO-CONTINUATION CONFIGURATION
# ============================================================================

C_FACTORS_DOWN = [
    0.995,
    0.990,
]

C_FACTORS_UP = [
    1.005,
    1.010,
]

# Root solver.
ROOT_METHOD = "hybr"
ROOT_MAXFEV = 60
ROOT_TOL = 1.0e-9

# Branch continuity guards.
#
# These are deliberately loose enough not to reject a genuine small
# continuous movement, while preventing an accidental jump to a distant
# branch.
MAX_JUMP_RE_HZ = 0.25
MAX_JUMP_IM_HZ = 0.0010

# Candidate fallback seeds around the previous accepted root.
#
# The first seed is always the previous accepted root.
FALLBACK_RE_OFFSETS = [
    0.0,
    +0.01,
    -0.01,
]

FALLBACK_IM_OFFSETS = [
    0.0,
    +5.0e-5,
    -5.0e-5,
]


# ============================================================================
# UTILITIES
# ============================================================================

def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fmt_complex(z: complex) -> str:
    return f"{z.real:+.12e}{z.imag:+.12e}i"


def safe_float(x):
    if x is None:
        return None
    try:
        x = float(x)
        if not np.isfinite(x):
            return None
        return x
    except Exception:
        return None


# ============================================================================
# LOAD AUTHORITATIVE SOLVER
# ============================================================================

def load_authoritative_solver():
    if not RES_FILE.exists():
        raise FileNotFoundError(
            f"Authoritative solver not found:\n{RES_FILE}"
        )

    spec = importlib.util.spec_from_file_location(
        "stage5I_3_resonance_authoritative",
        RES_FILE,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Could not construct solver import spec.")

    solver = importlib.util.module_from_spec(spec)

    # Important:
    # The authoritative file expects __file__ to exist.
    solver.__file__ = str(RES_FILE)
    solver.__name__ = "stage5I_3_resonance_authoritative"

    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))

    spec.loader.exec_module(solver)

    # ------------------------------------------------------------------
    # Compatibility shim for the radial-action patch.
    #
    # The patch does:
    #     from stage5I_3_resonance import DEFAULT_COMPLEX_RADIAL_N
    #
    # so the symbol must be (a) present on this module object, and
    # (b) the module must be registered in sys.modules under the exact
    # name "stage5I_3_resonance" -- otherwise Python will try to import
    # a separate file under that name (or fail) instead of reusing this
    # already-loaded, in-memory-patched object.
    # ------------------------------------------------------------------

    if not hasattr(solver, "DEFAULT_COMPLEX_RADIAL_N"):
        solver.DEFAULT_COMPLEX_RADIAL_N = RADIAL_N

    sys.modules["stage5I_3_resonance"] = solver

    return solver


# ============================================================================
# LOAD PATCH
# ============================================================================

def load_radial_patch():
    if not PATCH_FILE.exists():
        raise FileNotFoundError(
            f"Radial-action patch not found:\n{PATCH_FILE}"
        )

    spec = importlib.util.spec_from_file_location(
        "stage5I_3c_radial_action_patch_runtime",
        PATCH_FILE,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Could not construct patch import spec.")

    patch = importlib.util.module_from_spec(spec)

    patch.__file__ = str(PATCH_FILE)
    patch.__name__ = "stage5I_3c_radial_action_patch_runtime"

    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))

    spec.loader.exec_module(patch)

    return patch


# ============================================================================
# C INJECTION
# ============================================================================

def set_complex_C(solver, C_value: float) -> None:
    """
    Change only the C-related globals that actually exist in the
    authoritative complex solver.

    This is deliberately done in memory.

    The complex API itself does not accept C explicitly, so the complex
    machinery needs this controlled runtime mutation.
    """

    changed = []

    for name in (
        "C_METHODS",
        "C_TABLE",
        "C",
    ):
        if hasattr(solver, name):
            setattr(solver, name, float(C_value))
            changed.append(name)

    if not changed:
        raise RuntimeError(
            "Could not find any authoritative C global "
            "(expected C_METHODS / C_TABLE / C)."
        )


def get_complex_C_state(solver) -> dict:
    state = {}

    for name in (
        "C_METHODS",
        "C_TABLE",
        "C",
    ):
        if hasattr(solver, name):
            state[name] = float(getattr(solver, name))

    return state


# ============================================================================
# LIGHT RING
# ============================================================================

def get_light_ring(solver, C_value: float):
    """
    Explicit-C real light-ring calculation.

    This is the already validated API from the preflight:
        find_light_ring(m, C=...)
    """

    result = solver.find_light_ring(
        M_MODE,
        C=C_value,
        Omega=OMEGA,
        gamma=GAMMA,
        h0=H0,
        g=G_GRAV,
        r_min=1.0e-3,
        r_max=R_B,
    )

    if result is None:
        return None, None

    r_sp, omega_lr = result

    if r_sp is None or omega_lr is None:
        return None, None

    return float(r_sp), float(omega_lr)


# ============================================================================
# COMPLEX RESONANCE EVALUATION
# ============================================================================

def evaluate_resonance(
    solver,
    patch,
    f_complex_hz: complex,
    C_value: float,
    r_sp: float,
    omega_lr: float,
    radial_n: int = RADIAL_N,
):
    """
    Evaluate the same resonance architecture used in B4:

        Res = R * exp(2 i S) - 1

    where S is the validated complex radial action.

    Returns a dictionary with all diagnostics.
    """

    # C must already be installed into the complex solver.
    state = get_complex_C_state(solver)

    omega = 2.0 * np.pi * f_complex_hz

    # ----------------------------------------------------------------------
    # Last scattering point
    # ----------------------------------------------------------------------

    r_minus_real = solver.find_last_scattering_point(
        omega,
        M_MODE,
        r_sp,
    )

    if r_minus_real is None:
        raise RuntimeError("Last scattering point returned None.")

    r_minus_real = float(np.real(r_minus_real))

    # ----------------------------------------------------------------------
    # Complex turning point
    # ----------------------------------------------------------------------

    r_turn = solver.find_turning_point_complex(
        omega,
        M_MODE,
        r_minus_real,
    )

    if r_turn is None:
        raise RuntimeError("Complex turning point returned None.")

    r_turn = complex(r_turn)

    # ----------------------------------------------------------------------
    # Hamiltonian residual at turning point
    # ----------------------------------------------------------------------

    H_turn = solver.hamiltonian(
        omega,
        0j,
        M_MODE,
        r_turn,
    )

    H_turn = complex(H_turn)

    # ----------------------------------------------------------------------
    # Complex radial action
    # ----------------------------------------------------------------------

    action = patch.radial_action_complex_v2(
        omega,
        M_MODE,
        r_turn,
        r_B=R_B,
        n=radial_n,
    )

    action = complex(action)

    # ----------------------------------------------------------------------
    # Reflection coefficient
    # ----------------------------------------------------------------------

    R = solver.reflection_coefficient(
        omega,
        M_MODE,
        r_sp,
        omega_lr=omega_lr,
    )

    R = complex(R)

    # ----------------------------------------------------------------------
    # Resonance residual
    # ----------------------------------------------------------------------

    Res = R * np.exp(2j * action) - 1.0
    Res = complex(Res)

    Q = None
    if f_complex_hz.imag != 0.0:
        Q = abs(f_complex_hz.real / (2.0 * f_complex_hz.imag))

    return {
        "C": float(C_value),
        "f_R_Hz": float(f_complex_hz.real),
        "f_I_Hz": float(f_complex_hz.imag),
        "Q": safe_float(Q),

        "r_sp_m": float(r_sp),
        "r_minus_real_m": float(r_minus_real),
        "r_turn_re_m": float(r_turn.real),
        "r_turn_im_m": float(r_turn.imag),

        "omega_lr_rad_s": float(omega_lr),
        "f_lr_Hz": float(omega_lr / (2.0 * np.pi)),

        "H_turn_re": float(H_turn.real),
        "H_turn_im": float(H_turn.imag),
        "H_turn_abs": float(abs(H_turn)),

        "action_re": float(action.real),
        "action_im": float(action.imag),
        "action_abs": float(abs(action)),

        "R_re": float(R.real),
        "R_im": float(R.imag),
        "R_abs": float(abs(R)),

        "Res_re": float(Res.real),
        "Res_im": float(Res.imag),
        "Res_abs": float(abs(Res)),

        "radial_n": int(radial_n),
        "complex_C_state": state,
    }


# ============================================================================
# RESIDUAL FOR ROOT SOLVER
# ============================================================================

def make_residual_function(
    solver,
    patch,
    C_value: float,
    r_sp: float,
    omega_lr: float,
):
    """
    scipy root works with two real variables:

        x[0] = f_R
        x[1] = f_I

    and solves

        Re(Res) = 0
        Im(Res) = 0
    """

    def residual(x):
        f = complex(float(x[0]), float(x[1]))

        try:
            result = evaluate_resonance(
                solver=solver,
                patch=patch,
                f_complex_hz=f,
                C_value=C_value,
                r_sp=r_sp,
                omega_lr=omega_lr,
                radial_n=RADIAL_N,
            )

            res_re = result["Res_re"]
            res_im = result["Res_im"]

            if not np.isfinite(res_re) or not np.isfinite(res_im):
                return np.array([1.0e6, 1.0e6], dtype=float)

            return np.array(
                [res_re, res_im],
                dtype=float,
            )

        except Exception:
            # Root solver should see a finite penalty rather than crashing.
            return np.array([1.0e6, 1.0e6], dtype=float)

    return residual


# ============================================================================
# SINGLE CONTINUATION STEP
# ============================================================================

def continuation_step(
    solver,
    patch,
    previous_root: complex,
    C_value: float,
):
    """
    Follow one small C step starting from the previous accepted root.

    Several very local seeds are allowed, but every accepted candidate must
    satisfy the branch-jump guards.
    """

    # ----------------------------------------------------------------------
    # Explicit real C light ring
    # ----------------------------------------------------------------------

    r_sp, omega_lr = get_light_ring(solver, C_value)

    if r_sp is None:
        return {
            "status": "NO_LIGHT_RING",
            "C": float(C_value),
        }

    f_lr = omega_lr / (2.0 * np.pi)

    # ----------------------------------------------------------------------
    # Install C into complex solver
    # ----------------------------------------------------------------------

    set_complex_C(solver, C_value)

    # ----------------------------------------------------------------------
    # Evaluate previous root before solving
    # ----------------------------------------------------------------------

    try:
        seed_eval = evaluate_resonance(
            solver=solver,
            patch=patch,
            f_complex_hz=previous_root,
            C_value=C_value,
            r_sp=r_sp,
            omega_lr=omega_lr,
            radial_n=RADIAL_N,
        )

        seed_res_abs = seed_eval["Res_abs"]

    except Exception as exc:
        seed_eval = None
        seed_res_abs = None
        seed_error = repr(exc)

    # ----------------------------------------------------------------------
    # Build local seeds
    # ----------------------------------------------------------------------

    seeds = []

    for dre in FALLBACK_RE_OFFSETS:
        for dim in FALLBACK_IM_OFFSETS:

            seed = complex(
                previous_root.real + dre,
                previous_root.imag + dim,
            )

            # Put exact previous root first.
            if abs(dre) == 0.0 and abs(dim) == 0.0:
                seeds.insert(0, seed)
            else:
                seeds.append(seed)

    best = None

    # ----------------------------------------------------------------------
    # Root attempts
    # ----------------------------------------------------------------------

    residual_function = make_residual_function(
        solver=solver,
        patch=patch,
        C_value=C_value,
        r_sp=r_sp,
        omega_lr=omega_lr,
    )

    for seed in seeds:

        try:
            sol = root(
                residual_function,
                np.array(
                    [seed.real, seed.imag],
                    dtype=float,
                ),
                method=ROOT_METHOD,
                tol=ROOT_TOL,
                options={
                    "maxfev": ROOT_MAXFEV,
                },
            )

        except Exception:
            continue

        candidate = complex(
            float(sol.x[0]),
            float(sol.x[1]),
        )

        # Evaluate candidate explicitly at the same N.
        try:
            candidate_eval = evaluate_resonance(
                solver=solver,
                patch=patch,
                f_complex_hz=candidate,
                C_value=C_value,
                r_sp=r_sp,
                omega_lr=omega_lr,
                radial_n=RADIAL_N,
            )

        except Exception:
            continue

        jump = candidate - previous_root

        jump_re = abs(jump.real)
        jump_im = abs(jump.imag)

        candidate_eval["root_success"] = bool(sol.success)
        candidate_eval["root_message"] = str(sol.message)
        candidate_eval["root_nfev"] = int(getattr(sol, "nfev", -1))

        candidate_eval["seed_f_R_Hz"] = float(seed.real)
        candidate_eval["seed_f_I_Hz"] = float(seed.imag)

        candidate_eval["delta_f_R_Hz"] = float(jump.real)
        candidate_eval["delta_f_I_Hz"] = float(jump.imag)

        candidate_eval["jump_abs_Hz"] = float(abs(jump))

        candidate_eval["jump_guard_re_pass"] = (
            jump_re <= MAX_JUMP_RE_HZ
        )

        candidate_eval["jump_guard_im_pass"] = (
            jump_im <= MAX_JUMP_IM_HZ
        )

        candidate_eval["branch_guard_pass"] = (
            candidate_eval["jump_guard_re_pass"]
            and candidate_eval["jump_guard_im_pass"]
        )

        # Root quality score:
        # residual is primary; branch jump is secondary.
        score = (
            candidate_eval["Res_abs"]
            + 0.1 * abs(jump.real)
            + 10.0 * abs(jump.imag)
        )

        candidate_eval["_score"] = float(score)

        if best is None or score < best["_score"]:
            best = candidate_eval

    # ----------------------------------------------------------------------
    # No candidate
    # ----------------------------------------------------------------------

    if best is None:
        result = {
            "status": "ROOT_FAILED",
            "C": float(C_value),
            "r_sp_m": float(r_sp),
            "f_lr_Hz": float(f_lr),
            "seed_f_R_Hz": float(previous_root.real),
            "seed_f_I_Hz": float(previous_root.imag),
            "seed_res_abs": seed_res_abs,
            "radial_n": int(RADIAL_N),
        }

        if seed_eval is None:
            result["seed_eval_error"] = seed_error

        return result

    # ----------------------------------------------------------------------
    # Branch guard
    # ----------------------------------------------------------------------

    if not best["branch_guard_pass"]:
        best["status"] = "BRANCH_GUARD_FAIL"
    elif best["Res_abs"] > 5.0e-3:
        # A candidate should be genuinely close to resonance.
        best["status"] = "RESIDUAL_TOO_LARGE"
    elif not best["root_success"]:
        # Numerical solver did not report formal convergence.
        # Keep the candidate but do NOT call it a clean continuation.
        best["status"] = "ROOT_CANDIDATE_NO_SOLVER_CONVERGENCE"
    else:
        best["status"] = "PASS"

    best.pop("_score", None)

    best["seed_res_abs"] = seed_res_abs

    if seed_eval is not None:
        best["seed_eval_f_R_Hz"] = seed_eval["f_R_Hz"]
        best["seed_eval_f_I_Hz"] = seed_eval["f_I_Hz"]
        best["seed_eval_res_re"] = seed_eval["Res_re"]
        best["seed_eval_res_im"] = seed_eval["Res_im"]

    return best


# ============================================================================
# BASELINE
# ============================================================================

def evaluate_baseline(solver, patch):
    banner("BASELINE REPRODUCTION")

    r_sp, omega_lr = get_light_ring(solver, C0)

    if r_sp is None:
        raise RuntimeError("Baseline light ring unexpectedly absent.")

    set_complex_C(solver, C0)

    baseline_root = complex(
        BASELINE_F_RE,
        BASELINE_F_IM,
    )

    result = evaluate_resonance(
        solver=solver,
        patch=patch,
        f_complex_hz=baseline_root,
        C_value=C0,
        r_sp=r_sp,
        omega_lr=omega_lr,
        radial_n=RADIAL_N,
    )

    result["status"] = "BASELINE_PASS"

    result["root_success"] = True
    result["root_message"] = "Validated B4-0 baseline supplied as seed."
    result["root_nfev"] = 0

    print(f"C0          = {C0:.12e}")
    print(
        f"r_sp        = {result['r_sp_m'] * 1.0e3:.12f} mm"
    )
    print(
        f"f_lr        = {result['f_lr_Hz']:.12f} Hz"
    )
    print(
        f"baseline f  = "
        f"{result['f_R_Hz']:+.12f}"
        f"{result['f_I_Hz']:+.12f} i Hz"
    )
    print(
        f"|Res|       = {result['Res_abs']:.12e}"
    )
    print(
        f"Q           = {result['Q']:.6e}"
    )
    print(
        f"r_turn      = "
        f"{result['r_turn_re_m']:+.12e}"
        f"{result['r_turn_im_m']:+.12e} i m"
    )

    return result


# ============================================================================
# ADDITIONAL: INDEPENDENT ROOT SOLVE AT C0 ITSELF
# ============================================================================

def solve_baseline_independently(solver, patch):
    """
    Solve for the resonance root directly AT C0, using the same root-finding
    machinery as the continuation steps -- instead of trusting the supplied
    B4-0 baseline value as-is.

    This provides:
      1. An independent cross-check of the B4-0 baseline (does root() find
         the same f_R, f_I from a nearby seed, or does it drift elsewhere?)
      2. A genuinely root-solved point at C0, with residual comparable to
         the neighboring continuation points (~1e-11/1e-12), instead of the
         supplied baseline's looser 1.14e-4.
      3. Two clean derivative intervals that straddle C0 symmetrically
         (0.995C0 -> C0) and (C0 -> 1.005C0), instead of skipping over it.
    """

    banner("INDEPENDENT ROOT SOLVE AT C0")

    r_sp, omega_lr = get_light_ring(solver, C0)

    if r_sp is None:
        raise RuntimeError("C0 light ring unexpectedly absent.")

    set_complex_C(solver, C0)

    seed = complex(BASELINE_F_RE, BASELINE_F_IM)

    result = continuation_step(
        solver=solver,
        patch=patch,
        previous_root=seed,
        C_value=C0,
    )

    status = result.get("status", "UNKNOWN")

    print(f"seed              = {seed.real:+.12f}{seed.imag:+.12f} i Hz")
    print(f"status            = {status}")

    if result.get("f_R_Hz") is not None:
        print(
            f"independent root  = "
            f"{result['f_R_Hz']:+.12f}"
            f"{result['f_I_Hz']:+.12f} i Hz"
        )
        print(f"|Res|             = {result['Res_abs']:.12e}")

        delta_from_baseline = complex(
            result["f_R_Hz"] - BASELINE_F_RE,
            result["f_I_Hz"] - BASELINE_F_IM,
        )
        print(
            f"delta vs supplied baseline = "
            f"{delta_from_baseline.real:+.6e}"
            f"{delta_from_baseline.imag:+.6e} i Hz"
        )

    if status != "PASS":
        print(
            "WARNING: independent solve at C0 did not reach PASS status; "
            "the supplied baseline should not be treated as independently "
            "re-confirmed by this check."
        )

    result["status"] = (
        "C0_INDEPENDENT_PASS" if status == "PASS" else status
    )

    return result


# ============================================================================
# MAIN
# ============================================================================

def main():

    banner(
        "STAGE 6.5B4-1A — MICRO C-CONTINUATION "
        "OF THE BASELINE RESONANCE @ N=1000"
    )

    print()
    print("Configuration:")
    print(f"  C0              = {C0:.12e}")
    print(f"  mode m          = {M_MODE}")
    print(f"  radial N        = {RADIAL_N}")
    print(f"  root method     = {ROOT_METHOD}")
    print(f"  root maxfev     = {ROOT_MAXFEV}")
    print(f"  root tol        = {ROOT_TOL:.3e}")
    print()
    print("Downward chain:")
    print("  C0 -> 0.995 C0 -> 0.990 C0")
    print()
    print("Upward chain:")
    print("  C0 -> 1.005 C0 -> 1.010 C0")
    print()
    print("Important:")
    print("  authoritative solver is READ-ONLY")
    print("  C mutation is IN-MEMORY ONLY")
    print("  no blind resonance scan")
    print("  no Kerr/GW calibration")
    print("  no frequency rescaling")
    print("  no long-wave approximation")

    # ----------------------------------------------------------------------
    # Load solver
    # ----------------------------------------------------------------------

    banner("AUTHORITATIVE LOAD")

    solver = load_authoritative_solver()
    print("Authoritative solver load: PASS")
    print(
        "Compatibility shim: "
        f"DEFAULT_COMPLEX_RADIAL_N={solver.DEFAULT_COMPLEX_RADIAL_N} "
        "installed and registered as 'stage5I_3_resonance' in sys.modules"
    )

    # ----------------------------------------------------------------------
    # API audit
    # ----------------------------------------------------------------------

    required_solver = [
        "hamiltonian",
        "solve_p_complex",
        "find_light_ring",
        "find_last_scattering_point",
        "find_turning_point_complex",
        "reflection_coefficient",
    ]

    for name in required_solver:
        if not hasattr(solver, name):
            raise RuntimeError(
                f"Required authoritative API missing: {name}"
            )

    print("Required authoritative APIs: PASS")

    # ----------------------------------------------------------------------
    # Load patch
    # ----------------------------------------------------------------------

    patch = load_radial_patch()

    if not hasattr(patch, "radial_action_complex_v2"):
        raise RuntimeError(
            "radial_action_complex_v2 missing from patch."
        )

    print("Radial-action patch load: PASS")

    # ----------------------------------------------------------------------
    # C injection audit
    # ----------------------------------------------------------------------

    banner("COMPLEX C-INJECTION AUDIT")

    probe_values = [
        0.995 * C0,
        C0,
        1.005 * C0,
    ]

    probe_values = list(probe_values)

    probe_H = []

    probe_omega = 2.0 * np.pi * BASELINE_F_RE

    for C_value in probe_values:

        set_complex_C(solver, C_value)

        H = solver.hamiltonian(
            probe_omega,
            0j,
            M_MODE,
            0.020,
        )

        H = complex(H)

        probe_H.append(H)

        print(
            f"C={C_value:.12e}  "
            f"H={fmt_complex(H)}"
        )

    d01 = abs(probe_H[1] - probe_H[0])
    d12 = abs(probe_H[2] - probe_H[1])

    print()
    print(
        f"|H(C0)-H(0.995C0)| = {d01:.12e}"
    )
    print(
        f"|H(1.005C0)-H(C0)|  = {d12:.12e}"
    )

    if d01 <= 1.0e-10 or d12 <= 1.0e-10:
        raise RuntimeError(
            "Complex C injection audit failed: "
            "C response is numerically indistinguishable."
        )

    print("COMPLEX C-INJECTION: PASS")

    # ----------------------------------------------------------------------
    # Baseline
    # ----------------------------------------------------------------------

    baseline = evaluate_baseline(
        solver,
        patch,
    )

    # ------------------------------------------------------------------
    # NEW: independent cross-check root-solve directly at C0
    # ------------------------------------------------------------------

    c0_independent = solve_baseline_independently(solver, patch)

    # ----------------------------------------------------------------------
    # Results
    # ----------------------------------------------------------------------

    results = [
        baseline,
        c0_independent,
    ]

    # ----------------------------------------------------------------------
    # Downward continuation
    # ----------------------------------------------------------------------

    banner("DOWNWARD CONTINUATION")

    previous_root = complex(
        baseline["f_R_Hz"],
        baseline["f_I_Hz"],
    )

    for factor in C_FACTORS_DOWN:

        C_value = factor * C0

        print()
        print(
            f"--- C = {factor:.6f} C0 "
            f"= {C_value:.12e} ---"
        )
        print(
            f"previous root = "
            f"{previous_root.real:+.12f}"
            f"{previous_root.imag:+.12f} i Hz"
        )

        result = continuation_step(
            solver=solver,
            patch=patch,
            previous_root=previous_root,
            C_value=C_value,
        )

        results.append(result)

        status = result.get("status", "UNKNOWN")

        print(
            f"status        = {status}"
        )

        if result.get("f_R_Hz") is not None:
            print(
                f"candidate      = "
                f"{result['f_R_Hz']:+.12f}"
                f"{result['f_I_Hz']:+.12f} i Hz"
            )

        if result.get("Res_abs") is not None:
            print(
                f"|Res|          = "
                f"{result['Res_abs']:.12e}"
            )

        if result.get("delta_f_R_Hz") is not None:
            print(
                f"delta f_R      = "
                f"{result['delta_f_R_Hz']:+.12e} Hz"
            )

        if result.get("delta_f_I_Hz") is not None:
            print(
                f"delta f_I      = "
                f"{result['delta_f_I_Hz']:+.12e} Hz"
            )

        if result.get("root_success") is not None:
            print(
                f"root success    = "
                f"{result['root_success']}"
            )

        if result.get("branch_guard_pass") is not None:
            print(
                f"branch guard    = "
                f"{result['branch_guard_pass']}"
            )

        # IMPORTANT:
        # Continue only from a genuinely accepted PASS.
        if status == "PASS":
            previous_root = complex(
                result["f_R_Hz"],
                result["f_I_Hz"],
            )
        else:
            print(
                "CHAIN STOP: this branch was not accepted; "
                "do not force continuation."
            )
            break

    # ----------------------------------------------------------------------
    # Upward continuation
    # ----------------------------------------------------------------------

    banner("UPWARD CONTINUATION")

    previous_root = complex(
        baseline["f_R_Hz"],
        baseline["f_I_Hz"],
    )

    for factor in C_FACTORS_UP:

        C_value = factor * C0

        print()
        print(
            f"--- C = {factor:.6f} C0 "
            f"= {C_value:.12e} ---"
        )
        print(
            f"previous root = "
            f"{previous_root.real:+.12f}"
            f"{previous_root.imag:+.12f} i Hz"
        )

        result = continuation_step(
            solver=solver,
            patch=patch,
            previous_root=previous_root,
            C_value=C_value,
        )

        results.append(result)

        status = result.get("status", "UNKNOWN")

        print(
            f"status        = {status}"
        )

        if result.get("f_R_Hz") is not None:
            print(
                f"candidate      = "
                f"{result['f_R_Hz']:+.12f}"
                f"{result['f_I_Hz']:+.12f} i Hz"
            )

        if result.get("Res_abs") is not None:
            print(
                f"|Res|          = "
                f"{result['Res_abs']:.12e}"
            )

        if result.get("delta_f_R_Hz") is not None:
            print(
                f"delta f_R      = "
                f"{result['delta_f_R_Hz']:+.12e} Hz"
            )

        if result.get("delta_f_I_Hz") is not None:
            print(
                f"delta f_I      = "
                f"{result['delta_f_I_Hz']:+.12e} Hz"
            )

        if result.get("root_success") is not None:
            print(
                f"root success    = "
                f"{result['root_success']}"
            )

        if result.get("branch_guard_pass") is not None:
            print(
                f"branch guard    = "
                f"{result['branch_guard_pass']}"
            )

        if status == "PASS":
            previous_root = complex(
                result["f_R_Hz"],
                result["f_I_Hz"],
            )
        else:
            print(
                "CHAIN STOP: this branch was not accepted; "
                "do not force continuation."
            )
            break

    # ----------------------------------------------------------------------
    # Derivative extraction
    # ----------------------------------------------------------------------

    banner("LOCAL C-DERIVATIVE AUDIT")

    valid = [
        r for r in results
        if r.get("status") in ("PASS", "C0_INDEPENDENT_PASS")
        and r.get("C") is not None
        and r.get("f_R_Hz") is not None
        and r.get("f_I_Hz") is not None
    ]

    derivative_rows = []

    if len(valid) >= 2:

        valid_sorted = sorted(
            valid,
            key=lambda x: x["C"],
        )

        for a, b in zip(valid_sorted[:-1], valid_sorted[1:]):

            dC = b["C"] - a["C"]

            if dC == 0.0:
                continue

            dfR_dC = (
                b["f_R_Hz"] - a["f_R_Hz"]
            ) / dC

            dfI_dC = (
                b["f_I_Hz"] - a["f_I_Hz"]
            ) / dC

            derivative_rows.append(
                {
                    "C_a": a["C"],
                    "C_b": b["C"],
                    "df_R_dC_Hz_per_C": dfR_dC,
                    "df_I_dC_Hz_per_C": dfI_dC,
                }
            )

            print(
                f"{a['C']:.12e} -> {b['C']:.12e}"
            )
            print(
                f"  df_R/dC = {dfR_dC:+.12e} Hz/C"
            )
            print(
                f"  df_I/dC = {dfI_dC:+.12e} Hz/C"
            )

    else:
        print(
            "Not enough accepted continuation points "
            "for a local derivative."
        )

    # ----------------------------------------------------------------------
    # Final verdict
    # ----------------------------------------------------------------------

    banner("B4-1A VERDICT")

    downward_pass = [
        r for r in results
        if r.get("status") == "PASS"
        and r.get("C", C0) < C0
    ]

    upward_pass = [
        r for r in results
        if r.get("status") == "PASS"
        and r.get("C", C0) > C0
    ]

    if len(downward_pass) >= 2 and len(upward_pass) >= 2:

        verdict = (
            "LOCAL C-CONTINUATION PASS: "
            "both downward and upward micro-branches "
            "were followed continuously."
        )

    elif len(downward_pass) >= 1 or len(upward_pass) >= 1:

        verdict = (
            "PARTIAL LOCAL C-CONTINUATION: "
            "at least one small C branch step was followed, "
            "but full two-sided continuation was not established."
        )

    else:

        verdict = (
            "LOCAL C-CONTINUATION UNRESOLVED: "
            "no small-step branch was accepted."
        )

    print(verdict)

    c0_status = c0_independent.get("status", "UNKNOWN")
    if c0_status == "C0_INDEPENDENT_PASS":
        print(
            "C0 INDEPENDENT CROSS-CHECK: PASS -- root solved directly at "
            "C0 confirms the supplied B4-0 baseline."
        )
    else:
        print(
            f"C0 INDEPENDENT CROSS-CHECK: {c0_status} -- the supplied "
            "baseline was NOT independently re-confirmed by direct "
            "root-solving at C0."
        )

    print()
    print("Interpretation rule:")
    print(
        "  This stage tests numerical continuity of the resonance branch."
    )
    print(
        "  It does NOT by itself establish a new fundamental interaction."
    )
    print(
        "  Failure is NOT interpreted as absence of a C effect."
    )

    if derivative_rows:
        print()
        print("Derivative availability:")
        print(
            f"  {len(derivative_rows)} local derivative intervals available."
        )

    # ----------------------------------------------------------------------
    # Save JSON
    # ----------------------------------------------------------------------

    payload = {
        "stage": "6.5B4-1A",
        "description": (
            "Micro C-continuation around validated baseline resonance, "
            "with independent root-solve cross-check at C0"
        ),
        "C0": C0,
        "Omega": OMEGA,
        "gamma": GAMMA,
        "h0": H0,
        "g": G_GRAV,
        "r_B": R_B,
        "m": M_MODE,
        "radial_n": RADIAL_N,

        "baseline_root": {
            "f_R_Hz": BASELINE_F_RE,
            "f_I_Hz": BASELINE_F_IM,
        },

        "C_factors_down": C_FACTORS_DOWN,
        "C_factors_up": C_FACTORS_UP,

        "jump_guards": {
            "max_jump_re_Hz": MAX_JUMP_RE_HZ,
            "max_jump_im_Hz": MAX_JUMP_IM_HZ,
        },

        "results": results,
        "derivatives": derivative_rows,
        "verdict": verdict,
        "c0_independent_status": c0_status,
    }

    JSON_FILE.write_text(
        json.dumps(
            payload,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # ----------------------------------------------------------------------
    # Save CSV
    # ----------------------------------------------------------------------

    csv_rows = []

    for r in results:

        row = {}

        for key, value in r.items():

            # Dictionaries are represented as JSON strings in CSV.
            if isinstance(value, dict):
                row[key] = json.dumps(
                    value,
                    sort_keys=True,
                )
            else:
                row[key] = value

        csv_rows.append(row)

    if csv_rows:

        fieldnames = []

        for row in csv_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

        with CSV_FILE.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )

            writer.writeheader()
            writer.writerows(csv_rows)

    print()
    print("CSV  ->", CSV_FILE)
    print("JSON ->", JSON_FILE)


if __name__ == "__main__":
    main()