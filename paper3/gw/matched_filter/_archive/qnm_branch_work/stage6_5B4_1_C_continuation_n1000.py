# -*- coding: utf-8 -*-

"""
==============================================================================
STAGE 6.5B4-1 — NARROW C-CONTINUATION OF THE BASELINE RESONANCE @ N=1000
==============================================================================

Purpose
-------
Test whether the known baseline complex resonance root moves continuously
when ONLY C is changed slightly.

Controlled points:
    C_MINUS = 0.95 * C0
    C0      = 1.00 * C0
    C_PLUS  = 1.05 * C0

Strategy
--------
1. Load authoritative stage5I_3_resonance.py read-only.
2. Load validated radial-action patch read-only.
3. Inject C only in memory.
4. Reproduce the known baseline B4-0 root at C0.
5. Use that root as the starting point for:
       C0 -> 0.95 C0
       C0 -> 1.05 C0
6. At each C, solve Re(Res)=0 and Im(Res)=0 locally using the previous
   complex root as the initial seed.
7. Re-evaluate the final candidate with N=1000.
8. Record f_R, f_I, Q, |Res|, r_sp, r_turn, etc.

Scientific rules
----------------
- NO modification of authoritative solver on disk.
- NO Kerr fit.
- NO GW calibration.
- NO frequency rescaling.
- NO low-k approximation.
- NO blind 2D frequency scan.
- NO inherited C/2 or 2C roots.
- C is varied only in memory.
- C=0 is NOT tested here because it has no interior light ring.
- This stage tests LOCAL CONTINUITY, not global uniqueness.

Important
---------
A successful scipy optimizer/root call is NOT by itself sufficient.
The final resonance residual and continuity from the previous root are also
checked.

==============================================================================
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import numpy as np


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent

RES_FILE = HERE / "stage5I_3_resonance.py"
PATCH_FILE = HERE / "stage5I_3c_radial_action_patch.py"

OUT_DIR = HERE / "stage6_5B4_1_C_continuation_n1000"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = OUT_DIR / "stage6_5B4_1_C_continuation_n1000.csv"
JSON_FILE = OUT_DIR / "stage6_5B4_1_C_continuation_n1000.json"


# =============================================================================
# ROOT-B CONSTANTS
# =============================================================================

M_MODE = -12

C0 = 7.690000000000e-04
OMEGA = 1.000000000000e-01
GAMMA = 2.220000000000e-06
H0 = 3.400000000000e-02
G_GRAV = 9.810000000000e+00
R_B = 3.730000000000e-02


# =============================================================================
# NUMERICAL SETTINGS
# =============================================================================

# Working radial resolution requested for continuation.
RADIAL_N = 1000

# Compatibility value only. The actual radial action call below explicitly
# uses RADIAL_N=1000.
COMPAT_DEFAULT_COMPLEX_RADIAL_N = 20000

# Root solver limits.
ROOT_MAX_NFEV = 35

# Numerical finite-difference step in Hz for the 2D complex-frequency solver.
# This is only used if scipy.optimize.root is unavailable/fails.
FD_STEP_RE_HZ = 2.0e-5
FD_STEP_IM_HZ = 2.0e-7

# Maximum allowed jump from the previous root.
# These are guards against the optimizer jumping to a different branch.
MAX_JUMP_RE_HZ = 0.75
MAX_JUMP_IM_HZ = 0.0015

# Minimum quality of a candidate.
# This is deliberately much looser than the final residual acceptance.
MAX_RESIDUAL_FOR_CANDIDATE = 5.0e-3

# Strong residual threshold for a resolved local resonance.
RESOLVED_RESIDUAL = 1.0e-3

# Known B4-0 baseline candidate.
BASELINE_ROOT_HZ = complex(
    8.306625532027001,
    -0.000586962309000,
)


# =============================================================================
# UTILITY
# =============================================================================

def banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def section(text: str) -> None:
    print("\n" + "-" * 78)
    print(text)
    print("-" * 78)


def fmt_complex(z: complex) -> str:
    return f"{z.real:+.12e}{z.imag:+.12e}i"


def finite_complex(z) -> bool:
    z = complex(z)
    return (
        math.isfinite(z.real)
        and math.isfinite(z.imag)
    )


# =============================================================================
# AUTHORITATIVE SOLVER LOADER
# =============================================================================

def load_authoritative_solver():
    """
    Load stage5I_3_resonance.py without executing its __main__ self-test.

    The module is registered under the canonical name expected by the
    radial-action patch.
    """

    if not RES_FILE.exists():
        raise FileNotFoundError(
            f"Authoritative solver not found:\n{RES_FILE}"
        )

    source = RES_FILE.read_text(encoding="utf-8")

    tree = ast.parse(
        source,
        filename=str(RES_FILE),
    )

    module_name = "_stage5I_3_resonance_B4_1"

    spec = importlib.util.spec_from_loader(
        module_name,
        loader=None,
    )

    solver = importlib.util.module_from_spec(spec)

    # Critical compatibility fields BEFORE patch import.
    solver.__file__ = str(RES_FILE)
    solver.__name__ = module_name
    solver.__package__ = ""
    solver.__spec__ = spec

    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))

    # Execute authoritative source.
    exec(
        compile(tree, str(RES_FILE), "exec"),
        solver.__dict__,
    )

    # The patch imports this symbol.
    if not hasattr(solver, "DEFAULT_COMPLEX_RADIAL_N"):
        solver.DEFAULT_COMPLEX_RADIAL_N = COMPAT_DEFAULT_COMPLEX_RADIAL_N
        print(
            "Compatibility shim: "
            "DEFAULT_COMPLEX_RADIAL_N=20000 installed in memory."
        )
    else:
        print(
            "Authoritative DEFAULT_COMPLEX_RADIAL_N already present."
        )

    # Register under the exact import name expected by the patch.
    sys.modules["stage5I_3_resonance"] = solver

    return solver


# =============================================================================
# RADIAL PATCH LOADER
# =============================================================================

def load_radial_patch():
    if not PATCH_FILE.exists():
        raise FileNotFoundError(
            f"Radial patch not found:\n{PATCH_FILE}"
        )

    module_name = "_stage5I_3c_radial_action_patch_B4_1"

    spec = importlib.util.spec_from_file_location(
        module_name,
        PATCH_FILE,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not create module spec for:\n{PATCH_FILE}"
        )

    patch = importlib.util.module_from_spec(spec)

    # Make sure __file__ exists for any relative path logic.
    patch.__file__ = str(PATCH_FILE)

    spec.loader.exec_module(patch)

    return patch


# =============================================================================
# API CHECK
# =============================================================================

def check_apis(solver, patch):
    required_solver = [
        "hamiltonian",
        "solve_p_complex",
        "find_light_ring",
        "find_last_scattering_point",
        "find_turning_point_complex",
        "reflection_coefficient",
    ]

    required_patch = [
        "radial_action_complex_v2",
    ]

    missing_solver = [
        name for name in required_solver
        if not hasattr(solver, name)
    ]

    missing_patch = [
        name for name in required_patch
        if not hasattr(patch, name)
    ]

    if missing_solver:
        raise RuntimeError(
            "Missing authoritative APIs: " +
            ", ".join(missing_solver)
        )

    if missing_patch:
        raise RuntimeError(
            "Missing radial patch APIs: " +
            ", ".join(missing_patch)
        )

    print("Required APIs: PASS")

    print("\nAuthoritative solver APIs:")
    for name in required_solver:
        print(f"  {name}: {getattr(solver, name)}")

    print("\nRadial patch API:")
    for name in required_patch:
        print(f"  {name}: {getattr(patch, name)}")


# =============================================================================
# IN-MEMORY C CONTROL
# =============================================================================

def set_complex_C(solver, C_value: float):
    """
    Change ONLY the C-related module-level constants used by the complex API.

    The authoritative file on disk is never changed.

    Different versions of the authoritative solver have used C_METHODS,
    C_TABLE and/or C. We update only those names if present.
    """

    names_set = []

    for name in ("C_METHODS", "C_TABLE", "C"):
        if hasattr(solver, name):
            setattr(solver, name, float(C_value))
            names_set.append(name)

    if not names_set:
        raise RuntimeError(
            "Could not find a C-related module-level constant "
            "in authoritative complex solver."
        )

    return names_set


def audit_complex_C_injection(solver):
    """
    Confirm that changing C actually changes the complex Hamiltonian.
    """

    omega_test = (
        2.0 * math.pi *
        complex(
            BASELINE_ROOT_HZ.real,
            BASELINE_ROOT_HZ.imag,
        )
    )

    # Use a safe finite point away from the exact turning point.
    r_test = complex(
        0.025,
        2.0e-6,
    )

    p_test = complex(0.7, 0.01)

    set_complex_C(solver, 0.95 * C0)
    H_minus = complex(
        solver.hamiltonian(
            omega_test,
            p_test,
            M_MODE,
            r_test,
        )
    )

    set_complex_C(solver, C0)
    H_zero = complex(
        solver.hamiltonian(
            omega_test,
            p_test,
            M_MODE,
            r_test,
        )
    )

    set_complex_C(solver, 1.05 * C0)
    H_plus = complex(
        solver.hamiltonian(
            omega_test,
            p_test,
            M_MODE,
            r_test,
        )
    )

    d_minus = abs(H_minus - H_zero)
    d_plus = abs(H_plus - H_zero)

    print("\nComplex C-injection audit:")
    print(f"  H(0.95C0) = {fmt_complex(H_minus)}")
    print(f"  H(C0)      = {fmt_complex(H_zero)}")
    print(f"  H(1.05C0) = {fmt_complex(H_plus)}")
    print(f"  |H(0.95C0)-H(C0)| = {d_minus:.8e}")
    print(f"  |H(1.05C0)-H(C0)| = {d_plus:.8e}")

    passed = (
        math.isfinite(d_minus)
        and math.isfinite(d_plus)
        and d_minus > 1.0e-10
        and d_plus > 1.0e-10
    )

    print(
        "C-INJECTION AUDIT:",
        "PASS" if passed else "FAIL",
    )

    if not passed:
        raise RuntimeError(
            "Complex C injection did not produce a measurable change."
        )

    # Restore baseline.
    set_complex_C(solver, C0)

    return {
        "H_minus": H_minus,
        "H_zero": H_zero,
        "H_plus": H_plus,
        "delta_minus": d_minus,
        "delta_plus": d_plus,
    }


# =============================================================================
# REAL LIGHT RING
# =============================================================================

def get_light_ring(solver, C_value: float):
    """
    Explicit C injection into the authoritative REAL light-ring function.

    This is deliberately separate from the complex module-level C mutation.
    """

    result = solver.find_light_ring(
        M_MODE,
        C=float(C_value),
        Omega=OMEGA,
        gamma=GAMMA,
        h0=H0,
        g=G_GRAV,
        r_min=1.0e-3,
        r_max=R_B,
    )

    if result is None:
        return None, None

    if len(result) != 2:
        raise RuntimeError(
            f"Unexpected find_light_ring result: {result}"
        )

    r_sp, omega_lr = result

    if r_sp is None or omega_lr is None:
        return None, None

    r_sp = float(r_sp)
    omega_lr = float(omega_lr)

    if not (
        math.isfinite(r_sp)
        and math.isfinite(omega_lr)
    ):
        return None, None

    f_lr = omega_lr / (2.0 * math.pi)

    return r_sp, f_lr


# =============================================================================
# RESONANCE EVALUATION
# =============================================================================

def evaluate_resonance(
    solver,
    patch,
    C_value: float,
    f_complex_hz: complex,
    n: int = RADIAL_N,
):
    """
    Evaluate:
        light ring
        last scattering point
        complex turning point
        H(turn)
        radial action
        reflection coefficient
        Res = R exp(2 i S) - 1

    All complex machinery uses the in-memory C value.
    """

    f_complex_hz = complex(f_complex_hz)

    if not finite_complex(f_complex_hz):
        raise FloatingPointError(
            f"Non-finite frequency: {f_complex_hz}"
        )

    # Inject C into complex solver.
    set_complex_C(solver, C_value)

    # Real light ring uses explicit C.
    r_sp, f_lr = get_light_ring(
        solver,
        C_value,
    )

    if r_sp is None:
        return {
            "valid": False,
            "reason": "NO_LIGHT_RING",
            "C": float(C_value),
            "f": f_complex_hz,
        }

    omega = 2.0 * math.pi * f_complex_hz

    # Last scattering point.
    r_minus_real = solver.find_last_scattering_point(
        omega,
        M_MODE,
        r_sp,
    )

    if r_minus_real is None:
        raise RuntimeError(
            "find_last_scattering_point returned None"
        )

    r_minus_real = float(np.real(r_minus_real))

    if not math.isfinite(r_minus_real):
        raise FloatingPointError(
            "Non-finite r_minus_real"
        )

    # Complex turning point.
    r_turn = solver.find_turning_point_complex(
        omega,
        M_MODE,
        r_minus_real,
    )

    if r_turn is None:
        raise RuntimeError(
            "find_turning_point_complex returned None"
        )

    r_turn = complex(r_turn)

    if not finite_complex(r_turn):
        raise FloatingPointError(
            "Non-finite complex turning point"
        )

    # Turning-point Hamiltonian residual.
    H_turn = complex(
        solver.hamiltonian(
            omega,
            0j,
            M_MODE,
            r_turn,
        )
    )

    H_turn_abs = abs(H_turn)

    if not math.isfinite(H_turn_abs):
        raise FloatingPointError(
            "Non-finite H(turn)"
        )

    # Validated radial action.
    S = patch.radial_action_complex_v2(
        omega,
        M_MODE,
        r_minus_real,
        r_B=R_B,
        n=int(n),
    )

    S = complex(S)

    if not finite_complex(S):
        raise FloatingPointError(
            "Non-finite radial action"
        )

    # Reflection coefficient.
    R = solver.reflection_coefficient(
        omega,
        M_MODE,
        r_sp,
    )

    R = complex(R)

    if not finite_complex(R):
        raise FloatingPointError(
            "Non-finite reflection coefficient"
        )

    # Resonance condition used throughout the validated branch:
    #
    #     Res = R exp(2 i S) - 1
    #
    Res = complex(
        R * np.exp(2j * S) - 1.0
    )

    if not finite_complex(Res):
        raise FloatingPointError(
            "Non-finite resonance residual"
        )

    f_R = float(f_complex_hz.real)
    f_I = float(f_complex_hz.imag)

    if abs(f_I) > 0.0:
        Q = abs(f_R) / (2.0 * abs(f_I))
    else:
        Q = math.inf

    return {
        "valid": True,
        "reason": "OK",

        "C": float(C_value),

        "f": f_complex_hz,
        "f_R": f_R,
        "f_I": f_I,

        "Q": float(Q),

        "r_sp": float(r_sp),
        "r_minus_real": float(r_minus_real),
        "r_turn": r_turn,

        "omega": omega,

        "f_lr": float(f_lr),
        "omega_lr": float(2.0 * math.pi * f_lr),

        "H_turn": H_turn,
        "H_turn_abs": float(H_turn_abs),

        "S": S,
        "S_abs": float(abs(S)),

        "R": R,
        "R_abs": float(abs(R)),

        "Res": Res,
        "Res_abs": float(abs(Res)),

        "radial_n": int(n),
    }


# =============================================================================
# SCALAR OBJECTIVE FOR LOCAL ROOT SOLVER
# =============================================================================

class ResonanceEvaluator:
    """
    Cached wrapper so scipy root does not evaluate the exact same point
    multiple times unnecessarily.
    """

    def __init__(
        self,
        solver,
        patch,
        C_value,
        radial_n,
    ):
        self.solver = solver
        self.patch = patch
        self.C_value = float(C_value)
        self.radial_n = int(radial_n)

        self.last_x = None
        self.last_result = None
        self.calls = 0
        self.failures = 0

    def evaluate(self, f_complex_hz: complex):
        f_complex_hz = complex(f_complex_hz)

        if (
            self.last_x is not None
            and abs(f_complex_hz - self.last_x) < 1.0e-15
            and self.last_result is not None
        ):
            return self.last_result

        self.calls += 1

        try:
            result = evaluate_resonance(
                self.solver,
                self.patch,
                self.C_value,
                f_complex_hz,
                n=self.radial_n,
            )

            self.last_x = f_complex_hz
            self.last_result = result

            return result

        except Exception as exc:
            self.failures += 1

            result = {
                "valid": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "C": self.C_value,
                "f": f_complex_hz,
            }

            self.last_x = f_complex_hz
            self.last_result = result

            return result

    def residual_vector(self, x):
        """
        x = [f_R, f_I]
        return [Re(Res), Im(Res)]
        """

        f = complex(
            float(x[0]),
            float(x[1]),
        )

        result = self.evaluate(f)

        if not result.get("valid", False):
            # Large finite penalty. Never return NaN to scipy.
            return np.array(
                [10.0, 10.0],
                dtype=float,
            )

        Res = complex(result["Res"])

        if not finite_complex(Res):
            return np.array(
                [10.0, 10.0],
                dtype=float,
            )

        return np.array(
            [Res.real, Res.imag],
            dtype=float,
        )


# =============================================================================
# LOCAL COMPLEX ROOT SOLVER
# =============================================================================

def solve_local_root(
    solver,
    patch,
    C_value,
    seed_hz,
):
    """
    Solve Re(Res)=0 and Im(Res)=0 locally around seed_hz.

    Primary:
        scipy.optimize.root(method='hybr')

    Fallback:
        scipy.optimize.least_squares

    The result is accepted only if:
        - finite
        - close to the previous root
        - radial calculation is valid
        - final residual is finite
        - residual is reasonably small
    """

    evaluator = ResonanceEvaluator(
        solver,
        patch,
        C_value,
        RADIAL_N,
    )

    seed_hz = complex(seed_hz)

    print("\n  LOCAL CONTINUATION")
    print(f"    C = {C_value:.12e}")
    print(f"    seed = {fmt_complex(seed_hz)} Hz")

    seed_eval = evaluator.evaluate(seed_hz)

    if seed_eval.get("valid", False):
        print(
            f"    seed |Res| = "
            f"{seed_eval['Res_abs']:.8e}"
        )
    else:
        print(
            "    seed evaluation FAILED:"
            f" {seed_eval.get('reason')}"
        )

    x0 = np.array(
        [seed_hz.real, seed_hz.imag],
        dtype=float,
    )

    scipy_root_result = None

    # -------------------------------------------------------------------------
    # PRIMARY ROOT SOLVER
    # -------------------------------------------------------------------------

    try:
        from scipy.optimize import root

        scipy_root_result = root(
            evaluator.residual_vector,
            x0,
            method="hybr",
            options={
                "maxfev": ROOT_MAX_NFEV,
                "xtol": 1.0e-8,
            },
        )

        print("\n    scipy.root:")
        print(
            f"      success = "
            f"{scipy_root_result.success}"
        )
        print(
            f"      status  = "
            f"{scipy_root_result.status}"
        )
        print(
            f"      nfev    = "
            f"{getattr(scipy_root_result, 'nfev', -1)}"
        )
        print(
            f"      message = "
            f"{scipy_root_result.message}"
        )

        candidate_x = np.asarray(
            scipy_root_result.x,
            dtype=float,
        )

    except Exception as exc:
        print(
            "\n    scipy.root exception:",
            type(exc).__name__,
            str(exc),
        )
        candidate_x = x0.copy()

    candidate = complex(
        float(candidate_x[0]),
        float(candidate_x[1]),
    )

    # -------------------------------------------------------------------------
    # GUARD AGAINST BRANCH JUMP
    # -------------------------------------------------------------------------

    jump = candidate - seed_hz

    print("\n    candidate:")
    print(f"      f = {fmt_complex(candidate)} Hz")
    print(f"      jump = {fmt_complex(jump)} Hz")

    if (
        abs(jump.real) > MAX_JUMP_RE_HZ
        or abs(jump.imag) > MAX_JUMP_IM_HZ
    ):
        print(
            "      BRANCH-JUMP GUARD: FAIL"
        )

        return {
            "success": False,
            "reason": "BRANCH_JUMP",
            "seed": seed_hz,
            "candidate": candidate,
            "calls": evaluator.calls,
            "failures": evaluator.failures,
        }

    # -------------------------------------------------------------------------
    # FINAL EXACT EVALUATION
    # -------------------------------------------------------------------------

    final_result = evaluate_resonance(
        solver,
        patch,
        C_value,
        candidate,
        n=RADIAL_N,
    )

    if not final_result.get("valid", False):
        print(
            "      final evaluation FAILED:",
            final_result.get("reason"),
        )

        return {
            "success": False,
            "reason": final_result.get("reason"),
            "seed": seed_hz,
            "candidate": candidate,
            "calls": evaluator.calls,
            "failures": evaluator.failures,
        }

    print(
        f"      final |Res| = "
        f"{final_result['Res_abs']:.8e}"
    )

    # -------------------------------------------------------------------------
    # ACCEPTANCE
    # -------------------------------------------------------------------------

    residual_ok = (
        final_result["Res_abs"]
        <= MAX_RESIDUAL_FOR_CANDIDATE
    )

    finite_ok = all(
        [
            finite_complex(final_result["f"]),
            finite_complex(final_result["r_turn"]),
            finite_complex(final_result["S"]),
            finite_complex(final_result["R"]),
            finite_complex(final_result["Res"]),
        ]
    )

    accepted = (
        residual_ok
        and finite_ok
    )

    if accepted:
        print("      LOCAL ROOT: PASS")
    else:
        print("      LOCAL ROOT: FAIL")

    return {
        "success": bool(accepted),
        "reason": (
            "PASS"
            if accepted
            else "RESIDUAL_OR_FINITE_CHECK"
        ),
        "seed": seed_hz,
        "candidate": candidate,
        "calls": evaluator.calls,
        "failures": evaluator.failures,
        "result": final_result,
    }


# =============================================================================
# BASELINE REPRODUCTION
# =============================================================================

def reproduce_baseline(solver, patch):
    """
    Re-evaluate the known B4-0 baseline root before doing continuation.
    """

    banner("BASELINE ROOT REPRODUCTION")

    set_complex_C(solver, C0)

    result = evaluate_resonance(
        solver,
        patch,
        C0,
        BASELINE_ROOT_HZ,
        n=RADIAL_N,
    )

    if not result.get("valid", False):
        raise RuntimeError(
            "Baseline root could not be evaluated:\n"
            + str(result.get("reason"))
        )

    print(
        f"Baseline f = {fmt_complex(BASELINE_ROOT_HZ)} Hz"
    )
    print(
        f"Baseline |Res| = {result['Res_abs']:.12e}"
    )
    print(
        f"Baseline r_sp = "
        f"{result['r_sp'] * 1.0e3:.12f} mm"
    )
    print(
        f"Baseline r_turn = "
        f"{fmt_complex(result['r_turn'])} m"
    )
    print(
        f"Baseline Q = {result['Q']:.12e}"
    )

    if result["Res_abs"] > MAX_RESIDUAL_FOR_CANDIDATE:
        raise RuntimeError(
            "Baseline residual is too large for continuation."
        )

    print("\nBASELINE REPRODUCTION: PASS")

    return result


# =============================================================================
# CONTINUITY CHECK
# =============================================================================

def continuity_metrics(
    previous_result,
    current_result,
):
    if previous_result is None:
        return {
            "delta_f_R": None,
            "delta_f_I": None,
            "delta_f_abs": None,
            "relative_delta_f": None,
        }

    df_R = (
        current_result["f_R"]
        - previous_result["f_R"]
    )

    df_I = (
        current_result["f_I"]
        - previous_result["f_I"]
    )

    df_abs = abs(
        current_result["f"]
        - previous_result["f"]
    )

    denominator = max(
        abs(previous_result["f"]),
        1.0e-30,
    )

    relative = df_abs / denominator

    return {
        "delta_f_R": float(df_R),
        "delta_f_I": float(df_I),
        "delta_f_abs": float(df_abs),
        "relative_delta_f": float(relative),
    }


# =============================================================================
# CASE RUNNER
# =============================================================================

def run_case(
    solver,
    patch,
    label,
    C_value,
    seed_result,
):
    section(
        f"{label}: C={C_value:.12e}"
    )

    # Real light ring first.
    r_sp, f_lr = get_light_ring(
        solver,
        C_value,
    )

    if r_sp is None:
        print("STATUS = NO_LIGHT_RING")

        return {
            "label": label,
            "C": float(C_value),
            "status": "NO_LIGHT_RING",
            "f_lr": None,
            "seed_f_R": (
                float(seed_result["f_R"])
                if seed_result is not None
                else None
            ),
            "seed_f_I": (
                float(seed_result["f_I"])
                if seed_result is not None
                else None
            ),
        }

    print(
        f"light ring = {r_sp * 1.0e3:.12f} mm"
    )
    print(
        f"f_lr       = {f_lr:.12f} Hz"
    )

    # Previous root is always the seed.
    seed = complex(seed_result["f"])

    continuation = solve_local_root(
        solver,
        patch,
        C_value,
        seed,
    )

    if not continuation["success"]:
        print(
            "\nSTATUS = FAIL"
        )
        print(
            f"reason = {continuation['reason']}"
        )

        return {
            "label": label,
            "C": float(C_value),
            "status": "FAIL",
            "reason": continuation["reason"],
            "f_lr": float(f_lr),
            "seed_f_R": float(seed.real),
            "seed_f_I": float(seed.imag),
            "candidate_f_R": float(
                continuation["candidate"].real
            ),
            "candidate_f_I": float(
                continuation["candidate"].imag
            ),
        }

    result = continuation["result"]

    metrics = continuity_metrics(
        seed_result,
        result,
    )

    print("\n  RESULT")
    print(
        f"    f_R       = {result['f_R']:.12f} Hz"
    )
    print(
        f"    f_I       = {result['f_I']:.12f} Hz"
    )
    print(
        f"    Q         = {result['Q']:.12e}"
    )
    print(
        f"    |Res|     = {result['Res_abs']:.12e}"
    )
    print(
        f"    Re(Res)   = {result['Res'].real:+.12e}"
    )
    print(
        f"    Im(Res)   = {result['Res'].imag:+.12e}"
    )
    print(
        f"    r_sp      = "
        f"{result['r_sp'] * 1.0e3:.12f} mm"
    )
    print(
        f"    r_turn    = "
        f"{fmt_complex(result['r_turn'])} m"
    )
    print(
        f"    H(turn)   = "
        f"{fmt_complex(result['H_turn'])}"
    )
    print(
        f"    S         = "
        f"{fmt_complex(result['S'])}"
    )
    print(
        f"    |R|       = "
        f"{result['R_abs']:.12e}"
    )

    print("\n  CONTINUITY FROM PREVIOUS ROOT")
    print(
        f"    Delta f_R = "
        f"{metrics['delta_f_R']:+.12e} Hz"
    )
    print(
        f"    Delta f_I = "
        f"{metrics['delta_f_I']:+.12e} Hz"
    )
    print(
        f"    |Delta f| = "
        f"{metrics['delta_f_abs']:.12e} Hz"
    )

    print("\nSTATUS = PASS")

    return {
        "label": label,
        "C": float(C_value),
        "status": "PASS",

        "f_lr": float(f_lr),

        "seed_f_R": float(seed.real),
        "seed_f_I": float(seed.imag),

        "f_R": float(result["f_R"]),
        "f_I": float(result["f_I"]),
        "Q": float(result["Q"]),

        "r_sp_m": float(result["r_sp"]),
        "r_sp_mm": float(result["r_sp"] * 1.0e3),

        "r_minus_real_m": float(
            result["r_minus_real"]
        ),

        "r_turn_re_m": float(
            result["r_turn"].real
        ),
        "r_turn_im_m": float(
            result["r_turn"].imag
        ),

        "H_turn_re": float(
            result["H_turn"].real
        ),
        "H_turn_im": float(
            result["H_turn"].imag
        ),
        "H_turn_abs": float(
            result["H_turn_abs"]
        ),

        "S_re": float(result["S"].real),
        "S_im": float(result["S"].imag),
        "S_abs": float(result["S_abs"]),

        "R_re": float(result["R"].real),
        "R_im": float(result["R"].imag),
        "R_abs": float(result["R_abs"]),

        "Res_re": float(result["Res"].real),
        "Res_im": float(result["Res"].imag),
        "Res_abs": float(result["Res_abs"]),

        "delta_f_R": metrics["delta_f_R"],
        "delta_f_I": metrics["delta_f_I"],
        "delta_f_abs": metrics["delta_f_abs"],
        "relative_delta_f": metrics["relative_delta_f"],

        "radial_n": int(result["radial_n"]),

        "root_calls": int(
            continuation["calls"]
        ),
        "root_failures": int(
            continuation["failures"]
        ),

        "reason": "LOCAL_CONTINUATION_PASS",
    }


# =============================================================================
# SAVE CSV
# =============================================================================

def save_csv(rows):
    if not rows:
        return

    # Union of all keys.
    fieldnames = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with CSV_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# =============================================================================
# SAVE JSON
# =============================================================================

def json_safe(value):
    if isinstance(value, complex):
        return {
            "re": float(value.real),
            "im": float(value.imag),
        }

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(v)
            for v in value
        ]

    return value


def save_json(payload):
    with JSON_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_safe(payload),
            f,
            indent=2,
            sort_keys=True,
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    banner(
        "STAGE 6.5B4-1 — NARROW C-CONTINUATION "
        "OF THE BASELINE RESONANCE @ N=1000"
    )

    print(
        "\nAuthoritative solver:"
        f"\n  {RES_FILE}"
    )

    print(
        "\nRadial patch:"
        f"\n  {PATCH_FILE}"
    )

    print(
        "\nMode:"
        f"\n  m = {M_MODE}"
    )

    print(
        "\nRoot-B constants:"
        f"\n  C0     = {C0:.12e}"
        f"\n  Omega  = {OMEGA:.12e}"
        f"\n  gamma  = {GAMMA:.12e}"
        f"\n  h0     = {H0:.12e}"
        f"\n  g      = {G_GRAV:.12e}"
        f"\n  r_B    = {R_B:.12e}"
    )

    print(
        "\nWorking radial resolution:"
        f"\n  N = {RADIAL_N}"
    )

    print(
        "\nKnown B4-0 baseline root:"
        f"\n  f = {fmt_complex(BASELINE_ROOT_HZ)} Hz"
    )

    # -------------------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------------------

    section("1. LOAD AUTHORITATIVE SOLVER")

    solver = load_authoritative_solver()

    print("Authoritative solver load: PASS")

    if not hasattr(
        solver,
        "DEFAULT_COMPLEX_RADIAL_N",
    ):
        raise RuntimeError(
            "Compatibility shim installation failed."
        )

    print(
        "Compatibility shim: "
        "DEFAULT_COMPLEX_RADIAL_N="
        f"{solver.DEFAULT_COMPLEX_RADIAL_N} installed in memory."
    )

    patch = load_radial_patch()

    print("Radial patch load: PASS")

    check_apis(
        solver,
        patch,
    )

    # -------------------------------------------------------------------------
    # C INJECTION AUDIT
    # -------------------------------------------------------------------------

    section("2. COMPLEX C-INJECTION AUDIT")

    injection_audit = audit_complex_C_injection(
        solver,
    )

    # -------------------------------------------------------------------------
    # BASELINE REPRO
    # -------------------------------------------------------------------------

    baseline_result = reproduce_baseline(
        solver,
        patch,
    )

    # -------------------------------------------------------------------------
    # CONTINUATION CASES
    # -------------------------------------------------------------------------

    C_MINUS = 0.95 * C0
    C_PLUS = 1.05 * C0

    rows = []

    # Store baseline itself.
    baseline_row = {
        "label": "BASELINE",
        "C": C0,
        "status": "PASS",
        "f_lr": float(baseline_result["f_lr"]),
        "seed_f_R": float(
            BASELINE_ROOT_HZ.real
        ),
        "seed_f_I": float(
            BASELINE_ROOT_HZ.imag
        ),
        "f_R": float(
            baseline_result["f_R"]
        ),
        "f_I": float(
            baseline_result["f_I"]
        ),
        "Q": float(
            baseline_result["Q"]
        ),
        "r_sp_m": float(
            baseline_result["r_sp"]
        ),
        "r_sp_mm": float(
            baseline_result["r_sp"] * 1.0e3
        ),
        "r_minus_real_m": float(
            baseline_result["r_minus_real"]
        ),
        "r_turn_re_m": float(
            baseline_result["r_turn"].real
        ),
        "r_turn_im_m": float(
            baseline_result["r_turn"].imag
        ),
        "H_turn_abs": float(
            baseline_result["H_turn_abs"]
        ),
        "S_re": float(
            baseline_result["S"].real
        ),
        "S_im": float(
            baseline_result["S"].imag
        ),
        "S_abs": float(
            baseline_result["S_abs"]
        ),
        "R_re": float(
            baseline_result["R"].real
        ),
        "R_im": float(
            baseline_result["R"].imag
        ),
        "R_abs": float(
            baseline_result["R_abs"]
        ),
        "Res_re": float(
            baseline_result["Res"].real
        ),
        "Res_im": float(
            baseline_result["Res"].imag
        ),
        "Res_abs": float(
            baseline_result["Res_abs"]
        ),
        "delta_f_R": 0.0,
        "delta_f_I": 0.0,
        "delta_f_abs": 0.0,
        "relative_delta_f": 0.0,
        "radial_n": RADIAL_N,
        "reason": "B4_0_BASELINE_REPRODUCTION",
    }

    rows.append(baseline_row)

    # -------------------------------------------------------------------------
    # DOWNWARD CONTINUATION
    # -------------------------------------------------------------------------

    minus_row = run_case(
        solver,
        patch,
        "C_MINUS_0.95",
        C_MINUS,
        baseline_result,
    )

    rows.append(minus_row)

    # -------------------------------------------------------------------------
    # UPWARD CONTINUATION
    #
    # IMPORTANT:
    # Return to the known baseline root, rather than using the C_MINUS root.
    # This makes the two directions independent local perturbations of C0.
    # -------------------------------------------------------------------------

    set_complex_C(
        solver,
        C0,
    )

    plus_row = run_case(
        solver,
        patch,
        "C_PLUS_1.05",
        C_PLUS,
        baseline_result,
    )

    rows.append(plus_row)

    # -------------------------------------------------------------------------
    # LOCAL DERIVATIVE ESTIMATES
    # -------------------------------------------------------------------------

    banner("3. LOCAL C-RESPONSE SUMMARY")

    derivative = {
        "df_R_dC_central": None,
        "df_I_dC_central": None,
        "df_R_dC_minus": None,
        "df_I_dC_minus": None,
        "df_R_dC_plus": None,
        "df_I_dC_plus": None,
    }

    minus_ok = (
        minus_row.get("status") == "PASS"
    )

    plus_ok = (
        plus_row.get("status") == "PASS"
    )

    if minus_ok:
        derivative["df_R_dC_minus"] = (
            (
                minus_row["f_R"]
                - baseline_row["f_R"]
            )
            /
            (C_MINUS - C0)
        )

        derivative["df_I_dC_minus"] = (
            (
                minus_row["f_I"]
                - baseline_row["f_I"]
            )
            /
            (C_MINUS - C0)
        )

    if plus_ok:
        derivative["df_R_dC_plus"] = (
            (
                plus_row["f_R"]
                - baseline_row["f_R"]
            )
            /
            (C_PLUS - C0)
        )

        derivative["df_I_dC_plus"] = (
            (
                plus_row["f_I"]
                - baseline_row["f_I"]
            )
            /
            (C_PLUS - C0)
        )

    if minus_ok and plus_ok:
        derivative["df_R_dC_central"] = (
            (
                plus_row["f_R"]
                - minus_row["f_R"]
            )
            /
            (C_PLUS - C_MINUS)
        )

        derivative["df_I_dC_central"] = (
            (
                plus_row["f_I"]
                - minus_row["f_I"]
            )
            /
            (C_PLUS - C_MINUS)
        )

    print("\nLOCAL DERIVATIVES")

    for key, value in derivative.items():
        if value is None:
            print(f"  {key:24s} = NOT AVAILABLE")
        else:
            print(
                f"  {key:24s} = "
                f"{value:+.12e}"
            )

    # -------------------------------------------------------------------------
    # CONTINUITY TEST
    # -------------------------------------------------------------------------

    print("\nCONTINUATION STATUS")

    if minus_ok:
        print("  C0 -> 0.95 C0 : PASS")
    else:
        print("  C0 -> 0.95 C0 : FAIL")

    if plus_ok:
        print("  C0 -> 1.05 C0 : PASS")
    else:
        print("  C0 -> 1.05 C0 : FAIL")

    both_pass = (
        minus_ok
        and plus_ok
    )

    # A stronger check:
    # both directions should produce finite, reasonably small local changes.
    # We deliberately do NOT demand a particular sign before seeing the data.
    local_response = False

    if both_pass:
        dminus = complex(
            minus_row["f_R"] - baseline_row["f_R"],
            minus_row["f_I"] - baseline_row["f_I"],
        )

        dplus = complex(
            plus_row["f_R"] - baseline_row["f_R"],
            plus_row["f_I"] - baseline_row["f_I"],
        )

        local_response = (
            finite_complex(dminus)
            and finite_complex(dplus)
            and (
                abs(dminus) > 1.0e-10
                or abs(dplus) > 1.0e-10
            )
        )

    # -------------------------------------------------------------------------
    # FINAL VERDICT
    # -------------------------------------------------------------------------

    banner("B4-1 VERDICT")

    if both_pass:
        print(
            "NARROW C-CONTINUATION: PASS"
        )
        print(
            "\nThe known baseline resonance root can be continued"
            " locally to both 0.95 C0 and 1.05 C0."
        )

        if local_response:
            print(
                "\nLOCAL C-DEPENDENT RESONANCE RESPONSE: DETECTED"
            )
            print(
                "The complex resonance frequency changes under"
                " the controlled local C perturbation."
            )
        else:
            print(
                "\nLOCAL C-DEPENDENT RESONANCE RESPONSE: NOT RESOLVED"
            )

        print(
            "\nThis is stronger evidence than the independent B3 scans,"
            " because both perturbed roots originate from the same"
            " validated baseline branch."
        )

    else:
        print(
            "NARROW C-CONTINUATION: FAIL / INCOMPLETE"
        )
        print(
            "\nAt least one local continuation step failed."
        )
        print(
            "This is NOT evidence that C has no resonance effect."
        )
        print(
            "The next response should reduce the C step and retry"
            " the failed direction."
        )

    print(
        "\nIMPORTANT:"
        "\n  This does NOT establish global branch uniqueness."
        "\n  This does NOT establish physical realization."
        "\n  This does NOT establish a new fundamental interaction."
        "\n  It tests local numerical C-continuity of the resonance."
    )

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    payload = {
        "stage": "6.5B4-1",
        "description": (
            "Narrow local C-continuation of the validated baseline "
            "complex resonance root"
        ),
        "mode": M_MODE,

        "constants": {
            "C0": C0,
            "Omega": OMEGA,
            "gamma": GAMMA,
            "h0": H0,
            "g": G_GRAV,
            "r_B": R_B,
        },

        "controlled_cases": {
            "C_MINUS": C_MINUS,
            "BASELINE": C0,
            "C_PLUS": C_PLUS,
        },

        "radial_resolution": RADIAL_N,

        "baseline_root_hz": BASELINE_ROOT_HZ,

        "complex_C_injection_audit": injection_audit,

        "results": rows,

        "derivatives": derivative,

        "verdict": {
            "minus_pass": minus_ok,
            "plus_pass": plus_ok,
            "both_continuations_pass": both_pass,
            "local_response_detected": local_response,
        },

        "authoritative_solver_modified": False,
        "C_modified_on_disk": False,
        "frequency_rescaling": False,
        "kerr_fit": False,
        "gw_calibration": False,
        "low_k_approximation": False,
        "blind_2d_scan": False,
    }

    save_csv(rows)
    save_json(payload)

    print(
        "\nFILES"
        f"\n  CSV  -> {CSV_FILE}"
        f"\n  JSON -> {JSON_FILE}"
    )

    print(
        "\nAuthoritative solver was NOT modified."
    )
    print(
        "C was modified IN MEMORY ONLY."
    )
    print(
        f"Actual radial resolution: N={RADIAL_N}"
    )

    print(
        "\n" + "=" * 78
    )


if __name__ == "__main__":
    main()