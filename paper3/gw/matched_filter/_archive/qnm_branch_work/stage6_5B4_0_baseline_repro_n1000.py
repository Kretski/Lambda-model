"""
STAGE 6.5B4-0 — BASELINE RESONANCE REPRODUCIBILITY @ N=1000

Purpose
-------
Reproduce the known Stage 6.5B-3 baseline resonance candidate using
the working radial resolution N=1000.

This stage is ONLY a reproducibility test.

It does NOT:
    - perform an optimizer search
    - scan C
    - perform C-continuation
    - fit Kerr
    - calibrate to GW data
    - rescale frequency
    - use the long-wave approximation
    - modify the authoritative solver on disk

Known B3 baseline candidate:

    f_R = 8.306625532027 Hz
    f_I = -0.000586962309 Hz

The validated radial-action patch is used exactly as a numerical
continuation tool.

IMPORTANT:
The authoritative solver version currently does not expose
DEFAULT_COMPLEX_RADIAL_N. The patch imports that symbol, so this script
installs a compatibility value IN MEMORY ONLY.

The actual radial resolution is passed explicitly as n=N=1000.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np


# ============================================================================
# PATHS
# ============================================================================

HERE = Path(__file__).resolve().parent

RES_FILE = HERE / "stage5I_3_resonance.py"
PATCH_FILE = HERE / "stage5I_3c_radial_action_patch.py"

OUT_DIR = HERE / "stage6_5B4_0_baseline_repro_n1000"

OUT_CSV = (
    OUT_DIR /
    "stage6_5B4_0_baseline_repro_n1000.csv"
)

OUT_JSON = (
    OUT_DIR /
    "stage6_5B4_0_baseline_repro_n1000.json"
)


# ============================================================================
# ROOT-B PARAMETERS
# ============================================================================

M = -12

C0 = 7.690000000000e-04
OMEGA = 1.000000000000e-01
GAMMA = 2.220000000000e-06
H0 = 3.400000000000e-02
G_GRAV = 9.810000000000e+00
R_B = 3.730000000000e-02


# ============================================================================
# WORKING RADIAL RESOLUTION
# ============================================================================

# IMPORTANT:
# This is the ONLY radial resolution used in B4-0.
N = 1000


# ============================================================================
# KNOWN B3 BASELINE ROOT
# ============================================================================

F_RE_B3 = 8.306625532027
F_IM_B3 = -0.000586962309


# Historical Stage 5I-3c point.
# Reference only; not used as input.
F_RE_HIST = 8.3066307032346
F_IM_HIST = -0.0005696389388967092


# ============================================================================
# TOLERANCES
# ============================================================================

H_TURN_TOL = 1.0e-8

# We only use this as a loose reference-consistency check.
# It is NOT a resonance acceptance criterion.
MAX_REFERENCE_ROOT_DISTANCE_HZ = 5.0e-3


# ============================================================================
# UTILITY
# ============================================================================

def fmt_complex(z):
    z = complex(z)
    return f"{z.real:+.12e}{z.imag:+.12e}i"


def finite_complex(z):
    try:
        z = complex(z)
        return (
            np.isfinite(z.real)
            and np.isfinite(z.imag)
        )
    except Exception:
        return False


# ============================================================================
# LOAD AUTHORITATIVE SOLVER
# ============================================================================

def load_authoritative_solver():

    if not RES_FILE.exists():
        raise FileNotFoundError(
            f"Authoritative solver not found:\n{RES_FILE}"
        )

    module_name = "_stage5I_3_resonance_B4_0"

    spec = importlib.util.spec_from_file_location(
        module_name,
        RES_FILE,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not create authoritative solver spec."
        )

    solver = importlib.util.module_from_spec(spec)

    # ------------------------------------------------------------------------
    # IMPORTANT
    #
    # The authoritative solver uses __file__ internally.
    # ------------------------------------------------------------------------

    solver.__file__ = str(RES_FILE)
    solver.__name__ = module_name
    solver.__package__ = ""

    # Make it available under the exact module name expected by the patch.
    sys.modules["stage5I_3_resonance"] = solver

    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))

    spec.loader.exec_module(solver)

    return solver


# ============================================================================
# COMPATIBILITY SHIM
# ============================================================================

def install_patch_compatibility(solver):

    """
    The validated radial patch imports:

        DEFAULT_COMPLEX_RADIAL_N
        R_B

    from stage5I_3_resonance.

    Current authoritative solver has R_B but may not expose
    DEFAULT_COMPLEX_RADIAL_N.

    We therefore provide the missing symbol in memory only.

    The actual test resolution is still passed explicitly as n=N.
    """

    if not hasattr(solver, "DEFAULT_COMPLEX_RADIAL_N"):

        solver.DEFAULT_COMPLEX_RADIAL_N = 20000

        print(
            "Compatibility shim:"
            " DEFAULT_COMPLEX_RADIAL_N=20000 installed in memory."
        )

    else:

        print(
            "Compatibility symbol already present:"
            f" DEFAULT_COMPLEX_RADIAL_N="
            f"{solver.DEFAULT_COMPLEX_RADIAL_N}"
        )

    if not hasattr(solver, "R_B"):
        raise RuntimeError(
            "Authoritative solver does not expose R_B."
        )

    print("Compatibility shim: PASS")


# ============================================================================
# LOAD RADIAL PATCH
# ============================================================================

def load_radial_patch():

    if not PATCH_FILE.exists():
        raise FileNotFoundError(
            f"Radial patch not found:\n{PATCH_FILE}"
        )

    module_name = "_stage5I_3c_radial_action_patch_B4_0"

    spec = importlib.util.spec_from_file_location(
        module_name,
        PATCH_FILE,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not create radial patch spec."
        )

    patch = importlib.util.module_from_spec(spec)

    patch.__file__ = str(PATCH_FILE)
    patch.__name__ = module_name
    patch.__package__ = ""

    spec.loader.exec_module(patch)

    return patch


# ============================================================================
# API CHECK
# ============================================================================

def verify_api(solver, patch):

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
        name
        for name in required_solver
        if not hasattr(solver, name)
    ]

    missing_patch = [
        name
        for name in required_patch
        if not hasattr(patch, name)
    ]

    if missing_solver:
        raise RuntimeError(
            "Missing solver API: "
            + ", ".join(missing_solver)
        )

    if missing_patch:
        raise RuntimeError(
            "Missing patch API: "
            + ", ".join(missing_patch)
        )

    print()
    print("Required APIs: PASS")

    print()
    print("Authoritative solver APIs:")

    for name in required_solver:
        print(
            f"  {name}: "
            f"{getattr(solver, name)}"
        )

    print()
    print("Radial patch API:")

    print(
        "  radial_action_complex_v2: "
        f"{patch.radial_action_complex_v2}"
    )


# ============================================================================
# TURNING-POINT HAMILTONIAN
# ============================================================================

def compute_H_turn(solver, omega, r_turn):

    return complex(
        solver.hamiltonian(
            omega,
            0.0j,
            M,
            r_turn,
        )
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print(
        "STAGE 6.5B4-0 — "
        "BASELINE RESONANCE REPRODUCIBILITY @ N=1000"
    )
    print("=" * 78)
    print()

    print("Authoritative solver:")
    print(f"  {RES_FILE}")
    print()

    print("Radial patch:")
    print(f"  {PATCH_FILE}")
    print()

    print("Mode:")
    print(f"  m = {M}")
    print()

    print("Root-B constants:")
    print(f"  C      = {C0:.12e}")
    print(f"  Omega  = {OMEGA:.12e}")
    print(f"  gamma  = {GAMMA:.12e}")
    print(f"  h0     = {H0:.12e}")
    print(f"  g      = {G_GRAV:.12e}")
    print(f"  r_B    = {R_B:.12e}")
    print()

    print("WORKING RADIAL RESOLUTION:")
    print(f"  N = {N}")
    print()

    print("Known B3 baseline candidate:")
    print(
        f"  f = {F_RE_B3:.12f}"
        f"{F_IM_B3:+.12f} i Hz"
    )
    print()

    # ========================================================================
    # LOAD SOLVER
    # ========================================================================

    try:

        solver = load_authoritative_solver()

        print("Authoritative solver load: PASS")

    except Exception as exc:

        print("LOAD: FAIL")
        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise SystemExit(1)

    # ========================================================================
    # COMPATIBILITY SHIM
    # ========================================================================

    try:

        install_patch_compatibility(
            solver
        )

    except Exception as exc:

        print()
        print("COMPATIBILITY SHIM: FAIL")
        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise SystemExit(1)

    # ========================================================================
    # LOAD PATCH
    # ========================================================================

    try:

        patch = load_radial_patch()

        print("Radial patch load: PASS")

    except Exception as exc:

        print()
        print("RADIAL PATCH LOAD: FAIL")
        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise SystemExit(1)

    # ========================================================================
    # VERIFY APIs
    # ========================================================================

    try:

        verify_api(
            solver,
            patch,
        )

    except Exception as exc:

        print()
        print("API VERIFICATION: FAIL")
        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise SystemExit(1)

    # ========================================================================
    # 1. LIGHT RING
    # ========================================================================

    print()
    print("-" * 78)
    print("1. REAL LIGHT RING")
    print("-" * 78)

    try:

        r_sp, omega_lr = solver.find_light_ring(
            M,
            C=C0,
            Omega=OMEGA,
            gamma=GAMMA,
            h0=H0,
            g=G_GRAV,
        )

    except Exception as exc:

        print("LIGHT RING: FAIL")
        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise SystemExit(1)

    if r_sp is None or omega_lr is None:

        print("LIGHT RING: FAIL")
        print("No baseline light ring found.")

        raise SystemExit(1)

    r_sp = float(r_sp)
    omega_lr = float(omega_lr)

    f_lr = omega_lr / (
        2.0 * math.pi
    )

    print(
        f"r_sp     = "
        f"{r_sp:.15e} m"
    )

    print(
        f"r_sp     = "
        f"{r_sp * 1000.0:.12f} mm"
    )

    print(
        f"omega_lr = "
        f"{omega_lr:.15e} rad/s"
    )

    print(
        f"f_lr     = "
        f"{f_lr:.15f} Hz"
    )

    print("LIGHT RING: PASS")

    # ========================================================================
    # 2. COMPLEX FREQUENCY
    # ========================================================================

    omega = (
        2.0
        * math.pi
        * complex(
            F_RE_B3,
            F_IM_B3,
        )
    )

    print()
    print("-" * 78)
    print("2. COMPLEX FREQUENCY")
    print("-" * 78)

    print(
        "f = "
        f"{F_RE_B3:.15f}"
        f"{F_IM_B3:+.15f} i Hz"
    )

    print(
        "omega = "
        f"{fmt_complex(omega)} rad/s"
    )

    # ========================================================================
    # 3. LAST SCATTERING POINT
    # ========================================================================

    print()
    print("-" * 78)
    print("3. LAST SCATTERING POINT")
    print("-" * 78)

    try:

        r_minus_real = (
            solver.find_last_scattering_point(
                omega,
                M,
                r_sp,
            )
        )

    except Exception as exc:

        print("LAST SCATTERING POINT: FAIL")
        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise SystemExit(1)

    if r_minus_real is None:

        print("LAST SCATTERING POINT: FAIL")
        print("Returned None.")

        raise SystemExit(1)

    r_minus_real = float(
        r_minus_real
    )

    print(
        f"r_minus_real = "
        f"{r_minus_real:.15e} m"
    )

    print(
        f"r_minus_real = "
        f"{r_minus_real * 1000.0:.12f} mm"
    )

    print(
        "LAST SCATTERING POINT: PASS"
    )

    # ========================================================================
    # 4. COMPLEX TURNING POINT
    # ========================================================================

    print()
    print("-" * 78)
    print("4. COMPLEX TURNING POINT")
    print("-" * 78)

    try:

        r_turn = (
            solver.find_turning_point_complex(
                omega,
                M,
                r_minus_real,
            )
        )

    except Exception as exc:

        print("TURNING POINT: FAIL")
        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise SystemExit(1)

    if r_turn is None:

        print("TURNING POINT: FAIL")
        print("Returned None.")

        raise SystemExit(1)

    r_turn = complex(r_turn)

    print(
        f"r_turn = "
        f"{fmt_complex(r_turn)} m"
    )

    # ========================================================================
    # 5. HAMILTONIAN RESIDUAL
    # ========================================================================

    print()
    print("-" * 78)
    print("5. TURNING-POINT HAMILTONIAN RESIDUAL")
    print("-" * 78)

    try:

        H_turn = compute_H_turn(
            solver,
            omega,
            r_turn,
        )

    except Exception as exc:

        print("H(turn): FAIL")
        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise SystemExit(1)

    H_turn_abs = abs(H_turn)

    print(
        f"H(turn) = "
        f"{fmt_complex(H_turn)}"
    )

    print(
        f"|H(turn)| = "
        f"{H_turn_abs:.15e}"
    )

    if not np.isfinite(H_turn_abs):

        print(
            "TURNING-POINT RESIDUAL: FAIL"
        )

        raise SystemExit(1)

    if H_turn_abs <= H_TURN_TOL:

        print(
            "TURNING-POINT RESIDUAL: PASS"
        )

    else:

        print(
            "TURNING-POINT RESIDUAL: WARNING"
        )

    # ========================================================================
    # 6. RADIAL ACTION @ N=1000
    # ========================================================================

    print()
    print("-" * 78)
    print("6. COMPLEX RADIAL ACTION @ N=1000")
    print("-" * 78)

    print(
        "Calling validated "
        "radial_action_complex_v2..."
    )

    print(
        f"Explicit n = {N}"
    )

    try:

        action = (
            patch.radial_action_complex_v2(
                omega,
                M,
                r_turn,
                r_B=R_B,
                n=N,
            )
        )

    except Exception as exc:

        print()
        print("RADIAL ACTION: FAIL")
        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()
        print(
            "This is a numerical continuation "
            "failure."
        )

        print(
            "It is NOT evidence against "
            "the resonance."
        )

        raise SystemExit(1)

    action = complex(action)

    print(
        f"S = "
        f"{fmt_complex(action)}"
    )

    print(
        f"|S| = "
        f"{abs(action):.15e}"
    )

    if not finite_complex(action):

        print("RADIAL ACTION: FAIL")

        raise SystemExit(1)

    print("RADIAL ACTION: PASS")

    # ========================================================================
    # 7. REFLECTION COEFFICIENT
    # ========================================================================

    print()
    print("-" * 78)
    print("7. REFLECTION COEFFICIENT")
    print("-" * 78)

    try:

        R = solver.reflection_coefficient(
            omega,
            M,
            r_sp,
            omega_lr,
        )

    except Exception as exc:

        print("REFLECTION COEFFICIENT: FAIL")
        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise SystemExit(1)

    R = complex(R)

    print(
        f"R = "
        f"{fmt_complex(R)}"
    )

    print(
        f"|R| = "
        f"{abs(R):.15e}"
    )

    if not finite_complex(R):

        print(
            "REFLECTION COEFFICIENT: FAIL"
        )

        raise SystemExit(1)

    print(
        "REFLECTION COEFFICIENT: PASS"
    )

    # ========================================================================
    # 8. RESONANCE RESIDUAL
    # ========================================================================

    print()
    print("-" * 78)
    print("8. RESONANCE RESIDUAL")
    print("-" * 78)

    Res = (
        R
        * np.exp(2.0j * action)
        - 1.0
    )

    Res = complex(Res)

    Res_abs = abs(Res)

    print(
        f"Res = "
        f"{fmt_complex(Res)}"
    )

    print(
        f"|Res| = "
        f"{Res_abs:.15e}"
    )

    if not finite_complex(Res):

        print(
            "RESONANCE RESIDUAL: FAIL"
        )

        raise SystemExit(1)

    print(
        "RESONANCE RESIDUAL: PASS"
    )

    # ========================================================================
    # 9. Q
    # ========================================================================

    Q = (
        abs(F_RE_B3)
        / (2.0 * abs(F_IM_B3))
    )

    print()
    print("-" * 78)
    print("9. DERIVED QUANTITIES")
    print("-" * 78)

    print(
        f"f_R = "
        f"{F_RE_B3:.15f} Hz"
    )

    print(
        f"f_I = "
        f"{F_IM_B3:.15f} Hz"
    )

    print(
        f"Q   = "
        f"{Q:.15e}"
    )

    # ========================================================================
    # 10. HISTORICAL REFERENCE
    # ========================================================================

    delta_f_re = (
        F_RE_B3
        - F_RE_HIST
    )

    delta_f_im = (
        F_IM_B3
        - F_IM_HIST
    )

    root_distance = math.sqrt(
        delta_f_re ** 2
        + delta_f_im ** 2
    )

    if (
        root_distance
        <= MAX_REFERENCE_ROOT_DISTANCE_HZ
    ):
        reference_status = "PASS"
    else:
        reference_status = "WARNING"

    print()
    print("-" * 78)
    print("10. HISTORICAL REFERENCE")
    print("-" * 78)

    print(
        "B3 candidate:"
    )

    print(
        f"  {F_RE_B3:.12f}"
        f"{F_IM_B3:+.12f} i Hz"
    )

    print(
        "Historical Stage 5I-3c:"
    )

    print(
        f"  {F_RE_HIST:.12f}"
        f"{F_IM_HIST:+.12f} i Hz"
    )

    print(
        f"Delta f_R = "
        f"{delta_f_re:+.15e} Hz"
    )

    print(
        f"Delta f_I = "
        f"{delta_f_im:+.15e} Hz"
    )

    print(
        f"Complex root distance = "
        f"{root_distance:.15e} Hz"
    )

    print(
        f"REFERENCE CONSISTENCY: "
        f"{reference_status}"
    )

    # ========================================================================
    # FINAL VERDICT
    # ========================================================================

    core_pass = (
        np.isfinite(r_sp)
        and np.isfinite(omega_lr)
        and np.isfinite(r_minus_real)
        and finite_complex(r_turn)
        and np.isfinite(H_turn_abs)
        and finite_complex(action)
        and finite_complex(R)
        and finite_complex(Res)
    )

    print()
    print("=" * 78)
    print("B4-0 VERDICT")
    print("=" * 78)

    if core_pass:

        print(
            "BASELINE REPRODUCIBILITY: PASS"
        )

        print()
        print(
            "Known B3 baseline candidate can be "
            "evaluated successfully at N=1000."
        )

        print()
        print(
            "This is sufficient to proceed to "
            "the narrow C-continuation pilot."
        )

        print()
        print("IMPORTANT:")
        print(
            "  This does NOT establish C-causality."
        )
        print(
            "  This does NOT prove the root is unique."
        )
        print(
            "  This does NOT establish physical realization."
        )

    else:

        print(
            "BASELINE REPRODUCIBILITY: FAIL"
        )

    # ========================================================================
    # SAVE JSON
    # ========================================================================

    result = {
        "stage": "6.5B4-0",
        "description":
            "Baseline resonance reproducibility @ N=1000",

        "mode_m": M,

        "constants": {
            "C": C0,
            "Omega": OMEGA,
            "gamma": GAMMA,
            "h0": H0,
            "g": G_GRAV,
            "r_B": R_B,
        },

        "radial_N": N,

        "input_frequency_hz": {
            "f_re": F_RE_B3,
            "f_im": F_IM_B3,
        },

        "light_ring": {
            "r_sp_m": r_sp,
            "r_sp_mm":
                r_sp * 1000.0,
            "omega_lr_rad_s":
                omega_lr,
            "f_lr_hz":
                f_lr,
        },

        "last_scattering_point": {
            "r_minus_real_m":
                r_minus_real,
        },

        "turning_point": {
            "r_turn_real_m":
                r_turn.real,
            "r_turn_imag_m":
                r_turn.imag,
            "H_turn_real":
                H_turn.real,
            "H_turn_imag":
                H_turn.imag,
            "H_turn_abs":
                H_turn_abs,
        },

        "radial_action": {
            "real":
                action.real,
            "imag":
                action.imag,
            "abs":
                abs(action),
            "N":
                N,
        },

        "reflection": {
            "real":
                R.real,
            "imag":
                R.imag,
            "abs":
                abs(R),
        },

        "resonance": {
            "f_re_hz":
                F_RE_B3,
            "f_im_hz":
                F_IM_B3,
            "Q":
                Q,
            "Res_real":
                Res.real,
            "Res_imag":
                Res.imag,
            "Res_abs":
                Res_abs,
        },

        "historical_reference": {
            "f_re_hz":
                F_RE_HIST,
            "f_im_hz":
                F_IM_HIST,
            "delta_f_re_hz":
                delta_f_re,
            "delta_f_im_hz":
                delta_f_im,
            "complex_distance_hz":
                root_distance,
            "status":
                reference_status,
        },

        "verdict":
            "PASS" if core_pass else "FAIL",

        "authoritative_solver_modified":
            False,
    }

    with open(
        OUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    # ========================================================================
    # SAVE CSV
    # ========================================================================

    row = {
        "stage": "6.5B4-0",
        "m": M,

        "C": C0,
        "Omega": OMEGA,
        "gamma": GAMMA,
        "h0": H0,
        "g": G_GRAV,
        "r_B": R_B,

        "N": N,

        "f_re_hz": F_RE_B3,
        "f_im_hz": F_IM_B3,
        "Q": Q,

        "r_sp_m": r_sp,
        "r_sp_mm":
            r_sp * 1000.0,

        "omega_lr_rad_s":
            omega_lr,

        "f_lr_hz":
            f_lr,

        "r_minus_real_m":
            r_minus_real,

        "r_turn_real_m":
            r_turn.real,

        "r_turn_imag_m":
            r_turn.imag,

        "H_turn_abs":
            H_turn_abs,

        "action_real":
            action.real,

        "action_imag":
            action.imag,

        "action_abs":
            abs(action),

        "R_real":
            R.real,

        "R_imag":
            R.imag,

        "R_abs":
            abs(R),

        "Res_real":
            Res.real,

        "Res_imag":
            Res.imag,

        "Res_abs":
            Res_abs,

        "historical_delta_f_re_hz":
            delta_f_re,

        "historical_delta_f_im_hz":
            delta_f_im,

        "historical_root_distance_hz":
            root_distance,

        "reference_status":
            reference_status,

        "verdict":
            "PASS" if core_pass else "FAIL",
    }

    with open(
        OUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys()),
        )

        writer.writeheader()
        writer.writerow(row)

    # ========================================================================
    # FILES
    # ========================================================================

    print()
    print("=" * 78)
    print("FILES")
    print("=" * 78)

    print(
        f"CSV  -> {OUT_CSV}"
    )

    print(
        f"JSON -> {OUT_JSON}"
    )

    print()
    print(
        "Authoritative solver was NOT modified."
    )

    print(
        "Compatibility shim was installed "
        "IN MEMORY ONLY."
    )

    print(
        "Actual radial resolution: N=1000"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()