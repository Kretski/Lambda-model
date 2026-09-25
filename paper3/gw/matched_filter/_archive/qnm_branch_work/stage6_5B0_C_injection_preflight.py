
"""
stage6_5B0_C_injection_preflight.py
===================================

STAGE 6.5B-0
C-INJECTION PREFLIGHT

Purpose
-------
Before attempting any complex resonance root search, prove that the
controlled parameter C is actually reaching the authoritative solver.

This is deliberately a SMALL test.

It does NOT:
    - modify stage5I_3_resonance.py
    - use AST rewriting
    - modify function defaults
    - fit Kerr
    - calibrate to GW data
    - rescale frequencies
    - use a low-k approximation
    - search for complex resonances

It ONLY evaluates the authoritative light-ring machinery with:

    C = 0
    C = C0 / 2
    C = C0
    C = 2 C0

All other Root-B parameters remain fixed at their authoritative values.

Expected qualitative result
---------------------------
For the validated baseline model:

    C = 0       -> NO INTERIOR LIGHT RING
    C = C0/2    -> light ring at a different radius/frequency
    C = C0      -> baseline light ring
    C = 2*C0    -> light ring at a different radius/frequency

This preflight MUST pass before Stage 6.5B resonance root solving.

Authoritative solver:
    stage5I_3_resonance.py
"""

from __future__ import annotations

import csv
import inspect
import json
import sys
from pathlib import Path

import numpy as np


# ============================================================================
# PATHS
# ============================================================================

HERE = Path(__file__).resolve().parent

RES_FILE = HERE / "stage5I_3_resonance.py"

OUTPUT_DIR = HERE / "stage6_5B0_C_injection_preflight"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = OUTPUT_DIR / "stage6_5B0_C_injection_results.csv"
JSON_FILE = OUTPUT_DIR / "stage6_5B0_C_injection_results.json"


# ============================================================================
# EXPERIMENT PARAMETERS
# ============================================================================

M = -12

# Validated baseline circulation.
BASE_C = 7.690000000000e-04


CASES = [
    ("C_ZERO", 0.0),
    ("C_HALF", 0.5 * BASE_C),
    ("BASELINE", BASE_C),
    ("C_DOUBLE", 2.0 * BASE_C),
]


# ============================================================================
# LOAD AUTHORITATIVE SOLVER
# ============================================================================

def load_authoritative_solver():
    """
    Import stage5I_3_resonance.py normally.

    IMPORTANT:
        The authoritative solver is NOT modified.

    We deliberately do NOT:
        - exec() modified source
        - AST-patch constants
        - alter module globals
        - alter function defaults
    """

    if not RES_FILE.exists():
        raise FileNotFoundError(
            f"Authoritative solver not found:\n{RES_FILE}"
        )

    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))

    module_name = "stage5I_3_resonance"

    # Remove an accidentally cached copy from a previous interactive run.
    if module_name in sys.modules:
        del sys.modules[module_name]

    import importlib

    solver = importlib.import_module(module_name)

    return solver


# ============================================================================
# HELPER: SHOW FUNCTION SIGNATURE
# ============================================================================

def show_signature(func, name):
    """
    Print the actual authoritative Python signature.

    This is a diagnostic only.
    """

    try:
        sig = inspect.signature(func)
    except Exception as exc:
        print(f"  {name} signature: <unavailable: {exc}>")
        return None

    print(f"  {name}{sig}")

    return sig


# ============================================================================
# HELPER: CALL FIND_LIGHT_RING WITH EXPLICIT C
# ============================================================================

def call_find_light_ring(solver, m, C_value):
    """
    Call the authoritative find_light_ring() with C explicitly supplied.

    We inspect the actual signature first.

    Expected authoritative signature:

        find_light_ring(
            m,
            C=C_METHODS,
            Omega=OMEGA,
            gamma=GAMMA,
            h0=H0,
            g=G_GRAV,
            r_min=...,
            r_max=...
        )

    If C is not an explicit parameter, we FAIL HARD.

    This is intentional.

    We do NOT silently modify globals or function defaults because that
    would make the causal experiment ambiguous.
    """

    func = solver.find_light_ring

    sig = inspect.signature(func)

    if "C" not in sig.parameters:
        raise RuntimeError(
            "AUTHORITATIVE API ERROR:\n"
            "find_light_ring() does not expose an explicit 'C' parameter.\n"
            "Refusing to modify globals/defaults."
        )

    # ------------------------------------------------------------------
    # Use the authoritative values for everything except C.
    #
    # We pass only parameters that actually exist in the authoritative
    # function signature.
    # ------------------------------------------------------------------

    kwargs = {
        "C": float(C_value),
    }

    # Keep all non-C parameters at authoritative values.
    #
    # This is not necessary for the current solver because its defaults
    # already contain these values, but passing them explicitly makes the
    # experiment auditable and protects against accidental environment
    # changes.
    #
    # IMPORTANT:
    # We obtain these from the authoritative module itself.
    #

    authoritative_parameter_map = {
        "Omega": "OMEGA",
        "gamma": "GAMMA",
        "h0": "H0",
        "g": "G_GRAV",
        "r_min": None,
        "r_max": "R_B",
    }

    for parameter_name, module_name in authoritative_parameter_map.items():

        if parameter_name not in sig.parameters:
            continue

        if module_name is None:
            # Do not override r_min if the authoritative default is used.
            continue

        if not hasattr(solver, module_name):
            raise RuntimeError(
                f"Authoritative solver is missing expected constant "
                f"{module_name!r}."
            )

        kwargs[parameter_name] = getattr(
            solver,
            module_name,
        )

    result = func(
        m,
        **kwargs,
    )

    return result


# ============================================================================
# VERIFY FUNCTION DEFAULTS HAVE NOT BEEN MODIFIED
# ============================================================================

def inspect_default_C(solver):
    """
    Show the C-related default in find_light_ring().

    This is informational.

    The important point is that we do NOT change it.

    Explicit C is supplied at call time.
    """

    func = solver.find_light_ring
    sig = inspect.signature(func)

    if "C" not in sig.parameters:
        return None

    parameter = sig.parameters["C"]

    return parameter.default


# ============================================================================
# RUN ONE CASE
# ============================================================================

def run_case(solver, label, C_value):
    """
    Run one controlled C case.

    No resonance search occurs here.
    """

    print()
    print("-" * 78)
    print(f"{label}: C={C_value:.12e} m^2/s")
    print("-" * 78)

    # ------------------------------------------------------------------
    # Authoritative constants
    # ------------------------------------------------------------------

    C_methods = getattr(
        solver,
        "C_METHODS",
        None,
    )

    C_table = getattr(
        solver,
        "C_TABLE",
        None,
    )

    Omega = getattr(
        solver,
        "OMEGA",
        None,
    )

    gamma = getattr(
        solver,
        "GAMMA",
        None,
    )

    h0 = getattr(
        solver,
        "H0",
        None,
    )

    g_grav = getattr(
        solver,
        "G_GRAV",
        None,
    )

    r_B = getattr(
        solver,
        "R_B",
        None,
    )

    print()
    print("authoritative module constants:")
    print(f"  C_METHODS = {C_methods}")
    print(f"  C_TABLE   = {C_table}")
    print(f"  OMEGA     = {Omega}")
    print(f"  GAMMA     = {gamma}")
    print(f"  H0        = {h0}")
    print(f"  G_GRAV    = {g_grav}")
    print(f"  R_B       = {r_B}")

    # ------------------------------------------------------------------
    # Function signature
    # ------------------------------------------------------------------

    print()
    print("authoritative function signature:")
    show_signature(
        solver.find_light_ring,
        "find_light_ring",
    )

    # ------------------------------------------------------------------
    # Default C -- diagnostic only
    # ------------------------------------------------------------------

    default_C = inspect_default_C(solver)

    print()
    print(
        "find_light_ring default C = "
        f"{default_C!r}"
    )

    print(
        "explicit C supplied to this case = "
        f"{C_value:.12e}"
    )

    # ------------------------------------------------------------------
    # CRITICAL TEST
    #
    # Explicit C is supplied here.
    # ------------------------------------------------------------------

    try:

        r_sp, omega_lr = call_find_light_ring(
            solver,
            M,
            C_value,
        )

    except Exception as exc:

        print()
        print("  CALL FAILED")
        print(f"  {type(exc).__name__}: {exc}")

        return {
            "case": label,
            "C_requested": float(C_value),
            "C_methods_module": (
                None
                if C_methods is None
                else float(C_methods)
            ),
            "C_table_module": (
                None
                if C_table is None
                else float(C_table)
            ),
            "light_ring": False,
            "r_sp_m": None,
            "f_lr_hz": None,
            "call_ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    # ------------------------------------------------------------------
    # No interior light ring
    # ------------------------------------------------------------------

    if r_sp is None or omega_lr is None:

        print()
        print("  LIGHT RING: NONE")

        return {
            "case": label,
            "C_requested": float(C_value),
            "C_methods_module": (
                None
                if C_methods is None
                else float(C_methods)
            ),
            "C_table_module": (
                None
                if C_table is None
                else float(C_table)
            ),
            "light_ring": False,
            "r_sp_m": None,
            "f_lr_hz": None,
            "call_ok": True,
            "error_type": None,
            "error": None,
        }

    # ------------------------------------------------------------------
    # Valid light ring
    # ------------------------------------------------------------------

    r_sp = float(np.real(r_sp))
    omega_lr = float(np.real(omega_lr))

    f_lr = omega_lr / (2.0 * np.pi)

    print()
    print(
        f"  LIGHT RING: r_sp={r_sp * 1e3:.12f} mm"
    )

    print(
        f"  FREQUENCY:  f_lr={f_lr:.12f} Hz"
    )

    return {
        "case": label,
        "C_requested": float(C_value),
        "C_methods_module": (
            None
            if C_methods is None
            else float(C_methods)
        ),
        "C_table_module": (
            None
            if C_table is None
            else float(C_table)
        ),
        "light_ring": True,
        "r_sp_m": r_sp,
        "f_lr_hz": f_lr,
        "omega_lr_rad_s": omega_lr,
        "call_ok": True,
        "error_type": None,
        "error": None,
    }


# ============================================================================
# VALIDATION
# ============================================================================

def validate_results(results):
    """
    Strict validation of the C-only preflight.

    Required:

        1. All calls succeed.
        2. Baseline has a light ring.
        3. C=0 has no interior light ring.
        4. C/2 has a light ring.
        5. 2C has a light ring.
        6. C/2, C, 2C do NOT all return the same light-ring location.
        7. Baseline agrees with the known validated benchmark.
    """

    by_case = {
        row["case"]: row
        for row in results
    }

    failures = []

    # ------------------------------------------------------------------
    # 1. All calls successful
    # ------------------------------------------------------------------

    for row in results:

        if not row["call_ok"]:

            failures.append(
                f"{row['case']}: authoritative call failed"
            )

    # ------------------------------------------------------------------
    # 2. Required cases exist
    # ------------------------------------------------------------------

    required_cases = {
        "C_ZERO",
        "C_HALF",
        "BASELINE",
        "C_DOUBLE",
    }

    missing = required_cases - set(by_case)

    for label in sorted(missing):

        failures.append(
            f"missing case: {label}"
        )

    if failures:
        return False, failures

    # ------------------------------------------------------------------
    # 3. C=0 must have NO interior light ring
    # ------------------------------------------------------------------

    if by_case["C_ZERO"]["light_ring"]:

        failures.append(
            "C_ZERO unexpectedly has an interior light ring"
        )

    # ------------------------------------------------------------------
    # 4. C/2 must have light ring
    # ------------------------------------------------------------------

    if not by_case["C_HALF"]["light_ring"]:

        failures.append(
            "C_HALF has no interior light ring"
        )

    # ------------------------------------------------------------------
    # 5. Baseline must have light ring
    # ------------------------------------------------------------------

    if not by_case["BASELINE"]["light_ring"]:

        failures.append(
            "BASELINE has no interior light ring"
        )

    # ------------------------------------------------------------------
    # 6. 2C must have light ring
    # ------------------------------------------------------------------

    if not by_case["C_DOUBLE"]["light_ring"]:

        failures.append(
            "C_DOUBLE has no interior light ring"
        )

    # Stop if any required light-ring result is missing.
    if failures:
        return False, failures

    # ------------------------------------------------------------------
    # 7. Baseline numerical benchmark
    #
    # Known validated Stage 6.5 result:
    #
    # r_sp ~= 20.087461 mm
    # f_lr  ~= 8.8354431574 Hz
    #
    # We use deliberately reasonable tolerances here.
    # ------------------------------------------------------------------

    baseline_r = by_case["BASELINE"]["r_sp_m"]
    baseline_f = by_case["BASELINE"]["f_lr_hz"]

    expected_r = 20.087461e-3
    expected_f = 8.8354431574

    r_error = abs(baseline_r - expected_r)
    f_error = abs(baseline_f - expected_f)

    if r_error > 5.0e-7:

        failures.append(
            "BASELINE r_sp does not reproduce the validated "
            f"benchmark: error={r_error:.3e} m"
        )

    if f_error > 1.0e-6:

        failures.append(
            "BASELINE f_lr does not reproduce the validated "
            f"benchmark: error={f_error:.3e} Hz"
        )

    # ------------------------------------------------------------------
    # 8. C/2 and 2C must produce a genuine response
    # ------------------------------------------------------------------

    r_half = by_case["C_HALF"]["r_sp_m"]
    r_base = by_case["BASELINE"]["r_sp_m"]
    r_double = by_case["C_DOUBLE"]["r_sp_m"]

    f_half = by_case["C_HALF"]["f_lr_hz"]
    f_base = by_case["BASELINE"]["f_lr_hz"]
    f_double = by_case["C_DOUBLE"]["f_lr_hz"]

    # If all locations are effectively identical, C is not reaching
    # the calculation.
    r_spread = max(
        r_half,
        r_base,
        r_double,
    ) - min(
        r_half,
        r_base,
        r_double,
    )

    f_spread = max(
        f_half,
        f_base,
        f_double,
    ) - min(
        f_half,
        f_base,
        f_double,
    )

    if r_spread < 1.0e-9:

        failures.append(
            "C/2, C, and 2C produce indistinguishable r_sp"
        )

    if f_spread < 1.0e-9:

        failures.append(
            "C/2, C, and 2C produce indistinguishable f_lr"
        )

    # ------------------------------------------------------------------
    # 9. Requested C values must actually differ
    # ------------------------------------------------------------------

    requested = [
        by_case["C_ZERO"]["C_requested"],
        by_case["C_HALF"]["C_requested"],
        by_case["BASELINE"]["C_requested"],
        by_case["C_DOUBLE"]["C_requested"],
    ]

    if len(set(requested)) != 4:

        failures.append(
            "Internal test definition error: requested C values "
            "are not distinct"
        )

    return len(failures) == 0, failures


# ============================================================================
# SAVE
# ============================================================================

def save_results(results, passed, failures):
    """
    Save both CSV and JSON.
    """

    fieldnames = [
        "case",
        "C_requested",
        "C_methods_module",
        "C_table_module",
        "light_ring",
        "r_sp_m",
        "f_lr_hz",
        "omega_lr_rad_s",
        "call_ok",
        "error_type",
        "error",
    ]

    with open(
        CSV_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in results:
            writer.writerow(row)

    payload = {
        "stage": "6.5B-0",
        "description": "C injection preflight",
        "authoritative_solver": str(RES_FILE),
        "mode_m": M,
        "baseline_C": BASE_C,
        "authoritative_solver_modified": False,
        "passed": bool(passed),
        "failures": failures,
        "results": results,
    }

    with open(
        JSON_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            indent=2,
        )


# ============================================================================
# SUMMARY
# ============================================================================

def print_summary(results, passed, failures):
    """

    Print final scientific interpretation.

    """

    print()
    print("=" * 78)
    print("STAGE 6.5B-0 — C-INJECTION PREFLIGHT SUMMARY")
    print("=" * 78)

    print()
    print(
        f"{'case':>12} "
        f"{'C requested':>16} "
        f"{'LR':>7} "
        f"{'r_sp [mm]':>18} "
        f"{'f_lr [Hz]':>18}"
    )

    print("-" * 78)

    for row in results:

        if row["light_ring"]:

            r_text = (
                f"{row['r_sp_m'] * 1e3:.12f}"
            )

            f_text = (
                f"{row['f_lr_hz']:.12f}"
            )

        else:

            r_text = "NONE"
            f_text = "NONE"

        print(
            f"{row['case']:>12} "
            f"{row['C_requested']:16.9e} "
            f"{str(row['light_ring']):>7} "
            f"{r_text:>18} "
            f"{f_text:>18}"
        )

    print()
    print("=" * 78)

    if passed:

        print("PREFLIGHT: PASS")
        print()
        print(
            "Explicit C injection is reaching the authoritative "
            "light-ring calculation."
        )
        print()
        print(
            "The C=0 / C0/2 / C0 / 2C causal control is therefore "
            "ready for the next stage."
        )
        print()
        print(
            "NEXT STEP:"
        )
        print(
            "Run the resonance-level test using the same explicit-C "
            "injection architecture."
        )

    else:

        print("PREFLIGHT: FAIL")
        print()
        print(
            "Do NOT proceed to the resonance root search."
        )
        print()
        print(
            "Failures:"
        )

        for failure in failures:

            print(
                f"  - {failure}"
            )

    print()
    print(
        "No authoritative solver file was modified."
    )

    print("=" * 78)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 78)
    print(
        "STAGE 6.5B-0 — C-INJECTION PREFLIGHT"
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
        "Mode:"
    )
    print(
        f"  m = {M}"
    )

    print()
    print(
        "Baseline circulation:"
    )
    print(
        f"  C0 = {BASE_C:.12e} m^2/s"
    )

    print()
    print(
        "Controlled cases:"
    )

    for label, C_value in CASES:

        print(
            f"  {label:>10}: "
            f"C = {C_value:.12e}"
        )

    print()
    print(
        "Rules:"
    )
    print(
        "  * authoritative solver loaded normally"
    )
    print(
        "  * no AST rewriting"
    )
    print(
        "  * no modification of globals"
    )
    print(
        "  * no modification of function defaults"
    )
    print(
        "  * only explicit C argument is varied"
    )
    print(
        "  * no resonance root search"
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

    # ------------------------------------------------------------------
    # Load authoritative solver
    # ------------------------------------------------------------------

    print()
    print(
        "-" * 78
    )
    print(
        "LOADING AUTHORITATIVE SOLVER"
    )
    print(
        "-" * 78
    )

    try:

        solver = load_authoritative_solver()

    except Exception as exc:

        print()
        print(
            "FATAL: could not load authoritative solver."
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise

    print()
    print(
        "Loaded:"
    )
    print(
        f"  {solver.__file__}"
    )

    # ------------------------------------------------------------------
    # Check required API
    # ------------------------------------------------------------------

    required = [
        "find_light_ring",
        "C_METHODS",
        "OMEGA",
        "GAMMA",
        "H0",
        "G_GRAV",
        "R_B",
    ]

    missing = [
        name
        for name in required
        if not hasattr(solver, name)
    ]

    if missing:

        raise RuntimeError(
            "Authoritative solver is missing required symbols:\n"
            + "\n".join(
                f"  {name}"
                for name in missing
            )
        )

    # ------------------------------------------------------------------
    # Show authoritative API
    # ------------------------------------------------------------------

    print()
    print(
        "-" * 78
    )
    print(
        "AUTHORITATIVE API CHECK"
    )
    print(
        "-" * 78
    )

    show_signature(
        solver.find_light_ring,
        "find_light_ring",
    )

    default_C = inspect_default_C(solver)

    print()
    print(
        f"find_light_ring default C = {default_C!r}"
    )

    print(
        f"authoritative C_METHODS   = "
        f"{solver.C_METHODS!r}"
    )

    # ------------------------------------------------------------------
    # Run cases
    # ------------------------------------------------------------------

    results = []

    for label, C_value in CASES:

        result = run_case(
            solver,
            label,
            C_value,
        )

        results.append(result)

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    passed, failures = validate_results(
        results
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    save_results(
        results,
        passed,
        failures,
    )

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------

    print_summary(
        results,
        passed,
        failures,
    )

    print()
    print(
        "OUTPUT FILES"
    )
    print(
        f"CSV  -> {CSV_FILE}"
    )
    print(
        f"JSON -> {JSON_FILE}"
    )
    print()

    # ------------------------------------------------------------------
    # Exit status
    # ------------------------------------------------------------------

    if not passed:

        raise SystemExit(1)


if __name__ == "__main__":
    main()

