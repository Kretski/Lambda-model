"""
STAGE 6.5B-0.9 — STEP-77 COMPLEX CONTINUATION DIAGNOSTIC
==========================================================

Purpose
-------
Diagnose the long-range complex-p continuation failure observed in
STAGE 6.5B-0.8 near continuation step ~77.

This test DOES NOT:
  - vary C
  - search for a resonance root
  - modify the authoritative solver on disk
  - modify solver globals/defaults
  - perform a Kerr fit
  - use GW calibration
  - rescale frequency
  - use the low-k approximation

It tests only numerical continuation robustness.

At the failure region it compares:
  1. previous-point seed
  2. linear predictor
  3. half-step predictor
  4. quarter-step predictor
  5. smaller radial step sizes

The accepted physical branch is NOT changed silently.
Every successful candidate is checked by the Hamiltonian residual.
"""

from pathlib import Path
import sys
import importlib.util
import inspect
import numpy as np


# ============================================================================
# PATHS
# ============================================================================

HERE = Path(__file__).resolve().parent

RES_FILE = HERE / "stage5I_3_resonance.py"
PATCH_FILE = HERE / "stage5I_3c_radial_action_patch.py"


# ============================================================================
# BASELINE CONSTANTS
# ============================================================================

M_MODE = -12

C0 = 7.690000000000e-04
OMEGA0 = 1.000000000000e-01
GAMMA0 = 2.220000000000e-06
H00 = 3.400000000000e-02
G0 = 9.810000000000e+00
R_B = 3.730000000000e-02

# Validated benchmark
F_RE = 8.306630716900
F_IM = -0.000566435000

OMEGA = 2.0 * np.pi * (F_RE + 1j * F_IM)

# We reproduce the region where 6.5B-0.8 failed.
# The previous run failed at approximately:
# r = 0.030292165318095216 + 7.795280621568865e-06 i
#
# The complex turning point was:
# r_turn = 0.030265079519957144 + 7.82540995636762e-06 i

EXPECTED_R_TURN = (
    0.030265079519957144
    + 7.82540995636762e-06j
)

EXPECTED_FAILURE_R = (
    0.030292165318095216
    + 7.795280621568865e-06j
)

# Original radial action resolution.
N_ORIGINAL = 20000

# Diagnostic resolution:
# use the same global path, but explicitly inspect several step sizes.
N_DIAGNOSTIC = 20000

# Number of steps to inspect around the known failure.
WINDOW_STEPS = 12


# ============================================================================
# LOAD AUTHORITATIVE SOLVER
# ============================================================================

def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"File not found:\n{path}")

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for:\n{path}")

    module = importlib.util.module_from_spec(spec)

    # IMPORTANT:
    # The authoritative solver expects __file__ in some contexts.
    module.__file__ = str(path)
    module.__name__ = name

    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


# ============================================================================
# HAMILTONIAN RESIDUAL
# ============================================================================

def h_residual(solver, omega, p, m, r):
    """
    Absolute Hamiltonian residual.
    """
    H = solver.hamiltonian(omega, p, m, r)
    return abs(H)


# ============================================================================
# NEWTON SOLVER WITH MULTIPLE SEEDS
# ============================================================================

def try_seed(solver, omega, m, r, seed):
    """
    Call authoritative solve_p_complex with one explicit seed.

    No modification of solver code.
    """

    try:
        p = solver.solve_p_complex(
            omega,
            m,
            r,
            p_initial=complex(seed),
        )
    except Exception as exc:
        return {
            "success": False,
            "p": None,
            "residual": np.inf,
            "error": repr(exc),
        }

    if p is None:
        return {
            "success": False,
            "p": None,
            "residual": np.inf,
            "error": "solve_p_complex returned None",
        }

    try:
        residual = h_residual(
            solver,
            omega,
            p,
            m,
            r,
        )
    except Exception as exc:
        return {
            "success": False,
            "p": complex(p),
            "residual": np.inf,
            "error": f"Residual evaluation failed: {exc!r}",
        }

    return {
        "success": np.isfinite(residual),
        "p": complex(p),
        "residual": float(residual),
        "error": None,
    }


# ============================================================================
# COMPARE CANDIDATE ROOTS
# ============================================================================

def print_candidate(label, result, p_reference=None):
    if not result["success"]:
        print(f"    {label:24s}: FAIL")
        print(f"        error    = {result['error']}")
        return

    p = result["p"]
    residual = result["residual"]

    if p_reference is None:
        distance = np.nan
    else:
        distance = abs(p - p_reference)

    print(
        f"    {label:24s}: PASS  "
        f"p={p.real:+.12e}{p.imag:+.12e}i  "
        f"|H|={residual:.3e}  "
        f"|dp|={distance:.3e}"
    )


# ============================================================================
# BUILD LOCAL PATH
# ============================================================================

def build_path(r_turn, r_target, n):
    """
    Same straight complex-r path convention used by the diagnostic.

    We intentionally do not call radial_action_complex_v2 here because
    we need to inspect every continuation step individually.
    """

    return np.linspace(
        complex(r_turn),
        complex(r_target),
        int(n),
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 78)
    print("STAGE 6.5B-0.9 — STEP-77 COMPLEX CONTINUATION DIAGNOSTIC")
    print("=" * 78)

    print()
    print("Authoritative solver:")
    print(f"  {RES_FILE}")

    print()
    print("Mode:")
    print(f"  m = {M_MODE}")

    print()
    print("Benchmark:")
    print(
        f"  f = {F_RE:.12f} {F_IM:+.12f} i Hz"
    )
    print(
        f"  omega = {OMEGA.real:.12e}"
        f" {OMEGA.imag:+.12e} i rad/s"
    )

    print()
    print("Rules:")
    print("  * no C variation")
    print("  * no resonance root search")
    print("  * no Kerr fit")
    print("  * no GW calibration")
    print("  * no frequency rescaling")
    print("  * no low-k approximation")
    print("  * no authoritative-file modification")
    print("  * diagnostic only")

    # ----------------------------------------------------------------------
    # LOAD
    # ----------------------------------------------------------------------

    print()
    print("-" * 78)
    print("1. LOADING AUTHORITATIVE SOLVER")
    print("-" * 78)

    solver = load_module(
        RES_FILE,
        "_stage5I3_step77_diagnostic",
    )

    print("  Loaded successfully.")

    required = [
        "hamiltonian",
        "solve_p_complex",
        "find_light_ring",
        "find_last_scattering_point",
        "find_turning_point_complex",
    ]

    print()
    print("Required API:")

    for name in required:
        status = "FOUND" if hasattr(solver, name) else "MISSING"
        print(f"  {name:35s}: {status}")

    missing = [x for x in required if not hasattr(solver, x)]

    if missing:
        raise RuntimeError(
            "Missing authoritative API: " + ", ".join(missing)
        )

    print()
    print("Signatures:")

    print(
        "  solve_p_complex:",
        inspect.signature(solver.solve_p_complex),
    )

    print(
        "  hamiltonian:",
        inspect.signature(solver.hamiltonian),
    )

    # ----------------------------------------------------------------------
    # LIGHT RING
    # ----------------------------------------------------------------------

    print()
    print("-" * 78)
    print("2. BASELINE LIGHT RING")
    print("-" * 78)

    r_sp, omega_lr = solver.find_light_ring(
        M_MODE,
        C=C0,
        Omega=OMEGA0,
        gamma=GAMMA0,
        h0=H00,
        g=G0,
        r_min=1.0e-3,
        r_max=R_B,
    )

    if r_sp is None:
        raise RuntimeError("Baseline light ring not found.")

    print(
        f"  r_sp     = {r_sp * 1e3:.12f} mm"
    )

    print(
        f"  omega_lr = {omega_lr:.12e} rad/s"
    )

    print(
        f"  f_lr     = {omega_lr/(2*np.pi):.12f} Hz"
    )

    # ----------------------------------------------------------------------
    # LAST SCATTERING POINT
    # ----------------------------------------------------------------------

    print()
    print("-" * 78)
    print("3. TURNING-POINT RECONSTRUCTION")
    print("-" * 78)

    r_minus_real = solver.find_last_scattering_point(
        OMEGA,
        M_MODE,
        r_sp,
    )

    if r_minus_real is None:
        raise RuntimeError(
            "find_last_scattering_point returned None."
        )

    print(
        f"  r_minus_real = {r_minus_real:.15f} m"
    )

    r_turn = solver.find_turning_point_complex(
        OMEGA,
        M_MODE,
        r_minus_real,
    )

    if r_turn is None:
        raise RuntimeError(
            "find_turning_point_complex returned None."
        )

    print(
        f"  r_turn = {r_turn.real:.15f}"
        f" {r_turn.imag:+.15e} i m"
    )

    print()
    print("  Difference from previous validated turning point:")

    print(
        f"    Delta r = {abs(r_turn - EXPECTED_R_TURN):.6e} m"
    )

    H_turn = solver.hamiltonian(
        OMEGA,
        0j,
        M_MODE,
        r_turn,
    )

    print(
        f"    |H(turn)| = {abs(H_turn):.6e}"
    )

    # ----------------------------------------------------------------------
    # FULL PATH
    # ----------------------------------------------------------------------

    print()
    print("-" * 78)
    print("4. REPRODUCE CONTINUATION PATH")
    print("-" * 78)

    path = build_path(
        r_turn,
        R_B,
        N_DIAGNOSTIC,
    )

    dr = abs(path[1] - path[0])

    print(
        f"  N = {N_DIAGNOSTIC}"
    )
    print(
        f"  |dr| = {dr:.12e} m"
    )
    print(
        f"  r_start = {path[0]}"
    )
    print(
        f"  r_end   = {path[-1]}"
    )

    # ----------------------------------------------------------------------
    # WALK UNTIL FAILURE
    # ----------------------------------------------------------------------

    print()
    print("-" * 78)
    print("5. LOCATE FIRST FAILURE")
    print("-" * 78)

    p_prev = None
    p_prevprev = None

    failure_index = None
    history = []

    for i in range(1, len(path)):

        r = path[i]

        # First step: use zero seed exactly as original patch does.
        if p_prev is None:
            seeds = [
                ("previous p", 0j),
            ]
        else:

            seeds = [
                ("previous p", p_prev),
            ]

            if p_prevprev is not None:
                predictor = p_prev + (p_prev - p_prevprev)

                seeds.append(
                    ("linear predictor", predictor)
                )

                seeds.append(
                    (
                        "half predictor",
                        p_prev + 0.5 * (p_prev - p_prevprev),
                    )
                )

            # Always include the previous p as primary continuation seed.

        result = None
        chosen_label = None

        for label, seed in seeds:

            trial = try_seed(
                solver,
                OMEGA,
                M_MODE,
                r,
                seed,
            )

            if trial["success"]:
                result = trial
                chosen_label = label
                break

        if result is None:
            failure_index = i

            print()
            print(
                f"  FIRST FAILURE at step {i}"
            )

            print(
                f"    r = {r.real:.15f}"
                f" {r.imag:+.15e} i m"
            )

            print(
                f"    distance from r_turn = "
                f"{abs(r-r_turn):.12e} m"
            )

            break

        p_current = result["p"]

        history.append(
            {
                "i": i,
                "r": r,
                "p": p_current,
                "residual": result["residual"],
                "seed": chosen_label,
            }
        )

        p_prevprev = p_prev
        p_prev = p_current

    if failure_index is None:

        print()
        print("  FULL PATH COMPLETED.")
        print("  No continuation failure detected.")

        print()
        print("=" * 78)
        print("DIAGNOSTIC VERDICT: NO FAILURE REPRODUCED")
        print("=" * 78)

        return

    # ----------------------------------------------------------------------
    # LOCAL HISTORY
    # ----------------------------------------------------------------------

    print()
    print("-" * 78)
    print("6. LOCAL HISTORY AROUND FAILURE")
    print("-" * 78)

    start = max(
        0,
        len(history) - WINDOW_STEPS,
    )

    for row in history[start:]:

        r = row["r"]
        p = row["p"]

        print(
            f"  step={row['i']:6d}  "
            f"r={r.real:.12f}  "
            f"p={p.real:+.8e}{p.imag:+.8e}i  "
            f"|H|={row['residual']:.3e}  "
            f"seed={row['seed']}"
        )

    # ----------------------------------------------------------------------
    # EXACT FAILURE POINT — MULTIPLE SEEDS
    # ----------------------------------------------------------------------

    print()
    print("-" * 78)
    print("7. MULTI-SEED TEST AT FAILURE")
    print("-" * 78)

    i = failure_index
    r_fail = path[i]

    print(
        f"  step = {i}"
    )

    print(
        f"  r    = {r_fail.real:.15f}"
        f" {r_fail.imag:+.15e} i m"
    )

    if len(history) >= 1:
        p1 = history[-1]["p"]
    else:
        p1 = 0j

    if len(history) >= 2:
        p2 = history[-2]["p"]
    else:
        p2 = p1

    predictor = p1 + (p1 - p2)

    seeds = [
        ("previous p", p1),
        ("linear predictor", predictor),
        ("half predictor",
         p1 + 0.5 * (p1 - p2)),
        ("quarter predictor",
         p1 + 0.25 * (p1 - p2)),
        ("zero", 0j),
    ]

    for label, seed in seeds:
        result = try_seed(
            solver,
            OMEGA,
            M_MODE,
            r_fail,
            seed,
        )

        print_candidate(
            label,
            result,
            p_reference=p1,
        )

    # ----------------------------------------------------------------------
    # REDUCED STEP TEST
    # ----------------------------------------------------------------------

    print()
    print("-" * 78)
    print("8. REDUCED-STEP TEST")
    print("-" * 78)

    print(
        "Testing whether the failure is caused simply by"
    )
    print(
        "the radial step being too large."
    )

    print()

    r_prev = path[i - 1]

    total_delta = r_fail - r_prev

    print(
        f"  original step |dr| = {abs(total_delta):.12e} m"
    )

    for divisor in [2, 4, 8, 16]:

        print()
        print(
            f"  --- subdivision factor {divisor} ---"
        )

        p_start = p1

        success_count = 0

        for j in range(1, divisor + 1):

            rr = (
                r_prev
                + total_delta * (j / divisor)
            )

            result = try_seed(
                solver,
                OMEGA,
                M_MODE,
                rr,
                p_start,
            )

            if not result["success"]:

                print(
                    f"    substep {j}/{divisor}: FAIL"
                )
                print(
                    f"      r = {rr}"
                )
                print(
                    f"      error = {result['error']}"
                )

                break

            p_new = result["p"]

            print(
                f"    substep {j}/{divisor}: PASS  "
                f"|H|={result['residual']:.3e}  "
                f"|dp|={abs(p_new-p_start):.6e}"
            )

            p_start = p_new
            success_count += 1

        print(
            f"    successful substeps = "
            f"{success_count}/{divisor}"
        )

    # ----------------------------------------------------------------------
    # BRANCH COMPARISON
    # ----------------------------------------------------------------------

    print()
    print("-" * 78)
    print("9. BRANCH CONSISTENCY CHECK")
    print("-" * 78)

    print(
        "The purpose here is NOT to select a new branch."
    )

    print(
        "We only report whether successful seeds converge"
    )

    print(
        "to mutually consistent p values."
    )

    candidate_results = []

    for label, seed in seeds:

        result = try_seed(
            solver,
            OMEGA,
            M_MODE,
            r_fail,
            seed,
        )

        if result["success"]:
            candidate_results.append(
                (label, result["p"], result["residual"])
            )

    if len(candidate_results) == 0:

        print("  No successful candidate roots.")

    else:

        for label, p, residual in candidate_results:

            print(
                f"  {label:24s} "
                f"p={p.real:+.12e}"
                f"{p.imag:+.12e}i  "
                f"|H|={residual:.3e}"
            )

        if len(candidate_results) >= 2:

            ref = candidate_results[0][1]

            print()
            print("  Pairwise distance from first successful root:")

            for label, p, residual in candidate_results[1:]:

                print(
                    f"    {label:24s} "
                    f"|dp|={abs(p-ref):.6e}"
                )

    # ----------------------------------------------------------------------
    # VERDICT
    # ----------------------------------------------------------------------

    print()
    print("=" * 78)
    print("DIAGNOSTIC VERDICT")
    print("=" * 78)

    if candidate_results:

        min_res = min(
            x[2] for x in candidate_results
        )

        if min_res < 1e-8:

            print()
            print(
                "SUCCESSFUL ROOT(S) FOUND AT THE FAILURE POINT."
            )

            print()
            print(
                "Interpretation:"
            )

            print(
                "  The previous failure is consistent with"
            )

            print(
                "  a numerical continuation/seed problem,"
            )

            print(
                "  not evidence of absence of a complex radial branch."
            )

        else:

            print()
            print(
                "Candidates exist but residual quality is poor."
            )

            print(
                "Further local Hamiltonian/branch analysis is required."
            )

    else:

        print()
        print(
            "NO SUCCESSFUL ROOT AT THE ORIGINAL STEP."
        )

        print()
        print(
            "Check the reduced-step results above."
        )

        print(
            "If subdivision succeeds, the problem is step-size"
        )

        print(
            "or continuation robustness."
        )

        print(
            "If subdivision also fails, inspect the local branch"
        )

        print(
            "structure around this radial location."
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This diagnostic does NOT establish or reject a resonance."
    )

    print(
        "It only diagnoses the numerical continuation failure."
    )

    print()
    print(
        "Authoritative solver was not modified."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()