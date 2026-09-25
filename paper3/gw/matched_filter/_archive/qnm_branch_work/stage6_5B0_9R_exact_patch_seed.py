"""
STAGE 6.5B-0.9R — EXACT PATCH-SEED CONTINUATION DIAGNOSTIC
============================================================

Purpose
-------
Reproduce the complex radial continuation using the SAME initial
near-turning-point analytic seed used by stage5I_3c_radial_action_patch.py.

This corrects the previous diagnostic (6.5B-0.9), which incorrectly
started continuation with p_initial=0 and therefore failed at step 1.

We now test:

    H(p,r) ~= H_r * (r-r_minus)
              + 1/2 * H_pp * p^2

giving

    p_seed = sqrt(-2 H_r (r-r_minus) / H_pp)

exactly as used by the validated patch.

No:
  - C variation
  - resonance root search
  - Kerr fit
  - GW calibration
  - frequency rescaling
  - low-k approximation
  - authoritative solver modification

The purpose is purely numerical continuation diagnosis.
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
# CONSTANTS
# ============================================================================

M_MODE = -12

C0 = 7.690000000000e-04
OMEGA0 = 1.000000000000e-01
GAMMA0 = 2.220000000000e-06
H00 = 3.400000000000e-02
G0 = 9.810000000000e+00
R_B = 3.730000000000e-02

F_RE = 8.306630716900
F_IM = -0.000566435000

OMEGA = 2.0 * np.pi * (F_RE + 1j * F_IM)

N = 20000


# ============================================================================
# LOAD MODULE
# ============================================================================

def load_module(path, name):

    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot create import spec for {path}"
        )

    module = importlib.util.module_from_spec(spec)

    module.__file__ = str(path)
    module.__name__ = name

    sys.modules[name] = module

    spec.loader.exec_module(module)

    return module


# ============================================================================
# HAMILTONIAN DERIVATIVES
# ============================================================================

def compute_H_r(
    solver,
    omega,
    p,
    m,
    r,
    eps=1.0e-9,
):
    """
    Complex central finite difference for dH/dr.
    """

    H_plus = solver.hamiltonian(
        omega,
        p,
        m,
        r + eps,
    )

    H_minus = solver.hamiltonian(
        omega,
        p,
        m,
        r - eps,
    )

    return (
        H_plus - H_minus
    ) / (2.0 * eps)


def compute_H_pp(
    solver,
    omega,
    m,
    r,
    eps=1.0e-5,
):
    """
    Second derivative d²H/dp² evaluated at p=0.
    """

    H0 = solver.hamiltonian(
        omega,
        0j,
        m,
        r,
    )

    Hp = solver.hamiltonian(
        omega,
        complex(eps),
        m,
        r,
    )

    Hm = solver.hamiltonian(
        omega,
        complex(-eps),
        m,
        r,
    )

    return (
        Hp - 2.0 * H0 + Hm
    ) / (eps ** 2)


# ============================================================================
# RESIDUAL
# ============================================================================

def residual(
    solver,
    omega,
    p,
    m,
    r,
):
    H = solver.hamiltonian(
        omega,
        p,
        m,
        r,
    )

    return abs(H)


# ============================================================================
# TRY AUTHORITATIVE COMPLEX SOLVER
# ============================================================================

def try_solve(
    solver,
    omega,
    m,
    r,
    seed,
):

    try:

        p = solver.solve_p_complex(
            omega,
            m,
            r,
            p_initial=complex(seed),
        )

    except Exception as exc:

        return None, np.inf, repr(exc)

    if p is None:

        return None, np.inf, "solve_p_complex returned None"

    try:

        res = residual(
            solver,
            omega,
            p,
            m,
            r,
        )

    except Exception as exc:

        return (
            complex(p),
            np.inf,
            f"residual failed: {exc!r}",
        )

    return (
        complex(p),
        float(res),
        None,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 78)
    print("STAGE 6.5B-0.9R — EXACT PATCH-SEED CONTINUATION DIAGNOSTIC")
    print("=" * 78)

    print()
    print("Purpose:")
    print(
        "  Reproduce the exact near-turning-point seed used by"
    )
    print(
        "  stage5I_3c_radial_action_patch.py."
    )

    print()
    print("No C variation.")
    print("No resonance root search.")
    print("No authoritative-file modification.")

    # ----------------------------------------------------------------------
    # LOAD SOLVER
    # ----------------------------------------------------------------------

    print()
    print("-" * 78)
    print("1. LOAD AUTHORITATIVE SOLVER")
    print("-" * 78)

    solver = load_module(
        RES_FILE,
        "_stage5I3_exact_patch_seed",
    )

    print("  Loaded successfully.")

    print()
    print(
        "solve_p_complex:",
        inspect.signature(
            solver.solve_p_complex
        ),
    )

    print(
        "hamiltonian:",
        inspect.signature(
            solver.hamiltonian
        ),
    )

    # ----------------------------------------------------------------------
    # LIGHT RING
    # ----------------------------------------------------------------------

    print()
    print("-" * 78)
    print("2. LIGHT RING")
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

    print(
        f"  r_sp = {r_sp * 1e3:.12f} mm"
    )

    print(
        f"  f_lr = {omega_lr/(2*np.pi):.12f} Hz"
    )

    # ----------------------------------------------------------------------
    # TURNING POINT
    # ----------------------------------------------------------------------

    print()
    print("-" * 78)
    print("3. COMPLEX TURNING POINT")
    print("-" * 78)

    r_minus_real = solver.find_last_scattering_point(
        OMEGA,
        M_MODE,
        r_sp,
    )

    r_turn = solver.find_turning_point_complex(
        OMEGA,
        M_MODE,
        r_minus_real,
    )

    print(
        f"  r_minus_real = {r_minus_real:.15f} m"
    )

    print(
        f"  r_turn = {r_turn.real:.15f}"
        f" {r_turn.imag:+.15e} i m"
    )

    H_turn = solver.hamiltonian(
        OMEGA,
        0j,
        M_MODE,
        r_turn,
    )

    print(
        f"  |H(turn)| = {abs(H_turn):.6e}"
    )

    # ----------------------------------------------------------------------
    # BUILD PATH
    # ----------------------------------------------------------------------

    path = np.linspace(
        complex(r_turn),
        complex(R_B),
        N,
    )

    dr = path[1] - path[0]

    print()
    print("-" * 78)
    print("4. PATH")
    print("-" * 78)

    print(f"  N = {N}")
    print(f"  dr = {abs(dr):.12e} m")

    # ----------------------------------------------------------------------
    # EXACT PATCH SEED
    # ----------------------------------------------------------------------

    r1 = path[1]

    print()
    print("-" * 78)
    print("5. EXACT PATCH FIRST-STEP SEED")
    print("-" * 78)

    # This is the formula used in stage5I_3c_radial_action_patch.py:
    #
    # H ~= H_r * (r-r_minus) + 1/2 H_pp p²
    #
    # p ~= sqrt(-2 H_r (r-r_minus) / H_pp)

    Hr = compute_H_r(
        solver,
        OMEGA,
        0j,
        M_MODE,
        r_turn,
    )

    Hpp = compute_H_pp(
        solver,
        OMEGA,
        M_MODE,
        r_turn,
    )

    radicand = (
        -2.0
        * Hr
        * (r1 - r_turn)
        / Hpp
    )

    p_seed_plus = np.sqrt(radicand)
    p_seed_minus = -p_seed_plus

    print(
        f"  H_r  = {Hr.real:+.12e}"
        f" {Hr.imag:+.12e} i"
    )

    print(
        f"  H_pp = {Hpp.real:+.12e}"
        f" {Hpp.imag:+.12e} i"
    )

    print(
        f"  radicand = {radicand.real:+.12e}"
        f" {radicand.imag:+.12e} i"
    )

    print()
    print(
        f"  p_seed + = {p_seed_plus.real:+.12e}"
        f" {p_seed_plus.imag:+.12e} i"
    )

    print(
        f"  p_seed - = {p_seed_minus.real:+.12e}"
        f" {p_seed_minus.imag:+.12e} i"
    )

    # ----------------------------------------------------------------------
    # TEST BOTH INITIAL BRANCH SIGNS
    # ----------------------------------------------------------------------

    print()
    print("-" * 78)
    print("6. FIRST STEP — BOTH ANALYTIC SEED SIGNS")
    print("-" * 78)

    p_plus, res_plus, err_plus = try_solve(
        solver,
        OMEGA,
        M_MODE,
        r1,
        p_seed_plus,
    )

    if p_plus is None:

        print("  + seed: FAIL")
        print(f"    {err_plus}")

    else:

        print("  + seed: PASS")
        print(
            f"    p = {p_plus.real:+.12e}"
            f" {p_plus.imag:+.12e} i"
        )
        print(
            f"    |H| = {res_plus:.6e}"
        )

    p_minus, res_minus, err_minus = try_solve(
        solver,
        OMEGA,
        M_MODE,
        r1,
        p_seed_minus,
    )

    if p_minus is None:

        print("  - seed: FAIL")
        print(f"    {err_minus}")

    else:

        print("  - seed: PASS")
        print(
            f"    p = {p_minus.real:+.12e}"
            f" {p_minus.imag:+.12e} i"
        )
        print(
            f"    |H| = {res_minus:.6e}"
        )

    # Choose the plus branch first, matching np.sqrt convention.
    if p_plus is not None:

        p_prev = p_plus

    elif p_minus is not None:

        p_prev = p_minus

    else:

        print()
        print("=" * 78)
        print("VERDICT: ANALYTIC FIRST-STEP SEED ALSO FAILS")
        print("=" * 78)

        print()
        print(
            "This would be a genuine numerical issue immediately"
        )
        print(
            "at the turning-point continuation step."
        )

        return

    # ----------------------------------------------------------------------
    # CONTINUE AND RECORD
    # ----------------------------------------------------------------------

    print()
    print("-" * 78)
    print("7. CONTINUATION")
    print("-" * 78)

    print(
        "Using the successful analytic first-step root as"
    )
    print(
        "the seed for subsequent continuation."
    )

    p_prevprev = None

    first_failure = None
    history = []

    # Start at index 1 because it was solved above.
    history.append(
        {
            "i": 1,
            "r": r1,
            "p": p_prev,
            "res": residual(
                solver,
                OMEGA,
                p_prev,
                M_MODE,
                r1,
            ),
        }
    )

    # Only print selected milestones.
    milestones = {
        2,
        3,
        4,
        5,
        10,
        20,
        40,
        60,
        70,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        90,
        100,
    }

    for i in range(2, len(path)):

        r = path[i]

        # Primary continuation seed = previous p.
        seed = p_prev

        p_new, res_new, err = try_solve(
            solver,
            OMEGA,
            M_MODE,
            r,
            seed,
        )

        if p_new is None:

            first_failure = i

            print()
            print(
                f"  FIRST FAILURE: step {i}"
            )

            print(
                f"    r = {r.real:.15f}"
                f" {r.imag:+.15e} i"
            )

            print(
                f"    |dr from previous| = "
                f"{abs(r-path[i-1]):.12e} m"
            )

            print(
                f"    error = {err}"
            )

            break

        dp = abs(p_new - p_prev)

        history.append(
            {
                "i": i,
                "r": r,
                "p": p_new,
                "res": res_new,
                "dp": dp,
            }
        )

        if i in milestones:

            print(
                f"  step={i:5d}  "
                f"r={r.real:.12f}  "
                f"p={p_new.real:+.8e}"
                f"{p_new.imag:+.8e}i  "
                f"|H|={res_new:.3e}  "
                f"|dp|={dp:.3e}"
            )

        p_prevprev = p_prev
        p_prev = p_new

    # ----------------------------------------------------------------------
    # FAILURE REGION MULTI-SEED
    # ----------------------------------------------------------------------

    if first_failure is not None:

        print()
        print("-" * 78)
        print("8. MULTI-SEED TEST AT ACTUAL FAILURE")
        print("-" * 78)

        i = first_failure
        r_fail = path[i]

        p1 = history[-1]["p"]

        if len(history) >= 2:
            p2 = history[-2]["p"]
        else:
            p2 = p1

        predictor = p1 + (p1 - p2)

        seeds = [
            ("previous p", p1),
            ("linear predictor", predictor),
            (
                "half predictor",
                p1 + 0.5 * (p1 - p2),
            ),
            (
                "quarter predictor",
                p1 + 0.25 * (p1 - p2),
            ),
        ]

        print(
            f"  failure step = {i}"
        )

        print(
            f"  r = {r_fail.real:.15f}"
            f" {r_fail.imag:+.15e} i m"
        )

        for label, seed in seeds:

            p_test, res_test, err_test = try_solve(
                solver,
                OMEGA,
                M_MODE,
                r_fail,
                seed,
            )

            if p_test is None:

                print()
                print(
                    f"  {label:20s}: FAIL"
                )
                print(
                    f"    {err_test}"
                )

            else:

                print()
                print(
                    f"  {label:20s}: PASS"
                )
                print(
                    f"    p = {p_test.real:+.12e}"
                    f" {p_test.imag:+.12e} i"
                )
                print(
                    f"    |H| = {res_test:.6e}"
                )
                print(
                    f"    |dp from previous| = "
                    f"{abs(p_test-p1):.6e}"
                )

    # ----------------------------------------------------------------------
    # VERDICT
    # ----------------------------------------------------------------------

    print()
    print("=" * 78)
    print("DIAGNOSTIC VERDICT")
    print("=" * 78)

    if first_failure is None:

        print()
        print(
            "FULL CONTINUATION COMPLETED."
        )

        print()
        print(
            "The step-77 failure from 6.5B-0.8 was NOT reproduced"
        )

        print(
            "when the exact patch initialization was used."
        )

        print()
        print(
            "Next step: compare the exact radial-action implementation"
        )

        print(
            "against this continuation trace."
        )

    else:

        print()
        print(
            f"Continuation failed at step {first_failure}."
        )

        print()
        print(
            "This result is now meaningful because the exact"
        )

        print(
            "near-turning-point analytic seed was used."
        )

        print()
        print(
            "Inspect the multi-seed result above:"
        )

        print(
            "  * alternate seed succeeds"
        )
        print(
            "      -> continuation seed / branch tracking issue"
        )

        print(
            "  * all seeds fail"
        )

        print(
            "      -> local complex radial structure needs inspection"
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This test neither establishes nor rejects a resonance."
    )

    print(
        "It diagnoses only the numerical complex-p continuation."
    )

    print()
    print(
        "Authoritative solver was not modified."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()