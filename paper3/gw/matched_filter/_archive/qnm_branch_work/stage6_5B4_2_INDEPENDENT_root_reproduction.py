# -*- coding: utf-8 -*-
"""
STAGE 6.5B4-2
INDEPENDENT ROOT-B REPRODUCTION

Purpose
-------
Independently reproduce frozen Root-B points using a custom damped
Newton solver, without:

    - scipy.optimize.root(method="hybr")
    - scipy.optimize.minimize(method="Nelder-Mead")
    - continuation seeding
    - experimental q=2 frequencies
    - EXP_B
    - branch tracking between different m values

The physical residual is the SAME validated residual used by
Stage 5I-4:

    Res(f) = R * exp(2 i S) - 1

implemented through:

    resonance_factor_v2(
        omega=...,
        m=...,
        r_sp=...,
        omega_lr=...,
        complex_action=True
    )

IMPORTANT
---------
This is an algorithmically independent ROOT SOLVER test, not a
re-derivation of the physical residual.

The residual itself remains the validated double-precision implementation
from stage5I_3_resonance_v2.py.

The solver uses:
    - central finite-difference Jacobian
    - explicit 2x2 Newton solve
    - backtracking damping
    - deterministic local perturbations
    - no SciPy root solver

Frozen Root-B points:
    m=-4
    m=-5
    m=-6
    m=-11
    m=-12
    m=-13
    m=-14

No experimental values are used anywhere in this script.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


# ============================================================================
# PATH / IMPORT
# ============================================================================

HERE = Path(__file__).resolve().parent

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


# Exact validated API from Stage 5I-4
from stage5I_3_resonance_v2 import (
    resonance_factor as resonance_factor_v2,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_CSV = HERE / "stage6_5B4_2_INDEPENDENT_root_reproduction.csv"
OUTPUT_LOG = HERE / "stage6_5B4_2_INDEPENDENT_root_reproduction.log"


# Frozen Root-B points.
#
# These are NOT experimental q=2 values.
#
# m=-6 has the precision available from the previous frozen table.
#
FROZEN_ROOTS = {
    -4.0: (6.636443687771, -0.421971108577),
    -5.0: (6.989518102571, -0.363141434134),
    -6.0: (7.327835886000, -0.318302218000),
    -11.0: (8.812972738000, -0.188208416000),
    -12.0: (9.080269182000, -0.169919319000),
    -13.0: (9.341131317000, -0.152555812000),
    -14.0: (9.596712868000, -0.135838317000),
}


# ---------------------------------------------------------------------------
# Numerical limits
# ---------------------------------------------------------------------------

RES_TOL = 1.0e-10

STEP_TOL_RE = 1.0e-10
STEP_TOL_IM = 1.0e-10

MAX_ITER = 35

# Finite-difference step in Hz.
FD_STEP_RE = 2.0e-5
FD_STEP_IM = 2.0e-5

# Backtracking
MAX_BACKTRACK = 12
DAMP_MIN = 2.0 ** (-MAX_BACKTRACK)

# Physical domain used by Stage 5I-4.
COMPLEX_IM_MIN = -5.0
COMPLEX_IM_MAX = -1.0e-12

# Keep positive real frequency.
FRE_MIN = 1.0e-8


# ============================================================================
# LOCAL START OFFSETS
# ============================================================================
#
# IMPORTANT:
#
# We deliberately do NOT start exactly at the frozen root.
#
# Each frozen root gets a deterministic local perturbation. If the
# independent Newton solver returns to the same root with tiny residual,
# this is stronger evidence that the root is not merely an artifact of
# scipy.optimize.root(method="hybr") starting from the continuation point.
#
# We use two local starts per frozen point.
#
# They are small enough to remain in the local basin but large enough
# that the solver has to actually perform iterations.
#
START_OFFSETS = [
    (+1.0e-2, +1.0e-2),
    (-1.0e-2, +5.0e-3),
]


# ============================================================================
# LOGGING
# ============================================================================

def log(message: str) -> None:
    print(message, flush=True)

    with open(OUTPUT_LOG, "a", encoding="utf-8") as fp:
        fp.write(message + "\n")


# ============================================================================
# RESIDUAL
# ============================================================================

def safe_complex_res(
    f_re: float,
    f_im: float,
    m: float,
):
    """
    Evaluate the validated complex resonance residual.

    Returns
    -------
    complex or None
    """

    if not (
        np.isfinite(f_re)
        and np.isfinite(f_im)
        and np.isfinite(m)
    ):
        return None

    if f_re <= FRE_MIN:
        return None

    if not (COMPLEX_IM_MIN <= f_im <= COMPLEX_IM_MAX):
        return None

    omega = 2.0 * np.pi * (
        f_re + 1j * f_im
    )

    try:
        z = resonance_factor_v2(
            omega=omega,
            m=m,
            r_sp=None,
            omega_lr=None,
            complex_action=True,
        )

    except TypeError:
        # The Stage 5I-4 validated API requires r_sp and omega_lr.
        #
        # This branch is intentionally NOT allowed to silently guess
        # physics. The proper values must therefore be obtained below.
        return None

    except Exception:
        return None

    try:
        zr = float(np.real(z))
        zi = float(np.imag(z))
    except Exception:
        return None

    if not (
        np.isfinite(zr)
        and np.isfinite(zi)
    ):
        return None

    return complex(zr, zi)


# ============================================================================
# PHYSICAL GEOMETRY
# ============================================================================

def get_geometry(m: float):
    """
    Obtain r_sp and omega_lr using the same validated geometry functions
    used by Stage 5I-4.

    Imported lazily so that this script remains explicit about what it uses.
    """

    from stage5I_3_resonance_v2 import (
        find_light_ring,
    )

    try:
        r_sp, omega_lr = find_light_ring(m)
    except Exception:
        return None, None

    return r_sp, omega_lr


# ============================================================================
# RESIDUAL WITH GEOMETRY
# ============================================================================

def make_residual_function(m: float):
    """
    Build a local residual evaluator for a fixed m.
    """

    from stage5I_3_resonance_v2 import (
        find_light_ring,
    )

    try:
        r_sp, omega_lr = find_light_ring(m)
    except Exception as exc:
        raise RuntimeError(
            f"find_light_ring failed for m={m}: {exc}"
        ) from exc

    if not np.isfinite(r_sp):
        raise RuntimeError(
            f"Non-finite r_sp for m={m}: {r_sp}"
        )

    if not np.isfinite(omega_lr):
        raise RuntimeError(
            f"Non-finite omega_lr for m={m}: {omega_lr}"
        )

    def residual(f_re: float, f_im: float):
        if not (
            np.isfinite(f_re)
            and np.isfinite(f_im)
        ):
            return None

        if f_re <= FRE_MIN:
            return None

        if not (
            COMPLEX_IM_MIN
            <= f_im
            <= COMPLEX_IM_MAX
        ):
            return None

        omega = 2.0 * np.pi * (
            f_re + 1j * f_im
        )

        try:
            z = resonance_factor_v2(
                omega=omega,
                m=m,
                r_sp=r_sp,
                omega_lr=omega_lr,
                complex_action=True,
            )
        except Exception:
            return None

        try:
            zr = float(np.real(z))
            zi = float(np.imag(z))
        except Exception:
            return None

        if not (
            np.isfinite(zr)
            and np.isfinite(zi)
        ):
            return None

        return np.array(
            [zr, zi],
            dtype=np.float64,
        )

    return residual, r_sp, omega_lr


# ============================================================================
# NORM
# ============================================================================

def residual_norm(F):
    if F is None:
        return np.inf

    return float(
        np.hypot(
            F[0],
            F[1],
        )
    )


# ============================================================================
# FINITE DIFFERENCE JACOBIAN
# ============================================================================

def finite_difference_jacobian(
    residual,
    x,
):
    """
    Central finite-difference 2x2 Jacobian.

    x = [f_re, f_im]
    F = [Re(Res), Im(Res)]
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    J = np.empty(
        (2, 2),
        dtype=np.float64,
    )

    # ------------------------------------------------------------
    # derivative wrt f_re
    # ------------------------------------------------------------

    xp = x.copy()
    xm = x.copy()

    xp[0] += FD_STEP_RE
    xm[0] -= FD_STEP_RE

    Fp = residual(
        float(xp[0]),
        float(xp[1]),
    )

    Fm = residual(
        float(xm[0]),
        float(xm[1]),
    )

    if Fp is None or Fm is None:
        return None

    J[:, 0] = (
        Fp - Fm
    ) / (
        2.0 * FD_STEP_RE
    )

    # ------------------------------------------------------------
    # derivative wrt f_im
    # ------------------------------------------------------------

    xp = x.copy()
    xm = x.copy()

    xp[1] += FD_STEP_IM
    xm[1] -= FD_STEP_IM

    Fp = residual(
        float(xp[0]),
        float(xp[1]),
    )

    Fm = residual(
        float(xm[0]),
        float(xm[1]),
    )

    if Fp is None or Fm is None:
        return None

    J[:, 1] = (
        Fp - Fm
    ) / (
        2.0 * FD_STEP_IM
    )

    if not np.all(
        np.isfinite(J)
    ):
        return None

    return J


# ============================================================================
# CUSTOM DAMPED NEWTON
# ============================================================================

def independent_newton(
    residual,
    x0,
    m,
):
    """
    Independent 2D damped Newton solver.

    No scipy.optimize.root.
    No scipy.optimize.minimize.

    Returns
    -------
    dict
    """

    x = np.asarray(
        x0,
        dtype=np.float64,
    ).copy()

    F = residual(
        float(x[0]),
        float(x[1]),
    )

    if F is None:
        return {
            "success": False,
            "reason": "INVALID_INITIAL_RESIDUAL",
            "x": x,
            "F": None,
            "iterations": 0,
        }

    norm0 = residual_norm(F)
    norm = norm0

    history = []

    for iteration in range(
        1,
        MAX_ITER + 1,
    ):

        history.append(
            {
                "iteration": iteration,
                "f_re": float(x[0]),
                "f_im": float(x[1]),
                "res_norm": float(norm),
            }
        )

        # --------------------------------------------------------
        # convergence
        # --------------------------------------------------------

        if norm <= RES_TOL:
            return {
                "success": True,
                "reason": "RESIDUAL_CONVERGED",
                "x": x,
                "F": F,
                "iterations": iteration - 1,
                "initial_norm": norm0,
                "final_norm": norm,
                "history": history,
            }

        # --------------------------------------------------------
        # Jacobian
        # --------------------------------------------------------

        J = finite_difference_jacobian(
            residual,
            x,
        )

        if J is None:
            return {
                "success": False,
                "reason": "JACOBIAN_FAILED",
                "x": x,
                "F": F,
                "iterations": iteration - 1,
                "initial_norm": norm0,
                "final_norm": norm,
                "history": history,
            }

        # --------------------------------------------------------
        # condition number
        # --------------------------------------------------------

        try:
            condJ = np.linalg.cond(J)
        except Exception:
            condJ = np.inf

        if not np.isfinite(condJ):
            return {
                "success": False,
                "reason": "JACOBIAN_ILL_CONDITIONED",
                "x": x,
                "F": F,
                "iterations": iteration - 1,
                "initial_norm": norm0,
                "final_norm": norm,
                "history": history,
            }

        # --------------------------------------------------------
        # Newton step
        #
        # J dx = -F
        # --------------------------------------------------------

        try:
            dx = np.linalg.solve(
                J,
                -F,
            )
        except np.linalg.LinAlgError:
            return {
                "success": False,
                "reason": "LINEAR_SOLVE_FAILED",
                "x": x,
                "F": F,
                "iterations": iteration - 1,
                "initial_norm": norm0,
                "final_norm": norm,
                "history": history,
            }

        if not np.all(
            np.isfinite(dx)
        ):
            return {
                "success": False,
                "reason": "NONFINITE_NEWTON_STEP",
                "x": x,
                "F": F,
                "iterations": iteration - 1,
                "initial_norm": norm0,
                "final_norm": norm,
                "history": history,
            }

        # --------------------------------------------------------
        # Step-size sanity
        # --------------------------------------------------------

        if (
            abs(dx[0]) > 1.0
            or abs(dx[1]) > 1.0
        ):
            scale = max(
                1.0,
                abs(dx[0]),
                abs(dx[1]),
            )

            dx = dx / scale

        # --------------------------------------------------------
        # Backtracking line search
        # --------------------------------------------------------

        accepted = False

        damping = 1.0

        best_x = None
        best_F = None
        best_norm = np.inf

        for _ in range(
            MAX_BACKTRACK + 1
        ):

            candidate = (
                x + damping * dx
            )

            f_re_new = float(
                candidate[0]
            )

            f_im_new = float(
                candidate[1]
            )

            Fcand = residual(
                f_re_new,
                f_im_new,
            )

            if Fcand is not None:

                ncand = residual_norm(
                    Fcand
                )

                if ncand < best_norm:
                    best_x = candidate.copy()
                    best_F = Fcand.copy()
                    best_norm = ncand

                # Accept if there is actual decrease.
                if ncand < norm:
                    accepted = True
                    x = candidate
                    F = Fcand
                    norm = ncand
                    break

            damping *= 0.5

            if damping < DAMP_MIN:
                break

        if not accepted:

            # One final possibility:
            # accept the best valid candidate if it improved
            # the residual.
            if (
                best_x is not None
                and best_norm < norm
            ):
                x = best_x
                F = best_F
                norm = best_norm
                continue

            return {
                "success": False,
                "reason": "LINE_SEARCH_FAILED",
                "x": x,
                "F": F,
                "iterations": iteration,
                "initial_norm": norm0,
                "final_norm": norm,
                "history": history,
            }

        # --------------------------------------------------------
        # step convergence
        # --------------------------------------------------------

        actual_step = damping * dx

        if (
            abs(actual_step[0])
            <= STEP_TOL_RE
            and
            abs(actual_step[1])
            <= STEP_TOL_IM
        ):

            if norm <= RES_TOL:
                return {
                    "success": True,
                    "reason": "STEP_AND_RESIDUAL_CONVERGED",
                    "x": x,
                    "F": F,
                    "iterations": iteration,
                    "initial_norm": norm0,
                    "final_norm": norm,
                    "history": history,
                }

    return {
        "success": False,
        "reason": "MAX_ITER",
        "x": x,
        "F": F,
        "iterations": MAX_ITER,
        "initial_norm": norm0,
        "final_norm": norm,
        "history": history,
    }


# ============================================================================
# SINGLE TEST
# ============================================================================

def run_single_test(
    m,
    frozen,
    offset,
    residual,
):
    """
    Run one independent local solve.
    """

    f0_re = (
        frozen[0]
        + offset[0]
    )

    f0_im = (
        frozen[1]
        + offset[1]
    )

    t0 = time.time()

    result = independent_newton(
        residual=residual,
        x0=np.array(
            [f0_re, f0_im],
            dtype=np.float64,
        ),
        m=m,
    )

    elapsed = time.time() - t0

    x = result["x"]

    f_re = float(x[0])
    f_im = float(x[1])

    F = result["F"]

    if F is None:
        final_res = np.inf
    else:
        final_res = residual_norm(F)

    delta_re = (
        f_re - frozen[0]
    )

    delta_im = (
        f_im - frozen[1]
    )

    distance = math.hypot(
        delta_re,
        delta_im,
    )

    return {
        "m": m,
        "start_f_re": f0_re,
        "start_f_im": f0_im,
        "root_f_re": f_re,
        "root_f_im": f_im,
        "frozen_f_re": frozen[0],
        "frozen_f_im": frozen[1],
        "delta_re": delta_re,
        "delta_im": delta_im,
        "distance_to_frozen": distance,
        "abs_res": final_res,
        "iterations": result["iterations"],
        "initial_res": result.get(
            "initial_norm",
            np.nan,
        ),
        "reason": result["reason"],
        "success": result["success"],
        "elapsed_s": elapsed,
    }


# ============================================================================
# CSV
# ============================================================================

FIELDNAMES = [
    "m",
    "start_f_re",
    "start_f_im",
    "root_f_re",
    "root_f_im",
    "frozen_f_re",
    "frozen_f_im",
    "delta_re",
    "delta_im",
    "distance_to_frozen",
    "abs_res",
    "iterations",
    "initial_res",
    "reason",
    "success",
    "elapsed_s",
]


# ============================================================================
# MAIN
# ============================================================================

def main():

    # Fresh log
    with open(
        OUTPUT_LOG,
        "w",
        encoding="utf-8",
    ) as fp:
        fp.write(
            "STAGE 6.5B4-2 — "
            "INDEPENDENT ROOT-B REPRODUCTION\n"
        )

    log("")
    log("=" * 78)
    log(
        "STAGE 6.5B4-2 — "
        "INDEPENDENT ROOT-B REPRODUCTION"
    )
    log("=" * 78)

    log("")
    log("Solver:")
    log("  Custom damped Newton")
    log("  Central finite-difference Jacobian")
    log("  Explicit 2x2 linear solve")
    log("  Backtracking line search")

    log("")
    log("NOT USED:")
    log("  scipy.optimize.root(method='hybr')")
    log("  scipy.optimize.minimize(method='Nelder-Mead')")
    log("  continuation")
    log("  experimental q=2 frequencies")
    log("  EXP_B")

    log("")
    log("Frozen points:")

    for m, root0 in FROZEN_ROOTS.items():
        log(
            f"  m={m:6.1f} : "
            f"{root0[0]:.12f} "
            f"{root0[1]:+.12f} i"
        )

    log("")
    log(
        f"Residual tolerance = {RES_TOL:.1e}"
    )

    log(
        f"FD steps = "
        f"{FD_STEP_RE:.1e}, "
        f"{FD_STEP_IM:.1e}"
    )

    log(
        f"Local starts per root = "
        f"{len(START_OFFSETS)}"
    )

    # ------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------

    csv_file = open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    )

    writer = csv.DictWriter(
        csv_file,
        fieldnames=FIELDNAMES,
    )

    writer.writeheader()

    all_results = []

    # ------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------

    for m, frozen in FROZEN_ROOTS.items():

        log("")
        log("-" * 78)
        log(
            f"m = {m:.1f}"
        )
        log("-" * 78)

        t_m = time.time()

        try:
            residual, r_sp, omega_lr = (
                make_residual_function(m)
            )

        except Exception as exc:

            log(
                f"[FAIL] geometry setup: {exc}"
            )

            for idx, offset in enumerate(
                START_OFFSETS,
                start=1,
            ):

                row = {
                    "m": m,
                    "start_f_re":
                        frozen[0] + offset[0],
                    "start_f_im":
                        frozen[1] + offset[1],
                    "root_f_re":
                        np.nan,
                    "root_f_im":
                        np.nan,
                    "frozen_f_re":
                        frozen[0],
                    "frozen_f_im":
                        frozen[1],
                    "delta_re":
                        np.nan,
                    "delta_im":
                        np.nan,
                    "distance_to_frozen":
                        np.nan,
                    "abs_res":
                        np.inf,
                    "iterations":
                        0,
                    "initial_res":
                        np.inf,
                    "reason":
                        "GEOMETRY_SETUP_FAILED",
                    "success":
                        False,
                    "elapsed_s":
                        np.nan,
                }

                writer.writerow(row)
                csv_file.flush()
                all_results.append(row)

            continue

        log(
            f"r_sp      = {r_sp:.12e}"
        )

        log(
            f"omega_lr  = {omega_lr:.12e}"
        )

        # --------------------------------------------------------
        # Verify frozen point itself
        # --------------------------------------------------------

        F_frozen = residual(
            frozen[0],
            frozen[1],
        )

        frozen_res = residual_norm(
            F_frozen
        )

        log("")
        log(
            "Frozen-point residual:"
        )

        log(
            f"  |Res(frozen)| = "
            f"{frozen_res:.6e}"
        )

        # --------------------------------------------------------
        # Independent local starts
        # --------------------------------------------------------

        for idx, offset in enumerate(
            START_OFFSETS,
            start=1,
        ):

            log("")
            log(
                f"START {idx}/"
                f"{len(START_OFFSETS)}"
            )

            log(
                f"  offset = "
                f"({offset[0]:+.6e}, "
                f"{offset[1]:+.6e}) Hz"
            )

            row = run_single_test(
                m=m,
                frozen=frozen,
                offset=offset,
                residual=residual,
            )

            writer.writerow(row)
            csv_file.flush()

            all_results.append(row)

            log(
                f"  start = "
                f"{row['start_f_re']:.12f} "
                f"{row['start_f_im']:+.12f} i"
            )

            log(
                f"  root  = "
                f"{row['root_f_re']:.12f} "
                f"{row['root_f_im']:+.12f} i"
            )

            log(
                f"  |Res| = "
                f"{row['abs_res']:.6e}"
            )

            log(
                f"  Δroot = "
                f"({row['delta_re']:+.6e}, "
                f"{row['delta_im']:+.6e}) Hz"
            )

            log(
                f"  distance = "
                f"{row['distance_to_frozen']:.6e} Hz"
            )

            log(
                f"  iterations = "
                f"{row['iterations']}"
            )

            log(
                f"  reason = "
                f"{row['reason']}"
            )

            log(
                f"  elapsed = "
                f"{row['elapsed_s']:.3f} s"
            )

        log("")
        log(
            f"m={m:.1f} finished in "
            f"{time.time() - t_m:.3f} s"
        )

    csv_file.close()

    # =========================================================================
    # SUMMARY
    # =========================================================================

    log("")
    log("=" * 78)
    log("FINAL SUMMARY")
    log("=" * 78)

    successful = [
        r for r in all_results
        if bool(r["success"])
    ]

    log("")
    log(
        f"Total independent solves : "
        f"{len(all_results)}"
    )

    log(
        f"Successful solves        : "
        f"{len(successful)}"
    )

    log(
        f"Failed solves            : "
        f"{len(all_results) - len(successful)}"
    )

    if successful:

        max_res = max(
            float(r["abs_res"])
            for r in successful
        )

        max_distance = max(
            float(r["distance_to_frozen"])
            for r in successful
        )

        log("")
        log(
            f"max |Res| among PASS      = "
            f"{max_res:.6e}"
        )

        log(
            f"max distance to frozen   = "
            f"{max_distance:.6e} Hz"
        )

    log("")
    log(
        f"CSV : {OUTPUT_CSV}"
    )

    log(
        f"LOG : {OUTPUT_LOG}"
    )

    log("")
    log("=" * 78)
    log(
        "INTERPRETATION"
    )
    log("=" * 78)

    log("")
    log(
        "A PASS means the independent Newton solver, "
        "starting from a local perturbation, "
        "returned to the frozen Root-B point "
        "with |Res| below tolerance."
    )

    log("")
    log(
        "This supports numerical reproducibility of "
        "the frozen Root-B roots under a different "
        "root-finding algorithm."
    )

    log("")
    log(
        "It does NOT establish a physical Kerr mapping "
        "or an experimental q=2 identification."
    )

    log("")
    log("DONE.")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()