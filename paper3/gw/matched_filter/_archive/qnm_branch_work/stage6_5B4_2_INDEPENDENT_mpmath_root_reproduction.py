```python
"""
stage6_5B4_2_INDEPENDENT_mpmath_root_reproduction.py
=====================================================

STAGE 6.5B4-2
INDEPENDENT SOLVER REPRODUCTION OF FROZEN ROOT-B POINTS

Purpose
-------
Test whether previously validated Root-B roots are artifacts of
scipy.optimize.root(method='hybr').

This test intentionally does NOT:
    - perform continuation in m
    - use experimental q=2 frequencies
    - use scipy.optimize.root('hybr')
    - use refine_and_polish_candidate()
    - select the branch using experimental data

Instead:
    1. Take frozen mathematical Root-B points.
    2. Use the SAME physical residual function.
    3. Solve Re(Res)=0, Im(Res)=0 with mpmath.findroot.
    4. Use 50-digit arithmetic for the outer root finder.
    5. Start locally around each frozen point.
    6. Verify the resulting root against the original residual.

IMPORTANT
---------
The physical residual machinery is intentionally reused. The independence
being tested here is the ROOT-FINDING METHOD, not a complete re-derivation
of the physical model.

Representative frozen points:
    m = -4, -5, -6, -11, -12, -13

The exact -11/-13 values should be filled from the validated Stage 5I-29 CSV
before execution. Do NOT reconstruct or interpolate them.

Outputs
-------
    stage6_5B4_2_INDEPENDENT_results.csv
    stage6_5B4_2_INDEPENDENT_run.log

PASS criterion
--------------
A point is considered independently reproduced if:

    1. mpmath converges to a finite complex root
    2. |Res| < RES_TOL
    3. root remains close to the frozen Root-B point
    4. no experimental value was used

The primary comparison is the complex frequency itself.
"""

import csv
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import mpmath as mp


# ============================================================
# PATH
# ============================================================

HERE = Path(__file__).resolve().parent

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


# ============================================================
# IMPORT EXISTING PHYSICAL RESIDUAL MACHINERY
# ============================================================
#
# We deliberately reuse the already validated physical residual.
# The solver itself is NOT reused.
#

try:
    from stage5I_3_resonance_v2 import find_light_ring
    from stage5I_4_full_m_scan_v6_FIXED import (
        resonance_factor,
    )
except ImportError as e:
    print(f"[ERROR] Could not import residual machinery: {e}")
    print("Check that this script is in qnm_branch_work.")
    sys.exit(1)


# ============================================================
# CONFIGURATION
# ============================================================

MP_DPS = 50

# Residual tolerance for the independently evaluated root.
RES_TOL = 1.0e-10

# Maximum allowed displacement from frozen root.
#
# This is deliberately generous compared with the observed numerical
# precision, but small enough to prevent convergence to a completely
# different resonance.
MAX_DISTANCE_RE_HZ = 0.25
MAX_DISTANCE_IM_HZ = 0.25

# mpmath Newton/secant starting offsets around frozen root.
#
# No continuation is performed. Every point is treated independently.
SEED_OFFSETS = [
    (0.0, 0.0),
    (+1.0e-4, 0.0),
    (-1.0e-4, 0.0),
    (0.0, +1.0e-4),
    (0.0, -1.0e-4),
    (+1.0e-3, 0.0),
    (-1.0e-3, 0.0),
    (0.0, +1.0e-3),
    (0.0, -1.0e-3),
]


# ============================================================
# FROZEN ROOT-B DATA
# ============================================================
#
# IMPORTANT:
# These are mathematical frozen roots from the blind continuation.
#
# DO NOT put experimental q=2 values here.
#
# -4, -5, -6 are taken from the completed blind continuation.
#
# -11, -12, -13:
#   -12 is known exactly.
#   -11 and -13 MUST be replaced with the exact Stage 5I-29 values.
#

FROZEN_ROOTS = {
    -4.0:  (6.636443687771, -0.421971108577),
    -5.0:  (6.989518102571, -0.363141434134),
    -6.0:  (7.327835886000, -0.318302218000),

    # --------------------------------------------------------
    # FILL THESE TWO FROM stage5I_29 CSV.
    # DO NOT INTERPOLATE.
    # --------------------------------------------------------

    -11.0: (None, None),
    -12.0: (8.306630716900, -0.000566435000),
    -13.0: (None, None),
}


# ============================================================
# OUTPUT FILES
# ============================================================

RESULTS_CSV = HERE / "stage6_5B4_2_INDEPENDENT_results.csv"
RUN_LOG = HERE / "stage6_5B4_2_INDEPENDENT_run.log"


# ============================================================
# LOGGING
# ============================================================

def log(message):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp}  {message}"

    print(line, flush=True)

    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


# ============================================================
# ORIGINAL RESIDUAL WRAPPER
# ============================================================
#
# resonance_factor() is the validated physical residual implementation.
#
# It returns the complex residual machinery in the original numerical
# implementation. We use it only as an oracle for evaluating Res(f).
#
# The root finder below is completely independent.
# ============================================================

def residual_numpy(m, f_re, f_im):
    """
    Evaluate the validated physical residual at complex frequency

        f = f_re + i f_im

    Returns complex Res.
    """

    f_complex = complex(float(f_re), float(f_im))

    value = resonance_factor(
        f_complex,
        int(round(m)),
    )

    if value is None:
        raise FloatingPointError(
            f"resonance_factor returned None at "
            f"m={m}, f={f_complex}"
        )

    value = complex(value)

    if not (np.isfinite(value.real) and np.isfinite(value.imag)):
        raise FloatingPointError(
            f"non-finite residual at m={m}, f={f_complex}"
        )

    return value


# ============================================================
# MP WRAPPER
# ============================================================

def residual_mp(m, fre, fim):
    """
    mpmath-facing residual.

    NOTE:
    The physical residual implementation itself is still double precision
    because it is the already validated model implementation.

    mpmath is used for:
        - independent 2D root solving
        - high-precision finite-difference Jacobian
        - independent convergence logic

    This distinction is intentional and documented.
    """

    fre_float = float(fre)
    fim_float = float(fim)

    z = residual_numpy(m, fre_float, fim_float)

    return (
        mp.mpf(str(z.real)),
        mp.mpf(str(z.imag)),
    )


# ============================================================
# INDEPENDENT ROOT SOLVER
# ============================================================

def solve_independent(m, frozen_re, frozen_im):
    """
    Independently solve:

        Re Res(f_re + i f_im) = 0
        Im Res(f_re + i f_im) = 0

    using mpmath.findroot.

    No continuation between different m values.
    """

    attempts = []

    for d_re, d_im in SEED_OFFSETS:

        x0 = mp.mpf(str(frozen_re + d_re))
        y0 = mp.mpf(str(frozen_im + d_im))

        # Second point for secant-style 2D findroot.
        x1 = x0 + mp.mpf("1e-5")
        y1 = y0 + mp.mpf("1e-5")

        try:

            def F(x, y):
                return residual_mp(m, x, y)

            t0 = time.time()

            root = mp.findroot(
                F,
                (x0, y0),
                solver="mdnewton",
                tol=mp.mpf("1e-30"),
                maxsteps=40,
                verify=False,
            )

            elapsed = time.time() - t0

            # mpmath can return matrix-like objects depending on version.
            if hasattr(root, "__len__"):
                fre_mp = root[0]
                fim_mp = root[1]
            else:
                raise RuntimeError(
                    "Unexpected mpmath root return type"
                )

            fre = float(fre_mp)
            fim = float(fim_mp)

            if not (
                np.isfinite(fre)
                and np.isfinite(fim)
            ):
                raise FloatingPointError("non-finite root")

            # Independent residual verification.
            res = residual_numpy(m, fre, fim)
            abs_res = abs(res)

            jump_re = abs(fre - frozen_re)
            jump_im = abs(fim - frozen_im)

            attempts.append({
                "f_re": fre,
                "f_im": fim,
                "abs_res": abs_res,
                "jump_re": jump_re,
                "jump_im": jump_im,
                "elapsed": elapsed,
                "seed_re": float(x0),
                "seed_im": float(y0),
            })

            if (
                abs_res < RES_TOL
                and jump_re <= MAX_DISTANCE_RE_HZ
                and jump_im <= MAX_DISTANCE_IM_HZ
                and fim <= 1e-9
            ):
                return attempts[-1]

        except Exception as exc:

            attempts.append({
                "error": f"{type(exc).__name__}: {exc}",
                "seed_re": float(x0),
                "seed_im": float(y0),
            })

    return None


# ============================================================
# CSV INITIALIZATION
# ============================================================

def initialize_csv():

    if RESULTS_CSV.exists():
        return

    with RESULTS_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "m",
            "frozen_re_Hz",
            "frozen_im_Hz",
            "independent_re_Hz",
            "independent_im_Hz",
            "abs_res",
            "delta_re_Hz",
            "delta_im_Hz",
            "elapsed_s",
            "seed_re_Hz",
            "seed_im_Hz",
            "status",
        ])

        f.flush()
        os.fsync(f.fileno())


# ============================================================
# WRITE RESULT IMMEDIATELY
# ============================================================

def append_result(row):

    with RESULTS_CSV.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)
        writer.writerow(row)

        f.flush()
        os.fsync(f.fileno())


# ============================================================
# MAIN
# ============================================================

def main():

    mp.mp.dps = MP_DPS

    initialize_csv()

    log("=" * 78)
    log("STAGE 6.5B4-2 -- INDEPENDENT mpmath ROOT REPRODUCTION")
    log("=" * 78)
    log(f"mpmath precision: {MP_DPS} decimal digits")
    log(f"Residual tolerance: {RES_TOL:.1e}")
    log("")
    log("IMPORTANT:")
    log("  No continuation between m values.")
    log("  No experimental q=2 frequencies.")
    log("  No scipy.optimize.root('hybr').")
    log("  No refine_and_polish_candidate().")
    log("  Same validated physical residual only.")
    log("")

    frozen = [
        (m, values)
        for m, values in FROZEN_ROOTS.items()
        if values[0] is not None and values[1] is not None
    ]

    missing = [
        m
        for m, values in FROZEN_ROOTS.items()
        if values[0] is None or values[1] is None
    ]

    if missing:
        log(
            "WARNING: missing frozen points: "
            + ", ".join(f"{m:g}" for m in missing)
        )
        log(
            "These points will NOT be guessed or interpolated."
        )
        log("")

    passed = 0
    failed = 0

    for m, (frozen_re, frozen_im) in frozen:

        log("-" * 78)
        log(
            f"m={m:+.1f}  "
            f"frozen={frozen_re:+.12f}"
            f"{frozen_im:+.12f}i Hz"
        )

        t0 = time.time()

        result = solve_independent(
            m,
            frozen_re,
            frozen_im,
        )

        total_elapsed = time.time() - t0

        if result is None:

            failed += 1

            log(
                f"m={m:+.1f}  FAIL -- "
                f"no independently validated root"
            )

            append_result([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                f"{m:.6f}",
                f"{frozen_re:.12f}",
                f"{frozen_im:.12f}",
                "",
                "",
                "",
                "",
                "",
                f"{total_elapsed:.3f}",
                "",
                "",
                "FAIL",
            ])

            continue

        passed += 1

        log(
            f"m={m:+.1f}  PASS"
        )

        log(
            f"  independent root = "
            f"{result['f_re']:+.12f}"
            f"{result['f_im']:+.12f}i Hz"
        )

        log(
            f"  |Res| = {result['abs_res']:.6e}"
        )

        log(
            f"  ΔRe = {result['jump_re']:.6e} Hz"
        )

        log(
            f"  ΔIm = {result['jump_im']:.6e} Hz"
        )

        log(
            f"  seed = "
            f"{result['seed_re']:+.12f}"
            f"{result['seed_im']:+.12f}i Hz"
        )

        log(
            f"  solver time = "
            f"{result['elapsed']:.3f} s"
        )

        append_result([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"{m:.6f}",
            f"{frozen_re:.12f}",
            f"{frozen_im:.12f}",
            f"{result['f_re']:.12f}",
            f"{result['f_im']:.12f}",
            f"{result['abs_res']:.6e}",
            f"{result['jump_re']:.6e}",
            f"{result['jump_im']:.6e}",
            f"{total_elapsed:.3f}",
            f"{result['seed_re']:.12f}",
            f"{result['seed_im']:.12f}",
            "PASS",
        ])

    log("")
    log("=" * 78)
    log("FINAL SUMMARY")
    log("=" * 78)
    log(f"Points tested: {len(frozen)}")
    log(f"PASS: {passed}")
    log(f"FAIL: {failed}")

    if missing:
        log(
            "Not tested because exact frozen values are missing: "
            + ", ".join(f"{m:g}" for m in missing)
        )

    log("")
    log(f"CSV -> {RESULTS_CSV}")
    log(f"LOG -> {RUN_LOG}")
    log("=" * 78)


if __name__ == "__main__":
    main()
```
