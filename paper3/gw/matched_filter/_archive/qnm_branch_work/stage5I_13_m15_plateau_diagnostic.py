"""
stage5I_13_m15_plateau_diagnostic.py
========================================

STAGE 5I-13 - m=-15 PLATEAU DIAGNOSTIC (genuine absence vs coverage gap)

CONTEXT: Stage 5I-12 found that all 7 seeds around the linear-
extrapolation prediction for m=-15 converge (via Nelder-Mead) to
essentially the SAME point (Re~9.681, Im~-0.305), with a STABLE
plateau value |Res|~0.53-0.58 -- but hybr's subsequent root-polish
REJECTS every one of them (never reaching MAX_VALIDATED_ABS_RES=1e-8).

As correctly noted: "NOT FOUND is not mathematical proof of absence."
This script does NOT lower the validation threshold -- instead it
investigates the SHAPE of Res(f) directly around the plateau, via a
dense 2D grid scan (same unchanged safe_complex_res()), to distinguish:

  (a) GENUINE LOCAL MINIMUM, NOT A ROOT: |Res| bottoms out near the
      plateau value across a wide search box and never approaches
      zero nearby -- honest evidence the branch does not continue to
      m=-15.

  (b) NARROW/MISSED TRUE ROOT NEARBY: a substantially better point
      (|Res| much smaller) exists just outside Stage 5I-12's 7-point
      seed grid -- meaning that search was simply insufficient.

No threshold is changed. No q-label is assigned. No physics or solver
code is modified -- only the diagnostic scan pattern is new.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stage5I_3_resonance_v2 import find_light_ring
from stage5I_4_full_m_scan_v6_FIXED import (
    safe_complex_res,
    refine_and_polish_candidate,
    MAX_VALIDATED_ABS_RES,
)


TARGET_M = -15

RE_RANGE = (9.4, 9.9)
IM_RANGE = (-0.6, -0.05)

RE_POINTS = 60
IM_POINTS = 60

PLATEAU_REFERENCE_ABS_RES = 0.53
IMPROVEMENT_FACTOR_THRESHOLD = 3.0


def dense_grid_scan(m, r_sp, omega_lr):
    re_grid = np.linspace(RE_RANGE[0], RE_RANGE[1], RE_POINTS)
    im_grid = np.linspace(IM_RANGE[0], IM_RANGE[1], IM_POINTS)

    print(f"  Dense grid scan: Re in {RE_RANGE} ({RE_POINTS} pts) x "
          f"Im in {IM_RANGE} ({IM_POINTS} pts) = {RE_POINTS*IM_POINTS} points")
    print()

    grid_vals = np.full((IM_POINTS, RE_POINTS), np.inf, dtype=float)

    n_finite = 0
    for j, f_im in enumerate(im_grid):
        for i, f_re in enumerate(re_grid):
            z = safe_complex_res(float(f_re), float(f_im), m, r_sp, omega_lr)
            if z is not None:
                val = abs(z)
                if np.isfinite(val):
                    grid_vals[j, i] = val
                    n_finite += 1

    print(f"  Finite evaluations: {n_finite}/{RE_POINTS*IM_POINTS}")
    print()

    if n_finite == 0:
        print("  ERROR: no finite evaluations anywhere in this box.")
        return None, None, None, grid_vals, re_grid, im_grid

    finite_mask = np.isfinite(grid_vals)
    masked = np.where(finite_mask, grid_vals, np.inf)
    flat_idx = np.argmin(masked)
    j_min, i_min = np.unravel_index(flat_idx, grid_vals.shape)
    global_min_val = grid_vals[j_min, i_min]
    global_min_re = re_grid[i_min]
    global_min_im = im_grid[j_min]

    return global_min_re, global_min_im, global_min_val, grid_vals, re_grid, im_grid


def local_minima_survey(grid_vals, re_grid, im_grid, top_n=10):
    n_im, n_re = grid_vals.shape
    minima = []

    for j in range(1, n_im - 1):
        for i in range(1, n_re - 1):
            val = grid_vals[j, i]
            if not np.isfinite(val):
                continue
            neighborhood = grid_vals[j-1:j+2, i-1:i+2]
            if val <= np.nanmin(neighborhood):
                minima.append((re_grid[i], im_grid[j], val))

    minima.sort(key=lambda x: x[2])

    deduped = []
    for re, im, val in minima:
        if all(np.hypot(re - d[0], im - d[1]) > 0.03 for d in deduped):
            deduped.append((re, im, val))
        if len(deduped) >= top_n:
            break

    return deduped


def main():
    print("=" * 90)
    print("STAGE 5I-13 - m=-15 PLATEAU DIAGNOSTIC")
    print("=" * 90)
    print()
    print(f"Question: is the Stage 5I-12 plateau (|Res|~{PLATEAU_REFERENCE_ABS_RES})")
    print("at m=-15 a genuine local minimum with no nearby root, or does a")
    print("wider/denser scan reveal a substantially better point that")
    print("Nelder-Mead simply missed?")
    print()
    print("NO threshold is changed. NO physics/solver code is modified.")
    print()

    r_sp, omega_lr = find_light_ring(TARGET_M)
    if r_sp is None:
        print("ERROR: no light ring found for m=-15.")
        return

    print(f"m={TARGET_M}: light ring r_sp={r_sp*1e3:.4f} mm, "
          f"omega_lr={omega_lr:.4f} rad/s")
    print()

    global_min_re, global_min_im, global_min_val, grid_vals, re_grid, im_grid = \
        dense_grid_scan(TARGET_M, r_sp, omega_lr)

    if global_min_re is None:
        return

    print(f"  GLOBAL minimum on this grid:")
    print(f"    Re={global_min_re:.5f} Hz  Im={global_min_im:.5f} Hz  "
          f"|Res|={global_min_val:.5f}")
    print()

    improvement = PLATEAU_REFERENCE_ABS_RES / global_min_val if global_min_val > 0 else np.inf
    print(f"  Improvement vs Stage 5I-12 plateau "
          f"({PLATEAU_REFERENCE_ABS_RES}): {improvement:.2f}x")
    print()

    print("  Top local minima found on the grid (landscape survey):")
    minima = local_minima_survey(grid_vals, re_grid, im_grid, top_n=10)
    for re, im, val in minima:
        print(f"      Re={re:.5f} Hz  Im={im:.5f} Hz  |Res|={val:.5f}")
    print()

    print("=" * 90)
    print("DIAGNOSIS")
    print("=" * 90)
    print()

    if improvement < IMPROVEMENT_FACTOR_THRESHOLD:
        print(f"  The dense grid's global minimum ({global_min_val:.5f}) is NOT")
        print(f"  substantially better than Stage 5I-12's plateau "
              f"({PLATEAU_REFERENCE_ABS_RES}).")
        print(f"  This is consistent with a GENUINE LOCAL MINIMUM, not a")
        print(f"  missed root: |Res| bottoms out around this value across")
        print(f"  a wide search box, rather than approaching zero nearby.")
        print()
        print(f"  This does NOT yet PROVE the branch is absent at m=-15")
        print(f"  (the search box could still be too narrow, or the true")
        print(f"  root could require a different region of the complex")
        print(f"  plane) -- but it is meaningfully stronger evidence for")
        print(f"  genuine termination than Stage 5I-12 alone.")
    else:
        print(f"  The dense grid found a point {improvement:.1f}x better than")
        print(f"  Stage 5I-12's plateau -- Nelder-Mead's coarser seed grid")
        print(f"  MISSED a meaningfully deeper minimum. Attempting full")
        print(f"  refine_and_polish_candidate() at this better point now:")
        print()

        pole = refine_and_polish_candidate(
            f0_re=global_min_re,
            f0_im_hint=global_min_im,
            m=TARGET_M,
            r_sp=r_sp,
            omega_lr=omega_lr,
            real_axis_abs_res=float("nan"),
            is_complex_seed=True,
        )

        if pole is not None:
            print(f"  VALIDATED ROOT FOUND: Re={pole['f_re']:.8f} Hz "
                  f"Im={pole['f_im']:.8f} Hz  |Res|={pole['abs_res']:.4e}")
            print(f"  Stage 5I-12's NOT FOUND result for m=-15 was a search-")
            print(f"  coverage gap, not evidence of branch termination.")
        else:
            print(f"  Still REJECTED by hybr even from this better-looking")
            print(f"  seed. The grid minimum, despite being numerically lower,")
            print(f"  does not correspond to a genuine root either.")

    print()
    print("As agreed: no q-label assigned. This is a numerical")
    print("characterization step only.")


if __name__ == "__main__":
    main()
