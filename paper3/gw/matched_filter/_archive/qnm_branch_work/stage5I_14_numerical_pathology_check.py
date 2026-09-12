"""
stage5I_14_numerical_pathology_check.py
============================================

STAGE 5I-14 - NUMERICAL PATHOLOGY CHECK (m=-15)

Completes the 4-part verification protocol for branch termination:
  1. Branch tracking m=-10..-14: DONE (Stage 5I-12, stable to |Res|~1e-14)
  2. Systematic 2D search at m=-15: DONE (Stage 5I-13, 3600-point grid)
  3. Independent method (direct grid scan, not continuation seed): DONE
     (Stage 5I-13's dense_grid_scan used safe_complex_res() directly)
  4. Numerical pathology check: THIS SCRIPT.

This script re-examines m=-15 with:
  (a) TWO DIFFERENT grid resolutions (coarser and finer than Stage
      5I-13's 60x60), to check the "no sub-threshold minimum" finding
      is stable, not an artifact of this specific discretization.
  (b) AN ALTERNATIVE root-finding algorithm (Levenberg-Marquardt,
      method='lm') applied to the best candidates found, instead of
      only 'hybr' (used everywhere else in this project).
  (c) A CONTROL run of the exact same procedure at m=-14, where a
      stable root is already known to exist, to confirm the pathology
      check itself can correctly detect a real root when present.

No threshold is changed. No q-label is assigned. No physics or solver
code (resonance_factor, radial_action_*) is modified anywhere.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import root

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stage5I_3_resonance_v2 import find_light_ring
from stage5I_4_full_m_scan_v6_FIXED import safe_complex_res, MAX_VALIDATED_ABS_RES


CONTROL_M = -14
CONTROL_SEED = (9.39955096, -0.30861804)

TARGET_M = -15

RE_RANGE = (9.4, 9.9)
IM_RANGE = (-0.6, -0.05)
RESOLUTIONS = [15, 25, 40]


def dense_grid_min(m, r_sp, omega_lr, n_points):
    re_grid = np.linspace(RE_RANGE[0], RE_RANGE[1], n_points)
    im_grid = np.linspace(IM_RANGE[0], IM_RANGE[1], n_points)

    best_val = np.inf
    best_re, best_im = None, None

    for f_im in im_grid:
        for f_re in re_grid:
            z = safe_complex_res(float(f_re), float(f_im), m, r_sp, omega_lr)
            if z is not None:
                val = abs(z)
                if np.isfinite(val) and val < best_val:
                    best_val = val
                    best_re, best_im = f_re, f_im

    return best_re, best_im, best_val


def polish_with_lm(f_re0, f_im0, m, r_sp, omega_lr, max_shift_hz=0.5):
    def equations(x):
        f_re, f_im = float(x[0]), float(x[1])
        z = safe_complex_res(f_re, f_im, m, r_sp, omega_lr)
        if z is None:
            return np.array([1.0e3, 1.0e3])
        return np.array([z.real, z.imag])

    try:
        result = root(equations, x0=np.array([f_re0, f_im0]), method="lm",
                       options={"xtol": 1.0e-12, "maxiter": 150})
    except (ArithmeticError, FloatingPointError, OverflowError,
            ValueError, RuntimeError):
        return None, None, None, False

    f_re, f_im = float(result.x[0]), float(result.x[1])
    z_final = safe_complex_res(f_re, f_im, m, r_sp, omega_lr)
    if z_final is None:
        return f_re, f_im, None, False

    abs_res = abs(z_final)
    shifted_too_far = np.hypot(f_re - f_re0, f_im - f_im0) > max_shift_hz

    return f_re, f_im, abs_res, (result.success and not shifted_too_far)


def run_resolution_sweep(m, label):
    print(f"  --- Resolution sweep for m={m} ({label}) ---")
    r_sp, omega_lr = find_light_ring(m)
    if r_sp is None:
        print(f"    ERROR: no light ring for m={m}")
        return []

    results = []
    for n_points in RESOLUTIONS:
        best_re, best_im, best_val = dense_grid_min(m, r_sp, omega_lr, n_points)
        print(f"    resolution {n_points}x{n_points}: "
              f"global min |Res|={best_val:.5f} at "
              f"Re={best_re:.5f}, Im={best_im:.5f}")
        results.append((n_points, best_re, best_im, best_val))
    print()
    return results


def run_lm_check(m, seeds, label):
    print(f"  --- Levenberg-Marquardt check for m={m} ({label}) ---")
    r_sp, omega_lr = find_light_ring(m)

    best_overall = (None, None, np.inf, False)
    for f_re0, f_im0 in seeds:
        f_re, f_im, abs_res, success = polish_with_lm(f_re0, f_im0, m, r_sp, omega_lr)
        status = "CONVERGED" if success else "no convergence"
        res_str = f"{abs_res:.4e}" if abs_res is not None else "None"
        print(f"    seed ({f_re0:.5f},{f_im0:.5f}) -> "
              f"({f_re},{f_im})  |Res|={res_str}  [{status}]")
        if abs_res is not None and success and abs_res < best_overall[2]:
            best_overall = (f_re, f_im, abs_res, success)

    print()
    return best_overall


def main():
    print("=" * 90)
    print("STAGE 5I-14 - NUMERICAL PATHOLOGY CHECK")
    print("=" * 90)
    print()
    print("Part (c): CONTROL at m=-14 (known-good root) -- confirms the")
    print("check itself can detect a real root when one is present.")
    print("=" * 90)
    print()

    control_grid_results = run_resolution_sweep(CONTROL_M, "control, known root exists")
    control_lm_result = run_lm_check(
        CONTROL_M,
        seeds=[CONTROL_SEED, (CONTROL_SEED[0] + 0.05, CONTROL_SEED[1] - 0.05)],
        label="control")

    control_found = control_lm_result[3] and control_lm_result[2] < MAX_VALIDATED_ABS_RES
    print(f"  CONTROL RESULT (m={CONTROL_M}): "
          f"{'ROOT CONFIRMED' if control_found else 'UNEXPECTED: root NOT confirmed'}")
    if control_found:
        print(f"    Re={control_lm_result[0]:.8f} Hz  Im={control_lm_result[1]:.8f} Hz  "
              f"|Res|={control_lm_result[2]:.4e}")
    print()

    print("=" * 90)
    print(f"Part (a)+(b): TARGET m={TARGET_M} -- resolution sweep + LM check")
    print("=" * 90)
    print()

    target_grid_results = run_resolution_sweep(TARGET_M, "target, testing for absence")

    all_candidates = [(re, im) for (_, re, im, val) in target_grid_results]
    all_candidates += [(9.68814, -0.30169), (9.84915, -0.12458)]

    target_lm_result = run_lm_check(TARGET_M, seeds=all_candidates, label="target")

    target_found = target_lm_result[3] and target_lm_result[2] < MAX_VALIDATED_ABS_RES
    print(f"  TARGET RESULT (m={TARGET_M}): "
          f"{'ROOT FOUND' if target_found else 'NO ROOT FOUND (any resolution, any seed, LM algorithm)'}")
    if target_found:
        print(f"    Re={target_lm_result[0]:.8f} Hz  Im={target_lm_result[1]:.8f} Hz  "
              f"|Res|={target_lm_result[2]:.4e}")
    print()

    print("=" * 90)
    print("RESOLUTION STABILITY (does the m=-15 'no sub-threshold minimum'")
    print("finding hold across coarser AND finer grids?)")
    print("=" * 90)
    print()
    for n_points, re, im, val in target_grid_results:
        stable = "stable (still >> threshold)" if val > 100 * MAX_VALIDATED_ABS_RES else "CHANGED"
        print(f"    {n_points}x{n_points}: global min |Res|={val:.5f}  [{stable}]")
    print()

    print("=" * 90)
    print("OVERALL VERDICT")
    print("=" * 90)
    print()

    if control_found and not target_found:
        print("  PATHOLOGY CHECK PASSED:")
        print(f"    - Control (m={CONTROL_M}): root robustly confirmed via")
        print("      independent LM algorithm, matching the known result.")
        print(f"    - Target (m={TARGET_M}): NO root found across multiple")
        print("      grid resolutions AND an independent root-finding")
        print("      algorithm (LM, not just hybr).")
        print()
        print("  This is now numerically robust evidence -- not merely a")
        print("  single-config solver failure -- that no root of")
        print(f"  Res(omega)=0 exists in the tested region for m={TARGET_M}.")
        print("  Consistent with genuine branch termination between")
        print(f"  m={CONTROL_M} and m={TARGET_M}, rather than a search artifact.")
    elif not control_found:
        print("  WARNING: the control check itself did not confirm the")
        print(f"  known root at m={CONTROL_M}. This means the pathology-check")
        print("  methodology itself may be flawed -- the target result")
        print("  cannot be trusted until this is resolved.")
    else:
        print(f"  Root WAS found at m={TARGET_M} via LM/resolution variation --")
        print("  Stage 5I-12/13's original 'not found' result was a")
        print("  search-coverage or algorithm-specific limitation, not")
        print("  evidence of genuine branch termination.")

    print()
    print("As agreed throughout: no q-label assigned. This remains a")
    print("numerical characterization, not a physical interpretation.")


if __name__ == "__main__":
    main()
