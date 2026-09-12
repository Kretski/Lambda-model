"""
stage5I_12_deep_branch_scan.py
==================================

STAGE 5I-12 - DEEP BRANCH SCAN (linear extrapolation + multi-seed grid)

CONTEXT: Stage 5I-11's naive continuation (seed_(m+1) = root_m) showed
the solver can JUMP BASINS -- continuing the deep root found at m=-10
into m=-11 converged instead to the (unrelated) q=1 root there. This
does NOT prove the deep branch is absent at m=-11 -- it only shows
that a single previous-root seed is not a reliable way to track it,
since Nelder-Mead + hybr polish will happily converge to whichever
basin is nearest the seed.

FIX (this script, not a physics or solver change): use the
ALREADY-OBSERVED SMOOTH TREND across four known deep-branch points
(m=-11..-14) to PREDICT where the branch should sit at a new target m
(m=-10, m=-15), via linear extrapolation in Re(m) and Im(m)
separately. Then try a GRID of seeds around that prediction (varying
Im by fixed offsets) and let the existing, UNCHANGED
refine_and_polish_candidate() (from stage5I_4_full_m_scan_v6_FIXED.py)
do the actual complex refinement + hybr polish for each seed. Among
all validated roots found, keep the one closest to the PREDICTED
(Re, Im) point.

NO q-label is assigned. NO change to find_light_ring(),
refine_and_polish_candidate(), or any resonance_factor/radial_action
code. Only the SEEDING STRATEGY is new.

KNOWN DEEP-BRANCH ANCHOR POINTS:
    m=-11: Re=8.54558431  Im=-0.34109901
    m=-12: Re=8.83787835  Im=-0.32801295
    m=-13: Re=9.12188682  Im=-0.31733520
    m=-14: Re=9.39955096  Im=-0.30861804

Usage:
    python stage5I_12_deep_branch_scan.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stage5I_3_resonance_v2 import find_light_ring
from stage5I_4_full_m_scan_v6_FIXED import refine_and_polish_candidate


ANCHOR_POINTS = {
    -11: (8.54558431, -0.34109901),
    -12: (8.83787835, -0.32801295),
    -13: (9.12188682, -0.31733520),
    -14: (9.39955096, -0.30861804),
}

IM_SEED_OFFSETS = [0.0, -0.02, 0.02, -0.05, 0.05, -0.10, 0.10]

TARGET_M_VALUES = [-10, -15]

MAX_ACCEPT_DISTANCE_HZ = 0.5


def linear_extrapolate(anchor_points, target_m):
    m_vals = np.array(sorted(anchor_points.keys()), dtype=float)
    re_vals = np.array([anchor_points[int(m)][0] for m in m_vals])
    im_vals = np.array([anchor_points[int(m)][1] for m in m_vals])

    re_fit = np.polyfit(m_vals, re_vals, deg=1)
    im_fit = np.polyfit(m_vals, im_vals, deg=1)

    re_pred = np.polyval(re_fit, target_m)
    im_pred = np.polyval(im_fit, target_m)

    re_resid = re_vals - np.polyval(re_fit, m_vals)
    im_resid = im_vals - np.polyval(im_fit, m_vals)

    return re_pred, im_pred, re_resid, im_resid


def scan_target_m(target_m, re_pred, im_pred):
    r_sp, omega_lr = find_light_ring(target_m)
    if r_sp is None:
        print(f"  m={target_m}: no light ring found.")
        return None

    print(f"  m={target_m}: light ring r_sp={r_sp*1e3:.4f} mm, "
          f"omega_lr={omega_lr:.4f} rad/s")
    print(f"  Predicted (Re, Im) = ({re_pred:.5f}, {im_pred:.5f}) Hz")
    print()

    candidates_found = []

    for im_offset in IM_SEED_OFFSETS:
        seed_im = im_pred + im_offset
        print(f"    seed: Re={re_pred:.5f} Hz, Im={seed_im:+.5f} Hz "
              f"(offset {im_offset:+.3f})")

        pole = refine_and_polish_candidate(
            f0_re=re_pred,
            f0_im_hint=seed_im,
            m=target_m,
            r_sp=r_sp,
            omega_lr=omega_lr,
            real_axis_abs_res=float("nan"),
            is_complex_seed=True,
        )

        if pole is None:
            print("        -> no validated root from this seed")
            continue

        distance = np.hypot(pole["f_re"] - re_pred, pole["f_im"] - im_pred)
        print(f"        -> validated root: Re={pole['f_re']:.6f} Hz "
              f"Im={pole['f_im']:.6f} Hz  |Res|={pole['abs_res']:.2e}  "
              f"distance_from_pred={distance:.5f} Hz")

        candidates_found.append((pole, distance))

    print()

    if not candidates_found:
        print(f"  m={target_m}: NO validated roots found across the "
              f"entire seed grid.")
        return None

    unique_roots = []
    for pole, distance in candidates_found:
        if all(np.hypot(pole["f_re"] - u[0]["f_re"],
                         pole["f_im"] - u[0]["f_im"]) > 0.01
               for u in unique_roots):
            unique_roots.append((pole, distance))

    print(f"  m={target_m}: {len(unique_roots)} distinct validated "
          f"root(s) found across the seed grid:")
    for pole, distance in sorted(unique_roots, key=lambda x: x[1]):
        print(f"      Re={pole['f_re']:.6f} Hz  Im={pole['f_im']:.6f} Hz  "
              f"|Res|={pole['abs_res']:.2e}  distance={distance:.5f} Hz")
    print()

    best_pole, best_distance = min(unique_roots, key=lambda x: x[1])

    if best_distance > MAX_ACCEPT_DISTANCE_HZ:
        print(f"  m={target_m}: closest root (distance={best_distance:.5f} Hz) "
              f"exceeds MAX_ACCEPT_DISTANCE_HZ={MAX_ACCEPT_DISTANCE_HZ} -- "
              f"NOT accepted as a plausible branch continuation.")
        return None

    return best_pole, best_distance


def main():
    print("=" * 90)
    print("STAGE 5I-12 - DEEP BRANCH SCAN (linear extrapolation + multi-seed grid)")
    print("=" * 90)
    print()
    print("No changes to find_light_ring(), refine_and_polish_candidate(),")
    print("or any resonance_factor/radial_action code. Only the SEEDING")
    print("strategy is new.")
    print()

    print("Anchor points:")
    for m in sorted(ANCHOR_POINTS.keys()):
        re, im = ANCHOR_POINTS[m]
        print(f"    m={m}: Re={re:.8f} Hz  Im={im:.8f} Hz")
    print()

    results = {}

    for target_m in TARGET_M_VALUES:

        print("-" * 90)
        print(f"TARGET m = {target_m}")
        print("-" * 90)

        re_pred, im_pred, re_resid, im_resid = linear_extrapolate(
            ANCHOR_POINTS, target_m)

        print(f"  Linear fit residuals at anchor points:")
        print(f"    Re residuals: {np.round(re_resid, 6)} Hz "
              f"(max |resid| = {np.max(np.abs(re_resid)):.6f} Hz)")
        print(f"    Im residuals: {np.round(im_resid, 6)} Hz "
              f"(max |resid| = {np.max(np.abs(im_resid)):.6f} Hz)")
        print()

        outcome = scan_target_m(target_m, re_pred, im_pred)
        results[target_m] = outcome

        if outcome is not None:
            pole, distance = outcome
            print(f"  m={target_m} RESULT: FOUND")
            print(f"    Re={pole['f_re']:.8f} Hz  Im={pole['f_im']:.8f} Hz")
            print(f"    |Res|={pole['abs_res']:.4e}")
            print(f"    distance from linear prediction: {distance:.5f} Hz")
        else:
            print(f"  m={target_m} RESULT: NOT FOUND (branch does not "
                  f"continue cleanly here, or seed grid was insufficient)")
        print()

    print("=" * 90)
    print("OVERALL VERDICT")
    print("=" * 90)
    print()

    n_found = sum(1 for v in results.values() if v is not None)
    print(f"  {n_found}/{len(TARGET_M_VALUES)} target m-values yielded a "
          f"plausible branch continuation.")
    print()

    if n_found == len(TARGET_M_VALUES):
        print("  Both m=-10 and m=-15 continue the deep branch smoothly.")
        print("  This substantially strengthens the case for a genuine,")
        print("  wider deep-branch family extending beyond the original")
        print("  m=-11..-14 observation window.")
    elif n_found == 0:
        print("  Neither target m continued the branch. The m=-11..-14")
        print("  deep-branch observation may be a locally isolated")
        print("  feature, not a broader family.")
    else:
        found_m = [m for m, v in results.items() if v is not None]
        missing_m = [m for m, v in results.items() if v is None]
        print(f"  Partial continuation: found at m={found_m}, "
              f"not found at m={missing_m}.")
        print("  This asymmetry itself is informative.")

    print()
    print("As agreed: NO q-label is assigned to any result here. This")
    print("remains a numerical branch-tracking observation, to be")
    print("interpreted physically only after further, independent checks.")


if __name__ == "__main__":
    main()
