"""
stage5I_27_intermediate_m_targets.py
======================================
STAGE 5I-27 — INTERMEDIATE m TARGETS FOR TERMINATION MAPPING

CONTEXT (from Stage 5I-10 analysis):
  m=-10: FOUND  Re=8.24266, Im=-0.35718  (strong confirmation, 4 seeds agree)
  m=-15: NOT FOUND  (solver finds Re≈9.68, Im≈-0.305 with residual ~0.55
                     — not a valid root, but consistently same non-root point)

  The branch therefore terminates somewhere in (-15, -10).
  Stage 5I-18 localized it further to m_c ~ (-14.80, -14.75) via
  adaptive bisection from the quartet anchor points m=-13,-14.

  This script tests INTERMEDIATE m values using seeds INTERPOLATED
  from the confirmed quartet (m=-11, -12, -13, -14) plus the m=-10
  confirmation from Stage 5I-10.

PURPOSE:
  Map branch smoothness approaching the termination.
  Distinguish: gradual approach vs abrupt disappearance.
  Provide independent corroboration of m_c localization
  WITHOUT relying on fractional-m scans suspected of noise.

DESIGN:
  5 intermediate targets: m = -11.5, -12.5, -13.5, -14.0, -14.5
  Seeds interpolated linearly from confirmed quartet.
  Each target: 4x5 seed grid (Re offset × Im offset).
  NO modification of pipeline physics.
  Reports: found/not-found, residual, distance from interpolated prediction.

INSTRUCTIONS:
  Place in same directory as:
    stage5I_3_resonance_v2.py
    stage5I_4_full_m_scan_v6_FIXED.py
  Run: python stage5I_27_intermediate_m_targets.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from stage5I_3_resonance_v2 import find_light_ring
    from stage5I_4_full_m_scan_v6_FIXED import refine_and_polish_candidate
except ImportError as e:
    print(f"[ERROR] Cannot import pipeline modules: {e}")
    sys.exit(1)

# ── Confirmed quartet + Stage 5I-10 confirmation ─────────────────────────────
# (m, Re, Im) — all independently validated
CONFIRMED_POINTS = {
    -10: (8.24266,  -0.35718),   # Stage 5I-10 confirmed
    -11: (8.54558,  -0.34110),   # quartet
    -12: (8.83788,  -0.32801),   # quartet
    -13: (9.12189,  -0.31734),   # quartet
    -14: (9.39955,  -0.30862),   # quartet
}

# ── Intermediate targets to test ──────────────────────────────────────────────
# Choose values that span the (-15, -10) interval non-uniformly,
# denser near the suspected termination
INTERMEDIATE_M_VALUES = [-11.5, -12.5, -13.0, -13.5, -14.5]

# ── Search grid around interpolated prediction ────────────────────────────────
RE_OFFSETS = [-0.05, -0.02, 0.0, +0.02, +0.05]
IM_OFFSETS = [-0.05, -0.02, 0.0, +0.02, +0.05]

MAX_ACCEPT_DISTANCE_HZ = 0.25   # accept poles within 0.25 Hz of prediction

sep = "=" * 72


def linear_interp(m_target, confirmed):
    """
    Linear interpolation of (Re, Im) at m_target from confirmed points.
    Uses the two nearest confirmed points.
    """
    m_arr = np.array(sorted(confirmed.keys()))
    re_arr = np.array([confirmed[m][0] for m in m_arr])
    im_arr = np.array([confirmed[m][1] for m in m_arr])

    # Find bracketing points
    idx = np.searchsorted(m_arr, m_target)
    if idx == 0:
        i1, i2 = 0, 1
    elif idx >= len(m_arr):
        i1, i2 = len(m_arr)-2, len(m_arr)-1
    else:
        i1, i2 = idx-1, idx

    m1, m2 = m_arr[i1], m_arr[i2]
    if m2 == m1:
        return re_arr[i1], im_arr[i1]

    t = (m_target - m1) / (m2 - m1)
    re_pred = re_arr[i1] + t*(re_arr[i2] - re_arr[i1])
    im_pred = im_arr[i1] + t*(im_arr[i2] - im_arr[i1])
    return float(re_pred), float(im_pred)


def search_at_m(m, re_pred, im_pred, r_sp, omega_lr):
    """
    Grid search around (re_pred, im_pred). Returns best pole or None.
    Stops early if a very close match is found.
    """
    best = None
    best_dist = float('inf')

    for d_re in RE_OFFSETS:
        for d_im in IM_OFFSETS:
            seed_re = re_pred + d_re
            seed_im = im_pred + d_im

            pole = refine_and_polish_candidate(
                f0_re=seed_re,
                f0_im_hint=seed_im,
                m=m,
                r_sp=r_sp,
                omega_lr=omega_lr,
                real_axis_abs_res=float("nan"),
                is_complex_seed=True,
            )

            if pole is None:
                continue

            dist = np.hypot(pole['f_re'] - re_pred, pole['f_im'] - im_pred)
            if dist < MAX_ACCEPT_DISTANCE_HZ and dist < best_dist:
                best = pole
                best_dist = dist
                if dist < 0.005:   # very close match — stop early
                    return best, best_dist

    return best, best_dist


def main():
    print(sep)
    print("STAGE 5I-27 — INTERMEDIATE m TARGETS FOR TERMINATION MAPPING")
    print(sep)
    print()
    print("Confirmed anchor points (quartet + Stage 5I-10):")
    print(f"  {'m':>6}  {'Re_confirmed':>14}  {'Im_confirmed':>14}")
    for m in sorted(CONFIRMED_POINTS):
        re, im = CONFIRMED_POINTS[m]
        print(f"  {m:>6}  {re:>14.6f}  {im:>14.6f}")
    print()
    print(f"Intermediate targets: {INTERMEDIATE_M_VALUES}")
    print(f"Search grid: {len(RE_OFFSETS)}×{len(IM_OFFSETS)} = "
          f"{len(RE_OFFSETS)*len(IM_OFFSETS)} seeds per target")
    print(f"Accept radius: {MAX_ACCEPT_DISTANCE_HZ} Hz from interpolated prediction")
    print()

    results = {}

    for m in INTERMEDIATE_M_VALUES:
        print(sep)
        print(f"TARGET: m = {m}")
        print(sep)

        re_pred, im_pred = linear_interp(m, CONFIRMED_POINTS)
        print(f"  Interpolated prediction: Re={re_pred:.6f}, Im={im_pred:.6f}")

        r_sp, omega_lr = find_light_ring(m)
        if r_sp is None:
            print(f"  [SKIP] find_light_ring returned None for m={m}")
            results[m] = None
            continue

        print(f"  Light ring: r_sp={r_sp:.6f}, omega_lr={omega_lr:.6f}")

        pole, dist = search_at_m(m, re_pred, im_pred, r_sp, omega_lr)

        if pole is not None:
            print(f"  FOUND:")
            print(f"    Re={pole['f_re']:.6f}  Im={pole['f_im']:.6f}  "
                  f"|Res|={pole['abs_res']:.2e}")
            print(f"    Distance from prediction: {dist:.4f} Hz")
            print(f"    Re deviation: {pole['f_re']-re_pred:+.4f} Hz")
            print(f"    Im deviation: {pole['f_im']-im_pred:+.4f} Hz")
            results[m] = pole
        else:
            print(f"  NOT FOUND in {len(RE_OFFSETS)*len(IM_OFFSETS)}-seed search")
            print(f"  (centered on prediction Re={re_pred:.4f}, Im={im_pred:.4f})")
            results[m] = None

        print()

    # ── Summary table ─────────────────────────────────────────────────────────
    print(sep)
    print("SUMMARY TABLE")
    print(sep)
    print()

    # Build full table: confirmed + intermediate
    all_m = sorted(list(CONFIRMED_POINTS.keys()) + INTERMEDIATE_M_VALUES)

    print(f"  {'m':>7}  {'Re':>12}  {'Im':>12}  {'|Res|':>10}  {'Status':>20}")
    print("  " + "-"*70)

    for m in all_m:
        if m in CONFIRMED_POINTS and m not in INTERMEDIATE_M_VALUES:
            re, im = CONFIRMED_POINTS[m]
            print(f"  {m:>7.1f}  {re:>12.6f}  {im:>12.6f}  {'<1e-8':>10}  "
                  f"{'CONFIRMED (anchor)':>20}")
        elif m in results:
            pole = results[m]
            if pole is not None:
                print(f"  {m:>7.1f}  {pole['f_re']:>12.6f}  {pole['f_im']:>12.6f}  "
                      f"{pole['abs_res']:>10.2e}  {'FOUND':>20}")
            else:
                re_pred, im_pred = linear_interp(m, CONFIRMED_POINTS)
                print(f"  {m:>7.1f}  {'(pred:'+f'{re_pred:.3f})':>12}  "
                      f"{'(pred:'+f'{im_pred:.3f})':>12}  {'---':>10}  "
                      f"{'NOT FOUND':>20}")

    print()

    # ── Physical interpretation ───────────────────────────────────────────────
    found_m  = [m for m in INTERMEDIATE_M_VALUES if results.get(m) is not None]
    miss_m   = [m for m in INTERMEDIATE_M_VALUES if results.get(m) is None]

    print(sep)
    print("PHYSICAL INTERPRETATION")
    print(sep)
    print()

    if not miss_m:
        print("ALL intermediate targets FOUND.")
        print("Branch is smooth throughout (-15, -10).")
        print("Termination must be at m < min(targets) or search missed it.")
        print("→ Next: bisect below m=" + str(min(INTERMEDIATE_M_VALUES)))

    elif not found_m:
        print("ALL intermediate targets NOT FOUND.")
        print("Branch terminates at m > " + str(max(INTERMEDIATE_M_VALUES)))
        print("i.e., between m=-10 (found) and m=" + str(max(INTERMEDIATE_M_VALUES)))
        print("→ Next: bisect in (m=-10, m=" + str(max(INTERMEDIATE_M_VALUES)) + ")")

    else:
        # Mixed: found some, not others → we can bracket more tightly
        last_found = max(found_m)  # most negative m where found
        first_miss = min(miss_m)   # first m where not found (more negative)
        print(f"Branch confirmed at m >= {last_found} (interpolated)")
        print(f"Branch absent at m <= {first_miss}")
        print(f"Termination bracket: ({first_miss}, {last_found})")
        print()
        print("Smoothness check (if multiple found points):")
        if len(found_m) >= 2:
            for i in range(len(found_m)-1):
                m1, m2 = found_m[i], found_m[i+1]
                p1, p2 = results[m1], results[m2]
                dRe_dm = (p2['f_re'] - p1['f_re']) / (m2 - m1)
                dIm_dm = (p2['f_im'] - p1['f_im']) / (m2 - m1)
                print(f"  m={m1}→{m2}: dRe/dm={dRe_dm:.4f}, dIm/dm={dIm_dm:.4f}")
            print("  (Compare with quartet: dRe/dm≈0.278, dIm/dm≈0.009)")

    print()
    print("NOTE ON m_c=-14.794 (Stage 5I-18):")
    print("  That localization used Stage 5I-16 validated anchors m=-13,-14.")
    print("  This script provides INDEPENDENT corroboration via different")
    print("  intermediate points. Consistent termination bracket = stronger.")
    print("  Discrepant brackets = re-examine seeding/coverage assumptions.")


if __name__ == "__main__":
    main()
