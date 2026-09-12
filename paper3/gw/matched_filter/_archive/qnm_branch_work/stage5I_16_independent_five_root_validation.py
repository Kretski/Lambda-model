"""
stage5I_16_independent_five_root_validation.py
====================================================

STAGE 5I-16 - INDEPENDENT VALIDATION OF THE FIVE DEEP-BRANCH ROOTS

CONTEXT: Stage 5I-15 fit a smooth quadratic to five (Re, Im) pairs at
m=-10..-14 and found excellent smoothness. As correctly pointed out:
that test proves the FIVE NUMBERS lie on a smooth curve -- it does NOT
independently prove each number is itself a validated root of the
original solver, since those numbers were GIVEN inputs to that fit,
not re-derived from scratch there.

THIS SCRIPT closes that gap with LEAVE-ONE-OUT validation, stronger
than simply re-running the solver seeded at each already-known root
(which would only confirm "the solver stays where you put it"):

  For each target m in {-10,-11,-12,-13,-14}:
    1. Take the OTHER FOUR known deep-branch points (NOT the target's
       own known root).
    2. Fit a quadratic to Re(m) and Im(m) using ONLY those four points.
    3. Predict a SEED for the target m -- contains NO information
       about the target's own previously-found root value.
    4. Run the ORIGINAL, UNCHANGED refine_and_polish_candidate()
       (Nelder-Mead + hybr polish) from that seed.
    5. Compare the freshly-converged root to the previously-reported
       value, and report |Res| at the final converged point.

No threshold changed. No q-label assigned. No physics or solver code
modified -- only the seeding/validation logic is new.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stage5I_3_resonance_v2 import find_light_ring
from stage5I_4_full_m_scan_v6_FIXED import (
    refine_and_polish_candidate,
    MAX_VALIDATED_ABS_RES,
)


KNOWN_ROOTS = {
    -10: (8.24265780, -0.35718004),
    -11: (8.54558431, -0.34109901),
    -12: (8.83787835, -0.32801295),
    -13: (9.12188682, -0.31733520),
    -14: (9.39955096, -0.30861804),
}


def leave_one_out_predict(target_m, known_roots, degree=2):
    other_m = sorted(m for m in known_roots if m != target_m)
    m_vals = np.array(other_m, dtype=float)
    re_vals = np.array([known_roots[m][0] for m in other_m])
    im_vals = np.array([known_roots[m][1] for m in other_m])

    re_fit = np.polyfit(m_vals, re_vals, deg=degree)
    im_fit = np.polyfit(m_vals, im_vals, deg=degree)

    re_pred = float(np.polyval(re_fit, target_m))
    im_pred = float(np.polyval(im_fit, target_m))

    re_train_resid = re_vals - np.polyval(re_fit, m_vals)
    im_train_resid = im_vals - np.polyval(im_fit, m_vals)
    re_rmse = float(np.sqrt(np.mean(re_train_resid ** 2)))
    im_rmse = float(np.sqrt(np.mean(im_train_resid ** 2)))

    return re_pred, im_pred, re_rmse, im_rmse


def validate_one_m(target_m):
    print(f"  --- Target m={target_m} (leave-one-out) ---")

    re_pred, im_pred, re_rmse, im_rmse = leave_one_out_predict(
        target_m, KNOWN_ROOTS, degree=2)

    known_re, known_im = KNOWN_ROOTS[target_m]

    print(f"    Training points (excluding m={target_m}): "
          f"{sorted(m for m in KNOWN_ROOTS if m != target_m)}")
    print(f"    Quadratic fit train-RMSE: Re={re_rmse:.6f} Hz, "
          f"Im={im_rmse:.6f} Hz")
    print(f"    Leave-one-out PREDICTED seed: "
          f"Re={re_pred:.5f} Hz, Im={im_pred:.5f} Hz")
    print(f"    (For reference only, NOT used as input) previously-"
          f"reported root: Re={known_re:.8f} Hz, Im={known_im:.8f} Hz")
    print(f"    Seed-to-known-root distance: "
          f"{np.hypot(re_pred - known_re, im_pred - known_im):.5f} Hz")
    print()

    r_sp, omega_lr = find_light_ring(target_m)
    if r_sp is None:
        print(f"    ERROR: no light ring found for m={target_m}")
        return None

    pole = refine_and_polish_candidate(
        f0_re=re_pred,
        f0_im_hint=im_pred,
        m=target_m,
        r_sp=r_sp,
        omega_lr=omega_lr,
        real_axis_abs_res=float("nan"),
        is_complex_seed=True,
    )

    if pole is None:
        print(f"    RESULT: NO VALIDATED ROOT from the independent "
              f"leave-one-out seed.")
        print()
        return None

    diff_re = pole["f_re"] - known_re
    diff_im = pole["f_im"] - known_im
    diff_dist = np.hypot(diff_re, diff_im)

    print(f"    RESULT: independently converged to")
    print(f"      Re={pole['f_re']:.10f} Hz  Im={pole['f_im']:.10f} Hz")
    print(f"      |Res|={pole['abs_res']:.4e}")
    print(f"    Difference vs previously-reported value: "
          f"dRe={diff_re:+.8f} Hz, dIm={diff_im:+.8f} Hz "
          f"(distance={diff_dist:.8f} Hz)")
    print()

    return dict(m=target_m, f_re=pole["f_re"], f_im=pole["f_im"],
                abs_res=pole["abs_res"], diff_dist=diff_dist,
                re_pred=re_pred, im_pred=im_pred)


def main():
    print("=" * 90)
    print("STAGE 5I-16 - INDEPENDENT LEAVE-ONE-OUT VALIDATION OF THE FIVE")
    print("DEEP-BRANCH ROOTS (m=-10..-14)")
    print("=" * 90)
    print()
    print("For each target m, the seed is predicted from a quadratic fit")
    print("to the OTHER FOUR known points only -- the target's own")
    print("previously-reported root value is NEVER used as input to its")
    print("own seed. The ORIGINAL, UNCHANGED solver")
    print("(refine_and_polish_candidate: Nelder-Mead + hybr polish) then")
    print("runs fresh from that independently-derived seed.")
    print()

    results = []
    for target_m in sorted(KNOWN_ROOTS.keys(), reverse=True):
        result = validate_one_m(target_m)
        results.append(result)

    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print()

    n_validated = sum(1 for r in results if r is not None)
    print(f"  {n_validated}/{len(KNOWN_ROOTS)} roots independently "
          f"re-validated via leave-one-out seeding.")
    print()

    if n_validated > 0:
        print(f"  {'m':>4} {'|Res|':>12} {'distance from prior value':>26}")
        print("  " + "-" * 46)
        for r in results:
            if r is not None:
                print(f"  {r['m']:>4} {r['abs_res']:>12.4e} "
                      f"{r['diff_dist']:>26.8f} Hz")
        print()

        max_res = max(r["abs_res"] for r in results if r is not None)
        max_dist = max(r["diff_dist"] for r in results if r is not None)

        print(f"  Worst-case |Res| across all 5: {max_res:.4e}")
        print(f"  Worst-case distance from prior value: {max_dist:.8f} Hz")
        print()

    print("=" * 90)
    print("VERDICT")
    print("=" * 90)
    print()

    if n_validated == len(KNOWN_ROOTS):
        all_machine_precision = all(
            r["abs_res"] < MAX_VALIDATED_ABS_RES for r in results if r is not None)
        all_close_to_prior = all(
            r["diff_dist"] < 1.0e-4 for r in results if r is not None)

        if all_machine_precision and all_close_to_prior:
            print("  ALL FIVE roots independently re-validated:")
            print(f"    - Every root converged to |Res| < "
                  f"{MAX_VALIDATED_ABS_RES:.0e} (machine-precision-level).")
            print("    - Every root matches its previously-reported value")
            print("      to within 1e-4 Hz, despite being seeded WITHOUT")
            print("      any knowledge of that value (leave-one-out).")
            print()
            print("  This is now genuinely strong evidence: 5 independently")
            print("  validated roots, forming a smooth, distinct branch")
            print("  (per Stage 5I-15), clearly separated from q=2, with a")
            print("  numerically robust absence result at m=-15 (Stage")
            print("  5I-14). This combination is a credible candidate for a")
            print("  genuine additional resonance family with finite-m")
            print("  termination -- still requiring physical interpretation")
            print("  and, ideally, comparison with any independent theory")
            print("  or additional experimental data before publication.")
        else:
            print("  All five roots converged, but not all met the strict")
            print("  precision/consistency bar -- see per-m table above for")
            print("  which points need further attention.")
    else:
        missing = [m for m, r in zip(sorted(KNOWN_ROOTS.keys(), reverse=True), results)
                   if r is None]
        print(f"  NOT all five roots were independently re-validated.")
        print(f"  Missing/failed: m={missing}")
        print("  The smooth-branch claim from Stage 5I-15 should be")
        print("  treated with caution until these are resolved.")


if __name__ == "__main__":
    main()
