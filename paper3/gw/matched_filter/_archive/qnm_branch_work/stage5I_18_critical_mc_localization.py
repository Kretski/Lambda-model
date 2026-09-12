"""
stage5I_18_critical_mc_localization.py
===========================================

STAGE 5I-18 - CRITICAL m_c LOCALIZATION

CONTEXT: the deep resonance branch has been independently validated
(Stage 5I-16, leave-one-out) over m=-10..-14 and, per further
continuation, over consecutive m-values down to m=-14.75, with the
next tested point (m=-14.80) failing to yield a validated root. This
does NOT yet distinguish a genuine branch ENDPOINT near m_c in
(-14.80,-14.75) from a search/seeding failure that a denser seed grid
could still resolve.

THIS SCRIPT localizes m_c via adaptive coarse-to-fine bisection
(step sizes 0.005 -> 0.001 -> 0.0001), using at EACH tested m a much
denser multi-seed search (a grid of Re x Im offsets around a
linear-extrapolated prediction from the two most recently confirmed
points) than the handful of seeds used in earlier stages -- so a
"NO ROOT" result here is a much stronger statement.

It also tracks d(Re)/dm and d(Im)/dm via finite differences between
consecutive confirmed points as m approaches the bracket edge, as a
DIAGNOSTIC indicator only (not proof) of a fold/saddle-node-type
termination mechanism.

ANCHOR_POINTS below should be edited to your own most recent confirmed
(m, Re, Im) points near the suspected boundary (e.g. from an
m=-14.00..-14.75 continuation run) before running for an accurate
localization -- the placeholders here use only the m=-13,-14 points
independently validated in this session.

No threshold changed. No q-label assigned. No physics/solver code
modified -- only the search/seeding and bisection logic is new.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stage5I_3_resonance_v2 import find_light_ring
from stage5I_4_full_m_scan_v6_FIXED import refine_and_polish_candidate


# EDIT: most recent confirmed (m, Re, Im) points near the suspected
# boundary. Replace with your actual continuation results (e.g. the
# m=-14.70, -14.75 confirmed points) for an accurate run.
ANCHOR_POINTS = {
    -13.00: (9.12188682, -0.31733520),
    -14.00: (9.39955096, -0.30861804),
}

STEP_SEQUENCE = [0.005, 0.001, 0.0001]

SEARCH_M_START = -14.75   # last known/assumed ROOT
SEARCH_M_END = -14.80     # last known/assumed NO-ROOT

RE_OFFSETS = [-0.005, 0.0, 0.005]
IM_OFFSETS = [-0.02, -0.01, 0.0, 0.01, 0.02]

MAX_ACCEPT_DISTANCE_HZ = 0.3


def linear_predict(m_target, m1, point1, m2, point2):
    re1, im1 = point1
    re2, im2 = point2
    if m2 == m1:
        return re1, im1
    t = (m_target - m1) / (m2 - m1)
    re_pred = re1 + t * (re2 - re1)
    im_pred = im1 + t * (im2 - im1)
    return re_pred, im_pred


def dense_search_at_m(m, re_pred, im_pred, r_sp, omega_lr,
                       early_stop_distance=0.01):
    """
    Try seeds around the prediction. Stops EARLY once a validated root
    within early_stop_distance of the prediction is found (a
    legitimate efficiency measure -- once a very close match is found,
    continuing to exhaustively try all remaining offset combinations
    cannot meaningfully change the "ROOT FOUND" conclusion for this m,
    only waste compute). If no such close match is found, all seeds
    are tried and the closest overall result is returned (unchanged
    from the exhaustive behaviour).
    """
    found = []
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
            if pole is not None:
                distance = np.hypot(pole["f_re"] - re_pred, pole["f_im"] - im_pred)
                if distance < MAX_ACCEPT_DISTANCE_HZ:
                    found.append((pole, distance))
                    if distance < early_stop_distance:
                        return pole

    if not found:
        return None

    best_pole, best_distance = min(found, key=lambda x: x[1])
    return best_pole


def main():
    print("=" * 90)
    print("STAGE 5I-18 - CRITICAL m_c LOCALIZATION")
    print("=" * 90)
    print()
    n_seeds = len(RE_OFFSETS) * len(IM_OFFSETS)
    print("Adaptive coarse-to-fine bisection with a DENSE multi-seed search")
    print(f"({len(RE_OFFSETS)}x{len(IM_OFFSETS)}={n_seeds} seeds per tested m)")
    print("-- much denser than earlier stages, so a NO ROOT result here is")
    print("meaningfully stronger evidence.")
    print()
    print("EDIT ANCHOR_POINTS/SEARCH_M_START/SEARCH_M_END at the top of this")
    print("file with your actual most-recent confirmed points before relying")
    print("on this run for a real localization.")
    print()

    anchors_sorted = sorted(ANCHOR_POINTS.items())
    (m1, point1), (m2, point2) = anchors_sorted[-2], anchors_sorted[-1]
    print(f"Extrapolation anchors: m={m1} -> {point1}, m={m2} -> {point2}")
    print()

    confirmed = dict(ANCHOR_POINTS)

    m_root_side = SEARCH_M_START
    m_noroot_side = SEARCH_M_END

    print(f"Initial bracket: root-side m={m_root_side}, "
          f"no-root-side m={m_noroot_side}")
    print()

    for step in STEP_SEQUENCE:

        print("-" * 90)
        print(f"BISECTION PASS: step size = {step}")
        print("-" * 90)

        direction = np.sign(m_noroot_side - m_root_side)
        if direction == 0:
            print("  Bracket already collapsed to a single point.")
            break

        m_test = m_root_side
        last_good_m = m_root_side

        while abs(m_test - m_root_side) < abs(m_noroot_side - m_root_side) - 1e-12:
            m_test = round(m_test + direction * step, 10)

            recent = sorted(confirmed.items())[-2:]
            (rm1, rp1), (rm2, rp2) = recent[0], recent[1]
            re_pred, im_pred = linear_predict(m_test, rm1, rp1, rm2, rp2)

            r_sp, omega_lr = find_light_ring(m_test)
            if r_sp is None:
                print(f"    m={m_test}: no light ring, skipping")
                continue

            pole = dense_search_at_m(m_test, re_pred, im_pred, r_sp, omega_lr)

            if pole is not None:
                print(f"    m={m_test:.4f}: ROOT  Re={pole['f_re']:.6f}  "
                      f"Im={pole['f_im']:.6f}  |Res|={pole['abs_res']:.2e}")
                confirmed[m_test] = (pole["f_re"], pole["f_im"])
                last_good_m = m_test
            else:
                print(f"    m={m_test:.4f}: NO ROOT (dense {n_seeds}-seed search)")
                m_noroot_side = m_test
                break
        else:
            print("    Reached the far end of the bracket without a "
                  "NO-ROOT point at this step size.")

        m_root_side = last_good_m
        print()
        print(f"  Updated bracket after this pass: "
              f"root-side m={m_root_side}, no-root-side m={m_noroot_side}")
        print()

    print("=" * 90)
    print("LOCALIZED m_c BRACKET")
    print("=" * 90)
    print()
    print(f"  Last confirmed ROOT at:     m = {m_root_side}")
    print(f"  First confirmed NO-ROOT at: m = {m_noroot_side}")
    lo, hi = sorted([m_root_side, m_noroot_side])
    print(f"  m_c lies in the interval ({lo}, {hi})")
    print()

    print("=" * 90)
    print("DERIVATIVE TRACKING NEAR THE BRACKET EDGE")
    print("(diagnostic only -- diverging derivatives would be consistent")
    print("with a fold/saddle-node-type termination, not proof of one)")
    print("=" * 90)
    print()

    sorted_confirmed = sorted(confirmed.items())
    print(f"  {'m':>10} {'Re':>12} {'Im':>12} {'dRe/dm':>12} {'dIm/dm':>12}")
    print("  " + "-" * 62)
    prev_m, prev_re, prev_im = None, None, None
    for m, (re, im) in sorted_confirmed[-8:]:
        if prev_m is not None:
            dre_dm = (re - prev_re) / (m - prev_m)
            dim_dm = (im - prev_im) / (m - prev_m)
            print(f"  {m:>10.4f} {re:>12.6f} {im:>12.6f} "
                  f"{dre_dm:>12.4f} {dim_dm:>12.4f}")
        else:
            print(f"  {m:>10.4f} {re:>12.6f} {im:>12.6f} "
                  f"{'--':>12} {'--':>12}")
        prev_m, prev_re, prev_im = m, re, im

    print()
    print("=" * 90)
    print("SUGGESTED SUMMARY LANGUAGE")
    print("=" * 90)
    print()
    print("  \"A distinct smooth numerical resonance branch has been")
    print("  independently validated over multiple continuous m-values")
    print(f"  and appears to terminate in the interval m in ({lo:.4f}, {hi:.4f}),")
    print(f"  localized via adaptive bisection with a dense "
          f"({len(RE_OFFSETS)}x{len(IM_OFFSETS)})-seed")
    print("  search at each tested point. No claim is made here about the")
    print("  underlying termination mechanism.\"")


if __name__ == "__main__":
    main()
