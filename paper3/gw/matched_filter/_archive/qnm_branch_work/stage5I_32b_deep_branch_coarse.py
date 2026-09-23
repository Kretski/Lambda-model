"""
stage5I_32_deep_branch_continuation.py
=========================================
STAGE 5I-32b (COARSE) — FULL-RANGE CONTINUATION OF THE UNASSIGNED DEEP-POLE BRANCH

Purpose
-------
Closes the last unexplored numerical possibility for q3/q4 correspondence.
Five isolated points on an unassigned, strongly-damped mathematical branch
were previously found (stage5I_26, stage5I_28) at scattered m values:

    m=-17: 11.801550 - 2.085053i
    m=-13: 10.443057 - 1.557246i
    m=-12: 10.244066 - 1.635369i
    m=-11: 10.054074 - 1.716312i
    m=-4:   9.310230 - 2.102358i

These were never continuation-tracked across the full experimental range
(m=-21 to -4). This script does that: anchors on the best-conditioned of
the five points and walks outward in both directions, seeded ONLY from
the immediately preceding accepted root at each step -- never from
experimental q3/q4 values.

This is explicitly NOT a search for q3/q4 at any cost. It is a closure
step: does this specific mathematical branch, followed honestly across
its full range, ever approach the experimental q3/q4 curves, or does it
not? Either answer is reported as-is.

Rules (same discipline as stage5I_27/28/29/30)
-------------------------------------------------
1. Authoritative solver files are read-only.
2. Each step seeded only from the immediately preceding accepted root.
3. Same acceptance/jump-guard criteria as prior continuation scripts,
   not relaxed to force a result.
4. Experimental q3/q4 values used ONLY in the final post-hoc comparison.
5. Expanded Im bound (-2.5, from stage5I_26) carried forward since this
   branch is already known to live below the old -1.5 boundary.

Output
------
  stage5I_32b_deep_branch_coarse_m21_to_m4.csv
  console: per-step log + full post-hoc comparison table vs exp q3/q4
"""

import sys
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from stage5I_3_resonance_v2 import find_light_ring
    from stage5I_4_full_m_scan_v6_FIXED import refine_and_polish_candidate
    import stage5I_4_full_m_scan_v6_FIXED as _pipeline_mod
except ImportError as e:
    print(f"[ERROR] Cannot import pipeline modules: {e}")
    sys.exit(1)


# ============================================================================
# CONFIG
# ============================================================================

# Anchor: m=-12, the best-conditioned of the five known deep points
# (found independently in both stage5I_26 and confirmed in stage5I_28's
# blind 2D discovery cross-family; |Res| < 1e-13 in prior runs).
ANCHOR_M = -12
ANCHOR_F_RE = 10.244066
ANCHOR_F_IM = -1.635369

TARGET_M_DOWN = -21   # walk toward more negative m
TARGET_M_UP = -4      # walk toward less negative m

DM = 0.5
RES_ACCEPT = 1.0e-8
MAX_JUMP_RE_HZ = 1.0
MAX_JUMP_IM_HZ = 1.0

# Expanded Im bound carried forward from stage5I_26 -- this branch lives
# below the old -1.5 boundary, confirmed not a wall-attractor artifact.
COMPLEX_IM_MIN_EXPANDED = -3.0

FALLBACK_RE_OFFSETS = [0.0, +0.02, -0.02, +0.05, -0.05]
FALLBACK_IM_OFFSETS = [0.0, +0.02, -0.02, +0.05, -0.05]

sep = "=" * 74


def patch_im_bound(new_min):
    for attr in ("COMPLEX_IM_MIN", "IM_MIN", "F_IM_MIN", "COMPLEX_IM_LOWER"):
        if hasattr(_pipeline_mod, attr):
            original = getattr(_pipeline_mod, attr)
            setattr(_pipeline_mod, attr, new_min)
            return attr, original
    print("  WARNING: could not find an Im-bound attribute to patch.")
    return None, None


def restore_im_bound(attr, original):
    if attr is not None:
        setattr(_pipeline_mod, attr, original)


def try_seed(m, seed_re, seed_im):
    if seed_im < COMPLEX_IM_MIN_EXPANDED:
        return None
    r = find_light_ring(m)
    if r is None or r[0] is None:
        return None
    r_sp, omega_lr = r
    try:
        pole = refine_and_polish_candidate(
            f0_re=seed_re, f0_im_hint=seed_im, m=m,
            r_sp=r_sp, omega_lr=omega_lr,
            real_axis_abs_res=float("nan"), is_complex_seed=True,
        )
    except Exception:
        return None
    if pole is None:
        return None
    try:
        fre = float(pole["f_re"]); fim = float(pole["f_im"])
        abs_res = float(pole.get("abs_res", np.inf))
    except Exception:
        return None
    if not (np.isfinite(fre) and np.isfinite(fim) and np.isfinite(abs_res)):
        return None
    if fim > 1e-9:
        return None
    return {"f_re": fre, "f_im": fim, "abs_res": abs_res}


def continuation_step(m, previous_root):
    seeds = []
    for dre in FALLBACK_RE_OFFSETS:
        for dim in FALLBACK_IM_OFFSETS:
            seed = (previous_root[0] + dre, previous_root[1] + dim)
            if dre == 0.0 and dim == 0.0:
                seeds.insert(0, seed)
            else:
                seeds.append(seed)

    best = None
    best_score = None
    for sre, sim in seeds:
        pole = try_seed(m, sre, sim)
        if pole is None or pole["abs_res"] >= RES_ACCEPT:
            continue
        jump_re = abs(pole["f_re"] - previous_root[0])
        jump_im = abs(pole["f_im"] - previous_root[1])
        if jump_re > MAX_JUMP_RE_HZ or jump_im > MAX_JUMP_IM_HZ:
            continue
        score = pole["abs_res"] + 0.1 * jump_re + 0.1 * jump_im
        if best is None or score < best_score:
            best, best_score = pole, score
        if best is not None and best["abs_res"] < 1e-13 and jump_re < 0.01 and jump_im < 0.01:
            break

    if best is None:
        return None, "NO_CONTINUATION_POLE"
    return best, "PASS"


def walk(direction, start_m, target_m, first_prev, branch):
    previous_root = first_prev
    m = start_m
    while True:
        m = round(m + direction * DM, 6)
        if direction < 0 and m < target_m - 1e-9:
            break
        if direction > 0 and m > target_m + 1e-9:
            break

        pole, status = continuation_step(m, previous_root)
        print(f"m={m:+7.3f}  status={status}", end="")
        if status == "PASS":
            print(f"  f={pole['f_re']:+.9f}{pole['f_im']:+.9f}i Hz "
                  f"|Res|={pole['abs_res']:.2e}")
            branch[m] = pole
            previous_root = (pole["f_re"], pole["f_im"])
        else:
            print("  -- CHAIN STOP: continuation not accepted here")
            break


def main():
    print(sep)
    print("STAGE 5I-32b (COARSE) — DEEP-POLE BRANCH CONTINUATION, m=-21 to -4")
    print("Closing the last unexplored numerical possibility for q3/q4")
    print(sep)
    print()
    print(f"Anchor  : m={ANCHOR_M}, f={ANCHOR_F_RE:+.6f}{ANCHOR_F_IM:+.6f}i Hz")
    print(f"Range   : m={TARGET_M_DOWN} to m={TARGET_M_UP}")
    print(f"Step    : dm={DM}")
    print(f"|Res| acceptance : {RES_ACCEPT}")
    print(f"Jump guards      : dRe<{MAX_JUMP_RE_HZ}, dIm<{MAX_JUMP_IM_HZ} Hz")
    print(f"Im bound         : {COMPLEX_IM_MIN_EXPANDED} (expanded, from stage5I_26)")
    print()
    print("NOTE: seeded ONLY from the previous accepted root at each step.")
    print("      Experimental q3/q4 values are NOT used during continuation.")
    print("      This is a closure test, not a targeted search.")
    print()

    attr, original = patch_im_bound(COMPLEX_IM_MIN_EXPANDED)

    branch = {ANCHOR_M: {"f_re": ANCHOR_F_RE, "f_im": ANCHOR_F_IM, "abs_res": 0.0}}

    try:
        print(sep)
        print(f"DOWNWARD (m={ANCHOR_M} -> {TARGET_M_DOWN})")
        print(sep)
        walk(-1, ANCHOR_M, TARGET_M_DOWN, (ANCHOR_F_RE, ANCHOR_F_IM), branch)

        print()
        print(sep)
        print(f"UPWARD (m={ANCHOR_M} -> {TARGET_M_UP})")
        print(sep)
        walk(+1, ANCHOR_M, TARGET_M_UP, (ANCHOR_F_RE, ANCHOR_F_IM), branch)

    finally:
        restore_im_bound(attr, original)

    out_csv = HERE / "stage5I_32b_deep_branch_coarse_m21_to_m4.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["m", "f_re_Hz", "f_im_Hz", "abs_res"])
        for mm in sorted(branch.keys(), reverse=True):
            p = branch[mm]
            w.writerow([mm, f"{p['f_re']:.9f}", f"{p['f_im']:.9f}",
                        f"{p['abs_res']:.3e}"])
    print()
    print(f"Saved -> {out_csv}")
    print(f"Total branch points recovered: {len(branch)}")

    # ------------------------------------------------------------------
    # Post-hoc comparison vs experimental q3 AND q4 (used ONLY here)
    # ------------------------------------------------------------------
    exp_q3 = {
        -21: 11.73305339, -20: 11.48642482, -19: 11.23953467, -18: 10.99149058,
        -17: 10.74739399, -16: 10.50612174, -15: 10.26056441, -14: 10.01552194,
        -13: 9.76372149,  -12: 9.515540164, -11: 9.257350105, -10: 8.992990952,
        -9: 8.725865804,  -8: 8.452768701,  -7: 8.169428269,  -6: 7.869851632,
        -5: 7.563690856,  -4: 7.236964453,
    }
    exp_q4 = {
        -21: 12.04931005, -20: 11.79180863, -19: 11.54107841, -18: 11.29593696,
        -17: 11.04921008, -16: 10.80589014, -15: 10.55883689, -14: 10.3128259,
        -13: 10.07244017, -12: 9.824297904, -11: 9.578550689, -10: 9.327218691,
        -9: 9.067763134,  -8: 8.808992769,  -7: 8.540722055,  -6: 8.269296643,
        -5: 7.986183324,  -4: 7.696262081,
    }

    print()
    print(sep)
    print("POST-HOC COMPARISON: deep branch vs experimental q3 AND q4")
    print("(comparison performed AFTER continuation -- never used as seed)")
    print(sep)
    print(f"{'m':>5} {'branch Re':>11} {'branch Im':>11} {'exp q3':>9} "
          f"{'dRe(q3)':>9} {'exp q4':>9} {'dRe(q4)':>9}")

    rel_errors_q3 = []
    rel_errors_q4 = []
    for mm in sorted(exp_q3.keys(), reverse=True):
        candidates = [k for k in branch.keys() if abs(k - mm) < 0.05]
        if not candidates:
            print(f"{mm:>5}   -- not reached by continuation --")
            continue
        p = branch[candidates[0]]
        d_q3 = p["f_re"] - exp_q3[mm]
        d_q4 = p["f_re"] - exp_q4[mm]
        rel_q3 = 100 * abs(d_q3) / exp_q3[mm]
        rel_q4 = 100 * abs(d_q4) / exp_q4[mm]
        rel_errors_q3.append(rel_q3)
        rel_errors_q4.append(rel_q4)
        print(f"{mm:>5} {p['f_re']:11.4f} {p['f_im']:11.4f} "
              f"{exp_q3[mm]:9.4f} {d_q3:+9.4f} "
              f"{exp_q4[mm]:9.4f} {d_q4:+9.4f}")

    print()
    if rel_errors_q3:
        print(f"Mean |rel. error| vs q3: {np.mean(rel_errors_q3):.2f}%  "
              f"(min: {np.min(rel_errors_q3):.2f}%, max: {np.max(rel_errors_q3):.2f}%)")
        print(f"Mean |rel. error| vs q4: {np.mean(rel_errors_q4):.2f}%  "
              f"(min: {np.min(rel_errors_q4):.2f}%, max: {np.max(rel_errors_q4):.2f}%)")
        print()
        best_q3 = np.min(rel_errors_q3)
        best_q4 = np.min(rel_errors_q4)
        if best_q3 < 1.0 or best_q4 < 1.0:
            print("VERDICT: at least one point comes within 1% of q3 or q4 --")
            print("         worth a closer, targeted look at that specific m.")
        else:
            print("VERDICT: no point on this branch approaches q3 or q4 to")
            print("         within 1% anywhere across the tested range.")
            print("         This branch is NOT q3/q4. Reported as a closed,")
            print("         honest null result -- not a failed search.")
    else:
        print("No overlap between the recovered branch and the experimental")
        print("m-range -- comparison inconclusive, not negative.")

    print()
    print(sep)


if __name__ == "__main__":
    main()
