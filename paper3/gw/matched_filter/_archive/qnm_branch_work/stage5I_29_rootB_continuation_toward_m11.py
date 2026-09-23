"""
stage5I_29_rootB_continuation_toward_m11.py
=============================================
STAGE 5I-29 — ROOT-B q=2 CONTINUATION FROM m=-14 TOWARD m=-11

Purpose
-------
The validated Root-B q=2 branch (stage5J_rootB_continuation.py) was only
continued in ONE direction: from m=-14 toward m=-15 (more negative m),
reaching m~-14.9055 before breakdown.

The comparison table against experimental q=2 needs Root-B values at the
INTEGER m points -11, -12, -13 as well -- these were never continued.
The v6 full-m-scan discovery grid only found "Root A" at those points
(the smoothly-continuous-but-worse-matching branch), not Root B.

This script continues Root-B in the OTHER direction: m=-14 -> -11,
using small steps (dm=-0.1, i.e. m increasing toward less negative),
seeded ONLY from the immediately preceding accepted root (true
continuation, same discipline as stage5J_rootB_continuation.py and
stage5I_27/28).

No experimental frequency is used as a seed or selection criterion.

Rules (same as the rest of the Stage 5I/5J/6.5 family)
--------------------------------------------------------
1. Authoritative stage5I_3_resonance_v2.py is read-only.
2. Each step is seeded from the immediately preceding accepted root.
3. A step is accepted only if |Res| < RES_ACCEPT.
4. Branch-jump guards prevent accidental jumps to a distant root
   (e.g. back to q=1, or to Root A).
5. Experimental q=2 values are used ONLY at the very end, as a
   post-hoc comparison -- never during the continuation itself.

Output
------
  stage5I_29_rootB_continuation_toward_m11.csv
  console: per-step log + final comparison table vs experimental q=2
"""

import sys
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# ── Import the SAME validated pipeline used throughout Stage 5I/5J ────────────
try:
    from stage5I_3_resonance_v2 import find_light_ring
    from stage5I_4_full_m_scan_v6_FIXED import refine_and_polish_candidate
    import stage5I_4_full_m_scan_v6_FIXED as _pipeline_mod
except ImportError as e:
    print(f"[ERROR] Cannot import pipeline modules: {e}")
    print("Place this script in the same directory as:")
    print("  stage5I_3_resonance_v2.py")
    print("  stage5I_4_full_m_scan_v6_FIXED.py")
    sys.exit(1)


# ============================================================================
# CONFIG
# ============================================================================

# Validated Root-B anchor at m=-14 (from Stage 5J/5K, reproduced independently
# in Stage 6.5B4-1A with |Res|=3.6e-14 -- see STATUS.md).
START_M = -14
START_F_RE = 9.596712868
START_F_IM = -0.135838317

# Target: continue toward less negative m, i.e. m = -13, -12, -11.
TARGET_M = -11

# Step size. Small steps, same discipline as the validated m=-14 -> -14.9055
# continuation (which used dm=-1e-2 coarse / -5e-4 fine). Here we use a
# moderate step since we are filling in known-smooth territory (Root-B was
# already shown continuous across m=-19..-4 on the *experimental* side;
# we are just confirming the *model* side tracks continuously too).
DM = 0.1

# Acceptance / branch-jump guards
RES_ACCEPT = 1.0e-8
MAX_JUMP_RE_HZ = 0.30
MAX_JUMP_IM_HZ = 0.30

# Local fallback seed offsets around the previous accepted root, in case the
# exact previous-root seed does not converge cleanly for a given step.
FALLBACK_RE_OFFSETS = [0.0, +0.02, -0.02, +0.05, -0.05]
FALLBACK_IM_OFFSETS = [0.0, +0.01, -0.01, +0.03, -0.03]

sep = "=" * 74


# ============================================================================
# SINGLE STEP
# ============================================================================

def try_seed(m, seed_re, seed_im):
    """Refine one (Re, Im) seed at azimuthal number m."""
    r = find_light_ring(m)
    if r is None or r[0] is None:
        return None, None

    r_sp, omega_lr = r

    try:
        pole = refine_and_polish_candidate(
            f0_re=seed_re,
            f0_im_hint=seed_im,
            m=m,
            r_sp=r_sp,
            omega_lr=omega_lr,
            real_axis_abs_res=float("nan"),
            is_complex_seed=True,
        )
    except Exception:
        return None, None

    if pole is None:
        return None, None

    try:
        fre = float(pole["f_re"])
        fim = float(pole["f_im"])
        abs_res = float(pole.get("abs_res", np.inf))
    except Exception:
        return None, None

    if not (np.isfinite(fre) and np.isfinite(fim) and np.isfinite(abs_res)):
        return None, None
    if fim > 1e-9:
        return None, None

    return {"f_re": fre, "f_im": fim, "abs_res": abs_res}, r_sp


def continuation_step(m, previous_root):
    """
    Attempt one continuation step at azimuthal number m, seeded from the
    previous accepted root. Returns (pole_dict_or_None, status_str).
    """
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
        pole, _ = try_seed(m, sre, sim)
        if pole is None:
            continue
        if pole["abs_res"] >= RES_ACCEPT:
            continue

        jump_re = abs(pole["f_re"] - previous_root[0])
        jump_im = abs(pole["f_im"] - previous_root[1])
        if jump_re > MAX_JUMP_RE_HZ or jump_im > MAX_JUMP_IM_HZ:
            continue

        score = pole["abs_res"] + 0.1 * jump_re + 0.1 * jump_im
        if best is None or score < best_score:
            best = pole
            best_score = score

    if best is None:
        return None, "NO_CONTINUATION_POLE"

    return best, "PASS"


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(sep)
    print("STAGE 5I-29 — ROOT-B q=2 CONTINUATION: m=-14 -> m=-11")
    print(sep)
    print()
    print(f"Start   : m={START_M}, f={START_F_RE:+.9f}{START_F_IM:+.9f}i Hz")
    print(f"Target  : m={TARGET_M}")
    print(f"Step    : dm={DM}")
    print(f"|Res| acceptance : {RES_ACCEPT}")
    print(f"Jump guards      : dRe<{MAX_JUMP_RE_HZ}, dIm<{MAX_JUMP_IM_HZ} Hz")
    print()
    print("NOTE: seeded ONLY from the previous accepted root at each step.")
    print("      Experimental q=2 values are NOT used during continuation.")
    print()

    branch = {START_M: {"f_re": START_F_RE, "f_im": START_F_IM, "abs_res": 0.0}}
    previous_root = (START_F_RE, START_F_IM)

    m = START_M
    # m increases toward TARGET_M (e.g. -14 -> -13.9 -> ... -> -11.0)
    n_steps = int(round((TARGET_M - START_M) / DM))

    print(sep)
    print("CONTINUATION")
    print(sep)

    for step in range(1, n_steps + 1):
        m = round(START_M + step * DM, 6)

        pole, status = continuation_step(m, previous_root)

        print(f"m={m:+7.3f}  status={status}", end="")

        if status == "PASS":
            print(
                f"  f={pole['f_re']:+.9f}{pole['f_im']:+.9f}i Hz "
                f"|Res|={pole['abs_res']:.2e}"
            )
            branch[m] = pole
            previous_root = (pole["f_re"], pole["f_im"])
        else:
            print("  -- CHAIN STOP: continuation not accepted here")
            break

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    out_csv = HERE / "stage5I_29_rootB_continuation_toward_m11.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["m", "f_re_Hz", "f_im_Hz", "abs_res"])
        for mm in sorted(branch.keys(), reverse=True):
            p = branch[mm]
            w.writerow([mm, f"{p['f_re']:.9f}", f"{p['f_im']:.9f}",
                        f"{p['abs_res']:.3e}"])

    print()
    print(f"Saved -> {out_csv}")

    # ------------------------------------------------------------------
    # Post-hoc comparison vs experimental q=2 (integer m only)
    # ------------------------------------------------------------------
    exp_q2 = {
        -13: 9.342201757, -12: 9.082091798, -11: 8.811238183,
    }

    print()
    print(sep)
    print("POST-HOC COMPARISON: Root-B continuation vs experimental q=2")
    print("(integer m points only -- comparison performed AFTER continuation)")
    print(sep)
    print(f"{'m':>5} {'model Re':>12} {'exp q2':>12} {'delta':>10} {'rel %':>8}")

    for mm in sorted(exp_q2.keys(), reverse=True):
        # find the closest continued point to this integer m
        candidates = [k for k in branch.keys() if abs(k - mm) < 0.05]
        if not candidates:
            print(f"{mm:>5}   -- not reached by continuation --")
            continue
        key = candidates[0]
        p = branch[key]
        d = p["f_re"] - exp_q2[mm]
        rel = 100 * abs(d) / exp_q2[mm]
        print(f"{mm:>5} {p['f_re']:12.6f} {exp_q2[mm]:12.6f} "
              f"{d:+10.6f} {rel:7.4f}%")

    print()
    print(sep)


if __name__ == "__main__":
    main()
