"""
stage5I_30_rootB_continuation_toward_m4.py
=============================================
STAGE 5I-30 — ROOT-B q=2 CONTINUATION FROM m=-14 TOWARD m=-4

Purpose
-------
Extends the previously validated Root-B q=2 continuation
(stage5I_29_rootB_continuation_toward_m11.py, which reached m=-14 -> -11
cleanly: 31 points, max residual 1.5e-14, no branch jumps) further toward
m=-4.

The comparison table currently uses discovery-grid values (not Root-B
continuation) for m=-10 through m=-4, where the relative error grows
smoothly from 0.053% (m=-10) to 1.011% (m=-4). This script tests whether
that growth is a genuine feature of the Root-B branch, or a branch-
selection artifact of the discovery grid (i.e. whether continuation from
the validated anchor reaches a different, better-matching branch in that
region).

Same discipline as stage5I_29 throughout:
  - each step seeded ONLY from the immediately preceding accepted root;
  - no experimental frequency used as a seed, ever;
  - identical acceptance/jump-guard criteria, NOT relaxed partway through;
  - experimental comparison performed only as a post-hoc final step.

Output
------
  stage5I_30_rootB_continuation_toward_m4.csv
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

# Validated Root-B anchor at m=-14 (identical to stage5I_29).
START_M = -14
START_F_RE = 9.596712868
START_F_IM = -0.135838317

# Extended target: continue toward less negative m, all the way to m=-4.
TARGET_M = -4

# SAME step size and SAME acceptance criteria as stage5I_29 -- not relaxed.
DM = 0.1
RES_ACCEPT = 1.0e-8
MAX_JUMP_RE_HZ = 0.30
MAX_JUMP_IM_HZ = 0.30

FALLBACK_RE_OFFSETS = [0.0, +0.02, -0.02, +0.05, -0.05]
FALLBACK_IM_OFFSETS = [0.0, +0.01, -0.01, +0.03, -0.03]

sep = "=" * 74


# ============================================================================
# SINGLE STEP (identical logic to stage5I_29)
# ============================================================================

def try_seed(m, seed_re, seed_im):
    r = find_light_ring(m)
    if r is None or r[0] is None:
        return None, None
    r_sp, omega_lr = r
    try:
        pole = refine_and_polish_candidate(
            f0_re=seed_re, f0_im_hint=seed_im, m=m,
            r_sp=r_sp, omega_lr=omega_lr,
            real_axis_abs_res=float("nan"), is_complex_seed=True,
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
    print("STAGE 5I-30 — ROOT-B q=2 CONTINUATION: m=-14 -> m=-4")
    print("(extends stage5I_29's validated m=-14 -> -11 run)")
    print(sep)
    print()
    print(f"Start   : m={START_M}, f={START_F_RE:+.9f}{START_F_IM:+.9f}i Hz")
    print(f"Target  : m={TARGET_M}")
    print(f"Step    : dm={DM}  (SAME as stage5I_29, not relaxed)")
    print(f"|Res| acceptance : {RES_ACCEPT}  (SAME as stage5I_29)")
    print(f"Jump guards      : dRe<{MAX_JUMP_RE_HZ}, dIm<{MAX_JUMP_IM_HZ} Hz  (SAME)")
    print()
    print("NOTE: seeded ONLY from the previous accepted root at each step.")
    print("      Experimental q=2 values are NOT used during continuation.")
    print("      Acceptance criteria are NOT changed partway through this run.")
    print()

    branch = {START_M: {"f_re": START_F_RE, "f_im": START_F_IM, "abs_res": 0.0}}
    previous_root = (START_F_RE, START_F_IM)

    m = START_M
    n_steps = int(round((TARGET_M - START_M) / DM))

    print(sep)
    print("CONTINUATION")
    print(sep)

    chain_broke_at = None
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
            chain_broke_at = m
            break

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    out_csv = HERE / "stage5I_30_rootB_continuation_toward_m4.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["m", "f_re_Hz", "f_im_Hz", "abs_res"])
        for mm in sorted(branch.keys(), reverse=True):
            p = branch[mm]
            w.writerow([mm, f"{p['f_re']:.9f}", f"{p['f_im']:.9f}",
                        f"{p['abs_res']:.3e}"])
    print()
    print(f"Saved -> {out_csv}")
    if chain_broke_at is not None:
        print(f"NOTE: chain did not reach target m={TARGET_M}; "
              f"stopped at m={chain_broke_at}. This is reported as-is, "
              f"not forced further.")

    # ------------------------------------------------------------------
    # Post-hoc comparison vs experimental q=2 (integer m only)
    # ------------------------------------------------------------------
    exp_q2 = {
        -10: 8.533427811, -9: 8.246009901, -8: 7.945602232,
        -7: 7.633791333, -6: 7.298338, -5: 6.948591527, -4: 6.57003913,
    }
    # Discovery-grid values currently used in the addendum table, for
    # direct before/after comparison at each point this run reaches.
    discovery_grid = {
        -10: 8.53792493, -9: 8.25362645, -8: 7.95839174,
        -7: 7.65039249, -6: 7.32783589, -5: 6.98951810, -4: 6.63644369,
    }

    print()
    print(sep)
    print("POST-HOC COMPARISON: Root-B continuation vs experimental q=2")
    print("vs previously-used discovery-grid values")
    print("(comparison performed AFTER continuation)")
    print(sep)
    print(f"{'m':>5} {'RootB cont.':>13} {'exp q2':>12} {'RootB rel%':>11} "
          f"{'discov.grid':>13} {'discov rel%':>12}")

    for mm in sorted(exp_q2.keys(), reverse=True):
        candidates = [k for k in branch.keys() if abs(k - mm) < 0.05]
        exp_val = exp_q2[mm]
        dg_val = discovery_grid[mm]
        dg_rel = 100 * abs(dg_val - exp_val) / exp_val
        if not candidates:
            print(f"{mm:>5}   -- not reached by continuation --  "
                  f"{exp_val:12.6f}   {'--':>11} "
                  f"{dg_val:13.6f} {dg_rel:12.4f}")
            continue
        key = candidates[0]
        p = branch[key]
        rb_rel = 100 * abs(p["f_re"] - exp_val) / exp_val
        print(f"{mm:>5} {p['f_re']:13.6f} {exp_val:12.6f} {rb_rel:11.4f} "
              f"{dg_val:13.6f} {dg_rel:12.4f}")

    print()
    print(sep)


if __name__ == "__main__":
    main()
