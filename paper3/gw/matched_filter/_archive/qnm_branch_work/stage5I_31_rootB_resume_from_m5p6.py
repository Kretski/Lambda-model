"""
stage5I_31_rootB_resume_from_m5p6.py
======================================
RESUME Root-B continuation from the last confirmed point (m=-5.600) to
the original target m=-4.0. Saves ~5.5 hours by not recomputing the
already-confirmed m=-14 -> -5.6 portion.

Same discipline as stage5I_29/30: seeded only from the previous accepted
root, same acceptance/jump-guard criteria, no experimental frequency used
as a seed.
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
except ImportError as e:
    print(f"[ERROR] Cannot import pipeline modules: {e}")
    sys.exit(1)

# ── RESUME POINT (last confirmed PASS from the running log) ──────────────
START_M = -5.6
START_F_RE = 7.194422146
START_F_IM = -0.334876498

TARGET_M = -4.0
DM = 0.1
RES_ACCEPT = 1.0e-8
MAX_JUMP_RE_HZ = 0.30
MAX_JUMP_IM_HZ = 0.30

FALLBACK_RE_OFFSETS = [0.0, +0.02, -0.02, +0.05, -0.05]
FALLBACK_IM_OFFSETS = [0.0, +0.01, -0.01, +0.03, -0.03]

sep = "=" * 74


def try_seed(m, seed_re, seed_im):
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
        # Early-exit: first seed IS the previous root -- if it converges
        # cleanly (typical case, as seen in the running log), don't waste
        # time trying the other 24 fallback seeds.
        if best is not None and best["abs_res"] < 1e-13 and jump_re < 0.01 and jump_im < 0.01:
            break

    if best is None:
        return None, "NO_CONTINUATION_POLE"
    return best, "PASS"


def main():
    print(sep)
    print(f"RESUME: m={START_M} -> m={TARGET_M}")
    print(sep)
    print(f"Start: f={START_F_RE:+.9f}{START_F_IM:+.9f}i Hz")
    print("Early-exit enabled: stop trying fallback seeds once a clean,")
    print("near-zero-jump match is found (matches observed log behavior).")
    print()

    branch = {START_M: {"f_re": START_F_RE, "f_im": START_F_IM, "abs_res": 0.0}}
    previous_root = (START_F_RE, START_F_IM)

    n_steps = int(round((TARGET_M - START_M) / DM))
    chain_broke_at = None

    for step in range(1, n_steps + 1):
        m = round(START_M + step * DM, 6)
        pole, status = continuation_step(m, previous_root)
        print(f"m={m:+7.3f}  status={status}", end="")
        if status == "PASS":
            print(f"  f={pole['f_re']:+.9f}{pole['f_im']:+.9f}i Hz "
                  f"|Res|={pole['abs_res']:.2e}")
            branch[m] = pole
            previous_root = (pole["f_re"], pole["f_im"])
        else:
            print("  -- CHAIN STOP")
            chain_broke_at = m
            break

    out_csv = HERE / "stage5I_31_rootB_resume_results.csv"
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
        print(f"NOTE: chain stopped at m={chain_broke_at}, target was {TARGET_M}")

    # Post-hoc comparison
    exp_q2 = {-5: 6.948591527, -4: 6.57003913}
    discovery_grid = {-5: 6.98951810, -4: 6.63644369}
    print()
    print(sep)
    print("POST-HOC COMPARISON (resumed portion)")
    print(sep)
    print(f"{'m':>5} {'RootB cont.':>13} {'exp q2':>12} {'RootB rel%':>11} "
          f"{'discov.grid':>13} {'discov rel%':>12}")
    for mm in sorted(exp_q2.keys(), reverse=True):
        candidates = [k for k in branch.keys() if abs(k - mm) < 0.05]
        exp_val = exp_q2[mm]
        dg_val = discovery_grid[mm]
        dg_rel = 100 * abs(dg_val - exp_val) / exp_val
        if not candidates:
            print(f"{mm:>5}   -- not reached --  {exp_val:12.6f}   "
                  f"{'--':>11} {dg_val:13.6f} {dg_rel:12.4f}")
            continue
        p = branch[candidates[0]]
        rb_rel = 100 * abs(p["f_re"] - exp_val) / exp_val
        print(f"{mm:>5} {p['f_re']:13.6f} {exp_val:12.6f} {rb_rel:11.4f} "
              f"{dg_val:13.6f} {dg_rel:12.4f}")
    print()
    print("IMPORTANT: merge this CSV with stage5I_30's earlier output")
    print("(m=-14 to -5.6, already confirmed) to get the full m=-14..-4 branch.")


if __name__ == "__main__":
    main()
