"""
stage5I_32d_deep_branch_resume_downward.py
=============================================
RESUME the deep-pole branch continuation downward from the last
successfully reached point (m=-14) toward m=-21, closing the chain-break
gap left by stage5I_32c.

Same discipline: seeded only from the previous accepted root, no
experimental values used until the final post-hoc comparison. Same
fast/coarse settings (DM=0.5, reduced fallback seeds, loosened early-exit)
that successfully completed the upward direction in stage5I_32c.
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

# Resume point: last successfully reached point in stage5I_32c
START_M = -14
START_F_RE = 10.649226
START_F_IM = -1.481607

TARGET_M = -21
DM = 0.5
RES_ACCEPT = 1.0e-8
MAX_JUMP_RE_HZ = 1.0
MAX_JUMP_IM_HZ = 1.0
COMPLEX_IM_MIN_EXPANDED = -3.0

FALLBACK_RE_OFFSETS = [0.0, +0.1, -0.1]
FALLBACK_IM_OFFSETS = [0.0, +0.1, -0.1]

sep = "=" * 74


def patch_im_bound(new_min):
    for attr in ("COMPLEX_IM_MIN", "IM_MIN", "F_IM_MIN", "COMPLEX_IM_LOWER"):
        if hasattr(_pipeline_mod, attr):
            original = getattr(_pipeline_mod, attr)
            setattr(_pipeline_mod, attr, new_min)
            return attr, original
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
        if best is not None and best["abs_res"] < 1e-10 and jump_re < 0.5 and jump_im < 0.5:
            break

    if best is None:
        return None, "NO_CONTINUATION_POLE"
    return best, "PASS"


def main():
    print(sep)
    print(f"RESUME DOWNWARD: m={START_M} -> m={TARGET_M}")
    print(sep)
    print(f"Start: f={START_F_RE:+.6f}{START_F_IM:+.6f}i Hz")
    print()

    attr, original = patch_im_bound(COMPLEX_IM_MIN_EXPANDED)

    branch = {START_M: {"f_re": START_F_RE, "f_im": START_F_IM, "abs_res": 0.0}}
    previous_root = (START_F_RE, START_F_IM)
    chain_broke_at = None

    try:
        m = START_M
        n_steps = int(round((START_M - TARGET_M) / DM))
        for step in range(1, n_steps + 1):
            m = round(START_M - step * DM, 6)
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
    finally:
        restore_im_bound(attr, original)

    out_csv = HERE / "stage5I_32d_deep_branch_downward_resume.csv"
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

    exp_q3 = {
        -21: 11.73305339, -20: 11.48642482, -19: 11.23953467, -18: 10.99149058,
        -17: 10.74739399, -16: 10.50612174, -15: 10.26056441,
    }
    exp_q4 = {
        -21: 12.04931005, -20: 11.79180863, -19: 11.54107841, -18: 11.29593696,
        -17: 11.04921008, -16: 10.80589014, -15: 10.55883689,
    }

    print()
    print(sep)
    print("POST-HOC COMPARISON (downward resume portion)")
    print(sep)
    print(f"{'m':>5} {'branch Re':>11} {'exp q3':>9} {'dRe(q3)':>9} "
          f"{'exp q4':>9} {'dRe(q4)':>9}")
    for mm in sorted(exp_q3.keys(), reverse=True):
        candidates = [k for k in branch.keys() if abs(k - mm) < 0.05]
        if not candidates:
            print(f"{mm:>5}   -- not reached --")
            continue
        p = branch[candidates[0]]
        d_q3 = p["f_re"] - exp_q3[mm]
        d_q4 = p["f_re"] - exp_q4[mm]
        rel_q3 = 100 * abs(d_q3) / exp_q3[mm]
        rel_q4 = 100 * abs(d_q4) / exp_q4[mm]
        print(f"{mm:>5} {p['f_re']:11.4f} {exp_q3[mm]:9.4f} {d_q3:+9.4f} "
              f"({rel_q3:.2f}%)  {exp_q4[mm]:9.4f} {d_q4:+9.4f} ({rel_q4:.2f}%)")


if __name__ == "__main__":
    main()
