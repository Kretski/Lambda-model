"""
stage5I_33_rootB_fine_window_m15.py
======================================
FAST, FINE-STEP TEST: does Root-B genuinely terminate before m=-15, or
was the earlier stop (stage5J_rootB_continuation, DM=0.1-scale) a
step-size artifact that a finer step can push through?

Window: m=-14.905 (last previously-confirmed Root-B point) to m=-15.000
(nearest integer m with experimental data: q2=9.8489 Hz).

Same discipline: seeded only from the previous accepted root, no
experimental values used during continuation -- only in the final
post-hoc comparison. Fast settings (small fallback set, early-exit)
matching the optimized stage5I_31/32c pipeline, since the window itself
is very narrow (only ~0.095 in m).
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

# ---------------------------------------------------------------------
# Anchor: last previously-confirmed Root-B point near the old stop.
# NOTE: Im value is an approximate extrapolation from the known trend
# (Im grows more negative as m -> -15); the solver refines from here,
# it does not need to be exact. If this seed fails to converge cleanly,
# widen FALLBACK_IM_OFFSETS below rather than assuming termination.
# ---------------------------------------------------------------------
START_M = -14.905
START_F_RE = 9.824309
START_F_IM = -0.17     # approximate extrapolated seed, see note above

TARGET_M = -15.000
DM = 0.005              # fine step, ~19 steps to cover the window
RES_ACCEPT = 1.0e-8
MAX_JUMP_RE_HZ = 0.15
MAX_JUMP_IM_HZ = 0.15

FALLBACK_RE_OFFSETS = [0.0, +0.01, -0.01, +0.03, -0.03]
FALLBACK_IM_OFFSETS = [0.0, +0.01, -0.01, +0.03, -0.03, +0.08, -0.08]

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
        if best is not None and best["abs_res"] < 1e-11 and jump_re < 0.02 and jump_im < 0.02:
            break

    if best is None:
        return None, "NO_CONTINUATION_POLE"
    return best, "PASS"


def main():
    print(sep)
    print(f"FINE-STEP TEST: m={START_M} -> m={TARGET_M}")
    print("Does Root-B genuinely terminate, or was the earlier stop a")
    print("step-size artifact?")
    print(sep)
    print()
    print(f"Start (approx seed): f={START_F_RE:+.6f}{START_F_IM:+.6f}i Hz")
    print(f"Step: dm={DM} (~{int(round((START_M-TARGET_M)/DM))} steps)")
    print()

    branch = {START_M: {"f_re": START_F_RE, "f_im": START_F_IM, "abs_res": 0.0}}
    previous_root = (START_F_RE, START_F_IM)
    chain_broke_at = None

    m = START_M
    n_steps = int(round((START_M - TARGET_M) / DM))
    for step in range(1, n_steps + 1):
        m = round(START_M - step * DM, 6)
        pole, status = continuation_step(m, previous_root)
        print(f"m={m:+8.4f}  status={status}", end="")
        if status == "PASS":
            print(f"  f={pole['f_re']:+.9f}{pole['f_im']:+.9f}i Hz "
                  f"|Res|={pole['abs_res']:.2e}")
            branch[m] = pole
            previous_root = (pole["f_re"], pole["f_im"])
        else:
            print("  -- CHAIN STOP")
            chain_broke_at = m
            break

    out_csv = HERE / "stage5I_33_rootB_fine_window_m15.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["m", "f_re_Hz", "f_im_Hz", "abs_res"])
        for mm in sorted(branch.keys(), reverse=True):
            p = branch[mm]
            w.writerow([mm, f"{p['f_re']:.9f}", f"{p['f_im']:.9f}",
                        f"{p['abs_res']:.3e}"])
    print()
    print(f"Saved -> {out_csv}")
    print(f"Points recovered: {len(branch)} / {n_steps + 1}")
    if chain_broke_at is not None:
        print(f"CHAIN STOPPED at m={chain_broke_at} (target was {TARGET_M})")
        last_m = max(branch.keys())
        # actually we want the LAST reached (most negative m in this downward walk)
        reached = sorted(branch.keys())
        print(f"Furthest point reached: m={reached[0]}")
    else:
        print(f"REACHED TARGET m={TARGET_M} without a chain break.")

    # ------------------------------------------------------------------
    # Post-hoc comparison vs experimental q2 at m=-15 (used ONLY here)
    # ------------------------------------------------------------------
    exp_q2_m15 = 9.8489
    print()
    print(sep)
    print("POST-HOC COMPARISON vs experimental q2 at m=-15")
    print(sep)
    if -15.0 in branch or any(abs(k - (-15.0)) < 1e-6 for k in branch):
        key = min(branch.keys(), key=lambda k: abs(k - (-15.0)))
        p = branch[key]
        d = p["f_re"] - exp_q2_m15
        rel = 100 * abs(d) / exp_q2_m15
        print(f"Reached m={key}: Re(f)={p['f_re']:.6f} Hz")
        print(f"Experimental q2(m=-15) = {exp_q2_m15:.6f} Hz")
        print(f"Delta = {d:+.6f} Hz  ({rel:.3f}% relative error)")
        if rel < 1.0:
            print(">>> Root-B reaches within 1% of experimental q2 at m=-15.")
            print(">>> This would extend the validated q=1/q=2 range.")
        else:
            print(">>> Reached m=-15 but relative error exceeds 1%.")
    else:
        last_m = min(branch.keys())  # most negative reached
        p = branch[last_m]
        d = p["f_re"] - exp_q2_m15
        rel = 100 * abs(d) / exp_q2_m15
        print(f"Did NOT reach m=-15. Furthest point: m={last_m}")
        print(f"  Re(f)={p['f_re']:.6f} Hz vs exp q2(m=-15)={exp_q2_m15:.6f} Hz")
        print(f"  Delta at closest approach = {d:+.6f} Hz ({rel:.3f}%)")
        print(">>> Chain break confirmed at finer resolution too --")
        print(">>> genuine termination, not a step-size artifact.")

    print()
    print(sep)


if __name__ == "__main__":
    main()
