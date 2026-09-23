"""
stage5I_27_q3_continuation_search.py
=====================================
STAGE 5I-27 — CONTINUATION SEARCH FOR THE q=3 BRANCH
              (seeded from the q=2 Root-B family, NEVER from experiment)

PURPOSE
-------
The branch map (Re vs m, -Im vs m) showed:
  * q=1 and q=2 (Root B) model poles land on the experimental curves;
  * the five NEW deep poles (Im < -1.5) form their OWN smooth family that
    sits ABOVE experimental q=4 and in a much stronger damping regime
    (-Im ~ 1.5-2.1 Hz) -> they are NOT the experimental q=3/q=4 overtones;
  * the v6 "q=3" candidates were partly mislabelled (they sit near the
    experimental q=2 curve, not q=3).

So the genuine, smoothly-continuous q=3 branch has NOT yet been isolated.
This script looks for it the RIGHT way:

  branch identity is established by CONTINUATION, not by nearest experiment.

METHOD (deliberately experiment-blind)
--------------------------------------
1. Anchor on the already-validated q=2 Root-B pole at a starting m
   (default m=-14, the anchor used in Stage 5J/5K).
2. Build q=3 seeds ONLY from model quantities:
     - real part: a modest positive offset above the q=2 Re at that m
       (the next radial branch sits higher in Re), scanned over a small
       grid so we do not hand-pick it;
     - imaginary part: a grid BETWEEN the q=2 damping and the deep-pole
       damping (q=2 has -Im ~ 0.1-0.4 Hz; deep poles ~1.5-2.1 Hz), i.e.
       the physically-expected intermediate window for the next overtone.
   NOTHING here uses the experimental q=3 frequency.
3. At each m, refine every seed with the SAME validated NM->hybr machinery
   used in stage5I_26 (refine_and_polish_candidate). Keep only
   |Res| < RES_ACCEPT poles.
4. Among accepted poles at each m, pick the continuation-consistent one:
   the pole closest (in the complex plane) to the PREVIOUS accepted q=3
   pole — NOT the one closest to any experimental value. The first m is
   seeded from the offset grid; every subsequent m is seeded from the
   previous accepted pole (true continuation).
5. Walk m downward and upward from the anchor, stopping a direction when
   no accepted, continuation-consistent pole is found.

ONLY AFTER the branch is built do we (separately, at the very end) report
how it compares to experimental q=3 — as a TEST of the branch, never as
its definition.

OUTPUT
------
  stage5I_27_q3_branch.csv     - the continuation-built q=3 branch
  console                      - per-m log + final experiment comparison

Place next to:
  stage5I_3_resonance_v2.py
  stage5I_4_full_m_scan_v6_FIXED.py
Run:
  python stage5I_27_q3_continuation_search.py
"""

import sys
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# ── Import the SAME validated pipeline used by stage5I_26 ─────────────────────
try:
    from stage5I_3_resonance_v2 import find_light_ring
    from stage5I_4_full_m_scan_v6_FIXED import refine_and_polish_candidate
    import stage5I_4_full_m_scan_v6_FIXED as _pipeline_mod
except ImportError as e:
    print(f"[ERROR] Cannot import pipeline modules: {e}")
    print("Place this script in the same directory as stage5I_3_resonance_v2.py")
    print("and stage5I_4_full_m_scan_v6_FIXED.py")
    sys.exit(1)

# ── CONFIG ────────────────────────────────────────────────────────────────────

# Anchor: the validated q=2 Root-B pole. (Re, Im) at the anchor m.
# From STATUS.md / Stage 5J-5K, the Root-B q=2 family at m=-14 is near:
#   f_re ~ 9.5967 Hz, f_im ~ -0.1358 Hz
# We anchor the SEARCH on this, but the q=3 branch we seek sits ABOVE it.
ANCHOR_M = -14
ANCHOR_Q2_RE = 9.5967
ANCHOR_Q2_IM = -0.1358

# q=3 seed construction at the anchor (MODEL-ONLY, experiment-blind):
#   Re seed = q2_Re + dRe, scanned over a small positive grid
#   (the next radial branch lies above q=2 in Re; we do NOT hand-pick dRe)
Q3_DRE_GRID = [0.20, 0.30, 0.40, 0.50, 0.60]         # Hz above q=2 Re
#   Im seed = intermediate damping window between q=2 and the deep poles
Q3_IM_GRID  = [-0.20, -0.30, -0.45, -0.60, -0.80]    # Hz

# Expanded Im bound (same as stage5I_26) so intermediate-damping poles
# are not clipped.
COMPLEX_IM_MIN_EXPANDED = -2.5

# Acceptance
RES_ACCEPT = 1.0e-8          # a pole must satisfy |Res| < this
CONT_MAX_JUMP_RE = 0.60      # Hz: max Re move between adjacent m for same branch
CONT_MAX_JUMP_IM = 0.60      # Hz: max Im move between adjacent m
DEDUP_TOL = 5.0e-3           # Hz: treat as same pole if within this

# m range to walk (downward and upward from anchor)
M_MIN = -21
M_MAX = -4

# Guard so we never accidentally seed from experiment:
# (there is simply no experimental array imported in this file at all.)

sep = "=" * 74


def patch_im_bound(new_min):
    for attr in ("COMPLEX_IM_MIN", "IM_MIN", "F_IM_MIN", "COMPLEX_IM_LOWER"):
        if hasattr(_pipeline_mod, attr):
            original = getattr(_pipeline_mod, attr)
            setattr(_pipeline_mod, attr, new_min)
            return attr, original
    print("  WARNING: could not find an Im-bound attribute to patch; "
          "deep seeds may be clipped at the module default.")
    return None, None


def restore_im_bound(attr, original):
    if attr is not None:
        setattr(_pipeline_mod, attr, original)


def try_seed(m, seed_re, seed_im, r_sp, omega_lr):
    """Refine a single (Re,Im) seed. Return dict pole or None."""
    if seed_im < COMPLEX_IM_MIN_EXPANDED:
        return None
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
        return None
    if pole is None:
        return None
    if not (np.isfinite(pole["f_re"]) and np.isfinite(pole["f_im"])):
        return None
    if pole.get("abs_res", 1.0) >= RES_ACCEPT:
        return None
    # physical pole must be in lower half plane
    if pole["f_im"] > 1e-9:
        return None
    return pole


def collect_accepted(m, seeds, r_sp, omega_lr):
    """Refine all seeds at m, dedup, return list of accepted poles."""
    accepted = []
    for sre, sim in seeds:
        p = try_seed(m, sre, sim, r_sp, omega_lr)
        if p is None:
            continue
        dup = False
        for q in accepted:
            if (abs(p["f_re"] - q["f_re"]) < DEDUP_TOL and
                    abs(p["f_im"] - q["f_im"]) < DEDUP_TOL):
                dup = True
                break
        if not dup:
            accepted.append(p)
    return accepted


def pick_continuation(accepted, prev_pole):
    """
    Among accepted poles, choose the one closest to prev_pole in the
    complex plane, subject to the jump guards. Returns pole or None.
    prev_pole is a (Re, Im) tuple = the previous accepted q=3 pole.
    THIS IS THE ONLY SELECTION RULE — no experiment involved.
    """
    best = None
    best_d = None
    for p in accepted:
        dre = abs(p["f_re"] - prev_pole[0])
        dim = abs(p["f_im"] - prev_pole[1])
        if dre > CONT_MAX_JUMP_RE or dim > CONT_MAX_JUMP_IM:
            continue
        d = np.hypot(dre, dim)
        if best is None or d < best_d:
            best, best_d = p, d
    return best


def walk(direction, start_m, first_prev, r_sp_cache):
    """
    Walk m in `direction` (+1 or -1) starting just past start_m.
    first_prev = (Re, Im) of the anchor-accepted q=3 pole to continue from.
    Returns list of (m, pole) accepted along the walk.
    """
    out = []
    prev = first_prev
    m = start_m + direction
    while M_MIN <= m <= M_MAX:
        r = find_light_ring(m)
        if r is None or r[0] is None:
            print(f"  m={m:>4}: light ring None -> stop this direction")
            break
        r_sp, omega_lr = r
        # seed the next m ONLY from the previous accepted pole (continuation)
        seeds = [
            (prev[0], prev[1]),
            (prev[0] + 0.05, prev[1]),
            (prev[0] - 0.05, prev[1]),
            (prev[0], prev[1] - 0.05),
            (prev[0], prev[1] + 0.05),
        ]
        accepted = collect_accepted(m, seeds, r_sp, omega_lr)
        pick = pick_continuation(accepted, prev) if accepted else None
        if pick is None:
            print(f"  m={m:>4}: no continuation-consistent pole "
                  f"({len(accepted)} accepted, none within jump guard) -> stop")
            break
        print(f"  m={m:>4}: Re={pick['f_re']:8.4f}  Im={pick['f_im']:8.4f}  "
              f"|Res|={pick['abs_res']:.2e}")
        out.append((m, pick))
        prev = (pick["f_re"], pick["f_im"])
        m += direction
    return out


def main():
    print(sep)
    print("STAGE 5I-27 — q=3 CONTINUATION SEARCH (experiment-blind)")
    print(sep)
    print(f"Anchor m            : {ANCHOR_M}")
    print(f"Anchor q=2 Root-B   : Re={ANCHOR_Q2_RE}  Im={ANCHOR_Q2_IM}")
    print(f"q=3 Re-offset grid  : {Q3_DRE_GRID} Hz above q=2 Re")
    print(f"q=3 Im seed grid    : {Q3_IM_GRID} Hz (intermediate damping)")
    print(f"Expanded Im bound   : {COMPLEX_IM_MIN_EXPANDED}")
    print(f"|Res| acceptance    : {RES_ACCEPT}")
    print(f"Continuation guards : dRe<{CONT_MAX_JUMP_RE}, dIm<{CONT_MAX_JUMP_IM} Hz")
    print()
    print("NOTE: no experimental frequency is used anywhere as a seed or")
    print("      as a selection rule. Branch identity = continuation only.")
    print()

    attr, original = patch_im_bound(COMPLEX_IM_MIN_EXPANDED)

    branch = {}  # m -> pole
    try:
        # ---- anchor: find the q=3 pole at ANCHOR_M from the offset grid ----
        print(sep)
        print(f"ANCHOR m={ANCHOR_M}: locating q=3 candidate from model offsets")
        print(sep)
        r = find_light_ring(ANCHOR_M)
        if r is None or r[0] is None:
            print("  anchor light ring None -> abort")
            return
        r_sp, omega_lr = r
        anchor_seeds = [(ANCHOR_Q2_RE + dre, sim)
                        for dre in Q3_DRE_GRID for sim in Q3_IM_GRID]
        accepted = collect_accepted(ANCHOR_M, anchor_seeds, r_sp, omega_lr)

        # remove any pole that is actually the q=2 pole itself (too close to anchor)
        q3_candidates = [p for p in accepted
                         if not (abs(p["f_re"] - ANCHOR_Q2_RE) < 0.05 and
                                 abs(p["f_im"] - ANCHOR_Q2_IM) < 0.05)]
        if not q3_candidates:
            print("  No distinct q=3 candidate at anchor from model-only seeds.")
            print("  -> The next branch above q=2 is not reachable with these")
            print("     offsets. Report as: q=3 branch not located (experiment-blind).")
            return

        # pick the LOWEST-Re distinct candidate above q=2 (closest new branch)
        q3_candidates.sort(key=lambda p: p["f_re"])
        anchor_pole = q3_candidates[0]
        print(f"  anchor q=3 candidate: Re={anchor_pole['f_re']:8.4f}  "
              f"Im={anchor_pole['f_im']:8.4f}  |Res|={anchor_pole['abs_res']:.2e}")
        print(f"  ({len(q3_candidates)} distinct candidate(s) above q=2 found)")
        branch[ANCHOR_M] = anchor_pole
        anchor_prev = (anchor_pole["f_re"], anchor_pole["f_im"])

        # ---- walk downward (m more negative) ----
        print()
        print(f"Continuation DOWNWARD (m: {ANCHOR_M} -> {M_MIN})")
        down = walk(-1, ANCHOR_M, anchor_prev, {})
        for m, p in down:
            branch[m] = p

        # ---- walk upward (m less negative) ----
        print()
        print(f"Continuation UPWARD (m: {ANCHOR_M} -> {M_MAX})")
        up = walk(+1, ANCHOR_M, anchor_prev, {})
        for m, p in up:
            branch[m] = p

    finally:
        restore_im_bound(attr, original)

    # ---- save branch ----
    if branch:
        out_csv = HERE / "stage5I_27_q3_branch.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["m", "f_re_Hz", "f_im_Hz", "abs_res"])
            for m in sorted(branch.keys(), reverse=True):
                p = branch[m]
                w.writerow([m, f"{p['f_re']:.6f}", f"{p['f_im']:.6f}",
                            f"{p['abs_res']:.3e}"])
        print()
        print(f"Saved {len(branch)} branch points -> {out_csv}")

    # ---- ONLY NOW: compare to experiment, as a TEST (not a definition) ----
    # Experimental q=3 (qnms.csv, EXP B) — used ONLY here, after the branch
    # is already built by continuation.
    exp_q3 = {
        -21: 11.73305339, -20: 11.48642482, -19: 11.23953467, -18: 10.99149058,
        -17: 10.74739399, -16: 10.50612174, -15: 10.26056441, -14: 10.01552194,
        -13: 9.76372149,  -12: 9.515540164, -11: 9.257350105, -10: 8.992990952,
        -9: 8.725865804,  -8: 8.452768701,  -7: 8.169428269,  -6: 7.869851632,
        -5: 7.563690856,  -4: 7.236964453,
    }
    print()
    print(sep)
    print("POST-HOC TEST: continuation-built branch vs experimental q=3")
    print("(comparison only — did NOT influence branch construction)")
    print(sep)
    print(f"{'m':>4} {'model Re':>10} {'exp q3':>10} {'dRe':>9} {'rel%':>8} {'model Im':>10}")
    rels = []
    for m in sorted(branch.keys(), reverse=True):
        p = branch[m]
        if m in exp_q3:
            d = p["f_re"] - exp_q3[m]
            rel = 100 * abs(d) / exp_q3[m]
            rels.append(rel)
            print(f"{m:>4} {p['f_re']:10.4f} {exp_q3[m]:10.4f} "
                  f"{d:9.4f} {rel:7.3f}% {p['f_im']:10.4f}")
    if rels:
        print()
        print(f"  mean |rel| = {np.mean(rels):.3f}% ,  max |rel| = {np.max(rels):.3f}%")
        print()
        if np.mean(rels) < 0.1:
            print("  VERDICT: continuation branch matches experimental q=3 at the")
            print("           same level as q=1/q=2 -> strong q=3 identification.")
        elif np.mean(rels) < 1.0:
            print("  VERDICT: continuation branch is close to experimental q=3 but")
            print("           not at q=1/q=2 precision -> candidate q=3, not established.")
        else:
            print("  VERDICT: continuation branch does NOT track experimental q=3.")
            print("           This smooth branch is a distinct model feature; the")
            print("           experimental q=3 branch remains unlocated.")
    else:
        print("  No overlap with experimental q=3 m-range.")


if __name__ == "__main__":
    main()
