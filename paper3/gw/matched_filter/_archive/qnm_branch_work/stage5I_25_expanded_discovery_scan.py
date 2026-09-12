"""
stage5I_25_expanded_discovery_scan.py
==========================================

STAGE 5I-25 -- EXPANDED FULL-m DISCOVERY SCAN

Per the agreed plan: Stage 5I-24 showed root B satisfies the SAME
frozen stage5I_6 assignment criteria at 4 diagnostic points
(m=-11..-14). This does NOT establish whether root-B-like points exist
systematically across the FULL m=-21..-4 range, or whether they were a
local coincidence at those 4 points.

THIS SCRIPT: re-runs Stage 5I-4's OWN complex-plane discovery
methodology (same underlying resonance_factor/refine_and_polish_candidate
functions, same physics) for ALL 18 m values, with an EXPANDED Re
search window per m (wider than the original's
COMPLEX_DISCOVERY_RE_ABOVE_FLR=3.5 Hz).

Every discovered pole is evaluated against the EXACT SAME frozen
stage5I_6_branch_assignment.py criteria (FREQ_TOLERANCE_HZ, continuity)
-- NO new tolerance, NO new rule.

DOES NOT MODIFY:
  - stage5I_4_full_m_scan_v6_FIXED_results.csv (untouched)
  - stage5I_6_branch_assignment.py (imported, unchanged)
  - stage5I_7_validation_report.py

Output written to a NEW, separate file.
"""

import sys
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stage5I_4_full_m_scan_v6_FIXED import (
    find_light_ring,
    safe_complex_res,
    refine_and_polish_candidate,
    M_VALUES,
)
from stage5I_6_branch_assignment import EXP_B, FREQ_TOLERANCE_HZ


EXPANDED_RE_ABOVE_FLR = 5.0
EXPANDED_RE_POINTS = 80
EXPANDED_IM_VALUES = [-0.05, -0.1, -0.15, -0.2, -0.25, -0.3, -0.35,
                        -0.4, -0.5, -0.7, -0.9, -1.2]
TOP_N_CANDIDATES = 10

OUTPUT_CSV = HERE / "stage5I_25_expanded_results.csv"


def expanded_complex_discovery(m, r_sp, omega_lr, f_lr):
    f_re_grid = np.linspace(f_lr + 0.05, f_lr + EXPANDED_RE_ABOVE_FLR,
                              EXPANDED_RE_POINTS)

    grid_vals = np.full((len(EXPANDED_IM_VALUES), len(f_re_grid)), np.inf)

    for j, f_im in enumerate(EXPANDED_IM_VALUES):
        for i, f_re in enumerate(f_re_grid):
            z = safe_complex_res(float(f_re), float(f_im), m, r_sp, omega_lr)
            if z is not None:
                val = abs(z)
                if np.isfinite(val):
                    grid_vals[j, i] = val

    candidates = []
    n_im, n_re = grid_vals.shape
    for j in range(n_im):
        for i in range(n_re):
            val = grid_vals[j, i]
            if not np.isfinite(val):
                continue
            neighbours = []
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    if dj == 0 and di == 0:
                        continue
                    jj, ii = j + dj, i + di
                    if 0 <= jj < n_im and 0 <= ii < n_re:
                        neighbours.append(grid_vals[jj, ii])
            if neighbours and val <= min(neighbours):
                candidates.append({
                    "f_re": float(f_re_grid[i]),
                    "f_im": float(EXPANDED_IM_VALUES[j]),
                    "abs_res": float(val),
                })

    candidates.sort(key=lambda c: c["abs_res"])
    deduped = []
    for c in candidates:
        if all(abs(c["f_re"] - d["f_re"]) > 0.1 for d in deduped):
            deduped.append(c)
        if len(deduped) >= TOP_N_CANDIDATES:
            break

    return deduped


def validate_candidates(m, candidates, r_sp, omega_lr):
    validated = []
    for cand in candidates:
        pole = refine_and_polish_candidate(
            cand["f_re"], cand["f_im"], m, r_sp, omega_lr,
            real_axis_abs_res=float("nan"), is_complex_seed=True,
        )
        if pole is not None:
            validated.append(pole)
    return validated


def dedupe_poles(poles, min_separation=0.02):
    poles_sorted = sorted(poles, key=lambda p: p["f_re"])
    unique = []
    for p in poles_sorted:
        matched = False
        for idx, u in enumerate(unique):
            if abs(p["f_re"] - u["f_re"]) <= min_separation and \
               abs(p["f_im"] - u["f_im"]) <= min_separation:
                if p["abs_res"] < u["abs_res"]:
                    unique[idx] = p
                matched = True
                break
        if not matched:
            unique.append(p)
    return unique


def main():
    print("=" * 100)
    print("STAGE 5I-25 -- EXPANDED FULL-m DISCOVERY SCAN")
    print("=" * 100)
    print()
    print(f"Widened window: {EXPANDED_RE_ABOVE_FLR} Hz above f_lr "
          f"({EXPANDED_RE_POINTS} pts) x {len(EXPANDED_IM_VALUES)} Im values")
    print()
    print("stage5I_4/6/7 outputs remain UNTOUCHED. Frozen assignment")
    print(f"criteria imported unchanged (FREQ_TOLERANCE_HZ={FREQ_TOLERANCE_HZ}).")
    print()

    all_rows = []
    q2_freq_candidates_by_m = {}

    for m in M_VALUES:
        r_sp, omega_lr = find_light_ring(m)
        if r_sp is None:
            print(f"[m={m}] no light ring, skipping")
            continue

        f_lr = omega_lr / (2 * np.pi)
        print(f"[m={m}] f_lr={f_lr:.4f} Hz, scanning...")

        candidates = expanded_complex_discovery(m, r_sp, omega_lr, f_lr)
        validated = validate_candidates(m, candidates, r_sp, omega_lr)
        unique = dedupe_poles(validated)

        print(f"  {len(unique)} unique validated pole(s) found")

        exp_q2 = EXP_B[m][1] if m in EXP_B else None

        for pole in unique:
            dq2 = pole["f_re"] - exp_q2 if exp_q2 is not None else np.nan
            passes = abs(dq2) < FREQ_TOLERANCE_HZ if exp_q2 is not None else False

            row = dict(m=m, f_re=pole["f_re"], f_im=pole["f_im"],
                       abs_res=pole["abs_res"], dq2=dq2,
                       passes_freq_criterion=passes)
            all_rows.append(row)

            if passes:
                q2_freq_candidates_by_m.setdefault(m, []).append(pole)

            flag = " <-- passes frozen q=2 freq criterion" if passes else ""
            print(f"    Re={pole['f_re']:.6f} Im={pole['f_im']:.6f} "
                  f"|Res|={pole['abs_res']:.2e} dq2={dq2:+.6f}{flag}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["m", "f_re", "f_im", "abs_res",
                                                  "dq2", "passes_freq_criterion"])
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print()
    print(f"Results written to: {OUTPUT_CSV}")

    print()
    print("=" * 100)
    print("COVERAGE SUMMARY")
    print("=" * 100)
    print()
    print(f"m-values with >=1 pole passing frozen q=2 frequency criterion: "
          f"{len(q2_freq_candidates_by_m)}/{len(M_VALUES)}")
    print()

    for m in sorted(q2_freq_candidates_by_m.keys(), reverse=True):
        best = min(q2_freq_candidates_by_m[m], key=lambda p: abs(p["f_re"] - EXP_B[m][1]))
        dq2 = best["f_re"] - EXP_B[m][1]
        print(f"  m={m}: Re={best['f_re']:.6f} dq2={dq2:+.6f} |Res|={best['abs_res']:.2e}")

    print()
    covered_m = set(q2_freq_candidates_by_m.keys())
    missing_m = set(M_VALUES) - covered_m
    print(f"Missing (no pole found passing frozen criterion): "
          f"{sorted(missing_m, reverse=True)}")

    if len(covered_m) >= 3:
        m_sorted = sorted(covered_m)
        re_sorted = [min(q2_freq_candidates_by_m[m], key=lambda p: abs(p["f_re"] - EXP_B[m][1]))["f_re"]
                     for m in m_sorted]
        print()
        print("Continuity check on newly-covered points (Re vs m):")
        for i in range(1, len(m_sorted)):
            gap = m_sorted[i] - m_sorted[i - 1]
            step = re_sorted[i] - re_sorted[i - 1]
            print(f"  m={m_sorted[i-1]} -> m={m_sorted[i]} (gap={gap}): "
                  f"Re step = {step:+.6f}")

    print()
    print("=" * 100)
    print("As agreed: this is a coverage/sensitivity finding, evaluated with")
    print("UNCHANGED frozen criteria. stage5I_4/6/7 remain the official")
    print("baseline until/unless a formal decision is made to supersede them.")


if __name__ == "__main__":
    main()
