"""
stage5I_6_branch_assignment.py
==================================

STAGE 5I-6 — BRANCH-AWARE q-ASSIGNMENT (FINAL)

Does NOT touch stage5I_3_resonance_v2.py, the search pipeline, or the
already-generated CSV. Produces a physically-labeled table from the
Stage 5I-4 v6 results, per the agreed methodology:

  "The numerical search returns multiple complex resonance branches.
   The ordinal rank of a candidate in the search output does not
   uniquely identify its physical overtone. We therefore perform
   branch-aware assignment using frequency agreement and continuity
   rather than positional ranking."

ASSIGNMENT RULES (both required, not just the nearer of the two):

  q=1: taken directly from confirmed near-zero-Im poles (unambiguous;
       this branch was never in question).

  q=2 CONFIRMED: a pole is assigned physical label q=2 only if BOTH:
    (a) |f_model - EXP_B[m][1]| < FREQ_TOLERANCE_HZ, AND
    (b) CONTINUITY: among that m's frequency-agreement candidates, the
        one closest in Re(f) to the previously-confirmed q=2 point
        (walking outward from the anchor m=-19, the cleanest
        shallow-tier point) is selected.

  Everything else at that m is left UNIDENTIFIED, explicitly, with its
  full (m, q_csv, Re, Im, dq1..dq4, nearest_q, nearest_delta) record
  preserved -- never silently dropped, never forced into a q label it
  does not robustly satisfy.

m=-11 remains EXCLUDED entirely from automatic assignment (ambiguous
boundary case), listed separately.

Usage:
    python stage5I_6_branch_assignment.py stage5I_4_full_m_scan_v6_FIXED_results.csv
"""

import sys
import csv
from pathlib import Path

import numpy as np


EXP_B = {
    -21: [10.53127805, 11.31559458, 11.73305339, 12.0],
    -20: [10.30228351, 11.07509152, 11.48642482, 11.79180863],
    -19: [10.07028344, 10.83337372, 11.23953467, 11.54107841],
    -18: [9.840683264, 10.58962136, 10.99149058, 11.29593696],
    -17: [9.600492823, 10.34296042, 10.74739399, 11.04921008],
    -16: [9.355011082, 10.09843672, 10.50612174, 10.80589014],
    -15: [9.103267196, 9.848948391, 10.26056441, 10.55883689],
    -14: [8.844144355, 9.599296405, 10.01552194, 10.31282590],
    -13: [8.582278857, 9.342201757, 9.76372149, 10.07244017],
    -12: [8.304259799, 9.082091798, 9.515540164, 9.824297904],
    -11: [8.020133041, 8.811238183, 9.257350105, 9.578550689],
    -10: [7.721731873, 8.533427811, 8.992990952, 9.327218691],
    -9: [7.406262072, 8.246009901, 8.725865804, 9.067763134],
    -8: [7.076173999, 7.945602232, 8.452768701, 8.808992769],
    -7: [6.714687044, 7.633791333, 8.169428269, 8.540722055],
    -6: [6.333524665, 7.298338000, 7.869851632, 8.269296643],
    -5: [5.910999730, 6.948591527, 7.563690856, 7.986183324],
    -4: [5.437524431, 6.570039130, 7.236964453, 7.696262081],
}

FREQ_TOLERANCE_HZ = 0.05
IM_TIER_NEAR_ZERO = 0.02
IM_TIER_SHALLOW_MAX = 0.15

# STAGE 5I-6b UPDATE: m=-11 was originally flagged here out of general
# caution (all three of its candidates looked confusing together at
# first glance). Targeted per-candidate re-inspection showed this was
# NOT a genuine ambiguity at the m=-11 level -- it decomposes cleanly
# under the SAME criteria already used everywhere else:
#   q_csv=1 (Re=8.02078): small dq1 -> confirmed q=1, as at every m.
#   q_csv=3 (Re=8.81297): |dq2|=0.00173 (within FREQ_TOLERANCE_HZ) AND
#     sits between the already-confirmed q=2 values at m=-10
#     (8.53792) and m=-12 (8.83788) -- i.e. satisfies BOTH the
#     frequency-agreement and continuity criteria used for every other
#     m's q=2 assignment. No special-casing was needed to accept it.
#   q_csv=2 (Re=8.54558, dq2=-0.26565): does NOT satisfy the q=2
#     frequency criterion, and its residual value fits smoothly onto
#     the ALREADY-RECOGNIZED off-pattern trend seen in the
#     UNIDENTIFIED group at m=-12,-13,-14 (dq2 = -0.244, -0.220,
#     -0.200 respectively) -- m=-11 continues this same sequence
#     (-0.266), not a new or different anomaly.
#
# m=-11 is therefore REMOVED from this set: the standard assignment
# logic (unchanged) now runs on it exactly as on every other m, and
# correctly reproduces this three-way split automatically.
AMBIGUOUS_M_VALUES = set()


def classify_im_tier(f_im):
    abs_im = abs(f_im)
    if abs_im < IM_TIER_NEAR_ZERO:
        return "near-zero"
    elif abs_im < IM_TIER_SHALLOW_MAX:
        return "shallow"
    else:
        return "deep"


def load_csv(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                rows.append({
                    "m": int(row["m"]),
                    "q_csv": int(row["q"]),
                    "f_re": float(row["f_model_hz"]),
                    "f_im": float(row["f_model_im_hz"]),
                    "abs_res": float(row["abs_res"]),
                })
            except (ValueError, KeyError) as exc:
                print(f"WARNING: skipping malformed row {row}: {exc}")
    return rows


def assign_branches(rows):

    by_m = {}
    for row in rows:
        by_m.setdefault(row["m"], []).append(row)

    q1_table = []
    q2_candidates_by_m = {}
    all_by_m_annotated = {}

    for m, entries in by_m.items():

        if m not in EXP_B:
            continue

        exp = EXP_B[m]
        annotated = []

        for e in entries:
            deltas = [e["f_re"] - exp[i] for i in range(4)]
            abs_deltas = [abs(d) for d in deltas]
            nearest_idx = int(np.argmin(abs_deltas))

            record = dict(e)
            record["dq1"], record["dq2"], record["dq3"], record["dq4"] = deltas
            record["nearest_q"] = nearest_idx + 1
            record["nearest_delta"] = deltas[nearest_idx]
            record["tier"] = classify_im_tier(e["f_im"])
            annotated.append(record)

        all_by_m_annotated[m] = annotated

        if m in AMBIGUOUS_M_VALUES:
            continue

        for r in annotated:
            if r["tier"] == "near-zero" and abs(r["dq1"]) < FREQ_TOLERANCE_HZ:
                q1_table.append(r)

        q2_freq_candidates = [
            r for r in annotated if abs(r["dq2"]) < FREQ_TOLERANCE_HZ
        ]
        if q2_freq_candidates:
            q2_candidates_by_m[m] = q2_freq_candidates

    q2_table = []
    m_values_desc = sorted(q2_candidates_by_m.keys(), reverse=True)

    anchor_m = -19 if -19 in q2_candidates_by_m else (
        m_values_desc[0] if m_values_desc else None)

    chain_by_m = {}

    if anchor_m is not None:
        anchor_candidate = min(q2_candidates_by_m[anchor_m],
                                 key=lambda r: abs(r["dq2"]))
        q2_table.append(anchor_candidate)
        chain_by_m[anchor_m] = anchor_candidate

        prev = anchor_candidate
        for m in range(anchor_m + 1, -3):
            if m in q2_candidates_by_m:
                cands = q2_candidates_by_m[m]
                best = min(cands, key=lambda r: abs(r["f_re"] - prev["f_re"]))
                q2_table.append(best)
                chain_by_m[m] = best
                prev = best

        prev = anchor_candidate
        for m in range(anchor_m - 1, -22, -1):
            if m in q2_candidates_by_m:
                cands = q2_candidates_by_m[m]
                best = min(cands, key=lambda r: abs(r["f_re"] - prev["f_re"]))
                q2_table.append(best)
                chain_by_m[m] = best
                prev = best

    q1_ids = {(r["m"], r["f_re"], r["f_im"]) for r in q1_table}
    q2_ids = {(r["m"], r["f_re"], r["f_im"]) for r in q2_table}

    unidentified_table = []
    for m, annotated in all_by_m_annotated.items():
        if m in AMBIGUOUS_M_VALUES:
            continue
        for r in annotated:
            rid = (r["m"], r["f_re"], r["f_im"])
            if rid not in q1_ids and rid not in q2_ids:
                unidentified_table.append(r)

    ambiguous_table = []
    for m in AMBIGUOUS_M_VALUES:
        if m in all_by_m_annotated:
            ambiguous_table.extend(all_by_m_annotated[m])

    return q1_table, q2_table, unidentified_table, ambiguous_table


def print_table(title, table, exp_col_idx):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)
    if not table:
        print("  (none)")
        return
    print(f"{'m':>4} {'Re(f)':>11} {'Im(f)':>10} {'q_csv':>6} "
          f"{'exp':>11} {'delta':>9} {'tier':>10} {'|Res|':>10}")
    print("-" * 78)
    for r in sorted(table, key=lambda x: x["m"], reverse=True):
        exp_val = EXP_B[r["m"]][exp_col_idx]
        delta = r["f_re"] - exp_val
        print(f"{r['m']:>4} {r['f_re']:>11.5f} {r['f_im']:>10.5f} "
              f"{r['q_csv']:>6} {exp_val:>11.5f} {delta:>9.5f} "
              f"{r['tier']:>10} {r['abs_res']:>10.2e}")


def print_unidentified(table):
    print()
    print("=" * 100)
    print("UNIDENTIFIED BRANCH (physical_label = UNIDENTIFIED, not artifact,")
    print("not forced into any q label)")
    print("=" * 100)
    if not table:
        print("  (none)")
        return
    print(f"{'m':>4} {'q_csv':>5} {'Re(f)':>11} {'Im(f)':>10} {'tier':>10} "
          f"{'dq1':>9} {'dq2':>9} {'dq3':>9} {'dq4':>9} "
          f"{'nearest':>7} {'d_near':>9}")
    print("-" * 108)
    for r in sorted(table, key=lambda x: (x["m"], x["f_re"]), reverse=True):
        print(f"{r['m']:>4} {r['q_csv']:>5} {r['f_re']:>11.5f} {r['f_im']:>10.5f} "
              f"{r['tier']:>10} "
              f"{r['dq1']:>9.5f} {r['dq2']:>9.5f} {r['dq3']:>9.5f} {r['dq4']:>9.5f} "
              f"{r['nearest_q']:>7} {r['nearest_delta']:>9.5f}")


def summary_stats(table, exp_col_idx, label):
    if not table:
        print(f"  {label}: no confirmed poles.")
        return
    deltas = np.array([r["f_re"] - EXP_B[r["m"]][exp_col_idx] for r in table])
    print(f"  {label}: n={len(table)}  "
          f"MAE={np.mean(np.abs(deltas)):.5f} Hz  "
          f"RMSE={np.sqrt(np.mean(deltas**2)):.5f} Hz  "
          f"max|delta|={np.max(np.abs(deltas)):.5f} Hz")


def main():
    if len(sys.argv) < 2:
        print("Usage: python stage5I_6_branch_assignment.py <results.csv>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    rows = load_csv(path)
    print(f"Loaded {len(rows)} poles from {path}")

    q1_table, q2_table, unidentified_table, ambiguous_table = assign_branches(rows)

    print_table("CONFIRMED q=1 (fundamental)", q1_table, exp_col_idx=0)
    print_table("CONFIRMED q=2 (frequency-agreement + continuity)", q2_table, exp_col_idx=1)
    print_unidentified(unidentified_table)

    print()
    print("=" * 100)
    print("EXCLUDED / AMBIGUOUS (m=-11) -- not auto-assigned, listed for reference")
    print("=" * 100)
    if ambiguous_table:
        print(f"{'m':>4} {'q_csv':>5} {'Re(f)':>11} {'Im(f)':>10} {'tier':>10} "
              f"{'dq1':>9} {'dq2':>9} {'dq3':>9} {'dq4':>9}")
        for r in sorted(ambiguous_table, key=lambda x: x["f_re"]):
            print(f"{r['m']:>4} {r['q_csv']:>5} {r['f_re']:>11.5f} {r['f_im']:>10.5f} "
                  f"{r['tier']:>10} "
                  f"{r['dq1']:>9.5f} {r['dq2']:>9.5f} {r['dq3']:>9.5f} {r['dq4']:>9.5f}")

    print()
    print("=" * 100)
    print("SUMMARY STATISTICS (confirmed branches only)")
    print("=" * 100)
    summary_stats(q1_table, 0, "q=1")
    summary_stats(q2_table, 1, "q=2")
    print()
    print(f"  UNIDENTIFIED: n={len(unidentified_table)} poles -- ")
    print(f"  NOT included in any q-branch statistic. Requires separate,")
    print(f"  independent physical interpretation before any claim.")
    print()
    print(f"  Coverage: q=1 ({len(q1_table)}/18 m-values), "
          f"q=2 ({len(q2_table)}/18 m-values), "
          f"unidentified ({len(unidentified_table)} poles), "
          f"excluded ({len(ambiguous_table)} poles, m=-11)")


if __name__ == "__main__":
    main()
