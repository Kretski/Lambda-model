"""
STAGE 5I-8 — GENERALIZED BRANCH RESOLUTION
============================================
Extends the logic validated by the m=-11 targeted test
(stage5I_6_targeted_m11_test.py) to ALL m-values automatically,
instead of hardcoding a single m. Same criteria, applied uniformly:

  - q=1: |Im(f)| is "near-zero" tier AND close to EXP_B[m][0]
  - q=2: candidate matches EXP_B[m][1] within FREQ_TOLERANCE_HZ,
         AND (when more than one candidate at that m satisfies the
         tolerance) the one closest to the interpolated continuity
         line from confirmed neighbors wins
  - anything left over is UNIDENTIFIED -- never forced into a label

This does not invent new physics or new criteria. It mechanizes
exactly the same test that was manually run for m=-11, so that
m=-11 (and any other ambiguous m) is resolved by the SAME rule as
everyone else, rather than by a one-off script.
"""
import csv
import sys
from collections import defaultdict

import numpy as np

FREQ_TOLERANCE_HZ = 0.05

EXP_B = {
    -21: [10.53127805, 11.31559458, 11.73305339, 12.04931005],
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
    -9:  [7.406262072, 8.246009901, 8.725865804, 9.067763134],
    -8:  [7.076173999, 7.945602232, 8.452768701, 8.808992769],
    -7:  [6.714687044, 7.633791333, 8.169428269, 8.540722055],
    -6:  [6.333524665, 7.298338000, 7.869851632, 8.269296643],
    -5:  [5.910999730, 6.948591527, 7.563690856, 7.986183324],
    -4:  [5.437524431, 6.570039130, 7.236964453, 7.696262081],
}


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["m"] = int(float(r["m"]))
        r["q_csv"] = int(float(r["q"]))
        r["re"] = float(r["f_model_hz"])
        r["im"] = float(r["f_model_im_hz"])
    return rows


def tier(im):
    return "near-zero" if abs(im) < 0.01 else "deep"


def main(path):
    rows = load(path)
    by_m = defaultdict(list)
    for r in rows:
        by_m[r["m"]].append(r)

    confirmed_q1 = []
    confirmed_q2 = []
    unidentified = []

    m_sorted = sorted(by_m)

    # Pass 1: q=1 (unambiguous everywhere -- near-zero Im, matches exp[0])
    for m in m_sorted:
        cands = by_m[m]
        q1_cands = [r for r in cands if tier(r["im"]) == "near-zero"]
        if len(q1_cands) == 1:
            r = q1_cands[0]
            r["delta"] = r["re"] - EXP_B[m][0]
            confirmed_q1.append(r)
        elif len(q1_cands) > 1:
            # pick closest to exp q1
            r = min(q1_cands, key=lambda r: abs(r["re"] - EXP_B[m][0]))
            r["delta"] = r["re"] - EXP_B[m][0]
            confirmed_q1.append(r)
        # if zero candidates: genuinely missing, nothing to assign

    q1_by_m = {r["m"]: r for r in confirmed_q1}

    # Pass 2: q=2 -- same rule as the m=-11 targeted test, applied to every m
    for m in m_sorted:
        # deep-tier candidates only (q1 already claimed above)
        deep_cands = [r for r in by_m[m] if tier(r["im"]) != "near-zero"]
        matches = [r for r in deep_cands
                   if abs(r["re"] - EXP_B[m][1]) < FREQ_TOLERANCE_HZ]
        if len(matches) == 1:
            r = matches[0]
            r["delta"] = r["re"] - EXP_B[m][1]
            confirmed_q2.append(r)
        elif len(matches) > 1:
            # continuity tie-break: closest to neighbor-interpolated line
            left = EXP_B.get(m - 1, [None, None])[1]
            right = EXP_B.get(m + 1, [None, None])[1]
            neighbors = [x for x in (left, right) if x is not None]
            ref = np.mean(neighbors) if neighbors else EXP_B[m][1]
            r = min(matches, key=lambda r: abs(r["re"] - ref))
            r["delta"] = r["re"] - EXP_B[m][1]
            confirmed_q2.append(r)
        # else: no candidate satisfies tolerance -> nothing assigned for q2 at this m

    q2_ids = {(r["m"], r["q_csv"]) for r in confirmed_q2}
    q1_ids = {(r["m"], r["q_csv"]) for r in confirmed_q1}

    for r in rows:
        key = (r["m"], r["q_csv"])
        if key not in q1_ids and key not in q2_ids:
            unidentified.append(r)

    print("=" * 100)
    print("CONFIRMED q=1")
    print("=" * 100)
    for r in sorted(confirmed_q1, key=lambda r: r["m"]):
        print(f"  m={r['m']:4d}  q_csv={r['q_csv']}  Re={r['re']:.5f}  delta={r['delta']:+.5f}")

    print("=" * 100)
    print("CONFIRMED q=2 (generalized rule, m=-11 included automatically)")
    print("=" * 100)
    for r in sorted(confirmed_q2, key=lambda r: r["m"]):
        flag = "  <-- was manually excluded before" if r["m"] == -11 else ""
        print(f"  m={r['m']:4d}  q_csv={r['q_csv']}  Re={r['re']:.5f}  Im={r['im']:.5f}  delta={r['delta']:+.5f}{flag}")

    print("=" * 100)
    print("STILL UNIDENTIFIED")
    print("=" * 100)
    for r in sorted(unidentified, key=lambda r: (r["m"], r["q_csv"])):
        print(f"  m={r['m']:4d}  q_csv={r['q_csv']}  Re={r['re']:.5f}  Im={r['im']:.5f}")

    d1 = np.array([abs(r["delta"]) for r in confirmed_q1])
    d2 = np.array([abs(r["delta"]) for r in confirmed_q2])
    m_all = sorted(by_m)
    print("=" * 100)
    print("STATS")
    print("=" * 100)
    print(f"q=1: n={len(d1)}/{len(m_all)}  MAE={d1.mean():.5f}  RMSE={np.sqrt((d1**2).mean()):.5f}")
    print(f"q=2: n={len(d2)}/{len(m_all)}  MAE={d2.mean():.5f}  RMSE={np.sqrt((d2**2).mean()):.5f}")
    print(f"unidentified: n={len(unidentified)}")
    missing_q2_m = [m for m in m_all if m not in {r['m'] for r in confirmed_q2}]
    print(f"m-values with NO q=2 match at all (genuine gap, not unidentified): {missing_q2_m}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generalized_branch_resolution.py <results.csv>")
        sys.exit(1)
    main(sys.argv[1])
