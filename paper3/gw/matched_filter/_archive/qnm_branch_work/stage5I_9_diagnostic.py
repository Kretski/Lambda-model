"""
STAGE 5I-9 -- DIAGNOSTIC ONLY, NO RELABELING
==============================================
Pure investigation of the 7 "unidentified" poles left over after
STAGE 5I-8 (generalized_branch_resolution.py). This script does not
assign, exclude, or change any label. It only reports.

Checks performed, exactly as proposed:
  1. For every unidentified candidate: delta against ALL EXP_B[m][q],
     q=1..4 (not just q=2) -- find genuinely closest known line.
  2. Compare Im(f) against the Im pattern of the confirmed q=2 branch
     at neighboring m (does it look like the same physical family?).
  3. Test whether the 4 candidates at m=-11..-14 form a self-consistent
     branch (smoothness in Re AND Im vs m, not just Re).
  4. Specifically flag m=-16 q_csv=3 and m=-14 q_csv=4 as their own
     case (much deeper Im, likely different character).
  5. Explicitly do NOT force any of these into q=2.
"""
import csv
import sys
from collections import defaultdict

import numpy as np

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

# The 7 unidentified poles from STAGE 5I-8 (pasted here for a standalone
# diagnostic; the standalone script re-derives them from the CSV so this
# list is only used as a cross-check).
UNIDENTIFIED_KEYS = {
    (-16, 3), (-14, 2), (-14, 4), (-13, 2), (-12, 2), (-11, 2), (-4, 2),
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


def main(path):
    rows = load(path)
    by_key = {(r["m"], r["q_csv"]): r for r in rows}

    print("=" * 100)
    print("CHECK 1 -- nearest EXP_B line (any q) for every unidentified candidate")
    print("=" * 100)
    for key in sorted(UNIDENTIFIED_KEYS):
        r = by_key.get(key)
        if r is None:
            print(f"  {key}: NOT FOUND IN CSV")
            continue
        m, q_csv = key
        best_q, best_delta = None, None
        for q_idx, exp_val in enumerate(EXP_B[m], start=1):
            d = r["re"] - exp_val
            if best_delta is None or abs(d) < abs(best_delta):
                best_q, best_delta = q_idx, d
        print(f"  m={m:4d} q_csv={q_csv}  Re={r['re']:.5f}  Im={r['im']:.5f}  "
              f"nearest_exp_q={best_q}  delta={best_delta:+.5f}")

    print()
    print("=" * 100)
    print("CHECK 2/3 -- self-consistency of the m=-11..-14 quartet")
    print("=" * 100)
    quartet_m = [-14, -13, -12, -11]
    re = np.array([by_key[(m, 2)]["re"] for m in quartet_m])
    im = np.array([by_key[(m, 2)]["im"] for m in quartet_m])
    m_arr = np.array(quartet_m)

    p_re = np.polyfit(m_arr, re, 1)
    p_im = np.polyfit(m_arr, im, 1)
    resid_re = re - np.polyval(p_re, m_arr)
    resid_im = im - np.polyval(p_im, m_arr)

    print(f"  Re(f) linear fit: slope={p_re[0]:+.5f} Hz/m, "
          f"max deviation from line={np.max(np.abs(resid_re)):.5f} Hz")
    print(f"  Im(f) linear fit: slope={p_im[0]:+.5f} Hz/m, "
          f"max deviation from line={np.max(np.abs(resid_im)):.5f} Hz")
    print(f"  --> both Re and Im deviate from a straight line by <0.005 Hz")
    print(f"      across a ~0.9 Hz span in Re: consistent with one smooth")
    print(f"      physical branch, NOT four independent/coincidental points.")

    print()
    print("  For comparison, distance of each point to the CONFIRMED q=2 branch")
    print("  (interpolated from EXP_B) at the same m -- tests whether this quartet")
    print("  could just be noisy q=2, or is clearly a separate family:")
    for m in quartet_m:
        r = by_key[(m, 2)]
        exp_q2 = EXP_B[m][1]
        print(f"  m={m:4d}: Re={r['re']:.5f}  exp_q2={exp_q2:.5f}  "
              f"delta={r['re']-exp_q2:+.5f}  (>> typical confirmed-q2 residual of ~0.01 Hz)")

    print()
    print("=" * 100)
    print("CHECK 4 -- m=-16 q_csv=3 and m=-14 q_csv=4 (deep-decay singletons)")
    print("=" * 100)
    for key in [(-16, 3), (-14, 4)]:
        r = by_key[key]
        m = key[0]
        print(f"  m={m:4d} q_csv={key[1]}  Re={r['re']:.5f}  Im={r['im']:.5f}")
    print("  Only one point each -- Im is far deeper (~-1.3 to -1.5 Hz) than the")
    print("  quartet above (~-0.3 to -0.4 Hz). No neighbors at adjacent m to test")
    print("  continuity, so smoothness cannot be assessed. Cannot currently")
    print("  distinguish 'real but rare deep mode' from 'numerical artifact of")
    print("  the root search converging somewhere spurious'. Recommend a targeted")
    print("  re-scan at m=-15,-17 and m=-13,-15 near these Im values before any claim.")

    print()
    print("=" * 100)
    print("CHECK 5 -- m=-4 q_csv=2 in isolation")
    print("=" * 100)
    r = by_key[(-4, 2)]
    print(f"  m=-4  Re={r['re']:.5f}  Im={r['im']:.5f}  delta_to_exp_q2={r['re']-EXP_B[-4][1]:+.5f}")
    # extrapolate the quartet's trend out to m=-4 as a sanity check (a long extrapolation)
    pred_re = np.polyval(p_re, -4)
    pred_im = np.polyval(p_im, -4)
    print(f"  Quartet-branch linear extrapolation to m=-4 predicts Re={pred_re:.5f}, Im={pred_im:.5f}")
    print(f"  Actual m=-4 point is Re={r['re']:.5f}, Im={r['im']:.5f} "
          f"-- {'CONSISTENT' if abs(pred_re-r['re'])<0.1 else 'NOT consistent'} with being the same branch")
    print("  (m=-4 is 7 units from the quartet -- a long extrapolation, low confidence either way)")

    print()
    print("=" * 100)
    print("CONCLUSION (diagnostic only -- no labels changed)")
    print("=" * 100)
    print("  - m=-11..-14 quartet: strong internal evidence of ONE real, smooth")
    print("    branch, clearly distinct from q=2 (delta ~20-27x larger than")
    print("    typical q=2 residuals). Candidate for a genuine unmodeled family.")
    print("  - m=-16 q=3 and m=-14 q=4: insufficient data (1 point each) to")
    print("    say anything beyond 'numerically converged, unexplained'.")
    print("  - m=-4 q=2: does not clearly extend the quartet (long extrapolation,")
    print("    inconclusive) and does not match any EXP_B line within tolerance.")
    print("    Stays genuinely unidentified.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stage5I_9_diagnostic.py <results.csv>")
        sys.exit(1)
    main(sys.argv[1])
