"""
stage5K_1_observable_extraction_rootB.py
===========================================
STAGE 5K.1 (Root B) -- extract ringdown observables (f_R, tau, Q) from
the VALIDATED portion of the Root-B branch trajectory only.

Root B (see stage5J_rootB_continuation.py) is adopted as the PRIMARY
WORKING q=2 branch for this analysis: it is smoothly continuous with
the rest of the q=2 family across m=-19..-4 (unlike Root A, whose
Im(f)=-0.309 at m=-14 is a 3x discontinuous jump from its neighbors),
and it tracks cleanly to m~-14.9055 -- much farther than Root A's
m~-14.7725 -- before an abrupt, non-fold breakdown. Root A's original
observables (stage5K_1_observables.csv) are RETAINED as a documented
secondary/diagnostic branch, not deleted or overwritten.

Reads stage5J_rootB_branch_trajectory.csv, keeps only rows with
abs_res < RES_VALID (1e-8, same threshold used everywhere else in this
pipeline), and stops strictly at the last validated point -- no
extrapolation past the breakdown.

Convention: omega = 2*pi*(f_re + i*f_im), so
    f_R (Hz) = f_re
    tau (s)  = 1 / (2*pi*|f_im|)
    Q        = f_re / (2*|f_im|)
"""

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE / "stage5J_rootB_branch_trajectory.csv"
OUTPUT_CSV = HERE / "stage5K_1_observables_rootB.csv"

RES_VALID = 1.0e-8


def main():
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found. Run stage5J_rootB_continuation.py first.")
        sys.exit(1)

    rows = []
    with open(INPUT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            m = float(row["m"])
            f_re = float(row["f_re"])
            f_im = float(row["f_im"])
            abs_res = float(row["abs_res"])
            if abs_res >= RES_VALID or not np.isfinite(abs_res):
                continue
            rows.append({"m": m, "f_re": f_re, "f_im": f_im, "abs_res": abs_res})

    if not rows:
        print(f"ERROR: no validated rows found (all abs_res >= {RES_VALID:.1e}).")
        sys.exit(1)

    by_m = {}
    for row in rows:
        if row["m"] not in by_m or row["abs_res"] < by_m[row["m"]]["abs_res"]:
            by_m[row["m"]] = row
    rows = sorted(by_m.values(), key=lambda r: r["m"], reverse=True)

    out_rows = []
    for row in rows:
        f_re, f_im = row["f_re"], row["f_im"]
        tau_s = 1.0 / (2.0 * np.pi * abs(f_im))
        Q = f_re / (2.0 * abs(f_im))
        out_rows.append({
            "m": row["m"], "f_re": f_re, "f_im": f_im,
            "f_R_hz": f_re, "tau_s": tau_s, "Q": Q,
            "abs_res": row["abs_res"],
        })

    print("=" * 100)
    print(f"STAGE 5K.1 (Root B) -- VALIDATED OBSERVABLES (n={len(out_rows)} points, "
          f"m in [{out_rows[-1]['m']:.4f}, {out_rows[0]['m']:.4f}])")
    print("=" * 100)
    print(f"{'m':>10}  {'f_R (Hz)':>12}  {'tau (s)':>12}  {'Q':>10}  {'|Res|':>10}")
    for r in out_rows[::10] + [out_rows[-1]]:  # print every 10th + the last, table is long
        print(f"{r['m']:+10.4f}  {r['f_R_hz']:12.6f}  {r['tau_s']:12.6f}  "
              f"{r['Q']:10.4f}  {r['abs_res']:10.3e}")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print()
    print(f"LAST VALIDATED POINT: m = {out_rows[-1]['m']:.4f}")
    print("Nothing beyond this m is included -- no extrapolation into the breakdown region.")
    print(f"written to {OUTPUT_CSV} ({len(out_rows)} rows total; only every 10th printed above)")


if __name__ == "__main__":
    main()
