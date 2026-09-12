"""
stage5K_1_observable_extraction.py
=====================================
STAGE 5K.1 -- extract ringdown observables (f_R, tau, Q) from the
VALIDATED portion of the Stage 5J branch trajectory only.

Reads stage5J_branch_trajectory.csv (written by
stage5J_branch_continuation.py) and keeps only rows with
abs_res < RES_VALID -- the same acceptance threshold used everywhere
else in this pipeline (1e-8). This deliberately EXCLUDES the
breakdown row itself and everything after it: the table and any plot
built from it stops at the last validated point (m ~ -14.7725) with
no extrapolation past it, so nothing here visually or numerically
implies knowledge of what happens for m beyond that point.

Convention: the residual-equations use omega = 2*pi*(f_re + i*f_im),
so:
    f_R (Hz)     = f_re
    omega_I      = 2*pi*f_im
    tau (s)      = 1 / |omega_I| = 1 / (2*pi*|f_im|)
    Q            = omega_R / (2*|omega_I|) = f_re / (2*|f_im|)

Writes a clean CSV (m, f_re, f_im, f_R_hz, tau_s, Q) and prints it,
sorted by m from -14.0 toward the critical region.
"""

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE / "stage5J_branch_trajectory.csv"
OUTPUT_CSV = HERE / "stage5K_1_observables.csv"

RES_VALID = 1.0e-8


def main():
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found. Run stage5J_branch_continuation.py first.")
        sys.exit(1)

    rows = []
    with open(INPUT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            m = float(row["m"])
            f_re = float(row["f_re"])
            f_im = float(row["f_im"])
            abs_res = float(row["abs_res"])
            if abs_res >= RES_VALID or not np.isfinite(abs_res):
                continue  # not validated -- excluded, no extrapolation
            rows.append({"m": m, "f_re": f_re, "f_im": f_im, "abs_res": abs_res})

    if not rows:
        print("ERROR: no validated rows found (all abs_res >= "
              f"{RES_VALID:.1e}). Nothing to extract.")
        sys.exit(1)

    # De-duplicate identical m values (coarse+fine phases overlap at
    # several m's) -- keep the row with the smaller abs_res at each m.
    by_m = {}
    for row in rows:
        if row["m"] not in by_m or row["abs_res"] < by_m[row["m"]]["abs_res"]:
            by_m[row["m"]] = row
    rows = sorted(by_m.values(), key=lambda r: r["m"], reverse=True)  # -14.0 -> more negative

    out_rows = []
    for row in rows:
        f_re, f_im = row["f_re"], row["f_im"]
        omega_I = 2.0 * np.pi * f_im
        tau_s = 1.0 / abs(omega_I)
        Q = f_re / (2.0 * abs(f_im))
        out_rows.append({
            "m": row["m"], "f_re": f_re, "f_im": f_im,
            "f_R_hz": f_re, "tau_s": tau_s, "Q": Q,
            "abs_res": row["abs_res"],
        })

    print("=" * 100)
    print(f"STAGE 5K.1 -- VALIDATED OBSERVABLES (n={len(out_rows)} points, "
          f"m in [{out_rows[-1]['m']:.4f}, {out_rows[0]['m']:.4f}])")
    print("=" * 100)
    print(f"{'m':>10}  {'f_R (Hz)':>12}  {'tau (s)':>12}  {'Q':>10}  {'|Res|':>10}")
    for r in out_rows:
        print(f"{r['m']:+10.4f}  {r['f_R_hz']:12.6f}  {r['tau_s']:12.6f}  "
              f"{r['Q']:10.4f}  {r['abs_res']:10.3e}")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print()
    print(f"LAST VALIDATED POINT: m = {out_rows[-1]['m']:.4f}")
    print("Nothing beyond this m is included -- no extrapolation into the")
    print("breakdown region.")
    print()
    print(f"written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
