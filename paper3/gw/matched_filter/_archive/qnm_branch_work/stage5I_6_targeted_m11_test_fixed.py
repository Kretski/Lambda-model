"""
stage5I_6_targeted_m11_test.py
==============================

STAGE 5I-6 TARGETED INDEPENDENT TEST — m=-11

Purpose
-------
Independently validate the m=-11 branch assignment BEFORE changing
stage5I_6_branch_assignment.py or stage5I_7_validation_report.py.

IMPORTANT:
    This script intentionally does NOT import stage5I_6_branch_assignment.py.
    It operates directly on the raw Stage 5I-4 v6 CSV.

The test asks:

1. Does m=-11 contain the expected q=1 pole?
2. Is q_csv=3 a valid candidate for physical q=2 based on frequency agreement?
3. Does q_csv=3 satisfy the same continuity criterion using the already
   confirmed neighboring q=2 points m=-10 and m=-12?
4. Is q_csv=2 clearly rejected as physical q=2 by the frequency criterion?
5. Does q_csv=2 continue the already observed off-pattern residual trend?
6. Is there any competing m=-11 candidate that also satisfies the q=2
   frequency criterion?

Only if ALL required checks pass do we print:

    TARGETED m=-11 TEST: PASS

This script does NOT modify any file.

Usage
-----
    python stage5I_6_targeted_m11_test.py \
        stage5I_4_full_m_scan_v6_FIXED_results.csv
"""

import sys
import csv
from pathlib import Path

import numpy as np


# ============================================================
# EXPERIMENTAL FREQUENCIES
# ============================================================

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


# ============================================================
# FIXED VALIDATION PARAMETERS
# ============================================================

FREQ_TOLERANCE_HZ = 0.05

M_TARGET = -11

# Neighboring q=2 assignments already confirmed independently
M_LEFT = -10
M_RIGHT = -12

# Expected q=2 frequencies
EXP_Q2_TARGET = EXP_B[M_TARGET][1]
EXP_Q2_LEFT = EXP_B[M_LEFT][1]
EXP_Q2_RIGHT = EXP_B[M_RIGHT][1]


# ============================================================
# CSV LOADER
# ============================================================

def load_csv(path):
    rows = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for raw in reader:
            row = {}

            for key, value in raw.items():
                if value is None:
                    continue

                key = key.strip()
                value = value.strip()

                if key in ("m", "q", "q_csv"):
                    try:
                        row[key] = int(float(value))
                    except ValueError:
                        row[key] = int(value)

                elif key in (
                    "f_re",
                    "f_im",
                    "exp",
                    "delta",
                    "residual",
                    "abs_res",
                    "|Res|",
                ):
                    try:
                        row[key] = float(value)
                    except ValueError:
                        pass

                else:
                    row[key] = value

            rows.append(row)

    return rows


# ============================================================
# COLUMN HELPERS
# ============================================================

def get_value(row, *names):
    for name in names:
        if name in row:
            return row[name]

    raise KeyError(
        f"None of the columns {names} found. "
        f"Available columns: {list(row.keys())}"
    )


def f_re(row):
    return float(get_value(row, "f_re", "Re(f)", "re", "real", "f_model_hz"))


def f_im(row):
    return float(get_value(row, "f_im", "Im(f)", "im", "imag", "f_model_im_hz"))


def q_csv(row):
    return int(get_value(row, "q_csv", "q"))


def m_value(row):
    return int(get_value(row, "m"))


# ============================================================
# DISPLAY
# ============================================================

def print_candidate(row, label):
    re = f_re(row)
    im = f_im(row)

    dq1 = re - EXP_B[M_TARGET][0]
    dq2 = re - EXP_B[M_TARGET][1]
    dq3 = re - EXP_B[M_TARGET][2]
    dq4 = re - EXP_B[M_TARGET][3]

    print(
        f"{label:<18} "
        f"q_csv={q_csv(row):<2d} "
        f"Re={re:11.8f} "
        f"Im={im:11.8f} "
        f"dq1={dq1:+.8f} "
        f"dq2={dq2:+.8f} "
        f"dq3={dq3:+.8f} "
        f"dq4={dq4:+.8f}"
    )


# ============================================================
# MAIN TARGETED TEST
# ============================================================

def main():

    if len(sys.argv) < 2:
        print(
            "Usage: python stage5I_6_targeted_m11_test.py "
            "<stage5I_4_full_m_scan_v6_FIXED_results.csv>"
        )
        sys.exit(1)

    path = Path(sys.argv[1])

    if not path.exists():
        print(f"ERROR: file not found: {path}")
        sys.exit(1)

    rows = load_csv(path)

    target_rows = [
        r for r in rows
        if m_value(r) == M_TARGET
    ]

    print()
    print("=" * 100)
    print("STAGE 5I-6 TARGETED INDEPENDENT TEST — m=-11")
    print("=" * 100)
    print()
    print(f"Input CSV: {path}")
    print(f"Total poles in CSV: {len(rows)}")
    print(f"m target: {M_TARGET}")
    print()

    if not target_rows:
        print("FAIL: no m=-11 poles found.")
        sys.exit(2)

    print(f"Found {len(target_rows)} poles for m={M_TARGET}")
    print()

    for row in sorted(target_rows, key=f_re):
        print_candidate(row, "candidate")

    print()
    print("-" * 100)
    print("REFERENCE q=2 CONTINUITY ANCHORS")
    print("-" * 100)

    print(
        f"m={M_LEFT}:  confirmed q=2 experimental frequency = "
        f"{EXP_Q2_LEFT:.8f} Hz"
    )

    print(
        f"m={M_TARGET}: experimental q=2 frequency = "
        f"{EXP_Q2_TARGET:.8f} Hz"
    )

    print(
        f"m={M_RIGHT}: confirmed q=2 experimental frequency = "
        f"{EXP_Q2_RIGHT:.8f} Hz"
    )

    print()
    print(
        "Continuity direction: "
        f"m={M_LEFT} -> m={M_TARGET} -> m={M_RIGHT}"
    )

    # ========================================================
    # FIND CANDIDATES
    # ========================================================

    q1_candidates = []
    q2_candidates = []

    for row in target_rows:

        re = f_re(row)

        dq1 = abs(re - EXP_B[M_TARGET][0])
        dq2 = abs(re - EXP_B[M_TARGET][1])

        if dq1 < FREQ_TOLERANCE_HZ:
            q1_candidates.append(row)

        if dq2 < FREQ_TOLERANCE_HZ:
            q2_candidates.append(row)

    # ========================================================
    # CHECK 1 — q=1
    # ========================================================

    print()
    print("-" * 100)
    print("CHECK 1 — q=1 candidate")
    print("-" * 100)

    q1_pass = len(q1_candidates) == 1

    if q1_pass:
        row = q1_candidates[0]
        print_candidate(row, "q=1 PASS")
        print(
            f"Frequency residual = "
            f"{f_re(row) - EXP_B[M_TARGET][0]:+.8f} Hz"
        )
    else:
        print(
            f"FAIL: expected exactly one q=1 frequency candidate, "
            f"found {len(q1_candidates)}"
        )

    # ========================================================
    # CHECK 2 — q_csv=3 is q=2 frequency candidate
    # ========================================================

    print()
    print("-" * 100)
    print("CHECK 2 — q_csv=3 frequency agreement with EXP q=2")
    print("-" * 100)

    q3_rows = [
        r for r in target_rows
        if q_csv(r) == 3
    ]

    if len(q3_rows) != 1:
        print(
            f"FAIL: expected exactly one q_csv=3 candidate, "
            f"found {len(q3_rows)}"
        )
        q3_row = None
        q3_freq_pass = False
    else:
        q3_row = q3_rows[0]

        dq2_signed = f_re(q3_row) - EXP_Q2_TARGET
        dq2_abs = abs(dq2_signed)

        q3_freq_pass = dq2_abs < FREQ_TOLERANCE_HZ

        print_candidate(q3_row, "q_csv=3")

        print(
            f"Experimental q=2 = {EXP_Q2_TARGET:.8f} Hz"
        )
        print(
            f"Model Re(f)      = {f_re(q3_row):.8f} Hz"
        )
        print(
            f"delta            = {dq2_signed:+.8f} Hz"
        )
        print(
            f"|delta|          = {dq2_abs:.8f} Hz"
        )
        print(
            f"tolerance        = {FREQ_TOLERANCE_HZ:.8f} Hz"
        )

        print(
            "RESULT: "
            + ("PASS" if q3_freq_pass else "FAIL")
        )

    # ========================================================
    # CHECK 3 — q_csv=3 continuity
    # ========================================================

    print()
    print("-" * 100)
    print("CHECK 3 — q_csv=3 continuity")
    print("-" * 100)

    if q3_row is None:
        continuity_pass = False
    else:

        candidate_re = f_re(q3_row)

        # The candidate must sit between the neighboring
        # confirmed q=2 frequencies in the same ordering.
        lower_neighbor = min(EXP_Q2_LEFT, EXP_Q2_RIGHT)
        upper_neighbor = max(EXP_Q2_LEFT, EXP_Q2_RIGHT)

        ordering_pass = (
            lower_neighbor < candidate_re < upper_neighbor
        )

        # Local distance to each confirmed q=2 anchor.
        d_left = abs(candidate_re - EXP_Q2_LEFT)
        d_right = abs(candidate_re - EXP_Q2_RIGHT)

        # The experimental target itself must also lie between
        # the two neighboring q=2 points.
        exp_ordering_pass = (
            lower_neighbor < EXP_Q2_TARGET < upper_neighbor
        )

        print(
            f"left  anchor m={M_LEFT}:  "
            f"{EXP_Q2_LEFT:.8f} Hz"
        )

        print(
            f"target m={M_TARGET}: "
            f"{candidate_re:.8f} Hz"
        )

        print(
            f"right anchor m={M_RIGHT}: "
            f"{EXP_Q2_RIGHT:.8f} Hz"
        )

        print()

        print(
            f"Candidate inside neighbor interval: "
            f"{ordering_pass}"
        )

        print(
            f"Experimental q=2 inside neighbor interval: "
            f"{exp_ordering_pass}"
        )

        print(
            f"Distance to left anchor:  {d_left:.8f} Hz"
        )

        print(
            f"Distance to right anchor: {d_right:.8f} Hz"
        )

        continuity_pass = (
            ordering_pass
            and exp_ordering_pass
        )

        print(
            "RESULT: "
            + ("PASS" if continuity_pass else "FAIL")
        )

    # ========================================================
    # CHECK 4 — q_csv=2 must NOT qualify as q=2
    # ========================================================

    print()
    print("-" * 100)
    print("CHECK 4 — q_csv=2 rejection as physical q=2")
    print("-" * 100)

    q2_csv_rows = [
        r for r in target_rows
        if q_csv(r) == 2
    ]

    if len(q2_csv_rows) != 1:
        print(
            f"FAIL: expected exactly one q_csv=2 candidate, "
            f"found {len(q2_csv_rows)}"
        )
        q2_rejection_pass = False
    else:

        q2_csv_row = q2_csv_rows[0]

        delta_q2 = f_re(q2_csv_row) - EXP_Q2_TARGET
        abs_delta_q2 = abs(delta_q2)

        q2_rejection_pass = (
            abs_delta_q2 >= FREQ_TOLERANCE_HZ
        )

        print_candidate(q2_csv_row, "q_csv=2")

        print(
            f"delta to EXP q=2 = {delta_q2:+.8f} Hz"
        )

        print(
            f"|delta|           = {abs_delta_q2:.8f} Hz"
        )

        print(
            f"tolerance         = {FREQ_TOLERANCE_HZ:.8f} Hz"
        )

        print(
            "RESULT: "
            + (
                "PASS — correctly rejected"
                if q2_rejection_pass
                else "FAIL — still within q=2 tolerance"
            )
        )

    # ========================================================
    # CHECK 5 — off-pattern continuity of q_csv=2 residual
    # ========================================================

    print()
    print("-" * 100)
    print("CHECK 5 — q_csv=2 off-pattern residual trend")
    print("-" * 100)

    # Existing unidentified q=2 residual sequence:
    #
    # m=-14 : -0.19974545
    # m=-13 : -0.22031494
    # m=-12 : -0.24421345
    #
    # m=-11 should continue the same direction if it is
    # genuinely the same unidentified branch.

    known_trend = {
        -14: -0.19974545,
        -13: -0.22031494,
        -12: -0.24421345,
    }

    expected_m11_delta = (
        f_re(q2_csv_rows[0]) - EXP_Q2_TARGET
        if q2_csv_rows
        else None
    )

    if expected_m11_delta is None:

        offpattern_pass = False

    else:

        # More negative than m=-12.
        monotonic_extension = (
            expected_m11_delta < known_trend[-12]
        )

        # Check that it is not remotely close to zero/q=2.
        clearly_off_branch = (
            abs(expected_m11_delta) > FREQ_TOLERANCE_HZ
        )

        offpattern_pass = (
            monotonic_extension
            and clearly_off_branch
        )

        print(
            "Known off-pattern q=2 residual sequence:"
        )

        for m in sorted(known_trend):
            print(
                f"  m={m:3d}: dq2={known_trend[m]:+.8f} Hz"
            )

        print(
            f"  m={M_TARGET:3d}: dq2={expected_m11_delta:+.8f} Hz"
        )

        print()

        print(
            f"Continues more-negative trend: "
            f"{monotonic_extension}"
        )

        print(
            f"Clearly outside q=2 tolerance: "
            f"{clearly_off_branch}"
        )

        print(
            "RESULT: "
            + ("PASS" if offpattern_pass else "FAIL")
        )

    # ========================================================
    # CHECK 6 — no competing q=2 candidate
    # ========================================================

    print()
    print("-" * 100)
    print("CHECK 6 — uniqueness of q=2 frequency candidate")
    print("-" * 100)

    unique_q2_candidate = (
        len(q2_candidates) == 1
        and q3_row is not None
        and q2_candidates[0] is q3_row
    )

    print(
        f"Number of m=-11 candidates inside "
        f"|delta_q2| < {FREQ_TOLERANCE_HZ} Hz: "
        f"{len(q2_candidates)}"
    )

    if q2_candidates:
        for r in q2_candidates:
            print_candidate(r, "q=2 candidate")

    print(
        "RESULT: "
        + (
            "PASS — q_csv=3 is unique"
            if unique_q2_candidate
            else "FAIL — competing q=2 candidate exists"
        )
    )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    all_pass = (
        q1_pass
        and q3_freq_pass
        and continuity_pass
        and q2_rejection_pass
        and offpattern_pass
        and unique_q2_candidate
    )

    print()
    print("=" * 100)

    if all_pass:

        print("TARGETED m=-11 TEST: PASS")
        print("=" * 100)

        print()
        print("Independent conclusion:")
        print()
        print(
            "1. q_csv=1 remains the q=1 pole."
        )

        print(
            "2. q_csv=3 satisfies the q=2 frequency criterion."
        )

        print(
            "3. q_csv=3 lies on the same local frequency-continuity "
            "sequence as the confirmed m=-10 and m=-12 q=2 branch."
        )

        print(
            "4. q_csv=2 is decisively outside the q=2 frequency tolerance."
        )

        print(
            "5. q_csv=2 continues the previously observed "
            "off-pattern residual trend."
        )

        print(
            "6. No competing m=-11 pole satisfies the q=2 "
            "frequency criterion."
        )

        print()
        print(
            "RECOMMENDATION:"
        )
        print(
            "m=-11 may now be released from the blanket exclusion "
            "and processed by the standard branch-assignment logic."
        )

        print()
        print(
            "IMPORTANT:"
        )
        print(
            "This script has NOT modified stage5I_6 or stage5I_7."
        )

        print()

        sys.exit(0)

    else:

        print("TARGETED m=-11 TEST: FAIL")
        print("=" * 100)

        print()
        print(
            "Do NOT change the branch assignment or summary statistics."
        )

        print()
        sys.exit(3)


if __name__ == "__main__":
    main()