"""
stage5I_7_validation_report.py
==================================

STAGE 5I-7 — VALIDATION STATISTICS REPORT (FINAL, PRE-PAPER)

Does NOT touch stage5I_3_resonance_v2.py, the search pipeline, or
stage5I_6_branch_assignment.py's assignment logic -- imports and
reuses assign_branches() from that module unchanged. This script only
adds the full descriptive-statistics layer requested before any
paper-level claim:

  For each group (q=1, q=2, m=-11 [separate], UNIDENTIFIED):
    - n (count)
    - coverage (out of 18 m-values, where applicable)
    - MAE
    - RMSE
    - median |delta|
    - max |delta|  (explicitly attributed to its m-value -- outliers
      are SHOWN, not hidden)
    - mean signed delta (checks for systematic bias, not just scatter)
    - std (of signed delta)

  NO p-values or "model significance" claims are computed here --
  per the agreed methodology, that is a separate, later step, only
  after the branch assignment itself is confirmed stable.

Usage:
    python stage5I_7_validation_report.py stage5I_4_full_m_scan_v6_FIXED_results.csv
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stage5I_6_branch_assignment import (
    EXP_B,
    load_csv,
    assign_branches,
)


def full_stats(deltas, label, m_values=None):
    deltas = np.asarray(deltas, dtype=float)
    n = len(deltas)

    print(f"  {label}")
    print(f"    n              = {n}")

    if n == 0:
        print(f"    (no data)")
        print()
        return

    abs_deltas = np.abs(deltas)
    mae = np.mean(abs_deltas)
    rmse = np.sqrt(np.mean(deltas ** 2))
    median_abs = np.median(abs_deltas)
    max_abs = np.max(abs_deltas)
    mean_signed = np.mean(deltas)
    std_signed = np.std(deltas)

    max_idx = int(np.argmax(abs_deltas))
    max_m_str = f" (m={m_values[max_idx]})" if m_values is not None else ""

    print(f"    MAE            = {mae:.5f} Hz")
    print(f"    RMSE           = {rmse:.5f} Hz")
    print(f"    median |delta| = {median_abs:.5f} Hz")
    print(f"    max |delta|    = {max_abs:.5f} Hz{max_m_str}")
    print(f"    mean(delta)    = {mean_signed:+.5f} Hz  "
          f"(sign: {'model > exp on average' if mean_signed > 0 else 'model < exp on average' if mean_signed < 0 else 'unbiased'})")
    print(f"    std(delta)     = {std_signed:.5f} Hz")
    print()


def coverage_line(n_found, total=18):
    pct = 100.0 * n_found / total
    return f"{n_found}/{total} m-values ({pct:.1f}%)"


def main():
    if len(sys.argv) < 2:
        print("Usage: python stage5I_7_validation_report.py <results.csv>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    rows = load_csv(path)
    print(f"Loaded {len(rows)} poles from {path}")

    q1_table, q2_table, unidentified_table, ambiguous_table = assign_branches(rows)

    print()
    print("=" * 100)
    print("STAGE 5I-7 — FULL VALIDATION STATISTICS REPORT")
    print("=" * 100)
    print()
    print("No p-values or model-significance claims computed here --")
    print("that step follows only after this branch assignment is")
    print("independently confirmed stable.")
    print()

    print("-" * 100)
    print(f"GROUP: q=1 (fundamental)   coverage: {coverage_line(len(q1_table))}")
    print("-" * 100)
    deltas_q1 = [r["f_re"] - EXP_B[r["m"]][0] for r in q1_table]
    m_q1 = [r["m"] for r in q1_table]
    full_stats(deltas_q1, "q=1 vs experimental q=1", m_q1)

    print("-" * 100)
    print(f"GROUP: q=2 (continuity-confirmed)   coverage: {coverage_line(len(q2_table))}")
    print("-" * 100)
    deltas_q2 = [r["f_re"] - EXP_B[r["m"]][1] for r in q2_table]
    m_q2 = [r["m"] for r in q2_table]
    full_stats(deltas_q2, "q=2 vs experimental q=2", m_q2)

    if deltas_q2:
        max_idx = int(np.argmax(np.abs(deltas_q2)))
        print(f"    NOTE: largest q=2 residual is at m={m_q2[max_idx]} "
              f"(delta={deltas_q2[max_idx]:+.5f} Hz). This point is RETAINED")
        print(f"    in the statistics above, not excluded as an outlier --")
        print(f"    the q=2 branch is reported honestly as less clean than q=1.")
        print()

    print("-" * 100)
    print(f"GROUP: m=-11 (EXCLUDED/AMBIGUOUS)   n={len(ambiguous_table)} poles")
    print("-" * 100)
    print("  Not a validated branch match -- reporting nearest-line delta")
    print("  for each candidate pole at m=-11 individually, for reference")
    print("  only. NOT included in q=1 or q=2 statistics above.")
    print()
    for r in sorted(ambiguous_table, key=lambda x: x["f_re"]):
        print(f"    q_csv={r['q_csv']}  Re(f)={r['f_re']:.5f} Hz  "
              f"Im(f)={r['f_im']:.5f} Hz  tier={r['tier']:>10}  "
              f"nearest=q{r['nearest_q']}  delta={r['nearest_delta']:+.5f} Hz")
    print()

    print("-" * 100)
    print(f"GROUP: UNIDENTIFIED   n={len(unidentified_table)} poles")
    print("-" * 100)
    print("  These poles are numerically converged (|Res| at machine")
    print("  precision) but do NOT satisfy the frequency-agreement +")
    print("  continuity criteria for q=1 or q=2. Reporting nearest-line")
    print("  delta for reference ONLY -- NOT a validated match, NOT")
    print("  included in any branch statistic.")
    print()
    for r in sorted(unidentified_table, key=lambda x: (x["m"], x["f_re"]), reverse=True):
        print(f"    m={r['m']:>4}  q_csv={r['q_csv']}  Re(f)={r['f_re']:.5f} Hz  "
              f"Im(f)={r['f_im']:.5f} Hz  tier={r['tier']:>10}  "
              f"nearest=q{r['nearest_q']}  delta={r['nearest_delta']:+.5f} Hz")
    print()

    print("=" * 100)
    print("SUGGESTED SUMMARY LANGUAGE (for reference)")
    print("=" * 100)
    print()
    q2_mae = np.mean(np.abs(deltas_q2)) if deltas_q2 else float("nan")
    print("  \"The fundamental branch (q=1) is reproduced with sub-")
    print("  millihertz-to-few-millihertz frequency agreement across")
    print(f"  {coverage_line(len(q1_table))}, while a continuity-identified q=2")
    print(f"  branch is also recovered across {coverage_line(len(q2_table))},")
    print("  with larger but still systematic frequency residuals")
    print(f"  (MAE={q2_mae:.5f} Hz).")
    print(f"  {len(unidentified_table)} additional numerically converged poles")
    print("  remain unassigned rather than being forced into experimental")
    if ambiguous_table:
        excluded_m = sorted(set(r["m"] for r in ambiguous_table))
        print(f"  overtone labels. {len(excluded_m)} m-value(s) "
              f"({excluded_m}) are excluded from")
        print("  automatic branch assignment as ambiguous boundary case(s).\"")
    else:
        print("  overtone labels.\"")


if __name__ == "__main__":
    main()
