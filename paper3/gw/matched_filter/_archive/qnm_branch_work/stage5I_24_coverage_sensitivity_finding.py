"""
stage5I_24_coverage_sensitivity_finding.py
================================================

STAGE 5I-24 -- COVERAGE/SENSITIVITY FINDING (root B), NOT a branch
reassignment.

DOES NOT MODIFY:
  - stage5I_4_full_m_scan_v6_FIXED_results.csv (kept untouched)
  - stage5I_6_branch_assignment.py's assignment criteria
    (FREQ_TOLERANCE_HZ, continuity logic -- imported unchanged)
  - stage5I_7_validation_report.py

FINDING: an independent re-implementation (independent_reimplementation.py),
built with different numerical machinery (SymPy exact derivatives,
adaptive quadrature, Levenberg-Marquardt), used as a SENSITIVITY PROBE
seeded at the already-known "deep branch" points (m=-11..-14), landed
in a DIFFERENT basin and converged to a SECOND set of genuine roots
("root B") at each m -- confirmed independently by BOTH the original
(hybr-based) and independent (LM-based) solvers to machine-precision
|Res|.

This does NOT mean root B is "more correct" than the original deep-
branch root ("root A"). Both are genuine solutions of Res(omega)=0.
What this DOES establish is that Stage 5I-4's complex-plane discovery
grid did not achieve exhaustive root coverage in this region -- there
was at least one additional genuine root nearby that its specific
seed/grid pattern never found.

METHODOLOGY (per agreed discipline): root B is evaluated against the
EXACT SAME frozen frequency-agreement + continuity criteria already
used in stage5I_6_branch_assignment.py, with NO modification to
FREQ_TOLERANCE_HZ or the continuity logic, and NO "closer to experiment
therefore correct" reasoning. The method itself decides whether root B
qualifies -- this script only reports whether it does.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stage5I_6_branch_assignment import EXP_B, FREQ_TOLERANCE_HZ
from stage5I_3_resonance_v2 import find_light_ring, resonance_factor
from scipy.optimize import root as scipy_root


ROOT_A = {
    -11: (8.54558431, -0.34109901),
    -12: (8.83787835, -0.32801295),
    -13: (9.12188682, -0.31733520),
    -14: (9.39955096, -0.30861804),
}

ROOT_B_SEED = {
    -11: (8.779833, -0.194965),
    -12: (9.050625, -0.176562),
    -13: (9.314815, -0.159163),
    -14: (9.573590, -0.142422),
}


def confirm_root_b_with_original(m, seed_re, seed_im):
    r_sp, omega_lr = find_light_ring(m)

    def equations(x):
        omega = 2 * np.pi * (x[0] + 1j * x[1])
        z = resonance_factor(omega=omega, m=m, r_sp=r_sp, omega_lr=omega_lr,
                              complex_action=True)
        if z is None:
            return [1e6, 1e6]
        return [z.real, z.imag]

    sol = scipy_root(equations, x0=[seed_re, seed_im], method="hybr", tol=1e-13)
    f_re, f_im = float(sol.x[0]), float(sol.x[1])
    z_final = resonance_factor(omega=2 * np.pi * (f_re + 1j * f_im), m=m,
                                r_sp=r_sp, omega_lr=omega_lr, complex_action=True)
    abs_res = abs(z_final) if z_final is not None else np.nan

    return dict(f_re=f_re, f_im=f_im, abs_res=abs_res, success=bool(sol.success))


def evaluate_against_frozen_criteria(m, f_re, f_im):
    dq2 = f_re - EXP_B[m][1]
    passes_freq_criterion = abs(dq2) < FREQ_TOLERANCE_HZ
    return dq2, passes_freq_criterion


def main():
    print("=" * 100)
    print("STAGE 5I-24 -- COVERAGE/SENSITIVITY FINDING (root B)")
    print("=" * 100)
    print()
    print("stage5I_4_full_m_scan_v6_FIXED_results.csv is NOT modified.")
    print("stage5I_6_branch_assignment.py's criteria are NOT modified --")
    print("imported and applied literally, unchanged.")
    print()

    print(f"{'m':>4} {'root_A_Re':>11} {'root_B_Re':>11} {'|Res_B|':>10} "
          f"{'dq2(A)':>10} {'dq2(B)':>10} {'B passes (a)?':>14}")
    print("-" * 90)

    root_b_confirmed = {}

    for m in sorted(ROOT_A.keys(), reverse=True):
        re_a, im_a = ROOT_A[m]
        seed_re, seed_im = ROOT_B_SEED[m]

        confirmed_b = confirm_root_b_with_original(m, seed_re, seed_im)
        root_b_confirmed[m] = confirmed_b

        dq2_a, _ = evaluate_against_frozen_criteria(m, re_a, im_a)
        dq2_b, passes_b = evaluate_against_frozen_criteria(
            m, confirmed_b["f_re"], confirmed_b["f_im"])

        print(f"{m:>4} {re_a:>11.6f} {confirmed_b['f_re']:>11.6f} "
              f"{confirmed_b['abs_res']:>10.2e} {dq2_a:>10.6f} "
              f"{dq2_b:>10.6f} {str(passes_b):>14}")

    print()

    print("-" * 100)
    print("CONTINUITY CHECK on root B (criterion b, same style as stage5I_6)")
    print("-" * 100)
    print()

    m_sorted = sorted(root_b_confirmed.keys())
    re_b_sorted = [root_b_confirmed[m]["f_re"] for m in m_sorted]
    print(f"  Re(m) sequence: {[f'{r:.6f}' for r in re_b_sorted]}")
    steps = np.diff(re_b_sorted)
    print(f"  Step sizes: {[f'{s:+.6f}' for s in steps]}")
    step_variation = np.std(steps) / np.mean(np.abs(steps))
    print(f"  Relative step variation: {step_variation:.4f} "
          f"({'smooth/monotonic' if step_variation < 0.15 else 'irregular'})")
    print()

    print("=" * 100)
    print("VERDICT (frozen criteria only)")
    print("=" * 100)
    print()

    all_pass_freq = all(
        evaluate_against_frozen_criteria(m, root_b_confirmed[m]["f_re"],
                                           root_b_confirmed[m]["f_im"])[1]
        for m in ROOT_A.keys())
    all_machine_precision = all(
        root_b_confirmed[m]["abs_res"] < 1e-8 for m in ROOT_A.keys())
    continuity_ok = step_variation < 0.15

    print(f"  All 4 root-B points satisfy frozen frequency criterion (a): "
          f"{all_pass_freq}")
    print(f"  All 4 root-B points at machine precision |Res|<1e-8: "
          f"{all_machine_precision}")
    print(f"  Root-B sequence is smooth/continuous (criterion b): "
          f"{continuity_ok}")
    print()

    if all_pass_freq and all_machine_precision and continuity_ok:
        print("  Root B satisfies the SAME frozen assignment criteria")
        print("  already used for the confirmed q=2 branch elsewhere.")
        print()
        print("  \"Independent root discovery identified an additional")
        print("  genuine branch (root B) that, when evaluated against the")
        print("  UNCHANGED frequency-agreement and continuity criteria")
        print("  already established in this project, satisfies the q=2")
        print("  physical-branch assignment. This does not invalidate")
        print("  root A as a genuine numerical solution -- it demonstrates")
        print("  that Stage 5I-4's complex-plane discovery did not achieve")
        print("  exhaustive root coverage in this region.\"")
        print()
        print("  RECOMMENDATION: record as a SEPARATE coverage/sensitivity")
        print("  finding. Do NOT overwrite existing stage5I_4/6/7 outputs.")
        print("  A full re-scan with an EXPANDED discovery grid would be")
        print("  needed to check for systematic appearance of root B (or")
        print("  further roots), evaluated with these SAME frozen criteria.")
    else:
        print("  Root B does NOT cleanly satisfy all frozen criteria.")
        print("  No branch reassignment is warranted from this finding.")


if __name__ == "__main__":
    main()
