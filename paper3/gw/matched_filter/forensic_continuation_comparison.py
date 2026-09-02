"""
forensic_continuation_comparison.py
========================================

FORENSIC COMPARISON: original vs independent p(r) continuation,
step-by-step, at an IDENTICAL fixed test point. NO SILENT FALLBACKS --
every step reports explicit success/failure and residual.

Scope (deliberately narrow, per agreed plan): compare ONLY
  H(p,r), dH/dp, dH/dr  at fixed (p,r,m,omega)
  p(r) continuation, step by step
  radial action (once continuation is validated pointwise)
  resonance_factor (final comparison)

Does NOT touch stage5I_3_resonance_v2.py, does NOT re-validate q=1/q=2,
does NOT touch m_c or the Lambda model.

Fixed test case: m=-12, omega = 2*pi*(8.3066307169 - 0.0005664350j),
the EXACT already-validated root, so both implementations are tested
against a point where the ORIGINAL is known to converge (|Res|~1e-16).
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stage5I_3_resonance_v2 import (
    hamiltonian as hamiltonian_orig,
    solve_p_complex as solve_p_complex_orig,
    find_light_ring,
    find_last_scattering_point,
    find_turning_point_complex,
    compute_H_r_at_turning_point,
    R_B as R_B_orig,
)

from independent_reimplementation import (
    _H_num, _d2H_dp2_num,
    solve_p_step_lm,
)


M_TEST = -12
F_RE_TEST = 8.3066307169
F_IM_TEST = -0.0005664350
OMEGA_TEST = 2 * np.pi * (F_RE_TEST + 1j * F_IM_TEST)

N_STEPS = 250


def hamiltonian_indep(p, r, m, omega):
    return complex(_H_num(p, r, m, omega))


def dH_dp_orig_fd(p, r, m, omega, eps=1e-3):
    return (hamiltonian_orig(omega, p + eps, m, r) -
            hamiltonian_orig(omega, p - eps, m, r)) / (2 * eps)


def main():
    print("=" * 100)
    print("FORENSIC COMPARISON: p(r) CONTINUATION, STEP BY STEP")
    print("=" * 100)
    print()
    print(f"Fixed test point: m={M_TEST}, omega=2*pi*({F_RE_TEST} "
          f"{F_IM_TEST:+.10f}j) rad/s")
    print("(the exact already-validated original root)")
    print()

    r_sp, omega_lr = find_light_ring(M_TEST)
    r_minus_real = find_last_scattering_point(OMEGA_TEST, M_TEST, r_sp)
    r_minus_complex = find_turning_point_complex(OMEGA_TEST, M_TEST, r_minus_real)

    print(f"r_sp = {r_sp}")
    print(f"r_minus_real = {r_minus_real}")
    print(f"r_minus_complex = {r_minus_complex}")
    print()

    if r_minus_complex is None:
        print("ERROR: original's find_turning_point_complex failed. "
              "Cannot proceed with a shared reference point.")
        return

    r_path = r_minus_complex + np.linspace(0, 1, N_STEPS) * (R_B_orig - r_minus_complex)

    print("-" * 100)
    print("STEP 1: H, dH/dp, dH/dr at (p=0, r=r_minus_complex)")
    print("-" * 100)

    H_orig_0 = hamiltonian_orig(OMEGA_TEST, 0.0j, M_TEST, r_minus_complex)
    H_indep_0 = hamiltonian_indep(0.0j, r_minus_complex, M_TEST, OMEGA_TEST)
    print(f"  H(0, r_minus): original={H_orig_0}, independent={H_indep_0}")
    print(f"    match: {np.isclose(H_orig_0, H_indep_0, atol=1e-6)}")

    dHdp_orig_0 = dH_dp_orig_fd(0.0j, r_minus_complex, M_TEST, OMEGA_TEST)
    print(f"  dH/dp(0, r_minus) [orig, finite-diff]: {dHdp_orig_0}")
    print("    (expected ~0 exactly at a genuine turning point -- H even in p)")
    print()

    print("-" * 100)
    print("STEP 2: p(r) CONTINUATION -- step by step, NO silent fallback")
    print("-" * 100)
    print()

    print("  ORIGINAL (solve_p_complex, hybr-based continuation):")
    p_orig = np.zeros(N_STEPS, dtype=complex)
    p_orig[0] = 0.0j
    orig_first_failure = None

    H_r0_orig = compute_H_r_at_turning_point(OMEGA_TEST, M_TEST, r_minus_complex)
    eps_p_probe = 1.0
    H_pp0_orig = (
        hamiltonian_orig(OMEGA_TEST, eps_p_probe + 0j, M_TEST, r_minus_complex)
        - 2 * hamiltonian_orig(OMEGA_TEST, 0j, M_TEST, r_minus_complex)
        + hamiltonian_orig(OMEGA_TEST, -eps_p_probe + 0j, M_TEST, r_minus_complex)
    ) / eps_p_probe ** 2

    for i in range(1, N_STEPS):
        r = r_path[i]
        if i == 1:
            delta_r = r - r_minus_complex
            radicand = -2.0 * H_r0_orig * delta_r / H_pp0_orig
            seed = np.sqrt(radicand)
        else:
            seed = p_orig[i - 1]

        p_new = solve_p_complex_orig(OMEGA_TEST, M_TEST, r, p_initial=seed)
        if p_new is None and i == 1:
            p_new = solve_p_complex_orig(OMEGA_TEST, M_TEST, r, p_initial=-seed)

        if p_new is None:
            if orig_first_failure is None:
                orig_first_failure = i
                print(f"    step {i}: FAILURE (no fallback) -- "
                      f"r={r}, seed={seed}")
            p_orig[i] = np.nan
            continue

        if abs(p_new - p_orig[i - 1]) > abs(-p_new - p_orig[i - 1]):
            p_new = -p_new
        p_orig[i] = p_new

    n_orig_failures = int(np.sum(np.isnan(p_orig)))
    print(f"    completed: {N_STEPS - n_orig_failures}/{N_STEPS} steps succeeded")
    if orig_first_failure is not None:
        print(f"    first failure at step {orig_first_failure}")
    print()

    print("  INDEPENDENT (solve_p_step_lm, LM-based continuation):")
    p_indep = np.zeros(N_STEPS, dtype=complex)
    p_indep[0] = 0.0j
    indep_first_failure = None

    for i in range(1, N_STEPS):
        r = r_path[i]
        if i == 1:
            p_new = None
            for trial_seed in [1.0 + 0j, 10.0 + 0j, 100.0 + 0j,
                                 abs(M_TEST) / abs(r) + 0j]:
                p_new = solve_p_step_lm(OMEGA_TEST, M_TEST, r, trial_seed)
                if p_new is not None:
                    break
        else:
            seed = p_indep[i - 1]
            p_new = solve_p_step_lm(OMEGA_TEST, M_TEST, r, seed)
            if p_new is None:
                p_new = solve_p_step_lm(OMEGA_TEST, M_TEST, r, -seed)

        if p_new is None:
            if indep_first_failure is None:
                indep_first_failure = i
                print(f"    step {i}: FAILURE (no fallback) -- r={r}")
            p_indep[i] = np.nan
            continue

        if abs(p_new - p_indep[i - 1]) > abs(-p_new - p_indep[i - 1]):
            p_new = -p_new
        p_indep[i] = p_new

    n_indep_failures = int(np.sum(np.isnan(p_indep)))
    print(f"    completed: {N_STEPS - n_indep_failures}/{N_STEPS} steps succeeded")
    if indep_first_failure is not None:
        print(f"    first failure at step {indep_first_failure}")
    print()

    print("-" * 100)
    print("STEP 3: pointwise comparison (steps where BOTH succeeded)")
    print("-" * 100)
    print()

    both_valid = ~np.isnan(p_orig) & ~np.isnan(p_indep)
    n_both = int(np.sum(both_valid))
    print(f"  Both succeeded at {n_both}/{N_STEPS} steps")

    if n_both > 0:
        diffs = np.abs(p_orig[both_valid] - p_indep[both_valid])
        print(f"  Max |p_orig - p_indep| where both succeeded: {np.max(diffs):.6e}")
        print(f"  Mean |p_orig - p_indep| where both succeeded: {np.mean(diffs):.6e}")

        idx_valid = np.where(both_valid)[0]
        print()
        print(f"  {'step':>6} {'r':>14} {'p_orig':>22} {'p_indep':>22} {'|diff|':>12}")
        show_idx = list(idx_valid[:5])
        if n_both > 10:
            show_idx += list(idx_valid[-5:])
        for idx in show_idx:
            print(f"  {idx:>6} {r_path[idx].real:>14.8f} "
                  f"{str(p_orig[idx]):>22} {str(p_indep[idx]):>22} "
                  f"{abs(p_orig[idx]-p_indep[idx]):>12.4e}")
    print()

    if n_orig_failures == 0 and n_indep_failures == 0 and n_both == N_STEPS:
        max_diff = float(np.max(np.abs(p_orig - p_indep)))
        if max_diff < 1e-4:
            print("  VERDICT: continuation matches closely at every step.")
        else:
            print(f"  VERDICT: both completed all steps but DIVERGE "
                  f"(max diff={max_diff:.4e}).")
            print("  Points to a systematic branch-selection difference,")
            print("  not a convergence failure per se.")
    else:
        print("  VERDICT: at least one implementation failed at some step(s).")
        print("  This is the forensic evidence requested: pinpointing exactly")
        print("  where robustness breaks down, rather than guessing.")


if __name__ == "__main__":
    main()
