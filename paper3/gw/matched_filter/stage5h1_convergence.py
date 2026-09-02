"""
stage5h1_convergence.py
===========================

STAGE 5H-1 (standalone): does the null cross-family Λ bias found in
Stage 5G stabilize with more realizations, or was it a finite-sample
artifact (n_realizations=4 in 5G)?

Split into its own script (separate from stage5h2_grid_expansion.py)
to keep each run within the memory-safe pycbc.match() call budget
(~3000 calls) established across Stages 5-5G. Run this first; run
stage5h2_grid_expansion.py in a SEPARATE process afterward (a fresh
`python3` invocation releases any accumulated memory).

METHOD: Lambda_true=0 null test, cross-family combinations A->B and
B->A only (same-family A->A/B->B already confirmed stable and small
in Stage 5F-2), at increasing n_realizations = 4, 10, 20, for two
bands: [20,100) Hz (Stage 5G's worst case: 50-75% boundary saturation)
and [20,400) Hz (Stage 5G's cleanest case: no boundary hits).

INTERPRETATION:
  - If cross-family mean/std STABILIZE to consistent nonzero values as
    N grows -> supports a genuine systematic (not finite-sample noise).
  - If they trend toward zero or remain wildly unstable -> Stage 5G's
    specific numbers were finite-sample artifacts, not established bias.
  - [20,100) Hz boundary-fraction behavior is diagnostic in its own
    right: if it does NOT decrease with N (staying at 50-75%
    regardless of sample size), this points toward Stage 5H-2's
    hypothesis (pathological non-identifiability, not a finite-sample
    issue) rather than simply needing more realizations.
"""

import numpy as np
from pathlib import Path
import sys
import gc
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from waveform import cosmological_K_factor
from likelihood import aligo_like_psd
from stage5_cross_waveform_validation import (
    WAVEFORM_FAMILIES, grid_search_lambda_generic,
)


def run_null_combo(m1, m2, K_z, distance_Mpc, f_lo, f_hi, Lambda_grid,
                    inject_fn, recover_fn, n_realizations, seed_base):
    duration = 8.0
    df = 1.0 / duration
    f = np.arange(f_lo, f_hi, df)
    psd = aligo_like_psd(f)

    estimates = []
    rng = np.random.default_rng(seed_base)
    for real_idx in range(n_realizations):
        sigma = np.sqrt(psd / (4 * df))
        noise = rng.normal(0, sigma) + 1j * rng.normal(0, sigma)
        h_inject = inject_fn(f, m1, m2, 0.0, K_z, distance_Mpc=distance_Mpc)
        data = h_inject + noise

        _, _, Lam_ml, _ = grid_search_lambda_generic(
            data, f, psd, df, m1, m2, K_z, Lambda_grid, recover_fn,
            distance_Mpc=distance_Mpc)
        estimates.append(Lam_ml)
        gc.collect()

    estimates = np.array(estimates)
    grid_lo, grid_hi = Lambda_grid[0], Lambda_grid[-1]
    boundary_hits = np.sum((estimates <= grid_lo) | (estimates >= grid_hi))

    return dict(mean=np.mean(estimates), median=np.median(estimates),
                std=np.std(estimates), n=n_realizations,
                boundary_fraction=boundary_hits / n_realizations)


def main():
    print("#" * 72)
    print("# STAGE 5H-1 — CONVERGENCE TEST")
    print("#" * 72)
    print()

    m1, m2 = 35.6, 30.6
    z = 0.09
    distance_Mpc = 440.0
    K_z = cosmological_K_factor(z)

    family_names = list(WAVEFORM_FAMILIES.keys())
    fn_A, fn_B = WAVEFORM_FAMILIES[family_names[0]], WAVEFORM_FAMILIES[family_names[1]]

    Lambda_grid = np.linspace(-5.0, 5.0, 25)
    N_values = [4, 10, 20]
    bands = [(20.0, 100.0), (20.0, 400.0)]

    total_calls = 2 * 2 * sum(N_values) * len(Lambda_grid)
    print(f"  Planned pycbc.match() calls: {total_calls}")
    print()

    results = {}
    for f_lo, f_hi in bands:
        print(f"  Band [{f_lo:.0f},{f_hi:.0f}) Hz:")
        print(f"  {'N':>5}  {'A->B mean':>10}  {'A->B std':>10}  "
              f"{'B->A mean':>10}  {'B->A std':>10}  {'A->B bound%':>12}  "
              f"{'B->A bound%':>12}")
        print("  " + "-" * 82)

        band_results = []
        for N in N_values:
            r_AB = run_null_combo(m1, m2, K_z, distance_Mpc, f_lo, f_hi,
                                   Lambda_grid, fn_A, fn_B, N,
                                   seed_base=hash((f_lo, f_hi, N, "AB")) % (2**32))
            r_BA = run_null_combo(m1, m2, K_z, distance_Mpc, f_lo, f_hi,
                                   Lambda_grid, fn_B, fn_A, N,
                                   seed_base=hash((f_lo, f_hi, N, "BA")) % (2**32))

            print(f"  {N:>5}  {r_AB['mean']:>10.3f}  {r_AB['std']:>10.3f}  "
                  f"{r_BA['mean']:>10.3f}  {r_BA['std']:>10.3f}  "
                  f"{r_AB['boundary_fraction']*100:>11.1f}%  "
                  f"{r_BA['boundary_fraction']*100:>11.1f}%")

            band_results.append(dict(N=N, AB=r_AB, BA=r_BA))
        results[(f_lo, f_hi)] = band_results
        print()

    # ── Diagnosis ────────────────────────────────────────────────────────
    print("=" * 72)
    print("DIAGNOSIS")
    print("=" * 72)
    print()

    narrow_band = results[(20.0, 100.0)]
    wide_band = results[(20.0, 400.0)]

    narrow_boundary_trend = [r["AB"]["boundary_fraction"] for r in narrow_band]
    print(f"  [20,100)Hz A->B boundary fraction vs N: "
          f"{[f'{b*100:.0f}%' for b in narrow_boundary_trend]}")

    if max(narrow_boundary_trend) - min(narrow_boundary_trend) < 0.15:
        print("  Boundary fraction stays roughly CONSTANT across N -- this is")
        print("  NOT a finite-sample effect. Supports Stage 5H-2's hypothesis")
        print("  that this band has a fundamental non-identifiability issue.")
    else:
        print("  Boundary fraction CHANGES with N -- may partially be a")
        print("  finite-sample effect, requiring the grid-expansion test")
        print("  (5H-2) to fully characterize.")

    print()
    wide_ab_means = [r["AB"]["mean"] for r in wide_band]
    wide_ab_stds = [r["AB"]["std"] for r in wide_band]
    print(f"  [20,400)Hz A->B mean vs N: {[f'{m:.3f}' for m in wide_ab_means]}")
    print(f"  [20,400)Hz A->B std vs N:  {[f'{s:.3f}' for s in wide_ab_stds]}")

    if max(wide_ab_stds) > 0 and (wide_ab_stds[0] / wide_ab_stds[-1] > 1.5
                                    if wide_ab_stds[-1] > 0 else False):
        print("  Std DECREASES with N (as expected for statistical convergence).")
        print("  The mean bias value should be trusted more at higher N.")
    print()

    # ── Plot ──────────────────────────────────────────────────────────────
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    for (f_lo, f_hi), band_results in results.items():
        Ns = [r["N"] for r in band_results]
        ab_means = [r["AB"]["mean"] for r in band_results]
        ab_stds = [r["AB"]["std"] for r in band_results]
        label = f"[{f_lo:.0f},{f_hi:.0f})Hz A->B"
        ax.errorbar(Ns, ab_means, yerr=ab_stds, fmt="o-", capsize=4, label=label)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("N realizations")
    ax.set_ylabel(r"$\Lambda_{\rm ML}$ (true=0)")
    ax.set_title("Stage 5H-1: Convergence with increasing N")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out / "stage5h1_convergence.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
