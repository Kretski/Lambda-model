"""
stage5h4_highN_focused.py
=============================

STAGE 5H-4 (focused, high-N): resolve a specific anomaly found in
Stage 5H-3 for [20,400)Hz B->A: empirical std across N=5 realizations
(0.2538) is ~18x larger than the mean Fisher-curvature error (0.0139)
from individual realizations. This means each individual likelihood
peak is narrow (well-localized), but the PEAK LOCATION itself jumps
substantially between different noise draws -- unlike A->B, where
Fisher error and empirical std are of the same order.

This script does NOT repeat the expensive coarse+fine two-stage search
from 5H-3. Since 5H-3 already established reliable approximate
locations (A->B~-0.17, B->A~-0.11) with NO boundary saturation, we go
straight to a FINE-ONLY search centered on those locations, but with
N=20 realizations (4x more than 5H-3's N=5) to properly resolve
whether:

  (a) B->A's large empirical std reflects genuine multi-modal or
      widely-scattered peak locations across realizations (a real,
      larger-than-A->B statistical uncertainty on the cross-family
      bias), or
  (b) N=5 was simply too few samples and the true std is closer to
      A->B's, with the 5H-3 result an unlucky small-sample fluctuation.

SCOPE: intentionally narrow (only [20,400)Hz, only A->B/B->A, only
Lambda_true=0) to stay within the memory-safe compute budget
(~3000-3500 pycbc.match() calls) while directly answering the specific
open question from 5H-3, rather than attempting the full mass/band/
Lambda_true factorial sweep in one step.
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


def fine_only_search(m1, m2, K_z, distance_Mpc, f_lo, f_hi,
                      inject_fn, recover_fn, center, n_realizations,
                      seed_base, half_width=2.0, resolution=0.05):
    duration = 8.0
    df = 1.0 / duration
    f = np.arange(f_lo, f_hi, df)
    psd = aligo_like_psd(f)

    n_pts = int(2 * half_width / resolution) + 1
    grid = np.linspace(center - half_width, center + half_width, n_pts)

    estimates = []
    fisher_errs = []
    rng = np.random.default_rng(seed_base)

    for real_idx in range(n_realizations):
        sigma = np.sqrt(psd / (4 * df))
        noise = rng.normal(0, sigma) + 1j * rng.normal(0, sigma)
        h_inject = inject_fn(f, m1, m2, 0.0, K_z, distance_Mpc=distance_Mpc)
        data = h_inject + noise

        _, _, Lam_ml, Lam_err = grid_search_lambda_generic(
            data, f, psd, df, m1, m2, K_z, grid, recover_fn,
            distance_Mpc=distance_Mpc)
        estimates.append(Lam_ml)
        if np.isfinite(Lam_err):
            fisher_errs.append(Lam_err)
        gc.collect()

    estimates = np.array(estimates)
    grid_lo, grid_hi = grid[0], grid[-1]
    boundary_hits = np.sum((estimates <= grid_lo) | (estimates >= grid_hi))

    return dict(
        mean=np.mean(estimates), median=np.median(estimates),
        std=np.std(estimates),
        mean_fisher_err=np.mean(fisher_errs) if fisher_errs else np.nan,
        boundary_fraction=boundary_hits / n_realizations,
        n=n_realizations, raw=estimates)


def main():
    print("#" * 72)
    print("# STAGE 5H-4 — HIGH-N FOCUSED TEST ([20,400)Hz, N=20)")
    print("#" * 72)
    print()
    print("  Resolving: Stage 5H-3 found B->A std (0.254) >> mean Fisher")
    print("  error (0.014), ~18x discrepancy, at N=5. Is this genuine")
    print("  wide scatter in peak location across realizations, or a")
    print("  small-sample artifact?")
    print()

    m1, m2 = 35.6, 30.6
    z = 0.09
    distance_Mpc = 440.0
    K_z = cosmological_K_factor(z)
    f_lo, f_hi = 20.0, 400.0

    family_names = list(WAVEFORM_FAMILIES.keys())
    fn_A, fn_B = WAVEFORM_FAMILIES[family_names[0]], WAVEFORM_FAMILIES[family_names[1]]

    n_realizations = 20
    total_calls = 2 * n_realizations * 81
    print(f"  Planned pycbc.match() calls: {total_calls}")
    print()

    r_AB = fine_only_search(
        m1, m2, K_z, distance_Mpc, f_lo, f_hi, fn_A, fn_B,
        center=-0.17, n_realizations=n_realizations,
        seed_base=hash(("5H4", "AB")) % (2**32))
    r_BA = fine_only_search(
        m1, m2, K_z, distance_Mpc, f_lo, f_hi, fn_B, fn_A,
        center=-0.11, n_realizations=n_realizations,
        seed_base=hash(("5H4", "BA")) % (2**32))

    print(f"  {'Combo':>8}  {'mean':>8}  {'median':>8}  {'std':>8}  "
          f"{'mean Fisher':>12}  {'std/Fisher':>10}  {'boundary%':>10}")
    print("  " + "-" * 76)
    for label, r in [("A->B", r_AB), ("B->A", r_BA)]:
        ratio = r["std"] / r["mean_fisher_err"] if r["mean_fisher_err"] > 0 else np.nan
        print(f"  {label:>8}  {r['mean']:>8.4f}  {r['median']:>8.4f}  "
              f"{r['std']:>8.4f}  {r['mean_fisher_err']:>12.4f}  "
              f"{ratio:>10.2f}  {r['boundary_fraction']*100:>9.1f}%")

    print()
    print("  Raw A->B estimates:", np.round(r_AB["raw"], 3))
    print("  Raw B->A estimates:", np.round(r_BA["raw"], 3))
    print()

    # ── Diagnosis ────────────────────────────────────────────────────────
    print("=" * 72)
    print("DIAGNOSIS")
    print("=" * 72)
    print()

    ratio_AB = r_AB["std"] / r_AB["mean_fisher_err"] if r_AB["mean_fisher_err"] > 0 else np.nan
    ratio_BA = r_BA["std"] / r_BA["mean_fisher_err"] if r_BA["mean_fisher_err"] > 0 else np.nan

    print(f"  A->B: std/Fisher ratio = {ratio_AB:.2f}  (5H-3 N=5 comparison "
          f"context: A->B was well-behaved)")
    print(f"  B->A: std/Fisher ratio = {ratio_BA:.2f}  (5H-3 N=5 gave ~18)")
    print()

    if ratio_BA > 5:
        print("  ┌────────────────────────────────────────────────────────┐")
        print("  │  CONFIRMED at N=20: B->A shows genuine realization-to-   │")
        print("  │  realization scatter in peak LOCATION far exceeding the  │")
        print("  │  local (Fisher) curvature uncertainty of any single      │")
        print("  │  peak. This is NOT a small-sample artifact -- it         │")
        print("  │  indicates the B->A cross-family likelihood surface has  │")
        print("  │  additional structure (e.g. secondary local maxima, or   │")
        print("  │  noise-dependent peak wandering) not captured by a       │")
        print("  │  single Fisher-matrix error bar. Any reported B->A bias  │")
        print("  │  must use the EMPIRICAL std, not the Fisher error, as    │")
        print("  │  its uncertainty.                                        │")
        print("  └────────────────────────────────────────────────────────┘")
    else:
        print("  ┌────────────────────────────────────────────────────────┐")
        print("  │  RESOLVED at N=20: the large std/Fisher ratio seen at    │")
        print("  │  N=5 in Stage 5H-3 was a small-sample fluctuation. With  │")
        print("  │  more realizations, B->A's empirical scatter is now      │")
        print("  │  consistent with its local Fisher-curvature precision,   │")
        print("  │  similar to A->B's behavior.                             │")
        print("  └────────────────────────────────────────────────────────┘")

    print()
    print(f"  Updated [20,400)Hz cross-family estimate (N=20):")
    print(f"    A->B = {r_AB['mean']:.4f} +/- {r_AB['std']:.4f}")
    print(f"    B->A = {r_BA['mean']:.4f} +/- {r_BA['std']:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(r_AB["raw"], bins=10, alpha=0.6, color="firebrick", label="A->B")
    ax.hist(r_BA["raw"], bins=10, alpha=0.6, color="forestgreen", label="B->A")
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_xlabel(r"$\Lambda_{\rm ML}$ (true=0)")
    ax.set_ylabel("Count")
    ax.set_title("5H-4: Distribution of peak locations (N=20)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(["A->B", "B->A"], [ratio_AB, ratio_BA], color=["firebrick", "forestgreen"])
    ax.axhline(1.0, color="k", lw=1, ls="--", label="Fisher-consistent (ratio=1)")
    ax.set_ylabel("Empirical std / mean Fisher error")
    ax.set_title("5H-4: Fisher-consistency check")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Stage 5H-4: High-N focused test", fontsize=11)
    plt.tight_layout()
    path = out / "stage5h4_highN_focused.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
