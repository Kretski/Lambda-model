"""
stage5g_null_frequency_conditioning.py
==========================================

STAGE 5G: null cross-family test across a finer grid of frequency bands,
with more realizations, to determine whether the extreme values found
at [20,100) Hz in Stage 5F-1 (A->B=-2.10, B->A=+1.90, at Lambda_true=0.1)
represent a genuine systematic effect or estimator instability in a
narrow, low-SNR-content frequency band.

WHY THIS TEST, NOT A PHYSICAL EXPLANATION YET:

Stage 5F-1 found bias DECREASING with increasing upper frequency cutoff
-- the opposite of the "merger/ringdown is the problem" hypothesis. This
is either:
  (a) A genuine, unexpected physical finding (low-frequency inspiral
      phase differences between IMRPhenomD/SEOBNRv4 dominate the bias),
  (b) An artifact of restricting the analysis to a narrow band with
      correspondingly little frequency-domain information, making the
      Lambda grid search poorly constrained (wide, unstable posteriors)
      regardless of any real physics, or
  (c) A specific numerical/conditioning issue at the [20,100) Hz band
      edge (e.g. interacting with the RuntimeError seen during the
      Stage 5F run, which did not halt execution but was not localized).

This script tests specifically for (b): using Lambda_true=0 (null test)
across FIVE frequency bands with MANY MORE realizations per band/family
combination, and reporting full distributional diagnostics (mean,
median, std, 90% interval, boundary fraction) rather than a single
point estimate -- exactly the discipline already used in Stage 3/4's
Mode C null trials, now applied to the cross-family frequency-band
question.

If narrow bands show LARGE SCATTER (not just large mean bias) even at
Lambda_true=0, this points to (b): estimator instability from limited
frequency content, not a genuine low-frequency physical effect.
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


def run_null_test_single_band(m1, m2, K_z, distance_Mpc, f_lo, f_hi,
                                Lambda_grid, n_realizations=8):
    """
    Lambda_true=0 null test for all four family combinations (A->A,
    A->B, B->A, B->B) in a single frequency band, with full
    distributional diagnostics (not just mean/scatter).
    """
    duration = 8.0
    df = 1.0 / duration
    f = np.arange(f_lo, f_hi, df)
    psd = aligo_like_psd(f)

    family_names = list(WAVEFORM_FAMILIES.keys())
    fn_A, fn_B = WAVEFORM_FAMILIES[family_names[0]], WAVEFORM_FAMILIES[family_names[1]]
    combos = [("A", "A", fn_A, fn_A), ("A", "B", fn_A, fn_B),
              ("B", "A", fn_B, fn_A), ("B", "B", fn_B, fn_B)]

    results = {}
    for inj_label, rec_label, inject_fn, recover_fn in combos:
        estimates = []
        rng = np.random.default_rng(
            hash((f_lo, f_hi, inj_label, rec_label)) % (2**32))
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

        results[(inj_label, rec_label)] = dict(
            mean=np.mean(estimates), median=np.median(estimates),
            std=np.std(estimates),
            p05=np.percentile(estimates, 5) if len(estimates) > 1 else estimates[0],
            p95=np.percentile(estimates, 95) if len(estimates) > 1 else estimates[0],
            boundary_fraction=boundary_hits / n_realizations,
            n=n_realizations, raw=estimates)

    return results


def main():
    print("#" * 72)
    print("# STAGE 5G — NULL CROSS-FAMILY TEST WITH FREQUENCY CONDITIONING")
    print("#" * 72)
    print()

    m1, m2 = 35.6, 30.6
    z = 0.09
    distance_Mpc = 440.0
    K_z = cosmological_K_factor(z)

    bands = [(20.0, 100.0), (20.0, 200.0), (20.0, 400.0)]
    # Wider grid than 5F to avoid boundary-hit contamination of the
    # diagnostic itself (Stage 3/4's Mode C lesson applied here)
    Lambda_grid = np.linspace(-5.0, 5.0, 61)
    n_realizations = 4

    print(f"  Lambda_true = 0.0 (null test) across {len(bands)} bands")
    print(f"  Lambda search grid: [{Lambda_grid[0]}, {Lambda_grid[-1]}], "
          f"n_realizations={n_realizations}")
    print()

    all_band_results = {}

    for f_lo, f_hi in bands:
        print(f"  --- Band [{f_lo:.0f}, {f_hi:.0f}) Hz ---")
        results = run_null_test_single_band(
            m1, m2, K_z, distance_Mpc, f_lo, f_hi, Lambda_grid, n_realizations)
        all_band_results[(f_lo, f_hi)] = results

        print(f"  {'Combo':>8}  {'mean':>8}  {'median':>8}  {'std':>8}  "
              f"{'90% interval':>20}  {'boundary%':>10}")
        print("  " + "-" * 72)
        for (inj, rec), r in results.items():
            print(f"  {inj}->{rec:>5}  {r['mean']:>8.3f}  {r['median']:>8.3f}  "
                  f"{r['std']:>8.3f}  [{r['p05']:>7.3f},{r['p95']:>7.3f}]  "
                  f"{r['boundary_fraction']*100:>9.1f}%")
        print()

    # ── Diagnosis ────────────────────────────────────────────────────────
    print("=" * 72)
    print("DIAGNOSIS")
    print("=" * 72)
    print()

    print(f"  {'Band':>16}  {'A->A std':>10}  {'A->B std':>10}  "
          f"{'B->A std':>10}  {'B->B std':>10}")
    print("  " + "-" * 62)
    for (f_lo, f_hi), results in all_band_results.items():
        std_AA = results[("A", "A")]["std"]
        std_AB = results[("A", "B")]["std"]
        std_BA = results[("B", "A")]["std"]
        std_BB = results[("B", "B")]["std"]
        print(f"  [{f_lo:.0f},{f_hi:.0f}) Hz     {std_AA:>10.3f}  {std_AB:>10.3f}  "
              f"{std_BA:>10.3f}  {std_BB:>10.3f}")

    print()

    # Check: does scatter shrink monotonically as band widens?
    narrow_std = np.mean([all_band_results[bands[0]][("A", "B")]["std"],
                          all_band_results[bands[0]][("B", "A")]["std"]])
    wide_std = np.mean([all_band_results[bands[-1]][("A", "B")]["std"],
                        all_band_results[bands[-1]][("B", "A")]["std"]])

    print(f"  Cross-family scatter at narrowest band [{bands[0][0]:.0f},"
          f"{bands[0][1]:.0f}): {narrow_std:.3f}")
    print(f"  Cross-family scatter at widest band [{bands[-1][0]:.0f},"
          f"{bands[-1][1]:.0f}):    {wide_std:.3f}")
    print()

    if narrow_std > 3 * wide_std:
        print("  ┌────────────────────────────────────────────────────────┐")
        print("  │  FINDING: cross-family SCATTER (not just mean bias) is  │")
        print("  │  much larger in narrow bands. This supports hypothesis  │")
        print("  │  (b): the Lambda grid search is POORLY CONSTRAINED with │")
        print("  │  limited frequency-domain information, producing wide,  │")
        print("  │  unstable estimates -- not necessarily a genuine low-   │")
        print("  │  frequency PHYSICAL effect. Stage 5F-1's extreme values  │")
        print("  │  (-2.1, +1.9) at [20,100)Hz are likely ESTIMATOR         │")
        print("  │  INSTABILITY from limited band content, not evidence    │")
        print("  │  that low-frequency inspiral is where the 'true' bias   │")
        print("  │  originates.                                            │")
        print("  └────────────────────────────────────────────────────────┘")
    else:
        print("  ┌────────────────────────────────────────────────────────┐")
        print("  │  FINDING: cross-family scatter does NOT shrink sharply   │")
        print("  │  with wider bands. Stage 5F-1's narrow-band bias may be  │")
        print("  │  a genuine effect requiring physical explanation, not    │")
        print("  │  simply an estimator-stability artifact.                 │")
        print("  └────────────────────────────────────────────────────────┘")

    same_family_std = np.mean([
        np.mean([all_band_results[b][("A", "A")]["std"],
                 all_band_results[b][("B", "B")]["std"]])
        for b in bands
    ])
    cross_family_std = np.mean([
        np.mean([all_band_results[b][("A", "B")]["std"],
                 all_band_results[b][("B", "A")]["std"]])
        for b in bands
    ])
    print()
    print(f"  Overall same-family scatter (all bands): {same_family_std:.3f}")
    print(f"  Overall cross-family scatter (all bands): {cross_family_std:.3f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    f_highs = [b[1] for b in bands]
    ax = axes[0]
    for combo, col, label in [(("A", "A"), "steelblue", "A->A"),
                               (("A", "B"), "firebrick", "A->B"),
                               (("B", "A"), "forestgreen", "B->A"),
                               (("B", "B"), "purple", "B->B")]:
        means = [all_band_results[b][combo]["mean"] for b in bands]
        stds = [all_band_results[b][combo]["std"] for b in bands]
        ax.errorbar(f_highs, means, yerr=stds, fmt="o-", color=col,
                   label=label, capsize=4)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Upper frequency cutoff [Hz]")
    ax.set_ylabel(r"Recovered $\Lambda_{\rm ML}$ (true=0)")
    ax.set_title("5G: Null test mean +/- scatter vs band")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for combo, col, label in [(("A", "A"), "steelblue", "A->A"),
                               (("A", "B"), "firebrick", "A->B"),
                               (("B", "A"), "forestgreen", "B->A"),
                               (("B", "B"), "purple", "B->B")]:
        stds = [all_band_results[b][combo]["std"] for b in bands]
        ax.plot(f_highs, stds, "o-", color=col, label=label)
    ax.set_xlabel("Upper frequency cutoff [Hz]")
    ax.set_ylabel(r"Scatter (std) of $\Lambda_{\rm ML}$")
    ax.set_title("5G: Estimator scatter vs band width")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Stage 5G: Null cross-family test with frequency conditioning",
                 fontsize=11)
    plt.tight_layout()
    path = out / "stage5g_null_frequency_conditioning.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
