"""
stage5h3_fine_grid_convergence.py
=====================================

STAGE 5H-3: does the cross-family Lambda_ML estimate converge to a
STABLE, PRECISE value once boundary saturation is resolved (5H-2), and
is the [20,400)Hz "0.417" value a genuine likelihood maximum or a grid-
discretization artifact?

METHOD: COARSE-TO-FINE grid search (standard practice), NOT a brute-
force fine grid across the full range. A direct fine grid (ΔΛ=0.025)
spanning +/-40 would require ~700,000 pycbc.match() calls per band/
combo/realization -- far beyond the memory-safe budget established in
prior stages. Instead:

  STAGE A (coarse): scan the WIDE range [-40,+40] at coarse resolution
    (~81 points, ΔΛ~1.0) to locate the approximate global maximum and
    confirm it is comfortably INTERIOR to the range (not pinned at the
    +/-40 boundary -- if it were, this would itself indicate the
    degeneracy extends even further, requiring yet another widening).

  STAGE B (fine): given the coarse maximum location, scan a NARROW
    window (+/-2 around the coarse max) at fine resolution (ΔΛ=0.05)
    to precisely locate the maximum and compute the local likelihood
    curvature (Fisher-matrix-style error estimate), directly answering
    whether "0.417" (or whatever value emerges) is a genuine, well-
    defined maximum (narrow curvature, small Fisher error) or an
    artifact of coarse grid discretization (the fine search would
    reveal a much flatter, less localized peak).

Also computes and reports delta-log-likelihood around the maximum, so
a reader can independently judge whether the peak is well-constrained.

Run in a SEPARATE process from stage5h1/5h2 (fresh `python3` call) to
avoid any accumulated memory from prior stages.
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
    WAVEFORM_FAMILIES, grid_search_lambda_generic, log_likelihood_generic,
)


def coarse_to_fine_search(m1, m2, K_z, distance_Mpc, f_lo, f_hi,
                            inject_fn, recover_fn, n_realizations, seed_base,
                            coarse_width=40.0, coarse_n=81,
                            fine_half_width=2.0, fine_resolution=0.05):
    """
    Two-stage search for a single band/combo, across n_realizations.
    Returns per-realization (coarse_max, fine_max, fine_err, delta_logL_curve).
    """
    duration = 8.0
    df = 1.0 / duration
    f = np.arange(f_lo, f_hi, df)
    psd = aligo_like_psd(f)

    coarse_grid = np.linspace(-coarse_width, coarse_width, coarse_n)

    results = []
    rng = np.random.default_rng(seed_base)

    for real_idx in range(n_realizations):
        sigma = np.sqrt(psd / (4 * df))
        noise = rng.normal(0, sigma) + 1j * rng.normal(0, sigma)
        h_inject = inject_fn(f, m1, m2, 0.0, K_z, distance_Mpc=distance_Mpc)
        data = h_inject + noise

        # Stage A: coarse
        _, logL_coarse, Lam_coarse, _ = grid_search_lambda_generic(
            data, f, psd, df, m1, m2, K_z, coarse_grid, recover_fn,
            distance_Mpc=distance_Mpc)
        gc.collect()

        coarse_at_boundary = (abs(Lam_coarse - coarse_grid[0]) < 1e-6 or
                               abs(Lam_coarse - coarse_grid[-1]) < 1e-6)

        # Stage B: fine, centered on coarse max
        fine_n = int(2 * fine_half_width / fine_resolution) + 1
        fine_grid = np.linspace(Lam_coarse - fine_half_width,
                                  Lam_coarse + fine_half_width, fine_n)

        _, logL_fine, Lam_fine, Lam_err = grid_search_lambda_generic(
            data, f, psd, df, m1, m2, K_z, fine_grid, recover_fn,
            distance_Mpc=distance_Mpc)
        gc.collect()

        # Delta-logL diagnostic: how much does logL drop moving away
        # from the fine maximum by the full fine_half_width?
        delta_logL_edge = np.max(logL_fine) - logL_fine[0]

        results.append(dict(
            coarse_max=Lam_coarse, coarse_at_boundary=coarse_at_boundary,
            fine_max=Lam_fine, fine_err=Lam_err,
            delta_logL_edge=delta_logL_edge))

    return results


def summarize_band_combo(m1, m2, K_z, distance_Mpc, f_lo, f_hi,
                          inject_fn, recover_fn, combo_label,
                          n_realizations, seed_base):
    results = coarse_to_fine_search(
        m1, m2, K_z, distance_Mpc, f_lo, f_hi, inject_fn, recover_fn,
        n_realizations, seed_base)

    fine_maxes = np.array([r["fine_max"] for r in results])
    fine_errs = np.array([r["fine_err"] for r in results
                           if np.isfinite(r["fine_err"])])
    n_coarse_boundary = sum(r["coarse_at_boundary"] for r in results)
    mean_delta_logL = np.mean([r["delta_logL_edge"] for r in results])

    print(f"  [{f_lo:.0f},{f_hi:.0f})Hz {combo_label}:")
    print(f"    Coarse max at +/-40 boundary: "
          f"{n_coarse_boundary}/{n_realizations} realizations")
    print(f"    Fine max: mean={np.mean(fine_maxes):.4f}, "
          f"std={np.std(fine_maxes):.4f}")
    if len(fine_errs) > 0:
        print(f"    Fisher-curvature error (mean): {np.mean(fine_errs):.4f}")
    print(f"    Mean delta-logL from peak to fine-window edge "
          f"(+/-2): {mean_delta_logL:.2f}")
    print(f"    (Large delta-logL => well-localized peak; small/flat")
    print(f"    delta-logL => the '{np.mean(fine_maxes):.3f}' value is")
    print(f"    poorly constrained even within this narrow window.)")
    print()

    return dict(mean=np.mean(fine_maxes), std=np.std(fine_maxes),
                fisher_err=np.mean(fine_errs) if len(fine_errs) > 0 else np.nan,
                n_boundary=n_coarse_boundary, mean_delta_logL=mean_delta_logL,
                raw=results)


def main():
    print("#" * 72)
    print("# STAGE 5H-3 — COARSE-TO-FINE GRID CONVERGENCE")
    print("#" * 72)
    print()
    print("  Method: coarse scan [-40,+40] (81 pts) -> fine scan +/-2 around")
    print("  coarse max (ΔΛ=0.05, 81 pts). N=6 realizations per band/combo.")
    print()

    m1, m2 = 35.6, 30.6
    z = 0.09
    distance_Mpc = 440.0
    K_z = cosmological_K_factor(z)

    family_names = list(WAVEFORM_FAMILIES.keys())
    fn_A, fn_B = WAVEFORM_FAMILIES[family_names[0]], WAVEFORM_FAMILIES[family_names[1]]

    n_realizations = 5
    bands = [(20.0, 100.0), (20.0, 400.0)]

    total_calls = len(bands) * 2 * n_realizations * (81 + 81)
    print(f"  Planned pycbc.match() calls: {total_calls}")
    print()

    all_results = {}
    for f_lo, f_hi in bands:
        print(f"  === Band [{f_lo:.0f},{f_hi:.0f})Hz ===")
        r_AB = summarize_band_combo(
            m1, m2, K_z, distance_Mpc, f_lo, f_hi, fn_A, fn_B, "A->B",
            n_realizations, seed_base=hash((f_lo, f_hi, "AB")) % (2**32))
        r_BA = summarize_band_combo(
            m1, m2, K_z, distance_Mpc, f_lo, f_hi, fn_B, fn_A, "B->A",
            n_realizations, seed_base=hash((f_lo, f_hi, "BA")) % (2**32))
        all_results[(f_lo, f_hi)] = dict(AB=r_AB, BA=r_BA)

    # ── Diagnosis ────────────────────────────────────────────────────────
    print("=" * 72)
    print("DIAGNOSIS")
    print("=" * 72)
    print()

    for (f_lo, f_hi), results in all_results.items():
        print(f"  [{f_lo:.0f},{f_hi:.0f})Hz:")
        for label, r in [("A->B", results["AB"]), ("B->A", results["BA"])]:
            well_localized = r["mean_delta_logL"] > 2.0
            print(f"    {label}: mean={r['mean']:.4f} +/- {r['std']:.4f}  "
                  f"boundary_hits={r['n_boundary']}/{n_realizations}  "
                  f"well_localized={'YES' if well_localized else 'NO'}")
        print()

    print("  Overall verdict:")
    any_boundary = any(results["AB"]["n_boundary"] > 0 or
                        results["BA"]["n_boundary"] > 0
                        for results in all_results.values())
    any_flat = any(results["AB"]["mean_delta_logL"] < 2.0 or
                   results["BA"]["mean_delta_logL"] < 2.0
                   for results in all_results.values())

    if any_boundary:
        print("  Some realizations still hit the +/-40 coarse boundary --")
        print("  the degeneracy in at least one band/combo extends beyond")
        print("  even this wide range. Further widening or a fundamentally")
        print("  different (non-grid-search) approach is needed for those.")
    if any_flat:
        print("  Some peaks are POORLY LOCALIZED (small delta-logL) even in")
        print("  the fine window -- the reported mean value should NOT be")
        print("  treated as a precise measurement, only as a rough estimate")
        print("  of where a shallow likelihood plateau sits.")
    if not any_boundary and not any_flat:
        print("  All tested band/combo cases show well-localized, boundary-")
        print("  free maxima. Values can be treated as reasonably precise")
        print("  cross-family systematic estimates for THIS specific")
        print("  waveform-family pair and mass configuration.")

    print()
    print("  [20,400)Hz cross-family offset specifically:")
    r400_AB = all_results[(20.0, 400.0)]["AB"]
    r400_BA = all_results[(20.0, 400.0)]["BA"]
    print(f"    A->B = {r400_AB['mean']:.4f} +/- {r400_AB['std']:.4f} "
          f"(Fisher err: {r400_AB['fisher_err']:.4f})")
    print(f"    B->A = {r400_BA['mean']:.4f} +/- {r400_BA['std']:.4f} "
          f"(Fisher err: {r400_BA['fisher_err']:.4f})")
    if r400_AB["mean_delta_logL"] > 2.0 and r400_BA["mean_delta_logL"] > 2.0:
        print("    Both peaks are well-localized -- if this were used as a")
        print("    systematic-uncertainty floor, it should be treated as a")
        print("    NUISANCE PARAMETER in any future H1+L1 analysis, not")
        print("    corrected away or ignored.")

    # ── Plot ──────────────────────────────────────────────────────────────
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    labels, means, stds = [], [], []
    for (f_lo, f_hi), results in all_results.items():
        for combo_label, r in [("A->B", results["AB"]), ("B->A", results["BA"])]:
            labels.append(f"[{f_lo:.0f},{f_hi:.0f})\n{combo_label}")
            means.append(r["mean"])
            stds.append(r["std"])
    ax.bar(labels, means, yerr=stds, capsize=5,
           color=["firebrick", "forestgreen"] * 2)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel(r"Fine-grid $\Lambda_{\rm ML}$ (true=0)")
    ax.set_title("5H-3: Coarse-to-fine converged estimates")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    delta_logLs = []
    for (f_lo, f_hi), results in all_results.items():
        for combo_label, r in [("A->B", results["AB"]), ("B->A", results["BA"])]:
            delta_logLs.append(r["mean_delta_logL"])
    ax.bar(labels, delta_logLs, color=["steelblue", "purple"] * 2)
    ax.axhline(2.0, color="k", lw=1, ls="--", label="well-localized threshold")
    ax.set_ylabel(r"Mean $\Delta\ln L$ (peak to window edge)")
    ax.set_title("5H-3: Peak localization quality")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Stage 5H-3: Coarse-to-fine grid convergence", fontsize=11)
    plt.tight_layout()
    path = out / "stage5h3_fine_grid_convergence.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
