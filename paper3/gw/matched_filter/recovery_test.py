"""
recovery_test.py
====================

THE CRITICAL PROOF-OF-CONCEPT TEST, before touching any real GWOSC data
or any lalsimulation-based full waveform.

Question: is Lambda identifiable via matched filtering at all, when the
dispersion enters directly as an analytic PHASE term in the frequency-
domain waveform (no intermediate time-domain frequency extraction)?

This decouples the Lambda-identifiability question from every problem
diagnosed in the legacy time-domain scripts (Hilbert bias, zero-crossing
instability, envelope gating, median filtering, PN-order mismatch) by
construction: injection and recovery use the IDENTICAL leading-order
TaylorF2 + Lambda-phase model, exactly matched, in a standard matched-
filter framework.

If this test does not pass, the phase-based approach itself has a
problem (not extraction methodology), and must be fixed before any
further investment -- including before installing lalsimulation for a
production-grade waveform.

If this test PASSES, we have concrete evidence that the phase-based
matched-filter parameterization of Lambda is sound, and the next
legitimate step is Stage 2: inject into REAL GWOSC detector noise
(not synthetic aLIGO-like colored noise), then Stage 3: analyze real
GW150914 strain with this corrected pipeline.
"""

import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from waveform import cosmological_K_factor
from likelihood import aligo_like_psd, grid_search_lambda, snr_optimal
from synthetic_injection import make_frequency_grid, generate_injection


def run_recovery_test():
    print("=" * 72)
    print("MATCHED-FILTER LAMBDA RECOVERY TEST (synthetic, phase-domain)")
    print("=" * 72)
    print()

    # ── Setup: GW150914-like source parameters ─────────────────────────
    m1, m2 = 35.6, 30.6
    z = 0.09
    distance_Mpc = 440.0
    K_z = cosmological_K_factor(z)

    f, df = make_frequency_grid(f_min=20.0, f_max=400.0, duration=8.0)
    psd = aligo_like_psd(f)

    print(f"  Source: m1={m1} Msun, m2={m2} Msun, z={z}, D={distance_Mpc} Mpc")
    print(f"  K(z) = {K_z:.4e} s")
    print(f"  Frequency grid: {f[0]:.1f}-{f[-1]:.1f} Hz, df={df:.4f} Hz, "
          f"{len(f)} bins")
    print()

    # Report the optimal SNR at Lambda=0 for context (sanity check that
    # the waveform amplitude scaling is reasonable)
    snr0 = snr_optimal(f, psd, df, m1, m2, 0.0, K_z, distance_Mpc=distance_Mpc)
    print(f"  Optimal SNR at Lambda=0 (no noise realization): {snr0:.2f}")
    print()

    # ── Injection/recovery grid ──────────────────────────────────────────
    Lambda_true_values = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
    Lambda_grid = np.linspace(-2.0, 2.0, 401)  # search grid for ML estimate

    n_realizations = 5  # multiple noise realizations per Lambda_true, for
                         # an honest scatter-based error estimate in
                         # addition to the Fisher-curvature estimate

    print(f"  Testing Lambda_true = {Lambda_true_values}")
    print(f"  {n_realizations} independent noise realizations per value")
    print()

    results = []

    print(f"  {'Lambda_true':>12}  {'Lambda_ML (mean)':>18}  "
          f"{'scatter (std)':>14}  {'Fisher_err (mean)':>18}  "
          f"{'sigma_from_true':>16}")
    print("  " + "-" * 84)

    for Lam_true in Lambda_true_values:
        ml_estimates = []
        fisher_errs = []

        for real_idx in range(n_realizations):
            data = generate_injection(m1, m2, Lam_true, K_z, f, df, psd,
                                       distance_Mpc=distance_Mpc,
                                       seed=1000 + real_idx, add_noise=True)
            Lgrid, logL, Lam_ml, Lam_err = grid_search_lambda(
                data, f, psd, df, m1, m2, K_z, Lambda_grid,
                distance_Mpc=distance_Mpc)
            ml_estimates.append(Lam_ml)
            if np.isfinite(Lam_err):
                fisher_errs.append(Lam_err)

        ml_estimates = np.array(ml_estimates)
        mean_ml = np.mean(ml_estimates)
        scatter = np.std(ml_estimates)
        mean_fisher_err = np.mean(fisher_errs) if fisher_errs else np.nan

        sigma_from_true = abs(mean_ml - Lam_true) / scatter if scatter > 0 else np.nan

        results.append(dict(Lambda_true=Lam_true, mean_ml=mean_ml,
                             scatter=scatter, fisher_err=mean_fisher_err,
                             sigma=sigma_from_true, all_estimates=ml_estimates))

        print(f"  {Lam_true:>12.3f}  {mean_ml:>18.4f}  {scatter:>14.4f}  "
              f"{mean_fisher_err:>18.4f}  {sigma_from_true:>16.2f}")

    print()

    # ── Pass/fail ──────────────────────────────────────────────────────
    print("=" * 72)
    print("VALIDATION")
    print("=" * 72)
    print()

    all_pass = True
    for r in results:
        # Criterion: recovered mean within 2x scatter of true value
        # (a reasonable Gaussian-consistency check across realizations)
        within_tolerance = abs(r["mean_ml"] - r["Lambda_true"]) < 2 * r["scatter"] \
            if r["scatter"] > 0 else (abs(r["mean_ml"] - r["Lambda_true"]) < 0.05)
        status = "PASS" if within_tolerance else "FAIL"
        if not within_tolerance:
            all_pass = False
        print(f"  Lambda_true={r['Lambda_true']:.3f}: recovered "
              f"{r['mean_ml']:.4f}+/-{r['scatter']:.4f}  [{status}]")

    print()
    if all_pass:
        print("  ┌────────────────────────────────────────────────────────┐")
        print("  │  PASS: matched-filter phase-domain Lambda recovery is   │")
        print("  │  consistent across the tested range.                   │")
        print("  │                                                        │")
        print("  │  This DECOUPLES the identifiability question from all  │")
        print("  │  time-domain extraction bias found in the legacy       │")
        print("  │  scripts (Hilbert, zero-crossing, STFT ridge).         │")
        print("  │                                                        │")
        print("  │  NEXT STEP: inject into REAL GWOSC detector noise      │")
        print("  │  (not synthetic aLIGO-shaped noise), then analyze      │")
        print("  │  real GW150914 strain with this pipeline.              │")
        print("  └────────────────────────────────────────────────────────┘")
    else:
        print("  ┌────────────────────────────────────────────────────────┐")
        print("  │  FAIL: even the phase-domain matched-filter approach   │")
        print("  │  does not cleanly recover injected Lambda.             │")
        print("  │                                                        │")
        print("  │  Do NOT proceed to real GWOSC noise or GW150914 data   │")
        print("  │  until this is resolved. Likely causes: waveform       │")
        print("  │  normalization, grid resolution, or a sign/units error │")
        print("  │  in the Lambda phase term -- check waveform.py before  │")
        print("  │  any further pipeline development.                     │")
        print("  └────────────────────────────────────────────────────────┘")

    # ── Plot ──────────────────────────────────────────────────────────────
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    Lt = [r["Lambda_true"] for r in results]
    Lml = [r["mean_ml"] for r in results]
    Lsc = [r["scatter"] for r in results]
    ax.errorbar(Lt, Lml, yerr=Lsc, fmt="o-", color="steelblue",
                capsize=4, markersize=8, label="recovered (mean +/- scatter)")
    lims = [min(Lt) - 0.2, max(Lt) + 0.2]
    ax.plot(lims, lims, "k--", lw=1.5, label="ideal recovery")
    ax.set_xlabel(r"Injected $\Lambda_{\rm true}$")
    ax.set_ylabel(r"Recovered $\Lambda_{\rm ML}$")
    ax.set_title("Matched-filter Lambda recovery\n(phase-domain, "
                 f"{n_realizations} realizations)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    example = results[2]  # Lambda_true=0.05 case, illustrative
    data_example = generate_injection(
        m1, m2, example["Lambda_true"], K_z, f, df, psd,
        distance_Mpc=distance_Mpc, seed=1000, add_noise=True)
    Lgrid_ex, logL_ex, _, _ = grid_search_lambda(
        data_example, f, psd, df, m1, m2, K_z, Lambda_grid,
        distance_Mpc=distance_Mpc)
    ax.plot(Lgrid_ex, logL_ex - np.max(logL_ex), color="firebrick", lw=2)
    ax.axvline(example["Lambda_true"], color="k", ls="--", lw=1.5,
              label=fr"true $\Lambda$={example['Lambda_true']}")
    ax.set_xlabel(r"$\Lambda$")
    ax.set_ylabel(r"$\ln L(\Lambda) - \ln L_{\rm max}$")
    ax.set_title(fr"Example likelihood curve ($\Lambda_{{\rm true}}$="
                f"{example['Lambda_true']})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 1)

    plt.suptitle("Matched-filter Lambda recovery test (synthetic, phase-domain)",
                 fontsize=11)
    plt.tight_layout()
    path = out / "matched_filter_recovery.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {path}")

    return results, all_pass


if __name__ == "__main__":
    run_recovery_test()
