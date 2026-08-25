"""
stage2_real_noise_recovery.py
=================================

STAGE 2: matched-filter Lambda recovery using REAL GWOSC detector noise,
with SYNTHETIC injected waveforms (never the real GW150914 event itself).

RULE (per the agreed plan): the waveform model is UNCHANGED from Stage 1
(waveform.py, likelihood.py, synthetic_injection.py are reused exactly
as-is). Only the noise source changes: from synthetic aLIGO-shaped
Gaussian noise to a PSD estimated from real H1 strain, with the actual
GW150914 signal region excluded so this remains a clean injection/
recovery test.

THREE TEST SERIES:

  A — Zero injection: Lambda_true=0, GR-only waveform + real-noise-
      matched colored Gaussian realizations. Expect Lambda_fit ~ 0.

  B — Nonzero injection: Lambda_true in {0.01, ..., 1.0}, GR+Lambda
      waveform + same noise treatment. Expect Lambda_fit ~ Lambda_true.

  C — Noise-only null test: real-noise-matched colored Gaussian
      realizations with NO injected signal at all. The matched filter
      is still run (correlating against a GR+Lambda template it was
      never given a real signal for). Expect: no coherent, statistically
      significant Lambda recovery -- if the likelihood "finds" a
      confident nonzero Lambda from pure noise, that is a critical
      pipeline problem independent of any injection.

METHODOLOGY NOTE ON REAL-NOISE PSD:
  We estimate a PSD from a segment of REAL H1 strain (Welch's method,
  same estimator as in the legacy diagnostic scripts), using a time
  window that EXCLUDES the GW150914 merger (offset well away from the
  event GPS time), so this segment contains only detector noise. This
  measured PSD is then used to (a) whiten/color synthetic Gaussian
  noise realizations with the SAME spectral shape as the real detector
  (more realistic than the analytic aLIGO-like fit used in Stage 1),
  and (b) as the noise model in the matched-filter inner product.

  We do NOT inject anything into the actual real strain time series in
  this script -- the "real GWOSC noise" contribution is a colored
  Gaussian realization drawn from the REAL measured PSD, which is the
  standard, well-understood way to test pipeline behavior under
  realistic noise statistics without the complications of the real
  strain's non-Gaussian glitches/artifacts (a separate, harder test).
"""

import numpy as np
import h5py
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).parent.parent / "matched_filter"))
from waveform import waveform_frequency_domain, cosmological_K_factor
from likelihood import (noise_weighted_inner_product, log_likelihood,
                         grid_search_lambda, snr_optimal)


# ══════════════════════════════════════════════════════════════════════════
# Real-noise PSD estimation
# ══════════════════════════════════════════════════════════════════════════

def estimate_real_psd(h1_path, gps_center_avoid, avoid_half_window=64.0,
                       segment_duration=32.0, fs_expected=4096.0):
    """
    Load a segment of REAL H1 strain that EXCLUDES the region within
    +/- avoid_half_window seconds of gps_center_avoid (the GW150914
    merger GPS time), and estimate its PSD via Welch's method.

    Returns (freqs, psd) with freqs in Hz, psd in strain^2/Hz.
    """
    with h5py.File(h1_path, "r") as f:
        strain = f["strain"]["Strain"]
        attrs = strain.attrs
        fs = 1.0 / attrs["Xspacing"]
        gps_start = attrs["Xstart"]
        n_total = attrs["Npoints"]

        # Pick a segment safely BEFORE the excluded region (start of file)
        idx_avoid_lo = int(round((gps_center_avoid - avoid_half_window - gps_start) * fs))
        idx_seg_len = int(round(segment_duration * fs))

        idx_hi = max(0, idx_avoid_lo - 10)  # end just before the avoid zone
        idx_lo = max(0, idx_hi - idx_seg_len)

        if idx_hi - idx_lo < idx_seg_len // 2:
            # Not enough room before; use a segment well after instead
            idx_avoid_hi = int(round((gps_center_avoid + avoid_half_window - gps_start) * fs))
            idx_lo = idx_avoid_hi + 10
            idx_hi = min(n_total, idx_lo + idx_seg_len)

        data = np.array(strain[idx_lo:idx_hi])

    print(f"  Real-noise PSD segment: {len(data)} samples "
          f"({len(data)/fs:.1f} s), fs={fs} Hz")
    print(f"  Segment GPS range: [{gps_start + idx_lo/fs:.1f}, "
          f"{gps_start + idx_hi/fs:.1f}]  "
          f"(excludes merger at {gps_center_avoid} +/- {avoid_half_window}s)")

    freqs, psd = welch(data, fs=fs, nperseg=int(4 * fs))
    return freqs, psd, fs


def psd_on_grid(freqs_measured, psd_measured, f_target):
    """Interpolate the measured PSD onto the target frequency grid,
    with a floor to avoid zero/negative values from estimation noise."""
    psd_interp = np.interp(f_target, freqs_measured, psd_measured)
    floor = np.percentile(psd_measured[psd_measured > 0], 1)
    psd_interp = np.maximum(psd_interp, floor)
    return psd_interp


# ══════════════════════════════════════════════════════════════════════════
# Injection using real-noise-shaped colored Gaussian realizations
# ══════════════════════════════════════════════════════════════════════════

def generate_injection_real_noise_shaped(m1, m2, Lambda_true, K_z, f, df,
                                           psd_real, distance_Mpc=440.0,
                                           tc=0.0, phi_c=0.0, seed=0,
                                           add_signal=True):
    """
    Same construction as synthetic_injection.generate_injection, but
    using the REAL-measured PSD (psd_real) instead of the analytic
    aLIGO-like fit. If add_signal=False, returns pure colored noise
    (for the Test C null test).
    """
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(psd_real / (4 * df))
    noise = rng.normal(0, sigma) + 1j * rng.normal(0, sigma)

    if not add_signal:
        return noise

    h = waveform_frequency_domain(f, m1, m2, Lambda_true, K_z, tc, phi_c,
                                   distance_Mpc)
    return h + noise


# ══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("STAGE 2 — MATCHED-FILTER RECOVERY WITH REAL GWOSC NOISE")
    print("=" * 72)
    print()

    h1_path = r"C:\Users\Lenovo\Desktop\z\H-H1_LOSC_4_V1-1126256640-4096.hdf5"
    gps_merger = 1126259462.4

    print("[Step 1] Estimating PSD from REAL H1 strain (excluding merger)...")
    freqs_measured, psd_measured, fs = estimate_real_psd(
        h1_path, gps_merger, avoid_half_window=64.0, segment_duration=32.0)
    print()

    # ── Setup (identical to Stage 1) ────────────────────────────────────
    m1, m2 = 35.6, 30.6
    z = 0.09
    distance_Mpc = 440.0
    K_z = cosmological_K_factor(z)

    f_min, f_max, duration = 20.0, 400.0, 8.0
    df = 1.0 / duration
    f = np.arange(f_min, f_max, df)

    psd_real_grid = psd_on_grid(freqs_measured, psd_measured, f)

    print(f"[Step 2] Frequency grid: {f[0]:.1f}-{f[-1]:.1f} Hz, "
          f"df={df:.4f} Hz, {len(f)} bins")
    print(f"  Real PSD at 100 Hz: {np.interp(100, f, psd_real_grid):.3e} strain^2/Hz")
    print()

    snr0 = snr_optimal(f, psd_real_grid, df, m1, m2, 0.0, K_z,
                        distance_Mpc=distance_Mpc)
    print(f"  Optimal SNR at Lambda=0 with REAL noise PSD: {snr0:.2f}")
    print()

    Lambda_grid = np.linspace(-2.0, 2.0, 401)
    n_realizations = 5

    # ── Test A + B: injection grid (Lambda=0 and nonzero) ────────────────
    Lambda_true_values = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]

    print("=" * 72)
    print("TEST A+B — INJECTION/RECOVERY WITH REAL-NOISE-SHAPED COLORED GAUSSIAN")
    print("=" * 72)
    print()
    print(f"  {'Lambda_true':>12}  {'Lambda_ML (mean)':>18}  "
          f"{'scatter (std)':>14}  {'sigma_from_true':>16}")
    print("  " + "-" * 68)

    results_AB = []
    for Lam_true in Lambda_true_values:
        ml_estimates = []
        for real_idx in range(n_realizations):
            data = generate_injection_real_noise_shaped(
                m1, m2, Lam_true, K_z, f, df, psd_real_grid,
                distance_Mpc=distance_Mpc, seed=2000 + real_idx,
                add_signal=True)
            _, _, Lam_ml, _ = grid_search_lambda(
                data, f, psd_real_grid, df, m1, m2, K_z, Lambda_grid,
                distance_Mpc=distance_Mpc)
            ml_estimates.append(Lam_ml)

        ml_estimates = np.array(ml_estimates)
        mean_ml, scatter = np.mean(ml_estimates), np.std(ml_estimates)
        sigma_from_true = abs(mean_ml - Lam_true) / scatter if scatter > 0 else np.nan

        results_AB.append(dict(Lambda_true=Lam_true, mean_ml=mean_ml,
                                scatter=scatter, sigma=sigma_from_true))

        print(f"  {Lam_true:>12.3f}  {mean_ml:>18.4f}  {scatter:>14.4f}  "
              f"{sigma_from_true:>16.2f}")

    print()

    # ── Test C: noise-only null test ──────────────────────────────────────
    print("=" * 72)
    print("TEST C — NOISE-ONLY NULL TEST (no injected signal)")
    print("=" * 72)
    print()

    null_estimates = []
    for real_idx in range(n_realizations * 2):  # more realizations for null
        data_noise_only = generate_injection_real_noise_shaped(
            m1, m2, 0.0, K_z, f, df, psd_real_grid,
            distance_Mpc=distance_Mpc, seed=3000 + real_idx,
            add_signal=False)
        _, _, Lam_ml, _ = grid_search_lambda(
            data_noise_only, f, psd_real_grid, df, m1, m2, K_z, Lambda_grid,
            distance_Mpc=distance_Mpc)
        null_estimates.append(Lam_ml)

    null_estimates = np.array(null_estimates)
    print(f"  {len(null_estimates)} pure-noise realizations, no signal injected")
    print(f"  Lambda_ML values: {np.round(null_estimates, 3)}")
    print(f"  Mean: {np.mean(null_estimates):.4f}, "
          f"Std: {np.std(null_estimates):.4f}")
    print()
    print("  Expectation: values should scatter widely/randomly across the")
    print("  search grid with no consistent, statistically significant")
    print("  preferred Lambda (since there is no coherent signal to match).")

    null_scatter = np.std(null_estimates)
    null_range = np.max(null_estimates) - np.min(null_estimates)
    null_consistent_with_noise = null_range > 0.5 * (Lambda_grid[-1] - Lambda_grid[0]) * 0.1
    # crude check: null estimates should NOT cluster tightly (which would
    # indicate the likelihood has a spurious preferred Lambda even from
    # pure noise)

    print()
    print(f"  Null-test scatter: {null_scatter:.4f}, range: {null_range:.4f}")

    # ── Diagnosis ──────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("STAGE 2 DIAGNOSIS")
    print("=" * 72)
    print()

    ab_pass = all(
        abs(r["mean_ml"] - r["Lambda_true"]) < 2 * r["scatter"]
        if r["scatter"] > 0 else abs(r["mean_ml"] - r["Lambda_true"]) < 0.05
        for r in results_AB
    )

    print(f"  Test A+B (injection/recovery): {'PASS' if ab_pass else 'FAIL'}")
    print(f"  Test C (noise-only null): scatter={null_scatter:.4f} "
          f"(qualitative check -- see plot for distribution)")
    print()

    if ab_pass:
        print("  Real-noise-shaped PSD does not introduce recovery bias beyond")
        print("  Stage 1 (synthetic aLIGO-like noise). The matched-filter,")
        print("  phase-domain approach remains sound under realistic detector")
        print("  noise SPECTRAL SHAPE.")
        print()
        print("  NEXT STEP (Stage 3): analyze the REAL GW150914 strain segment")
        print("  itself (not a synthetic injection) with this pipeline.")
        print("  Note: Stage 2 used colored GAUSSIAN noise matched to the real")
        print("  PSD, not the actual non-Gaussian real strain time series --")
        print("  Stage 3 introduces real strain's potential non-Gaussianities")
        print("  and glitches as a further, separate consideration.")
    else:
        print("  Recovery degraded under real-noise spectral shape relative to")
        print("  Stage 1. Investigate PSD estimation, grid resolution, or")
        print("  frequency-band mismatch before proceeding to Stage 3.")

    # ── Plots ──────────────────────────────────────────────────────────────
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.loglog(freqs_measured, psd_measured, color="gray", lw=0.5, alpha=0.6,
              label="raw Welch estimate")
    ax.loglog(f, psd_real_grid, color="firebrick", lw=2,
              label="interpolated (analysis grid)")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"PSD [strain$^2$/Hz]")
    ax.set_title("Real H1 noise PSD (merger excluded)")
    ax.set_xlim(10, 500)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[1]
    Lt = [r["Lambda_true"] for r in results_AB]
    Lml = [r["mean_ml"] for r in results_AB]
    Lsc = [r["scatter"] for r in results_AB]
    ax.errorbar(Lt, Lml, yerr=Lsc, fmt="o-", color="steelblue", capsize=4,
                markersize=8, label="recovered")
    lims = [min(Lt) - 0.2, max(Lt) + 0.2]
    ax.plot(lims, lims, "k--", lw=1.5, label="ideal")
    ax.set_xlabel(r"Injected $\Lambda_{\rm true}$")
    ax.set_ylabel(r"Recovered $\Lambda_{\rm ML}$")
    ax.set_title("Stage 2: recovery with real-noise-shaped PSD")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.hist(null_estimates, bins=15, color="gray", edgecolor="k", alpha=0.7)
    ax.axvline(0, color="firebrick", lw=1.5, ls="--", label=r"$\Lambda=0$")
    ax.set_xlabel(r"Recovered $\Lambda_{\rm ML}$ (no signal injected)")
    ax.set_ylabel("Count")
    ax.set_title("Test C: noise-only null test")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Stage 2: Matched-filter Lambda recovery with real GWOSC noise",
                 fontsize=11)
    plt.tight_layout()
    path = out / "stage2_real_noise_recovery.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
