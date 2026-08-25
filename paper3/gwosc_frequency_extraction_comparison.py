"""
gwosc_frequency_extraction_comparison.py
============================================

ISOLATING THE EXTRACTION METHOD: is the bias fundamental to the Hilbert
transform, or specific to how instantaneous_frequency() implements it?

gwosc_extraction_ablation.py (Test A) showed that even a noiseless,
unenveloped analytic chirp produces enormous Lambda bias (2338 sigma)
when frequency is extracted via Hilbert transform, versus ~0 sigma when
the exact analytic f(t) is used directly. This script compares FIVE
independent frequency-extraction methods on the IDENTICAL analytic
chirp, to determine which methods are usable and which are not:

  1. TRUE analytical f(t)      -- ground truth (oracle), sanity check
  2. Direct phase derivative   -- d(unwrapped phase)/dt computed from
                                   the KNOWN analytic phase (not from a
                                   reconstructed signal) -- upper bound
                                   on what any phase-based method could
                                   achieve
  3. Hilbert transform         -- scipy.signal.hilbert + phase diff
                                   (the method used in the main pipeline)
  4. Zero-crossing frequency   -- classic, crude, but has NO systematic
                                   phase-derivative bias by construction
  5. STFT ridge extraction     -- short-time Fourier transform, track
                                   the peak-magnitude bin per time frame
                                   (standard time-frequency ridge method,
                                   used in gravitational-wave chirp
                                   tracking literature)

For each method, report:
  - RMSE(f_estimated - f_true)
  - bias (mean signed error)
  - Lambda_fit and its significance vs Lambda=0

This determines whether ANY simple, non-matched-filter method can be
trusted for this test, or whether matched filtering is unavoidable.
"""

import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from gwosc_chirp_dispersion_test import (
    pn_chirp_mass, pn_frequency_of_time, fit_lambda_from_delta_t,
    cosmological_K_factor, C_SI, G_SI, MSUN_SI
)
from scipy.signal import hilbert, stft


def generate_clean_chirp(m1=35.6, m2=30.6, fs=4096.0, duration=2.0,
                          tau_min=0.3, tau_max=2.3):
    """Same construction as Test A in the ablation script."""
    Mc_msun = pn_chirp_mass(m1, m2)
    Mc_SI = Mc_msun * MSUN_SI

    n = int(fs * duration)
    t = np.arange(n) / fs
    tau = tau_max - t  # decreasing from tau_max to tau_min
    f_true = (1 / np.pi) * (5 / (256 * tau)) ** (3 / 8) * \
        (G_SI * Mc_SI / C_SI ** 3) ** (-5 / 8)

    phase_true = 2 * np.pi * np.cumsum(f_true) / fs
    signal = np.sin(phase_true)

    return t, signal, f_true, phase_true, fs, Mc_msun


def method_1_oracle(t, f_true, **kw):
    return t[:-1], f_true[:-1]


def method_2_direct_phase_derivative(t, phase_true, fs, **kw):
    """Upper bound: derivative of the KNOWN analytic phase (not
    reconstructed from the signal). This is what any phase-tracking
    method would achieve in the noiseless, infinite-precision limit."""
    f_est = np.diff(np.unwrap(phase_true)) / (2 * np.pi) * fs
    return t[:-1], f_est


def method_3_hilbert(signal, fs, t, **kw):
    analytic = hilbert(signal)
    phase = np.unwrap(np.angle(analytic))
    f_est = np.diff(phase) / (2 * np.pi) * fs
    return t[:-1], f_est


def method_4_zero_crossing(signal, fs, t, **kw):
    """
    Instantaneous frequency from the spacing between successive
    positive-going zero crossings: f = 1 / (2 * dt_between_crossings)
    for a signal with roughly one zero-crossing pair per cycle.
    """
    sign = np.sign(signal)
    crossings = np.where(np.diff(sign) > 0)[0]  # positive-going only
    if len(crossings) < 3:
        return np.array([]), np.array([])
    t_cross = t[crossings]
    dt_cross = np.diff(t_cross)
    f_est = 1.0 / dt_cross
    t_est = t_cross[:-1] + dt_cross / 2  # midpoint of each crossing pair
    return t_est, f_est


def method_5_stft_ridge(signal, fs, t, nperseg=256, **kw):
    """
    Short-time Fourier transform, track the frequency bin of maximum
    magnitude in each time frame (standard ridge-extraction approach).
    """
    f_bins, t_frames, Zxx = stft(signal, fs=fs, nperseg=nperseg,
                                   noverlap=nperseg - 8)
    mag = np.abs(Zxx)
    peak_bin = np.argmax(mag, axis=0)
    f_est = f_bins[peak_bin]
    t_est = t_frames
    return t_est, f_est


def evaluate_method(t_est, f_est, t_true, f_true, Mc_msun, z,
                     tau_max=2.3, f_range=(15, 500)):
    """
    Compute RMSE/bias against ground truth (via interpolation onto a
    common time grid), then fit Lambda against the 0PN model.
    """
    if len(f_est) < 10:
        return None

    valid = (f_est > f_range[0]) & (f_est < f_range[1])
    t_est_v = t_est[valid]
    f_est_v = f_est[valid]

    if len(f_est_v) < 10:
        return None

    # Interpolate ground truth onto estimated time points for RMSE
    f_true_interp = np.interp(t_est_v, t_true, f_true)
    residual = f_est_v - f_true_interp
    rmse = np.sqrt(np.mean(residual ** 2))
    bias = np.mean(residual)

    # Fit Lambda: predicted arrival time for each estimated frequency,
    # using the SAME 0PN model + tau=tau_max-t construction
    tau_dense = np.logspace(-4, np.log10(3.0), 5000)
    f_dense = pn_frequency_of_time(tau_dense, Mc_msun)
    order = np.argsort(f_dense)
    tau_of_f = np.interp(f_est_v, f_dense[order], tau_dense[order])
    t_pred = tau_max - tau_of_f
    delta_t = t_est_v - t_pred
    delta_t_err = np.full_like(delta_t, 1.0 / 4096.0)

    try:
        Lambda_fit, Lambda_err, t0_fit, K = fit_lambda_from_delta_t(
            f_est_v, delta_t, delta_t_err, z)
    except Exception:
        Lambda_fit, Lambda_err = float("nan"), float("nan")

    sigma = abs(Lambda_fit) / Lambda_err if Lambda_err > 0 else float("nan")

    return dict(rmse=rmse, bias=bias, Lambda_fit=Lambda_fit,
                Lambda_err=Lambda_err, sigma=sigma, n_points=len(f_est_v))


def main():
    out = Path(__file__).parent / "gwosc_results"
    out.mkdir(exist_ok=True)

    z = 0.09
    tau_max = 2.3
    t, signal, f_true, phase_true, fs, Mc_msun = generate_clean_chirp(
        tau_max=tau_max)

    print("=" * 72)
    print("FREQUENCY-EXTRACTION METHOD COMPARISON")
    print("=" * 72)
    print()
    print(f"  Signal: pure analytic 0PN chirp, no noise, no envelope")
    print(f"  Chirp mass: {Mc_msun:.3f} Msun, true f range: "
          f"{f_true.min():.1f}-{f_true.max():.1f} Hz")
    print()

    methods = {
        "1_oracle": lambda: method_1_oracle(t, f_true),
        "2_direct_phase_deriv": lambda: method_2_direct_phase_derivative(
            t, phase_true, fs),
        "3_hilbert": lambda: method_3_hilbert(signal, fs, t),
        "4_zero_crossing": lambda: method_4_zero_crossing(signal, fs, t),
        "5_stft_ridge": lambda: method_5_stft_ridge(signal, fs, t),
    }

    results = {}
    for name, fn in methods.items():
        t_est, f_est = fn()
        r = evaluate_method(t_est, f_est, t, f_true, Mc_msun, z, tau_max=tau_max)
        results[name] = r
        if r is None:
            print(f"  {name:>24}: insufficient points")
        else:
            print(f"  {name:>24}: RMSE={r['rmse']:>8.3f} Hz  bias={r['bias']:>8.3f} Hz  "
                  f"Lambda={r['Lambda_fit']:>12.4f}  sigma={r['sigma']:>10.2f}  "
                  f"n={r['n_points']}")

    print()
    print("=" * 72)
    print("DIAGNOSIS")
    print("=" * 72)
    print()

    r2 = results.get("2_direct_phase_deriv")
    r3 = results.get("3_hilbert")
    r4 = results.get("4_zero_crossing")
    r5 = results.get("5_stft_ridge")

    if r2 and r2["sigma"] < 5:
        print("  Direct phase derivative (method 2) is UNBIASED, confirming the")
        print("  phase-derivative APPROACH is sound in principle -- the issue is")
        print("  specific to Hilbert's signal RECONSTRUCTION of the phase, not")
        print("  the differentiation step itself.")
    print()

    candidates = []
    for name, r in [("zero_crossing", r4), ("stft_ridge", r5)]:
        if r and r["sigma"] < 5:
            candidates.append(name)

    if candidates:
        print(f"  VIABLE ALTERNATIVES FOUND: {', '.join(candidates)}")
        print("  These methods recover Lambda=0 within statistical error on the")
        print("  clean analytic test and should be substituted for the Hilbert")
        print("  transform in the main pipeline before any real-data re-analysis.")
    else:
        print("  NO SIMPLE METHOD TESTED HERE RECOVERS Lambda=0 RELIABLY.")
        print("  This suggests the bias may be more fundamental than the specific")
        print("  extraction algorithm -- possibly in how delta_t(f) is computed")
        print("  or in the PN-inversion (tau_of_f) step shared by all methods.")
        print("  Matched filtering (cross-correlation against a template bank)")
        print("  remains the recommended path forward, as it does not rely on")
        print("  any form of single-cycle instantaneous-frequency estimation.")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(t, f_true, "k-", lw=2, label="true f(t)")
    for name, fn in methods.items():
        if name == "1_oracle":
            continue
        t_est, f_est = fn()
        valid = (f_est > 15) & (f_est < 500)
        ax.plot(t_est[valid], f_est[valid], ".", markersize=2, alpha=0.6,
                label=name)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title("Extracted frequency tracks vs ground truth")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    names = [k for k in results if results[k] is not None]
    sigmas = [results[k]["sigma"] for k in names]
    colors = ["forestgreen" if s < 5 else "firebrick" for s in sigmas]
    ax.barh(names, sigmas, color=colors)
    ax.axvline(5, color="k", ls="--", lw=1, label="5-sigma threshold")
    ax.set_xlabel(r"$|\Lambda_{\rm fit}| / \sigma_\Lambda$")
    ax.set_title("Lambda bias significance by method")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")

    plt.suptitle("Frequency-extraction method comparison (clean analytic chirp)",
                 fontsize=11)
    plt.tight_layout()
    path = out / "extraction_method_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
