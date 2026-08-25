"""
gwosc_injection_recovery_test.py
====================================

THE CRITICAL TEST BEFORE ANY PHYSICAL INTERPRETATION:

The 51.9-sigma "detection" from gwosc_chirp_dispersion_test.py on real
GW150914 H1 data used a leading-order (0PN) Newtonian baseline compared
against Hilbert-transform frequency tracking. Before spending effort on
a higher-PN-order baseline, we must answer a cheaper, more fundamental
question:

    If we inject a signal with Lambda=0 (by construction) into REALISTIC
    detector noise, and run it through the EXACT SAME extraction and
    fitting pipeline, do we recover Lambda=0 (within errors), or do we
    recover a large spurious Lambda -- as the real-data run suggests?

If Lambda_recovered is large even when Lambda_true=0, this proves the
51.9-sigma "detection" on GW150914 was pipeline/model bias, not
evidence of dispersion. This is the necessary control before any
physical claim.

METHOD:

  1. Generate a synthetic chirp using the SAME 0PN Newtonian phase model
     used in the main pipeline (so the model perfectly matches the
     injection -- if Lambda_recovered != 0 even here, the bug is in the
     extraction/fitting code itself, not in PN-order mismatch).
  2. Inject this into REALISTIC noise: colored, using an approximate
     aLIGO-like noise PSD (not flat white noise, which was shown to be
     an inadequate test for the whitening pathway).
  3. Run the exact same load -> whiten -> bandpass -> extract -> fit
     pipeline as the main script.
  4. Repeat for several known injected Lambda values (0, and a few
     nonzero test values) to build a recovery curve.
  5. Report explicitly: does Lambda_recovered match Lambda_true within
     the fitted error bars?

This determines whether the pipeline is fit for physical interpretation
BEFORE any higher-PN-order investment.
"""

import numpy as np
import h5py
from pathlib import Path
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from gwosc_chirp_dispersion_test import (
    load_strain, whiten, bandpass, instantaneous_frequency,
    pn_chirp_mass, pn_frequency_of_time, fit_lambda_from_delta_t,
    cosmological_K_factor, EVENT_CATALOG, C_SI, G_SI, MSUN_SI
)
from scipy.signal import medfilt


def aligo_like_psd(f):
    """
    Approximate aLIGO design-sensitivity PSD (simplified analytic fit),
    strain^2/Hz. Not the exact aLIGO curve, but captures the right
    qualitative shape: rising steeply below ~20 Hz, a broad sensitive
    band 20-300 Hz, rising again above a few hundred Hz. This is enough
    to test whether whitening against COLORED (not flat) noise breaks
    the pipeline the way flat noise did not.
    """
    f = np.maximum(f, 1.0)
    f0 = 215.0
    x = f / f0
    S = 1e-49 * (x**-4.14 - 5*x**-2 + 111*(1 - x**2 + 0.5*x**4)/(1 + 0.5*x**2))
    return np.abs(S) + 1e-47  # floor to avoid negative/zero from the fit shape


def generate_injection(Lambda_true, m1, m2, z, fs=4096.0, duration=32.0,
                        seed=0, snr_scale=25.0):
    """
    Generate a synthetic H1-like strain segment containing:
      - colored (aLIGO-like) Gaussian noise
      - an injected 0PN Newtonian chirp with masses (m1, m2), optionally
        modified by a Lambda-dispersion time shift per frequency (to
        allow testing nonzero Lambda_true as well as Lambda_true=0)

    The dispersion is injected by shifting the arrival TIME of each
    instantaneous frequency component by delta_t(f) = -(Lambda/2c^3) *
    (2 pi f)^2 * K(z), i.e. by pre-warping the phase evolution so that,
    if analyzed correctly, a fit would recover Lambda_true.
    """
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    t = np.arange(n) / fs
    merger_t = duration / 2

    Mc_msun = pn_chirp_mass(m1, m2)
    Mc_SI = Mc_msun * MSUN_SI
    K = cosmological_K_factor(z)

    # Time-domain phase construction with dispersion pre-warp:
    # standard 0PN: f(tau) with tau = time before merger
    # dispersion shifts arrival time by delta_t(f) = -(Lambda/2c^3)*omega^2*K
    # equivalently: at fixed frequency f, the wave arrives delta_t(f)
    # EARLIER (for Lambda>0) or later, relative to the GR prediction.
    # We build this by first computing GR tau(f), then subtracting
    # delta_t(f) from the emission-frame time axis before regenerating.

    tau = np.maximum(merger_t - t, 1e-4)
    f_inst_GR = (1 / np.pi) * (5 / (256 * tau)) ** (3 / 8) * \
        (G_SI * Mc_SI / C_SI ** 3) ** (-5 / 8)
    f_inst_GR = np.clip(f_inst_GR, 1.0, 2000.0)

    omega_inst = 2 * np.pi * f_inst_GR
    delta_t_inject = -(Lambda_true / (2 * C_SI ** 3)) * omega_inst ** 2 * K

    # Warp the time axis: the phase that would have arrived at time t
    # in GR now arrives at t + delta_t_inject (dispersion delays/advances
    # each frequency component)
    t_warped = t + delta_t_inject

    # Re-derive instantaneous frequency vs the WARPED time axis by
    # interpolation (so integrating this phase reproduces the injected
    # dispersion when later compared against the undispersed GR model)
    order = np.argsort(t_warped)
    f_of_t_warped = np.interp(t, t_warped[order], f_inst_GR[order])
    f_of_t_warped = np.clip(f_of_t_warped, 1.0, 2000.0)

    phase = 2 * np.pi * np.cumsum(f_of_t_warped) / fs
    amplitude_envelope = np.exp(-((t - merger_t) / 4.0) ** 2) * (t < merger_t + 0.05)
    signal = amplitude_envelope * np.sin(phase)

    # Colored noise: generate via inverse-FFT filtering of white noise
    white_noise = rng.normal(0, 1, n)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    psd = aligo_like_psd(freqs)
    amplitude_spectrum = np.sqrt(psd * fs / 2)
    noise_fft = np.fft.rfft(white_noise) * amplitude_spectrum
    colored_noise = np.fft.irfft(noise_fft, n=n)
    colored_noise /= np.std(colored_noise)
    colored_noise *= 1e-23  # realistic strain noise floor scale

    # Scale injected signal to a target-ish SNR (crude, just needs to be
    # detectable by the crude Hilbert-transform extraction)
    signal_scaled = signal * snr_scale * np.std(colored_noise) / \
        (np.std(signal[amplitude_envelope > 0.5]) + 1e-30)

    strain = colored_noise + signal_scaled
    gps_start = 1126259462.4 - duration / 2

    return t, strain, fs, gps_start, merger_t, Mc_msun


def save_as_gwosc_hdf5(path, strain, fs, gps_start):
    with h5py.File(path, "w") as f:
        g = f.create_group("strain")
        ds = g.create_dataset("Strain", data=strain)
        ds.attrs["Npoints"] = len(strain)
        ds.attrs["Xspacing"] = 1 / fs
        ds.attrs["Xstart"] = gps_start
        ds.attrs["Xunits"] = "second"


def run_pipeline_on_injection(strain_path, gps_merger, Mc_msun, z):
    """Run the same extraction+fit pipeline as the main script."""
    t_h1, strain_h1, fs = load_strain(strain_path, gps_merger, half_window=16.0)

    strain_h1_white_full = whiten(strain_h1, fs)
    strain_h1_white = bandpass(strain_h1_white_full, fs, f_lo=20, f_hi=400)

    idx_center = len(t_h1) // 2
    window_samples = int(0.5 * fs)
    margin_samples = int(0.01 * fs)
    idx_lo = max(0, idx_center - window_samples)
    idx_hi = max(idx_lo + 1, idx_center - margin_samples)

    seg = strain_h1_white[idx_lo:idx_hi]
    t_seg = t_h1[idx_lo:idx_hi]

    t_freq, inst_freq, envelope = instantaneous_frequency(seg, fs)
    t_freq_abs = t_seg[0] + t_freq

    mf_len = 15 if len(inst_freq) > 15 else (len(inst_freq) // 2) * 2 + 1
    if mf_len >= 3:
        inst_freq = medfilt(inst_freq, kernel_size=mf_len)

    env_threshold = np.percentile(envelope, 40)
    valid = (inst_freq > 20) & (inst_freq < 350) & (envelope[:-1] > env_threshold)

    if valid.sum() < 10:
        for pctl in [30, 20, 10, 5]:
            env_threshold = np.percentile(envelope, pctl)
            valid = (inst_freq > 20) & (inst_freq < 350) & \
                    (envelope[:-1] > env_threshold)
            if valid.sum() >= 10:
                break

    f_obs = inst_freq[valid]
    t_obs = t_freq_abs[valid]

    if len(f_obs) < 10:
        return None  # insufficient points

    tau_before_merger_obs = gps_merger - t_obs
    tau_dense = np.logspace(-4, np.log10(2.0), 5000)
    f_dense = pn_frequency_of_time(tau_dense, Mc_msun)
    order = np.argsort(f_dense)
    tau_of_f = np.interp(f_obs, f_dense[order], tau_dense[order])

    t_PN_predicted = gps_merger - tau_of_f
    delta_t = t_obs - t_PN_predicted

    df_dt = np.gradient(f_dense[order], tau_dense[order])
    df_dt_at_f = np.interp(f_obs, f_dense[order], -df_dt)
    f_jitter = np.std(np.diff(f_obs)) if len(f_obs) > 1 else 1.0
    delta_t_err = np.full_like(delta_t, 1.0 / fs) + \
        f_jitter / np.maximum(df_dt_at_f, 1e-3)

    Lambda_fit, Lambda_err, t0_fit, K = fit_lambda_from_delta_t(
        f_obs, delta_t, delta_t_err, z)

    return dict(Lambda_fit=Lambda_fit, Lambda_err=Lambda_err,
                n_points=len(f_obs))


def main():
    out = Path(__file__).parent / "gwosc_results"
    out.mkdir(exist_ok=True)

    m1, m2, z = 35.6, 30.6, 0.09  # GW150914-like

    # Injected Lambda test values: 0, and several nonzero values spanning
    # the scale suggested by the spurious real-data fit (~1.4 m^2), to
    # see if the pipeline can distinguish them at all.
    Lambda_true_values = [0.0, 0.5, 1.0, 2.0, 5.0]

    print("=" * 72)
    print("INJECTION/RECOVERY TEST: does the pipeline recover Lambda_true?")
    print("=" * 72)
    print()
    print("  Using SAME 0PN Newtonian model for injection AND fitting")
    print("  baseline -- if Lambda_recovered != Lambda_true even here,")
    print("  the bug is in extraction/fitting, not PN-order mismatch.")
    print()
    print(f"  {'Lambda_true':>12}  {'Lambda_fit':>14}  {'Lambda_err':>12}  "
          f"{'n_sigma_from_true':>18}  {'n_points':>10}")
    print("  " + "-" * 74)

    results = []
    for i, Lam_true in enumerate(Lambda_true_values):
        t, strain, fs, gps_start, merger_t, Mc = generate_injection(
            Lam_true, m1, m2, z, seed=100 + i)
        gps_merger = gps_start + merger_t

        tmp_path = out / f"_tmp_injection_{i}.hdf5"
        save_as_gwosc_hdf5(tmp_path, strain, fs, gps_start)

        result = run_pipeline_on_injection(tmp_path, gps_merger, Mc, z)
        tmp_path.unlink()  # cleanup

        if result is None:
            print(f"  {Lam_true:>12.3f}  {'(insufficient points)':>14}")
            results.append(dict(Lambda_true=Lam_true, Lambda_fit=np.nan,
                                 Lambda_err=np.nan, n_points=0))
            continue

        Lf, Le = result["Lambda_fit"], result["Lambda_err"]
        n_sigma_from_true = abs(Lf - Lam_true) / Le if Le > 0 else float("nan")

        print(f"  {Lam_true:>12.3f}  {Lf:>14.4f}  {Le:>12.4f}  "
              f"{n_sigma_from_true:>18.2f}  {result['n_points']:>10}")

        results.append(dict(Lambda_true=Lam_true, Lambda_fit=Lf,
                             Lambda_err=Le, n_points=result["n_points"]))

    print()

    # ── Diagnosis ────────────────────────────────────────────────────────
    zero_case = results[0]
    print("=" * 72)
    print("DIAGNOSIS")
    print("=" * 72)
    print()

    if np.isnan(zero_case["Lambda_fit"]):
        print("  Lambda=0 injection failed to produce enough points -- cannot")
        print("  diagnose pipeline bias. Investigate extraction gate first.")
    else:
        bias_at_zero = zero_case["Lambda_fit"]
        bias_sigma = abs(bias_at_zero) / zero_case["Lambda_err"] \
            if zero_case["Lambda_err"] > 0 else float("nan")

        print(f"  Lambda_true=0 recovers Lambda_fit={bias_at_zero:.4f} "
              f"+/- {zero_case['Lambda_err']:.4f} ({bias_sigma:.1f} sigma from 0)")
        print()

        if bias_sigma > 5:
            print("  ┌────────────────────────────────────────────────────────┐")
            print("  │  PIPELINE BIAS CONFIRMED.                              │")
            print("  │  Even with a PERFECTLY MATCHED 0PN model (injection    │")
            print("  │  and fitting use the identical baseline), Lambda=0     │")
            print("  │  is NOT recovered. This proves the 51.9-sigma          │")
            print("  │  \"detection\" on real GW150914 data is a pipeline/     │")
            print("  │  extraction artifact, not evidence of dispersion.      │")
            print("  │  Root cause is in the extraction step (Hilbert phase   │")
            print("  │  tracking, envelope gating, or median filtering),      │")
            print("  │  NOT in PN-order mismatch. Fix extraction before       │")
            print("  │  investing in higher-PN-order baselines.               │")
            print("  └────────────────────────────────────────────────────────┘")
        else:
            print("  Lambda=0 is recovered within statistical error. The")
            print("  pipeline itself is not obviously biased at Lambda=0.")
            print("  The 51.9-sigma real-data result may then be attributable")
            print("  to PN-order mismatch (0PN vs. the true >=3.5PN waveform)")
            print("  rather than a pipeline bug -- higher-PN-order baseline")
            print("  should be the next investment.")

    # ── Plot recovery curve ──────────────────────────────────────────────
    Lt = np.array([r["Lambda_true"] for r in results])
    Lf = np.array([r["Lambda_fit"] for r in results])
    Le = np.array([r["Lambda_err"] for r in results])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    valid_mask = ~np.isnan(Lf)
    ax.errorbar(Lt[valid_mask], Lf[valid_mask], yerr=Le[valid_mask],
                fmt="o", color="steelblue", markersize=8, capsize=4,
                label="recovered Lambda")
    lims = [min(Lt.min(), Lf[valid_mask].min() if valid_mask.any() else 0) - 0.5,
            max(Lt.max(), Lf[valid_mask].max() if valid_mask.any() else 5) + 0.5]
    ax.plot(lims, lims, "k--", lw=1.5, label="perfect recovery (y=x)")
    ax.set_xlabel(r"Injected $\Lambda_{\rm true}$ [m$^2$]")
    ax.set_ylabel(r"Recovered $\Lambda_{\rm fit}$ [m$^2$]")
    ax.set_title("Injection/recovery test\n(same 0PN model for injection and fit)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out / "injection_recovery_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
