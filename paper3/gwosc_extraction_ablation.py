"""
gwosc_extraction_ablation.py
================================

LOCALIZING THE BIAS: which stage of the extraction pipeline introduces
the Lambda != 0 bias confirmed by gwosc_injection_recovery_test.py?

The injection/recovery test proved the pipeline is biased (Lambda=0
injected, Lambda=-1.9053 recovered), but did not identify WHICH stage
of the extraction is responsible:

    Hilbert transform phase tracking?
    Envelope-based SNR gating?
    Median filtering?
    Some interaction between chirp shape + amplitude envelope + Hilbert?

This script tests two independent things:

  TEST A — ANALYTIC PHASE, NO WAVEFORM GENERATOR
    Generate a pure cosine with a KNOWN, EXACT instantaneous frequency
    law (the same 0PN f(t) formula), with NO amplitude envelope, NO
    noise, NO merger cutoff. Run it through each extraction stage
    incrementally. If bias appears even here, the bug is in the
    Hilbert/gating/filtering MATH itself, independent of any waveform
    realism.

  TEST B — STAGE-BY-STAGE ABLATION ON THE REALISTIC INJECTION
    Take the same colored-noise, amplitude-enveloped injection used in
    gwosc_injection_recovery_test.py (Lambda_true=0), and run it through
    each pipeline stage cumulatively:
      A: ideal (ground-truth) instantaneous frequency directly (no
         extraction at all -- this is the "perfect extraction" oracle)
      B: + Hilbert transform phase tracking (raw, no cleanup)
      C: + envelope-based SNR gate
      D: + median filter smoothing
      E: full current pipeline (identical to injection_recovery_test)

    Report Lambda_fit at each stage to localize exactly where the bias
    enters.
"""

import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from gwosc_chirp_dispersion_test import (
    pn_chirp_mass, pn_frequency_of_time, fit_lambda_from_delta_t,
    cosmological_K_factor, C_SI, G_SI, MSUN_SI, instantaneous_frequency
)
from gwosc_injection_recovery_test import aligo_like_psd
from scipy.signal import hilbert, medfilt


# ══════════════════════════════════════════════════════════════════════════
# TEST A — PURE ANALYTIC PHASE, NO WAVEFORM REALISM
# ══════════════════════════════════════════════════════════════════════════

def test_A_analytic_phase(m1=35.6, m2=30.6, z=0.09, fs=4096.0, duration=2.0):
    """
    Pure sinusoid with EXACT 0PN instantaneous frequency law, no
    envelope, no noise, no merger singularity cutoff (we stop safely
    before tau->0). Tests whether Hilbert-transform frequency tracking
    itself is unbiased for a clean, noiseless signal.
    """
    print("=" * 72)
    print("TEST A — PURE ANALYTIC PHASE (no noise, no envelope, no cutoff)")
    print("=" * 72)
    print()

    Mc_msun = pn_chirp_mass(m1, m2)
    Mc_SI = Mc_msun * MSUN_SI

    n = int(fs * duration)
    t = np.arange(n) / fs
    # Keep well away from merger singularity: tau in [0.3, 2.3] s
    tau = 2.3 - t  # decreasing from 2.3 to 0.3 over the segment
    f_true = (1 / np.pi) * (5 / (256 * tau)) ** (3 / 8) * \
        (G_SI * Mc_SI / C_SI ** 3) ** (-5 / 8)

    phase = 2 * np.pi * np.cumsum(f_true) / fs
    signal = np.sin(phase)  # unit amplitude, no envelope, no noise

    print(f"  Signal: pure sin(phase), duration={duration}s, fs={fs}")
    print(f"  True frequency range: {f_true.min():.2f} - {f_true.max():.2f} Hz")
    print()

    # Stage-by-stage
    results = {}

    # Stage 1: ideal (ground truth) -- oracle, should give Lambda~0 trivially
    results["A1_ideal"] = extract_and_fit(t[:-1], f_true[:-1], Mc_msun, z,
                                            label="ideal ground-truth f(t)")

    # Stage 2: raw Hilbert transform (no gating, no smoothing)
    t_freq, inst_freq, envelope = instantaneous_frequency(signal, fs)
    # Use full range, no gate
    valid_range = (inst_freq > 20) & (inst_freq < 500)
    results["A2_hilbert_raw"] = extract_and_fit(
        t[:-1][valid_range], inst_freq[valid_range], Mc_msun, z,
        label="raw Hilbert transform, band-limited only")

    # Stage 3: + median filter
    inst_freq_mf = medfilt(inst_freq, kernel_size=15)
    valid_mf = (inst_freq_mf > 20) & (inst_freq_mf < 500)
    results["A3_hilbert_medfilt"] = extract_and_fit(
        t[:-1][valid_mf], inst_freq_mf[valid_mf], Mc_msun, z,
        label="Hilbert + median filter")

    print()
    print("  Summary (Test A):")
    print(f"  {'Stage':>28}  {'Lambda_fit':>14}  {'Lambda_err':>12}  {'sigma':>10}")
    print("  " + "-" * 70)
    for key, r in results.items():
        if r is None:
            print(f"  {key:>28}  (insufficient points)")
            continue
        sigma = abs(r["Lambda_fit"]) / r["Lambda_err"] if r["Lambda_err"] > 0 else float("nan")
        print(f"  {key:>28}  {r['Lambda_fit']:>14.4f}  {r['Lambda_err']:>12.4f}  {sigma:>10.2f}")
    print()

    return results


def extract_and_fit(t_arr, f_arr, Mc_msun, z, label=""):
    """Given (t, f) pairs, fit Lambda against the 0PN prediction."""
    if len(f_arr) < 10:
        return None

    # We need a reference merger time; use the max-frequency point's
    # implied tau as an anchor (since this synthetic test has no
    # explicit "merger" -- we anchor to the true model at t=duration
    # using the KNOWN construction: tau = 2.3 - t)
    tau_dense = np.logspace(-4, np.log10(3.0), 5000)
    f_dense = pn_frequency_of_time(tau_dense, Mc_msun)
    order = np.argsort(f_dense)

    tau_of_f = np.interp(f_arr, f_dense[order], tau_dense[order])
    # In this synthetic construction, tau = 2.3 - t_true, so the
    # "predicted" t for a given f is t_pred = 2.3 - tau_of_f
    t_pred = 2.3 - tau_of_f
    delta_t = t_arr - t_pred

    delta_t_err = np.full_like(delta_t, 1.0 / 4096.0)  # 1 sample uncertainty floor

    try:
        Lambda_fit, Lambda_err, t0_fit, K = fit_lambda_from_delta_t(
            f_arr, delta_t, delta_t_err, z)
    except Exception as e:
        print(f"    [{label}] fit failed: {e}")
        return None

    return dict(Lambda_fit=Lambda_fit, Lambda_err=Lambda_err, n_points=len(f_arr))


# ══════════════════════════════════════════════════════════════════════════
# TEST B — STAGE-BY-STAGE ABLATION ON REALISTIC INJECTION
# ══════════════════════════════════════════════════════════════════════════

def test_B_realistic_ablation(m1=35.6, m2=30.6, z=0.09, fs=4096.0,
                                duration=32.0, seed=100, snr_scale=25.0):
    """
    Same realistic injection as gwosc_injection_recovery_test.py
    (Lambda_true=0), but track Lambda_fit after EACH stage of
    extraction, cumulatively, to localize where bias enters.
    """
    print("=" * 72)
    print("TEST B — STAGE-BY-STAGE ABLATION (realistic injection, Lambda_true=0)")
    print("=" * 72)
    print()

    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    t = np.arange(n) / fs
    merger_t = duration / 2

    Mc_msun = pn_chirp_mass(m1, m2)
    Mc_SI = Mc_msun * MSUN_SI

    tau = np.maximum(merger_t - t, 1e-4)
    f_inst_GR = (1 / np.pi) * (5 / (256 * tau)) ** (3 / 8) * \
        (G_SI * Mc_SI / C_SI ** 3) ** (-5 / 8)
    f_inst_GR = np.clip(f_inst_GR, 1.0, 2000.0)

    phase = 2 * np.pi * np.cumsum(f_inst_GR) / fs
    amplitude_envelope = np.exp(-((t - merger_t) / 4.0) ** 2) * (t < merger_t + 0.05)
    signal = amplitude_envelope * np.sin(phase)

    white_noise = rng.normal(0, 1, n)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    psd = aligo_like_psd(freqs)
    amplitude_spectrum = np.sqrt(psd * fs / 2)
    noise_fft = np.fft.rfft(white_noise) * amplitude_spectrum
    colored_noise = np.fft.irfft(noise_fft, n=n)
    colored_noise /= np.std(colored_noise)
    colored_noise *= 1e-23

    signal_scaled = signal * snr_scale * np.std(colored_noise) / \
        (np.std(signal[amplitude_envelope > 0.5]) + 1e-30)

    strain_clean = signal_scaled          # no noise, for isolating envelope/Hilbert effects
    strain_noisy = colored_noise + signal_scaled  # full realistic case

    gps_merger = 1126259462.4  # dummy anchor, only relative timing matters here

    results = {}

    # Stage B0: ideal ground-truth frequency (oracle) -- sanity check only
    idx_window = slice(int((merger_t - 0.5) * fs), int((merger_t - 0.01) * fs))
    f_ideal = f_inst_GR[idx_window]
    t_ideal = t[idx_window]
    results["B0_ideal_oracle"] = extract_and_fit(
        t_ideal, f_ideal, Mc_msun, z, label="ideal oracle (ground truth)")

    # Stage B1: Hilbert on CLEAN signal (envelope, no noise) -- isolates
    # whether the amplitude envelope itself (independent of noise)
    # biases Hilbert phase tracking
    t_freq, inst_freq_clean, env_clean = instantaneous_frequency(
        strain_clean[idx_window.start:idx_window.stop + 1], fs)
    t_freq_abs = t_ideal[0] + t_freq
    valid = (inst_freq_clean > 20) & (inst_freq_clean < 350)
    results["B1_hilbert_clean_signal"] = extract_and_fit(
        t_freq_abs[valid], inst_freq_clean[valid], Mc_msun, z,
        label="Hilbert on clean (noiseless) enveloped signal")

    # Stage B2: Hilbert on NOISY signal, no gate, no filter
    t_freq_n, inst_freq_noisy, env_noisy = instantaneous_frequency(
        strain_noisy[idx_window.start:idx_window.stop + 1], fs)
    t_freq_n_abs = t_ideal[0] + t_freq_n
    valid_n = (inst_freq_noisy > 20) & (inst_freq_noisy < 350)
    results["B2_hilbert_noisy_raw"] = extract_and_fit(
        t_freq_n_abs[valid_n], inst_freq_noisy[valid_n], Mc_msun, z,
        label="Hilbert on noisy signal, band-gate only")

    # Stage B3: + envelope-based SNR gate (as in main pipeline)
    env_threshold = np.percentile(env_noisy, 40)
    valid_env = valid_n & (env_noisy[:len(valid_n)] > env_threshold)
    results["B3_plus_envelope_gate"] = extract_and_fit(
        t_freq_n_abs[valid_env], inst_freq_noisy[valid_env], Mc_msun, z,
        label="+ envelope SNR gate (40th percentile)")

    # Stage B4: + median filter (full current pipeline)
    mf_len = 15 if len(inst_freq_noisy) > 15 else (len(inst_freq_noisy) // 2) * 2 + 1
    inst_freq_mf = medfilt(inst_freq_noisy, kernel_size=mf_len)
    valid_full = (inst_freq_mf > 20) & (inst_freq_mf < 350) & \
                 (env_noisy[:len(inst_freq_mf)] > env_threshold)
    results["B4_full_pipeline"] = extract_and_fit(
        t_freq_n_abs[valid_full], inst_freq_mf[valid_full], Mc_msun, z,
        label="+ median filter (full current pipeline)")

    print()
    print("  Summary (Test B) -- cumulative ablation:")
    print(f"  {'Stage':>30}  {'Lambda_fit':>14}  {'Lambda_err':>12}  "
          f"{'sigma':>10}  {'n_pts':>7}")
    print("  " + "-" * 82)
    for key, r in results.items():
        if r is None:
            print(f"  {key:>30}  (insufficient points)")
            continue
        sigma = abs(r["Lambda_fit"]) / r["Lambda_err"] if r["Lambda_err"] > 0 else float("nan")
        print(f"  {key:>30}  {r['Lambda_fit']:>14.4f}  {r['Lambda_err']:>12.4f}  "
              f"{sigma:>10.2f}  {r['n_points']:>7}")
    print()

    return results


# ══════════════════════════════════════════════════════════════════════════
def main():
    out = Path(__file__).parent / "gwosc_results"
    out.mkdir(exist_ok=True)

    results_A = test_A_analytic_phase()
    results_B = test_B_realistic_ablation()

    # ── Diagnosis ────────────────────────────────────────────────────────
    print("=" * 72)
    print("LOCALIZATION DIAGNOSIS")
    print("=" * 72)
    print()

    def get_sigma(r):
        if r is None or r["Lambda_err"] <= 0:
            return float("nan")
        return abs(r["Lambda_fit"]) / r["Lambda_err"]

    sigA1 = get_sigma(results_A.get("A1_ideal"))
    sigA2 = get_sigma(results_A.get("A2_hilbert_raw"))
    sigA3 = get_sigma(results_A.get("A3_hilbert_medfilt"))

    print("  Test A (pure analytic phase, no noise/envelope):")
    print(f"    Ideal oracle:          {sigA1:.2f} sigma from 0")
    print(f"    Raw Hilbert:           {sigA2:.2f} sigma from 0")
    print(f"    Hilbert + medfilt:     {sigA3:.2f} sigma from 0")
    print()
    if sigA2 > 5:
        print("  => Hilbert transform frequency tracking is ITSELF biased even")
        print("     on a clean, noiseless, unenveloped chirp. The bug is in the")
        print("     core phase-derivative method, not noise or envelope effects.")
    elif sigA3 > 5 and sigA2 < 5:
        print("  => Median filtering introduces the bias; raw Hilbert tracking")
        print("     is fine on clean signals.")
    else:
        print("  => Test A stages are all consistent with Lambda=0: the bias")
        print("     requires noise and/or amplitude envelope to appear (see Test B).")
    print()

    sigB0 = get_sigma(results_B.get("B0_ideal_oracle"))
    sigB1 = get_sigma(results_B.get("B1_hilbert_clean_signal"))
    sigB2 = get_sigma(results_B.get("B2_hilbert_noisy_raw"))
    sigB3 = get_sigma(results_B.get("B3_plus_envelope_gate"))
    sigB4 = get_sigma(results_B.get("B4_full_pipeline"))

    print("  Test B (realistic injection, cumulative ablation):")
    print(f"    B0 ideal oracle:            {sigB0:.2f} sigma")
    print(f"    B1 Hilbert, clean signal:   {sigB1:.2f} sigma")
    print(f"    B2 Hilbert, noisy, raw:     {sigB2:.2f} sigma")
    print(f"    B3 + envelope gate:         {sigB3:.2f} sigma")
    print(f"    B4 + median filter (full):  {sigB4:.2f} sigma")
    print()

    stages = [("B0->B1 (add amplitude envelope)", sigB0, sigB1),
              ("B1->B2 (add colored noise)", sigB1, sigB2),
              ("B2->B3 (add envelope SNR gate)", sigB2, sigB3),
              ("B3->B4 (add median filter)", sigB3, sigB4)]

    print("  Bias introduced at each transition:")
    for name, before, after in stages:
        if np.isnan(before) or np.isnan(after):
            print(f"    {name}: (insufficient data)")
            continue
        jump = after - before
        flag = "  <-- LARGEST JUMP" if abs(jump) == max(
            abs(a - b) for _, b, a in stages if not np.isnan(a) and not np.isnan(b)
        ) else ""
        print(f"    {name}: {before:.1f} -> {after:.1f} sigma (delta={jump:+.1f}){flag}")

    print()
    print("  ROOT CAUSE: the stage with the largest sigma jump above is the")
    print("  primary contributor to pipeline bias and should be fixed or")
    print("  replaced first.")


if __name__ == "__main__":
    main()
