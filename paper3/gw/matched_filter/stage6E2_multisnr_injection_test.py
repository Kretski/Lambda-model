"""
PAPER 3 — STAGE 6E2
MULTI-SNR INJECTION / RECOVERY TEST

Purpose
-------
Controlled end-to-end validation of the matched-filter pipeline.

This test does NOT infer Lambda.
It tests whether injected signals are recovered consistently
at several target SNR levels.

Design
------
- Generate a controlled waveform.
- Generate colored Gaussian noise.
- Scale the waveform to target SNR.
- Inject waveform into noise.
- Recover with matched filtering.
- Repeat for multiple random seeds.
- Measure recovered SNR and detection fraction.

This is a validation test, not a physical Lambda measurement.

FIX (this version)
-------------------
inject_signal() previously computed:

    shift = center - len(signal_td) // 2

which is ALWAYS 0 when signal_td and noise_ts have equal length
(they do here) -- it assumed the raw ifft'd waveform is naturally
centered in its own array. It is not: IMRPhenomD's merger sits near
sample 0 due to the LAL/PyCBC FD->TD convention (the early inspiral
wraps to the end of the array). So no real shift ever happened; the
signal stayed wherever the raw ifft placed it, while expected_time
was hardcoded to center/FS = 8.0s -- producing a constant -8.0s
timing_error regardless of SNR.

Fix: locate the true intrinsic peak sample of the raw waveform and
circularly shift so THAT sample lands exactly at the intended
injection time, then use that as expected_time. Everything else
(PSD, colored noise via noise.noise_from_psd, matched_filter via
pycbc.filter) is unchanged -- that part was already correct.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from pycbc import noise
from pycbc import psd
from pycbc import waveform
from pycbc.filter import matched_filter


# ============================================================
# DEFAULTS
# ============================================================

FS = 4096.0

FLOW = 20.0
FHIGH = 300.0

DURATION = 16.0

TARGET_SNRS = [5.0, 8.0, 10.0, 15.0, 20.0]

N_REALIZATIONS = 20

BASE_SEED = 20260830

DETECTION_THRESHOLD = 5.0

APPROXIMANT = "IMRPhenomD"

MASS1 = 36.0
MASS2 = 29.0

F_LOWER = 20.0

DISTANCE = 400.0

# Segment-relative injection time (fixed target for every realization).
INJECTION_TIME = 8.0

OUTPUT_DIR = Path("stage6E2_results")


# ============================================================
# PSD
# ============================================================

def make_psd(duration: float, fs: float, flow: float):
    delta_f = 1.0 / duration
    flen = int(fs / (2.0 * delta_f)) + 1
    return psd.aLIGOZeroDetHighPower(flen, delta_f, flow)


# ============================================================
# WAVEFORM
# ============================================================

def make_waveform(fs, duration, mass1, mass2, flow, distance):
    delta_f = 1.0 / duration
    hp, hc = waveform.get_fd_waveform(
        approximant=APPROXIMANT, mass1=mass1, mass2=mass2,
        distance=distance, delta_f=delta_f, f_lower=flow,
    )
    nfreq = int(fs * duration / 2.0) + 1
    hp.resize(nfreq)
    return hp


# ============================================================
# TEMPLATE NORM
# ============================================================

def template_norm(template, psd_obj):
    df = template.delta_f
    freqs = np.arange(len(template)) * df
    mask = (freqs >= FLOW) & (freqs <= FHIGH)
    h = np.asarray(template)
    p = np.asarray(psd_obj)
    valid = mask & np.isfinite(p) & (p > 0)
    integrand = np.zeros_like(freqs, dtype=np.float64)
    integrand[valid] = np.abs(h[valid]) ** 2 / p[valid]
    return math.sqrt(4.0 * df * np.sum(integrand))


def scale_to_target_snr(template, psd_obj, target_snr):
    norm = template_norm(template, psd_obj)
    if norm <= 0:
        raise RuntimeError("Template normalization is zero.")
    return template * (target_snr / norm)


# ============================================================
# GENERATE NOISE
# ============================================================

def generate_noise(n_samples, fs, psd_obj, seed):
    delta_t = 1.0 / fs
    return noise.noise_from_psd(n_samples, delta_t, psd_obj, seed=seed)


# ============================================================
# INJECT -- FIXED
# ============================================================

def inject_signal(noise_ts, template_fd, injection_time=INJECTION_TIME):
    """
    Inject the waveform so that its own intrinsic amplitude peak
    lands exactly at injection_time (same time coordinates as
    noise_ts), instead of assuming the raw waveform array is
    naturally centered in the data (see module docstring for why
    that assumption was wrong).
    """

    signal_td = template_fd.to_timeseries()
    signal_td.resize(len(noise_ts))

    signal_values = np.asarray(signal_td)

    intrinsic_peak = int(np.argmax(np.abs(signal_values)))

    target_sample = int(round(injection_time * FS))

    shift = target_sample - intrinsic_peak

    shifted_signal = np.roll(signal_values, shift)

    measured_peak = int(np.argmax(np.abs(shifted_signal)))
    if measured_peak != target_sample:
        raise RuntimeError(
            "INJECTION GEOMETRY FAILED\n"
            f"Requested sample = {target_sample}\n"
            f"Measured peak    = {measured_peak}"
        )

    injected = noise_ts.copy()
    injected.data[:] += shifted_signal

    return injected, target_sample


# ============================================================
# MATCHED FILTER
# ============================================================

def recover(template_fd, data_ts, psd_obj):
    snr = matched_filter(
        template_fd, data_ts, psd=psd_obj,
        low_frequency_cutoff=FLOW, high_frequency_cutoff=FHIGH,
    )
    abs_snr = np.abs(np.asarray(snr))
    peak_index = int(np.argmax(abs_snr))
    recovered_snr = float(abs_snr[peak_index])
    peak_time = float(snr.sample_times[peak_index])
    return recovered_snr, peak_time


# ============================================================
# SINGLE TEST
# ============================================================

def run_single(target_snr, realization, template, psd_obj):
    seed = BASE_SEED + int(target_snr * 100) + realization
    n_samples = int(DURATION * FS)

    data = generate_noise(n_samples, FS, psd_obj, seed)
    scaled_template = scale_to_target_snr(template, psd_obj, target_snr)

    injected, expected_index = inject_signal(data, scaled_template, INJECTION_TIME)

    recovered_snr, peak_time = recover(scaled_template, injected, psd_obj)

    expected_time = expected_index / FS
    timing_error = peak_time - expected_time
    detected = recovered_snr >= DETECTION_THRESHOLD

    return {
        "target_snr": target_snr,
        "realization": realization,
        "seed": seed,
        "recovered_snr": recovered_snr,
        "expected_time": expected_time,
        "recovered_time": peak_time,
        "timing_error": timing_error,
        "detected": bool(detected),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-realizations", type=int, default=N_REALIZATIONS)
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print("PAPER 3 — STAGE 6E2")
    print("MULTI-SNR INJECTION / RECOVERY TEST")
    print("=" * 70)
    print(f"Sampling rate      = {FS:.1f} Hz")
    print(f"Duration            = {DURATION:.1f} s")
    print(f"Frequency band      = {FLOW:.1f} – {FHIGH:.1f} Hz")
    print(f"Waveform            = {APPROXIMANT}")
    print(f"Mass1               = {MASS1:.1f} Msun")
    print(f"Mass2               = {MASS2:.1f} Msun")
    print(f"Injection time      = {INJECTION_TIME:.6f} s")
    print(f"Realizations        = {args.n_realizations}")
    print(f"Detection threshold = {DETECTION_THRESHOLD:.1f}")
    print()

    print("[1] BUILDING PSD")
    psd_obj = make_psd(DURATION, FS, FLOW)
    print(f"PSD bins = {len(psd_obj)}")
    print()

    print("[2] GENERATING TEMPLATE")
    template = make_waveform(FS, DURATION, MASS1, MASS2, F_LOWER, DISTANCE)
    print(f"Template bins = {len(template)}")
    print(f"delta_f = {template.delta_f:.8f} Hz")
    base_norm = template_norm(template, psd_obj)
    print(f"Base template norm = {base_norm:.6e}")
    print()

    results = []
    total = len(TARGET_SNRS) * args.n_realizations
    counter = 0

    print("[3] RUNNING INJECTION CAMPAIGN")
    print()

    for target_snr in TARGET_SNRS:
        print(f"Target SNR = {target_snr:.1f}")
        for realization in range(args.n_realizations):
            counter += 1
            result = run_single(target_snr, realization, template, psd_obj)
            results.append(result)
            print(
                f"  {counter:3d}/{total:3d} seed={result['seed']} "
                f"recovered={result['recovered_snr']:.3f} "
                f"detected={result['detected']} "
                f"dt={result['timing_error']:+.6f} s"
            )
        print()

    print("=" * 70)
    print("STAGE 6E2 SUMMARY")
    print("=" * 70)

    summary = []
    for target in TARGET_SNRS:
        subset = [r for r in results if r["target_snr"] == target]
        recovered = np.array([r["recovered_snr"] for r in subset])
        timing = np.array([abs(r["timing_error"]) for r in subset])
        detected = np.array([r["detected"] for r in subset])

        mean_snr = float(np.mean(recovered))
        std_snr = float(np.std(recovered, ddof=1) if len(recovered) > 1 else 0.0)
        detection_fraction = float(np.mean(detected))
        mean_abs_timing = float(np.mean(timing))

        row = {
            "target_snr": target,
            "mean_recovered_snr": mean_snr,
            "std_recovered_snr": std_snr,
            "detection_fraction": detection_fraction,
            "mean_abs_timing_error_s": mean_abs_timing,
            "n": len(subset),
        }
        summary.append(row)

        print(
            f"SNR {target:5.1f} | recovered = {mean_snr:8.3f} ± {std_snr:7.3f} | "
            f"detection = {100*detection_fraction:6.1f}% | "
            f"|dt| = {mean_abs_timing:.6e} s"
        )

    json_path = output_dir / "stage6E2_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "configuration": {
                    "fs": FS, "duration": DURATION, "flow": FLOW, "fhigh": FHIGH,
                    "approximant": APPROXIMANT, "mass1": MASS1, "mass2": MASS2,
                    "distance": DISTANCE, "injection_time": INJECTION_TIME,
                    "detection_threshold": DETECTION_THRESHOLD,
                    "n_realizations": args.n_realizations,
                },
                "summary": summary,
                "individual_results": results,
            },
            f, indent=2,
        )
    print()
    print(f"Saved: {json_path}")

    csv_path = output_dir / "stage6E2_results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("target_snr,realization,seed,recovered_snr,expected_time,recovered_time,timing_error,detected\n")
        for r in results:
            f.write(
                f"{r['target_snr']},{r['realization']},{r['seed']},"
                f"{r['recovered_snr']:.12g},{r['expected_time']:.12g},"
                f"{r['recovered_time']:.12g},{r['timing_error']:.12g},{int(r['detected'])}\n"
            )
    print(f"Saved: {csv_path}")

    targets = [x["target_snr"] for x in summary]
    means = [x["mean_recovered_snr"] for x in summary]
    stds = [x["std_recovered_snr"] for x in summary]

    plt.figure(figsize=(8, 6))
    plt.errorbar(targets, means, yerr=stds, fmt="o", capsize=4, label="Recovered SNR")
    plt.plot(targets, targets, "--", label="Ideal y=x")
    plt.xlabel("Injected target SNR")
    plt.ylabel("Recovered SNR")
    plt.title("Stage 6E2 — Injection / Recovery")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot1 = output_dir / "stage6E2_snr_recovery.png"
    plt.savefig(plot1, dpi=200)
    plt.close()
    print(f"Saved: {plot1}")

    fractions = [100.0 * x["detection_fraction"] for x in summary]
    plt.figure(figsize=(8, 6))
    plt.plot(targets, fractions, "o-")
    plt.xlabel("Injected target SNR")
    plt.ylabel("Detection fraction [%]")
    plt.title("Stage 6E2 — Detection Efficiency")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot2 = output_dir / "stage6E2_detection_efficiency.png"
    plt.savefig(plot2, dpi=200)
    plt.close()
    print(f"Saved: {plot2}")

    print()
    print("=" * 70)
    print("STAGE 6E2 COMPLETE")
    print("=" * 70)
    print()
    print("This is an injection/recovery validation.")
    print("No Lambda inference was performed.")
    print("No physical Lambda claim is made.")
    print()


if __name__ == "__main__":
    main()
