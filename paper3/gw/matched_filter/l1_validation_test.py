#!/usr/bin/env python3
"""
sterile_lambda_analysis.py
===========================

Corrected replacement for the previous "UNBIASED LIGO STERILE ANALYSIS"
script.

WHAT WAS WRONG WITH THE PREVIOUS VERSION (for the record, see chat):

  1. The "H1 off-source noise" and "H1 event signal" were both built with
     np.random.normal(...) PLUS a hand-injected lambda_val=20.8 term. The
     extractor then "recovered" +20.8 -- which is circular: you inject the
     number, then measure the number you injected. No real strain data was
     read anywhere in that script.
  2. The waveform model used sin(4*pi*t) phase modulation -- the toy model
     already shown (earlier in this session) to be structurally
     non-identifiable at |Lambda| above ~0.3 rad of modulation depth.
  3. The "sign-flip test" flipped s(t) -> -s(t), which is a global pi phase
     shift (sin(x) -> sin(x+pi)), NOT equivalent to Lambda -> -Lambda in
     this phase-modulation model. Its own 0.90 threshold correctly returned
     "NO / random noise" even on the circular data -- the test's own logic
     already contradicted the headline claim.

WHAT THIS SCRIPT DOES INSTEAD:

  1. Loads REAL GWOSC/LOSC H1 (and optionally L1) strain via HDF5 -- same
     loader convention as stage6E3H-R.py.
  2. Uses the PHYSICAL Lambda dispersion phase model from waveform.py
     (Delta_Psi(f) = -(4*pi^3*Lambda*K(z)/c^3) * f^3), not a toy sinusoid.
  3. Runs a template-bank matched-filter search for the best-fit Lambda on
     a set of OFF-SOURCE (timeslide) segments -- background distribution,
     with NO injected value anywhere.
  4. Runs the SAME search on the actual ON-SOURCE segment around the real
     event GPS time -- this is the actual measurement.
  5. Reports the on-source value's z-score / percentile against the
     background distribution -- a real null test, replacing the broken
     sign-flip test. This is a diagnostic calibration test, not a claim of
     detection.

Usage:
    python sterile_lambda_analysis.py --h1-data /path/to/H-H1_GWOSC_....hdf5 \
        [--l1-data /path/to/L-L1_GWOSC_....hdf5] \
        [--n-offsource 100] [--lambda-min -30] [--lambda-max 30] [--lambda-step 0.1]
"""

import os
import sys
import csv
import argparse
import numpy as np
import h5py

from pycbc.types import TimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.psd import welch, interpolate
from pycbc.filter import matched_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from waveform import lambda_phase_correction, cosmological_K_factor


# ============================================================
# CONFIG
# ============================================================

EVENT_GPS = 1126259462.0          # GW150914
SAMPLE_RATE = 4096.0
DURATION = 16.0

F_LOW = 20.0
F_HIGH = 300.0

REDSHIFT = 0.09
K_Z = cosmological_K_factor(REDSHIFT)

OFFSOURCE_SPACING = 32.0
OUTPUT_DIR = "sterile_results"


# ============================================================
# HDF5 LOADER  (same convention as stage6E3H-R.py)
# ============================================================

def read_losc_hdf5(path):
    with h5py.File(path, "r") as f:
        if "strain/Strain" not in f:
            raise KeyError("Expected strain/Strain dataset in HDF5 file.")
        data = np.asarray(f["strain/Strain"][:], dtype=np.float64)

        if "meta/GPSstart" not in f:
            raise RuntimeError("HDF5 file has no meta/GPSstart.")
        gps_start = float(np.asarray(f["meta/GPSstart"]))

        duration = None
        if "meta/Duration" in f:
            duration = float(np.asarray(f["meta/Duration"]))

        detector = "UNKNOWN"
        if "meta/Detector" in f:
            try:
                detector = str(np.asarray(f["meta/Detector"]).astype(str))
            except Exception:
                detector = "UNKNOWN"

    n = len(data)
    fs = n / duration if (duration is not None and duration > 0) else SAMPLE_RATE
    delta_t = 1.0 / fs

    ts = TimeSeries(data, delta_t=delta_t, epoch=gps_start)

    print(f"  Detector    = {detector}")
    print(f"  GPS start   = {gps_start:.3f}")
    print(f"  Samples     = {n}")
    print(f"  Sample rate = {fs:.3f} Hz")
    print(f"  Duration    = {n * delta_t:.3f} s")

    return ts, detector


# ============================================================
# GR TEMPLATE  (leading-order TaylorF2 via pycbc IMRPhenomD for a
# realistic time-domain reference; Lambda phase applied in the
# frequency domain via apply_lambda_phase, matching waveform.py)
# ============================================================

def generate_gr_template():
    hp, _ = get_td_waveform(
        approximant="IMRPhenomD",
        mass1=36.0,
        mass2=29.0,
        delta_t=1.0 / SAMPLE_RATE,
        f_lower=F_LOW,
    )
    template = hp.copy()
    target_n = int(DURATION * SAMPLE_RATE)
    arr = np.asarray(template)

    if len(arr) > target_n:
        arr = arr[-target_n:]
        template = TimeSeries(arr, delta_t=1.0 / SAMPLE_RATE)
    elif len(arr) < target_n:
        padded = np.zeros(target_n, dtype=np.float64)
        padded[-len(arr):] = arr
        template = TimeSeries(padded, delta_t=1.0 / SAMPLE_RATE)
    else:
        template = TimeSeries(arr, delta_t=1.0 / SAMPLE_RATE)

    template = template - np.mean(template)
    return template


def apply_lambda_phase(template, lam):
    n = len(template)
    dt = float(template.delta_t)
    freq = np.fft.rfftfreq(n, d=dt)
    hf = np.fft.rfft(np.asarray(template))
    phase = lambda_phase_correction(freq, lam, K_Z)
    deformed_hf = hf * np.exp(1j * phase)
    out = np.fft.irfft(deformed_hf, n=n)
    return TimeSeries(out, delta_t=dt, epoch=template.start_time)


# ============================================================
# PSD
# ============================================================

def make_psd(data):
    seg_len = int(4.0 * SAMPLE_RATE)
    psd = welch(data, seg_len=seg_len, seg_stride=seg_len // 2)
    target_delta_f = 1.0 / DURATION
    if abs(psd.delta_f - target_delta_f) > 1e-12:
        psd = interpolate(psd, target_delta_f)
    return psd


# ============================================================
# TEMPLATE BANK MATCHED-FILTER RECOVERY (NO INJECTION -- this
# runs directly on whatever segment it is given, real or
# off-source, and returns the best-fit Lambda for THAT segment)
# ============================================================

def build_template_bank(gr_template, lambda_grid):
    templates = []
    for lam in lambda_grid:
        templates.append((float(lam), apply_lambda_phase(gr_template, float(lam))))
    return templates


def recover_lambda(data, templates, psd):
    best_lambda = None
    best_score = -np.inf
    for lam, template in templates:
        try:
            snr = matched_filter(
                template, data, psd=psd,
                low_frequency_cutoff=F_LOW, high_frequency_cutoff=F_HIGH,
            )
            score = float(np.max(np.abs(np.asarray(snr))))
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_lambda = lam
        except Exception:
            continue
    return best_lambda, best_score


# ============================================================
# SEGMENT EXTRACTION
# ============================================================

def extract_segment(full_data, gps):
    dt = float(full_data.delta_t)
    start = int(round((gps - float(full_data.start_time)) / dt))
    n = int(DURATION / dt)
    end = start + n
    if start < 0 or end > len(full_data):
        raise ValueError("Requested segment outside data.")
    segment = full_data[start:end].copy()
    segment = segment - np.mean(segment)
    return segment


def build_offsource_times(gps_start, duration, event_offset, n):
    """Timeslide off-source times, symmetric around the event, avoiding it."""
    event_time = gps_start + event_offset
    times = []
    k = 1
    while len(times) < n:
        t = event_time - k * OFFSOURCE_SPACING
        if t < gps_start + DURATION:
            break
        if t + DURATION > gps_start + duration:
            k += 1
            continue
        times.append(t)
        k += 1
    k = 1
    while len(times) < n:
        t = event_time + k * OFFSOURCE_SPACING
        if t + DURATION > gps_start + duration:
            break
        if t < gps_start + DURATION:
            k += 1
            continue
        times.append(t)
        k += 1
    return times[:n]


# ============================================================
# ANALYSIS FOR ONE DETECTOR
# ============================================================

def analyze_detector(name, data_path, gr_template, lambda_grid, n_offsource):
    print()
    print("=" * 80)
    print(f"[{name}] LOADING REAL DATA")
    print("=" * 80)
    full_data, detector = read_losc_hdf5(data_path)
    full_data = full_data.astype(np.float64)

    gps_start = float(full_data.start_time)
    duration = len(full_data) * float(full_data.delta_t)
    event_offset = EVENT_GPS - gps_start

    print()
    print(f"[{name}] BUILDING TEMPLATE BANK ({len(lambda_grid)} templates)")
    templates = build_template_bank(gr_template, lambda_grid)

    # --- Off-source background (NO injection anywhere) ---
    print()
    print(f"[{name}] OFF-SOURCE BACKGROUND ({n_offsource} timeslides, no injection)")
    offsource_gps = build_offsource_times(gps_start, duration, event_offset, n_offsource)
    if len(offsource_gps) < n_offsource:
        print(f"  WARNING: only {len(offsource_gps)} usable off-source segments found "
              f"(requested {n_offsource})")

    background_lambdas = []
    for i, gps in enumerate(offsource_gps):
        segment = extract_segment(full_data, gps)
        psd = make_psd(segment)
        lam, score = recover_lambda(segment, templates, psd)
        if lam is not None:
            background_lambdas.append(lam)
        print(f"  offsource {i+1:3d}/{len(offsource_gps)}  lambda_null={lam}  score={score:.3f}")

    background_lambdas = np.array(background_lambdas, dtype=float)
    bg_mean = float(np.mean(background_lambdas)) if len(background_lambdas) else float("nan")
    bg_std = float(np.std(background_lambdas)) if len(background_lambdas) else float("nan")

    # --- On-source: the actual real event segment, NOT injected ---
    print()
    print(f"[{name}] ON-SOURCE MEASUREMENT (real strain, GPS={EVENT_GPS})")
    event_segment = extract_segment(full_data, EVENT_GPS - DURATION / 2.0)
    event_psd = make_psd(event_segment)
    on_source_lambda, on_source_score = recover_lambda(event_segment, templates, event_psd)

    z_score = ((on_source_lambda - bg_mean) / bg_std
               if (on_source_lambda is not None and bg_std > 0) else float("nan"))

    print(f"  Recovered on-source Lambda = {on_source_lambda}")
    print(f"  Background mean / std      = {bg_mean:+.4f} / {bg_std:.4f}")
    print(f"  z-score vs background      = {z_score:+.4f}")

    return {
        "detector": name,
        "n_offsource": len(background_lambdas),
        "background_mean": bg_mean,
        "background_std": bg_std,
        "on_source_lambda": on_source_lambda,
        "on_source_score": on_source_score,
        "z_score": z_score,
        "background_samples": background_lambdas.tolist(),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Sterile Lambda analysis (real data, no injection)")
    parser.add_argument("--h1-data", required=True, help="Path to H1 GWOSC HDF5 file")
    parser.add_argument("--l1-data", default=None, help="Optional path to L1 GWOSC HDF5 file")
    parser.add_argument("--n-offsource", type=int, default=100)
    parser.add_argument("--lambda-min", type=float, default=-30.0)
    parser.add_argument("--lambda-max", type=float, default=30.0)
    parser.add_argument("--lambda-step", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    lambda_grid = np.arange(args.lambda_min, args.lambda_max + 0.5 * args.lambda_step,
                             args.lambda_step)

    print("=" * 80)
    print("STERILE LAMBDA ANALYSIS -- REAL DATA, NO INJECTION")
    print("=" * 80)
    print(f"Analysis band   = {F_LOW:.1f}-{F_HIGH:.1f} Hz")
    print(f"Lambda grid     = [{args.lambda_min}, {args.lambda_max}] step={args.lambda_step} "
          f"({len(lambda_grid)} templates)")
    print(f"K(z={REDSHIFT}) = {K_Z:.4e} s")

    gr_template = generate_gr_template()

    results = [analyze_detector("H1", args.h1_data, gr_template, lambda_grid, args.n_offsource)]
    if args.l1_data:
        results.append(analyze_detector("L1", args.l1_data, gr_template, lambda_grid, args.n_offsource))

    summary_path = os.path.join(OUTPUT_DIR, "sterile_analysis_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["detector", "n_offsource", "background_mean", "background_std",
                      "on_source_lambda", "on_source_score", "z_score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})

    print()
    print("=" * 80)
    print(f"SUMMARY WRITTEN TO: {summary_path}")
    print("=" * 80)
    print()
    print("IMPORTANT: no Lambda value is injected anywhere in this script. Both the")
    print("background distribution and the on-source value come from real strain.")
    print("A large |z_score| here would be the first thing worth taking seriously --")
    print("but it still would not resolve the separate open question of whether this")
    print("Lambda is the same physical quantity/units as the Fermi-LAT-bounded Lambda")
    print("elsewhere in the project (Lambda < 1.4421e-53 m^2).")


if __name__ == "__main__":
    main()