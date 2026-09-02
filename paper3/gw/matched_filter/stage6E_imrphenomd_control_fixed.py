#!/usr/bin/env python3
"""
PAPER 3 — STAGE 6E (IMRPhenomD) — FIXED VERSION
================================================

IMRPhenomD GR-only real-event control with GPS-correct segment extraction.

FIXES:
    A — Frequency-grid correction:
        Properly extract IMRPhenomD bins from 20–300 Hz instead of taking
        the first bins from 0 Hz.

    B — Time/phase-maximized matched filter:
        Instead of concluding failure from a single fixed phase/time,
        find the maximum complex overlap across the 12-second segment.

IMPORTANT:
    This is a GR-only control.
    Lambda is fixed to exactly zero.
    No Lambda estimation is performed.

This requires LALSuite (lal, lalsimulation).
"""

import argparse
import importlib.util
from pathlib import Path

import h5py
import numpy as np

try:
    import lal
    import lalsimulation as lalsim
except ImportError as exc:
    raise ImportError(
        "\nLALSimulation is required for Stage 6E (IMRPhenomD).\n"
        "Install with: conda install -c conda-forge lalsuite\n"
    ) from exc


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_FS = 4096.0

M1 = 35.60
M2 = 30.60
DISTANCE_MPC = 440.0
REDSHIFT = 0.09

F_LOW = 20.0
F_HIGH = 300.0

SEGMENT_DURATION = 12.0
HALF_WINDOW = SEGMENT_DURATION / 2.0

# GW150914 merger GPS reference used by previous stages.
GPS_MERGER = 1126259462.4

LAMBDA = 0.0

# LAL approximant: IMRPhenomD
APPROX_NAME = "IMRPhenomD"


# ============================================================
# HDF5 INSPECTION
# ============================================================

def inspect_hdf5(filename):
    print("=" * 78)
    print("[1] HDF5 INSPECTION")
    print("=" * 78)

    with h5py.File(filename, "r") as f:
        print("HDF5 keys:")

        for key in f.keys():
            obj = f[key]
            print(f"  {key}: shape={obj.shape} dtype={obj.dtype}")

            if hasattr(obj, "attrs"):
                if "x0" in obj.attrs:
                    print(f"    x0 = {obj.attrs['x0']}")

                if "dx" in obj.attrs:
                    print(f"    dx = {obj.attrs['dx']}")

                if "Xstart" in obj.attrs:
                    print(f"    Xstart = {obj.attrs['Xstart']}")

                if "Xspacing" in obj.attrs:
                    print(f"    Xspacing = {obj.attrs['Xspacing']}")


# ============================================================
# FIND STRAIN DATASET
# ============================================================

def find_strain_dataset(h5):
    if "Strain" in h5:
        return h5["Strain"]

    candidates = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            if obj.ndim == 1:
                candidates.append(obj)

    h5.visititems(visitor)

    if not candidates:
        raise RuntimeError("No 1D strain dataset found.")

    return candidates[0]


# ============================================================
# LOAD GPS-CORRECT LOCAL SEGMENT
# ============================================================

def load_h1_segment(filename, fs, tc_local):
    print("=" * 78)
    print("[2] GPS-CORRECT H1 SEGMENT")
    print("=" * 78)

    expected_n = int(round(SEGMENT_DURATION * fs))

    with h5py.File(filename, "r") as f:

        ds = find_strain_dataset(f)
        n_total = len(ds)

        if "x0" in ds.attrs:
            gps_start = float(ds.attrs["x0"])
        elif "Xstart" in ds.attrs:
            gps_start = float(ds.attrs["Xstart"])
        else:
            raise RuntimeError("No GPS start attribute found.")

        if "dx" in ds.attrs:
            dx = float(ds.attrs["dx"])
            fs_file = 1.0 / dx
        elif "Xspacing" in ds.attrs:
            dx = float(ds.attrs["Xspacing"])
            fs_file = 1.0 / dx
        else:
            fs_file = fs

        if abs(fs_file - fs) > 1e-6:
            print(f"WARNING: requested fs={fs}, file fs={fs_file}")

        gps_tc = GPS_MERGER + (tc_local - HALF_WINDOW)
        gps_seg_start = GPS_MERGER - HALF_WINDOW
        gps_seg_end = GPS_MERGER + HALF_WINDOW

        print(f"Dataset:              {ds.name}")
        print(f"Samples:               {n_total}")
        print(f"File fs:               {fs_file:.9f} Hz")
        print(f"GPS start:             {gps_start:.6f}")
        print(f"GPS merger reference:  {GPS_MERGER:.6f}")
        print(f"Local tc:               {tc_local:.6f} s")
        print(f"GPS tc:                {gps_tc:.6f}")
        print(f"Segment GPS start:     {gps_seg_start:.6f}")
        print(f"Segment GPS end:       {gps_seg_end:.6f}")

        idx_center = int(round((gps_tc - gps_start) * fs_file))
        idx_lo = int(round((gps_seg_start - gps_start) * fs_file))
        idx_hi = idx_lo + expected_n

        print()
        print("Calculated indices:")
        print(f"  center = {idx_center}")
        print(f"  lo     = {idx_lo}")
        print(f"  hi     = {idx_hi}")
        print(f"  n      = {expected_n}")

        if idx_lo < 0 or idx_hi > n_total:
            raise ValueError(
                f"\nRequested GPS segment outside dataset.\n"
                f"idx_lo={idx_lo}, idx_hi={idx_hi}, n_total={n_total}"
            )

        data = np.asarray(ds[idx_lo:idx_hi], dtype=float)

    if len(data) != expected_n:
        raise RuntimeError(f"Expected {expected_n} samples, got {len(data)}")

    print()
    print("Segment loaded successfully.")
    print(f"  samples = {len(data)}")
    print(f"  duration = {len(data) / fs:.9f} s")
    print(f"  mean = {np.mean(data):.12e}")
    print(f"  std  = {np.std(data):.12e}")

    return data


# ============================================================
# FFT
# ============================================================

def make_frequency_domain(data, fs):
    n = len(data)
    x = data - np.mean(data)
    window = np.hanning(n)
    xw = x * window
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fd = np.fft.rfft(xw)
    return freqs, fd


# ============================================================
# PSD
# ============================================================

def estimate_psd(data, fs):
    """
    Simple periodogram PSD (no scipy required).
    """
    try:
        from scipy.signal import welch
        f, pxx = welch(
            data,
            fs=fs,
            window="hann",
            nperseg=min(4096, len(data)),
            noverlap=min(2048, len(data) // 2),
            detrend="constant",
            scaling="density"
        )
        return f, pxx
    except Exception as exc:
        print(f"[WARNING] scipy.signal.welch failed: {exc}")
        print("[WARNING] Using simple periodogram fallback")
        
        # Simple periodogram
        x = data - np.mean(data)
        n = len(x)
        fft = np.fft.rfft(x)
        psd = np.abs(fft) ** 2 / (fs * n)
        freq = np.fft.rfftfreq(n, d=1.0 / fs)
        
        return freq, psd


def interpolate_psd(psd_f, psd, target_f):
    safe = np.maximum(psd, 1e-60)
    return np.interp(target_f, psd_f, safe, left=safe[0], right=safe[-1])


# ============================================================
# INNER PRODUCT
# ============================================================

def inner_product(a, b, psd, df):
    return 4.0 * np.real(np.sum(np.conjugate(a) * b / psd) * df)


def complex_inner_product(a, b, psd, df):
    """
    Complex-valued inner product (without taking real part).
    Used for time/phase maximization.
    """
    return 4.0 * np.sum(np.conjugate(a) * b / psd) * df


# ============================================================
# IMRPHENOMD TEMPLATE - Using SimInspiralChooseFDWaveform
# ============================================================

def generate_imrphenomd_template(f_band, m1, m2, distance_mpc):
    """
    Generate IMRPhenomD frequency-domain template using SimInspiralChooseFDWaveform.
    
    FIX A: Proper frequency-grid extraction.
           LAL array starts at 0 Hz, we need to extract the correct bins
           for the [20, 300] Hz band.
    
    Returns:
        h_plus_fd: complex array of plus polarization
        h_cross_fd: complex array of cross polarization
    """
    
    # Convert to SI
    m1_si = m1 * lal.MSUN_SI
    m2_si = m2 * lal.MSUN_SI
    distance_si = distance_mpc * 1.0e6 * lal.PC_SI
    
    # Nonspinning
    spin1x = spin1y = spin1z = 0.0
    spin2x = spin2y = spin2z = 0.0
    
    inclination = 0.0
    phi_ref = 0.0
    long_asc_nodes = 0.0
    eccentricity = 0.0
    mean_per_ano = 0.0
    
    # Frequency grid parameters
    f_min = f_band[0]
    f_max = f_band[-1]
    delta_f = f_band[1] - f_band[0]
    f_ref = 0.0  # Reference frequency
    
    # Create params
    params = lal.CreateDict()
    
    # Use SimInspiralChooseFDWaveform - the official LALSimulation API
    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        m1_si,          # mass1
        m2_si,          # mass2
        spin1x,         # spin1x
        spin1y,         # spin1y
        spin1z,         # spin1z
        spin2x,         # spin2x
        spin2y,         # spin2y
        spin2z,         # spin2z
        distance_si,    # distance
        inclination,    # inclination
        phi_ref,        # phi_ref
        long_asc_nodes, # long_asc_nodes
        eccentricity,   # eccentricity
        mean_per_ano,   # mean_per_ano
        delta_f,        # delta_f
        f_min,          # f_min
        f_max,          # f_max
        f_ref,          # f_ref
        params,         # params
        lalsim.IMRPhenomD,  # approximant
    )
    
    hp_data = np.asarray(hp.data.data, dtype=complex)
    hc_data = np.asarray(hc.data.data, dtype=complex)
    
    # --------------------------------------------------------
    # FIX A: Proper frequency-grid extraction
    # LAL frequency series starts at f = 0 Hz.
    # Select the actual requested [20, 300] Hz bins.
    # --------------------------------------------------------
    
    i0 = int(round(f_min / delta_f))
    i1 = i0 + len(f_band)
    
    if i1 > len(hp_data):
        raise RuntimeError(
            f"IMRPhenomD FD array too short: "
            f"need [{i0}:{i1}], have {len(hp_data)} bins"
        )
    
    hp_data = hp_data[i0:i1]
    hc_data = hc_data[i0:i1]
    
    # Explicit frequency-grid validation.
    f_selected = np.arange(i0, i1) * delta_f
    
    if not np.allclose(f_selected, f_band, rtol=0.0, atol=1e-10):
        raise RuntimeError(
            "IMRPhenomD frequency grid does not match data grid"
        )
    
    return hp_data.astype(complex), hc_data.astype(complex)


# ============================================================
# TIME/PHASE MAXIMIZATION
# ============================================================

def maximize_complex_overlap(d_band, template, psd_band, df, fs, max_shift_seconds=0.5):
    """
    FIX B: Time/phase-maximized matched filter.
    
    Find the maximum complex overlap between data and template
    across time shifts and phase.
    
    Returns:
        max_match: maximum match value
        best_shift_samples: best time shift in samples
        best_phase: best phase in radians
        max_snr: maximum SNR
    """
    
    # Calculate the complex inner product (without taking real part)
    z = complex_inner_product(d_band, template, psd_band, df)
    
    # Phase maximization is trivial: |z| gives the phase-maximized value
    # The phase that maximizes Re(e^{-iφ} * z) is φ = arg(z)
    max_abs_z = np.abs(z)
    best_phase = np.angle(z)
    
    # For time shifts, we need to work in time domain
    # Convert to time domain
    n_data = len(d_band) * 2 - 1  # Approximate length of time series
    fs_actual = 1.0 / (df * n_data / 2)  # Rough estimate
    
    # Pad template to same length as data
    template_td = np.fft.irfft(template, n=n_data)
    data_td = np.fft.irfft(d_band, n=n_data)
    
    # Compute cross-correlation
    cross_corr = np.correlate(data_td, template_td, mode='same')
    
    # Find the shift that maximizes |cross_corr|
    best_idx = np.argmax(np.abs(cross_corr))
    best_shift = best_idx - len(data_td) // 2
    
    # Maximum match from time shift
    max_match_time = np.max(np.abs(cross_corr)) / (np.sqrt(dd) * np.sqrt(hh))
    
    # Combined maximization (time + phase)
    # For the best time shift, compute the complex inner product
    shifted_template = np.roll(template_td, best_shift)
    shifted_template_fd = np.fft.rfft(shifted_template)[:len(d_band)]
    
    z_shifted = complex_inner_product(d_band, shifted_template_fd, psd_band, df)
    max_match_combined = np.abs(z_shifted) / (np.sqrt(dd) * np.sqrt(hh))
    best_phase_combined = np.angle(z_shifted)
    
    return {
        'max_abs_z': max_abs_z,
        'best_phase': best_phase,
        'best_shift_samples': best_shift,
        'max_match_time': max_match_time,
        'max_match_combined': max_match_combined,
        'best_phase_combined': best_phase_combined,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h1", required=True, help="Path to H1 HDF5 strain file")
    parser.add_argument("--fs", type=float, default=DEFAULT_FS)
    parser.add_argument("--tc", type=float, default=6.0, help="LOCAL tc inside 12-second segment")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 78)
    print("STAGE 6E — IMRPHENOMD GR-ONLY REAL-EVENT CONTROL (FIXED)")
    print("=" * 78)
    print()
    print("FIXES APPLIED:")
    print("  A — Frequency-grid correction (proper 20-300 Hz extraction)")
    print("  B — Time/phase-maximized matched filter")
    print()
    print("Configuration:")
    print(f"  m1, m2       = {M1:.2f}, {M2:.2f} Msun")
    print(f"  Distance     = {DISTANCE_MPC:.1f} Mpc")
    print(f"  Redshift     = {REDSHIFT:.3f}")
    print(f"  Band         = {F_LOW:.0f}–{F_HIGH:.0f} Hz")
    print(f"  fs           = {args.fs:.1f} Hz")
    print(f"  tc LOCAL     = {args.tc:.6f} s")
    print(f"  GPS merger   = {GPS_MERGER:.6f} s")
    print(f"  Approximant  = {APPROX_NAME}")
    print()
    print("Lambda = 0 EXACTLY")
    print("No Lambda estimation is performed.")
    print()

    # --------------------------------------------------------
    # 1. Inspect
    # --------------------------------------------------------
    inspect_hdf5(args.h1)

    # --------------------------------------------------------
    # 2. Load GPS-correct segment
    # --------------------------------------------------------
    data = load_h1_segment(args.h1, args.fs, args.tc)

    # --------------------------------------------------------
    # 3. FFT
    # --------------------------------------------------------
    print()
    print("=" * 78)
    print("[3] FREQUENCY DOMAIN")
    print("=" * 78)

    freqs, data_fd = make_frequency_domain(data, args.fs)
    df = freqs[1] - freqs[0]

    print(f"FFT bins = {len(freqs)}")
    print(f"df       = {df:.8f} Hz")

    # --------------------------------------------------------
    # 4. PSD
    # --------------------------------------------------------
    print()
    print("=" * 78)
    print("[4] PSD")
    print("=" * 78)

    psd_f, psd = estimate_psd(data, args.fs)

    mask = (freqs >= F_LOW) & (freqs <= F_HIGH)
    f_band = freqs[mask]

    psd_band = interpolate_psd(psd_f, psd, f_band)

    print(f"Band bins  = {len(f_band)}")
    print(f"PSD median = {np.median(psd_band):.6e}")
    print(f"PSD min    = {np.min(psd_band):.6e}")
    print(f"PSD max    = {np.max(psd_band):.6e}")

    # --------------------------------------------------------
    # 5. IMRPHENOMD TEMPLATE
    # --------------------------------------------------------
    print()
    print("=" * 78)
    print("[5] IMRPHENOMD GR TEMPLATE")
    print("=" * 78)

    hp_fd, hc_fd = generate_imrphenomd_template(
        f_band,
        M1,
        M2,
        DISTANCE_MPC
    )

    # Use plus polarization only (for face-on aligned system)
    template = hp_fd

    if len(template) != len(f_band):
        raise RuntimeError(
            f"Template length mismatch: {len(template)} vs {len(f_band)}"
        )

    print(f"Template bins = {len(template)}")
    print(f"f min         = {f_band[0]:.6f} Hz")
    print(f"f max         = {f_band[-1]:.6f} Hz")
    print(f"max |h(f)|    = {np.max(np.abs(template)):.6e}")

    # --------------------------------------------------------
    # 6. Fixed-phase inner products
    # --------------------------------------------------------

    d_band = data_fd[mask]

    hh = inner_product(template, template, psd_band, df)
    dd = inner_product(d_band, d_band, psd_band, df)
    dh_fixed = inner_product(d_band, template, psd_band, df)

    denom = np.sqrt(max(dd, 0.0) * max(hh, 0.0))
    match_fixed = dh_fixed / denom if denom > 0 else np.nan
    snr_fixed = dh_fixed / np.sqrt(max(hh, 1e-300))

    print()
    print("=" * 78)
    print("[6] FIXED-PHASE GR-ONLY CONTROL")
    print("=" * 78)

    print(f"<d|d> = {dd:.8e}")
    print(f"<h|h> = {hh:.8e}")
    print(f"<d|h> = {dh_fixed:.8e}")
    print()
    print(f"sqrt(<d|d>) = {np.sqrt(max(dd, 0.0)):.8f}")
    print(f"sqrt(<h|h>) = {np.sqrt(max(hh, 0.0)):.8f}")
    print(f"match (fixed phase) = {match_fixed:.8f}")
    print(f"SNR (fixed phase)   = {snr_fixed:.4f}")

    # --------------------------------------------------------
    # 7. Time/phase-maximized control (FIX B)
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("[7] TIME/PHASE-MAXIMIZED GR CONTROL")
    print("=" * 78)

    # Compute complex inner product
    z = complex_inner_product(d_band, template, psd_band, df)
    max_abs_z = np.abs(z)
    best_phase = np.angle(z)
    match_phase_max = max_abs_z / denom if denom > 0 else np.nan
    snr_phase_max = max_abs_z / np.sqrt(max(hh, 1e-300))

    print(f"Complex <d|h> = {z:.8e}")
    print(f"|z|            = {max_abs_z:.8e}")
    print(f"best phase     = {best_phase:.6f} rad")
    print(f"match (phase max) = {match_phase_max:.8f}")
    print(f"SNR (phase max)   = {snr_phase_max:.4f}")

    # Time shift maximization (simplified)
    # Convert to time domain for cross-correlation
    n_fft = len(d_band) * 2 - 1
    template_td = np.fft.irfft(template, n=n_fft)
    data_td = np.fft.irfft(d_band, n=n_fft)
    
    # Cross-correlation
    cross_corr = np.correlate(data_td, template_td, mode='same')
    max_cross = np.max(np.abs(cross_corr))
    best_idx = np.argmax(np.abs(cross_corr))
    best_shift = best_idx - len(data_td) // 2
    
    match_time_max = max_cross / (np.sqrt(dd) * np.sqrt(hh))
    
    print()
    print(f"best time shift  = {best_shift} samples")
    print(f"match (time max) = {match_time_max:.8f}")

    # Combined time + phase maximization
    shifted_template = np.roll(template_td, best_shift)
    shifted_template_fd = np.fft.rfft(shifted_template)[:len(d_band)]
    
    z_shifted = complex_inner_product(d_band, shifted_template_fd, psd_band, df)
    match_combined = np.abs(z_shifted) / denom if denom > 0 else np.nan
    best_phase_combined = np.angle(z_shifted)
    snr_combined = np.abs(z_shifted) / np.sqrt(max(hh, 1e-300))

    print()
    print(f"combined |z|      = {np.abs(z_shifted):.8e}")
    print(f"best phase        = {best_phase_combined:.6f} rad")
    print(f"match (combined)  = {match_combined:.8f}")
    print(f"SNR (combined)    = {snr_combined:.4f}")

    # --------------------------------------------------------
    # 8. Interpretation
    # --------------------------------------------------------
    print()
    print("=" * 78)
    print("[8] INTERPRETATION")
    print("=" * 78)

    print()
    print("Summary of matches:")
    print(f"  Fixed phase:        {match_fixed:.8f}")
    print(f"  Phase-maximized:    {match_phase_max:.8f}")
    print(f"  Time-maximized:     {match_time_max:.8f}")
    print(f"  Combined (time+phase): {match_combined:.8f}")
    print()
    print("Expected values for GW150914:")
    print("  - Match:       > 0.95 (ideally > 0.98)")
    print("  - SNR:         ~ 20-25")
    print()

    if np.isfinite(match_combined):
        if match_combined > 0.95:
            print("✅ STAGE 6E PASSES:")
            print("   IMRPhenomD matches the real data well.")
            print("   The infrastructure works with realistic waveforms.")
            print("   The Lambda effect (if any) deserves investigation.")
        elif match_combined > 0.80:
            print("⚠️ STAGE 6E PARTIAL:")
            print("   The match is moderate but not optimal.")
            print("   May need parameter tuning or better waveform.")
        else:
            print("❌ STAGE 6E FAILS:")
            print("   IMRPhenomD does NOT match the data.")
            print("   Even with time/phase maximization, the match is low.")
            print("   Possible causes:")
            print("     1. Incorrect masses (M1, M2)")
            print("     2. Incorrect distance")
            print("     3. The signal may require NR waveforms")
            print("     4. PSD or data preprocessing issues")
    else:
        print("❌ STAGE 6E ERROR: Could not compute match.")

    print()
    print("=" * 78)
    print("STAGE 6E COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()