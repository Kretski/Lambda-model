"""
stage6E_taylorf2_gr_control.py
================================

STAGE 6E — TAYLORF2 GR-ONLY CONTROL (No LALSuite required)

Purpose
-------
Test whether the matched-filter infrastructure can recover the
real GW150914 merger using a TaylorF2 GR waveform (without Lambda).

This is a critical control experiment that does NOT require LALSuite.

    If Stage 6E PASSES:
        The infrastructure works with TaylorF2, and the Lambda
        discrepancy in Stage 6A/6D needs further investigation.

    If Stage 6E FAILS:
        The infrastructure cannot recover the GR signal from
        real data, meaning the problem is in the data processing,
        PSD estimation, or likelihood normalization.

This experiment does NOT estimate Lambda.
It does NOT claim detection.
It is a pure infrastructure validation.
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from waveform import (
    waveform_frequency_domain,
    cosmological_K_factor,
)
from likelihood import noise_weighted_inner_product


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

# GW150914 parameters (from Stage 6)
M1 = 35.6
M2 = 30.6
DISTANCE_MPC = 440.0
Z = 0.09

F_LOW = 20.0
F_HIGH = 300.0

# Data segment (12 seconds, same as Stage 6D)
HALF_WINDOW = 6.0
TC = 5.99  # GPS center within segment
FS = 4096.0


# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------

def load_real_segment(h1_path, fs, tc, half_window):
    """
    Load real H1 segment using Stage 3 loader.
    """
    
    # Import stage3 locally to avoid circular imports
    try:
        import stage3_real_strain_validation as stage3
    except ImportError as exc:
        print("[ERROR] Could not import stage3_real_strain_validation")
        raise exc
    
    print("=" * 78)
    print("[1] LOADING H1 DATA")
    print("=" * 78)
    print()
    
    print(f"Requested tc       = {tc:.6f} s")
    print(f"Half-window        = {half_window:.6f} s")
    print(f"Total duration     = {2.0 * half_window:.6f} s")
    print()
    
    try:
        result = stage3.load_real_segment(
            h1_path,
            fs=fs,
            tc=tc,
            half_window=half_window
        )
    except TypeError:
        result = stage3.load_real_segment(
            h1_path,
            fs,
            tc,
            half_window
        )
    
    if isinstance(result, tuple):
        data = result[0]
        if len(result) > 1:
            meta = result[1:]
        else:
            meta = ()
    else:
        data = result
        meta = ()
    
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        data = np.ravel(data)
    
    print("Loaded H1 segment successfully.")
    print(f"Samples            = {len(data)}")
    print(f"Expected samples    = {int(round(12.0 * fs))}")
    print(f"Duration [s]        = {len(data) / fs:.9f}")
    print(f"mean                = {np.mean(data):.12e}")
    print(f"std                 = {np.std(data):.12e}")
    print()
    
    return data, meta


def estimate_psd_from_data(data, fs):
    """
    Estimate PSD from real data using Stage 3 estimator or fallback.
    """
    
    try:
        import stage3_real_strain_validation as stage3
        if hasattr(stage3, "estimate_psd_from_segment"):
            print("Using stage3.estimate_psd_from_segment")
            psd_freq, psd = stage3.estimate_psd_from_segment(data, fs)
            return psd_freq, psd
    except Exception as e:
        print(f"[WARNING] Stage3 PSD failed: {e}")
    
    # Fallback: simple periodogram
    print("[WARNING] Using periodogram PSD fallback")
    x = data - np.mean(data)
    n = len(x)
    fft = np.fft.rfft(x)
    psd = np.abs(fft) ** 2 / (fs * n)
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    
    return freq, psd


def interpolate_psd(freq, psd, target_freq):
    """Interpolate PSD to target frequency grid."""
    
    valid = (
        np.isfinite(freq)
        & np.isfinite(psd)
        & (freq >= 0)
        & (psd > 0)
    )
    
    freq = freq[valid]
    psd = psd[valid]
    
    if len(freq) < 2:
        raise RuntimeError("Insufficient PSD data.")
    
    return np.interp(target_freq, freq, psd)


def to_frequency_domain(data, fs):
    """Real FFT with explicit convention."""
    
    n = len(data)
    data_demean = data - np.mean(data)
    fd = np.fft.rfft(data_demean)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    df = fs / n
    
    return freqs, fd, df


def normalized_match(a_fd, b_fd, psd, df):
    """Normalized match between two waveforms."""
    
    aa = noise_weighted_inner_product(a_fd, a_fd, psd, df)
    bb = noise_weighted_inner_product(b_fd, b_fd, psd, df)
    ab = noise_weighted_inner_product(a_fd, b_fd, psd, df)
    
    denom = np.sqrt(max(aa * bb, 0.0))
    
    if denom <= 0:
        return np.nan
    
    return ab / denom


def align_template_to_data(data_fd, template_fd, psd, df, max_shift_seconds=0.1):
    """
    Align template to data by maximizing match over time shifts.
    """
    
    # Rough estimate of sampling rate from df
    n = len(data_fd) * 2 - 1  # Approximate length of time series
    fs = 1.0 / (df * n / 2)  # Rough estimate
    
    max_shift_samples = int(max_shift_seconds * fs)
    if max_shift_samples < 1:
        max_shift_samples = 1
    
    best_match = -np.inf
    best_shift = 0
    
    # Convert to time domain for shifting
    data_td = np.fft.irfft(data_fd)
    template_td = np.fft.irfft(template_fd)
    
    # Pad to same length
    if len(template_td) < len(data_td):
        template_td = np.pad(template_td, (0, len(data_td) - len(template_td)))
    elif len(template_td) > len(data_td):
        template_td = template_td[:len(data_td)]
        data_td = data_td[:len(template_td)]
    
    # Try integer sample shifts
    shifts = np.arange(-max_shift_samples, max_shift_samples + 1, 1)
    
    for shift in shifts:
        if shift >= 0:
            shifted_template = np.roll(template_td, shift)
        else:
            shifted_template = np.roll(template_td, shift)
        
        # Convert back to frequency domain
        shifted_fd = np.fft.rfft(shifted_template)
        
        # Truncate to data length
        if len(shifted_fd) > len(data_fd):
            shifted_fd = shifted_fd[:len(data_fd)]
        elif len(shifted_fd) < len(data_fd):
            shifted_fd = np.pad(shifted_fd, (0, len(data_fd) - len(shifted_fd)))
        
        match = normalized_match(data_fd, shifted_fd, psd, df)
        
        if not np.isnan(match) and match > best_match:
            best_match = match
            best_shift = shift
    
    return best_match, best_shift


# -------------------------------------------------------------------------
# Main experiment
# -------------------------------------------------------------------------

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--h1",
        required=True,
        help="Path to GW150914 H1 HDF5 file"
    )
    
    parser.add_argument(
        "--fs",
        type=float,
        default=FS,
        help="Sampling rate [Hz]"
    )
    
    parser.add_argument(
        "--tc",
        type=float,
        default=TC,
        help="GPS center time [s]"
    )
    
    parser.add_argument(
        "--half_window",
        type=float,
        default=HALF_WINDOW,
        help="Half window duration [s]"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )
    
    args = parser.parse_args()
    
    print("=" * 78)
    print("STAGE 6E — TAYLORF2 GR-ONLY CONTROL")
    print("=" * 78)
    print()
    
    print("Configuration:")
    print(f"  m1, m2:              {M1:.2f}, {M2:.2f} Msun")
    print(f"  Distance:             {DISTANCE_MPC:.1f} Mpc")
    print(f"  Redshift:             {Z:.3f}")
    print(f"  Frequency band:       {F_LOW:.0f}–{F_HIGH:.0f} Hz")
    print(f"  Waveform:             TaylorF2")
    print(f"  Segment duration:     {2.0*args.half_window:.1f} s")
    print(f"  Sampling rate:        {args.fs:.1f} Hz")
    print()
    print("This is a control experiment:")
    print("  - Real H1 data + GR-only template (Lambda=0)")
    print("  - No Lambda estimation")
    print("  - Tests whether the infrastructure can recover GR")
    print("  - TaylorF2 is used (no LALSuite required)")
    print()
    
    # ------------------------------------------------------------------
    # Load real H1 data
    # ------------------------------------------------------------------
    
    data, meta = load_real_segment(
        args.h1,
        args.fs,
        args.tc,
        args.half_window
    )
    
    # ------------------------------------------------------------------
    # Data to frequency domain
    # ------------------------------------------------------------------
    
    print("=" * 78)
    print("[2] DATA FREQUENCY DOMAIN")
    print("=" * 78)
    print()
    
    f_data, data_fd, df = to_frequency_domain(data, args.fs)
    
    print(f"FFT bins:       {len(f_data)}")
    print(f"df:             {df:.6f} Hz")
    print()
    
    # ------------------------------------------------------------------
    # PSD estimation
    # ------------------------------------------------------------------
    
    print("=" * 78)
    print("[3] PSD ESTIMATION")
    print("=" * 78)
    print()
    
    psd_freq, psd_raw = estimate_psd_from_data(data, args.fs)
    psd = interpolate_psd(psd_freq, psd_raw, f_data)
    
    band = (
        (f_data >= F_LOW)
        & (f_data <= F_HIGH)
        & np.isfinite(psd)
        & (psd > 0)
    )
    
    print(f"Band bins:      {np.sum(band)}")
    print(f"PSD median:     {np.median(psd[band]):.12e}")
    print(f"PSD min:        {np.min(psd[band]):.12e}")
    print(f"PSD max:        {np.max(psd[band]):.12e}")
    print()
    
    # ------------------------------------------------------------------
    # Generate GR template (TaylorF2)
    # ------------------------------------------------------------------
    
    print("=" * 78)
    print("[4] GENERATING TAYLORF2 GR TEMPLATE")
    print("=" * 78)
    print()
    
    K_z = cosmological_K_factor(Z)
    
    # Generate template on data frequency grid
    h_fd = np.zeros_like(data_fd, dtype=complex)
    mask = (f_data >= F_LOW) & (f_data <= F_HIGH)
    
    h_fd[mask] = waveform_frequency_domain(
        f_data[mask],
        M1,
        M2,
        0.0,  # Lambda=0 for GR template
        K_z,
        tc=0.0,
        phi_c=0.0,
        distance_Mpc=DISTANCE_MPC,
    )
    
    print(f"Template bins:  {np.sum(mask)}")
    print(f"max |h(f)|:     {np.max(np.abs(h_fd)):.12e}")
    print()
    
    # ------------------------------------------------------------------
    # Inner products
    # ------------------------------------------------------------------
    
    print("=" * 78)
    print("[5] INNER PRODUCTS")
    print("=" * 78)
    print()
    
    dd = noise_weighted_inner_product(data_fd, data_fd, psd, df)
    hh = noise_weighted_inner_product(h_fd, h_fd, psd, df)
    dh = noise_weighted_inner_product(data_fd, h_fd, psd, df)
    
    print(f"<d|d> =          {dd:.12e}")
    print(f"sqrt(<d|d>) =    {np.sqrt(dd):.12e}")
    print(f"<h|h> =          {hh:.12e}")
    print(f"sqrt(<h|h>) =    {np.sqrt(hh):.12e}")
    print(f"<d|h> =          {dh:.12e}")
    print()
    
    # ------------------------------------------------------------------
    # Match and SNR
    # ------------------------------------------------------------------
    
    print("=" * 78)
    print("[6] MATCH AND SNR")
    print("=" * 78)
    print()
    
    match = normalized_match(data_fd, h_fd, psd, df)
    snr = dh / np.sqrt(hh)
    
    print(f"Match:           {match:.8f}")
    print(f"SNR (optimal):   {snr:.4f}")
    print()
    
    # ------------------------------------------------------------------
    # Alignment optimization
    # ------------------------------------------------------------------
    
    print("=" * 78)
    print("[7] TIME ALIGNMENT")
    print("=" * 78)
    print()
    
    best_match, best_shift = align_template_to_data(
        data_fd,
        h_fd,
        psd,
        df,
        max_shift_seconds=0.1
    )
    
    print(f"Best match:      {best_match:.8f}")
    print(f"Best shift:      {best_shift} samples")
    print()
    
    # ------------------------------------------------------------------
    # Frequency band contribution
    # ------------------------------------------------------------------
    
    if args.verbose:
        
        print("=" * 78)
        print("[8] FREQUENCY BAND CONTRIBUTION")
        print("=" * 78)
        print()
        
        weighted_data = np.zeros_like(f_data)
        weighted_template = np.zeros_like(f_data)
        weighted_cross = np.zeros_like(f_data)
        
        valid = band
        
        weighted_data[valid] = (
            4.0 * df * np.abs(data_fd[valid]) ** 2 / psd[valid]
        )
        weighted_template[valid] = (
            4.0 * df * np.abs(h_fd[valid]) ** 2 / psd[valid]
        )
        weighted_cross[valid] = (
            4.0 * df * np.real(np.conjugate(data_fd[valid]) * h_fd[valid]) / psd[valid]
        )
        
        print(f"{'Band':>12} {'data':>14} {'template':>14} {'cross':>14}")
        print("-" * 56)
        
        for lo, hi in [
            (20, 50),
            (50, 100),
            (100, 150),
            (150, 200),
            (200, 250),
            (250, 300),
        ]:
            
            m = valid & (f_data >= lo) & (f_data < hi)
            
            if np.sum(m) > 0:
                print(
                    f"{lo:3d}-{hi:3d} Hz   "
                    f"{np.sum(weighted_data[m]):12.6e}   "
                    f"{np.sum(weighted_template[m]):12.6e}   "
                    f"{np.sum(weighted_cross[m]):12.6e}"
                )
    
    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    
    print("=" * 78)
    print("[9] INTERPRETATION")
    print("=" * 78)
    print()
    
    print("Expected values for GW150914:")
    print("  - Match:       > 0.95 (ideally > 0.98)")
    print("  - SNR:         ~ 20-25")
    print()
    
    if match > 0.95:
        print("✅ STAGE 6E PASSES:")
        print("   The GR-only template matches the real data well.")
        print("   The infrastructure works, and the Lambda effect")
        print("   (if any) is real and deserves further investigation.")
    elif match > 0.80:
        print("⚠️ STAGE 6E PARTIAL:")
        print("   The match is moderate but not optimal.")
        print("   There may be issues with:")
        print("     - PSD estimation")
        print("     - Template parameters (masses, distance)")
        print("     - Data normalization")
        print("     - Time alignment")
        print("   Or TaylorF2 may not be sufficient for real data.")
    else:
        print("❌ STAGE 6E FAILS:")
        print("   The GR-only template does NOT match the data.")
        print("   The matched-filter infrastructure cannot recover")
        print("   the GW150914 signal with this setup.")
        print()
        print("   Possible causes:")
        print("     1. Incorrect PSD normalization")
        print("     2. Incorrect template parameters")
        print("     3. TaylorF2 is insufficient for real data")
        print("     4. Fundamental problem with the likelihood")
        print("     5. Data preprocessing issues")
    
    print()
    print("=" * 78)
    print("STAGE 6E COMPLETE")
    print("=" * 78)
    print()
    print("This is a control experiment only.")
    print("No Lambda inference is performed.")
    print("No non-zero Lambda claim is made.")


if __name__ == "__main__":
    main()