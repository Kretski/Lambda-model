#!/usr/bin/env python3
"""
PAPER 3 — STAGE 6E
TaylorF2 GR-only real-event control with GPS-correct segment extraction.

IMPORTANT:
This is a GR-only control.
Lambda is fixed to exactly zero.
No Lambda estimation is performed.

The HDF5 file contains a long GPS-time series. Therefore tc is interpreted
as a LOCAL time inside the extracted 12-second segment, while the segment
itself is anchored to the known GW150914 GPS merger time.
"""

import argparse
import importlib.util
from pathlib import Path

import h5py
import numpy as np


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


# ============================================================
# IMPORT LOCAL WAVEFORM MODULE
# ============================================================

def load_waveform_module():
    here = Path(__file__).resolve().parent
    path = here / "waveform.py"

    spec = importlib.util.spec_from_file_location(
        "paper3_waveform",
        path
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load waveform.py: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


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

    # fallback: recursively find 1D dataset
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

        # ----------------------------------------------------
        # Read GPS metadata
        # ----------------------------------------------------

        if "x0" in ds.attrs:
            gps_start = float(ds.attrs["x0"])
        elif "Xstart" in ds.attrs:
            gps_start = float(ds.attrs["Xstart"])
        else:
            raise RuntimeError(
                "No GPS start attribute (x0/Xstart) found."
            )

        if "dx" in ds.attrs:
            dx = float(ds.attrs["dx"])
            fs_file = 1.0 / dx
        elif "Xspacing" in ds.attrs:
            dx = float(ds.attrs["Xspacing"])
            fs_file = 1.0 / dx
        else:
            fs_file = fs

        if abs(fs_file - fs) > 1e-6:
            print(
                f"WARNING: requested fs={fs}, "
                f"file fs={fs_file}"
            )

        # ----------------------------------------------------
        # Convert LOCAL tc into GPS time
        #
        # Segment convention:
        #
        #   local t = 0       -> GPS_MERGER - 6 s
        #   local t = 6       -> GPS_MERGER
        #   local t = 12      -> GPS_MERGER + 6 s
        #
        # Therefore:
        #
        # GPS(tc) = GPS_MERGER + (tc_local - 6)
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Convert GPS times to indices
        # ----------------------------------------------------

        idx_center = int(
            round((gps_tc - gps_start) * fs_file)
        )

        idx_lo = int(
            round((gps_seg_start - gps_start) * fs_file)
        )

        idx_hi = idx_lo + expected_n

        print()
        print("Calculated indices:")
        print(f"  center = {idx_center}")
        print(f"  lo     = {idx_lo}")
        print(f"  hi     = {idx_hi}")
        print(f"  n      = {expected_n}")

        # ----------------------------------------------------
        # Bounds check
        # ----------------------------------------------------

        if idx_lo < 0 or idx_hi > n_total:
            raise ValueError(
                "\nRequested GPS segment outside dataset.\n"
                f"gps_start={gps_start}\n"
                f"gps_segment_start={gps_seg_start}\n"
                f"gps_segment_end={gps_seg_end}\n"
                f"idx_lo={idx_lo}\n"
                f"idx_hi={idx_hi}\n"
                f"n_total={n_total}"
            )

        data = np.asarray(ds[idx_lo:idx_hi], dtype=float)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if len(data) != expected_n:
        raise RuntimeError(
            f"Expected {expected_n} samples, got {len(data)}"
        )

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

    # Remove DC offset only.
    x = data - np.mean(data)

    window = np.hanning(n)

    # Windowed FFT.
    xw = x * window

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fd = np.fft.rfft(xw)

    return freqs, fd


# ============================================================
# PSD
# ============================================================

def estimate_psd(data, fs):
    """
    Welch PSD using scipy if available.
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
        raise RuntimeError(
            "scipy.signal.welch is required for Stage 6E.\n"
            f"Original error: {exc}"
        )


# ============================================================
# INTERPOLATE PSD
# ============================================================

def interpolate_psd(psd_f, psd, target_f):
    safe = np.maximum(psd, 1e-60)

    return np.interp(
        target_f,
        psd_f,
        safe,
        left=safe[0],
        right=safe[-1]
    )


# ============================================================
# INNER PRODUCT
# ============================================================

def inner_product(a, b, psd, df):
    """
    Real-valued frequency-domain noise-weighted product.

    This is a diagnostic/control implementation.
    """

    return 4.0 * np.real(
        np.sum(
            np.conjugate(a) * b / psd
        ) * df
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--h1",
        required=True,
        help="Path to H1 HDF5 strain file"
    )

    parser.add_argument(
        "--fs",
        type=float,
        default=DEFAULT_FS
    )

    parser.add_argument(
        "--tc",
        type=float,
        default=6.0,
        help="LOCAL tc inside 12-second extracted segment"
    )

    parser.add_argument(
        "--verbose",
        action="store_true"
    )

    args = parser.parse_args()

    print("=" * 78)
    print("STAGE 6E — TAYLORF2 GR-ONLY REAL-EVENT CONTROL")
    print("=" * 78)

    print()
    print("Configuration:")
    print(f"  m1, m2       = {M1:.2f}, {M2:.2f} Msun")
    print(f"  Distance     = {DISTANCE_MPC:.1f} Mpc")
    print(f"  Redshift     = {REDSHIFT:.3f}")
    print(f"  Band         = {F_LOW:.0f}–{F_HIGH:.0f} Hz")
    print(f"  fs           = {args.fs:.1f} Hz")
    print(f"  tc LOCAL     = {args.tc:.6f} s")
    print(f"  GPS merger   = {GPS_MERGER:.6f} s")
    print()
    print("Lambda = 0 EXACTLY")
    print("No Lambda estimation is performed.")
    print()
    print("This is a GR-only real-event control.")
    print()

    # --------------------------------------------------------
    # 1. Inspect
    # --------------------------------------------------------

    inspect_hdf5(args.h1)

    # --------------------------------------------------------
    # 2. Load GPS-correct segment
    # --------------------------------------------------------

    data = load_h1_segment(
        args.h1,
        args.fs,
        args.tc
    )

    # --------------------------------------------------------
    # 3. FFT
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("[3] FREQUENCY DOMAIN")
    print("=" * 78)

    freqs, data_fd = make_frequency_domain(
        data,
        args.fs
    )

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

    psd_f, psd = estimate_psd(
        data,
        args.fs
    )

    mask = (
        (freqs >= F_LOW) &
        (freqs <= F_HIGH)
    )

    f_band = freqs[mask]

    psd_band = interpolate_psd(
        psd_f,
        psd,
        f_band
    )

    print(f"Band bins  = {len(f_band)}")
    print(f"PSD median = {np.median(psd_band):.6e}")
    print(f"PSD min    = {np.min(psd_band):.6e}")
    print(f"PSD max    = {np.max(psd_band):.6e}")

    # --------------------------------------------------------
    # 5. TEMPLATE — SAME FREQUENCY GRID AS DATA
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("[5] TAYLORF2 GR TEMPLATE")
    print("=" * 78)

    waveform = load_waveform_module()

    # IMPORTANT:
    # Use exactly the same frequency bins as the data.
    # Do not create a second frequency mask/grid.

    f_band = freqs[
        (freqs >= F_LOW) &
        (freqs <= F_HIGH)
    ]

    # TaylorF2 returns (h_plus, h_cross) as a tuple or 2D array
    # For this control, we use h_plus only (index 0)
    template_result = waveform.taylorf2_leading_order(
        f_band,
        M1,
        M2,
        tc=0.0,
        phi_c=0.0,
        distance_Mpc=DISTANCE_MPC
    )

    # Handle different return types
    if isinstance(template_result, tuple):
        # It's a tuple (h_plus, h_cross)
        template = np.asarray(template_result[0], dtype=complex)
    elif isinstance(template_result, np.ndarray):
        if template_result.ndim == 1:
            template = template_result
        elif template_result.ndim == 2:
            # Take first row (h_plus)
            template = template_result[0, :]
        else:
            raise RuntimeError(
                f"Unexpected template shape: {template_result.shape}"
            )
    else:
        template = np.asarray(template_result, dtype=complex)

    if template.ndim != 1:
        raise RuntimeError(
            f"Template must be 1-D, got shape={template.shape}"
        )

    if len(template) != len(f_band):
        raise RuntimeError(
            "TaylorF2 returned an unexpected number of samples.\n"
            f"frequency bins = {len(f_band)}\n"
            f"template bins  = {len(template)}"
        )

    print(f"Template bins = {len(template)}")
    print(f"f min         = {f_band[0]:.6f} Hz")
    print(f"f max         = {f_band[-1]:.6f} Hz")
    print(f"max |h(f)|    = {np.max(np.abs(template)):.6e}")

    # --------------------------------------------------------
    # 6. Inner products on matched frequency bands
    # --------------------------------------------------------

    d_band = data_fd[
        (freqs >= F_LOW) &
        (freqs <= F_HIGH)
    ]

    hh = inner_product(
        template,
        template,
        psd_band,
        df
    )

    dd = inner_product(
        d_band,
        d_band,
        psd_band,
        df
    )

    dh = inner_product(
        d_band,
        template,
        psd_band,
        df
    )

    denom = np.sqrt(
        max(dd, 0.0) *
        max(hh, 0.0)
    )

    match = dh / denom if denom > 0 else np.nan

    # --------------------------------------------------------
    # 7. Report
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("[6] GR-ONLY MATCHED-FILTER CONTROL")
    print("=" * 78)

    print(f"<d|d> = {dd:.8e}")
    print(f"<h|h> = {hh:.8e}")
    print(f"<d|h> = {dh:.8e}")

    print()
    print(f"sqrt(<d|d>) = {np.sqrt(max(dd, 0.0)):.8f}")
    print(f"sqrt(<h|h>) = {np.sqrt(max(hh, 0.0)):.8f}")
    print(f"match        = {match:.8f}")

    print()
    print("=" * 78)
    print("[7] INTERPRETATION")
    print("=" * 78)

    if np.isfinite(match):
        print(f"GR-only match = {match:.8f}")

    print()
    print("Expected values for GW150914:")
    print("  - Match:       > 0.95 (ideally > 0.98)")
    print("  - SNR:         ~ 20-25")
    print()

    if np.isfinite(match):
        if match > 0.95:
            print("✅ STAGE 6E PASSES:")
            print("   The GR-only template matches the real data well.")
            print("   The infrastructure works with TaylorF2.")
        elif match > 0.80:
            print("⚠️ STAGE 6E PARTIAL:")
            print("   The match is moderate but not optimal.")
            print("   TaylorF2 may not be sufficient for real data.")
        else:
            print("❌ STAGE 6E FAILS:")
            print("   The GR-only template does NOT match the data.")
            print("   TaylorF2 is insufficient for GW150914 real data.")
            print("   This explains why Stage 6A/6D gave poor results.")
    else:
        print("❌ STAGE 6E ERROR:")
        print("   Could not compute match.")

    print()
    print("This is a REAL-EVENT GR CONTROL.")
    print()
    print("Lambda is fixed to:")
    print("  Lambda = 0")
    print()
    print("No non-zero Lambda inference is performed.")
    print("No Lambda claim is made.")

    print()
    print("=" * 78)
    print("STAGE 6E COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()