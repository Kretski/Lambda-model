"""
PAPER 3 — STAGE 6D.5
CROSS-CHECK OF STAGE 6C / 6D NORMALIZATION

Purpose:
    Directly compare the inner-product normalization used by
    Stage 6C and Stage 6D for the same H1 segment and frequency band.

This is an integrity diagnostic.

It does NOT estimate Lambda.
It does NOT claim GW detection.
It does NOT modify the physical model.
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np


# ----------------------------------------------------------------------
# PATH SETUP
# ----------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


# ----------------------------------------------------------------------
# IMPORT EXISTING PIPELINE
# ----------------------------------------------------------------------

try:
    import stage3_real_strain_validation as stage3
except Exception as exc:
    print("[ERROR] Could not import stage3_real_strain_validation")
    print(exc)
    sys.exit(1)

try:
    import waveform
except Exception as exc:
    print("[ERROR] Could not import waveform")
    print(exc)
    sys.exit(1)


# ----------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------

F_LOW = 20.0
F_HIGH = 300.0

M1 = 35.60
M2 = 30.60
DISTANCE_MPC = 440.0
Z = 0.09

LAMBDA = 0.0

DEFAULT_TC = 5.99


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def safe_float(x):
    try:
        return float(np.asarray(x))
    except Exception:
        return float(x)


def summarize(name, value):
    value = safe_float(value)
    print(f"{name:<30} = {value:.12e}")


def ratio(a, b):
    if abs(b) < 1e-300:
        return np.nan
    return a / b


# ----------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------

def load_segment(h1_path, fs, gps_merger):
    """
    Load exactly the same type of real H1 segment used by Stage 6C/6D.

    Stage-3 API (real signature):
        load_real_segment(h1_path, gps_center, half_window,
                           fs_override=None)

    IMPORTANT: the analysis segment is always centered on the fixed
    event GPS merger time (EVENT_CATALOG), NOT on the template tc
    scan value. tc is a LOCAL offset inside this fixed window (used
    only by build_template()) and must never be passed as gps_center
    -- conflating the two was the root cause of the earlier crashes.

    Stage 6D uses a 12-second segment, therefore:
        half_window = 6.0 s
    """

    print("=" * 78)
    print("[1] LOADING H1 DATA")
    print("=" * 78)

    half_window = 6.0

    print(f"GPS merger (fixed) = {gps_merger:.6f} s")
    print(f"Half-window        = {half_window:.6f} s")
    print(f"Total duration     = {2.0 * half_window:.6f} s")

    result = stage3.load_real_segment(
        h1_path,
        gps_merger,
        half_window,
        fs_override=fs,
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

    print()
    print("Loaded H1 segment successfully.")

    print(f"Samples            = {len(data)}")
    print(f"Expected samples    = {int(round(12.0 * fs))}")
    print(f"Duration [s]        = {len(data) / fs:.9f}")

    print(f"mean                = {np.mean(data):.12e}")
    print(f"std                 = {np.std(data):.12e}")

    return data, meta


# ----------------------------------------------------------------------
# FFT
# ----------------------------------------------------------------------

def to_frequency_domain(data, fs):
    """
    Explicit one-sided real FFT.

    Keep this implementation visible so Stage 6D.5 can report
    exactly which convention is being used.
    """

    n = len(data)

    data_demean = data - np.mean(data)

    fd = np.fft.rfft(data_demean)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    df = fs / n

    return freqs, fd, df


# ----------------------------------------------------------------------
# PSD
# ----------------------------------------------------------------------

def estimate_psd(data, fs):
    """
    Try the existing PSD estimator from Stage 3.

    If unavailable, fall back to a simple periodogram.
    The fallback is explicitly reported.
    """

    estimator_names = [
        "estimate_psd_from_segment",
        "estimate_psd",
    ]

    for name in estimator_names:

        if hasattr(stage3, name):

            fn = getattr(stage3, name)

            freq_guess = np.fft.rfftfreq(len(data), d=1.0 / fs)

            attempts = [
                lambda: fn(data, fs, freq_guess),
                lambda: fn(data, fs),
                lambda: fn(data=data, fs=fs),
                lambda: fn(data, sample_rate=fs),
                lambda: fn(data=data, sample_rate=fs),
            ]

            for attempt in attempts:
                try:
                    result = attempt()

                    if isinstance(result, tuple):
                        psd = np.asarray(result[0], dtype=float)
                        psd_freq = np.asarray(result[1], dtype=float)
                    else:
                        psd = np.asarray(result, dtype=float)
                        psd_freq = np.fft.rfftfreq(
                            len(data),
                            d=1.0 / fs
                        )

                    print(f"PSD estimator = stage3.{name}")

                    return psd_freq, psd

                except Exception:
                    pass

    print("[WARNING] Existing PSD estimator unavailable.")
    print("          Using periodogram fallback.")

    x = data - np.mean(data)

    n = len(x)

    fft = np.fft.rfft(x)

    psd = (
        np.abs(fft) ** 2
        / (fs * n)
    )

    freq = np.fft.rfftfreq(
        n,
        d=1.0 / fs
    )

    return freq, psd


# ----------------------------------------------------------------------
# PSD INTERPOLATION
# ----------------------------------------------------------------------

def interpolate_psd(freq, psd, target_freq):

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

    return np.interp(
        target_freq,
        freq,
        psd
    )


# ----------------------------------------------------------------------
# INNER PRODUCT
# ----------------------------------------------------------------------

def inner_product(a_fd, b_fd, psd, df):

    valid = (
        np.isfinite(a_fd)
        & np.isfinite(b_fd)
        & np.isfinite(psd)
        & (psd > 0)
    )

    value = (
        4.0
        * df
        * np.real(
            np.sum(
                np.conjugate(a_fd[valid])
                * b_fd[valid]
                / psd[valid]
            )
        )
    )

    return float(value)


# ----------------------------------------------------------------------
# TEMPLATE
# ----------------------------------------------------------------------

def build_template(freq, tc):

    mask = (
        (freq >= F_LOW)
        & (freq <= F_HIGH)
    )

    f = freq[mask]

    # Try existing waveform API.
    if hasattr(waveform, "waveform_frequency_domain"):

        h = waveform.waveform_frequency_domain(
            f,
            M1,
            M2,
            LAMBDA,
            waveform.cosmological_K_factor(Z),
            tc=tc,
            phi_c=0.0,
            distance_Mpc=DISTANCE_MPC,
        )

        return f, np.asarray(h, dtype=complex), mask

    raise RuntimeError(
        "waveform_frequency_domain() not found."
    )


# ----------------------------------------------------------------------
# MAIN DIAGNOSTIC
# ----------------------------------------------------------------------

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
        default=4096.0
    )

    parser.add_argument(
        "--tc",
        type=float,
        default=DEFAULT_TC
    )

    args = parser.parse_args()

    print()
    print("=" * 78)
    print("PAPER 3 — STAGE 6D.5")
    print("STAGE 6C / 6D NORMALIZATION CROSS-CHECK")
    print("=" * 78)

    print()
    print("H1:")
    print(args.h1)

    print(f"fs        = {args.fs:.6f} Hz")
    print(f"tc        = {args.tc:.6f} s")
    print(f"band      = {F_LOW:.1f} – {F_HIGH:.1f} Hz")
    print(f"Lambda    = {LAMBDA:.6e}")

    # ------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------

    gps_merger = stage3.EVENT_CATALOG["GW150914"]["gps_merger"]

    print(f"gps_merger (fixed) = {gps_merger:.6f} s")

    data, meta = load_segment(
        args.h1,
        args.fs,
        gps_merger,
    )

    n = len(data)

    duration = n / args.fs

    print()
    print("=" * 78)
    print("[2] SEGMENT")
    print("=" * 78)

    summarize("samples", n)
    summarize("duration [s]", duration)
    summarize("mean", np.mean(data))
    summarize("std", np.std(data))

    # ------------------------------------------------------------------
    # FFT
    # ------------------------------------------------------------------

    print()
    print("=" * 78)
    print("[3] FFT")
    print("=" * 78)

    freq, data_fd, df = to_frequency_domain(
        data,
        args.fs
    )

    print(f"FFT convention     = rFFT")
    print(f"FFT bins           = {len(freq)}")
    print(f"df                 = {df:.12e} Hz")

    # ------------------------------------------------------------------
    # PSD
    # ------------------------------------------------------------------

    print()
    print("=" * 78)
    print("[4] PSD")
    print("=" * 78)

    psd_freq, psd_raw = estimate_psd(
        data,
        args.fs
    )

    psd = interpolate_psd(
        psd_freq,
        psd_raw,
        freq
    )

    band = (
        (freq >= F_LOW)
        & (freq <= F_HIGH)
        & np.isfinite(psd)
        & (psd > 0)
    )

    print(f"Band bins          = {np.sum(band)}")
    print(
        f"PSD median         = "
        f"{np.median(psd[band]):.12e}"
    )
    print(
        f"PSD min            = "
        f"{np.min(psd[band]):.12e}"
    )
    print(
        f"PSD max            = "
        f"{np.max(psd[band]):.12e}"
    )

    # ------------------------------------------------------------------
    # TEMPLATE
    # ------------------------------------------------------------------

    print()
    print("=" * 78)
    print("[5] TEMPLATE")
    print("=" * 78)

    f_template, h_band, template_mask = build_template(
        freq,
        args.tc
    )

    h_fd = np.zeros_like(freq, dtype=complex)

    h_fd[template_mask] = h_band

    print(f"Template bins      = {len(h_band)}")
    print(
        f"max |h(f)|         = "
        f"{np.max(np.abs(h_band)):.12e}"
    )

    # ------------------------------------------------------------------
    # INNER PRODUCTS
    # ------------------------------------------------------------------

    print()
    print("=" * 78)
    print("[6] RAW INNER PRODUCTS")
    print("=" * 78)

    dd = inner_product(
        data_fd[band],
        data_fd[band],
        psd[band],
        df
    )

    hh = inner_product(
        h_fd[band],
        h_fd[band],
        psd[band],
        df
    )

    dh = inner_product(
        data_fd[band],
        h_fd[band],
        psd[band],
        df
    )

    match = dh / np.sqrt(
        max(dd * hh, 1e-300)
    )

    summarize("<d|d>", dd)
    summarize("sqrt(<d|d>)", np.sqrt(dd))

    summarize("<h|h>", hh)
    summarize("sqrt(<h|h>)", np.sqrt(hh))

    summarize("<d|h>", dh)
    summarize("match", match)

    # ------------------------------------------------------------------
    # COMPONENT RATIOS
    # ------------------------------------------------------------------

    print()
    print("=" * 78)
    print("[7] NORMALIZATION DIAGNOSTIC")
    print("=" * 78)

    print()
    print("Reference: Stage 6D reported")
    print("    <d|d> = 4.24527432e+10")
    print("    <h|h> = 1.32300285e+03")
    print("    sqrt(<h|h>) = 36.37310609")
    print()

    dd_6d = 4.24527432e10
    hh_6d = 1.32300285e3

    print(
        f"Current / Stage6D dd = "
        f"{ratio(dd, dd_6d):.12e}"
    )

    print(
        f"Current / Stage6D hh = "
        f"{ratio(hh, hh_6d):.12e}"
    )

    print(
        f"Current sqrt(dd) / Stage6D sqrt(dd) = "
        f"{ratio(np.sqrt(dd), np.sqrt(dd_6d)):.12e}"
    )

    print(
        f"Current sqrt(hh) / Stage6D sqrt(hh) = "
        f"{ratio(np.sqrt(hh), np.sqrt(hh_6d)):.12e}"
    )

    # ------------------------------------------------------------------
    # EXPECTED 3000x CHECK
    # ------------------------------------------------------------------

    print()
    print("=" * 78)
    print("[8] 3000x NORMALIZATION CHECK")
    print("=" * 78)

    ratio_dd = ratio(dd_6d, dd)
    ratio_hh = ratio(hh_6d, hh)

    print(
        f"Stage6D dd / current dd = "
        f"{ratio_dd:.6e}"
    )

    print(
        f"Stage6D hh / current hh = "
        f"{ratio_hh:.6e}"
    )

    print()
    print(
        "If dd differs by ~1e4 while hh remains similar,"
    )
    print(
        "the main discrepancy is in DATA normalization / FFT / PSD."
    )

    print(
        "If hh differs strongly while dd remains similar,"
    )
    print(
        "the main discrepancy is in TEMPLATE normalization."
    )

    print(
        "If both scale by approximately the same factor,"
    )
    print(
        "the PSD / global inner-product convention is suspect."
    )

    # ------------------------------------------------------------------
    # DATA VS TEMPLATE AMPLITUDE
    # ------------------------------------------------------------------

    print()
    print("=" * 78)
    print("[9] FREQUENCY-DOMAIN AMPLITUDE CHECK")
    print("=" * 78)

    for target in [50.0, 100.0, 150.0, 200.0, 250.0]:

        idx = np.argmin(
            np.abs(freq - target)
        )

        print(
            f"{freq[idx]:8.3f} Hz  "
            f"|data|={abs(data_fd[idx]):.12e}  "
            f"|template|={abs(h_fd[idx]):.12e}  "
            f"PSD={psd[idx]:.12e}"
        )

    # ------------------------------------------------------------------
    # PSD-WEIGHTED CONTRIBUTION
    # ------------------------------------------------------------------

    print()
    print("=" * 78)
    print("[10] INNER-PRODUCT CONTRIBUTION")
    print("=" * 78)

    weighted_data = np.zeros_like(freq)

    weighted_template = np.zeros_like(freq)

    valid = band

    weighted_data[valid] = (
        4.0
        * df
        * np.abs(data_fd[valid]) ** 2
        / psd[valid]
    )

    weighted_template[valid] = (
        4.0
        * df
        * np.abs(h_fd[valid]) ** 2
        / psd[valid]
    )

    for lo, hi in [
        (20, 50),
        (50, 100),
        (100, 150),
        (150, 200),
        (200, 250),
        (250, 300),
    ]:

        m = (
            valid
            & (freq >= lo)
            & (freq < hi)
        )

        print(
            f"{lo:3d}-{hi:3d} Hz   "
            f"data={np.sum(weighted_data[m]):.6e}   "
            f"template={np.sum(weighted_template[m]):.6e}"
        )

    # ------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------

    print()
    print("=" * 78)
    print("STAGE 6D.5 FINAL")
    print("=" * 78)

    print()
    print(f"<d|d> = {dd:.12e}")
    print(f"<h|h> = {hh:.12e}")
    print(f"<d|h> = {dh:.12e}")
    print(f"match = {match:.12e}")

    print()
    print("This is a normalization/integrity diagnostic only.")
    print("No Lambda inference is performed.")
    print("No non-zero Lambda claim is made.")

    print()
    print("=" * 78)
    print("STAGE 6D.5 COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()