#!/usr/bin/env python3
"""
PAPER 3 — STAGE 6D
FULL tc-ALIGNMENT DIAGNOSTIC  (FIXED)

Purpose
-------
Test whether the real GW150914 H1 strain can be aligned with the
Lambda=0 GR template by scanning the coalescence time tc.

IMPORTANT
---------
This is an alignment/integrity diagnostic.

It does NOT estimate a physical non-zero Lambda.
It does NOT constitute evidence for Lambda != 0.

The diagnostic keeps Lambda fixed at zero and scans only tc.

FIX NOTES (vs. original)
-------------------------
- load_h1_data() now calls load_real_segment() with its real
  signature: (h1_path, gps_center, half_window, fs_override).
  The original blind (path, fs) / (path, fs=fs) attempts always
  raised TypeError because gps_center and half_window are required
  positional arguments.
- Event masses/distance/redshift are now taken from
  stage3_real_strain_validation.EVENT_CATALOG instead of being
  hardcoded (36.0, 29.0) in make_template().
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))

if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import stage3_real_strain_validation as stage3
except Exception as exc:
    print("[FATAL] Could not import stage3_real_strain_validation")
    print(exc)
    sys.exit(1)

try:
    import waveform
except Exception as exc:
    print("[FATAL] Could not import waveform")
    print(exc)
    sys.exit(1)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

F_LOW = 20.0
F_HIGH = 300.0

DEFAULT_FS = 4096.0
DEFAULT_DURATION = 8.0
DEFAULT_EVENT = "GW150914"

# Full local tc scan.
TC_MIN = 0.0
TC_MAX = 12.0
TC_STEP = 0.01

LAMBDA = 0.0

# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------


def banner(title: str):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def safe_float(x):
    try:
        return float(np.real(x))
    except Exception:
        return np.nan


def normalized_match(dh, dd, hh):
    denom = np.sqrt(max(dd * hh, 0.0))

    if denom <= 0:
        return np.nan

    return safe_float(dh / denom)


# ---------------------------------------------------------------------
# HDF5 loading
# ---------------------------------------------------------------------


def load_h1_data(path, fs, gps_center, half_window):
    """
    Call the Stage-3 real-data loader with its actual signature:

        load_real_segment(h1_path, gps_center, half_window,
                           fs_override=None)

    This reuses the corrected (x0/dx-aware) GPS handling from
    Stage 3 instead of implementing a second competing HDF5 loader.
    """

    fn = getattr(stage3, "load_real_segment", None)

    if fn is None:
        raise RuntimeError(
            "stage3_real_strain_validation.load_real_segment not found."
        )

    try:
        result = fn(
            path,
            gps_center,
            half_window,
            fs_override=fs,
        )
    except Exception as exc:
        raise RuntimeError(
            f"load_real_segment() call failed: {exc}"
        ) from exc

    if result is None:
        raise RuntimeError("load_real_segment() returned None.")

    return result


# ---------------------------------------------------------------------
# Generic extraction
# ---------------------------------------------------------------------


def extract_array(result):
    """
    Extract strain array from common Stage-3 return formats.

    load_real_segment() returns (data, fs, seg_gps_start) -- a tuple
    whose first ndarray element is the strain array.
    """

    if isinstance(result, np.ndarray):
        return result.astype(float)

    if isinstance(result, (tuple, list)):
        for item in result:
            if isinstance(item, np.ndarray):
                if item.ndim == 1:
                    return item.astype(float)

    if isinstance(result, dict):
        keys = [
            "strain",
            "data",
            "h1",
            "strain_data",
            "segment",
        ]

        for key in keys:
            if key in result:
                value = result[key]

                if isinstance(value, np.ndarray):
                    return value.astype(float)

    raise RuntimeError(
        "Could not extract a 1-D strain array from Stage-3 loader output."
    )


# ---------------------------------------------------------------------
# Frequency-domain conversion
# ---------------------------------------------------------------------


def to_fd(data, fs):
    """
    Standard real FFT convention.

    The diagnostic keeps this isolated so that FFT normalization can be
    compared directly with the convention used elsewhere in the pipeline.
    """

    data = np.asarray(data, dtype=float)

    n = len(data)

    if n < 2:
        raise ValueError("Not enough samples for FFT.")

    window = np.hanning(n)

    x = data * window

    fd = np.fft.rfft(x)

    freq = np.fft.rfftfreq(n, d=1.0 / fs)

    return freq, fd


# ---------------------------------------------------------------------
# PSD
# ---------------------------------------------------------------------


def estimate_psd(data, fs):
    """
    Conservative Welch PSD estimator.

    Used only if the Stage-3 PSD function cannot be accessed.
    """

    try:
        from scipy.signal import welch

        n = len(data)

        nperseg = min(4096, n)

        freq, psd = welch(
            data,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            detrend="constant",
            scaling="density",
        )

        return freq, psd

    except Exception as exc:
        raise RuntimeError(
            "Could not estimate PSD with scipy.signal.welch"
        ) from exc


def interpolate_psd(psd_f, psd, target_f):
    """
    Interpolate PSD onto FFT frequencies.
    """

    result = np.interp(
        target_f,
        psd_f,
        psd,
        left=np.nan,
        right=np.nan,
    )

    return result


# ---------------------------------------------------------------------
# Inner product
# ---------------------------------------------------------------------


def inner_product(
    d_fd,
    h_fd,
    psd,
    df,
    mask,
):
    """
    Frequency-domain noise-weighted inner product.
    """

    d = d_fd[mask]
    h = h_fd[mask]
    p = psd[mask]

    valid = (
        np.isfinite(p)
        & (p > 0)
    )

    d = d[valid]
    h = h[valid]
    p = p[valid]

    if len(d) == 0:
        return np.nan

    return safe_float(
        4.0 * np.real(np.sum(
            np.conjugate(d) * h / p
        )) * df
    )


def template_norm(h_fd, psd, df, mask):
    h = h_fd[mask]
    p = psd[mask]

    valid = np.isfinite(p) & (p > 0)

    h = h[valid]
    p = p[valid]

    return safe_float(
        4.0 * np.sum(
            np.abs(h) ** 2 / p
        ) * df
    )


def data_norm(d_fd, psd, df, mask):
    d = d_fd[mask]
    p = psd[mask]

    valid = np.isfinite(p) & (p > 0)

    d = d[valid]
    p = p[valid]

    return safe_float(
        4.0 * np.sum(
            np.abs(d) ** 2 / p
        ) * df
    )


# ---------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------


def make_template(
    f,
    tc,
    lambda_value,
    m1,
    m2,
    K_z,
    distance_Mpc,
):
    """
    Generate the Lambda-model frequency-domain template using the
    real event parameters (masses/distance) instead of hardcoded
    placeholder values.
    """

    fn = getattr(waveform, "waveform_frequency_domain", None)

    if fn is None:
        raise RuntimeError(
            "waveform.waveform_frequency_domain() not found."
        )

    h = fn(
        f,
        m1,
        m2,
        lambda_value,
        K_z,
        tc=tc,
        phi_c=0.0,
        distance_Mpc=distance_Mpc,
    )

    return np.asarray(h, dtype=complex)


# ---------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------


def main():

    parser = argparse.ArgumentParser(
        description="Stage 6D full tc alignment diagnostic"
    )

    parser.add_argument(
        "--h1",
        required=True,
        help="Path to GW150914 H1 HDF5 file",
    )

    parser.add_argument(
        "--event",
        default=DEFAULT_EVENT,
        help="Event name in EVENT_CATALOG",
    )

    parser.add_argument(
        "--fs",
        type=float,
        default=DEFAULT_FS,
        help="Sampling frequency",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help="Analysis duration in seconds (window = duration + 4s, "
             "same convention as Stage 6A/6B/6C)",
    )

    parser.add_argument(
        "--tc-min",
        type=float,
        default=TC_MIN,
        help="Minimum local tc",
    )

    parser.add_argument(
        "--tc-max",
        type=float,
        default=TC_MAX,
        help="Maximum local tc",
    )

    parser.add_argument(
        "--tc-step",
        type=float,
        default=TC_STEP,
        help="tc scan step",
    )

    args = parser.parse_args()

    banner(
        "PAPER 3 — STAGE 6D\n"
        "FULL tc-ALIGNMENT DIAGNOSTIC"
    )

    event = stage3.EVENT_CATALOG[args.event]
    m1 = event["m1"]
    m2 = event["m2"]
    distance_Mpc = event["distance_Mpc"]
    z = event["z"]
    K_z = waveform.cosmological_K_factor(z)
    gps_merger = event["gps_merger"]

    print()
    print("H1 file:")
    print(args.h1)

    print()
    print("Event:")
    print(f"{args.event}  (m1={m1:.2f}, m2={m2:.2f} Msun, "
          f"D={distance_Mpc:.1f} Mpc, z={z})")

    print()
    print("Sampling frequency:")
    print(f"{args.fs:.3f} Hz")

    print()
    print("Frequency band:")
    print(f"{F_LOW:.1f} – {F_HIGH:.1f} Hz")

    print()
    print("Lambda:")
    print(f"{LAMBDA:.6e}")

    print()
    print("tc scan:")
    print(
        f"[{args.tc_min:.3f}, "
        f"{args.tc_max:.3f}] s "
        f"step={args.tc_step:.4f} s"
    )

    # -----------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------

    banner("[1] LOADING REAL H1 DATA")

    half_window = args.duration / 2.0 + 2.0

    result = load_h1_data(
        args.h1,
        args.fs,
        gps_merger,
        half_window,
    )

    data = extract_array(result)

    # actual fs used (in case fs_override differed from metadata)
    if isinstance(result, (tuple, list)) and len(result) >= 2:
        fs_used = float(result[1])
    else:
        fs_used = args.fs

    print(f"Samples = {len(data)}")
    print(f"Duration = {len(data) / fs_used:.6f} s")
    print(
        f"mean = {np.mean(data):.6e}"
    )
    print(
        f"std  = {np.std(data):.6e}"
    )

    # -----------------------------------------------------------------
    # FFT
    # -----------------------------------------------------------------

    banner("[2] FREQUENCY DOMAIN")

    f, d_fd = to_fd(data, fs_used)

    df = f[1] - f[0]

    print(f"Frequency bins = {len(f)}")
    print(f"df = {df:.8f} Hz")

    # -----------------------------------------------------------------
    # PSD
    # -----------------------------------------------------------------

    banner("[3] PSD")

    psd_f, psd_raw = estimate_psd(data, fs_used)

    psd = interpolate_psd(
        psd_f,
        psd_raw,
        f,
    )

    mask = (
        (f >= F_LOW)
        & (f <= F_HIGH)
        & np.isfinite(psd)
        & (psd > 0)
    )

    print(
        f"Valid bins in band = {np.sum(mask)}"
    )

    print(
        f"PSD median = "
        f"{np.median(psd[mask]):.6e}"
    )

    # -----------------------------------------------------------------
    # Data norm
    # -----------------------------------------------------------------

    banner("[4] DATA NORMALIZATION")

    dd = data_norm(
        d_fd,
        psd,
        df,
        mask,
    )

    print(
        f"<d|d> = {dd:.8e}"
    )

    print(
        f"sqrt(<d|d>) = "
        f"{np.sqrt(abs(dd)):.8f}"
    )

    # -----------------------------------------------------------------
    # tc scan
    # -----------------------------------------------------------------

    banner("[5] FULL tc SCAN")

    tc_values = np.arange(
        args.tc_min,
        args.tc_max + 0.5 * args.tc_step,
        args.tc_step,
    )

    matches = []
    overlaps = []
    template_snrs = []

    best = None

    print(
        f"Testing {len(tc_values)} tc values..."
    )

    for i, tc in enumerate(tc_values):

        try:

            h_fd = make_template(
                f,
                tc,
                LAMBDA,
                m1,
                m2,
                K_z,
                distance_Mpc,
            )

            hh = template_norm(
                h_fd,
                psd,
                df,
                mask,
            )

            dh = inner_product(
                d_fd,
                h_fd,
                psd,
                df,
                mask,
            )

            match = normalized_match(
                dh,
                dd,
                hh,
            )

            matches.append(match)
            overlaps.append(dh)
            template_snrs.append(
                np.sqrt(abs(hh))
            )

            if np.isfinite(match):

                if best is None:
                    best = (
                        match,
                        tc,
                        dh,
                        hh,
                    )

                elif match > best[0]:
                    best = (
                        match,
                        tc,
                        dh,
                        hh,
                    )

        except Exception as exc:

            print(
                f"[WARN] tc={tc:.4f}: {exc}"
            )

            matches.append(np.nan)
            overlaps.append(np.nan)
            template_snrs.append(np.nan)

        if (
            (i + 1) % 100 == 0
            or i == len(tc_values) - 1
        ):
            print(
                f"  {i+1:5d}/"
                f"{len(tc_values)}"
            )

    matches = np.asarray(matches)
    overlaps = np.asarray(overlaps)
    template_snrs = np.asarray(template_snrs)

    # -----------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------

    banner("[6] ALIGNMENT RESULT")

    valid = np.isfinite(matches)

    if not np.any(valid):

        print(
            "[FAIL] No valid tc values produced a match."
        )

        sys.exit(2)

    idx = np.nanargmax(matches)

    best_match = matches[idx]
    best_tc = tc_values[idx]

    best_dh = overlaps[idx]
    best_hh = template_snrs[idx] ** 2

    print()
    print(
        f"Best tc          = "
        f"{best_tc:.6f} s"
    )

    print(
        f"Maximum match    = "
        f"{best_match:.8f}"
    )

    print(
        f"<d|h>            = "
        f"{best_dh:.8e}"
    )

    print(
        f"<h|h>            = "
        f"{best_hh:.8e}"
    )

    print(
        f"Template SNR      = "
        f"{np.sqrt(abs(best_hh)):.8f}"
    )

    # -----------------------------------------------------------------
    # Nearby values
    # -----------------------------------------------------------------

    banner("[7] LOCAL PEAK")

    lo = max(0, idx - 5)
    hi = min(len(tc_values), idx + 6)

    print()
    print(
        "       tc [s]              match"
    )
    print(
        "------------------------------------------"
    )

    for j in range(lo, hi):

        print(
            f"{tc_values[j]:14.6f}    "
            f"{matches[j]: .10f}"
        )

    # -----------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------

    results_dir = os.path.join(
        HERE,
        "results",
    )

    os.makedirs(
        results_dir,
        exist_ok=True,
    )

    csv_path = os.path.join(
        results_dir,
        "stage6D_tc_scan.csv",
    )

    table = np.column_stack(
        [
            tc_values,
            matches,
            overlaps,
            template_snrs,
        ]
    )

    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        header="tc,match,dh,template_snr",
        comments="",
    )

    print()
    print(
        f"CSV saved: {csv_path}"
    )

    # -----------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------

    banner("[8] PLOT")

    try:

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.plot(
            tc_values,
            matches,
            linewidth=1.0,
        )

        ax.axvline(
            best_tc,
            linestyle="--",
            label=f"best tc = {best_tc:.4f} s",
        )

        ax.axhline(
            0.0,
            linestyle=":",
        )

        ax.set_xlabel(
            "tc [s]"
        )

        ax.set_ylabel(
            "normalized match"
        )

        ax.set_title(
            "GW150914 H1 — Stage 6D tc Alignment"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

        fig.tight_layout()

        plot_path = os.path.join(
            results_dir,
            "stage6D_tc_alignment.svg",
        )

        fig.savefig(
            plot_path,
            format="svg",
        )

        plt.close(fig)

        print(
            f"SVG saved: {plot_path}"
        )

    except Exception as exc:

        print(
            "[WARN] Plot generation failed:"
        )

        print(exc)

    # -----------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------

    banner("FINAL INTERPRETATION")

    print()

    if best_match >= 0.5:

        print(
            "[RESULT] Strong alignment detected."
        )

        print(
            "The real H1 data contain a region that "
            "is substantially correlated with the GR template."
        )

        print(
            "Proceed to the next diagnostic only after "
            "checking the tc/GPS convention."
        )

    elif best_match >= 0.1:

        print(
            "[RESULT] Weak/moderate alignment."
        )

        print(
            "The template and data show some correlation, "
            "but the match is not yet strong enough for a "
            "physical Lambda inference."
        )

    else:

        print(
            "[RESULT] No strong alignment."
        )

        print(
            "The maximum match remains small across the "
            "entire tc scan."
        )

        print(
            "Do NOT estimate a physical non-zero Lambda "
            "from this result."
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Stage 6D is an alignment/integrity test."
    )

    print(
        "It does NOT establish Lambda != 0."
    )

    print(
        "A physical Lambda inference requires a validated "
        "real-event matched-filter alignment first."
    )

    banner("STAGE 6D COMPLETE")


if __name__ == "__main__":
    main()
