"""
stage6A_frequency_robustness.py
================================

STAGE 6A — FREQUENCY-BAND ROBUSTNESS

Purpose
-------
Test whether the previously obtained Lambda estimate is robust to the
choice of observational frequency band.

IMPORTANT:
    This stage intentionally reuses the EXISTING validated Stage-3
    waveform + likelihood pipeline.

    It does NOT:
        - rewrite the likelihood
        - change the Lambda phase model
        - fit masses, distance, tc, or phi_c
        - select a frequency band because of the observed Lambda value
        - use Lambda = -2.78 as an optimization target

The previously obtained value

    Lambda_reference = -2.7800

is recorded only as an external/reference value. It is NOT used to
choose the bands or alter the likelihood.

Bands tested by default
-----------------------
    20-100 Hz
    20-150 Hz
    20-200 Hz
    30-200 Hz
    30-300 Hz

For each band we calculate:

    Lambda_ML
    Fisher curvature error
    match(Lambda_ML, Lambda=0)
    Delta_logL = logL(Lambda_ML) - logL(Lambda=0)
    distance_from_zero = |Lambda_ML| / sigma_Lambda

Output
------
    results/stage6A_frequency_robustness.csv
    results/stage6A_frequency_robustness.png

Example
-------
python stage6A_frequency_robustness.py ^
    --h1 "C:\\Users\\Lenovo\\Desktop\\GravOptAdaptiveE-main\\H1_GW150914_4096s.hdf5" ^
    --fs 4096
"""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
# IMPORT EXISTING VALIDATED PIPELINE
# ======================================================================

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from stage3_real_strain_validation import (
    EVENT_CATALOG,
    load_real_segment,
    to_frequency_domain,
    estimate_psd_from_segment,
)

from likelihood import (
    grid_search_lambda,
    log_likelihood,
    noise_weighted_inner_product,
)

from waveform import (
    waveform_frequency_domain,
    cosmological_K_factor,
)


# ======================================================================
# CONFIGURATION
# ======================================================================

REFERENCE_LAMBDA = -2.7800

DEFAULT_BANDS = [
    (20.0, 100.0),
    (20.0, 150.0),
    (20.0, 200.0),
    (30.0, 200.0),
    (30.0, 300.0),
]

DEFAULT_DURATION = 8.0

# Exactly the same broad grid convention used by Stage 3.
DEFAULT_GRID = np.linspace(-5.0, 5.0, 1001)


# ======================================================================
# UTILITIES
# ======================================================================

def compute_match(
    data_fd,
    f,
    psd,
    df,
    m1,
    m2,
    K_z,
    Lambda,
    distance_Mpc,
):
    """
    Normalized noise-weighted match between the real data and a
    Lambda waveform.

        match(d,h) = <d|h> /
                     sqrt(<d|d><h|h>)

    This is a diagnostic quantity only.
    """

    h = waveform_frequency_domain(
        f,
        m1,
        m2,
        Lambda,
        K_z,
        distance_Mpc=distance_Mpc,
    )

    dd = noise_weighted_inner_product(
        data_fd,
        data_fd,
        f,
        psd,
        df,
    )

    hh = noise_weighted_inner_product(
        h,
        h,
        f,
        psd,
        df,
    )

    if dd <= 0.0 or hh <= 0.0:
        return np.nan

    dh = noise_weighted_inner_product(
        data_fd,
        h,
        f,
        psd,
        df,
    )

    return dh / np.sqrt(dd * hh)


def interpolate_logL_at_zero(Lambda_grid, logL):
    """
    Lambda=0 is normally an exact grid point for the symmetric Stage-3
    grid. Keep interpolation as a defensive fallback.
    """

    Lambda_grid = np.asarray(Lambda_grid)
    logL = np.asarray(logL)

    exact = np.where(np.isclose(Lambda_grid, 0.0))[0]

    if len(exact):
        return float(logL[exact[0]])

    return float(np.interp(0.0, Lambda_grid, logL))


def fisher_error_from_grid(Lambda_grid, logL):
    """
    Reproduce the Stage-3 local-curvature prescription.

    This deliberately does NOT introduce a different uncertainty
    estimator.
    """

    i_max = int(np.argmax(logL))

    if i_max <= 0 or i_max >= len(Lambda_grid) - 1:
        return np.nan

    dL = Lambda_grid[1] - Lambda_grid[0]

    d2logL = (
        logL[i_max + 1]
        - 2.0 * logL[i_max]
        + logL[i_max - 1]
    ) / dL ** 2

    if d2logL < 0.0:
        return float(np.sqrt(-1.0 / d2logL))

    return np.nan


# ======================================================================
# SINGLE-BAND ANALYSIS
# ======================================================================

def analyze_band(
    strain_td,
    fs,
    f_min,
    f_max,
    duration,
    m1,
    m2,
    K_z,
    distance_Mpc,
    Lambda_grid,
):
    """
    Analyze exactly one frequency band using the existing Stage-3
    Fourier/PSD + likelihood machinery.
    """

    # --------------------------------------------------------------
    # Same Stage-3 conversion, but with this band's limits.
    # --------------------------------------------------------------

    f, data_fd, df = to_frequency_domain(
        strain_td,
        fs,
        f_min,
        f_max,
        duration,
    )

    # --------------------------------------------------------------
    # Same Stage-3 PSD estimator.
    # --------------------------------------------------------------

    psd = estimate_psd_from_segment(
        strain_td,
        fs,
        f,
    )

    # --------------------------------------------------------------
    # Existing likelihood grid search.
    # --------------------------------------------------------------

    (
        Lgrid,
        logL,
        Lambda_ml,
        Lambda_err,
    ) = grid_search_lambda(
        data_fd,
        f,
        psd,
        df,
        m1,
        m2,
        K_z,
        Lambda_grid,
        distance_Mpc=distance_Mpc,
    )

    # --------------------------------------------------------------
    # Reference GR likelihood.
    # --------------------------------------------------------------

    logL_zero = interpolate_logL_at_zero(
        Lgrid,
        logL,
    )

    logL_max = float(np.max(logL))

    delta_logL = logL_max - logL_zero

    # --------------------------------------------------------------
    # Match of ML waveform against the actual data.
    # --------------------------------------------------------------

    match_ml = compute_match(
        data_fd,
        f,
        psd,
        df,
        m1,
        m2,
        K_z,
        Lambda_ml,
        distance_Mpc,
    )

    # --------------------------------------------------------------
    # Match of GR waveform against data.
    # --------------------------------------------------------------

    match_gr = compute_match(
        data_fd,
        f,
        psd,
        df,
        m1,
        m2,
        K_z,
        0.0,
        distance_Mpc,
    )

    # --------------------------------------------------------------
    # Reproduce Stage-3 Fisher curvature independently as a check.
    # --------------------------------------------------------------

    curvature_error = fisher_error_from_grid(
        Lgrid,
        logL,
    )

    # Normally this should equal Lambda_err from likelihood.py.
    # Prefer the existing likelihood.py result, while recording the
    # independent calculation for audit/debugging.
    if np.isfinite(Lambda_err):
        sigma = float(Lambda_err)
    else:
        sigma = float(curvature_error)

    if np.isfinite(sigma) and sigma > 0.0:
        distance_from_zero = abs(float(Lambda_ml)) / sigma
    else:
        distance_from_zero = np.nan

    return {
        "f_min_Hz": float(f_min),
        "f_max_Hz": float(f_max),
        "band": f"{f_min:.0f}-{f_max:.0f} Hz",
        "Lambda_ML": float(Lambda_ml),
        "Fisher_error": float(sigma),
        "match_ML": float(match_ml),
        "match_GR": float(match_gr),
        "Delta_logL": float(delta_logL),
        "distance_from_zero_sigma": float(distance_from_zero),
        "logL_ML": float(logL_max),
        "logL_Lambda0": float(logL_zero),
        "n_frequency_bins": int(len(f)),
        "df_Hz": float(df),
    }, Lgrid, logL


# ======================================================================
# MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Stage 6A frequency-band robustness test."
    )

    parser.add_argument(
        "--h1",
        required=True,
        help="Path to H1 HDF5 strain file.",
    )

    parser.add_argument(
        "--event",
        default="GW150914",
        choices=list(EVENT_CATALOG.keys()),
    )

    parser.add_argument(
        "--fs",
        type=float,
        default=None,
        help="Sampling frequency if HDF5 metadata does not contain it.",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help="Analysis duration in seconds.",
    )

    parser.add_argument(
        "--grid-min",
        type=float,
        default=-5.0,
        help="Minimum Lambda for likelihood grid.",
    )

    parser.add_argument(
        "--grid-max",
        type=float,
        default=5.0,
        help="Maximum Lambda for likelihood grid.",
    )

    parser.add_argument(
        "--grid-points",
        type=int,
        default=1001,
        help="Number of Lambda grid points.",
    )

    args = parser.parse_args()

    h1_path = Path(args.h1)

    if not h1_path.exists():
        raise FileNotFoundError(
            f"H1 file does not exist:\n{h1_path}"
        )

    # --------------------------------------------------------------
    # Event parameters
    # --------------------------------------------------------------

    event = EVENT_CATALOG[args.event]

    m1 = event["m1"]
    m2 = event["m2"]
    distance_Mpc = event["distance_Mpc"]
    z = event["z"]

    K_z = cosmological_K_factor(z)

    gps_merger = event["gps_merger"]

    # --------------------------------------------------------------
    # Lambda grid
    # --------------------------------------------------------------

    Lambda_grid = np.linspace(
        args.grid_min,
        args.grid_max,
        args.grid_points,
    )

    # --------------------------------------------------------------
    # Load ONE fixed time-domain segment.
    #
    # This is critical:
    #
    # frequency robustness means changing only the frequency band.
    # We therefore keep the underlying time-domain data identical.
    # --------------------------------------------------------------

    strain_td, fs, seg_start = load_real_segment(
        str(h1_path),
        gps_merger,
        half_window=args.duration / 2.0 + 2.0,
        fs_override=args.fs,
    )

    # --------------------------------------------------------------
    # Header
    # --------------------------------------------------------------

    print()
    print("#" * 78)
    print("# STAGE 6A — FREQUENCY-BAND ROBUSTNESS")
    print("#" * 78)
    print()

    print(f"Input H1:              {h1_path}")
    print(f"Event:                 {args.event}")
    print(f"m1, m2:                {m1:.2f}, {m2:.2f} Msun")
    print(f"Distance:              {distance_Mpc:.1f} Mpc")
    print(f"Redshift:              {z}")
    print(f"Sampling rate:         {fs:.3f} Hz")
    print(f"Duration:              {args.duration:.3f} s")
    print(f"Reference Lambda:      {REFERENCE_LAMBDA:.4f}")
    print()
    print(
        "IMPORTANT: Lambda=-2.7800 is a reference value only."
    )
    print(
        "It is NOT used to select frequency bands or tune the likelihood."
    )
    print()

    print(
        "Underlying time-domain segment is FIXED across all bands."
    )
    print()

    # --------------------------------------------------------------
    # Output directory
    # --------------------------------------------------------------

    out = HERE / "results"
    out.mkdir(exist_ok=True)

    csv_path = out / "stage6A_frequency_robustness.csv"
    png_path = out / "stage6A_frequency_robustness.png"

    # --------------------------------------------------------------
    # Run all bands
    # --------------------------------------------------------------

    results = []
    curves = []

    for f_min, f_max in DEFAULT_BANDS:

        print("=" * 78)
        print(
            f"ANALYSIS BAND: {f_min:.0f}–{f_max:.0f} Hz"
        )
        print("=" * 78)

        result, Lgrid, logL = analyze_band(
            strain_td=strain_td,
            fs=fs,
            f_min=f_min,
            f_max=f_max,
            duration=args.duration,
            m1=m1,
            m2=m2,
            K_z=K_z,
            distance_Mpc=distance_Mpc,
            Lambda_grid=Lambda_grid,
        )

        results.append(result)
        curves.append(
            (
                result["band"],
                Lgrid.copy(),
                logL.copy(),
            )
        )

        print(
            f"Lambda_ML:              "
            f"{result['Lambda_ML']:.6f}"
        )

        if np.isfinite(result["Fisher_error"]):
            print(
                f"Fisher error:            "
                f"{result['Fisher_error']:.6f}"
            )
        else:
            print(
                "Fisher error:            undefined"
            )

        print(
            f"Match (ML):              "
            f"{result['match_ML']:.8f}"
        )

        print(
            f"Match (GR, Lambda=0):    "
            f"{result['match_GR']:.8f}"
        )

        print(
            f"Delta logL:              "
            f"{result['Delta_logL']:.8f}"
        )

        print(
            f"|Lambda_ML| / sigma:     "
            f"{result['distance_from_zero_sigma']:.4f}"
        )

        print(
            f"Frequency bins:          "
            f"{result['n_frequency_bins']}"
        )

        print()

    # ==================================================================
    # SUMMARY
    # ==================================================================

    print()
    print("#" * 78)
    print("# STAGE 6A SUMMARY")
    print("#" * 78)
    print()

    print(
        f"{'Band':>12} "
        f"{'Lambda_ML':>14} "
        f"{'sigma':>14} "
        f"{'match':>12} "
        f"{'DeltaLogL':>14} "
        f"{'|L|/sigma':>14}"
    )

    print("-" * 82)

    for r in results:

        print(
            f"{r['band']:>12} "
            f"{r['Lambda_ML']:>14.6f} "
            f"{r['Fisher_error']:>14.6f} "
            f"{r['match_ML']:>12.8f} "
            f"{r['Delta_logL']:>14.6f} "
            f"{r['distance_from_zero_sigma']:>14.4f}"
        )

    print()

    # ==================================================================
    # REFERENCE-VALUE COMPARISON
    # ==================================================================

    print("#" * 78)
    print("# REFERENCE VALUE CHECK")
    print("#" * 78)
    print()

    print(
        f"Previously obtained reference Lambda: "
        f"{REFERENCE_LAMBDA:.4f}"
    )

    print()
    print(
        "Distance of each independent ML estimate from the reference:"
    )

    for r in results:

        difference = (
            r["Lambda_ML"] - REFERENCE_LAMBDA
        )

        print(
            f"  {r['band']:>12}: "
            f"Lambda_ML - Lambda_reference = "
            f"{difference:+.6f}"
        )

    print()

    # ==================================================================
    # CSV
    # ==================================================================

    fieldnames = list(results[0].keys())

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as fp:

        writer = csv.DictWriter(
            fp,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for r in results:
            writer.writerow(r)

    print(f"Saved CSV -> {csv_path}")

    # ==================================================================
    # PLOT 1 — Lambda ML vs frequency band
    # ==================================================================

    x = np.arange(len(results))

    lambda_ml = np.array(
        [r["Lambda_ML"] for r in results]
    )

    lambda_sigma = np.array(
        [r["Fisher_error"] for r in results]
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.errorbar(
        x,
        lambda_ml,
        yerr=lambda_sigma,
        fmt="o-",
        capsize=5,
        markersize=7,
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.2,
        label="Lambda = 0",
    )

    ax.axhline(
        REFERENCE_LAMBDA,
        linestyle=":",
        linewidth=1.5,
        label="reference Lambda = -2.78",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [r["band"] for r in results]
    )

    ax.set_xlabel(
        "Analysis frequency band"
    )

    ax.set_ylabel(
        r"$\Lambda_{\rm ML}$"
    )

    ax.set_title(
        "Stage 6A — Frequency-band robustness"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        png_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved plot -> {png_path}")

    # ==================================================================
    # BASIC ROBUSTNESS DIAGNOSTIC
    # ==================================================================

    finite_ml = lambda_ml[
        np.isfinite(lambda_ml)
    ]

    if len(finite_ml) >= 2:

        spread = np.max(finite_ml) - np.min(finite_ml)

        mean_ml = np.mean(finite_ml)

        print()
        print("#" * 78)
        print("# BASIC FREQUENCY ROBUSTNESS DIAGNOSTIC")
        print("#" * 78)
        print()

        print(
            f"Mean Lambda_ML:         {mean_ml:.6f}"
        )

        print(
            f"Band-to-band spread:     {spread:.6f}"
        )

        print()

        print(
            "This diagnostic is descriptive only."
        )

        print(
            "It does NOT define a statistical detection threshold."
        )

        print(
            "It does NOT turn Stage 6A into a final Lambda constraint."
        )

    print()
    print("#" * 78)
    print("# STAGE 6A COMPLETE")
    print("#" * 78)
    print()


if __name__ == "__main__":
    main()