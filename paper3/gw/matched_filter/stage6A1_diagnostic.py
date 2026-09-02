"""
stage6A1_diagnostic.py
======================

STAGE 6A.1 — DIAGNOSTIC OF STAGE 6A FREQUENCY ROBUSTNESS

Purpose
-------
Diagnose the Stage 6A result before proceeding to Stage 6B.

This script deliberately preserves the existing:
    waveform.py
    likelihood.py
    Stage-3 data loading / FFT / PSD pipeline

It does NOT change the Lambda physics.

Main diagnostic questions
--------------------------
1. Is Lambda = -5 merely a lower-grid boundary artifact?
2. Does the likelihood continue increasing toward more negative Lambda?
3. Does Lambda = -2.78 remain competitive?
4. Does Stage 6A reproduce the existing likelihood machinery?
5. Are the normalized matches extremely small for all relevant Lambda
   values?

Lambda grid
-----------
Diagnostic grid:
    -20 <= Lambda <= +5

This wider grid is NOT an inference choice. It is specifically used to
determine whether the previous [-5,+5] grid artificially truncated the
likelihood maximum.

Reference value
---------------
    Lambda_reference = -2.7800

This value is ONLY evaluated diagnostically.
It is never used to select a frequency band or tune the fit.

Outputs
-------
    results/stage6A1_diagnostic.csv
    results/stage6A1_likelihood_curves.png
"""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
# IMPORT EXISTING PIPELINE
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
# CONSTANTS
# ======================================================================

REFERENCE_LAMBDA = -2.7800

BANDS = [
    (20.0, 100.0),
    (20.0, 150.0),
    (20.0, 200.0),
    (30.0, 200.0),
    (30.0, 300.0),
]

# Diagnostic grid deliberately wider than Stage 6A.
GRID_MIN = -20.0
GRID_MAX = 5.0
GRID_POINTS = 2501


# ======================================================================
# MATCH
# ======================================================================

def normalized_match(
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
    Normalized noise-weighted overlap:

        <d|h> / sqrt(<d|d><h|h>)

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

    return float(dh / np.sqrt(dd * hh))


# ======================================================================
# LIKELIHOOD AT EXACT LAMBDA
# ======================================================================

def evaluate_lambda(
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
    Evaluate exactly one Lambda value using the EXISTING likelihood.py.
    """

    return float(
        log_likelihood(
            data_fd,
            f,
            psd,
            df,
            m1,
            m2,
            Lambda,
            K_z,
            distance_Mpc=distance_Mpc,
        )
    )


# ======================================================================
# LOCAL CURVATURE
# ======================================================================

def local_fisher_error(Lambda_grid, logL):
    """
    Same local three-point curvature prescription used by likelihood.py.

    Returns:
        sigma, status

    status:
        interior
        lower_boundary
        upper_boundary
        invalid_curvature
    """

    i = int(np.argmax(logL))

    if i == 0:
        return np.nan, "lower_boundary"

    if i == len(Lambda_grid) - 1:
        return np.nan, "upper_boundary"

    dL = Lambda_grid[1] - Lambda_grid[0]

    d2 = (
        logL[i + 1]
        - 2.0 * logL[i]
        + logL[i - 1]
    ) / dL**2

    if d2 >= 0.0:
        return np.nan, "invalid_curvature"

    return float(np.sqrt(-1.0 / d2)), "interior"


# ======================================================================
# SINGLE BAND
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
    Run the existing likelihood machinery on one frequency band.
    """

    # --------------------------------------------------------------
    # Frequency-domain data
    # --------------------------------------------------------------

    f, data_fd, df = to_frequency_domain(
        strain_td,
        fs,
        f_min,
        f_max,
        duration,
    )

    # --------------------------------------------------------------
    # PSD
    # --------------------------------------------------------------

    psd = estimate_psd_from_segment(
        strain_td,
        fs,
        f,
    )

    # --------------------------------------------------------------
    # Existing grid search
    # --------------------------------------------------------------

    Lgrid, logL, Lambda_ml, Lambda_err = grid_search_lambda(
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
    # Maximum likelihood
    # --------------------------------------------------------------

    i_max = int(np.argmax(logL))

    Lambda_ml = float(Lgrid[i_max])
    logL_ml = float(logL[i_max])

    # --------------------------------------------------------------
    # Exact Lambda = 0
    # --------------------------------------------------------------

    logL_zero = evaluate_lambda(
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
    # Exact Lambda = -2.78
    # --------------------------------------------------------------

    logL_reference = evaluate_lambda(
        data_fd,
        f,
        psd,
        df,
        m1,
        m2,
        K_z,
        REFERENCE_LAMBDA,
        distance_Mpc,
    )

    # --------------------------------------------------------------
    # Likelihood differences
    # --------------------------------------------------------------

    delta_logL_ml_vs_zero = logL_ml - logL_zero

    delta_logL_reference_vs_zero = (
        logL_reference - logL_zero
    )

    delta_logL_ml_vs_reference = (
        logL_ml - logL_reference
    )

    # --------------------------------------------------------------
    # Matches
    # --------------------------------------------------------------

    match_zero = normalized_match(
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

    match_reference = normalized_match(
        data_fd,
        f,
        psd,
        df,
        m1,
        m2,
        K_z,
        REFERENCE_LAMBDA,
        distance_Mpc,
    )

    match_ml = normalized_match(
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
    # Fisher diagnostic
    # --------------------------------------------------------------

    sigma, status = local_fisher_error(
        Lgrid,
        logL,
    )

    # --------------------------------------------------------------
    # Grid-edge distance
    # --------------------------------------------------------------

    lower_boundary_distance = Lambda_ml - GRID_MIN
    upper_boundary_distance = GRID_MAX - Lambda_ml

    # --------------------------------------------------------------
    # Likelihood monotonicity near lower edge
    # --------------------------------------------------------------

    n_edge = min(50, len(logL) - 1)

    lower_edge_slope = np.polyfit(
        Lgrid[:n_edge],
        logL[:n_edge],
        1,
    )[0]

    # --------------------------------------------------------------
    # Return
    # --------------------------------------------------------------

    result = {
        "band": f"{f_min:.0f}-{f_max:.0f} Hz",
        "f_min_Hz": f_min,
        "f_max_Hz": f_max,

        "Lambda_ML": Lambda_ml,

        "Fisher_error": (
            float(sigma)
            if np.isfinite(sigma)
            else np.nan
        ),

        "maximum_status": status,

        "logL_ML": logL_ml,
        "logL_Lambda0": logL_zero,
        "logL_Lambda_reference": logL_reference,

        "DeltaLogL_ML_vs_0":
            delta_logL_ml_vs_zero,

        "DeltaLogL_ref_vs_0":
            delta_logL_reference_vs_zero,

        "DeltaLogL_ML_vs_ref":
            delta_logL_ml_vs_reference,

        "match_ML": match_ml,
        "match_Lambda0": match_zero,
        "match_reference": match_reference,

        "lower_grid": GRID_MIN,
        "upper_grid": GRID_MAX,

        "lower_boundary_distance":
            lower_boundary_distance,

        "lower_edge_logL_slope":
            lower_edge_slope,

        "n_frequency_bins":
            len(f),

        "df_Hz":
            df,
    }

    return result, Lgrid, logL


# ======================================================================
# MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Stage 6A.1 diagnostic."
    )

    parser.add_argument(
        "--h1",
        required=True,
        help="Path to H1 GW150914 HDF5 strain file.",
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
        help="Sampling frequency override.",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
    )

    args = parser.parse_args()

    h1_path = Path(args.h1)

    if not h1_path.exists():
        raise FileNotFoundError(
            f"H1 file not found:\n{h1_path}"
        )

    event = EVENT_CATALOG[args.event]

    m1 = event["m1"]
    m2 = event["m2"]
    distance_Mpc = event["distance_Mpc"]
    z = event["z"]
    gps_merger = event["gps_merger"]

    K_z = cosmological_K_factor(z)

    Lambda_grid = np.linspace(
        GRID_MIN,
        GRID_MAX,
        GRID_POINTS,
    )

    # --------------------------------------------------------------
    # One fixed time-domain segment
    # --------------------------------------------------------------

    strain_td, fs, seg_start = load_real_segment(
        str(h1_path),
        gps_merger,
        half_window=args.duration / 2.0 + 2.0,
        fs_override=args.fs,
    )

    # --------------------------------------------------------------
    # Output directory
    # --------------------------------------------------------------

    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)

    csv_path = (
        results_dir /
        "stage6A1_diagnostic.csv"
    )

    plot_path = (
        results_dir /
        "stage6A1_likelihood_curves.png"
    )

    # --------------------------------------------------------------
    # Header
    # --------------------------------------------------------------

    print()
    print("#" * 78)
    print("# STAGE 6A.1 — Λ LIKELIHOOD DIAGNOSTIC")
    print("#" * 78)
    print()

    print(f"H1:                    {h1_path}")
    print(f"Event:                 {args.event}")
    print(f"m1:                    {m1:.2f} Msun")
    print(f"m2:                    {m2:.2f} Msun")
    print(f"Distance:              {distance_Mpc:.2f} Mpc")
    print(f"Redshift:              {z:.5f}")
    print(f"fs:                    {fs:.3f} Hz")
    print(f"Duration:              {args.duration:.3f} s")
    print()
    print(
        f"Diagnostic Lambda grid: "
        f"[{GRID_MIN:.1f}, {GRID_MAX:.1f}]"
    )
    print(
        f"Reference Lambda:       "
        f"{REFERENCE_LAMBDA:.4f}"
    )
    print()
    print(
        "The reference value is evaluated only as a diagnostic."
    )
    print(
        "It is NOT used to select or tune the grid."
    )
    print()

    # --------------------------------------------------------------
    # Run
    # --------------------------------------------------------------

    results = []
    curves = []

    for f_min, f_max in BANDS:

        print("=" * 78)
        print(
            f"DIAGNOSTIC BAND: "
            f"{f_min:.0f}–{f_max:.0f} Hz"
        )
        print("=" * 78)

        result, Lgrid, logL = analyze_band(
            strain_td,
            fs,
            f_min,
            f_max,
            args.duration,
            m1,
            m2,
            K_z,
            distance_Mpc,
            Lambda_grid,
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
            f"Lambda_ML:               "
            f"{result['Lambda_ML']:.6f}"
        )

        print(
            f"Maximum status:           "
            f"{result['maximum_status']}"
        )

        print(
            f"Fisher error:             "
            f"{result['Fisher_error']}"
        )

        print()
        print(
            f"logL(Lambda=0):           "
            f"{result['logL_Lambda0']:.8f}"
        )

        print(
            f"logL(Lambda=-2.78):       "
            f"{result['logL_Lambda_reference']:.8f}"
        )

        print(
            f"logL(Lambda_ML):          "
            f"{result['logL_ML']:.8f}"
        )

        print()
        print(
            f"DeltaLogL ML vs 0:        "
            f"{result['DeltaLogL_ML_vs_0']:.8f}"
        )

        print(
            f"DeltaLogL -2.78 vs 0:    "
            f"{result['DeltaLogL_ref_vs_0']:.8f}"
        )

        print(
            f"DeltaLogL ML vs -2.78:   "
            f"{result['DeltaLogL_ML_vs_ref']:.8f}"
        )

        print()
        print(
            f"match(Lambda=0):          "
            f"{result['match_Lambda0']:.8e}"
        )

        print(
            f"match(Lambda=-2.78):      "
            f"{result['match_reference']:.8e}"
        )

        print(
            f"match(Lambda_ML):         "
            f"{result['match_ML']:.8e}"
        )

        print()
        print(
            f"Lower-edge logL slope:    "
            f"{result['lower_edge_logL_slope']:.8e}"
        )

        print(
            f"Frequency bins:           "
            f"{result['n_frequency_bins']}"
        )

        print()

    # ==================================================================
    # SUMMARY TABLE
    # ==================================================================

    print()
    print("#" * 78)
    print("# STAGE 6A.1 SUMMARY")
    print("#" * 78)
    print()

    print(
        f"{'Band':>12} "
        f"{'Lambda_ML':>12} "
        f"{'sigma':>12} "
        f"{'status':>18} "
        f"{'ΔlogL ML/0':>14} "
        f"{'ΔlogL ref/0':>14}"
    )

    print("-" * 100)

    for r in results:

        print(
            f"{r['band']:>12} "
            f"{r['Lambda_ML']:>12.5f} "
            f"{r['Fisher_error']:>12.5f} "
            f"{r['maximum_status']:>18} "
            f"{r['DeltaLogL_ML_vs_0']:>14.6f} "
            f"{r['DeltaLogL_ref_vs_0']:>14.6f}"
        )

    # ==================================================================
    # INTERPRETATION FLAGS
    # ==================================================================

    print()
    print("#" * 78)
    print("# DIAGNOSTIC FLAGS")
    print("#" * 78)
    print()

    for r in results:

        if r["maximum_status"] == "lower_boundary":

            print(
                f"[BOUNDARY] {r['band']}: "
                f"ML is at Lambda={GRID_MIN:.1f}. "
                f"The previous [-5,+5] grid was insufficient."
            )

        elif r["maximum_status"] == "interior":

            print(
                f"[INTERIOR] {r['band']}: "
                f"ML={r['Lambda_ML']:.5f} "
                f"with finite local curvature."
            )

        elif r["maximum_status"] == "upper_boundary":

            print(
                f"[BOUNDARY] {r['band']}: "
                f"ML reached upper grid boundary."
            )

        else:

            print(
                f"[CURVATURE] {r['band']}: "
                f"ML exists but local Fisher curvature "
                f"is not valid."
            )

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

    print()
    print(
        f"Saved diagnostic CSV -> {csv_path}"
    )

    # ==================================================================
    # LIKELIHOOD CURVES
    # ==================================================================

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    for band, Lgrid, logL in curves:

        relative_logL = logL - np.max(logL)

        ax.plot(
            Lgrid,
            relative_logL,
            label=band,
            linewidth=1.5,
        )

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.2,
        label="Lambda = 0",
    )

    ax.axvline(
        REFERENCE_LAMBDA,
        linestyle=":",
        linewidth=1.5,
        label="reference Lambda = -2.78",
    )

    ax.axvline(
        GRID_MIN,
        linestyle="-.",
        linewidth=1.0,
        label="diagnostic lower boundary",
    )

    ax.set_xlabel(
        r"$\Lambda$"
    )

    ax.set_ylabel(
        r"$\ln L(\Lambda)-\max[\ln L]$"
    )

    ax.set_title(
        "Stage 6A.1 — Lambda likelihood curves"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        plot_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved likelihood plot -> {plot_path}"
    )

    # ==================================================================
    # FINAL MESSAGE
    # ==================================================================

    print()
    print("#" * 78)
    print("# STAGE 6A.1 COMPLETE")
    print("#" * 78)
    print()

    print(
        "Interpretation rule:"
    )

    print(
        "If Lambda_ML remains at the lower boundary, "
        "do NOT quote it as a measured Lambda."
    )

    print(
        "If an interior maximum appears, inspect the "
        "likelihood curve and compare it with Lambda=-2.78."
    )

    print(
        "Do not proceed to Stage 6B until these diagnostics "
        "have been inspected."
    )

    print()


if __name__ == "__main__":
    main()