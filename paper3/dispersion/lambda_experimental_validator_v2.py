"""
PAPER 3 — EXPERIMENTAL LAMBDA VALIDATOR v2

Purpose
-------
Fit and compare dispersion models against externally supplied
experimental data.

Models
------
1. Linear:
       omega = a*k

2. Lambda dispersion:
       omega = c*k*sqrt(1 + 2*Lambda*k^2)

3. Linear + quartic:
       omega = a*k + b*k^4

The script reports:
    - fitted parameters
    - parameter uncertainty
    - R^2
    - RMSE
    - residual statistics
    - AIC
    - Delta AIC
    - model preference
    - CSV results
    - JSON statistics
    - SVG diagnostic plot

IMPORTANT
---------
This script does NOT automatically identify an arbitrary quartic
coefficient beta4 with Lambda.

A Lambda claim requires the Lambda-model itself to provide the
best-supported description of the measured dispersion.
"""

from pathlib import Path
import json
import math
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ================================================================
# PATHS
# ================================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_DATA = DATA_DIR / "experimental.csv"


# ================================================================
# CONSTANTS
# ================================================================

C_LIGHT = 299792458.0


# ================================================================
# MODELS
# ================================================================

def linear_model(k, a):
    return a * k


def lambda_model(k, c, lam):
    argument = 1.0 + 2.0 * lam * k**2

    # Prevent invalid square roots during fitting.
    argument = np.maximum(argument, 0.0)

    return c * k * np.sqrt(argument)


def linear_quartic_model(k, a, b):
    return a * k + b * k**4


# ================================================================
# BASIC STATISTICS
# ================================================================

def calculate_r2(y, y_pred):

    residual = y - y_pred

    ss_res = np.sum(residual**2)
    ss_tot = np.sum((y - np.mean(y))**2)

    if ss_tot == 0:
        return float("nan")

    return 1.0 - ss_res / ss_tot


def calculate_rmse(y, y_pred):

    residual = y - y_pred

    return float(
        np.sqrt(np.mean(residual**2))
    )


def calculate_aic(y, y_pred, n_parameters):

    n = len(y)

    residual = y - y_pred

    rss = np.sum(residual**2)

    # Protect against log(0)
    rss = max(rss, 1e-300)

    return float(
        n * np.log(rss / n)
        + 2 * n_parameters
    )


# ================================================================
# NUMERICAL FIT
# ================================================================

def least_squares_fit(
    model,
    k,
    y,
    initial,
    lower=None,
    upper=None,
):
    """
    Lightweight Gauss-Newton / numerical least-squares fitter.

    Avoids requiring scipy for the core pipeline.
    """

    params = np.array(
        initial,
        dtype=float,
    )

    if lower is None:
        lower = np.full_like(
            params,
            -np.inf,
        )

    if upper is None:
        upper = np.full_like(
            params,
            np.inf,
        )

    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    params = np.clip(
        params,
        lower,
        upper,
    )

    damping = 1e-6

    for _ in range(500):

        prediction = model(
            k,
            *params,
        )

        residual = y - prediction

        # Numerical Jacobian
        J = np.zeros(
            (len(k), len(params))
        )

        for j in range(len(params)):

            step = 1e-6 * max(
                abs(params[j]),
                1.0,
            )

            p2 = params.copy()
            p2[j] += step

            prediction2 = model(
                k,
                *p2,
            )

            J[:, j] = (
                prediction2 - prediction
            ) / step

        A = (
            J.T @ J
            + damping * np.eye(len(params))
        )

        b = J.T @ residual

        try:
            delta = np.linalg.solve(
                A,
                b,
            )
        except np.linalg.LinAlgError:
            break

        new_params = params + delta

        new_params = np.clip(
            new_params,
            lower,
            upper,
        )

        old_error = np.sum(
            residual**2
        )

        new_prediction = model(
            k,
            *new_params,
        )

        new_error = np.sum(
            (y - new_prediction)**2
        )

        if new_error < old_error:

            params = new_params

            damping *= 0.7

        else:

            damping *= 10.0

        if np.linalg.norm(delta) < 1e-10:

            break

    prediction = model(
        k,
        *params,
    )

    residual = y - prediction

    return params, prediction, residual


# ================================================================
# PARAMETER UNCERTAINTY
# ================================================================

def estimate_parameter_uncertainty(
    model,
    k,
    y,
    params,
):

    n = len(k)
    p = len(params)

    prediction = model(
        k,
        *params,
    )

    residual = y - prediction

    sigma2 = (
        np.sum(residual**2)
        / max(n - p, 1)
    )

    J = np.zeros(
        (n, p)
    )

    for j in range(p):

        step = 1e-6 * max(
            abs(params[j]),
            1.0,
        )

        p2 = params.copy()
        p2[j] += step

        prediction2 = model(
            k,
            *p2,
        )

        J[:, j] = (
            prediction2 - prediction
        ) / step

    try:

        covariance = (
            sigma2
            * np.linalg.inv(
                J.T @ J
            )
        )

        uncertainty = np.sqrt(
            np.maximum(
                np.diag(covariance),
                0.0,
            )
        )

    except np.linalg.LinAlgError:

        uncertainty = np.full(
            p,
            np.nan,
        )

    return uncertainty


# ================================================================
# LOAD DATA
# ================================================================

def load_data(path):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"Data file not found: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "k",
        "omega",
    }

    missing = required - set(
        df.columns
    )

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    df = df[
        ["k", "omega"]
    ].dropna()

    df = df.sort_values(
        "k"
    )

    if len(df) < 5:

        raise ValueError(
            "At least 5 experimental points are required."
        )

    return df


# ================================================================
# FIT ALL MODELS
# ================================================================

def fit_models(df):

    k = df["k"].to_numpy(
        dtype=float
    )

    omega = df["omega"].to_numpy(
        dtype=float
    )

    # ------------------------------------------------------------
    # Initial velocity estimate
    # ------------------------------------------------------------

    positive_k = k != 0

    if np.any(positive_k):

        c0 = np.median(
            omega[positive_k]
            / k[positive_k]
        )

    else:

        c0 = 1.0

    c0 = max(
        abs(c0),
        1e-12,
    )

    results = {}

    # ============================================================
    # LINEAR
    # ============================================================

    p, pred, residual = least_squares_fit(
        linear_model,
        k,
        omega,
        initial=[c0],
    )

    uncertainty = estimate_parameter_uncertainty(
        linear_model,
        k,
        omega,
        p,
    )

    results["linear"] = {
        "parameters": {
            "a": float(p[0]),
        },
        "uncertainty": {
            "a": float(uncertainty[0]),
        },
        "prediction": pred,
        "residual": residual,
        "r2": calculate_r2(
            omega,
            pred,
        ),
        "rmse": calculate_rmse(
            omega,
            pred,
        ),
        "rss": float(
            np.sum(residual**2)
        ),
        "aic": calculate_aic(
            omega,
            pred,
            1,
        ),
    }

    # ============================================================
    # LAMBDA MODEL
    # ============================================================

    p, pred, residual = least_squares_fit(
        lambda_model,
        k,
        omega,
        initial=[
            c0,
            0.0,
        ],
        lower=[
            0.0,
            0.0,
        ],
    )

    uncertainty = estimate_parameter_uncertainty(
        lambda_model,
        k,
        omega,
        p,
    )

    results["lambda"] = {
        "parameters": {
            "c": float(p[0]),
            "Lambda": float(p[1]),
        },
        "uncertainty": {
            "c": float(uncertainty[0]),
            "Lambda": float(uncertainty[1]),
        },
        "prediction": pred,
        "residual": residual,
        "r2": calculate_r2(
            omega,
            pred,
        ),
        "rmse": calculate_rmse(
            omega,
            pred,
        ),
        "rss": float(
            np.sum(residual**2)
        ),
        "aic": calculate_aic(
            omega,
            pred,
            2,
        ),
    }

    # ============================================================
    # LINEAR + QUARTIC
    # ============================================================

    p, pred, residual = least_squares_fit(
        linear_quartic_model,
        k,
        omega,
        initial=[
            c0,
            0.0,
        ],
    )

    uncertainty = estimate_parameter_uncertainty(
        linear_quartic_model,
        k,
        omega,
        p,
    )

    results["linear_quartic"] = {
        "parameters": {
            "a": float(p[0]),
            "beta4": float(p[1]),
        },
        "uncertainty": {
            "a": float(uncertainty[0]),
            "beta4": float(uncertainty[1]),
        },
        "prediction": pred,
        "residual": residual,
        "r2": calculate_r2(
            omega,
            pred,
        ),
        "rmse": calculate_rmse(
            omega,
            pred,
        ),
        "rss": float(
            np.sum(residual**2)
        ),
        "aic": calculate_aic(
            omega,
            pred,
            2,
        ),
    }

    return k, omega, results


# ================================================================
# REPORT
# ================================================================

def create_report(
    df,
    results,
    source_path,
):

    k = df["k"].to_numpy(
        dtype=float
    )

    omega = df["omega"].to_numpy(
        dtype=float
    )

    print()
    print("=" * 70)
    print("PAPER 3 — EXPERIMENTAL LAMBDA VALIDATOR v2")
    print("=" * 70)

    print()
    print("[1] DATA")
    print("-" * 70)

    print(
        f"Source: {source_path}"
    )

    print(
        f"Points: {len(df)}"
    )

    print(
        f"k range: "
        f"[{k.min():.6e}, {k.max():.6e}]"
    )

    print(
        f"omega range: "
        f"[{omega.min():.6e}, {omega.max():.6e}]"
    )

    print()
    print("[2] MODEL FITS")
    print("-" * 70)

    for name, result in results.items():

        print()
        print(name)

        for key, value in result[
            "parameters"
        ].items():

            uncertainty = result[
                "uncertainty"
            ][key]

            print(
                f"  {key:10s} = "
                f"{value:.10e} "
                f"+/- "
                f"{uncertainty:.3e}"
            )

        print(
            f"  R²   = "
            f"{result['r2']:.8f}"
        )

        print(
            f"  RMSE = "
            f"{result['rmse']:.8e}"
        )

        print(
            f"  RSS  = "
            f"{result['rss']:.8e}"
        )

        print(
            f"  AIC  = "
            f"{result['aic']:.8f}"
        )

    # ============================================================
    # AIC COMPARISON
    # ============================================================

    aic_values = {
        name: result["aic"]
        for name, result
        in results.items()
    }

    best_model = min(
        aic_values,
        key=aic_values.get,
    )

    print()
    print("[3] MODEL COMPARISON")
    print("-" * 70)

    for name, aic in aic_values.items():

        delta = (
            aic
            - aic_values[best_model]
        )

        print(
            f"{name:18s} "
            f"AIC = {aic:12.6f} "
            f"ΔAIC = {delta:12.6f}"
        )

    print()
    print(
        f"Preferred model by AIC: "
        f"{best_model}"
    )

    # ============================================================
    # LAMBDA RESULT
    # ============================================================

    lam = results[
        "lambda"
    ]["parameters"]["Lambda"]

    lam_sigma = results[
        "lambda"
    ]["uncertainty"]["Lambda"]

    lambda_aic = results[
        "lambda"
    ]["aic"]

    delta_aic_lambda = (
        lambda_aic
        - aic_values[best_model]
    )

    print()
    print("[4] LAMBDA RESULT")
    print("-" * 70)

    print(
        f"Lambda = "
        f"{lam:.10e}"
    )

    print(
        f"Uncertainty = "
        f"{lam_sigma:.3e}"
    )

    print(
        f"Lambda-model R² = "
        f"{results['lambda']['r2']:.8f}"
    )

    print(
        f"Lambda-model ΔAIC = "
        f"{delta_aic_lambda:.6f}"
    )

    # ============================================================
    # INTERPRETATION
    # ============================================================

    print()
    print("[5] INTERPRETATION")
    print("-" * 70)

    if best_model == "lambda":

        print(
            "Λ-model is statistically preferred by AIC "
            "for this dataset."
        )

        print(
            "This supports Λ as a fitted dispersion "
            "parameter for this dataset."
        )

    elif best_model == "linear_quartic":

        print(
            "Linear + quartic model is preferred."
        )

        print(
            "The dataset contains quartic dispersion, "
            "but this result alone does NOT establish "
            "the Λ-model."
        )

    else:

        print(
            "Linear dispersion is preferred."
        )

        print(
            "No statistically supported Λ correction "
            "is identified by this dataset."
        )

    print()
    print(
        "IMPORTANT: model preference is not equivalent "
        "to experimental proof of the underlying theory."
    )

    # ============================================================
    # SAVE CSV
    # ============================================================

    csv_rows = []

    for name, result in results.items():

        row = {
            "model": name,
            "r2": result["r2"],
            "rmse": result["rmse"],
            "rss": result["rss"],
            "aic": result["aic"],
        }

        for key, value in result[
            "parameters"
        ].items():

            row[
                f"parameter_{key}"
            ] = value

            row[
                f"uncertainty_{key}"
            ] = result[
                "uncertainty"
            ][key]

        csv_rows.append(row)

    csv_path = (
        RESULTS_DIR
        / "lambda_validation_results.csv"
    )

    pd.DataFrame(
        csv_rows
    ).to_csv(
        csv_path,
        index=False,
    )

    # ============================================================
    # JSON
    # ============================================================

    json_data = {
        "source": str(
            Path(source_path)
        ),
        "n_points": len(df),
        "best_model": best_model,
        "models": {},
    }

    for name, result in results.items():

        json_data[
            "models"
        ][name] = {
            "parameters": result[
                "parameters"
            ],
            "uncertainty": result[
                "uncertainty"
            ],
            "r2": result["r2"],
            "rmse": result["rmse"],
            "rss": result["rss"],
            "aic": result["aic"],
        }

    json_path = (
        RESULTS_DIR
        / "lambda_validation_statistics.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_data,
            f,
            indent=2,
        )

    # ============================================================
    # SVG
    # ============================================================

    k_plot = np.linspace(
        k.min(),
        k.max(),
        500,
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.scatter(
        k,
        omega,
        label="Experimental data",
        s=30,
    )

    plt.plot(
        k_plot,
        linear_model(
            k_plot,
            results["linear"][
                "parameters"
            ]["a"],
        ),
        label="Linear",
    )

    plt.plot(
        k_plot,
        lambda_model(
            k_plot,
            results["lambda"][
                "parameters"
            ]["c"],
            results["lambda"][
                "parameters"
            ]["Lambda"],
        ),
        label="Λ-model",
    )

    plt.plot(
        k_plot,
        linear_quartic_model(
            k_plot,
            results[
                "linear_quartic"
            ]["parameters"]["a"],
            results[
                "linear_quartic"
            ]["parameters"]["beta4"],
        ),
        label="Linear + quartic",
    )

    plt.xlabel("k")
    plt.ylabel("ω")

    plt.title(
        "Paper 3 — Experimental Dispersion Model Comparison"
    )

    plt.legend()
    plt.grid(True)

    svg_path = (
        RESULTS_DIR
        / "lambda_validation.svg"
    )

    plt.tight_layout()

    plt.savefig(
        svg_path,
        format="svg",
    )

    plt.close()

    # ============================================================
    # FINAL
    # ============================================================

    print()
    print("[6] FILES")
    print("-" * 70)

    print(
        f"CSV:  {csv_path}"
    )

    print(
        f"JSON: {json_path}"
    )

    print(
        f"SVG:  {svg_path}"
    )

    print()
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


# ================================================================
# MAIN
# ================================================================

def main():

    if len(sys.argv) > 1:

        data_path = Path(
            sys.argv[1]
        )

    else:

        data_path = DEFAULT_DATA

    print()
    print("=" * 70)
    print("PAPER 3 — EXPERIMENTAL LAMBDA VALIDATOR v2")
    print("=" * 70)

    print(
        f"Data file: {data_path}"
    )

    try:

        df = load_data(
            data_path
        )

        k, omega, results = fit_models(
            df
        )

        create_report(
            df,
            results,
            data_path,
        )

    except Exception as exc:

        print()
        print("ERROR")
        print("-" * 70)
        print(str(exc))
        print()

        raise


if __name__ == "__main__":
    main()