"""
Paper 3 — Real Experimental Dataset Validator

Independent model comparison for externally sourced dispersion data.

Models
------
1. Linear:
       omega = a*k

2. Lambda:
       omega = c*k*sqrt(1 + 2*Lambda*k^2)

3. Linear + quartic:
       omega = a*k + beta4*k^4

Outputs
-------
CSV
JSON
SVG

IMPORTANT
---------
A statistically preferred model is not automatically evidence
for the underlying physical theory.
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------

def model_linear(k, a):
    return a * k


def model_lambda(k, c, Lambda):
    return c * k * np.sqrt(
        1.0 + 2.0 * Lambda * k**2
    )


def model_linear_quartic(k, a, beta4):
    return a * k + beta4 * k**4


# ---------------------------------------------------------------------
# STATISTICS
# ---------------------------------------------------------------------

def calculate_statistics(y, prediction, n_parameters):

    residuals = y - prediction

    rss = float(
        np.sum(residuals**2)
    )

    n = len(y)

    rmse = float(
        np.sqrt(rss / n)
    )

    tss = float(
        np.sum((y - np.mean(y))**2)
    )

    if tss > 0:
        r2 = float(
            1.0 - rss / tss
        )
    else:
        r2 = float("nan")

    if rss > 0:
        aic = float(
            n * np.log(rss / n)
            + 2 * n_parameters
        )
    else:
        aic = float("-inf")

    return {
        "RSS": rss,
        "RMSE": rmse,
        "R2": r2,
        "AIC": aic,
    }


# ---------------------------------------------------------------------
# FITTING
# ---------------------------------------------------------------------

def fit_model(
    name,
    func,
    k,
    omega,
    sigma=None,
    p0=None,
):

    if sigma is None:

        popt, pcov = curve_fit(
            func,
            k,
            omega,
            p0=p0,
            maxfev=100000,
        )

    else:

        popt, pcov = curve_fit(
            func,
            k,
            omega,
            sigma=sigma,
            absolute_sigma=True,
            p0=p0,
            maxfev=100000,
        )

    prediction = func(
        k,
        *popt,
    )

    stats = calculate_statistics(
        omega,
        prediction,
        len(popt),
    )

    errors = np.sqrt(
        np.maximum(
            np.diag(pcov),
            0.0,
        )
    )

    return {
        "name": name,
        "parameters": popt,
        "uncertainties": errors,
        "prediction": prediction,
        **stats,
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Paper 3 experimental Lambda validator"
        )
    )

    parser.add_argument(
        "csv",
        help=(
            "CSV containing k,omega "
            "and optionally sigma"
        ),
    )

    args = parser.parse_args()

    path = Path(args.csv)

    if not path.exists():
        raise FileNotFoundError(path)

    print("=" * 70)
    print("PAPER 3 — REAL EXPERIMENTAL LAMBDA VALIDATOR")
    print("=" * 70)

    # ---------------------------------------------------------------
    # DATA
    # ---------------------------------------------------------------

    df = pd.read_csv(path)

    required = {"k", "omega"}

    if not required.issubset(df.columns):
        raise ValueError(
            "CSV must contain columns: k, omega"
        )

    df = df.dropna(
        subset=["k", "omega"]
    ).copy()

    df = df.sort_values("k")

    k = df["k"].to_numpy(
        dtype=float
    )

    omega = df["omega"].to_numpy(
        dtype=float
    )

    sigma = None

    if "sigma" in df.columns:

        sigma = df["sigma"].to_numpy(
            dtype=float
        )

    print()
    print("[1] DATA")
    print("-" * 70)

    print(f"Source: {path}")
    print(f"Points: {len(df)}")
    print(
        f"k range: "
        f"[{k.min():.6e}, {k.max():.6e}]"
    )
    print(
        f"omega range: "
        f"[{omega.min():.6e}, {omega.max():.6e}]"
    )

    # ---------------------------------------------------------------
    # FITS
    # ---------------------------------------------------------------

    print()
    print("[2] MODEL FITS")
    print("-" * 70)

    fits = []

    fits.append(
        fit_model(
            "linear",
            model_linear,
            k,
            omega,
            sigma=sigma,
            p0=[1.0],
        )
    )

    fits.append(
        fit_model(
            "lambda",
            model_lambda,
            k,
            omega,
            sigma=sigma,
            p0=[1.0, 0.01],
        )
    )

    fits.append(
        fit_model(
            "linear_quartic",
            model_linear_quartic,
            k,
            omega,
            sigma=sigma,
            p0=[1.0, 0.01],
        )
    )

    for result in fits:

        print()
        print(result["name"])

        for value, error in zip(
            result["parameters"],
            result["uncertainties"],
        ):

            print(
                f"  parameter = "
                f"{value:.10e} "
                f"+/- {error:.3e}"
            )

        print(
            f"  R²   = {result['R2']:.8f}"
        )

        print(
            f"  RMSE = {result['RMSE']:.8e}"
        )

        print(
            f"  RSS  = {result['RSS']:.8e}"
        )

        print(
            f"  AIC  = {result['AIC']:.8f}"
        )

    # ---------------------------------------------------------------
    # AIC
    # ---------------------------------------------------------------

    aics = np.array([
        x["AIC"]
        for x in fits
    ])

    best_index = int(
        np.argmin(aics)
    )

    best = fits[best_index]

    print()
    print("[3] MODEL COMPARISON")
    print("-" * 70)

    for result in fits:

        delta_aic = (
            result["AIC"]
            - best["AIC"]
        )

        print(
            f"{result['name']:<18}"
            f" AIC = {result['AIC']:12.6f}"
            f" ΔAIC = {delta_aic:12.6f}"
        )

    print()
    print(
        f"Preferred model by AIC: "
        f"{best['name']}"
    )

    # ---------------------------------------------------------------
    # LAMBDA
    # ---------------------------------------------------------------

    lambda_result = next(
        x for x in fits
        if x["name"] == "lambda"
    )

    Lambda = float(
        lambda_result["parameters"][1]
    )

    Lambda_sigma = float(
        lambda_result["uncertainties"][1]
    )

    delta_aic_lambda = (
        lambda_result["AIC"]
        - best["AIC"]
    )

    print()
    print("[4] LAMBDA")
    print("-" * 70)

    print(
        f"Lambda = "
        f"{Lambda:.10e}"
    )

    print(
        f"Uncertainty = "
        f"{Lambda_sigma:.3e}"
    )

    print(
        f"Lambda R² = "
        f"{lambda_result['R2']:.8f}"
    )

    print(
        f"Lambda ΔAIC = "
        f"{delta_aic_lambda:.6f}"
    )

    # ---------------------------------------------------------------
    # RESIDUALS
    # ---------------------------------------------------------------

    print()
    print("[5] RESIDUALS")
    print("-" * 70)

    output_df = df.copy()

    for result in fits:

        name = result["name"]

        output_df[
            f"{name}_prediction"
        ] = result["prediction"]

        output_df[
            f"{name}_residual"
        ] = (
            omega
            - result["prediction"]
        )

    # ---------------------------------------------------------------
    # SAVE CSV
    # ---------------------------------------------------------------

    output_csv = (
        RESULTS_DIR
        / "real_dataset_validation.csv"
    )

    output_df.to_csv(
        output_csv,
        index=False,
    )

    # ---------------------------------------------------------------
    # JSON
    # ---------------------------------------------------------------

    statistics = {
        "source": str(path),
        "points": len(df),
        "models": {},
        "preferred_model": best["name"],
    }

    for result in fits:

        statistics["models"][
            result["name"]
        ] = {
            "parameters": [
                float(x)
                for x in result["parameters"]
            ],
            "uncertainties": [
                float(x)
                for x in result["uncertainties"]
            ],
            "R2": result["R2"],
            "RMSE": result["RMSE"],
            "RSS": result["RSS"],
            "AIC": result["AIC"],
        }

    statistics[
        "lambda_result"
    ] = {
        "Lambda": Lambda,
        "uncertainty": Lambda_sigma,
        "delta_AIC": delta_aic_lambda,
    }

    output_json = (
        RESULTS_DIR
        / "real_dataset_validation.json"
    )

    output_json.write_text(
        json.dumps(
            statistics,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------
    # SVG
    # ---------------------------------------------------------------

    x = np.linspace(
        k.min(),
        k.max(),
        500,
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        k,
        omega,
        label="Experimental data",
        zorder=5,
    )

    for result in fits:

        y = {
            "linear": model_linear,
            "lambda": model_lambda,
            "linear_quartic":
                model_linear_quartic,
        }[result["name"]](
            x,
            *result["parameters"],
        )

        plt.plot(
            x,
            y,
            label=result["name"],
        )

    plt.xlabel("k")
    plt.ylabel("ω")
    plt.title(
        "Paper 3 — Experimental Dispersion Model Comparison"
    )

    plt.legend()
    plt.grid(True)

    output_svg = (
        RESULTS_DIR
        / "real_dataset_validation.svg"
    )

    plt.tight_layout()
    plt.savefig(
        output_svg,
        format="svg",
    )

    plt.close()

    # ---------------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------------

    print()
    print("[6] FILES")
    print("-" * 70)

    print(
        f"CSV:  {output_csv}"
    )

    print(
        f"JSON: {output_json}"
    )

    print(
        f"SVG:  {output_svg}"
    )

    print()
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    print()
    print(
        "IMPORTANT: statistical model preference "
        "does not by itself establish the physical "
        "validity of the Lambda theory."
    )


if __name__ == "__main__":
    main()