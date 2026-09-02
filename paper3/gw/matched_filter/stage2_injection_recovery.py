"""
stage2_injection_recovery.py
============================

STAGE 2 — SYNTHETIC LAMBDA INJECTION / RECOVERY

Purpose
-------
Test whether Lambda is identifiable by the EXISTING validated
waveform.py + likelihood.py pipeline, without using real GW150914
strain.

This deliberately isolates Lambda identifiability from real-signal
waveform mismatch.

For each injected Lambda_true:

    synthetic data = h(Lambda_true) + Gaussian noise

and recovery is performed with the SAME waveform model and the SAME
likelihood/grid-search implementation.

Tests:
    Lambda_true = 0
    Lambda_true = -1
    Lambda_true = -2
    Lambda_true = -2.78
    Lambda_true = -4

Diagnostics:
    Lambda_ML
    recovery error
    Fisher error
    match at Lambda_ML
    match at Lambda_true
    match at Lambda=0
    Delta logL (ML - GR)
    Delta logL (true - GR)
    boundary fraction

IMPORTANT
---------
Lambda=-2.78 is NOT used to tune the grid or select any band.
It is simply one of the pre-declared injection values.

The frequency band is fixed for the primary test.

This is an identifiability test, NOT an observational constraint.
"""

import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from waveform import (
    waveform_frequency_domain,
    cosmological_K_factor,
)

from likelihood import (
    aligo_like_psd,
    grid_search_lambda,
    noise_weighted_inner_product,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

M1 = 35.6
M2 = 30.6

DISTANCE_MPC = 440.0
Z = 0.09

TC = 0.0
PHI_C = 0.0

# Fixed primary analysis band.
F_LO = 20.0
F_HI = 300.0

# The grid is deliberately wider than the reference value.
LAMBDA_GRID = np.linspace(-5.0, 5.0, 201)

# Multiple realizations are important: one realization is not enough
# to characterize estimator behavior.
N_REALIZATIONS = 16

# Explicitly declared injections.
INJECTED_LAMBDAS = np.array([
    0.0,
    -1.0,
    -2.0,
    -2.78,
    -4.0,
])

BASE_SEED = 260830


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def compute_match(h1, h2, f, psd, df):
    """
    Normalized noise-weighted overlap:

        match = <h1|h2> /
                sqrt(<h1|h1><h2|h2>)

    No time/phase maximization is performed here because the synthetic
    injection and recovery use identical tc and phi_c by construction.
    """
    h1h1 = noise_weighted_inner_product(h1, h1, f, psd, df)
    h2h2 = noise_weighted_inner_product(h2, h2, f, psd, df)

    if h1h1 <= 0.0 or h2h2 <= 0.0:
        return np.nan

    overlap = noise_weighted_inner_product(h1, h2, f, psd, df)

    return overlap / np.sqrt(h1h1 * h2h2)


def make_frequency_grid(f_lo, f_hi, duration):
    """
    Frequency resolution is fixed by the synthetic observation duration:

        df = 1 / duration
    """
    df = 1.0 / duration
    f = np.arange(f_lo, f_hi, df)

    return f, df


def generate_complex_noise(psd, df, rng):
    """
    Generate complex frequency-domain Gaussian noise consistent with
    the convention used by the existing Stage 5G null-test code:

        sigma = sqrt(Sn / (4 df))

        n = N_real + i N_imag
    """
    sigma = np.sqrt(psd / (4.0 * df))

    return (
        rng.normal(0.0, sigma)
        + 1j * rng.normal(0.0, sigma)
    )


def recover_single_realization(
    f,
    psd,
    df,
    K_z,
    Lambda_true,
    rng,
):
    """
    Inject one Lambda value and recover it using the existing
    grid_search_lambda() implementation.
    """

    # -------------------------------------------------------------
    # Injection
    # -------------------------------------------------------------

    h_true = waveform_frequency_domain(
        f,
        M1,
        M2,
        Lambda_true,
        K_z,
        TC,
        PHI_C,
        DISTANCE_MPC,
    )

    noise = generate_complex_noise(psd, df, rng)

    data = h_true + noise

    # -------------------------------------------------------------
    # Recovery
    # -------------------------------------------------------------

    grid, logL, Lambda_ml, Lambda_err = grid_search_lambda(
        data,
        f,
        psd,
        df,
        M1,
        M2,
        K_z,
        LAMBDA_GRID,
        TC,
        PHI_C,
        DISTANCE_MPC,
    )

    i_ml = int(np.argmax(logL))

    # -------------------------------------------------------------
    # Templates for diagnostic matches
    # -------------------------------------------------------------

    h_ml = waveform_frequency_domain(
        f,
        M1,
        M2,
        Lambda_ml,
        K_z,
        TC,
        PHI_C,
        DISTANCE_MPC,
    )

    h_gr = waveform_frequency_domain(
        f,
        M1,
        M2,
        0.0,
        K_z,
        TC,
        PHI_C,
        DISTANCE_MPC,
    )

    # -------------------------------------------------------------
    # Match diagnostics
    # -------------------------------------------------------------

    match_ml = compute_match(
        h_true,
        h_ml,
        f,
        psd,
        df,
    )

    match_true = compute_match(
        h_true,
        h_true,
        f,
        psd,
        df,
    )

    match_gr = compute_match(
        h_true,
        h_gr,
        f,
        psd,
        df,
    )

    # -------------------------------------------------------------
    # Likelihood diagnostics
    # -------------------------------------------------------------

    logL_ml = logL[i_ml]

    # Evaluate exactly at Lambda_true and Lambda=0.
    logL_true = None
    logL_gr = None

    for Lam, value in zip(grid, logL):
        if np.isclose(Lam, Lambda_true):
            logL_true = value
        if np.isclose(Lam, 0.0):
            logL_gr = value

    # Since Lambda_true=-2.78 is not necessarily a grid point under
    # arbitrary grid definitions, evaluate it explicitly if necessary.
    if logL_true is None:
        from likelihood import log_likelihood

        logL_true = log_likelihood(
            data,
            f,
            psd,
            df,
            M1,
            M2,
            Lambda_true,
            K_z,
            TC,
            PHI_C,
            DISTANCE_MPC,
        )

    if logL_gr is None:
        from likelihood import log_likelihood

        logL_gr = log_likelihood(
            data,
            f,
            psd,
            df,
            M1,
            M2,
            0.0,
            K_z,
            TC,
            PHI_C,
            DISTANCE_MPC,
        )

    delta_logL_ml_gr = logL_ml - logL_gr
    delta_logL_true_gr = logL_true - logL_gr

    # -------------------------------------------------------------
    # Boundary diagnostic
    # -------------------------------------------------------------

    at_boundary = (
        np.isclose(Lambda_ml, LAMBDA_GRID[0])
        or
        np.isclose(Lambda_ml, LAMBDA_GRID[-1])
    )

    return {
        "Lambda_true": Lambda_true,
        "Lambda_ml": Lambda_ml,
        "recovery_error": Lambda_ml - Lambda_true,
        "fisher_sigma": Lambda_err,
        "match_ml": match_ml,
        "match_true": match_true,
        "match_gr": match_gr,
        "delta_logL_ml_gr": delta_logL_ml_gr,
        "delta_logL_true_gr": delta_logL_true_gr,
        "at_boundary": at_boundary,
    }


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Stage 2 synthetic Lambda injection/recovery"
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
        help="Synthetic observation duration in seconds.",
    )

    parser.add_argument(
        "--nreal",
        type=int,
        default=N_REALIZATIONS,
        help="Number of noise realizations per Lambda injection.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=BASE_SEED,
        help="Base RNG seed.",
    )

    args = parser.parse_args()

    duration = args.duration
    n_realizations = args.nreal

    # -------------------------------------------------------------
    # Header
    # -------------------------------------------------------------

    print("=" * 78)
    print("STAGE 2 — SYNTHETIC LAMBDA INJECTION / RECOVERY")
    print("=" * 78)
    print()

    print(f"m1, m2:              {M1:.2f}, {M2:.2f} Msun")
    print(f"Distance:             {DISTANCE_MPC:.1f} Mpc")
    print(f"Redshift:              {Z:.3f}")
    print(f"Duration:              {duration:.3f} s")
    print(f"Frequency band:        {F_LO:.0f}–{F_HI:.0f} Hz")
    print(f"Lambda grid:           [{LAMBDA_GRID[0]:.2f}, "
          f"{LAMBDA_GRID[-1]:.2f}]")
    print(f"Grid points:           {len(LAMBDA_GRID)}")
    print(f"Realizations/Lambda:   {n_realizations}")
    print(f"Base RNG seed:          {args.seed}")
    print()

    print("Injected Lambda values:")
    for value in INJECTED_LAMBDAS:
        print(f"    {value: .4f}")

    print()
    print("IMPORTANT:")
    print("  Lambda=-2.7800 is an injection value only.")
    print("  It is NOT used to select the grid or analysis band.")
    print("  Injection and recovery use the SAME waveform model.")
    print()

    # -------------------------------------------------------------
    # Frequency grid
    # -------------------------------------------------------------

    f, df = make_frequency_grid(
        F_LO,
        F_HI,
        duration,
    )

    psd = aligo_like_psd(f)

    K_z = cosmological_K_factor(Z)

    print(f"K(z):                  {K_z:.6e} s")
    print(f"df:                    {df:.6f} Hz")
    print(f"Frequency bins:        {len(f)}")
    print()

    # -------------------------------------------------------------
    # Experiment
    # -------------------------------------------------------------

    all_results = []

    for lambda_index, Lambda_true in enumerate(INJECTED_LAMBDAS):

        print("=" * 78)
        print(
            f"INJECTION: Lambda_true = {Lambda_true:.4f}"
        )
        print("=" * 78)

        lambda_results = []

        for realization in range(n_realizations):

            # Deterministic independent RNG stream for every
            # Lambda / realization combination.
            seed = (
                args.seed
                + 100000 * lambda_index
                + realization
            )

            rng = np.random.default_rng(seed)

            result = recover_single_realization(
                f,
                psd,
                df,
                K_z,
                Lambda_true,
                rng,
            )

            result["realization"] = realization + 1
            result["seed"] = seed

            lambda_results.append(result)
            all_results.append(result)

            print(
                f"  #{realization + 1:02d}: "
                f"ML={result['Lambda_ml']: .4f}   "
                f"error={result['recovery_error']: .4f}   "
                f"sigma={result['fisher_sigma']: .4f}   "
                f"match={result['match_ml']:.6f}   "
                f"dLogL={result['delta_logL_ml_gr']:.4f}"
            )

            gc.collect()

        estimates = np.array(
            [r["Lambda_ml"] for r in lambda_results]
        )

        errors = np.array(
            [r["recovery_error"] for r in lambda_results]
        )

        sigmas = np.array(
            [r["fisher_sigma"] for r in lambda_results]
        )

        matches = np.array(
            [r["match_ml"] for r in lambda_results]
        )

        boundary_fraction = np.mean(
            [r["at_boundary"] for r in lambda_results]
        )

        print()
        print(
            f"  SUMMARY Lambda_true={Lambda_true:.4f}"
        )
        print(
            f"    mean recovered:      {np.mean(estimates): .6f}"
        )
        print(
            f"    median recovered:    {np.median(estimates): .6f}"
        )
        print(
            f"    std recovered:       {np.std(estimates): .6f}"
        )
        print(
            f"    mean error:          {np.mean(errors): .6f}"
        )
        print(
            f"    RMSE:                "
            f"{np.sqrt(np.mean(errors ** 2)):.6f}"
        )
        print(
            f"    mean Fisher sigma:   {np.nanmean(sigmas): .6f}"
        )
        print(
            f"    mean ML match:       {np.mean(matches): .6f}"
        )
        print(
            f"    boundary fraction:   {boundary_fraction * 100:.1f}%"
        )
        print()

    # -----------------------------------------------------------------
    # Convert to arrays
    # -----------------------------------------------------------------

    import csv

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)

    csv_path = out / "stage2_injection_recovery.csv"

    fieldnames = [
        "Lambda_true",
        "realization",
        "seed",
        "Lambda_ml",
        "recovery_error",
        "fisher_sigma",
        "match_ml",
        "match_true",
        "match_gr",
        "delta_logL_ml_gr",
        "delta_logL_true_gr",
        "at_boundary",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:

        writer = csv.DictWriter(
            fh,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in all_results:
            writer.writerow(row)

    # -----------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------

    print()
    print("#" * 78)
    print("# STAGE 2 SUMMARY")
    print("#" * 78)
    print()

    print(
        f"{'Lambda_true':>12} "
        f"{'mean ML':>12} "
        f"{'std ML':>12} "
        f"{'mean err':>12} "
        f"{'RMSE':>12} "
        f"{'mean match':>12}"
    )

    print("-" * 78)

    summary_rows = []

    for Lambda_true in INJECTED_LAMBDAS:

        rows = [
            r for r in all_results
            if np.isclose(r["Lambda_true"], Lambda_true)
        ]

        ml = np.array([r["Lambda_ml"] for r in rows])
        err = np.array([r["recovery_error"] for r in rows])
        match = np.array([r["match_ml"] for r in rows])

        mean_ml = np.mean(ml)
        std_ml = np.std(ml)
        mean_err = np.mean(err)
        rmse = np.sqrt(np.mean(err ** 2))
        mean_match = np.mean(match)

        summary_rows.append({
            "Lambda_true": Lambda_true,
            "mean_ml": mean_ml,
            "std_ml": std_ml,
            "mean_err": mean_err,
            "rmse": rmse,
            "mean_match": mean_match,
        })

        print(
            f"{Lambda_true:12.4f} "
            f"{mean_ml:12.6f} "
            f"{std_ml:12.6f} "
            f"{mean_err:12.6f} "
            f"{rmse:12.6f} "
            f"{mean_match:12.6f}"
        )

    # -----------------------------------------------------------------
    # Null-test diagnosis
    # -----------------------------------------------------------------

    null_rows = [
        r for r in all_results
        if np.isclose(r["Lambda_true"], 0.0)
    ]

    null_ml = np.array(
        [r["Lambda_ml"] for r in null_rows]
    )

    null_error = np.array(
        [r["recovery_error"] for r in null_rows]
    )

    null_mean = np.mean(null_ml)
    null_std = np.std(null_ml)

    print()
    print("#" * 78)
    print("# NULL TEST — Lambda_true = 0")
    print("#" * 78)
    print()

    print(f"Mean recovered Lambda:   {null_mean:.6f}")
    print(f"Std recovered Lambda:    {null_std:.6f}")
    print(f"Mean recovery error:     {np.mean(null_error):.6f}")

    if abs(null_mean) < max(null_std, 1e-12):
        print()
        print(
            "NULL DIAGNOSTIC: no obvious systematic non-zero "
            "Lambda bias in this realization ensemble."
        )
    else:
        print()
        print(
            "NULL DIAGNOSTIC: recovered Lambda shows a systematic "
            "offset comparable to or larger than the ensemble scatter."
        )
        print(
            "This requires investigation before interpreting non-zero "
            "Lambda injections."
        )

    # -----------------------------------------------------------------
    # Identifiability diagnosis
    # -----------------------------------------------------------------

    print()
    print("#" * 78)
    print("# IDENTIFIABILITY DIAGNOSTIC")
    print("#" * 78)
    print()

    for row in summary_rows:

        Lambda_true = row["Lambda_true"]

        print(
            f"Lambda_true={Lambda_true: .4f}  "
            f"mean_recovered={row['mean_ml']: .4f}  "
            f"RMSE={row['rmse']:.4f}  "
            f"mean_match={row['mean_match']:.6f}"
        )

    print()
    print(
        "Interpretation rule:"
    )
    print(
        "  Successful injection/recovery means the recovered Lambda "
        "tracks the injected Lambda across the declared grid,"
    )
    print(
        "  while the Lambda_true=0 null remains centered near zero."
    )
    print()
    print(
        "This does NOT constitute an observational detection or "
        "constraint on GW150914."
    )

    # -----------------------------------------------------------------
    # Plot 1 — recovered Lambda versus injected Lambda
    # -----------------------------------------------------------------

    fig = plt.figure(figsize=(8, 6))

    x = np.array(
        [r["Lambda_true"] for r in summary_rows]
    )

    y = np.array(
        [r["mean_ml"] for r in summary_rows]
    )

    yerr = np.array(
        [r["std_ml"] for r in summary_rows]
    )

    plt.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        capsize=4,
        label="Recovered mean ± scatter",
    )

    lim_min = min(
        np.min(x),
        np.min(y - yerr),
    ) - 0.5

    lim_max = max(
        np.max(x),
        np.max(y + yerr),
    ) + 0.5

    plt.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        "--",
        label="Ideal recovery",
    )

    plt.xlabel(r"Injected $\Lambda$")
    plt.ylabel(r"Recovered $\Lambda_{\rm ML}$")
    plt.title("Stage 2 — Synthetic Lambda Injection / Recovery")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot1 = out / "stage2_injection_recovery.png"

    plt.savefig(
        plot1,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # -----------------------------------------------------------------
    # Plot 2 — recovery error
    # -----------------------------------------------------------------

    fig = plt.figure(figsize=(8, 6))

    for Lambda_true in INJECTED_LAMBDAS:

        rows = [
            r for r in all_results
            if np.isclose(r["Lambda_true"], Lambda_true)
        ]

        errors = np.array(
            [r["recovery_error"] for r in rows]
        )

        xvals = np.full(
            len(errors),
            Lambda_true,
        )

        jitter = np.linspace(
            -0.08,
            0.08,
            len(errors),
        )

        plt.scatter(
            xvals + jitter,
            errors,
            alpha=0.7,
        )

    plt.axhline(
        0.0,
        linestyle="--",
    )

    plt.xlabel(r"Injected $\Lambda$")
    plt.ylabel(r"$\Lambda_{\rm recovered}-\Lambda_{\rm true}$")
    plt.title("Stage 2 — Recovery error")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot2 = out / "stage2_recovery_error.png"

    plt.savefig(
        plot2,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print(f"Saved CSV:   {csv_path}")
    print(f"Saved plot:  {plot1}")
    print(f"Saved plot:  {plot2}")

    print()
    print("=" * 78)
    print("STAGE 2 COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()