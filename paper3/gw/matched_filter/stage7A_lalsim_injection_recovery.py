"""
stage7A_alternative_injection_recovery.py
=========================================

STAGE 7A (ALTERNATIVE) — TAYLORF2 WAVEFORM + LAMBDA INJECTION / RECOVERY

This is a fallback version that does NOT require LALSuite.
It uses the existing TaylorF2 waveform generator from waveform.py
instead of LALSimulation.

Purpose
-------
Test Lambda identifiability using a higher-order TaylorF2 waveform
instead of the leading-order 0PN waveform used in earlier stages.

Note: This is NOT a realistic GR waveform (like IMRPhenomD), but it
allows testing the identifiability concept without LALSuite dependency.
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from waveform import (
    C_SI,
    lambda_phase_correction,
    cosmological_K_factor,
    waveform_frequency_domain,
)
from likelihood import (
    aligo_like_psd,
    noise_weighted_inner_product,
)


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

M1 = 35.6
M2 = 30.6

DISTANCE_MPC = 440.0
Z = 0.09

F_LO = 20.0
F_HI = 300.0

# Deliberately broad grid.
# -2.78 is NOT used to define this grid.
LAMBDA_GRID = np.linspace(-5.0, 5.0, 401)

LAMBDA_INJECTIONS = [
    0.0,
    -1.0,
    -2.0,
    -2.78,
    -4.0,
]

N_REALIZATIONS = 16
BASE_SEED = 260830


# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------

def generate_taylorf2_fd(f, m1, m2, distance_mpc, z, tc=0.0, phi_c=0.0):
    """
    Generate TaylorF2 frequency-domain waveform using waveform.py.
    
    This is the same waveform generator used in earlier stages.
    """
    K_z = cosmological_K_factor(z)
    
    h = waveform_frequency_domain(
        f,
        m1,
        m2,
        0.0,  # Lambda=0 for the base waveform
        K_z,
        tc=tc,
        phi_c=phi_c,
        distance_Mpc=distance_mpc,
    )
    
    return np.asarray(h, dtype=complex)


def lambda_waveform_from_gr(h_gr, f, Lambda, K_z):
    """
    Apply the exact Lambda phase correction from waveform.py
    to the GR frequency-domain waveform.
    """
    delta_psi = lambda_phase_correction(f, Lambda, K_z)
    return h_gr * np.exp(1j * delta_psi)


def normalized_match(a, b, f, psd, df):
    """
    Noise-weighted normalized match.

        M = <a|b> / sqrt(<a|a><b|b>)

    No time/phase maximization is performed here.
    """

    aa = noise_weighted_inner_product(a, a, f, psd, df)
    bb = noise_weighted_inner_product(b, b, f, psd, df)
    ab = noise_weighted_inner_product(a, b, f, psd, df)

    denom = np.sqrt(max(aa * bb, 0.0))

    if denom <= 0:
        return np.nan

    return ab / denom


def fisher_sigma_from_logL(logL, grid, i_max):
    """
    Local parabolic curvature estimate.
    """
    if i_max <= 0 or i_max >= len(grid) - 1:
        return np.nan

    dL = grid[1] - grid[0]

    d2 = (
        logL[i_max + 1]
        - 2.0 * logL[i_max]
        + logL[i_max - 1]
    ) / (dL ** 2)

    if d2 >= 0:
        return np.nan

    return np.sqrt(-1.0 / d2)


def lambda_logL(
    data,
    f,
    psd,
    df,
    h_gr,
    K_z,
    Lambda,
):
    """
    Likelihood evaluated directly using the GR waveform plus
    Lambda phase correction.
    """

    h = lambda_waveform_from_gr(
        h_gr,
        f,
        Lambda,
        K_z,
    )

    dh = noise_weighted_inner_product(
        data,
        h,
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

    return dh - 0.5 * hh


def recover_lambda(
    data,
    f,
    psd,
    df,
    h_gr,
    K_z,
    lambda_grid,
):
    """
    Grid-search Lambda using the GR waveform.
    """

    logL = np.array([
        lambda_logL(
            data,
            f,
            psd,
            df,
            h_gr,
            K_z,
            Lam,
        )
        for Lam in lambda_grid
    ])

    i_max = int(np.argmax(logL))
    Lambda_ml = float(lambda_grid[i_max])

    sigma = fisher_sigma_from_logL(
        logL,
        lambda_grid,
        i_max,
    )

    return logL, Lambda_ml, sigma, i_max


# -------------------------------------------------------------------------
# Main experiment
# -------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--realizations",
        type=int,
        default=N_REALIZATIONS,
        help="Realizations per Lambda",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=BASE_SEED,
        help="Base RNG seed",
    )

    args = parser.parse_args()

    n_realizations = args.realizations

    print("=" * 78)
    print("STAGE 7A (ALTERNATIVE) — TAYLORF2 + LAMBDA INJECTION / RECOVERY")
    print("=" * 78)
    print()

    print(f"m1, m2:              {M1:.2f}, {M2:.2f} Msun")
    print(f"Distance:             {DISTANCE_MPC:.1f} Mpc")
    print(f"Redshift:              {Z:.3f}")
    print(f"Frequency band:        {F_LO:.0f}–{F_HI:.0f} Hz")
    print(f"Lambda grid:           [{LAMBDA_GRID[0]:.2f}, {LAMBDA_GRID[-1]:.2f}]")
    print(f"Grid points:           {len(LAMBDA_GRID)}")
    print(f"Realizations/Lambda:   {n_realizations}")
    print(f"Base RNG seed:         {args.seed}")
    print()

    print("Injected Lambda values:")
    for Lam in LAMBDA_INJECTIONS:
        print(f"    {Lam:8.4f}")

    print()
    print("IMPORTANT:")
    print("  Lambda=-2.7800 is an injection value only.")
    print("  It is NOT used to select the grid, band, seed, or estimator.")
    print("  Injection and recovery share the same Lambda phase prescription.")
    print()
    print("NOTE: This is a TaylorF2 waveform (not IMRPhenomD from LALSuite).")
    print("      It still tests Lambda identifiability with a higher-order")
    print("      waveform than the leading-order 0PN used in Stage 2.")
    print()

    K_z = cosmological_K_factor(Z)

    print(f"K(z):                  {K_z:.6e} s")
    print()

    # ---------------------------------------------------------------------
    # Generate GR waveform
    # ---------------------------------------------------------------------

    print("=" * 78)
    print("GENERATING TaylorF2 GR WAVEFORM")
    print("=" * 78)
    print()

    # Generate frequency grid
    df = 0.1  # Hz
    f = np.arange(F_LO, F_HI + df, df)

    h_gr = generate_taylorf2_fd(
        f,
        M1,
        M2,
        DISTANCE_MPC,
        Z,
        tc=0.0,
        phi_c=0.0,
    )

    if len(f) < 10:
        raise RuntimeError(
            "Too few frequency bins generated."
        )

    psd = aligo_like_psd(f)

    print(f"Frequency bins:        {len(f)}")
    print(f"df:                    {df:.6f} Hz")
    print(f"f_min:                 {f[0]:.3f} Hz")
    print(f"f_max:                 {f[-1]:.3f} Hz")
    print()

    # ---------------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------------

    all_results = []

    for lambda_true in LAMBDA_INJECTIONS:

        print("=" * 78)
        print(
            f"INJECTION: Lambda_true = {lambda_true:.4f}"
        )
        print("=" * 78)

        estimates = []
        errors = []
        sigmas = []
        matches_ml = []
        matches_true = []
        matches_gr = []
        delta_logLs = []

        for idx in range(n_realizations):

            seed = (
                args.seed
                + int(round((lambda_true + 10.0) * 10000))
                + idx
            )

            rng = np.random.default_rng(seed)

            # -------------------------------------------------------------
            # Construct Lambda-injected waveform
            # -------------------------------------------------------------

            h_true = lambda_waveform_from_gr(
                h_gr,
                f,
                lambda_true,
                K_z,
            )

            # -------------------------------------------------------------
            # Colored complex Gaussian noise.
            # -------------------------------------------------------------

            sigma_noise = np.sqrt(
                psd / (4.0 * df)
            )

            noise = (
                rng.normal(0.0, sigma_noise)
                +
                1j * rng.normal(0.0, sigma_noise)
            )

            data = h_true + noise

            # -------------------------------------------------------------
            # Lambda recovery
            # -------------------------------------------------------------

            logL, Lambda_ml, fisher_sigma, i_max = recover_lambda(
                data,
                f,
                psd,
                df,
                h_gr,
                K_z,
                LAMBDA_GRID,
            )

            # -------------------------------------------------------------
            # Templates
            # -------------------------------------------------------------

            h_ml = lambda_waveform_from_gr(
                h_gr,
                f,
                Lambda_ml,
                K_z,
            )

            h_true_template = h_true

            h_gr_template = lambda_waveform_from_gr(
                h_gr,
                f,
                0.0,
                K_z,
            )

            # -------------------------------------------------------------
            # Matches
            # -------------------------------------------------------------

            match_ml = normalized_match(
                h_true,
                h_ml,
                f,
                psd,
                df,
            )

            match_true = normalized_match(
                h_true,
                h_true_template,
                f,
                psd,
                df,
            )

            match_gr = normalized_match(
                h_true,
                h_gr_template,
                f,
                psd,
                df,
            )

            # -------------------------------------------------------------
            # Delta log likelihood relative to GR
            # -------------------------------------------------------------

            i_gr = int(
                np.argmin(
                    np.abs(LAMBDA_GRID - 0.0)
                )
            )

            delta_logL = (
                logL[i_max]
                - logL[i_gr]
            )

            error = Lambda_ml - lambda_true

            estimates.append(Lambda_ml)
            errors.append(error)
            sigmas.append(fisher_sigma)
            matches_ml.append(match_ml)
            matches_true.append(match_true)
            matches_gr.append(match_gr)
            delta_logLs.append(delta_logL)

            print(
                f"  #{idx + 1:02d}: "
                f"ML={Lambda_ml:8.4f}   "
                f"error={error:8.4f}   "
                f"sigma={fisher_sigma:8.5f}   "
                f"match={match_ml:.6f}   "
                f"match_true={match_true:.6f}   "
                f"match_gr={match_gr:.6f}   "
                f"dLogL={delta_logL:.4f}"
            )

            gc.collect()

        # -----------------------------------------------------------------
        # Summary
        # -----------------------------------------------------------------

        estimates = np.asarray(estimates)
        errors = np.asarray(errors)
        sigmas = np.asarray(sigmas)
        matches_ml = np.asarray(matches_ml)
        matches_true = np.asarray(matches_true)
        matches_gr = np.asarray(matches_gr)
        delta_logLs = np.asarray(delta_logLs)

        boundary_fraction = np.mean(
            (estimates <= LAMBDA_GRID[0])
            |
            (estimates >= LAMBDA_GRID[-1])
        )

        mean_recovered = np.mean(estimates)
        std_recovered = np.std(estimates)
        mean_error = np.mean(errors)
        rmse = np.sqrt(np.mean(errors ** 2))

        mean_sigma = np.nanmean(sigmas)
        mean_match = np.nanmean(matches_ml)
        mean_match_true = np.nanmean(matches_true)
        mean_match_gr = np.nanmean(matches_gr)
        mean_delta_logL = np.mean(delta_logLs)

        print()
        print(
            f"  SUMMARY Lambda_true={lambda_true:.4f}"
        )
        print(
            f"    mean recovered:       {mean_recovered: .6f}"
        )
        print(
            f"    median recovered:     {np.median(estimates): .6f}"
        )
        print(
            f"    std recovered:        {std_recovered: .6f}"
        )
        print(
            f"    mean error:           {mean_error: .6f}"
        )
        print(
            f"    RMSE:                 {rmse: .6f}"
        )
        print(
            f"    mean Fisher sigma:    {mean_sigma: .6f}"
        )
        print(
            f"    mean match ML:        {mean_match:.6f}"
        )
        print(
            f"    mean match_true:      {mean_match_true:.6f}"
        )
        print(
            f"    mean match_gr:        {mean_match_gr:.6f}"
        )
        print(
            f"    mean DeltaLogL:       {mean_delta_logL:.6f}"
        )
        print(
            f"    boundary fraction:    {boundary_fraction * 100:.1f}%"
        )

        all_results.append(
            {
                "Lambda_true": lambda_true,
                "mean_ML": mean_recovered,
                "median_ML": np.median(estimates),
                "std_ML": std_recovered,
                "mean_error": mean_error,
                "RMSE": rmse,
                "mean_sigma": mean_sigma,
                "mean_match": mean_match,
                "mean_match_true": mean_match_true,
                "mean_match_gr": mean_match_gr,
                "mean_DeltaLogL": mean_delta_logL,
                "boundary_fraction": boundary_fraction,
            }
        )

    # ---------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------

    print()
    print("#" * 78)
    print("# STAGE 7A SUMMARY")
    print("#" * 78)
    print()

    print(
        f"{'Lambda_true':>12} "
        f"{'mean ML':>12} "
        f"{'std ML':>10} "
        f"{'RMSE':>10} "
        f"{'match_true':>12} "
        f"{'match_gr':>10} "
        f"{'DeltaLogL':>12}"
    )

    print("-" * 86)

    for r in all_results:

        print(
            f"{r['Lambda_true']:12.4f} "
            f"{r['mean_ML']:12.6f} "
            f"{r['std_ML']:10.6f} "
            f"{r['RMSE']:10.6f} "
            f"{r['mean_match_true']:12.6f} "
            f"{r['mean_match_gr']:10.6f} "
            f"{r['mean_DeltaLogL']:12.6f}"
        )

    # ---------------------------------------------------------------------
    # Null diagnostic
    # ---------------------------------------------------------------------

    null_result = next(
        r for r in all_results
        if abs(r["Lambda_true"]) < 1e-12
    )

    print()
    print("#" * 78)
    print("# NULL TEST")
    print("#" * 78)
    print()

    print(
        f"Mean recovered Lambda:   "
        f"{null_result['mean_ML']:.8f}"
    )

    print(
        f"Std recovered Lambda:    "
        f"{null_result['std_ML']:.8f}"
    )

    print(
        f"RMSE:                    "
        f"{null_result['RMSE']:.8f}"
    )

    if abs(null_result["mean_ML"]) < 3.0 * max(
        null_result["std_ML"],
        1e-12,
    ):
        print()
        print(
            "NULL DIAGNOSTIC: no obvious systematic non-zero "
            "Lambda bias detected."
        )
    else:
        print()
        print(
            "NULL DIAGNOSTIC: possible systematic Lambda bias "
            "detected; investigate before interpretation."
        )

    # ---------------------------------------------------------------------
    # Identifiability diagnostic
    # ---------------------------------------------------------------------

    print()
    print("#" * 78)
    print("# IDENTIFIABILITY DIAGNOSTIC")
    print("#" * 78)
    print()

    for r in all_results:

        print(
            f"Lambda_true={r['Lambda_true']:7.4f} "
            f"mean_ML={r['mean_ML']:9.4f} "
            f"RMSE={r['RMSE']:8.4f} "
            f"match_true={r['mean_match_true']:.6f} "
            f"match_gr={r['mean_match_gr']:.6f}"
        )

    print()
    print("Interpretation:")
    print()
    print(
        "  1. Recovery accuracy:"
    )
    print(
        "       mean Lambda_ML should track Lambda_true."
    )
    print()
    print(
        "  2. Null behavior:"
    )
    print(
        "       Lambda_true=0 should remain centered near zero."
    )
    print()
    print(
        "  3. Template fidelity:"
    )
    print(
        "       match_true should remain high (trivially 1.0)."
    )
    print()
    print(
        "  4. Lambda distinguishability:"
    )
    print(
        "       match_gr should decrease as |Lambda_true| increases."
    )
    print()
    print(
        "  5. Detection strength:"
    )
    print(
        "       DeltaLogL should increase when the injected Lambda"
    )
    print(
        "       becomes increasingly different from GR."
    )
    print()
    print(
        "This is a synthetic validation only."
    )
    print(
        "It does NOT constitute an observational measurement of Lambda"
    )
    print(
        "from GW150914."
    )

    # ---------------------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------------------

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)

    csv_path = out / "stage7A_alternative_injection_recovery.csv"

    with open(csv_path, "w", encoding="utf-8") as fp:

        fp.write(
            "Lambda_true,mean_ML,median_ML,std_ML,"
            "mean_error,RMSE,mean_sigma,"
            "mean_match,mean_match_true,mean_match_gr,"
            "mean_DeltaLogL,boundary_fraction\n"
        )

        for r in all_results:

            fp.write(
                f"{r['Lambda_true']:.8f},"
                f"{r['mean_ML']:.8f},"
                f"{r['median_ML']:.8f},"
                f"{r['std_ML']:.8f},"
                f"{r['mean_error']:.8f},"
                f"{r['RMSE']:.8f},"
                f"{r['mean_sigma']:.8f},"
                f"{r['mean_match']:.8f},"
                f"{r['mean_match_true']:.8f},"
                f"{r['mean_match_gr']:.8f},"
                f"{r['mean_DeltaLogL']:.8f},"
                f"{r['boundary_fraction']:.8f}\n"
            )

    print()
    print(f"Saved CSV:   {csv_path}")

    print()
    print("#" * 78)
    print("STAGE 7A COMPLETE")
    print("#" * 78)


if __name__ == "__main__":
    main()