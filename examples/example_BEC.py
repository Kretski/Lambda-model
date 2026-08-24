"""
example_BEC.py
================

Worked example: testing the Lambda-model against Bogoliubov dispersion
in a Bose-Einstein condensate.

PHYSICS BACKGROUND:

The Bogoliubov dispersion relation for excitations in a weakly-interacting
BEC is:

    omega(k) = c_s * k * sqrt(1 + (xi*k/2)^2)

where:
    c_s = sound speed = sqrt(g*n0/m)   [g=interaction strength, n0=density]
    xi  = healing length = hbar/(m*c_s)

This has EXACTLY the Lambda-model form omega = c*k*sqrt(1+Lambda*k^2) with

    Lambda = xi^2 / 4

This is not a coincidence: both the Lambda-model and Bogoliubov theory
describe a quadratic UV completion of a linear (phononic) dispersion
relation, controlled by a single length scale.

WHAT THIS SCRIPT DOES:

  1. Generates synthetic "experimental" Bogoliubov data for a realistic
     BEC (e.g. Rb-87 in a dilute trap).
  2. Fits the Lambda-model to this data using the validator.
  3. Compares the fitted Lambda to the theoretical xi^2/4.
  4. Shows how sensitive the fit is to experimental noise.

HOW TO USE WITH YOUR OWN DATA:

  Replace the synthetic data generation in generate_synthetic_data() with
  your own measured (k, omega) pairs from Bragg spectroscopy or similar,
  then run the same fitting procedure.
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "paper3"))
from lambda_experimental_validator import fit_lambda, lambda_from_BEC, print_report


def bogoliubov_omega(k, c_s, xi):
    """Exact Bogoliubov dispersion relation."""
    return c_s * k * np.sqrt(1 + (xi * k / 2)**2)


def generate_synthetic_data(c_s, xi, k_range, noise_level=0.02, n_points=15,
                             seed=0):
    """Generate synthetic Bragg-spectroscopy-like data with noise."""
    rng = np.random.default_rng(seed)
    k_data = np.linspace(k_range[0], k_range[1], n_points)
    omega_true = bogoliubov_omega(k_data, c_s, xi)
    omega_data = omega_true * (1 + rng.normal(0, noise_level, size=k_data.shape))
    return k_data, omega_data


def main():
    print("=" * 68)
    print("EXAMPLE: BEC (Bogoliubov dispersion) — Lambda-model test")
    print("=" * 68)
    print()

    # ── Realistic BEC parameters (Rb-87, typical dilute trap) ────────────
    c_s = 5.0e-3     # m/s, typical sound speed in dilute BEC
    xi = 5.0e-7      # m, typical healing length (~0.5 micron)

    Lambda_theory = lambda_from_BEC(xi)

    print(f"  BEC parameters:")
    print(f"    Sound speed c_s = {c_s:.4e} m/s")
    print(f"    Healing length xi = {xi:.4e} m")
    print(f"    Theoretical Lambda = xi^2/4 = {Lambda_theory:.4e} m^2")
    print()

    # ── Generate synthetic data and fit ───────────────────────────────────
    k_max = 3.0 / xi   # probe up to a few times the inverse healing length
    k_data, omega_data = generate_synthetic_data(
        c_s, xi, k_range=(0.1 / xi, k_max), noise_level=0.02, n_points=15)

    print(f"  Simulated {len(k_data)} data points, k in "
          f"[{k_data[0]:.2e}, {k_data[-1]:.2e}] 1/m, 2% noise")
    print()

    result = fit_lambda(k_data, omega_data, c_fixed=c_s)
    print_report(result, domain_name="BEC", theory_Lambda=Lambda_theory)

    # ── Plot ────────────────────────────────────────────────────────────
    out = Path(__file__).parent / "figures"
    out.mkdir(exist_ok=True)

    k_smooth = np.linspace(k_data[0], k_data[-1], 200)
    omega_theory_curve = bogoliubov_omega(k_smooth, c_s, xi)
    omega_fit_curve = c_s * k_smooth * np.sqrt(
        np.maximum(1 + result["Lambda"] * k_smooth**2, 0))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_data * xi, omega_data / c_s / (1/xi), "o", color="steelblue",
             markersize=6, label="synthetic Bragg data (2% noise)")
    ax.plot(k_smooth * xi, omega_theory_curve / c_s / (1/xi), "k-", lw=2,
             label="exact Bogoliubov")
    ax.plot(k_smooth * xi, omega_fit_curve / c_s / (1/xi), "r--", lw=2,
             label=f"Lambda-model fit (Lambda={result['Lambda']:.2e})")
    ax.set_xlabel(r"$k \xi$ (dimensionless)")
    ax.set_ylabel(r"$\omega / (c_s/\xi)$ (dimensionless)")
    ax.set_title("BEC Bogoliubov dispersion vs Lambda-model fit")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    path = out / "example_BEC_fit.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")

    print()
    print("HOW TO USE WITH YOUR OWN DATA:")
    print("  Replace c_s, xi above with your measured sound speed and")
    print("  healing length, and replace generate_synthetic_data() with")
    print("  your own (k, omega) measurements from Bragg spectroscopy.")


if __name__ == "__main__":
    main()
