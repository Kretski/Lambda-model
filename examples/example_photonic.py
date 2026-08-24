"""
example_photonic.py
======================

Worked example: testing the Lambda-model's INTERNAL fitting machinery
against synthetic fourth-order dispersion (FOD) data in the photonic
k-space convention.

PHYSICS BACKGROUND:

The Lambda-model dispersion omega(k) = c*k*sqrt(1+Lambda*k^2) has an
EXACT, closed-form quartic term in omega^2(k), not omega(k) itself:

    omega(k)^2 = c^2*k^2 + beta4_k * k^4,     beta4_k = c^2 * Lambda

(An earlier version of this codebase used omega(k) = c*k + (beta4_k/24)*k^4,
which is WRONG: direct Taylor expansion of c*k*sqrt(1+Lambda*k^2) gives a
CUBIC leading correction term to omega(k), c*k + (Lambda*c/2)*k^3 + ...,
not a quartic one. The quartic structure only appears cleanly in
omega^2(k). See lambda_from_photonic() docstring for the corrected
derivation.)

STATUS -- READ BEFORE USING WITH REAL DATA:

  beta4_k, as defined here, is NOT (yet) connected to any standard,
  independently-measurable photonics quantity. Standard telecom
  dispersion coefficients (beta_2, beta_3, beta_4 = d^n(beta)/d(omega)^n)
  are Taylor coefficients of the propagation constant around a nonzero
  pump/operating frequency -- a structurally different expansion from
  the near-k=0 form used by the Lambda-model. Even in the fiber-optic
  event-horizon / "pure-quartic-soliton" literature (the closest real
  physical analogue), the relevant co-moving-frame frequency near a
  group-velocity-matched point behaves as a quartic MINIMUM with no
  linear term at all -- a different mathematical structure from the
  deformed-light-cone form used here. Deriving beta4_k from real,
  measured beta_n values requires additional theoretical work that has
  not been done. Treat this example as a test of the FITTING CODE only,
  not as evidence that any real photonic system follows the Lambda-model.

WHAT THIS SCRIPT DOES:

  1. Generates synthetic omega(k) data from the Lambda-model formula
     itself (with noise), using an assumed beta4_k.
  2. Fits the Lambda-model to that synthetic data.
  3. Confirms the fit recovers the input Lambda -- i.e. sanity-checks
     fit_lambda(), nothing more.
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "paper3"))
from lambda_experimental_validator import (
    fit_lambda, lambda_from_photonic, print_report
)


def photonic_omega(k, c, beta4_k):
    """Lambda-model dispersion using the (corrected) photonic beta4_k mapping."""
    Lambda = lambda_from_photonic(beta4_k, c)
    return c * k * np.sqrt(np.maximum(1 + Lambda * k**2, 0))


def generate_synthetic_transmission_data(c, beta4_k, k_range,
                                          noise_level=0.005, n_points=18,
                                          seed=2):
    """Generate synthetic transmission-spectroscopy-like data with noise."""
    rng = np.random.default_rng(seed)
    k_data = np.linspace(k_range[0], k_range[1], n_points)
    omega_true = photonic_omega(k_data, c, beta4_k)
    omega_data = omega_true * (1 + rng.normal(0, noise_level, size=k_data.shape))
    return k_data, omega_data


def main():
    print("=" * 68)
    print("EXAMPLE: Photonic k-space quartic dispersion -- FITTING CODE TEST")
    print("(NOT a validated physical prediction -- see script header)")
    print("=" * 68)
    print()

    c = 2.998e8 / 1.45          # m/s, phase velocity in fiber (n~1.45)

    # beta4_k is the coefficient of k^4 in omega^2(k) = c^2 k^2 + beta4_k k^4
    # (units m^4/s^2). Chosen here purely to give an O(1e-3) illustrative
    # Lambda -- NOT taken from any measured photonic dataset.
    Lambda_illustrative = 1.6e-3   # m^2, arbitrary illustrative target
    beta4_k = Lambda_illustrative * c**2

    Lambda_theory = lambda_from_photonic(beta4_k, c)

    print(f"  Illustrative parameters (NOT measured data):")
    print(f"    Phase velocity c = {c:.4e} m/s")
    print(f"    beta4_k (coeff. of k^4 in omega^2(k)) = {beta4_k:.4e} m^4/s^2")
    print(f"    Theoretical Lambda = beta4_k/c^2 = {Lambda_theory:.4e} m^2")
    print()

    # ── Generate synthetic data and fit ───────────────────────────────────
    k_max = 5.0e7  # 1/m, illustrative probe range
    k_data, omega_data = generate_synthetic_transmission_data(
        c, beta4_k, k_range=(1e6, k_max), noise_level=0.005, n_points=18)

    print(f"  Simulated {len(k_data)} data points, k in "
          f"[{k_data[0]:.2e}, {k_data[-1]:.2e}] 1/m, 0.5% noise")
    print()

    result = fit_lambda(k_data, omega_data, c_fixed=c)
    print_report(result, domain_name="Photonic (k-space, illustrative)",
                 theory_Lambda=Lambda_theory, synthetic_selfcheck=True)

    # ── Plot ────────────────────────────────────────────────────────────
    out = Path(__file__).parent / "figures"
    out.mkdir(exist_ok=True)

    k_smooth = np.linspace(k_data[0], k_data[-1], 200)
    omega_theory_curve = photonic_omega(k_smooth, c, beta4_k)
    omega_fit_curve = c * k_smooth * np.sqrt(
        np.maximum(1 + result["Lambda"] * k_smooth**2, 0))
    omega_linear = c * k_smooth

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_data * 1e-6, omega_data * 1e-12, "o", color="steelblue",
             markersize=6, label="synthetic data (0.5% noise)")
    ax.plot(k_smooth * 1e-6, omega_theory_curve * 1e-12, "k-", lw=2,
             label="exact Lambda-model curve")
    ax.plot(k_smooth * 1e-6, omega_fit_curve * 1e-12, "r--", lw=2,
             label="fit result")
    ax.plot(k_smooth * 1e-6, omega_linear * 1e-12, "gray", lw=1, ls=":",
             label="linear dispersion (Lambda=0)")
    ax.set_xlabel(r"$k$ [rad/$\mu$m]")
    ax.set_ylabel(r"$\omega$ [Trad/s]")
    ax.set_title("Photonic k-space fit -- fitting-code sanity check only")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    path = out / "example_photonic_fit.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")

    print()
    print("This script does NOT demonstrate that any real photonic system")
    print("follows the Lambda-model. It confirms the fitting code correctly")
    print("recovers a known Lambda from noisy synthetic data. Connecting")
    print("beta4_k to a measurable quantity in a real waveguide is an open")
    print("problem -- see script header.")


if __name__ == "__main__":
    main()
