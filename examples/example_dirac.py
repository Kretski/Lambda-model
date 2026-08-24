"""
example_dirac.py
===================

*** SPECULATIVE EXAMPLE -- NOT A VALIDATED PHYSICAL PREDICTION ***

Originally framed as a "trigonal-warping-corrected Dirac dispersion"
worked example. On review, the cited mechanism does not support the
formula used:

  Fu, L., PRL 103, 266801 (2009) derives a hexagonal warping term that
  is CUBIC in momentum:
      H = v_F(k_x*sigma_y - k_y*sigma_x) + (lambda_w/2)(k_+^3+k_-^3)*sigma_z
  giving
      E(k)^2 = v_F^2 k^2 + lambda_w^2 k^6 cos^2(3*theta)
  -- an ANISOTROPIC k^6 correction to the energy, not the isotropic k^4
  correction this script's Lambda-model form requires. No literature-
  supported derivation of Lambda=(1-eta^2)*v_F^2/4 from real trigonal-
  warping physics is currently known.

This script is retained ONLY as a demonstration of the fitting code
(fit_lambda / lambda_from_dirac), not as evidence for any real material.
The "ARPES-like" data below is synthetic, generated FROM the same
formula being tested -- the fit is guaranteed to recover the input
Lambda regardless of whether the underlying physical mapping is real.
Do not cite this script's output as experimental support for the model.

WHAT THIS SCRIPT DOES:

  1. Generates synthetic data from the Lambda-model formula itself
     (with noise).
  2. Fits the Lambda-model.
  3. Confirms the fit recovers the input -- i.e. sanity-checks
     fit_lambda(), nothing more.

IF YOU WANT TO ACTUALLY TEST THIS AGAINST REAL PHYSICS:
  Generate comparison data from an independent tight-binding or k.p
  model of your material (not from this formula), or from real ARPES
  band data, and fit that instead.
"""

import numpy as np
import sys
import warnings
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "paper3"))
from lambda_experimental_validator import fit_lambda, lambda_from_dirac, print_report


def dirac_warped_omega(k, v_F, eta):
    """
    Lambda-model dispersion using the (speculative) Dirac eta mapping.
    Matches the Lambda-model form exactly BY CONSTRUCTION -- see module
    docstring. This is not derived from an independent physical model.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # already warned once at top level
        Lambda = lambda_from_dirac(eta, v_F)
    return v_F * k * np.sqrt(np.maximum(1 + Lambda * k**2, 0))


def generate_synthetic_ARPES(v_F, eta, k_range, noise_level=0.015,
                              n_points=20, seed=1):
    """Generate synthetic (k, E) data with noise, FROM the same formula
    being tested. This is a fitting-code sanity check, not a physics test."""
    rng = np.random.default_rng(seed)
    k_data = np.linspace(k_range[0], k_range[1], n_points)
    E_true = dirac_warped_omega(k_data, v_F, eta)
    E_data = E_true * (1 + rng.normal(0, noise_level, size=k_data.shape))
    return k_data, E_data


def main():
    print("=" * 68)
    print("EXAMPLE: Dirac material eta-mapping -- FITTING CODE TEST ONLY")
    print("SPECULATIVE -- not supported by the cited literature.")
    print("See script header before using this for anything else.")
    print("=" * 68)
    print()

    v_F = 5.0e5       # m/s, illustrative Fermi velocity
    eta = 0.25        # dimensionless, illustrative

    warnings.warn(
        "example_dirac.py demonstrates a SPECULATIVE, non-validated "
        "mapping. See script header.", stacklevel=2,
    )
    Lambda_theory = lambda_from_dirac(eta, v_F)

    print(f"  Illustrative parameters (NOT measured data):")
    print(f"    Fermi velocity v_F = {v_F:.4e} m/s")
    print(f"    eta (speculative warping parameter) = {eta}")
    print(f"    Lambda = (1-eta^2)*v_F^2/4 = {Lambda_theory:.4e} m^2/s^2")
    print()

    k_max = 0.3e9
    k_data, E_data = generate_synthetic_ARPES(
        v_F, eta, k_range=(0.02e9, k_max), noise_level=0.015, n_points=20)

    print(f"  Simulated {len(k_data)} data points, "
          f"k in [{k_data[0]:.2e}, {k_data[-1]:.2e}] 1/m, 1.5% noise")
    print()

    result = fit_lambda(k_data, E_data, c_fixed=v_F)
    print_report(result, domain_name="Dirac material (speculative)",
                 theory_Lambda=Lambda_theory, synthetic_selfcheck=True)

    out = Path(__file__).parent / "figures"
    out.mkdir(exist_ok=True)

    k_smooth = np.linspace(k_data[0], k_data[-1], 200)
    E_theory_curve = dirac_warped_omega(k_smooth, v_F, eta)
    E_fit_curve = v_F * k_smooth * np.sqrt(
        np.maximum(1 + result["Lambda"] * k_smooth**2, 0))
    E_linear = v_F * k_smooth

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_data * 1e-9, E_data * 1e3, "o", color="steelblue", markersize=6,
             label="synthetic data (1.5% noise)")
    ax.plot(k_smooth * 1e-9, E_theory_curve * 1e3, "k-", lw=2,
             label="exact Lambda-model curve")
    ax.plot(k_smooth * 1e-9, E_fit_curve * 1e3, "r--", lw=2,
             label="fit result")
    ax.plot(k_smooth * 1e-9, E_linear * 1e3, "gray", lw=1, ls=":",
             label="bare Dirac cone (eta=0)")
    ax.set_xlabel(r"$k$ [nm$^{-1}$]")
    ax.set_ylabel(r"$E$ [meV] (arb. norm.)")
    ax.set_title("Dirac eta-mapping fit -- fitting-code sanity check only")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    path = out / "example_dirac_fit.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")

    print()
    print("This script does NOT demonstrate that any real Dirac material")
    print("follows the Lambda-model. See script header for why the")
    print("underlying mapping is not currently supported by the cited")
    print("physics.")


if __name__ == "__main__":
    main()
