"""
lambda_experimental_validator.py
===================================

Command-line tool for experimentalists to test whether their measured
dispersion relation omega(k) is consistent with the Lambda-model form:

    omega(k) = c * k * sqrt(1 + Lambda * k^2)

USAGE (from your own data):

    python lambda_experimental_validator.py --omega data_omega.csv --k data_k.csv

CSV files should contain a single column of numbers (no header), with
matching lengths. Units are up to the user -- Lambda will be returned in
[unit of k]^-2, i.e. consistent with the units used in the input.

USAGE (built-in domain examples):

    python lambda_experimental_validator.py --domain BEC --xi 1e-6 --cs 0.01
    python lambda_experimental_validator.py --domain dirac --eta 0.3 --vF 1e6
    python lambda_experimental_validator.py --domain photonic --beta4k 1e13 --c 3e8

WHAT IT DOES:

  1. Fits Lambda (and optionally c) by least-squares to the model
     (omega/(c*k))^2 - 1 = Lambda * k^2
  2. Reports the fit quality (R^2, residuals)
  3. Reports whether the data is consistent with Lambda=0 (pure GR / linear
     dispersion) at the given noise level
  4. For domain examples, compares the fitted Lambda to the theoretical
     mapping

STATUS OF THE DOMAIN MAPPINGS (read before using --domain):

  - BEC (lambda_from_BEC):      physically established. omega(k) =
    c_s*k*sqrt(1+(xi*k/2)^2) is the exact Bogoliubov dispersion relation
    for a weakly-interacting BEC; Lambda = xi^2/4 follows directly.

  - dirac (lambda_from_dirac):  SPECULATIVE, NOT VALIDATED. See the
    function docstring -- the cited hexagonal/trigonal-warping literature
    (Fu, PRL 103, 266801 (2009)) describes a k^3 term in the Hamiltonian,
    giving an ANISOTROPIC k^6 correction to E(k)^2, not the isotropic k^4
    form used here. No literature-supported derivation of
    Lambda=(1-eta^2)*v_F^2/4 from real trigonal-warping physics currently
    exists. This function is kept for exploratory use only.

  - photonic (lambda_from_photonic): the formula below is dimensionally
    self-consistent and correctly derived from the Lambda-model itself,
    but beta4_k is NOT (yet) connected to any standard, independently
    measurable photonics quantity. See the function docstring.
"""

import argparse
import numpy as np
import sys
from pathlib import Path


# ── Domain-specific Lambda mappings ──────────────────────────────────────
def lambda_from_BEC(xi):
    """
    Bose-Einstein condensate healing length mapping.
    Lambda = xi^2 / 4
    xi = healing length [m], typically ~0.1-1 micron for dilute BEC.

    STATUS: physically established. The Bogoliubov dispersion for
    phonons in a weakly-interacting BEC is exactly
        omega(k) = c_s * k * sqrt(1 + (xi*k/2)^2)
    which is the Lambda-model form with Lambda = xi^2/4 (verified by
    direct coefficient matching against the standard Bogoliubov result
    omega^2 = c_s^2 k^2 + (hbar^2/4m^2) k^4).
    """
    return xi**2 / 4.0


def lambda_from_dirac(eta, v_F):
    """
    *** SPECULATIVE -- NOT VALIDATED AGAINST ANY LITERATURE DERIVATION ***

    Lambda = (1 - eta^2) * v_F^2 / 4

    This formula was originally motivated by analogy with the BEC mapping
    (same algebraic shape) and re-justified in a later draft by citing
    trigonal/hexagonal warping in Dirac materials. That citation does not
    actually support this formula:

      Fu, L., PRL 103, 266801 (2009) derives a hexagonal warping term
      that is CUBIC in momentum in the Hamiltonian:
          H = v_F(k_x*sigma_y - k_y*sigma_x) + (lambda_w/2)(k_+^3+k_-^3)*sigma_z
      which gives
          E(k)^2 = v_F^2 k^2 + lambda_w^2 k^6 cos^2(3*theta)
      -- an ANISOTROPIC k^6 correction, not the isotropic k^4 correction
      this function computes. No power of eta or angular averaging turns
      a k^6 term into a k^4 term.

    No literature-supported derivation of an isotropic k^4 (Lambda-model)
    correction to the Dirac dispersion is currently known to the authors.
    This function is retained for exploratory/illustrative use only and
    should NOT be presented as a validated physical mapping. Do not use
    its output as a claimed experimental prediction.
    """
    import warnings
    warnings.warn(
        "lambda_from_dirac() uses a SPECULATIVE, non-literature-derived "
        "formula. See function docstring before using this result.",
        stacklevel=2,
    )
    return (1 - eta**2) * v_F**2 / 4.0


def lambda_from_photonic(beta4_k, c=2.998e8):
    """
    Photonic k-space fourth-order dispersion mapping.

    DEFINITION USED HERE: beta4_k is the coefficient of k^4 in the
    expansion of omega^2(k) (NOT omega(k) itself):

        omega(k)^2 = c^2*k^2 + beta4_k * k^4 + ...

    This is the mathematically correct point to look for a quartic term:
    expanding the Lambda-model dispersion directly,

        omega(k) = c*k*sqrt(1+Lambda*k^2)
        =>  omega(k)^2 = c^2 k^2 (1 + Lambda k^2) = c^2 k^2 + c^2 Lambda k^4

    exactly, with NO higher-order terms. Matching coefficients of k^4:

        beta4_k = c^2 * Lambda   =>   Lambda = beta4_k / c^2

    (units of beta4_k: [omega^2]/[k^4] = m^4/s^2, so that Lambda =
    beta4_k/c^2 comes out in m^2 as required.)

    IMPORTANT CORRECTION (previous versions of this function used
    Lambda = beta4_k/(12c) with beta4_k defined via a claimed expansion
    omega(k) = c*k + (beta4_k/24)*k^4. That expansion is WRONG: direct
    Taylor expansion of omega(k)=c*k*sqrt(1+Lambda*k^2) gives
        omega(k) = c*k + (Lambda*c/2)*k^3 + O(k^5)
    i.e. the leading correction to omega(k) itself is CUBIC in k, not
    quartic -- there is no k^4 term in omega(k) at this order at all.
    The quartic term only appears cleanly in omega^2(k), as used above.)

    STATUS: this formula is now a correctly-derived, dimensionally
    consistent identity WITHIN the Lambda-model. It is NOT yet connected
    to any standard, independently-measurable photonics quantity --
    standard telecom dispersion coefficients (beta_2, beta_3, beta_4 =
    d^n(beta)/d(omega)^n) are Taylor coefficients of the propagation
    constant around a nonzero pump/operating frequency, a structurally
    different expansion (see analysis in project notes: even in the
    fiber-optic event-horizon / "pure-quartic-soliton" literature, the
    co-moving-frame frequency near a group-velocity-matched point behaves
    as a quartic MINIMUM with no linear term, not as a deformed light
    cone). Deriving beta4_k from real beta_n measurements requires
    additional theoretical work not yet done. Treat beta4_k as an
    abstract input for now, not a directly-measured lab quantity.
    """
    return beta4_k / (c**2)


# ── Core fitting routine ──────────────────────────────────────────────────
def fit_lambda(k_data, omega_data, c_fixed=None):
    """
    Fit the Lambda-model dispersion relation to data.

    If c_fixed is None, both c and Lambda are fit (nonlinear).
    If c_fixed is given, only Lambda is fit (linear regression).

    Returns: dict with fitted parameters, R^2, and consistency-with-zero test.
    """
    k_data = np.asarray(k_data, dtype=float)
    omega_data = np.asarray(omega_data, dtype=float)

    if len(k_data) != len(omega_data):
        raise ValueError("k and omega arrays must have the same length")
    if len(k_data) < 3:
        raise ValueError("Need at least 3 data points for a meaningful fit")

    if c_fixed is not None:
        c = c_fixed
        # Linear fit: y = (omega/(c*k))^2 - 1 = Lambda * k^2
        y = (omega_data / (c * k_data))**2 - 1
        x = k_data**2

        Lambda_fit = np.sum(x * y) / np.sum(x * x)
        y_pred = Lambda_fit * x
        residuals = y - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        # Standard error of the slope (linear regression through origin)
        n = len(x)
        if n > 1:
            sigma2 = ss_res / (n - 1)
            se_Lambda = np.sqrt(sigma2 / np.sum(x * x))
        else:
            se_Lambda = float("nan")

    else:
        # Nonlinear fit for both c and Lambda using scipy
        from scipy.optimize import curve_fit

        def model(k, c, Lambda):
            return c * k * np.sqrt(np.maximum(1 + Lambda * k**2, 0))

        c_guess = np.mean(omega_data / k_data)
        popt, pcov = curve_fit(model, k_data, omega_data,
                                p0=[c_guess, 0.01], maxfev=10000)
        c, Lambda_fit = popt
        se_Lambda = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else float("nan")

        omega_pred = model(k_data, c, Lambda_fit)
        residuals = omega_data - omega_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((omega_data - np.mean(omega_data))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Consistency-with-zero: is Lambda statistically distinguishable from 0?
    if not np.isnan(se_Lambda) and se_Lambda > 0:
        n_sigma = abs(Lambda_fit) / se_Lambda
    else:
        n_sigma = float("nan")

    return {
        "Lambda": Lambda_fit,
        "Lambda_stderr": se_Lambda,
        "c": c,
        "R_squared": r_squared,
        "n_sigma_from_zero": n_sigma,
        "n_points": len(k_data),
    }


def print_report(result, domain_name=None, theory_Lambda=None,
                  synthetic_selfcheck=False):
    print("=" * 68)
    print("Lambda-MODEL FIT REPORT")
    print("=" * 68)
    print()
    print(f"  Data points used:        {result['n_points']}")
    print(f"  Fitted c:                {result['c']:.6e}")
    print(f"  Fitted Lambda:           {result['Lambda']:.6e}")
    print(f"  Standard error:          {result['Lambda_stderr']:.6e}")
    print(f"  R^2:                     {result['R_squared']:.6f}")
    print()

    n_sigma = result["n_sigma_from_zero"]
    if not np.isnan(n_sigma):
        print(f"  Significance vs Lambda=0: {n_sigma:.2f} sigma")
        if n_sigma < 2:
            print("  => Data is CONSISTENT with Lambda=0 (no detected dispersion)")
        elif n_sigma < 5:
            print("  => Data shows MARGINAL evidence for Lambda != 0")
        else:
            print("  => Data shows SIGNIFICANT evidence for Lambda != 0")
    print()

    if domain_name and theory_Lambda is not None:
        rel_diff = abs(result["Lambda"] - theory_Lambda) / abs(theory_Lambda) \
            if theory_Lambda != 0 else float("nan")
        print(f"  Domain: {domain_name}")
        print(f"  Theoretical Lambda (from domain mapping): {theory_Lambda:.6e}")
        print(f"  Fitted Lambda:                             {result['Lambda']:.6e}")
        print(f"  Relative difference:                       {rel_diff:.4f}")
        print()
        if synthetic_selfcheck:
            print("  NOTE: this comparison uses synthetic data generated FROM")
            print("  the same theoretical formula being tested. It validates the")
            print("  fitting procedure's correctness, NOT the physical validity")
            print("  of the domain mapping itself. Do not cite this as evidence")
            print("  that real experimental data follows this mapping.")
            print()
        if rel_diff < 0.1:
            print("  => Fitted Lambda is CONSISTENT with the domain-specific")
            print("     theoretical mapping (within 10%).")
        else:
            print("  => Fitted Lambda DIFFERS from the theoretical mapping by")
            print("     more than 10%. Either the mapping does not apply to")
            print("     this system, or additional physics is present.")
    print()


def load_csv_column(path):
    """Load a single column of numbers from a CSV file (no header)."""
    return np.loadtxt(path, delimiter=",")


def main():
    parser = argparse.ArgumentParser(
        description="Fit the Lambda-model dispersion relation to your data, "
                    "or compute the theoretical Lambda for a known domain."
    )
    parser.add_argument("--omega", type=str, help="CSV file with omega values")
    parser.add_argument("--k", type=str, help="CSV file with k values")
    parser.add_argument("--c", type=float, default=None,
                         help="Fix the speed parameter c (optional)")

    parser.add_argument("--domain", type=str, choices=["BEC", "dirac", "photonic"],
                         help="Compute theoretical Lambda for a known domain")
    parser.add_argument("--xi", type=float, help="[BEC] healing length")
    parser.add_argument("--cs", type=float, help="[BEC] sound speed (informational)")
    parser.add_argument("--eta", type=float, help="[dirac, SPECULATIVE] warping param")
    parser.add_argument("--vF", type=float, help="[dirac, SPECULATIVE] Fermi velocity")
    parser.add_argument("--beta4k", type=float,
                         help="[photonic] coefficient of k^4 in omega^2(k), m^4/s^2")

    args = parser.parse_args()

    if args.domain:
        if args.domain == "BEC":
            if args.xi is None:
                sys.exit("--xi required for --domain BEC")
            Lam = lambda_from_BEC(args.xi)
            print(f"Theoretical Lambda (BEC, xi={args.xi}): {Lam:.6e}")
        elif args.domain == "dirac":
            if args.eta is None or args.vF is None:
                sys.exit("--eta and --vF required for --domain dirac")
            print("WARNING: the Dirac mapping is speculative and not "
                  "literature-validated. See lambda_from_dirac() docstring.")
            Lam = lambda_from_dirac(args.eta, args.vF)
            print(f"Theoretical Lambda (Dirac, eta={args.eta}, vF={args.vF}): "
                  f"{Lam:.6e}")
        elif args.domain == "photonic":
            if args.beta4k is None:
                sys.exit("--beta4k required for --domain photonic "
                          "(coefficient of k^4 in omega^2(k), units m^4/s^2)")
            c_val = args.c if args.c else 2.998e8
            Lam = lambda_from_photonic(args.beta4k, c_val)
            print(f"Theoretical Lambda (photonic, beta4_k={args.beta4k}, "
                  f"c={c_val}): {Lam:.6e}")
        return

    if args.omega and args.k:
        k_data = load_csv_column(args.k)
        omega_data = load_csv_column(args.omega)
        result = fit_lambda(k_data, omega_data, c_fixed=args.c)
        print_report(result)
        return

    # No arguments: run a self-test with synthetic data
    print("No --omega/--k data or --domain specified. Running self-test with")
    print("synthetic data (Lambda_true=0.05, 1% noise)...\n")

    rng = np.random.default_rng(0)
    k_data = np.linspace(0.5, 5.0, 12)
    Lambda_true = 0.05
    omega_true = k_data * np.sqrt(1 + Lambda_true * k_data**2)
    omega_data = omega_true * (1 + rng.normal(0, 0.01, size=k_data.shape))

    result = fit_lambda(k_data, omega_data, c_fixed=1.0)
    print_report(result)
    print(f"(True Lambda was {Lambda_true} for this self-test)")


if __name__ == "__main__":
    main()
