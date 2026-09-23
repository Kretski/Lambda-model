"""
Tests for fit_lambda() in paper3/dispersion/lambda_experimental_validator.py.

Every test calls the repository's own fitting routine on noisy synthetic
data. None of them passes by construction: each one fails if fit_lambda()
returns a biased value, a wrong standard error, or a good fit to data that
do not follow the Lambda-model. The recovery and null tests repeat the fit
over many noise realisations, so the reported standard error is checked
statistically, not just the point estimate.

These are code checks. They are not physical or experimental results.

Model convention: omega = c k sqrt(1 + Lambda k^2).

Run from the repository root:  python -m pytest paper3/dispersion/tests
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lambda_experimental_validator import fit_lambda  # noqa: E402


K = np.linspace(0.15, 1.2, 60)
C_TRUE = 1.0
SIGMA = 2.0e-3           # absolute noise on omega
N_REALISATIONS = 300


def lambda_model(k, c, lam):
    return c * k * np.sqrt(1.0 + lam * k**2)


def reduced_chi2(k, omega, result, sigma):
    pred = lambda_model(k, result["c"], result["Lambda"])
    dof = len(k) - 2
    return np.sum(((omega - pred) / sigma) ** 2) / dof


# ---------------------------------------------------------------------
# 1. Recovery: unbiased, and the reported standard error is honest
# ---------------------------------------------------------------------

def test_lambda_model_recovery():
    rng = np.random.default_rng(1)
    lam_true = 0.05
    pulls = []
    for _ in range(N_REALISATIONS):
        omega = lambda_model(K, C_TRUE, lam_true) + rng.normal(0, SIGMA, K.size)
        r = fit_lambda(K, omega)            # c and Lambda both free
        pulls.append((r["Lambda"] - lam_true) / r["Lambda_stderr"])
    pulls = np.array(pulls)

    # Unbiased: mean pull consistent with 0.
    assert abs(pulls.mean()) < 4 / np.sqrt(N_REALISATIONS)
    # Honest error bar: pull width close to 1.
    assert 0.85 < pulls.std() < 1.15
    # About 95% of fits contain the true value within 2 stderr.
    assert 0.90 < np.mean(np.abs(pulls) < 2) < 0.99


def test_lambda_recovery_with_fixed_c():
    rng = np.random.default_rng(2)
    lam_true = 0.05
    fits = []
    for _ in range(N_REALISATIONS):
        omega = lambda_model(K, C_TRUE, lam_true) + rng.normal(0, SIGMA, K.size)
        fits.append(fit_lambda(K, omega, c_fixed=C_TRUE)["Lambda"])
    fits = np.array(fits)
    assert abs(fits.mean() - lam_true) < 4 * fits.std() / np.sqrt(N_REALISATIONS)


# ---------------------------------------------------------------------
# 2. Null control: Lambda = 0 data must not look significant too often
# ---------------------------------------------------------------------

def test_lambda_zero_control():
    rng = np.random.default_rng(3)
    n_sigma = []
    for _ in range(N_REALISATIONS):
        omega = lambda_model(K, C_TRUE, 0.0) + rng.normal(0, SIGMA, K.size)
        n_sigma.append(fit_lambda(K, omega)["n_sigma_from_zero"])
    n_sigma = np.array(n_sigma)

    # Nominal false-positive rate at 3 sigma is 0.27%; allow up to 2%.
    assert np.mean(n_sigma > 3) < 0.02
    # The 2-sigma rate should be near its nominal 4.6%.
    assert np.mean(n_sigma > 2) < 0.10


# ---------------------------------------------------------------------
# 3. Model discrimination: data that are NOT Lambda-model must fit badly
# ---------------------------------------------------------------------

def test_pure_quartic_is_not_lambda_model():
    """omega = c k + beta k^4 is not of the Lambda-model form.

    For a strong enough quartic term, the Lambda-model fit must leave
    residuals far above the noise, while genuine Lambda-model data give
    reduced chi^2 close to 1. The model that generated the data is never
    used to judge the fit.
    """
    rng = np.random.default_rng(4)
    beta = 0.3

    omega_quartic = C_TRUE * K + beta * K**4 + rng.normal(0, SIGMA, K.size)
    chi2_quartic = reduced_chi2(K, omega_quartic, fit_lambda(K, omega_quartic), SIGMA)

    omega_lambda = lambda_model(K, C_TRUE, 0.05) + rng.normal(0, SIGMA, K.size)
    chi2_lambda = reduced_chi2(K, omega_lambda, fit_lambda(K, omega_lambda), SIGMA)

    assert 0.5 < chi2_lambda < 1.6
    assert chi2_quartic > 10


def test_weak_quartic_is_indistinguishable():
    """Documents a limitation rather than a success.

    For a weak quartic term (beta = 0.02) at this noise level and k-range,
    omega = c k + beta k^4 and the Lambda-model are practically degenerate:
    the Lambda-model fits it with reduced chi^2 near 1. A good Lambda-model
    fit alone therefore does NOT show that data follow the Lambda-model.
    If this test starts failing, the discrimination power has changed and
    the statement above should be re-examined.
    """
    rng = np.random.default_rng(5)
    omega = C_TRUE * K + 0.02 * K**4 + rng.normal(0, SIGMA, K.size)
    chi2 = reduced_chi2(K, omega, fit_lambda(K, omega), SIGMA)
    assert chi2 < 2.0


# ---------------------------------------------------------------------
# 4. Input validation
# ---------------------------------------------------------------------

def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        fit_lambda(K, K[:-1])


def test_rejects_too_few_points():
    with pytest.raises(ValueError):
        fit_lambda(K[:2], K[:2])
