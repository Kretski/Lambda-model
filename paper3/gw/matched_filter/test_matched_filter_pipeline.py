"""
Tests for the frequency-domain matched-filter machinery in waveform.py and
likelihood.py.

Every test calls the repository's own functions (lambda_phase_correction,
cosmological_K_factor, waveform_frequency_domain, grid_search_lambda) and can
fail. Recovery and null tests use many coloured-Gaussian-noise realisations,
so the reported error bar is checked statistically, not just the point
estimate.

These are code checks on synthetic data with known masses, distance and
coalescence parameters. They are not evidence about Lambda in any real
gravitational-wave event.

Replaces an earlier version of this file whose tests re-implemented the
formulas inline (with the obsolete 1 + 2 Lambda k^2 convention) and compared
data against the exact injected signal, so they could not fail.

Run from the repository root:
    python -m pytest paper3/gw/matched_filter/test_matched_filter_pipeline.py
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from waveform import (  # noqa: E402
    C_SI,
    MPC_SI,
    cosmological_K_factor,
    lambda_phase_correction,
    waveform_frequency_domain,
)
from likelihood import aligo_like_psd, grid_search_lambda, snr_optimal  # noqa: E402


# Fixed synthetic configuration (GW150914-like masses, SNR ~ 25).
M1, M2 = 36.0, 29.0
DIST_MPC = 1500.0
Z = 0.09
DF = 0.25
F = np.arange(20.0, 300.0, DF)
PSD = aligo_like_psd(F)
K_Z = cosmological_K_factor(Z)
# Grid kept inside one phase-aliasing period (about +-0.57 here).
GRID = np.arange(-0.5, 0.5 + 1e-12, 0.005)
N_REALISATIONS = 100


def coloured_noise(rng):
    """Complex Gaussian noise whose inner products have the right variance:
    E[<n|h>] = 0 and Var[<n|h>] = <h|h> for the inner product in likelihood.py."""
    sigma = np.sqrt(PSD / (4.0 * DF))
    return rng.normal(0, sigma) + 1j * rng.normal(0, sigma)


def recover(lam_true, rng, k_template=K_Z):
    h = waveform_frequency_domain(F, M1, M2, lam_true, K_Z, distance_Mpc=DIST_MPC)
    data = h + coloured_noise(rng)
    _, _, lam_ml, lam_err = grid_search_lambda(
        data, F, PSD, DF, M1, M2, k_template, GRID, distance_Mpc=DIST_MPC)
    return lam_ml, lam_err


# ---------------------------------------------------------------------
# 1. Building blocks
# ---------------------------------------------------------------------

def test_signal_is_in_a_realistic_snr_range():
    snr = snr_optimal(F, PSD, DF, M1, M2, 0.0, K_Z, distance_Mpc=DIST_MPC)
    assert 10 < snr < 60


def test_K_factor_small_redshift_limit():
    """K(z) -> z / H0 for z -> 0 (analytic limit, independent of the code)."""
    z = 0.005
    H0_si = 67.4e3 / MPC_SI
    assert abs(cosmological_K_factor(z) / (z / H0_si) - 1) < 0.01


def test_phase_correction_scaling_and_sign():
    f = np.array([50.0, 100.0])
    d = lambda_phase_correction(f, 1.0, K_Z)
    assert np.isclose(d[1] / d[0], 8.0)                       # f^3
    assert np.isclose(lambda_phase_correction(f, 2.0, K_Z)[0], 2 * d[0])  # linear in Lambda
    assert d[0] < 0                                            # sign convention
    expected = -(4 * np.pi**3 * K_Z / C_SI**3) * 50.0**3
    assert np.isclose(d[0], expected)


# ---------------------------------------------------------------------
# 2. Recovery: unbiased, and the curvature error bar is honest
# ---------------------------------------------------------------------

def test_lambda_recovery_is_unbiased_with_calibrated_errors():
    rng = np.random.default_rng(10)
    lam_true = 0.1
    pulls = []
    for _ in range(N_REALISATIONS):
        lam_ml, lam_err = recover(lam_true, rng)
        assert np.isfinite(lam_err)
        pulls.append((lam_ml - lam_true) / lam_err)
    pulls = np.array(pulls)
    assert abs(pulls.mean()) < 4 / np.sqrt(N_REALISATIONS)
    assert 0.8 < pulls.std() < 1.25


# ---------------------------------------------------------------------
# 3. Null control
# ---------------------------------------------------------------------

def test_lambda_zero_control():
    rng = np.random.default_rng(11)
    n_sigma = []
    for _ in range(N_REALISATIONS):
        lam_ml, lam_err = recover(0.0, rng)
        n_sigma.append(abs(lam_ml) / lam_err)
    n_sigma = np.array(n_sigma)
    assert np.mean(n_sigma > 3) < 0.03
    assert np.mean(n_sigma > 2) < 0.12


# ---------------------------------------------------------------------
# 4. Documented limitation: only the product Lambda * K(z) is measured
# ---------------------------------------------------------------------

def test_only_lambda_times_K_is_identified():
    """If the template uses K(z) wrong by a factor 2, the recovered Lambda is
    off by the inverse factor. The phase depends on Lambda*K(z) only, so any
    error in the assumed distance/redshift goes straight into Lambda."""
    rng = np.random.default_rng(12)
    lam_true = 0.1
    fits = [recover(lam_true, rng, k_template=K_Z / 2)[0] for _ in range(30)]
    assert abs(np.mean(fits) / (2 * lam_true) - 1) < 0.1
