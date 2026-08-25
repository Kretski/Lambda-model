"""
likelihood.py
================

Frequency-domain matched-filter likelihood for the GR+Lambda waveform
model, and a simple grid-search / Fisher-matrix-based Lambda inference.

STANDARD GW LIKELIHOOD (frequency domain, colored Gaussian noise):

    ln L(Lambda) = -(1/2) * <d - h(Lambda) | d - h(Lambda)>
                 = <d|h> - (1/2)<h|h> + const

where the noise-weighted inner product is

    <a|b> = 4 * Re[ integral_fmin^fmax  a*(f) b(f) / S_n(f) df ]

and S_n(f) is the one-sided noise PSD.

We do NOT attempt sky location, inclination, polarization, or full
parameter estimation here -- this is a PROOF OF CONCEPT that Lambda is
identifiable via matched filtering given a KNOWN, FIXED set of other
parameters (masses, distance, coalescence time/phase). This isolates
the Lambda-identifiability question from the much larger problem of
full parameter estimation, exactly as recommended: a synthetic
matched-filter test BEFORE attempting real GW150914 data.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from waveform import waveform_frequency_domain, cosmological_K_factor


def aligo_like_psd(f):
    """Same approximate aLIGO-shaped PSD as used in the legacy scripts,
    for consistency."""
    f = np.maximum(np.asarray(f, dtype=float), 1.0)
    f0 = 215.0
    x = f / f0
    S = 1e-49 * (x ** -4.14 - 5 * x ** -2 +
                 111 * (1 - x ** 2 + 0.5 * x ** 4) / (1 + 0.5 * x ** 2))
    return np.abs(S) + 1e-47


def noise_weighted_inner_product(a, b, f, psd, df):
    """<a|b> = 4 * Re[ sum a*(f) b(f) / S_n(f) * df ]"""
    integrand = np.conj(a) * b / psd
    return 4.0 * np.real(np.sum(integrand) * df)


def log_likelihood(data_fd, f, psd, df, m1, m2, Lambda, K_z,
                    tc=0.0, phi_c=0.0, distance_Mpc=440.0):
    """
    ln L(Lambda) = <d|h> - 0.5*<h|h>  (dropping the Lambda-independent
    <d|d> term, which cancels in any Lambda comparison/ratio)
    """
    h = waveform_frequency_domain(f, m1, m2, Lambda, K_z, tc, phi_c,
                                   distance_Mpc)
    dh = noise_weighted_inner_product(data_fd, h, f, psd, df)
    hh = noise_weighted_inner_product(h, h, f, psd, df)
    return dh - 0.5 * hh


def snr_optimal(f, psd, df, m1, m2, Lambda, K_z, tc=0.0, phi_c=0.0,
                 distance_Mpc=440.0):
    """Optimal (matched-filter) SNR of the template itself: sqrt(<h|h>)."""
    h = waveform_frequency_domain(f, m1, m2, Lambda, K_z, tc, phi_c,
                                   distance_Mpc)
    hh = noise_weighted_inner_product(h, h, f, psd, df)
    return np.sqrt(max(hh, 0.0))


def grid_search_lambda(data_fd, f, psd, df, m1, m2, K_z,
                        Lambda_grid, tc=0.0, phi_c=0.0, distance_Mpc=440.0):
    """
    Evaluate ln L(Lambda) over a grid, return the grid, the log-
    likelihoods, the maximum-likelihood Lambda, and a Gaussian-
    approximation 1-sigma width from the curvature at the peak
    (standard Fisher-matrix-style error estimate from a parabolic fit
    near the maximum).
    """
    logL = np.array([
        log_likelihood(data_fd, f, psd, df, m1, m2, Lam, K_z, tc, phi_c,
                        distance_Mpc)
        for Lam in Lambda_grid
    ])

    i_max = np.argmax(logL)
    Lambda_ml = Lambda_grid[i_max]

    # Parabolic (Fisher-matrix-like) error estimate from local curvature
    if 0 < i_max < len(Lambda_grid) - 1:
        dL = Lambda_grid[1] - Lambda_grid[0]
        d2logL = (logL[i_max + 1] - 2 * logL[i_max] + logL[i_max - 1]) / dL ** 2
        if d2logL < 0:
            Lambda_err = np.sqrt(-1.0 / d2logL)
        else:
            Lambda_err = np.nan
    else:
        Lambda_err = np.nan

    return Lambda_grid, logL, Lambda_ml, Lambda_err
