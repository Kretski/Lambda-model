"""
synthetic_injection.py
==========================

Generates a synthetic frequency-domain "data" segment: a known-Lambda
waveform injected into colored Gaussian noise realized directly in the
frequency domain (the natural domain for the matched-filter approach,
avoiding any time-domain FFT/windowing subtleties).
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from waveform import waveform_frequency_domain
from likelihood import aligo_like_psd


def make_frequency_grid(f_min=20.0, f_max=400.0, duration=8.0):
    """
    Frequency grid consistent with a time-domain segment of the given
    duration (df = 1/duration, standard FFT frequency resolution).
    """
    df = 1.0 / duration
    f = np.arange(f_min, f_max, df)
    return f, df


def generate_injection(m1, m2, Lambda_true, K_z, f, df, psd,
                        distance_Mpc=440.0, tc=0.0, phi_c=0.0,
                        seed=0, add_noise=True):
    """
    data(f) = h(f; Lambda_true) + n(f)

    n(f) is complex Gaussian noise with the correct frequency-domain
    variance for the given one-sided PSD and frequency resolution df:
    Var[Re(n)] = Var[Im(n)] = S_n(f) / (4*df)  (standard convention for
    a one-sided PSD and this inner-product normalization).
    """
    h = waveform_frequency_domain(f, m1, m2, Lambda_true, K_z, tc, phi_c,
                                   distance_Mpc)

    if not add_noise:
        return h

    rng = np.random.default_rng(seed)
    sigma = np.sqrt(psd / (4 * df))
    noise = rng.normal(0, sigma) + 1j * rng.normal(0, sigma)

    return h + noise
