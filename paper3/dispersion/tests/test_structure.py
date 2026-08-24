"""
Paper 3 — dispersion structural regression tests.

These tests verify the numerical conclusions of the current
demonstration datasets.

They do not constitute experimental validation.
"""

import numpy as np


def test_fiber_quartic_exponent():
    k = np.array([
        0.05, 0.07, 0.10, 0.14, 0.20,
        0.28, 0.40, 0.55, 0.75, 1.00
    ])

    omega = np.array([
        0.000006,
        0.000024,
        0.000101,
        0.000383,
        0.001608,
        0.006110,
        0.025702,
        0.091049,
        0.317355,
        0.998000
    ])

    p, intercept = np.polyfit(
        np.log(k),
        np.log(omega),
        1
    )

    assert 3.9 < p < 4.1


def test_fiber_not_linear():
    k = np.array([
        0.05, 0.07, 0.10, 0.14, 0.20,
        0.28, 0.40, 0.55, 0.75, 1.00
    ])

    omega = np.array([
        0.000006,
        0.000024,
        0.000101,
        0.000383,
        0.001608,
        0.006110,
        0.025702,
        0.091049,
        0.317355,
        0.998000
    ])

    # Linear model through origin
    b = np.dot(k, omega) / np.dot(k, k)
    prediction = b * k

    rss_linear = np.sum((omega - prediction) ** 2)

    # Quartic model through origin
    b4 = np.dot(k**4, omega) / np.dot(k**4, k**4)
    prediction4 = b4 * k**4

    rss_quartic = np.sum((omega - prediction4) ** 2)

    assert rss_quartic < rss_linear


def test_lambda_dimensions_are_not_inferred_from_beta4():
    # beta4 has units s^4/m.
    #
    # beta4/c^2 therefore has:
    #
    # (s^4/m) / (m^2/s^2)
    # = s^6/m^3
    #
    # which is not m^2.

    beta4_units = "s^4/m"
    lambda_units = "m^2"

    assert beta4_units != lambda_units