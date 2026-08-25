"""
waveform.py
==============

Frequency-domain inspiral waveform model, GR baseline plus a Lambda
dispersion phase correction, for matched-filter recovery testing.

WHY FREQUENCY DOMAIN, WHY PHASE-BASED DISPERSION:

All prior attempts in this line of work (Hilbert transform, zero-
crossing, STFT ridge -- see paper3/gw/*.py, marked legacy/diagnostic)
tried to extract an instantaneous time-domain frequency track from
noisy strain and compare it against a time-domain PN prediction. Every
one of these showed severe, method-specific systematic bias, confirmed
by injection/recovery testing.

The standard, robust alternative used throughout LIGO/Virgo/KAGRA data
analysis is to work in the FREQUENCY DOMAIN and put any physics
modification directly into the WAVEFORM PHASE model, then use matched
filtering (or full likelihood inference) against the strain -- never
extracting an intermediate "instantaneous frequency" at all. This
removes the entire class of extraction-method bias identified in the
diagnostic scripts.

GR BASELINE (stationary phase approximation, leading + next-order PN):

    Psi_GR(f) = 2*pi*f*tc - phi_c - pi/4
                + (3/128) * (pi * Mc * f)^(-5/3)
                * [1 + ... higher PN terms, leading order used here]

We use the leading-order (Newtonian, 0PN) stationary-phase amplitude
and phase, which is the standard TaylorF2 leading term:

    Psi_GR(f) = 2*pi*f*tc - phi_c - pi/4
                + (3/(128*eta)) * (pi*Mc*f)^(-5/3)

    A_GR(f) = sqrt(5/24) * pi^(-2/3) * Mc^(5/6) * f^(-7/6) / D_eff

This is intentionally NOT full TaylorF2 with all PN correction terms
(that requires lalsimulation for a production-grade implementation).
For a PROOF-OF-CONCEPT test of whether the Lambda dispersion parameter
is identifiable via matched filtering at all -- decoupled from
real-waveform PN accuracy -- the leading-order term is sufficient,
exactly as leading-order 0PN was sufficient (and clearly labeled as
such) in the legacy time-domain scripts.

LAMBDA DISPERSION PHASE:

Physical motivation (Paper 2, FLRW section): a photon/GW of frequency
omega propagating a comoving distance picks up a dispersion-induced
time delay

    Delta T(omega) = -(Lambda / 2c^3) * omega^2 * K(z)

A time delay Delta T(f) at frequency f corresponds to an ADDITIONAL
phase in the frequency-domain waveform:

    Delta Psi(f) = 2*pi*f * Delta T(f)
                 = -(Lambda / c^3) * (2*pi*f)^2 * pi*f * K(z)
                 = -(4*pi^3*Lambda*K(z)/c^3) * f^3

So the total phase model is:

    Psi(f; Lambda) = Psi_GR(f) - (4*pi^3*Lambda*K(z)/c^3) * f^3

This f^3 correction is the frequency-domain analogue of the f^2
time-domain delta_t signature used (and shown to be extraction-biased)
in the legacy scripts -- but here it enters directly as an analytic
phase term in a matched-filter framework, with no intermediate
frequency-extraction step.
"""

import numpy as np

G_SI = 6.674e-11
C_SI = 2.998e8
MSUN_SI = 1.989e30
MPC_SI = 3.0857e22


def chirp_mass_SI(m1_msun, m2_msun):
    m1, m2 = m1_msun * MSUN_SI, m2_msun * MSUN_SI
    return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)


def symmetric_mass_ratio(m1_msun, m2_msun):
    m1, m2 = m1_msun, m2_msun
    return (m1 * m2) / (m1 + m2) ** 2


def taylorf2_leading_order(f, m1_msun, m2_msun, tc=0.0, phi_c=0.0,
                            distance_Mpc=440.0):
    """
    Leading-order (Newtonian) TaylorF2 stationary-phase waveform.

    Returns the complex frequency-domain strain h(f) = A(f)*exp(i*Psi(f))
    for f > 0 (positive-frequency part only; use with a real-signal
    matched filter convention that handles this consistently).
    """
    Mc = chirp_mass_SI(m1_msun, m2_msun)
    eta = symmetric_mass_ratio(m1_msun, m2_msun)
    D = distance_Mpc * MPC_SI

    f = np.asarray(f, dtype=float)
    f_safe = np.maximum(f, 1e-6)

    # Leading-order amplitude (Newtonian quadrupole, SPA)
    A = np.sqrt(5.0 / 24.0) * np.pi ** (-2.0 / 3.0) * \
        (G_SI * Mc / C_SI ** 3) ** (5.0 / 6.0) * C_SI / D * f_safe ** (-7.0 / 6.0)

    # Leading-order phase (0PN term of TaylorF2)
    v = (np.pi * G_SI * Mc / C_SI ** 3 * f_safe) ** (1.0 / 3.0)
    Psi_GR = 2 * np.pi * f_safe * tc - phi_c - np.pi / 4 + \
        (3.0 / 128.0) * (np.pi * G_SI * Mc / C_SI ** 3 * f_safe) ** (-5.0 / 3.0)

    return A, Psi_GR


def lambda_phase_correction(f, Lambda, K_z):
    """
    Delta_Psi(f) = -(4*pi^3*Lambda*K(z)/c^3) * f^3

    See module docstring for derivation.
    """
    f = np.asarray(f, dtype=float)
    return -(4.0 * np.pi ** 3 * Lambda * K_z / C_SI ** 3) * f ** 3


def waveform_frequency_domain(f, m1_msun, m2_msun, Lambda, K_z,
                               tc=0.0, phi_c=0.0, distance_Mpc=440.0):
    """
    Full frequency-domain waveform including the Lambda dispersion
    phase correction. Returns complex h(f).
    """
    A, Psi_GR = taylorf2_leading_order(f, m1_msun, m2_msun, tc, phi_c,
                                        distance_Mpc)
    Psi_total = Psi_GR + lambda_phase_correction(f, Lambda, K_z)
    return A * np.exp(1j * Psi_total)


def cosmological_K_factor(z, H0=67.4, Om=0.315, OL=0.685):
    """K(z) = integral_0^z (1+z')^2/H(z') dz' [seconds]. Same as Paper 2."""
    from scipy.integrate import quad

    def H_of_zp(zp):
        return H0 * np.sqrt(Om * (1 + zp) ** 3 + OL) * 1e3 / MPC_SI

    def integrand(zp):
        return (1 + zp) ** 2 / H_of_zp(zp)

    val, _ = quad(integrand, 0, z)
    return val
