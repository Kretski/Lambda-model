"""
lalsim_waveform.py
======================

Thin wrapper around lalsimulation's SimInspiralFD, producing frequency-
domain waveforms compatible with the Lambda-phase-correction machinery
in waveform.py, likelihood.py, and the Stage 3/4 pipeline.

WHY THIS FILE EXISTS:

Stage 4 (stage4_coherent_h1l1_validation.py) showed that the hand-rolled
0PN vs 1PN TaylorF2 baselines produce wildly different Lambda point
estimates on real GW150914 data (Λ=+6.57 vs Λ=−1.27). This proves the
hand-rolled waveform approach cannot distinguish a genuine Lambda signal
from PN-truncation error. The only way forward is to use INDEPENDENTLY
VALIDATED waveform models (not derived by us), so that cross-family
comparisons test genuine model systematics rather than our own
possible derivation errors.

This module generates GR baseline waveforms via lalsimulation
(IMRPhenomD as the primary choice: fast, frequency-domain-native,
well-validated against numerical relativity), and adds the SAME
Lambda dispersion phase term used throughout this repository:

    Delta_Psi(f) = -(4*pi^3*Lambda*K(z)/c^3) * f^3

This is added AFTER lalsimulation generates the GR waveform, exactly
as in waveform.py -- we do not modify lalsimulation's internals, only
post-multiply the returned frequency-domain waveform by
exp(i*Delta_Psi(f)). This keeps the Lambda parameterization identical
across all waveform families tested (hand-rolled 0PN/1PN in Stage 4,
IMRPhenomD/SEOBNR here), so cross-family comparisons isolate GR-baseline
systematics from Lambda-model systematics.
"""

import numpy as np
from waveform import lambda_phase_correction

try:
    import lal
    import lalsimulation as lalsim
    LAL_AVAILABLE = True
except ImportError:
    LAL_AVAILABLE = False

C_SI = 2.998e8
MSUN_SI = 1.989e30
MPC_SI = 3.0857e22


AVAILABLE_APPROXIMANTS = {
    "IMRPhenomD": "IMRPhenomD",
    "IMRPhenomPv2": "IMRPhenomPv2",
    "SEOBNRv4": "SEOBNRv4",
    "TaylorF2": "TaylorF2",
}


def _check_lal():
    if not LAL_AVAILABLE:
        raise ImportError(
            "lalsimulation is not importable in this Python environment. "
            "This module requires a working LALSuite installation (see "
            "STAGE5 setup notes -- typically via WSL on Windows, since "
            "LALSuite has no native Windows wheel/conda package)."
        )


def generate_gr_waveform_lalsim(f_min, f_max, df, m1_msun, m2_msun,
                                 distance_Mpc=440.0, inclination=0.0,
                                 approximant="IMRPhenomD",
                                 spin1z=0.0, spin2z=0.0,
                                 f_max_generation=None):
    """
    Generate a frequency-domain GR waveform via lalsimulation, on a
    uniform frequency grid [f_min, f_max) with spacing df, matching the
    grid convention used elsewhere in this repository (f = f_min +
    n*df).

    GENERATION BAND vs ANALYSIS BAND (important):

    lalsimulation's SimInspiralFD uses the requested f_max to set the
    internal generation bandwidth/sample rate. For time-domain-native
    models like SEOBNRv4 (generated in the time domain, then FFT'd),
    this bandwidth must be wide enough to resolve the FULL physical
    merger-ringdown content for the given masses, REGARDLESS of what
    narrower band we actually want to analyze. Requesting a narrow
    f_max (e.g. 100 Hz, well below the ringdown frequency for typical
    BBH masses) causes SEOBNRv4 to fail with "Ringdown frequency >
    Nyquist frequency" -- a hard physical/numerical constraint, not a
    bug in this wrapper.

    FIX: always GENERATE on a wide, fixed band (f_max_generation,
    default 1024 Hz -- safely above ringdown for stellar-mass BBH),
    then CROP the result down to the requested [f_min, f_max) analysis
    band. This separates "how wide a band lalsimulation needs to do
    its job correctly" from "how wide a band we want to analyze" --
    the two are conflated if f_max is used for both, which is exactly
    what caused Stage 5F-1's narrow-band ([20,100) Hz) test to fail.

    Returns (f_grid, h_plus_complex) where h_plus_complex is the
    frequency-domain plus-polarization strain, restricted to the
    requested [f_min, f_max) ANALYSIS band and resampled onto the exact
    f_grid via linear interpolation of real/imaginary parts.
    """
    _check_lal()

    if f_max_generation is None:
        f_max_generation = max(f_max, 1024.0)

    m1_SI = m1_msun * lal.MSUN_SI
    m2_SI = m2_msun * lal.MSUN_SI
    distance_SI = distance_Mpc * 1e6 * lal.PC_SI

    approx = lalsim.GetApproximantFromString(AVAILABLE_APPROXIMANTS[approximant])

    hp, hc = lalsim.SimInspiralFD(
        m1_SI, m2_SI,
        0.0, 0.0, spin1z,      # spin1x, spin1y, spin1z
        0.0, 0.0, spin2z,      # spin2x, spin2y, spin2z
        distance_SI,
        inclination,
        0.0,                    # phiRef
        0.0,                    # longAscNodes
        0.0,                    # eccentricity
        0.0,                    # meanPerAno
        df,                      # deltaF
        f_min,                   # f_min
        f_max_generation,        # f_max -- WIDE generation band, see above
        f_min,                   # f_ref
        None,                     # LALDict (no extra params)
        approx,
    )

    # hp.data.data is a complex numpy array on lalsimulation's internal
    # frequency grid: f_lal[k] = k * df, k = 0, 1, 2, ...
    n_lal = hp.data.length
    f_lal = np.arange(n_lal) * df

    h_lal = np.array(hp.data.data)

    # Resample onto the requested ANALYSIS grid (f_min, f_min+df, ...,
    # <f_max) -- narrower than the generation band, per the above.
    f_grid = np.arange(f_min, f_max, df)

    h_real = np.interp(f_grid, f_lal, np.real(h_lal))
    h_imag = np.interp(f_grid, f_lal, np.imag(h_lal))
    h_grid = h_real + 1j * h_imag

    return f_grid, h_grid


def waveform_lalsim_with_lambda(f, m1_msun, m2_msun, Lambda, K_z,
                                 tc=0.0, phi_c=0.0, distance_Mpc=440.0,
                                 approximant="IMRPhenomD", df=None,
                                 f_max_generation=None):
    """
    Drop-in replacement for waveform.waveform_frequency_domain(), using
    a validated lalsimulation approximant for the GR part, with the
    SAME Lambda dispersion phase term applied on top.

    NOTE ON tc/phi_c: lalsimulation's SimInspiralFD returns a waveform
    referenced to its own internal time/phase convention (merger near
    t=0 in the time-domain sense, with a corresponding linear phase
    term already present in the frequency domain). To keep this
    consistent with the tc/phi_c convention used in waveform.py's
    hand-rolled model, we apply an ADDITIONAL linear-in-f phase shift
    exp(i*2*pi*f*tc - i*phi_c) on top of the lalsimulation output. This
    matches how tc/phi_c are used for the H1/L1 time-shift in Stage 4's
    joint likelihood.

    df must be provided explicitly (lalsimulation requires a fixed
    frequency spacing to generate the waveform); if the input f array
    is not uniformly spaced with spacing df, results will not align
    correctly -- this is checked.
    """
    _check_lal()

    f = np.asarray(f, dtype=float)
    if df is None:
        df = f[1] - f[0]
    if not np.allclose(np.diff(f), df, rtol=1e-6):
        raise ValueError("f array must be uniformly spaced with spacing df "
                          "for lalsimulation waveform generation")

    f_min, f_max = f[0], f[-1] + df

    f_grid, h_gr = generate_gr_waveform_lalsim(
        f_min, f_max, df, m1_msun, m2_msun, distance_Mpc=distance_Mpc,
        approximant=approximant, f_max_generation=f_max_generation)

    if not np.allclose(f_grid, f, rtol=1e-9):
        # Should not happen if df/f_min/f_max are consistent, but guard
        # with an explicit re-interpolation rather than silently
        # misaligning arrays
        h_real = np.interp(f, f_grid, np.real(h_gr))
        h_imag = np.interp(f, f_grid, np.imag(h_gr))
        h_gr = h_real + 1j * h_imag

    # Additional tc/phi_c linear phase (see docstring)
    extra_phase = 2 * np.pi * f * tc - phi_c
    h_gr = h_gr * np.exp(1j * extra_phase)

    # Lambda dispersion phase -- canonical implementation in waveform.py
    Delta_Psi = lambda_phase_correction(f, Lambda, K_z)
    h_total = h_gr * np.exp(1j * Delta_Psi)

    return h_total


def check_lalsimulation_available():
    """Quick diagnostic: is lalsimulation importable and functional?"""
    if not LAL_AVAILABLE:
        print("lalsimulation: NOT AVAILABLE")
        print("  Install via: pip install lalsuite  (Linux/macOS/WSL only)")
        return False

    print(f"lalsimulation: available")
    print(f"  LAL version: {lal.__version__}")

    try:
        f_grid, h = generate_gr_waveform_lalsim(
            f_min=20.0, f_max=400.0, df=1.0 / 8.0,
            m1_msun=35.6, m2_msun=30.6, distance_Mpc=440.0,
            approximant="IMRPhenomD")
        print(f"  Test waveform generated: {len(f_grid)} frequency bins, "
              f"|h|_max = {np.max(np.abs(h)):.3e}")
        return True
    except Exception as e:
        print(f"  Test waveform generation FAILED: {e}")
        return False


if __name__ == "__main__":
    check_lalsimulation_available()
