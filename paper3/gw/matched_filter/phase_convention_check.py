#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convention_check.py (also: residual-first-framework scanner/gw/f4_coefficient.py)
— dispersion-phase convention check (no GW data involved).

Model:  ω² = c²k²(1 + Λk²)  ⇔  E² = p²c² + A₄p⁴c⁴,  Λ = A₄(ħc)² = (λ_A/2π)²,  Λ in m².

Two phase prescriptions exist for the α = 4 modified dispersion
(LVK, GWTC-4.0 Tests of GR II, arXiv:2603.19020, §3.1):
  (P) particle velocity  — LVK up to GWTC-3, Mirshekari–Yunes–Will, and the LIV
      option of LALSimulation:   δΨ = +(4π³/3) Λ I₄(z) f³ / c³
  (G) group velocity     — LVK from GWTC-4.0, consistent with WKB
      (Ezquiaga et al. 2022):     δΨ = −4π³ Λ I₄(z) f³ / c³
The two differ by (1 − α) = −3 in size AND sign.
I₄(z) = (c/H₀) ∫₀^z (1+z')² / E(z') dz'.

What this script checks: which prescription LALSimulation's built-in LIV term
implements. Expected: ratio c3_LAL / c3_pred ≈ +1 for (P) and ≈ −1/3 for (G).
Agreement with (P) is a SOFTWARE/CONVENTION verification only; it does not make
(P) the physical choice. Analyses of the Λ model use (G), and injections must
add the (G) phase explicitly instead of using the built-in LIV term.

Method:
  1. IMRPhenomXPHM in the frequency domain without and with the LIV term
     (α = 4, given λ_eff and sign of A).
  2. δΨ(f) = unwrap(arg(h_LIV / h_GR)) over the signal band.
  3. Fit δΨ = c0 + c1 f + c2 f² + c3 f³ (c0, c1 absorb phase/time conventions);
     a pure f³ term gives c2 ≈ 0.
  4. Compare c3 with (P) and (G); z from the luminosity distance with Planck15
     and Planck18.

Result obtained (lal 7.7.1, lalsimulation 6.2.1): pure f³ phase (c2 ~ 1e-20,
residual 7e-16 rad), sign follows sign(A), ratio to (P) = 0.9983, i.e. ratio to
(G) = −0.333; LALSimulation's λ_eff equals λ_A.

  python convention_check.py
"""
import sys

import numpy as np

try:
    import lal
    import lalsimulation as ls
except ImportError as e:
    sys.exit(f"lalsuite is required: {e}")

C = 299792458.0
M1, M2 = 36.0, 29.0
DIST_MPC = 400.0
INCL = 0.4
F_LOW, F_REF, F_MAX = 20.0, 20.0, 1024.0
DELTA_F = 1.0 / 16
ALPHA = 4.0
LOG10_LAMBDA_EFF = -3.5          # λ_eff [m]; chosen so that δΨ ~ 1 rad in band
BAND = (25.0, 400.0)


def liv_setters():
    names = {
        "enable": "SimInspiralWaveformParamsInsertEnableLIV",
        "loglambda": "SimInspiralWaveformParamsInsertNonGRLIVLogLambdaEff",
        "alpha": "SimInspiralWaveformParamsInsertNonGRLIVAlpha",
        "sign": "SimInspiralWaveformParamsInsertNonGRLIVASign",
    }
    missing = [v for v in names.values() if not hasattr(ls, v)]
    if missing:
        print("LIV setters not found under the expected names:", missing)
        print("Available LIV-related functions:")
        for n in sorted(dir(ls)):
            if "LIV" in n or "Liv" in n:
                print("  ", n)
        sys.exit(1)
    return {k: getattr(ls, v) for k, v in names.items()}


def waveform(liv=None):
    params = lal.CreateDict()
    if liv is not None:
        s = liv_setters()
        s["enable"](params, 1)
        s["loglambda"](params, LOG10_LAMBDA_EFF)
        s["alpha"](params, ALPHA)
        s["sign"](params, float(liv))
    hp, hc = ls.SimInspiralChooseFDWaveform(
        M1 * lal.MSUN_SI, M2 * lal.MSUN_SI, 0, 0, 0, 0, 0, 0,
        DIST_MPC * 1e6 * lal.PC_SI, INCL, 0.0, 0.0, 0.0, 0.0,
        DELTA_F, F_LOW, F_MAX, F_REF, params, ls.IMRPhenomXPHM)
    f = hp.f0 + DELTA_F * np.arange(hp.data.length)
    return f, hp.data.data.copy()


def measured_c3(sign):
    f, h0 = waveform(None)
    _, h1 = waveform(sign)
    m = (f >= BAND[0]) & (f <= BAND[1]) & (np.abs(h0) > 0)
    dpsi = np.unwrap(np.angle(h1[m] / h0[m]))
    fm = f[m]
    A = np.column_stack([np.ones_like(fm), fm, fm ** 2, fm ** 3])
    c, *_ = np.linalg.lstsq(A, dpsi, rcond=None)
    resid = dpsi - A @ c
    A3 = np.column_stack([np.ones_like(fm), fm, fm ** 3])
    c3only, *_ = np.linalg.lstsq(A3, dpsi, rcond=None)
    return c, float(np.sqrt(np.mean(resid ** 2))), float(np.ptp(dpsi)), c3only


def I4(z, cosmo):
    from scipy import integrate
    H0 = cosmo.H0.to("1/s").value
    E = lambda zz: cosmo.efunc(zz)
    val, _ = integrate.quad(lambda zz: (1 + zz) ** 2 / E(zz), 0, z)
    return C / H0 * val


def main():
    print(f"lal {lal.VCSInfo.version}, lalsimulation {ls.__version__ if hasattr(ls, '__version__') else '?'}")
    print(f"α = {ALPHA}, log10 λ_eff = {LOG10_LAMBDA_EFF} (λ_eff = {10 ** LOG10_LAMBDA_EFF:.3e} m), "
          f"D_L = {DIST_MPC} Mpc, band {BAND} Hz\n")

    results = {}
    for sign in (+1, -1):
        c, rms, span, c3only = measured_c3(sign)
        results[sign] = c[3]
        print(f"sign(A) = {sign:+d}: δΨ span {span:.3f} rad; fit c2 = {c[2]:+.3e}, c3 = {c[3]:+.6e} rad/Hz³, "
              f"rms residual {rms:.2e} rad; pure-f³ fit c3 = {c3only[2]:+.6e}")
    if abs(results[+1]) < 1e-20:
        print("\nδΨ is zero: the dispersion term is NOT applied for this approximant/call. Stop.")
        return

    lam = 10 ** LOG10_LAMBDA_EFF
    Lambda = (lam / (2 * np.pi)) ** 2
    print(f"\nIf λ_eff = λ_A:  Λ = (λ_A/2π)² = {Lambda:.4e} m²,  A₄ = Λ/(ħc)² = "
          f"{Lambda / (lal.HBAR_SI * C / lal.QE_SI) ** 2:.4e} eV⁻²")

    try:
        from astropy import units as u
        from astropy.cosmology import Planck15, Planck18, z_at_value
    except ImportError:
        print("astropy not available; cannot compute I₄(z)")
        return
    print("\nratio c3_LAL / c3_pred:  expected ≈ +1 for (P) particle velocity, ≈ −1/3 for (G) group velocity")
    for cname, cosmo in (("Planck15", Planck15), ("Planck18", Planck18)):
        z = float(z_at_value(cosmo.luminosity_distance, DIST_MPC * u.Mpc))
        i4 = I4(z, cosmo)
        predP = (4 * np.pi ** 3 / 3) * Lambda * i4 / C ** 3      # particle velocity
        predG = -4 * np.pi ** 3 * Lambda * i4 / C ** 3           # group velocity
        print(f"  {cname}: z = {z:.5f}, I₄ = {i4:.5e} m")
        for sign in (+1, -1):
            print(f"     sign {sign:+d}:  (P) {results[sign] / predP:+.4f}    (G) {results[sign] / predG:+.4f}")
    print("\nLALSimulation implements (P). The Λ-model analyses use (G); add that phase explicitly for injections.")


if __name__ == "__main__":
    main()
