#!/usr/bin/env python3
"""
fisher_alpha.py -- why alpha = 0 measures and alpha >= 2.5 does not
===================================================================

The injection campaign established, for GW150914 with the (36, 36)
template over 20-512 Hz:

    alpha    local maxima   dRho^2 (largest inj)   slope
    0.0          2-3              +24.88           0.94
    2.5           0                +1.68           1.00
    3.0           0                +2.28           1.02
    3.5           0                +1.67           0.98
    4.0           0                -0.06           0.98

Unit slope everywhere means the injected phase reaches the likelihood
at full strength, so this is not a transfer bug. What differs is
whether the likelihood can DISCRIMINATE the value once t_c and phi_c
have been maximised over.

This script computes that directly, with no matched filtering. The MDR
phase derivative is f^(alpha-1); the nuisance directions are
d(Psi)/d(phi_c) ~ 1 and d(Psi)/d(t_c) ~ f. Under the noise- and
amplitude-weighted inner product

    (a, b) = sum_f  [ 4 |h(f)|^2 / S_n(f) ] a(f) b(f) df

the fraction of Fisher information on B that survives projecting out
{1, f} is

    R(alpha) = (v_perp, v_perp) / (v, v),     v(f) = f^(alpha-1)

R near 1 means B is nearly orthogonal to the nuisance parameters and is
measurable; R near 0 means t_c and phi_c absorb the deformation and no
amount of SNR or grid refinement will recover it.

The prediction to check: R should be large at alpha = 0 and collapse
for alpha >= 2.5, matching the injection results above. If it does,
the pipeline's alpha-dependence is explained analytically rather than
merely observed.

Requires: numpy, scipy, pycbc, h5py   (uses the real PSD and template)
Usage:    python3 fisher_alpha.py
"""

import json

import numpy as np

import mdr_search as M

# ======================================================================
EVENT_GPS = 1126259462.428
H1 = "H-H1_GWOSC_4KHZ_R1-1126257415-4096.hdf5"
TEMPLATE = (36.0, 36.0, 0.0, 0.0)

ALPHAS = [0.0, 0.5, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# Injection results to compare against, from injection_alpha*_wide.json
OBSERVED = {0.0: 24.88, 2.5: 1.68, 3.0: 2.28, 3.5: 1.67, 4.0: -0.06}

OUT_JSON = "fisher_alpha.json"


def main():
    print("=" * 72)
    print("FISHER INFORMATION ON B AFTER PROJECTING OUT t_c AND phi_c")
    print("=" * 72)

    ts = M.condition(M.load_gwosc(H1))
    block = ts.time_slice(EVENT_GPS - M.BLOCK / 2, EVENT_GPS + M.BLOCK / 2)
    psd = M.block_psd(block)
    hp = M.fd_template(TEMPLATE, 1.0 / M.BLOCK, len(psd))

    f = psd.sample_frequencies.numpy()
    sn = psd.numpy()
    amp = np.abs(hp.numpy())

    band = (f >= M.F_LOW) & (f <= M.F_HIGH) & (sn > 0)
    f = f[band]
    w = 4.0 * amp[band] ** 2 / sn[band] * (1.0 / M.BLOCK)
    snr2 = w.sum()

    print(f"  band        : {M.F_LOW:.0f}-{M.F_HIGH:.0f} Hz")
    print(f"  template    : {TEMPLATE[:2]}")
    print(f"  optimal SNR : {np.sqrt(snr2):.2f}")
    print()

    def ip(a, b):
        return float(np.sum(w * a * b))

    # orthonormalise the nuisance directions {1, f} under this metric
    basis = []
    for raw in (np.ones_like(f), f):
        v = raw.copy()
        for b in basis:
            v -= ip(v, b) * b
        v /= np.sqrt(ip(v, v))
        basis.append(v)

    print(f"  {'alpha':>6} {'retained':>10} {'sigma ratio':>12} "
          f"{'observed dRho2':>16}")
    rows = []
    for a in ALPHAS:
        v = f ** (a - 1.0)
        vp = v.copy()
        for b in basis:
            vp -= ip(vp, b) * b
        retained = ip(vp, vp) / ip(v, v)
        ratio = 1.0 / np.sqrt(retained) if retained > 0 else np.inf
        obs = OBSERVED.get(a)
        obs_s = f"{obs:+.2f}" if obs is not None else "-"
        print(f"  {a:>6.1f} {100*retained:>9.2f}% {ratio:>12.1f} "
              f"{obs_s:>16}")
        rows.append(dict(alpha=a, retained=float(retained),
                         sigma_ratio=float(ratio), observed_drho2=obs))

    print()
    print("=" * 72)
    print("READING")
    print("=" * 72)

    r0 = next(r["retained"] for r in rows if r["alpha"] == 0.0)
    r4 = next(r["retained"] for r in rows if r["alpha"] == 4.0)
    print(f"  alpha = 0 keeps {100*r0:.1f}% of the information on B;")
    print(f"  alpha = 4 keeps {100*r4:.2f}%.")
    print(f"  Ratio: {r0/max(r4,1e-30):.0f}x more information at alpha = 0.")
    print()

    checked = [r for r in rows if r["observed_drho2"] is not None]
    if len(checked) >= 3:
        x = np.array([r["retained"] for r in checked])
        y = np.array([max(r["observed_drho2"], 0.0) for r in checked])
        if x.std() > 0 and y.std() > 0:
            c = float(np.corrcoef(x, y)[0, 1])
            print(f"  Correlation between retained information and the")
            print(f"  measured injection discrimination: {c:+.3f}")
            if c > 0.8:
                print()
                print("  => The alpha-dependence of the pipeline is explained.")
                print("     Where t_c and phi_c absorb the MDR phase, no SNR")
                print("     and no grid refinement will recover B. This is a")
                print("     property of the signal and band, not a defect of")
                print("     the implementation.")

    print()
    print("  Note: retained information depends on the band. A lower")
    print("  F_LOW lengthens the inspiral and helps low alpha; raising")
    print("  F_HIGH helps high alpha only if the signal still has cycles")
    print("  there, which a 31 solar-mass chirp mass largely does not.")

    with open(OUT_JSON, "w") as fh:
        json.dump(dict(event_gps=EVENT_GPS, template=list(TEMPLATE),
                       f_low=M.F_LOW, f_high=M.F_HIGH,
                       optimal_snr=float(np.sqrt(snr2)), rows=rows),
                  fh, indent=2)
    print(f"\n  written: {OUT_JSON}")


if __name__ == "__main__":
    main()
