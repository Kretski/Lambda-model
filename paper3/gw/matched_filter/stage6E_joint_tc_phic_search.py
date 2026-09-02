"""
stage6E_joint_tc_phic_search.py
=================================

STAGE 6E — JOINT tc + phi_c MATCHED-FILTER SEARCH (IFFT TRICK)

Purpose
-------
Stages 6A-6D all fixed phi_c=0 and only scanned tc. That means any
genuine signal alignment could have been destroyed by residual phase
mismatch even at the correct tc -- exactly what the Stage 6C
component diagnostic suggested (dd, hh were both physically
reasonable in scale, but dh collapsed to near zero, the signature of
destructive phase interference).

This script fixes that by using the standard GW matched-filter
"complex SNR time series" trick: for a fixed Lambda, the template's
dependence on tc is a pure linear phase term

    h(f; tc) = h0(f) * exp(2*pi*i*f*tc)

and its dependence on phi_c is a pure constant phase rotation

    h(f; tc, phi_c) = h(f; tc) * exp(-i*phi_c)

Because of this, <h|h> does NOT depend on tc or phi_c (only the masses
and distance do), and the tc-and-phi_c-maximized overlap has a closed
form:

    Z(tc)        = sum_f  conj(d(f)) * h0(f) * exp(2*pi*i*f*tc) / Sn(f) * df
    dh_max(tc)   = 4 * |Z(tc)|                  (optimized over phi_c)
    phi_c_best   = arg(Z(tc))
    match(tc)    = dh_max(tc) / sqrt(<d|d> * <h|h>)

Z(tc) evaluated on the sample grid tc_n = n/fs is exactly the inverse
FFT of the (zero-padded, one-sided) frequency-domain integrand -- so
the ENTIRE tc axis (at full 1/fs resolution, far finer than our
earlier 0.01-0.02s manual scans) is obtained from a single ifft call.

CRITICAL: This script reuses stage3_real_strain_validation's
to_frequency_domain() and estimate_psd_from_segment() UNCHANGED. It
does not reimplement its own FFT/PSD convention (that is what caused
the spurious ~1e4-1e7x normalization discrepancies in the
independently-written Stage 6D / 6D.5 wrapper scripts).

Lambda is fixed to 0 throughout -- this is an alignment diagnostic,
not a Lambda estimate.
"""

import argparse
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from stage3_real_strain_validation import (
    EVENT_CATALOG,
    load_real_segment,
    to_frequency_domain,
    estimate_psd_from_segment,
)

from waveform import waveform_frequency_domain, cosmological_K_factor


def main():

    parser = argparse.ArgumentParser(
        description="Stage 6E joint tc+phi_c matched-filter search "
                     "(IFFT trick), built on the validated Stage-3 "
                     "pipeline."
    )

    parser.add_argument("--h1", required=True)
    parser.add_argument("--event", default="GW150914",
                         choices=list(EVENT_CATALOG.keys()))
    parser.add_argument("--fs", type=float, default=None,
                         help="Sampling frequency override if HDF5 "
                              "metadata is missing it.")
    parser.add_argument("--duration", type=float, default=8.0,
                         help="Analysis window duration in seconds "
                              "(must match Stage 6A-6C convention).")
    parser.add_argument("--f-min", type=float, default=20.0)
    parser.add_argument("--f-max", type=float, default=300.0)
    parser.add_argument("--tc-search-max", type=float, default=None,
                         help="Only report the best tc within "
                              "[0, tc-search-max]. Default: full "
                              "duration window.")

    args = parser.parse_args()

    h1_path = Path(args.h1)
    if not h1_path.exists():
        raise FileNotFoundError(f"H1 file does not exist:\n{h1_path}")

    event = EVENT_CATALOG[args.event]
    m1 = event["m1"]
    m2 = event["m2"]
    distance_Mpc = event["distance_Mpc"]
    z = event["z"]
    K_z = cosmological_K_factor(z)
    gps_merger = event["gps_merger"]

    print()
    print("#" * 78)
    print("# STAGE 6E — JOINT tc + phi_c MATCHED-FILTER SEARCH")
    print("#" * 78)
    print()
    print(f"Event:      {args.event}  (m1={m1:.2f}, m2={m2:.2f} Msun, "
          f"D={distance_Mpc:.1f} Mpc)")
    print(f"Band:       {args.f_min:.0f}-{args.f_max:.0f} Hz")
    print(f"Duration:   {args.duration:.3f} s")
    print(f"Lambda:     0.0 (fixed, pure GR baseline)")
    print()
    print("Reusing stage3_real_strain_validation.to_frequency_domain() "
          "and estimate_psd_from_segment() unchanged -- no independent "
          "FFT/PSD reimplementation.")
    print()

    # ------------------------------------------------------------
    # Load (fixed GPS merger center, same convention as 6A-6D)
    # ------------------------------------------------------------

    strain_td, fs, seg_start = load_real_segment(
        str(h1_path), gps_merger,
        half_window=args.duration / 2.0 + 2.0,
        fs_override=args.fs,
    )

    # ------------------------------------------------------------
    # Validated Stage-3 FFT + PSD
    # ------------------------------------------------------------

    f, data_fd, df = to_frequency_domain(
        strain_td, fs, args.f_min, args.f_max, args.duration,
    )

    psd = estimate_psd_from_segment(strain_td, fs, f)

    # ------------------------------------------------------------
    # dd (band-limited, using the SAME convention as Stage 6A-6C)
    # ------------------------------------------------------------

    dd = 4.0 * np.real(np.sum(np.conj(data_fd) * data_fd / psd)) * df

    print(f"<d|d>          = {dd:.6e}")
    print(f"sqrt(<d|d>)    = {np.sqrt(max(dd, 0)):.6f}")
    print()

    # ------------------------------------------------------------
    # Baseline template h0(f) = h(f; tc=0, phi_c=0)
    # ------------------------------------------------------------

    h0 = waveform_frequency_domain(
        f, m1, m2, 0.0, K_z, tc=0.0, phi_c=0.0,
        distance_Mpc=distance_Mpc,
    )

    hh = 4.0 * np.real(np.sum(np.conj(h0) * h0 / psd)) * df

    print(f"<h|h>          = {hh:.6e}   (tc/phi_c-independent)")
    print(f"sqrt(<h|h>)    = {np.sqrt(max(hh, 0)):.6f}")
    print()

    if dd <= 0 or hh <= 0:
        print("ERROR: dd or hh is non-positive; cannot proceed.")
        return

    # ------------------------------------------------------------
    # Build the full-length, zero-padded, one-sided integrand and
    # take the IFFT to get Z(tc) for the ENTIRE tc axis at once.
    # ------------------------------------------------------------

    N = int(round(args.duration * fs))
    Nf = N // 2 + 1  # length of a full rfft array for this N

    freqs_full = np.fft.rfftfreq(N, d=1.0 / fs)
    band_mask = (freqs_full >= args.f_min) & (freqs_full < args.f_max)

    if np.sum(band_mask) != len(f):
        print("WARNING: band bin count mismatch between local "
              "recomputation and to_frequency_domain() output "
              f"({np.sum(band_mask)} vs {len(f)}). Proceeding, but "
              "double-check f_min/f_max/duration consistency.")

    A_full = np.zeros(Nf, dtype=complex)
    A_full[band_mask] = df * np.conj(data_fd) * h0 / psd

    B = np.zeros(N, dtype=complex)
    B[:Nf] = A_full
    # B[Nf:] left as zero: one-sided (positive-frequency-only)
    # convention, standard for the complex matched-filter SNR
    # time series.

    Z = N * np.fft.ifft(B)  # complex array, length N

    tc_axis = np.arange(N) / fs  # tc_n = n / fs, n = 0..N-1

    dh_max = 4.0 * np.abs(Z)
    match = dh_max / np.sqrt(dd * hh)
    phi_c_best = np.angle(Z)

    # ------------------------------------------------------------
    # Restrict search range if requested
    # ------------------------------------------------------------

    if args.tc_search_max is not None:
        search_mask = tc_axis <= args.tc_search_max
    else:
        search_mask = np.ones(N, dtype=bool)

    idx_local = np.argmax(match[search_mask])
    idx = np.where(search_mask)[0][idx_local]

    best_tc = tc_axis[idx]
    best_match = match[idx]
    best_phi_c = phi_c_best[idx]
    best_dh = dh_max[idx]

    print("=" * 78)
    print("RESULT (joint tc + phi_c optimum)")
    print("=" * 78)
    print(f"Best tc            = {best_tc:.6f} s")
    print(f"Best phi_c         = {best_phi_c:.6f} rad")
    print(f"dh_max(tc)         = {best_dh:.6e}")
    print(f"Maximum match      = {best_match:.6f}")
    print()

    # Local peak table around the optimum
    lo = max(0, idx - 5)
    hi = min(N, idx + 6)
    print("Local peak (tc, match, phi_c):")
    for j in range(lo, hi):
        marker = "  <-- best" if j == idx else ""
        print(f"  tc={tc_axis[j]:10.6f}  match={match[j]: .8f}  "
              f"phi_c={phi_c_best[j]: .6f}{marker}")
    print()

    # Also report match at phi_c=0 (i.e. Re[Z], not |Z|) at the same
    # tc, for direct comparison against Stage 6A-6C, which always
    # fixed phi_c=0.
    dh_phi0 = 4.0 * np.real(Z[idx])
    match_phi0 = dh_phi0 / np.sqrt(dd * hh)
    print(f"For comparison, match at this tc with phi_c FIXED to 0 "
          f"(Stage 6A-6C convention): {match_phi0:.6f}")
    print()

    if best_match >= 0.5:
        print("INTERPRETATION: Strong alignment recovered once phi_c "
              "is allowed to float. This strongly supports the "
              "hypothesis that Stage 6A-6C's near-zero match was "
              "caused by phase (phi_c) misalignment, not a deeper "
              "amplitude/PSD/normalization problem.")
    elif best_match >= 0.15:
        print("INTERPRETATION: Partial improvement. phi_c misalignment "
              "was likely A contributing factor, but not sufficient "
              "alone to explain the original near-zero match -- "
              "further investigation needed (e.g. higher-order PN "
              "phase terms missing from the leading-order template, "
              "or residual PSD/calibration issues).")
    else:
        print("INTERPRETATION: Still no strong alignment even with "
              "phi_c optimized. The near-zero match is NOT primarily "
              "a tc/phi_c alignment issue -- likely candidates: "
              "missing PN phase terms in the leading-order waveform "
              "model, PSD estimation quality, or calibration/units "
              "issues in the strain data itself.")

    print()
    print("#" * 78)
    print("# STAGE 6E COMPLETE")
    print("#" * 78)
    print()


if __name__ == "__main__":
    main()
