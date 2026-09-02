"""
stage6C_component_diagnostic.py
=================================

STAGE 6C — INNER PRODUCT COMPONENT DIAGNOSTIC

Purpose
-------
Stage 6B ruled out pure tc misalignment as the (sole) cause of the
near-zero match. This script breaks the match down into its raw
components:

    dd = <d|d>   (data self-overlap)
    hh = <h|h>   (template self-overlap = optimal SNR^2)
    dh = <d|h>   (data-template overlap)

and compares them against known physical scales for GW150914:

    - Network matched-filter SNR for GW150914 is ~24 (H1 alone is
      roughly ~20). So sqrt(hh) -- the template's OWN optimal SNR in
      this noise -- should land somewhere in that ballpark IF the
      template amplitude and PSD are both correctly scaled.
    - Real LIGO strain amplitude near merger is of order 1e-21.
    - Typical aLIGO noise PSD in the 100-300 Hz band is of order
      1e-46 to 1e-44 strain^2/Hz (a few times higher than the design
      minimum of ~1e-47 near 150 Hz for O1-era data).

Wildly different orders of magnitude here point directly at which
piece (template amplitude, PSD, or FFT normalization) is broken.

This script fixes Lambda=0 and uses the tc value found to be the true
merger offset (default 2.0s, matching the corrected GPS-based
localization -- override with --tc if needed).
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

from likelihood import noise_weighted_inner_product

from waveform import waveform_frequency_domain, cosmological_K_factor


def main():

    parser = argparse.ArgumentParser(
        description="Stage 6C inner-product component diagnostic."
    )

    parser.add_argument("--h1", required=True)
    parser.add_argument("--event", default="GW150914",
                         choices=list(EVENT_CATALOG.keys()))
    parser.add_argument("--fs", type=float, default=None)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--f-min", type=float, default=20.0)
    parser.add_argument("--f-max", type=float, default=300.0)
    parser.add_argument("--tc", type=float, default=2.0,
                         help="Merger offset (s) into the analysis "
                              "window to use for the template.")

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
    print("# STAGE 6C — INNER PRODUCT COMPONENT DIAGNOSTIC")
    print("#" * 78)
    print()
    print(f"Band:      {args.f_min:.0f}-{args.f_max:.0f} Hz")
    print(f"tc used:   {args.tc:.4f} s")
    print(f"Lambda:    0.0 (pure GR)")
    print()

    strain_td, fs, seg_start = load_real_segment(
        str(h1_path), gps_merger,
        half_window=args.duration / 2.0 + 2.0,
        fs_override=args.fs,
    )

    f, data_fd, df = to_frequency_domain(
        strain_td, fs, args.f_min, args.f_max, args.duration,
    )

    psd = estimate_psd_from_segment(strain_td, fs, f)

    h = waveform_frequency_domain(
        f, m1, m2, 0.0, K_z, tc=args.tc, distance_Mpc=distance_Mpc,
    )

    dd = noise_weighted_inner_product(data_fd, data_fd, f, psd, df)
    hh = noise_weighted_inner_product(h, h, f, psd, df)
    dh = noise_weighted_inner_product(data_fd, h, f, psd, df)

    print("=" * 78)
    print("RAW INNER PRODUCTS")
    print("=" * 78)
    print(f"  <d|d> = {dd:.6e}   (sqrt = {np.sqrt(max(dd,0)):.6e})")
    print(f"  <h|h> = {hh:.6e}   (sqrt = {np.sqrt(max(hh,0)):.6e})")
    print(f"  <d|h> = {dh:.6e}")
    print()

    if dd > 0 and hh > 0:
        match = dh / np.sqrt(dd * hh)
        print(f"  match = <d|h> / sqrt(<d|d><h|h>) = {match:.6f}")
    print()

    print("=" * 78)
    print("PHYSICAL SANITY CHECKS")
    print("=" * 78)
    print()
    print(f"  Template optimal SNR  sqrt(<h|h>) = {np.sqrt(max(hh,0)):.4f}")
    print("    Expected ballpark for GW150914 in H1 alone: ~15-20")
    print("    (network SNR ~24 across H1+L1)")
    print()
    print(f"  Data 'optimal SNR'    sqrt(<d|d>) = {np.sqrt(max(dd,0)):.4f}")
    print("    (data includes noise + signal; not directly comparable")
    print("    to template SNR, but should NOT differ by many orders")
    print("    of magnitude from sqrt(<h|h>) if scales are consistent)")
    print()

    ratio = np.sqrt(max(hh, 0)) / np.sqrt(max(dd, 1e-300))
    print(f"  Ratio sqrt(hh)/sqrt(dd) = {ratio:.6e}")
    print("    If this ratio is huge (>>100) or tiny (<<0.01), the")
    print("    template amplitude and the data/PSD are on incompatible")
    print("    scales -- check waveform.py amplitude normalization")
    print("    and/or PSD units in estimate_psd_from_segment.")
    print()

    # Amplitude sanity check: |h(f)| at a representative frequency
    idx_100 = int(np.argmin(np.abs(f - 100.0)))
    print(f"  |h(f={f[idx_100]:.1f} Hz)| = {np.abs(h[idx_100]):.6e}")
    print("    (this is a frequency-domain amplitude, not directly")
    print("    strain amplitude, but should not be absurdly larger/")
    print("    smaller than data_fd at the same frequency)")
    print(f"  |data_fd(f={f[idx_100]:.1f} Hz)| = "
          f"{np.abs(data_fd[idx_100]):.6e}")
    print()

    # PSD sanity check
    idx_150 = int(np.argmin(np.abs(f - 150.0)))
    print(f"  PSD at f={f[idx_150]:.1f} Hz: {psd[idx_150]:.6e}")
    print("    Typical aLIGO PSD in this band: ~1e-46 to 1e-44 "
          "strain^2/Hz")
    print()

    print("#" * 78)
    print("# STAGE 6C COMPLETE")
    print("#" * 78)
    print()


if __name__ == "__main__":
    main()
