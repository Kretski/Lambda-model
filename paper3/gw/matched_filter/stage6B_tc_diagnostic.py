"""
stage6B_tc_diagnostic.py
=========================

STAGE 6B — tc (MERGER TIME) MISALIGNMENT DIAGNOSTIC

Purpose
-------
Test the hypothesis that the near-zero match values seen in Stage 6A
are caused by a merger-time (tc) misalignment between the analysis
window and the waveform template, NOT by anything Lambda-related.

Reasoning
---------
The template model (waveform.py) assumes the merger occurs at t=0 of
the analysis window (tc=0.0, hardcoded everywhere upstream). But the
way stage3's to_frequency_domain() slices the loaded segment means
the true merger time can fall several seconds INTO the window instead
of at its start. If that is the whole story, then:

    - With tc=0 fixed (current behavior): match should be near zero.
    - Scanning tc over a small range should show a SHARP, NARROW peak
      in match(GR, tc) right at the true merger offset, with the peak
      match rising close to 1.0 (since masses/distance are already
      the correct GW150914 catalog values).

A sharp, high peak confirms pure time misalignment (easy fix: shift
the data or fit tc). A weak/broad/absent peak means there is ALSO a
separate problem (PSD, FFT normalization convention, amplitude scale,
etc.) that this diagnostic will help isolate.

This script intentionally:
    - Fixes Lambda = 0 (pure GR baseline) for the whole scan.
    - Does NOT touch the Lambda phase model or grid_search_lambda.
    - Reuses the existing validated Stage-3 loading / PSD / frequency
      pipeline unchanged.

Example
-------
python stage6B_tc_diagnostic.py ^
    --h1 "C:\\Users\\Lenovo\\Desktop\\GravOptAdaptiveE-main\\H1_GW150914_4096s.hdf5" ^
    --fs 4096
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

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


DEFAULT_DURATION = 8.0
DEFAULT_F_MIN = 20.0
DEFAULT_F_MAX = 300.0

# tc scan range and resolution (seconds)
TC_MIN = 0.0
TC_MAX = 4.0
TC_STEP = 0.02


def compute_match(data_fd, f, psd, df, m1, m2, K_z, Lambda, tc,
                   distance_Mpc):
    """
    Normalized noise-weighted match between the real data and a
    Lambda/tc waveform.

        match(d,h) = <d|h> / sqrt(<d|d><h|h>)
    """

    h = waveform_frequency_domain(
        f, m1, m2, Lambda, K_z, tc=tc, distance_Mpc=distance_Mpc,
    )

    dd = noise_weighted_inner_product(data_fd, data_fd, f, psd, df)
    hh = noise_weighted_inner_product(h, h, f, psd, df)

    if dd <= 0.0 or hh <= 0.0:
        return np.nan

    dh = noise_weighted_inner_product(data_fd, h, f, psd, df)

    return dh / np.sqrt(dd * hh)


def main():

    parser = argparse.ArgumentParser(
        description="Stage 6B tc misalignment diagnostic."
    )

    parser.add_argument("--h1", required=True,
                         help="Path to H1 HDF5 strain file.")
    parser.add_argument("--event", default="GW150914",
                         choices=list(EVENT_CATALOG.keys()))
    parser.add_argument("--fs", type=float, default=None,
                         help="Sampling frequency if HDF5 metadata "
                              "does not contain it.")
    parser.add_argument("--duration", type=float,
                         default=DEFAULT_DURATION,
                         help="Analysis duration in seconds.")
    parser.add_argument("--f-min", type=float, default=DEFAULT_F_MIN)
    parser.add_argument("--f-max", type=float, default=DEFAULT_F_MAX)
    parser.add_argument("--tc-min", type=float, default=TC_MIN)
    parser.add_argument("--tc-max", type=float, default=TC_MAX)
    parser.add_argument("--tc-step", type=float, default=TC_STEP)

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
    print("# STAGE 6B — tc MISALIGNMENT DIAGNOSTIC")
    print("#" * 78)
    print()
    print(f"Input H1:              {h1_path}")
    print(f"Event:                 {args.event}")
    print(f"m1, m2:                {m1:.2f}, {m2:.2f} Msun")
    print(f"Distance:              {distance_Mpc:.1f} Mpc")
    print(f"Analysis band:         {args.f_min:.0f}-{args.f_max:.0f} Hz")
    print(f"Duration:              {args.duration:.3f} s")
    print(f"tc scan:               {args.tc_min:.3f} to {args.tc_max:.3f} s, "
          f"step {args.tc_step:.3f} s")
    print()
    print("Lambda is fixed to 0 (pure GR baseline) for this entire scan.")
    print()

    # Same fixed time-domain segment convention as Stage 6A.
    strain_td, fs, seg_start = load_real_segment(
        str(h1_path),
        gps_merger,
        half_window=args.duration / 2.0 + 2.0,
        fs_override=args.fs,
    )

    f, data_fd, df = to_frequency_domain(
        strain_td, fs, args.f_min, args.f_max, args.duration,
    )

    psd = estimate_psd_from_segment(strain_td, fs, f)

    tc_grid = np.arange(args.tc_min, args.tc_max + args.tc_step / 2,
                         args.tc_step)

    matches = np.full(len(tc_grid), np.nan)

    print(f"Scanning {len(tc_grid)} tc values...")
    print()

    for i, tc in enumerate(tc_grid):
        matches[i] = compute_match(
            data_fd, f, psd, df, m1, m2, K_z, 0.0, tc, distance_Mpc,
        )

    finite = np.isfinite(matches)

    if not np.any(finite):
        print("ERROR: all match values are NaN/undefined. "
              "Check dd/hh individually (likely a PSD or amplitude "
              "problem, unrelated to tc).")
        return

    i_best = int(np.nanargmax(matches))
    tc_best = tc_grid[i_best]
    match_best = matches[i_best]

    # Simple peak-sharpness diagnostic: width where match > half of
    # (match_best - baseline), baseline = median match over the scan.
    baseline = float(np.nanmedian(matches))
    half_level = baseline + 0.5 * (match_best - baseline)
    above = matches >= half_level
    if np.any(above):
        idx_above = np.where(above)[0]
        width_s = (tc_grid[idx_above[-1]] - tc_grid[idx_above[0]])
    else:
        width_s = float("nan")

    print("=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"Best-fit tc:             {tc_best:.4f} s")
    print(f"Match at best tc:        {match_best:.6f}")
    print(f"Match at tc=0 (current): {matches[0]:.6f}")
    print(f"Median match over scan:  {baseline:.6f}")
    print(f"Approx peak width:       {width_s:.4f} s "
          f"(narrower = more likely a pure time-offset artifact)")
    print()

    if match_best > 0.5:
        print("INTERPRETATION: Sharp, high match recovered at nonzero "
              "tc strongly supports the tc-misalignment hypothesis.")
        print(f"  -> Fix: use tc = {tc_best:.4f} s (or fit tc jointly) "
              "instead of the hardcoded tc=0.0 used throughout "
              "waveform.py / likelihood.py / stage6A.")
    elif match_best > 0.15:
        print("INTERPRETATION: Partial improvement at nonzero tc. "
              "tc misalignment is likely A factor, but not the whole "
              "story -- there may also be a PSD/normalization/amplitude "
              "issue to investigate separately.")
    else:
        print("INTERPRETATION: No strong peak found anywhere in this "
              "tc range. The problem is likely NOT (only) tc "
              "misalignment -- check <d|d> and <h|h> individually, "
              "PSD units, and FFT normalization convention next.")

    print()

    # Plot
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    png_path = out / "stage6B_tc_diagnostic.png"

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(tc_grid, matches, "-", lw=1.5)
    ax.axvline(tc_best, color="firebrick", ls="--",
               label=f"best tc = {tc_best:.3f} s")
    ax.axvline(0.0, color="gray", ls=":", label="tc = 0 (current code)")
    ax.set_xlabel("tc (s)")
    ax.set_ylabel("match(GR, tc)")
    ax.set_title(f"Stage 6B — tc scan, {args.f_min:.0f}-{args.f_max:.0f} Hz")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"Saved plot -> {png_path}")
    print()
    print("#" * 78)
    print("# STAGE 6B COMPLETE")
    print("#" * 78)
    print()


if __name__ == "__main__":
    main()
