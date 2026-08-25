"""
stage3_real_strain_validation.py
====================================

STAGE 3: validation against REAL H1 strain (not colored-Gaussian
approximations of its PSD), with the actual GW150914 event analysis
gated behind three mandatory checks.

CRITICAL DISTINCTION FROM STAGE 2:
  Stage 2 injected synthetic waveforms into colored GAUSSIAN noise
  shaped to match the real PSD -- this validates robustness to the
  real noise SPECTRUM but not to non-Gaussian features (glitches,
  spectral lines, non-stationarity) actually present in real detector
  data. Stage 3 injects into and analyzes the ACTUAL real strain time
  series.

FOUR MODES, run in this fixed order, with later modes only interpreted
if earlier ones pass:

  A — GR (Lambda=0) injection into real off-source H1 noise.
      Uses a segment of real strain WITHOUT the GW150914 event, adds a
      synthetic Lambda=0 waveform on top of the ACTUAL real strain
      (not a Gaussian approximation), recovers Lambda via matched
      filtering against the real segment's own PSD.
      Expectation: Lambda_fit ~ 0.

  B — Nonzero-Lambda injection into the SAME real off-source noise.
      Same real strain segment, injected Lambda in {0.01,...,1.0}.
      Expectation: Lambda_fit ~ Lambda_true.

  C — Time-slide / off-source null trials.
      Many independent real off-source H1 segments (no injection),
      matched-filtered against the GR+Lambda template bank.
      Expectation: Lambda_ML values scatter without a consistent
      significant nonzero preference (same check as Stage 2's Test C,
      now on real strain rather than Gaussian noise).

  D — THE ACTUAL GW150914 EVENT.
      Only reached if A, B, C all pass. Printed and plotted, but
      EXPLICITLY labeled as an EXPLORATORY RESULT regardless of the
      number obtained -- not a claim, not a constraint, not a
      detection, until independently reproduced (H1+L1 coherent
      analysis, full PN waveform, proper Bayesian priors) well beyond
      the scope of this proof-of-concept pipeline.

Usage:
    python stage3_real_strain_validation.py --h1 <path-to-H1-strain.hdf5>
"""

import argparse
import numpy as np
import h5py
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).parent))
from waveform import waveform_frequency_domain, cosmological_K_factor
from likelihood import grid_search_lambda, snr_optimal


EVENT_CATALOG = {
    "GW150914": dict(gps_merger=1126259462.4, m1=35.6, m2=30.6,
                      distance_Mpc=440.0, z=0.09),
}


# ══════════════════════════════════════════════════════════════════════════
# Real strain loading and frequency-domain conditioning
# ══════════════════════════════════════════════════════════════════════════

def load_real_segment(h1_path, gps_center, half_window):
    """Load a real strain segment [gps_center - half_window,
    gps_center + half_window] with metadata."""
    with h5py.File(h1_path, "r") as f:
        strain = f["strain"]["Strain"]
        attrs = strain.attrs
        fs = 1.0 / attrs["Xspacing"]
        gps_start = attrs["Xstart"]
        n_total = attrs["Npoints"]

        idx_center = int(round((gps_center - gps_start) * fs))
        idx_half = int(round(half_window * fs))
        idx_lo = max(0, idx_center - idx_half)
        idx_hi = min(n_total, idx_center + idx_half)

        data = np.array(strain[idx_lo:idx_hi])
        seg_gps_start = gps_start + idx_lo / fs

    return data, fs, seg_gps_start


def to_frequency_domain(strain_td, fs, f_min, f_max, duration_target):
    """
    Take a real time-domain strain segment, window it, and transform to
    the frequency domain on a grid consistent with duration_target
    (df = 1/duration_target), restricted to [f_min, f_max].

    If the segment is longer than duration_target, uses only the last
    duration_target seconds (closest to any embedded merger, for the
    real-event case) or the first duration_target seconds (for
    off-source noise segments, direction does not matter).
    """
    n_target = int(round(duration_target * fs))
    if len(strain_td) > n_target:
        strain_td = strain_td[-n_target:]
    elif len(strain_td) < n_target:
        raise ValueError(f"Segment too short: {len(strain_td)} < {n_target} samples")

    window = np.hanning(len(strain_td))
    strain_windowed = strain_td * window
    # Correct for window power loss (standard normalization)
    win_norm = np.sqrt(np.mean(window ** 2))

    strain_fd_full = np.fft.rfft(strain_windowed) / fs / win_norm
    freqs_full = np.fft.rfftfreq(len(strain_td), d=1 / fs)

    band = (freqs_full >= f_min) & (freqs_full < f_max)
    df = 1.0 / duration_target

    return freqs_full[band], strain_fd_full[band], df


def estimate_psd_from_segment(strain_td, fs, f_target):
    """Welch PSD estimate from a real time-domain segment, interpolated
    onto f_target with a floor."""
    freqs, psd = welch(strain_td, fs=fs, nperseg=min(len(strain_td), int(4 * fs)))
    psd_interp = np.interp(f_target, freqs, psd)
    floor = np.percentile(psd[psd > 0], 1)
    return np.maximum(psd_interp, floor)


# ══════════════════════════════════════════════════════════════════════════
# Mode A/B: injection into real off-source strain
# ══════════════════════════════════════════════════════════════════════════

def inject_into_real_strain(strain_fd, f, df, m1, m2, Lambda_true, K_z,
                             distance_Mpc, scale=1.0):
    """
    Add a synthetic waveform ON TOP OF the real strain's frequency-
    domain representation (already loaded). scale allows adjusting the
    injected amplitude if needed for SNR control; default scale=1.0
    uses the physical amplitude from the given distance.
    """
    h = waveform_frequency_domain(f, m1, m2, Lambda_true, K_z,
                                   distance_Mpc=distance_Mpc / scale)
    return strain_fd + h


# ══════════════════════════════════════════════════════════════════════════
def run_mode_AB(h1_path, gps_offsource_center, m1, m2, K_z, distance_Mpc,
                 f_min, f_max, duration, Lambda_grid, n_realizations=10):
    print("=" * 72)
    print("MODE A+B — INJECTION INTO REAL OFF-SOURCE H1 STRAIN")
    print("=" * 72)
    print()

    # Load a LARGE off-source block (not just duration+10s) so that
    # different "realizations" are genuinely independent, non-heavily-
    # overlapping noise draws -- a too-small block was the cause of the
    # previously observed scatter=0.0000 (each "realization" was nearly
    # the same overlapping window, not an independent noise draw).
    block_td, fs, block_start = load_real_segment(
        h1_path, gps_offsource_center,
        half_window=duration * n_realizations / 2 + 60)
    print(f"  Off-source block: GPS [{block_start:.1f}, "
          f"{block_start + len(block_td)/fs:.1f}], {len(block_td)/fs:.1f}s")
    print(f"  ({n_realizations} independent {duration}s realizations drawn from this block)")
    print()

    seg_samples = int(duration * fs)
    max_start = len(block_td) - seg_samples

    Lambda_true_values = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]

    # Fix the independent off-source windows ONCE (same across all
    # Lambda_true values, for a fair like-for-like comparison)
    rng = np.random.default_rng(0)
    if max_start > n_realizations:
        starts = rng.choice(max_start, size=n_realizations, replace=False)
    else:
        starts = np.linspace(0, max_start, n_realizations, dtype=int)

    # Report SNR using the first window's PSD, for context
    sub0 = block_td[starts[0]:starts[0] + seg_samples]
    f0, _, df0 = to_frequency_domain(sub0, fs, f_min, f_max, duration)
    psd0 = estimate_psd_from_segment(sub0, fs, f0)
    snr0 = snr_optimal(f0, psd0, df0, m1, m2, 0.0, K_z, distance_Mpc=distance_Mpc)
    print(f"  Optimal SNR at Lambda=0 (physical distance {distance_Mpc} Mpc, "
          f"first window): {snr0:.2f}")
    print()

    print(f"  {'Lambda_true':>12}  {'Lambda_ML (mean)':>18}  "
          f"{'scatter (std)':>14}  {'sigma_from_true':>16}")
    print("  " + "-" * 68)

    results = []
    for Lam_true in Lambda_true_values:
        ml_estimates = []
        for start in starts:
            sub_td = block_td[start:start + seg_samples]
            f_r, strain_fd_r, df_r = to_frequency_domain(
                sub_td, fs, f_min, f_max, duration)
            psd_r = estimate_psd_from_segment(sub_td, fs, f_r)

            data_fd = inject_into_real_strain(
                strain_fd_r, f_r, df_r, m1, m2, Lam_true, K_z, distance_Mpc)

            _, _, Lam_ml, _ = grid_search_lambda(
                data_fd, f_r, psd_r, df_r, m1, m2, K_z, Lambda_grid,
                distance_Mpc=distance_Mpc)
            ml_estimates.append(Lam_ml)

        ml_estimates = np.array(ml_estimates)
        mean_ml, scatter = np.mean(ml_estimates), np.std(ml_estimates)
        sigma = abs(mean_ml - Lam_true) / scatter if scatter > 0 else np.nan
        results.append(dict(Lambda_true=Lam_true, mean_ml=mean_ml,
                             scatter=scatter, sigma=sigma))
        print(f"  {Lam_true:>12.3f}  {mean_ml:>18.4f}  {scatter:>14.4f}  "
              f"{sigma:>16.2f}")

    print()
    ab_pass = all(
        abs(r["mean_ml"] - r["Lambda_true"]) < 2 * r["scatter"]
        if r["scatter"] > 0 else abs(r["mean_ml"] - r["Lambda_true"]) < 0.05
        for r in results
    )
    print(f"  MODE A+B: {'PASS' if ab_pass else 'FAIL'}")
    print()

    return results, ab_pass, f0, psd0, df0


# ══════════════════════════════════════════════════════════════════════════
def run_mode_C(h1_path, gps_avoid, m1, m2, K_z, distance_Mpc, f_min, f_max,
               duration, n_trials=200, grid_widths=(5.0, 10.0, 20.0),
               grid_points_per_unit=100):
    """
    Adaptive boundary widening: start with Lambda_grid=[-5,5], and if
    boundary hits exceed threshold, WIDEN THE GRID and re-run the SAME
    200 null trials (same off-source segments, same random seed) --
    critical for a fair comparison, since otherwise a change in result
    could be due to different noise realizations rather than the grid
    change itself.
    """
    print("=" * 72)
    print("MODE C — TIME-SLIDE / OFF-SOURCE NULL TRIALS (real strain, no injection)")
    print("=" * 72)
    print()

    max_width = max(grid_widths)
    block_td, fs, block_start = load_real_segment(
        h1_path, gps_avoid, half_window=duration * n_trials / 2 + 60)

    seg_samples = int(duration * fs)
    max_start = len(block_td) - seg_samples

    print(f"  Off-source block: {len(block_td)/fs:.1f}s, {n_trials} trials")
    print()

    # Fix the trial segments ONCE, before any grid widening, so all grid
    # widths are tested against the IDENTICAL set of noise realizations
    rng = np.random.default_rng(1)
    if max_start > n_trials:
        starts = rng.choice(max_start, size=n_trials, replace=False)
    else:
        starts = np.linspace(0, max_start, n_trials, dtype=int)

    # Pre-load and pre-condition each trial segment once (reused across
    # grid widths -- only the search grid changes)
    trial_data = []
    for start in starts:
        sub_td = block_td[start:start + seg_samples]
        f_r, strain_fd_r, df_r = to_frequency_domain(sub_td, fs, f_min, f_max, duration)
        psd_r = estimate_psd_from_segment(sub_td, fs, f_r)
        trial_data.append((f_r, strain_fd_r, df_r, psd_r))

    null_estimates = None
    final_grid = None
    c_pass = False

    for width in grid_widths:
        Lambda_grid = np.linspace(-width, width, int(2 * width * grid_points_per_unit) + 1)

        estimates = []
        for f_r, strain_fd_r, df_r, psd_r in trial_data:
            _, _, Lam_ml, _ = grid_search_lambda(
                strain_fd_r, f_r, psd_r, df_r, m1, m2, K_z, Lambda_grid,
                distance_Mpc=distance_Mpc)
            estimates.append(Lam_ml)

        null_estimates = np.array(estimates)
        final_grid = Lambda_grid

        mean_v, median_v, std_v = (np.mean(null_estimates), np.median(null_estimates),
                                     np.std(null_estimates))
        p05, p95 = np.percentile(null_estimates, [5, 95])

        grid_lo, grid_hi = Lambda_grid[0], Lambda_grid[-1]
        boundary_hits = np.sum((null_estimates <= grid_lo) | (null_estimates >= grid_hi))
        boundary_fraction = boundary_hits / n_trials
        boundary_pass = boundary_fraction < 0.05

        hist, edges = np.histogram(null_estimates, bins=20)
        max_cluster_frac = np.max(hist) / n_trials
        cluster_pass = max_cluster_frac < 0.20

        n_pass = n_trials >= 100

        print(f"  --- Grid width +/-{width} ({len(Lambda_grid)} points) ---")
        print(f"    Mean={mean_v:.4f}  Median={median_v:.4f}  Std={std_v:.4f}  "
              f"90%=[{p05:.4f},{p95:.4f}]")
        print(f"    Boundary hits: {boundary_hits}/{n_trials} "
              f"({boundary_fraction:.3f}) -> {'PASS' if boundary_pass else 'FAIL'}")
        print(f"    Max cluster fraction: {max_cluster_frac:.3f} "
              f"-> {'PASS' if cluster_pass else 'FAIL'}")
        print()

        if boundary_pass and cluster_pass and n_pass:
            c_pass = True
            print(f"  Converged at grid width +/-{width}: all diagnostics PASS.")
            print()
            break
        elif not boundary_pass and width < max_width:
            print(f"  Boundary hits still too high -- widening grid...")
            print()
        else:
            print(f"  Diagnostics did not clear even at widest tested grid "
                  f"(+/-{width}).")
            if not boundary_pass:
                print("  Persistent boundary hits at wide grids suggest the null")
                print("  likelihood may be poorly conditioned under pure noise,")
                print("  rather than simply needing a wider search range.")
            print()

    print(f"  MODE C OVERALL: {'PASS' if c_pass else 'FAIL'} "
          f"(final grid tested: +/-{final_grid[-1]:.1f})")
    print()

    return null_estimates, c_pass, final_grid


# ══════════════════════════════════════════════════════════════════════════
def run_mode_D(h1_path, event_name, f_min, f_max, duration, Lambda_grid):
    print("=" * 72)
    print(f"MODE D — ACTUAL {event_name} EVENT (EXPLORATORY ONLY)")
    print("=" * 72)
    print()
    print("  ┌────────────────────────────────────────────────────────┐")
    print("  │  WHATEVER NUMBER FOLLOWS IS EXPLORATORY, NOT A RESULT.  │")
    print("  │  Not a detection. Not a constraint. Not for citation.   │")
    print("  │  Single detector, leading-order waveform, grid-search   │")
    print("  │  point estimate only -- no priors, no PN systematics,   │")
    print("  │  no H1+L1 coherence, no glitch vetoing.                 │")
    print("  └────────────────────────────────────────────────────────┘")
    print()

    ev = EVENT_CATALOG[event_name]
    K_z = cosmological_K_factor(ev["z"])

    strain_td, fs, seg_start = load_real_segment(
        h1_path, ev["gps_merger"], half_window=duration / 2 + 2)
    f, strain_fd, df = to_frequency_domain(strain_td, fs, f_min, f_max, duration)
    psd = estimate_psd_from_segment(strain_td, fs, f)

    Lgrid, logL, Lam_ml, Lam_err = grid_search_lambda(
        strain_fd, f, psd, df, ev["m1"], ev["m2"], K_z, Lambda_grid,
        distance_Mpc=ev["distance_Mpc"])

    print(f"  Grid-search maximum-likelihood Lambda: {Lam_ml:.4f}")
    print(f"  Fisher-curvature error estimate: {Lam_err:.4f}" if np.isfinite(Lam_err)
          else "  Fisher-curvature error estimate: undefined (edge of grid or flat likelihood)")
    print()
    print("  This number is recorded for the repository's audit trail. It is")
    print("  NOT interpreted as evidence for or against nonzero Lambda.")

    return Lam_ml, Lam_err, Lgrid, logL


# ══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h1", required=True)
    parser.add_argument("--event", default="GW150914")
    parser.add_argument("--skip-D", action="store_true",
                         help="Skip Mode D even if A/B/C pass (for audit-only runs)")
    args = parser.parse_args()

    ev = EVENT_CATALOG[args.event]
    m1, m2, distance_Mpc, z = ev["m1"], ev["m2"], ev["distance_Mpc"], ev["z"]
    K_z = cosmological_K_factor(z)
    gps_merger = ev["gps_merger"]

    f_min, f_max, duration = 20.0, 400.0, 8.0
    Lambda_grid_AB = np.linspace(-5.0, 5.0, 1001)  # fixed grid for A+B (Lambda known/small)

    # Off-source anchor: well before the event, avoiding it entirely
    gps_offsource = gps_merger - 500.0

    print("#" * 72)
    print(f"# STAGE 3 — REAL STRAIN VALIDATION — {args.event}")
    print("#" * 72)
    print()

    results_AB, ab_pass, f_bg, psd_bg, df_bg = run_mode_AB(
        args.h1, gps_offsource, m1, m2, K_z, distance_Mpc,
        f_min, f_max, duration, Lambda_grid_AB)

    null_estimates, c_pass, Lambda_grid_C = run_mode_C(
        args.h1, gps_offsource, m1, m2, K_z, distance_Mpc,
        f_min, f_max, duration, n_trials=200,
        grid_widths=(5.0, 10.0, 20.0))

    print("=" * 72)
    print("GATE CHECK BEFORE MODE D")
    print("=" * 72)
    print()
    print(f"  Mode A+B (injection/recovery): {'PASS' if ab_pass else 'FAIL'}")
    print(f"  Mode C (null trials):          {'PASS' if c_pass else 'FAIL'}")
    print()

    Lam_ml_D, Lam_err_D = None, None
    if ab_pass and c_pass and not args.skip_D:
        print("  Both gates passed. Proceeding to Mode D (exploratory).")
        print()
        Lam_ml_D, Lam_err_D, Lgrid_D, logL_D = run_mode_D(
            args.h1, args.event, f_min, f_max, duration, Lambda_grid_C)
    elif args.skip_D:
        print("  --skip-D specified: Mode D not run.")
    else:
        print("  ┌────────────────────────────────────────────────────────┐")
        print("  │  GATE FAILED. Mode D (GW150914) is NOT run.             │")
        print("  │  Fix Mode A/B/C issues before analyzing the real event. │")
        print("  └────────────────────────────────────────────────────────┘")

    # ── Plots ──────────────────────────────────────────────────────────────
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)

    n_panels = 3 if Lam_ml_D is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 2:
        axes = list(axes) + [None]

    ax = axes[0]
    Lt = [r["Lambda_true"] for r in results_AB]
    Lml = [r["mean_ml"] for r in results_AB]
    Lsc = [r["scatter"] for r in results_AB]
    ax.errorbar(Lt, Lml, yerr=Lsc, fmt="o-", color="steelblue", capsize=4,
                markersize=8)
    lims = [min(Lt) - 0.2, max(Lt) + 0.2]
    ax.plot(lims, lims, "k--", lw=1.5, label="ideal")
    ax.set_xlabel(r"Injected $\Lambda_{\rm true}$")
    ax.set_ylabel(r"Recovered $\Lambda_{\rm ML}$")
    ax.set_title("Mode A+B: real off-source strain + injection")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(null_estimates, bins=10, color="gray", edgecolor="k", alpha=0.7)
    ax.axvline(0, color="firebrick", lw=1.5, ls="--")
    ax.set_xlabel(r"Recovered $\Lambda_{\rm ML}$")
    ax.set_ylabel("Count")
    ax.set_title("Mode C: real off-source null trials")
    ax.grid(True, alpha=0.3)

    if Lam_ml_D is not None:
        ax = axes[2]
        ax.plot(Lgrid_D, logL_D - np.max(logL_D), color="firebrick", lw=2)
        ax.axvline(Lam_ml_D, color="k", ls="--", lw=1.5,
                  label=f"ML: {Lam_ml_D:.3f}")
        ax.set_xlabel(r"$\Lambda$")
        ax.set_ylabel(r"$\ln L - \ln L_{\rm max}$")
        ax.set_title(f"Mode D: {args.event} (EXPLORATORY)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Stage 3: Real strain validation — {args.event}", fontsize=11)
    plt.tight_layout()
    path = out / f"stage3_{args.event}_validation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
