"""
stage5h9_fixed_grid_control.py
==================================

STAGE 5H-9: does the f_max dependence found in Stage 5H-8 (saturating
power law, n=2.70 at 100-250Hz vs n=1.62 at 250-400Hz) survive when
using a SINGLE FIXED Lambda search grid across all f_max values,
instead of 5H-8's per-point ADAPTIVE half-width (23.62 at 100Hz down
to 2.00 at 350-400Hz)?

WHY THIS CONTROL IS NECESSARY: 5H-8's adaptive half-width was a search-
window heuristic, not a physics assumption -- but it means the grid
RESOLUTION (points per unit Lambda) also varied substantially across
f_max values (same resolution_points=41 spread over very different
window widths). Part of the observed saturation could in principle be
an artifact of this varying resolution/window combination, not a
genuine property of the underlying likelihood surface.

METHOD: repeat the SAME 7-point f_max scan (100...400 Hz), but with a
SINGLE FIXED grid [-20, +20] (wide enough to contain the largest
100Hz-band peak separation seen in 5H-8, ~12.6) at FIXED resolution,
for every f_max value. If the fitted n and the saturation pattern
(lower-half vs upper-half exponent disagreement) persist under this
fixed-grid control, the f_max-dependence finding is robust to the
adaptive-window methodology. If the pattern changes substantially,
5H-8's specific numbers were partly a windowing artifact and the
underlying dependence needs re-characterization.
"""

import numpy as np
from pathlib import Path
import sys
import gc
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from waveform import cosmological_K_factor
from likelihood import aligo_like_psd
from stage5_cross_waveform_validation import (
    WAVEFORM_FAMILIES, grid_search_lambda_generic,
)


def measure_peak_spacing_fixed_grid(m1, m2, K_z, distance_Mpc, f_lo, f_hi,
                                      inject_fn, recover_fn, grid,
                                      n_realizations, seed_base):
    duration = 8.0
    df = 1.0 / duration
    f = np.arange(f_lo, f_hi, df)
    psd = aligo_like_psd(f)

    rng = np.random.default_rng(seed_base)
    spacings = []

    for real_idx in range(n_realizations):
        sigma = np.sqrt(psd / (4 * df))
        noise = rng.normal(0, sigma) + 1j * rng.normal(0, sigma)
        h_inject = inject_fn(f, m1, m2, 0.0, K_z, distance_Mpc=distance_Mpc)
        data = h_inject + noise

        Lgrid, logL, Lam_ml, _ = grid_search_lambda_generic(
            data, f, psd, df, m1, m2, K_z, grid, recover_fn,
            distance_Mpc=distance_Mpc)
        gc.collect()

        peak_idx, _ = find_peaks(logL, prominence=1.0)
        global_max_idx = np.argmax(logL)
        if global_max_idx not in peak_idx:
            peak_idx = np.append(peak_idx, global_max_idx)

        if len(peak_idx) >= 2:
            locs = Lgrid[peak_idx]
            heights = logL[peak_idx]
            order = np.argsort(heights)[::-1]
            locs = locs[order]
            spacing = abs(locs[0] - locs[1])
            spacings.append(spacing)

    return spacings


def fit_power_law(f_values, s_values):
    log_f = np.log(f_values)
    log_s = np.log(s_values)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_f, log_s)
    return -slope, std_err, r_value ** 2


def main():
    print("#" * 72)
    print("# STAGE 5H-9 — FIXED-GRID CONTROL")
    print("#" * 72)
    print()
    print("  Repeating 5H-8's 7-point f_max scan with a SINGLE FIXED")
    print("  Lambda grid [-20,+20] for all points, instead of the")
    print("  per-point adaptive half-width used in 5H-8.")
    print()

    m1, m2 = 35.6, 30.6
    z = 0.09
    distance_Mpc = 440.0
    K_z = cosmological_K_factor(z)
    f_lo = 20.0

    family_names = list(WAVEFORM_FAMILIES.keys())
    fn_A, fn_B = WAVEFORM_FAMILIES[family_names[0]], WAVEFORM_FAMILIES[family_names[1]]

    f_max_values = [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0]
    n_realizations = 4
    resolution_points = 61
    fixed_grid = np.linspace(-20.0, 20.0, resolution_points)

    total_calls = len(f_max_values) * 2 * n_realizations * resolution_points
    print(f"  Fixed grid: [-20, +20], {resolution_points} points "
          f"(resolution = {40.0/resolution_points:.3f})")
    print(f"  Planned pycbc.match() calls: {total_calls}")
    print()

    results = {"AB": [], "BA": []}

    for f_max in f_max_values:
        spacings_AB = measure_peak_spacing_fixed_grid(
            m1, m2, K_z, distance_Mpc, f_lo, f_max, fn_A, fn_B,
            fixed_grid, n_realizations,
            seed_base=hash(("5H9", f_max, "AB")) % (2**32))
        spacings_BA = measure_peak_spacing_fixed_grid(
            m1, m2, K_z, distance_Mpc, f_lo, f_max, fn_B, fn_A,
            fixed_grid, n_realizations,
            seed_base=hash(("5H9", f_max, "BA")) % (2**32))

        mean_AB = np.mean(spacings_AB) if spacings_AB else np.nan
        mean_BA = np.mean(spacings_BA) if spacings_BA else np.nan

        print(f"  f_max={f_max:>5.0f} Hz  "
              f"A->B={mean_AB:>7.3f} (n={len(spacings_AB)}/{n_realizations})  "
              f"B->A={mean_BA:>7.3f} (n={len(spacings_BA)}/{n_realizations})")

        results["AB"].append(mean_AB)
        results["BA"].append(mean_BA)

    print()

    print("=" * 72)
    print("POWER-LAW REGRESSION (fixed grid)")
    print("=" * 72)
    print()

    f_arr = np.array(f_max_values)
    s_AB = np.array(results["AB"])
    s_BA = np.array(results["BA"])

    valid_AB = np.isfinite(s_AB) & (s_AB > 0)
    valid_BA = np.isfinite(s_BA) & (s_BA > 0)

    n_AB, r2_AB = np.nan, np.nan
    if valid_AB.sum() >= 3:
        n_AB, err_AB, r2_AB = fit_power_law(f_arr[valid_AB], s_AB[valid_AB])
        print(f"  A->B: n = {n_AB:.3f} +/- {err_AB:.3f}, R^2 = {r2_AB:.4f}")
    else:
        print("  A->B: insufficient valid points for fit")

    n_BA, r2_BA = np.nan, np.nan
    if valid_BA.sum() >= 3:
        n_BA, err_BA, r2_BA = fit_power_law(f_arr[valid_BA], s_BA[valid_BA])
        print(f"  B->A: n = {n_BA:.3f} +/- {err_BA:.3f}, R^2 = {r2_BA:.4f}")
    else:
        print("  B->A: insufficient valid points for fit")
    print()

    mid = len(f_max_values) // 2
    n_lower = n_upper = np.nan
    if valid_AB[:mid + 1].sum() >= 2 and valid_AB[mid:].sum() >= 2:
        f_lower = f_arr[:mid + 1][valid_AB[:mid + 1]]
        s_AB_lower = s_AB[:mid + 1][valid_AB[:mid + 1]]
        f_upper = f_arr[mid:][valid_AB[mid:]]
        s_AB_upper = s_AB[mid:][valid_AB[mid:]]
        n_lower, _, r2_lower = fit_power_law(f_lower, s_AB_lower)
        n_upper, _, r2_upper = fit_power_law(f_upper, s_AB_upper)
        print(f"  A->B lower-half (100-250Hz) n = {n_lower:.3f}")
        print(f"  A->B upper-half (250-400Hz) n = {n_upper:.3f}")

    # ── Diagnosis ────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("DIAGNOSIS: does 5H-8's finding survive the fixed-grid control?")
    print("=" * 72)
    print()

    n_saturates = np.isfinite(n_lower) and np.isfinite(n_upper) and \
        abs(n_lower - n_upper) > 0.5

    print(f"  5H-8 (adaptive grid): n_overall~2.39, saturates "
          f"(lower=2.70, upper=1.62)")
    if np.isfinite(n_AB):
        print(f"  5H-9 (fixed grid):    n_overall={n_AB:.2f}, "
              f"saturates={'YES' if n_saturates else 'NO'} "
              f"(lower={n_lower:.2f}, upper={n_upper:.2f})")
    print()

    if n_saturates:
        print("  ┌────────────────────────────────────────────────────────┐")
        print("  │  5H-8's SATURATION FINDING SURVIVES the fixed-grid       │")
        print("  │  control. The f_max-dependent, non-power-law behavior    │")
        print("  │  is NOT an artifact of the adaptive search-window        │")
        print("  │  methodology. This strengthens the conclusion that the   │")
        print("  │  cross-family Lambda bias is genuinely band-dependent in │")
        print("  │  a complex (non-power-law) way, and single-band          │")
        print("  │  estimates cannot be interpreted as physical Lambda      │")
        print("  │  measurements without further band-systematics           │")
        print("  │  characterization.                                       │")
        print("  └────────────────────────────────────────────────────────┘")
    else:
        print("  ┌────────────────────────────────────────────────────────┐")
        print("  │  Saturation pattern WEAKENS or disappears under the      │")
        print("  │  fixed-grid control. Part of 5H-8's apparent saturation  │")
        print("  │  may have been an adaptive-window artifact. The          │")
        print("  │  underlying f_max dependence should be re-characterized  │")
        print("  │  using this fixed-grid methodology as the reference.     │")
        print("  └────────────────────────────────────────────────────────┘")

    # ── Plot ──────────────────────────────────────────────────────────────
    try:
        out = Path(__file__).parent / "results"
        out.mkdir(exist_ok=True)
    except OSError:
        out = Path.home() / "lambda_model_results"
        out.mkdir(exist_ok=True, parents=True)
        print(f"  WARNING: could not write to /mnt/c results dir, "
              f"using {out} instead.")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(f_arr[valid_AB], s_AB[valid_AB], "o-", color="firebrick",
             label=f"A->B fixed-grid (n={n_AB:.2f})" if np.isfinite(n_AB) else "A->B")
    ax.loglog(f_arr[valid_BA], s_BA[valid_BA], "o-", color="forestgreen",
             label=f"B->A fixed-grid (n={n_BA:.2f})" if np.isfinite(n_BA) else "B->A")
    ax.set_xlabel(r"$f_{\max}$ [Hz]")
    ax.set_ylabel(r"Peak spacing in $\Lambda$")
    ax.set_title("Stage 5H-9: Fixed-grid control")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    path = out / "stage5h9_fixed_grid_control.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
