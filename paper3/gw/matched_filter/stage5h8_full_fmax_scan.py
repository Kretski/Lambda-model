"""
stage5h8_full_fmax_scan.py
==============================

STAGE 5H-8: full 7-point f_max scan to properly fit the power-law
exponent n in S(f_max) = A * f_max^(-n), rather than relying on 2-3
individual point comparisons (5H-6, 5H-7).

CONTEXT: 5H-6 rejected n=3 (naive f_c=f_max aliasing). 5H-7's 300Hz
point favored n~1.9 over n=3, but a single additional point cannot
establish a reliable scaling law -- this requires proper log-space
linear regression across many points, with an explicit R^2 to quantify
how well ANY power law fits the data at all.

ADDITIONAL CRITICAL CHECK (saturation): if S(f_max) does NOT follow a
clean power law but instead SATURATES (plateaus) at high f_max, this
would indicate no genuine runaway aliasing scaling exists at all --
the apparent bias is simply a band-dependent numerical sensitivity,
not a mechanism with a well-defined asymptotic law. This is tested by
comparing the fitted n from the FULL 7-point set against a fit using
only the upper half of frequencies (250-400Hz) vs lower half
(100-250Hz): if n differs substantially between these subsets, no
single global power law describes the data -- saturation or a more
complex dependence is present.

BOTH A->B and B->A are fit independently, and their exponents compared:
if n_AB ~ n_BA, this supports a COMMON underlying numerical/band-edge
mechanism (not a family-specific effect). If they differ substantially,
the mechanism is more complex than a shared band-edge artifact.

Frequencies tested: 100, 150, 200, 250, 300, 350, 400 Hz (7 points).
N=6 realizations per point/combo, resolution=41 points per fine-grid
search, keeping total cost within the established memory-safe budget.
"""

import numpy as np
from pathlib import Path
import sys
import gc
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from waveform import cosmological_K_factor, C_SI
from likelihood import aligo_like_psd
from stage5_cross_waveform_validation import (
    WAVEFORM_FAMILIES, grid_search_lambda_generic,
)


def predicted_spacing_n3(f_c, K_z):
    return C_SI ** 3 / (2 * np.pi ** 2 * K_z * f_c ** 3)


def measure_peak_spacing(m1, m2, K_z, distance_Mpc, f_lo, f_hi,
                           inject_fn, recover_fn, center, half_width,
                           n_realizations, seed_base, resolution_points):
    duration = 8.0
    df = 1.0 / duration
    f = np.arange(f_lo, f_hi, df)
    psd = aligo_like_psd(f)

    grid = np.linspace(center - half_width, center + half_width,
                        resolution_points)

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
    n_fit = -slope
    return n_fit, std_err, r_value ** 2


def main():
    print("#" * 72)
    print("# STAGE 5H-8 — FULL 7-POINT f_max SCAN")
    print("#" * 72)
    print()

    m1, m2 = 35.6, 30.6
    z = 0.09
    distance_Mpc = 440.0
    K_z = cosmological_K_factor(z)
    f_lo = 20.0

    family_names = list(WAVEFORM_FAMILIES.keys())
    fn_A, fn_B = WAVEFORM_FAMILIES[family_names[0]], WAVEFORM_FAMILIES[family_names[1]]

    f_max_values = [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0]
    n_realizations = 6
    resolution_points = 41
    total_calls = len(f_max_values) * 2 * n_realizations * resolution_points
    print(f"  Planned pycbc.match() calls: {total_calls}")
    print(f"  f_max values: {f_max_values}")
    print()

    results = {"AB": [], "BA": []}

    for f_max in f_max_values:
        half_width = max(2.0, 0.492 * (400.0 / f_max) ** 2.0 * 3)

        spacings_AB = measure_peak_spacing(
            m1, m2, K_z, distance_Mpc, f_lo, f_max, fn_A, fn_B,
            center=0.0, half_width=half_width, n_realizations=n_realizations,
            seed_base=hash(("5H8", f_max, "AB")) % (2**32),
            resolution_points=resolution_points)
        spacings_BA = measure_peak_spacing(
            m1, m2, K_z, distance_Mpc, f_lo, f_max, fn_B, fn_A,
            center=0.0, half_width=half_width, n_realizations=n_realizations,
            seed_base=hash(("5H8", f_max, "BA")) % (2**32),
            resolution_points=resolution_points)

        mean_AB = np.mean(spacings_AB) if spacings_AB else np.nan
        mean_BA = np.mean(spacings_BA) if spacings_BA else np.nan

        print(f"  f_max={f_max:>5.0f} Hz  half_width={half_width:>6.2f}  "
              f"A->B={mean_AB:>7.3f} (n={len(spacings_AB)}/{n_realizations})  "
              f"B->A={mean_BA:>7.3f} (n={len(spacings_BA)}/{n_realizations})")

        results["AB"].append(mean_AB)
        results["BA"].append(mean_BA)

    print()

    print("=" * 72)
    print("POWER-LAW REGRESSION")
    print("=" * 72)
    print()

    f_arr = np.array(f_max_values)
    s_AB = np.array(results["AB"])
    s_BA = np.array(results["BA"])

    valid_AB = np.isfinite(s_AB) & (s_AB > 0)
    valid_BA = np.isfinite(s_BA) & (s_BA > 0)

    n_AB, err_AB, r2_AB = fit_power_law(f_arr[valid_AB], s_AB[valid_AB])
    n_BA, err_BA, r2_BA = fit_power_law(f_arr[valid_BA], s_BA[valid_BA])

    print(f"  A->B: n = {n_AB:.3f} +/- {err_AB:.3f}, R^2 = {r2_AB:.4f}")
    print(f"  B->A: n = {n_BA:.3f} +/- {err_BA:.3f}, R^2 = {r2_BA:.4f}")
    print()

    print("  Saturation check (upper vs lower frequency half):")
    mid = len(f_max_values) // 2
    f_lower, s_AB_lower = f_arr[:mid + 1][valid_AB[:mid + 1]], s_AB[:mid + 1][valid_AB[:mid + 1]]
    f_upper, s_AB_upper = f_arr[mid:][valid_AB[mid:]], s_AB[mid:][valid_AB[mid:]]

    n_consistent = None
    if len(f_lower) >= 2 and len(f_upper) >= 2:
        n_lower, _, r2_lower = fit_power_law(f_lower, s_AB_lower)
        n_upper, _, r2_upper = fit_power_law(f_upper, s_AB_upper)
        print(f"    A->B lower-half (100-250Hz) n = {n_lower:.3f} (R^2={r2_lower:.3f})")
        print(f"    A->B upper-half (250-400Hz) n = {n_upper:.3f} (R^2={r2_upper:.3f})")
        n_consistent = abs(n_lower - n_upper) < 0.5
    else:
        print("    Insufficient valid points for sub-range comparison.")
    print()

    print("=" * 72)
    print("DIAGNOSIS")
    print("=" * 72)
    print()

    families_agree = abs(n_AB - n_BA) < 0.5
    good_fit = r2_AB > 0.8 and r2_BA > 0.8

    print(f"  A->B and B->A exponents agree (|dn|<0.5): {families_agree}")
    print(f"  Both fits have R^2 > 0.8: {good_fit}")
    if n_consistent is not None:
        print(f"  Upper/lower-half exponents consistent (no saturation): {n_consistent}")
    print()

    if good_fit and families_agree and n_consistent:
        print("  ┌────────────────────────────────────────────────────────┐")
        print("  │  CLEAN COMMON POWER LAW ESTABLISHED. Both A->B and B->A  │")
        print("  │  follow S(f_max) ~ f_max^-n with a SHARED exponent,      │")
        print(f"  │  (n_AB={n_AB:.2f}, n_BA={n_BA:.2f}), high R^2, and no        │")
        print("  │  saturation across the tested range. This is strong      │")
        print("  │  evidence of a COMMON NUMERICAL/BAND-EDGE MECHANISM      │")
        print("  │  (not family-specific physics). The [20,400)Hz results   │")
        print("  │  from earlier Stage 5 tests should be treated as         │")
        print("  │  band-dependent numerical artifacts, NOT physical        │")
        print("  │  waveform-family bias, until a properly conditioned      │")
        print("  │  (tapered) analysis is performed.                        │")
        print("  └────────────────────────────────────────────────────────┘")
    elif good_fit and not n_consistent:
        print("  ┌────────────────────────────────────────────────────────┐")
        print("  │  POWER LAW FITS WELL OVERALL BUT SATURATES. The          │")
        print("  │  upper/lower-half exponents disagree substantially --    │")
        print("  │  no single global power law describes the full range.    │")
        print("  │  The apparent Λ bias is BAND-DEPENDENT in a more          │")
        print("  │  complex way than a pure power law; treat any single-    │")
        print("  │  band cross-family bias number as NOT YET physically     │")
        print("  │  interpretable.                                          │")
        print("  └────────────────────────────────────────────────────────┘")
    else:
        print("  ┌────────────────────────────────────────────────────────┐")
        print("  │  NO CLEAN POWER LAW ESTABLISHED (low R^2 and/or          │")
        print("  │  family disagreement). The apparent n~1.9 pattern from   │")
        print("  │  5H-6/5H-7 does not hold up under the full 7-point       │")
        print("  │  scan. Multimodality/scatter likely dominated by noise   │")
        print("  │  or small-N effects at each point; a fundamentally       │")
        print("  │  different diagnostic (direct phase-curve inspection)    │")
        print("  │  is warranted instead of further grid scans.             │")
        print("  └────────────────────────────────────────────────────────┘")

    # ── Plot ──────────────────────────────────────────────────────────────
    # WSL/mnt-C filesystem can intermittently throw I/O errors on mkdir
    # (observed during long-running scripts); fall back to a native WSL
    # path if the /mnt/c results directory is not writable, so the
    # already-completed analysis is never lost to a save-step failure.
    out = Path(__file__).parent / "results"
    try:
        out.mkdir(exist_ok=True)
    except OSError:
        out = Path.home() / "lambda_model_results"
        out.mkdir(exist_ok=True, parents=True)
        print(f"  WARNING: could not write to /mnt/c results dir, "
              f"using {out} instead.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.loglog(f_arr[valid_AB], s_AB[valid_AB], "o-", color="firebrick",
             label=f"A->B (n={n_AB:.2f})")
    ax.loglog(f_arr[valid_BA], s_BA[valid_BA], "o-", color="forestgreen",
             label=f"B->A (n={n_BA:.2f})")
    f_smooth = np.linspace(90, 420, 100)
    n3_curve = [predicted_spacing_n3(fc, K_z) for fc in f_smooth]
    ax.loglog(f_smooth, n3_curve, "k--", lw=1, alpha=0.6, label="n=3 reference")
    ax.set_xlabel(r"$f_{\max}$ [Hz]")
    ax.set_ylabel(r"Peak spacing in $\Lambda$")
    ax.set_title("Stage 5H-8: Full 7-point scan")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[1]
    if len(f_arr[valid_AB]) > 0:
        f0 = f_arr[valid_AB][0]
        s0 = s_AB[valid_AB][0]
        residuals_AB = np.log(s_AB[valid_AB]) - (np.log(s0) -
                                                   n_AB * (np.log(f_arr[valid_AB]) - np.log(f0)))
        ax.plot(f_arr[valid_AB], residuals_AB, "o-", color="firebrick", label="A->B residuals")
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel(r"$f_{\max}$ [Hz]")
    ax.set_ylabel("log-residual from fit")
    ax.set_title("Stage 5H-8: Power-law fit residuals")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Stage 5H-8: Full f_max scan and power-law regression", fontsize=11)
    plt.tight_layout()
    path = out / "stage5h8_full_fmax_scan.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
