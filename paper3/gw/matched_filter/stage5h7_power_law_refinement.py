"""
stage5h7_power_law_refinement.py
====================================

STAGE 5H-7 (minimal, targeted version): Stage 5H-6 rejected the naive
"f_c = f_max" aliasing hypothesis (n=3 exactly), but a free two-point
fit to the SAME data revealed an unexpectedly clean empirical power
law with n ~ 1.9 for BOTH A->B (1.885) and B->A (1.948) -- consistent
between the two cross-family directions, which would be a strange
coincidence if this were just noise/multimodality with no underlying
scaling law at all.

Rather than immediately running the originally-proposed full 7-point
frequency scan (expensive), this script adds ONE decisive third
measurement at f_max=300 Hz, which the n=3 and n~1.9 hypotheses predict
very differently (1.150 vs 0.846 -- a 36% difference, cheaply
resolvable with a modest number of realizations):

  - If the 300Hz measurement lands near 1.15 -> supports the ORIGINAL
    n=3 hypothesis being correct in principle, with f_max=200Hz simply
    being an outlier/small-sample fluctuation in 5H-6.
  - If it lands near 0.85 -> supports a genuine but MODIFIED aliasing
    mechanism with effective n~1.9, warranting the full frequency scan
    (broader 5H-7) to characterize precisely.
  - If it lands somewhere else entirely -> neither simple power law
    holds, and the two-point "n~1.9" agreement between A->B/B->A was
    coincidental; genuine unexplained structure remains, and the
    broader multimodality characterization (independent of any
    aliasing hypothesis) becomes the priority.

This keeps compute cost minimal while directly discriminating between
the two live hypotheses before committing to the expensive 7-point
scan.
"""

import numpy as np
from pathlib import Path
import sys
import gc
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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


def main():
    print("#" * 72)
    print("# STAGE 5H-7 (minimal) — THIRD-POINT POWER-LAW DISCRIMINATION")
    print("#" * 72)
    print()

    m1, m2 = 35.6, 30.6
    z = 0.09
    distance_Mpc = 440.0
    K_z = cosmological_K_factor(z)
    f_lo = 20.0
    f_test = 300.0

    family_names = list(WAVEFORM_FAMILIES.keys())
    fn_A, fn_B = WAVEFORM_FAMILIES[family_names[0]], WAVEFORM_FAMILIES[family_names[1]]

    n_realizations = 8
    resolution_points = 121
    total_calls = 2 * n_realizations * resolution_points
    print(f"  Planned pycbc.match() calls: {total_calls}")
    print()

    pred_n3 = predicted_spacing_n3(f_test, K_z)
    n_emp = 1.9
    anchor_spacing_400 = 0.492
    pred_n_emp = anchor_spacing_400 * (400.0 / f_test) ** n_emp

    print(f"  f_max = {f_test:.0f} Hz")
    print(f"  Prediction if n=3 (original): {pred_n3:.3f}")
    print(f"  Prediction if n~1.9 (empirical): {pred_n_emp:.3f}")
    print()

    spacings_AB = measure_peak_spacing(
        m1, m2, K_z, distance_Mpc, f_lo, f_test, fn_A, fn_B,
        center=0.0, half_width=5.0, n_realizations=n_realizations,
        seed_base=hash(("5H7", f_test, "AB")) % (2**32),
        resolution_points=resolution_points)
    spacings_BA = measure_peak_spacing(
        m1, m2, K_z, distance_Mpc, f_lo, f_test, fn_B, fn_A,
        center=0.0, half_width=5.0, n_realizations=n_realizations,
        seed_base=hash(("5H7", f_test, "BA")) % (2**32),
        resolution_points=resolution_points)

    mean_AB = np.mean(spacings_AB) if spacings_AB else np.nan
    mean_BA = np.mean(spacings_BA) if spacings_BA else np.nan

    print(f"  Measured A->B spacing: {mean_AB:.3f} "
          f"(n={len(spacings_AB)}/{n_realizations} had 2+ peaks)")
    print(f"  Measured B->A spacing: {mean_BA:.3f} "
          f"(n={len(spacings_BA)}/{n_realizations} had 2+ peaks)")
    print()

    # ── Diagnosis ────────────────────────────────────────────────────────
    print("=" * 72)
    print("DIAGNOSIS")
    print("=" * 72)
    print()

    dist_to_n3 = abs(mean_AB - pred_n3) if np.isfinite(mean_AB) else np.inf
    dist_to_n_emp = abs(mean_AB - pred_n_emp) if np.isfinite(mean_AB) else np.inf

    print(f"  |measured - n=3 prediction|:    {dist_to_n3:.3f}")
    print(f"  |measured - n~1.9 prediction|:  {dist_to_n_emp:.3f}")
    print()

    if dist_to_n_emp < dist_to_n3 * 0.6:
        print("  ┌────────────────────────────────────────────────────────┐")
        print("  │  n~1.9 POWER LAW SUPPORTED. The measured 300Hz spacing   │")
        print("  │  is substantially closer to the empirical n~1.9          │")
        print("  │  prediction than the original n=3 hypothesis. This       │")
        print("  │  suggests a MODIFIED aliasing mechanism -- the effective │")
        print("  │  characteristic frequency scales sub-linearly with       │")
        print("  │  f_max, plausibly because SNR-weighted sensitivity for   │")
        print("  │  these masses is concentrated at frequencies below       │")
        print("  │  f_max, not at f_max itself. This IS still a numerical/  │")
        print("  │  band-edge effect, just with a different scaling than    │")
        print("  │  initially guessed -- warrants the full 7-point scan to  │")
        print("  │  pin down the precise mechanism and confirm before       │")
        print("  │  treating [20,400)Hz results as physical.                │")
        print("  └────────────────────────────────────────────────────────┘")
    elif dist_to_n3 < dist_to_n_emp * 0.6:
        print("  ┌────────────────────────────────────────────────────────┐")
        print("  │  n=3 HYPOTHESIS SUPPORTED after all. The 200Hz point     │")
        print("  │  from 5H-6 was likely a small-sample (N=6) fluctuation.  │")
        print("  │  Original aliasing hypothesis (f_c=f_max, n=3) remains   │")
        print("  │  viable -- recommend re-measuring the 200Hz point with   │")
        print("  │  higher N before drawing final conclusions.              │")
        print("  └────────────────────────────────────────────────────────┘")
    else:
        print("  ┌────────────────────────────────────────────────────────┐")
        print("  │  NEITHER simple power law is clearly favored. The        │")
        print("  │  apparent n~1.9 agreement between A->B and B->A at the   │")
        print("  │  200Hz point was likely coincidental, or small-N noise   │")
        print("  │  dominates all these measurements. No clean scaling law  │")
        print("  │  has been established. Multimodality should be treated  │")
        print("  │  as UNEXPLAINED structure requiring either much higher   │")
        print("  │  N per point, or a fundamentally different diagnostic    │")
        print("  │  (e.g. direct inspection of the Delta_Psi(f) phase       │")
        print("  │  curve itself, not just the resulting match statistic).  │")
        print("  └────────────────────────────────────────────────────────┘")

    # ── Plot ──────────────────────────────────────────────────────────────
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    f_data = [400.0, 300.0, 200.0]
    measured_data = [0.492, mean_AB, 1.817]

    f_smooth = np.linspace(150, 450, 100)
    n3_curve = [predicted_spacing_n3(fc, K_z) for fc in f_smooth]
    n_emp_curve = [anchor_spacing_400 * (400.0 / fc) ** n_emp for fc in f_smooth]

    ax.loglog(f_smooth, n3_curve, "k--", lw=1.5, label="n=3 prediction")
    ax.loglog(f_smooth, n_emp_curve, "b--", lw=1.5, label="n~1.9 prediction")
    ax.loglog(f_data, measured_data, "o", color="firebrick", markersize=10,
             label="measured (A->B)", zorder=5)
    ax.set_xlabel(r"$f_{\max}$ [Hz]")
    ax.set_ylabel(r"Peak spacing in $\Lambda$")
    ax.set_title("Stage 5H-7: Third-point power-law discrimination")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    path = out / "stage5h7_power_law_refinement.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
