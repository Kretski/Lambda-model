"""
stage5h6_aliasing_confirmation.py
=====================================

STAGE 5H-6: confirm the phase-wrap ALIASING hypothesis from 5H-5's
analytical check.

ANALYTICAL PREDICTION (5H-5): the observed ~0.48-0.50 peak spacing in
[20,400)Hz matches EXACTLY the phase-wrap interval

    Delta_Lambda_wrap(f_c) = c^3 / (2*pi^2*K(z)*f_c^3)

at f_c ~ 400 Hz (the upper band edge), for BOTH A->B (implied f_c =
399.3 Hz) and B->A (402.1 Hz).

DEFINITIVE TEST: if this is genuine aliasing from the band edge (not a
physical waveform-family effect), then changing f_max should shift the
peak spacing in a PREDICTABLE way, following the same formula. If we
instead used a lower f_max (e.g. 200 Hz), the predicted spacing would
be Delta_Lambda_wrap(200) ~ 3.88 -- nearly 8x wider than at 400 Hz. If
the ACTUAL measured peak spacing at f_max=200Hz also comes out near
3.88, this is decisive confirmation of aliasing. If the spacing does
NOT track f_max this way, the phase-wrap hypothesis is wrong and
genuine multimodality remains on the table.

This is a single, cheap, targeted test -- much less expensive than the
originally proposed general "fine-grid-resolution scan", because we
now have a SPECIFIC quantitative prediction to test rather than a
qualitative resolution-independence check.
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


def predicted_spacing(f_c, K_z):
    return C_SI ** 3 / (2 * np.pi ** 2 * K_z * f_c ** 3)


def measure_peak_spacing(m1, m2, K_z, distance_Mpc, f_lo, f_hi,
                           inject_fn, recover_fn, center, half_width,
                           n_realizations, seed_base, resolution_points=161):
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
    print("# STAGE 5H-6 — ALIASING CONFIRMATION (VARY f_max)")
    print("#" * 72)
    print()

    m1, m2 = 35.6, 30.6
    z = 0.09
    distance_Mpc = 440.0
    K_z = cosmological_K_factor(z)
    f_lo = 20.0

    family_names = list(WAVEFORM_FAMILIES.keys())
    fn_A, fn_B = WAVEFORM_FAMILIES[family_names[0]], WAVEFORM_FAMILIES[family_names[1]]

    test_configs = [
        dict(f_max=400.0, center=-0.17, half_width=2.0, n_pts=81),
        dict(f_max=200.0, center=0.0, half_width=8.0, n_pts=161),
    ]

    n_realizations = 6
    total_calls = sum(2 * n_realizations * c["n_pts"] for c in test_configs)
    print(f"  Planned pycbc.match() calls: {total_calls}")
    print()

    results = []
    for cfg in test_configs:
        f_max = cfg["f_max"]
        predicted = predicted_spacing(f_max, K_z)
        print(f"  f_max = {f_max:.0f} Hz  (predicted spacing = {predicted:.3f})")

        spacings_AB = measure_peak_spacing(
            m1, m2, K_z, distance_Mpc, f_lo, f_max, fn_A, fn_B,
            cfg["center"], cfg["half_width"], n_realizations,
            seed_base=hash(("5H6", f_max, "AB")) % (2**32),
            resolution_points=cfg["n_pts"])
        spacings_BA = measure_peak_spacing(
            m1, m2, K_z, distance_Mpc, f_lo, f_max, fn_B, fn_A,
            cfg["center"], cfg["half_width"], n_realizations,
            seed_base=hash(("5H6", f_max, "BA")) % (2**32),
            resolution_points=cfg["n_pts"])

        mean_AB = np.mean(spacings_AB) if spacings_AB else np.nan
        mean_BA = np.mean(spacings_BA) if spacings_BA else np.nan

        print(f"    Measured A->B spacing: {mean_AB:.3f} "
              f"(n={len(spacings_AB)}/{n_realizations} had 2+ peaks)")
        print(f"    Measured B->A spacing: {mean_BA:.3f} "
              f"(n={len(spacings_BA)}/{n_realizations} had 2+ peaks)")
        print()

        results.append(dict(f_max=f_max, predicted=predicted,
                            measured_AB=mean_AB, measured_BA=mean_BA))

    # ── Diagnosis ────────────────────────────────────────────────────────
    print("=" * 72)
    print("DIAGNOSIS")
    print("=" * 72)
    print()

    print(f"  {'f_max':>8}  {'predicted':>10}  {'meas A->B':>10}  "
          f"{'meas B->A':>10}  {'ratio A->B':>11}")
    print("  " + "-" * 56)
    for r in results:
        ratio = r["measured_AB"] / r["predicted"] if np.isfinite(r["measured_AB"]) else np.nan
        print(f"  {r['f_max']:>8.0f}  {r['predicted']:>10.3f}  "
              f"{r['measured_AB']:>10.3f}  {r['measured_BA']:>10.3f}  "
              f"{ratio:>11.2f}")

    print()

    ratios = [r["measured_AB"] / r["predicted"] for r in results
              if np.isfinite(r["measured_AB"]) and r["predicted"] > 0]

    if len(ratios) >= 2 and all(0.7 < ratio < 1.4 for ratio in ratios):
        print("  ┌────────────────────────────────────────────────────────┐")
        print("  │  ALIASING CONFIRMED. Measured peak spacing tracks the    │")
        print("  │  predicted phase-wrap formula within ~30% across         │")
        print("  │  different f_max values (spacing scales as f_max^-3,     │")
        print("  │  as predicted by the cubic Lambda phase term). This is   │")
        print("  │  a NUMERICAL ALIASING ARTIFACT of the finite analysis    │")
        print("  │  band, not physical waveform-family multimodality.       │")
        print("  │                                                          │")
        print("  │  CONSEQUENCE FOR STAGE 5 OVERALL: the cross-family bias  │")
        print("  │  numbers measured in 5D/5F/5G/5H-1..5H-4 for [20,400)Hz  │")
        print("  │  need to be RE-EXAMINED for aliasing contamination.      │")
        print("  │  Recommended fix: apply a frequency-domain taper near    │")
        print("  │  the band edges before matched filtering (standard GW    │")
        print("  │  practice) and re-run the key diagnostics.               │")
        print("  └────────────────────────────────────────────────────────┘")
    else:
        print("  Spacing does NOT clearly track the f_max^-3 prediction.")
        print("  The phase-wrap aliasing hypothesis is NOT confirmed by this")
        print("  test. Genuine multimodality remains a live possibility and")
        print("  requires the originally-proposed broader diagnostic.")

    # ── Plot ──────────────────────────────────────────────────────────────
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    f_maxes = [r["f_max"] for r in results]
    predicted_vals = [r["predicted"] for r in results]
    measured_AB = [r["measured_AB"] for r in results]
    measured_BA = [r["measured_BA"] for r in results]

    ax.loglog(f_maxes, predicted_vals, "k--", lw=2, marker="s",
             label=r"predicted ($\propto f_{\max}^{-3}$)")
    ax.loglog(f_maxes, measured_AB, "o-", color="firebrick",
             label="measured A->B")
    ax.loglog(f_maxes, measured_BA, "o-", color="forestgreen",
             label="measured B->A")
    ax.set_xlabel(r"$f_{\max}$ [Hz]")
    ax.set_ylabel(r"Peak spacing in $\Lambda$")
    ax.set_title("Stage 5H-6: Aliasing confirmation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    path = out / "stage5h6_aliasing_confirmation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
