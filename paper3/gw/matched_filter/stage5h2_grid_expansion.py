"""
stage5h2_grid_expansion.py
==============================

STAGE 5H-2 (standalone): does the [20,100) Hz cross-family Lambda_ML
keep growing in magnitude as the search grid widens, indicating no
interior likelihood maximum exists (pathological non-identifiability),
or does it converge to a finite value once the grid is wide enough?

Split into its own script (separate from stage5h1_convergence.py) to
keep each run within the memory-safe pycbc.match() call budget
established across Stages 5-5G. Run stage5h1_convergence.py FIRST in
a separate process, then this script.

CONTEXT: Stage 5G found A->B=-4.833 (50% boundary hits) and
B->A=+4.958 (75% boundary hits) at [20,100) Hz on a [-5,+5] grid, with
only n_realizations=4. This tests whether widening the grid to
[-10,+10] and [-20,+20] reveals a genuine interior maximum, or whether
the estimate keeps running toward the new boundary each time.
"""

import numpy as np
from pathlib import Path
import sys
import gc
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from waveform import cosmological_K_factor
from likelihood import aligo_like_psd
from stage5_cross_waveform_validation import (
    WAVEFORM_FAMILIES, grid_search_lambda_generic,
)


def run_null_combo(m1, m2, K_z, distance_Mpc, f_lo, f_hi, Lambda_grid,
                    inject_fn, recover_fn, n_realizations, seed_base):
    duration = 8.0
    df = 1.0 / duration
    f = np.arange(f_lo, f_hi, df)
    psd = aligo_like_psd(f)

    estimates = []
    rng = np.random.default_rng(seed_base)
    for real_idx in range(n_realizations):
        sigma = np.sqrt(psd / (4 * df))
        noise = rng.normal(0, sigma) + 1j * rng.normal(0, sigma)
        h_inject = inject_fn(f, m1, m2, 0.0, K_z, distance_Mpc=distance_Mpc)
        data = h_inject + noise

        _, _, Lam_ml, _ = grid_search_lambda_generic(
            data, f, psd, df, m1, m2, K_z, Lambda_grid, recover_fn,
            distance_Mpc=distance_Mpc)
        estimates.append(Lam_ml)
        gc.collect()

    estimates = np.array(estimates)
    grid_lo, grid_hi = Lambda_grid[0], Lambda_grid[-1]
    boundary_hits = np.sum((estimates <= grid_lo) | (estimates >= grid_hi))

    return dict(mean=np.mean(estimates), median=np.median(estimates),
                std=np.std(estimates), n=n_realizations,
                boundary_fraction=boundary_hits / n_realizations)


def main():
    print("#" * 72)
    print("# STAGE 5H-2 — GRID EXPANSION TEST ([20,100) Hz)")
    print("#" * 72)
    print()
    print("  Question: does Lambda_ML keep growing as the search grid")
    print("  widens, indicating pathological non-identifiability?")
    print()

    m1, m2 = 35.6, 30.6
    z = 0.09
    distance_Mpc = 440.0
    K_z = cosmological_K_factor(z)

    family_names = list(WAVEFORM_FAMILIES.keys())
    fn_A, fn_B = WAVEFORM_FAMILIES[family_names[0]], WAVEFORM_FAMILIES[family_names[1]]

    f_lo, f_hi = 20.0, 100.0
    grid_widths = [5.0, 10.0, 20.0]
    n_realizations = 4

    total_calls = 2 * n_realizations * sum(int(2 * w * 4) + 1 for w in grid_widths)
    print(f"  Planned pycbc.match() calls: {total_calls}")
    print()

    print(f"  {'Grid':>12}  {'A->B mean':>10}  {'A->B bound%':>12}  "
          f"{'B->A mean':>10}  {'B->A bound%':>12}")
    print("  " + "-" * 62)

    results = []
    for width in grid_widths:
        Lambda_grid = np.linspace(-width, width, int(2 * width * 4) + 1)

        r_AB = run_null_combo(m1, m2, K_z, distance_Mpc, f_lo, f_hi,
                               Lambda_grid, fn_A, fn_B, n_realizations,
                               seed_base=hash((width, "AB")) % (2**32))
        r_BA = run_null_combo(m1, m2, K_z, distance_Mpc, f_lo, f_hi,
                               Lambda_grid, fn_B, fn_A, n_realizations,
                               seed_base=hash((width, "BA")) % (2**32))

        print(f"  +/-{width:>8.0f}  {r_AB['mean']:>10.3f}  "
              f"{r_AB['boundary_fraction']*100:>11.1f}%  "
              f"{r_BA['mean']:>10.3f}  {r_BA['boundary_fraction']*100:>11.1f}%")

        results.append(dict(width=width, AB=r_AB, BA=r_BA))

    print()

    ab_means = [abs(r["AB"]["mean"]) for r in results]
    ab_widths = [r["width"] for r in results]

    print(f"  |A->B mean| / grid_width ratio: "
          f"{[f'{m/w:.3f}' for m, w in zip(ab_means, ab_widths)]}")
    print(f"  (Ratio staying roughly CONSTANT => estimate scales with grid,")
    print(f"  i.e. runaway/non-identifiable. Ratio DECREASING => converging")
    print(f"  to a finite interior value.)")
    print()

    all_saturated = all(r["AB"]["boundary_fraction"] > 0.3 for r in results)
    last_clean = results[-1]["AB"]["boundary_fraction"] < 0.1

    if all_saturated:
        print("  ┌────────────────────────────────────────────────────────┐")
        print("  │  FINDING: boundary saturation PERSISTS even at +/-20.    │")
        print("  │  PATHOLOGICAL NON-IDENTIFIABILITY in [20,100)Hz -- the    │")
        print("  │  cross-family likelihood has no interior maximum here.   │")
        print("  │  Any single 'Lambda_ML' from a finite grid in this band  │")
        print("  │  is an artifact of grid truncation, not a measurement.   │")
        print("  │  EXCLUDE this band from physical bias characterization   │")
        print("  │  until this degeneracy is understood.                    │")
        print("  └────────────────────────────────────────────────────────┘")
    elif last_clean:
        print("  ┌────────────────────────────────────────────────────────┐")
        print("  │  FINDING: boundary saturation RESOLVES as the grid       │")
        print("  │  widens -- converges to a finite interior value. The     │")
        print("  │  earlier [-5,+5] grid was simply too narrow; this band   │")
        print("  │  DOES have a well-defined cross-family bias, which       │")
        print("  │  should be re-measured with this wider grid.             │")
        print("  └────────────────────────────────────────────────────────┘")
    else:
        print("  FINDING: intermediate -- saturation partially resolves but")
        print("  does not fully disappear at +/-20. Further widening or a")
        print("  profile-likelihood approach (not grid search) may be needed.")

    # ── Plot ──────────────────────────────────────────────────────────────
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    widths = [r["width"] for r in results]
    ab_means_signed = [r["AB"]["mean"] for r in results]
    ba_means_signed = [r["BA"]["mean"] for r in results]
    ab_bound = [r["AB"]["boundary_fraction"] * 100 for r in results]
    ba_bound = [r["BA"]["boundary_fraction"] * 100 for r in results]
    ax.plot(widths, ab_means_signed, "o-", color="firebrick", label="A->B mean")
    ax.plot(widths, ba_means_signed, "o-", color="forestgreen", label="B->A mean")
    ax2 = ax.twinx()
    ax2.plot(widths, ab_bound, "s--", color="firebrick", alpha=0.5,
             label="A->B boundary%")
    ax2.plot(widths, ba_bound, "s--", color="forestgreen", alpha=0.5,
             label="B->A boundary%")
    ax2.set_ylabel("Boundary hit %")
    ax.set_xlabel("Grid half-width")
    ax.set_ylabel(r"$\Lambda_{\rm ML}$ mean")
    ax.set_title("Stage 5H-2: Grid expansion, [20,100)Hz")
    ax.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out / "stage5h2_grid_expansion.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
