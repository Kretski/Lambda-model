"""
stage5f_bias_decomposition.py
=================================

STAGE 5F: localize the source of the large cross-family Λ bias found
in Stage 5C/5D (A->B bias=+0.19, B->A bias=-0.19, ~7.6x statistical
scatter -- see STAGE5 results). This does NOT attempt to "fix" the
bias; it attempts to LOCATE where it comes from, following the
scientific discipline used throughout this repository: diagnose before
treating.

TWO INDEPENDENT DECOMPOSITIONS:

  5F-1 — FREQUENCY-BAND DEPENDENCE.
    Repeat the cross-family recovery test restricted to progressively
    wider frequency bands: [20,100) Hz (inspiral-only, low-frequency),
    [20,200) Hz (mid), [20,400) Hz (full band, matching Stage 5's
    original test, includes merger/ringdown for these masses).
    If bias grows with the upper frequency cutoff, this points to the
    merger/ringdown region (or the high-frequency part of inspiral) as
    the dominant source -- consistent with the hypothesis that
    IMRPhenomD and SEOBNRv4 diverge most in their merger-ringdown
    modeling, not in the early inspiral where both are well-constrained
    by the same PN theory.

  5F-2 — FULL WAVEFORM-FAMILY MATRIX (A->A, A->B, B->A, B->B).
    Stage 5C/5D only tested A->A as the control, implicitly assuming
    B->B would behave the same way. This explicitly tests B->B as an
    independent control, completing the 2x2 matrix. If B->B behaves
    like A->A (bias ~0, small scatter), this confirms the bias is
    specifically a CROSS-family effect, not a problem with waveform
    family B in isolation.

Uses the SAME waveform families, Λ phase model, and pycbc-based
matched-filter machinery as stage5_cross_waveform_validation.py --
this script imports directly from it rather than duplicating logic.
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


def run_recovery_single(m1, m2, K_z, distance_Mpc, f, psd, df,
                         Lambda_grid, inject_fn, recover_fn,
                         Lambda_true, n_realizations=2, seed_base=0):
    """Single injection/recovery measurement, reused across both 5F-1
    and 5F-2 to avoid duplicating the core loop logic."""
    ml_estimates = []
    rng = np.random.default_rng(seed_base)
    for real_idx in range(n_realizations):
        sigma = np.sqrt(psd / (4 * df))
        noise = rng.normal(0, sigma) + 1j * rng.normal(0, sigma)
        h_inject = inject_fn(f, m1, m2, Lambda_true, K_z,
                              distance_Mpc=distance_Mpc)
        data = h_inject + noise

        _, _, Lam_ml, _ = grid_search_lambda_generic(
            data, f, psd, df, m1, m2, K_z, Lambda_grid, recover_fn,
            distance_Mpc=distance_Mpc)
        ml_estimates.append(Lam_ml)
        gc.collect()

    ml_estimates = np.array(ml_estimates)
    return np.mean(ml_estimates), np.std(ml_estimates)


# ══════════════════════════════════════════════════════════════════════════
def run_5F1_frequency_dependence(m1, m2, K_z, distance_Mpc):
    print("=" * 72)
    print("MODE 5F-1 — FREQUENCY-BAND DEPENDENCE OF CROSS-FAMILY BIAS")
    print("=" * 72)
    print()

    duration = 8.0
    df = 1.0 / duration
    bands = [(20.0, 100.0, "inspiral-only"),
             (20.0, 200.0, "mid-band"),
             (20.0, 400.0, "full band (Stage 5 original)")]

    Lambda_true = 0.1  # fixed at one representative value for this scan
    Lambda_grid = np.linspace(-2.0, 2.0, 81)

    family_names = list(WAVEFORM_FAMILIES.keys())
    fn_A, fn_B = WAVEFORM_FAMILIES[family_names[0]], WAVEFORM_FAMILIES[family_names[1]]

    print(f"  Fixed Lambda_true = {Lambda_true}")
    print()
    print(f"  {'Band':>28}  {'A->A bias':>12}  {'A->B bias':>12}  "
          f"{'B->A bias':>12}")
    print("  " + "-" * 72)

    results = []
    for f_lo, f_hi, label in bands:
        f = np.arange(f_lo, f_hi, df)
        psd = aligo_like_psd(f)

        mean_AA, _ = run_recovery_single(
            m1, m2, K_z, distance_Mpc, f, psd, df, Lambda_grid,
            fn_A, fn_A, Lambda_true, seed_base=hash((f_lo, f_hi, "AA")) % (2**32))
        mean_AB, _ = run_recovery_single(
            m1, m2, K_z, distance_Mpc, f, psd, df, Lambda_grid,
            fn_A, fn_B, Lambda_true, seed_base=hash((f_lo, f_hi, "AB")) % (2**32))
        mean_BA, _ = run_recovery_single(
            m1, m2, K_z, distance_Mpc, f, psd, df, Lambda_grid,
            fn_B, fn_A, Lambda_true, seed_base=hash((f_lo, f_hi, "BA")) % (2**32))

        bias_AA = mean_AA - Lambda_true
        bias_AB = mean_AB - Lambda_true
        bias_BA = mean_BA - Lambda_true

        print(f"  [{f_lo:.0f},{f_hi:.0f}) Hz {label:>18}  {bias_AA:>12.4f}  "
              f"{bias_AB:>12.4f}  {bias_BA:>12.4f}")

        results.append(dict(f_lo=f_lo, f_hi=f_hi, label=label,
                             bias_AA=bias_AA, bias_AB=bias_AB, bias_BA=bias_BA))

    print()
    ab_biases = [abs(r["bias_AB"]) for r in results]
    if ab_biases[-1] > ab_biases[0] * 1.5:
        print("  FINDING: |bias| grows substantially with upper frequency cutoff.")
        print("  This points to the merger/ringdown / high-frequency inspiral")
        print("  region as the dominant source of cross-family disagreement,")
        print("  NOT the low-frequency inspiral where both approximants are")
        print("  tightly constrained by the same well-established PN theory.")
    elif ab_biases[-1] < ab_biases[0] * 0.67:
        print("  FINDING: |bias| DECREASES with upper frequency cutoff --")
        print("  unexpected; the low-frequency/inspiral region contributes")
        print("  more than merger/ringdown. Requires further investigation.")
    else:
        print("  FINDING: |bias| is roughly CONSTANT across frequency bands.")
        print("  This suggests the discrepancy is not localized to a specific")
        print("  frequency region (e.g. merger) but is a broadband effect,")
        print("  possibly in the overall phase/amplitude normalization")
        print("  convention between approximants, or in how the Lambda phase")
        print("  term interacts with the waveform across the full band.")

    print()
    return results


# ══════════════════════════════════════════════════════════════════════════
def run_5F2_full_matrix(m1, m2, K_z, distance_Mpc):
    print("=" * 72)
    print("MODE 5F-2 — FULL WAVEFORM-FAMILY MATRIX (A->A, A->B, B->A, B->B)")
    print("=" * 72)
    print()

    duration = 8.0
    df = 1.0 / duration
    f_min, f_max = 20.0, 400.0
    f = np.arange(f_min, f_max, df)
    psd = aligo_like_psd(f)

    Lambda_true_values = [0.0, 0.1, 0.5]
    Lambda_grid = np.linspace(-2.0, 2.0, 81)

    family_names = list(WAVEFORM_FAMILIES.keys())
    fn_A, fn_B = WAVEFORM_FAMILIES[family_names[0]], WAVEFORM_FAMILIES[family_names[1]]

    combos = [("A", "A", fn_A, fn_A), ("A", "B", fn_A, fn_B),
              ("B", "A", fn_B, fn_A), ("B", "B", fn_B, fn_B)]

    matrix = {combo[:2]: [] for combo in combos}

    for Lambda_true in Lambda_true_values:
        print(f"  Lambda_true = {Lambda_true}")
        for inj_label, rec_label, inject_fn, recover_fn in combos:
            mean_ml, scatter = run_recovery_single(
                m1, m2, K_z, distance_Mpc, f, psd, df, Lambda_grid,
                inject_fn, recover_fn, Lambda_true,
                seed_base=hash((Lambda_true, inj_label, rec_label)) % (2**32))
            bias = mean_ml - Lambda_true
            matrix[(inj_label, rec_label)].append(bias)
            print(f"    {inj_label}->{rec_label}: Lambda_ML={mean_ml:>8.4f}  "
                  f"bias={bias:>8.4f}  scatter={scatter:.4f}")
        print()

    print("  SUMMARY MATRIX (mean |bias| across tested Lambda_true values):")
    print()
    print(f"  {'Injection':>10}  {'Recovery':>10}  {'mean |bias|':>12}")
    print("  " + "-" * 38)
    for (inj, rec), biases in matrix.items():
        mean_abs_bias = np.mean(np.abs(biases))
        print(f"  {inj:>10}  {rec:>10}  {mean_abs_bias:>12.4f}")

    print()
    bb_bias = np.mean(np.abs(matrix[("B", "B")]))
    aa_bias = np.mean(np.abs(matrix[("A", "A")]))
    cross_bias = np.mean([np.mean(np.abs(matrix[("A", "B")])),
                           np.mean(np.abs(matrix[("B", "A")]))])

    print(f"  Same-family average (A->A, B->B): {np.mean([aa_bias, bb_bias]):.4f}")
    print(f"  Cross-family average (A->B, B->A): {cross_bias:.4f}")
    print()

    if bb_bias < 0.05 and aa_bias < 0.05:
        print("  FINDING: Both same-family controls (A->A AND B->B) show small")
        print("  bias. This confirms the large bias is SPECIFICALLY a cross-")
        print("  family effect, not a problem with either waveform family in")
        print("  isolation. The Lambda estimator itself is sound when the")
        print("  recovery template matches the injection's underlying model.")
    else:
        print("  FINDING: At least one same-family control (A->A or B->B) also")
        print("  shows non-trivial bias. This suggests the issue may not be")
        print("  purely a cross-family effect -- investigate the Lambda phase")
        print("  model's interaction with waveform structure more broadly.")

    return matrix


# ══════════════════════════════════════════════════════════════════════════
def main():
    print("#" * 72)
    print("# STAGE 5F — BIAS DECOMPOSITION")
    print("#" * 72)
    print()

    m1, m2 = 35.6, 30.6
    z = 0.09
    distance_Mpc = 440.0
    K_z = cosmological_K_factor(z)

    freq_results = run_5F1_frequency_dependence(m1, m2, K_z, distance_Mpc)
    matrix_results = run_5F2_full_matrix(m1, m2, K_z, distance_Mpc)

    # ── Plots ────────────────────────────────────────────────────────────
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    f_highs = [r["f_hi"] for r in freq_results]
    bias_AB = [r["bias_AB"] for r in freq_results]
    bias_BA = [r["bias_BA"] for r in freq_results]
    bias_AA = [r["bias_AA"] for r in freq_results]
    ax.plot(f_highs, bias_AA, "o-", color="steelblue", label="A->A control")
    ax.plot(f_highs, bias_AB, "o-", color="firebrick", label="A->B cross")
    ax.plot(f_highs, bias_BA, "o-", color="forestgreen", label="B->A cross")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Upper frequency cutoff [Hz]")
    ax.set_ylabel(r"$\Lambda$ bias (at $\Lambda_{\rm true}=0.1$)")
    ax.set_title("5F-1: Bias vs frequency band")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    combo_labels = ["A->A", "A->B", "B->A", "B->B"]
    mean_biases = [np.mean(np.abs(matrix_results[(c[0], c[2])]))
                   for c in [("A", "->", "A"), ("A", "->", "B"),
                             ("B", "->", "A"), ("B", "->", "B")]]
    colors = ["steelblue", "firebrick", "firebrick", "steelblue"]
    ax.bar(combo_labels, mean_biases, color=colors)
    ax.set_ylabel("Mean |bias| across tested Λ_true")
    ax.set_title("5F-2: Full waveform-family matrix")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Stage 5F: Bias decomposition", fontsize=11)
    plt.tight_layout()
    path = out / "stage5f_bias_decomposition.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
