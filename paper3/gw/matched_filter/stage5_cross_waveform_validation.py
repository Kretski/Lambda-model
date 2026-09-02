"""
stage5_cross_waveform_validation.py
=======================================

STAGE 5: cross-waveform-family injection/recovery using INDEPENDENTLY
VALIDATED waveform models (via lalsimulation), following the plan
agreed after Stage 4 revealed a dominant PN-order systematic
(0PN=+6.57 vs 1PN=-1.27 on real GW150914 -- see STAGE4_FINAL_STATUS.md).

WHY CROSS-FAMILY, NOT JUST A BETTER SINGLE WAVEFORM:

Stage 4's hand-rolled 0PN/1PN comparison could not distinguish "the
Lambda estimator is unreliable" from "our own PN truncation is the
specific problem" -- both waveforms were derived by us, so a shared
derivation error could affect both identically. Using lalsimulation's
independently-implemented, NR-calibrated approximants (IMRPhenomD,
SEOBNRv4) removes that ambiguity: if Lambda_bias is small when
comparing two independently-developed, validated models, the Lambda
estimator itself is likely sound and residual systematics are
genuinely small. If Lambda_bias remains large even between two
validated models, the problem is more fundamental (e.g. in how the
Lambda phase term interacts with realistic merger-ringdown structure,
not captured by the leading-order phase-only test done so far).

MODES (mirroring the agreed Stage 5 plan):

  5A — lalsimulation availability check (see lalsim_waveform.py).

  5B — validated waveform generation sanity check: generate GR (Λ=0)
       waveforms with 2+ approximants, confirm they are finite,
       physically reasonable (correct chirp mass scaling, SNR in a
       sane range), and visibly different from each other (confirming
       they are genuinely independent implementations, not the same
       code twice).

  5C — Λ injection: for a grid of Λ_true values, generate GR+Λ
       waveforms using waveform family A.

  5D — CROSS-FAMILY RECOVERY:
         A -> A   (control: inject with A, recover with A)
         A -> B   (cross: inject with A, recover with B)
         B -> A   (cross: inject with B, recover with A)
       For each, report Λ_bias = Λ_recovered - Λ_true.

  5E — Quantify Λ_bias vs statistical error (Fisher curvature), for
       both the control and cross-family cases, and give an explicit
       verdict: is Λ_bias small compared to σ_stat (waveform choice is
       a minor systematic), or comparable/larger (waveform choice
       remains the dominant uncertainty, as found in Stage 4)?

This mirrors Stage 3/4's gating discipline: real events are NOT
touched in this script. Stage 5 uses synthetic injections into
colored Gaussian noise (matching the real-noise-PSD-shaped approach
of Stage 2), isolating the waveform-family-mismatch question from
real-strain non-Gaussianity, exactly as Stage 2 isolated real-PSD
robustness from real-strain robustness before Stage 3 introduced real
strain.
"""

import numpy as np
from pathlib import Path
import sys
import gc
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from waveform import cosmological_K_factor
from likelihood import aligo_like_psd, noise_weighted_inner_product
from lalsim_waveform import (
    waveform_lalsim_with_lambda, check_lalsimulation_available,
    LAL_AVAILABLE,
)


# ══════════════════════════════════════════════════════════════════════════
# Generic likelihood/grid-search machinery, parameterized by waveform_fn
# (mirrors likelihood.py and stage4's pluggable-waveform_fn pattern)
# ══════════════════════════════════════════════════════════════════════════

def phase_time_maximized_overlap(h1, h2, f, psd, df, f_min):
    """
    Match maximized over BOTH an unknown overall phase AND an unknown
    time shift between h1 and h2.

    IMPLEMENTATION NOTE (important): this delegates to PyCBC's
    `pycbc.filter.matchedfilter.match()`, a well-tested, standard
    implementation of exactly this operation, rather than a hand-rolled
    IFFT-based construction.

    WHY NOT HAND-ROLL THIS: an earlier version of this function
    attempted a custom IFFT-based time-shift maximization. Verification
    against a controlled synthetic test (two waveforms differing by a
    KNOWN 50ms time shift) revealed the hand-rolled version recovered
    the WRONG time shift even after fixing an initial sign-convention
    bug -- the remaining discrepancy was not resolved after multiple
    debugging attempts. Given the well-known subtlety of getting
    one-sided-spectrum, zero-padded IFFT matched-filter conventions
    exactly right, this repository uses PyCBC's extensively-tested
    implementation instead of an unverified custom one. This follows
    the same principle applied throughout this repository: do not
    present a "fix" as working without a passing verification test.

    Requires: pip install pycbc  (same WSL environment as lalsuite).
    """
    try:
        from pycbc.types import FrequencySeries
        from pycbc.filter import match as pycbc_match
    except ImportError:
        raise ImportError(
            "pycbc is required for phase_time_maximized_overlap(). "
            "Install via: pip install pycbc  (in the same environment "
            "as lalsuite). A hand-rolled IFFT-based alternative was "
            "attempted during development and FAILED a controlled "
            "verification test (see function docstring / STAGE5 dev "
            "notes) -- do not reimplement without a passing test "
            "against a known time-shift case first."
        )

    psd_series = FrequencySeries(psd, delta_f=df, epoch=0)
    h1_series = FrequencySeries(h1, delta_f=df, epoch=0)
    h2_series = FrequencySeries(h2, delta_f=df, epoch=0)

    m, index = pycbc_match(h1_series, h2_series, psd=psd_series,
                            low_frequency_cutoff=f_min)

    # NOTE: 'index' is the raw pycbc sample-index of the best-fit time
    # shift, in the convention of pycbc's internal implicit time series.
    # We deliberately do NOT convert this to seconds here: an earlier
    # hand-rolled attempt at this exact conversion (see git history /
    # dev notes) contained an unverified indexing/sign error. The match
    # VALUE (m) is the quantity actually used by the likelihood; the
    # raw index is returned only as an unconverted diagnostic.
    return float(m), int(index)


def log_likelihood_generic(data_fd, f, psd, df, m1, m2, Lambda, K_z,
                            waveform_fn, tc=0.0, phi_c=0.0,
                            distance_Mpc=440.0, **waveform_kwargs):
    """
    TIME-AND-PHASE-MAXIMIZED likelihood proxy: ln L ~ max_t|<d|h(t)>| - 0.5*<h|h>.

    WHY TIME-MAXIMIZATION, NOT JUST PHASE (see phase_time_maximized_overlap
    docstring for the full explanation): Mode 5B found that a simple
    constant-phase maximization (|<d|h>|) was NOT sufficient to recover
    a sensible match between IMRPhenomD and SEOBNRv4 (match=0.0067,
    should be >0.9) -- the two approximants differ in their internal
    coalescence-time placement within SimInspiralFD's output, which is
    a LINEAR-IN-f phase difference that only an explicit time-shift
    maximization can remove. This is standard practice in real GW
    matched-filter searches, used here for exactly the reason it
    exists: comparing independently-generated templates that do not
    share an a priori common time convention. Uses PyCBC's tested
    match() implementation -- see phase_time_maximized_overlap().
    """
    h = waveform_fn(f, m1, m2, Lambda, K_z, tc=tc, phi_c=phi_c,
                     distance_Mpc=distance_Mpc, **waveform_kwargs)
    match, _ = phase_time_maximized_overlap(data_fd, h, f, psd, df, f[0])
    norm_d = np.sqrt(noise_weighted_inner_product(data_fd, data_fd, f, psd, df))
    norm_h = np.sqrt(noise_weighted_inner_product(h, h, f, psd, df))
    dh_maximized = match * norm_d * norm_h  # undo match normalization
    hh = norm_h ** 2
    return dh_maximized - 0.5 * hh


def grid_search_lambda_generic(data_fd, f, psd, df, m1, m2, K_z,
                                Lambda_grid, waveform_fn, tc=0.0, phi_c=0.0,
                                distance_Mpc=440.0, **waveform_kwargs):
    logL = np.array([
        log_likelihood_generic(data_fd, f, psd, df, m1, m2, Lam, K_z,
                                waveform_fn, tc, phi_c, distance_Mpc,
                                **waveform_kwargs)
        for Lam in Lambda_grid
    ])
    i_max = np.argmax(logL)
    Lambda_ml = Lambda_grid[i_max]

    if 0 < i_max < len(Lambda_grid) - 1:
        dL = Lambda_grid[1] - Lambda_grid[0]
        d2logL = (logL[i_max + 1] - 2 * logL[i_max] + logL[i_max - 1]) / dL ** 2
        Lambda_err = np.sqrt(-1.0 / d2logL) if d2logL < 0 else np.nan
    else:
        Lambda_err = np.nan

    return Lambda_grid, logL, Lambda_ml, Lambda_err


def waveform_family_A(f, m1, m2, Lambda, K_z, tc=0.0, phi_c=0.0,
                       distance_Mpc=440.0):
    return waveform_lalsim_with_lambda(
        f, m1, m2, Lambda, K_z, tc=tc, phi_c=phi_c,
        distance_Mpc=distance_Mpc, approximant="IMRPhenomD")


def waveform_family_B(f, m1, m2, Lambda, K_z, tc=0.0, phi_c=0.0,
                       distance_Mpc=440.0):
    return waveform_lalsim_with_lambda(
        f, m1, m2, Lambda, K_z, tc=tc, phi_c=phi_c,
        distance_Mpc=distance_Mpc, approximant="SEOBNRv4")


WAVEFORM_FAMILIES = {"A_IMRPhenomD": waveform_family_A,
                      "B_SEOBNRv4": waveform_family_B}


# ══════════════════════════════════════════════════════════════════════════
def run_mode_5A():
    print("=" * 72)
    print("MODE 5A — LALSIMULATION AVAILABILITY CHECK")
    print("=" * 72)
    print()
    ok = check_lalsimulation_available()
    print()
    if not ok:
        print("  Stage 5 CANNOT proceed without a working lalsimulation")
        print("  installation. See module docstrings for setup notes.")
    return ok


def run_mode_5B(m1, m2, K_z, distance_Mpc, f, df):
    print("=" * 72)
    print("MODE 5B — VALIDATED WAVEFORM GENERATION SANITY CHECK")
    print("=" * 72)
    print()

    psd = aligo_like_psd(f)
    results = {}

    for name, wf_fn in WAVEFORM_FAMILIES.items():
        try:
            h = wf_fn(f, m1, m2, 0.0, K_z, distance_Mpc=distance_Mpc)
            hh = noise_weighted_inner_product(h, h, f, psd, df)
            snr = np.sqrt(max(hh, 0.0))
            finite = np.all(np.isfinite(h))
            results[name] = dict(h=h, snr=snr, finite=finite)
            print(f"  {name}: SNR={snr:.2f}, all finite={finite}, "
                  f"|h|_max={np.max(np.abs(h)):.3e}")
        except Exception as e:
            print(f"  {name}: FAILED -- {e}")
            results[name] = None

    print()

    if all(r is not None for r in results.values()):
        names = list(results.keys())
        h1, h2 = results[names[0]]["h"], results[names[1]]["h"]
        psd_local = aligo_like_psd(f)
        match, best_idx = phase_time_maximized_overlap(
            h1, h2, f, psd_local, df, f[0])
        print(f"  Time+phase-maximized match between {names[0]} and "
              f"{names[1]} (Lambda=0): {match:.4f}")
        print(f"  (Best-fit sample index: {best_idx}, raw pycbc convention;")
        print(f"  not converted to seconds -- match value is the")
        print(f"  quantity that matters for the likelihood.)")
        print(f"  (IFFT-based match, standard GW technique -- corrects for")
        print(f"  differing internal coalescence-time conventions between")
        print(f"  approximants, not just a constant phase offset. Expect")
        print(f"  high (>0.9 typically) but not exactly 1.)")

    print()
    b_pass = all(r is not None and r["finite"] and r["snr"] > 1
                 for r in results.values())

    match_pass = True
    if all(r is not None for r in results.values()):
        match_pass = match > 0.8
        if not match_pass:
            print()
            print("  ┌────────────────────────────────────────────────────────┐")
            print("  │  MATCH TOO LOW. The two waveform families do not        │")
            print("  │  represent the same physical source consistently --     │")
            print("  │  cross-family recovery (Mode 5C/5D) would measure an    │")
            print("  │  arbitrary phase/time convention offset, not genuine    │")
            print("  │  waveform-model systematics. STOPPING before 5C/5D.     │")
            print("  └────────────────────────────────────────────────────────┘")

    b_pass = b_pass and match_pass
    print(f"  MODE 5B: {'PASS' if b_pass else 'FAIL'}")
    print()
    return b_pass, results


def run_mode_5CD(m1, m2, K_z, distance_Mpc, f, psd, df,
                  Lambda_grid, n_realizations=2):
    print("=" * 72)
    print("MODE 5C/5D — Λ INJECTION AND CROSS-FAMILY RECOVERY")
    print("=" * 72)
    print()

    Lambda_true_values = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
    family_names = list(WAVEFORM_FAMILIES.keys())
    pairs = [(family_names[0], family_names[0], "A->A control"),
             (family_names[0], family_names[1], "A->B cross"),
             (family_names[1], family_names[0], "B->A cross")]

    all_results = {label: [] for _, _, label in pairs}

    for Lam_true in Lambda_true_values:
        print(f"  Lambda_true = {Lam_true}")
        for inject_name, recover_name, label in pairs:
            inject_fn = WAVEFORM_FAMILIES[inject_name]
            recover_fn = WAVEFORM_FAMILIES[recover_name]

            ml_estimates = []
            rng = np.random.default_rng(hash((Lam_true, label)) % (2**32))
            for real_idx in range(n_realizations):
                sigma = np.sqrt(psd / (4 * df))
                noise = rng.normal(0, sigma) + 1j * rng.normal(0, sigma)
                h_inject = inject_fn(f, m1, m2, Lam_true, K_z,
                                      distance_Mpc=distance_Mpc)
                data = h_inject + noise

                _, _, Lam_ml, _ = grid_search_lambda_generic(
                    data, f, psd, df, m1, m2, K_z, Lambda_grid, recover_fn,
                    distance_Mpc=distance_Mpc)
                ml_estimates.append(Lam_ml)
                gc.collect()  # pycbc FrequencySeries objects accumulate
                              # quickly across the ~thousands of match()
                              # calls in the Lambda grid search; explicit
                              # collection avoids memory pressure/swap

            ml_estimates = np.array(ml_estimates)
            mean_ml, scatter = np.mean(ml_estimates), np.std(ml_estimates)
            bias = mean_ml - Lam_true

            all_results[label].append(dict(
                Lambda_true=Lam_true, mean_ml=mean_ml, scatter=scatter,
                bias=bias))

            print(f"    {label:>15}: Lambda_ML={mean_ml:>8.4f}  "
                  f"scatter={scatter:>7.4f}  bias={bias:>8.4f}")
        print()

    return all_results, pairs


def run_mode_5E(all_results, pairs):
    print("=" * 72)
    print("MODE 5E — QUANTIFY BIAS VS STATISTICAL ERROR")
    print("=" * 72)
    print()

    print(f"  {'Comparison':>15}  {'mean|bias|':>12}  {'mean scatter':>14}  "
          f"{'bias/scatter':>14}  verdict")
    print("  " + "-" * 78)

    verdicts = {}
    for _, _, label in pairs:
        results = all_results[label]
        biases = np.array([abs(r["bias"]) for r in results])
        scatters = np.array([r["scatter"] for r in results if r["scatter"] > 0])

        mean_bias = np.mean(biases)
        mean_scatter = np.mean(scatters) if len(scatters) > 0 else np.nan
        ratio = mean_bias / mean_scatter if mean_scatter > 0 else np.nan

        if np.isnan(ratio):
            verdict = "insufficient scatter data"
        elif ratio < 1:
            verdict = "waveform bias < statistical error (GOOD)"
        elif ratio < 5:
            verdict = "waveform bias comparable to statistical error"
        else:
            verdict = "waveform bias DOMINATES (as in Stage 4)"

        verdicts[label] = (mean_bias, mean_scatter, ratio, verdict)
        print(f"  {label:>15}  {mean_bias:>12.4f}  {mean_scatter:>14.4f}  "
              f"{ratio:>14.2f}  {verdict}")

    print()

    control_label = pairs[0][2]
    cross_labels = [p[2] for p in pairs[1:]]

    control_ratio = verdicts[control_label][2]
    cross_ratios = [verdicts[label][2] for label in cross_labels]

    print("  SUMMARY:")
    print(f"    Control (A->A) bias/scatter: {control_ratio:.2f}")
    print(f"    Cross-family bias/scatter:   "
          f"{', '.join(f'{r:.2f}' for r in cross_ratios)}")
    print()

    if all(r < 5 for r in cross_ratios):
        print("  ┌────────────────────────────────────────────────────────┐")
        print("  │  Cross-family bias is NOT dominant. Unlike Stage 4's    │")
        print("  │  hand-rolled 0PN/1PN comparison, independently          │")
        print("  │  validated waveform families largely agree on Λ.       │")
        print("  │  This is evidence the Λ estimator itself is sound, and  │")
        print("  │  Stage 4's large swing was likely due to the specific   │")
        print("  │  hand-rolled PN truncation, not a fundamental problem   │")
        print("  │  with matched-filter Λ inference.                       │")
        print("  └────────────────────────────────────────────────────────┘")
    else:
        print("  ┌────────────────────────────────────────────────────────┐")
        print("  │  Cross-family bias REMAINS large. This indicates the    │")
        print("  │  Λ-phase-correction approach interacts with waveform    │")
        print("  │  structure (e.g. merger-ringdown) in a way not          │")
        print("  │  captured by the simple f^3 phase term, independent of  │")
        print("  │  which specific PN order or approximant is used.        │")
        print("  │  This is a more fundamental finding requiring further   │")
        print("  │  investigation before any real-event analysis.          │")
        print("  └────────────────────────────────────────────────────────┘")

    return verdicts


# ══════════════════════════════════════════════════════════════════════════
def main():
    print("#" * 72)
    print("# STAGE 5 — CROSS-WAVEFORM-FAMILY VALIDATION")
    print("#" * 72)
    print()

    if not run_mode_5A():
        print("Aborting Stage 5: lalsimulation not available.")
        return

    print()
    m1, m2 = 35.6, 30.6
    z = 0.09
    distance_Mpc = 440.0
    K_z = cosmological_K_factor(z)

    f_min, f_max, duration = 20.0, 400.0, 8.0
    df = 1.0 / duration
    f = np.arange(f_min, f_max, df)
    psd = aligo_like_psd(f)

    b_pass, _ = run_mode_5B(m1, m2, K_z, distance_Mpc, f, df)
    if not b_pass:
        print("Aborting Stage 5: waveform generation sanity check failed.")
        return

    Lambda_grid = np.linspace(-2.0, 2.0, 81)
    all_results, pairs = run_mode_5CD(
        m1, m2, K_z, distance_Mpc, f, psd, df, Lambda_grid)

    verdicts = run_mode_5E(all_results, pairs)

    # ── Plot ──────────────────────────────────────────────────────────────
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["steelblue", "firebrick", "forestgreen"]
    for (_, _, label), col in zip(pairs, colors):
        results = all_results[label]
        Lt = [r["Lambda_true"] for r in results]
        Lml = [r["mean_ml"] for r in results]
        Lsc = [r["scatter"] for r in results]
        ax.errorbar(Lt, Lml, yerr=Lsc, fmt="o-", color=col, capsize=4,
                    markersize=7, label=label)

    lims = [-0.2, 1.2]
    ax.plot(lims, lims, "k--", lw=1.5, label="ideal recovery")
    ax.set_xlabel(r"Injected $\Lambda_{\rm true}$")
    ax.set_ylabel(r"Recovered $\Lambda_{\rm ML}$")
    ax.set_title("Stage 5: cross-waveform-family Λ recovery\n"
                 "(IMRPhenomD vs SEOBNRv4, synthetic real-PSD-shaped noise)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out / "stage5_cross_waveform_recovery.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
