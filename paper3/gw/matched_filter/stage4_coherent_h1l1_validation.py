"""
stage4_coherent_h1l1_validation.py
======================================

STAGE 4: joint H1+L1 coherent matched-filter Lambda inference, gated
behind its own off-source null trials and injection/recovery tests --
exactly the same discipline as Stage 3, now applied to the coherent
(multi-detector) estimator rather than H1 alone.

WHAT "COHERENT" MEANS HERE:

Lambda is a single, shared parameter across both detectors (it is a
property of the wave's propagation, not of either detector). The joint
log-likelihood is

    ln L_joint(Lambda) = ln L_H1(Lambda; tc) + ln L_L1(Lambda; tc + dt_H1L1)

where dt_H1L1 is the H1-to-L1 arrival time delay. We do NOT search over
sky location to determine dt_H1L1 from first principles (that requires
antenna-pattern functions and a full (RA, dec, polarization, inclination)
parameter set well beyond this proof-of-concept). Instead we FIX
dt_H1L1 to the published GW150914 value from the LIGO discovery paper
(Abbott et al. 2016, PRL 116, 061102): L1 lags H1 by approximately
6.9 ms. This is an explicit, documented simplification -- see
LIMITATIONS below.

WHAT waveform.py AND likelihood.py CONTRIBUTE UNCHANGED:

Neither file is modified. This script imports and reuses
`waveform_frequency_domain`, `cosmological_K_factor` from waveform.py,
and `noise_weighted_inner_product` from likelihood.py directly. The
joint likelihood is a thin wrapper summing two independent
single-detector log-likelihoods (each already validated in Stage 3),
NOT a redefinition of the underlying physics model.

FOUR MODES, same discipline as Stage 3:

  4A/4B — load H1+L1, verify GPS alignment, report the fixed dt_H1L1
          and per-detector optimal SNR (sanity checks only).

  4C — H1+L1 coherent OFF-SOURCE NULL TRIALS (no injected signal).
       Synchronized (same GPS window, correctly time-shifted) H1/L1
       off-source segments, joint likelihood grid search.
       Expectation: no consistent nonzero Lambda preference.

  4D — H1+L1 coherent INJECTION/RECOVERY.
       Same off-source segments as 4C, now with a known Lambda_true
       waveform injected coherently into BOTH streams (correctly
       time-shifted for L1). Expectation: Lambda_fit ~ Lambda_true.

  4E — GW150914 COHERENT ANALYSIS (exploratory only, gated behind
       4C+4D passing). Joint H1+L1 grid-search point estimate.

  4F — MINIMAL WAVEFORM-SYSTEMATICS CHECK.
       Repeats 4E with an alternative (1PN-extended) GR phase, defined
       LOCALLY in this script (waveform.py is not modified), to see
       whether the Mode 4E point estimate is stable against a simple,
       well-documented next-order PN correction. This is a MINIMAL
       systematics check (two waveform prescriptions), not a
       comprehensive systematics budget -- see LIMITATIONS.

LIMITATIONS (read before interpreting any Mode 4E/4F number):

  - dt_H1L1 is FIXED to the published value, not fit for coherence.
    A genuine coherent search would marginalize over sky location.
  - No antenna-pattern-based relative amplitude/polarization treatment
    between H1 and L1 -- both detectors currently use the same
    `distance_Mpc` normalization, which is a simplification.
  - Leading-order (0PN) baseline in Mode 4E; only a single alternative
    (1PN) tested in Mode 4F, not a full waveform-systematics ensemble.
  - No Bayesian priors, no glitch vetoing, no calibration-uncertainty
    propagation -- same limitations as Stage 3 Mode D, now doubled
    across two detectors.

Usage:
    python stage4_coherent_h1l1_validation.py --h1 <H1.hdf5> --l1 <L1.hdf5> --event GW150914
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from waveform import waveform_frequency_domain, cosmological_K_factor, C_SI
from likelihood import noise_weighted_inner_product
from stage3_real_strain_validation import (
    load_real_segment, to_frequency_domain, estimate_psd_from_segment,
    EVENT_CATALOG,
)


# Published H1->L1 arrival time delay for GW150914 (Abbott et al. 2016,
# PRL 116, 061102): L1 lags H1 by ~6.9 ms. FIXED, not fit -- see module
# docstring LIMITATIONS.
DT_H1L1_GW150914 = 6.9e-3  # seconds, L1 arrives AFTER H1


# ══════════════════════════════════════════════════════════════════════════
# Joint (coherent) likelihood -- thin wrapper, waveform.py/likelihood.py
# themselves are untouched
# ══════════════════════════════════════════════════════════════════════════

def log_likelihood_single(data_fd, f, psd, df, m1, m2, Lambda, K_z,
                           tc=0.0, phi_c=0.0, distance_Mpc=440.0,
                           waveform_fn=waveform_frequency_domain):
    """Identical structure to likelihood.log_likelihood, but accepts a
    pluggable waveform_fn so Mode 4F can substitute the 1PN-extended
    model without touching waveform.py."""
    h = waveform_fn(f, m1, m2, Lambda, K_z, tc, phi_c, distance_Mpc)
    dh = noise_weighted_inner_product(data_fd, h, f, psd, df)
    hh = noise_weighted_inner_product(h, h, f, psd, df)
    return dh - 0.5 * hh


def log_likelihood_joint(data_H1, f_H1, psd_H1, df_H1,
                          data_L1, f_L1, psd_L1, df_L1,
                          m1, m2, Lambda, K_z, dt_H1L1,
                          tc=0.0, phi_c=0.0, distance_Mpc=440.0,
                          waveform_fn=waveform_frequency_domain):
    """ln L_joint(Lambda) = ln L_H1(Lambda; tc) + ln L_L1(Lambda; tc+dt_H1L1)"""
    logL_H1 = log_likelihood_single(data_H1, f_H1, psd_H1, df_H1, m1, m2,
                                     Lambda, K_z, tc=tc, phi_c=phi_c,
                                     distance_Mpc=distance_Mpc,
                                     waveform_fn=waveform_fn)
    logL_L1 = log_likelihood_single(data_L1, f_L1, psd_L1, df_L1, m1, m2,
                                     Lambda, K_z, tc=tc + dt_H1L1, phi_c=phi_c,
                                     distance_Mpc=distance_Mpc,
                                     waveform_fn=waveform_fn)
    return logL_H1 + logL_L1


def grid_search_lambda_joint(data_H1, f_H1, psd_H1, df_H1,
                              data_L1, f_L1, psd_L1, df_L1,
                              m1, m2, K_z, dt_H1L1, Lambda_grid,
                              tc=0.0, phi_c=0.0, distance_Mpc=440.0,
                              waveform_fn=waveform_frequency_domain):
    """Same structure as likelihood.grid_search_lambda, joint version."""
    logL = np.array([
        log_likelihood_joint(data_H1, f_H1, psd_H1, df_H1,
                              data_L1, f_L1, psd_L1, df_L1,
                              m1, m2, Lam, K_z, dt_H1L1, tc, phi_c,
                              distance_Mpc, waveform_fn)
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


# ══════════════════════════════════════════════════════════════════════════
# Mode 4F: minimal 1PN-extended waveform, defined locally (waveform.py
# is NOT modified)
# ══════════════════════════════════════════════════════════════════════════

def waveform_1PN_extended(f, m1_msun, m2_msun, Lambda, K_z,
                           tc=0.0, phi_c=0.0, distance_Mpc=440.0):
    """
    Leading-order amplitude (unchanged from waveform.py) plus the
    standard 1PN phase correction added to the TaylorF2 stationary
    phase, on top of the SAME Lambda dispersion term used everywhere
    else in this repository. This is the well-known, textbook 1PN
    coefficient (e.g. Blanchet 2014 Living Reviews, Eq. for TaylorF2):

        Psi(f) = Psi_0PN(f) * [1 + (3715/756 + 55/9 * eta) * v^2 / ... ]

    implemented directly here as an ADDITIVE term to keep the function
    self-contained; this is a MINIMAL systematics probe, not a
    production-grade 1PN implementation.
    """
    from waveform import chirp_mass_SI, symmetric_mass_ratio, MSUN_SI, MPC_SI

    Mc = chirp_mass_SI(m1_msun, m2_msun)
    eta = symmetric_mass_ratio(m1_msun, m2_msun)
    D = distance_Mpc * MPC_SI
    G_SI = 6.674e-11

    f = np.asarray(f, dtype=float)
    f_safe = np.maximum(f, 1e-6)

    A = np.sqrt(5.0 / 24.0) * np.pi ** (-2.0 / 3.0) * \
        (G_SI * Mc / C_SI ** 3) ** (5.0 / 6.0) * C_SI / D * f_safe ** (-7.0 / 6.0)

    x = (np.pi * G_SI * Mc / C_SI ** 3 * f_safe) ** (1.0 / 3.0)
    # 0PN term (identical to waveform.py)
    Psi_0PN = (3.0 / 128.0) * (np.pi * G_SI * Mc / C_SI ** 3 * f_safe) ** (-5.0 / 3.0)
    # Standard 1PN fractional correction (textbook TaylorF2 coefficient),
    # expressed via the total-mass-dependent PN expansion parameter v:
    m_total_SI = (m1_msun + m2_msun) * MSUN_SI
    v = (np.pi * G_SI * m_total_SI / C_SI ** 3 * f_safe) ** (1.0 / 3.0)
    pn1_fraction = (3715.0 / 756.0 + 55.0 / 9.0 * eta) * v ** 2

    Psi_GR = 2 * np.pi * f_safe * tc - phi_c - np.pi / 4 + Psi_0PN * (1 + pn1_fraction)

    Delta_Psi = -(4.0 * np.pi ** 3 * Lambda * K_z / C_SI ** 3) * f_safe ** 3
    Psi_total = Psi_GR + Delta_Psi

    return A * np.exp(1j * Psi_total)


# ══════════════════════════════════════════════════════════════════════════
# Loading and alignment helpers
# ══════════════════════════════════════════════════════════════════════════

def load_and_condition(path, gps_center, half_window, f_min, f_max, duration):
    strain_td, fs, seg_start = load_real_segment(path, gps_center, half_window)
    f, strain_fd, df = to_frequency_domain(strain_td, fs, f_min, f_max, duration)
    psd = estimate_psd_from_segment(strain_td, fs, f)
    return strain_td, fs, seg_start, f, strain_fd, df, psd


# ══════════════════════════════════════════════════════════════════════════
def run_mode_4AB(h1_path, l1_path, event_name, f_min, f_max, duration):
    print("=" * 72)
    print("MODE 4A/4B — LOAD H1+L1, VERIFY ALIGNMENT, SANITY CHECKS")
    print("=" * 72)
    print()

    ev = EVENT_CATALOG[event_name]
    gps_merger = ev["gps_merger"]

    _, fs_h1, start_h1, f_h1, fd_h1, df_h1, psd_h1 = load_and_condition(
        h1_path, gps_merger, duration / 2 + 2, f_min, f_max, duration)
    _, fs_l1, start_l1, f_l1, fd_l1, df_l1, psd_l1 = load_and_condition(
        l1_path, gps_merger, duration / 2 + 2, f_min, f_max, duration)

    print(f"  H1: fs={fs_h1} Hz, segment GPS start={start_h1:.3f}")
    print(f"  L1: fs={fs_l1} Hz, segment GPS start={start_l1:.3f}")
    print(f"  Fixed dt_H1L1 = {DT_H1L1_GW150914*1e3:.1f} ms "
          f"(published GW150914 value, NOT fit for coherence -- see LIMITATIONS)")
    print()

    if abs(fs_h1 - fs_l1) > 1e-6:
        print("  WARNING: H1 and L1 sample rates differ -- frequency grids may")
        print("  not align correctly. Proceeding, but verify results carefully.")
    if not np.allclose(f_h1, f_l1):
        print("  WARNING: H1 and L1 frequency grids are not identical.")
    else:
        print("  Frequency grids match between H1 and L1. OK.")
    print()

    return ev, f_h1, df_h1


# ══════════════════════════════════════════════════════════════════════════
def run_mode_4C(h1_path, l1_path, gps_avoid, m1, m2, K_z, distance_Mpc,
                 f_min, f_max, duration, n_trials=200,
                 grid_widths=(5.0, 10.0, 20.0), grid_points_per_unit=100):
    print("=" * 72)
    print("MODE 4C — H1+L1 COHERENT OFF-SOURCE NULL TRIALS")
    print("=" * 72)
    print()

    half_window = duration * n_trials / 2 + 60
    block_h1, fs_h1, start_h1 = load_real_segment(h1_path, gps_avoid, half_window)
    block_l1, fs_l1, start_l1 = load_real_segment(l1_path, gps_avoid, half_window)

    seg_samples = int(duration * fs_h1)
    max_start = min(len(block_h1), len(block_l1)) - seg_samples

    print(f"  Off-source blocks: H1 {len(block_h1)/fs_h1:.1f}s, "
          f"L1 {len(block_l1)/fs_l1:.1f}s, {n_trials} trials")
    print()

    rng = np.random.default_rng(1)
    if max_start > n_trials:
        starts = rng.choice(max_start, size=n_trials, replace=False)
    else:
        starts = np.linspace(0, max_start, n_trials, dtype=int)

    # Pre-condition each trial (H1 and L1 use the SAME start index, i.e.
    # synchronized GPS windows -- the dt_H1L1 shift is applied inside the
    # joint likelihood via tc offset, not by desynchronizing the windows)
    trial_data = []
    for start in starts:
        sub_h1 = block_h1[start:start + seg_samples]
        sub_l1 = block_l1[start:start + seg_samples]

        f_h1, fd_h1, df_h1 = to_frequency_domain(sub_h1, fs_h1, f_min, f_max, duration)
        psd_h1 = estimate_psd_from_segment(sub_h1, fs_h1, f_h1)
        f_l1, fd_l1, df_l1 = to_frequency_domain(sub_l1, fs_l1, f_min, f_max, duration)
        psd_l1 = estimate_psd_from_segment(sub_l1, fs_l1, f_l1)

        trial_data.append((fd_h1, f_h1, psd_h1, df_h1, fd_l1, f_l1, psd_l1, df_l1))

    null_estimates = None
    final_grid = None
    c_pass = False

    for width in grid_widths:
        Lambda_grid = np.linspace(-width, width,
                                    int(2 * width * grid_points_per_unit) + 1)
        estimates = []
        for (fd_h1, f_h1, psd_h1, df_h1, fd_l1, f_l1, psd_l1, df_l1) in trial_data:
            _, _, Lam_ml, _ = grid_search_lambda_joint(
                fd_h1, f_h1, psd_h1, df_h1, fd_l1, f_l1, psd_l1, df_l1,
                m1, m2, K_z, DT_H1L1_GW150914, Lambda_grid,
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
        elif not boundary_pass and width < max(grid_widths):
            print(f"  Boundary hits still too high -- widening grid...")
            print()

    print(f"  MODE 4C OVERALL: {'PASS' if c_pass else 'FAIL'}")
    print()

    return null_estimates, c_pass, final_grid, trial_data


# ══════════════════════════════════════════════════════════════════════════
def run_mode_4D(trial_data, m1, m2, K_z, distance_Mpc, Lambda_grid,
                 n_realizations=10):
    print("=" * 72)
    print("MODE 4D — H1+L1 COHERENT INJECTION/RECOVERY")
    print("=" * 72)
    print()
    print(f"  Reusing the same {len(trial_data)} off-source segments as Mode 4C")
    print(f"  (using first {n_realizations} of them), now with a coherent")
    print(f"  Lambda_true injection added to both H1 and L1 (L1 time-shifted")
    print(f"  by dt_H1L1={DT_H1L1_GW150914*1e3:.1f} ms).")
    print()

    Lambda_true_values = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]

    print(f"  {'Lambda_true':>12}  {'Lambda_ML (mean)':>18}  "
          f"{'scatter (std)':>14}  {'sigma_from_true':>16}")
    print("  " + "-" * 68)

    results = []
    for Lam_true in Lambda_true_values:
        ml_estimates = []
        for (fd_h1, f_h1, psd_h1, df_h1, fd_l1, f_l1, psd_l1, df_l1) in \
                trial_data[:n_realizations]:

            h_h1 = waveform_frequency_domain(f_h1, m1, m2, Lam_true, K_z,
                                              tc=0.0, distance_Mpc=distance_Mpc)
            h_l1 = waveform_frequency_domain(f_l1, m1, m2, Lam_true, K_z,
                                              tc=DT_H1L1_GW150914,
                                              distance_Mpc=distance_Mpc)
            data_h1 = fd_h1 + h_h1
            data_l1 = fd_l1 + h_l1

            _, _, Lam_ml, _ = grid_search_lambda_joint(
                data_h1, f_h1, psd_h1, df_h1, data_l1, f_l1, psd_l1, df_l1,
                m1, m2, K_z, DT_H1L1_GW150914, Lambda_grid,
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
    d_pass = all(
        abs(r["mean_ml"] - r["Lambda_true"]) < 2 * r["scatter"]
        if r["scatter"] > 0 else abs(r["mean_ml"] - r["Lambda_true"]) < 0.05
        for r in results
    )
    print(f"  MODE 4D: {'PASS' if d_pass else 'FAIL'}")
    print()

    return results, d_pass


# ══════════════════════════════════════════════════════════════════════════
def run_mode_4E(h1_path, l1_path, event_name, f_min, f_max, duration, Lambda_grid):
    print("=" * 72)
    print(f"MODE 4E — {event_name} COHERENT H1+L1 ANALYSIS (EXPLORATORY ONLY)")
    print("=" * 72)
    print()
    print("  ┌────────────────────────────────────────────────────────┐")
    print("  │  WHATEVER NUMBER FOLLOWS IS EXPLORATORY, NOT A RESULT.  │")
    print("  │  Fixed dt_H1L1 (not fit). No antenna pattern. No sky    │")
    print("  │  location marginalization. Leading-order waveform.      │")
    print("  └────────────────────────────────────────────────────────┘")
    print()

    ev = EVENT_CATALOG[event_name]
    K_z = cosmological_K_factor(ev["z"])
    gps_merger = ev["gps_merger"]

    _, _, _, f_h1, fd_h1, df_h1, psd_h1 = load_and_condition(
        h1_path, gps_merger, duration / 2 + 2, f_min, f_max, duration)
    _, _, _, f_l1, fd_l1, df_l1, psd_l1 = load_and_condition(
        l1_path, gps_merger, duration / 2 + 2, f_min, f_max, duration)

    Lgrid, logL, Lam_ml, Lam_err = grid_search_lambda_joint(
        fd_h1, f_h1, psd_h1, df_h1, fd_l1, f_l1, psd_l1, df_l1,
        ev["m1"], ev["m2"], K_z, DT_H1L1_GW150914, Lambda_grid,
        distance_Mpc=ev["distance_Mpc"])

    print(f"  Joint H1+L1 grid-search maximum-likelihood Lambda: {Lam_ml:.4f}")
    print(f"  Fisher-curvature error estimate: {Lam_err:.4f}" if np.isfinite(Lam_err)
          else "  Fisher-curvature error estimate: undefined")
    print()

    return Lam_ml, Lam_err, Lgrid, logL, f_h1, fd_h1, psd_h1, df_h1, \
        f_l1, fd_l1, psd_l1, df_l1, ev, K_z


# ══════════════════════════════════════════════════════════════════════════
def run_mode_4F(f_h1, fd_h1, psd_h1, df_h1, f_l1, fd_l1, psd_l1, df_l1,
                 ev, K_z, Lambda_grid):
    print("=" * 72)
    print("MODE 4F — MINIMAL WAVEFORM-SYSTEMATICS CHECK (0PN vs 1PN)")
    print("=" * 72)
    print()
    print("  Repeating Mode 4E with a 1PN-extended phase (defined locally in")
    print("  this script, waveform.py unchanged) to test point-estimate")
    print("  stability against the simplest next-order PN correction.")
    print()

    Lgrid_1PN, logL_1PN, Lam_ml_1PN, Lam_err_1PN = grid_search_lambda_joint(
        fd_h1, f_h1, psd_h1, df_h1, fd_l1, f_l1, psd_l1, df_l1,
        ev["m1"], ev["m2"], K_z, DT_H1L1_GW150914, Lambda_grid,
        distance_Mpc=ev["distance_Mpc"], waveform_fn=waveform_1PN_extended)

    print(f"  0PN (Mode 4E baseline) vs 1PN (this mode):")
    print(f"    Lambda_ML difference indicates SYSTEMATIC (not statistical)")
    print(f"    uncertainty from waveform choice alone.")
    print()
    print(f"  1PN Lambda_ML: {Lam_ml_1PN:.4f}  "
          f"(Fisher err: {Lam_err_1PN:.4f})" if np.isfinite(Lam_err_1PN)
          else f"  1PN Lambda_ML: {Lam_ml_1PN:.4f}  (Fisher err: undefined)")
    print()
    print("  NOTE: this is a MINIMAL two-waveform check, not a comprehensive")
    print("  systematics budget. A full treatment requires a wider ensemble")
    print("  of independently-validated waveform models (e.g. via")
    print("  lalsimulation), well beyond this proof-of-concept.")

    return Lam_ml_1PN, Lam_err_1PN, Lgrid_1PN, logL_1PN


# ══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h1", required=True)
    parser.add_argument("--l1", required=True)
    parser.add_argument("--event", default="GW150914")
    parser.add_argument("--skip-E", action="store_true",
                         help="Skip Modes 4E/4F even if 4C/4D pass")
    args = parser.parse_args()

    ev = EVENT_CATALOG[args.event]
    m1, m2, distance_Mpc, z = ev["m1"], ev["m2"], ev["distance_Mpc"], ev["z"]
    K_z = cosmological_K_factor(z)
    gps_merger = ev["gps_merger"]

    f_min, f_max, duration = 20.0, 400.0, 8.0
    Lambda_grid_D = np.linspace(-5.0, 5.0, 1001)

    gps_offsource = gps_merger - 500.0

    print("#" * 72)
    print(f"# STAGE 4 — COHERENT H1+L1 VALIDATION — {args.event}")
    print("#" * 72)
    print()

    run_mode_4AB(args.h1, args.l1, args.event, f_min, f_max, duration)

    null_estimates, c_pass, Lambda_grid_C, trial_data = run_mode_4C(
        args.h1, args.l1, gps_offsource, m1, m2, K_z, distance_Mpc,
        f_min, f_max, duration)

    results_4D, d_pass = run_mode_4D(
        trial_data, m1, m2, K_z, distance_Mpc, Lambda_grid_D)

    print("=" * 72)
    print("GATE CHECK BEFORE MODE 4E")
    print("=" * 72)
    print()
    print(f"  Mode 4C (coherent null trials):        {'PASS' if c_pass else 'FAIL'}")
    print(f"  Mode 4D (coherent injection/recovery):  {'PASS' if d_pass else 'FAIL'}")
    print()

    mode4E_result = None
    mode4F_result = None

    if c_pass and d_pass and not args.skip_E:
        print("  Both gates passed. Proceeding to Mode 4E (exploratory) "
              "and Mode 4F (systematics).")
        print()
        (Lam_ml_E, Lam_err_E, Lgrid_E, logL_E, f_h1, fd_h1, psd_h1, df_h1,
         f_l1, fd_l1, psd_l1, df_l1, ev_, K_z_) = run_mode_4E(
            args.h1, args.l1, args.event, f_min, f_max, duration, Lambda_grid_C)
        mode4E_result = (Lam_ml_E, Lam_err_E)

        Lam_ml_F, Lam_err_F, Lgrid_F, logL_F = run_mode_4F(
            f_h1, fd_h1, psd_h1, df_h1, f_l1, fd_l1, psd_l1, df_l1,
            ev_, K_z_, Lambda_grid_C)
        mode4F_result = (Lam_ml_F, Lam_err_F)

        print("=" * 72)
        print("SYSTEMATICS SUMMARY")
        print("=" * 72)
        print()
        print(f"  0PN (Mode 4E): Lambda = {Lam_ml_E:.4f}")
        print(f"  1PN (Mode 4F): Lambda = {Lam_ml_F:.4f}")
        print(f"  Difference:    {abs(Lam_ml_E - Lam_ml_F):.4f}")
        print()
        print("  Both numbers remain EXPLORATORY. Neither is a physical result.")
    elif args.skip_E:
        print("  --skip-E specified: Modes 4E/4F not run.")
    else:
        print("  ┌────────────────────────────────────────────────────────┐")
        print("  │  GATE FAILED. Modes 4E/4F (GW150914 coherent) NOT run.  │")
        print("  │  Fix Mode 4C/4D issues before analyzing the real event. │")
        print("  └────────────────────────────────────────────────────────┘")

    # ── Plots ────────────────────────────────────────────────────────────
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)

    n_panels = 3 if mode4E_result else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 2:
        axes = list(axes) + [None]

    ax = axes[0]
    Lt = [r["Lambda_true"] for r in results_4D]
    Lml = [r["mean_ml"] for r in results_4D]
    Lsc = [r["scatter"] for r in results_4D]
    ax.errorbar(Lt, Lml, yerr=Lsc, fmt="o-", color="steelblue", capsize=4,
                markersize=8)
    lims = [min(Lt) - 0.2, max(Lt) + 0.2]
    ax.plot(lims, lims, "k--", lw=1.5, label="ideal")
    ax.set_xlabel(r"Injected $\Lambda_{\rm true}$")
    ax.set_ylabel(r"Recovered $\Lambda_{\rm ML}$")
    ax.set_title("Mode 4D: coherent H1+L1 injection/recovery")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(null_estimates, bins=15, color="gray", edgecolor="k", alpha=0.7)
    ax.axvline(0, color="firebrick", lw=1.5, ls="--")
    ax.set_xlabel(r"Recovered $\Lambda_{\rm ML}$")
    ax.set_ylabel("Count")
    ax.set_title("Mode 4C: coherent H1+L1 null trials")
    ax.grid(True, alpha=0.3)

    if mode4E_result:
        ax = axes[2]
        ax.plot(Lgrid_E, logL_E - np.max(logL_E), color="firebrick", lw=2,
                label="0PN (Mode 4E)")
        ax.plot(Lgrid_F, logL_F - np.max(logL_F), color="steelblue", lw=2,
                ls="--", label="1PN (Mode 4F)")
        ax.axvline(mode4E_result[0], color="firebrick", lw=1, ls=":")
        ax.axvline(mode4F_result[0], color="steelblue", lw=1, ls=":")
        ax.set_xlabel(r"$\Lambda$")
        ax.set_ylabel(r"$\ln L - \ln L_{\rm max}$")
        ax.set_title(f"Mode 4E/4F: {args.event} coherent (EXPLORATORY)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-3, 3)

    plt.suptitle(f"Stage 4: Coherent H1+L1 validation — {args.event}", fontsize=11)
    plt.tight_layout()
    path = out / f"stage4_{args.event}_coherent.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
