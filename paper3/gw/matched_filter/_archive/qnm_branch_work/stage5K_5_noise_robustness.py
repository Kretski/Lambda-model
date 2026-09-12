"""
stage5K_5_noise_robustness.py
================================
STAGE 5K.5 -- noise robustness of QNM ringdown parameter recovery.

For each of the same validated (f_R, tau) points used in Stage 5K.3,
injects white Gaussian noise at several target optimal matched-filter
SNRs, repeats recovery (nonlinear least-squares fit of the same
damped-sinusoid model) across many independent noise realizations per
SNR, and reports:

  - bias:    mean(recovered - true) over realizations
  - std:     scatter of recovered values over realizations
  - success rate: fraction of realizations where the fit converged to
    within a generous sanity bound of the true parameters (rather
    than diverging to a nonsense local minimum) -- a simple proxy for
    "detection efficiency" at this SNR, not a full matched-filter
    detection statistic

SNR convention: optimal (matched-filter) SNR = ||h||_2 / sigma, where
||h||_2 = sqrt(sum(h_i^2)) over the sampled signal and sigma is the
per-sample white-noise standard deviation. This is the simplest
well-defined convention for synthetic white noise; it is NOT the same
as detector strain-noise-PSD-weighted SNR used in real GW pipelines,
which requires a noise PSD model this stage deliberately does not
introduce yet.

Recovery seeding: p0 is the true parameters perturbed by a FIXED
random jitter (see JITTER_FRAC) drawn once per (m, SNR, trial) --
representing "we know roughly where to look from a coarse search",
not "we already know the answer". The same jitter distribution is
used at every SNR so results across SNR are comparable.
"""

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
OBSERVABLES_CSV = HERE / "stage5K_1_observables.csv"

TARGET_MS = [-14.0000, -14.5000, -14.7725]
SNR_LEVELS = [30, 20, 10, 7]
N_TRIALS = 30

A_INJECT = 1.0
PHI_INJECT = 0.3
SAMPLE_RATE_HZ = 4096.0
N_TAU_DURATION = 8.0

JITTER_FRAC = 0.05  # seed perturbation, as a fraction of the true value
SUCCESS_REL_TOL = 0.10  # a fit counts as "successful" if f_R and tau are
                        # both recovered within this relative tolerance

RNG_SEED = 20260912  # fixed for reproducibility


def ringdown(t, A, tau, f_R, phi):
    return A * np.exp(-t / tau) * np.cos(2.0 * np.pi * f_R * t + phi)


def load_observables():
    rows = []
    with open(OBSERVABLES_CSV, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({"m": float(row["m"]), "f_R": float(row["f_R_hz"]),
                         "tau": float(row["tau_s"])})
    return rows


def nearest_row(rows, m_target):
    return min(rows, key=lambda r: abs(r["m"] - m_target))


def main():
    if not OBSERVABLES_CSV.exists():
        print(f"ERROR: {OBSERVABLES_CSV} not found. Run "
              "stage5K_1_observable_extraction.py first.")
        sys.exit(1)

    rows = load_observables()
    rng = np.random.default_rng(RNG_SEED)

    print("=" * 100)
    print("STAGE 5K.5 -- NOISE ROBUSTNESS (SNR SWEEP)")
    print("=" * 100)
    print(f"SNR levels: {SNR_LEVELS}, trials per (m, SNR): {N_TRIALS}")
    print()

    summary_rows = []

    for m_target in TARGET_MS:
        row = nearest_row(rows, m_target)
        f_R_true, tau_true, m_actual = row["f_R"], row["tau"], row["m"]

        duration = N_TAU_DURATION * tau_true
        t = np.arange(0.0, duration, 1.0 / SAMPLE_RATE_HZ)
        h_clean = ringdown(t, A_INJECT, tau_true, f_R_true, PHI_INJECT)
        h_norm = np.sqrt(np.sum(h_clean ** 2))

        print(f"--- m={m_actual:+.4f}  f_R_true={f_R_true:.6f} Hz  "
              f"tau_true={tau_true:.6f} s ---")

        for snr in SNR_LEVELS:
            sigma = h_norm / snr
            f_R_recs, tau_recs, successes = [], [], []

            for trial in range(N_TRIALS):
                noise = rng.normal(0.0, sigma, size=t.shape)
                h_obs = h_clean + noise

                jitter = 1.0 + rng.uniform(-JITTER_FRAC, JITTER_FRAC, size=4)
                p0 = [A_INJECT * jitter[0], tau_true * jitter[1],
                      f_R_true * jitter[2], PHI_INJECT * jitter[3]]

                try:
                    popt, _ = curve_fit(ringdown, t, h_obs, p0=p0, maxfev=20000)
                except RuntimeError:
                    continue

                _, tau_rec, f_R_rec, _ = popt
                f_R_recs.append(f_R_rec)
                tau_recs.append(tau_rec)
                success = (abs(f_R_rec - f_R_true) / f_R_true < SUCCESS_REL_TOL and
                           abs(tau_rec - tau_true) / tau_true < SUCCESS_REL_TOL)
                successes.append(success)

            n_ok = len(f_R_recs)
            if n_ok == 0:
                print(f"  SNR={snr:>3}  ALL {N_TRIALS} FITS FAILED TO CONVERGE")
                summary_rows.append({"m": m_actual, "snr": snr, "n_converged": 0,
                                      "n_trials": N_TRIALS, "success_rate": 0.0,
                                      "f_R_bias": None, "f_R_std": None,
                                      "tau_bias": None, "tau_std": None})
                continue

            f_R_arr = np.array(f_R_recs)
            tau_arr = np.array(tau_recs)
            f_R_bias = float(np.mean(f_R_arr) - f_R_true)
            f_R_std = float(np.std(f_R_arr))
            tau_bias = float(np.mean(tau_arr) - tau_true)
            tau_std = float(np.std(tau_arr))
            success_rate = float(np.mean(successes))

            print(f"  SNR={snr:>3}  converged={n_ok}/{N_TRIALS}  "
                  f"success_rate={success_rate:.2f}  "
                  f"f_R: bias={f_R_bias:+.3e} Hz std={f_R_std:.3e} Hz  "
                  f"tau: bias={tau_bias:+.3e} s std={tau_std:.3e} s")

            summary_rows.append({"m": m_actual, "snr": snr, "n_converged": n_ok,
                                  "n_trials": N_TRIALS, "success_rate": success_rate,
                                  "f_R_bias": f_R_bias, "f_R_std": f_R_std,
                                  "tau_bias": tau_bias, "tau_std": tau_std})
        print()

    out_csv = HERE / "stage5K_5_noise_robustness_summary.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    print("=" * 100)
    print("HOW TO READ THIS:")
    print("  - success_rate near 1.0 at high SNR, dropping at low SNR, is the")
    print("    expected shape -- it shows detectability degrading gracefully")
    print("  - a std that grows as SNR drops (roughly like 1/SNR) is consistent")
    print("    with the expected statistical scaling of matched-filter parameter")
    print("    estimation")
    print("  - a persistent BIAS that does not shrink toward 0 as SNR increases")
    print("    would indicate a problem in the recovery method itself, not noise")
    print(f"written to {out_csv}")


if __name__ == "__main__":
    main()
