"""
stage5K_5_noise_robustness_rootB.py
======================================
STAGE 5K.5 (Root B) -- noise robustness of QNM ringdown parameter
recovery, same protocol as the Root-A version (SNR 30/20/10/7, 30
trials each), applied to the Root-B branch, for an honest A/B
comparison between the two candidate resonances under identical
recovery methodology.
"""

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
OBSERVABLES_CSV = HERE / "stage5K_1_observables_rootB.csv"

TARGET_MS = [-14.0000, -14.5000, -14.9050]
SNR_LEVELS = [30, 20, 10, 7]
N_TRIALS = 30

A_INJECT = 1.0
PHI_INJECT = 0.3
SAMPLE_RATE_HZ = 4096.0
N_TAU_DURATION = 8.0

JITTER_FRAC = 0.05
SUCCESS_REL_TOL = 0.10
RNG_SEED = 20260912


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
              "stage5K_1_observable_extraction_rootB.py first.")
        sys.exit(1)

    rows = load_observables()
    rng = np.random.default_rng(RNG_SEED)

    print("=" * 100)
    print("STAGE 5K.5 (Root B) -- NOISE ROBUSTNESS (SNR SWEEP)")
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
                    popt, _ = curve_fit(
                        ringdown, t, h_obs, p0=p0, maxfev=20000,
                        bounds=([0.0, 1e-6, 0.0, -np.pi],
                                [np.inf, np.inf, np.inf, np.pi]))
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

    out_csv = HERE / "stage5K_5_noise_robustness_summary_rootB.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    print("=" * 100)
    print(f"written to {out_csv}")
    print("Compare against stage5K_5_noise_robustness_summary.csv (Root A) for")
    print("an A/B recoverability comparison between the two candidate resonances.")


if __name__ == "__main__":
    main()
