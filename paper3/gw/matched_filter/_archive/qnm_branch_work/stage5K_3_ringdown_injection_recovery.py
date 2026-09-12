"""
stage5K_3_ringdown_injection_recovery.py
===========================================
STAGE 5K.3 -- synthetic ringdown injection/recovery (noiseless).

Takes 2-3 VALIDATED points from stage5K_1_observables.csv (Stage
5K.1's output -- only rows with |Res| < 1e-8 from the actual solved
QNM branch, nothing invented), generates a clean damped-sinusoid
ringdown

    h(t) = A * exp(-t/tau) * cos(2*pi*f_R*t + phi),   t >= 0

for each chosen point, and recovers (f_R, tau) from the NOISELESS
time series via nonlinear least-squares fit of the same functional
form. This is the cleanest possible bridge test: can a recovery
pipeline extract the QNM parameters we validated, from a waveform
built out of those exact parameters, before any noise is involved?

Noise injection and an SNR sweep are explicitly Stage 5K.5, not here.
Keeping this step noiseless isolates one question only: is the
recovery method itself unbiased on a clean signal. If it isn't even
unbiased here, an SNR sweep on top of it would be meaningless.
"""

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
OBSERVABLES_CSV = HERE / "stage5K_1_observables.csv"

# Validated m's to inject (nearest available row is used for each)
TARGET_MS = [-14.0000, -14.5000, -14.7725]

A_INJECT = 1.0
PHI_INJECT = 0.3  # rad, arbitrary fixed phase
SAMPLE_RATE_HZ = 4096.0
N_TAU_DURATION = 8.0  # signal duration in units of tau


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

    print("=" * 100)
    print("STAGE 5K.3 -- SYNTHETIC RINGDOWN INJECTION / RECOVERY (noiseless)")
    print("=" * 100)
    print()

    results = []
    for m_target in TARGET_MS:
        row = nearest_row(rows, m_target)
        f_R_true, tau_true, m_actual = row["f_R"], row["tau"], row["m"]

        duration = N_TAU_DURATION * tau_true
        t = np.arange(0.0, duration, 1.0 / SAMPLE_RATE_HZ)
        h = ringdown(t, A_INJECT, tau_true, f_R_true, PHI_INJECT)

        # Recovery: nonlinear least-squares fit of the same functional
        # form. Seed deliberately off from the injected values, so a
        # good fit isn't just reporting the seed back.
        p0 = [A_INJECT * 0.8, tau_true * 1.2, f_R_true * 1.001, 0.0]
        try:
            popt, pcov = curve_fit(ringdown, t, h, p0=p0, maxfev=20000)
        except RuntimeError as exc:
            print(f"m={m_actual:+.4f}  RECOVERY FAILED: {exc}")
            continue

        A_rec, tau_rec, f_R_rec, phi_rec = popt
        # (A, phi) -> (-A, phi+pi) represents the identical waveform;
        # normalize sign of A for a meaningful comparison.
        if A_rec < 0:
            A_rec = -A_rec
            phi_rec = phi_rec + np.pi

        f_R_err = f_R_rec - f_R_true
        tau_err = tau_rec - tau_true

        print(f"m={m_actual:+.4f} (requested {m_target:+.4f})")
        print(f"    injected:  f_R={f_R_true:.6f} Hz  tau={tau_true:.6f} s")
        print(f"    recovered: f_R={f_R_rec:.6f} Hz  tau={tau_rec:.6f} s  "
              f"A={A_rec:.4f}  phi={phi_rec:.4f}")
        print(f"    error:     d(f_R)={f_R_err:+.3e} Hz  d(tau)={tau_err:+.3e} s  "
              f"(rel: {f_R_err / f_R_true:+.2e}, {tau_err / tau_true:+.2e})")
        print()

        results.append({
            "m": m_actual, "f_R_true": f_R_true, "tau_true": tau_true,
            "f_R_rec": f_R_rec, "tau_rec": tau_rec,
            "f_R_err_rel": f_R_err / f_R_true, "tau_err_rel": tau_err / tau_true,
        })

    print("=" * 100)
    if results:
        max_fR_err = max(abs(r["f_R_err_rel"]) for r in results)
        max_tau_err = max(abs(r["tau_err_rel"]) for r in results)
        print(f"max relative error across all points: "
              f"f_R={max_fR_err:.2e}, tau={max_tau_err:.2e}")
        print()
        print("This is a NOISELESS sanity check only: it tests whether the")
        print("recovery method (fitting the same functional form back to a")
        print("clean signal) is unbiased -- not whether the QNM is")
        print("detectable in realistic noise. That is Stage 5K.5.")
    else:
        print("No points recovered -- see failures above.")


if __name__ == "__main__":
    main()
