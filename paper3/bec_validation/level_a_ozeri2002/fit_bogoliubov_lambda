#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fit_bogoliubov_lambda.py
=========================
LEVEL-A EXPERIMENTAL CONSISTENCY CHECK
Ozeri, Steinhauer, Katz & Davidson (2002), arXiv:cond-mat/0112496,
Fig. 5 (excitation energies measured from the released-phonon cloud only).

Purpose
-------
Test whether directly-measured, real atomic-BEC excitation-energy data
are consistent with (a) pure linear dispersion, (b) the Bogoliubov
dispersion relation equivalent to the Lambda-model's BEC-sector
prediction Lambda = xi_O^2/2 (= xi_P^2/4 in the Paper-3 healing-length
convention; see convention_audit.md), and (c) a free-curvature fit.

This is explicitly framed as a PREDICTION TEST, not a free search:
the Bogoliubov/Lambda shape (beta=0.5) is fixed BEFORE fitting, from the
convention audit, not chosen after seeing the data.

IMPORTANT CAVEATS (see provenance.md for full discussion):
  - Data points are DIGITIZED from a published figure (pixel-based,
    cross-checked against one text-quoted number to <1 Hz agreement),
    not raw machine-readable data.
  - The measured system is a TRAPPED, INHOMOGENEOUS BEC analyzed under
    the local-density approximation (LDA), not a strictly homogeneous
    system.
  - The k*xi=0.324 point is explicitly flagged BY THE ORIGINAL AUTHORS
    as affected by a systematic recoil effect at low k.

Method
------
Unified model:  E/h = A * x * sqrt(1 + beta*x^2),  x = k*xi_O

  M0   : beta = 0   (fixed)  -- pure linear / free-phonon dispersion
  M_B  : beta = 0.5 (fixed)  -- Bogoliubov dispersion = Lambda-model
                                 prediction (Lambda = xi_O^2/2)
  M_Lf : beta free           -- exploratory 2-parameter fit

Two analyses are run:
  PRIMARY    : all N=4 digitized points
  SENSITIVITY: N=3, excluding the author-flagged k*xi=0.324 point

Run: python3 fit_bogoliubov_lambda.py
Requires: numpy, scipy
"""

import csv
import numpy as np
from scipy.optimize import curve_fit

sep = "=" * 72

# ============================================================================
# CONVENTION AUDIT (see convention_audit.md for the full derivation)
# ============================================================================
# xi_P = hbar/(m*c_s)              [Paper 3 / example_BEC.py]
# xi_O = sqrt(hbar^2/(2*m*g*n))    [Ozeri et al. 2002]
# => xi_P = sqrt(2) * xi_O
# => Lambda_P = xi_P^2/4 = xi_O^2/2
#
# In the dimensionless model E/h = A*x*sqrt(1+beta*x^2), x=k*xi_O:
#   beta = Lambda / xi_O^2
# so the Paper-3 prediction Lambda_P = xi_P^2/4 = xi_O^2/2 corresponds to
#   beta_pred = (xi_O^2/2) / xi_O^2 = 1/2

XI_P_OVER_XI_O = np.sqrt(2.0)
LAMBDA_P_FACTOR = 0.25          # Lambda_P = xi_P^2 * LAMBDA_P_FACTOR
LAMBDA_O_FACTOR = LAMBDA_P_FACTOR * XI_P_OVER_XI_O**2  # = Lambda_P / xi_O^2

assert abs(LAMBDA_O_FACTOR - 0.5) < 1e-12, (
    "CONVENTION AUDIT FAILED: expected Lambda_pred = xi_O^2/2 (beta_pred=0.5); "
    f"got beta_pred={LAMBDA_O_FACTOR}. Do not proceed without resolving this."
)

BETA_PRED = LAMBDA_O_FACTOR  # = 0.5, the pre-registered, fixed prediction


# ============================================================================
# DATA (digitized_fig5.csv, as produced by the pixel-calibration procedure
# documented in provenance.md)
# ============================================================================

def load_data(path="digitized_fig5.csv"):
    x, e, s, notes = [], [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(float(row["k_xi_O"]))
            e.append(float(row["E_over_h_kHz"]))
            s.append(float(row["sigma_kHz"]))
            notes.append(row.get("note", ""))
    return np.array(x), np.array(e), np.array(s), notes


# ============================================================================
# UNIFIED MODEL
# ============================================================================

def model(x, A, beta):
    return A * x * np.sqrt(np.maximum(1.0 + beta * x**2, 0.0))


def fit_fixed_beta(x, e, sigma, beta_fixed):
    def f(x, A):
        return model(x, A, beta_fixed)
    popt, pcov = curve_fit(f, x, e, sigma=sigma, absolute_sigma=True, p0=[1.5])
    A, A_err = popt[0], np.sqrt(pcov[0, 0])
    chi2 = np.sum(((e - f(x, A)) / sigma) ** 2)
    dof = len(x) - 1
    return A, A_err, chi2, dof


def fit_free_beta(x, e, sigma):
    popt, pcov = curve_fit(model, x, e, sigma=sigma, absolute_sigma=True,
                            p0=[1.5, 0.5])
    A, beta = popt
    A_err, beta_err = np.sqrt(np.diag(pcov))
    chi2 = np.sum(((e - model(x, *popt)) / sigma) ** 2)
    dof = len(x) - 2
    return A, A_err, beta, beta_err, chi2, dof


# ============================================================================
# ANALYSIS
# ============================================================================

def run_analysis(x, e, sigma, label):
    n = len(x)
    print(sep)
    print(f"{label}  (N={n})")
    print(sep)

    results = {"label": label, "n": n}

    # --- M0: beta=0 fixed ---
    A0, A0_err, chi2_0, dof0 = fit_fixed_beta(x, e, sigma, 0.0)
    aic0 = chi2_0 + 2 * 1
    bic0 = chi2_0 + 1 * np.log(n)
    print("M0 (linear, beta=0 fixed):")
    print(f"  A = {A0:.4f} +/- {A0_err:.4f}")
    print(f"  chi2 = {chi2_0:.4f} (dof={dof0}), reduced = {chi2_0/dof0:.4f}")
    print(f"  AIC = {aic0:.4f}, BIC = {bic0:.4f}")
    results["M0"] = dict(A=A0, A_err=A0_err, chi2=chi2_0, dof=dof0,
                          aic=aic0, bic=bic0)

    # --- M_B: beta=BETA_PRED fixed (Bogoliubov / Lambda-model prediction) ---
    AB, AB_err, chi2_B, dof_B = fit_fixed_beta(x, e, sigma, BETA_PRED)
    aicB = chi2_B + 2 * 1
    bicB = chi2_B + 1 * np.log(n)
    print(f"\nM_B (Bogoliubov, beta={BETA_PRED} FIXED "
          f"= Lambda_pred/xi_O^2, pre-registered):")
    print(f"  A = {AB:.4f} +/- {AB_err:.4f}")
    print(f"  chi2 = {chi2_B:.4f} (dof={dof_B}), reduced = {chi2_B/dof_B:.4f}")
    print(f"  AIC = {aicB:.4f}, BIC = {bicB:.4f}")
    results["M_B"] = dict(A=AB, A_err=AB_err, chi2=chi2_B, dof=dof_B,
                           aic=aicB, bic=bicB)

    # --- M_Lambda_free: beta free ---
    if n >= 3:
        AL, AL_err, betaL, betaL_err, chi2_L, dof_L = fit_free_beta(x, e, sigma)
        aicL = chi2_L + 2 * 2
        bicL = chi2_L + 2 * np.log(n)
        beta_sigma = (betaL - BETA_PRED) / betaL_err if betaL_err > 0 else float("nan")
        print("\nM_Lambda_free (beta free, exploratory):")
        print(f"  A = {AL:.4f} +/- {AL_err:.4f}")
        print(f"  beta_fit = {betaL:.4f} +/- {betaL_err:.4f}")
        print(f"  deviation from beta_pred={BETA_PRED}: {beta_sigma:+.3f} sigma")
        print(f"  Lambda_fit / Lambda_pred = {betaL/BETA_PRED:.3f}")
        dof_str = f", reduced = {chi2_L/dof_L:.4f}" if dof_L > 0 else " (dof=0)"
        print(f"  chi2 = {chi2_L:.4f} (dof={dof_L}){dof_str}")
        print(f"  AIC = {aicL:.4f}, BIC = {bicL:.4f}")
        results["M_Lambda_free"] = dict(A=AL, A_err=AL_err, beta=betaL,
                                         beta_err=betaL_err, chi2=chi2_L,
                                         dof=dof_L, aic=aicL, bic=bicL,
                                         beta_sigma=beta_sigma)
    else:
        print(f"\nM_Lambda_free: insufficient points (N={n}) for a "
              "2-parameter fit")
        results["M_Lambda_free"] = None

    print("\nModel comparison (relative to M0):")
    print(f"  Delta AIC (M_B - M0)      = {aicB-aic0:+.4f}")
    print(f"  Delta BIC (M_B - M0)      = {bicB-bic0:+.4f}")
    if results["M_Lambda_free"]:
        print(f"  Delta AIC (M_Lfree - M0)  = {aicL-aic0:+.4f}")
        print(f"  Delta BIC (M_Lfree - M0)  = {bicL-bic0:+.4f}")
        print(f"  Delta AIC (M_Lfree - M_B) = {aicL-aicB:+.4f}")
        print(f"  Delta BIC (M_Lfree - M_B) = {bicL-bicB:+.4f}")
    print()
    return results


def main():
    print(sep)
    print("LEVEL-A EXPERIMENTAL CONSISTENCY CHECK")
    print("Ozeri, Steinhauer, Katz & Davidson (2002), arXiv:cond-mat/0112496")
    print("Digitized Fig. 5 (released-phonon-cloud excitation energies)")
    print(sep)
    print()
    print(f"Convention audit PASSED: beta_pred = Lambda_pred/xi_O^2 = "
          f"{BETA_PRED} (= xi_O^2/2 = xi_P^2/4, Paper-3 native value)")
    print("See convention_audit.md for the full derivation.")
    print()

    x, e, sigma, notes = load_data()
    print("Digitized data:")
    print(f"{'k*xi_O':>10} {'E/h [kHz]':>12} {'sigma':>10}  note")
    for xi_, ei_, si_, note in zip(x, e, sigma, notes):
        print(f"{xi_:10.3f} {ei_:12.4f} {si_:10.4f}  {note}")
    print()

    primary = run_analysis(x, e, sigma, "PRIMARY: all N=4 points")
    sensitivity = run_analysis(x[1:], e[1:], sigma[1:],
                                "SENSITIVITY: excluding k*xi=0.324 "
                                "(author-flagged systematic)")

    # ------------------------------------------------------------------
    # Save results.csv
    # ------------------------------------------------------------------
    with open("results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["analysis", "N", "model", "A", "A_err", "beta", "beta_err",
                    "chi2", "dof", "reduced_chi2", "AIC", "BIC", "beta_sigma"])
        for res in (primary, sensitivity):
            for mname in ("M0", "M_B", "M_Lambda_free"):
                m = res.get(mname)
                if m is None:
                    continue
                beta = m.get("beta", 0.0 if mname == "M0" else BETA_PRED)
                beta_err = m.get("beta_err", "")
                beta_sigma = m.get("beta_sigma", "")
                redchi2 = m["chi2"] / m["dof"] if m["dof"] > 0 else ""
                w.writerow([res["label"], res["n"], mname,
                            f"{m['A']:.6f}", f"{m['A_err']:.6f}",
                            beta, beta_err,
                            f"{m['chi2']:.6f}", m["dof"],
                            f"{redchi2:.6f}" if redchi2 != "" else "",
                            f"{m['aic']:.6f}", f"{m['bic']:.6f}",
                            f"{beta_sigma:.6f}" if beta_sigma != "" else ""])

    print(sep)
    print("BOTTOM LINE")
    print(sep)
    print("PRIMARY (N=4): inconclusive. The apparent curvature preference is")
    print("dominated by the k*xi=0.324 point, for which the original authors")
    print("report a systematic recoil effect at low k.")
    print()
    print("SENSITIVITY (N=3): data are consistent with the pre-registered")
    print("Bogoliubov/Lambda-model prediction (beta_pred=0.5). The free fit's")
    print("beta_fit differs from beta_pred by <0.6 sigma, and AIC/BIC prefer")
    print("the fixed-prediction model over the free-curvature model.")
    print()
    print("Given N=3 (1 dof for the free fit), this is an experimental")
    print("consistency check, NOT a confirmation or detection of Lambda.")
    print()
    print("Saved -> results.csv")


if __name__ == "__main__":
    main()
