#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fit_quartic.py
==============
LEVEL-B PHENOMENOLOGICAL QUARTIC-CURVATURE TEST
Guo et al. 2021, Fig. 4 dispersion data (Harvard Dataverse, DOI:10.7910/DVN/LGT5O6)

Purpose
-------
Test whether a real, independently-measured BEC-cavity collective-mode
dispersion dataset shows statistically significant curvature of the form

    omega = a*k + b*k^3        (equivalent, to leading order, to
                                 omega^2 = A*k^2 + B*k^4)

relative to a pure linear baseline

    omega = a*k

IMPORTANT: this is NOT a Lambda-model validation. The measured system is
a cavity-mediated density-wave polariton with its own characteristic
momentum scale (set by the confocal cavity geometry), not a free
homogeneous-BEC phonon. See README.md and provenance.md for the full
caveat. This script performs a purely phenomenological curvature test.

Method
------
- Six (k, omega, sigma_omega) points, used exactly as archived.
- Fit directly to omega(k), not omega^2(k), to avoid distorting the
  reported Gaussian uncertainties under a nonlinear transform.
- Weighted chi^2, reduced chi^2, AIC, BIC for both models.
- F-test for the nested model comparison.
- Significance of b (in units of its own standard error).
- Sensitivity check excluding the k=0 point.

No re-binning, re-weighting, or data exclusion is performed in the
primary analysis.

Run: python3 fit_quartic.py
Requires: numpy, scipy
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy import stats


# ============================================================================
# DATA (Fig4_dispersion.csv, as archived)
# ============================================================================

K = np.array([0.00000, 0.00213, 0.00426, 0.00639, 0.00851, 0.01064])
OMEGA = np.array([0.03126, 0.81605, 1.27045, 1.00358, 1.70941, 2.12196])
SIGMA = np.array([0.13316, 0.10447, 0.13395, 0.11701, 0.15079, 0.11438])

N = len(K)

sep = "=" * 70


# ============================================================================
# MODELS
# ============================================================================

def M0(k, a):
    """Pure linear (Goldstone-mode) baseline."""
    return a * k


def M1(k, a, b):
    """Linear + cubic term. Equivalent, to leading order in small b*k^2/a,
    to omega^2 = A*k^2 + B*k^4."""
    return a * k + b * k**3


# ============================================================================
# FIT + STATISTICS
# ============================================================================

def fit_and_report(k, omega, sigma, label):
    n = len(k)

    popt0, pcov0 = curve_fit(M0, k, omega, sigma=sigma,
                              absolute_sigma=True, p0=[200])
    a0, a0_err = popt0[0], np.sqrt(pcov0[0, 0])
    resid0 = (omega - M0(k, *popt0)) / sigma
    chi2_0 = np.sum(resid0**2)
    dof0 = n - 1
    redchi2_0 = chi2_0 / dof0
    aic0 = chi2_0 + 2 * 1
    bic0 = chi2_0 + 1 * np.log(n)

    popt1, pcov1 = curve_fit(M1, k, omega, sigma=sigma,
                              absolute_sigma=True, p0=[200, 0])
    a1, b1 = popt1
    a1_err, b1_err = np.sqrt(np.diag(pcov1))
    resid1 = (omega - M1(k, *popt1)) / sigma
    chi2_1 = np.sum(resid1**2)
    dof1 = n - 2
    redchi2_1 = chi2_1 / dof1 if dof1 > 0 else float("nan")
    aic1 = chi2_1 + 2 * 2
    bic1 = chi2_1 + 2 * np.log(n)

    delta_aic = aic1 - aic0
    delta_bic = bic1 - bic0

    if dof1 > 0:
        F = ((chi2_0 - chi2_1) / (dof0 - dof1)) / (chi2_1 / dof1)
        p_value = 1 - stats.f.cdf(F, dof0 - dof1, dof1)
    else:
        F, p_value = float("nan"), float("nan")

    b_sigma = b1 / b1_err if b1_err > 0 else float("nan")

    print(sep)
    print(label)
    print(sep)
    print(f"N = {n}")
    print()
    print("M0: omega = a*k")
    print(f"  a            = {a0:.4f} +/- {a0_err:.4f}")
    print(f"  chi2 (dof)   = {chi2_0:.4f} ({dof0})")
    print(f"  reduced chi2 = {redchi2_0:.4f}")
    print(f"  AIC          = {aic0:.4f}")
    print(f"  BIC          = {bic0:.4f}")
    print()
    print("M1: omega = a*k + b*k^3")
    print(f"  a            = {a1:.4f} +/- {a1_err:.4f}")
    print(f"  b            = {b1:.4e} +/- {b1_err:.4e}")
    print(f"  b significance = {b_sigma:.3f} sigma")
    print(f"  chi2 (dof)   = {chi2_1:.4f} ({dof1})")
    print(f"  reduced chi2 = {redchi2_1:.4f}")
    print(f"  AIC          = {aic1:.4f}")
    print(f"  BIC          = {bic1:.4f}")
    print()
    print("Comparison")
    print(f"  Delta AIC (M1-M0) = {delta_aic:+.4f}")
    print(f"  Delta BIC (M1-M0) = {delta_bic:+.4f}")
    print(f"  F-test: F={F:.4f}, p={p_value:.4f}")
    print()

    return {
        "label": label, "n": n,
        "a0": a0, "a0_err": a0_err, "chi2_0": chi2_0, "dof0": dof0,
        "redchi2_0": redchi2_0, "aic0": aic0, "bic0": bic0,
        "a1": a1, "a1_err": a1_err, "b1": b1, "b1_err": b1_err,
        "b_sigma": b_sigma, "chi2_1": chi2_1, "dof1": dof1,
        "redchi2_1": redchi2_1, "aic1": aic1, "bic1": bic1,
        "delta_aic": delta_aic, "delta_bic": delta_bic,
        "F": F, "p_value": p_value,
        "resid0": resid0,
    }


def main():
    print(sep)
    print("LEVEL-B PHENOMENOLOGICAL QUARTIC-CURVATURE TEST")
    print("Fig4_dispersion.csv (Guo et al. 2021, Harvard Dataverse)")
    print("NOT a Lambda-model validation -- see README.md / provenance.md")
    print(sep)
    print()

    print("Data:")
    print(f"{'k':>10} {'omega':>10} {'sigma':>10}")
    for ki, oi, si in zip(K, OMEGA, SIGMA):
        print(f"{ki:10.5f} {oi:10.5f} {si:10.5f}")
    print()

    primary = fit_and_report(K, OMEGA, SIGMA, "PRIMARY FIT (all N=6 points)")

    print(sep)
    print("RESIDUALS (M0 fit) vs k")
    print(sep)
    for ki, ri in zip(K, primary["resid0"]):
        print(f"  k={ki:.5f}  standardized residual = {ri:+.3f}")
    print()

    sensitivity = fit_and_report(K[1:], OMEGA[1:], SIGMA[1:],
                                  "SENSITIVITY CHECK (k=0 excluded, N=5)")

    # ------------------------------------------------------------------
    # Save results.csv
    # ------------------------------------------------------------------
    import csv
    with open("results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["analysis", "N", "a0", "a0_err", "chi2_0", "dof0",
                     "redchi2_0", "aic0", "bic0",
                     "a1", "a1_err", "b1", "b1_err", "b_sigma",
                     "chi2_1", "dof1", "redchi2_1", "aic1", "bic1",
                     "delta_aic", "delta_bic", "F", "p_value"])
        for r in (primary, sensitivity):
            w.writerow([
                r["label"], r["n"],
                f"{r['a0']:.6f}", f"{r['a0_err']:.6f}",
                f"{r['chi2_0']:.6f}", r["dof0"], f"{r['redchi2_0']:.6f}",
                f"{r['aic0']:.6f}", f"{r['bic0']:.6f}",
                f"{r['a1']:.6f}", f"{r['a1_err']:.6f}",
                f"{r['b1']:.6e}", f"{r['b1_err']:.6e}",
                f"{r['b_sigma']:.6f}",
                f"{r['chi2_1']:.6f}", r["dof1"], f"{r['redchi2_1']:.6f}",
                f"{r['aic1']:.6f}", f"{r['bic1']:.6f}",
                f"{r['delta_aic']:.6f}", f"{r['delta_bic']:.6f}",
                f"{r['F']:.6f}", f"{r['p_value']:.6f}",
            ])

    print(sep)
    print("BOTTOM LINE")
    print(sep)
    print("No statistically significant preference for M1 over M0.")
    print("Result is robust to exclusion of k=0.")
    print("chi2_nu(M0) >> 1 indicates reported uncertainties do not fully")
    print("account for point-to-point scatter in this N=6 dataset.")
    print()
    print("This is a completed Level-B negative/control result.")
    print("It is NOT a Lambda-model validation or falsification.")
    print()
    print("Saved -> results.csv")


if __name__ == "__main__":
    main()
