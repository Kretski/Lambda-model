#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Steinhauer et al. (2002) BEC spectrum residual test -- V3
=========================================================
PRL 88, 120407 (2002), arXiv:cond-mat/0111438

H0 : LDA Bogoliubov spectrum (Eq. 4), omega = hbar k^2 / (2 m S(k))
H1 : H0 + extra quartic term, gamma_total = (1 + g_extra) * hbar^2/(4 m^2)

New in V3 (relative to steinhauer_test.py):
  1. Gaussian prior on mu/h = 1.91 +/- 0.09 kHz (independently measured in
     the paper) added as an extra residual term in chi^2.
  2. Covariance matrix and mu--g_extra correlation from the fit Jacobian.
  3. Profile likelihood for g_extra (mu re-fitted at each g) -> 68%/95% CI.
  5. Optional low-k cut scan (--kmin-scan): repeats the fit for fixed k_min
     thresholds to check that g_extra does not depend on the low-k region.
  4. Null test: pseudo-experiments generated from H0 with the published mu
     (and a re-drawn mu "measurement" for the prior) -> empirical distribution
     of Delta chi^2 under H0 -> empirical p-value.

g_extra is a relative correction to the Bogoliubov k^4 coefficient.
It is NOT an estimate of the fundamental Lambda of the GW / strong-field
model: no theoretical bridge between the two exists.

Usage:
    python steinhauer_test_v3.py [path/to/data.csv] [--nnull 2000] [--kmin-scan]
CSV columns: k_um_inv, f_khz, sigma_khz
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist

HBAR = 1.054571817e-34
M_RB87 = 1.44316060e-25
GAMMA_STD = HBAR**2 / (4.0 * M_RB87**2)
C_EFF = 2.0e-3                  # m/s, published, only for unit bookkeeping

MU_PUB_KHZ = 1.91               # published time-averaged mu/h
MU_ERR_KHZ = 0.09
K_CLEAN_MAX = 6.8e6             # s-wave scattering visible above this (paper)

MU_BOUNDS = (0.2, 10.0)
G_BOUNDS = (-0.99, 100.0)


# ------------------------------------------------------------------ physics
def mu_joule(mu_khz):
    return mu_khz * 1e3 * 2.0 * np.pi * HBAR


def structure_factor_lda(k, mu):
    kin = HBAR**2 * k**2 / (2.0 * M_RB87)
    a = 2.0 * mu / kin
    sa = np.sqrt(a)
    t1 = (3.0 + a) / (4.0 * a**2)
    t2 = ((3.0 + 2.0 * a - a**2) / (16.0 * a**2.5)
          * (np.pi + 2.0 * np.arctan((a - 1.0) / (2.0 * sa))))
    return 15.0 / 4.0 * (t1 - t2)


def c_ld(k, mu):
    S = structure_factor_lda(k, mu)
    return HBAR * k / (2.0 * M_RB87) * np.sqrt(np.maximum(S**-2 - 1.0, 0.0))


def f_model(k, mu_khz, g=0.0):
    """Frequency in Hz."""
    mu = mu_joule(mu_khz)
    c = c_ld(k, mu)
    w2 = c**2 * k**2 + GAMMA_STD * (1.0 + g) * k**4
    return np.sqrt(np.maximum(w2, 0.0)) / (2.0 * np.pi)


# ------------------------------------------------------------------ fitting
def residuals(theta, k, f, s, model, mu_prior):
    mu_khz = theta[0]
    g = theta[1] if model == "H1" else 0.0
    r = (f_model(k, mu_khz, g) - f) / s
    if mu_prior is not None:
        r = np.append(r, (mu_khz - mu_prior) / MU_ERR_KHZ)
    return r


def fit(k, f, s, model, mu_prior=MU_PUB_KHZ, g_fixed=None):
    if model == "H0":
        x0, lo, hi = [MU_PUB_KHZ], [MU_BOUNDS[0]], [MU_BOUNDS[1]]
        fun = lambda th: residuals(th, k, f, s, "H0", mu_prior)
    elif g_fixed is not None:   # profile: H1 with g fixed, mu free
        x0, lo, hi = [MU_PUB_KHZ], [MU_BOUNDS[0]], [MU_BOUNDS[1]]
        fun = lambda th: residuals([th[0], g_fixed], k, f, s, "H1", mu_prior)
    else:
        x0 = [MU_PUB_KHZ, 0.0]
        lo, hi = [MU_BOUNDS[0], G_BOUNDS[0]], [MU_BOUNDS[1], G_BOUNDS[1]]
        fun = lambda th: residuals(th, k, f, s, "H1", mu_prior)
    res = least_squares(fun, x0, bounds=(lo, hi), x_scale="jac")
    return res, float(np.sum(res.fun**2))


def covariance(res):
    J = res.jac
    try:
        cov = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(J.T @ J)
    return cov


def profile_g(k, f, s, g_grid, mu_prior=MU_PUB_KHZ):
    return np.array([fit(k, f, s, "H1", mu_prior, g_fixed=g)[1] for g in g_grid])


def interval_from_profile(g_grid, prof, level):
    ok = g_grid[prof - prof.min() <= level]
    return (ok.min(), ok.max()) if ok.size else (np.nan, np.nan)


def null_test(k, s, nnull, seed, use_prior=True):
    """Pseudo-experiments from H0 at the published mu."""
    rng = np.random.default_rng(seed)
    f_true = f_model(k, MU_PUB_KHZ, 0.0)
    d = np.empty(nnull)
    g_hat = np.empty(nnull)
    for i in range(nnull):
        f_sim = f_true + rng.normal(0.0, s)
        prior = MU_PUB_KHZ + rng.normal(0.0, MU_ERR_KHZ) if use_prior else None
        _, c0 = fit(k, f_sim, s, "H0", prior)
        r1, c1 = fit(k, f_sim, s, "H1", prior)
        d[i] = c0 - c1
        g_hat[i] = r1.x[1]
    return d, g_hat


# ------------------------------------------------------------------ analysis
def analyse(k, f, s, label, nnull, seed):
    n = len(k)
    print("\n" + "=" * 72)
    print(f"{label}   (N = {n}, k = {k.min()/1e6:.2f}-{k.max()/1e6:.2f} um^-1)")
    print("=" * 72)

    out = {}
    for tag, prior in [("no prior", None), ("mu prior", MU_PUB_KHZ)]:
        r0, c0 = fit(k, f, s, "H0", prior)
        r1, c1 = fit(k, f, s, "H1", prior)
        cov = covariance(r1)
        sig = np.sqrt(np.diag(cov))
        rho = cov[0, 1] / (sig[0] * sig[1])
        dchi = c0 - c1
        print(f"\n[{tag}]")
        print(f"  H0: mu/h = {r0.x[0]:.4f} kHz                       chi2 = {c0:.3f}")
        print(f"  H1: mu/h = {r1.x[0]:.4f} +/- {sig[0]:.4f} kHz")
        print(f"      g_extra = {r1.x[1]:+.4f} +/- {sig[1]:.4f}          chi2 = {c1:.3f}")
        print(f"      corr(mu, g_extra) = {rho:+.3f}")
        print(f"  Delta chi2 = {dchi:.3f}   Delta AIC = {2 - dchi:+.3f}   "
              f"Delta BIC = {np.log(n) - dchi:+.3f}   "
              f"asymptotic p = {chi2_dist.sf(max(dchi, 0), 1):.4g}")
        out[tag] = dict(r0=r0, r1=r1, c0=c0, c1=c1, dchi=dchi, rho=rho)

    # profile likelihood (with prior)
    g_grid = np.linspace(-0.9, 2.0, 291)
    prof = profile_g(k, f, s, g_grid)
    ci68 = interval_from_profile(g_grid, prof, 1.0)
    ci95 = interval_from_profile(g_grid, prof, 3.84)
    print("\n[profile likelihood, mu prior]")
    print(f"  g_extra 68% CI: [{ci68[0]:+.3f}, {ci68[1]:+.3f}]")
    print(f"  g_extra 95% CI: [{ci95[0]:+.3f}, {ci95[1]:+.3f}]")
    print(f"  -> Lambda_extra 95% CI: [{ci95[0]*GAMMA_STD/C_EFF**2:+.2e}, "
          f"{ci95[1]*GAMMA_STD/C_EFF**2:+.2e}] m^2  (bookkeeping only)")

    # null test
    print(f"\n[null test: {nnull} pseudo-experiments from H0, mu = {MU_PUB_KHZ} kHz]")
    d_null, g_null = null_test(k, s, nnull, seed)
    d_obs = out["mu prior"]["dchi"]
    p_emp = (np.sum(d_null >= d_obs) + 1) / (nnull + 1)
    print(f"  null Delta chi2: median {np.median(d_null):.3f}, "
          f"95th pct {np.percentile(d_null, 95):.3f}, 99th pct {np.percentile(d_null, 99):.3f}")
    print(f"  null g_extra: median {np.median(g_null):+.4f}, "
          f"std {np.std(g_null):.4f}  (bias check: median should be ~0)")
    print(f"  observed Delta chi2 = {d_obs:.3f}  ->  empirical p = {p_emp:.4f}")

    # goodness of fit of H0 itself
    c0 = out["mu prior"]["c0"]
    p_gof = chi2_dist.sf(c0, n)   # N data + 1 prior - 1 parameter = N dof
    print(f"\n[H0 goodness of fit, mu prior]  chi2 = {c0:.2f}, dof = {n}, p = {p_gof:.4f}")

    pulls = (f - f_model(k, out["mu prior"]["r0"].x[0])) / s
    runs = 1 + np.sum(np.sign(pulls[1:]) != np.sign(pulls[:-1]))
    npos, nneg = np.sum(pulls > 0), np.sum(pulls < 0)
    exp_runs = 1 + 2 * npos * nneg / max(npos + nneg, 1)
    print(f"  pulls: {np.round(pulls, 2)}")
    print(f"  sign runs = {runs} (expected ~{exp_runs:.1f} for random residuals)")

    return dict(out=out, g_grid=g_grid, prof=prof, d_null=d_null,
                d_obs=d_obs, p_emp=p_emp, pulls=pulls)


def kmin_scan(k, f, s, kmins, nnull, seed):
    """Robustness scan: drop points below a fixed k threshold (not selected by pull)."""
    print("\n" + "=" * 72)
    print("LOW-k CUT SCAN (mu prior; thresholds fixed in k, not chosen by residuals)")
    print("=" * 72)
    print("k_min   N   chi2_H0/dof   p_gof    g_extra          dchi2   p_emp   g 95% CI")
    g_grid = np.linspace(-0.9, 2.0, 291)
    rows = []
    for j, kmin in enumerate(kmins):
        m = k >= kmin * 1e6
        n = int(m.sum())
        if n < 4:
            continue
        kk, ff, ss = k[m], f[m], s[m]
        r0, c0 = fit(kk, ff, ss, "H0")
        r1, c1 = fit(kk, ff, ss, "H1")
        sg = np.sqrt(np.diag(covariance(r1)))[1]
        lo, hi = interval_from_profile(g_grid, profile_g(kk, ff, ss, g_grid), 3.84)
        d_null, _ = null_test(kk, ss, nnull, seed + 100 + j)
        d = c0 - c1
        p_emp = (np.sum(d_null >= d) + 1) / (nnull + 1)
        p_gof = chi2_dist.sf(c0, n)
        print(f"{kmin:5.2f} {n:3d}   {c0:6.1f}/{n:<3d}    {p_gof:.4f}   {r1.x[1]:+.3f}+/-{sg:.3f}   "
              f"{d:5.2f}   {p_emp:.3f}   [{lo:+.2f}, {hi:+.2f}]")
        rows.append(dict(k_min=kmin, N=n, chi2_H0=c0, p_gof=p_gof, g_extra=r1.x[1],
                         g_err=sg, dchi2=d, p_emp=p_emp, g_lo95=lo, g_hi95=hi))
    return pd.DataFrame(rows)


def plot(k, f, s, res_all, res_clean, fname):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    kk = np.linspace(0.2e6, k.max() * 1.05, 400)
    a = ax[0, 0]
    a.errorbar(k / 1e6, f / 1e3, s / 1e3, fmt="o", ms=3, label="input data")
    a.plot(kk / 1e6, f_model(kk, MU_PUB_KHZ) / 1e3, label="H0, mu = 1.91 kHz")
    r1 = res_all["out"]["mu prior"]["r1"]
    a.plot(kk / 1e6, f_model(kk, *r1.x) / 1e3, "--", label="H1 fit")
    a.set_xlabel("k (um$^{-1}$)"); a.set_ylabel("f (kHz)"); a.legend()

    a = ax[0, 1]
    a.axhline(0, c="k", lw=0.5)
    a.plot(k / 1e6, res_all["pulls"], "o-")
    a.axvline(K_CLEAN_MAX / 1e6, c="gray", ls=":")
    a.set_xlabel("k (um$^{-1}$)"); a.set_ylabel("H0 pull (mu prior)")

    a = ax[1, 0]
    for r, lab in [(res_all, "all"), (res_clean, "k < 6.8")]:
        a.plot(r["g_grid"], r["prof"] - r["prof"].min(), label=lab)
    for lvl in (1.0, 3.84):
        a.axhline(lvl, c="gray", ls=":")
    a.set_ylim(0, 10); a.set_xlabel("g_extra"); a.set_ylabel("profile $\\Delta\\chi^2$")
    a.legend()

    a = ax[1, 1]
    for r, lab in [(res_all, "all"), (res_clean, "k < 6.8")]:
        h = a.hist(r["d_null"], bins=40, histtype="step", density=True, label=f"null, {lab}")
        a.axvline(r["d_obs"], color=h[2][0].get_edgecolor(), ls="--")
    x = np.linspace(0.01, 12, 200)
    a.plot(x, chi2_dist.pdf(x, 1), "k:", label="$\\chi^2_1$")
    a.set_xlabel("$\\Delta\\chi^2$ (H0 - H1)"); a.set_ylim(0, 1.5); a.legend()

    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    print(f"\nPlot saved: {fname}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=os.path.join(here, "steinhauer_fig3a.csv"))
    ap.add_argument("--nnull", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--kmin-scan", action="store_true",
                    help="also run the low-k cut robustness scan")
    ap.add_argument("--kmins", type=float, nargs="+",
                    default=[0.0, 0.5, 0.8, 1.2, 1.5, 2.0],
                    help="k_min thresholds in um^-1 for the scan")
    args = ap.parse_args()

    df = pd.read_csv(args.csv).dropna(subset=["k_um_inv", "f_khz", "sigma_khz"])
    df = df.sort_values("k_um_inv")
    k = df.k_um_inv.to_numpy(float) * 1e6
    f = df.f_khz.to_numpy(float) * 1e3
    s = df.sigma_khz.to_numpy(float) * 1e3
    if np.any(s <= 0):
        sys.exit("All sigma_khz values must be > 0.")

    print("STEINHAUER 2002 -- BEC RESIDUAL TEST V3")
    print(f"input: {args.csv}")
    print("NOTE: g_extra is a relative correction to the Bogoliubov k^4 term,")
    print("      not a measurement of the fundamental Lambda.")

    res_all = analyse(k, f, s, "ALL POINTS", args.nnull, args.seed)
    m = k < K_CLEAN_MAX
    res_clean = analyse(k[m], f[m], s[m], "CLEAN REGION k < 6.8 um^-1", args.nnull, args.seed + 1)

    plot(k, f, s, res_all, res_clean, os.path.join(os.getcwd(), "steinhauer_test_v3.png"))

    if args.kmin_scan:
        scan = kmin_scan(k, f, s, args.kmins, min(args.nnull, 1000), args.seed)
        out = os.path.join(os.getcwd(), "steinhauer_kmin_scan.csv")
        scan.to_csv(out, index=False)
        print(f"\nScan table saved: {out}")


if __name__ == "__main__":
    main()
