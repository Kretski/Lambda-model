"""
A2_chromatic_shadow_generalized.py
=====================================

GENERALIZED dispersion-family version of A2_chromatic_shadow.py.

Original (n=1, from Paper 2):
    H = (1/2)[g^mn k_m k_n + Lambda k_loc^4]
    -> delta_b ~ Lambda * E^2   (confirmed numerically, slope -2.000)

Generalized family (chat-session derivation):
    H_n = (1/2)[g^mn k_m k_n + Lambda_n k_loc^(2n+2)]

DERIVED PREDICTION (Cramer's-rule scaling argument, not assumed):
    k_loc^2 is homogeneous degree 2 in E (confirmed: both terms in
    zamo_kloc2 scale as E^2).
    => RHS of the implicit-differentiation system ~ k_loc^(2n+2) ~ E^(2n+2)
    => Jacobian J (built purely from the GR H, Lambda=0) is homogeneous
       degree 2 in E (same as the GR mass-shell itself)
       => det(J) ~ E^4
    => db/dLambda_n = (J . rhs)/det(J) ~ (E^2 * E^(2n+2)) / E^4 = E^(2n)

    So: delta_b ~ Lambda_n * E^(2n)

    n=1 -> E^2  (must reproduce the original Paper 2 result EXACTLY --
                 this is the built-in regression test, --n 1)
    n=2 -> E^4
    n=3 -> E^6
    n=4 -> E^8

Usage:
    python A2_chromatic_shadow_generalized.py --n 1   # regression test
    python A2_chromatic_shadow_generalized.py --n 2
    python A2_chromatic_shadow_generalized.py --n 3
    python A2_chromatic_shadow_generalized.py --n 4
    python A2_chromatic_shadow_generalized.py --n 1 --n 2 --n 3 --n 4 --table
        (runs all four, prints the n -> measured exponent p table)
"""

import argparse
import math
import numpy as np
from scipy.optimize import fsolve
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

M  = 1.0
RS = 2 * M
A  = 0.9


# ── Metric ────────────────────────────────────────────────────────────────
def kerr_eq(r):
    Delta = r**2 - RS*r + A**2
    Sigma = r**2
    g_tt = -(1 - RS*r/Sigma)
    g_tp = -RS*r*A/Sigma
    g_pp = r**2 + A**2 + RS*r*A**2/Sigma
    det = g_tt*g_pp - g_tp**2
    return dict(tt=g_pp/det, tp=-g_tp/det, pp=g_tt/det,
                g_tt_cov=g_tt, g_tp_cov=g_tp, g_pp_cov=g_pp, Delta=Delta)


def zamo_velocity(r):
    gI = kerr_eq(r)
    gtt, gtp, gpp = gI['g_tt_cov'], gI['g_tp_cov'], gI['g_pp_cov']
    Omega = -gtp/gpp
    bracket = gtt + 2*gtp*Omega + gpp*Omega**2
    if bracket >= 0:
        return None, None
    nt = 1.0/math.sqrt(-bracket)
    nph = Omega*nt
    return nt, nph


def zamo_kloc2(r, b, E):
    """k_loc^2 with ZAMO observer at energy E. Homogeneous degree 2 in E
    (both terms below scale as E^2 -- confirms the scaling argument
    above)."""
    gI = kerr_eq(r)
    L = b*E
    nt, nph = zamo_velocity(r)
    if nt is None:
        return 0.0
    gmunu = gI['tt']*E**2 - 2*gI['tp']*E*L + gI['pp']*L**2
    n_dot_k = -nt*E + nph*L
    return max(gmunu + n_dot_k**2, 0.0)


def H_eq(r, b, Lam, E, n=1):
    """Generalized dispersion-family Hamiltonian:
        H_n = 1/2[g^mn k_m k_n + Lambda_n (k_loc^2)^(n+1)]
    n=1 reproduces the original H = 1/2[g^mn k_m k_n + Lambda k_loc^4]
    exactly, since (k_loc^2)^2 = k_loc^4."""
    gI = kerr_eq(r)
    L = b*E
    gr = gI['tt']*E**2 - 2*gI['tp']*E*L + gI['pp']*L**2
    return 0.5*(gr + Lam*zamo_kloc2(r, b, E)**(n+1))


def dH_dr_num(r, b, Lam, E, eps=1e-7, n=1):
    return (H_eq(r+eps, b, Lam, E, n=n) - H_eq(r-eps, b, Lam, E, n=n)) / (2*eps)


# ── Find photon ring at given (E, Lambda_n, n) ──────────────────────────────
def find_ring(r0, b0, Lam, E, n=1):
    """Circular photon orbit: H_n=0 and dH_n/dr=0."""
    def sys(rb):
        r_, b_ = rb
        if r_ <= A or r_ > RS*8:
            return [1e10, 1e10]
        return [H_eq(r_, b_, Lam, E, n=n), dH_dr_num(r_, b_, Lam, E, n=n)]
    sol = fsolve(sys, [r0, b0], full_output=True)
    r_f, b_f = sol[0]
    res = max(abs(x) for x in sol[1]['fvec'])
    return (r_f, b_f) if res < 1e-6 and r_f > A and r_f < RS*8 else (None, None)


# ── Analytical: db/dLambda_n at given E (implicit differentiation) ─────────
def db_dLam_at_E(r_ph, b_ph, E, n=1, eps_r=1e-5, eps_b=1e-5, eps_L=1e-7):
    """Computes db/dLambda_n|_{Lam=0} at given E via implicit
    differentiation of {H_n=0, dH_n/dr=0}."""
    J11 = dH_dr_num(r_ph, b_ph, 0.0, E, n=n)
    J12 = (H_eq(r_ph, b_ph+eps_b, 0.0, E, n=n) - H_eq(r_ph, b_ph-eps_b, 0.0, E, n=n)) / (2*eps_b)
    J21 = (dH_dr_num(r_ph+eps_r, b_ph, 0.0, E, n=n) - dH_dr_num(r_ph-eps_r, b_ph, 0.0, E, n=n)) / (2*eps_r)
    J22 = (dH_dr_num(r_ph, b_ph+eps_b, 0.0, E, n=n) - dH_dr_num(r_ph, b_ph-eps_b, 0.0, E, n=n)) / (2*eps_b)

    kl2 = zamo_kloc2(r_ph, b_ph, E)
    rhs1 = -0.5*kl2**(n+1)

    def dH_dLam_r(r_):
        return 0.5*zamo_kloc2(r_, b_ph, E)**(n+1)

    rhs2 = -(dH_dLam_r(r_ph+eps_L) - dH_dLam_r(r_ph-eps_L)) / (2*eps_L)

    det = J11*J22 - J12*J21
    if abs(det) < 1e-20:
        return None, None
    dr = (rhs1*J22 - rhs2*J12) / det
    db = (J11*rhs2 - J21*rhs1) / det
    return dr, db


def run_for_n(n, make_plot=True):
    """Runs the full E-sweep + power-law fit for one value of n.
    Returns (n, measured_exponent_pro, measured_exponent_ret, predicted=2n)."""
    out = Path("A2_generalized_results")
    out.mkdir(exist_ok=True)

    r_pro0, b_pro0 = 1.557855, 2.844421
    r_ret0, b_ret0 = 3.910268, -6.832319

    predicted_p = 2 * n

    print("=" * 68)
    print(f"n = {n}  (H_n = 1/2[g.k + Lambda_n * k_loc^{2*n+2}])"
          f"  predicted db/dLambda_n ~ E^{predicted_p}")
    print("=" * 68)

    E_values_GR = np.array([0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0])
    b_pro_GR_E, b_ret_GR_E = [], []
    for E in E_values_GR:
        rp, bp = find_ring(r_pro0, b_pro0, 0.0, E, n=n)
        rr, br = find_ring(r_ret0, b_ret0, 0.0, E, n=n)
        b_pro_GR_E.append(bp)
        b_ret_GR_E.append(br)
    b_pro_GR_E = np.array([x for x in b_pro_GR_E if x is not None])
    b_ret_GR_E = np.array([x for x in b_ret_GR_E if x is not None])
    print(f"  GR achromaticity check: std(b_pro)={np.std(b_pro_GR_E):.2e}  "
          f"std(b_ret)={np.std(b_ret_GR_E):.2e}  (both should be ~0)")

    E_values = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0])
    db_dL_pro_arr, db_dL_ret_arr = [], []
    r_pro_E_arr, r_ret_E_arr = [], []

    for E in E_values:
        rp, bp = find_ring(r_pro0, b_pro0, 0.0, E, n=n)
        rr, br = find_ring(r_ret0, b_ret0, 0.0, E, n=n)
        if rp is None or rr is None:
            db_dL_pro_arr.append(float('nan'))
            db_dL_ret_arr.append(float('nan'))
            r_pro_E_arr.append(float('nan'))
            r_ret_E_arr.append(float('nan'))
            continue
        _, db_p = db_dLam_at_E(rp, bp, E, n=n)
        _, db_r = db_dLam_at_E(rr, br, E, n=n)
        db_dL_pro_arr.append(db_p)
        db_dL_ret_arr.append(db_r)
        r_pro_E_arr.append(rp)
        r_ret_E_arr.append(rr)

    E_arr = E_values
    db_p_arr = np.array(db_dL_pro_arr)
    db_r_arr = np.array(db_dL_ret_arr)

    measured = {}
    for label, db_arr in [("pro", db_p_arr), ("ret", db_r_arr)]:
        mask = np.isfinite(db_arr) & (np.abs(db_arr) > 1e-10)
        if mask.sum() < 3:
            print(f"  {label}: insufficient data for fit")
            measured[label] = float('nan')
            continue
        E_fit = E_arr[mask]
        db_fit = np.abs(db_arr[mask])
        p_fit, logC = np.polyfit(np.log(E_fit), np.log(db_fit), 1)
        measured[label] = p_fit
        C = np.exp(logC)
        print(f"  {label:>3}: |db/dLambda_n| = {C:.6g} * E^{p_fit:.4f}   "
              f"(predicted {predicted_p}, residual {p_fit-predicted_p:+.4f})")

    if make_plot:
        fig, ax = plt.subplots(figsize=(6, 5))
        mask_p = np.isfinite(db_p_arr)
        mask_r = np.isfinite(db_r_arr)
        ax.loglog(E_arr[mask_p], np.abs(db_p_arr[mask_p]), 'bo-', label='prograde')
        ax.loglog(E_arr[mask_r], np.abs(db_r_arr[mask_r]), 'rs-', label='retrograde')
        if mask_p.sum() >= 2:
            idx = mask_p.sum() // 2
            E_ref = E_arr[mask_p]
            ref_vals = np.abs(db_p_arr[mask_p])
            ax.loglog(E_ref, ref_vals[idx]*(E_ref/E_ref[idx])**predicted_p,
                       'k--', alpha=0.6, label=f'predicted $E^{{{predicted_p}}}$')
        ax.set_xlabel('E')
        ax.set_ylabel(r'$|db/d\Lambda_n|$')
        ax.set_title(f'n={n}: dispersion-family chromatic coefficient')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        path = out / f'A2_generalized_n{n}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved -> {path}")

    print()
    return n, measured.get("pro", float('nan')), measured.get("ret", float('nan')), predicted_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, action="append", required=True,
                     help="Dispersion-family member(s) to run. Repeat for "
                          "multiple, e.g. --n 1 --n 2 --n 3 --n 4")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    rows = []
    for n in args.n:
        rows.append(run_for_n(n, make_plot=not args.no_plot))

    print("=" * 68)
    print("SUMMARY TABLE: n -> measured exponent p vs predicted 2n")
    print("=" * 68)
    print(f"{'n':>3}  {'H_n term':>18}  {'predicted p':>12}  "
          f"{'measured p (pro)':>17}  {'measured p (ret)':>17}")
    for n, p_pro, p_ret, predicted in rows:
        term = f"Lambda_{n} k^{2*n+2}"
        print(f"{n:>3}  {term:>18}  {predicted:>12}  "
              f"{p_pro:>17.4f}  {p_ret:>17.4f}")

    if 1 in args.n:
        row1 = [r for r in rows if r[0] == 1][0]
        if abs(row1[1] - 2.0) < 0.05 and abs(row1[2] - 2.0) < 0.05:
            print("\nREGRESSION TEST (n=1): PASS -- reproduces Paper 2's "
                  "E^2 result to within 0.05.")
        else:
            print("\nREGRESSION TEST (n=1): FAIL -- does not reproduce "
                  "Paper 2's E^2 result. Check the generalization before "
                  "trusting n>=2 results.")


if __name__ == "__main__":
    main()
