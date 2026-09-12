"""
stage5I_1_2_dispersion_radial_p.py
======================================

STAGE 5I-1: Dispersion relation F(k) and Doppler-shifted branches
omega_D^+-(p,m,r), exactly as in Smaniotto et al. 2025 (arXiv:2502.11209v3)
Methods, Eq. (4) and Eq. (6). NO Lambda term -- this is the authors'
own baseline model, reproduced independently.

STAGE 5I-2: Radial momentum p(r; omega, m) solved from the FULL
dispersion condition H=0 (Eq. 7-8), not assumed to take a simplified
low-k WKB form. This matters because F(k) = (gk + gamma*k^3)tanh(h0*k)
is NOT a simple power law -- k enters through k = sqrt(p^2 + (m/r)^2),
and the tanh(h0*k) factor makes this a transcendental equation in p at
fixed (omega, m, r).

PHYSICAL SETUP (EXP B, Table SI / Methods):
    C      = 769 mm^2/s     (circulation; Methods gives 769+/-5,
                              Table SI gives 769+/-4 -- both tested
                              as a sensitivity check, not resolved)
    Omega  = 0.10 rad/s     (residual solid-body rotation)
    gamma  = 2.22e-6 m^3/s^2 (surface tension / density, at 1.70 K)
    h0     = 34 mm          (equilibrium interface height)
    g      = 9.81 m/s^2     (standard gravity)
    r_B    = 37.3 mm        (outer boundary)

All lengths converted to SI (metres) internally; C in m^2/s, h0 in m.

NO Lambda dispersion term is introduced anywhere in this file. This is
the authors' baseline model only, per the agreed Stage 5I plan.
"""

import numpy as np
from scipy.optimize import brentq


# ── EXP B physical parameters (SI units) ─────────────────────────────────
G_GRAV = 9.81          # m/s^2
GAMMA = 2.22e-6        # m^3/s^2 (surface tension / density at 1.70 K)
H0 = 34.0e-3           # m (equilibrium interface height)
R_B = 37.3e-3          # m (outer boundary)

# Circulation, two named values for the sensitivity check requested in
# the plan (Methods text: 769+/-5 mm^2/s; Table SI: 769+/-4 mm^2/s).
# Central values are identical to the precision quoted; kept as
# separate named constants for explicit bookkeeping.
C_METHODS = 769e-6      # m^2/s
C_TABLE = 769e-6        # m^2/s
OMEGA = 0.10            # rad/s (residual solid-body rotation)


# ── Dispersion function F(k), Eq. (6) ────────────────────────────────────
def F_dispersion(k, gamma=GAMMA, h0=H0, g=G_GRAV):
    """F(k) = (g*k + gamma*k^3) * tanh(h0*k). k is the wavevector magnitude."""
    k = np.asarray(k, dtype=float)
    return (g * k + gamma * k ** 3) * np.tanh(h0 * k)


# ── Doppler-shifted branches omega_D^+-(p, m, r), Eq. (4) ────────────────
def omega_D(p, m, r, C=C_METHODS, Omega=OMEGA, gamma=GAMMA, h0=H0, g=G_GRAV):
    """
    omega_D^+-(p, m, r) = m*C/r^2 + m*Omega +/- sqrt(F(k)),
    k = sqrt(p^2 + (m/r)^2). Returns (omega_plus, omega_minus).
    """
    k = np.sqrt(p ** 2 + (m / r) ** 2)
    doppler = m * C / r ** 2 + m * Omega
    sqrtF = np.sqrt(np.maximum(F_dispersion(k, gamma, h0, g), 0.0))
    return doppler + sqrtF, doppler - sqrtF


def omega_D_plus_at_p0(m, r, C=C_METHODS, Omega=OMEGA, gamma=GAMMA, h0=H0,
                        g=G_GRAV):
    """omega_D^+(p=0, m, r), Eq. (10) -- effective potential barrier."""
    k0 = np.abs(m) / r
    doppler = m * C / r ** 2 + m * Omega
    sqrtF = np.sqrt(np.maximum(F_dispersion(k0, gamma, h0, g), 0.0))
    return doppler + sqrtF


# ── Radial momentum p(r; omega, m), full transcendental solve ───────────
def solve_p(omega, m, r, C=C_METHODS, Omega=OMEGA, gamma=GAMMA, h0=H0,
            g=G_GRAV, p_max_search=2000.0):
    """
    Solve H=0, equivalently omega = omega_D^+(p,m,r) (the branch used
    for the light-ring / potential-barrier construction, Eq. 10), for
    the radial wavenumber p at fixed (omega, m, r), via full root-
    finding on the transcendental equation -- NOT a low-k or high-k
    asymptotic approximation.

    Returns p (real, >=0) if a propagating solution exists, or None if
    evanescent at this (omega, r).
    """
    doppler = m * C / r ** 2 + m * Omega
    target = omega - doppler  # must equal sqrt(F(k)) if p real

    def residual(p_squared):
        p = np.sqrt(max(p_squared, 0.0))
        k = np.sqrt(p ** 2 + (m / r) ** 2)
        return F_dispersion(k, gamma, h0, g) - target ** 2

    p2_lo, p2_hi = 0.0, p_max_search ** 2
    r_lo = residual(p2_lo)
    r_hi = residual(p2_hi)

    if r_lo * r_hi > 0:
        return None  # no propagating root in range: evanescent here

    p2_root = brentq(residual, p2_lo, p2_hi, xtol=1e-12, rtol=1e-12)
    return np.sqrt(max(p2_root, 0.0))


def find_light_ring(m, C=C_METHODS, Omega=OMEGA, gamma=GAMMA, h0=H0,
                     g=G_GRAV, r_min=1.0e-3, r_max=R_B):
    """
    Locate the light ring: radius r_sp maximizing omega_D^+(p=0,m,r)
    for m<0 (counterrotating), satisfying Eq. (12). Returns
    (r_sp, omega_lightring) or (None, None) if no interior maximum
    exists in [r_min, r_max].
    """
    r_grid = np.linspace(r_min, r_max, 20000)
    omega_grid = np.array([
        omega_D_plus_at_p0(m, r, C, Omega, gamma, h0, g) for r in r_grid
    ])

    idx_max = np.argmax(omega_grid)

    if idx_max == 0 or idx_max == len(r_grid) - 1:
        return None, None

    def d_omega_dr(r):
        eps = 1e-7
        return (omega_D_plus_at_p0(m, r + eps, C, Omega, gamma, h0, g) -
                omega_D_plus_at_p0(m, r - eps, C, Omega, gamma, h0, g)) / (2 * eps)

    r_lo = r_grid[max(idx_max - 1, 0)]
    r_hi = r_grid[min(idx_max + 1, len(r_grid) - 1)]

    try:
        d_lo = d_omega_dr(r_lo)
        d_hi = d_omega_dr(r_hi)
        if d_lo * d_hi < 0:
            r_sp = brentq(d_omega_dr, r_lo, r_hi, xtol=1e-10)
        else:
            r_sp = r_grid[idx_max]
    except Exception:
        r_sp = r_grid[idx_max]

    omega_lr = omega_D_plus_at_p0(m, r_sp, C, Omega, gamma, h0, g)
    return r_sp, omega_lr


# ── Self-test / sanity check ──────────────────────────────────────────────
def main():
    print("=" * 72)
    print("STAGE 5I-1/5I-2 — BASELINE DISPERSION AND RADIAL MOMENTUM")
    print("=" * 72)
    print()
    print("  EXP B parameters (SI units):")
    print(f"    C      = {C_METHODS*1e6:.1f} mm^2/s")
    print(f"    Omega  = {OMEGA} rad/s")
    print(f"    gamma  = {GAMMA:.3e} m^3/s^2")
    print(f"    h0     = {H0*1e3:.1f} mm")
    print(f"    r_B    = {R_B*1e3:.1f} mm")
    print()

    print("  Test 1 - F(k) sanity check (low-k should be ~ g*k, gravity-")
    print("  wave dominated; high-k should show gamma*k^3 capillary term):")
    for k_test in [1.0, 10.0, 100.0, 1000.0]:
        F_val = F_dispersion(k_test)
        F_gravity_only = G_GRAV * k_test * np.tanh(H0 * k_test)
        print(f"    k={k_test:>7.1f} /m: F(k)={F_val:.4e}, "
              f"gravity-only approx={F_gravity_only:.4e}, "
              f"ratio={F_val/F_gravity_only:.4f}")
    print()

    print("  Test 2 - light-ring location for several m values:")
    print(f"  {'m':>6}  {'r_sp (mm)':>12}  {'omega_lr/2pi (Hz)':>18}")
    print("  " + "-" * 44)
    for m in [-4, -8, -12, -16, -20]:
        r_sp, omega_lr = find_light_ring(m)
        if r_sp is not None:
            f_lr = omega_lr / (2 * np.pi)
            print(f"  {m:>6}  {r_sp*1e3:>12.4f}  {f_lr:>18.4f}")
        else:
            print(f"  {m:>6}  {'no interior max':>12}  {'--':>18}")
    print()

    print("  Test 3 - radial momentum p(r) at fixed omega, m (spot check):")
    m_test = -12
    r_sp, omega_lr = find_light_ring(m_test)
    if r_sp is not None:
        print(f"    m={m_test}: light ring at r_sp={r_sp*1e3:.3f} mm, "
              f"f_lr={omega_lr/(2*np.pi):.4f} Hz")
        for f_test in [omega_lr/(2*np.pi) * 0.9, omega_lr/(2*np.pi) * 1.1]:
            omega_test = 2 * np.pi * f_test
            p_val = solve_p(omega_test, m_test, r_sp)
            status = f"p={p_val:.4f} /m" if p_val is not None else "evanescent"
            print(f"    f={f_test:.4f} Hz at r=r_sp: {status}")
    print()

    print("  Baseline dispersion/radial-momentum machinery constructed.")
    print("  NO Lambda term introduced anywhere in this file.")


if __name__ == "__main__":
    main()
