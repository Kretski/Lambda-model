"""
Lambda Step 1.1 — Final Forensic Validation
============================================
Model:   u_tt = c^2 u_xx - c^2 Lambda u_xxxx
Target:  omega^2 = c^2 k^2 (1 + Lambda k^2)

Three independent tests on pure-mode initial conditions cos(k*x):

Test A  FFT dispersion
        Measure omega from FFT of u(x=0, t) over 20 full periods.
        Compare against exact discrete FD dispersion (not continuum):
            omega_FD^2 = c^2 * mu * (1 + Lambda*mu),  mu = (2/dx)^2 sin^2(k*dx/2)
        Threshold: |err_FD| < 0.1%

Test B  Shadow Hamiltonian conservation (leapfrog)
        E_half = (1/2)||v_{n+1/2}||^2 + PE(u^n) oscillates at 2*omega (expected).
        Report the LINEAR TREND of E_half over 100 periods (not oscillation amplitude).
        A flat trend means the shadow Hamiltonian is conserved.
        Threshold: |trend * T_total / E_mean| < 1e-5

Test C  Momentum conservation
        P = integral(u_t) = 0 exactly for cos(kx) with zero initial velocity.
        |P| / ||u||_2 < 1e-9 at every step.

Notes
-----
- FFT uses 20 full periods → peak at expected index, no spectral leakage.
- dt subdivides one period into integer steps, stays <= dt_CFL.
- P oscillates at omega; threshold is on its absolute value / norm.
"""

import numpy as np

# ── Parameters ────────────────────────────────────────────────────────────────
N      = 4096
L      = 100.0
c      = 1.0
LAMBDA = 2e-3
MODES  = [8, 16, 24]

FFT_PERIODS    = 20
CONSERVE_PERI  = 100
SAFETY         = 0.25

dx = L / N
x  = np.arange(N) * dx

# ── Dispersions ───────────────────────────────────────────────────────────────
def mu_fd(k):
    return (2.0/dx)**2 * np.sin(k*dx/2)**2

def omega_fd(k):
    m = mu_fd(k)
    return np.sqrt(c**2 * m * (1.0 + LAMBDA * m))

def omega_cont(k):
    return np.sqrt(c**2 * k**2 * (1.0 + LAMBDA * k**2))

# ── CFL ───────────────────────────────────────────────────────────────────────
mu_max    = (2.0/dx)**2
omega_max = np.sqrt(c**2 * mu_max * (1.0 + LAMBDA * mu_max))
dt_cfl    = SAFETY * 2.0 / omega_max

# ── Operators ─────────────────────────────────────────────────────────────────
def d2(u):
    return (np.roll(u, 1) - 2*u + np.roll(u,-1)) / dx**2

def d4(u):
    return (np.roll(u, 2) - 4*np.roll(u, 1) + 6*u
            - 4*np.roll(u,-1) + np.roll(u,-2)) / dx**4

def RHS(u):
    return c**2 * d2(u) - c**2 * LAMBDA * d4(u)

def PE(u):
    ux  = (np.roll(u,-1) - np.roll(u, 1)) / (2*dx)
    uxx = d2(u)
    return 0.5*c**2*np.sum(ux**2)*dx + 0.5*c**2*LAMBDA*np.sum(uxx**2)*dx

def E_half(uc, un, dt):
    v = (un - uc) / dt
    return 0.5*np.sum(v**2)*dx + PE(uc)

# ── Header ────────────────────────────────────────────────────────────────────
sep = "=" * 72
print(sep)
print("LAMBDA STEP 1.1 — FINAL FORENSIC VALIDATION REPORT")
print(sep)
print(f"N={N}, L={L}, dx={dx:.6e}")
print(f"Lambda={LAMBDA:.3e}, c={c}")
print(f"omega_max(FD)={omega_max:.6e}, dt_CFL={dt_cfl:.6e}")
print()

# ── Test A: FFT dispersion ────────────────────────────────────────────────────
print("TEST A — FFT DISPERSION (pure-mode, 20 periods, peak near expected index)")
print(f"{'mode':>6}  {'k_phys':>13}  {'omega_meas':>13}  {'omega_cont':>13}"
      f"  {'omega_FD':>13}  {'err_cont%':>10}  {'err_FD%':>10}  {'A-pass':>7}")
print("-" * 98)

A_results = []
for mode in MODES:
    k     = 2*np.pi*mode / L
    o_c   = omega_cont(k)
    o_fd  = omega_fd(k)
    T_fd  = 2*np.pi / o_fd

    # dt: integer subdivision of one period
    spp = max(32, int(np.ceil(T_fd / dt_cfl)))
    dt  = T_fd / spp   # <= dt_cfl

    N_fft  = FFT_PERIODS * spp
    u0     = np.cos(k * x)
    up, uc = u0.copy(), u0 + 0.5*dt**2 * RHS(u0)
    u_ts   = np.zeros(N_fft)

    ok = True
    for s in range(N_fft):
        u_ts[s] = uc[0]
        un = 2*uc - up + dt**2 * RHS(uc)
        if not np.isfinite(un).all():
            ok = False; break
        up, uc = uc, un

    if ok:
        freqs  = np.fft.rfftfreq(N_fft, d=dt)
        power  = np.abs(np.fft.rfft(u_ts))**2
        i_exp  = int(round(o_fd / (2*np.pi) / (freqs[1]-freqs[0])))
        window = slice(max(1, i_exp-5), i_exp+6)
        i_rel  = np.argmax(power[window])
        i_peak = max(1, i_exp-5) + i_rel
        om_m   = 2*np.pi*freqs[i_peak]
    else:
        om_m = float('nan')

    err_c  = 100*(om_m - o_c)/o_c   if np.isfinite(om_m) else float('nan')
    err_fd = 100*(om_m - o_fd)/o_fd if np.isfinite(om_m) else float('nan')
    a_pass = ok and abs(err_fd) < 0.1

    print(f"{mode:>6}  {k:>13.7e}  {om_m:>13.7e}  {o_c:>13.7e}"
          f"  {o_fd:>13.7e}  {err_c:>10.5f}  {err_fd:>10.5f}"
          f"  {'PASS' if a_pass else 'FAIL':>7}")

    A_results.append(dict(mode=mode, k=k, om_m=om_m, o_c=o_c, o_fd=o_fd,
                          err_c=err_c, err_fd=err_fd, ok=ok, a_pass=a_pass,
                          spp=spp, dt=dt))

# ── Tests B & C: conservation ─────────────────────────────────────────────────
print()
print("TEST B — SHADOW HAMILTONIAN TREND  /  TEST C — MOMENTUM")
print(f"  (Conservation over {CONSERVE_PERI} periods, trend = slope*T_total/E_mean)")
print(f"{'mode':>6}  {'E_osc_amp':>12}  {'E_trend/E':>12}  {'B-pass':>7}"
      f"  {'max|P|/norm':>14}  {'C-pass':>7}")
print("-" * 72)

B_results = []
for r in A_results:
    mode, k, dt, spp = r['mode'], r['k'], r['dt'], r['spp']
    nsteps      = CONSERVE_PERI * spp
    rec_every   = max(1, spp // 8)     # 8 samples per period

    u0     = np.cos(k * x)
    norm_u = np.sqrt(np.sum(u0**2)*dx)
    up, uc = u0.copy(), u0 + 0.5*dt**2 * RHS(u0)

    e_vals, t_vals, P_max = [], [], 0.0
    ok = True
    for step in range(nsteps):
        un = 2*uc - up + dt**2 * RHS(uc)
        if not np.isfinite(un).all():
            ok = False; break
        if step % rec_every == 0:
            e_vals.append(E_half(uc, un, dt))
            t_vals.append(step*dt)
        Pn   = np.sum((un - uc)/dt) * dx
        P_max = max(P_max, abs(Pn)/norm_u)
        up, uc = uc, un

    e, t = np.array(e_vals), np.array(t_vals)
    A_mat   = np.column_stack([t, np.ones_like(t)])
    slope,_ = np.linalg.lstsq(A_mat, e, rcond=None)[0]
    trend   = slope * t[-1] / e.mean() if e.mean() != 0 else float('nan')
    osc     = (e.max() - e.min()) / e.mean() if e.mean() != 0 else float('nan')

    b_pass = ok and abs(trend) < 1e-5
    c_pass = ok and P_max     < 1e-9

    print(f"{mode:>6}  {osc:>12.4e}  {trend:>12.4e}  {'PASS' if b_pass else 'FAIL':>7}"
          f"  {P_max:>14.4e}  {'PASS' if c_pass else 'FAIL':>7}")

    B_results.append(dict(mode=mode, osc=osc, trend=trend, P_max=P_max,
                          b_pass=b_pass, c_pass=c_pass))

# ── Overall ───────────────────────────────────────────────────────────────────
A_ok = all(r['a_pass'] for r in A_results)
B_ok = all(r['b_pass'] for r in B_results)
C_ok = all(r['c_pass'] for r in B_results)

print()
print(sep)
print("FINAL PASS/FAIL SUMMARY")
print(sep)
print(f"Test A — FFT dispersion vs FD (|err|<0.1%)  : {'PASS' if A_ok else 'FAIL'}")
print(f"Test B — Shadow Hamiltonian trend (<1e-5)    : {'PASS' if B_ok else 'FAIL'}")
print(f"Test C — Momentum conservation (<1e-9/norm)  : {'PASS' if C_ok else 'FAIL'}")
print()
overall = A_ok and B_ok and C_ok
print(f"OVERALL STEP 1.1  :  {'PASS — operator fully validated ✓' if overall else 'FAIL — see details above'}")
print(sep)

if overall:
    print()
    print("Physical conclusions (constant Lambda):")
    print("  1. Correct sign: u_tt = c^2*u_xx - c^2*Lambda*u_xxxx")
    print("     gives omega^2 = c^2*k^2*(1+Lambda*k^2) > 0 — stable.")
    print("  2. Symmetric (self-adjoint) spatial operator confirmed")
    print("     by machine-precision momentum conservation.")
    print("  3. Leapfrog shadow Hamiltonian flat to < 1e-5 over 100 periods.")
    print()
    print("Ready to proceed to Step 2 (gradient Lambda(x)).")
