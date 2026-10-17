#!/usr/bin/env python3
"""Corrected eikonal quantities for Paper 2, Section 7 (Table 3, Eq. 20), full dispersive Hamiltonian.
H = 1/2 [ g^{mn} k_m k_n + Lambda (k_loc^2)^2 ], ZAMO k_loc, Kerr equatorial plane, G = c = M = 1.
Omega_ph = (dH/dk_phi)/(dH/dk_t) with ALL terms of H.
Lyapunov exponent from linearising Hamilton's equations about the circular orbit:
  r' = dH/dk_r, k_r' = -dH/dr  ->  mu^2 = -H_{k_r k_r} H_{rr} (affine), lambda_Lyap = mu / |dt/dlambda|.
Validation: Schwarzschild, Lambda = 0: Omega_ph = lambda_Lyap = 1/(3 sqrt 3)."""
import sympy as sp, mpmath as mp
mp.mp.dps = 30
r, kr, L, kt, Lam, a = sp.symbols('r k_r L k_t Lambda a', real=True)
Dl = r**2 - 2*r + a**2; Sig = r**2; A = (r**2 + a**2)**2 - a**2*Dl
gtt = -A/(Sig*Dl); gtp = -2*a*r/(Sig*Dl); gpp = (Dl - a**2)/(Sig*Dl); grr = Dl/Sig
N = sp.sqrt(Sig*Dl/A); Om = 2*a*r/A
gkk = gtt*kt**2 + 2*gtp*kt*L + gpp*L**2 + grr*kr**2
H = (gkk + Lam*(gkk + ((kt + Om*L)/N)**2)**2)/2
args = (r, kr, L, kt, Lam, a)
f = {name: sp.lambdify(args, ex, 'mpmath') for name, ex in dict(
    H=H, Hr=sp.diff(H, r), Hrr=sp.diff(H, r, 2), Hkk=sp.diff(H, kr, 2),
    HL=sp.diff(H, L), Hkt=sp.diff(H, kt)).items()}

def eikonal(spin, lam, guess, E=1):
    rr_, LL = mp.findroot(lambda x, y: [f['H'](x, 0, y, -E, lam, spin), f['Hr'](x, 0, y, -E, lam, spin)], guess)
    p = (rr_, 0, LL, -E, lam, spin)
    Omega = f['HL'](*p)/f['Hkt'](*p)
    lyap = mp.sqrt(-f['Hkk'](*p)*f['Hrr'](*p))/abs(f['Hkt'](*p))
    return rr_, LL/E, Omega, lyap

r0, b0, O0, l0 = eikonal(mp.mpf(0), 0, (3.0, 5.196))
print(f"Validation (Schwarzschild, Lambda=0): Omega={mp.nstr(O0,10)}  lambda_Lyap={mp.nstr(l0,10)}  exact 1/(3sqrt3)={mp.nstr(1/(3*mp.sqrt(3)),10)}")

spin = mp.mpf('0.9')
for kind, guess in (('prograde', (1.5578, 2.8444)), ('retrograde', (3.9103, -6.8323))):
    print(f"\n{kind} (a = 0.9, E = 1):  Lambda   r_ph   Omega_ph   lambda_Lyap   dOmega/Omega   dlambda/lambda")
    base = eikonal(spin, 0, guess)
    for lam in (0, 1e-4, 1e-3, 1e-2):
        rr_, bb, OO, ll = eikonal(spin, mp.mpf(lam), guess)
        print(f"   {lam:>6}  {mp.nstr(rr_,8)}  {mp.nstr(OO,8)}  {mp.nstr(ll,8)}  {mp.nstr((OO-base[2])/base[2],6):>12}  {mp.nstr((ll-base[3])/base[3],6):>12}")
    h = mp.mpf('1e-8')
    for E in (1, 2):
        p_, m_ = eikonal(spin, h, guess, E), eikonal(spin, -h, guess, E)
        dO = (p_[2]-m_[2])/(2*h)/base[2]/E**2; dl = (p_[3]-m_[3])/(2*h)/base[3]/E**2
        print(f"   first order at E={E}: dOmega/Omega = {mp.nstr(dO,6)} Lambda_hat E^2 ; dlambda/lambda = {mp.nstr(dl,6)} Lambda_hat E^2")
