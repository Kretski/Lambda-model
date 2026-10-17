#!/usr/bin/env python3
"""Eikonal D_Lambda for Kerr photon rings with the Paper 2 Hamiltonian (notes/EIKONAL_D_LAMBDA.md).
H = 1/2 [ g^{mn} k_m k_n + Lambda (k_loc^2)^2 ],  k_loc^2 = g^{mn} k_m k_n + (n.k)^2 (ZAMO normal n),
Kerr equatorial plane, G = c = M = 1, signature (-,+,+,+), k = (-E, 0, 0, L), a = 0.9.
Circular null orbit: H = 0, dH/dr = 0. Orbital frequency Omega_ph = (dH/dL)/(dH/dk_t) (= dphi/dt).
Method A: numerical (mpmath findroot, central differences in Lambda).
Method B: implicit differentiation with exact symbolic derivatives (sympy)."""
import mpmath as mp
import sympy as sp
mp.mp.dps = 40
A_SPIN = sp.Rational(9, 10)

# ---------- symbolic Hamiltonian ----------
r, L, Lam, kt = sp.symbols('r L Lambda k_t', real=True)
a = A_SPIN
Dl = r**2 - 2*r + a**2; Sig = r**2; Akerr = (r**2 + a**2)**2 - a**2*Dl
gtt = -Akerr/(Sig*Dl); gtp = -2*a*r/(Sig*Dl); gpp = (Dl - a**2)/(Sig*Dl)
N = sp.sqrt(Sig*Dl/Akerr); Om = 2*a*r/Akerr
gkk = gtt*kt**2 + 2*gtp*kt*L + gpp*L**2
H = (gkk + Lam*(gkk + ((kt + Om*L)/N)**2)**2)/2
F = sp.Matrix([H, sp.diff(H, r)])
Omega = sp.diff(H, L)/sp.diff(H, kt)
Hn = sp.lambdify((r, L, kt, Lam), H, 'mpmath')
dHr = sp.lambdify((r, L, kt, Lam), sp.diff(H, r), 'mpmath')
Omn = sp.lambdify((r, L, kt, Lam), Omega, 'mpmath')
GUESS = {'pro': (1.5578, 2.8444), 'ret': (3.9103, -6.8323)}

def ring(kind, lam, E=1):
    r0, b0 = GUESS[kind]
    rr, LL = mp.findroot(lambda x, y: [Hn(x, y, -E, lam), dHr(x, y, -E, lam)], (r0, b0*E))
    return rr, LL/E, Omn(rr, LL, -E, lam)

print("1) Paper 2 setup reproduced (E = 1): photon-ring radii and impact parameters")
for lam in ('0', '1e-4', '1e-3', '1e-2'):
    rp, bp, Op = ring('pro', mp.mpf(lam)); rr_, br, Or = ring('ret', mp.mpf(lam))
    print(f"   Lambda={lam:>5}: pro r={mp.nstr(rp,7)} b={mp.nstr(bp,7)} Omega={mp.nstr(Op,7)} | "
          f"ret r={mp.nstr(rr_,7)} b={mp.nstr(br,7)} Omega={mp.nstr(Or,7)}")

print("\n2) First-order coefficients at Lambda = 0, two independent methods (E = 1)")
coef = {}
for kind in ('pro', 'ret'):
    h = mp.mpf('1e-8')
    numA = (ring(kind, h)[2] - ring(kind, -h)[2])/(2*h)
    r0, b0, O0 = ring(kind, 0)
    pt = {r: r0, L: b0, kt: -1, Lam: 0}
    J = F.jacobian([r, L]).subs(pt).evalf(30); dF = sp.diff(F, Lam).subs(pt).evalf(30)
    dr, dL = list(-J.LUsolve(dF))
    numB = (sp.diff(Omega, Lam) + sp.diff(Omega, r)*dr + sp.diff(Omega, L)*dL).subs(pt).evalf(20)
    coef[kind] = (float(O0), float(numB), float(dL))
    print(f"   {kind}: Omega0={mp.nstr(O0,10)}  db/dLambda={sp.N(dL,10)}  dOmega/dLambda: A={mp.nstr(numA,10)}  B={sp.N(numB,10)}  rel={float(numB)/float(O0):+.6f}")

print("\n3) Energy scaling of dOmega/dLambda (expected exactly E^2)")
for E in (1, 2, 4):
    h = mp.mpf('1e-8')
    s = [(ring(k, h, E)[2] - ring(k, -h, E)[2])/(2*h)/E**2 for k in ('pro', 'ret')]
    print(f"   E={E}: (dOmega/dLambda)/E^2  pro={mp.nstr(s[0],10)}  ret={mp.nstr(s[1],10)}")

(Op0, cp, _), (Or0, cr, _) = coef['pro'], coef['ret']
kp = cp*Op0**2            # delta omega_+ = l * cp * Lambda_hat * (l Omega_pro)^2
km = (-cr)*Or0**2         # delta omega_- = l * d|Omega_ret| , with |Omega_ret| growing
print("\n4) Eikonal D_Lambda (geometric units, Lambda_hat = Lambda/M^2, omega in 1/M), E = omega ~ l|Omega|:")
print(f"   delta omega(+l) = {kp:.6f} Lambda_hat l^3 ;  delta omega(-l) = {km:.6f} Lambda_hat l^3")
print(f"   D_Lambda^eik = {kp-km:.6f} Lambda_hat l^3")
G, c, Msun = 6.674e-11, 2.998e8, 1.989e30
for label, M, lam in (("10 Msun ringdown, Lambda = 7.4e-12 m^2 (GWTC-4.0 bound)", 10*Msun, 7.4e-12),
                      ("10 Msun, Lambda = 2.6e-70 m^2 (Planck-scale estimate)", 10*Msun, 2.6e-70)):
    Mg = G*M/c**2; lamhat = lam/Mg**2
    D_hz = (kp - km)*lamhat*8*c**3/(G*M)/(2*mp.pi)
    print(f"   {label}: Lambda_hat = {lamhat:.2e}, D(l=2) = {float(D_hz):.2e} Hz")
