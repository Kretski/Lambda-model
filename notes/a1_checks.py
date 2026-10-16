#!/usr/bin/env python3
"""A1 symbolic checks on Kerr (Boyer-Lindquist, signature -+++, G = c = 1).
1) Eikonal symbol: g^{mn} k_m k_n + (n.k)^2 == gamma^{ij} k_i k_j for the ZAMO (normal) observer.
2) Field equation: the Euler-Lagrange equation of
       L = -1/2 sqrt(-g) [ g^{mn} d_m phi d_n phi + Lambda (D^2 phi)^2 ],
   D^2 = Laplacian of the t = const slice, equals
       +sqrt(-g) [ Box phi - (Lambda/N) D^2( N D^2 phi ) ]   (checked at random points, a test field).
3) Separation: Schwarzschild (a = 0) separates with spherical harmonics; for a != 0 the
   Carter constant is not conserved by the eikonal Hamiltonian (Poisson bracket != 0)."""
import random
import sympy as sp

t, r, th, ph = sp.symbols('t r theta phi', real=True)
M, a, Lam = sp.symbols('M a Lambda', positive=True)
X = [t, r, th, ph]
Sig = r**2 + a**2*sp.cos(th)**2
Dl = r**2 - 2*M*r + a**2
A = (r**2 + a**2)**2 - a**2*Dl*sp.sin(th)**2
ginv = sp.zeros(4)
ginv[0, 0] = -A/(Sig*Dl); ginv[0, 3] = ginv[3, 0] = -2*M*a*r/(Sig*Dl)
ginv[1, 1] = Dl/Sig; ginv[2, 2] = 1/Sig; ginv[3, 3] = (Dl - a**2*sp.sin(th)**2)/(Sig*Dl*sp.sin(th)**2)
N = sp.sqrt(Sig*Dl/A); Om = 2*M*a*r/A                    # lapse, ZAMO angular velocity
gam = sp.diag(Sig/Dl, Sig, A*sp.sin(th)**2/Sig)           # induced metric of t = const
gaminv = gam.inv()
sqrtg = Sig*sp.sin(th)                                    # sqrt(-g) for Kerr in BL
sqrtgam = sp.sqrt(gam.det())

# 1) eikonal symbol
k = sp.symbols('k_t k_r k_th k_ph', real=True)
kk = sum(ginv[i, j]*k[i]*k[j] for i in range(4) for j in range(4))
ndotk = (k[0] + Om*k[3])/N
lhs = kk + ndotk**2
rhs = sum(gaminv[i, j]*k[i+1]*k[j+1] for i in range(3) for j in range(3))
print("1) g^{mn}k_m k_n + (n.k)^2 - gamma^{ij}k_i k_j =", sp.simplify(lhs - rhs))

# 2) field equation, numerically at random points with a test field
phi = sp.Function('phi')(*X)
def D2(f):
    s = 0
    for i, xi in enumerate((r, th, ph)):
        for j, xj in enumerate((r, th, ph)):
            if gaminv[i, j] != 0:
                s += sp.diff(sqrtgam*gaminv[i, j]*sp.diff(f, xj), xi)
    return s/sqrtgam
grad2 = sum(ginv[i, j]*sp.diff(phi, X[i])*sp.diff(phi, X[j]) for i in range(4) for j in range(4))
L = -sp.Rational(1, 2)*sqrtg*(grad2 + Lam*D2(phi)**2)
EL = sp.euler_equations(L, phi, X)[0].lhs
box = sum(sp.diff(sqrtg*ginv[i, j]*sp.diff(phi, X[j]), X[i]) for i in range(4) for j in range(4))/sqrtg
expected = sqrtg*(box - Lam/N*D2(N*D2(phi)))     # sympy's EL convention: dL/dphi - d(dL/dphi_m) + ...
w, m = sp.Rational(3, 10), 2
test = sp.exp(sp.I*(m*ph - w*t))*(r**3/(r**2 + 1))*sp.sin(th)**2*sp.cos(th)
diff = (EL - expected).subs(phi, test).doit()
vals = {M: 1, a: sp.Rational(7, 10), Lam: sp.Rational(1, 10)}
random.seed(1)
worst = 0
for _ in range(4):
    pt = {t: random.uniform(0, 1), r: random.uniform(3, 8), th: random.uniform(0.3, 2.8), ph: random.uniform(0, 6)}
    num = complex(sp.N(diff.subs(vals).subs(pt), 30))
    ref = complex(sp.N(expected.subs(phi, test).doit().subs(vals).subs(pt), 30))
    worst = max(worst, abs(num)/max(1e-30, abs(ref)))
print(f"2) |EL - expected| / |expected| at 4 random points (a = 0.7, Lambda = 0.1): {worst:.1e}")

# 3a) Schwarzschild separation: D^2 acting on R(r) Y_lm with l = 2, m = 1
R = sp.Function('R')(r)
Y = sp.sin(th)*sp.cos(th)*sp.exp(sp.I*ph)                 # proportional to Y_21
f0 = R*Y
D2s = sp.simplify((D2(f0)/Y).subs(a, 0))
print("3a) a = 0: D^2(R Y_21)/Y_21 depends on theta?", sp.simplify(D2s).has(th))

# 3b) Kerr, a != 0: Poisson bracket of the Carter constant with the eikonal Hamiltonian
#     H = 1/2 [ g^{mn} k_m k_n + Lambda (gamma^{ij} k_i k_j)^2 ]
pt_, pr_, pth_, pph_ = k
H0 = sp.Rational(1, 2)*kk
HL = sp.Rational(1, 2)*(kk + Lam*rhs**2)
E, Lz = -pt_, pph_
Q = pth_**2 + sp.cos(th)**2*(a**2*(-E**2) + Lz**2/sp.sin(th)**2)   # Carter constant (massless)
def PB(F, G):
    s = 0
    for q, p in ((t, pt_), (r, pr_), (th, pth_), (ph, pph_)):
        s += sp.diff(F, q)*sp.diff(G, p) - sp.diff(F, p)*sp.diff(G, q)
    return s
base = {M: 1, a: sp.Rational(9, 10), Lam: sp.Rational(1, 10), t: 0, r: sp.Rational(37, 10), th: sp.Rational(11, 10),
        ph: 0, pt_: -1, pth_: sp.Rational(2, 10), pph_: sp.Rational(25, 10)}
def on_shell(H, pts):
    sol = sp.nsolve(H.subs(pts), pr_, 1.0)          # choose p_r so that H = 0 (null ray)
    d = dict(pts); d[pr_] = sol
    return d
for label, H, pts in (("GR, a = 0.9", H0, base),
                      ("Lambda = 0.1, a = 0.9", HL, base),
                      ("Lambda = 0.1, a = 0 (Schwarzschild)", HL, {**base, a: 0})):
    d = on_shell(H, pts)
    print(f"3b) on shell (H = {float(H.subs(d)):.1e}): {{Q, H}} for {label}: {float(PB(Q, H).subs(d)):+.3e}")
