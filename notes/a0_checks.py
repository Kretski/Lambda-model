#!/usr/bin/env python3
"""A0 symbolic checks (flat space, preferred frame u = d/dt), signature (-,+,+,+), hbar = c = 1.
1) The Paper 1 quartic term (Lambda/2) Y^2, Y = |grad phi|^2, drops out of the linearised field equation.
2) The candidate L2 = 1/2 phi_t^2 - 1/2 |grad phi|^2 - Lambda/2 (lap phi)^2 gives omega^2 = k^2 + Lambda k^4.
3) Averaged plane-wave energy density per unit amplitude: k^2 (1 + Lambda k^2) / 2 (positive for Lambda >= 0)."""
import sympy as sp

t, x, y, z, Lam, eps, w, k = sp.symbols('t x y z Lambda epsilon omega k', real=True)
X = [t, x, y, z]
phi = sp.Function('phi')(t, x, y, z)
grad2 = sum(sp.diff(phi, v)**2 for v in (x, y, z))
lap = sum(sp.diff(phi, v, 2) for v in (x, y, z))

L1 = -sp.Rational(1, 2) * (-sp.diff(phi, t)**2 + grad2 + Lam / 2 * grad2**2)
eom1 = sp.euler_equations(L1, phi, X)[0].lhs
psi = sp.Function('psi')(t, x, y, z)
lin1 = sp.simplify(sp.diff(sp.expand(eom1.subs(phi, eps * psi).doit()), eps).subs(eps, 0))
print("1) linearised field equation of the Paper 1 term:", lin1, "| contains Lambda:", lin1.has(Lam))

L2 = sp.Rational(1, 2) * sp.diff(phi, t)**2 - sp.Rational(1, 2) * grad2 - sp.Rational(1, 2) * Lam * lap**2
eom2 = sp.euler_equations(L2, phi, X)[0].lhs
pw = sp.exp(sp.I * (k * x - w * t))
disp = sp.simplify(eom2.subs(phi, pw).doit() / pw)
print("2) candidate on a plane wave:", sp.factor(disp), "=> omega^2 =", sp.solve(disp, w**2))

print("3) averaged energy density per unit amplitude: k^2 (1 + Lambda k^2) / 2")
