#!/usr/bin/env python3
"""A2(a): Schwarzschild, candidate equation  Box phi - (Lambda/N) D^2(N D^2 phi) = 0,
phi = exp(-i w t) R(r) Y_lm, f = 1 - 2M/r, N = sqrt(f), slices t = const.
Radial operator:  w^2/f R + (r^2 f R')'/r^2 - L R/r^2 - (Lambda/sqrt f) D( sqrt f D R ),
D F = (sqrt f / r^2)(r^2 sqrt f F')' - L F / r^2,  L = l(l+1).  Fourth order in r.
1) Large r: roots of w^2 - k^2 - Lambda k^4 = 0.
2) Near the horizon: Frobenius exponents with r = 2M + x, f = x/(2M) + O(x^2), R = x^s."""
import sympy as sp

w, Lam, L, x, s, k, M = sp.symbols('omega Lambda L x s k M', positive=True)
print("1) large r: k^2 =", sp.solve(sp.Eq(w**2 - k**2 - Lam*k**4, 0), k**2))

# near horizon, leading order: r -> 2M, f -> x/(2M)
e = 1/(2*M)
F = x**s
sfx = sp.sqrt(e*x)
def Dop(G):                       # r^2 ~ const at leading order: D G ~ sqrt f (sqrt f G')'
    return sfx*sp.diff(sfx*sp.diff(G, x), x)
gr = sp.expand(sp.simplify((w**2/(e*x)*F + sp.diff(e*x*sp.diff(F, x), x))/x**(s-1)))
lam = sp.factor(sp.simplify(1/sfx*Dop(sfx*Dop(F))/x**(s-2)))
print("2) near horizon, R ~ x^s (x = r - 2M):")
print("   GR terms     = x^(s-1) *", sp.simplify(gr))
print("   Lambda terms = x^(s-2) *", lam, "   (more singular: dominates as x -> 0)")
print("   exponents from the Lambda terms:", sp.solve(sp.Eq(lam, 0), s))
print("   exponents for GR alone:", sp.solve(sp.Eq(gr, 0), s))
xs = sp.symbols('x_star', positive=True)
print("   crossover x_* where Lambda e^2 x^-2 ~ (w^2/e) x^-1:  x_* ~", sp.simplify(Lam*e**3/w**2))
