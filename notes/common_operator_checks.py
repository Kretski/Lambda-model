#!/usr/bin/env python3
"""Checks for notes/COMMON_GEOMETRIC_OPERATOR.md (flat space, preferred frame u = d/dt, c = 1).
Each Lagrangian is varied with sympy's Euler-Lagrange routine and evaluated on plane waves.
1) scalar:  L = 1/2 phi_t^2 - 1/2 |grad phi|^2 - Lam_s/2 (lap phi)^2
2) photon:  L = 1/2 |E|^2 - 1/2 |B|^2 - Lam_g/2 sum_{k,j} (d_k B_j)^2,  E = -grad A0 - A_t, B = curl A
   (built from E and B only, hence gauge invariant; both transverse polarisations checked; no birefringence)
3) tensor:  L = 1/2 h_t^2 - 1/2 |grad h|^2 - Lam_T/2 (lap h)^2 for each TT polarisation amplitude h
   (the M_Pl^2/8 normalisation and the sum over i,j reduce to this per polarisation)."""
import sympy as sp
t, x, y, z, w, k = sp.symbols('t x y z omega k', real=True)
Ls, Lg, LT = sp.symbols('Lambda_phi Lambda_gamma Lambda_g', positive=True)
X = [t, x, y, z]; S = [x, y, z]
lap = lambda f: sum(sp.diff(f, v, 2) for v in S)
pw = sp.exp(sp.I*(k*z - w*t))   # propagation along z

def dispersion(L, fields, test):
    eqs = sp.euler_equations(L, fields, X)
    res = [sp.simplify((e.lhs.subs(test).doit())/pw) for e in eqs]
    return [sp.factor(r) for r in res if r != 0]

# 1) scalar
phi = sp.Function('phi')(*X)
L1 = sp.Rational(1,2)*sp.diff(phi,t)**2 - sp.Rational(1,2)*sum(sp.diff(phi,v)**2 for v in S) - Ls/2*lap(phi)**2
print("1) scalar:", dispersion(L1, [phi], {phi: pw}))

# 2) photon
A0, Ax, Ay, Az = [sp.Function(n)(*X) for n in ('A0','Ax','Ay','Az')]
A = [Ax, Ay, Az]
E = [-sp.diff(A0, S[i]) - sp.diff(A[i], t) for i in range(3)]
B = [sp.diff(Az, y) - sp.diff(Ay, z), sp.diff(Ax, z) - sp.diff(Az, x), sp.diff(Ay, x) - sp.diff(Ax, y)]
L2 = sp.Rational(1,2)*sum(e**2 for e in E) - sp.Rational(1,2)*sum(b**2 for b in B) \
     - Lg/2*sum(sp.diff(b, v)**2 for b in B for v in S)
# gauge invariance: L built from E, B; check explicitly under A_mu -> A_mu + d_mu chi
chi = sp.Function('chi')(*X)
shift = {A0: A0 + sp.diff(chi, t), Ax: Ax - sp.diff(chi, x), Ay: Ay - sp.diff(chi, y), Az: Az - sp.diff(chi, z)}
print("2) photon: gauge variation of L =", sp.simplify((L2.subs(shift).doit() - L2).expand()))
for name, test in (("x-polarised", {A0: 0, Ax: pw, Ay: 0, Az: 0}), ("y-polarised", {A0: 0, Ax: 0, Ay: pw, Az: 0})):
    print(f"   {name}:", dispersion(L2, [A0, Ax, Ay, Az], test))

# 3) tensor (one TT polarisation amplitude)
h = sp.Function('h')(*X)
L3 = sp.Rational(1,2)*sp.diff(h,t)**2 - sp.Rational(1,2)*sum(sp.diff(h,v)**2 for v in S) - LT/2*lap(h)**2
print("3) tensor:", dispersion(L3, [h], {h: pw}))
