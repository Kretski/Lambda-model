"""
stage5I_7_m15_continuation.py
================================
Parameter continuation of the q=2 branch from its validated pole at
m=-14 down through m=-15, tracking the pole via Newton continuation
(seeded from the previous step's root) and monitoring the residual-
equations' Jacobian conditioning -- to distinguish a genuine
fold/saddle-node branch termination (Jacobian singular, |Res| grows
near a critical m) from the pipeline simply failing to converge
(Jacobian well-conditioned throughout, solver just needs a better
seed).

Uses safe_complex_res() (from stage5I_4_full_m_scan_v6_FIXED, which
wraps resonance_factor_v2 from stage5I_3_resonance_v2) as the SAME
residual function that validated every other pole in v6_FIXED's
results -- no new physics, no new tolerance, no reimplementation.

CAVEAT (unverified assumption, flagged rather than silently assumed):
this script calls find_light_ring(m) at FRACTIONAL m values
(m=-14.01, -14.02, ... down to -15.00) as part of the continuation.
find_light_ring()'s internals were not inspected here -- if it
assumes integer m anywhere internally, this will fail loudly (an
exception or a clearly wrong r_sp/omega_lr) rather than silently
producing bad numbers, since every step prints its inputs and
degrades to a clean stop on failure. If it errors immediately at the
first fractional step, that itself is useful information: it means
m must be treated as a discrete quantum number in that function and
this continuation approach needs adapting (e.g. re-deriving r_sp
directly instead of via find_light_ring), not that anything here is
wrong.

Stops as soon as EITHER root() reports failure OR |Res| exceeds
1e-4 (well above the 1e-8 validation threshold used elsewhere, but a
useful early-warning breakdown point) -- whichever comes first -- so
the printed trace shows exactly where continuation broke down.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import root

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stage5I_3_resonance_v2 import find_light_ring
from stage5I_4_full_m_scan_v6_FIXED import safe_complex_res

# Validated q=2 pole at m=-14, from v6_FIXED's own results.
M_START = -14
F_RE_START = 9.39955096
F_IM_START = -0.30861804

M_END = -15.0
DM = -0.01  # continuation step; direction is toward m=-15

ROOT_XTOL = 1.0e-12
ROOT_MAXFEV = 300
JAC_EPS = 1.0e-5  # finite-difference step (Hz) for the Jacobian
BREAKDOWN_ABS_RES = 1.0e-4  # stop continuation if |Res| exceeds this


def equations(x, m, r_sp, omega_lr):
    f_re, f_im = float(x[0]), float(x[1])
    z = safe_complex_res(f_re, f_im, m, r_sp, omega_lr)
    if z is None:
        return np.array([1.0e3, 1.0e3])
    return np.array([z.real, z.imag])


def numerical_jacobian(x, m, r_sp, omega_lr, eps=JAC_EPS):
    """2x2 finite-difference Jacobian of equations() at x, w.r.t.
    (Re f, Im f). Column j is d(equations)/dx_j."""
    J = np.zeros((2, 2))
    f0 = equations(x, m, r_sp, omega_lr)
    for j in range(2):
        xp = x.copy()
        xp[j] += eps
        fp = equations(xp, m, r_sp, omega_lr)
        J[:, j] = (fp - f0) / eps
    return J


def main():
    print("=" * 100)
    print("STAGE 5I -- q=2 BRANCH CONTINUATION FROM m=-14 TOWARD m=-15")
    print("=" * 100)
    print()

    m = float(M_START)
    x = np.array([F_RE_START, F_IM_START])

    rows = []
    while m >= M_END - 1.0e-9:
        try:
            r_sp, omega_lr = find_light_ring(m)
        except Exception as exc:
            print(f"m={m:+.4f}  find_light_ring RAISED {type(exc).__name__}: {exc}")
            print("    -> stopping: find_light_ring likely assumes integer m "
                  "(see module docstring caveat).")
            break

        if r_sp is None:
            print(f"m={m:+.4f}  ERROR: no light ring found -- stopping continuation.")
            break

        try:
            result = root(lambda xx: equations(xx, m, r_sp, omega_lr), x0=x,
                          method="hybr",
                          options={"xtol": ROOT_XTOL, "maxfev": ROOT_MAXFEV})
        except (ArithmeticError, FloatingPointError, OverflowError,
                ValueError, RuntimeError) as exc:
            print(f"m={m:+.4f}  root() RAISED {type(exc).__name__}: {exc} -- stopping.")
            break

        x_new = np.asarray(result.x, dtype=float)
        z_final = safe_complex_res(x_new[0], x_new[1], m, r_sp, omega_lr)
        abs_res = abs(z_final) if z_final is not None else np.nan

        J = numerical_jacobian(x_new, m, r_sp, omega_lr)
        try:
            det_J = np.linalg.det(J)
            cond_J = np.linalg.cond(J)
        except np.linalg.LinAlgError:
            det_J, cond_J = np.nan, np.nan

        print(f"m={m:+.4f}  f_re={x_new[0]:.8f}  f_im={x_new[1]:.8f}  "
              f"|Res|={abs_res:.4e}  success={result.success}  "
              f"det(J)={det_J:.4e}  cond(J)={cond_J:.4e}")

        rows.append({"m": m, "f_re": x_new[0], "f_im": x_new[1],
                     "abs_res": abs_res, "success": bool(result.success),
                     "det_J": det_J, "cond_J": cond_J})

        broke_down = (not result.success) or (np.isfinite(abs_res) and abs_res > BREAKDOWN_ABS_RES)
        if broke_down:
            print(f"    -> continuation broke down at m={m:.4f} "
                  f"(success={result.success}, |Res|={abs_res:.4e})")
            break

        x = x_new
        m += DM

    print()
    print("=" * 100)
    print("HOW TO READ THIS:")
    print("  - a genuine fold/saddle-node termination shows det(J) -> 0")
    print("    (equivalently cond(J) -> very large) BEFORE |Res| blows up --")
    print("    the root literally runs out of room to exist, not just hard to find")
    print("  - if |Res| stays small and det(J) stays O(1) all the way to m=-15 and")
    print("    the loop completes cleanly, the branch continues fine end-to-end --")
    print("    meaning the earlier all-REJECTED discovery-grid result was a")
    print("    seeding/search-space issue specific to that grid, not a real")
    print("    termination")
    print("  - an abrupt jump in det(J)/cond(J) with NO gradual trend beforehand")
    print("    is more consistent with the solver losing the branch than with a")
    print("    genuine physical fold -- a real fold should show a smooth approach")


if __name__ == "__main__":
    main()
