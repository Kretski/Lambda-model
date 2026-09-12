"""
stage5I_8_critical_m_refinement.py
=====================================
Fine-resolution continuation across the critical region identified by
stage5I_7_m15_continuation.py: cond(J) was ~1.0018 (flat) through
m=-14.77, then jumped to 1.3e5 at m=-14.78 and 4.2e5 at m=-14.79
(where root() reported success=False and continuation stopped). That
run used dm=0.01 and a single fixed Jacobian finite-difference step
(JAC_EPS=1e-5) -- too coarse in m to see HOW the transition happens,
and with no check on whether the Jacobian itself is numerically
trustworthy that close to the transition.

This script:
  1. Steps m in [-14.77, -14.80] at dm=5e-4 (60x finer than before).
  2. Reports SINGULAR VALUES (via SVD) of the residual-equations
     Jacobian, not just det(J) -- a real fold shows sigma_min -> 0
     while sigma_max stays finite; det(J) alone (product of both
     singular values, signed) can swing wildly from sign/rounding
     interactions even when sigma_min is behaving smoothly, which is
     exactly the kind of artifact flagged after the previous run.
  3. At EVERY step, computes the Jacobian at three different
     finite-difference step sizes (1e-4, 1e-5, 1e-6 Hz) and reports
     all three sigma_min values side by side. If they agree to ~1-2
     significant figures, the Jacobian estimate is trustworthy at that
     m. If they disagree sharply, that alone is a warning that the
     residual function's own numerical noise floor (from the nested
     radial_action_complex continuation) is being probed, not real
     curvature -- and no singularity claim should be based on that
     step's numbers.

Same residual function (safe_complex_res) and same root() settings as
stage5I_7_m15_continuation.py -- no new physics, no new tolerance.
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

# Seed: the last well-converged point from stage5I_7 (m=-14.77, |Res|~3e-14)
M_START = -14.7700
F_RE_START = 9.61008156
F_IM_START = -0.30301171

M_END = -14.8000
DM = -5.0e-4

ROOT_XTOL = 1.0e-12
ROOT_MAXFEV = 300
JAC_EPS_LIST = [1.0e-4, 1.0e-5, 1.0e-6]
BREAKDOWN_ABS_RES = 1.0e-3


def equations(x, m, r_sp, omega_lr):
    f_re, f_im = float(x[0]), float(x[1])
    z = safe_complex_res(f_re, f_im, m, r_sp, omega_lr)
    if z is None:
        return np.array([1.0e3, 1.0e3])
    return np.array([z.real, z.imag])


def numerical_jacobian(x, m, r_sp, omega_lr, eps):
    J = np.zeros((2, 2))
    f0 = equations(x, m, r_sp, omega_lr)
    for j in range(2):
        xp = x.copy()
        xp[j] += eps
        fp = equations(xp, m, r_sp, omega_lr)
        J[:, j] = (fp - f0) / eps
    return J


def main():
    print("=" * 110)
    print("STAGE 5I -- FINE CRITICAL-REGION REFINEMENT, m in "
          f"[{M_START}, {M_END}], dm={DM}")
    print("=" * 110)
    print()

    m = float(M_START)
    x = np.array([F_RE_START, F_IM_START])

    header = (f"{'m':>10}  {'f_re':>12}  {'f_im':>12}  {'|Res|':>10}  "
              f"{'succ':>5}  " +
              "  ".join(f"sigma_min@{eps:.0e}" for eps in JAC_EPS_LIST))
    print(header)

    while m >= M_END - 1.0e-9:
        r_sp, omega_lr = find_light_ring(m)
        if r_sp is None:
            print(f"m={m:+.4f}  ERROR: no light ring found -- stopping.")
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

        sigma_mins = []
        sigma_maxs = []
        for eps in JAC_EPS_LIST:
            J = numerical_jacobian(x_new, m, r_sp, omega_lr, eps)
            try:
                svals = np.linalg.svd(J, compute_uv=False)
                sigma_mins.append(float(np.min(svals)))
                sigma_maxs.append(float(np.max(svals)))
            except np.linalg.LinAlgError:
                sigma_mins.append(np.nan)
                sigma_maxs.append(np.nan)

        sig_str = "  ".join(f"{s:.4e}" for s in sigma_mins)
        print(f"{m:+10.4f}  {x_new[0]:12.6f}  {x_new[1]:12.6f}  "
              f"{abs_res:10.3e}  {str(result.success):>5}  {sig_str}")

        # Flag Jacobian-estimate disagreement (candidate numerical-noise regime)
        finite = [s for s in sigma_mins if np.isfinite(s) and s > 0]
        if len(finite) >= 2 and (max(finite) / min(finite) > 10.0):
            print(f"    WARNING: sigma_min estimates disagree by "
                  f">10x across JAC_EPS at m={m:.4f} -- Jacobian is not "
                  f"trustworthy here, do not read a singularity into this row.")

        # Break on the residual itself, not on hybr's internal success flag:
        # success=True does not guarantee a small residual, and success=False
        # does not guarantee a bad one. result.success is still printed above
        # as diagnostic information, but is not used to decide breakdown.
        broke_down = (not np.isfinite(abs_res)) or (abs_res > BREAKDOWN_ABS_RES)
        if broke_down:
            print(f"    -> continuation broke down at m={m:.4f} "
                  f"(success={result.success}, |Res|={abs_res:.4e})")
            break

        x = x_new
        m += DM

    print()
    print("=" * 110)
    print("HOW TO READ THIS:")
    print("  - trust a sigma_min trend only where the three JAC_EPS estimates agree")
    print("    (no WARNING line) -- rows with a WARNING are testing numerical noise,")
    print("    not curvature, and should be excluded from any fold/no-fold conclusion")
    print("  - among the trustworthy rows, a genuine fold shows sigma_min decreasing")
    print("    smoothly toward 0 while sigma_max stays roughly constant")
    print("  - an abrupt jump in |Res| with no corresponding smooth sigma_min decay in")
    print("    the trustworthy rows beforehand still points to solver/branch-tracking")
    print("    failure rather than a resolved physical fold -- pseudo-arclength")
    print("    continuation (parametrizing by arc length instead of m) remains the")
    print("    test that can actually distinguish the two")


if __name__ == "__main__":
    main()
