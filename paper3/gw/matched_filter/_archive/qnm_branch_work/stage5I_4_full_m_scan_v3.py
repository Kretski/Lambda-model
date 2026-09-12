"""
stage5I_4_full_m_scan_v3.py

Stage 5I-4: adaptive resonance-search redesign.

Physics is taken unchanged from stage5I_3_resonance_v2.py.
Only the search strategy is redesigned:

    coarse scan
      -> cheap radial evaluation
      -> local minima
      -> higher-accuracy real refinement
      -> complex Nelder-Mead
      -> true Re/Im root polish
      -> pole clustering

No experimental frequencies are used as seeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar, root

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stage5I_3_resonance_v2 import (
    R_B,
    find_light_ring,
    omega_D_plus_at_p0,
    reflection_coefficient,
    radial_action_real,
    resonance_factor as resonance_factor_v2,
    refine_complex_resonance as refine_complex_v2,
)


# ---------------------------------------------------------------------------
# SEARCH SETTINGS
# ---------------------------------------------------------------------------

# Cheap discovery stage.
COARSE_N = 350
COARSE_RADIAL_N = 64
COARSE_TURNING_SCAN = 1000

# Real-axis refinement.
REAL_REFINE_RADIAL_N = 128
REAL_REFINE_TURNING_SCAN = 1800

# Complex stage uses the validated v2 implementation.
TOP_N = 12
LOCAL_HALF_WIDTH = 0.12
MIN_POLE_SEPARATION_HZ = 0.015
DEFAULT_IM0 = -0.005


# ---------------------------------------------------------------------------
# CHEAPER OUTER TURNING-POINT SEARCH
# ---------------------------------------------------------------------------

def find_last_scattering_point_fast(
    omega,
    m,
    r_sp,
    n_scan=1000,
):
    """
    Same physical definition as v2:
    below the light ring, choose the outermost real turning point.

    Only the bracketing grid is reduced during the cheap search.
    The root itself is still solved with Brent's method.
    """
    omega = float(np.real(omega))
    omega_lr = omega_D_plus_at_p0(m, r_sp)

    if omega >= omega_lr:
        return float(r_sp)

    r_grid = np.linspace(5.0e-3, R_B, int(n_scan))
    values = np.array([
        omega_D_plus_at_p0(m, r) - omega
        for r in r_grid
    ])

    # Search from outside inward: first sign change = outermost root.
    for i in range(len(r_grid) - 2, -1, -1):
        a = values[i]
        b = values[i + 1]

        if not (np.isfinite(a) and np.isfinite(b)):
            continue

        if a == 0.0:
            return float(r_grid[i])

        if a * b < 0.0:
            try:
                return float(brentq(
                    lambda r: omega_D_plus_at_p0(m, r) - omega,
                    r_grid[i],
                    r_grid[i + 1],
                    xtol=1.0e-11,
                    rtol=1.0e-11,
                ))
            except (ValueError, RuntimeError):
                return None

    return None


# ---------------------------------------------------------------------------
# REAL-AXIS RESONANCE FACTOR WITH CONTROLLED RADIAL RESOLUTION
# ---------------------------------------------------------------------------

def resonance_factor_real_fast(
    f,
    m,
    r_sp,
    omega_lr,
    radial_n=COARSE_RADIAL_N,
    turning_scan=COARSE_TURNING_SCAN,
):
    """
    Exact Eq. (17) physics on the real axis, with controllable numerical
    resolution.  This does NOT replace solve_p() or alter the Hamiltonian.
    """
    omega = 2.0 * np.pi * float(f)

    r_minus = find_last_scattering_point_fast(
        omega,
        m,
        r_sp,
        n_scan=turning_scan,
    )

    if r_minus is None:
        return None

    try:
        R = reflection_coefficient(
            omega + 0j,
            m,
            r_sp,
            omega_lr,
        )

        if not (np.isfinite(R.real) and np.isfinite(R.imag)):
            return None

        action = radial_action_real(
            omega,
            m,
            r_minus,
            n=radial_n,
        )

        z = R * np.exp(2.0j * action) - 1.0

        if not (np.isfinite(z.real) and np.isfinite(z.imag)):
            return None

        return z

    except (
        ArithmeticError,
        FloatingPointError,
        OverflowError,
        ValueError,
        RuntimeError,
    ):
        return None


# ---------------------------------------------------------------------------
# COARSE SCAN
# ---------------------------------------------------------------------------

def coarse_scan(
    m,
    f_min,
    f_max,
    n_coarse=COARSE_N,
):
    r_sp, omega_lr = find_light_ring(m)

    if r_sp is None:
        return [], None, None

    f_grid = np.linspace(f_min, f_max, n_coarse)
    abs_res = np.full(n_coarse, np.inf)

    print(
        f"[m={m}] coarse scan: {n_coarse} points | "
        f"radial_n={COARSE_RADIAL_N} | "
        f"turning_scan={COARSE_TURNING_SCAN}"
    )

    for i, f in enumerate(f_grid):
        z = resonance_factor_real_fast(
            f,
            m,
            r_sp,
            omega_lr,
            radial_n=COARSE_RADIAL_N,
            turning_scan=COARSE_TURNING_SCAN,
        )

        if z is not None:
            abs_res[i] = abs(z)

        if (i + 1) % max(1, n_coarse // 10) == 0:
            print(
                f"    {i + 1:4d}/{n_coarse} "
                f"({100.0 * (i + 1) / n_coarse:5.1f}%)"
            )

    finite = np.isfinite(abs_res)
    if not np.any(finite):
        return [], r_sp, omega_lr

    candidates = []

    for i in range(1, n_coarse - 1):
        if not np.isfinite(abs_res[i]):
            continue

        if abs_res[i] <= abs_res[i - 1] and abs_res[i] <= abs_res[i + 1]:
            candidates.append(i)

    # Always keep the globally best finite point as fallback.
    finite_idx = np.flatnonzero(finite)
    best_idx = finite_idx[np.argmin(abs_res[finite])]

    if int(best_idx) not in candidates:
        candidates.append(int(best_idx))

    candidates.sort(key=lambda i: abs_res[i])
    candidates = candidates[:TOP_N]

    return [
        {
            "f_coarse": float(f_grid[i]),
            "abs_res_coarse": float(abs_res[i]),
        }
        for i in candidates
    ], r_sp, omega_lr


# ---------------------------------------------------------------------------
# REAL-AXIS LOCAL REFINEMENT
# ---------------------------------------------------------------------------

def refine_real_candidate(
    candidate,
    m,
    f_min,
    f_max,
    r_sp,
    omega_lr,
    half_width=LOCAL_HALF_WIDTH,
):
    f0 = candidate["f_coarse"]

    lo = max(f_min, f0 - half_width)
    hi = min(f_max, f0 + half_width)

    def objective(f):
        z = resonance_factor_real_fast(
            f,
            m,
            r_sp,
            omega_lr,
            radial_n=REAL_REFINE_RADIAL_N,
            turning_scan=REAL_REFINE_TURNING_SCAN,
        )

        if z is None:
            return 1.0e100

        value = abs(z)
        return float(value) if np.isfinite(value) else 1.0e100

    result = minimize_scalar(
        objective,
        bounds=(lo, hi),
        method="bounded",
        options={
            "xatol": 2.0e-9,
            "maxiter": 120,
        },
    )

    if not result.success or not np.isfinite(result.fun):
        return None

    return {
        "f_real": float(result.x),
        "abs_res": float(result.fun),
        "coarse_f": float(f0),
    }


def cluster_real_candidates(candidates):
    candidates = sorted(candidates, key=lambda x: x["f_real"])
    unique = []

    for c in candidates:
        if not unique:
            unique.append(c)
            continue

        if abs(c["f_real"] - unique[-1]["f_real"]) < MIN_POLE_SEPARATION_HZ:
            if c["abs_res"] < unique[-1]["abs_res"]:
                unique[-1] = c
        else:
            unique.append(c)

    return unique


# ---------------------------------------------------------------------------
# COMPLEX REFINEMENT + TRUE ROOT POLISH
# ---------------------------------------------------------------------------

def complex_refine_and_polish(
    candidate,
    m,
    r_sp,
    omega_lr,
    im0=DEFAULT_IM0,
):
    """
    First use the validated v2 complex minimizer.
    Then solve Re(Res)=0, Im(Res)=0 directly.
    """
    f0 = candidate["f_real"]

    refined = refine_complex_v2(
        f0=f0,
        im0=im0,
        resonance_factor=resonance_factor_v2,
        m=m,
        re_half_width=LOCAL_HALF_WIDTH,
        im_min=-1.5,
        im_max=0.0,
    )

    if refined is None:
        return None

    x0 = np.array([refined["f_re"], refined["f_im"]], dtype=float)

    def equations(x):
        f_re = float(x[0])
        f_im = float(x[1])

        if abs(f_re - f0) > 4.0 * LOCAL_HALF_WIDTH:
            return np.array([1.0e3, 1.0e3])

        if not (-1.5 <= f_im <= 0.0):
            return np.array([1.0e3, 1.0e3])

        omega = 2.0 * np.pi * (f_re + 1j * f_im)

        try:
            z = resonance_factor_v2(
                omega=omega,
                m=m,
                r_sp=r_sp,
                omega_lr=omega_lr,
                complex_action=True,
            )
        except Exception:
            return np.array([1.0e3, 1.0e3])

        if z is None or not (
            np.isfinite(z.real) and np.isfinite(z.imag)
        ):
            return np.array([1.0e3, 1.0e3])

        return np.array([z.real, z.imag])

    polished = root(
        equations,
        x0,
        method="hybr",
        options={
            "xtol": 1.0e-10,
            "maxfev": 100,
        },
    )

    if polished.success:
        f_re = float(polished.x[0])
        f_im = float(polished.x[1])
        residual = float(np.linalg.norm(equations(polished.x)))

        if (
            abs(f_re - f0) <= 4.0 * LOCAL_HALF_WIDTH
            and -1.5 <= f_im <= 0.0
            and np.isfinite(residual)
        ):
            return {
                "f_re": f_re,
                "f_im": f_im,
                "omega_re": 2.0 * np.pi * f_re,
                "omega_im": 2.0 * np.pi * f_im,
                "abs_res": residual,
                "real_axis_abs_res": candidate["abs_res"],
                "root_success": True,
                "root_message": str(polished.message),
            }

    # Do not discard a useful Nelder-Mead result just because root polish
    # failed. Mark it explicitly.
    return {
        "f_re": refined["f_re"],
        "f_im": refined["f_im"],
        "omega_re": refined["omega_re"],
        "omega_im": refined["omega_im"],
        "abs_res": refined["abs_res"],
        "real_axis_abs_res": candidate["abs_res"],
        "root_success": False,
        "root_message": str(polished.message),
    }


# ---------------------------------------------------------------------------
# ONE m MODE
# ---------------------------------------------------------------------------

def solve_one_m(
    m,
    f_min,
    f_max,
    expected_n=4,
):
    print("=" * 78)
    print(f"STAGE 5I-4 v3   m={m}")
    print("=" * 78)

    candidates, r_sp, omega_lr = coarse_scan(
        m,
        f_min,
        f_max,
    )

    if r_sp is None:
        print("No light ring.")
        return []

    print(
        f"light ring: r_sp={r_sp * 1e3:.6f} mm, "
        f"f_lr={omega_lr / (2.0 * np.pi):.10f} Hz"
    )

    if not candidates:
        print("No coarse candidates.")
        return []

    print("\nCoarse candidates:")
    for c in candidates:
        print(
            f"    f={c['f_coarse']:.8f} Hz "
            f"|Res|={c['abs_res_coarse']:.4e}"
        )

    print("\nReal-axis local refinement:")
    real_candidates = []

    for c in candidates:
        refined = refine_real_candidate(
            c,
            m,
            f_min,
            f_max,
            r_sp,
            omega_lr,
        )

        if refined is not None:
            real_candidates.append(refined)
            print(
                f"    f={refined['f_real']:.10f} Hz "
                f"|Res|={refined['abs_res']:.4e}"
            )

    real_candidates = cluster_real_candidates(real_candidates)

    print(
        f"\nDistinct real-axis candidates: "
        f"{len(real_candidates)}"
    )

    poles = []

    print("\nComplex refinement + root polish:")

    for c in real_candidates:
        print(
            f"    starting f={c['f_real']:.10f} Hz"
        )

        pole = complex_refine_and_polish(
            c,
            m,
            r_sp,
            omega_lr,
        )

        if pole is None:
            print("        FAILED")
            continue

        poles.append(pole)

        print(
            f"        Re={pole['f_re']:.10f} Hz "
            f"Im={pole['f_im']:.10f} Hz "
            f"|Res|={pole['abs_res']:.4e} "
            f"root={pole['root_success']}"
        )

    poles.sort(key=lambda x: x["f_re"])

    unique = []
    for p in poles:
        if not unique:
            unique.append(p)
            continue

        if abs(p["f_re"] - unique[-1]["f_re"]) >= MIN_POLE_SEPARATION_HZ:
            unique.append(p)
        elif p["abs_res"] < unique[-1]["abs_res"]:
            unique[-1] = p

    unique = unique[:expected_n]

    print("\nFINAL POLES:")
    for q, p in enumerate(unique, start=1):
        print(
            f"    q={q} "
            f"f={p['f_re']:.10f} Hz "
            f"Im={p['f_im']:.10f} Hz "
            f"|Res|={p['abs_res']:.4e}"
        )

    return unique


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("STAGE 5I-4 v3 — ADAPTIVE RESONANCE SEARCH")
    print("=" * 78)
    print()
    print("Physics source: stage5I_3_resonance_v2.py")
    print("No experimental frequencies are used as seeds.")
    print()
    print("Search:")
    print("    coarse real scan")
    print("      -> local minima")
    print("      -> higher-accuracy real refinement")
    print("      -> complex Nelder-Mead")
    print("      -> Re/Im root polish")
    print("      -> pole clustering")
    print()

    # First run only the m=-12 regression/performance test.
    solve_one_m(
        m=-12,
        f_min=7.0,
        f_max=10.5,
        expected_n=4,
    )

    print()
    print("=" * 78)
    print("STAGE 5I-4 v3 COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
