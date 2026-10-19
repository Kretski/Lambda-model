"""
stage5I_4_full_m_scan_v6_FIXED.py
=================================

STAGE 5I-4 v6 - FULL m-SCAN WITH COMPLEX-PLANE OVERTONE DISCOVERY

v6 changes vs v5
-----------------
v5 found q=1 (fundamental) reliably for every m, but found ZERO
overtones (q=2,3,4): 18/72 poles total.

Diagnosis (Stage 5I-4, two-part):

  1. radial_action_complex() and find_turning_point_complex() in
     stage5I_3_resonance_v2.py both seeded their Newton continuations
     AT or immediately adjacent to the light-ring saddle point
     (r_minus = r_sp for omega above f_lr, the overtone regime),
     where the relevant derivatives vanish by definition (Eq. 12).
     FIXED in stage5I_3_resonance_v2.py (three fixes total -- see
     that file's docstrings and STAGE5I_FINAL_STATUS.md for details):
       - radial_action_complex(): local saddle-point (hyperbolic)
         expansion for the degenerate-H_r first-step seed.
       - find_turning_point_complex(): incremental continuation in
         Im(omega) instead of a single-shot solve.
       - resonance_factor(): conceptual fix -- r_minus=r_sp above f_lr
         is a fixed reference point, not a root to continue.

  2. EVEN WITH those fixes, real-axis-seeded local refinement still
     failed to find overtones (broad, damped features don't produce a
     narrow real-axis |Res| dip). FIX (this file): genuine
     COMPLEX-PLANE discovery -- scan directly in (f_re, f_im) above
     f_lr, at Im values matching the paper's stated overtone damping
     scale (up to -1.2 Hz), then refine each local minimum found
     there.

RESULT (as run by the user, 2026-08-26): 39/72 poles found, including
genuine overtone (q>=2) branches for the first time. See
STAGE5I_FINAL_STATUS.md for the full validation/branch-assignment
follow-up (stage5I_6_branch_assignment.py, stage5I_7_validation_report.py).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar, minimize, root


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from stage5I_3_resonance_v2 import (
        R_B,
        find_light_ring,
        omega_D_plus_at_p0,
        radial_action_real,
        reflection_coefficient,
        resonance_factor as resonance_factor_v2,
    )
except ImportError as exc:
    raise ImportError(
        "\nCould not import stage5I_3_resonance_v2.py\n\n"
        "Make sure the FIXED version (with the Stage 5I-4 saddle-point,\n"
        "incremental-continuation, and conceptual r_minus=r_sp fixes) is\n"
        "in this directory, alongside:\n\n"
        "    stage5I_1_2_dispersion_radial_p.py\n"
        "    stage5I_3_resonance_v2.py\n"
        "    stage5I_4_full_m_scan_v6_FIXED.py\n"
    ) from exc


EXP_B = {
    -21: [10.53127805, 11.31559458, 11.73305339, 12.0],
    -20: [10.30228351, 11.07509152, 11.48642482, 11.79180863],
    -19: [10.07028344, 10.83337372, 11.23953467, 11.54107841],
    -18: [9.840683264, 10.58962136, 10.99149058, 11.29593696],
    -17: [9.600492823, 10.34296042, 10.74739399, 11.04921008],
    -16: [9.355011082, 10.09843672, 10.50612174, 10.80589014],
    -15: [9.103267196, 9.848948391, 10.26056441, 10.55883689],
    -14: [8.844144355, 9.599296405, 10.01552194, 10.31282590],
    -13: [8.582278857, 9.342201757, 9.76372149, 10.07244017],
    -12: [8.304259799, 9.082091798, 9.515540164, 9.824297904],
    -11: [8.020133041, 8.811238183, 9.257350105, 9.578550689],
    -10: [7.721731873, 8.533427811, 8.992990952, 9.327218691],
    -9: [7.406262072, 8.246009901, 8.725865804, 9.067763134],
    -8: [7.076173999, 7.945602232, 8.452768701, 8.808992769],
    -7: [6.714687044, 7.633791333, 8.169428269, 8.540722055],
    -6: [6.333524665, 7.298338000, 7.869851632, 8.269296643],
    -5: [5.910999730, 6.948591527, 7.563690856, 7.986183324],
    -4: [5.437524431, 6.570039130, 7.236964453, 7.696262081],
}

M_VALUES = list(range(-21, -3))
EXPECTED_POLES = 4

COARSE_N = 400
COARSE_RADIAL_N = 64
COARSE_TURNING_SCAN = 400

REAL_REFINE_RADIAL_N = 64
REAL_REFINE_TURNING_SCAN = 800
REAL_REFINE_HALF_WIDTH = 0.10

REAL_CANDIDATE_SEPARATION_HZ = 0.025
MIN_POLE_SEPARATION_HZ = 0.015

COMPLEX_DISCOVERY_RE_ABOVE_FLR = 3.5
COMPLEX_DISCOVERY_RE_POINTS = 60
COMPLEX_DISCOVERY_IM_VALUES = [-0.05, -0.1, -0.2, -0.35, -0.5, -0.7, -0.9, -1.2]
COMPLEX_DISCOVERY_TOP_N = 8

COMPLEX_RE_HALF_WIDTH = 0.20
COMPLEX_IM_MIN = -1.5
COMPLEX_IM_MAX = 0.0
FALLBACK_IM_SHIFTS = [0.0, -0.02, -0.05, -0.1]

ROOT_XTOL = 1.0e-12
ROOT_MAXFEV = 150
MAX_VALIDATED_ABS_RES = 1.0e-8
MAX_ROOT_SHIFT_HZ = 0.30

REGRESSION_M = -12
BENCHMARK_F_RE = 8.3066307169
BENCHMARK_F_IM = -0.0005664350
BENCHMARK_F_TOL = 1.0e-5
BENCHMARK_RES_TOL = 1.0e-8

OUTPUT_CSV = HERE / "stage5I_4_full_m_scan_v6_FIXED_results.csv"


def find_last_scattering_point_fast(omega, m, r_sp, n_scan=COARSE_TURNING_SCAN):
    omega = float(np.real(omega))
    omega_lr = omega_D_plus_at_p0(m, r_sp)
    if omega >= omega_lr:
        return float(r_sp)
    r_grid = np.linspace(5.0e-3, R_B, int(n_scan))
    values = np.array([omega_D_plus_at_p0(m, r) - omega for r in r_grid])
    for i in range(len(r_grid) - 2, -1, -1):
        a, b = values[i], values[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a == 0.0:
            return float(r_grid[i])
        if a * b < 0.0:
            try:
                return float(brentq(
                    lambda r: omega_D_plus_at_p0(m, r) - omega,
                    r_grid[i], r_grid[i + 1], xtol=1.0e-11, rtol=1.0e-11))
            except (ValueError, RuntimeError):
                return None
    return None


def resonance_factor_real_fast(f, m, r_sp, omega_lr,
                                 radial_n=COARSE_RADIAL_N,
                                 turning_scan=COARSE_TURNING_SCAN):
    omega = 2.0 * np.pi * float(f)
    r_minus = find_last_scattering_point_fast(omega, m, r_sp, n_scan=turning_scan)
    if r_minus is None:
        return None
    try:
        R = reflection_coefficient(omega + 0j, m, r_sp, omega_lr)
        if not (np.isfinite(R.real) and np.isfinite(R.imag)):
            return None
        action = radial_action_real(omega, m, r_minus, n=radial_n)
        z = R * np.exp(2.0j * action) - 1.0
        if not (np.isfinite(z.real) and np.isfinite(z.imag)):
            return None
        return z
    except (ArithmeticError, FloatingPointError, OverflowError,
            ValueError, RuntimeError):
        return None


def safe_complex_res(f_re, f_im, m, r_sp, omega_lr):
    if not (np.isfinite(f_re) and np.isfinite(f_im)):
        return None
    if not (COMPLEX_IM_MIN <= f_im <= COMPLEX_IM_MAX):
        return None
    omega = 2.0 * np.pi * (f_re + 1j * f_im)
    try:
        z = resonance_factor_v2(omega=omega, m=m, r_sp=r_sp,
                                  omega_lr=omega_lr, complex_action=True)
    except (ArithmeticError, FloatingPointError, OverflowError,
            ValueError, RuntimeError):
        return None
    if z is None:
        return None
    if not (np.isfinite(z.real) and np.isfinite(z.imag)):
        return None
    return z


def coarse_scan(m, f_min, f_max):
    r_sp, omega_lr = find_light_ring(m)
    if r_sp is None:
        return [], None, None
    f_grid = np.linspace(f_min, f_max, COARSE_N)
    abs_res = np.full(COARSE_N, np.inf, dtype=float)
    print(f"[m={m}] coarse real-axis scan: {COARSE_N} points | "
          f"radial_n={COARSE_RADIAL_N} | turning_scan={COARSE_TURNING_SCAN}")
    for i, f in enumerate(f_grid):
        z = resonance_factor_real_fast(f, m, r_sp, omega_lr,
                                         radial_n=COARSE_RADIAL_N,
                                         turning_scan=COARSE_TURNING_SCAN)
        if z is not None:
            value = abs(z)
            if np.isfinite(value):
                abs_res[i] = value
        if (i + 1) % max(1, COARSE_N // 10) == 0:
            print(f"    {i + 1:4d}/{COARSE_N} ({100.0*(i+1)/COARSE_N:5.1f}%)")
    finite = np.isfinite(abs_res)
    if not np.any(finite):
        return [], r_sp, omega_lr
    candidates = []
    for i in range(1, COARSE_N - 1):
        if not np.isfinite(abs_res[i]):
            continue
        left, center, right = abs_res[i - 1], abs_res[i], abs_res[i + 1]
        if not (np.isfinite(left) and np.isfinite(right)):
            continue
        if center <= left and center <= right:
            candidates.append({"f_coarse": float(f_grid[i]),
                                "abs_res_coarse": float(center)})
    finite_indices = np.flatnonzero(finite)
    best_idx = finite_indices[np.argmin(abs_res[finite])]
    if not candidates:
        candidates.append({"f_coarse": float(f_grid[best_idx]),
                            "abs_res_coarse": float(abs_res[best_idx])})
    candidates.sort(key=lambda x: x["f_coarse"])
    clustered = []
    for candidate in candidates:
        if not clustered:
            clustered.append(candidate)
            continue
        if abs(candidate["f_coarse"] - clustered[-1]["f_coarse"]) < REAL_CANDIDATE_SEPARATION_HZ:
            if candidate["abs_res_coarse"] < clustered[-1]["abs_res_coarse"]:
                clustered[-1] = candidate
        else:
            clustered.append(candidate)
    return clustered, r_sp, omega_lr


def refine_real_candidate(candidate, m, f_min, f_max, r_sp, omega_lr):
    f0 = candidate["f_coarse"]
    lo = max(f_min, f0 - REAL_REFINE_HALF_WIDTH)
    hi = min(f_max, f0 + REAL_REFINE_HALF_WIDTH)
    def objective(f):
        z = resonance_factor_real_fast(f, m, r_sp, omega_lr,
                                         radial_n=REAL_REFINE_RADIAL_N,
                                         turning_scan=REAL_REFINE_TURNING_SCAN)
        if z is None:
            return 1.0e100
        value = abs(z)
        return float(value) if np.isfinite(value) else 1.0e100
    try:
        result = minimize_scalar(objective, bounds=(lo, hi), method="bounded",
                                   options={"xatol": 2.0e-8, "maxiter": 100})
    except (ArithmeticError, FloatingPointError, OverflowError,
            ValueError, RuntimeError):
        return None
    if not result.success or not np.isfinite(result.fun):
        return None
    return {"f_real": float(result.x), "abs_res": float(result.fun),
            "coarse_f": float(f0)}


def cluster_real_candidates(candidates):
    candidates = sorted(candidates, key=lambda x: x["f_real"])
    unique = []
    for candidate in candidates:
        if not unique:
            unique.append(candidate)
            continue
        if abs(candidate["f_real"] - unique[-1]["f_real"]) < REAL_CANDIDATE_SEPARATION_HZ:
            if candidate["abs_res"] < unique[-1]["abs_res"]:
                unique[-1] = candidate
        else:
            unique.append(candidate)
    return unique


def complex_plane_discovery(m, r_sp, omega_lr, f_lr):
    f_re_grid = np.linspace(f_lr + 0.05, f_lr + COMPLEX_DISCOVERY_RE_ABOVE_FLR,
                              COMPLEX_DISCOVERY_RE_POINTS)
    print(f"[m={m}] complex-plane discovery: "
          f"Re in [{f_re_grid[0]:.3f},{f_re_grid[-1]:.3f}] Hz x "
          f"{len(COMPLEX_DISCOVERY_IM_VALUES)} Im values")
    grid_vals = np.full((len(COMPLEX_DISCOVERY_IM_VALUES), len(f_re_grid)),
                         np.inf, dtype=float)
    for j, f_im in enumerate(COMPLEX_DISCOVERY_IM_VALUES):
        for i, f_re in enumerate(f_re_grid):
            z = safe_complex_res(float(f_re), float(f_im), m, r_sp, omega_lr)
            if z is not None:
                val = abs(z)
                if np.isfinite(val):
                    grid_vals[j, i] = val
    candidates = []
    n_im = len(COMPLEX_DISCOVERY_IM_VALUES)
    n_re = len(f_re_grid)
    for j in range(n_im):
        for i in range(n_re):
            val = grid_vals[j, i]
            if not np.isfinite(val):
                continue
            neighbours = []
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    if dj == 0 and di == 0:
                        continue
                    jj, ii = j + dj, i + di
                    if 0 <= jj < n_im and 0 <= ii < n_re:
                        neighbours.append(grid_vals[jj, ii])
            if neighbours and val <= min(neighbours):
                candidates.append({"f_re": float(f_re_grid[i]),
                                    "f_im": float(COMPLEX_DISCOVERY_IM_VALUES[j]),
                                    "abs_res": float(val)})
    candidates.sort(key=lambda c: c["abs_res"])
    deduped = []
    for c in candidates:
        if all(abs(c["f_re"] - d["f_re"]) > 0.15 for d in deduped):
            deduped.append(c)
        if len(deduped) >= COMPLEX_DISCOVERY_TOP_N:
            break
    print(f"[m={m}] complex-plane candidates found: {len(deduped)}")
    for c in deduped:
        print(f"    Re={c['f_re']:.6f} Hz  Im={c['f_im']:.6f} Hz  "
              f"|Res|={c['abs_res']:.4e}")
    return deduped


def complex_minimize_from_seed(f0_re, f0_im, m, r_sp, omega_lr):
    lo_re = f0_re - COMPLEX_RE_HALF_WIDTH
    hi_re = f0_re + COMPLEX_RE_HALF_WIDTH
    def objective(x):
        f_re, f_im = float(x[0]), float(x[1])
        if not (lo_re <= f_re <= hi_re):
            return 1.0e100
        if not (COMPLEX_IM_MIN <= f_im <= COMPLEX_IM_MAX):
            return 1.0e100
        z = safe_complex_res(f_re, f_im, m, r_sp, omega_lr)
        if z is None:
            return 1.0e100
        value = abs(z)
        return float(np.log1p(value)) if np.isfinite(value) else 1.0e100
    try:
        result = minimize(objective, x0=np.array([f0_re, f0_im]),
                            method="Nelder-Mead",
                            options={"xatol": 1.0e-8, "fatol": 1.0e-10, "maxiter": 800})
    except (ArithmeticError, FloatingPointError, OverflowError,
            ValueError, RuntimeError):
        return None
    if not np.all(np.isfinite(result.x)):
        return None
    f_re, f_im = float(result.x[0]), float(result.x[1])
    if not (lo_re <= f_re <= hi_re):
        return None
    if not (COMPLEX_IM_MIN <= f_im <= COMPLEX_IM_MAX):
        return None
    z = safe_complex_res(f_re, f_im, m, r_sp, omega_lr)
    if z is None:
        return None
    abs_res = abs(z)
    if not np.isfinite(abs_res):
        return None
    return {"f_re": f_re, "f_im": f_im, "abs_res": float(abs_res),
            "nfev": int(getattr(result, "nfev", -1))}


def polish_root(seed_f_re, seed_f_im, m, r_sp, omega_lr, f0, real_axis_abs_res):
    def equations(x):
        f_re, f_im = float(x[0]), float(x[1])
        if not (np.isfinite(f_re) and np.isfinite(f_im)):
            return np.array([1.0e3, 1.0e3])
        if abs(f_re - f0) > MAX_ROOT_SHIFT_HZ:
            return np.array([1.0e3, 1.0e3])
        if f_im > COMPLEX_IM_MAX or f_im < COMPLEX_IM_MIN:
            return np.array([1.0e3, 1.0e3])
        z = safe_complex_res(f_re, f_im, m, r_sp, omega_lr)
        if z is None:
            return np.array([1.0e3, 1.0e3])
        return np.array([z.real, z.imag])
    x0 = np.array([seed_f_re, seed_f_im])
    try:
        result = root(equations, x0=x0, method="hybr",
                       options={"xtol": ROOT_XTOL, "maxfev": ROOT_MAXFEV})
    except (ArithmeticError, FloatingPointError, OverflowError,
            ValueError, RuntimeError):
        return None
    if not np.all(np.isfinite(result.x)):
        return None
    f_re, f_im = float(result.x[0]), float(result.x[1])
    z_final = safe_complex_res(f_re, f_im, m, r_sp, omega_lr)
    if z_final is None:
        return None
    abs_res = abs(z_final)
    if not result.success:
        return None
    if not np.isfinite(abs_res):
        return None
    if f_im >= 0.0:
        return None
    if abs(f_re - f0) > MAX_ROOT_SHIFT_HZ:
        return None
    if abs_res >= MAX_VALIDATED_ABS_RES:
        return None
    return {"f_re": f_re, "f_im": f_im, "abs_res": float(abs_res),
            "real_axis_abs_res": float(real_axis_abs_res),
            "root_success": True, "root_message": str(result.message),
            "root_nfev": int(getattr(result, "nfev", -1)),
            "seed_f": f0, "seed_f_re": seed_f_re, "seed_f_im": seed_f_im}


def refine_and_polish_candidate(f0_re, f0_im_hint, m, r_sp, omega_lr,
                                  real_axis_abs_res, is_complex_seed=False):
    if is_complex_seed:
        im_seeds = [f0_im_hint + shift for shift in FALLBACK_IM_SHIFTS]
    else:
        im_seeds = [f0_im_hint] + [s for s in FALLBACK_IM_SHIFTS if s != 0.0]
    validated = []
    for im_seed in im_seeds:
        print(f"        seed Im(f)={im_seed:+.6e} Hz")
        refined = complex_minimize_from_seed(f0_re, im_seed, m, r_sp, omega_lr)
        if refined is None:
            print("            Nelder-Mead failed")
            continue
        print(f"            NM: Re={refined['f_re']:.10f} Hz "
              f"Im={refined['f_im']:.10f} Hz |Res|={refined['abs_res']:.4e}")
        polished = polish_root(refined["f_re"], refined["f_im"], m, r_sp,
                                 omega_lr, f0_re, real_axis_abs_res)
        if polished is None:
            print("            hybr: REJECTED")
            continue
        print(f"            hybr: Re={polished['f_re']:.10f} Hz "
              f"Im={polished['f_im']:.10f} Hz |Res|={polished['abs_res']:.4e}")
        validated.append(polished)
    if not validated:
        return None
    validated.sort(key=lambda x: x["abs_res"])
    return validated[0]


def cluster_poles(poles):
    poles = sorted(poles, key=lambda x: (x["f_re"], x["f_im"]))
    unique = []
    for pole in poles:
        if not unique:
            unique.append(pole)
            continue
        previous = unique[-1]
        distance = np.hypot(pole["f_re"] - previous["f_re"],
                              pole["f_im"] - previous["f_im"])
        if distance < MIN_POLE_SEPARATION_HZ:
            if pole["abs_res"] < previous["abs_res"]:
                unique[-1] = pole
        else:
            unique.append(pole)
    return unique


def solve_one_m(m, f_min, f_max):
    print()
    print("=" * 78)
    print(f"STAGE 5I-4 v6   m={m}")
    print("=" * 78)
    candidates, r_sp, omega_lr = coarse_scan(m, f_min, f_max)
    if r_sp is None:
        print("No light ring.")
        return []
    f_lr = omega_lr / (2.0 * np.pi)
    print(f"light ring: r_sp={r_sp*1.0e3:.6f} mm f_lr={f_lr:.10f} Hz")
    real_poles = []
    if candidates:
        print()
        print(f"ALL real-axis candidates: {len(candidates)}")
        for candidate in candidates:
            print(f"    coarse f={candidate['f_coarse']:.8f} Hz "
                  f"|Res|={candidate['abs_res_coarse']:.4e}")
        real_candidates = []
        for candidate in candidates:
            refined = refine_real_candidate(candidate, m, f_min, f_max, r_sp, omega_lr)
            if refined is not None:
                real_candidates.append(refined)
        real_candidates = cluster_real_candidates(real_candidates)
        print()
        print(f"Distinct real candidates: {len(real_candidates)}")
        for candidate in real_candidates:
            f0 = candidate["f_real"]
            print()
            print(f"    starting real candidate f={f0:.10f} Hz")
            pole = refine_and_polish_candidate(
                f0, -5.0e-4, m, r_sp, omega_lr, candidate["abs_res"],
                is_complex_seed=False)
            if pole is None:
                print("        NO VALIDATED POLE")
                continue
            real_poles.append(pole)
    print()
    complex_candidates = complex_plane_discovery(m, r_sp, omega_lr, f_lr)
    complex_poles = []
    for cand in complex_candidates:
        print()
        print(f"    starting complex candidate "
              f"Re={cand['f_re']:.6f} Hz Im={cand['f_im']:.6f} Hz")
        pole = refine_and_polish_candidate(
            cand["f_re"], cand["f_im"], m, r_sp, omega_lr, cand["abs_res"],
            is_complex_seed=True)
        if pole is None:
            print("        NO VALIDATED POLE")
            continue
        complex_poles.append(pole)
    poles = cluster_poles(real_poles + complex_poles)
    poles.sort(key=lambda x: x["f_re"])
    print()
    print(f"Validated physical poles (all sources): {len(poles)}")
    final_poles = poles[:EXPECTED_POLES]
    print()
    print("FINAL MODEL POLES:")
    for q, pole in enumerate(final_poles, start=1):
        print(f"    q={q} f={pole['f_re']:.10f} Hz Im={pole['f_im']:.10f} Hz "
              f"|Res|={pole['abs_res']:.4e}")
    if len(final_poles) < EXPECTED_POLES:
        print()
        print(f"WARNING: only {len(final_poles)}/{EXPECTED_POLES} validated poles found.")
    return final_poles


def experimental_frequency_window():
    values = [float(f) for frequencies in EXP_B.values() for f in frequencies]
    f_min, f_max = min(values), max(values)
    margin = 0.15
    return (f_min - margin, f_max + margin)


def run_m12_regression():
    print()
    print("=" * 78)
    print("STAGE 5I-4 v6 - m=-12 REGRESSION")
    print("=" * 78)
    m = REGRESSION_M
    r_sp, omega_lr = find_light_ring(m)
    if r_sp is None:
        print("REGRESSION FAIL: no light ring.")
        return False
    f_min, f_max = 7.0, 10.5
    candidates, _, _ = coarse_scan(m, f_min, f_max)
    if not candidates:
        print("REGRESSION FAIL: no candidates.")
        return False
    real_candidates = []
    for candidate in candidates:
        refined = refine_real_candidate(candidate, m, f_min, f_max, r_sp, omega_lr)
        if refined is not None:
            real_candidates.append(refined)
    real_candidates = cluster_real_candidates(real_candidates)
    if not real_candidates:
        print("REGRESSION FAIL: no real-axis candidates.")
        return False
    candidate = min(real_candidates, key=lambda c: abs(c["f_real"] - BENCHMARK_F_RE))
    print()
    print(f"Regression candidate: {candidate['f_real']:.10f} Hz")
    pole = refine_and_polish_candidate(
        candidate["f_real"], -5.0e-4, m, r_sp, omega_lr, candidate["abs_res"],
        is_complex_seed=False)
    if pole is None:
        print("REGRESSION FAIL: no validated complex pole.")
        return False
    diff_re = abs(pole["f_re"] - BENCHMARK_F_RE)
    diff_im = abs(pole["f_im"] - BENCHMARK_F_IM)
    res_ok = pole["abs_res"] < BENCHMARK_RES_TOL
    re_ok = diff_re < BENCHMARK_F_TOL
    im_ok = diff_im < BENCHMARK_F_TOL
    print()
    print("REGRESSION RESULT:")
    print(f"    f_re = {pole['f_re']:.12f} Hz")
    print(f"    f_im = {pole['f_im']:.12f} Hz")
    print(f"    |Res| = {pole['abs_res']:.6e}")
    print(f"    |df_re| = {diff_re:.6e} Hz {'PASS' if re_ok else 'FAIL'}")
    print(f"    |df_im| = {diff_im:.6e} Hz {'PASS' if im_ok else 'FAIL'}")
    print(f"    |Res| < {BENCHMARK_RES_TOL:.0e}: {'PASS' if res_ok else 'FAIL'}")
    passed = re_ok and im_ok and res_ok
    print()
    print("REGRESSION CHECK PASSED." if passed else "REGRESSION CHECK FAILED.")
    return passed


def write_results_csv(rows, path=OUTPUT_CSV):
    fieldnames = ["m", "q", "f_model_hz", "f_model_im_hz", "abs_res",
                  "root_success", "root_nfev", "root_message", "seed_f_hz",
                  "real_axis_abs_res", "light_ring_hz", "experimental_hz",
                  "delta_hz", "relative_error_percent"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print()
    print("Results written to:")
    print(f"    {path}")


def print_summary(all_results):
    print()
    print("=" * 110)
    print("STAGE 5I-4 v6 - FINAL SUMMARY")
    print("=" * 110)
    print()
    print(f"{'m':>4} {'q':>3} {'model Re(f)':>16} {'model Im(f)':>16} "
          f"{'experiment':>16} {'delta':>14} {'|Res|':>12}")
    print("-" * 110)
    for row in all_results:
        print(f"{row['m']:>4} {row['q']:>3} {row['f_model_hz']:>16.8f} "
              f"{row['f_model_im_hz']:>16.8f} {row['experimental_hz']:>16.8f} "
              f"{row['delta_hz']:>14.8f} {row['abs_res']:>12.3e}")
    print()


def main():
    print("=" * 110)
    print("STAGE 5I-4 v6 - FULL m-SCAN WITH COMPLEX-PLANE OVERTONE DISCOVERY")
    print("=" * 110)
    print()
    print("Physics source: stage5I_3_resonance_v2.py (FIXED: saddle-point")
    print("seeding + incremental Im(omega) continuation + conceptual")
    print("r_minus=r_sp fix for the above-light-ring branch)")
    print()
    print("q=1 (fundamental): real-axis discovery, as in v5.")
    print("q=2,3,4 (overtones): NEW complex-plane discovery.")
    print()
    print("Experimental frequencies are NOT used as seeds anywhere.")
    print()
    regression_ok = run_m12_regression()
    if not regression_ok:
        print()
        print("=" * 78)
        print("FULL m-SCAN ABORTED")
        print("=" * 78)
        print()
        print("The m=-12 regression gate failed.")
        print("Do NOT interpret the full scan as validated.")
        return 1
    f_min, f_max = experimental_frequency_window()
    print()
    print("Global real-axis scan interval:")
    print(f"    [{f_min:.8f}, {f_max:.8f}] Hz")
    print()
    all_results = []
    for m in M_VALUES:
        experimental = EXP_B.get(m, [])
        poles = solve_one_m(m, f_min, f_max)
        light_ring = find_light_ring(m)
        light_ring_hz = (light_ring[1] / (2.0 * np.pi)
                          if light_ring[1] is not None else np.nan)
        for q, pole in enumerate(poles, start=1):
            if q <= len(experimental):
                experimental_hz = float(experimental[q - 1])
                delta_hz = pole["f_re"] - experimental_hz
                relative_error_percent = 100.0 * delta_hz / experimental_hz
            else:
                experimental_hz = np.nan
                delta_hz = np.nan
                relative_error_percent = np.nan
            all_results.append({
                "m": m, "q": q, "f_model_hz": pole["f_re"],
                "f_model_im_hz": pole["f_im"], "abs_res": pole["abs_res"],
                "root_success": pole["root_success"],
                "root_nfev": pole.get("root_nfev", -1),
                "root_message": pole.get("root_message", ""),
                "seed_f_hz": pole.get("seed_f", np.nan),
                "real_axis_abs_res": pole["real_axis_abs_res"],
                "light_ring_hz": light_ring_hz,
                "experimental_hz": experimental_hz,
                "delta_hz": delta_hz,
                "relative_error_percent": relative_error_percent,
            })
    write_results_csv(all_results)
    print_summary(all_results)
    total_expected = len(M_VALUES) * EXPECTED_POLES
    total_found = len(all_results)
    print("=" * 78)
    print("FINAL STATUS")
    print("=" * 78)
    print()
    print(f"m modes scanned: {len(M_VALUES)}")
    print(f"expected poles: {total_expected}")
    print(f"validated poles found: {total_found}")
    print()
    if total_found == total_expected:
        print("All requested q=1..4 poles were validated.")
    else:
        print("WARNING:")
        print("Not all requested q=1..4 poles were found.")
        print("This is a numerical/search result, not a reason")
        print("to manufacture missing poles or assign experimental")
        print("frequencies to unvalidated candidates.")
    print()
    print("STAGE 5I-4 v6 COMPLETE")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
