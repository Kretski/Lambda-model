"""
stage5I_4_full_m_scan_v5_fixed.py
=================================

STAGE 5I-4 v5 FIXED — FULL m-SCAN WITH STRICT POLE VALIDATION

Purpose
-------
Scan the full m-range using the validated v4 multi-Im-seed strategy.

IMPORTANT
---------
Experimental frequencies are NOT used as seeds.

Physics is imported unchanged from:

    stage5I_3_resonance_v2.py

Pipeline
--------
    coarse real-axis scan
        ->
    real-axis local refinement
        ->
    multiple imaginary seeds
        ->
    complex Nelder-Mead
        ->
    Re/Im root polish
        ->
    strict validation
        ->
    pole clustering
        ->
    validated poles only

A pole is accepted only if:

    root_success == True
    AND
    finite |Res|
    AND
    |Res| < RES_TOL
    AND
    finite Re(f), Im(f)
    AND
    F_MIN <= Re(f) <= F_MAX
    AND
    IM_MIN <= Im(f) <= IM_MAX

The code deliberately does NOT assign q=1,2,3,4.

It reports physical poles in ascending Re(f).

This is a diagnostic/model-validation scan.
It does not modify the physics in stage5I_3_resonance_v2.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from scipy.optimize import minimize, minimize_scalar, root


# ============================================================================
# IMPORT PHYSICS
# ============================================================================

HERE = Path(__file__).resolve().parent

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stage5I_3_resonance_v2 import (
    R_B,
    find_light_ring,
    resonance_factor,
)


# ============================================================================
# SETTINGS
# ============================================================================

# Full m range requested for the model scan.
M_VALUES = list(range(-21, -3))

# Frequency search range in Hz.
F_MIN = 7.0
F_MAX = 10.5

# Coarse scan.
N_COARSE = 350

# Keep only strongest real-axis minima before complex refinement.
TOP_N = 5

# Real-axis refinement.
REAL_HALF_WIDTH = 0.08

# Cheap radial integration for coarse search.
# Physics is unchanged; only numerical resolution is reduced
# during candidate discovery.
COARSE_RADIAL_N = 64

# Turning-point scan used by the imported physics functions.
# Printed only for reproducibility.
TURNING_SCAN = 1000

# Imaginary seeds from validated v4 diagnostic.
#
# Experimental frequencies are NOT used.
IM_SEEDS = [
    -1.0e-5,
    -5.0e-5,
    -1.0e-4,
    -2.5e-4,
    -5.0e-4,
    -1.0e-3,
    -2.0e-3,
    -5.0e-3,
    -1.0e-2,
    -2.0e-2,
    -5.0e-2,
    -1.0e-1,
]

# Allowed complex frequency half-plane.
IM_MIN = -0.10
IM_MAX = 0.0

# Root acceptance threshold.
RES_TOL = 1.0e-8

# Root solver tolerance.
ROOT_XTOL = 1.0e-10

# Pole clustering tolerance in Hz.
POLE_CLUSTER_TOL = 1.0e-6

# Number of radial points used by the imported v2 complex action.
COMPLEX_RADIAL_N = 100


# ============================================================================
# PATCH NUMERICAL RESOLUTION FOR SCAN
# ============================================================================

import stage5I_3_resonance_v2 as physics

physics.DEFAULT_RADIAL_N = COARSE_RADIAL_N
physics.DEFAULT_COMPLEX_RADIAL_N = COMPLEX_RADIAL_N


# ============================================================================
# RANGE VALIDATION HELPERS
# ============================================================================

def valid_complex_frequency(
    f_re: float,
    f_im: float,
) -> bool:
    """
    Check whether a complex frequency is inside the allowed scan domain.
    """

    if not np.isfinite(f_re):
        return False

    if not np.isfinite(f_im):
        return False

    if f_re < F_MIN or f_re > F_MAX:
        return False

    if f_im < IM_MIN or f_im > IM_MAX:
        return False

    return True


# ============================================================================
# SAFE RESONANCE EVALUATION
# ============================================================================

def safe_res(
    f_re: float,
    f_im: float,
    m: int,
    r_sp,
    omega_lr,
    complex_action: bool,
):
    """
    Evaluate Res safely.

    Parameters
    ----------
    f_re : float
        Real part of frequency in Hz.

    f_im : float
        Imaginary part of frequency in Hz.

    m : int
        Azimuthal/model index.

    r_sp :
        Light-ring radius returned by find_light_ring().

    omega_lr :
        Light-ring angular frequency returned by find_light_ring().

    complex_action : bool
        Passed unchanged to resonance_factor().

    Notes
    -----
    omega = 2*pi*f in rad/s.

    IMPORTANT:
    ----------
    r_sp and omega_lr are explicitly passed through the entire
    complex-refinement/root-polishing pipeline.
    """

    if not np.isfinite(f_re):
        return None

    if not np.isfinite(f_im):
        return None

    omega = 2.0 * np.pi * (
        float(f_re) + 1j * float(f_im)
    )

    try:

        z = resonance_factor(
            omega=omega,
            m=m,
            r_sp=r_sp,
            omega_lr=omega_lr,
            complex_action=complex_action,
        )

    except Exception:

        return None

    if z is None:
        return None

    try:

        value = abs(z)

    except Exception:

        return None

    if not np.isfinite(value):
        return None

    return z


# ============================================================================
# REAL AXIS COARSE SCAN
# ============================================================================

def coarse_real_scan(
    m: int,
    f_min: float,
    f_max: float,
    n_points: int,
    r_sp,
    omega_lr,
):
    """
    Find local minima of |Res| on the real axis.
    """

    f_grid = np.linspace(
        f_min,
        f_max,
        n_points,
    )

    values = np.full(
        n_points,
        np.inf,
        dtype=float,
    )

    for i, f in enumerate(f_grid):

        z = safe_res(
            f_re=f,
            f_im=0.0,
            m=m,
            r_sp=r_sp,
            omega_lr=omega_lr,
            complex_action=False,
        )

        if z is not None:
            values[i] = abs(z)

        if (
            (i + 1) % max(1, n_points // 10) == 0
        ):

            pct = 100.0 * (i + 1) / n_points

            print(
                f"      {i + 1:4d}/{n_points} "
                f"({pct:5.1f}%)",
                flush=True,
            )

    candidates = []

    for i in range(1, n_points - 1):

        if not np.isfinite(values[i]):
            continue

        if (
            values[i] <= values[i - 1]
            and
            values[i] <= values[i + 1]
        ):

            candidates.append(i)

    if not candidates:
        return []

    candidates.sort(
        key=lambda i: values[i]
    )

    candidates = candidates[:TOP_N]

    results = []

    for i in candidates:

        results.append({
            "f": float(f_grid[i]),
            "abs_res": float(values[i]),
        })

    return results


# ============================================================================
# REAL AXIS REFINEMENT
# ============================================================================

def refine_real_candidate(
    f0: float,
    m: int,
    r_sp,
    omega_lr,
):
    """
    Higher-accuracy 1-D real-axis refinement.
    """

    lo = max(
        F_MIN,
        f0 - REAL_HALF_WIDTH,
    )

    hi = min(
        F_MAX,
        f0 + REAL_HALF_WIDTH,
    )

    def objective(f):

        z = safe_res(
            f_re=float(f),
            f_im=0.0,
            m=m,
            r_sp=r_sp,
            omega_lr=omega_lr,
            complex_action=False,
        )

        if z is None:
            return 1.0e100

        value = abs(z)

        if not np.isfinite(value):
            return 1.0e100

        return float(value)

    try:

        result = minimize_scalar(
            objective,
            bounds=(lo, hi),
            method="bounded",
            options={
                "xatol": 1.0e-10,
                "maxiter": 500,
            },
        )

    except Exception:

        return None

    if not result.success:
        return None

    if not np.isfinite(result.x):
        return None

    if not np.isfinite(result.fun):
        return None

    if result.x < F_MIN or result.x > F_MAX:
        return None

    return {
        "f": float(result.x),
        "abs_res": float(result.fun),
    }


# ============================================================================
# COMPLEX NELDER-MEAD
# ============================================================================

def complex_nm(
    f0: float,
    im_seed: float,
    m: int,
    r_sp,
    omega_lr,
):
    """
    Complex minimization around a real-axis candidate.

    FIX:
        r_sp and omega_lr are explicitly passed into safe_res().
    """

    def objective(x):

        f_re = float(x[0])
        f_im = float(x[1])

        # Global frequency range.
        if f_re < F_MIN or f_re > F_MAX:
            return 1.0e100

        # Local real-frequency guard.
        if (
            f_re < f0 - 0.32
            or
            f_re > f0 + 0.32
        ):
            return 1.0e100

        # Allowed damped half-plane.
        if f_im < IM_MIN or f_im > IM_MAX:
            return 1.0e100

        z = safe_res(
            f_re=f_re,
            f_im=f_im,
            m=m,
            r_sp=r_sp,
            omega_lr=omega_lr,
            complex_action=True,
        )

        if z is None:
            return 1.0e100

        value = abs(z)

        if not np.isfinite(value):
            return 1.0e100

        return float(
            np.log1p(value)
        )

    # Start directly from the validated real-axis candidate
    # and one of the predefined imaginary seeds.
    x0 = np.array([
        float(f0),
        float(im_seed),
    ])

    if not valid_complex_frequency(
        x0[0],
        x0[1],
    ):

        return None

    try:

        result = minimize(
            objective,
            x0=x0,
            method="Nelder-Mead",
            options={
                "xatol": 1.0e-9,
                "fatol": 1.0e-10,
                "maxiter": 800,
            },
        )

    except Exception:

        return None

    if result is None:
        return None

    if not np.all(
        np.isfinite(result.x)
    ):

        return None

    f_re = float(result.x[0])
    f_im = float(result.x[1])

    # Global frequency-domain validation.
    if not valid_complex_frequency(
        f_re,
        f_im,
    ):

        return None

    # Local real-frequency guard.
    if (
        f_re < f0 - 0.32
        or
        f_re > f0 + 0.32
    ):

        return None

    z = safe_res(
        f_re=f_re,
        f_im=f_im,
        m=m,
        r_sp=r_sp,
        omega_lr=omega_lr,
        complex_action=True,
    )

    if z is None:
        return None

    residual = float(abs(z))

    if not np.isfinite(residual):
        return None

    return {
        "f_re": f_re,
        "f_im": f_im,
        "abs_res": residual,
        "nm_success": bool(result.success),
    }


# ============================================================================
# ROOT POLISH
# ============================================================================

def polish_root(
    nm_result,
    m: int,
    r_sp,
    omega_lr,
    f0: float,
):
    """
    Solve:

        Re(Res) = 0
        Im(Res) = 0

    This is the decisive validation step.

    FIXES
    -----
    1. r_sp and omega_lr are passed explicitly.
    2. Root output is checked against the allowed frequency domain.
    3. Root output is checked against the local candidate guard.
    4. |Res| is independently recalculated after root().
    """

    x0 = np.array([
        nm_result["f_re"],
        nm_result["f_im"],
    ], dtype=float)

    if not np.all(
        np.isfinite(x0)
    ):

        return None

    def equations(x):

        f_re = float(x[0])
        f_im = float(x[1])

        # Keep root evaluation inside the physical scan domain.
        if not valid_complex_frequency(
            f_re,
            f_im,
        ):

            return np.array([
                1.0e6,
                1.0e6,
            ])

        # Keep root near the real candidate.
        if (
            f_re < f0 - 0.32
            or
            f_re > f0 + 0.32
        ):

            return np.array([
                1.0e6,
                1.0e6,
            ])

        z = safe_res(
            f_re=f_re,
            f_im=f_im,
            m=m,
            r_sp=r_sp,
            omega_lr=omega_lr,
            complex_action=True,
        )

        if z is None:

            return np.array([
                1.0e6,
                1.0e6,
            ])

        return np.array([
            float(z.real),
            float(z.imag),
        ])

    try:

        result = root(
            equations,
            x0,
            method="hybr",
            options={
                "xtol": ROOT_XTOL,
                "maxfev": 500,
            },
        )

    except Exception:

        return None

    if result is None:
        return None

    if not hasattr(result, "x"):
        return None

    if not np.all(
        np.isfinite(result.x)
    ):

        return None

    f_re = float(result.x[0])
    f_im = float(result.x[1])

    # ------------------------------------------------------------
    # STRICT DOMAIN VALIDATION
    # ------------------------------------------------------------

    domain_ok = valid_complex_frequency(
        f_re,
        f_im,
    )

    local_ok = (
        f0 - 0.32
        <= f_re
        <= f0 + 0.32
    )

    if not domain_ok or not local_ok:

        return {
            "f_re": f_re,
            "f_im": f_im,
            "abs_res": np.inf,
            "root_success": False,
            "raw_root_success": bool(
                result.success
            ),
            "domain_ok": bool(domain_ok),
            "local_ok": bool(local_ok),
        }

    # ------------------------------------------------------------
    # FINAL RESIDUAL EVALUATION
    # ------------------------------------------------------------

    z = safe_res(
        f_re=f_re,
        f_im=f_im,
        m=m,
        r_sp=r_sp,
        omega_lr=omega_lr,
        complex_action=True,
    )

    if z is None:

        return {
            "f_re": f_re,
            "f_im": f_im,
            "abs_res": np.inf,
            "root_success": False,
            "raw_root_success": bool(
                result.success
            ),
            "domain_ok": bool(domain_ok),
            "local_ok": bool(local_ok),
        }

    residual = float(abs(z))

    if not np.isfinite(residual):

        return {
            "f_re": f_re,
            "f_im": f_im,
            "abs_res": np.inf,
            "root_success": False,
            "raw_root_success": bool(
                result.success
            ),
            "domain_ok": bool(domain_ok),
            "local_ok": bool(local_ok),
        }

    # ------------------------------------------------------------
    # STRICT ROOT ACCEPTANCE
    # ------------------------------------------------------------

    root_success = bool(
        result.success
        and
        domain_ok
        and
        local_ok
        and
        np.isfinite(f_re)
        and
        np.isfinite(f_im)
        and
        np.isfinite(residual)
        and
        residual < RES_TOL
    )

    return {
        "f_re": f_re,
        "f_im": f_im,
        "abs_res": residual,
        "root_success": root_success,
        "raw_root_success": bool(
            result.success
        ),
        "domain_ok": bool(domain_ok),
        "local_ok": bool(local_ok),
    }


# ============================================================================
# ONE REAL CANDIDATE — MULTI-SEED TEST
# ============================================================================

def validate_candidate(
    f0: float,
    m: int,
    r_sp,
    omega_lr,
):
    """
    Try every imaginary seed.

    Only validated roots are returned.
    """

    validated = []

    for im_seed in IM_SEEDS:

        print(
            f"        Im seed = {im_seed:+.6e} Hz",
            flush=True,
        )

        nm = complex_nm(
            f0=f0,
            im_seed=im_seed,
            m=m,
            r_sp=r_sp,
            omega_lr=omega_lr,
        )

        if nm is None:

            print(
                "            NM -> FAILED",
                flush=True,
            )

            continue

        print(
            f"            NM -> "
            f"Re={nm['f_re']:.10f} "
            f"Im={nm['f_im']:.10f} "
            f"|Res|={nm['abs_res']:.4e} "
            f"success={nm['nm_success']}",
            flush=True,
        )

        polished = polish_root(
            nm_result=nm,
            m=m,
            r_sp=r_sp,
            omega_lr=omega_lr,
            f0=f0,
        )

        if polished is None:

            print(
                "            ROOT -> FAILED",
                flush=True,
            )

            continue

        print(
            f"            ROOT -> "
            f"Re={polished['f_re']:.10f} "
            f"Im={polished['f_im']:.10f} "
            f"|Res|={polished['abs_res']:.4e} "
            f"success={polished['root_success']}",
            flush=True,
        )

        if polished["root_success"]:

            print(
                "            STATUS -> VALID",
                flush=True,
            )

            validated.append(
                polished
            )

        else:

            print(
                "            STATUS -> REJECTED",
                flush=True,
            )

    return validated


# ============================================================================
# CLUSTER VALIDATED POLES
# ============================================================================

def cluster_poles(
    poles,
    tolerance=POLE_CLUSTER_TOL,
):
    """
    Merge numerically identical poles reached from different seeds.
    """

    if not poles:
        return []

    poles = sorted(
        poles,
        key=lambda p: (
            p["f_re"],
            p["f_im"],
        ),
    )

    clusters = []

    for pole in poles:

        placed = False

        for cluster in clusters:

            representative = cluster[0]

            distance = np.sqrt(
                (
                    pole["f_re"]
                    -
                    representative["f_re"]
                ) ** 2
                +
                (
                    pole["f_im"]
                    -
                    representative["f_im"]
                ) ** 2
            )

            if distance < tolerance:

                cluster.append(pole)
                placed = True
                break

        if not placed:

            clusters.append(
                [pole]
            )

    result = []

    for cluster in clusters:

        best = min(
            cluster,
            key=lambda p: p["abs_res"],
        )

        result.append({
            "f_re": best["f_re"],
            "f_im": best["f_im"],
            "abs_res": best["abs_res"],
            "multiplicity": len(cluster),
        })

    result.sort(
        key=lambda p: p["f_re"]
    )

    return result


# ============================================================================
# SCAN ONE m
# ============================================================================

def scan_m(m: int):

    print()
    print("=" * 78)
    print(f"STAGE 5I-4 v5 FIXED   m={m}")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Light ring
    # ------------------------------------------------------------------

    r_sp, omega_lr = find_light_ring(m)

    if r_sp is None:

        print(
            f"[m={m}] ERROR: no light ring"
        )

        return []

    if omega_lr is None:

        print(
            f"[m={m}] ERROR: no light-ring frequency"
        )

        return []

    if not np.isfinite(r_sp):

        print(
            f"[m={m}] ERROR: non-finite r_sp"
        )

        return []

    if not np.isfinite(omega_lr):

        print(
            f"[m={m}] ERROR: non-finite omega_lr"
        )

        return []

    f_lr = (
        omega_lr
        /
        (2.0 * np.pi)
    )

    print(
        f"[m={m}] light ring: "
        f"r_sp={r_sp * 1.0e3:.9f} mm "
        f"f_lr={f_lr:.12f} Hz"
    )

    print(
        f"[m={m}] coarse scan: "
        f"{N_COARSE} points | "
        f"radial_n={COARSE_RADIAL_N} | "
        f"complex_radial_n={COMPLEX_RADIAL_N}"
    )

    # ------------------------------------------------------------------
    # Coarse scan
    # ------------------------------------------------------------------

    candidates = coarse_real_scan(
        m=m,
        f_min=F_MIN,
        f_max=F_MAX,
        n_points=N_COARSE,
        r_sp=r_sp,
        omega_lr=omega_lr,
    )

    if not candidates:

        print(
            f"[m={m}] no real-axis candidates"
        )

        return []

    print()
    print(
        f"[m={m}] coarse candidates:"
    )

    for c in candidates:

        print(
            f"    f={c['f']:.10f} Hz "
            f"|Res|={c['abs_res']:.4e}"
        )

    # ------------------------------------------------------------------
    # Real refinement
    # ------------------------------------------------------------------

    refined_candidates = []

    print()
    print(
        f"[m={m}] real-axis local refinement:"
    )

    for candidate in candidates:

        refined = refine_real_candidate(
            f0=candidate["f"],
            m=m,
            r_sp=r_sp,
            omega_lr=omega_lr,
        )

        if refined is None:

            print(
                f"    f={candidate['f']:.10f} "
                f"-> FAILED"
            )

            continue

        print(
            f"    f={refined['f']:.10f} Hz "
            f"|Res|={refined['abs_res']:.4e}"
        )

        refined_candidates.append(
            refined
        )

    if not refined_candidates:

        print(
            f"[m={m}] no refined real-axis candidates"
        )

        return []

    # ------------------------------------------------------------------
    # Remove duplicate real candidates
    # ------------------------------------------------------------------

    refined_candidates.sort(
        key=lambda x: x["f"]
    )

    distinct = []

    df = (
        (F_MAX - F_MIN)
        /
        max(
            1,
            N_COARSE - 1,
        )
    )

    for candidate in refined_candidates:

        if not distinct:

            distinct.append(candidate)
            continue

        if (
            abs(
                candidate["f"]
                -
                distinct[-1]["f"]
            )
            >
            0.5 * df
        ):

            distinct.append(candidate)

        elif (
            candidate["abs_res"]
            <
            distinct[-1]["abs_res"]
        ):

            distinct[-1] = candidate

    print()
    print(
        f"[m={m}] distinct real-axis candidates: "
        f"{len(distinct)}"
    )

    # ------------------------------------------------------------------
    # Complex validation
    # ------------------------------------------------------------------

    all_validated = []

    for candidate in distinct:

        f0 = candidate["f"]

        print()
        print(
            "-" * 78
        )
        print(
            f"[m={m}] REAL CANDIDATE "
            f"f0={f0:.10f} Hz"
        )
        print(
            "-" * 78
        )

        validated = validate_candidate(
            f0=f0,
            m=m,
            r_sp=r_sp,
            omega_lr=omega_lr,
        )

        all_validated.extend(
            validated
        )

    # ------------------------------------------------------------------
    # Cluster
    # ------------------------------------------------------------------

    clusters = cluster_poles(
        all_validated
    )

    print()
    print(
        f"[m={m}] VALIDATED POLES: "
        f"{len(clusters)}"
    )

    for i, pole in enumerate(
        clusters,
        start=1,
    ):

        print(
            f"    pole {i}: "
            f"f_re={pole['f_re']:.12f} Hz "
            f"f_im={pole['f_im']:.12f} Hz "
            f"|Res|={pole['abs_res']:.4e} "
            f"seeds={pole['multiplicity']}"
        )

    return clusters


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 78)
    print(
        "STAGE 5I-4 v5 FIXED — FULL m-SCAN"
    )
    print("=" * 78)

    print()
    print(
        "Physics source: stage5I_3_resonance_v2.py"
    )

    print(
        "Experimental frequencies are NOT used as seeds."
    )

    print()
    print(
        "Pipeline:"
    )

    print(
        "    coarse real scan"
    )

    print(
        "      -> real-axis refinement"
    )

    print(
        "      -> multi-Im complex seeds"
    )

    print(
        "      -> Nelder-Mead"
    )

    print(
        "      -> Re/Im root polish"
    )

    print(
        "      -> strict |Res| validation"
    )

    print(
        "      -> pole clustering"
    )

    print()
    print(
        f"m range = {M_VALUES[0]} ... {M_VALUES[-1]}"
    )

    print(
        f"frequency range = "
        f"[{F_MIN}, {F_MAX}] Hz"
    )

    print(
        f"Im range = "
        f"[{IM_MIN}, {IM_MAX}] Hz"
    )

    print(
        f"Im seeds = {len(IM_SEEDS)}"
    )

    print(
        f"acceptance threshold = "
        f"|Res| < {RES_TOL:.1e}"
    )

    print(
        f"pole cluster tolerance = "
        f"{POLE_CLUSTER_TOL:.1e} Hz"
    )

    print()
    print(
        "FIXES:"
    )

    print(
        "    r_sp / omega_lr propagated through complex refinement"
    )

    print(
        "    r_sp / omega_lr propagated through root polish"
    )

    print(
        "    strict post-root frequency-domain validation"
    )

    print(
        "    strict local-candidate validation"
    )

    all_results = {}

    for index, m in enumerate(
        M_VALUES,
        start=1,
    ):

        print()
        print(
            f"GLOBAL PROGRESS: "
            f"{index}/{len(M_VALUES)}"
        )

        try:

            poles = scan_m(m)

        except KeyboardInterrupt:

            print()
            print(
                "Interrupted by user."
            )

            break

        except Exception as exc:

            print()
            print(
                f"[m={m}] FATAL ERROR:"
            )

            print(
                f"    {type(exc).__name__}: {exc}"
            )

            poles = []

        all_results[m] = poles

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "FINAL VALIDATED POLES"
    )
    print("=" * 78)

    total = 0

    for m in M_VALUES:

        poles = all_results.get(
            m,
            [],
        )

        print()
        print(
            f"m={m}: "
            f"{len(poles)} validated pole(s)"
        )

        for i, pole in enumerate(
            poles,
            start=1,
        ):

            total += 1

            print(
                f"    pole {i}: "
                f"Re={pole['f_re']:.12f} Hz "
                f"Im={pole['f_im']:.12f} Hz "
                f"|Res|={pole['abs_res']:.4e} "
                f"seeds={pole['multiplicity']}"
            )

    print()
    print("=" * 78)
    print(
        f"TOTAL VALIDATED POLES = {total}"
    )
    print("=" * 78)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Only poles passing the strict root test are reported."
    )

    print(
        "No q=1/q=2/q=3/q=4 labels are assigned by this scan."
    )

    print(
        "Experimental frequencies were not used as seeds."
    )

    print(
        "The imported physics model was not modified."
    )

    print(
        "This result is suitable for the next model-vs-experiment comparison."
    )

    print()
    print(
        "STAGE 5I-4 v5 FIXED COMPLETE"
    )


if __name__ == "__main__":
    main()