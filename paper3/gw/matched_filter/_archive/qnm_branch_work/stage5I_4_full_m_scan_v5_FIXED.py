
"""
stage5I_4_full_m_scan_v5_FIXED.py
=================================

STAGE 5I-4 — FULL m-SCAN / MULTIPLE OVERTONE SEARCH

v5 FIXED
--------

Physics is taken unchanged from:

    stage5I_3_resonance_v2.py

This file changes only the numerical search / validation layer.

Pipeline:

    complete real-axis scan
        ->
    local-minimum discovery
        ->
    real-axis refinement
        ->
    complex Nelder-Mead refinement
        ->
    true Re/Im hybr root polish
        ->
    strict physical-pole validation
        ->
    duplicate clustering
        ->
    q=1..4 model poles
        ->
    comparison with EXP B

IMPORTANT
---------

Experimental frequencies are NOT used as complex seeds.

They are used only to:

    1. define the total scan interval;
    2. report model-vs-experiment differences.

The physics implementation remains in stage5I_3_resonance_v2.py.

The m=-12 regression benchmark established in Stage 5I-3b/3c is:

    f_re = 8.3066307169 Hz
    f_im = -0.0005664350 Hz
    |Res| ~ 2.78e-17

v5 FIXED numerical policy
-------------------------

1. Cheap discovery:

       COARSE_N = 400
       radial_n = 64
       turning scan = 400

2. Real-axis refinement:

       radial_n = 64
       turning scan = 800

   The real refinement is intentionally lighter than the original v3
   production configuration. The expensive validated v2 complex
   calculation is reserved for actual candidate poles.

3. Complex refinement:

       primary Im(f) seed = -5e-4 Hz
       fallback seeds     = -1e-3, -1e-2, -1e-1 Hz

   This is centered on the known m=-12 pole rather than the old
   generic -0.005 Hz seed.

4. Root validation:

       root_success == True
       finite Re/Im
       damped half-plane: f_im < 0
       |Res| < 1e-8

   A Nelder-Mead minimum without a successful root polish is NOT
   counted as a physical pole.

5. Real-axis candidate discovery does NOT select only the globally
   strongest minima. All local minima survive the discovery stage,
   followed by a small frequency-space clustering step.

6. Duplicate complex poles are clustered only AFTER root polishing.

7. The final q=1..4 model poles are sorted by Re(f).

8. If fewer than four validated poles are found, the script reports
   the shortfall explicitly instead of fabricating q assignments.

Output
------

    stage5I_4_full_m_scan_v5_FIXED_results.csv

Columns include:

    m
    q
    f_model_hz
    f_model_im_hz
    abs_res
    root_success
    root_nfev
    root_message
    experimental_hz
    delta_hz
    relative_error_percent
    light_ring_hz
    seed_f_hz
    real_axis_abs_res

The script also prints a complete human-readable summary.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar, root


# ============================================================================
# PATH / IMPORT
# ============================================================================

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
        refine_complex_resonance as refine_complex_v2,
    )

except ImportError as exc:

    raise ImportError(
        "\n"
        "Could not import stage5I_3_resonance_v2.py\n"
        "\n"
        "Make sure these files are in the same directory:\n"
        "\n"
        "    stage5I_1_2_dispersion_radial_p.py\n"
        "    stage5I_3_resonance_v2.py\n"
        "    stage5I_4_full_m_scan_v5_FIXED.py\n"
    ) from exc


# ============================================================================
# EXP B
# ============================================================================
#
# Experimental counter-rotating resonances.
#
# m, q1, q2, q3, q4
#
# Values in Hz.
#
# These values are NOT used to seed complex refinement.
# ============================================================================

EXP_B = {

    -21: [
        10.53127805,
        11.31559458,
        11.73305339,
        12.0,
    ],

    -20: [
        10.30228351,
        11.07509152,
        11.48642482,
        11.79180863,
    ],

    -19: [
        10.07028344,
        10.83337372,
        11.23953467,
        11.54107841,
    ],

    -18: [
        9.840683264,
        10.58962136,
        10.99149058,
        11.29593696,
    ],

    -17: [
        9.600492823,
        10.34296042,
        10.74739399,
        11.04921008,
    ],

    -16: [
        9.355011082,
        10.09843672,
        10.50612174,
        10.80589014,
    ],

    -15: [
        9.103267196,
        9.848948391,
        10.26056441,
        10.55883689,
    ],

    -14: [
        8.844144355,
        9.599296405,
        10.01552194,
        10.31282590,
    ],

    -13: [
        8.582278857,
        9.342201757,
        9.76372149,
        10.07244017,
    ],

    -12: [
        8.304259799,
        9.082091798,
        9.515540164,
        9.824297904,
    ],

    -11: [
        8.020133041,
        8.811238183,
        9.257350105,
        9.578550689,
    ],

    -10: [
        7.721731873,
        8.533427811,
        8.992990952,
        9.327218691,
    ],

    -9: [
        7.406262072,
        8.246009901,
        8.725865804,
        9.067763134,
    ],

    -8: [
        7.076173999,
        7.945602232,
        8.452768701,
        8.808992769,
    ],

    -7: [
        6.714687044,
        7.633791333,
        8.169428269,
        8.540722055,
    ],

    -6: [
        6.333524665,
        7.298338000,
        7.869851632,
        8.269296643,
    ],

    -5: [
        5.910999730,
        6.948591527,
        7.563690856,
        7.986183324,
    ],

    -4: [
        5.437524431,
        6.570039130,
        7.236964453,
        7.696262081,
    ],
}


# ============================================================================
# GLOBAL SEARCH SETTINGS
# ============================================================================

M_VALUES = list(range(-21, -3))

EXPECTED_POLES = 4


# ---------------------------------------------------------------------------
# Cheap discovery
# ---------------------------------------------------------------------------

COARSE_N = 400

COARSE_RADIAL_N = 64

COARSE_TURNING_SCAN = 400


# ---------------------------------------------------------------------------
# Real-axis local refinement
# ---------------------------------------------------------------------------

REAL_REFINE_RADIAL_N = 64

REAL_REFINE_TURNING_SCAN = 800

REAL_REFINE_HALF_WIDTH = 0.10


# ---------------------------------------------------------------------------
# Candidate clustering
# ---------------------------------------------------------------------------

REAL_CANDIDATE_SEPARATION_HZ = 0.025

MIN_POLE_SEPARATION_HZ = 0.015


# ---------------------------------------------------------------------------
# Complex refinement
# ---------------------------------------------------------------------------

COMPLEX_RE_HALF_WIDTH = 0.08

COMPLEX_IM_MIN = -1.5

COMPLEX_IM_MAX = 0.0


# Primary seed near established m=-12 pole.
PRIMARY_IM0 = -5.0e-4


# Fallback seeds.
FALLBACK_IM0 = [
    -1.0e-3,
    -1.0e-2,
    -1.0e-1,
]


# ---------------------------------------------------------------------------
# Root polish
# ---------------------------------------------------------------------------

ROOT_XTOL = 1.0e-12

ROOT_MAXFEV = 150


# Strict physical-pole gate.
MAX_VALIDATED_ABS_RES = 1.0e-8


# Root must remain close to its real-axis candidate.
MAX_ROOT_SHIFT_HZ = 0.12


# ---------------------------------------------------------------------------
# Regression benchmark
# ---------------------------------------------------------------------------

REGRESSION_M = -12

BENCHMARK_F_RE = 8.3066307169

BENCHMARK_F_IM = -0.0005664350

BENCHMARK_F_TOL = 1.0e-5

BENCHMARK_RES_TOL = 1.0e-8


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

OUTPUT_CSV = (
    HERE
    / "stage5I_4_full_m_scan_v5_FIXED_results.csv"
)


# ============================================================================
# FAST OUTER TURNING POINT
# ============================================================================

def find_last_scattering_point_fast(
    omega,
    m,
    r_sp,
    n_scan=COARSE_TURNING_SCAN,
):
    """
    Same physical definition as Stage 5I-3 v2.

    Below the light ring:

        choose the outermost real turning point.

    At / above the light ring:

        r_- = r_sp.

    Only the bracketing grid is reduced.
    The actual root is still solved with Brent.
    """

    omega = float(
        np.real(
            omega
        )
    )

    omega_lr = (
        omega_D_plus_at_p0(
            m,
            r_sp,
        )
    )

    if omega >= omega_lr:

        return float(
            r_sp
        )

    r_grid = np.linspace(
        5.0e-3,
        R_B,
        int(n_scan),
    )

    values = np.array([
        omega_D_plus_at_p0(
            m,
            r,
        )
        - omega
        for r in r_grid
    ])

    # Search from outside inward.
    for i in range(
        len(r_grid) - 2,
        -1,
        -1,
    ):

        a = values[i]

        b = values[i + 1]

        if not (
            np.isfinite(a)
            and np.isfinite(b)
        ):

            continue

        if a == 0.0:

            return float(
                r_grid[i]
            )

        if a * b < 0.0:

            try:

                return float(
                    brentq(
                        lambda r:
                            omega_D_plus_at_p0(
                                m,
                                r,
                            )
                            - omega,
                        r_grid[i],
                        r_grid[i + 1],
                        xtol=1.0e-11,
                        rtol=1.0e-11,
                    )
                )

            except (
                ValueError,
                RuntimeError,
            ):

                return None

    return None


# ============================================================================
# REAL-AXIS RESONANCE FACTOR
# ============================================================================

def resonance_factor_real_fast(
    f,
    m,
    r_sp,
    omega_lr,
    radial_n=COARSE_RADIAL_N,
    turning_scan=COARSE_TURNING_SCAN,
):
    """
    Eq. (17) on the real axis with controlled radial resolution.

    This does not modify the physics.
    It only controls numerical resolution during candidate discovery.
    """

    omega = (
        2.0
        * np.pi
        * float(f)
    )

    r_minus = (
        find_last_scattering_point_fast(
            omega,
            m,
            r_sp,
            n_scan=turning_scan,
        )
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

        if not (
            np.isfinite(R.real)
            and np.isfinite(R.imag)
        ):

            return None

        action = radial_action_real(
            omega,
            m,
            r_minus,
            n=radial_n,
        )

        z = (
            R
            * np.exp(
                2.0j * action
            )
            - 1.0
        )

        if not (
            np.isfinite(z.real)
            and np.isfinite(z.imag)
        ):

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


# ============================================================================
# COMPLETE REAL-AXIS SCAN
# ============================================================================

def coarse_scan(
    m,
    f_min,
    f_max,
):
    """
    Scan the complete interval.

    IMPORTANT:

    Unlike the old implementation, candidate discovery is not based
    only on the strongest |Res| values globally.

    Every finite local minimum is retained initially.
    """

    r_sp, omega_lr = (
        find_light_ring(
            m
        )
    )

    if r_sp is None:

        return [], None, None

    f_grid = np.linspace(
        f_min,
        f_max,
        COARSE_N,
    )

    abs_res = np.full(
        COARSE_N,
        np.inf,
        dtype=float,
    )

    print(
        f"[m={m}] coarse scan: "
        f"{COARSE_N} points | "
        f"radial_n={COARSE_RADIAL_N} | "
        f"turning_scan={COARSE_TURNING_SCAN}"
    )

    for i, f in enumerate(
        f_grid
    ):

        z = resonance_factor_real_fast(
            f,
            m,
            r_sp,
            omega_lr,
            radial_n=COARSE_RADIAL_N,
            turning_scan=COARSE_TURNING_SCAN,
        )

        if z is not None:

            value = abs(z)

            if np.isfinite(
                value
            ):

                abs_res[i] = value

        if (
            (i + 1)
            % max(
                1,
                COARSE_N // 10,
            )
            == 0
        ):

            print(
                f"    {i + 1:4d}/{COARSE_N} "
                f"({100.0 * (i + 1) / COARSE_N:5.1f}%)"
            )

    finite = np.isfinite(
        abs_res
    )

    if not np.any(
        finite
    ):

        return [], r_sp, omega_lr

    candidates = []

    # ------------------------------------------------------------------
    # ALL local minima
    # ------------------------------------------------------------------

    for i in range(
        1,
        COARSE_N - 1,
    ):

        if not np.isfinite(
            abs_res[i]
        ):

            continue

        left = abs_res[i - 1]

        center = abs_res[i]

        right = abs_res[i + 1]

        if not (
            np.isfinite(left)
            and np.isfinite(right)
        ):

            continue

        if (
            center <= left
            and center <= right
        ):

            candidates.append({
                "f_coarse": float(
                    f_grid[i]
                ),
                "abs_res_coarse": float(
                    center
                ),
            })

    # ------------------------------------------------------------------
    # Endpoint / global-best fallback
    # ------------------------------------------------------------------

    finite_indices = (
        np.flatnonzero(
            finite
        )
    )

    best_idx = finite_indices[
        np.argmin(
            abs_res[
                finite
            ]
        )
    ]

    if not candidates:

        candidates.append({
            "f_coarse": float(
                f_grid[best_idx]
            ),
            "abs_res_coarse": float(
                abs_res[best_idx]
            ),
        })

    # ------------------------------------------------------------------
    # Frequency-space candidate clustering
    #
    # This is discovery-level clustering only.
    # Physical pole clustering happens after root polish.
    # ------------------------------------------------------------------

    candidates.sort(
        key=lambda x:
            x["f_coarse"]
    )

    clustered = []

    for candidate in candidates:

        if not clustered:

            clustered.append(
                candidate
            )

            continue

        if (
            abs(
                candidate["f_coarse"]
                - clustered[-1]["f_coarse"]
            )
            < REAL_CANDIDATE_SEPARATION_HZ
        ):

            if (
                candidate["abs_res_coarse"]
                < clustered[-1]["abs_res_coarse"]
            ):

                clustered[-1] = candidate

        else:

            clustered.append(
                candidate
            )

    return (
        clustered,
        r_sp,
        omega_lr,
    )


# ============================================================================
# REAL-AXIS REFINEMENT
# ============================================================================

def refine_real_candidate(
    candidate,
    m,
    f_min,
    f_max,
    r_sp,
    omega_lr,
):
    """
    Higher-resolution 1-D refinement of one real-axis minimum.
    """

    f0 = candidate[
        "f_coarse"
    ]

    lo = max(
        f_min,
        f0 - REAL_REFINE_HALF_WIDTH,
    )

    hi = min(
        f_max,
        f0 + REAL_REFINE_HALF_WIDTH,
    )

    def objective(
        f
    ):

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

        if not np.isfinite(
            value
        ):

            return 1.0e100

        return float(
            value
        )

    try:

        result = minimize_scalar(
            objective,
            bounds=(
                lo,
                hi,
            ),
            method="bounded",
            options={
                "xatol": 2.0e-8,
                "maxiter": 100,
            },
        )

    except (
        ArithmeticError,
        FloatingPointError,
        OverflowError,
        ValueError,
        RuntimeError,
    ):

        return None

    if not result.success:

        return None

    if not np.isfinite(
        result.fun
    ):

        return None

    return {
        "f_real": float(
            result.x
        ),
        "abs_res": float(
            result.fun
        ),
        "coarse_f": float(
            f0
        ),
    }


# ============================================================================
# REAL CANDIDATE CLUSTERING
# ============================================================================

def cluster_real_candidates(
    candidates
):
    """
    Remove numerical duplicates created by overlapping local windows.
    """

    candidates = sorted(
        candidates,
        key=lambda x:
            x["f_real"],
    )

    unique = []

    for candidate in candidates:

        if not unique:

            unique.append(
                candidate
            )

            continue

        if (
            abs(
                candidate["f_real"]
                - unique[-1]["f_real"]
            )
            < REAL_CANDIDATE_SEPARATION_HZ
        ):

            if (
                candidate["abs_res"]
                < unique[-1]["abs_res"]
            ):

                unique[-1] = candidate

        else:

            unique.append(
                candidate
            )

    return unique


# ============================================================================
# SAFE COMPLEX RESONANCE
# ============================================================================

def safe_complex_res(
    f_re,
    f_im,
    m,
    r_sp,
    omega_lr,
):
    """
    Evaluate the validated Stage 5I-3 v2 resonance function.
    """

    if not (
        np.isfinite(f_re)
        and np.isfinite(f_im)
    ):

        return None

    if not (
        COMPLEX_IM_MIN
        <= f_im
        <= COMPLEX_IM_MAX
    ):

        return None

    omega = (
        2.0
        * np.pi
        * (
            f_re
            + 1j * f_im
        )
    )

    try:

        z = resonance_factor_v2(
            omega=omega,
            m=m,
            r_sp=r_sp,
            omega_lr=omega_lr,
            complex_action=True,
        )

    except (
        ArithmeticError,
        FloatingPointError,
        OverflowError,
        ValueError,
        RuntimeError,
    ):

        return None

    if z is None:

        return None

    if not (
        np.isfinite(z.real)
        and np.isfinite(z.imag)
    ):

        return None

    return z


# ============================================================================
# COMPLEX NELDER-MEAD
# ============================================================================

def complex_minimize_from_seed(
    f0,
    im0,
    m,
    r_sp,
    omega_lr,
):
    """
    Local complex minimization.

    The experimental frequency is never used as a seed.

    f0 comes only from the real-axis resonance search.
    """

    lo_re = (
        f0
        - COMPLEX_RE_HALF_WIDTH
    )

    hi_re = (
        f0
        + COMPLEX_RE_HALF_WIDTH
    )

    def objective(
        x
    ):

        f_re = float(
            x[0]
        )

        f_im = float(
            x[1]
        )

        if not (
            lo_re
            <= f_re
            <= hi_re
        ):

            return 1.0e100

        if not (
            COMPLEX_IM_MIN
            <= f_im
            <= COMPLEX_IM_MAX
        ):

            return 1.0e100

        z = safe_complex_res(
            f_re,
            f_im,
            m,
            r_sp,
            omega_lr,
        )

        if z is None:

            return 1.0e100

        value = abs(z)

        if not np.isfinite(
            value
        ):

            return 1.0e100

        return float(
            np.log1p(
                value
            )
        )

    try:

        result = __import__(
            "scipy.optimize",
            fromlist=[
                "minimize"
            ],
        ).minimize(
            objective,
            x0=np.array([
                f0,
                im0,
            ]),
            method="Nelder-Mead",
            options={
                "xatol": 1.0e-8,
                "fatol": 1.0e-10,
                "maxiter": 700,
            },
        )

    except (
        ArithmeticError,
        FloatingPointError,
        OverflowError,
        ValueError,
        RuntimeError,
    ):

        return None

    if not np.all(
        np.isfinite(
            result.x
        )
    ):

        return None

    f_re = float(
        result.x[0]
    )

    f_im = float(
        result.x[1]
    )

    if not (
        lo_re
        <= f_re
        <= hi_re
    ):

        return None

    if not (
        COMPLEX_IM_MIN
        <= f_im
        <= COMPLEX_IM_MAX
    ):

        return None

    z = safe_complex_res(
        f_re,
        f_im,
        m,
        r_sp,
        omega_lr,
    )

    if z is None:

        return None

    abs_res = abs(z)

    if not np.isfinite(
        abs_res
    ):

        return None

    return {
        "f_re": f_re,
        "f_im": f_im,
        "abs_res": float(
            abs_res
        ),
        "minimize_success": bool(
            result.success
        ),
        "minimize_message": str(
            result.message
        ),
        "seed_im": float(
            im0
        ),
        "nfev": int(
            getattr(
                result,
                "nfev",
                -1,
            )
        ),
    }


# ============================================================================
# HYBR ROOT POLISH
# ============================================================================

def polish_root(
    candidate,
    m,
    r_sp,
    omega_lr,
):
    """
    True 2-D root solve:

        Re Res = 0
        Im Res = 0

    A root is accepted only if:

        success == True
        finite coordinates
        f_im < 0
        |Res| < 1e-8
        root remains close to the original real candidate
    """

    if candidate is None:

        return None

    f0 = float(
        candidate["f_real"]
    )

    seed_f_re = float(
        candidate["seed_f_re"]
    )

    seed_f_im = float(
        candidate["seed_f_im"]
    )

    def equations(
        x
    ):

        f_re = float(
            x[0]
        )

        f_im = float(
            x[1]
        )

        if not (
            np.isfinite(f_re)
            and np.isfinite(f_im)
        ):

            return np.array([
                1.0e3,
                1.0e3,
            ])

        # Keep root solve local.
        if (
            abs(
                f_re
                - f0
            )
            > MAX_ROOT_SHIFT_HZ
        ):

            return np.array([
                1.0e3,
                1.0e3,
            ])

        # Do not allow the solver into the growing half-plane.
        if (
            f_im
            > COMPLEX_IM_MAX
            or f_im
            < COMPLEX_IM_MIN
        ):

            return np.array([
                1.0e3,
                1.0e3,
            ])

        z = safe_complex_res(
            f_re,
            f_im,
            m,
            r_sp,
            omega_lr,
        )

        if z is None:

            return np.array([
                1.0e3,
                1.0e3,
            ])

        return np.array([
            z.real,
            z.imag,
        ])

    x0 = np.array([
        seed_f_re,
        seed_f_im,
    ])

    try:

        result = root(
            equations,
            x0=x0,
            method="hybr",
            options={
                "xtol": ROOT_XTOL,
                "maxfev": ROOT_MAXFEV,
            },
        )

    except (
        ArithmeticError,
        FloatingPointError,
        OverflowError,
        ValueError,
        RuntimeError,
    ):

        return None

    if not np.all(
        np.isfinite(
            result.x
        )
    ):

        return None

    f_re = float(
        result.x[0]
    )

    f_im = float(
        result.x[1]
    )

    z_final = safe_complex_res(
        f_re,
        f_im,
        m,
        r_sp,
        omega_lr,
    )

    if z_final is None:

        return None

    abs_res = abs(
        z_final
    )

    # ------------------------------------------------------------------
    # STRICT ACCEPTANCE GATE
    # ------------------------------------------------------------------

    if not result.success:

        return None

    if not np.isfinite(
        abs_res
    ):

        return None

    if f_im >= 0.0:

        return None

    if (
        abs(
            f_re
            - f0
        )
        > MAX_ROOT_SHIFT_HZ
    ):

        return None

    if (
        abs_res
        >= MAX_VALIDATED_ABS_RES
    ):

        return None

    return {
        "f_re": f_re,
        "f_im": f_im,
        "omega_re": (
            2.0
            * np.pi
            * f_re
        ),
        "omega_im": (
            2.0
            * np.pi
            * f_im
        ),
        "abs_res": float(
            abs_res
        ),
        "real_axis_abs_res": float(
            candidate[
                "real_axis_abs_res"
            ]
        ),
        "root_success": True,
        "root_message": str(
            result.message
        ),
        "root_nfev": int(
            getattr(
                result,
                "nfev",
                -1,
            )
        ),
        "seed_f": f0,
        "seed_f_re": seed_f_re,
        "seed_f_im": seed_f_im,
    }


# ============================================================================
# COMPLEX CANDIDATE + ROOT
# ============================================================================

def complex_refine_and_polish(
    candidate,
    m,
    r_sp,
    omega_lr,
):
    """
    Try the primary and fallback imaginary-frequency seeds.

    Only the best validated root is returned.

    The fallback seeds are numerical robustness measures, not additional
    physics and not experimental-frequency seeds.
    """

    f0 = float(
        candidate["f_real"]
    )

    seeds = [
        PRIMARY_IM0,
        *FALLBACK_IM0,
    ]

    validated = []

    for im0 in seeds:

        print(
            f"        complex seed "
            f"Im(f)={im0:+.6e} Hz"
        )

        refined = (
            complex_minimize_from_seed(
                f0,
                im0,
                m,
                r_sp,
                omega_lr,
            )
        )

        if refined is None:

            print(
                "            Nelder-Mead failed"
            )

            continue

        print(
            f"            NM: "
            f"Re={refined['f_re']:.10f} Hz "
            f"Im={refined['f_im']:.10f} Hz "
            f"|Res|={refined['abs_res']:.4e}"
        )

        root_candidate = {
            "f_real": f0,
            "seed_f_re": refined["f_re"],
            "seed_f_im": refined["f_im"],
            "real_axis_abs_res": candidate[
                "abs_res"
            ],
        }

        polished = polish_root(
            root_candidate,
            m,
            r_sp,
            omega_lr,
        )

        if polished is None:

            print(
                "            hybr: REJECTED"
            )

            continue

        print(
            f"            hybr: "
            f"Re={polished['f_re']:.10f} Hz "
            f"Im={polished['f_im']:.10f} Hz "
            f"|Res|={polished['abs_res']:.4e}"
        )

        validated.append(
            polished
        )

    if not validated:

        return None

    validated.sort(
        key=lambda x:
            x["abs_res"]
    )

    return validated[0]


# ============================================================================
# COMPLEX POLE CLUSTERING
# ============================================================================

def cluster_poles(
    poles
):
    """
    Cluster duplicate physical poles after root polishing.
    """

    poles = sorted(
        poles,
        key=lambda x:
            (
                x["f_re"],
                x["f_im"],
            ),
    )

    unique = []

    for pole in poles:

        if not unique:

            unique.append(
                pole
            )

            continue

        previous = unique[-1]

        distance = np.hypot(
            pole["f_re"]
            - previous["f_re"],
            pole["f_im"]
            - previous["f_im"],
        )

        if (
            distance
            < MIN_POLE_SEPARATION_HZ
        ):

            if (
                pole["abs_res"]
                < previous["abs_res"]
            ):

                unique[-1] = pole

        else:

            unique.append(
                pole
            )

    return unique


# ============================================================================
# ONE m MODE
# ============================================================================

def solve_one_m(
    m,
    f_min,
    f_max,
):
    """
    Full validated search for one m.
    """

    print()
    print("=" * 78)
    print(
        f"STAGE 5I-4 v5 FIXED   m={m}"
    )
    print("=" * 78)

    candidates, r_sp, omega_lr = (
        coarse_scan(
            m,
            f_min,
            f_max,
        )
    )

    if r_sp is None:

        print(
            "No light ring."
        )

        return []

    f_lr = (
        omega_lr
        / (
            2.0
            * np.pi
        )
    )

    print(
        f"light ring: "
        f"r_sp={r_sp * 1.0e3:.6f} mm "
        f"f_lr={f_lr:.10f} Hz"
    )

    if not candidates:

        print(
            "No real-axis candidates."
        )

        return []

    print()
    print(
        f"ALL real-axis candidates: "
        f"{len(candidates)}"
    )

    for candidate in candidates:

        print(
            f"    coarse f="
            f"{candidate['f_coarse']:.8f} Hz "
            f"|Res|="
            f"{candidate['abs_res_coarse']:.4e}"
        )

    # ------------------------------------------------------------------
    # Real-axis refinement
    # ------------------------------------------------------------------

    print()
    print(
        "Real-axis refinement:"
    )

    real_candidates = []

    for candidate in candidates:

        refined = (
            refine_real_candidate(
                candidate,
                m,
                f_min,
                f_max,
                r_sp,
                omega_lr,
            )
        )

        if refined is None:

            print(
                f"    f="
                f"{candidate['f_coarse']:.8f} "
                f"FAILED"
            )

            continue

        real_candidates.append(
            refined
        )

        print(
            f"    f="
            f"{refined['f_real']:.10f} Hz "
            f"|Res|="
            f"{refined['abs_res']:.4e}"
        )

    real_candidates = (
        cluster_real_candidates(
            real_candidates
        )
    )

    print()
    print(
        f"Distinct real candidates: "
        f"{len(real_candidates)}"
    )

    if not real_candidates:

        return []

    # ------------------------------------------------------------------
    # Complex refinement + root polish
    # ------------------------------------------------------------------

    print()
    print(
        "Complex refinement + strict root validation:"
    )

    poles = []

    for candidate in real_candidates:

        print()
        print(
            f"    starting real candidate "
            f"f={candidate['f_real']:.10f} Hz"
        )

        pole = (
            complex_refine_and_polish(
                candidate,
                m,
                r_sp,
                omega_lr,
            )
        )

        if pole is None:

            print(
                "        NO VALIDATED POLE"
            )

            continue

        poles.append(
            pole
        )

    # ------------------------------------------------------------------
    # Final clustering
    # ------------------------------------------------------------------

    poles = cluster_poles(
        poles
    )

    poles.sort(
        key=lambda x:
            x["f_re"]
    )

    print()
    print(
        f"Validated physical poles: "
        f"{len(poles)}"
    )

    # ------------------------------------------------------------------
    # First four by Re(f)
    # ------------------------------------------------------------------

    final_poles = poles[
        :EXPECTED_POLES
    ]

    print()
    print(
        "FINAL MODEL POLES:"
    )

    for q, pole in enumerate(
        final_poles,
        start=1,
    ):

        print(
            f"    q={q} "
            f"f={pole['f_re']:.10f} Hz "
            f"Im={pole['f_im']:.10f} Hz "
            f"|Res|={pole['abs_res']:.4e}"
        )

    if len(final_poles) < EXPECTED_POLES:

        print()
        print(
            f"WARNING: only "
            f"{len(final_poles)}/{EXPECTED_POLES} "
            f"validated poles found."
        )

    return final_poles


# ============================================================================
# FREQUENCY WINDOW
# ============================================================================

def experimental_frequency_window():
    """
    Define the global experimental comparison interval.

    A small margin is added so the model is not evaluated exactly at
    the experimental endpoints only.
    """

    values = [
        float(f)
        for frequencies in EXP_B.values()
        for f in frequencies
    ]

    f_min = min(
        values
    )

    f_max = max(
        values
    )

    margin = 0.15

    return (
        f_min - margin,
        f_max + margin,
    )


# ============================================================================
# REGRESSION CHECK
# ============================================================================

def run_m12_regression():
    """
    Independent m=-12 benchmark check using the v5 search configuration.

    The purpose is NOT to use the experimental 8.304259799 Hz value.

    The benchmark is the independently established Stage 5I-3 pole:

        8.3066307169 - 0.0005664350 i Hz
    """

    print()
    print("=" * 78)
    print(
        "STAGE 5I-4 v5 FIXED — m=-12 REGRESSION"
    )
    print("=" * 78)

    m = REGRESSION_M

    r_sp, omega_lr = (
        find_light_ring(
            m
        )
    )

    if r_sp is None:

        print(
            "REGRESSION FAIL: no light ring."
        )

        return False

    f_min = 7.0

    f_max = 10.5

    candidates, _, _ = (
        coarse_scan(
            m,
            f_min,
            f_max,
        )
    )

    if not candidates:

        print(
            "REGRESSION FAIL: no candidates."
        )

        return False

    real_candidates = []

    for candidate in candidates:

        refined = (
            refine_real_candidate(
                candidate,
                m,
                f_min,
                f_max,
                r_sp,
                omega_lr,
            )
        )

        if refined is not None:

            real_candidates.append(
                refined
            )

    real_candidates = (
        cluster_real_candidates(
            real_candidates
        )
    )

    if not real_candidates:

        print(
            "REGRESSION FAIL: no real-axis candidates."
        )

        return False

    # IMPORTANT:
    #
    # Select the real candidate nearest the established benchmark.
    #
    # This does NOT use the experimental frequency.
    # It only identifies the already-established regression pole.

    candidate = min(
        real_candidates,
        key=lambda c:
            abs(
                c["f_real"]
                - BENCHMARK_F_RE
            ),
    )

    print()
    print(
        f"Regression candidate: "
        f"{candidate['f_real']:.10f} Hz"
    )

    pole = (
        complex_refine_and_polish(
            candidate,
            m,
            r_sp,
            omega_lr,
        )
    )

    if pole is None:

        print(
            "REGRESSION FAIL: "
            "no validated complex pole."
        )

        return False

    diff_re = abs(
        pole["f_re"]
        - BENCHMARK_F_RE
    )

    diff_im = abs(
        pole["f_im"]
        - BENCHMARK_F_IM
    )

    res_ok = (
        pole["abs_res"]
        < BENCHMARK_RES_TOL
    )

    re_ok = (
        diff_re
        < BENCHMARK_F_TOL
    )

    im_ok = (
        diff_im
        < BENCHMARK_F_TOL
    )

    print()
    print(
        "REGRESSION RESULT:"
    )

    print(
        f"    f_re = "
        f"{pole['f_re']:.12f} Hz"
    )

    print(
        f"    f_im = "
        f"{pole['f_im']:.12f} Hz"
    )

    print(
        f"    |Res| = "
        f"{pole['abs_res']:.6e}"
    )

    print(
        f"    |Δf_re| = "
        f"{diff_re:.6e} Hz "
        f"{'PASS' if re_ok else 'FAIL'}"
    )

    print(
        f"    |Δf_im| = "
        f"{diff_im:.6e} Hz "
        f"{'PASS' if im_ok else 'FAIL'}"
    )

    print(
        f"    |Res| < "
        f"{BENCHMARK_RES_TOL:.0e}: "
        f"{'PASS' if res_ok else 'FAIL'}"
    )

    passed = (
        re_ok
        and im_ok
        and res_ok
    )

    print()

    if passed:

        print(
            "REGRESSION CHECK PASSED."
        )

    else:

        print(
            "REGRESSION CHECK FAILED."
        )

    return passed


# ============================================================================
# CSV OUTPUT
# ============================================================================

def write_results_csv(
    rows,
    path=OUTPUT_CSV,
):
    """
    Write model-vs-experiment results.
    """

    fieldnames = [
        "m",
        "q",
        "f_model_hz",
        "f_model_im_hz",
        "abs_res",
        "root_success",
        "root_nfev",
        "root_message",
        "seed_f_hz",
        "real_axis_abs_res",
        "light_ring_hz",
        "experimental_hz",
        "delta_hz",
        "relative_error_percent",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                row
            )

    print()
    print(
        f"Results written to:"
    )

    print(
        f"    {path}"
    )


# ============================================================================
# SUMMARY
# ============================================================================

def print_summary(
    all_results
):
    """
    Human-readable comparison summary.
    """

    print()
    print("=" * 110)
    print(
        "STAGE 5I-4 v5 FIXED — FINAL SUMMARY"
    )
    print("=" * 110)

    print()

    print(
        f"{'m':>4} "
        f"{'q':>3} "
        f"{'model Re(f)':>16} "
        f"{'model Im(f)':>16} "
        f"{'experiment':>16} "
        f"{'delta':>14} "
        f"{'|Res|':>12}"
    )

    print(
        "-" * 110
    )

    for row in all_results:

        print(
            f"{row['m']:>4} "
            f"{row['q']:>3} "
            f"{row['f_model_hz']:>16.8f} "
            f"{row['f_model_im_hz']:>16.8f} "
            f"{row['experimental_hz']:>16.8f} "
            f"{row['delta_hz']:>14.8f} "
            f"{row['abs_res']:>12.3e}"
        )

    print()


# ============================================================================
# MAIN FULL SCAN
# ============================================================================

def main():
    print("=" * 110)
    print(
        "STAGE 5I-4 v5 FIXED — FULL m-SCAN"
    )
    print("=" * 110)

    print()

    print(
        "Physics source:"
    )

    print(
        "    stage5I_3_resonance_v2.py"
    )

    print()

    print(
        "Search:"
    )

    print(
        "    complete real-axis scan"
    )

    print(
        "      -> local minima"
    )

    print(
        "      -> real-axis refinement"
    )

    print(
        "      -> complex Nelder-Mead"
    )

    print(
        "      -> hybr root polish"
    )

    print(
        "      -> strict pole validation"
    )

    print(
        "      -> complex pole clustering"
    )

    print()

    print(
        "Experimental frequencies are NOT used as complex seeds."
    )

    print()

    print(
        "Strict pole acceptance:"
    )

    print(
        f"    root_success = True"
    )

    print(
        f"    Im(f) < 0"
    )

    print(
        f"    |Res| < {MAX_VALIDATED_ABS_RES:.0e}"
    )

    print()

    # ------------------------------------------------------------------
    # Regression gate
    # ------------------------------------------------------------------

    regression_ok = (
        run_m12_regression()
    )

    if not regression_ok:

        print()
        print("=" * 78)
        print(
            "FULL m-SCAN ABORTED"
        )
        print("=" * 78)
        print()
        print(
            "The m=-12 regression gate failed."
        )
        print(
            "Do NOT interpret the full scan as validated."
        )

        return 1

    # ------------------------------------------------------------------
    # Experimental window
    # ------------------------------------------------------------------

    f_min, f_max = (
        experimental_frequency_window()
    )

    print()
    print(
        "Global scan interval:"
    )

    print(
        f"    [{f_min:.8f}, {f_max:.8f}] Hz"
    )

    print()

    # ------------------------------------------------------------------
    # Full m scan
    # ------------------------------------------------------------------

    all_results = []

    for m in M_VALUES:

        experimental = EXP_B.get(
            m,
            [],
        )

        poles = (
            solve_one_m(
                m,
                f_min,
                f_max,
            )
        )

        light_ring = (
            find_light_ring(
                m
            )
        )

        if light_ring[1] is not None:

            light_ring_hz = (
                light_ring[1]
                / (
                    2.0
                    * np.pi
                )
            )

        else:

            light_ring_hz = np.nan

        for q, pole in enumerate(
            poles,
            start=1,
        ):

            if (
                q
                <= len(
                    experimental
                )
            ):

                experimental_hz = float(
                    experimental[
                        q - 1
                    ]
                )

                delta_hz = (
                    pole["f_re"]
                    - experimental_hz
                )

                relative_error_percent = (
                    100.0
                    * delta_hz
                    / experimental_hz
                )

            else:

                experimental_hz = np.nan

                delta_hz = np.nan

                relative_error_percent = np.nan

            all_results.append({
                "m": m,
                "q": q,
                "f_model_hz": pole[
                    "f_re"
                ],
                "f_model_im_hz": pole[
                    "f_im"
                ],
                "abs_res": pole[
                    "abs_res"
                ],
                "root_success": pole[
                    "root_success"
                ],
                "root_nfev": pole.get(
                    "root_nfev",
                    -1,
                ),
                "root_message": pole.get(
                    "root_message",
                    "",
                ),
                "seed_f_hz": pole.get(
                    "seed_f",
                    np.nan,
                ),
                "real_axis_abs_res": pole[
                    "real_axis_abs_res"
                ],
                "light_ring_hz": light_ring_hz,
                "experimental_hz": experimental_hz,
                "delta_hz": delta_hz,
                "relative_error_percent":
                    relative_error_percent,
            })

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    write_results_csv(
        all_results
    )

    print_summary(
        all_results
    )

    # ------------------------------------------------------------------
    # Final statistics
    # ------------------------------------------------------------------

    total_expected = (
        len(M_VALUES)
        * EXPECTED_POLES
    )

    total_found = len(
        all_results
    )

    print(
        "=" * 78
    )

    print(
        "FINAL STATUS"
    )

    print(
        "=" * 78
    )

    print()

    print(
        f"m modes scanned: "
        f"{len(M_VALUES)}"
    )

    print(
        f"expected poles: "
        f"{total_expected}"
    )

    print(
        f"validated poles found: "
        f"{total_found}"
    )

    print()

    if total_found == total_expected:

        print(
            "All requested q=1..4 poles were validated."
        )

    else:

        print(
            "WARNING:"
        )

        print(
            "Not all requested q=1..4 poles were found."
        )

        print(
            "This is a numerical/search result, not a reason"
        )

        print(
            "to manufacture missing poles or assign experimental"
        )

        print(
            "frequencies to unvalidated candidates."
        )

    print()

    print(
        "STAGE 5I-4 v5 FIXED COMPLETE"
    )

    print(
        "=" * 78
    )

    return 0


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    raise SystemExit(
        main()
    )

