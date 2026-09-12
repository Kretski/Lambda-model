"""
stage5I_4_full_m_scan_v2.py
===========================

STAGE 5I-4 v2 — FULL m-SCAN / MULTIPLE OVERTONE SEARCH

Purpose
-------
Compare the validated Stage 5I-3 v2 resonance model against the four
experimental counter-rotating resonances of EXP B for

    m = -21 ... -4

without fitting parameters and without forcing the model to the
experimental frequencies.

IMPORTANT:
    This file does NOT modify the physics of stage5I_3_resonance_v2.py.

It imports the validated v2 implementation and changes ONLY the
multiple-pole search / candidate-selection / pole-filtering layer.

Why v2 of the m-scan is needed
------------------------------
The previous Stage 5I-4 search selected only a small number of the
strongest real-axis minima globally.

That can fail for multiple overtones because:

    candidate 1
    candidate 2
    candidate 3
       ...
       all belong to the SAME pole

while weaker minima corresponding to q=2, q=3, q=4 are discarded
before complex refinement.

The present implementation therefore:

    1. scans the complete real-frequency interval;
    2. finds ALL local real-axis minima;
    3. suppresses only redundant minima that are extremely close;
    4. refines each remaining candidate locally in the complex plane;
    5. root-polishes the complex resonance with scipy.optimize.root;
    6. rejects numerical/non-physical results;
    7. clusters duplicate complex poles;
    8. sorts physical poles by Re(f);
    9. takes the first four model poles;
   10. compares them with experimental q=1..4.

The experimental frequencies are NOT used as complex-refinement seeds.

They are used only for:
    - defining the total scan interval;
    - reporting model-vs-experiment differences.

Physics source
--------------
    stage5I_3_resonance_v2.py

The baseline physical parameters originate in:
    stage5I_1_2_dispersion_radial_p.py

The baseline uses the full transcendental dispersion relation
and no Lambda term.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from scipy.optimize import minimize, root


# ============================================================================
# IMPORT VALIDATED STAGE 5I-3 v2
# ============================================================================

HERE = Path(__file__).resolve().parent

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


try:

    from stage5I_3_resonance_v2 import (
        R_B,
        find_light_ring,
        resonance_factor,
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
        "    stage5I_4_full_m_scan_v2.py\n"
    ) from exc


# ============================================================================
# EXP B DATA
# ============================================================================
#
# These are the four counter-rotating experimental resonances supplied in
# qnms.csv for EXP B.
#
# m, q1, q2, q3, q4
#
# Values are in Hz.
# ============================================================================

EXP_B = {

    -21: [
        10.53127805,
        11.31559458,
        11.73305339,
        12.04931005,
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
# NUMERICAL SETTINGS
# ============================================================================

# Dense real-axis scan.
N_COARSE = 10000

# The experiment defines the outer frequency window.
SEARCH_PADDING = 0.25

# Do not discard weaker minima globally.
#
# Instead, candidates are locally thinned by frequency.
CANDIDATE_SEPARATION = 0.015

# Maximum number of real-axis candidates sent to complex refinement.
# This is deliberately generous.
MAX_REAL_CANDIDATES = 100

# Complex minimization.
COMPLEX_RE_HALF_WIDTH = 0.045

# Physical damped half-plane.
IM_MIN = -1.5
IM_MAX = 0.0

# Initial imaginary seed.
IM0 = -0.01

# Root-polish tolerance.
ROOT_XTOL = 1.0e-10

# Pole acceptance.
ROOT_RESIDUAL_TARGET = 1.0e-8

# A stricter diagnostic threshold used in the output.
ROOT_RESIDUAL_GOOD = 1.0e-10

# Duplicate-pole clustering threshold in Hz.
POLE_CLUSTER_DISTANCE = 2.0e-4

# Reject poles that move too far away from the original real candidate.
#
# This is important: the old search allowed a candidate at 6.78 Hz
# to wander to a completely unrelated point.
MAX_REFINEMENT_SHIFT = 0.12

# A physical pole should have a negative imaginary part.
# A tiny positive value caused by numerical noise is tolerated.
UPPER_HALF_PLANE_TOL = 1.0e-10

# Four experimental modes.
N_MODES = 4


# ============================================================================
# SAFE RESONANCE EVALUATION
# ============================================================================

def safe_res(
    f_re,
    f_im,
    m,
):
    """
    Evaluate Res(omega) safely.

    Input:
        f_re, f_im in Hz.

    Internally:
        omega = 2*pi*f.
    """

    f = complex(
        float(f_re),
        float(f_im),
    )

    omega = (
        2.0
        * np.pi
        * f
    )

    try:

        z = resonance_factor(
            omega=omega,
            m=m,
            complex_action=(
                abs(f_im) > 1.0e-13
            ),
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

    try:

        if not (
            np.isfinite(z.real)
            and np.isfinite(z.imag)
        ):

            return None

    except Exception:

        return None

    return complex(z)


# ============================================================================
# REAL-AXIS SCAN
# ============================================================================

def scan_real_axis(
    m,
    f_min,
    f_max,
    n_coarse=N_COARSE,
):
    """
    Full real-axis scan.

    Returns:
        f_grid
        abs_res
        local_minima

    local_minima is deliberately NOT restricted to the strongest
    minima globally.

    This is the principal change relative to the old Stage 5I-4
    candidate selection.
    """

    f_grid = np.linspace(
        f_min,
        f_max,
        n_coarse,
    )

    abs_res = np.full(
        n_coarse,
        np.inf,
        dtype=float,
    )

    for i, f in enumerate(
        f_grid
    ):

        z = safe_res(
            f,
            0.0,
            m,
        )

        if z is None:

            continue

        value = abs(z)

        if np.isfinite(value):

            abs_res[i] = value

    finite = np.isfinite(
        abs_res
    )

    if not np.any(finite):

        return (
            f_grid,
            abs_res,
            [],
        )

    candidates = []

    for i in range(
        1,
        n_coarse - 1,
    ):

        if not np.isfinite(
            abs_res[i]
        ):

            continue

        left = abs_res[
            i - 1
        ]

        center = abs_res[
            i
        ]

        right = abs_res[
            i + 1
        ]

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
                "f": float(
                    f_grid[i]
                ),
                "abs_res": float(
                    center
                ),
                "index": int(i),
            })

    # Include the best finite point if there are no interior minima.
    if not candidates:

        finite_indices = np.flatnonzero(
            finite
        )

        best_i = finite_indices[
            np.argmin(
                abs_res[
                    finite_indices
                ]
            )
        ]

        candidates.append({
            "f": float(
                f_grid[best_i]
            ),
            "abs_res": float(
                abs_res[best_i]
            ),
            "index": int(best_i),
        })

    # Sort by frequency first.
    #
    # We want broad coverage of the frequency axis, not just the
    # deepest minima around the first pole.
    candidates.sort(
        key=lambda x: x["f"]
    )

    return (
        f_grid,
        abs_res,
        candidates,
    )


# ============================================================================
# CANDIDATE THINNING
# ============================================================================

def thin_real_candidates(
    candidates,
    min_separation=CANDIDATE_SEPARATION,
    max_candidates=MAX_REAL_CANDIDATES,
):
    """
    Remove redundant real-axis minima.

    Important:
        We do NOT select globally strongest minima.

    Instead, nearby minima are treated as potentially belonging to the
    same local basin. The strongest member of each small frequency
    neighborhood is retained.

    This prevents:

        20 candidates -> same pole

    while retaining weaker minima farther away that may correspond
    to q=2, q=3, q=4.
    """

    if not candidates:

        return []

    candidates = sorted(
        candidates,
        key=lambda x: x["f"],
    )

    selected = []

    # Local clustering by frequency.
    clusters = []

    current = [
        candidates[0]
    ]

    for candidate in candidates[1:]:

        if (
            candidate["f"]
            - current[-1]["f"]
            <= min_separation
        ):

            current.append(
                candidate
            )

        else:

            clusters.append(
                current
            )

            current = [
                candidate
            ]

    clusters.append(
        current
    )

    # Keep the deepest minimum in each local cluster.
    for cluster in clusters:

        best = min(
            cluster,
            key=lambda x:
                x["abs_res"],
        )

        selected.append(
            best
        )

    # If there are too many, use evenly distributed frequency bins.
    if len(selected) > max_candidates:

        selected_sorted = sorted(
            selected,
            key=lambda x:
                x["f"],
        )

        positions = np.linspace(
            0,
            len(selected_sorted) - 1,
            max_candidates,
        )

        indices = np.unique(
            np.round(
                positions
            ).astype(int)
        )

        selected = [
            selected_sorted[i]
            for i in indices
        ]

    selected.sort(
        key=lambda x:
            x["f"]
    )

    return selected


# ============================================================================
# COMPLEX LOCAL REFINEMENT
# ============================================================================

def complex_refine(
    m,
    f0,
):
    """
    Refine one real-axis candidate into a nearby complex resonance.

    Unlike the old Stage 5I-4 implementation, the allowed Re(f)
    displacement is intentionally small.

    Therefore a candidate around q=2 cannot simply jump to q=1.
    """

    lo_re = (
        f0
        - MAX_REFINEMENT_SHIFT
    )

    hi_re = (
        f0
        + MAX_REFINEMENT_SHIFT
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

            return 1.0e50

        if not (
            IM_MIN
            <= f_im
            <= IM_MAX
        ):

            return 1.0e50

        z = safe_res(
            f_re,
            f_im,
            m,
        )

        if z is None:

            return 1.0e50

        value = abs(z)

        if not np.isfinite(
            value
        ):

            return 1.0e50

        # log1p avoids huge objective dynamic range.
        return float(
            np.log1p(value)
        )

    result = minimize(
        objective,
        x0=np.array([
            f0,
            IM0,
        ]),
        method="Nelder-Mead",
        options={
            "xatol": 1.0e-9,
            "fatol": 1.0e-10,
            "maxiter": 800,
        },
    )

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

    # Strict local containment.
    if not (
        lo_re
        <= f_re
        <= hi_re
    ):

        return None

    if not (
        IM_MIN
        <= f_im
        <= IM_MAX
    ):

        return None

    z = safe_res(
        f_re,
        f_im,
        m,
    )

    if z is None:

        return None

    abs_res = abs(z)

    if not np.isfinite(
        abs_res
    ):

        return None

    return {
        "seed_f": f0,
        "f_re": f_re,
        "f_im": f_im,
        "abs_res": float(
            abs_res
        ),
        "minimize_success":
            bool(result.success),
        "minimize_message":
            str(result.message),
    }


# ============================================================================
# ROOT POLISH
# ============================================================================

def polish_root(
    m,
    refined,
):
    """
    Solve

        Re Res = 0
        Im Res = 0

    with scipy.optimize.root('hybr').

    Returns a physical pole candidate or None.
    """

    if refined is None:

        return None

    x0 = np.array([
        refined["f_re"],
        refined["f_im"],
    ])

    def equations(
        x
    ):

        f_re = float(
            x[0]
        )

        f_im = float(
            x[1]
        )

        # Keep root iterations in the damped half-plane and close
        # to the candidate. Returning a large finite residual rather
        # than throwing prevents scipy from wandering into overflow.
        if (
            f_im > 0.05
            or f_im < IM_MIN - 0.5
            or abs(
                f_re
                - refined["seed_f"]
            )
            > MAX_REFINEMENT_SHIFT * 1.5
        ):

            return np.array([
                1.0e3,
                1.0e3,
            ])

        z = safe_res(
            f_re,
            f_im,
            m,
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

    try:

        result = root(
            equations,
            x0,
            method="hybr",
            options={
                "xtol": ROOT_XTOL,
                "maxfev": 100,
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

    # Must remain close to the candidate that generated the root.
    if (
        abs(
            f_re
            - refined["seed_f"]
        )
        > MAX_REFINEMENT_SHIFT * 1.5
    ):

        return None

    # We seek damped resonances.
    if (
        f_im
        > UPPER_HALF_PLANE_TOL
    ):

        return None

    if (
        f_im
        < IM_MIN - 0.5
    ):

        return None

    z = safe_res(
        f_re,
        f_im,
        m,
    )

    if z is None:

        return None

    residual = abs(z)

    if not np.isfinite(
        residual
    ):

        return None

    # A genuine root must satisfy this.
    if residual > ROOT_RESIDUAL_TARGET:

        return None

    return {
        "seed_f":
            refined["seed_f"],

        "f_re":
            f_re,

        "f_im":
            f_im,

        "abs_res":
            float(residual),

        "success":
            bool(result.success),

        "nfev":
            int(
                getattr(
                    result,
                    "nfev",
                    -1,
                )
            ),

        "message":
            str(result.message),
    }


# ============================================================================
# POLE CLUSTERING
# ============================================================================

def cluster_poles(
    poles,
    cluster_distance=POLE_CLUSTER_DISTANCE,
):
    """
    Merge repeated convergences to the same complex pole.

    The old output showed the same pole many times because several
    real-axis minima in the same basin all converged to the identical
    complex root.

    Here we retain only the best-residual representative.
    """

    if not poles:

        return []

    poles = sorted(
        poles,
        key=lambda p:
            (
                p["f_re"],
                p["f_im"],
            )
    )

    clusters = []

    current = [
        poles[0]
    ]

    for pole in poles[1:]:

        ref = current[-1]

        distance = np.sqrt(
            (
                pole["f_re"]
                - ref["f_re"]
            ) ** 2
            +
            (
                pole["f_im"]
                - ref["f_im"]
            ) ** 2
        )

        if distance <= cluster_distance:

            current.append(
                pole
            )

        else:

            clusters.append(
                current
            )

            current = [
                pole
            ]

    clusters.append(
        current
    )

    unique = []

    for cluster in clusters:

        best = min(
            cluster,
            key=lambda p:
                p["abs_res"],
        )

        unique.append(
            best
        )

    unique.sort(
        key=lambda p:
            p["f_re"]
    )

    return unique


# ============================================================================
# PHYSICAL POLE FILTER
# ============================================================================

def filter_physical_poles(
    poles,
    f_min,
    f_max,
):
    """
    Final physical/numerical filtering.

    Conditions:

        - finite
        - root residual <= ROOT_RESIDUAL_TARGET
        - scipy root success
        - Re(f) inside search interval
        - Im(f) <= 0
        - no NaN/Inf
    """

    good = []

    for pole in poles:

        f_re = pole["f_re"]
        f_im = pole["f_im"]
        residual = pole["abs_res"]

        if not (
            np.isfinite(f_re)
            and np.isfinite(f_im)
            and np.isfinite(residual)
        ):

            continue

        if not pole["success"]:

            continue

        if residual > ROOT_RESIDUAL_TARGET:

            continue

        if (
            f_re < f_min
            or f_re > f_max
        ):

            continue

        if (
            f_im
            > UPPER_HALF_PLANE_TOL
        ):

            continue

        good.append(
            pole
        )

    return good


# ============================================================================
# FIND MODEL POLES FOR ONE m
# ============================================================================

def find_model_poles(
    m,
    f_min,
    f_max,
):
    """
    Full multiple-overtone search for one azimuthal mode.
    """

    print()
    print(
        "Running real-axis scan:"
        f" {N_COARSE} points"
    )

    (
        f_grid,
        abs_res,
        raw_candidates,
    ) = scan_real_axis(
        m,
        f_min,
        f_max,
        N_COARSE,
    )

    finite = np.isfinite(
        abs_res
    )

    if np.any(finite):

        best_index = np.nanargmin(
            np.where(
                finite,
                abs_res,
                np.nan,
            )
        )

        print(
            "Best real-axis point:"
        )

        print(
            f"    f = "
            f"{f_grid[best_index]:.10f} Hz"
        )

        print(
            f"    |Res| = "
            f"{abs_res[best_index]:.8e}"
        )

    print()

    print(
        "Raw real-axis local minima:"
        f" {len(raw_candidates)}"
    )

    candidates = thin_real_candidates(
        raw_candidates,
    )

    print(
        "Candidates after local thinning:"
        f" {len(candidates)}"
    )

    if candidates:

        print()
        print(
            "Selected real-axis candidates:"
        )

        for i, candidate in enumerate(
            candidates,
            start=1,
        ):

            print(
                f"    {i:3d}: "
                f"f={candidate['f']:.10f} Hz "
                f"|Res|="
                f"{candidate['abs_res']:.8e}"
            )

    # ------------------------------------------------------------------
    # Complex refinement
    # ------------------------------------------------------------------

    print()
    print(
        "Complex refinement + root polish:"
    )

    polished = []

    for i, candidate in enumerate(
        candidates,
        start=1,
    ):

        f0 = candidate[
            "f"
        ]

        print()
        print(
            f"    Candidate {i}: "
            f"f0={f0:.10f} Hz"
        )

        refined = complex_refine(
            m,
            f0,
        )

        if refined is None:

            print(
                "        complex refinement "
                "FAILED"
            )

            continue

        print(
            f"        NM: "
            f"f={refined['f_re']:.10f}"
            f"{refined['f_im']:+.10f}i Hz"
        )

        print(
            f"        NM |Res|="
            f"{refined['abs_res']:.8e}"
        )

        pole = polish_root(
            m,
            refined,
        )

        if pole is None:

            print(
                "        root polish: "
                "REJECTED"
            )

            continue

        print(
            f"        ROOT: "
            f"f={pole['f_re']:.10f}"
            f"{pole['f_im']:+.10f}i Hz"
        )

        print(
            f"        |Res|="
            f"{pole['abs_res']:.8e}"
        )

        print(
            f"        success="
            f"{pole['success']} "
            f"nfev={pole['nfev']}"
        )

        polished.append(
            pole
        )

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    filtered = filter_physical_poles(
        polished,
        f_min,
        f_max,
    )

    print()
    print(
        "Accepted physical roots before "
        "duplicate clustering:"
        f" {len(filtered)}"
    )

    # ------------------------------------------------------------------
    # Cluster duplicates
    # ------------------------------------------------------------------

    unique = cluster_poles(
        filtered
    )

    return (
        unique,
        candidates,
    )


# ============================================================================
# EXPERIMENT / MODEL MATCHING
# ============================================================================

def compare_four_modes(
    m,
    experimental,
    model_poles,
):
    """
    Direct q=1..4 comparison.

    The model poles are ordered by increasing Re(f).

    No fitting is performed.
    """

    experimental = list(
        experimental
    )

    model = sorted(
        model_poles,
        key=lambda p:
            p["f_re"]
    )

    print()
    print(
        "-" * 92
    )

    print(
        f"COMPARISON m={m}"
    )

    print(
        "-" * 92
    )

    print()

    print(
        " q | experimental Hz | model Re(f) Hz "
        "| delta Hz | rel. delta % | Im(f) Hz | |Res|"
    )

    print(
        "---+-----------------+----------------"
        "+-----------+---------------+-----------+----------"
    )

    n_match = min(
        N_MODES,
        len(model),
    )

    rows = []

    for q in range(
        N_MODES
    ):

        f_exp = float(
            experimental[q]
        )

        if q < n_match:

            pole = model[q]

            f_model = pole[
                "f_re"
            ]

            f_im = pole[
                "f_im"
            ]

            residual = pole[
                "abs_res"
            ]

            delta = (
                f_model
                - f_exp
            )

            relative = (
                100.0
                * delta
                / f_exp
            )

            print(
                f" {q+1:1d} | "
                f"{f_exp:15.9f} | "
                f"{f_model:14.9f} | "
                f"{delta:+9.6f} | "
                f"{relative:+13.6f} | "
                f"{f_im:+9.6f} | "
                f"{residual:.2e}"
            )

            rows.append({
                "q":
                    q + 1,
                "f_exp":
                    f_exp,
                "f_model":
                    f_model,
                "f_im":
                    f_im,
                "delta":
                    delta,
                "relative_percent":
                    relative,
                "abs_res":
                    residual,
            })

        else:

            print(
                f" {q+1:1d} | "
                f"{f_exp:15.9f} | "
                f"{'MISSING':>14} | "
                f"{'--':>9} | "
                f"{'--':>13} | "
                f"{'--':>9} | "
                f"{'--':>8}"
            )

            rows.append({
                "q":
                    q + 1,
                "f_exp":
                    f_exp,
                "f_model":
                    None,
                "f_im":
                    None,
                "delta":
                    None,
                "relative_percent":
                    None,
                "abs_res":
                    None,
            })

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    matched = [
        row
        for row in rows
        if row["f_model"] is not None
    ]

    if len(matched) == 4:

        deltas = np.array([
            row["delta"]
            for row in matched
        ])

        rms = float(
            np.sqrt(
                np.mean(
                    deltas ** 2
                )
            )
        )

        mean_abs = float(
            np.mean(
                np.abs(
                    deltas
                )
            )
        )

        print()
        print(
            "Four-mode match available."
        )

        print(
            f"    mean |delta f| = "
            f"{mean_abs:.8e} Hz"
        )

        print(
            f"    RMS delta f    = "
            f"{rms:.8e} Hz"
        )

        status = (
            "FOUR MODES FOUND"
        )

    else:

        print()
        print(
            f"Only {len(matched)}/4 "
            "model poles found."
        )

        status = (
            f"INCOMPLETE "
            f"({len(matched)}/4)"
        )

    return {
        "m":
            m,
        "rows":
            rows,
        "n_model":
            len(model),
        "status":
            status,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 100)

    print(
        "STAGE 5I-4 v2 — FULL m-SCAN"
    )

    print(
        "VALIDATED STAGE 5I-3 v2 "
        "MULTIPLE-OVERTONE SEARCH vs EXP B"
    )

    print("=" * 100)

    print()

    print(
        "Physics:"
    )

    print(
        "    stage5I_3_resonance_v2.py"
    )

    print(
        "    NO physics changes in this file."
    )

    print()

    print(
        "Experimental modes:"
    )

    print(
        "    -21 ... -4"
    )

    print(
        f"Number of m values: "
        f"{len(EXP_B)}"
    )

    print(
        "Four experimental resonances per m."
    )

    print()

    print(
        "Search strategy:"
    )

    print(
        f"    real-axis points        = "
        f"{N_COARSE}"
    )

    print(
        f"    search padding          = "
        f"{SEARCH_PADDING:.3f} Hz"
    )

    print(
        f"    candidate separation    = "
        f"{CANDIDATE_SEPARATION:.3f} Hz"
    )

    print(
        f"    max candidates          = "
        f"{MAX_REAL_CANDIDATES}"
    )

    print(
        f"    complex Re window       = "
        f"+/- {MAX_REFINEMENT_SHIFT:.3f} Hz"
    )

    print(
        f"    root residual target    = "
        f"{ROOT_RESIDUAL_TARGET:.1e}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "    No parameters are fitted."
    )

    print(
        "    Experimental frequencies are NOT used as"
    )

    print(
        "    complex-refinement seeds."
    )

    print()

    all_results = []

    # =========================================================================
    # m LOOP
    # =========================================================================

    for m in sorted(
        EXP_B.keys()
    ):

        experimental = EXP_B[
            m
        ]

        print()
        print()
        print(
            "=" * 100
        )

        print(
            f"m = {m}"
        )

        print(
            "=" * 100
        )

        print()

        print(
            "Experimental frequencies:"
        )

        for q, f in enumerate(
            experimental,
            start=1,
        ):

            print(
                f"    q={q}: "
                f"{f:.9f} Hz"
            )

        # ------------------------------------------------------------------
        # Light ring
        # ------------------------------------------------------------------

        r_sp, omega_lr = (
            find_light_ring(
                m
            )
        )

        if (
            r_sp is None
            or omega_lr is None
        ):

            print()
            print(
                "Light ring: NOT FOUND"
            )

            all_results.append({
                "m":
                    m,
                "status":
                    "NO LIGHT RING",
                "model_poles":
                    [],
                "comparison":
                    None,
            })

            continue

        f_lr = (
            omega_lr
            / (
                2.0
                * np.pi
            )
        )

        print()

        print(
            f"Light ring: "
            f"r_sp={r_sp * 1.0e3:.6f} mm  "
            f"f_lr={f_lr:.9f} Hz"
        )

        # ------------------------------------------------------------------
        # Search interval
        # ------------------------------------------------------------------

        f_min = max(
            0.0,
            min(experimental)
            - SEARCH_PADDING,
        )

        f_max = (
            max(experimental)
            + SEARCH_PADDING
        )

        print()

        print(
            f"Model search interval: "
            f"[{f_min:.6f}, "
            f"{f_max:.6f}] Hz"
        )

        # ------------------------------------------------------------------
        # Model poles
        # ------------------------------------------------------------------

        model_poles, candidates = (
            find_model_poles(
                m,
                f_min,
                f_max,
            )
        )

        print()
        print(
            "Unique physical model poles:"
            f" {len(model_poles)}"
        )

        if model_poles:

            for i, pole in enumerate(
                model_poles,
                start=1,
            ):

                print(
                    f"    pole {i}: "
                    f"{pole['f_re']:.10f}"
                    f"{pole['f_im']:+.10f}i Hz "
                    f"|Res|="
                    f"{pole['abs_res']:.3e}"
                )

        else:

            print(
                "    NONE"
            )

        # ------------------------------------------------------------------
        # Direct comparison
        # ------------------------------------------------------------------

        comparison = (
            compare_four_modes(
                m,
                experimental,
                model_poles,
            )
        )

        all_results.append({
            "m":
                m,
            "status":
                comparison["status"],
            "model_poles":
                model_poles,
            "comparison":
                comparison,
        })

    # =========================================================================
    # GLOBAL SUMMARY
    # =========================================================================

    print()
    print()
    print(
        "=" * 100
    )

    print(
        "GLOBAL STAGE 5I-4 v2 SUMMARY"
    )

    print(
        "=" * 100
    )

    print()

    print(
        " m | exp q1 | model q1 | "
        "exp q2 | model q2 | "
        "exp q3 | model q3 | "
        "exp q4 | model q4"
    )

    print(
        "---+--------+----------+"
        "--------+----------+"
        "--------+----------+"
        "--------+----------"
    )

    total_four_mode = 0

    total_poles = 0

    for result in all_results:

        m = result[
            "m"
        ]

        poles = result[
            "model_poles"
        ]

        total_poles += len(
            poles
        )

        if (
            len(poles)
            >= 4
        ):

            total_four_mode += 1

        exp = EXP_B[
            m
        ]

        model = sorted(
            poles,
            key=lambda p:
                p["f_re"]
        )

        values = []

        for q in range(4):

            values.append(
                f"{exp[q]:.4f}"
            )

            if q < len(model):

                values.append(
                    f"{model[q]['f_re']:.4f}"
                )

            else:

                values.append(
                    "   --  "
                )

        print(
            f"{m:3d} | "
            f"{values[0]:>6} | "
            f"{values[1]:>8} | "
            f"{values[2]:>6} | "
            f"{values[3]:>8} | "
            f"{values[4]:>6} | "
            f"{values[5]:>8} | "
            f"{values[6]:>6} | "
            f"{values[7]:>8}"
        )

    print()

    print(
        f"Total unique physical poles: "
        f"{total_poles}"
    )

    print(
        f"m values with >=4 model poles: "
        f"{total_four_mode}/{len(EXP_B)}"
    )

    # =========================================================================
    # GLOBAL DELTA STATISTICS
    # =========================================================================

    all_deltas = []

    for result in all_results:

        comparison = result[
            "comparison"
        ]

        if comparison is None:

            continue

        for row in comparison[
            "rows"
        ]:

            if row[
                "delta"
            ] is not None:

                all_deltas.append(
                    row["delta"]
                )

    if all_deltas:

        all_deltas = np.array(
            all_deltas,
            dtype=float,
        )

        print()

        print(
            "Global model-vs-experiment "
            "statistics for matched modes:"
        )

        print(
            f"    N matched = "
            f"{len(all_deltas)}"
        )

        print(
            f"    mean delta f = "
            f"{np.mean(all_deltas):+.8e} Hz"
        )

        print(
            f"    mean |delta f| = "
            f"{np.mean(np.abs(all_deltas)):.8e} Hz"
        )

        print(
            f"    RMS delta f = "
            f"{np.sqrt(np.mean(all_deltas**2)):.8e} Hz"
        )

        print(
            f"    max |delta f| = "
            f"{np.max(np.abs(all_deltas)):.8e} Hz"
        )

    else:

        print()

        print(
            "No model/experiment frequency pairs "
            "were available."
        )

    # =========================================================================
    # FINAL INTERPRETATION
    # =========================================================================

    print()
    print(
        "-" * 100
    )

    print(
        "INTERPRETATION"
    )

    print(
        "-" * 100
    )

    print()

    if total_four_mode == len(
        EXP_B
    ):

        print(
            "All m values produced at least "
            "four physical model poles."
        )

        print(
            "The four lowest-Re(f) poles have "
            "been compared directly with EXP B q=1..4."
        )

    else:

        print(
            "The model/search did NOT produce "
            "four physical poles for every m."
        )

        print(
            "This is a search/model result, NOT "
            "a forced four-mode fit."
        )

        print(
            "Missing poles must be investigated "
            "before concluding that the physical "
            "model fails to reproduce the "
            "experimental overtones."
        )

    print()

    print(
        "Stage 5I-3 v2 physics was kept unchanged."
    )

    print(
        "Only multiple-overtone candidate search, "
        "complex refinement containment, "
        "root filtering and pole clustering "
        "were changed in this file."
    )

    print()

    print(
        "=" * 100
    )

    print(
        "STAGE 5I-4 v2 COMPLETE"
    )

    print(
        "=" * 100
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()