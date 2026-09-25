"""
STAGE 6.5B-1 — RESONANCE-LEVEL C-CAUSALITY AUDIT
=================================================

Purpose
-------
Test whether the complex resonance responds causally to C.

Controlled cases:

    C_ZERO     = 0
    C_HALF     = C0/2
    BASELINE   = C0
    C_DOUBLE   = 2*C0

For every case with an interior light ring:

    1. verify explicit-C light ring
    2. inject ONLY C into complex solver IN MEMORY
    3. verify complex Hamiltonian responds to C
    4. locate real last-scattering point
    5. locate complex turning point
    6. continue the complex radial branch
    7. compute radial action
    8. compute reflection coefficient
    9. evaluate

           Res(omega) = R(omega) exp(2 i S(omega)) - 1

   10. locally minimize |Res| in complex-frequency space
   11. report f_R, f_I, Q and residual

IMPORTANT
---------
This is a mathematical/numerical causality test.

It does NOT establish:
    - a new fundamental interaction
    - a Kerr correspondence
    - a GW interpretation
    - an experimentally realized physical system

No:
    - Kerr fit
    - GW calibration
    - frequency rescaling
    - low-k approximation
    - Lambda term
    - authoritative solver modification

C HANDLING
----------
The real light-ring API exposes C explicitly.

The complex API uses module-level constants.

Therefore C is temporarily injected into the authoritative solver
module IN MEMORY ONLY.

The script explicitly fingerprints the complex Hamiltonian at fixed
(omega,p,r) for C/2, C and 2C.

If the Hamiltonian does not respond, the test aborts.

COMPATIBILITY SHIM
------------------
The validated radial-action patch imports:

    DEFAULT_COMPLEX_RADIAL_N

The authoritative solver currently does not expose that name.

Therefore this script installs:

    solver.DEFAULT_COMPLEX_RADIAL_N = 20000

IN MEMORY ONLY before loading the patch.

No source file is modified.
"""

from pathlib import Path
import sys
import importlib.util
import inspect
import json
import csv
import copy

import numpy as np
from scipy.optimize import minimize


# ============================================================================
# PATHS
# ============================================================================

HERE = Path(__file__).resolve().parent

RES_FILE = HERE / "stage5I_3_resonance.py"
PATCH_FILE = HERE / "stage5I_3c_radial_action_patch.py"

OUT_DIR = HERE / "stage6_5B1_resonance_causal_audit"
OUT_DIR.mkdir(exist_ok=True)

CSV_FILE = OUT_DIR / "stage6_5B1_resonance_causal_results.csv"
JSON_FILE = OUT_DIR / "stage6_5B1_resonance_causal_results.json"


# ============================================================================
# AUTHORITATIVE ROOT-B CONSTANTS
# ============================================================================

M_MODE = -12

C0 = 7.690000000000e-04
OMEGA0 = 1.000000000000e-01
GAMMA0 = 2.220000000000e-06
H00 = 3.400000000000e-02
G0 = 9.810000000000e+00
R_B = 3.730000000000e-02


# ============================================================================
# CONTROLLED C CASES
# ============================================================================

CASES = [
    ("C_ZERO", 0.0),
    ("C_HALF", 0.5 * C0),
    ("BASELINE", C0),
    ("C_DOUBLE", 2.0 * C0),
]


# ============================================================================
# KNOWN RESONANCE STARTING POINTS
# ============================================================================

# Historical Stage 5I-3 estimate.
F_RE_START = 8.3066307032346
F_IM_START = -0.0005696389388967092

# Later validated benchmark point.
F_RE_BENCH = 8.306630716900
F_IM_BENCH = -0.000566435000


# ============================================================================
# NUMERICAL CONTROLS
# ============================================================================

N_ACTION = 20000

H_TOL = 1.0e-8

MAX_SUBDIVISIONS = 12

F_RE_HALF_WIDTH = 0.08
F_IM_HALF_WIDTH = 0.004

F_RE_STEP = 0.01
F_IM_STEP = 0.0002


# ============================================================================
# MODULE LOADER
# ============================================================================

def load_module(path, name):

    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot create import spec for:\n{path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    module.__file__ = str(path)
    module.__name__ = name

    sys.modules[name] = module

    spec.loader.exec_module(module)

    return module


# ============================================================================
# LOAD RADIAL PATCH
# ============================================================================

def load_patch(solver):

    """
    Load the validated radial-action patch.

    The patch expects DEFAULT_COMPLEX_RADIAL_N, which is absent from
    the current authoritative solver.

    We install that symbol IN MEMORY ONLY before importing the patch.
    """

    print()
    print(
        "  Installing compatibility shim:"
    )

    if not hasattr(
        solver,
        "DEFAULT_COMPLEX_RADIAL_N",
    ):

        solver.DEFAULT_COMPLEX_RADIAL_N = N_ACTION

        print(
            "    solver.DEFAULT_COMPLEX_RADIAL_N = "
            f"{solver.DEFAULT_COMPLEX_RADIAL_N}"
        )

        print(
            "    [IN MEMORY ONLY]"
        )

    else:

        print(
            "    existing DEFAULT_COMPLEX_RADIAL_N = "
            f"{solver.DEFAULT_COMPLEX_RADIAL_N}"
        )

    if not hasattr(
        solver,
        "R_B",
    ):

        solver.R_B = R_B

        print(
            "    solver.R_B = "
            f"{solver.R_B}"
        )

        print(
            "    [IN MEMORY ONLY]"
        )

    # The patch imports literally:
    #
    #   from stage5I_3_resonance import ...
    #
    # Therefore expose the already loaded authoritative module
    # under that exact module name.

    sys.modules[
        "stage5I_3_resonance"
    ] = solver

    spec = importlib.util.spec_from_file_location(
        "_stage5I3c_patch_6_5B1",
        PATCH_FILE,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot load patch:\n{PATCH_FILE}"
        )

    patch = importlib.util.module_from_spec(
        spec
    )

    patch.__file__ = str(PATCH_FILE)
    patch.__name__ = "_stage5I3c_patch_6_5B1"

    sys.modules[
        patch.__name__
    ] = patch

    spec.loader.exec_module(
        patch
    )

    return patch


# ============================================================================
# C GLOBAL DISCOVERY
# ============================================================================

def discover_c_names(solver):

    candidates = [
        "C_METHODS",
        "C_TABLE",
        "C",
    ]

    return [
        name
        for name in candidates
        if hasattr(solver, name)
    ]


def snapshot_c_values(
    solver,
    names,
):

    return {
        name: getattr(
            solver,
            name,
        )
        for name in names
    }


def set_runtime_C(
    solver,
    names,
    C_value,
):

    for name in names:

        setattr(
            solver,
            name,
            float(C_value),
        )


def restore_runtime_C(
    solver,
    snapshot,
):

    for name, value in snapshot.items():

        setattr(
            solver,
            name,
            value,
        )


# ============================================================================
# COMPLEX HAMILTONIAN FINGERPRINT
# ============================================================================

def complex_H_at_C(
    solver,
    c_names,
    C_value,
    omega,
    p,
    m,
    r,
):

    old = snapshot_c_values(
        solver,
        c_names,
    )

    try:

        set_runtime_C(
            solver,
            c_names,
            C_value,
        )

        H = solver.hamiltonian(
            omega,
            p,
            m,
            r,
        )

        return complex(H)

    finally:

        restore_runtime_C(
            solver,
            old,
        )


# ============================================================================
# RESIDUAL
# ============================================================================

def H_residual(
    solver,
    omega,
    p,
    m,
    r,
):

    H = solver.hamiltonian(
        omega,
        p,
        m,
        r,
    )

    return float(
        abs(H)
    )


# ============================================================================
# COMPLEX ROOT CANDIDATE
# ============================================================================

def solve_p_candidate(
    solver,
    omega,
    m,
    r,
    seed,
):

    try:

        p = solver.solve_p_complex(
            omega,
            m,
            r,
            p_initial=complex(seed),
        )

    except Exception as exc:

        return (
            None,
            np.inf,
            repr(exc),
        )

    if p is None:

        return (
            None,
            np.inf,
            "solve_p_complex returned None",
        )

    p = complex(p)

    try:

        res = H_residual(
            solver,
            omega,
            p,
            m,
            r,
        )

    except Exception as exc:

        return (
            p,
            np.inf,
            f"residual failure: {exc!r}",
        )

    if not np.isfinite(res):

        return (
            p,
            np.inf,
            "non-finite residual",
        )

    if res > H_TOL:

        return (
            p,
            res,
            f"residual too large: {res:.6e}",
        )

    return (
        p,
        res,
        None,
    )


# ============================================================================
# BRANCH SELECTION
# ============================================================================

def choose_branch(
    candidates,
    p_previous,
):

    if not candidates:
        return None

    best = None
    best_distance = np.inf

    for p in candidates:

        p = complex(p)

        # The Hamiltonian radial equation has ±p roots.
        # Pick whichever sign is closest to the existing branch.

        if abs(-p - p_previous) < abs(
            p - p_previous
        ):

            p_use = -p

        else:

            p_use = p

        distance = abs(
            p_use - p_previous
        )

        if distance < best_distance:

            best_distance = distance
            best = p_use

    return best


# ============================================================================
# MULTI-SEED ROOT SOLVE
# ============================================================================

def solve_with_seeds(
    solver,
    omega,
    m,
    r,
    seeds,
    p_previous,
):

    candidates = []

    for label, seed in seeds:

        p, res, err = solve_p_candidate(
            solver,
            omega,
            m,
            r,
            seed,
        )

        if (
            p is not None
            and np.isfinite(res)
            and res <= H_TOL
        ):

            candidates.append(
                (
                    p,
                    res,
                    label,
                )
            )

    if not candidates:

        return (
            None,
            np.inf,
            None,
        )

    p_best = choose_branch(
        [
            x[0]
            for x in candidates
        ],
        p_previous,
    )

    for p, res, label in candidates:

        if abs(
            p - p_best
        ) < 1.0e-12:

            return (
                p_best,
                res,
                label,
            )

    return (
        p_best,
        H_residual(
            solver,
            omega,
            p_best,
            m,
            r,
        ),
        "branch-selected",
    )


# ============================================================================
# TURNING-POINT DERIVATIVES
# ============================================================================

def compute_H_r(
    solver,
    omega,
    m,
    r,
):

    eps = max(
        abs(r) * 1.0e-6,
        1.0e-9,
    )

    Hp = solver.hamiltonian(
        omega,
        0j,
        m,
        r + eps,
    )

    Hm = solver.hamiltonian(
        omega,
        0j,
        m,
        r - eps,
    )

    return (
        Hp - Hm
    ) / (
        2.0 * eps
    )


def compute_H_pp(
    solver,
    omega,
    m,
    r,
):

    # Use a modest momentum scale.
    eps = 1.0

    H0 = solver.hamiltonian(
        omega,
        0j,
        m,
        r,
    )

    Hp = solver.hamiltonian(
        omega,
        complex(eps),
        m,
        r,
    )

    Hm = solver.hamiltonian(
        omega,
        complex(-eps),
        m,
        r,
    )

    return (
        Hp
        - 2.0 * H0
        + Hm
    ) / (
        eps ** 2
    )


# ============================================================================
# INITIAL ANALYTIC SEED
# ============================================================================

def initial_turning_seed(
    solver,
    omega,
    m,
    r_turn,
    r_next,
):

    Hr = compute_H_r(
        solver,
        omega,
        m,
        r_turn,
    )

    Hpp = compute_H_pp(
        solver,
        omega,
        m,
        r_turn,
    )

    if (
        not np.isfinite(
            abs(Hr)
        )
        or not np.isfinite(
            abs(Hpp)
        )
    ):

        raise FloatingPointError(
            "Non-finite turning-point derivative."
        )

    radicand = (
        -2.0
        * Hr
        * (
            r_next
            - r_turn
        )
        / Hpp
    )

    if not np.isfinite(
        abs(radicand)
    ):

        raise FloatingPointError(
            "Non-finite analytic seed radicand."
        )

    return (
        np.sqrt(radicand),
        Hr,
        Hpp,
        radicand,
    )


# ============================================================================
# ADAPTIVE RADIAL ACTION
# ============================================================================

def radial_action_adaptive(
    solver,
    omega,
    m,
    r_turn,
    r_B,
    n=N_ACTION,
):

    """
    Complex radial continuation.

    Initial point:
        exact patch-style near-turning-point seed.

    Normal continuation:
        previous root
        linear predictor
        half predictor
        quarter predictor

    If all fail:
        recursively subdivide the radial interval.

    This is deliberately a numerical continuation repair.
    It does not modify the Hamiltonian.
    """

    coarse = np.linspace(
        complex(r_turn),
        complex(r_B),
        n,
    )

    r_first = coarse[1]

    p_seed, Hr, Hpp, radicand = (
        initial_turning_seed(
            solver,
            omega,
            m,
            r_turn,
            r_first,
        )
    )

    p_first, res_first, seed_label = (
        solve_with_seeds(
            solver,
            omega,
            m,
            r_first,
            [
                (
                    "analytic +",
                    p_seed,
                ),
                (
                    "analytic -",
                    -p_seed,
                ),
            ],
            0j,
        )
    )

    if p_first is None:

        raise FloatingPointError(
            "Analytic first-step seed failed."
        )

    points = [
        complex(r_turn),
        complex(r_first),
    ]

    momenta = [
        0j,
        complex(p_first),
    ]

    # ------------------------------------------------------------------
    # Recursive local bridge.
    # ------------------------------------------------------------------

    def bridge(
        r_left,
        p_left,
        r_right,
        p_previous_previous,
        depth,
    ):

        seeds = [
            (
                "previous",
                p_left,
            ),
        ]

        if p_previous_previous is not None:

            dp = (
                p_left
                - p_previous_previous
            )

            seeds.extend(
                [
                    (
                        "linear predictor",
                        p_left + dp,
                    ),
                    (
                        "half predictor",
                        p_left + 0.5 * dp,
                    ),
                    (
                        "quarter predictor",
                        p_left + 0.25 * dp,
                    ),
                ]
            )

        p_right, res, label = (
            solve_with_seeds(
                solver,
                omega,
                m,
                r_right,
                seeds,
                p_left,
            )
        )

        if p_right is not None:

            return [
                (
                    complex(r_right),
                    complex(p_right),
                )
            ]

        if depth >= MAX_SUBDIVISIONS:

            raise FloatingPointError(
                "Adaptive continuation exhausted "
                f"subdivisions at r={r_right}."
            )

        r_mid = (
            r_left
            + r_right
        ) / 2.0

        # First solve midpoint from left.
        left_piece = bridge(
            r_left,
            p_left,
            r_mid,
            p_previous_previous,
            depth + 1,
        )

        r_mid_solved = (
            left_piece[-1][0]
        )

        p_mid = (
            left_piece[-1][1]
        )

        # Then continue midpoint -> right.
        right_piece = bridge(
            r_mid_solved,
            p_mid,
            r_right,
            p_left,
            depth + 1,
        )

        return (
            left_piece
            + right_piece
        )

    # ------------------------------------------------------------------
    # Main continuation.
    # ------------------------------------------------------------------

    r_prev = complex(
        r_first
    )

    p_prev = complex(
        p_first
    )

    p_prevprev = None

    for i in range(
        2,
        len(coarse),
    ):

        r_target = complex(
            coarse[i]
        )

        seeds = [
            (
                "previous",
                p_prev,
            ),
        ]

        if p_prevprev is not None:

            dp = (
                p_prev
                - p_prevprev
            )

            seeds.extend(
                [
                    (
                        "linear predictor",
                        p_prev + dp,
                    ),
                    (
                        "half predictor",
                        p_prev
                        + 0.5 * dp,
                    ),
                    (
                        "quarter predictor",
                        p_prev
                        + 0.25 * dp,
                    ),
                ]
            )

        p_new, res, label = (
            solve_with_seeds(
                solver,
                omega,
                m,
                r_target,
                seeds,
                p_prev,
            )
        )

        if p_new is not None:

            points.append(
                r_target
            )

            momenta.append(
                complex(p_new)
            )

            p_prevprev = p_prev
            p_prev = complex(
                p_new
            )

            r_prev = r_target

            continue

        # --------------------------------------------------------------
        # Coarse step failed.
        # Bridge it adaptively.
        # --------------------------------------------------------------

        bridge_points = bridge(
            r_prev,
            p_prev,
            r_target,
            p_prevprev,
            0,
        )

        for rr, pp in bridge_points:

            points.append(
                complex(rr)
            )

            momenta.append(
                complex(pp)
            )

        if len(bridge_points) >= 2:

            p_prevprev = (
                bridge_points[-2][1]
            )

        else:

            p_prevprev = p_prev

        r_prev = (
            bridge_points[-1][0]
        )

        p_prev = (
            bridge_points[-1][1]
        )

    # ------------------------------------------------------------------
    # Clean duplicates.
    # ------------------------------------------------------------------

    clean_r = []
    clean_p = []

    for rr, pp in zip(
        points,
        momenta,
    ):

        if (
            clean_r
            and abs(
                rr - clean_r[-1]
            ) < 1.0e-18
        ):

            clean_p[-1] = pp

        else:

            clean_r.append(
                rr
            )

            clean_p.append(
                pp
            )

    clean_r = np.asarray(
        clean_r,
        dtype=complex,
    )

    clean_p = np.asarray(
        clean_p,
        dtype=complex,
    )

    if hasattr(
        np,
        "trapezoid",
    ):

        action = np.trapezoid(
            clean_p,
            clean_r,
        )

    else:

        action = np.trapz(
            clean_p,
            clean_r,
        )

    info = {
        "n_input": int(n),
        "n_used": int(
            len(clean_r)
        ),
        "max_abs_p": float(
            np.max(
                np.abs(clean_p)
            )
        ),
        "first_seed_label": seed_label,
        "H_r_abs": float(
            abs(Hr)
        ),
        "H_pp_abs": float(
            abs(Hpp)
        ),
        "radicand_abs": float(
            abs(radicand)
        ),
    }

    return (
        complex(action),
        info,
    )


# ============================================================================
# SINGLE COMPLEX RESONANCE EVALUATION
# ============================================================================

def evaluate_resonance(
    solver,
    C_value,
    f_re,
    f_im,
):

    c_names = discover_c_names(
        solver
    )

    old_c = snapshot_c_values(
        solver,
        c_names,
    )

    try:

        # --------------------------------------------------------------
        # Runtime C injection.
        # --------------------------------------------------------------

        set_runtime_C(
            solver,
            c_names,
            C_value,
        )

        omega = (
            2.0
            * np.pi
            * (
                float(f_re)
                + 1j * float(f_im)
            )
        )

        # --------------------------------------------------------------
        # Explicit-C light ring.
        # --------------------------------------------------------------

        r_sp, omega_lr = (
            solver.find_light_ring(
                M_MODE,
                C=C_value,
                Omega=OMEGA0,
                gamma=GAMMA0,
                h0=H00,
                g=G0,
                r_min=1.0e-3,
                r_max=R_B,
            )
        )

        if r_sp is None:

            return {
                "status": "NO_LIGHT_RING",
                "C": float(
                    C_value
                ),
                "f_re": float(
                    f_re
                ),
                "f_im": float(
                    f_im
                ),
            }

        # --------------------------------------------------------------
        # Last scattering point.
        # --------------------------------------------------------------

        try:

            r_minus_real = (
                solver.find_last_scattering_point(
                    omega,
                    M_MODE,
                    r_sp,
                )
            )

        except Exception as exc:

            return {
                "status":
                    "TURNING_POINT_REAL_FAILURE",
                "C": float(
                    C_value
                ),
                "f_re": float(
                    f_re
                ),
                "f_im": float(
                    f_im
                ),
                "r_sp": float(
                    r_sp
                ),
                "f_lr": float(
                    omega_lr
                    / (
                        2.0
                        * np.pi
                    )
                ),
                "error": repr(
                    exc
                ),
            }

        if r_minus_real is None:

            return {
                "status":
                    "TURNING_POINT_REAL_FAILURE",
                "C": float(
                    C_value
                ),
                "f_re": float(
                    f_re
                ),
                "f_im": float(
                    f_im
                ),
                "r_sp": float(
                    r_sp
                ),
                "f_lr": float(
                    omega_lr
                    / (
                        2.0
                        * np.pi
                    )
                ),
            }

        # --------------------------------------------------------------
        # Complex turning point.
        # --------------------------------------------------------------

        try:

            r_turn = (
                solver.find_turning_point_complex(
                    omega,
                    M_MODE,
                    r_minus_real,
                )
            )

        except Exception as exc:

            return {
                "status":
                    "TURNING_POINT_COMPLEX_FAILURE",
                "C": float(
                    C_value
                ),
                "f_re": float(
                    f_re
                ),
                "f_im": float(
                    f_im
                ),
                "r_sp": float(
                    r_sp
                ),
                "f_lr": float(
                    omega_lr
                    / (
                        2.0
                        * np.pi
                    )
                ),
                "r_minus_real":
                    float(
                        r_minus_real
                    ),
                "error": repr(
                    exc
                ),
            }

        if r_turn is None:

            return {
                "status":
                    "TURNING_POINT_COMPLEX_FAILURE",
                "C": float(
                    C_value
                ),
                "f_re": float(
                    f_re
                ),
                "f_im": float(
                    f_im
                ),
                "r_sp": float(
                    r_sp
                ),
                "f_lr": float(
                    omega_lr
                    / (
                        2.0
                        * np.pi
                    )
                ),
                "r_minus_real":
                    float(
                        r_minus_real
                    ),
            }

        # --------------------------------------------------------------
        # Turning-point residual.
        # --------------------------------------------------------------

        H_turn = solver.hamiltonian(
            omega,
            0j,
            M_MODE,
            r_turn,
        )

        # --------------------------------------------------------------
        # Radial action.
        # --------------------------------------------------------------

        try:

            action, action_info = (
                radial_action_adaptive(
                    solver,
                    omega,
                    M_MODE,
                    r_turn,
                    R_B,
                    N_ACTION,
                )
            )

        except Exception as exc:

            return {
                "status":
                    "RADIAL_ACTION_FAILURE",
                "C": float(
                    C_value
                ),
                "f_re": float(
                    f_re
                ),
                "f_im": float(
                    f_im
                ),
                "r_sp": float(
                    r_sp
                ),
                "f_lr": float(
                    omega_lr
                    / (
                        2.0
                        * np.pi
                    )
                ),
                "r_minus_real":
                    float(
                        r_minus_real
                    ),
                "r_turn_re":
                    float(
                        np.real(
                            r_turn
                        )
                    ),
                "r_turn_im":
                    float(
                        np.imag(
                            r_turn
                        )
                    ),
                "H_turn_abs":
                    float(
                        abs(
                            H_turn
                        )
                    ),
                "error": repr(
                    exc
                ),
            }

        # --------------------------------------------------------------
        # Reflection coefficient.
        # --------------------------------------------------------------

        try:

            R = (
                solver.reflection_coefficient(
                    omega,
                    M_MODE,
                    r_sp,
                    omega_lr,
                )
            )

        except Exception as exc:

            return {
                "status":
                    "REFLECTION_FAILURE",
                "C": float(
                    C_value
                ),
                "f_re": float(
                    f_re
                ),
                "f_im": float(
                    f_im
                ),
                "r_sp": float(
                    r_sp
                ),
                "f_lr": float(
                    omega_lr
                    / (
                        2.0
                        * np.pi
                    )
                ),
                "r_minus_real":
                    float(
                        r_minus_real
                    ),
                "r_turn_re":
                    float(
                        np.real(
                            r_turn
                        )
                    ),
                "r_turn_im":
                    float(
                        np.imag(
                            r_turn
                        )
                    ),
                "H_turn_abs":
                    float(
                        abs(
                            H_turn
                        )
                    ),
                "action_re":
                    float(
                        np.real(
                            action
                        )
                    ),
                "action_im":
                    float(
                        np.imag(
                            action
                        )
                    ),
                "error": repr(
                    exc
                ),
            }

        # --------------------------------------------------------------
        # Resonance residual.
        # --------------------------------------------------------------

        Res = (
            R
            * np.exp(
                2.0j * action
            )
            - 1.0
        )

        return {
            "status": "OK",

            "C": float(
                C_value
            ),

            "f_re": float(
                f_re
            ),

            "f_im": float(
                f_im
            ),

            "omega_re":
                float(
                    np.real(
                        omega
                    )
                ),

            "omega_im":
                float(
                    np.imag(
                        omega
                    )
                ),

            "r_sp":
                float(
                    r_sp
                ),

            "f_lr":
                float(
                    omega_lr
                    / (
                        2.0
                        * np.pi
                    )
                ),

            "r_minus_real":
                float(
                    r_minus_real
                ),

            "r_turn_re":
                float(
                    np.real(
                        r_turn
                    )
                ),

            "r_turn_im":
                float(
                    np.imag(
                        r_turn
                    )
                ),

            "H_turn_abs":
                float(
                    abs(
                        H_turn
                    )
                ),

            "action_re":
                float(
                    np.real(
                        action
                    )
                ),

            "action_im":
                float(
                    np.imag(
                        action
                    )
                ),

            "R_re":
                float(
                    np.real(
                        R
                    )
                ),

            "R_im":
                float(
                    np.imag(
                        R
                    )
                ),

            "R_abs":
                float(
                    abs(
                        R
                    )
                ),

            "Res_re":
                float(
                    np.real(
                        Res
                    )
                ),

            "Res_im":
                float(
                    np.imag(
                        Res
                    )
                ),

            "Res_abs":
                float(
                    abs(
                        Res
                    )
                ),

            "action_n_used":
                int(
                    action_info[
                        "n_used"
                    ]
                ),

            "max_abs_p":
                float(
                    action_info[
                        "max_abs_p"
                    ]
                ),

            "first_seed_label":
                action_info[
                    "first_seed_label"
                ],
        }

    finally:

        restore_runtime_C(
            solver,
            old_c,
        )


# ============================================================================
# Q FACTOR
# ============================================================================

def calculate_Q(
    f_re,
    f_im,
):

    if not np.isfinite(
        f_re
    ):

        return np.nan

    if not np.isfinite(
        f_im
    ):

        return np.nan

    if f_im == 0.0:

        return np.inf

    return abs(
        f_re
        / (
            2.0
            * f_im
        )
    )


# ============================================================================
# STARTING POINTS
# ============================================================================

def evaluate_starting_points(
    solver,
    C_value,
):

    starts = [
        (
            "historical_stage5I3",
            F_RE_START,
            F_IM_START,
        ),
        (
            "later_benchmark",
            F_RE_BENCH,
            F_IM_BENCH,
        ),
    ]

    results = []

    for label, f_re, f_im in starts:

        print()
        print(
            f"  Starting point: {label}"
        )

        print(
            f"    f = "
            f"{f_re:.12f}"
            f"{f_im:+.12f} i Hz"
        )

        try:

            result = evaluate_resonance(
                solver,
                C_value,
                f_re,
                f_im,
            )

        except Exception as exc:

            result = {
                "status": "EXCEPTION",
                "C": float(
                    C_value
                ),
                "f_re": float(
                    f_re
                ),
                "f_im": float(
                    f_im
                ),
                "error": repr(
                    exc
                ),
            }

        result[
            "start_label"
        ] = label

        results.append(
            result
        )

        if (
            result.get(
                "status"
            )
            == "OK"
        ):

            print(
                f"    |Res| = "
                f"{result['Res_abs']:.8e}"
            )

        else:

            print(
                "    STATUS = "
                f"{result.get('status')}"
            )

            if "error" in result:

                print(
                    f"    error = "
                    f"{result['error']}"
                )

    return results


# ============================================================================
# OBJECTIVE
# ============================================================================

def make_objective(
    solver,
    C_value,
):

    cache = {}

    def objective(x):

        f_re = float(
            x[0]
        )

        f_im = float(
            x[1]
        )

        if not (
            F_RE_START
            - F_RE_HALF_WIDTH
            <= f_re
            <= F_RE_START
            + F_RE_HALF_WIDTH
        ):

            return 100.0

        if not (
            F_IM_START
            - F_IM_HALF_WIDTH
            <= f_im
            <= F_IM_START
            + F_IM_HALF_WIDTH
        ):

            return 100.0

        key = (
            round(
                f_re,
                12,
            ),
            round(
                f_im,
                12,
            ),
        )

        if key in cache:

            return cache[key]

        try:

            result = (
                evaluate_resonance(
                    solver,
                    C_value,
                    f_re,
                    f_im,
                )
            )

        except Exception:

            cache[key] = 100.0

            return 100.0

        if (
            result.get(
                "status"
            )
            != "OK"
        ):

            cache[key] = 100.0

            return 100.0

        value = np.log10(
            max(
                result[
                    "Res_abs"
                ],
                1.0e-300,
            )
        )

        cache[key] = float(
            value
        )

        return float(
            value
        )

    objective.cache = cache

    return objective


# ============================================================================
# RESONANCE SEARCH
# ============================================================================

def search_resonance(
    solver,
    C_value,
):

    start_results = (
        evaluate_starting_points(
            solver,
            C_value,
        )
    )

    valid_starts = [
        x
        for x in start_results
        if x.get(
            "status"
        ) == "OK"
        and np.isfinite(
            x.get(
                "Res_abs",
                np.nan,
            )
        )
    ]

    if not valid_starts:

        return {
            "status":
                "NO_VALID_START",
            "C":
                float(
                    C_value
                ),
            "starts":
                start_results,
        }

    best_start = min(
        valid_starts,
        key=lambda x:
            x["Res_abs"],
    )

    x0 = np.array(
        [
            best_start[
                "f_re"
            ],
            best_start[
                "f_im"
            ],
        ],
        dtype=float,
    )

    print()
    print(
        "  Best starting point:"
    )

    print(
        f"    f = "
        f"{x0[0]:.12f}"
        f"{x0[1]:+.12f} i Hz"
    )

    print(
        f"    |Res| = "
        f"{best_start['Res_abs']:.8e}"
    )

    objective = make_objective(
        solver,
        C_value,
    )

    candidates = []

    # ------------------------------------------------------------------
    # Nelder-Mead
    # ------------------------------------------------------------------

    simplex = np.array(
        [
            x0,

            x0
            + np.array(
                [
                    F_RE_STEP,
                    0.0,
                ]
            ),

            x0
            + np.array(
                [
                    0.0,
                    F_IM_STEP,
                ]
            ),
        ]
    )

    print()
    print(
        "  Running Nelder-Mead..."
    )

    try:

        nm = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={
                "xatol":
                    1.0e-8,
                "fatol":
                    1.0e-7,
                "maxiter":
                    80,
                "initial_simplex":
                    simplex,
                "disp":
                    False,
            },
        )

        print(
            f"    success = "
            f"{nm.success}"
        )

        print(
            f"    iterations = "
            f"{nm.nit}"
        )

        print(
            f"    objective = "
            f"{nm.fun:.8e}"
        )

        try:

            result_nm = (
                evaluate_resonance(
                    solver,
                    C_value,
                    float(
                        nm.x[0]
                    ),
                    float(
                        nm.x[1]
                    ),
                )
            )

            if (
                result_nm.get(
                    "status"
                )
                == "OK"
            ):

                result_nm[
                    "optimizer"
                ] = "Nelder-Mead"

                result_nm[
                    "optimizer_fun"
                ] = float(
                    nm.fun
                )

                candidates.append(
                    result_nm
                )

        except Exception:
            pass

    except Exception as exc:

        print(
            "    Nelder-Mead exception:"
        )

        print(
            f"      {exc!r}"
        )

    # ------------------------------------------------------------------
    # Powell
    # ------------------------------------------------------------------

    print()
    print(
        "  Running Powell..."
    )

    try:

        pw = minimize(
            objective,
            x0,
            method="Powell",
            bounds=[
                (
                    F_RE_START
                    - F_RE_HALF_WIDTH,
                    F_RE_START
                    + F_RE_HALF_WIDTH,
                ),
                (
                    F_IM_START
                    - F_IM_HALF_WIDTH,
                    F_IM_START
                    + F_IM_HALF_WIDTH,
                ),
            ],
            options={
                "xtol":
                    1.0e-7,
                "ftol":
                    1.0e-7,
                "maxiter":
                    40,
                "disp":
                    False,
            },
        )

        print(
            f"    success = "
            f"{pw.success}"
        )

        print(
            f"    iterations = "
            f"{pw.nit}"
        )

        print(
            f"    objective = "
            f"{pw.fun:.8e}"
        )

        try:

            result_pw = (
                evaluate_resonance(
                    solver,
                    C_value,
                    float(
                        pw.x[0]
                    ),
                    float(
                        pw.x[1]
                    ),
                )
            )

            if (
                result_pw.get(
                    "status"
                )
                == "OK"
            ):

                result_pw[
                    "optimizer"
                ] = "Powell"

                result_pw[
                    "optimizer_fun"
                ] = float(
                    pw.fun
                )

                candidates.append(
                    result_pw
                )

        except Exception:
            pass

    except Exception as exc:

        print(
            "    Powell exception:"
        )

        print(
            f"      {exc!r}"
        )

    if not candidates:

        return {
            "status":
                "NO_VALID_OPTIMUM",
            "C":
                float(
                    C_value
                ),
            "starts":
                start_results,
        }

    best = min(
        candidates,
        key=lambda x:
            x["Res_abs"],
    )

    best[
        "starts"
    ] = start_results

    return best


# ============================================================================
# CASE PRINT
# ============================================================================

def print_case_result(
    name,
    C_value,
    result,
):

    print()
    print("=" * 78)
    print(
        f"{name}: C = "
        f"{C_value:.12e}"
    )
    print("=" * 78)

    status = result.get(
        "status"
    )

    print(
        f"STATUS = {status}"
    )

    if status == "NO_LIGHT_RING":

        print()
        print(
            "No interior light ring."
        )

        print(
            "No complex resonance search"
        )

        print(
            "is attempted for this case."
        )

        return

    if status != "OK":

        if "error" in result:

            print()
            print(
                "ERROR = "
                + str(
                    result[
                        "error"
                    ]
                )
            )

        return

    Q = calculate_Q(
        result[
            "f_re"
        ],
        result[
            "f_im"
        ],
    )

    print()

    print(
        f"  f_R       = "
        f"{result['f_re']:.12f} Hz"
    )

    print(
        f"  f_I       = "
        f"{result['f_im']:.12f} Hz"
    )

    print(
        f"  Q         = "
        f"{Q:.8e}"
    )

    print(
        f"  |Res|     = "
        f"{result['Res_abs']:.8e}"
    )

    print(
        f"  Re(Res)   = "
        f"{result['Res_re']:+.8e}"
    )

    print(
        f"  Im(Res)   = "
        f"{result['Res_im']:+.8e}"
    )

    print()

    print(
        f"  r_sp      = "
        f"{result['r_sp']*1e3:.12f} mm"
    )

    print(
        f"  f_lr      = "
        f"{result['f_lr']:.12f} Hz"
    )

    print(
        f"  r_minus   = "
        f"{result['r_minus_real']:.12f} m"
    )

    print(
        f"  r_turn    = "
        f"{result['r_turn_re']:.12e}"
        f"{result['r_turn_im']:+.12e} i m"
    )

    print(
        f"  |H_turn|  = "
        f"{result['H_turn_abs']:.6e}"
    )

    print(
        f"  action    = "
        f"{result['action_re']:+.8e}"
        f"{result['action_im']:+.8e} i"
    )

    print(
        f"  |R|       = "
        f"{result['R_abs']:.8e}"
    )

    print(
        f"  action points = "
        f"{result['action_n_used']}"
    )

    print(
        f"  max |p|   = "
        f"{result['max_abs_p']:.8e}"
    )

    print(
        f"  first seed = "
        f"{result['first_seed_label']}"
    )

    print(
        f"  optimizer  = "
        f"{result.get('optimizer', '--')}"
    )


# ============================================================================
# FLATTEN RESULT
# ============================================================================

def flatten_result(
    result,
):

    row = {}

    for key, value in result.items():

        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        ):

            row[key] = value

        elif value is None:

            row[key] = None

    if (
        result.get(
            "status"
        ) == "OK"
        and "f_re" in result
        and "f_im" in result
    ):

        row["Q"] = calculate_Q(
            result[
                "f_re"
            ],
            result[
                "f_im"
            ],
        )

    return row


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(
    results,
):

    rows = []

    for item in results:

        row = flatten_result(
            item[
                "result"
            ]
        )

        row[
            "case"
        ] = item[
            "name"
        ]

        row[
            "C"
        ] = item[
            "C"
        ]

        rows.append(
            row
        )

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row.keys()
        }
    )

    with open(
        CSV_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                row
            )

    serializable = []

    for item in results:

        result = copy.deepcopy(
            item[
                "result"
            ]
        )

        result[
            "case"
        ] = item[
            "name"
        ]

        result[
            "C"
        ] = item[
            "C"
        ]

        if "starts" in result:

            clean_starts = []

            for start in result[
                "starts"
            ]:

                clean_starts.append(
                    flatten_result(
                        start
                    )
                )

            result[
                "starts"
            ] = clean_starts

        serializable.append(
            result
        )

    with open(
        JSON_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            serializable,
            f,
            indent=2,
        )


# ============================================================================
# FINAL CAUSAL VERDICT
# ============================================================================

def causal_verdict(
    results,
):

    print()
    print("=" * 78)
    print(
        "RESONANCE-LEVEL C-CAUSALITY VERDICT"
    )
    print("=" * 78)

    valid = []

    for item in results:

        result = item[
            "result"
        ]

        if (
            result.get(
                "status"
            )
            == "OK"
        ):

            valid.append(
                (
                    item[
                        "name"
                    ],
                    item[
                        "C"
                    ],
                    result,
                )
            )

    print()

    if not valid:

        print(
            "NO VALID COMPLEX RESONANCE RESULT."
        )

        print()

        print(
            "CAUSAL TEST: INCONCLUSIVE"
        )

        print(
            "This is a numerical failure,"
        )

        print(
            "not evidence against resonance."
        )

        return

    print(
        "VALID RESONANCE RESULTS:"
    )

    for name, C_value, result in valid:

        Q = calculate_Q(
            result[
                "f_re"
            ],
            result[
                "f_im"
            ],
        )

        print(
            f"  {name:10s} "
            f"C={C_value:.6e} "
            f"f="
            f"{result['f_re']:.9f}"
            f"{result['f_im']:+.9f}i Hz "
            f"|Res|="
            f"{result['Res_abs']:.3e} "
            f"Q={Q:.3e}"
        )

    # ------------------------------------------------------------------
    # C dependence.
    # ------------------------------------------------------------------

    f_re_values = np.array(
        [
            result[
                "f_re"
            ]
            for _, _, result
            in valid
        ]
    )

    f_im_values = np.array(
        [
            result[
                "f_im"
            ]
            for _, _, result
            in valid
        ]
    )

    r_sp_values = np.array(
        [
            result[
                "r_sp"
            ]
            for _, _, result
            in valid
        ]
    )

    f_re_spread = float(
        np.ptp(
            f_re_values
        )
    )

    f_im_spread = float(
        np.ptp(
            f_im_values
        )
    )

    r_sp_spread = float(
        np.ptp(
            r_sp_values
        )
    )

    print()

    print(
        f"  f_R spread  = "
        f"{f_re_spread:.8e} Hz"
    )

    print(
        f"  f_I spread  = "
        f"{f_im_spread:.8e} Hz"
    )

    print(
        f"  r_sp spread = "
        f"{r_sp_spread*1e3:.8e} mm"
    )

    # ------------------------------------------------------------------
    # C=0 structural control.
    # ------------------------------------------------------------------

    c_zero_result = None

    for item in results:

        if item[
            "name"
        ] == "C_ZERO":

            c_zero_result = item[
                "result"
            ]

    c_zero_clean = (
        c_zero_result is not None
        and c_zero_result.get(
            "status"
        ) == "NO_LIGHT_RING"
    )

    # ------------------------------------------------------------------
    # Nonzero resonance response.
    # ------------------------------------------------------------------

    resonance_response = (
        f_re_spread > 1.0e-8
        or f_im_spread > 1.0e-10
    )

    print()

    if (
        c_zero_clean
        and resonance_response
    ):

        print(
            "CAUSAL TEST: "
            "NONZERO RESONANCE RESPONSE TO C"
        )

        print()

        print(
            "The complex resonance characteristics"
        )

        print(
            "change across controlled C values."
        )

        print(
            "C=0 removes the interior light-ring structure."
        )

        print()

        print(
            "This establishes parameter dependence"
        )

        print(
            "of the resonance-level mathematical model."
        )

        print()

        print(
            "It does NOT establish a new fundamental"
        )

        print(
            "interaction or an experimentally realized"
        )

        print(
            "physical system."
        )

    elif resonance_response:

        print(
            "CAUSAL TEST: "
            "RESONANCE RESPONSE DETECTED"
        )

        print()

        print(
            "However, the C=0 structural control was"
        )

        print(
            "not clean enough for the strongest verdict."
        )

    else:

        print(
            "CAUSAL TEST: "
            "NO RESOLVED RESONANCE RESPONSE"
        )

        print()

        print(
            "No significant f_R/f_I movement was resolved"
        )

        print(
            "within the tested frequency region."
        )

        print()

        print(
            "This is not automatically evidence that"
        )

        print(
            "C has no effect."
        )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 78)
    print(
        "STAGE 6.5B-1 — "
        "RESONANCE-LEVEL C-CAUSALITY AUDIT"
    )
    print("=" * 78)

    print()
    print(
        "Authoritative solver:"
    )

    print(
        f"  {RES_FILE}"
    )

    print()
    print(
        "Radial patch:"
    )

    print(
        f"  {PATCH_FILE}"
    )

    print()
    print(
        "Mode:"
    )

    print(
        f"  m = {M_MODE}"
    )

    print()
    print(
        "Baseline constants:"
    )

    print(
        f"  C0     = "
        f"{C0:.12e} m^2/s"
    )

    print(
        f"  Omega  = "
        f"{OMEGA0:.12e} rad/s"
    )

    print(
        f"  gamma  = "
        f"{GAMMA0:.12e}"
    )

    print(
        f"  h0     = "
        f"{H00:.12e} m"
    )

    print(
        f"  g      = "
        f"{G0:.12e} m/s^2"
    )

    print(
        f"  r_B    = "
        f"{R_B:.12e} m"
    )

    print()
    print(
        "Controlled cases:"
    )

    for name, C_value in CASES:

        print(
            f"  {name:10s}: "
            f"C = {C_value:.12e}"
        )

    print()
    print(
        "Rules:"
    )

    print(
        "  * authoritative solver loaded from disk"
    )

    print(
        "  * compatibility shim is IN MEMORY ONLY"
    )

    print(
        "  * no source modification"
    )

    print(
        "  * only C is runtime-injected"
    )

    print(
        "  * complex C response is independently audited"
    )

    print(
        "  * exact patch-style turning-point seed"
    )

    print(
        "  * adaptive complex branch continuation"
    )

    print(
        "  * no Kerr fit"
    )

    print(
        "  * no GW calibration"
    )

    print(
        "  * no frequency rescaling"
    )

    print(
        "  * no low-k approximation"
    )

    print(
        "  * C_ZERO without light ring is not force-fit"
    )

    # ==================================================================
    # LOAD AUTHORITATIVE SOLVER
    # ==================================================================

    print()
    print("-" * 78)
    print(
        "1. LOAD AUTHORITATIVE SOLVER"
    )
    print("-" * 78)

    solver = load_module(
        RES_FILE,
        "_stage5I3_authoritative_6_5B1",
    )

    print(
        "  Loaded successfully."
    )

    print()
    print(
        "  hamiltonian:",
        inspect.signature(
            solver.hamiltonian
        ),
    )

    print()
    print(
        "  solve_p_complex:",
        inspect.signature(
            solver.solve_p_complex
        ),
    )

    print()
    print(
        "  find_light_ring:",
        inspect.signature(
            solver.find_light_ring
        ),
    )

    # ==================================================================
    # LOAD PATCH WITH COMPATIBILITY SHIM
    # ==================================================================

    print()
    print("-" * 78)
    print(
        "2. LOAD RADIAL-ACTION PATCH"
    )
    print("-" * 78)

    try:

        patch = load_patch(
            solver
        )

    except Exception as exc:

        print()
        print(
            "PATCH LOAD: FAIL"
        )

        print(
            f"  {exc!r}"
        )

        print()
        print(
            "No resonance calculation was attempted."
        )

        print(
            "This is an API/import failure, not a resonance result."
        )

        return

    print()
    print(
        "  Patch loaded successfully."
    )

    print()
    print(
        "  radial_action_complex_v2:",
        inspect.signature(
            patch.radial_action_complex_v2
        ),
    )

    # ==================================================================
    # C INJECTION AUDIT
    # ==================================================================

    print()
    print("-" * 78)
    print(
        "3. COMPLEX C-INJECTION AUDIT"
    )
    print("-" * 78)

    c_names = discover_c_names(
        solver
    )

    print()
    print(
        "C-related globals:"
    )

    for name in c_names:

        print(
            f"  {name:12s} = "
            f"{getattr(solver, name):.12e}"
        )

    if not c_names:

        print()
        print(
            "C-INJECTION AUDIT: FAIL"
        )

        print(
            "No C-related module global was found."
        )

        return

    omega_test = (
        2.0
        * np.pi
        * (
            F_RE_START
            + 1j * F_IM_START
        )
    )

    r_test = 0.025
    p_test = 0j

    H_half = complex_H_at_C(
        solver,
        c_names,
        0.5 * C0,
        omega_test,
        p_test,
        M_MODE,
        r_test,
    )

    H_base = complex_H_at_C(
        solver,
        c_names,
        C0,
        omega_test,
        p_test,
        M_MODE,
        r_test,
    )

    H_double = complex_H_at_C(
        solver,
        c_names,
        2.0 * C0,
        omega_test,
        p_test,
        M_MODE,
        r_test,
    )

    print()
    print(
        "Complex Hamiltonian fingerprints:"
    )

    print(
        f"  C/2 = "
        f"{H_half.real:+.12e}"
        f"{H_half.imag:+.12e} i"
    )

    print(
        f"  C   = "
        f"{H_base.real:+.12e}"
        f"{H_base.imag:+.12e} i"
    )

    print(
        f"  2C  = "
        f"{H_double.real:+.12e}"
        f"{H_double.imag:+.12e} i"
    )

    d_half = abs(
        H_half
        - H_base
    )

    d_double = abs(
        H_double
        - H_base
    )

    print()

    print(
        f"  |H(C/2)-H(C)| = "
        f"{d_half:.8e}"
    )

    print(
        f"  |H(2C)-H(C)| = "
        f"{d_double:.8e}"
    )

    injection_ok = (
        np.isfinite(
            d_half
        )
        and np.isfinite(
            d_double
        )
        and (
            d_half > 1.0e-14
            or d_double > 1.0e-14
        )
    )

    print()

    print(
        "C-INJECTION AUDIT: "
        + (
            "PASS"
            if injection_ok
            else "FAIL"
        )
    )

    if not injection_ok:

        print()
        print(
            "ABORT."
        )

        print(
            "The complex Hamiltonian does not show a resolved"
        )

        print(
            "response to runtime C injection."
        )

        return

    # ==================================================================
    # CONTROLLED RESONANCE CASES
    # ==================================================================

    print()
    print("-" * 78)
    print(
        "4. CONTROLLED RESONANCE TEST"
    )
    print("-" * 78)

    results = []

    for name, C_value in CASES:

        print()
        print("#" * 78)

        print(
            f"RUNNING CASE: {name}"
        )

        print(
            f"C = {C_value:.12e} m^2/s"
        )

        print("#" * 78)

        try:

            # ----------------------------------------------------------
            # First explicitly check the light ring.
            # ----------------------------------------------------------

            r_sp, omega_lr = (
                solver.find_light_ring(
                    M_MODE,
                    C=C_value,
                    Omega=OMEGA0,
                    gamma=GAMMA0,
                    h0=H00,
                    g=G0,
                    r_min=1.0e-3,
                    r_max=R_B,
                )
            )

            if r_sp is None:

                result = {
                    "status":
                        "NO_LIGHT_RING",
                    "C":
                        float(
                            C_value
                        ),
                }

            else:

                print()
                print(
                    f"  light ring = "
                    f"{r_sp*1e3:.12f} mm"
                )

                print(
                    f"  f_lr = "
                    f"{omega_lr/(2*np.pi):.12f} Hz"
                )

                # ------------------------------------------------------
                # Now perform local resonance search.
                # ------------------------------------------------------

                result = search_resonance(
                    solver,
                    C_value,
                )

        except Exception as exc:

            result = {
                "status":
                    "EXCEPTION",
                "C":
                    float(
                        C_value
                    ),
                "error":
                    repr(
                        exc
                    ),
            }

        results.append(
            {
                "name":
                    name,
                "C":
                    float(
                        C_value
                    ),
                "result":
                    result,
            }
        )

        print_case_result(
            name,
            C_value,
            result,
        )

    # ==================================================================
    # SAVE
    # ==================================================================

    print()
    print("-" * 78)
    print(
        "5. SAVE RESULTS"
    )
    print("-" * 78)

    save_results(
        results
    )

    print()
    print(
        f"CSV  -> {CSV_FILE}"
    )

    print(
        f"JSON -> {JSON_FILE}"
    )

    # ==================================================================
    # FINAL VERDICT
    # ==================================================================

    causal_verdict(
        results
    )

    print()
    print("=" * 78)
    print(
        "STAGE 6.5B-1 COMPLETE"
    )
    print("=" * 78)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "A small |Res| is a resonance candidate."
    )

    print(
        "The causal quantity is its reproducible response"
    )

    print(
        "to controlled variation of C."
    )

    print(
        "No physical interpretation is claimed by this stage."
    )

    print()
    print(
        "Authoritative solver was NOT modified."
    )


if __name__ == "__main__":

    main()