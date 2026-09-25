#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
STAGE 6.5B — RESONANCE-LEVEL CAUSAL / INDEPENDENT PHYSICAL CONSEQUENCE AUDIT
===============================================================================

Purpose
-------
Test whether the Root-B resonance itself responds causally to the circulation
parameter C, rather than testing only the light-ring location.

SCIENTIFIC RULES
----------------
1. The authoritative resonance machinery remains:
       stage5I_3_resonance.py

2. No Kerr fitting.

3. No GW/event calibration.

4. No frequency rescaling.

5. No low-k approximation.

6. No new dispersion model is introduced.

7. Only C is varied:
       C = 0
       C = C0/2
       C = C0
       C = 2*C0

8. Omega, gamma, h0, g, r_B and m are otherwise kept exactly as defined
   by the authoritative solver.

9. The resonance condition is the actual model condition:

       Res(omega) = R(omega) * exp(2 i S(omega)) - 1

10. The first complex-radial step uses the validated Stage 5I-3c
    turning-point seed rather than starting the continuation at exactly
    p = 0.

INTERPRETATION
--------------
If the complex resonance root changes systematically when only C is changed,
that is evidence that the resonance is dynamically dependent on the
circulation term already present in the Root-B equations.

It is NOT, by itself, evidence for a new fundamental interaction.

The C -> 0 case is treated carefully:
if the interior light ring disappears, no artificial resonance root is forced.
That absence is itself recorded as a control result.
===============================================================================
"""

import ast
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import root


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent

RES_FILE = HERE / "stage5I_3_resonance.py"

OUT_DIR = (
    HERE
    / "stage6_5B_resonance_causal_audit"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_FILE = (
    OUT_DIR
    / "stage6_5B_resonance_results.csv"
)

JSON_FILE = (
    OUT_DIR
    / "stage6_5B_resonance_results.json"
)


# =============================================================================
# BASELINE C
# =============================================================================
#
# This is the validated Root-B baseline value from Stage 6.5.
#
# Only this parameter is deliberately varied in the causal test.
# =============================================================================

BASE_C = 7.690000000000e-04


# =============================================================================
# MODE
# =============================================================================

M = -12


# =============================================================================
# NUMERICAL INITIAL GUESS
# =============================================================================
#
# IMPORTANT:
# This is only a numerical starting point for the complex root solver.
# It is NOT a physical calibration or frequency mapping.
# =============================================================================

BASE_F_RE = 8.3066307032346
BASE_F_IM = -0.0005696389388967092

INITIAL_F = complex(
    BASE_F_RE,
    BASE_F_IM,
)


# =============================================================================
# ROOT SOLVER
# =============================================================================

ROOT_TOL = 1e-10


# =============================================================================
# FALLBACK COMPLEX RADIAL RESOLUTION
# =============================================================================

DEFAULT_N_FALLBACK = 10000


# =============================================================================
# CONTROLLED CONSTANT REPLACER
# =============================================================================

class ConstantReplacer(ast.NodeTransformer):
    """
    Replace selected simple module-level assignments.

    For this experiment we use it ONLY for C-related names.

    No function bodies are rewritten.
    """

    def __init__(self, replacements):
        super().__init__()

        self.replacements = replacements

    def visit_Assign(self, node):
        node = self.generic_visit(node)

        for target in node.targets:

            if isinstance(target, ast.Name):

                name = target.id

                if name in self.replacements:

                    node.value = ast.Constant(
                        value=float(
                            self.replacements[name]
                        )
                    )

        return node

    def visit_AnnAssign(self, node):
        node = self.generic_visit(node)

        if isinstance(
            node.target,
            ast.Name,
        ):

            name = node.target.id

            if name in self.replacements:

                node.value = ast.Constant(
                    value=float(
                        self.replacements[name]
                    )
                )

        return node


# =============================================================================
# LOAD AUTHORITATIVE SOLVER
# =============================================================================

def load_controlled_solver(C_value):
    """
    Load stage5I_3_resonance.py into an isolated module namespace.

    ONLY C is changed.

    The original stage5I_3_resonance.py on disk is never modified.

    Critical detail:
    the authoritative solver uses __file__, therefore __file__ must be
    explicitly supplied to the exec() namespace.
    """

    if not RES_FILE.exists():

        raise FileNotFoundError(
            "Authoritative resonance solver not found:\n"
            f"{RES_FILE}"
        )

    # -------------------------------------------------------------------------
    # Read authoritative source.
    # -------------------------------------------------------------------------

    source = RES_FILE.read_text(
        encoding="utf-8"
    )

    # -------------------------------------------------------------------------
    # Parse source.
    # -------------------------------------------------------------------------

    tree = ast.parse(
        source,
        filename=str(RES_FILE),
    )

    # -------------------------------------------------------------------------
    # ONLY C is changed.
    #
    # Do NOT override:
    #   Omega
    #   gamma
    #   h0
    #   g
    #   r_B
    #
    # Those remain exactly as defined in stage5I_3_resonance.py.
    # -------------------------------------------------------------------------

    replacements = {
        "C_METHODS": C_value,
        "C_TABLE": C_value,
        "C": C_value,
    }

    tree = ConstantReplacer(
        replacements
    ).visit(tree)

    ast.fix_missing_locations(
        tree
    )

    # -------------------------------------------------------------------------
    # Unique isolated module name.
    # -------------------------------------------------------------------------

    module_name = (
        "_stage5I_3_resonance_C_"
        f"{C_value:.12e}"
    )

    spec = (
        importlib.util.spec_from_loader(
            module_name,
            loader=None,
        )
    )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    # -------------------------------------------------------------------------
    # CRITICAL FIX:
    #
    # stage5I_3_resonance.py contains:
    #
    #     HERE = Path(__file__).resolve().parent
    #
    # exec() does not automatically provide __file__.
    # -------------------------------------------------------------------------

    module.__file__ = str(
        RES_FILE
    )

    module.__name__ = (
        module_name
    )

    # -------------------------------------------------------------------------
    # Make local modules importable.
    # -------------------------------------------------------------------------

    if str(HERE) not in sys.path:

        sys.path.insert(
            0,
            str(HERE)
        )

    # -------------------------------------------------------------------------
    # Execute authoritative solver.
    # -------------------------------------------------------------------------

    exec(
        compile(
            tree,
            str(RES_FILE),
            "exec",
        ),
        module.__dict__,
    )

    return module


# =============================================================================
# H_r AT TURNING POINT
# =============================================================================

def compute_H_r_at_turning_point(
    RB,
    omega,
    m,
    r_minus,
    eps=None,
):
    """
    Numerical dH/dr at p=0.

    Used only for the local turning-point seed.
    """

    if eps is None:

        if abs(r_minus) > 0:

            eps = (
                abs(r_minus)
                * 1e-6
            )

        else:

            eps = 1e-9

    H_plus = RB.hamiltonian(
        omega,
        0.0j,
        m,
        r_minus + eps,
    )

    H_minus = RB.hamiltonian(
        omega,
        0.0j,
        m,
        r_minus - eps,
    )

    return (
        H_plus - H_minus
    ) / (
        2.0 * eps
    )


# =============================================================================
# CORRECTED COMPLEX RADIAL ACTION
# =============================================================================

def radial_action_complex_v2(
    RB,
    omega,
    m,
    r_minus,
    r_B,
    n=None,
):
    """
    Corrected Stage 5I-3c complex radial action.

    The first continuation point is seeded using

        p ~= sqrt(-2 H_r dr / H_pp)

    rather than starting continuation at p=0.
    """

    if n is None:

        n = int(
            getattr(
                RB,
                "DEFAULT_COMPLEX_RADIAL_N",
                DEFAULT_N_FALLBACK,
            )
        )

    n = int(n)

    if n < 10:

        raise ValueError(
            f"Complex radial resolution too small: n={n}"
        )

    # -------------------------------------------------------------------------
    # Radial path.
    # -------------------------------------------------------------------------

    t = np.linspace(
        0.0,
        1.0,
        n,
    )

    r_path = (
        r_minus
        + t * (
            r_B
            - r_minus
        )
    )

    p_values = np.zeros(
        n,
        dtype=complex,
    )

    p_previous = 0.0j

    # -------------------------------------------------------------------------
    # H_pp finite difference at turning point.
    # -------------------------------------------------------------------------

    eps_p_probe = 1.0

    H0 = RB.hamiltonian(
        omega,
        0.0j,
        m,
        r_minus,
    )

    Hp = RB.hamiltonian(
        omega,
        eps_p_probe + 0j,
        m,
        r_minus,
    )

    Hm = RB.hamiltonian(
        omega,
        -eps_p_probe + 0j,
        m,
        r_minus,
    )

    H_pp_estimate = (
        Hp
        - 2.0 * H0
        + Hm
    ) / (
        eps_p_probe ** 2
    )

    # -------------------------------------------------------------------------
    # H_r finite difference.
    # -------------------------------------------------------------------------

    H_r = compute_H_r_at_turning_point(
        RB,
        omega,
        m,
        r_minus,
    )

    if not np.isfinite(
        H_pp_estimate.real
    ) or not np.isfinite(
        H_pp_estimate.imag
    ):

        raise FloatingPointError(
            "Non-finite H_pp estimate."
        )

    if not np.isfinite(
        H_r.real
    ) or not np.isfinite(
        H_r.imag
    ):

        raise FloatingPointError(
            "Non-finite H_r estimate."
        )

    # -------------------------------------------------------------------------
    # Continuation.
    # -------------------------------------------------------------------------

    for i, rr in enumerate(
        r_path
    ):

        # Exact turning point.
        if i == 0:

            p_values[i] = 0.0j

            continue

        # ---------------------------------------------------------------------
        # First step: analytical local seed.
        # ---------------------------------------------------------------------

        if i == 1:

            delta_r = (
                rr
                - r_minus
            )

            radicand = (
                -2.0
                * H_r
                * delta_r
                / H_pp_estimate
            )

            p_seed = np.sqrt(
                complex(
                    radicand
                )
            )

        else:

            # Continue from previous branch.
            p_seed = p_previous

        # ---------------------------------------------------------------------
        # Solve H=0 for p.
        # ---------------------------------------------------------------------

        p_new = RB.solve_p_complex(
            omega,
            m,
            rr,
            p_initial=p_seed,
        )

        # ---------------------------------------------------------------------
        # First-step opposite-branch fallback.
        # ---------------------------------------------------------------------

        if (
            p_new is None
            and i == 1
        ):

            p_new = (
                RB.solve_p_complex(
                    omega,
                    m,
                    rr,
                    p_initial=-p_seed,
                )
            )

        if p_new is None:

            raise FloatingPointError(
                "Complex p continuation failed "
                f"at step {i}, "
                f"r={rr:.16e}"
            )

        # ---------------------------------------------------------------------
        # Branch continuity.
        # ---------------------------------------------------------------------

        if (
            abs(
                p_new
                - p_previous
            )
            >
            abs(
                -p_new
                - p_previous
            )
        ):

            p_new = -p_new

        p_values[i] = p_new

        p_previous = p_new

    # -------------------------------------------------------------------------
    # Integrate p dr.
    # -------------------------------------------------------------------------

    if hasattr(
        np,
        "trapezoid",
    ):

        return np.trapezoid(
            p_values,
            r_path,
        )

    return np.trapz(
        p_values,
        r_path,
    )


# =============================================================================
# DIRECT RESONANCE FACTOR
# =============================================================================

def resonance_factor_direct(
    RB,
    omega,
    m,
    r_sp,
    omega_lr,
):
    """
    Evaluate:

        Res(omega)
          =
        R(omega) * exp(2 i S(omega)) - 1
    """

    # -------------------------------------------------------------------------
    # Real last-scattering point.
    # -------------------------------------------------------------------------

    r_minus_real = (
        RB.find_last_scattering_point(
            omega,
            m,
            r_sp,
        )
    )

    if r_minus_real is None:

        raise FloatingPointError(
            "Last-scattering-point search failed."
        )

    # -------------------------------------------------------------------------
    # Complex turning point.
    # -------------------------------------------------------------------------

    r_minus_complex = (
        RB.find_turning_point_complex(
            omega,
            m,
            r_minus_real,
        )
    )

    if r_minus_complex is None:

        raise FloatingPointError(
            "Complex turning-point search failed."
        )

    # -------------------------------------------------------------------------
    # Complex radial action.
    # -------------------------------------------------------------------------

    action = radial_action_complex_v2(
        RB,
        omega,
        m,
        r_minus_complex,
        getattr(
            RB,
            "R_B",
            r_sp,
        ),
    )

    # -------------------------------------------------------------------------
    # Reflection coefficient.
    # -------------------------------------------------------------------------

    R = RB.reflection_coefficient(
        omega,
        m,
        r_sp,
        omega_lr,
    )

    if R is None:

        raise FloatingPointError(
            "Reflection coefficient returned None."
        )

    # -------------------------------------------------------------------------
    # Actual resonance factor.
    # -------------------------------------------------------------------------

    Res = (
        R
        * np.exp(
            2.0j * action
        )
        - 1.0
    )

    return (
        complex(Res),
        complex(action),
        complex(R),
        complex(r_minus_complex),
    )


# =============================================================================
# ROOT RESIDUAL
# =============================================================================

def resonance_residual(
    x,
    RB,
    m,
    r_sp,
    omega_lr,
):
    """
    Complex resonance root represented as two real equations.

        x[0] = Re(f) in Hz
        x[1] = Im(f) in Hz

        omega = 2*pi*f

    Returns:

        [Re(Res), Im(Res)]
    """

    f = complex(
        float(x[0]),
        float(x[1]),
    )

    omega = (
        2.0
        * np.pi
        * f
    )

    try:

        Res, _, _, _ = (
            resonance_factor_direct(
                RB,
                omega,
                m,
                r_sp,
                omega_lr,
            )
        )

        if not (
            np.isfinite(
                Res.real
            )
            and np.isfinite(
                Res.imag
            )
        ):

            return np.array(
                [
                    1e6,
                    1e6,
                ],
                dtype=float,
            )

        return np.array(
            [
                Res.real,
                Res.imag,
            ],
            dtype=float,
        )

    except Exception:

        # A failed evaluation is NOT a root.
        return np.array(
            [
                1e6,
                1e6,
            ],
            dtype=float,
        )


# =============================================================================
# SOLVE ONE C CASE
# =============================================================================

def solve_case(
    C_value,
    label,
    initial_f,
):
    """
    Run one controlled C case.
    """

    print()
    print("-" * 78)
    print(
        f"{label}: "
        f"C={C_value:.12e} m^2/s"
    )
    print("-" * 78)

    # -------------------------------------------------------------------------
    # Load authoritative solver with ONLY C changed.
    # -------------------------------------------------------------------------

    RB = load_controlled_solver(
        C_value
    )

    # -------------------------------------------------------------------------
    # Check the resulting solver constants.
    #
    # This is deliberately printed so that accidental changes to other
    # parameters are visible immediately.
    # -------------------------------------------------------------------------

    print()
    print("  solver constants:")

    for name in (
        "C_METHODS",
        "C_TABLE",
        "OMEGA",
        "GAMMA",
        "H0",
        "G_GRAV",
        "R_B",
    ):

        if hasattr(
            RB,
            name,
        ):

            print(
                f"    {name:10s} = "
                f"{getattr(RB, name)}"
            )

    # -------------------------------------------------------------------------
    # Light ring.
    # -------------------------------------------------------------------------

    r_sp, omega_lr = (
        RB.find_light_ring(
            M
        )
    )

    # -------------------------------------------------------------------------
    # No light ring.
    #
    # This is NOT automatically a failure.
    # We do not manufacture a resonance search without trapping structure.
    # -------------------------------------------------------------------------

    if (
        r_sp is None
        or omega_lr is None
    ):

        print()
        print(
            "  LIGHT RING: ABSENT"
        )

        print(
            "  RESONANCE: NOT SEARCHED"
        )

        print(
            "  Reason: no interior light ring / trapping structure."
        )

        return {
            "label": label,
            "C": C_value,
            "light_ring": False,
            "r_sp_m": None,
            "f_lr_Hz": None,
            "root_found": False,
            "f_re_Hz": None,
            "f_im_Hz": None,
            "tau_s": None,
            "Q": None,
            "res_abs": None,
            "res_re": None,
            "res_im": None,
            "turning_point_re_m": None,
            "turning_point_im_m": None,
            "action_re": None,
            "action_im": None,
            "reflection_abs": None,
            "reflection_re": None,
            "reflection_im": None,
            "solver_success": None,
            "solver_nfev": None,
            "status": "NO_LIGHT_RING",
        }

    # -------------------------------------------------------------------------
    # Light-ring frequency.
    # -------------------------------------------------------------------------

    f_lr = (
        omega_lr
        / (
            2.0
            * np.pi
        )
    )

    print()
    print(
        f"  light ring: "
        f"r_sp={r_sp*1e3:.12f} mm"
    )

    print(
        f"  light-ring frequency: "
        f"{f_lr:.12f} Hz"
    )

    # -------------------------------------------------------------------------
    # Root initial guess.
    #
    # Same fixed numerical seed for all cases.
    # No C-dependent rescaling.
    # -------------------------------------------------------------------------

    x0 = np.array(
        [
            initial_f.real,
            initial_f.imag,
        ],
        dtype=float,
    )

    print()
    print(
        "  root initial guess:"
    )

    print(
        f"    f_R = "
        f"{x0[0]:.12f} Hz"
    )

    print(
        f"    f_I = "
        f"{x0[1]:+.12e} Hz"
    )

    # -------------------------------------------------------------------------
    # Root solve.
    # -------------------------------------------------------------------------

    result = root(
        resonance_residual,
        x0,
        args=(
            RB,
            M,
            r_sp,
            omega_lr,
        ),
        method="hybr",
        tol=ROOT_TOL,
    )

    # -------------------------------------------------------------------------
    # IMPORTANT:
    #
    # result.success alone is NOT accepted.
    # We must directly reevaluate Res at the returned root.
    # -------------------------------------------------------------------------

    f_root = complex(
        result.x[0],
        result.x[1],
    )

    omega_root = (
        2.0
        * np.pi
        * f_root
    )

    print()
    print(
        f"  scipy root success = "
        f"{result.success}"
    )

    print(
        f"  scipy message = "
        f"{result.message}"
    )

    print(
        f"  scipy nfev = "
        f"{result.nfev}"
    )

    if not result.success:

        print(
            "  ROOT: FAILED"
        )

        return {
            "label": label,
            "C": C_value,
            "light_ring": True,
            "r_sp_m": float(r_sp),
            "f_lr_Hz": float(f_lr),
            "root_found": False,
            "f_re_Hz": float(f_root.real),
            "f_im_Hz": float(f_root.imag),
            "tau_s": None,
            "Q": None,
            "res_abs": None,
            "res_re": None,
            "res_im": None,
            "turning_point_re_m": None,
            "turning_point_im_m": None,
            "action_re": None,
            "action_im": None,
            "reflection_abs": None,
            "reflection_re": None,
            "reflection_im": None,
            "solver_success": False,
            "solver_nfev": int(
                result.nfev
            ),
            "status": "ROOT_FAILED",
        }

    # -------------------------------------------------------------------------
    # Direct root reevaluation.
    # -------------------------------------------------------------------------

    try:

        (
            Res,
            action,
            R,
            r_turn,
        ) = resonance_factor_direct(
            RB,
            omega_root,
            M,
            r_sp,
            omega_lr,
        )

    except Exception as exc:

        print()
        print(
            "  ROOT RE-EVALUATION: FAILED"
        )

        print(
            f"  {type(exc).__name__}: "
            f"{exc}"
        )

        return {
            "label": label,
            "C": C_value,
            "light_ring": True,
            "r_sp_m": float(r_sp),
            "f_lr_Hz": float(f_lr),
            "root_found": False,
            "f_re_Hz": float(f_root.real),
            "f_im_Hz": float(f_root.imag),
            "tau_s": None,
            "Q": None,
            "res_abs": None,
            "res_re": None,
            "res_im": None,
            "turning_point_re_m": None,
            "turning_point_im_m": None,
            "action_re": None,
            "action_im": None,
            "reflection_abs": None,
            "reflection_re": None,
            "reflection_im": None,
            "solver_success": bool(
                result.success
            ),
            "solver_nfev": int(
                result.nfev
            ),
            "status": "ROOT_REEVAL_FAILED",
        }

    # -------------------------------------------------------------------------
    # Root quality.
    # -------------------------------------------------------------------------

    res_abs = abs(
        Res
    )

    # A very small residual is required before accepting the root.
    ROOT_RES_TOL = 1e-7

    if not np.isfinite(
        res_abs
    ):

        root_found = False

    elif res_abs > ROOT_RES_TOL:

        root_found = False

    else:

        root_found = True

    # -------------------------------------------------------------------------
    # Damping time and Q.
    #
    # omega = 2*pi*f
    #
    # For exp(-i omega t):
    #
    #   f_I < 0  => damping
    #
    # omega_I = 2*pi*f_I
    #
    # tau = 1 / |omega_I|
    #
    # Q = |omega_R| / (2 |omega_I|)
    # -------------------------------------------------------------------------

    omega_R = (
        omega_root.real
    )

    omega_I = (
        omega_root.imag
    )

    if (
        omega_I != 0.0
        and np.isfinite(
            omega_I
        )
    ):

        tau = (
            1.0
            / abs(
                omega_I
            )
        )

        Q = (
            abs(
                omega_R
            )
            /
            (
                2.0
                * abs(
                    omega_I
                )
            )
        )

    else:

        tau = np.inf
        Q = np.inf

    # -------------------------------------------------------------------------
    # Print diagnostics.
    # -------------------------------------------------------------------------

    print()

    if root_found:

        print(
            "  ROOT: ACCEPTED"
        )

    else:

        print(
            "  ROOT: REJECTED"
        )

    print(
        f"  f_R = "
        f"{f_root.real:.12f} Hz"
    )

    print(
        f"  f_I = "
        f"{f_root.imag:+.12e} Hz"
    )

    print(
        f"  tau = "
        f"{tau:.12e} s"
    )

    print(
        f"  Q   = "
        f"{Q:.12e}"
    )

    print(
        f"  |Res| = "
        f"{res_abs:.12e}"
    )

    print(
        f"  Re(Res) = "
        f"{Res.real:+.12e}"
    )

    print(
        f"  Im(Res) = "
        f"{Res.imag:+.12e}"
    )

    print(
        f"  action = "
        f"{action.real:+.12e}"
        f" {action.imag:+.12e}i"
    )

    print(
        f"  |R| = "
        f"{abs(R):.12e}"
    )

    print(
        f"  turning point = "
        f"{r_turn.real*1e3:.12f}"
        f" {r_turn.imag*1e3:+.12f}i mm"
    )

    # -------------------------------------------------------------------------
    # Status.
    # -------------------------------------------------------------------------

    if root_found:

        status = "PASS"

    else:

        status = "ROOT_RESIDUAL_TOO_LARGE"

    return {
        "label": label,
        "C": C_value,
        "light_ring": True,
        "r_sp_m": float(r_sp),
        "f_lr_Hz": float(f_lr),
        "root_found": bool(
            root_found
        ),
        "f_re_Hz": float(
            f_root.real
        ),
        "f_im_Hz": float(
            f_root.imag
        ),
        "tau_s": float(
            tau
        ),
        "Q": float(
            Q
        ),
        "res_abs": float(
            res_abs
        ),
        "res_re": float(
            Res.real
        ),
        "res_im": float(
            Res.imag
        ),
        "turning_point_re_m": float(
            r_turn.real
        ),
        "turning_point_im_m": float(
            r_turn.imag
        ),
        "action_re": float(
            action.real
        ),
        "action_im": float(
            action.imag
        ),
        "reflection_abs": float(
            abs(R)
        ),
        "reflection_re": float(
            R.real
        ),
        "reflection_im": float(
            R.imag
        ),
        "solver_success": bool(
            result.success
        ),
        "solver_nfev": int(
            result.nfev
        ),
        "status": status,
    }


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(results):

    if not results:

        return

    # -------------------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------------------

    fieldnames = sorted(
        {
            key
            for row in results
            for key in row.keys()
        }
    )

    with CSV_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in results:

            writer.writerow(
                row
            )

    # -------------------------------------------------------------------------
    # JSON
    # -------------------------------------------------------------------------

    with JSON_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            allow_nan=False,
        )


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary(results):

    print()
    print("=" * 78)
    print(
        "STAGE 6.5B — RESONANCE-LEVEL CAUSAL AUDIT"
    )
    print("=" * 78)

    print()

    header = (
        f"{'case':>14} "
        f"{'C':>14} "
        f"{'LR':>8} "
        f"{'root':>8} "
        f"{'f_R [Hz]':>16} "
        f"{'f_I [Hz]':>16} "
        f"{'Q':>14} "
        f"{'|Res|':>14}"
    )

    print(
        header
    )

    print(
        "-" * len(header)
    )

    for row in results:

        print(
            f"{row['label']:>14} "
            f"{row['C']:>14.6e} "
            f"{str(row['light_ring']):>8} "
            f"{str(row['root_found']):>8} "
            f"{str(row['f_re_Hz']):>16} "
            f"{str(row['f_im_Hz']):>16} "
            f"{str(row['Q']):>14} "
            f"{str(row['res_abs']):>14}"
        )

    # =========================================================================
    # BASELINE
    # =========================================================================

    baseline = next(
        (
            row
            for row in results
            if row["label"] == "BASELINE"
            and row["root_found"]
        ),
        None,
    )

    print()
    print("-" * 78)

    if baseline is None:

        print(
            "BASELINE RESONANCE: NOT VALIDATED"
        )

        print()
        print(
            "CAUSAL TEST: INCONCLUSIVE"
        )

        print(
            "No accepted baseline resonance root."
        )

        return

    print(
        "BASELINE RESONANCE: VALIDATED"
    )

    print(
        f"  f_R = "
        f"{baseline['f_re_Hz']:.12f} Hz"
    )

    print(
        f"  f_I = "
        f"{baseline['f_im_Hz']:+.12e} Hz"
    )

    print(
        f"  Q = "
        f"{baseline['Q']:.12e}"
    )

    print(
        f"  |Res| = "
        f"{baseline['res_abs']:.12e}"
    )

    # =========================================================================
    # C VARIATIONS
    # =========================================================================

    variations = [
        row
        for row in results
        if row["label"] in (
            "C_HALF",
            "C_DOUBLE",
        )
        and row["root_found"]
    ]

    print()
    print(
        "RESONANCE RESPONSE RELATIVE TO BASELINE"
    )

    print("-" * 78)

    any_resolved_response = False

    for row in variations:

        delta_r = (
            row["r_sp_m"]
            - baseline["r_sp_m"]
        )

        delta_f_R = (
            row["f_re_Hz"]
            - baseline["f_re_Hz"]
        )

        delta_f_I = (
            row["f_im_Hz"]
            - baseline["f_im_Hz"]
        )

        delta_Q = (
            row["Q"]
            - baseline["Q"]
        )

        print()
        print(
            f"{row['label']}:"
        )

        print(
            f"  Delta r_sp = "
            f"{delta_r:+.12e} m"
        )

        print(
            f"  Delta f_R  = "
            f"{delta_f_R:+.12e} Hz"
        )

        print(
            f"  Delta f_I  = "
            f"{delta_f_I:+.12e} Hz"
        )

        print(
            f"  Delta Q    = "
            f"{delta_Q:+.12e}"
        )

        # ---------------------------------------------------------------------
        # Numerical significance criterion.
        #
        # These are deliberately conservative absolute scales. They are not
        # claimed physical thresholds; they only prevent exact numerical
        # identity from being mistaken for a response.
        # ---------------------------------------------------------------------

        if (
            abs(delta_f_R) > 1e-10
            or abs(delta_f_I) > 1e-12
        ):

            any_resolved_response = True

    # =========================================================================
    # C = 0 CONTROL
    # =========================================================================

    c_zero = next(
        (
            row
            for row in results
            if row["label"] == "C_ZERO"
        ),
        None,
    )

    print()
    print(
        "C -> 0 CONTROL"
    )

    print("-" * 78)

    if c_zero is None:

        print(
            "C_ZERO result missing."
        )

    elif (
        c_zero["status"]
        == "NO_LIGHT_RING"
    ):

        print(
            "C -> 0: NO_INTERIOR_LIGHT_RING"
        )

        print(
            "No artificial resonance was searched."
        )

    elif c_zero["root_found"]:

        print(
            "C -> 0: RESONANCE ROOT FOUND"
        )

        print(
            f"  f_R = "
            f"{c_zero['f_re_Hz']:.12f} Hz"
        )

        print(
            f"  f_I = "
            f"{c_zero['f_im_Hz']:+.12e} Hz"
        )

        print(
            f"  |Res| = "
            f"{c_zero['res_abs']:.12e}"
        )

    else:

        print(
            f"C -> 0: {c_zero['status']}"
        )

    # =========================================================================
    # FINAL VERDICT
    # =========================================================================

    print()
    print("=" * 78)
    print(
        "CAUSAL AUDIT VERDICT"
    )
    print("=" * 78)

    c_half_ok = any(
        row["label"] == "C_HALF"
        and row["root_found"]
        for row in results
    )

    c_double_ok = any(
        row["label"] == "C_DOUBLE"
        and row["root_found"]
        for row in results
    )

    if (
        baseline is not None
        and c_half_ok
        and c_double_ok
        and any_resolved_response
    ):

        print(
            "C/2 -> ACCEPTED RESONANCE"
        )

        print(
            "C0  -> ACCEPTED RESONANCE"
        )

        print(
            "2C  -> ACCEPTED RESONANCE"
        )

        print()
        print(
            "RESULT: PASS"
        )

        print(
            "The complex resonance root shows a resolved"
        )

        print(
            "response to controlled variation of C."
        )

    elif (
        baseline is not None
        and any_resolved_response
    ):

        print(
            "RESULT: PARTIAL / INCOMPLETE"
        )

        print(
            "A resonance response is indicated, but not all"
        )

        print(
            "controlled C cases produced validated roots."
        )

    else:

        print(
            "RESULT: INCONCLUSIVE"
        )

        print(
            "No numerically resolved resonance-level C response"
        )

        print(
            "has been demonstrated with accepted roots."
        )

    print()
    print(
        "Scientific status:"
    )

    print(
        "This stage does NOT establish a new fundamental interaction."
    )

    print(
        "It tests whether the Root-B resonance condition itself"
    )

    print(
        "depends on the circulation parameter C."
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 78)
    print(
        "STAGE 6.5B — RESONANCE-LEVEL "
        "CAUSAL / INDEPENDENT PHYSICAL CONSEQUENCE AUDIT"
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
        "Mode:"
    )

    print(
        f"  m = {M}"
    )

    print()
    print(
        "Baseline circulation:"
    )

    print(
        f"  C0 = "
        f"{BASE_C:.12e} m^2/s"
    )

    print()
    print(
        "Controlled cases:"
    )

    print(
        "  C = 0"
    )

    print(
        "  C = C0/2"
    )

    print(
        "  C = C0"
    )

    print(
        "  C = 2*C0"
    )

    print()
    print(
        "Resonance condition:"
    )

    print(
        "  Res(omega) = R(omega) * exp(2 i S(omega)) - 1"
    )

    print()
    print(
        "Rules:"
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
        "  * only C is varied"
    )

    print()

    # =========================================================================
    # CASES
    # =========================================================================

    cases = [
        (
            0.0,
            "C_ZERO",
        ),
        (
            0.5 * BASE_C,
            "C_HALF",
        ),
        (
            BASE_C,
            "BASELINE",
        ),
        (
            2.0 * BASE_C,
            "C_DOUBLE",
        ),
    ]

    results = []

    # =========================================================================
    # RUN
    # =========================================================================

    for C_value, label in cases:

        try:

            row = solve_case(
                C_value,
                label,
                INITIAL_F,
            )

        except Exception as exc:

            # -----------------------------------------------------------------
            # Do NOT abort the entire experiment because one C case fails.
            #
            # Record it explicitly as a failed numerical case.
            # -----------------------------------------------------------------

            print()
            print(
                "  CASE EXCEPTION:"
            )

            print(
                f"  {type(exc).__name__}: "
                f"{exc}"
            )

            row = {
                "label": label,
                "C": C_value,
                "light_ring": None,
                "r_sp_m": None,
                "f_lr_Hz": None,
                "root_found": False,
                "f_re_Hz": None,
                "f_im_Hz": None,
                "tau_s": None,
                "Q": None,
                "res_abs": None,
                "res_re": None,
                "res_im": None,
                "turning_point_re_m": None,
                "turning_point_im_m": None,
                "action_re": None,
                "action_im": None,
                "reflection_abs": None,
                "reflection_re": None,
                "reflection_im": None,
                "solver_success": False,
                "solver_nfev": None,
                "status": (
                    "CASE_EXCEPTION:"
                    + type(exc).__name__
                ),
            }

        results.append(
            row
        )

    # =========================================================================
    # SAVE
    # =========================================================================

    save_results(
        results
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print_summary(
        results
    )

    print()
    print(
        "=" * 78
    )

    print(
        "OUTPUT FILES"
    )

    print(
        "=" * 78
    )

    print(
        f"CSV  -> {CSV_FILE}"
    )

    print(
        f"JSON -> {JSON_FILE}"
    )

    print()
    print(
        "Original authoritative solver was NOT modified."
    )

    print(
        "=" * 78
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()