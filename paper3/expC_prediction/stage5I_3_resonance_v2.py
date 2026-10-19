"""
stage5I_3_resonance_v2.py
==========================

STAGE 5I-3 — RESONANCE SEARCH (v2: fixed complex radial-action seeding)

This is a full, standalone copy of stage5I_3_resonance.py, with ONE
targeted change: radial_action_complex() has been replaced with a
version that seeds its FIRST continuation step analytically near the
turning point, instead of blindly seeding Newton's method at p=0 (a
point where the Hamiltonian's p-Jacobian is singular by construction,
since r_minus is a turning point). This eliminates a severe numerical
fragility diagnosed in Stage 5I-3b/5I-3c: the original implementation
returned None for omega perturbations as small as 1e-9 Hz near a
genuine resonance, because Newton continuation seeded exactly at the
singular branch point failed to converge for many nearby (omega,r)
combinations along the 300-step continuation path.

DIAGNOSIS SUMMARY (Stage 5I-3b/5I-3c):
  1. scipy.optimize.root('hybr') on the ORIGINAL radial_action_complex
     got stuck with ZERO movement when polishing the Nelder-Mead
     result for m=-12 (f=8.3066307032 - 0.0005696389j Hz), because its
     finite-difference Jacobian probing hit None at nearly every
     nearby point.
  2. Root cause: the first continuation step in radial_action_complex
     seeds solve_p_complex with p_initial=0.0j -- exactly at the
     turning point, where dH/dp=0 by definition (a classical turning
     point), making the [Re(H),Im(H)]=0 Newton system singular there.
  3. FIX: use the local quadratic (near-turning-point) expansion
         H(p,r) ~= H_r*(r-r_minus) + 0.5*H_pp*p^2
     to seed p ~= sqrt(-2*H_r*(r-r_minus)/H_pp) for the FIRST step
     only; all subsequent steps continue to use p_initial=p_previous
     exactly as in v1 (those steps are not at the singular point).
  4. BRANCH VERIFICATION: the positive sqrt root was confirmed, by
     direct comparison against the independently-validated real-axis
     solve_p() (from stage5I_1_2_dispersion_radial_p.py) just off the
     real axis, to agree to ~2e-5 -- the negative root disagreed by
     O(1) and is WRONG. (An earlier attempt using the negative-root
     convention, and a unit-confusion bug passing Hz instead of rad/s
     to find_last_scattering_point during diagnosis, are documented in
     stage5I_3c_radial_action_patch.py's history; both are resolved
     here.)
  5. VALIDATION: with this fix, resonance_factor(f_re, f_im) for
     m=-12 near the known resonance became smooth and differentiable
     (no more None returns for sub-1e-5 Hz perturbations), and a
     genuine scipy.optimize.root('hybr') polish, seeded at the
     Nelder-Mead result, converged to
         f_re = 8.3066307169 Hz
         f_im = -0.0005664350 Hz
         |Res| ~ 2.78e-17  (machine precision)
     confirming this is a TRUE pole of Res(omega)=0, not merely a good
     local minimum of |Res|.

Everything else in this file (dispersion functions, light-ring finder,
real-axis scan, Nelder-Mead complex refinement, Hessian/rho/reflection
coefficient machinery) is UNCHANGED from stage5I_3_resonance.py.

REGRESSION CHECK REQUIRED before running the full m-scan: confirm that
m=-12 still reproduces (Re,Im) ~ (8.3066307169, -0.0005664350) Hz with
|Res| ~ 1e-17 through the FULL pipeline (real scan -> Nelder-Mead ->
hybr polish) using this file's functions end-to-end, not just the
isolated diagnostic calls used during development. See
stage5I_3_v2_regression_check.py.

"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from scipy.optimize import (
    brentq,
    minimize,
    minimize_scalar,
    root,
)

from scipy.special import gamma as gamma_function


# ============================================================================
# IMPORT BASELINE
# ============================================================================

HERE = Path(__file__).resolve().parent

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


try:

    from stage5I_1_2_dispersion_radial_p import (
        C_METHODS,
        C_TABLE,
        OMEGA,
        GAMMA,
        H0,
        R_B,
        G_GRAV,
        F_dispersion,
        omega_D,
        omega_D_plus_at_p0,
        solve_p,
        find_light_ring,
    )

except ImportError as exc:

    raise ImportError(
        "\n"
        "Could not import stage5I_1_2_dispersion_radial_p.py\n"
        "\n"
        "Make sure both files are in the same directory:\n"
        "\n"
        "    stage5I_1_2_dispersion_radial_p.py\n"
        "    stage5I_3_resonance.py\n"
    ) from exc


# ============================================================================
# NUMERICAL SETTINGS
# ============================================================================

DEFAULT_N_COARSE = 5000

DEFAULT_TOP_N = 5

DEFAULT_LOCAL_HALF_WIDTH = 0.08

DEFAULT_RADIAL_N = 320

DEFAULT_COMPLEX_RADIAL_N = 300


# ============================================================================
# DISPERSION DERIVATIVES
# ============================================================================

def F_dispersion_derivatives(
    k,
    gamma=GAMMA,
    h0=H0,
    g=G_GRAV,
):
    """
    Analytic derivatives of

        F(k) = (g*k + gamma*k^3) tanh(h0*k)

    Returns

        F
        F'
        F''
    """

    k = float(k)

    t = np.tanh(
        h0 * k
    )

    sech2 = (
        1.0
        - t * t
    )

    A = (
        g * k
        + gamma * k ** 3
    )

    A1 = (
        g
        + 3.0 * gamma * k ** 2
    )

    A2 = (
        6.0 * gamma * k
    )

    F = (
        A * t
    )

    F1 = (
        A1 * t
        + A * h0 * sech2
    )

    F2 = (
        A2 * t
        + 2.0 * A1 * h0 * sech2
        - 2.0 * A * h0 ** 2 * sech2 * t
    )

    return F, F1, F2


# ============================================================================
# HAMILTONIAN
# ============================================================================

def hamiltonian(
    omega,
    p,
    m,
    r,
):
    """
    H = -1/2 (omega - omega_D^+)(omega - omega_D^-)

    Equivalent form:

        H =
            -1/2 (omega-D)^2
            + 1/2 F(k)

    where

        D = m*C/r^2 + m*Omega
        k = sqrt(p^2 + (m/r)^2)
    """

    k = np.sqrt(
        p ** 2
        + (m / r) ** 2
    )

    doppler = (
        m * C_METHODS / r ** 2
        + m * OMEGA
    )

    F = (
        g_dispersion(
            k
        )
    )

    return (
        -0.5
        * (
            omega
            - doppler
        ) ** 2
        + 0.5 * F
    )


def g_dispersion(k):
    """
    Complex-capable dispersion function.

    The baseline F_dispersion() deliberately casts to float.
    That is appropriate for its real-axis solve_p().

    Complex resonance refinement needs a complex-capable version.
    """

    return (
        G_GRAV * k
        + GAMMA * k ** 3
    ) * np.tanh(
        H0 * k
    )


# ============================================================================
# COMPLEX POSITIVE BRANCH
# ============================================================================

def omega_plus_complex(
    p,
    m,
    r,
):
    """
    Complex-capable version of omega_D^+.
    """

    k = np.sqrt(
        p ** 2
        + (m / r) ** 2
    )

    doppler = (
        m * C_METHODS / r ** 2
        + m * OMEGA
    )

    F = g_dispersion(
        k
    )

    return (
        doppler
        + np.sqrt(F)
    )


# ============================================================================
# HESSIAN
# ============================================================================

def hessian_H(
    omega,
    m,
    r_sp,
):
    """
    Hessian of H with respect to (r,p) at

        p = 0
        r = r_sp

    At p=0:

        k = |m|/r

    and

        d²k/dp² = 1/k.

    Therefore:

        H_pp = F'(k)/(2k)

    and, because H is even in p:

        H_rp = 0
    """

    r = float(
        r_sp
    )

    k = (
        abs(m)
        / r
    )

    F, F1, F2 = (
        F_dispersion_derivatives(
            k
        )
    )

    D = (
        m * C_METHODS / r ** 2
        + m * OMEGA
    )

    Q = (
        omega
        - D
    )

    D_r = (
        -2.0
        * m
        * C_METHODS
        / r ** 3
    )

    D_rr = (
        6.0
        * m
        * C_METHODS
        / r ** 4
    )

    Q_r = (
        -D_r
    )

    Q_rr = (
        -D_rr
    )

    k_r = (
        -k / r
    )

    k_rr = (
        2.0 * k / r ** 2
    )

    H_value = (
        -0.5 * Q ** 2
        + 0.5 * F
    )

    H_rr = (
        -(
            Q_r ** 2
            + Q * Q_rr
        )
        + 0.5
        * (
            F2 * k_r ** 2
            + F1 * k_rr
        )
    )

    H_pp = (
        0.5
        * F1
        / k
    )

    H_rp = 0.0

    det_H = (
        H_rr * H_pp
        - H_rp ** 2
    )

    return {
        "H": H_value,
        "H_rr": H_rr,
        "H_rp": H_rp,
        "H_pp": H_pp,
        "det_H": det_H,
    }


# ============================================================================
# RHO — EQ. (16)
# ============================================================================

def rho_parameter(
    omega,
    m,
    r_sp,
):
    """
    Eq. (16):

        rho =
            -H sign(H_pp)
            / sqrt(-det(Hessian))

    IMPORTANT:

    This function must work for both:

        omega = real

    and

        omega = complex.

    Therefore radicand is NEVER compared using

        radicand > 0

    because complex numbers cannot be ordered in Python.
    """

    omega = complex(
        omega
    )

    # Remove artificial numerical imaginary parts on the
    # real-frequency axis.
    if abs(
        omega.imag
    ) < 1.0e-14:

        omega_eval = float(
            omega.real
        )

    else:

        omega_eval = omega

    hess = hessian_H(
        omega_eval,
        m,
        r_sp,
    )

    H_value = hess[
        "H"
    ]

    H_pp = hess[
        "H_pp"
    ]

    det_H = hess[
        "det_H"
    ]

    # H_pp is real for the saddle geometry.
    H_pp_real = float(
        np.real(
            H_pp
        )
    )

    if H_pp_real >= 0.0:

        sign_Hpp = 1.0

    else:

        sign_Hpp = -1.0

    # ------------------------------------------------------------------
    # IMPORTANT FIX
    # ------------------------------------------------------------------
    #
    # Previous buggy code:
    #
    #     if radicand > 0:
    #
    # This crashes when radicand is complex.
    #
    # np.sqrt(complex_value) is exactly what we want here.
    #

    radicand = complex(
        -det_H
    )

    denominator = np.sqrt(
        radicand
    )

    if abs(
        denominator
    ) == 0.0:

        raise FloatingPointError(
            "Zero Hessian denominator."
        )

    rho = (
        -H_value
        * sign_Hpp
        / denominator
    )

    return rho, hess


# ============================================================================
# REFLECTION COEFFICIENT — EQ. (15)
# ============================================================================

def reflection_coefficient(
    omega,
    m,
    r_sp,
    omega_lr=None,
):
    """
    Eq. (15):

        R =
            -i
            (mu*rho)^(i*rho)
            exp[-rho(i + pi/2)]
            / sqrt(2*pi)
            Gamma(1/2 - i*rho)

    IMPORTANT:

        (mu*rho)^(i*rho)

    not:

        (mu*i*rho)^(i*rho)
    """

    omega = complex(
        omega
    )

    rho, _ = (
        rho_parameter(
            omega,
            m,
            r_sp,
        )
    )

    if omega_lr is None:

        _, omega_lr = (
            find_light_ring(
                m
            )
        )

        if omega_lr is None:

            raise RuntimeError(
                f"No light ring found for m={m}."
            )

    # Use real part to determine which side of the light ring
    # the complex point is associated with.
    if (
        omega.real
        < omega_lr
    ):

        mu = -1.0

    else:

        mu = +1.0

    base = (
        mu * rho
    )

    R = (
        -1j
        * np.power(
            base,
            1j * rho,
        )
        * np.exp(
            -rho
            * (
                1j
                + np.pi / 2.0
            )
        )
        / np.sqrt(
            2.0 * np.pi
        )
        * gamma_function(
            0.5
            - 1j * rho
        )
    )

    return R


# ============================================================================
# TURNING POINTS
# ============================================================================

def find_turning_points(
    omega,
    m,
    r_min=5.0e-3,
    r_max=R_B,
    n_scan=6000,
):
    """
    Find all real roots of

        omega_D^+(p=0,m,r) - omega = 0.
    """

    omega = float(
        np.real(
            omega
        )
    )

    r_grid = np.linspace(
        r_min,
        r_max,
        n_scan,
    )

    values = np.array([
        (
            omega_D_plus_at_p0(
                m,
                r,
            )
            - omega
        )
        for r in r_grid
    ])

    roots = []

    for i in range(
        len(r_grid) - 1
    ):

        a = values[i]

        b = values[i + 1]

        if not (
            np.isfinite(a)
            and np.isfinite(b)
        ):

            continue

        if a == 0.0:

            roots.append(
                r_grid[i]
            )

            continue

        if (
            a * b
            < 0.0
        ):

            try:

                rr = brentq(
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

                roots.append(
                    rr
                )

            except (
                ValueError,
                RuntimeError,
            ):

                pass

    roots.sort()

    unique = []

    for rr in roots:

        if not unique:

            unique.append(
                rr
            )

        elif abs(
            rr
            - unique[-1]
        ) > 1.0e-10:

            unique.append(
                rr
            )

    return unique


# ============================================================================
# LAST / OUTER TURNING POINT
# ============================================================================

def find_last_scattering_point(
    omega,
    m,
    r_sp,
):
    """
    Determine r_-.

    Below the light ring:

        r_- = outermost / last real turning point.

    At / above the light ring:

        r_- = r_sp.
    """

    omega_real = float(
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

    if (
        omega_real
        >= omega_lr
    ):

        return float(
            r_sp
        )

    roots = find_turning_points(
        omega_real,
        m,
    )

    if not roots:

        return None

    # CRITICAL:
    #
    # Eq. (17) starts from the LAST scattering point.
    #
    # For f=8.25 Hz and m=-12:
    #
    #     r_inner ~ 14.55 mm
    #     r_outer ~ 31.04 mm
    #
    # We need r_outer.
    return float(
        max(roots)
    )


# ============================================================================
# REAL RADIAL ACTION
# ============================================================================

def radial_action_real(
    omega,
    m,
    r_minus,
    r_B=R_B,
    n=DEFAULT_RADIAL_N,
):
    """
    I = integral_{r_-}^{r_B} p(r) dr

    Real-axis calculation.

    The substitution

        r = r_- + (r_B-r_-) x^2

    regularizes the square-root behaviour near the turning point.
    """

    omega = float(
        np.real(
            omega
        )
    )

    if (
        r_minus
        >= r_B
    ):

        raise ValueError(
            "r_minus must be < r_B."
        )

    x, weights = (
        np.polynomial
        .legendre
        .leggauss(
            n
        )
    )

    x = (
        0.5
        * (x + 1.0)
    )

    weights = (
        0.5
        * weights
    )

    span = (
        r_B
        - r_minus
    )

    r_values = (
        r_minus
        + span
        * x ** 2
    )

    p_values = np.empty(
        len(r_values),
        dtype=float,
    )

    for i, rr in enumerate(
        r_values
    ):

        p_value = solve_p(
            omega,
            m,
            float(rr),
        )

        if p_value is None:

            raise ValueError(
                "solve_p() returned None "
                "inside the propagating "
                "integration interval."
            )

        p_values[i] = float(
            p_value
        )

    jacobian = (
        2.0
        * span
        * x
    )

    integral = np.sum(
        weights
        * p_values
        * jacobian
    )

    return float(
        integral
    )


# ============================================================================
# COMPLEX p SOLVER
# ============================================================================

def solve_p_complex(
    omega,
    m,
    r,
    p_initial=0.0j,
):
    """
    Complex continuation of H(omega,p,m,r)=0.

    The two real equations are:

        Re H = 0
        Im H = 0
    """

    p_initial = complex(
        p_initial
    )

    def equations(
        x
    ):

        p = (
            x[0]
            + 1j * x[1]
        )

        H = hamiltonian(
            omega,
            p,
            m,
            r,
        )

        return np.array([
            H.real,
            H.imag,
        ])

    result = root(
        equations,
        np.array([
            p_initial.real,
            p_initial.imag,
        ]),
        method="hybr",
        options={
            "xtol": 1.0e-10,
            "maxfev": 300,
        },
    )

    if not result.success:

        return None

    p = (
        result.x[0]
        + 1j * result.x[1]
    )

    H = hamiltonian(
        omega,
        p,
        m,
        r,
    )

    if abs(
        H
    ) > 1.0e-7:

        return None

    return p


# ============================================================================
# COMPLEX TURNING POINT
# ============================================================================

def find_turning_point_complex(
    omega,
    m,
    r_real,
    n_steps=None,
):
    """
    Continue the real turning point into complex r.

    Solve:

        omega_D^+(p=0,m,r) = omega.

    STAGE 5I-4 FIX: for OVERTONES above the light-ring frequency,
    r_real = r_sp exactly (the light-ring saddle point), where
    d(omega_D^+)/dr = 0 by definition (Eq. 12). A single-shot hybr
    root-solve seeded at (r_real, 0) then has a degenerate/near-
    singular Jacobian at the START of its search, causing it to
    converge only for very small |Im(omega)| (diagnosed: succeeds up
    to Im(f)~-0.001 Hz, fails entirely for Im(f)<=-0.01 Hz) -- far
    short of the ~1 Hz damping scale the paper reports for overtones
    (Methods, Section D). This is the SAME class of degeneracy already
    diagnosed and fixed in radial_action_complex()'s first-step
    seeding (Stage 5I-3b/3c/5I-4), one level up: here it is
    find_turning_point_complex()'s OWN root-solve that starts at (or
    immediately adjacent to) the singular point, not just
    radial_action_complex()'s.

    FIX: standard numerical continuation. Instead of jumping directly
    from Im(omega)=0 to the full target Im(omega) in one hybr call,
    march in n_steps small increments, using each step's converged
    (Re(r), Im(r)) as the seed for the next. Each individual step is a
    SMALL perturbation away from an already-converged (well-seeded)
    point, avoiding the degenerate-Jacobian failure mode that a single
    large jump exhibits. Away from the light ring (turning points
    below it, where d(omega_D^+)/dr != 0), a single step already
    converges reliably (as in v1/v2), so n_steps=1 there recovers the
    original behaviour exactly.

    n_steps defaults to a value scaled to how far Im(omega) needs to
    move, using a small step size, so shallow (near-real-axis) cases
    remain cheap (few or one step) while deep (overtone-scale) cases
    get correspondingly finer continuation.
    """

    omega_im_target = float(
        omega.imag
    )

    if n_steps is None:

        # Small, fixed step size in Im(omega) (rad/s); scale the
        # number of steps to the total distance being covered.
        step_size = 0.02  # rad/s, ~0.003 Hz -- comfortably smaller
                            # than the ~0.001-0.01 Hz scale where the
                            # single-shot solve was found to break down
        n_steps = max(
            1,
            int(
                np.ceil(
                    abs(omega_im_target) / step_size
                )
            ),
        )

    r_seed = np.array([
        float(r_real),
        0.0,
    ])

    omega_re = float(
        omega.real
    )

    rr = None

    for step in range(
        1,
        n_steps + 1,
    ):

        omega_im_step = (
            omega_im_target
            * step
            / n_steps
        )

        omega_step = (
            omega_re
            + 1j * omega_im_step
        )

        def equations(
            x,
            _omega_step=omega_step,
        ):

            rr_trial = (
                x[0]
                + 1j * x[1]
            )

            value = (
                omega_plus_complex(
                    0.0j,
                    m,
                    rr_trial,
                )
                - _omega_step
            )

            return np.array([
                value.real,
                value.imag,
            ])

        result = root(
            equations,
            r_seed,
            method="hybr",
            options={
                "xtol": 1.0e-10,
                "maxfev": 300,
            },
        )

        if not result.success:

            return None

        r_seed = result.x

        rr = (
            result.x[0]
            + 1j * result.x[1]
        )

    if rr is None:

        return None

    residual = (
        omega_plus_complex(
            0.0j,
            m,
            rr,
        )
        - omega
    )

    if abs(
        residual
    ) > 1.0e-7:

        return None

    return rr


# ============================================================================
# COMPLEX RADIAL ACTION — v2 (fixed first-step seeding, Stage 5I-3b/3c)
# ============================================================================

def compute_H_r_at_turning_point(
    omega,
    m,
    r_minus,
    eps=None,
):
    """
    dH/dr at (p=0, r=r_minus), via finite differences.

    eps is scaled to the physical magnitude of r_minus (metres) to
    avoid catastrophic cancellation, following the same discipline
    established in hessian_at_saddle()'s physically-scaled eps_p/eps_r.
    """

    if eps is None:

        eps = (
            abs(r_minus) * 1.0e-6
            if abs(r_minus) > 0
            else 1.0e-9
        )

    H_plus = hamiltonian(
        omega,
        0.0j,
        m,
        r_minus + eps,
    )

    H_minus = hamiltonian(
        omega,
        0.0j,
        m,
        r_minus - eps,
    )

    return (
        (H_plus - H_minus)
        / (2.0 * eps)
    )


def compute_H_rr_at_turning_point(
    omega,
    m,
    r_minus,
    eps=None,
):
    """
    d^2H/dr^2 at (p=0, r=r_minus), via finite differences.

    STAGE 5I-4 DIAGNOSIS: when r_minus coincides with (or lies very
    close to) the light-ring radius r_sp -- the case relevant for
    overtones ABOVE the light-ring frequency, where
    find_last_scattering_point() returns r_minus = r_sp exactly --
    r_minus is not a simple 1D turning point but a genuine 2D SADDLE
    POINT of H(p,r): by definition of the light ring, BOTH dH/dp=0 AND
    dH/dr=0 there (Eq. 12 in the paper). compute_H_r_at_turning_point()
    then returns a value near zero (verified numerically: ~2e-5,
    versus ~1e4-1e6 at a genuine 1D turning point below the light
    ring), which makes the turning-point seed formula in
    radial_action_complex() degenerate (p_seed -> 0 regardless of
    delta_r), causing Nelder-Mead refinement to start from a
    meaningless point and fail entirely for every overtone candidate.

    This function supplies the SECOND radial derivative needed for the
    correct LOCAL SADDLE-POINT (hyperbolic) expansion in that case:

        H(p,r) ~= 0.5*H_pp*p^2 + H_pr*p*(r-r_sp) + 0.5*H_rr*(r-r_sp)^2

    Since H is even in p (k depends on p only via p^2), H_pr=0 exactly
    at p=0 (same fact already used in the reflection-coefficient
    Hessian, hessian_H()), so this simplifies to

        p ~= +/- (r - r_sp) * sqrt(-H_rr/H_pp)

    -- LINEAR in (r-r_sp), not sqrt(r-r_sp) as at a simple 1D turning
    point. This is used by radial_action_complex() only when the
    degenerate-H_r case is detected.
    """

    if eps is None:

        eps = (
            abs(r_minus) * 1.0e-6
            if abs(r_minus) > 0
            else 1.0e-9
        )

    H_plus = hamiltonian(
        omega,
        0.0j,
        m,
        r_minus + eps,
    )

    H_center = hamiltonian(
        omega,
        0.0j,
        m,
        r_minus,
    )

    H_minus = hamiltonian(
        omega,
        0.0j,
        m,
        r_minus - eps,
    )

    return (
        (H_plus - 2.0 * H_center + H_minus)
        / (eps ** 2)
    )


# Threshold below which H_r is treated as numerically degenerate
# (light-ring saddle point case), given the large scale gap observed
# between a genuine 1D turning point (H_r ~ 1e4-1e6) and the light
# ring itself (H_r ~ 2e-5), for this experimental configuration.
SADDLE_POINT_H_R_THRESHOLD = 1.0e-3


def radial_action_complex(
    omega,
    m,
    r_minus,
    r_B=R_B,
    n=DEFAULT_COMPLEX_RADIAL_N,
):
    """
    Local complex continuation of

        integral p(r) dr

    along the straight contour from r_- to r_B.

    v2 FIX (Stage 5I-3b/3c): the FIRST continuation step (i=1) seeds
    Newton's method (solve_p_complex) with an ANALYTIC estimate from
    the near-turning-point quadratic expansion of H,

        H(p,r) ~= H_r*(r-r_minus) + 0.5*H_pp*p^2
        => p ~= sqrt(-2*H_r*(r-r_minus)/H_pp)

    instead of the original's p_initial=0.0j, which sits EXACTLY at
    the turning point -- a branch point where dH/dp=0 by definition,
    making Newton's method's Jacobian singular there. This caused
    resonance_factor() to return None for omega perturbations as
    small as 1e-9 Hz near genuine resonances (diagnosed via failed
    scipy.optimize.root('hybr') polishing attempts).

    BRANCH: the POSITIVE sqrt root is used, verified by direct
    comparison against the independently-validated real-axis solve_p()
    (stage5I_1_2_dispersion_radial_p.py) just off the real axis, where
    it agreed to ~2e-5; the negative root disagreed by O(1) and is
    the WRONG branch for this continuation's sign convention.

    v2b FIX (Stage 5I-4): the above formula assumes H_r != 0 at
    r_minus, valid for a genuine 1D turning point BELOW the light-ring
    frequency. For OVERTONES above the light-ring frequency,
    find_last_scattering_point() returns r_minus = r_sp -- the light
    ring itself, a genuine 2D SADDLE POINT where H_r=0 too (Eq. 12).
    The turning-point formula then degenerates (p_seed -> 0
    regardless of delta_r), causing complex refinement to fail for
    every overtone candidate (diagnosed via systematic "Nelder-Mead
    failed" results for all real-axis candidates above f_lr in the
    Stage 5I-4 full m-scan).

    FIX: when |H_r| falls below SADDLE_POINT_H_R_THRESHOLD (the light-
    ring case), use the correct LOCAL SADDLE-POINT (hyperbolic)
    expansion instead, via compute_H_rr_at_turning_point():

        p ~= (r - r_minus) * sqrt(-H_rr/H_pp)

    (linear in delta_r, not sqrt(delta_r) -- see that function's
    docstring for the derivation). Away from the light ring (H_r
    non-negligible), behaviour is UNCHANGED from v2.

    All subsequent steps (i>=2) are UNCHANGED from v1/v2: they continue
    to seed from p_previous, since those points are not at the
    singular branch point and Newton continuation from an already-
    converged neighbour remains well-behaved there.
    """

    t = np.linspace(
        0.0,
        1.0,
        n,
    )

    r_path = (
        r_minus
        + t
        * (
            r_B
            - r_minus
        )
    )

    p_values = np.zeros(
        n,
        dtype=complex,
    )

    p_previous = 0.0j

    # Local curvature estimate at the turning point, used only to
    # build the analytic first-step seed (not the final answer for
    # that step -- solve_p_complex still refines it via Newton's
    # method afterward).
    eps_p_probe = 1.0  # 1/m, modest probe step

    H_pp_estimate = (
        hamiltonian(omega, eps_p_probe + 0j, m, r_minus)
        - 2 * hamiltonian(omega, 0j, m, r_minus)
        + hamiltonian(omega, -eps_p_probe + 0j, m, r_minus)
    ) / eps_p_probe ** 2

    H_r = compute_H_r_at_turning_point(
        omega,
        m,
        r_minus,
    )

    # v2b FIX: detect the light-ring saddle-point case (H_r
    # degenerate) and precompute H_rr for the hyperbolic seed formula
    # in that case. Cheap: one extra finite-difference evaluation,
    # only used when actually needed.
    is_saddle_point_case = (
        abs(H_r) < SADDLE_POINT_H_R_THRESHOLD
    )

    if is_saddle_point_case:

        H_rr = compute_H_rr_at_turning_point(
            omega,
            m,
            r_minus,
        )

    for i, rr in enumerate(
        r_path
    ):

        if i == 0:

            p_values[i] = 0.0j

            continue

        if i == 1:

            delta_r = (
                rr
                - r_minus
            )

            if is_saddle_point_case:

                # Hyperbolic saddle-point expansion (H_pr=0 exactly,
                # since H is even in p): see
                # compute_H_rr_at_turning_point()'s docstring.
                p_seed = (
                    delta_r
                    * np.sqrt(
                        -H_rr
                        / H_pp_estimate
                    )
                )

            else:

                # Original v2 parabolic turning-point expansion.
                radicand = (
                    -2.0
                    * H_r
                    * delta_r
                    / H_pp_estimate
                )

                # Positive branch -- see docstring BRANCH note above.
                p_seed = np.sqrt(
                    radicand
                )

        else:

            p_seed = p_previous

        p_new = solve_p_complex(
            omega,
            m,
            rr,
            p_initial=p_seed,
        )

        if (
            p_new is None
            and i == 1
        ):

            # Fallback only: try the other branch if the primary
            # seed somehow fails to converge for this particular
            # (omega, r) combination.
            p_new = solve_p_complex(
                omega,
                m,
                rr,
                p_initial=-p_seed,
            )

        if p_new is None:

            raise FloatingPointError(
                "Complex p continuation "
                f"failed at step {i}, r={rr}."
            )

        # Because H is even in p, both +p and -p
        # are roots. Pick the branch continuously
        # connected to the previous point.
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

    return np.trapezoid(
        p_values,
        r_path,
    )


# ============================================================================
# RESONANCE FACTOR — EQ. (17)
# ============================================================================

def resonance_factor(
    omega,
    m,
    r_sp=None,
    omega_lr=None,
    complex_action=True,
):
    """
    Eq. (17):

        Res(omega) =
            R(omega)
            exp(
                2 i integral_{r_-}^{r_B} p dr
            )
            - 1
    """

    omega = complex(
        omega
    )

    # ------------------------------------------------------------------
    # Light ring
    # ------------------------------------------------------------------

    if (
        r_sp is None
        or omega_lr is None
    ):

        r_sp_found, omega_lr_found = (
            find_light_ring(
                m
            )
        )

        if r_sp_found is None:

            return None

        if r_sp is None:

            r_sp = r_sp_found

        if omega_lr is None:

            omega_lr = omega_lr_found

    # ------------------------------------------------------------------
    # Reflection coefficient
    # ------------------------------------------------------------------

    try:

        R = reflection_coefficient(
            omega,
            m,
            r_sp,
            omega_lr,
        )

    except (
        ArithmeticError,
        FloatingPointError,
        ValueError,
        OverflowError,
    ):

        return None

    if not (
        np.isfinite(
            R.real
        )
        and np.isfinite(
            R.imag
        )
    ):

        return None

    # ------------------------------------------------------------------
    # Real-axis case
    # ------------------------------------------------------------------

    if (
        not complex_action
        or abs(
            omega.imag
        ) < 1.0e-13
    ):

        r_minus = (
            find_last_scattering_point(
                omega.real,
                m,
                r_sp,
            )
        )

        if r_minus is None:

            return None

        try:

            action = radial_action_real(
                omega.real,
                m,
                r_minus,
                R_B,
            )

        except (
            ArithmeticError,
            FloatingPointError,
            ValueError,
            RuntimeError,
        ):

            return None

    # ------------------------------------------------------------------
    # Complex case
    # ------------------------------------------------------------------

    else:

        r_minus_real = (
            find_last_scattering_point(
                omega.real,
                m,
                r_sp,
            )
        )

        if r_minus_real is None:

            return None

        # ------------------------------------------------------------
        # STAGE 5I-4 FIX: conceptual correction for the above-light-
        # ring branch (OVERTONES).
        #
        # find_last_scattering_point() returns r_minus_real = r_sp
        # EXACTLY whenever omega.real >= omega_lr, by the model's own
        # convention (paper, Resonance conditions: "for higher
        # frequencies, r_- = r_sp"). This is a FIXED reference point
        # (depends only on m, via find_light_ring(m)) -- it is NOT a
        # root of "omega_D^+(p=0,r) = omega.real" for omega.real !=
        # omega_lr, since omega_D^+(p=0,r_sp) = omega_lr BY DEFINITION
        # of r_sp, not omega.real.
        #
        # find_turning_point_complex() was designed to continuously
        # deform a genuine ROOT of that equation (the case below the
        # light ring, where r_minus_real truly solves it) into the
        # complex-omega plane. Applying it here asks hybr to search
        # for some OTHER complex r satisfying omega_D^+(0,r)=omega --
        # a well-defined but PHYSICALLY IRRELEVANT point, unrelated to
        # the model's r_-=r_sp convention, and moreover ill-conditioned
        # to solve for (confirmed: fails to converge even for
        # |Im(omega)| as small as 1e-4 rad/s when Re(omega) is not
        # very close to omega_lr).
        #
        # FIX: for this branch, r_minus does not need any omega-
        # dependent continuation at all -- it simply IS r_sp, for any
        # omega (real or complex) with Re(omega) >= omega_lr. Skip
        # find_turning_point_complex entirely in this case.
        # ------------------------------------------------------------

        if abs(r_minus_real - r_sp) < 1.0e-9:

            r_minus_complex = complex(
                r_sp,
                0.0,
            )

        else:

            r_minus_complex = (
                find_turning_point_complex(
                    omega,
                    m,
                    r_minus_real,
                )
            )

            if r_minus_complex is None:

                return None

        try:

            action = radial_action_complex(
                omega,
                m,
                r_minus_complex,
                R_B,
            )

        except (
            ArithmeticError,
            FloatingPointError,
            ValueError,
            RuntimeError,
        ):

            return None

    # ------------------------------------------------------------------
    # Eq. (17)
    # ------------------------------------------------------------------

    result = (
        R
        * np.exp(
            2.0j
            * action
        )
        - 1.0
    )

    if not (
        np.isfinite(
            result.real
        )
        and np.isfinite(
            result.imag
        )
    ):

        return None

    return result


# ============================================================================
# REAL-AXIS SEARCH
# ============================================================================

def find_resonances(
    m,
    f_min,
    f_max,
    resonance_factor,
    n_coarse=5000,
    top_n=10,
    local_half_width=0.08,
):
    """
    Real-axis resonance search.

    1. Dense real-axis scan.
    2. Local minima of |Res|.
    3. 1-D bounded refinement.

    No complex grid.
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

    r_sp, omega_lr = (
        find_light_ring(
            m
        )
    )

    if r_sp is None:

        return []

    # ------------------------------------------------------------------
    # Dense scan
    # ------------------------------------------------------------------

    for i, f in enumerate(
        f_grid
    ):

        try:

            z = resonance_factor(
                omega=(
                    2.0
                    * np.pi
                    * f
                    + 0.0j
                ),
                m=m,
                r_sp=r_sp,
                omega_lr=omega_lr,
                complex_action=False,
            )

            if z is None:

                continue

            value = abs(
                z
            )

            if np.isfinite(
                value
            ):

                abs_res[i] = (
                    value
                )

        except (
            OverflowError,
            FloatingPointError,
            ValueError,
            ArithmeticError,
        ):

            pass

    # ------------------------------------------------------------------
    # Check scan
    # ------------------------------------------------------------------

    finite_mask = np.isfinite(
        abs_res
    )

    if not np.any(
        finite_mask
    ):

        print(
            "DEBUG: real-axis scan "
            "produced no finite Res values."
        )

        return []

    # ------------------------------------------------------------------
    # Find local minima
    # ------------------------------------------------------------------

    candidates = []

    for i in range(
        1,
        len(f_grid) - 1
    ):

        if not np.isfinite(
            abs_res[i]
        ):

            continue

        if (
            abs_res[i]
            <= abs_res[i - 1]
            and
            abs_res[i]
            <= abs_res[i + 1]
        ):

            candidates.append(
                i
            )

    # ------------------------------------------------------------------
    # If no strict interior minimum exists,
    # report the best finite point.
    # ------------------------------------------------------------------

    if not candidates:

        finite_indices = (
            np.flatnonzero(
                finite_mask
            )
        )

        best_index = (
            finite_indices[
                np.argmin(
                    abs_res[
                        finite_mask
                    ]
                )
            ]
        )

        print(
            "DEBUG: no real-axis local "
            "minimum found."
        )

        print(
            f"DEBUG: finite samples = "
            f"{len(finite_indices)}/{n_coarse}"
        )

        print(
            f"DEBUG: best finite point = "
            f"{f_grid[best_index]:.10f} Hz"
        )

        print(
            f"DEBUG: best |Res| = "
            f"{abs_res[best_index]:.8e}"
        )

        return []

    # Strongest minima first.
    candidates = sorted(
        candidates,
        key=lambda i:
            abs_res[i],
    )[:top_n]

    # ------------------------------------------------------------------
    # Refine each candidate
    # ------------------------------------------------------------------

    df = (
        f_grid[1]
        - f_grid[0]
    )

    resonances = []

    for index in candidates:

        f0 = float(
            f_grid[index]
        )

        half_width = max(
            local_half_width,
            3.0 * df,
        )

        lo = max(
            f_min,
            f0 - half_width,
        )

        hi = min(
            f_max,
            f0 + half_width,
        )

        def objective(
            f
        ):

            try:

                z = resonance_factor(
                    omega=(
                        2.0
                        * np.pi
                        * f
                        + 0.0j
                    ),
                    m=m,
                    r_sp=r_sp,
                    omega_lr=omega_lr,
                    complex_action=False,
                )

                if z is None:

                    return 1.0e100

                value = abs(
                    z
                )

                if not np.isfinite(
                    value
                ):

                    return 1.0e100

                return float(
                    value
                )

            except (
                OverflowError,
                FloatingPointError,
                ValueError,
                ArithmeticError,
            ):

                return 1.0e100

        result = minimize_scalar(
            objective,
            bounds=(
                lo,
                hi,
            ),
            method="bounded",
            options={
                "xatol": 1.0e-10,
                "maxiter": 500,
            },
        )

        if not result.success:

            continue

        resonances.append({
            "f_real": float(
                result.x
            ),
            "omega_real": (
                2.0
                * np.pi
                * float(
                    result.x
                )
            ),
            "abs_res": float(
                result.fun
            ),
            "coarse_f": f0,
        })

    # ------------------------------------------------------------------
    # Remove duplicates
    # ------------------------------------------------------------------

    resonances.sort(
        key=lambda x:
            x["f_real"]
    )

    unique = []

    for r in resonances:

        if not unique:

            unique.append(
                r
            )

            continue

        if (
            abs(
                r["f_real"]
                - unique[-1]["f_real"]
            )
            > 0.5 * df
        ):

            unique.append(
                r
            )

        elif (
            r["abs_res"]
            < unique[-1]["abs_res"]
        ):

            unique[-1] = r

    return unique


# ============================================================================
# COMPLEX REFINEMENT
# ============================================================================

def refine_complex_resonance(
    f0,
    im0=-0.01,
    resonance_factor=None,
    m=None,
    re_half_width=0.08,
    im_min=-1.5,
    im_max=0.0,
):
    """
    Local complex minimization.

    Variables:

        x[0] = Re(f)
        x[1] = Im(f)

    omega = 2*pi*f.

    The objective is:

        log(1 + |Res|)
    """

    if resonance_factor is None:

        raise ValueError(
            "resonance_factor is required."
        )

    if m is None:

        raise ValueError(
            "m is required."
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

        # Local real-frequency window.
        if not (
            f0
            - 4.0
            * re_half_width
            <= f_re
            <= f0
            + 4.0
            * re_half_width
        ):

            return 1.0e100

        # Damped half-plane.
        if not (
            im_min
            <= f_im
            <= im_max
        ):

            return 1.0e100

        omega = (
            2.0
            * np.pi
            * (
                f_re
                + 1j * f_im
            )
        )

        try:

            z = resonance_factor(
                omega=omega,
                m=m,
                complex_action=True,
            )

            if z is None:

                return 1.0e100

            value = abs(
                z
            )

            if not np.isfinite(
                value
            ):

                return 1.0e100

            return float(
                np.log1p(
                    value
                )
            )

        except (
            OverflowError,
            FloatingPointError,
            ValueError,
            ArithmeticError,
        ):

            return 1.0e100

    result = minimize(
        objective,
        x0=np.array([
            f0,
            im0,
        ]),
        method="Nelder-Mead",
        options={
            "xatol": 1.0e-9,
            "fatol": 1.0e-10,
            "maxiter": 1000,
        },
    )

    f_re = float(
        result.x[0]
    )

    f_im = float(
        result.x[1]
    )

    # Reject runaway solution.
    if not (
        f0
        - 4.0
        * re_half_width
        <= f_re
        <= f0
        + 4.0
        * re_half_width
    ):

        return None

    if not (
        im_min
        <= f_im
        <= im_max
    ):

        return None

    abs_res = float(
        np.expm1(
            result.fun
        )
    )

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
        "abs_res": abs_res,
        "success": bool(
            result.success
        ),
        "message": str(
            result.message
        ),
    }


# ============================================================================
# DIAGNOSTIC
# ============================================================================

def print_turning_point_diagnostic(
    m,
    f,
    r_sp,
):
    """
    Print all real turning points and identify r_-.
    """

    omega = (
        2.0
        * np.pi
        * f
    )

    roots = find_turning_points(
        omega,
        m,
    )

    print(
        f"Turning points at "
        f"f={f:.3f} Hz:"
    )

    if not roots:

        print(
            "    none"
        )

        return None

    for i, rr in enumerate(
        roots,
        start=1,
    ):

        print(
            f"    r_{i} = "
            f"{rr * 1.0e3:.6f} mm"
        )

    r_minus = (
        find_last_scattering_point(
            omega,
            m,
            r_sp,
        )
    )

    print()

    print(
        "Using LAST / OUTER "
        "turning point:"
    )

    print(
        f"    r_- = "
        f"{r_minus * 1.0e3:.6f} mm"
    )

    return r_minus


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 78)
    print(
        "STAGE 5I-3 — RESONANCE SEARCH"
    )
    print("=" * 78)

    print()

    print(
        "Baseline: "
        "stage5I_1_2_dispersion_radial_p.py"
    )

    print(
        "Resonance condition:"
    )

    print(
        "    Res(omega) = "
        "R(omega) exp(2 i integral p dr) - 1"
    )

    print()

    print(
        "Search:"
    )

    print(
        "    real-axis dense scan"
    )

    print(
        "        -> local minima"
    )

    print(
        "        -> bounded 1-D refinement"
    )

    print(
        "        -> local complex refinement"
    )

    print()

    print(
        "NO coarse 2-D complex grid."
    )

    print()

    # ------------------------------------------------------------------
    # Test mode
    # ------------------------------------------------------------------

    m = -12

    f_min = 7.0

    f_max = 10.5

    print(
        f"Mode m = {m}"
    )

    print(
        f"Frequency range = "
        f"[{f_min:.3f}, {f_max:.3f}] Hz"
    )

    print()

    # ------------------------------------------------------------------
    # Light ring
    # ------------------------------------------------------------------

    r_sp, omega_lr = (
        find_light_ring(
            m
        )
    )

    if r_sp is None:

        print(
            "ERROR: no light ring found."
        )

        return

    f_lr = (
        omega_lr
        / (
            2.0
            * np.pi
        )
    )

    print(
        f"Light ring: "
        f"r_sp={r_sp * 1.0e3:.6f} mm, "
        f"f_lr={f_lr:.10f} Hz"
    )

    print()

    # ------------------------------------------------------------------
    # Turning-point diagnostic
    # ------------------------------------------------------------------

    f_probe = 8.25

    print_turning_point_diagnostic(
        m,
        f_probe,
        r_sp,
    )

    print()

    # ------------------------------------------------------------------
    # Hessian diagnostic
    # ------------------------------------------------------------------

    rho, hess = (
        rho_parameter(
            2.0
            * np.pi
            * f_probe,
            m,
            r_sp,
        )
    )

    print(
        "Hessian diagnostic:"
    )

    print(
        f"    H_pp  = "
        f"{hess['H_pp']:.8e}"
    )

    print(
        f"    det H = "
        f"{hess['det_H']:.8e}"
    )

    print(
        f"    rho   = "
        f"{rho:.8e}"
    )

    print()

    # ------------------------------------------------------------------
    # Reflection diagnostic
    # ------------------------------------------------------------------

    try:

        R_probe = (
            reflection_coefficient(
                2.0
                * np.pi
                * f_probe
                + 0.0j,
                m,
                r_sp,
                omega_lr,
            )
        )

        print(
            f"Reflection coefficient "
            f"at {f_probe:.3f} Hz:"
        )

        print(
            f"    R = "
            f"{R_probe.real:+.8e}"
            f"{R_probe.imag:+.8e}j"
        )

        print(
            f"    |R| = "
            f"{abs(R_probe):.8e}"
        )

    except Exception as exc:

        print(
            "Reflection coefficient "
            "diagnostic FAILED:"
        )

        print(
            f"    {type(exc).__name__}: {exc}"
        )

        return

    print()

    # ------------------------------------------------------------------
    # Real-axis probe
    # ------------------------------------------------------------------

    z_probe = (
        resonance_factor(
            omega=(
                2.0
                * np.pi
                * f_probe
                + 0.0j
            ),
            m=m,
            r_sp=r_sp,
            omega_lr=omega_lr,
            complex_action=False,
        )
    )

    if z_probe is None:

        print(
            "Probe resonance factor: None"
        )

    else:

        print(
            f"Probe at "
            f"{f_probe:.3f} Hz:"
        )

        print(
            f"    Res = "
            f"{z_probe.real:+.8e}"
            f"{z_probe.imag:+.8e}j"
        )

        print(
            f"    |Res| = "
            f"{abs(z_probe):.8e}"
        )

    print()

    # ------------------------------------------------------------------
    # Dense real-axis search
    # ------------------------------------------------------------------

    print(
        f"Running dense real-axis scan "
        f"({DEFAULT_N_COARSE} points)..."
    )

    resonances = (
        find_resonances(
            m=m,
            f_min=f_min,
            f_max=f_max,
            resonance_factor=resonance_factor,
            n_coarse=DEFAULT_N_COARSE,
            top_n=DEFAULT_TOP_N,
            local_half_width=DEFAULT_LOCAL_HALF_WIDTH,
        )
    )

    if not resonances:

        print()

        print(
            "No real-axis local minima found."
        )

        print()

        print(
            "STAGE 5I-3 STOPPED AFTER "
            "REAL-AXIS SEARCH."
        )

        return

    print()

    print(
        "Real-axis candidates:"
    )

    for candidate in resonances:

        print(
            f"    candidate: "
            f"f={candidate['f_real']:.10f} Hz "
            f"|Res|={candidate['abs_res']:.8e} "
            f"coarse={candidate['coarse_f']:.10f} Hz"
        )

    print()

    # ------------------------------------------------------------------
    # Complex refinement
    # ------------------------------------------------------------------

    print(
        "Local complex refinement:"
    )

    for candidate in resonances:

        f0 = (
            candidate[
                "f_real"
            ]
        )

        print()

        print(
            f"    Starting from "
            f"f0={f0:.10f} Hz"
        )

        refined = (
            refine_complex_resonance(
                f0=f0,
                im0=-0.01,
                resonance_factor=resonance_factor,
                m=m,
                re_half_width=DEFAULT_LOCAL_HALF_WIDTH,
                im_min=-1.5,
                im_max=0.0,
            )
        )

        if refined is None:

            print(
                "    complex refinement FAILED"
            )

        else:

            print(
                f"    f_re  = "
                f"{refined['f_re']:.10f} Hz"
            )

            print(
                f"    f_im  = "
                f"{refined['f_im']:.10f} Hz"
            )

            print(
                f"    omega_re = "
                f"{refined['omega_re']:.10f} rad/s"
            )

            print(
                f"    omega_im = "
                f"{refined['omega_im']:.10f} rad/s"
            )

            print(
                f"    |Res| = "
                f"{refined['abs_res']:.8e}"
            )

            print(
                f"    success = "
                f"{refined['success']}"
            )

            print(
                f"    message = "
                f"{refined['message']}"
            )

    print()

    print(
        "=" * 78
    )

    print(
        "STAGE 5I-3 COMPLETE"
    )

    print(
        "=" * 78
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()