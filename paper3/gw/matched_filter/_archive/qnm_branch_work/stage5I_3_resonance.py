"""
stage5I_3_resonance.py
======================

STAGE 5I-3 — RESONANCE SEARCH

Baseline:
    stage5I_1_2_dispersion_radial_p.py

Resonance condition:

    Res(omega) =
        R(omega) * exp(2 i integral_{r_-}^{r_B} p dr) - 1

Search strategy:

    dense real-axis scan
        ->
    local minima of |Res|
        ->
    bounded 1-D refinement
        ->
    local complex minimization
        ->
    TRUE complex root solve:
        Re(Res) = 0
        Im(Res) = 0

No coarse 2-D complex grid is used.

Important:
    The OUTER / LAST real turning point is used below the
    light-ring frequency.
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
# BASELINE IMPORT
# ============================================================================

HERE = Path(__file__).resolve().parent

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


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


# ============================================================================
# NUMERICAL SETTINGS
# ============================================================================

N_COARSE = 5000

TOP_N = 5

LOCAL_HALF_WIDTH = 0.08

RADIAL_N = 320

COMPLEX_RADIAL_N = 300


# ============================================================================
# COMPLEX-CAPABLE DISPERSION FUNCTION
# ============================================================================

def F_complex(k):
    """
    Complex-capable version of

        F(k) = (g k + gamma k^3) tanh(h0 k)

    The baseline F_dispersion() deliberately casts to float, which
    is correct for the real-axis baseline solve. Complex refinement
    needs this version.
    """

    return (
        G_GRAV * k
        + GAMMA * k ** 3
    ) * np.tanh(
        H0 * k
    )


# ============================================================================
# COMPLEX DISPERSION BRANCH
# ============================================================================

def omega_plus_complex(
    p,
    m,
    r,
):
    """
    Complex-capable omega_D^+.
    """

    k = np.sqrt(
        p ** 2
        + (m / r) ** 2
    )

    doppler = (
        m * C_METHODS / r ** 2
        + m * OMEGA
    )

    return (
        doppler
        + np.sqrt(
            F_complex(k)
        )
    )


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
    H(omega,p,r) =
        -1/2 (omega-D)^2
        +1/2 F(k)

    with

        D = m C/r^2 + m Omega
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

    return (
        -0.5
        * (
            omega
            - doppler
        ) ** 2
        + 0.5
        * F_complex(k)
    )


# ============================================================================
# DISPERSION DERIVATIVES
# ============================================================================

def F_derivatives(k):
    """
    Derivatives of

        F(k) = (g k + gamma k^3) tanh(h0 k)

    Returns:

        F
        F'
        F''
    """

    k = float(k)

    t = np.tanh(
        H0 * k
    )

    sech2 = (
        1.0
        - t * t
    )

    A = (
        G_GRAV * k
        + GAMMA * k ** 3
    )

    A1 = (
        G_GRAV
        + 3.0 * GAMMA * k ** 2
    )

    A2 = (
        6.0 * GAMMA * k
    )

    F = (
        A * t
    )

    F1 = (
        A1 * t
        + A * H0 * sech2
    )

    F2 = (
        A2 * t
        + 2.0 * A1 * H0 * sech2
        - 2.0
        * A
        * H0 ** 2
        * sech2
        * t
    )

    return F, F1, F2


# ============================================================================
# HESSIAN AT THE LIGHT RING
# ============================================================================

def hessian_H(
    omega,
    m,
    r_sp,
):
    """
    Hessian of H with respect to (r,p), evaluated at p=0,
    r=r_sp.

    H_pp = F'(k)/(2k)

    H_rp = 0

    det(H) = H_rr H_pp
    """

    r = float(
        r_sp
    )

    k = (
        abs(m)
        / r
    )

    F, F1, F2 = (
        F_derivatives(k)
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

    Q_r = -D_r

    Q_rr = -D_rr

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
    Eq. (16)-type saddle parameter.

    IMPORTANT:
    This function is valid for both real and complex omega.

    Never do:

        if radicand > 0:

    because radicand can be complex.
    """

    omega = complex(
        omega
    )

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

    H_pp_real = float(
        np.real(
            H_pp
        )
    )

    if H_pp_real >= 0.0:
        sign_Hpp = 1.0
    else:
        sign_Hpp = -1.0

    # IMPORTANT FIX:
    #
    # No ordering comparison is performed here.
    # Complex square root is allowed.
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
    Reflection coefficient.

    R =
        -i
        (mu rho)^(i rho)
        exp[-rho(i + pi/2)]
        / sqrt(2 pi)
        Gamma(1/2 - i rho)
    """

    omega = complex(
        omega
    )

    rho, _ = rho_parameter(
        omega,
        m,
        r_sp,
    )

    if omega_lr is None:

        _, omega_lr = (
            find_light_ring(
                m
            )
        )

        if omega_lr is None:

            raise RuntimeError(
                "No light ring found."
            )

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
# REAL TURNING POINTS
# ============================================================================

def find_turning_points(
    omega,
    m,
    r_min=5.0e-3,
    r_max=R_B,
    n_scan=6000,
):
    """
    Find real roots of

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

            except Exception:

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
# OUTER / LAST TURNING POINT
# ============================================================================

def find_last_scattering_point(
    omega,
    m,
    r_sp,
):
    """
    For omega below the light-ring frequency:

        r_- = outermost real turning point.

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
    n=RADIAL_N,
):
    """
    Compute

        I = integral_{r_-}^{r_B} p(r) dr

    using a turning-point regularizing substitution:

        r = r_- + (r_B-r_-) x^2.
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
            "r_minus >= r_B"
        )

    nodes, weights = (
        np.polynomial
        .legendre
        .leggauss(n)
    )

    x = (
        0.5
        * (
            nodes
            + 1.0
        )
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

        p = solve_p(
            omega,
            m,
            float(rr),
        )

        if p is None:

            raise ValueError(
                "solve_p returned None "
                "inside propagation region."
            )

        p_values[i] = float(
            p
        )

    jacobian = (
        2.0
        * span
        * x
    )

    return float(
        np.sum(
            weights
            * p_values
            * jacobian
        )
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
    Solve H(omega,p,r)=0 in the complex p plane.

    Uses two real equations:

        Re H = 0
        Im H = 0.
    """

    p_initial = complex(
        p_initial
    )

    def equations(x):

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

    if abs(H) > 1.0e-7:

        return None

    return p


# ============================================================================
# COMPLEX TURNING POINT
# ============================================================================

def find_turning_point_complex(
    omega,
    m,
    r_real,
):
    """
    Continue a real turning point into the complex r plane.
    """

    def equations(x):

        r = (
            x[0]
            + 1j * x[1]
        )

        value = (
            omega_plus_complex(
                0.0j,
                m,
                r,
            )
            - omega
        )

        return np.array([
            value.real,
            value.imag,
        ])

    result = root(
        equations,
        np.array([
            float(r_real),
            0.0,
        ]),
        method="hybr",
        options={
            "xtol": 1.0e-10,
            "maxfev": 300,
        },
    )

    if not result.success:

        return None

    r = (
        result.x[0]
        + 1j * result.x[1]
    )

    residual = (
        omega_plus_complex(
            0.0j,
            m,
            r,
        )
        - omega
    )

    if abs(
        residual
    ) > 1.0e-7:

        return None

    return r


# ============================================================================
# COMPLEX RADIAL ACTION
# ============================================================================

def radial_action_complex(
    omega,
    m,
    r_minus,
    r_B=R_B,
    n=COMPLEX_RADIAL_N,
):
    """
    Complex continuation of

        I = integral p(r) dr

    along the straight contour from complex r_- to real r_B.
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

    for i, r in enumerate(
        r_path
    ):

        if i == 0:

            p_values[i] = 0.0j

            continue

        p = solve_p_complex(
            omega,
            m,
            r,
            p_initial=p_previous,
        )

        if p is None:

            raise FloatingPointError(
                "Complex p continuation failed."
            )

        # H is even in p. Choose the branch
        # continuously connected to the previous point.
        if (
            abs(
                p
                - p_previous
            )
            >
            abs(
                -p
                - p_previous
            )
        ):

            p = -p

        p_values[i] = p

        p_previous = p

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
    Res(omega) =
        R(omega)
        exp(2 i integral_{r_-}^{r_B} p dr)
        - 1
    """

    omega = complex(
        omega
    )

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

    except Exception:

        return None

    if not (
        np.isfinite(R.real)
        and np.isfinite(R.imag)
    ):

        return None

    # ------------------------------------------------------------------
    # Real axis
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

        except Exception:

            return None

    # ------------------------------------------------------------------
    # Complex omega
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

        except Exception:

            return None

    result = (
        R
        * np.exp(
            2.0j
            * action
        )
        - 1.0
    )

    if not (
        np.isfinite(result.real)
        and np.isfinite(result.imag)
    ):

        return None

    return result


# ============================================================================
# REAL-AXIS RESONANCE SEARCH
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
    1. Dense real-axis scan.
    2. Find local minima of |Res|.
    3. Refine with bounded scalar minimization.

    No 2-D complex grid.
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
    # Dense real scan
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

            value = abs(z)

            if np.isfinite(
                value
            ):

                abs_res[i] = value

        except Exception:

            pass

    finite = np.isfinite(
        abs_res
    )

    if not np.any(
        finite
    ):

        print(
            "DEBUG: no finite values "
            "were obtained from real-axis scan."
        )

        return []

    # ------------------------------------------------------------------
    # Local minima
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

    if not candidates:

        finite_indices = (
            np.flatnonzero(
                finite
            )
        )

        best = finite_indices[
            np.argmin(
                abs_res[
                    finite
                ]
            )
        ]

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
            f"{f_grid[best]:.10f} Hz"
        )

        print(
            f"DEBUG: best |Res| = "
            f"{abs_res[best]:.8e}"
        )

        return []

    candidates = sorted(
        candidates,
        key=lambda i:
            abs_res[i],
    )[:top_n]

    df = (
        f_grid[1]
        - f_grid[0]
    )

    resonances = []

    # ------------------------------------------------------------------
    # 1-D refinement
    # ------------------------------------------------------------------

    for idx in candidates:

        f0 = float(
            f_grid[idx]
        )

        half = max(
            local_half_width,
            3.0 * df,
        )

        lo = max(
            f_min,
            f0 - half,
        )

        hi = min(
            f_max,
            f0 + half,
        )

        def objective(f):

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

                value = abs(z)

                if not np.isfinite(
                    value
                ):

                    return 1.0e100

                return float(value)

            except Exception:

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
# LOCAL COMPLEX MINIMIZATION
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

    Objective:

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

    def objective(x):

        f_re = float(
            x[0]
        )

        f_im = float(
            x[1]
        )

        if not (
            f0
            - 4.0 * re_half_width
            <= f_re
            <= f0
            + 4.0 * re_half_width
        ):

            return 1.0e100

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

            value = abs(z)

            if not np.isfinite(
                value
            ):

                return 1.0e100

            return float(
                np.log1p(value)
            )

        except Exception:

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

    if not (
        f0
        - 4.0 * re_half_width
        <= f_re
        <= f0
        + 4.0 * re_half_width
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
# TRUE COMPLEX ROOT SOLVE
# ============================================================================

def solve_complex_resonance_root(
    f_re0,
    f_im0,
    resonance_factor,
    m,
):
    """
    TRUE complex resonance solve.

    Directly solves:

        Re[Res] = 0
        Im[Res] = 0

    starting from the local minimum obtained from
    refine_complex_resonance().
    """

    def equations(x):

        f_re = float(
            x[0]
        )

        f_im = float(
            x[1]
        )

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

                return np.array([
                    1.0e6,
                    1.0e6,
                ])

            if not (
                np.isfinite(
                    z.real
                )
                and np.isfinite(
                    z.imag
                )
            ):

                return np.array([
                    1.0e6,
                    1.0e6,
                ])

            return np.array([
                z.real,
                z.imag,
            ])

        except Exception:

            return np.array([
                1.0e6,
                1.0e6,
            ])

    result = root(
        equations,
        np.array([
            f_re0,
            f_im0,
        ]),
        method="hybr",
        options={
            "xtol": 1.0e-10,
            "maxfev": 1000,
        },
    )

    f_re = float(
        result.x[0]
    )

    f_im = float(
        result.x[1]
    )

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

    except Exception:

        z = None

    if z is None:

        res_real = np.nan
        res_imag = np.nan
        abs_res = np.inf

    else:

        res_real = float(
            z.real
        )

        res_imag = float(
            z.imag
        )

        abs_res = float(
            abs(z)
        )

    return {
        "f_re": f_re,
        "f_im": f_im,

        "omega_re": float(
            omega.real
        ),

        "omega_im": float(
            omega.imag
        ),

        "res_real": res_real,
        "res_imag": res_imag,
        "abs_res": abs_res,

        "success": bool(
            result.success
        ),

        "message": str(
            result.message
        ),

        "nfev": int(
            result.nfev
        ),
    }


# ============================================================================
# TURNING POINT DIAGNOSTIC
# ============================================================================

def print_turning_point_diagnostic(
    m,
    f,
    r_sp,
):
    """
    Print real turning points and explicitly select
    the outer / last one.
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
        "Using LAST / OUTER turning point:"
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

    print(
        "        -> TRUE complex root solve"
    )

    print()

    print(
        "NO coarse 2-D complex grid."
    )

    print()

    # ------------------------------------------------------------------
    # Parameters
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
    # Turning point diagnostic
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
    # Resonance probe
    # ------------------------------------------------------------------

    try:

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

    except Exception as exc:

        print(
            "Probe resonance factor FAILED:"
        )

        print(
            f"    {type(exc).__name__}: {exc}"
        )

        return

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
    # Real-axis scan
    # ------------------------------------------------------------------

    print(
        f"Running dense real-axis scan "
        f"({N_COARSE} points)..."
    )

    resonances = (
        find_resonances(
            m=m,
            f_min=f_min,
            f_max=f_max,
            resonance_factor=resonance_factor,
            n_coarse=N_COARSE,
            top_n=TOP_N,
            local_half_width=LOCAL_HALF_WIDTH,
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
    # Complex refinement + true root
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

        # ==============================================================
        # STEP 1 — Nelder-Mead
        # ==============================================================

        refined = (
            refine_complex_resonance(
                f0=f0,
                im0=-0.01,
                resonance_factor=resonance_factor,
                m=m,
                re_half_width=LOCAL_HALF_WIDTH,
                im_min=-1.5,
                im_max=0.0,
            )
        )

        if refined is None:

            print(
                "    complex refinement FAILED"
            )

            continue

        print(
            "    Nelder-Mead:"
        )

        print(
            f"        f_re = "
            f"{refined['f_re']:.12f} Hz"
        )

        print(
            f"        f_im = "
            f"{refined['f_im']:.12f} Hz"
        )

        print(
            f"        omega_re = "
            f"{refined['omega_re']:.12f} rad/s"
        )

        print(
            f"        omega_im = "
            f"{refined['omega_im']:.12f} rad/s"
        )

        print(
            f"        |Res| = "
            f"{refined['abs_res']:.12e}"
        )

        print(
            f"        success = "
            f"{refined['success']}"
        )

        print()

        # ==============================================================
        # STEP 2 — TRUE COMPLEX ROOT
        # ==============================================================

        print(
            "    True complex root solve:"
        )

        root_result = (
            solve_complex_resonance_root(
                f_re0=refined["f_re"],
                f_im0=refined["f_im"],
                resonance_factor=resonance_factor,
                m=m,
            )
        )

        print(
            f"        success = "
            f"{root_result['success']}"
        )

        print(
            f"        f_re = "
            f"{root_result['f_re']:.12f} Hz"
        )

        print(
            f"        f_im = "
            f"{root_result['f_im']:.12f} Hz"
        )

        print(
            f"        omega_re = "
            f"{root_result['omega_re']:.12f} rad/s"
        )

        print(
            f"        omega_im = "
            f"{root_result['omega_im']:.12f} rad/s"
        )

        print(
            f"        Re(Res) = "
            f"{root_result['res_real']:+.12e}"
        )

        print(
            f"        Im(Res) = "
            f"{root_result['res_imag']:+.12e}"
        )

        print(
            f"        |Res| = "
            f"{root_result['abs_res']:.12e}"
        )

        print(
            f"        nfev = "
            f"{root_result['nfev']}"
        )

        print(
            f"        message = "
            f"{root_result['message']}"
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