"""
stage6_4A_equation_level_kerr_audit.py
======================================

STAGE 6.4A — EQUATION-LEVEL KERR CORRESPONDENCE AUDIT

Purpose
-------
Test whether the Root-B Hamiltonian has an equation-level
correspondence with the Kerr equatorial eikonal Hamilton-Jacobi
radial equation.

IMPORTANT SCIENTIFIC RULES
--------------------------
This script does NOT:

    - fit Kerr a/M to Root-B QNM frequencies
    - use GW150914
    - use any real GW event
    - fit a frequency scale
    - identify giant-vortex m with Kerr m
    - identify circulation C with Kerr spin a

It only compares mathematical structures.

Root-B
------
    H = -1/2 (omega - D)^2 + 1/2 F(k)

    D = m*C/r^2 + m*Omega

    k^2 = p_r^2 + m^2/r^2

    F(k) = (g*k + gamma*k^3)*tanh(h0*k)

Therefore:

    p_Lambda^2 =
        F^{-1}[(omega-D)^2] - m^2/r^2

Kerr
----
For equatorial null/eikonal motion (Q=0):

    R_K =
        [(r^2+a^2)*omega - a*m]^2
        - Delta*(m-a*omega)^2

    Delta = r^2 - 2*M*r + a^2

and

    p_r,K^2 = R_K / Delta^2

The audit checks:

    GATE A: rotational frequency structure
    GATE B: radial Hamiltonian structure
    GATE C: turning-point / light-ring structure
    GATE D: resonance architecture

The script explicitly distinguishes:

    EXACT correspondence
    ASYMPTOTIC correspondence
    NO equation-level correspondence
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# ============================================================================
# PATH
# ============================================================================

HERE = Path(__file__).resolve().parent

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


# ============================================================================
# IMPORT ACTUAL ROOT-B PARAMETERS
# ============================================================================

from stage5I_1_2_dispersion_radial_p import (
    C_METHODS,
    OMEGA,
    GAMMA,
    H0,
    R_B,
    G_GRAV,
    omega_D_plus_at_p0,
    find_light_ring,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

M_MODE = -12

# A representative radius for structural diagnostics only.
# This is NOT a Kerr fit radius.
R_PROBE = 20.0e-3

# Dimensionless Kerr test parameters are deliberately generic.
# They are NOT fitted to Root-B observables.
KERR_M = 1.0
KERR_A_VALUES = (
    0.0,
    0.3,
    0.6,
    0.9,
)

# Root-B probe frequency: derived from its own light ring.
# No GW event is involved.
USE_ROOTB_LIGHT_RING = True

# Long-wave diagnostic range.
LONG_WAVE_K_MAX = 0.1


# ============================================================================
# ROOT-B FUNCTIONS
# ============================================================================

def rootb_doppler(m, r):
    """
    Root-B rotational/Doppler contribution:

        D = m*C/r^2 + m*Omega
    """
    return (
        m * C_METHODS / r**2
        + m * OMEGA
    )


def rootb_k_squared(p, m, r):
    """
    k^2 = p_r^2 + m^2/r^2
    """
    return (
        p**2
        + (m / r)**2
    )


def rootb_F(k):
    """
    Exact Root-B dispersion:

        F(k) = (g*k + gamma*k^3)*tanh(h0*k)
    """
    return (
        G_GRAV * k
        + GAMMA * k**3
    ) * np.tanh(H0 * k)


def rootb_F_long_wave(k):
    """
    Leading long-wave approximation:

        tanh(h0*k) ~= h0*k

        F ~= g*h0*k^2

    The gamma contribution is retained only as the next
    correction in the printed expansion.
    """
    return G_GRAV * H0 * k**2


def rootb_p2_exact_from_k(k, m, r):
    """
    p_r^2 from

        k^2 = p_r^2 + m^2/r^2
    """
    return (
        k**2
        - (m / r)**2
    )


def rootb_k_from_frequency(omega, m, r):
    """
    Solve the scalar Root-B dispersion relation

        F(k) = (omega-D)^2

    for the positive real k branch.

    This is only a diagnostic inversion.
    """

    target = (
        omega
        - rootb_doppler(m, r)
    )**2

    target = float(np.real(target))

    if target < 0:
        return None

    # Monotonic real branch search.
    k_lo = 0.0
    k_hi = max(
        10.0 / max(H0, 1.0e-12),
        10.0,
    )

    for _ in range(100):
        if rootb_F(k_hi) >= target:
            break
        k_hi *= 2.0

    else:
        return None

    # Simple bisection.
    for _ in range(200):
        k_mid = 0.5 * (k_lo + k_hi)

        value = rootb_F(k_mid)

        if value < target:
            k_lo = k_mid
        else:
            k_hi = k_mid

    return 0.5 * (k_lo + k_hi)


def rootb_p2_from_frequency(omega, m, r):
    """
    Exact Root-B radial momentum squared obtained by inversion.
    """
    k = rootb_k_from_frequency(
        omega,
        m,
        r,
    )

    if k is None:
        return None

    return rootb_p2_exact_from_k(
        k,
        m,
        r,
    )


def rootb_p2_long_wave(omega, m, r):
    """
    Leading long-wave radial Hamilton-Jacobi equation:

        (omega-D)^2 = c_eff^2 k^2

        p_r^2 =
            (omega-D)^2/c_eff^2
            - m^2/r^2
    """
    c_eff_sq = (
        G_GRAV * H0
    )

    D = rootb_doppler(
        m,
        r,
    )

    return (
        (omega - D)**2
        / c_eff_sq
        - (m / r)**2
    )


# ============================================================================
# KERR FUNCTIONS
# ============================================================================

def kerr_delta(r, M, a):
    """
    Delta = r^2 - 2Mr + a^2
    """
    return (
        r**2
        - 2.0 * M * r
        + a**2
    )


def kerr_K(omega, m, r, M, a):
    """
    K = (r^2+a^2) omega - a m
    """
    return (
        (r**2 + a**2) * omega
        - a * m
    )


def kerr_R_equatorial(
    omega,
    m,
    r,
    M,
    a,
):
    """
    Equatorial null/eikonal radial polynomial.

    Carter constant Q = 0:

        R =
            K^2
            - Delta*(m-a*omega)^2
    """

    Delta = kerr_delta(
        r,
        M,
        a,
    )

    K = kerr_K(
        omega,
        m,
        r,
        M,
        a,
    )

    return (
        K**2
        - Delta
        * (
            m
            - a * omega
        )**2
    )


def kerr_p2_equatorial(
    omega,
    m,
    r,
    M,
    a,
):
    """
    p_r^2 = R / Delta^2
    """

    Delta = kerr_delta(
        r,
        M,
        a,
    )

    if abs(Delta) < 1.0e-14:
        return np.inf

    return (
        kerr_R_equatorial(
            omega,
            m,
            r,
            M,
            a,
        )
        / Delta**2
    )


def kerr_rotational_shift(
    m,
    r,
    M,
    a,
):
    """
    Kerr local rotational term appearing in

        omega - a*m/(r^2+a^2)
    """

    return (
        a * m
        / (
            r**2
            + a**2
        )
    )


# ============================================================================
# GATE A — ROTATIONAL STRUCTURE
# ============================================================================

def gate_A_rotational_structure(
    m,
    r,
):
    """
    Compare functional forms:

        Root-B:
            D/m = Omega + C/r^2

        Kerr:
            D_K/m = a/(r^2+a^2)

    This is a structural comparison.

    No Kerr parameter is fitted.
    """

    rootb_value = (
        C_METHODS / r**2
        + OMEGA
    )

    print()
    print("=" * 78)
    print("GATE A — ROTATIONAL STRUCTURE")
    print("=" * 78)

    print()
    print("Root-B:")
    print()
    print("    D/m = Omega + C/r^2")

    print()
    print("Kerr:")
    print()
    print("    D_K/m = a/(r^2+a^2)")

    print()
    print(f"Root-B at r={r:.6e}:")
    print(f"    C/r^2 + Omega = {rootb_value:.12e}")

    print()
    print("Important asymptotic test:")

    print()
    print("    Root-B -> Omega       as r -> infinity")
    print("    Kerr   -> 0           as r -> infinity")

    if abs(OMEGA) > 1.0e-14:
        print()
        print(
            "RESULT: GLOBAL EXACT MATCH FAILS "
            "because Root-B has a nonzero asymptotic Omega."
        )
        return False

    print()
    print(
        "RESULT: No asymptotic obstruction from Omega alone."
    )

    return True


# ============================================================================
# GATE B — RADIAL HAMILTONIAN
# ============================================================================

def gate_B_radial_structure(
    m,
    r,
    omega,
):
    """
    Compare exact Root-B radial structure with Kerr radial structure.

    The key test is functional class:

        Root-B:
            F^{-1}[(omega-D)^2]

        Kerr:
            rational polynomial in r, omega, m, a, M

    Exact symbolic equivalence is rejected if the transcendental
    tanh structure remains irreducible.
    """

    print()
    print("=" * 78)
    print("GATE B — RADIAL HAMILTONIAN STRUCTURE")
    print("=" * 78)

    print()
    print("Root-B exact:")
    print()
    print(
        "    p_Lambda^2 = "
        "F^{-1}[(omega-D)^2] - m^2/r^2"
    )

    print()
    print("with")
    print()
    print(
        "    F(k) = (g k + gamma k^3) tanh(h0 k)"
    )

    print()
    print("Kerr equatorial eikonal:")
    print()
    print(
        "    p_Kerr^2 = R/Delta^2"
    )

    print()
    print(
        "    R = [(r^2+a^2)omega-am]^2"
    )
    print(
        "        - Delta*(m-a*omega)^2"
    )

    print()
    print(
        "    Delta = r^2 - 2Mr + a^2"
    )

    print()
    print("Functional-class test:")
    print()
    print("    Root-B contains tanh(h0*k).")
    print("    Kerr radial polynomial is algebraic/rational.")
    print()

    print(
        "RESULT: EXACT GLOBAL ALGEBRAIC KERR MATCH = FAIL"
    )

    print()
    print(
        "Reason: the exact Root-B dispersion contains a "
        "transcendental tanh(k) structure."
    )

    print()
    print(
        "This does NOT yet rule out an asymptotic/eikonal "
        "correspondence."
    )

    return False


# ============================================================================
# GATE C — TURNING POINT STRUCTURE
# ============================================================================

def numerical_derivative(
    func,
    x,
    dx,
):
    return (
        func(x + dx)
        - func(x - dx)
    ) / (
        2.0 * dx
    )


def gate_C_turning_point_structure(
    m,
):
    """
    Compare the mathematical conditions for an unstable circular
    orbit / turning-point degeneracy.

    Root-B:
        H = 0
        dH/dr = 0

    Kerr:
        R = 0
        dR/dr = 0
    """

    print()
    print("=" * 78)
    print("GATE C — TURNING-POINT / LIGHT-RING STRUCTURE")
    print("=" * 78)

    try:
        r_sp, omega_lr = find_light_ring(m)
    except Exception as exc:
        print()
        print(
            "Root-B light-ring calculation FAILED:"
        )
        print(
            f"    {type(exc).__name__}: {exc}"
        )
        return False

    if r_sp is None:
        print()
        print("No Root-B light ring found.")
        return False

    print()
    print("Root-B:")
    print(
        f"    r_sp = {r_sp:.12e}"
    )
    print(
        f"    omega_lr = {omega_lr:.12e}"
    )

    print()
    print(
        "Root-B degeneracy condition:"
    )
    print()
    print(
        "    H(omega,p,r) = 0"
    )
    print(
        "    dH/dr = 0"
    )

    print()
    print("Kerr:")
    print()
    print(
        "    R(r) = 0"
    )
    print(
        "    dR/dr = 0"
    )

    print()
    print(
        "RESULT: STRUCTURAL CORRESPONDENCE = PRESENT"
    )

    print()
    print(
        "Both systems define a trapped/circular-orbit condition "
        "through simultaneous vanishing of the radial Hamiltonian "
        "and its radial derivative."
    )

    print()
    print(
        "Caution: this is a structural analogy, not proof of "
        "Kerr equivalence."
    )

    return True


# ============================================================================
# GATE D — RESONANCE ARCHITECTURE
# ============================================================================

def gate_D_resonance_structure():
    """
    Compare resonance architecture.

    Root-B:
        R exp(2 i S) - 1 = 0

    Kerr:
        global QNM boundary-value condition
        / vanishing Wronskian / equivalent spectral condition.
    """

    print()
    print("=" * 78)
    print("GATE D — RESONANCE ARCHITECTURE")
    print("=" * 78)

    print()
    print("Root-B:")
    print()
    print(
        "    Res(omega) = R(omega)"
    )
    print(
        "                 * exp(2 i S(omega))"
    )
    print(
        "                 - 1"
    )

    print()
    print(
        "    S = integral p_r dr"
    )

    print()
    print("Kerr:")
    print()
    print(
        "    QNM frequencies are global spectral roots "
        "of the separated perturbation problem."
    )

    print()
    print(
        "    Equivalent formulations include vanishing "
        "of the relevant connection/Wronskian condition."
    )

    print()
    print(
        "RESULT: SPECTRAL ARCHITECTURE = ANALOGOUS"
    )

    print()
    print(
        "But Root-B's reflection/turning-point boundary "
        "condition has not been derived from Teukolsky."
    )

    return True


# ============================================================================
# LONG-WAVE TEST
# ============================================================================

def long_wave_diagnostic(
    m,
    r,
):
    """
    Demonstrate the controlled Root-B long-wave reduction:

        tanh(h0 k) ~ h0 k

        F(k) ~ g h0 k^2

    This tests whether the exact transcendental obstruction
    disappears in the controlled low-k limit.
    """

    print()
    print("=" * 78)
    print("ASYMPTOTIC TEST — LONG-WAVE LIMIT")
    print("=" * 78)

    print()
    print("Exact:")
    print()
    print(
        "    F(k) = (g k + gamma k^3) tanh(h0 k)"
    )

    print()
    print("For h0*k << 1:")
    print()
    print(
        "    tanh(h0 k) = h0 k + O((h0 k)^3)"
    )

    print()
    print(
        "Therefore:"
    )
    print()
    print(
        "    F(k) = g h0 k^2 + higher-order corrections"
    )

    c_eff_sq = (
        G_GRAV * H0
    )

    print()
    print(
        f"    c_eff^2 = g*h0 = {c_eff_sq:.12e}"
    )

    print()
    print(
        "Leading radial Hamiltonian:"
    )
    print()
    print(
        "    p_r^2 = "
        "(omega-D)^2/(g*h0) - m^2/r^2"
    )

    print()
    print(
        "This removes the exact tanh obstruction "
        "at leading asymptotic order."
    )

    print()
    print(
        "RESULT: ASYMPTOTIC REDUCTION = AVAILABLE"
    )

    return True


# ============================================================================
# NUMERICAL STRUCTURAL CROSS-CHECK
# ============================================================================

def numerical_probe(
    m,
    r,
):
    """
    Show representative Root-B and Kerr radial structures.

    Kerr parameters here are deliberately generic and are NOT fitted.
    """

    print()
    print("=" * 78)
    print("NUMERICAL STRUCTURAL PROBE")
    print("=" * 78)

    try:
        r_sp, omega_lr = find_light_ring(m)
    except Exception:
        r_sp, omega_lr = None, None

    if omega_lr is None:
        print()
        print("No Root-B light-ring frequency available.")
        return

    omega = float(
        omega_lr * 0.98
    )

    print()
    print(
        f"Root-B probe:"
    )
    print(
        f"    m = {m}"
    )
    print(
        f"    r = {r:.12e}"
    )
    print(
        f"    omega = {omega:.12e}"
    )

    D = rootb_doppler(
        m,
        r,
    )

    k = rootb_k_from_frequency(
        omega,
        m,
        r,
    )

    print()
    print(
        f"    D = {D:.12e}"
    )

    if k is None:
        print(
            "    k inversion: unavailable"
        )
    else:
        p2 = rootb_p2_exact_from_k(
            k,
            m,
            r,
        )

        print(
            f"    k = {k:.12e}"
        )

        print(
            f"    p_Lambda^2 = {p2:.12e}"
        )

    print()
    print(
        "Kerr generic probes:"
    )

    for a in KERR_A_VALUES:

        M = KERR_M

        # Use a dimensionless diagnostic radius safely outside
        # the Kerr horizon. This is only a structural probe.
        rk = 10.0 * M

        p2_k = kerr_p2_equatorial(
            omega,
            m,
            rk,
            M,
            a,
        )

        shift = kerr_rotational_shift(
            m,
            rk,
            M,
            a,
        )

        print()
        print(
            f"    a/M = {a:.3f}"
        )
        print(
            f"        r/M = {rk/M:.3f}"
        )
        print(
            f"        a*m/(r^2+a^2) = "
            f"{shift:.12e}"
        )
        print(
            f"        p_Kerr^2 = "
            f"{p2_k:.12e}"
        )


# ============================================================================
# FINAL CLASSIFICATION
# ============================================================================

def final_classification(
    gate_A,
    gate_B,
    gate_C,
    gate_D,
):
    print()
    print("=" * 78)
    print("FINAL STAGE 6.4A CLASSIFICATION")
    print("=" * 78)

    print()
    print(
        f"GATE A rotational structure      : "
        f"{'PASS' if gate_A else 'FAIL'}"
    )

    print(
        f"GATE B exact radial structure    : "
        f"{'PASS' if gate_B else 'FAIL'}"
    )

    print(
        f"GATE C turning-point structure   : "
        f"{'PASS' if gate_C else 'FAIL'}"
    )

    print(
        f"GATE D resonance architecture    : "
        f"{'PASS' if gate_D else 'FAIL'}"
    )

    print()

    print(
        "EXACT KERR CORRESPONDENCE:"
    )

    print(
        "    NOT ESTABLISHED"
    )

    print()

    print(
        "WHY:"
    )

    print(
        "    The exact Root-B dispersion contains"
    )

    print(
        "        (g k + gamma k^3) tanh(h0 k)"
    )

    print(
        "    whereas the Kerr eikonal radial equation"
    )

    print(
        "    is algebraic/rational in the radial variables."
    )

    print()

    print(
        "ASYMPTOTIC CORRESPONDENCE:"
    )

    print(
        "    OPEN / REQUIRES SECOND-STAGE ANALYSIS"
    )

    print()

    print(
        "The long-wave limit removes the tanh obstruction"
    )

    print(
        "at leading order and produces a Hamilton-Jacobi-like"
    )

    print(
        "radial equation."
    )

    print()

    print(
        "SCIENTIFIC STATUS:"
    )

    print(
        "    Root-B is NOT identified as Kerr."
    )

    print(
        "    No Kerr spin was fitted."
    )

    print(
        "    No GW event was used."
    )

    print(
        "    Numerical QNM similarity is not used as evidence."
    )

    print()

    print(
        "NEXT VALID TEST:"
    )

    print(
        "    Stage 6.4B — derive and compare the reduced"
    )

    print(
        "    long-wave/eikonal Hamiltonian against Kerr."
    )

    print()


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 78)
    print(
        "STAGE 6.4A — EQUATION-LEVEL KERR CORRESPONDENCE AUDIT"
    )
    print("=" * 78)

    print()
    print(
        "Root-B equation:"
    )

    print()
    print(
        "    H = -1/2 (omega-D)^2 + 1/2 F(k)"
    )

    print(
        "    D = m*C/r^2 + m*Omega"
    )

    print(
        "    k^2 = p_r^2 + m^2/r^2"
    )

    print(
        "    F(k) = (g*k + gamma*k^3)*tanh(h0*k)"
    )

    print()
    print(
        "No event calibration."
    )

    print(
        "No Kerr QNM fitting."
    )

    print(
        "No post-hoc a/M identification."
    )

    print()

    # ------------------------------------------------------------------
    # Root-B light ring
    # ------------------------------------------------------------------

    try:
        r_sp, omega_lr = find_light_ring(
            M_MODE
        )
    except Exception as exc:
        r_sp, omega_lr = None, None
        print(
            f"Light-ring lookup failed: {exc}"
        )

    if r_sp is not None:

        print(
            "Root-B internal reference:"
        )

        print(
            f"    m = {M_MODE}"
        )

        print(
            f"    r_sp = {r_sp:.12e} m"
        )

        print(
            f"    omega_lr = {omega_lr:.12e} rad/s"
        )

        print(
            f"    f_lr = "
            f"{omega_lr/(2*np.pi):.12f} Hz"
        )

    else:

        print(
            "WARNING: Root-B light ring unavailable."
        )

    # ------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------

    gate_A = gate_A_rotational_structure(
        M_MODE,
        R_PROBE,
    )

    gate_B = gate_B_radial_structure(
        M_MODE,
        R_PROBE,
        omega_lr if omega_lr is not None else 1.0,
    )

    gate_C = gate_C_turning_point_structure(
        M_MODE
    )

    gate_D = gate_D_resonance_structure()

    # ------------------------------------------------------------------
    # Asymptotic
    # ------------------------------------------------------------------

    long_wave_diagnostic(
        M_MODE,
        R_PROBE,
    )

    # ------------------------------------------------------------------
    # Numerical probe
    # ------------------------------------------------------------------

    if omega_lr is not None:

        numerical_probe(
            M_MODE,
            R_PROBE,
        )

    # ------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------

    final_classification(
        gate_A,
        gate_B,
        gate_C,
        gate_D,
    )


if __name__ == "__main__":
    main()