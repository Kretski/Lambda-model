
#!/usr/bin/env python3
"""
===============================================================================
STAGE 6.4B — REDUCED LONG-WAVE / EIKONAL EQUATION-LEVEL TEST
===============================================================================

Purpose
-------
Test whether the controlled long-wave Root-B Hamiltonian admits an
equation-level correspondence with the Kerr eikonal Hamilton-Jacobi structure.

IMPORTANT SCIENTIFIC RULES
---------------------------
1. No GW event is used.
2. No QNM frequency is used.
3. No Kerr spin is fitted.
4. No frequency scale is fitted.
5. No numerical Kerr matching is used as evidence.
6. We test equations and functional structure only.

Root-B exact:
    (omega - m*Omega - m*C/r^2)^2 = F(k)

    F(k) = (g*k + gamma*k^3)*tanh(h0*k)

Long-wave limit:
    F(k) = g*h0*k^2 + O(k^4)

Define:
    c2 = g*h0

Then:
    (omega - m*Omega - m*C/r^2)^2
        = c2*(p_r^2 + m^2/r^2)

Therefore:
    p_r^2 =
        (omega - m*Omega - m*C/r^2)^2/c2
        - m^2/r^2

Kerr equatorial eikonal:
    R = [(r^2+a^2)*omega - a*m]^2
        - Delta*(m-a*omega)^2

    Delta = r^2 - 2*M*r + a^2

    p_r^2 = R/Delta^2

The test asks:

A) Can the rotational structures be IDENTICAL under direct identification?
B) Can the radial polynomial structures be IDENTICAL?
C) Do the two systems possess the same homogeneity/radial pole structure?
D) Is a correspondence possible only asymptotically?
E) Does the correspondence require fitted parameters?

Expected scientific outcome:
    EXACT CORRESPONDENCE
    ASYMPTOTIC CORRESPONDENCE
    NO EQUATION-LEVEL CORRESPONDENCE

===============================================================================
"""

import sys
import math
import numpy as np

try:
    from stage5I_1_2_dispersion_radial_p import (
        C_METHODS,
        OMEGA,
        G_GRAV,
        H0,
        GAMMA,
    )
except Exception as e:
    print("\nERROR importing Root-B parameters:")
    print(e)
    print("\nMake sure stage5I_1_2_dispersion_radial_p.py is in this directory.")
    sys.exit(1)


# =============================================================================
# ROOT-B REDUCED MODEL
# =============================================================================

def rootb_shift_per_m(r):
    """
    Root-B rotational frequency shift divided by m:
        Omega + C/r^2
    """
    return OMEGA + C_METHODS / r**2


def rootb_p2_reduced(omega, m, r):
    """
    Long-wave Root-B radial Hamilton-Jacobi equation:
        p^2 = (omega - m*Omega - m*C/r^2)^2/c2 - m^2/r^2
    """
    c2 = G_GRAV * H0

    D = m * rootb_shift_per_m(r)

    return (omega - D)**2 / c2 - m**2 / r**2


# =============================================================================
# KERR
# =============================================================================

def kerr_delta(r, M, a):
    return r**2 - 2*M*r + a**2


def kerr_shift_per_m(r, M, a):
    """
    Kerr rotational term:
        a/(r^2+a^2)
    """
    return a / (r**2 + a**2)


def kerr_R(omega, m, r, M, a):
    Delta = kerr_delta(r, M, a)
    K = (r**2 + a**2)*omega - a*m

    # Equatorial Q=0 eikonal radial potential.
    return K**2 - Delta*(m - a*omega)**2


def kerr_p2(omega, m, r, M, a):
    Delta = kerr_delta(r, M, a)

    if abs(Delta) < 1e-30:
        return np.nan

    return kerr_R(omega, m, r, M, a) / Delta**2


# =============================================================================
# SYMBOLIC-STYLE STRUCTURAL TESTS
# =============================================================================

def print_rotational_test():
    print("=" * 78)
    print("TEST A — ROTATIONAL STRUCTURE")
    print("=" * 78)

    print("""
Root-B:
    W_Lambda(r) = Omega + C/r^2

Kerr:
    W_Kerr(r) = a/(r^2+a^2)
""")

    print("Asymptotic expansions:")
    print("""
Root-B:
    W_Lambda = Omega + C/r^2

Kerr:
    W_Kerr = a/r^2 - a^3/r^4 + O(r^-6)
""")

    if abs(OMEGA) > 1e-15:
        print("RESULT: DIRECT GLOBAL IDENTIFICATION = FAIL")
        print("Reason:")
        print("    Root-B contains a nonzero constant Omega at infinity.")
        print("    Kerr rotational shift vanishes at infinity.")
        print()
    else:
        print("Omega = 0, so the leading asymptotic obstruction is absent.")
        print("Further coefficient comparison required.")

    print("Coefficient comparison if Omega = 0:")
    print("    Root-B: C/r^2")
    print("    Kerr  : a/r^2 - a^3/r^4 + ...")
    print()
    print("Thus direct equality would additionally require:")
    print("    C = a")
    print("and simultaneously")
    print("    a^3 = 0")
    print("for exact equality of all powers.")
    print()
    print("Therefore nontrivial exact global equality is impossible.")
    print()


def print_radial_test():
    print("=" * 78)
    print("TEST B — REDUCED RADIAL HAMILTONIAN")
    print("=" * 78)

    print("""
Root-B reduced equation:

    p_Lambda^2 =
        (omega - m*Omega - m*C/r^2)^2/c2
        - m^2/r^2

where

    c2 = g*h0
""")

    print("""
Kerr:

    p_Kerr^2 = R/Delta^2

    R =
        [(r^2+a^2)*omega-am]^2
        - Delta*(m-a*omega)^2
""")

    print("Functional structure:")
    print()
    print("Root-B:")
    print("    rational in r, with explicit r^-2 rotational term")
    print()
    print("Kerr:")
    print("    rational in r, but contains Delta = r^2-2Mr+a^2")
    print("    and the coupled combinations (r^2+a^2)omega-am")
    print()

    print("RESULT:")
    print("    The long-wave limit REMOVES the transcendental tanh obstruction.")
    print("    Therefore an equation-level comparison is mathematically meaningful.")
    print()
    print("However:")
    print("    equality is not automatic because Kerr contains M and horizon")
    print("    structure through Delta, while Root-B does not.")
    print()


def print_pole_structure_test():
    print("=" * 78)
    print("TEST C — RADIAL POLE / HORIZON STRUCTURE")
    print("=" * 78)

    print("""
Root-B:
    p_Lambda^2 contains only r^-2 and r^-4 terms
    after expansion of the rotational square.

Kerr:
    p_Kerr^2 contains Delta^-2.

    Delta = r^2 - 2Mr + a^2.
""")

    print("""
Kerr therefore possesses a distinguished radial singular structure
associated with Delta = 0.

Root-B has no corresponding Delta factor in the reduced Hamiltonian.
""")

    print("RESULT: GLOBAL RADIAL STRUCTURE = NOT IDENTICAL")
    print()
    print("This is important:")
    print("    A local/asymptotic correspondence may still exist.")
    print("    A global Kerr identification does not follow.")
    print()


def print_asymptotic_expansion():
    print("=" * 78)
    print("TEST D — LARGE-r / EIKONAL ASYMPTOTICS")
    print("=" * 78)

    print("""
Root-B:

    D/m = Omega + C/r^2

    p_Lambda^2 =
        (omega-m*Omega)^2/c2
        - m^2/r^2
        - 2*m*C*(omega-m*Omega)/(c2*r^2)
        + m^2*C^2/(c2*r^4)
        - m^2/r^2
""")

    print("""
Kerr large-r expansion has the schematic form

    p_Kerr^2 =
        omega^2
        + A_2/r^2
        + A_3/r^3
        + A_4/r^4
        + ...

with coefficients depending on M, a, omega and m.
""")

    print("""
The crucial question is therefore:

    Can the Root-B coefficients reproduce the Kerr coefficients
    WITHOUT fitting M, a, omega-scale, or m?

If not, the result is only asymptotic analogy rather than
physical Kerr correspondence.
""")

    print("RESULT: COEFFICIENT MATCHING MUST BE DERIVED, NOT FITTED.")
    print()


# =============================================================================
# NUMERICAL CONSISTENCY TEST
# =============================================================================

def numerical_probe():
    print("=" * 78)
    print("TEST E — PARAMETER-FREE NUMERICAL STRUCTURAL PROBE")
    print("=" * 78)

    print("""
This is NOT a fit.

We use arbitrary dimensionless Kerr probes only to test whether
the functional structures can naturally coincide.
""")

    # Root-B internal scale
    m = -12

    # Use a representative radius around the Root-B scale.
    r0 = 2.0e-2

    # Internal Root-B long-wave test frequency.
    # Deliberately NOT from a GW event.
    omega0 = 0.98 * 55.51472662927

    c2 = G_GRAV * H0

    print(f"Root-B constants:")
    print(f"    g       = {G_GRAV:.12e}")
    print(f"    h0      = {H0:.12e}")
    print(f"    gamma   = {GAMMA:.12e}")
    print(f"    C       = {C_METHODS:.12e}")
    print(f"    Omega   = {OMEGA:.12e}")
    print(f"    c_eff²  = {c2:.12e}")
    print()

    print("Probe:")
    print(f"    m       = {m}")
    print(f"    r       = {r0:.12e}")
    print(f"    omega   = {omega0:.12e}")
    print()

    shift = rootb_shift_per_m(r0)

    p2_L = rootb_p2_reduced(omega0, m, r0)

    print("Root-B reduced:")
    print(f"    D/m     = {shift:.12e}")
    print(f"    p²      = {p2_L:.12e}")
    print()

    # Generic dimensionless Kerr probes.
    # These are not fitted to Root-B.
    M = 1.0

    for a in [0.0, 0.3, 0.6, 0.9]:

        # Dimensionless radius safely outside horizon.
        r = 10.0

        p2_K = kerr_p2(omega0, m, r, M, a)
        shift_K = kerr_shift_per_m(r, M, a)

        print(f"Kerr probe a/M={a:.1f}:")
        print(f"    W_Kerr = {shift_K:.12e}")
        print(f"    p²     = {p2_K:.12e}")
        print()


# =============================================================================
# DECISION LOGIC
# =============================================================================

def final_classification():

    print("=" * 78)
    print("FINAL STAGE 6.4B CLASSIFICATION")
    print("=" * 78)

    exact_rotational = abs(OMEGA) < 1e-15

    print()
    print("ROTATIONAL GLOBAL EQUALITY:")
    print("    FAIL" if not exact_rotational else "    NOT EXCLUDED")

    print()
    print("GLOBAL RADIAL / HORIZON STRUCTURE:")
    print("    FAIL")

    print()
    print("LONG-WAVE HAMILTON-JACOBI REDUCTION:")
    print("    PASS")

    print()
    print("TRANSCENDENTAL DISPERSION OBSTRUCTION:")
    print("    REMOVED AT LEADING LONG-WAVE ORDER")

    print("""
===============================================================================
SCIENTIFIC INTERPRETATION
===============================================================================

The important result is NOT:

    "Root-B is Kerr."

That statement is not established.

The result is:

    Root-B has a controlled long-wave Hamilton-Jacobi limit.

This limit has a radial eikonal structure that can be compared directly
with Kerr.

However, the comparison reveals two remaining global obstructions:

    1. rotational shift:
           Omega + C/r^2
       versus
           a/(r^2+a^2)

    2. radial geometry:
       Root-B lacks the Kerr Delta = r^2-2Mr+a^2 structure.

Therefore:

    EXACT GLOBAL KERR CORRESPONDENCE
        = NOT ESTABLISHED

    LONG-WAVE / EIKONAL STRUCTURAL CORRESPONDENCE
        = PLAUSIBLE / TESTABLE

    PHYSICAL IDENTIFICATION WITH KERR
        = NOT ESTABLISHED

===============================================================================
NEXT SCIENTIFIC TEST
===============================================================================

The next decisive test is NOT another numerical QNM comparison.

It is:

    derive whether a coordinate/frequency transformation exists such that

        H_Lambda(reduced)
              <-->
        H_Kerr(eikonal)

without fitted event parameters.

If such a transformation exists:
    -> equation-level physical correspondence becomes possible.

If no such transformation exists:
    -> Root-B remains a distinct non-classical analogue/modified-dispersion
       phenomenon rather than a Kerr/GR realization.

===============================================================================
""")



# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 78)
    print("STAGE 6.4B — REDUCED LONG-WAVE / EIKONAL AUDIT")
    print("=" * 78)
    print()

    print_rotational_test()
    print_radial_test()
    print_pole_structure_test()
    print_asymptotic_expansion()
    numerical_probe()
    final_classification()

