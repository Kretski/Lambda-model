"""
convention_check.py -- validation of lambda_phase_correction against GWTC-4.0.

STATUS: closed, September 2026.

The phase formula in waveform.py has been verified against the published
GWTC-4.0 modified-dispersion expression (arXiv:2603.19020, source file
paperII__tests_of_propagation.tex, line 25):

    delta_Psi(f) = -pi D_alpha h^(alpha-2) (1+z)^(alpha-1) / c
                   * A_alpha f^(alpha-1)

    D_alpha = c (1+z)^(1-alpha) / H0
              * int_0^z (1+z')^(alpha-2) / E(z') dz'         (line 29)

Result: ratio 1.0000 at every redshift tested, fitted (1+z) exponent 0.000.
There is no residual redshift dependence, so the conversion of the published
A_4 bound into a bound on Lambda needs no correction factor.

Correspondence
--------------
    Lambda = hbar^2 c^3 A_4                  [m^3/s], with A_4 in J^-2
    Lambda_PaperII [m^2] = Lambda [m^3/s] / c

The second line follows because the Paper II Hamiltonian
H = 1/2 [ g^{mu nu} k_mu k_nu + Lambda (k_loc^ZAMO)^4 ] reduces in the flat
limit to omega^2 = c^2 k^2 (1 + Lambda k^2), whose k^2 coefficient carries
units of length squared.

Source of the bound
-------------------
GWTC-4.0, 83 events (43 from GWTC-3.0 plus 40 new from O4a), FAR <= 1e-3/yr,
BBH only -- BNS and NSBH are excluded from the MDR analysis.

    A_4 in [-620, +190] eV^-2       (90% credible, two-sided; Q_GR = 82.5%)
    ->  Lambda in [-7.24e-3, +2.22e-3] m^3/s
                 = [-2.41e-11, +7.40e-12] m^2

If the model requires Lambda > 0, as assumed in Paper II, only the positive
arm applies: Lambda <~ 7.40e-12 m^2.

CAVEAT -- SECTORS DIFFER
------------------------
This is a bound in the GRAVITON sector. The Lambda of Paper II is applied to
photon orbits (photon ring, black-hole shadow, chromaticity at radio
frequencies). Lorentz-violating dispersion in the photon and graviton sectors
are independent parameters; identifying the two requires a universality
assumption that the model does not postulate. The photon-sector bound is
Lambda <~ 1e-53 m^2 from Fermi-LAT gamma-ray-burst timing (Paper I), some 41
orders of magnitude tighter than the graviton-sector bound above. The gap is
accounted for by the E^2 D scaling of the quartic term: GeV photons versus
~1e-12 eV gravitons.

Prior art
---------
The flat-space chain omega^2 = k^2 + alpha k^4 -> A_4 -> GWTC-4.0 bound was
published independently in arXiv:2607.17431 (Araujo Filho et al., July 2026,
"Gravitational wave propagation in Horava-Lifshitz gravity"). That work does
not treat curved backgrounds -- no Kerr, photon ring, shadow, superradiance,
QNM or ZAMO content -- so the Paper II strong-field sector is unaffected.
Note also that alpha = 4 is identified in the GWTC-4.0 text as the case
corresponding to Horava-Lifshitz and extra-dimensional theories.

Usage
-----
    python convention_check.py

Re-run this after any change to lambda_phase_correction in waveform.py.
A ratio other than 1.0000 means the two formulas have drifted apart.
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit

# --- constants (SI) ---
h = 6.62607015e-34
hbar = h / (2 * np.pi)
c = 2.99792458e8
eV = 1.602176634e-19
MPC = 3.0857e22

# --- cosmology (same values as waveform.py) ---
H0, OM, OL = 67.4, 0.315, 0.685

# --- published GWTC-4.0 bound on A_4, in eV^-2 (90% credible, two-sided) ---
A4_BOUND_EV = (-620.0, 190.0)


def H_of_z(z):
    """Hubble rate in 1/s."""
    return H0 * np.sqrt(OM * (1 + z) ** 3 + OL) * 1e3 / MPC


def K_of_z(z):
    """K(z) = int_0^z (1+z')^2 / H(z') dz'  [seconds].

    This is the same integral as cosmological_K_factor() in waveform.py.
    """
    return quad(lambda zp: (1 + zp) ** 2 / H_of_z(zp), 0, z)[0]


def D_alpha(z, alpha=4):
    """LVK distance measure. For alpha=4 this equals c K(z) / (1+z)^3."""
    I = quad(lambda zp: (1 + zp) ** (alpha - 2) / H_of_z(zp), 0, z)[0]
    return c * (1 + z) ** (1 - alpha) * I


# ----------------------------------------------------------------------
# FORMULA 1 -- waveform.py, derived independently from the Paper II
#              Hamiltonian via group velocity and group delay
# ----------------------------------------------------------------------

def dpsi_kretski(f, A4_J, z):
    """delta_Psi(f) = -(4 pi^3 Lambda K(z) / c^3) f^3, with f observed."""
    Lam = hbar ** 2 * c ** 3 * A4_J                      # [m^3/s]
    return -(4 * np.pi ** 3 * Lam * K_of_z(z) / c ** 3) * f ** 3


# ----------------------------------------------------------------------
# FORMULA 2 -- GWTC-4.0, transcribed verbatim from the arXiv TeX source
#              (arXiv:2603.19020, paperII__tests_of_propagation.tex:25)
# ----------------------------------------------------------------------

def dpsi_lvk(f, A4_J, z, alpha=4):
    """delta_Psi(f) = -pi D_alpha h^(a-2) (1+z)^(a-1) / c * A_alpha f^(a-1)."""
    D = D_alpha(z, alpha)
    return -(np.pi * D * h ** (alpha - 2) * (1 + z) ** (alpha - 1) / c) \
        * A4_J * f ** (alpha - 1)


# ----------------------------------------------------------------------

def compare(A4_J=1e39, f=250.0, zs=(0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2)):
    """Compare the two formulas across redshift and report any mismatch."""
    print(f"A_4 = {A4_J:.3e} J^-2 = {A4_J*eV**2:.3e} eV^-2")
    print(f"Lambda = {hbar**2*c**3*A4_J:.3e} m^3/s   f = {f:.0f} Hz\n")
    print(f"{'z':>6s} {'K(z) [s]':>12s} {'D_4 [m]':>12s} "
          f"{'dPsi ours':>13s} {'dPsi LVK':>13s} {'ratio':>11s}")
    rows = []
    for z in zs:
        a = dpsi_kretski(f, A4_J, z)
        b = dpsi_lvk(f, A4_J, z)
        r = a / b if b != 0 else np.nan
        rows.append((z, r))
        print(f"{z:6.2f} {K_of_z(z):12.4e} {D_alpha(z):12.4e} "
              f"{a:13.4e} {b:13.4e} {r:11.4f}")

    zs_a = np.array([r[0] for r in rows])
    ra = np.array([abs(r[1]) for r in rows])

    # is the mismatch of the form C * (1+z)^n ?
    def model(z, logC, n):
        return logC + n * np.log(1 + z)

    try:
        (logC, n), _ = curve_fit(model, zs_a, np.log(ra))
        C = np.exp(logC)
        resid = np.max(np.abs(np.exp(model(zs_a, logC, n)) / ra - 1))
        print(f"\nmismatch ~ {C:.4f} * (1+z)^{n:+.3f}   "
              f"(max residual {resid*100:.2f}%)")
        if resid > 0.02:
            print("  mismatch is NOT a clean power of (1+z) -- the difference "
                  "is structural, check the transcription")
        elif abs(n) < 0.05 and abs(C - 1) < 0.02:
            print("  ->  CONVENTIONS AGREE. The bound conversion is correct.")
        elif abs(n) < 0.05:
            print(f"  ->  constant factor {C:.4f} only; divide the Lambda "
                  f"bound by it")
        else:
            print(f"  ->  redshift dependent: correct the bound by "
                  f"(1+z_typ)^{n:+.2f}")
            for zt in (0.3, 0.5):
                k = C * (1 + zt) ** n
                lo = hbar ** 2 * c ** 3 * A4_BOUND_EV[0] / eV ** 2 / k
                print(f"      at z_typ={zt}: factor {k:.3f} -> "
                      f"Lambda_min {lo:.2e} m^3/s = {lo/c:.2e} m^2")
    except Exception as e:
        print(f"\nfit failed: {e}")

    print("\ndistance cross-check (D_4 vs c K(z)/(1+z)^3):")
    for z in (0.1, 0.5):
        lhs, rhs = D_alpha(z), c * K_of_z(z) / (1 + z) ** 3
        print(f"  z={z}: D_4={lhs:.6e}  c*K/(1+z)^3={rhs:.6e}  "
              f"rel. diff {abs(lhs/rhs-1):.2e}")

    print("\npublished bound in each parametrisation:")
    for A_eV, lbl in zip(A4_BOUND_EV, ("lower", "upper")):
        A_J = A_eV / eV ** 2
        Lam = hbar ** 2 * c ** 3 * A_J
        print(f"  A_4 = {A_eV:+7.1f} eV^-2  ->  Lambda = {Lam:+.3e} m^3/s "
              f"= {Lam/c:+.3e} m^2   ({lbl})")
    print("  graviton sector; see the sector caveat in the module docstring")


if __name__ == "__main__":
    compare()
