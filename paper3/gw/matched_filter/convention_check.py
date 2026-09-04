"""
convention_check.py -- последната отворена точка във веригата Lambda <-> A_4.

Проверява дали преводът на публикуваната граница за A_4 в граница за Lambda
използва същата (1+z) конвенция като LVK.

Формулата в waveform.py (изведена независимо, съвпада ред по ред):

    delta_Psi(f) = -(4 pi^3 Lambda K(z) / c^3) f^3        [f = НАБЛЮДАВАНА]
    K(z) = int_0^z (1+z')^2 / H(z') dz'                   [секунди]
    Lambda = hbar^2 c^3 A_4                               [m^3/s], A_4 в J^-2

LVK/MYW дефинират разстоянието
    D_alpha = (1+z)^(1-alpha) / H0 * int_0^z (1+z')^(alpha-2) / E(z') dz'
което за alpha=4 дава D_4 = c K(z) / (1+z)^3.

ВАЖНО: (1+z)^-3 е част от ТЯХНАТА дефиниция на D_4, не липсващ множител в
K(z). Въпросът е дали в техния израз за delta_Psi има компенсиращ (1+z)^3.

КАК СЕ ПОЛЗВА
-------------
1. Отвори arXiv:2603.19020 (GWTC-4.0 TGR II) или arXiv:2112.06861 (GWTC-3),
   раздела за модифицирана дисперсия.
2. Препиши техния израз за delta_Psi(f) в dpsi_lvk() по-долу. НЕ по памет.
3. python convention_check.py

Скриптът намира степента n в (1+z)^n, с която двете формули се разминават.
n = 0 -> конвенциите съвпадат, преводът на границата е верен.
n != 0 -> границата се коригира с (1+z_typ)^n.
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit

# --- константи (SI) ---
h = 6.62607015e-34
hbar = h / (2 * np.pi)
c = 2.99792458e8
eV = 1.602176634e-19
MPC = 3.0857e22

# --- космология (както в waveform.py) ---
H0, OM, OL = 67.4, 0.315, 0.685


def H_of_z(z):
    return H0 * np.sqrt(OM * (1 + z) ** 3 + OL) * 1e3 / MPC   # [1/s]


def K_of_z(z):
    """K(z) = int (1+z')^2 / H(z') dz'  [секунди] -- същата като в кода."""
    return quad(lambda zp: (1 + zp) ** 2 / H_of_z(zp), 0, z)[0]


def D_alpha(z, alpha=4):
    """LVK дефиниция. За alpha=4 се свежда до c*K(z)/(1+z)^3."""
    I = quad(lambda zp: (1 + zp) ** (alpha - 2) / H_of_z(zp), 0, z)[0]
    return c * (1 + z) ** (1 - alpha) * I


def lambda_A(A4_J):
    """lambda_A = h c |A_alpha|^(1/(alpha-2)); за alpha=4: h c sqrt(A_4)."""
    return h * c * np.sqrt(abs(A4_J))


# ----------------------------------------------------------------------
# ФОРМУЛА 1 -- твоята (waveform.py), изведена независимо
# ----------------------------------------------------------------------

def dpsi_kretski(f, A4_J, z):
    Lam = hbar ** 2 * c ** 3 * A4_J                 # [m^3/s]
    return -(4 * np.pi ** 3 * Lam * K_of_z(z) / c ** 3) * f ** 3


# ----------------------------------------------------------------------
# ФОРМУЛА 2 -- ТУК ПРЕПИШИ ИЗРАЗА ОТ СТАТИЯТА
# ----------------------------------------------------------------------

def dpsi_lvk(f, A4_J, z, alpha=4):
    """GWTC-4.0 (arXiv:2603.19020), paperII__tests_of_propagation.tex, ред 25:

        delta_Psi(f) = -pi D_alpha h^(alpha-2) (1+z)^(alpha-1) / c
                       * A_alpha f^(alpha-1)

    с D_alpha = c(1+z)^(1-alpha)/H0 * int (1+z')^(alpha-2)/E(z') dz'  (ред 29)
    """
    D = D_alpha(z, alpha)
    return -(np.pi * D * h ** (alpha - 2) * (1 + z) ** (alpha - 1) / c) \
        * A4_J * f ** (alpha - 1)


# ----------------------------------------------------------------------

def compare(A4_J=1e39, f=250.0, zs=(0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2)):
    print(f"A_4 = {A4_J:.3e} J^-2 = {A4_J*eV**2:.3e} eV^-2")
    print(f"Lambda = {hbar**2*c**3*A4_J:.3e} m^3/s   f = {f:.0f} Hz\n")
    print(f"{'z':>6s} {'K(z) [s]':>12s} {'D_4 [m]':>12s} "
          f"{'dPsi твоя':>13s} {'dPsi LVK':>13s} {'отношение':>11s}")
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

    # разминаване от вида C * (1+z)^n ?
    def model(z, logC, n):
        return logC + n * np.log(1 + z)
    try:
        (logC, n), _ = curve_fit(model, zs_a, np.log(ra))
        pred = np.exp(model(zs_a, logC, n))
        resid = np.max(np.abs(pred / ra - 1))
        print(f"\nразминаване ~ {np.exp(logC):.4f} * (1+z)^{n:+.3f}   "
              f"(макс. остатък {resid*100:.2f}%)")
        if resid > 0.02:
            print("  разминаването НЕ е чиста степен на (1+z) -- "
                  "провери транскрипцията, разликата е структурна")
        elif abs(n) < 0.05 and abs(np.exp(logC) - 1) < 0.02:
            print("  ->  КОНВЕНЦИИТЕ СЪВПАДАТ. Преводът на границата е верен.")
        elif abs(n) < 0.05:
            print(f"  ->  само постоянен множител {np.exp(logC):.4f}; "
                  f"границата за Lambda се дели на него")
        else:
            print(f"  ->  зависи от z: границата се коригира с "
                  f"(1+z_typ)^{n:+.2f}")
            for zt in (0.3, 0.5):
                k = np.exp(logC) * (1 + zt) ** n
                print(f"      при z_typ={zt}: фактор {k:.3f}  ->  "
                      f"Lambda_max {7.2e-3/k:.2e} m^3/s = "
                      f"{7.2e-3/k/c:.2e} m^2")
    except Exception as e:
        print(f"\nфитът не мина: {e}")

    print("\nконтролна проверка на разстоянието:")
    for z in (0.1, 0.5):
        lhs, rhs = D_alpha(z), c * K_of_z(z) / (1 + z) ** 3
        print(f"  z={z}: D_4={lhs:.6e}  c*K/(1+z)^3={rhs:.6e}  "
              f"отн.разлика {abs(lhs/rhs-1):.2e}")


if __name__ == "__main__":
    compare()
