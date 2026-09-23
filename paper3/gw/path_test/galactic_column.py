"""
galactic_column.py -- пътни променливи от геометрията, без сигнала.

Идея: за извънгалактически източник почти целият път минава през IGM,
който е приблизително еднакъв във всички посоки. Единственото нещо по
пътя, което зависи силно от посоката, е колонната плътност през
Млечния път -- тъмноматерийното хало и барионният диск.

Затова пътната променлива не е "ъгъл до Слънцето", а

    Sigma(l, b) = int_0^smax rho(r(s)) ds

изчислена по посоката на събитието. Динамичният ѝ обхват е ~1.5 порядъка
между галактичния център и полюсите -- достатъчно, за да се търси наклон.

Зависимости: numpy.
"""

from __future__ import annotations
import numpy as np

# --- параметри на Млечния път (стандартни стойности) ---
R_SUN_KPC = 8.2            # разстояние Слънце -- галактичен център
R_S_NFW_KPC = 20.0         # мащабен радиус на NFW халото
RHO_SUN_GEV_CM3 = 0.4      # локална плътност на ТМ
S_MAX_KPC = 200.0          # вириален радиус, докъдето се интегрира

H_R_KPC = 2.6              # мащабна дължина на барионния диск
H_Z_KPC = 0.30             # мащабна височина на барионния диск


def _r_galactocentric(s, l_rad, b_rad, r_sun=R_SUN_KPC):
    """Галактоцентричен радиус на разстояние s по посока (l, b)."""
    return np.sqrt(r_sun**2 + s**2 - 2.0 * r_sun * s * np.cos(b_rad) * np.cos(l_rad))


def _rho_nfw(r, r_s=R_S_NFW_KPC, r_sun=R_SUN_KPC, rho_sun=RHO_SUN_GEV_CM3):
    """NFW профил, нормиран така, че rho(r_sun) = rho_sun."""
    def f(rr):
        rr = np.maximum(rr, 1e-6)
        return 1.0 / ((rr / r_s) * (1.0 + rr / r_s) ** 2)
    return rho_sun * f(r) / f(r_sun)


def _rho_disc(r_cyl, z, h_r=H_R_KPC, h_z=H_Z_KPC):
    """Двойно-експоненциален барионен диск (относителни единици)."""
    return np.exp(-r_cyl / h_r) * np.exp(-np.abs(z) / h_z)


def dm_column(l_deg, b_deg, s_max=S_MAX_KPC, n_steps=4000):
    """Колонна плътност на тъмната материя по посока (l, b).

    Връща в GeV/cm^2 (нормировката е без значение за наклона -- влиза
    в константата, щом работим с log10).
    """
    l = np.radians(np.atleast_1d(l_deg))
    b = np.radians(np.atleast_1d(b_deg))
    s = np.linspace(0.0, s_max, n_steps)
    # (n_dir, n_steps)
    r = _r_galactocentric(s[None, :], l[:, None], b[:, None])
    rho = _rho_nfw(r)
    kpc_cm = 3.0857e21
    col = np.trapezoid(rho, s, axis=1) * kpc_cm
    return col if col.size > 1 else float(col[0])


def baryon_column(l_deg, b_deg, s_max=60.0, n_steps=4000):
    """Барионна колонна плътност (относителни единици) -- втора пътна ос.

    Много по-силно зависи от галактичната ширина от ТМ колоната,
    затова е полезна като независима проверка.
    """
    l = np.radians(np.atleast_1d(l_deg))
    b = np.radians(np.atleast_1d(b_deg))
    s = np.linspace(0.0, s_max, n_steps)
    z = s[None, :] * np.sin(b[:, None])
    x = R_SUN_KPC - s[None, :] * np.cos(b[:, None]) * np.cos(l[:, None])
    y = -s[None, :] * np.cos(b[:, None]) * np.sin(l[:, None])
    r_cyl = np.hypot(x, y)
    col = np.trapezoid(_rho_disc(r_cyl, z), s, axis=1)
    return col if col.size > 1 else float(col[0])


def path_variable(l_deg, b_deg, kind="dm", log10=True):
    """Единна входна точка. kind in {'dm', 'baryon'}."""
    f = {"dm": dm_column, "baryon": baryon_column}[kind]
    v = f(l_deg, b_deg)
    return np.log10(v) if log10 else v


if __name__ == "__main__":
    print("Колонна плътност на ТМ по избрани посоки (log10 GeV/cm^2):")
    for nm, (l, b) in {
        "галактичен център (0,0)": (0.0, 0.0),
        "антицентър (180,0)": (180.0, 0.0),
        "северен полюс (0,+90)": (0.0, 90.0),
        "южен полюс (0,-90)": (0.0, -90.0),
        "(90,+30)": (90.0, 30.0),
    }.items():
        print(f"  {nm:<26s} log10 Sigma_DM = {np.log10(dm_column(l, b)):.3f}"
              f"   log10 Sigma_bar = {np.log10(baryon_column(l, b)):+.3f}")
