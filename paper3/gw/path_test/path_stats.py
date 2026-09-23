"""
path_stats.py -- ядро на теста "зависи ли аномалията от пътя на вълната".

Проверява една хипотеза:
    A_i = a + b * x_i + sum_k c_k * u_ki + shum

    A_i  -- аномална статистика, извлечена САМО от сигнала (z-стойност)
    x_i  -- пътна променлива, извлечена САМО от геометрията (без сигнала)
    u_ki -- ковариати (log SNR, log площ на локализацията) -- контрол на бъркащи фактори

Хипотезата "средата по пътя влияе на сигнала" е b != 0.
Нулевата хипотеза е b = 0.

Тук НЯМА произволни прагове. Резултатът е b +- sigma_b, p-стойност от
пермутация, и горна граница при зададено ниво на доверие.

Зависимости: numpy, scipy.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy import stats


# ----------------------------------------------------------------------
# резултат
# ----------------------------------------------------------------------

@dataclass
class FitResult:
    slope: float                 # b -- наклон по пътната променлива
    slope_err: float             # sigma_b
    intercept: float
    coef: np.ndarray             # всички коефициенти [a, b, c_1..c_k]
    cov: np.ndarray              # ковариационна матрица
    names: list[str]
    chi2: float
    dof: int
    err_scale: float             # sqrt(chi2/dof), ако е приложено мащабиране
    n: int
    p_perm: float | None = None  # двустранна p от пермутация
    perm_slopes: np.ndarray | None = None
    spearman_r: float | None = None
    spearman_p: float | None = None
    extra: dict = field(default_factory=dict)

    def bound(self, cl: float = 0.90, two_sided: bool = False) -> float:
        """Горна граница за |b| при ниво cl.

        Ако наклонът е съвместим с нула, това е числото, което се публикува:
        "|b| < bound при cl% доверие".
        """
        if two_sided:
            k = stats.norm.ppf(0.5 + cl / 2.0)
        else:
            k = stats.norm.ppf(cl)
        return abs(self.slope) + k * self.slope_err

    def z(self) -> float:
        return self.slope / self.slope_err if self.slope_err > 0 else np.nan

    def summary(self) -> str:
        L = []
        L.append(f"n = {self.n} събития, dof = {self.dof}")
        for nm, c, s in zip(self.names, self.coef, np.sqrt(np.diag(self.cov))):
            L.append(f"  {nm:<24s} = {c:+.4f} +- {s:.4f}")
        L.append(f"chi2/dof = {self.chi2/self.dof:.3f}  (мащаб на грешките x{self.err_scale:.3f})")
        L.append(f"наклон по пътя: b = {self.slope:+.4f} +- {self.slope_err:.4f}  "
                 f"({self.z():+.2f} sigma)")
        if self.p_perm is not None:
            L.append(f"пермутационна p = {self.p_perm:.4f}")
        if self.spearman_r is not None:
            L.append(f"частичен Spearman r = {self.spearman_r:+.3f} (p = {self.spearman_p:.4f})")
        L.append(f"90% горна граница: |b| < {self.bound(0.90):.4f}")
        return "\n".join(L)


# ----------------------------------------------------------------------
# претеглена регресия
# ----------------------------------------------------------------------

def _wls(y, sigma, X, rescale_errors=True):
    """Претеглен най-малки квадрати. Връща coef, cov, chi2, dof, scale."""
    w = 1.0 / np.asarray(sigma, float) ** 2
    W = np.diag(w)
    XtW = X.T @ W
    cov = np.linalg.inv(XtW @ X)
    coef = cov @ (XtW @ y)
    resid = y - X @ coef
    chi2 = float(resid @ (W @ resid))
    dof = len(y) - X.shape[1]
    scale = 1.0
    if rescale_errors and dof > 0 and chi2 > dof:
        # има допълнителен разсейв над заявените грешки -> инфлация,
        # за да не се получи фалшива значимост
        scale = np.sqrt(chi2 / dof)
        cov = cov * scale ** 2
    return coef, cov, chi2, dof, scale


def _design(x, covariates, standardize=True):
    """Строи матрицата на дизайна [1, x, u_1..u_k]."""
    n = len(x)
    cols = [np.ones(n), np.asarray(x, float)]
    names = ["intercept", "path"]
    if covariates:
        for nm, u in covariates.items():
            u = np.asarray(u, float)
            if standardize:
                s = u.std()
                u = (u - u.mean()) / (s if s > 0 else 1.0)
            cols.append(u)
            names.append(nm)
    return np.column_stack(cols), names


def fit_path_dependence(
    A,
    sigma_A,
    x,
    sigma_x=None,
    covariates=None,
    n_perm=20000,
    n_mc=200,
    rescale_errors=True,
    center_x=True,
    seed=0,
):
    """Основната функция.

    Параметри
    ---------
    A         : (n,)  аномална статистика на събитие (напр. z от W-Twin)
    sigma_A   : (n,)  грешка на A
    x         : (n,)  пътна променлива (напр. log10 на колонната плътност,
                      или log10(theta_min/deg))
    sigma_x   : (n,) или None -- несигурност на x от posterior-а на позицията.
                      Ако е зададена, се прави Монте Карло върху x.
    covariates: dict{name: array} -- ковариати за контрол, напр.
                      {"log_snr": ..., "log_area": ...}
    n_perm    : брой пермутации за нулевото разпределение на наклона
    n_mc      : брой Монте Карло тегления върху x (само ако sigma_x е зададена)

    Връща FitResult.
    """
    rng = np.random.default_rng(seed)
    A = np.asarray(A, float)
    sigma_A = np.asarray(sigma_A, float)
    x = np.asarray(x, float)
    n = len(A)
    if not (len(sigma_A) == len(x) == n):
        raise ValueError("A, sigma_A и x трябва да са с еднаква дължина")
    if n < 5:
        raise ValueError("нужни са поне 5 събития")

    x0 = x - x.mean() if center_x else x.copy()

    # --- основен фит (с Монте Карло върху x, ако има несигурност) ---
    if sigma_x is None:
        X, names = _design(x0, covariates)
        coef, cov, chi2, dof, scale = _wls(A, sigma_A, X, rescale_errors)
        b, sb = coef[1], np.sqrt(cov[1, 1])
    else:
        sigma_x = np.asarray(sigma_x, float)
        bs, vs = [], []
        for _ in range(n_mc):
            xd = rng.normal(x, sigma_x)
            xd = xd - xd.mean() if center_x else xd
            X, names = _design(xd, covariates)
            c, cv, chi2, dof, scale = _wls(A, sigma_A, X, rescale_errors)
            bs.append(c[1]); vs.append(cv[1, 1])
        bs = np.array(bs); vs = np.array(vs)
        b = bs.mean()
        sb = np.sqrt(vs.mean() + bs.var(ddof=1))   # закон на пълната дисперсия
        X, names = _design(x0, covariates)
        coef, cov, chi2, dof, scale = _wls(A, sigma_A, X, rescale_errors)
        coef[1] = b
        cov[1, 1] = sb ** 2

    # --- пермутационно нулево разпределение ---
    p_perm, perm_b = None, None
    if n_perm and n_perm > 0:
        perm_b = np.empty(n_perm)
        for i in range(n_perm):
            xp = rng.permutation(x0)          # разбива връзката път <-> сигнал,
            Xp, _ = _design(xp, covariates)   # но пази A с неговите ковариати
            cp, _, _, _, _ = _wls(A, sigma_A, Xp, rescale_errors)
            perm_b[i] = cp[1]
        p_perm = float((np.abs(perm_b) >= abs(b)).mean())

    # --- частичен Spearman (без модел за формата на зависимостта) ---
    sr, sp = None, None
    if covariates:
        U, _ = _design(np.zeros(n), covariates)      # [1, 0, u...]
        U = np.delete(U, 1, axis=1)
        rA = A - U @ np.linalg.lstsq(U, A, rcond=None)[0]
        rx = x0 - U @ np.linalg.lstsq(U, x0, rcond=None)[0]
        sr, sp = stats.spearmanr(rx, rA)
    else:
        sr, sp = stats.spearmanr(x0, A)

    return FitResult(
        slope=float(b), slope_err=float(sb), intercept=float(coef[0]),
        coef=coef, cov=cov, names=names, chi2=float(chi2), dof=int(dof),
        err_scale=float(scale), n=n, p_perm=p_perm, perm_slopes=perm_b,
        spearman_r=float(sr), spearman_p=float(sp),
        extra={"x_mean": float(x.mean())},
    )


# ----------------------------------------------------------------------
# разделяне близо/далеч -- груб, но напълно прозрачен втори тест
# ----------------------------------------------------------------------

def split_test(A, sigma_A, x, q=0.5):
    """Сравнява претеглените средни на A в двете половини по x.

    Не замества регресията -- служи като независима проверка, която
    не зависи от предположението за линейност.
    """
    A = np.asarray(A, float); sigma_A = np.asarray(sigma_A, float); x = np.asarray(x, float)
    thr = np.quantile(x, q)
    lo, hi = x <= thr, x > thr
    out = {}
    for nm, m in (("low", lo), ("high", hi)):
        w = 1 / sigma_A[m] ** 2
        out[nm + "_mean"] = float((A[m] * w).sum() / w.sum())
        out[nm + "_err"] = float(np.sqrt(1 / w.sum()))
        out[nm + "_n"] = int(m.sum())
    d = out["high_mean"] - out["low_mean"]
    sd = np.hypot(out["high_err"], out["low_err"])
    out["diff"] = float(d)
    out["diff_err"] = float(sd)
    out["diff_z"] = float(d / sd)
    out["threshold"] = float(thr)
    return out
