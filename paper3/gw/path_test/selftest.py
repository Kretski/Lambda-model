"""
selftest.py -- валидация на path_stats преди да се пуска върху реални данни.

Три проверки:
  1. НУЛА       -- когато няма зависимост, p-стойностите трябва да са
                   равномерни, а покритието на 90% интервал ~90%.
  2. ИНЖЕКЦИЯ   -- когато има зададен наклон, той трябва да се възстановява
                   без отклонение.
  3. БЪРКАЩ ФАКТОР -- когато A зависи от SNR, а пътната променлива случайно
                   корелира със SNR, тестът БЕЗ ковариат трябва да даде
                   фалшива значимост, а тестът С ковариат -- не.

Проверка 3 е причината, поради която ковариатите не са опция.
"""

import numpy as np
from path_stats import fit_path_dependence, split_test


def make_dataset(n, slope, rng, conf=0.0, sigma_A=1.0, extra_scatter=0.0):
    """conf -- сила на връзката SNR -> A и SNR -> x (бъркащ фактор)."""
    log_snr = rng.normal(1.1, 0.25, n)                     # log10 SNR ~ 8-25
    x = rng.normal(0, 1, n) + conf * (log_snr - log_snr.mean()) / 0.25
    A_true = slope * (x - x.mean()) + conf * (log_snr - log_snr.mean()) / 0.25
    A = A_true + rng.normal(0, sigma_A, n) + rng.normal(0, extra_scatter, n)
    return A, np.full(n, sigma_A), x, log_snr


def check_null(n_trials=400, n=90, seed=1):
    rng = np.random.default_rng(seed)
    ps, inside = [], 0
    for _ in range(n_trials):
        A, sA, x, ls = make_dataset(n, 0.0, rng)
        r = fit_path_dependence(A, sA, x, covariates={"log_snr": ls},
                                n_perm=400, seed=int(rng.integers(1e9)))
        ps.append(r.p_perm)
        if abs(r.slope) < 1.645 * r.slope_err:
            inside += 1
    ps = np.array(ps)
    print("1. НУЛА (без зависимост)")
    print(f"   дял p < 0.05        = {np.mean(ps < 0.05):.3f}   (очаквано ~0.05)")
    print(f"   дял p < 0.10        = {np.mean(ps < 0.10):.3f}   (очаквано ~0.10)")
    print(f"   покритие на 90% CI  = {inside/n_trials:.3f}   (очаквано ~0.90)")
    return ps


def check_injection(n_trials=300, n=90, slope=0.35, seed=2):
    rng = np.random.default_rng(seed)
    got, errs, sig = [], [], 0
    for _ in range(n_trials):
        A, sA, x, ls = make_dataset(n, slope, rng)
        r = fit_path_dependence(A, sA, x, covariates={"log_snr": ls},
                                n_perm=400, seed=int(rng.integers(1e9)))
        got.append(r.slope); errs.append(r.slope_err)
        sig += r.p_perm < 0.05
    got = np.array(got); errs = np.array(errs)
    print(f"\n2. ИНЖЕКЦИЯ (истински наклон b = {slope})")
    print(f"   възстановен b       = {got.mean():+.4f} +- {got.std(ddof=1):.4f}")
    print(f"   отклонение          = {got.mean()-slope:+.4f}")
    print(f"   средна оценка sigma = {errs.mean():.4f}  (реален разсейв {got.std(ddof=1):.4f})")
    print(f"   мощност при p<0.05  = {sig/n_trials:.3f}")


def check_confounder(n_trials=300, n=90, conf=0.8, seed=3):
    rng = np.random.default_rng(seed)
    fp_without, fp_with = 0, 0
    for _ in range(n_trials):
        A, sA, x, ls = make_dataset(n, 0.0, rng, conf=conf)
        s = int(rng.integers(1e9))
        r0 = fit_path_dependence(A, sA, x, n_perm=400, seed=s)
        r1 = fit_path_dependence(A, sA, x, covariates={"log_snr": ls},
                                 n_perm=400, seed=s)
        fp_without += r0.p_perm < 0.05
        fp_with += r1.p_perm < 0.05
    print(f"\n3. БЪРКАЩ ФАКТОР (A зависи от SNR, x корелира със SNR, истинско b = 0)")
    print(f"   фалшива значимост БЕЗ ковариат = {fp_without/n_trials:.3f}")
    print(f"   фалшива значимост С ковариат   = {fp_with/n_trials:.3f}   (очаквано ~0.05)")


def demo_report(n=90, seed=7):
    """Как изглежда един реален отчет при нулев резултат."""
    rng = np.random.default_rng(seed)
    A, sA, x, ls = make_dataset(n, 0.0, rng, sigma_A=1.0)
    sx = np.full(n, 0.15)          # несигурност на пътната променлива от skymap
    r = fit_path_dependence(A, sA, x, sigma_x=sx,
                            covariates={"log_snr": ls}, n_perm=20000, seed=0)
    print("\n" + "=" * 62)
    print("ПРИМЕРЕН ОТЧЕТ (синтетични данни, истинско b = 0)")
    print("=" * 62)
    print(r.summary())
    st = split_test(A, sA, x)
    print(f"\nразделяне ниско/високо: разлика = {st['diff']:+.3f} "
          f"+- {st['diff_err']:.3f}  ({st['diff_z']:+.2f} sigma)")


if __name__ == "__main__":
    check_null()
    check_injection()
    check_confounder()
    demo_report()
