"""
run_test.py -- свързва двата CSV файла и печата пълния отчет.

    python run_test.py anomaly.csv path_vars.csv

anomaly.csv трябва да съдържа:
    event, A, sigma_A, snr [, area_deg2]

  A       -- твоята аномална статистика (z от W-Twin Λ-профила,
             или residual z) -- ЕДНА, фиксирана предварително
  sigma_A -- нейната грешка
  snr     -- мрежово SNR на събитието (ковариат)
  area_deg2 -- площ на 90% локализация (ковариат, по избор)

path_vars.csv се произвежда от sky_path_vars.py

Отчита и трите пътни оси. Основната е log_dm_col; другите две се
докладват като вторични, включително когато не показват нищо.
"""

from __future__ import annotations
import sys
import numpy as np
import pandas as pd

from path_stats import fit_path_dependence, split_test

AXES = [
    ("log_dm_col", "Σ_DM  (тъмна материя, NFW хало)  [ОСНОВНА]"),
    ("log_bar_col", "Σ_baryon  (барионен диск)        [вторична]"),
    ("log_theta_min", "θ_min  (Слънчева система)        [вторична]"),
]


def main(anomaly_csv, path_csv, n_perm=20000):
    a = pd.read_csv(anomaly_csv)
    p = pd.read_csv(path_csv)
    df = a.merge(p, on="event", how="inner")
    if len(df) < 5:
        raise SystemExit(f"само {len(df)} съвпадащи събития -- нужни са поне 5")
    if len(df) < len(a):
        missing = sorted(set(a.event) - set(df.event))
        print(f"[!] без пътни променливи: {missing}\n")

    cov = {"log_snr": np.log10(df["snr"].to_numpy())}
    if "area_deg2" in df.columns:
        cov["log_area"] = np.log10(df["area_deg2"].to_numpy())

    print(f"събития: {len(df)}   ковариати: {list(cov)}\n")

    for col, label in AXES:
        if col not in df.columns:
            continue
        x = df[col].to_numpy()
        sx = df.get(col + "_err")
        sx = sx.to_numpy() if sx is not None else None
        if sx is not None and np.allclose(sx, 0):
            sx = None

        r = fit_path_dependence(
            df["A"].to_numpy(), df["sigma_A"].to_numpy(),
            x=x, sigma_x=sx, covariates=cov, n_perm=n_perm, seed=0,
        )
        st = split_test(df["A"].to_numpy(), df["sigma_A"].to_numpy(), x)

        print("=" * 64)
        print(label)
        print("=" * 64)
        print(r.summary())
        print(f"разделяне ниско/високо: разлика = {st['diff']:+.3f} "
              f"± {st['diff_err']:.3f}  ({st['diff_z']:+.2f}σ)")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
