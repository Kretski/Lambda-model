"""
fetch_catalog.py -- сваля каталожните метаданни от GWOSC и прави events.csv

    python fetch_catalog.py                 # сваля и обработва
    python fetch_catalog.py --offline       # ползва вече свалено копие
    python fetch_catalog.py --probe GW150914   # показва какво има за едно събитие

Прави три неща:
  1. Сваля таблицата на всички събития (с повторни опити и кеш -- заради
     таймаутите ти с GWOSC).
  2. Дедуплицира по commonName, като пази най-високата версия, и слага
     праг по FAR.
  3. Пише events.csv (event, gps, snr, far, distance) и anomaly_template.csv

RA/Dec НЯМА в таблицата -- позицията идва от скаймаповете, отделна стъпка.
--probe показва какви файлове са налични за конкретно събитие, за да видим
откъде да ги теглим, без да гадаем URL-и.

Зависимости: pandas (urllib е от стандартната библиотека).
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

import pandas as pd

CSV_URL = "https://gwosc.org/eventapi/csv/allevents/"
CACHE = "allevents_cache.csv"
UA = {"User-Agent": "path-test/1.0"}

# каталози, които влизат в теста -- уверените, без marginal
CONFIDENT = ("confident", "GWTC")


def fetch(url, dest=None, timeout=60, retries=5, backoff=3.0):
    """Сваляне с повторни опити. Връща bytes или записва във файл."""
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if dest:
                with open(dest, "wb") as f:
                    f.write(data)
                print(f"  -> {dest} ({len(data)/1024:.0f} kB)")
            return data
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            wait = backoff * (k + 1)
            print(f"  опит {k+1}/{retries} неуспешен ({e}); чакам {wait:.0f}s")
            time.sleep(wait)
    raise SystemExit(f"не успях да сваля {url}: {last}\n"
                     f"свали го ръчно в {CACHE} и пусни с --offline")


def load_table(offline=False):
    if offline or os.path.exists(CACHE):
        if os.path.exists(CACHE):
            print(f"ползвам кеш: {CACHE}")
            return pd.read_csv(CACHE)
        raise SystemExit(f"няма {CACHE}; пусни без --offline")
    print(f"сваляне: {CSV_URL}")
    fetch(CSV_URL, CACHE)
    return pd.read_csv(CACHE)


def _catalog_rank(name):
    """Приоритет на каталога: по-новият печели. Preliminary/auxiliary са най-ниско."""
    s = str(name).lower()
    if "prelim" in s or "auxiliar" in s or "marginal" in s:
        return -1.0
    import re
    m = re.search(r"gwtc-?(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else 0.0


def build_events(df, far_max=1e-3, confident_only=True):
    """Сливане на версиите по събитие + праг по FAR.

    Ключът е GPS (закръглен до секунда), а не името -- едно и също събитие
    се води под различни имена в различните каталози (GW190425 срещу
    GW190425_081805).

    За всяко поле се взима последната НЕПРАЗНА стойност по приоритет на
    каталога. Иначе се губят събития, чиито по-нови редове имат празен far.
    """
    n0 = len(df)
    df = df.copy()

    if confident_only and "catalog.shortName" in df:
        cat = df["catalog.shortName"].astype(str)
        keep = cat.str.contains("|".join(CONFIDENT), case=False, na=False)
        keep &= ~cat.str.contains("marginal|preliminary|auxiliary",
                                  case=False, na=False)
        df = df[keep]
        print(f"потвърдени каталози: {len(df)}/{n0}  "
              f"({sorted(set(df['catalog.shortName'].astype(str)))})")

    df["_rank"] = df["catalog.shortName"].map(_catalog_rank)
    df["_key"] = df["GPS"].round(0)
    df = df.sort_values(["_rank", "version"])

    fields = ["commonName", "GPS", "network_matched_filter_snr", "far",
              "luminosity_distance", "jsonurl", "catalog.shortName"]
    fields = [f for f in fields if f in df.columns]

    def merge(g):
        out = {}
        for f in fields:
            s = g[f].dropna()
            out[f] = s.iloc[-1] if len(s) else float("nan")
        out["n_versions"] = len(g)
        return pd.Series(out)

    merged = df.groupby("_key", sort=False).apply(
        merge, include_groups=False).reset_index(drop=True)
    print(f"след сливане по GPS: {len(merged)} уникални събития")

    n_missing_far = int(merged["far"].isna().sum()) if "far" in merged else 0
    if n_missing_far:
        print(f"  без far дори след сливане: {n_missing_far}")

    if "far" in merged:
        before = len(merged)
        merged = merged[merged["far"].notna() & (merged["far"] <= far_max)]
        print(f"след FAR <= {far_max:g}/yr: {len(merged)}/{before}")

    out = pd.DataFrame({
        "event": merged["commonName"].values,
        "gps": merged["GPS"].values,
        "snr": merged.get("network_matched_filter_snr").values,
        "far": merged.get("far").values,
        "distance_mpc": merged.get("luminosity_distance").values,
        "jsonurl": merged.get("jsonurl", pd.Series([""]*len(merged))).values,
        "ra_deg": "",
        "dec_deg": "",
        "skymap": "",
    })
    before = len(out)
    out = out[out["snr"].notna()]
    print(f"със SNR: {len(out)}/{before}")
    return out.sort_values("gps").reset_index(drop=True)


def probe(df, event):
    """Показва какво съдържа JSON-ът за едно събитие -- търсим скаймап."""
    row = df[df["event"] == event]
    if row.empty:
        raise SystemExit(f"няма събитие {event} в таблицата")
    url = row.iloc[0]["jsonurl"]
    print(f"{event}: {url}")
    data = json.loads(fetch(url).decode())
    ev = list(data.get("events", {}).values())
    if not ev:
        print(json.dumps(data, indent=2)[:2000]); return
    ev = ev[0]
    print("\nключове:", sorted(ev.keys()))
    for k in ("parameters", "strain", "skymap", "files"):
        if k in ev:
            print(f"\n--- {k} ---")
            print(json.dumps(ev[k], indent=2)[:1500])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--far-max", type=float, default=1e-3,
                    help="праг по FAR (1/yr), по подразбиране 1e-3")
    ap.add_argument("--all-catalogs", action="store_true",
                    help="включи и marginal каталозите")
    ap.add_argument("--probe", metavar="EVENT",
                    help="покажи наличните файлове за едно събитие")
    ap.add_argument("-o", "--out", default="events.csv")
    a = ap.parse_args()

    df = load_table(a.offline)
    print(f"общо редове в таблицата: {len(df)}")
    ev = build_events(df, far_max=a.far_max, confident_only=not a.all_catalogs)

    if a.probe:
        probe(ev, a.probe)
        return

    ev.to_csv(a.out, index=False)
    print(f"\n-> {a.out}   ({len(ev)} събития)")
    print(f"   SNR: {ev['snr'].min():.1f} .. {ev['snr'].max():.1f}")
    print("   ra_deg/dec_deg/skymap са празни -- попълват се на следващата стъпка")

    tmpl = ev[["event", "snr"]].copy()
    tmpl.insert(1, "A", "")
    tmpl.insert(2, "sigma_A", "")
    tmpl["area_deg2"] = ""
    tmpl.to_csv("anomaly_template.csv", index=False)
    print("-> anomaly_template.csv  (попълни колоните A и sigma_A от W-Twin)")

    # груба оценка на чувствителността
    n = len(ev)
    print(f"\nочаквано sigma_b при n={n} и sigma_A=1: ~{1.0/max(n-4,1)**0.5/0.29:.2f}")
    print("(за сравнение: при n=11 е ~1.0, което е безполезно)")


if __name__ == "__main__":
    main()
