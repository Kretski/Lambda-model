"""
fetch_skymaps.py -- намира, сваля и разархивира скаймаповете, после ги
свързва със събитията в events.csv.

Скаймапите НЕ са отделен файл на събитие. В event JSON-а стои:

    parameters.<release>.links.skymap
        -> .../IGWN-GWTC2p1-v2-PESkyMaps.tar.gz

тоест един архив за цял каталог. Затова:

  фаза 1  сваля event JSON за всяко събитие (кеширано в json_cache/)
          и събира уникалните URL-и на архивите
  фаза 2  сваля архивите (с продължаване при прекъсване)
  фаза 3  разархивира и индексира FITS файловете
  фаза 4  попълва колоната skymap в events.csv

    python fetch_skymaps.py                  # всичко
    python fetch_skymaps.py --phase 1        # само събиране на URL-и
    python fetch_skymaps.py --dry-run        # показва какво би свалило и колко

Внимание: архивите са големи (стотици MB на каталог). --dry-run първо.

Зависимости: pandas (останалото е стандартна библиотека).
"""

from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import sys
import tarfile
import time
import urllib.request
import urllib.error

import pandas as pd

UA = {"User-Agent": "path-test/1.0"}
JSON_CACHE = "json_cache"
TAR_DIR = "skymap_tars"
FITS_DIR = "skymaps"

# GW150914 или GW190425_081805
EVENT_TOKEN = re.compile(r"GW\d{6}(?:_\d{6})?")


# ----------------------------------------------------------------------
# мрежа
# ----------------------------------------------------------------------

def fetch_bytes(url, timeout=60, retries=5, backoff=3.0):
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            time.sleep(backoff * (k + 1))
    raise RuntimeError(f"{url}: {last}")


def download_resumable(url, dest, timeout=120, retries=8):
    """Сваляне с продължаване -- за големите архиви при нестабилна връзка."""
    tmp = dest + ".part"
    for k in range(retries):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        headers = dict(UA)
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                total = r.headers.get("Content-Length")
                mode = "ab" if have and r.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                with open(tmp, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        if total:
                            pct = 100 * have / (int(total) + (have - len(chunk) if mode == "ab" else 0))
                            print(f"\r    {have/1e6:8.1f} MB", end="", flush=True)
                        else:
                            print(f"\r    {have/1e6:8.1f} MB", end="", flush=True)
            print()
            os.replace(tmp, dest)
            return dest
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"\n    прекъснато ({e}); продължавам след {3*(k+1)}s")
            time.sleep(3 * (k + 1))
    raise RuntimeError(f"не успях да сваля {url}")


# ----------------------------------------------------------------------
# фаза 1 -- URL-и на архивите
# ----------------------------------------------------------------------

def event_json(event, url, cache=JSON_CACHE):
    os.makedirs(cache, exist_ok=True)
    p = os.path.join(cache, f"{event}.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    data = json.loads(fetch_bytes(url).decode())
    with open(p, "w") as f:
        json.dump(data, f)
    return data


ZENODO_DOI = re.compile(r"zenodo\.(\d+)")
SKYMAP_FILE = re.compile(r"(skymap|skyloc|localization)", re.I)
ARCHIVE_EXT = re.compile(r"\.(tar\.gz|tgz|tar|zip|fits(\.gz)?)$", re.I)


def zenodo_files(doi_or_url, cache=JSON_CACHE):
    """Резолвира Zenodo DOI до списък с файлове в записа."""
    m = ZENODO_DOI.search(str(doi_or_url))
    if not m:
        return []
    rec = m.group(1)
    os.makedirs(cache, exist_ok=True)
    p = os.path.join(cache, f"zenodo_{rec}.json")
    if os.path.exists(p):
        with open(p) as f:
            data = json.load(f)
    else:
        try:
            data = json.loads(fetch_bytes(
                f"https://zenodo.org/api/records/{rec}").decode())
        except Exception as e:
            print(f"    zenodo {rec}: {e}")
            return []
        with open(p, "w") as f:
            json.dump(data, f)
    out = []
    for fl in data.get("files", []) or []:
        key = fl.get("key") or fl.get("filename") or ""
        url = (fl.get("links") or {}).get("self") or fl.get("links", {}).get("download")
        if key and url:
            out.append((key, url, fl.get("size", 0)))
    return out


def collect_links(data):
    """Всички линкове от PE releases -- не само ключа 'skymap'.

    По-новите каталози (GWTC-4.x, 5.0) нямат ключ 'skymap' изобщо; там
    стои {'na': 'https://doi.org/10.5281/zenodo.NNNN'} -- DOI към записа с
    PE данните. Затова се събира всичко и се класифицира после.
    """
    evs = list(data.get("events", {}).values())
    if not evs:
        return []
    params = evs[0].get("parameters", {}) or {}
    out = []
    for rel_name, rel in params.items():
        if not isinstance(rel, dict):
            continue
        if rel.get("pipeline_type") != "pe":
            continue
        for key, url in (rel.get("links") or {}).items():
            if isinstance(url, str) and url:
                out.append((key, url, bool(rel.get("is_preferred"))))
    return out


ZEN_ID = re.compile(r"zenodo[./](?:records/|api/records/)?(\d{4,})", re.I)


def zenodo_record_id(url):
    m = ZEN_ID.search(url)
    return m.group(1) if m else None


def archive_name(url):
    """Име на файла -- за дедупликация на стар/нов Zenodo URL за същия файл."""
    parts = [p for p in url.rstrip("/").split("/") if p]
    for p in reversed(parts):
        if re.search(r"\.(tar\.gz|tgz|zip|fits(\.gz)?)$", p, re.I):
            return p
    return parts[-1] if parts else url


def zenodo_files(record_id, cache=JSON_CACHE):
    """Списък с файловете в Zenodo запис: [(име, размер, url)]."""
    os.makedirs(cache, exist_ok=True)
    p = os.path.join(cache, f"zenodo_{record_id}.json")
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
    else:
        d = json.loads(fetch_bytes(
            f"https://zenodo.org/api/records/{record_id}").decode())
        with open(p, "w") as f:
            json.dump(d, f)
    out = []
    for fl in d.get("files", []):
        name = fl.get("key") or fl.get("filename") or ""
        size = fl.get("size") or fl.get("filesize") or 0
        url = (fl.get("links") or {}).get("self") or (fl.get("links") or {}).get("download") or ""
        out.append((name, size, url))
    return out, d.get("metadata", {}).get("title", "")


SKYMAP_HINT = re.compile(r"sky|localiz|fits", re.I)

# архивите със скаймапове вътре в Zenodo записите
SKYMAP_ARCHIVE = re.compile(r"(skymap|skylocali[sz]ation)s?.*\.(tar\.gz|tgz|zip)$", re.I)


def zenodo_skymap_archives(zenodo, verbose=True):
    """Намира tar.gz архивите със скаймапове в Zenodo записите.

    Връща {име: (url, [събития], размер)}. Дедуплицира архиви с еднакъв
    размер и еднакво множество събития (GWTC-3 е публикуван двойно, веднъж
    като skymaps.tar.gz и веднъж като PESkyLocalizations.tar.gz).
    """
    found = {}
    for rid, evs in sorted(zenodo.items(), key=lambda kv: -len(kv[1])):
        try:
            files, _ = zenodo_files(rid)
        except Exception as e:
            print(f"  zenodo.{rid}: {e}")
            continue
        for nm, size, url in files:
            if SKYMAP_ARCHIVE.search(nm) and url:
                found[nm] = (url, evs, size)
                if verbose:
                    print(f"  {size/1e6:8.1f} MB  {nm[:60]}  ({len(evs)} събития)")

    # дедупликация по (размер, множество събития)
    seen, out = {}, {}
    for nm, (url, evs, size) in found.items():
        key = (size, tuple(sorted(set(evs))))
        if key in seen:
            print(f"  пропускам дубликат: {nm[:60]} == {seen[key][:60]}")
            continue
        seen[key] = nm
        out[nm] = (url, evs, size)
    return out



def phase1(ev, delay=0.2):
    """Събира линковете и ги класифицира: директен архив / Zenodo DOI / друго."""
    archives = {}     # име на файл -> (url, [събития])
    zenodo = {}       # record id -> [събития]
    other = {}
    miss = []
    for i, r in ev.iterrows():
        if not isinstance(r.get("jsonurl"), str) or not r["jsonurl"]:
            miss.append(r["event"]); continue
        try:
            d = event_json(r["event"], r["jsonurl"])
        except Exception as e:
            print(f"  {r['event']}: {e}"); miss.append(r["event"]); continue

        links = collect_links(d)
        if not links:
            miss.append(r["event"])
        got = False
        for key, url, pref in links:
            if re.search(r"\.(tar\.gz|tgz|zip|fits(\.gz)?)(/content)?$", url, re.I):
                nm = archive_name(url)
                archives.setdefault(nm, [url, []])[1].append(r["event"])
                got = True
            elif zenodo_record_id(url):
                zenodo.setdefault(zenodo_record_id(url), []).append(r["event"])
                got = True
            else:
                other.setdefault(url, []).append(r["event"])
        if not got and links:
            miss.append(r["event"])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(ev)} ...")
        time.sleep(delay)

    print(f"\nдиректни архиви: {len(archives)}")
    for nm, (url, evs) in sorted(archives.items(), key=lambda kv: -len(kv[1][1])):
        print(f"  {len(evs):4d} събития  {nm}")
    print(f"\nZenodo записи: {len(zenodo)}")
    for rid, evs in sorted(zenodo.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(evs):4d} събития  zenodo.{rid}")
    if other:
        print(f"\nдруги линкове: {len(other)}")
        for url, evs in list(other.items())[:5]:
            print(f"  {len(evs):4d} събития  {url[:90]}")
    if miss:
        print(f"\nбез никакъв PE линк: {len(miss)}  напр. {miss[:6]}")
    return archives, zenodo, miss


def inspect_zenodo(zenodo):
    """Показва какви файлове има във всеки Zenodo запис -- преди да се тегли."""
    for rid, evs in sorted(zenodo.items(), key=lambda kv: -len(kv[1])):
        print(f"\n{'='*66}\nzenodo.{rid}   ({len(evs)} събития)")
        try:
            files, title = zenodo_files(rid)
        except Exception as e:
            print(f"  не успях: {e}"); continue
        print(f"  {title[:80]}")
        print(f"  общо файлове: {len(files)}")
        hits = [f for f in files if SKYMAP_HINT.search(f[0])]
        show = hits if hits else files[:10]
        label = "изглеждат като скаймапове" if hits else "първите 10"
        print(f"  {label}:")
        for nm, size, _ in show[:15]:
            print(f"    {size/1e6:9.1f} MB  {nm[:70]}")
        if len(show) > 15:
            print(f"    ... още {len(show)-15}")


# ----------------------------------------------------------------------
# фази 2-3 -- сваляне и разархивиране
# ----------------------------------------------------------------------

def phase2(archives, zen_archives=None):
    """archives: {име: (url, [събития])}; zen_archives: {име: (url, [събития], размер)}"""
    os.makedirs(TAR_DIR, exist_ok=True)
    todo = {nm: (u, evs) for nm, (u, evs) in archives.items()}
    for nm, (u, evs, _sz) in (zen_archives or {}).items():
        todo.setdefault(nm, (u, evs))
    known = sum(sz for _, _, sz in (zen_archives or {}).values())
    if known:
        print(f"  известен обем: {known/1e6:.0f} MB (+ архивите с неизвестен размер)")
    paths = []
    for name, (u, _evs) in todo.items():
        dest = os.path.join(TAR_DIR, name)
        if os.path.exists(dest):
            print(f"  вече е свален: {name}")
        else:
            print(f"  {name}")
            download_resumable(u, dest)
        paths.append(dest)
    return paths


def phase3(tars):
    os.makedirs(FITS_DIR, exist_ok=True)
    n = 0
    for t in tars:
        print(f"  разархивирам {os.path.basename(t)}")
        try:
            with tarfile.open(t) as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    if not re.search(r"\.fits(\.gz)?$", m.name):
                        continue
                    out = os.path.join(FITS_DIR, os.path.basename(m.name))
                    if os.path.exists(out):
                        n += 1; continue
                    src = tf.extractfile(m)
                    with open(out, "wb") as f:
                        shutil.copyfileobj(src, f)
                    n += 1
        except tarfile.ReadError as e:
            print(f"    не е валиден архив ({e}) -- може да е един FITS файл")
            if re.search(r"\.fits(\.gz)?$", t):
                shutil.copy(t, FITS_DIR); n += 1
    print(f"  FITS файлове: {n}")
    return sorted(os.listdir(FITS_DIR))


# ----------------------------------------------------------------------
# фаза 4 -- свързване по име
# ----------------------------------------------------------------------

def build_index(files):
    """token -> списък от файлове. token е GW150914 или GW190425_081805."""
    idx = {}
    for f in files:
        for tok in EVENT_TOKEN.findall(f):
            idx.setdefault(tok, []).append(f)
    return idx


# Приоритет на waveform фамилиите. ФИКСИРА СЕ ПРЕДВАРИТЕЛНО и се записва
# в бележките към анализа. "Mixed"/"combined" са комбинираните резултати и
# са предпочитаният избор на LVK.
FAMILY_PRIORITY = [
    "mixed", "combined",
    "imrphenomxphm", "seobnrv5phm", "nrsur7dq4",
    "imrphenomxpnr", "seobnrv4phm", "imrphenomxp",
]


def pick_file(files, priority=FAMILY_PRIORITY):
    """Избира един файл на събитие по фиксиран приоритет на фамилиите."""
    low = {f: f.lower() for f in files}
    for p in priority:
        hits = sorted(f for f in files if p in low[f])
        if hits:
            return hits[0], p
    return sorted(files)[0], "азбучно"


def _utc_hhmmss(gps):
    """GPS -> 'hhmmss' по UTC. Суфиксът в имената е точно това."""
    try:
        from astropy.time import Time
        return Time(float(gps), format="gps").utc.strftime("%H%M%S")
    except Exception:
        return None


def match_event(event, idx, gps=None):
    """Точно съвпадение; после по датата; при няколко -- разрешаване по GPS.

    GW190521 и GW190521_074359 са РАЗЛИЧНИ събития, затова съвпадението
    само по дата не е достатъчно и при неяснота се ползва часът от GPS.
    """
    if event in idx:
        f, fam = pick_file(idx[event])
        return f, f"exact/{fam}"
    m = EVENT_TOKEN.match(str(event))
    date = m.group(0)[:8] if m else str(event)[:8]
    cands = {k: v for k, v in idx.items() if k.startswith(date)}
    if len(cands) == 1:
        f, fam = pick_file(next(iter(cands.values())))
        return f, f"by-date/{fam}"
    if len(cands) > 1:
        hhmmss = _utc_hhmmss(gps) if gps is not None else None
        if hhmmss:
            best, bestd = None, 1e9
            for k, v in cands.items():
                suf = k.split("_")[1] if "_" in k else None
                if not suf:
                    continue
                d = abs(int(suf) - int(hhmmss))
                if d < bestd:
                    best, bestd = v, d
            # 2 секунди толеранс в hhmmss представяне
            if best is not None and bestd <= 2:
                f, fam = pick_file(best)
                return f, f"by-gps/{fam}"
        return None, f"ambiguous({len(cands)})"
    return None, "missing"


def phase4(ev, files, out="events.csv"):
    idx = build_index(files)
    paths, how = [], []
    for e, g in zip(ev["event"], ev["gps"]):
        p, h = match_event(e, idx, gps=g)
        paths.append(os.path.join(FITS_DIR, p) if p else "")
        how.append(h)
    ev = ev.copy()
    ev["skymap"] = paths
    from collections import Counter
    kind = [h.split("/")[0] for h in how]
    fam = [h.split("/")[1] for h in how if "/" in h]
    print("  свързване:", dict(Counter(kind)))
    print("  waveform фамилия:", dict(Counter(fam)))
    bad = [e for e, h in zip(ev["event"], kind) if h not in ("exact", "by-date", "by-gps")]
    if bad:
        print(f"  без скаймап: {len(bad)}  напр. {bad[:8]}")
    ev.to_csv(out, index=False)
    print(f"-> {out}  ({(ev['skymap'] != '').sum()}/{len(ev)} със скаймап)")
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--events", default="events.csv")
    ap.add_argument("--phase", type=int, default=0, help="1..4, 0 = всички")
    ap.add_argument("--dry-run", action="store_true",
                    help="спира след фаза 1")
    ap.add_argument("--inspect", action="store_true",
                    help="показва съдържанието на Zenodo записите и спира")
    a = ap.parse_args()

    ev = pd.read_csv(a.events)
    if "jsonurl" not in ev.columns:
        raise SystemExit("events.csv няма колона jsonurl -- пусни fetch_catalog.py наново")

    print("ФАЗА 1: събиране на PE линкове")
    archives, zenodo, _ = phase1(ev)

    if a.inspect:
        print("\n\nИНСПЕКЦИЯ НА ZENODO ЗАПИСИТЕ")
        inspect_zenodo(zenodo)
        return
    if a.dry_run or a.phase == 1:
        print("\n(спирам тук; --inspect показва какво има в Zenodo записите)")
        return

    print("\nФАЗА 2: сваляне на архивите")
    print("  архиви със скаймапове в Zenodo записите:")
    zen_arch = zenodo_skymap_archives(zenodo)
    tars = phase2(archives, zen_arch)
    if a.phase == 2:
        return

    print("\nФАЗА 3: разархивиране")
    files = phase3(tars)
    if a.phase == 3:
        return

    print("\nФАЗА 4: свързване със събитията")
    phase4(ev, files, a.events)


if __name__ == "__main__":
    main()
