"""
sky_path_vars.py -- превръща posterior-а на позицията в пътна променлива.

Локализацията на GW събития е десетки до стотици кв. градуса, затова пътната
променлива не е число, а разпределение. Тук се тегли от скаймапа и се
маргинализира.

Изход за всяко събитие -- средно и std на:
    log10 Sigma_DM       колонна плътност на ТМ през халото
    log10 Sigma_baryon   барионна колонна плътност
    log10(theta_min/deg) до най-близкото тяло от Слънчевата система

    python sky_path_vars.py events.csv path_vars.csv
    python sky_path_vars.py events.csv path_vars.csv --limit 5   # проба
    python sky_path_vars.py events.csv path_vars.csv --samples 4000

Чете и двата HEALPix формата: плосък (една колона пиксели) и multiorder
(UNIQ/PROBDENSITY). Резултатите се кешират по събитие, така че прекъснат
пуск продължава оттам, докъдето е стигнал.

Зависимости: astropy, healpy, pandas, numpy.
"""

from __future__ import annotations
import argparse
import json
import os
import numpy as np
import pandas as pd

from galactic_column import dm_column, baryon_column

BODIES = ["sun", "moon", "jupiter", "venus", "mars", "saturn", "mercury"]
CACHE = "path_vars_cache"


# ----------------------------------------------------------------------
# четене на скаймап
# ----------------------------------------------------------------------

def _read_multiorder(path):
    """Multiorder (UNIQ/PROBDENSITY) -> (nside на ред, ipix nested, prob)."""
    import healpy as hp
    from astropy.table import Table
    t = Table.read(path)
    cols = {c.upper(): c for c in t.colnames}
    if "UNIQ" not in cols:
        raise ValueError("няма колона UNIQ")
    uniq = np.asarray(t[cols["UNIQ"]], dtype=np.int64)
    level = (np.log2(uniq // 4) // 2).astype(int)
    nside = (2 ** level).astype(np.int64)
    ipix = uniq - 4 * (nside ** 2)

    if "PROBDENSITY" in cols:
        dens = np.asarray(t[cols["PROBDENSITY"]], dtype=float)
        prob = dens * hp.nside2pixarea(nside)      # плътност * площ
    elif "PROB" in cols:
        prob = np.asarray(t[cols["PROB"]], dtype=float)
    else:
        raise ValueError(f"няма PROBDENSITY/PROB; колони: {t.colnames}")

    prob = np.clip(prob, 0, None)
    s = prob.sum()
    if s <= 0:
        raise ValueError("нулева вероятност")
    return nside, ipix, prob / s


def _read_flat(path):
    """Плосък HEALPix -> (nside, prob) в RING подредба."""
    import healpy as hp
    m = np.asarray(hp.read_map(path, dtype=np.float64), dtype=float)
    m = np.clip(m, 0, None)
    s = m.sum()
    if s <= 0:
        raise ValueError("нулева вероятност")
    return hp.npix2nside(len(m)), m / s


def sample_skymap(path, n=2000, rng=None):
    """Тегли (ra, dec) в градуси от скаймап. Работи и с двата формата."""
    import healpy as hp
    rng = rng or np.random.default_rng(0)

    try:
        nside_arr, ipix, prob = _read_multiorder(path)
        rows = rng.choice(len(prob), size=n, p=prob)
        ns = nside_arr[rows]
        th, ph = hp.pix2ang(ns, ipix[rows], nest=True)
        res = hp.nside2resol(ns)
        mode = "multiorder"
    except Exception:
        nside, m = _read_flat(path)
        idx = rng.choice(len(m), size=n, p=m)
        th, ph = hp.pix2ang(nside, idx)
        res = np.full(n, hp.nside2resol(nside))
        mode = "flat"

    # разпръскване вътре в пиксела, за да не излязат дискретни стойности
    th = np.clip(th + rng.normal(0, res / 3), 1e-6, np.pi - 1e-6)
    ph = ph + rng.normal(0, res / 3) / np.sin(th)
    return np.degrees(ph) % 360.0, 90.0 - np.degrees(th), mode


# ----------------------------------------------------------------------
# геометрия
# ----------------------------------------------------------------------

def theta_min_to_bodies(ra_deg, dec_deg, gps, bodies=BODIES):
    """Минимално ъглово разстояние (градуси) до тяло от Слънчевата система."""
    from astropy.coordinates import SkyCoord, get_body, solar_system_ephemeris
    from astropy.time import Time
    import astropy.units as u

    t = Time(float(gps), format="gps")
    src = SkyCoord(ra=np.asarray(ra_deg) * u.deg,
                   dec=np.asarray(dec_deg) * u.deg, frame="icrs")
    seps = []
    with solar_system_ephemeris.set("builtin"):
        for b in bodies:
            seps.append(src.separation(get_body(b, t).icrs).deg)
    return np.min(np.vstack(seps), axis=0)


def galactic_lb(ra_deg, dec_deg):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    g = SkyCoord(ra=np.asarray(ra_deg) * u.deg,
                 dec=np.asarray(dec_deg) * u.deg, frame="icrs").galactic
    return g.l.deg, g.b.deg


def _ms(v):
    v = np.atleast_1d(np.asarray(v, float))
    return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1 else 0.0)


def path_vars_for_event(gps, skymap=None, ra_deg=None, dec_deg=None,
                        n=2000, seed=0):
    rng = np.random.default_rng(seed)
    if skymap and isinstance(skymap, str) and skymap.strip() and os.path.exists(skymap):
        ra, dec, mode = sample_skymap(skymap, n=n, rng=rng)
    elif ra_deg is not None and not pd.isna(ra_deg):
        ra = np.array([float(ra_deg)]); dec = np.array([float(dec_deg)])
        mode = "point"
    else:
        return None

    l, b = galactic_lb(ra, dec)
    m_dm, s_dm = _ms(np.log10(np.atleast_1d(dm_column(l, b))))
    m_bar, s_bar = _ms(np.log10(np.atleast_1d(baryon_column(l, b))))
    m_th, s_th = _ms(np.log10(np.maximum(theta_min_to_bodies(ra, dec, gps), 1e-3)))
    return dict(
        log_dm_col=m_dm, log_dm_col_err=s_dm,
        log_bar_col=m_bar, log_bar_col_err=s_bar,
        log_theta_min=m_th, log_theta_min_err=s_th,
        n_sky_samples=int(len(ra)), skymap_mode=mode,
    )


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events", nargs="?", default="events.csv")
    ap.add_argument("out", nargs="?", default="path_vars.csv")
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0, help="само първите N събития")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(a.events)
    if a.limit:
        df = df.head(a.limit)
    os.makedirs(CACHE, exist_ok=True)

    rows, failed = [], []
    for i, r in df.iterrows():
        cp = os.path.join(CACHE, f"{r['event']}.json")
        if not a.no_cache and os.path.exists(cp):
            with open(cp) as f:
                v = json.load(f)
        else:
            try:
                v = path_vars_for_event(
                    gps=r["gps"], skymap=r.get("skymap"),
                    ra_deg=r.get("ra_deg"), dec_deg=r.get("dec_deg"),
                    n=a.samples, seed=i)
            except Exception as e:
                print(f"  {r['event']}: {type(e).__name__}: {e}")
                failed.append(r["event"]); continue
            if v is None:
                failed.append(r["event"]); continue
            with open(cp, "w") as f:
                json.dump(v, f)
        v["event"] = r["event"]
        rows.append(v)
        if len(rows) <= 3 or len(rows) % 25 == 0:
            print(f"  [{len(rows)}] {r['event']:<18s} "
                  f"log Sigma_DM={v['log_dm_col']:.3f}±{v['log_dm_col_err']:.3f}  "
                  f"log theta_min={v['log_theta_min']:+.3f}±{v['log_theta_min_err']:.3f}  "
                  f"({v['skymap_mode']})")

    if not rows:
        raise SystemExit("нито едно събитие не мина")
    out = pd.DataFrame(rows)
    out = out[["event"] + [c for c in out.columns if c != "event"]]
    out.to_csv(a.out, index=False)
    print(f"\n-> {a.out}  ({len(out)} събития)")
    print("  формати:", out["skymap_mode"].value_counts().to_dict())
    print(f"  log Sigma_DM обхват: {out['log_dm_col'].min():.2f} .. "
          f"{out['log_dm_col'].max():.2f}")
    print(f"  медианна несигурност на log Sigma_DM: "
          f"{out['log_dm_col_err'].median():.3f}")
    if failed:
        print(f"  пропуснати: {len(failed)}  {failed[:8]}")


if __name__ == "__main__":
    main()
