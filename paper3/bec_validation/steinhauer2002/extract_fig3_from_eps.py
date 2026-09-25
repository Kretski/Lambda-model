#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract the measured excitation spectrum from Fig. 3 of
Steinhauer, Ozeri, Katz, Davidson, PRL 88, 120407 (2002), arXiv:cond-mat/0111438
directly from the vector EPS (OriginLab) in the arXiv TeX source.

Input : figure3.eps, or the arXiv source archive (.tar.gz) containing it
Output: steinhauer2002_fig3_extracted.csv  (13 measured points)

Usage:
    python extract_fig3_from_eps.py arXiv-cond-mat0111438v1_tar.gz
    python extract_fig3_from_eps.py figure3.eps

Method:
  * marker centres = circle ("a") commands; error bars = vertical segments at the marker x
  * each panel calibrated from its own axis ticks (3a, inset, 3b)
  * f = Delta(3b) + hbar k^2 / (2m) / 2pi   (3b has the finest frequency scale)
  * sigma from the 3b error bars; where the bar is hidden inside the marker
    (caption: "for most points the error bars are not visible"), sigma = marker
    radius, flagged sigma_is_upper_bound = True
Checks printed at the end:
  * the three panels must agree (~0.001 kHz)
  * the authors' plotted LDA curve (mu = 1.91 kHz) must match Eq. (4)
"""
import sys, os, tarfile
import numpy as np
import pandas as pd

HBAR = 1.054571817e-34
M_RB87 = 1.44316060e-25


def read_eps(path):
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as tf:
            member = [m for m in tf.getmembers() if m.name.endswith("figure3.eps")][0]
            return tf.extractfile(member).read().decode("latin-1")
    with open(path, encoding="latin-1") as fh:
        return fh.read()


def body_lines(text):
    lines = text.split("\n")
    start = [i for i, l in enumerate(lines) if l.startswith("%%BeginDocument: figure3.eps")][0]
    endfont = [i for i, l in enumerate(lines) if i > start and l.startswith("%%EndFont")][-1]
    return lines[endfont + 1:]


def parse(L, a, b):
    circ, segs, cur = [], [], None
    for i in range(a, b):
        t = L[i].split()
        if len(t) >= 6 and t[-1] == "a":
            circ.append((float(t[0]), float(t[1]), float(t[2])))
        elif len(t) == 3 and t[2] == "m":
            cur = (float(t[0]), float(t[1]))
        elif len(t) == 3 and t[2] == "l" and cur:
            nxt = (float(t[0]), float(t[1]))
            segs.append((cur, nxt))
            cur = nxt
    return circ, segs


def cal(p0, v0, p1, v1):
    s = (p1 - p0) / (v1 - v0)
    return (lambda p: (p - p0) / s), abs(s)


def blocks(L):
    """Data blocks start at 'pathproc ... rectpath' lines; group them per panel by clip rect."""
    idx = [i for i, l in enumerate(L) if l.startswith("pathproc")]
    panels = {}
    for j, i in enumerate(idx):
        key = " ".join(L[i].split()[1:5])
        end = idx[j + 1] if j + 1 < len(idx) else len(L)
        panels.setdefault(key, []).append((i, end))
    return list(panels.values())


# calibration from axis ticks (tick positions read from the EPS, labels 0..14 / 0.0..1.2 etc.)
PANELS = {
    "3b": dict(X=cal(595, 0, 2705, 14), Y=cal(1943, 0, 270, 1.2)),
    "3a": dict(X=cal(594, 0, 2703, 14), Y=cal(1943, 0, 270, 14)),
    "in": dict(X=cal(600, 0, 2739, 3.0), Y=cal(2001, 0, 770, 1.0)),
}


def extract_points(L, rng, X, Y):
    (ca, cb), (ba, bb) = rng[0], rng[1]
    circ, _ = parse(L, ca, cb)
    _, bars = parse(L, ba, bb)
    (fx, _), (fy, sy) = X, Y
    rows = []
    for cx, cy, r in circ:
        ys = [v for s in bars if abs(s[0][0] - cx) < 1 and abs(s[1][0] - cx) < 1
              for v in (s[0][1], s[1][1])]
        sig = (max(ys) - min(ys)) / 2 / sy if ys else np.nan
        rows.append((fx(cx), fy(cy), sig, r / sy))
    return np.array(rows)


def structure_factor_lda(k, mu):
    a = 2 * mu / (HBAR**2 * k**2 / (2 * M_RB87))
    t1 = (3 + a) / (4 * a**2)
    t2 = (3 + 2 * a - a**2) / (16 * a**2.5) * (np.pi + 2 * np.arctan((a - 1) / (2 * np.sqrt(a))))
    return 15 / 4 * (t1 - t2)


def f_lda_khz(k_um, mu_khz=1.91):
    k = k_um * 1e6
    S = structure_factor_lda(k, mu_khz * 1e3 * 2 * np.pi * HBAR)
    return HBAR * k**2 / (2 * M_RB87 * S) / (2 * np.pi) / 1e3


def f_free_khz(k_um):
    return HBAR * (k_um * 1e6)**2 / (2 * M_RB87) / (2 * np.pi) / 1e3


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "arXiv-cond-mat0111438v1_tar.gz"
    out = sys.argv[2] if len(sys.argv) > 2 else "steinhauer2002_fig3_extracted.csv"
    L = body_lines(read_eps(src))
    groups = blocks(L)          # order in file: 3b, 3a, inset
    assert len(groups) == 3, f"expected 3 panels, found {len(groups)}"
    names = ["3b", "3a", "in"]
    res = {}
    for n, g in zip(names, groups):
        res[n] = extract_points(L, g, PANELS[n]["X"], PANELS[n]["Y"])
        assert len(res[n]) == 13, f"panel {n}: {len(res[n])} markers"

    b, a, i = res["3b"], res["3a"], res["in"]
    k = (b[:, 0] + a[:, 0] + i[:, 0]) / 3
    f = b[:, 1] + f_free_khz(k)
    hidden = np.isclose(b[:, 2], b[:, 3], rtol=0.02)
    df = pd.DataFrame(dict(
        k_um_inv=k.round(4), f_khz=f.round(4), sigma_khz=b[:, 2].round(4),
        delta_khz=b[:, 1].round(4), f_panel3a_khz=a[:, 1].round(4),
        sigma_is_upper_bound=hidden)).sort_values("k_um_inv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nwritten: {out}")

    # ---- consistency checks
    print("\nCHECKS")
    print("  max |k spread| between panels  : %.4f um^-1"
          % np.max(np.ptp(np.vstack([a[:, 0], b[:, 0], i[:, 0]]), axis=0)))
    print("  max |f(3a) - f(3b)|             : %.4f kHz" % np.max(np.abs(a[:, 1] - f)))
    vis = ~np.isnan(i[:, 1]) & (i[:, 0] < 3.0)
    print("  max |f(inset) - f(3b)|, k<3     : %.4f kHz" % np.max(np.abs(i[vis, 1] - f[vis])))
    # authors' LDA curve in panel 3b (second block of the first panel)
    _, curve = parse(L, *groups[0][2])
    fx, _ = PANELS["3b"]["X"]; fy, _ = PANELS["3b"]["Y"]
    pts = np.array([(fx(s[0][0]), fy(s[0][1])) for s in curve])
    pts = pts[pts[:, 0] > 0.05]
    d = pts[:, 1] - (f_lda_khz(pts[:, 0]) - f_free_khz(pts[:, 0]))
    print("  authors' LDA curve vs Eq.(4)    : max |diff| = %.4f kHz" % np.max(np.abs(d)))


if __name__ == "__main__":
    main()