#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py
=================
Публикационна многопанелна фигура (стил PRD/CoAD) от записаните JSONL данни.

Панели:
  (a) ROC криви по морфология (BBH / sine-Gaussian / WNB) срещу общия null
  (b) Detection efficiency vs SNR по морфология, биномни error bars,
      праг FAP=1% (p99 на null)
  (c) Null хистограма + 11-те GWTC-1 събития (както events_vs_null, изчистена)
  (d) Delay грешка |измерен - истински| по морфология (уверени възстановявания)
Опционално (e): twin разпределения срещу реалните събития, ако twin файловете
съществуват.

Употреба:
  python make_figures.py
Очаква в текущата директория:
  global_background_v3.jsonl, injections.jsonl, inj_sg.jsonl, inj_wnb.jsonl,
  events_v5.json; опционално inj_gw151226_twin.jsonl, inj_gw150914_twin.jsonl,
  inj_gw170608_twin.jsonl
Изход: figure_validation.png (300 dpi) + figure_validation.pdf
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG = "global_background_v3.jsonl"
INJ_SETS = [("BBH", "injections_v2.jsonl", "tab:blue"),
            ("Sine-Gaussian", "inj_sg_v2.jsonl", "tab:orange"),
            ("White-noise burst", "inj_wnb_v2.jsonl", "tab:green")]
TWINS = [("GW151226 twin", "inj_gw151226_twin_v2.jsonl", "GW151226"),
         ("GW150914 twin", "inj_gw150914_twin_v2.jsonl", "GW150914"),
         ("GW170608 twin", "inj_gw170608_twin_v2.jsonl", "GW170608")]
EVENTS = "events_v5.json"
SNR_BINS = [(5, 8), (8, 12), (12, 18), (18, 30)]


def jl(path):
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def roc(pos, neg):
    thr = np.sort(np.unique(np.concatenate([pos, neg])))[::-1]
    tpr = [np.mean(pos >= t) for t in thr]
    fpr = [np.mean(neg >= t) for t in thr]
    x = np.concatenate([[0], fpr, [1]])
    y = np.concatenate([[0], tpr, [1]])
    auc = np.trapezoid(y, x)
    return x, y, auc


def main():
    bg = np.array([r['max_corr_maxoff'] for r in jl(BG)])
    thr99 = np.percentile(bg, 99)
    ev = json.load(open(EVENTS))['events']

    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.26)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # ---------- (a) ROC ----------
    for name, path, col in INJ_SETS:
        if not os.path.exists(path):
            continue
        ic = np.array([r['max_corr'] for r in jl(path)])
        x, y, auc = roc(ic, bg)
        axA.plot(x, y, color=col, lw=2, label=f"{name} (AUC={auc:.3f})")
    axA.plot([0, 1], [0, 1], 'k:', lw=1)
    axA.set_xlabel("False positive rate")
    axA.set_ylabel("True positive rate")
    axA.set_title("(a) ROC vs. global null (n=10{,}200)".replace("{,}", ","))
    axA.legend(loc="lower right", fontsize=9)
    axA.set_xlim(0, 1); axA.set_ylim(0, 1.02)

    # ---------- (b) efficiency vs SNR ----------
    for name, path, col in INJ_SETS:
        if not os.path.exists(path):
            continue
        recs = jl(path)
        snr = np.array([r['snr_net'] for r in recs])
        ic = np.array([r['max_corr'] for r in recs])
        xs, ys, es = [], [], []
        for a, b in SNR_BINS:
            m = (snr >= a) & (snr < b)
            if m.sum() < 8:
                continue
            eff = np.mean(ic[m] >= thr99)
            xs.append((a + b) / 2)
            ys.append(eff)
            es.append(np.sqrt(eff * (1 - eff) / m.sum()))
        axB.errorbar(xs, ys, yerr=es, color=col, marker='o', lw=2,
                     capsize=3, label=name)
    axB.set_xlabel("Network SNR (injected)")
    axB.set_ylabel("Detection efficiency (FAP = 1%)")
    axB.set_title("(b) Efficiency at p99(null) threshold")
    axB.set_ylim(-0.03, 1.05)
    axB.axhline(1.0, color='gray', lw=0.5, ls=':')
    axB.legend(fontsize=9, loc="lower right")

    # ---------- (c) null + events ----------
    axC.hist(bg, bins=60, color='0.75', density=True,
             label="Global null (max-over-9)")
    axC.axvline(thr99, color='k', ls='--', lw=1, label="p99(null)")
    sig = ["GW150914", "GW170814", "GW170608", "GW170104", "GW170729"]
    for name, e in sorted(ev.items(), key=lambda x: x[1]['max_corr']):
        col = 'tab:red' if name in sig else 'tab:gray'
        axC.axvline(e['max_corr'], color=col, lw=1.6, alpha=0.9)
        axC.text(e['max_corr'], axC.get_ylim()[1] * 0.98, name.replace("GW", ""),
                 rotation=90, va='top', ha='right', fontsize=7,
                 color=col)
    axC.set_xlabel("H1–L1 max cross-correlation (best-of-9 offsets)")
    axC.set_ylabel("Density")
    axC.set_title("(c) GWTC-1 events vs. global null "
                  "(red: Holm-significant)")
    axC.legend(fontsize=9)

    # ---------- (d) delay accuracy ----------
    for name, path, col in INJ_SETS:
        if not os.path.exists(path):
            continue
        recs = jl(path)
        conf = [r for r in recs if r['max_corr'] >= thr99]
        if len(conf) < 5:
            continue
        err = np.abs([r['delay_ms_measured'] - r['delay_true_ms']
                      for r in conf])
        err = np.clip(err, 1e-3, None)
        axD.hist(np.log10(err), bins=24, alpha=0.55, color=col,
                 label=f"{name} (median={np.median(err):.2f} ms, n={len(conf)})")
    axD.axvline(np.log10(2.0), color='k', ls='--', lw=1, label="2 ms")
    axD.set_xlabel(r"log$_{10}$ |delay error| (ms)")
    axD.set_ylabel("Count")
    axD.set_title("(d) Delay reconstruction (confident recoveries)")
    axD.legend(fontsize=8)

    fig.suptitle("Validation of the max-correlation statistic on LIGO O1/O2 data",
                 fontsize=13, y=0.995)
    fig.savefig("figure_validation.png", dpi=300, bbox_inches='tight')
    fig.savefig("figure_validation.pdf", bbox_inches='tight')
    print("Записано: figure_validation.png / .pdf")

    # ---------- (e) twin панел, отделна фигура ----------
    have = [(t, p, evn) for t, p, evn in TWINS if os.path.exists(p)]
    if have:
        fig2, ax = plt.subplots(figsize=(7.5, 4.6))
        for i, (title, path, evname) in enumerate(have):
            tc = np.array([r['max_corr'] for r in jl(path)])
            y = np.full_like(tc, i, dtype=float) + (np.random.default_rng(0)
                                                    .uniform(-0.12, 0.12, len(tc)))
            ax.scatter(tc, y, s=18, alpha=0.65, color='tab:blue',
                       label="Twin injections" if i == 0 else None)
            ax.scatter([np.median(tc)], [i], marker='|', s=500, color='tab:blue')
            if evname in ev:
                ax.scatter([ev[evname]['max_corr']], [i], marker='*', s=180,
                           color='tab:red', zorder=5,
                           label="Real event" if i == 0 else None)
        ax.axvline(thr99, color='k', ls='--', lw=1, label="p99(null)")
        ax.set_yticks(range(len(have)))
        ax.set_yticklabels([t for t, _, _ in have])
        ax.set_xlabel("H1–L1 max cross-correlation (best-of-9 offsets)")
        ax.set_title("Event-twin injections vs. real events")
        ax.legend(fontsize=9, loc="lower right")
        fig2.tight_layout()
        fig2.savefig("figure_twins.png", dpi=300, bbox_inches='tight')
        fig2.savefig("figure_twins.pdf", bbox_inches='tight')
        print(f"Записано: figure_twins.png / .pdf ({len(have)} twin сета)")


if __name__ == "__main__":
    main()
