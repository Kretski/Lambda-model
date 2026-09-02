#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_injections.py
=======================
ROC/AUC и криви на чувствителност за max_corr от injection recovery.

  - ROC/AUC: инжекции (позитиви) срещу глобалния null (негативи),
    двете на max_corr при best-of-9. Bootstrap 95% CI.
  - Detection efficiency vs network SNR при праг = p99 на null-а
    (т.е. FAP=1% per window).
  - Efficiency vs обща маса (за SNR >= 12, където има какво да се дели).
  - Санитарна проверка: измерен срещу истински delay за уверените
    възстановявания (max_corr > праг).

Употреба:
  python analyze_injections.py --inj injections.jsonl --bg global_background_v3.jsonl
"""

import json, argparse
import numpy as np


def load_jsonl(path, key=None):
    out = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                out.append(r[key] if key else r)
            except Exception:
                pass
    return out


def auc_rank(pos, neg):
    """AUC чрез Mann-Whitney (ранков), без параметрични допускания."""
    x = np.concatenate([neg, pos])
    ranks = np.argsort(np.argsort(x)) + 1
    r_pos = ranks[len(neg):].sum()
    n1, n0 = len(pos), len(neg)
    return (r_pos - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inj", default="injections.jsonl")
    ap.add_argument("--bg", default="global_background_v3.jsonl")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    inj = load_jsonl(args.inj)
    bg = np.array(load_jsonl(args.bg, 'max_corr_maxoff'))
    if not inj or len(bg) == 0:
        print("Липсват данни."); return

    ic = np.array([r['max_corr'] for r in inj])
    snr = np.array([r['snr_net'] for r in inj])
    mt = np.array([r.get('mtotal', np.nan) for r in inj])
    morph = [r.get('morphology', 'bbh') for r in inj]
    from collections import Counter
    print(f"Морфологии: {dict(Counter(morph))}")
    print(f"Инжекции: {len(ic)}   Null: {len(bg)}")

    # ROC/AUC + bootstrap CI
    rng = np.random.default_rng(0)
    auc = auc_rank(ic, bg)
    boots = [auc_rank(rng.choice(ic, len(ic)), rng.choice(bg, len(bg)))
             for _ in range(500)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\nAUC (всички инжекции, SNR 5-30): {auc:.3f}  [95% CI {lo:.3f}-{hi:.3f}]")

    # AUC по SNR band — по-информативно от общия
    print("\nAUC по SNR диапазони:")
    for a, b in [(5, 8), (8, 12), (12, 18), (18, 30)]:
        m = (snr >= a) & (snr < b)
        if m.sum() >= 10:
            print(f"  SNR {a:>2}-{b:<2}: AUC={auc_rank(ic[m], bg):.3f}  (n={m.sum()})")

    # Efficiency при FAP=1% праг
    thr = np.percentile(bg, 99)
    print(f"\nПраг p99(null) = {thr:.3f}  ->  detection efficiency:")
    for a, b in [(5, 8), (8, 12), (12, 18), (18, 30)]:
        m = (snr >= a) & (snr < b)
        if m.sum() >= 10:
            eff = np.mean(ic[m] >= thr)
            n = m.sum()
            err = np.sqrt(eff * (1 - eff) / n)
            print(f"  SNR {a:>2}-{b:<2}: {100*eff:5.1f}% ± {100*err:.1f}%  (n={n})")

    print(f"\nEfficiency vs обща маса (SNR >= 12):")
    hi_snr = (snr >= 12) & ~np.isnan(mt)
    for a, b in [(14, 30), (30, 50), (50, 70), (70, 100)]:
        m = hi_snr & (mt >= a) & (mt < b)
        if m.sum() >= 8:
            eff = np.mean(ic[m] >= thr)
            print(f"  Mtot {a:>3}-{b:<3}: {100*eff:5.1f}%  (n={m.sum()})")

    # Санитарен delay тест
    conf = ic >= thr
    if conf.sum() >= 5:
        dtrue = np.array([r['delay_true_ms'] for r in inj])[conf]
        dmeas = np.array([r['delay_ms_measured'] for r in inj])[conf]
        err = dmeas - dtrue
        print(f"\nDelay проверка ({conf.sum()} уверени възстановявания):")
        print(f"  median |измерен - истински| = {np.median(np.abs(err)):.2f} ms")
        print(f"  дял с |грешка| < 2 ms: {100*np.mean(np.abs(err) < 2):.0f}%")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        ax.hist(bg, bins=50, alpha=0.6, density=True, label=f'Null (n={len(bg)})')
        ax.hist(ic, bins=50, alpha=0.6, density=True, label=f'Инжекции (n={len(ic)})')
        ax.axvline(thr, color='k', ls='--', label='p99(null)')
        ax.set_xlabel('max_corr (best-of-9)'); ax.set_ylabel('Плътност')
        ax.legend(); ax.set_title(f'AUC={auc:.3f}')
        ax = axes[1]
        ax.scatter(snr, ic, s=12, alpha=0.5)
        ax.axhline(thr, color='k', ls='--')
        ax.set_xlabel('Network SNR (инжектиран)'); ax.set_ylabel('max_corr')
        ax.set_title('Recovery vs SNR')
        fig.tight_layout(); fig.savefig('injection_recovery.png', dpi=130)
        print("\nЗаписано: injection_recovery.png")


if __name__ == "__main__":
    main()
