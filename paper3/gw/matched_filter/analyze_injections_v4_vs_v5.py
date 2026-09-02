#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_injections_v4_vs_v5.py
================================
A/B сравнение: TriAxis V4 (legacy composite score) срещу V5 (coherence-only,
continuous delay taper) — AUC по SNR bin, специално фокус върху слабия
регион SNR<8 където публикуваният V5 baseline е "ефективно сляп"
(BBH AUC 0.577, white-noise-burst AUC 0.523).

И двата score-а идват от ЕДНА И СЪЩА offset selection (по V5 правилото,
консистентно с null-а/events), т.е. сравнението изолира разликата в
SCORING формулата (binary vs continuous delay weight, coherence-only vs
3-axis blend), не разлика в кой прозорец е избран.

Изисква injections.jsonl да съдържа полето 'triaxis_v4_legacy' —
изисква патчнат injection_recovery.py (виж чат инструкциите) и
regenerated injections.jsonl.

Употреба:
  python analyze_injections_v4_vs_v5.py --inj injections.jsonl --bg global_background_v3.jsonl
"""

import json, argparse
import numpy as np


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
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


def bootstrap_ci(pos, neg, rng, n_boot=500):
    boots = [auc_rank(rng.choice(pos, len(pos)), rng.choice(neg, len(neg)))
              for _ in range(n_boot)]
    return np.percentile(boots, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inj", default="injections.jsonl")
    ap.add_argument("--bg", default="global_background_v3.jsonl")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    inj = load_jsonl(args.inj)
    bg = load_jsonl(args.bg)

    if not inj or not bg:
        print("Липсват данни."); return

    if 'triaxis_v4_legacy' not in inj[0]:
        print("ERROR: injections.jsonl няма поле 'triaxis_v4_legacy'.")
        print("Патчни injection_recovery.py (виж чат инструкциите) и "
              "regenerate injections.jsonl, после пусни отново.")
        return

    snr = np.array([r['snr_net'] for r in inj])
    v5_inj = np.array([r['triaxis_v5'] for r in inj])
    v4_inj = np.array([r['triaxis_v4_legacy'] for r in inj])
    morph = [r.get('morphology', 'bbh') for r in inj]
    from collections import Counter
    print(f"Морфологии: {dict(Counter(morph))}")
    print(f"Инжекции: {len(inj)}   Null: {len(bg)}")

    v5_bg = np.array([r['triaxis_score_maxoff'] for r in bg])
    v4_bg = np.array([r['v4_legacy_maxoff'] for r in bg])

    rng = np.random.default_rng(0)

    print("\n=== Общ AUC (SNR 5-30) ===")
    auc_v5 = auc_rank(v5_inj, v5_bg)
    lo5, hi5 = bootstrap_ci(v5_inj, v5_bg, rng)
    print(f"  V5 (модифициран): AUC={auc_v5:.3f}  [95% CI {lo5:.3f}-{hi5:.3f}]")

    auc_v4 = auc_rank(v4_inj, v4_bg)
    lo4, hi4 = bootstrap_ci(v4_inj, v4_bg, rng)
    print(f"  V4 (стар):        AUC={auc_v4:.3f}  [95% CI {lo4:.3f}-{hi4:.3f}]")

    print("\n=== AUC по SNR bin (ключовото сравнение) ===")
    print(f"{'SNR bin':<10} {'n':>5}  {'AUC V4 (стар)':>15}  {'AUC V5 (модиф.)':>17}  {'delta (V5-V4)':>14}")
    bins = [(4, 6), (6, 8), (8, 10), (10, 12), (12, 15), (15, 18), (18, 24), (24, 30)]
    rows = []
    for a, b in bins:
        m = (snr >= a) & (snr < b)
        if m.sum() < 10:
            continue
        auc4 = auc_rank(v4_inj[m], v4_bg)
        auc5 = auc_rank(v5_inj[m], v5_bg)
        delta = auc5 - auc4
        rows.append((a, b, m.sum(), auc4, auc5, delta))
        flag = "  <-- weak-SNR regime" if b <= 8 else ""
        print(f"{a:>3}-{b:<4}   {m.sum():>5}  {auc4:>15.3f}  {auc5:>17.3f}  {delta:>+14.3f}{flag}")

    weak = [r for r in rows if r[1] <= 8]
    if weak:
        mean_delta_weak = np.mean([r[5] for r in weak])
        print(f"\nСреден AUC delta (V5-V4) за SNR<=8: {mean_delta_weak:+.3f}")
        if mean_delta_weak > 0.02:
            print("-> V5 показва измерима подобрена чувствителност при слаб SNR.")
        elif mean_delta_weak < -0.02:
            print("-> V5 е ПО-ЗЛЕ от V4 при слаб SNR (проверете дали structure/time "
                  "осите на V4 носят реална информация, изгубена при coherence-only V5).")
        else:
            print("-> Няма измерима разлика при слаб SNR; V5's предимство (ако има) "
                  "идва от null unimodality/calibration, не от суровата чувствителност.")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        centers = [(a + b) / 2 for a, b, *_ in rows]
        auc4s = [r[3] for r in rows]
        auc5s = [r[4] for r in rows]
        ax.plot(centers, auc4s, 'o-', label='V4 (стар)')
        ax.plot(centers, auc5s, 's-', label='V5 (модифициран)')
        ax.axhline(0.5, color='gray', ls=':', label='случайно (AUC=0.5)')
        ax.axvline(8, color='red', ls='--', alpha=0.5, label='SNR=8 праг (публикуван)')
        ax.set_xlabel('Network SNR (инжектиран)')
        ax.set_ylabel('AUC')
        ax.set_title('V4 vs V5: чувствителност при слаб SNR')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig('v4_vs_v5_weak_snr.png', dpi=130)
        print("\nЗаписано: v4_vs_v5_weak_snr.png")


if __name__ == "__main__":
    main()
