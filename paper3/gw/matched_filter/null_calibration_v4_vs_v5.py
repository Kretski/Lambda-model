#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
null_calibration_v4_vs_v5.py
=============================
Директен тест на хипотезата "V5 подобрява null калибрацията, не raw
detection power" — сравнява разпределенията на V4 (v4_legacy_maxoff) и
V5 (triaxis_score_maxoff) background score-ове от global_background_v3.jsonl.

Не изисква нови injections — ползва вече наличния 10,200-реализационен null.

Употреба:
  python null_calibration_v4_vs_v5.py --bg global_background_v3.jsonl
"""

import json
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", default="global_background_v3.jsonl")
    args = ap.parse_args()

    bg = load_jsonl(args.bg)
    v5 = np.array([r['triaxis_score_maxoff'] for r in bg])
    v4 = np.array([r['v4_legacy_maxoff'] for r in bg])
    n = len(bg)
    print(f"Null реализации: {n}\n")

    for name, x in [("V4 (стар)", v4), ("V5 (модифициран)", v5)]:
        print(f"--- {name} ---")
        print(f"  mean={x.mean():.4f}  std={x.std():.4f}  "
              f"median={np.median(x):.4f}")
        print(f"  min={x.min():.4f}  max={x.max():.4f}")
        for pct in [50, 90, 95, 99, 99.9]:
            print(f"  P{pct:<5} = {np.percentile(x, pct):.4f}")
        # skewness / excess kurtosis (тежки опашки = потенциален проблем)
        mu, sigma = x.mean(), x.std()
        skew = np.mean(((x - mu) / sigma) ** 3) if sigma > 0 else float('nan')
        kurt = np.mean(((x - mu) / sigma) ** 4) - 3 if sigma > 0 else float('nan')
        print(f"  skewness={skew:+.3f}  excess kurtosis={kurt:+.3f}  "
              f"(по-голяма кюртоза = по-тежки опашки = по-рисков null)")
        print()

    # FAR при различни прагове (за референция: публикуваният detection
    # threshold е network SNR ~= 12-15, тук гледаме директно score-based
    # thresholds за да сравним V4/V5 на еднаква "operating point" логика)
    print("=== False-alarm rate при различни percentile прагове ===")
    print(f"{'percentile':>10}  {'threshold V4':>13}  {'FAR V4':>10}  "
          f"{'threshold V5':>13}  {'FAR V5':>10}")
    for pct in [90, 95, 99, 99.5, 99.9]:
        thr_v4 = np.percentile(v4, pct)
        thr_v5 = np.percentile(v5, pct)
        far_v4 = np.mean(v4 >= thr_v4)
        far_v5 = np.mean(v5 >= thr_v5)
        print(f"{pct:>10.1f}  {thr_v4:>13.4f}  {far_v4:>10.4f}  "
              f"{thr_v5:>13.4f}  {far_v5:>10.4f}")

    print("\n(По конструкция FAR при собствения percentile е ~еднакво за "
          "двата — по-показателно е да фиксираш ЕДИН абсолютен праг и "
          "видиш дали V4 и V5 дават различен FAR НА НЕГО, или да гледаш "
          "кюртозата/опашките по-горе за heavy-tail поведение.)")

    # Kolmogorov-Smirnov: различни ли са двете разпределения формално
    try:
        from scipy.stats import ks_2samp
        stat, pval = ks_2samp(v4, v5)
        print(f"\nKS test (V4 vs V5 null разпределения): D={stat:.4f}, p={pval:.2e}")
        if pval < 0.01:
            print("-> Разпределенията са статистически различни (p<0.01).")
        else:
            print("-> Няма силно доказателство за различна форма на разпределенията.")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
