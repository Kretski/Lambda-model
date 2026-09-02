#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
span_scan.py
==============
Тест на self-whitening хипотезата: score-ът на силните събития расте ли
с дължината на whitening контекста?

За всяко от 11-те GWTC-1 събития: fetch ±SPAN s, идентична обработка
(whiten 0.25/0.1, bandpass 30-500), best-of-9 max_corr — при
SPAN ∈ {10, 20, 45, 90} s.

Интерпретация:
  - Ако score расте със span за силните събития и е плосък за слабите
    -> self-whitening потвърден; null/event сравнението е консервативно
    (null няма сигнал за потискане); robustness секцията се пише с
    тази крива.
  - Ако е плоско навсякъде -> разминаването 0.338/0.525 идва от друго
    и трябва да го намерим, преди каквото и да е друго.

Употреба: python span_scan.py
"""

import numpy as np
from triaxis_analyzer_v5 import TriAxisAnalyzerV5

GW_EVENTS = {
    "GW150914": 1126259462.4, "GW151012": 1128678900.4, "GW151226": 1135136350.6,
    "GW170104": 1167559936.6, "GW170608": 1180922494.5, "GW170729": 1185389807.3,
    "GW170809": 1186302519.8, "GW170814": 1186741861.5, "GW170817": 1187008882.4,
    "GW170818": 1187058327.1, "GW170823": 1187529256.5,
}
SPANS = [10, 20, 45, 90]
SAMPLE_RATE = 4096
OFFSET_GRID_MS = [-100, -75, -50, -25, 0, 25, 50, 75, 100]
CONFIG = {'sample_rate': SAMPLE_RATE, 'window_size': 0.25,
          'bandpass_low': 30, 'bandpass_high': 500,
          'viterbi_freq_penalty': 50.0, 'skip_time_axis': True}


def score(analyzer, h1, l1, times, gps):
    best = 0.0
    for off in OFFSET_GRID_MS:
        r = analyzer.analyze(h1, l1, times, gps + off / 1000.0)
        if r and r['max_corr'] > best:
            best = r['max_corr']
    return best


def main():
    from gwpy.timeseries import TimeSeries
    analyzer = TriAxisAnalyzerV5(CONFIG)

    print(f"{'Събитие':<10} " + " ".join(f"±{s:>3}s" for s in SPANS))
    print("-" * (12 + 7 * len(SPANS)))
    rows = {}
    for name, gps in GW_EVENTS.items():
        vals = []
        for span in SPANS:
            try:
                h1 = TimeSeries.fetch_open_data("H1", gps - span, gps + span,
                                                sample_rate=SAMPLE_RATE,
                                                cache=False)
                l1 = TimeSeries.fetch_open_data("L1", gps - span, gps + span,
                                                sample_rate=SAMPLE_RATE,
                                                cache=False)
                h1p = h1.whiten(fftlength=0.25, overlap=0.1).bandpass(30, 500)
                l1p = l1.whiten(fftlength=0.25, overlap=0.1).bandpass(30, 500)
                vals.append(score(analyzer, h1p.value, l1p.value,
                                  h1p.times.value, gps))
            except Exception as e:
                vals.append(float('nan'))
                print(f"  ({name} ±{span}s: {type(e).__name__})")
        rows[name] = vals
        print(f"{name:<10} " + " ".join(f"{v:.3f}" for v in vals))

    print("\nОтносителна промяна ±90s спрямо ±10s (положително = self-whitening):")
    for name, vals in sorted(rows.items(),
                             key=lambda kv: -(kv[1][-1] - kv[1][0])):
        d = vals[-1] - vals[0]
        print(f"  {name}: {d:+.3f}  ({100*d/max(vals[0],1e-9):+.0f}%)")


if __name__ == "__main__":
    main()
