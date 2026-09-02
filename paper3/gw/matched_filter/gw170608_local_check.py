#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gw170608_local_check.py
=========================
Локална проверка на GW170608 (max_corr=0.317, delay=-10.99ms — ИЗВЪН
физическия ±10ms H1-L1 диапазон, събитие с известни H1 DQ особености).

Въпрос: локалният шум около събитието дава ли подобни корелационни пикове?
  - Ако ДА  -> 0.317 е свойство на локалното детекторно състояние
               (артефакт), събитието излиза от значимите.
  - Ако НЕ  -> 0.317 е локализирано на GPS-а на събитието; остава значимо,
               с бележка за граничния delay.

Метод: същата статистика (best-of-9 offsets, max_corr), приложена на
~80 прозореца на ±30..±130 s от събитието (изключена зона ±5 s около него),
в СЪЩИЯ непрекъснат data сегмент, със същото whitening.

Употреба:
  python gw170608_local_check.py
  python gw170608_local_check.py --event GW170814   # контролна проверка
"""

import argparse
import numpy as np
from triaxis_analyzer_v5 import TriAxisAnalyzerV5

GW_EVENTS = {
    "GW150914": 1126259462.4, "GW151012": 1128678900.4, "GW151226": 1135136350.6,
    "GW170104": 1167559936.6, "GW170608": 1180922494.5, "GW170729": 1185389807.3,
    "GW170809": 1186302519.8, "GW170814": 1186741861.5, "GW170817": 1187008882.4,
    "GW170818": 1187058327.1, "GW170823": 1187529256.5,
}

SAMPLE_RATE = 4096
OFFSET_GRID_MS = [-100, -75, -50, -25, 0, 25, 50, 75, 100]
LOCAL_SPAN = 130.0     # s от всяка страна
EXCLUDE = 5.0          # s изключена зона около събитието
STEP = 3.0             # s между локалните прозорци

CONFIG = {
    'sample_rate': SAMPLE_RATE, 'window_size': 0.25,
    'bandpass_low': 30, 'bandpass_high': 500,
    'viterbi_freq_penalty': 50.0, 'skip_time_axis': True
}


def best_of_9(analyzer, h1, l1, times, center):
    best = None
    for off in OFFSET_GRID_MS:
        r = analyzer.analyze(h1, l1, times, center + off / 1000.0)
        if r and (best is None or r['max_corr'] > best['max_corr']):
            best = r
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="GW170608")
    args = ap.parse_args()
    gps = GW_EVENTS[args.event]

    from gwpy.timeseries import TimeSeries
    print(f"Fetch {args.event} ±{LOCAL_SPAN + 10:.0f}s (един непрекъснат сегмент)...")
    h1 = TimeSeries.fetch_open_data("H1", gps - LOCAL_SPAN - 10, gps + LOCAL_SPAN + 10,
                                    sample_rate=SAMPLE_RATE, cache=False)
    l1 = TimeSeries.fetch_open_data("L1", gps - LOCAL_SPAN - 10, gps + LOCAL_SPAN + 10,
                                    sample_rate=SAMPLE_RATE, cache=False)
    h1_p = h1.whiten(fftlength=0.25, overlap=0.1).bandpass(30, 500)
    l1_p = l1.whiten(fftlength=0.25, overlap=0.1).bandpass(30, 500)
    h1s, l1s, times = h1_p.value, l1_p.value, h1_p.times.value

    analyzer = TriAxisAnalyzerV5(CONFIG)

    # Събитието — същата процедура
    ev = best_of_9(analyzer, h1s, l1s, times, gps)
    print(f"\n{args.event}: max_corr={ev['max_corr']:.3f}  "
          f"delay={ev['delay_ms']:+.2f}ms  width={ev['width_ms']:.1f}ms")

    # Локален mini-null
    offsets = [o for o in np.arange(-LOCAL_SPAN, LOCAL_SPAN + 0.1, STEP)
               if abs(o) > EXCLUDE]
    loc_corr, loc_delay = [], []
    for o in offsets:
        r = best_of_9(analyzer, h1s, l1s, times, gps + o)
        if r is None:
            continue
        loc_corr.append(r['max_corr'])
        loc_delay.append(r['delay_ms'])
    loc_corr = np.array(loc_corr)
    loc_delay = np.array(loc_delay)
    n = len(loc_corr)

    p_local = (1 + np.sum(loc_corr >= ev['max_corr'])) / (1 + n)
    print(f"\nЛокален null: {n} прозореца на ±{EXCLUDE:.0f}..{LOCAL_SPAN:.0f}s")
    print(f"  max_corr: median={np.median(loc_corr):.3f}  "
          f"p95={np.percentile(loc_corr, 95):.3f}  max={loc_corr.max():.3f}")
    print(f"  Събитието срещу локалния null: перцентил "
          f"{100 * np.mean(loc_corr < ev['max_corr']):.1f}%  p={p_local:.4f}")

    near_edge = np.sum(np.abs(loc_delay) > 9.0)
    print(f"  Локални delay-и: median|d|={np.median(np.abs(loc_delay)):.2f}ms, "
          f"{near_edge}/{n} прозореца с |d|>9ms")

    print("\nИНТЕРПРЕТАЦИЯ:")
    if p_local < 0.02:
        print(f"  Пикът е ЛОКАЛИЗИРАН на GPS-а на събитието (p_local={p_local:.4f}).")
        print("  Локалният шум НЕ произвежда подобни корелации -> резултатът")
        print("  за събитието устоява; delay-ят остава бележка, не дисквалификация.")
    elif p_local < 0.15:
        print(f"  Гранично (p_local={p_local:.4f}) — локалният шум се доближава.")
        print("  Препоръка: повтори с LOCAL_SPAN=300 за по-плътен локален null.")
    else:
        print(f"  Локалният шум произвежда сравними пикове (p_local={p_local:.4f}).")
        print("  max_corr на събитието е свойство на локалното детекторно")
        print("  състояние -> изважда се от значимите с бележка.")


if __name__ == "__main__":
    main()
