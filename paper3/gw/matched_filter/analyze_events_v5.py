#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_events_v5.py
======================
Преизчислява 11-те GWTC-1 събития с V5 score (best-of-9 offsets, същата
процедура като null-а в global_background_v3), и записва events_v5.json
в формат, съвместим с compare_events_to_null.py.

Употреба:
  python analyze_events_v5.py
  python compare_events_to_null.py --events events_v5.json --bg global_background_v3.jsonl --plot
"""

import json
import numpy as np
from triaxis_analyzer_v5 import TriAxisAnalyzerV5

GW_EVENTS = {
    "GW150914": (1126259462.4, "BBH"),
    "GW151012": (1128678900.4, "BBH"),
    "GW151226": (1135136350.6, "BBH"),
    "GW170104": (1167559936.6, "BBH"),
    "GW170608": (1180922494.5, "BBH"),
    "GW170729": (1185389807.3, "BBH"),
    "GW170809": (1186302519.8, "BBH"),
    "GW170814": (1186741861.5, "BBH"),
    "GW170817": (1187008882.4, "BNS"),
    "GW170818": (1187058327.1, "BBH"),
    "GW170823": (1187529256.5, "BBH"),
}

SAMPLE_RATE = 4096
OFFSET_GRID_MS = [-100, -75, -50, -25, 0, 25, 50, 75, 100]

CONFIG = {
    'sample_rate': SAMPLE_RATE,
    'window_size': 0.25,
    'bandpass_low': 30,
    'bandpass_high': 500,
    'viterbi_freq_penalty': 50.0,
    'skip_time_axis': True    # V5 score не ползва time оста
}


def main():
    from gwpy.timeseries import TimeSeries
    analyzer = TriAxisAnalyzerV5(CONFIG)
    out = {'events': {}}

    for name, (gps, typ) in GW_EVENTS.items():
        print(f"{name}...", flush=True)
        try:
            h1 = TimeSeries.fetch_open_data("H1", gps - 10, gps + 10,
                                            sample_rate=SAMPLE_RATE, cache=False)
            l1 = TimeSeries.fetch_open_data("L1", gps - 10, gps + 10,
                                            sample_rate=SAMPLE_RATE, cache=False)
        except Exception as e:
            print(f"  fetch провален: {e}")
            continue

        h1_p = h1.whiten(fftlength=0.25, overlap=0.1).bandpass(30, 500)
        l1_p = l1.whiten(fftlength=0.25, overlap=0.1).bandpass(30, 500)
        h1s, l1s, times = h1_p.value, l1_p.value, h1_p.times.value

        best = None
        for off in OFFSET_GRID_MS:
            r = analyzer.analyze(h1s, l1s, times, gps + off / 1000.0)
            if r and (best is None or r['triaxis_score'] > best['triaxis_score']):
                best = r
                best_off = off
        if best is None:
            print("  анализът върна None за всички offset-и")
            continue

        out['events'][name] = {
            'type': typ,
            'triaxis_score': float(best['triaxis_score']),
            'max_corr': float(best['max_corr']),
            'delay_ms': float(best['delay_ms']),
            'width_ms': float(best['width_ms']),
            'delay_weight': float(best['delay_weight']),
            'v4_legacy': float(best['triaxis_score_v4_legacy']),
            'best_offset_ms': best_off,
        }
        print(f"  V5={best['triaxis_score']:.3f}  max_corr={best['max_corr']:.3f}  "
              f"delay={best['delay_ms']:+.2f}ms (w={best['delay_weight']:.2f})  "
              f"offset={best_off:+d}ms  [v4 legacy={best['triaxis_score_v4_legacy']:.3f}]")

    with open('events_v5.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nЗаписано: events_v5.json ({len(out['events'])} събития)")
    print("Следваща стъпка: python compare_events_to_null.py "
          "--events events_v5.json --bg global_background_v3.jsonl --plot")


if __name__ == "__main__":
    main()
