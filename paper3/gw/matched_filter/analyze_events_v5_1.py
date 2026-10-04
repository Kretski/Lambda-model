#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_events_v5_1.py
======================
The 11 GWTC-1 events with the V5 and V5.1 scores side by side, using exactly the
same data processing and offset procedure as analyze_events_v5.py (whitening with
fftlength 0.25 s, band-pass 30–500 Hz, best of 9 window offsets).

Each version selects its own best offset. One V5.1 analyzer run gives both scores
(V5.1 keeps the V5 score in 'triaxis_score_v5'), so the two versions see identical
windows.

Output: events_v5_1.json. The V5.1 null distribution and ROC must be recomputed
before any V5.1 p-value is quoted.

Usage:
  python analyze_events_v5_1.py
"""

import json
import numpy as np
from triaxis_analyzer_v5_1 import TriAxisAnalyzerV51

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
    'skip_time_axis': True,
}


def main():
    from gwpy.timeseries import TimeSeries
    analyzer = TriAxisAnalyzerV51(CONFIG)
    out = {'events': {}}

    for name, (gps, typ) in GW_EVENTS.items():
        print(f"{name}...", flush=True)
        try:
            h1 = TimeSeries.fetch_open_data("H1", gps - 10, gps + 10, sample_rate=SAMPLE_RATE, cache=False)
            l1 = TimeSeries.fetch_open_data("L1", gps - 10, gps + 10, sample_rate=SAMPLE_RATE, cache=False)
        except Exception as e:
            print(f"  fetch failed: {e}")
            continue

        h1_p = h1.whiten(fftlength=0.25, overlap=0.1).bandpass(30, 500)
        l1_p = l1.whiten(fftlength=0.25, overlap=0.1).bandpass(30, 500)
        h1s, l1s, times = h1_p.value, l1_p.value, h1_p.times.value

        best5 = best51 = None
        off5 = off51 = None
        for off in OFFSET_GRID_MS:
            r = analyzer.analyze(h1s, l1s, times, gps + off / 1000.0)
            if r is None:
                continue
            if best5 is None or r['triaxis_score_v5'] > best5['triaxis_score_v5']:
                best5, off5 = r, off
            if best51 is None or r['triaxis_score'] > best51['triaxis_score']:
                best51, off51 = r, off
        if best51 is None:
            print("  analysis returned None for all offsets")
            continue

        out['events'][name] = {
            'type': typ,
            'v5': {'score': float(best5['triaxis_score_v5']), 'max_corr': float(best5['max_corr']),
                   'delay_ms': float(best5['delay_ms']), 'delay_weight': float(best5['delay_weight']),
                   'offset_ms': off5},
            'v5_1': {'score': float(best51['triaxis_score']),
                     'max_corr_weighted': float(best51['max_corr_weighted']),
                     'corr_at_peak': float(best51['corr_at_weighted_peak']),
                     'delay_ms': float(best51['delay_ms_weighted']), 'offset_ms': off51},
        }
        print(f"  V5   {best5['triaxis_score_v5']:.3f}  (delay {best5['delay_ms']:+7.2f} ms, w {best5['delay_weight']:.2f}, "
              f"offset {off5:+d} ms)")
        print(f"  V5.1 {best51['triaxis_score']:.3f}  (delay {best51['delay_ms_weighted']:+7.2f} ms, "
              f"|corr| {best51['corr_at_weighted_peak']:.3f}, offset {off51:+d} ms)")

    with open('events_v5_1.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nwritten events_v5_1.json ({len(out['events'])} events)")


if __name__ == "__main__":
    main()
