#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recompute_background_v5_1.py
============================
Recompute the global background (global_background_v3.jsonl) with V5 and V5.1 on
exactly the same windows: same epochs, same window GPS times, same 9-offset grid,
same data processing as analyze_events_v5.py (20 s around the epoch, whitening with
fftlength 0.25 s, band-pass 30–500 Hz).

V5 is recomputed as a reproduction check: its best-of-9 score must match the stored
'triaxis_score_maxoff' of the original file. If it does not, the data processing
differs from the original background script and the V5.1 numbers are NOT comparable.

Output: background_v5_1.jsonl (one line per window, both versions).

Usage:
  python recompute_background_v5_1.py --max-epochs 50      # quick reproduction check
  python recompute_background_v5_1.py                      # all 1700 epochs (resumes)
"""

import argparse
import collections
import json
import os
import time

import numpy as np
from triaxis_analyzer_v5_1 import TriAxisAnalyzerV51

SAMPLE_RATE = 4096
OFFSET_GRID_MS = [-100, -75, -50, -25, 0, 25, 50, 75, 100]
CONFIG = {'sample_rate': SAMPLE_RATE, 'window_size': 0.25, 'bandpass_low': 30,
          'bandpass_high': 500, 'viterbi_freq_penalty': 50.0, 'skip_time_axis': True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", default="global_background_v3.jsonl")
    ap.add_argument("--out", default="background_v5_1.jsonl")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--half", type=float, default=10.0, help="data half-length around the epoch [s]")
    args = ap.parse_args()

    from gwpy.timeseries import TimeSeries
    rows = [json.loads(l) for l in open(args.bg) if l.strip()]
    by_epoch = collections.OrderedDict()
    for r in rows:
        by_epoch.setdefault(r['epoch_gps'], []).append(r)
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            if l.strip():
                done.add(json.loads(l)['epoch_gps'])
    an = TriAxisAnalyzerV51(CONFIG)
    epochs = list(by_epoch)[:args.max_epochs] if args.max_epochs else list(by_epoch)
    t0 = time.time()
    n_new = 0
    with open(args.out, "a") as fo:
        for i, ep in enumerate(epochs, 1):
            if ep in done:
                continue
            try:
                h1 = TimeSeries.fetch_open_data("H1", ep - args.half, ep + args.half, sample_rate=SAMPLE_RATE, cache=True)
                l1 = TimeSeries.fetch_open_data("L1", ep - args.half, ep + args.half, sample_rate=SAMPLE_RATE, cache=True)
            except Exception as e:
                print(f"[{i}/{len(epochs)}] {ep}: fetch failed ({e})", flush=True)
                continue
            h1p = h1.whiten(fftlength=0.25, overlap=0.1).bandpass(30, 500)
            l1p = l1.whiten(fftlength=0.25, overlap=0.1).bandpass(30, 500)
            h1s, l1s, times = h1p.value, l1p.value, h1p.times.value
            for r in by_epoch[ep]:
                b5 = b51 = None
                for off in OFFSET_GRID_MS:
                    a = an.analyze(h1s, l1s, times, r['window_gps'] + off / 1000.0)
                    if a is None:
                        continue
                    if b5 is None or a['triaxis_score_v5'] > b5[0]:
                        b5 = (a['triaxis_score_v5'], off)
                    if b51 is None or a['triaxis_score'] > b51[0]:
                        b51 = (a['triaxis_score'], off, a['delay_ms_weighted'])
                if b51 is None:
                    continue
                fo.write(json.dumps({'epoch_gps': ep, 'window_gps': r['window_gps'],
                                     'v5_stored': r['triaxis_score_maxoff'], 'v5': b5[0], 'v5_offset_ms': b5[1],
                                     'v5_1': b51[0], 'v5_1_offset_ms': b51[1], 'v5_1_delay_ms': b51[2]}) + "\n")
            fo.flush()
            n_new += 1
            if n_new % 10 == 0:
                el = time.time() - t0
                print(f"[{i}/{len(epochs)}] {n_new} epochs in {el:.0f} s "
                      f"(about {el / n_new * (len(epochs) - i) / 60:.0f} min left)", flush=True)

    res = [json.loads(l) for l in open(args.out) if l.strip()]
    dv = np.array([abs(x['v5'] - x['v5_stored']) for x in res])
    print(f"\n{len(res)} windows. Reproduction of V5: max |V5 − stored| = {dv.max():.2e}, "
          f"fraction matching within 1e-6: {np.mean(dv < 1e-6):.3f}")
    v5 = np.array([x['v5'] for x in res]); v51 = np.array([x['v5_1'] for x in res])
    print(f"null zeros: V5 {np.mean(v5 == 0):.3f}  V5.1 {np.mean(v51 == 0):.3f};  "
          f"median V5 {np.median(v5):.3f}  V5.1 {np.median(v51):.3f}")


if __name__ == "__main__":
    main()
