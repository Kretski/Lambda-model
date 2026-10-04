#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
triaxis_analyzer_v5_1.py
========================
V5.1 score on top of V5 (subclass; V4 and V5 are not modified).

Change with respect to V5: the delay weight is applied BEFORE taking the maximum.

V5 took the maximum of |corr(tau)| over ALL lags of the 0.25 s window (about
±125 ms with mode='same') and only then multiplied by the weight of THAT lag.
A noise peak at a non-physical lag that is slightly higher than a genuine peak
inside ±10 ms therefore zeroed the score of a real signal, and the statistic
behaved more like a gate ("did the peak land within ±20 ms?") than a measure of
correlation at a physically allowed delay.

V5.1 uses
    coherence = clip(3 * max_tau |corr(tau)| * w(tau), 0, 1)
with the same continuous weight w (1 for |tau| <= 10 ms, linear to 0 at 20 ms),
i.e. the search is restricted to physically possible delays with a soft edge.

All V5 fields are kept; the V5 score is returned as 'triaxis_score_v5' for
comparison. Because the statistic changes, the null distribution, the injection
ROC and every p-value must be recomputed with V5.1 before it is used.

Usage:
  from triaxis_analyzer_v5_1 import TriAxisAnalyzerV51
  analyzer = TriAxisAnalyzerV51({..., 'skip_time_axis': True})
  r = analyzer.analyze(h1, l1, times, gps)    # r['triaxis_score'] is V5.1
"""

import numpy as np
from scipy.signal import correlate

from triaxis_analyzer_v5 import TriAxisAnalyzerV5, DELAY_FULL_MS, DELAY_ZERO_MS


def delay_weights(lags_ms):
    """Vectorised V5 weight: 1 for |d| <= 10 ms, linear to 0 at 20 ms."""
    d = np.abs(np.asarray(lags_ms, float))
    w = 1.0 - (d - DELAY_FULL_MS) / (DELAY_ZERO_MS - DELAY_FULL_MS)
    return np.clip(w, 0.0, 1.0)


class TriAxisAnalyzerV51(TriAxisAnalyzerV5):

    def coherence_axis(self, h1_window, l1_window):
        out = super().coherence_axis(h1_window, l1_window)      # V4/V5 fields unchanged
        h1n = (h1_window - np.mean(h1_window)) / (np.std(h1_window) + 1e-10)
        l1n = (l1_window - np.mean(l1_window)) / (np.std(l1_window) + 1e-10)
        corr = correlate(h1n, l1n, mode='same') / len(h1_window)  # identical to V4
        lags_ms = (np.arange(len(corr)) - len(corr) // 2) / self.sample_rate * 1000.0
        weighted = np.abs(corr) * delay_weights(lags_ms)
        k = int(np.argmax(weighted))
        out['max_corr_weighted'] = float(weighted[k])
        out['delay_ms_weighted'] = float(lags_ms[k])
        out['corr_at_weighted_peak'] = float(np.abs(corr[k]))
        return out

    def compute_scores(self, struct_feat, coh_feat, time_feat):
        scores = super().compute_scores(struct_feat, coh_feat, time_feat)   # V5 (+ legacy V4)
        coherence_v51 = float(np.clip(coh_feat['max_corr_weighted'] * 3.0, 0, 1))
        scores['triaxis_score_v5'] = scores['triaxis_score']
        scores['coherence_score_v5'] = scores['coherence_score']
        scores['coherence_score'] = coherence_v51
        scores['triaxis_score'] = coherence_v51
        if self.skip_time_axis:
            # time features are empty on the fast path, so the V4 legacy score is meaningless
            scores['triaxis_score_v4_legacy'] = None
        return scores
