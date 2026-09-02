#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
triaxis_analyzer_v5.py
========================
V5 score поверх V4 (subclass — V4 кодът не се пипа).

Промени спрямо V4, мотивирани от аутопсията и глобалния null (юли 2026):

1. КОХЕРЕНТНОСТ — непрекъснат delay член вместо бинарен.
   V4: coherence = clip(max_corr*3)*0.5 + (0.5 if |delay|<10ms else 0)
   Бинарният член добавяше точно 0.5 и създаваше бимодален null
   (два мода на 0.49/0.70, разлика 0.205 ≈ 0.45*0.5 — потвърдено на
   n=1800). V5: множителна непрекъсната тежест — пълна за |d|<=10ms
   (физическият H1-L1 light-travel диапазон), линейно затихваща до 0
   при 20ms. Null-ът става унимодален, score-ът остава в [0,1].

2. SCORE = САМО КОХЕРЕНТНОСТ. Structure и time осите излизат от
   score-а (остават изчислими за диагностика):
   - structure: 55% от тежестта ѝ е спектрална ентропия ВЪРХУ
     WHITENED данни (whitening-ът изравнява спектъра по дефиниция —
     признакът е константен на реални данни: std=0.008 на n=1800);
     local_consistency е cross-detector; максимумът ѝ го взима
     константен тон, не chirp (autopsy: tone=0.774 > chirp=0.714).
   - time: на синтетика дава най-висок score на ЧИСТ ШУМ (0.815)
     и по-нисък на 20σ chirp (0.664); corr с total на null = 0.089.
   Нови structure признаци (flux/curvature/anisotropy) влизат в
   score-а само след injection ROC gate с AUC доказуемо > 0.5.

3. skip_time_axis (config, default False) — прескача Viterbi
   (най-скъпото изчисление, O(n_f^2 * n_t) python цикли) за бързо
   генериране на голям null. Time признаците се попълват с
   _empty_time_features, интерфейсът не се променя.

Употреба:
  from triaxis_analyzer_v5 import TriAxisAnalyzerV5
  analyzer = TriAxisAnalyzerV5({..., 'skip_time_axis': True})
  r = analyzer.analyze(h1, l1, times, gps)   # r['triaxis_score'] е V5
  # r['triaxis_score_v4_legacy'] пази старата формула за сравнение
"""

import numpy as np
from triaxis_analyzer_v4 import TriAxisAnalyzerV4

DELAY_FULL_MS = 10.0   # пълна тежест до тук (H1-L1 light travel ~10 ms)
DELAY_ZERO_MS = 20.0   # нула тежест от тук нататък


def delay_weight(delay_ms):
    """Непрекъсната тежест на закъснението: 1 в [0,10ms], линейно към 0 при 20ms."""
    d = abs(float(delay_ms))
    if d <= DELAY_FULL_MS:
        return 1.0
    if d >= DELAY_ZERO_MS:
        return 0.0
    return 1.0 - (d - DELAY_FULL_MS) / (DELAY_ZERO_MS - DELAY_FULL_MS)


class TriAxisAnalyzerV5(TriAxisAnalyzerV4):

    def __init__(self, config=None):
        super().__init__(config)
        cfg = config or {}
        self.skip_time_axis = bool(cfg.get('skip_time_axis', False))

    def compute_scores(self, struct_feat, coh_feat, time_feat):
        # Легасі V4 скорове — за сравнение/диагностика
        legacy = super().compute_scores(struct_feat, coh_feat, time_feat)

        w = delay_weight(coh_feat['delay_ms'])
        coherence_v5 = float(np.clip(coh_feat['max_corr'] * 3.0, 0, 1) * w)

        return {
            'structure_score': legacy['structure_score'],   # диагностика
            'time_score': legacy['time_score'],             # диагностика
            'coherence_score': coherence_v5,
            'delay_weight': w,
            'triaxis_score': coherence_v5,                  # V5: score = coherence
            'triaxis_score_v4_legacy': legacy['triaxis_score'],
        }

    def analyze(self, h1_strain, l1_strain, times, center_time):
        if not self.skip_time_axis:
            return super().analyze(h1_strain, l1_strain, times, center_time)

        # Бърз път: без Viterbi
        h1_window, l1_window, window_times = self.extract_window(
            h1_strain, l1_strain, times, center_time)
        if len(h1_window) < 64:
            return None
        struct_features = self.structure_axis(h1_window, l1_window)
        coh_features = self.coherence_axis(h1_window, l1_window)
        time_features = self._empty_time_features()
        scores = self.compute_scores(struct_features, coh_features, time_features)

        out = {}
        out.update(struct_features)
        out.update(coh_features)
        out.update(time_features)
        out.update(scores)
        out['window_size'] = len(h1_window)
        out['window_times'] = window_times
        out['h1_window'] = h1_window
        out['l1_window'] = l1_window
        return out
