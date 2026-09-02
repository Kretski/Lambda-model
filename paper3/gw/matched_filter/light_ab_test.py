#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
light_ab_test.py
=================
БЪРЗА, олекотена V4-срещу-V5 проверка на чувствителност при слаб SNR.

НЕ е заместител на пълния injection_recovery.py / analyze_injections_v4_vs_v5.py
pipeline (PyCBC IMRPhenomD, gwpy.whiten() per-injection, best-of-9 offset scan) —
това е публикуемо-качествен, строг тест. Този скрипт е за БЪРЗА първа насока:
работи ли изобщо ефектът (V5 > V4 при слаб SNR или не), преди да чакаш часове
за пълния run.

Ускорения спрямо тежкия pipeline:
  1. Реалният H1/L1 шум се зарежда ЕДИН път (не при всяка инжекция).
  2. PSD (Welch) се смята ЕДИН път за целия зареден сегмент, не наново
     за всяка инжекция.
  3. Whitening + bandpass се прилагат като ЕДИН честотен филтър, изчислен
     веднъж — математически еквивалентно на "инжектирай в суров strain,
     после whitening" (линейна операция, комутира със сумиране), но без
     повторно ASD estimation при всяка инжекция.
  4. Waveform-ът е опростен аналитичен Njutonov inspiral chirp (numpy,
     без PyCBC) — достатъчен за да провокира coherence_axis/max_corr,
     НЕ е физически точен BBH waveform.
  5. Fixed single offset (t=0, мержърът в центъра на прозореца) —
     БЕЗ best-of-9 offset scan. Това подценява и двата детектора
     еднакво спрямо пълния pipeline, но не би трябвало да променя
     ПОСОКАТА на V5-vs-V4 разликата.

Употреба:
  python light_ab_test.py --h1-path H-H1_GWOSC_4KHZ_R1-1126257415-4096.hdf5 \\
                           --l1-path L-L1_GWOSC_4KHZ_R1-1126257415-4096.hdf5 \\
                           --snr-list 4,5,6,7,8,10,12 --n-per-snr 80

Очаквано време: секунди до минути за няколкостотин trials общо (вместо часове).
"""

import argparse
import numpy as np
import h5py
from scipy.signal import welch, correlate
import sys

sys.path.insert(0, ".")
from triaxis_analyzer_v5 import TriAxisAnalyzerV5

WINDOW_SIZE = 0.25   # секунди, колкото window_size в CONFIG на пълния pipeline
CONFIG = {
    'sample_rate': 4096, 'window_size': WINDOW_SIZE,
    'bandpass_low': 30, 'bandpass_high': 500,
    'viterbi_freq_penalty': 50.0, 'skip_time_axis': True,
}


def load_strain(path):
    with h5py.File(path, 'r') as f:
        strain = f['strain']['Strain'][:]
        gps_start = float(f['meta']['GPSstart'][()])
        duration = float(f['meta']['Duration'][()])
    sample_rate = len(strain) / duration
    return np.asarray(strain, dtype=np.float64), gps_start, sample_rate


def compute_whitening_filter(strain, sample_rate, f_low=30.0, f_high=500.0):
    """PSD (Welch) веднъж; връща честотна маска + 1/ASD за whitening+bandpass
    като единичен множител, приложим на всеки FFT сегмент от СЪЩАТА дължина."""
    freqs, psd = welch(strain, fs=sample_rate, nperseg=4096 * 4)
    return freqs, psd


def whiten_and_bandpass(seg, sample_rate, psd_freqs, psd, f_low=30.0, f_high=500.0):
    """Приложи whitening (деление на ASD) + твърд bandpass в честотния domain."""
    n = len(seg)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    psd_interp = np.interp(freqs, psd_freqs, psd)
    psd_interp[psd_interp <= 0] = np.inf
    seg_fd = np.fft.rfft(seg)
    white_fd = seg_fd / np.sqrt(psd_interp)
    mask = (freqs >= f_low) & (freqs <= f_high)
    white_fd = white_fd * mask
    out = np.fft.irfft(white_fd, n=n)
    # нормализация: приблизително unit-variance изход (достатъчно за
    # относително V4-vs-V5 сравнение, не за абсолютна калибрация)
    std = np.std(out)
    if std > 0:
        out = out / std
    return out


def simple_chirp(duration, sample_rate, snr_target, seg_for_psd, psd_freqs, psd,
                  f_low=30.0, f_high=400.0, rng=None):
    """Опростен аналитичен Newtonian inspiral chirp: f(t) ~ (tc-t)^(-3/8),
    амплитуда ~ f(t)^(2/3). Мержърът е в центъра на прозореца.
    НЕ е физически точен IMRPhenomD waveform — само достатъчен за
    провокиране на coherence между H1/L1 копия със същия delay.

    ВАЖНО: `duration` тук е дължината на САМИЯ сигнал (не целия шумов
    сегмент) — трябва да е сравнима с analyzer-ния window_size (0.25s),
    защото TriAxisAnalyzerV5 гледа само тесен прозорец около center.
    По-дълъг сигнал "харчи" SNR бюджета си извън прозореца, който
    детекторът реално вижда, и води до подценена ефективна амплитуда.
    Освен това е ДОБАВЕН gaussian envelope, който локализира енергията
    около мержъра — без него, дори кратък по номинал сигнал би имал
    почти константна амплитуда по цялата си дължина.
    """
    rng = rng or np.random.default_rng()
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate - duration / 2.0  # мержър близо до центъра
    tc = duration / 2.0 - 0.02

    tau = np.clip(tc - t, 1e-4, None)
    f_gw = f_low + (f_high - f_low) * (1.0 - (tau / tau[0]) ** (3.0 / 8.0))
    f_gw = np.clip(f_gw, f_low, f_high)
    phase = 2 * np.pi * np.cumsum(f_gw) / sample_rate
    amp_envelope = (f_gw / f_low) ** (2.0 / 3.0)
    # локализира енергията около мержъра — БЕЗ това, дори кратък сигнал
    # би имал ~константна амплитуда по цялата си дължина и щеше пак да
    # харчи SNR бюджет извън detection window-а
    gauss_env = np.exp(-((t - tc) ** 2) / (2 * (duration / 6.0) ** 2))
    taper = np.ones(n)
    post_merger = t > tc
    taper[post_merger] = np.exp(-(t[post_merger] - tc) / 0.01)
    waveform = amp_envelope * gauss_env * taper * np.sin(phase)
    waveform -= np.mean(waveform)

    # мащабиране до целеви SNR срещу PSD-то на дадения шумов сегмент
    # КЛЮЧОВ ФИКС: rfft трябва да се нормализира с /sample_rate (dt),
    # за да апроксимира физическия непрекъснат Фурие трансформ h̃(f) —
    # същата конвенция като validirania sterile_lambda_analysis.py
    # (seg_fd_physical = np.fft.rfft(seg) / sample_rate). Без това
    # sigmasq излиза сгрешено с фактор sample_rate^2, и изчислената
    # "правилна" амплитуда за дадено SNR излиза ~sample_rate пъти
    # по-малка от физически коректната — точно бъгът, който направи
    # инжекциите практически невидими (label=0 == label=1) в първия run.
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    wf_fd = np.fft.rfft(waveform) / sample_rate
    psd_interp = np.interp(freqs, psd_freqs, psd)
    psd_interp[psd_interp <= 0] = np.inf
    df = freqs[1] - freqs[0]
    band = (freqs >= f_low) & (freqs <= f_high)
    sigmasq = 4.0 * df * np.sum((np.abs(wf_fd[band]) ** 2) / psd_interp[band])
    if sigmasq <= 0 or not np.isfinite(sigmasq):
        return waveform * 0.0
    current_snr_unit = np.sqrt(sigmasq)
    scale = snr_target / current_snr_unit
    return waveform * scale


BURST_DURATION = 0.5  # секунди — сравнимо с analyzer window_size=0.25s;
                       # ПО-ДЪЛЪГ сигнал харчи SNR бюджет извън прозореца,
                       # който детекторът реално вижда (виж бележка в simple_chirp)


def run_trial(h1_full, l1_full, sample_rate, gps_start, analyzer,
              psd_freqs_h1, psd_h1, psd_freqs_l1, psd_l1,
              seg_duration, target_snr, rng, avoid_gps=None, avoid_window=20.0):
    n_pts = len(h1_full)
    max_start = n_pts - int(seg_duration * sample_rate) - 1
    while True:
        start_idx = int(rng.integers(int(100 * sample_rate), max_start))
        seg_gps = gps_start + start_idx / sample_rate
        if avoid_gps is None or abs(seg_gps - avoid_gps) > avoid_window:
            break
    n_seg = int(seg_duration * sample_rate)
    h1_raw = h1_full[start_idx:start_idx + n_seg].copy()
    l1_raw = l1_full[start_idx:start_idx + n_seg].copy()

    # --- background (noise-only) ---
    h1_bg = whiten_and_bandpass(h1_raw, sample_rate, psd_freqs_h1, psd_h1)
    l1_bg = whiten_and_bandpass(l1_raw, sample_rate, psd_freqs_l1, psd_l1)
    times = np.arange(n_seg) / sample_rate
    center = times[n_seg // 2]
    r_bg = analyzer.analyze(h1_bg, l1_bg, times, center)

    # --- foreground (injection), SAME noise realization ---
    if target_snr is not None:
        n_burst = int(BURST_DURATION * sample_rate)
        wf_h1_short = simple_chirp(BURST_DURATION, sample_rate, target_snr / np.sqrt(2),
                                    h1_raw, psd_freqs_h1, psd_h1, rng=rng)
        delay_samples = int(round(0.007 * sample_rate))  # ~7ms типичен H1-L1 delay
        wf_l1_short = np.roll(wf_h1_short, delay_samples)

        # постави кратката вълна в ЦЕНТЪРА на по-дългия суров сегмент
        wf_h1 = np.zeros(n_seg)
        wf_l1 = np.zeros(n_seg)
        mid = n_seg // 2
        a = mid - n_burst // 2
        b = a + n_burst
        wf_h1[a:b] = wf_h1_short
        wf_l1[a:b] = wf_l1_short

        h1_fg_raw = h1_raw + wf_h1
        l1_fg_raw = l1_raw + wf_l1
        h1_fg = whiten_and_bandpass(h1_fg_raw, sample_rate, psd_freqs_h1, psd_h1)
        l1_fg = whiten_and_bandpass(l1_fg_raw, sample_rate, psd_freqs_l1, psd_l1)
        r_fg = analyzer.analyze(h1_fg, l1_fg, times, center)
    else:
        r_fg = None

    return r_bg, r_fg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h1-path", required=True)
    ap.add_argument("--l1-path", required=True)
    ap.add_argument("--snr-list", default="4,5,6,7,8,10,12")
    ap.add_argument("--n-per-snr", type=int, default=80)
    ap.add_argument("--seg-duration", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="light_ab_results.json")
    args = ap.parse_args()

    print("Зареждам H1/L1 (веднъж)...")
    h1_full, gps_h1, sr_h1 = load_strain(args.h1_path)
    l1_full, gps_l1, sr_l1 = load_strain(args.l1_path)
    assert abs(sr_h1 - sr_l1) < 1e-6, "H1/L1 sample rate mismatch"
    sample_rate = sr_h1
    print(f"  H1: {len(h1_full)} samples @ {sample_rate:.1f} Hz, GPS start {gps_h1:.1f}")
    print(f"  L1: {len(l1_full)} samples @ {sample_rate:.1f} Hz, GPS start {gps_l1:.1f}")

    print("Смятам PSD (Welch, веднъж за всеки детектор)...")
    psd_freqs_h1, psd_h1 = compute_whitening_filter(h1_full, sample_rate)
    psd_freqs_l1, psd_l1 = compute_whitening_filter(l1_full, sample_rate)

    analyzer = TriAxisAnalyzerV5(CONFIG)
    rng = np.random.default_rng(args.seed)
    snr_list = [float(s) for s in args.snr_list.split(",")]

    results = []
    total = len(snr_list) * args.n_per_snr * 2  # bg + fg на всеки trial
    done = 0

    for snr in snr_list:
        for i in range(args.n_per_snr):
            r_bg, r_fg = run_trial(
                h1_full, l1_full, sample_rate, gps_h1, analyzer,
                psd_freqs_h1, psd_h1, psd_freqs_l1, psd_l1,
                args.seg_duration, snr, rng,
            )
            if r_bg:
                results.append({
                    "snr": snr, "label": 0,
                    "v5": r_bg["triaxis_score"],
                    "v4": r_bg["triaxis_score_v4_legacy"],
                })
            if r_fg:
                results.append({
                    "snr": snr, "label": 1,
                    "v5": r_fg["triaxis_score"],
                    "v4": r_fg["triaxis_score_v4_legacy"],
                })
            done += 2
            if done % 20 == 0:
                print(f"  {done}/{total} trials done")

    import json
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nЗаписано: {args.out}")

    # бърз AUC-по-SNR summary директно тук
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("(pip install scikit-learn за AUC таблица; резултатите вече са в JSON)")
        return

    print(f"\n{'SNR':>6}  {'n_bg':>5}  {'n_fg':>5}  {'AUC V4':>8}  {'AUC V5':>8}  {'delta':>8}")
    for snr in snr_list:
        bg = [r for r in results if r["snr"] == snr and r["label"] == 0]
        fg = [r for r in results if r["snr"] == snr and r["label"] == 1]
        if len(bg) < 5 or len(fg) < 5:
            continue
        y = np.array([0] * len(bg) + [1] * len(fg))
        s_v4 = np.array([r["v4"] for r in bg] + [r["v4"] for r in fg])
        s_v5 = np.array([r["v5"] for r in bg] + [r["v5"] for r in fg])
        auc_v4 = roc_auc_score(y, s_v4)
        auc_v5 = roc_auc_score(y, s_v5)
        print(f"{snr:>6.1f}  {len(bg):>5}  {len(fg):>5}  {auc_v4:>8.3f}  {auc_v5:>8.3f}  {auc_v5 - auc_v4:>+8.3f}")


if __name__ == "__main__":
    main()
