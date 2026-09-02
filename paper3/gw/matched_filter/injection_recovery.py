#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
injection_recovery.py
=======================
Етап 2: injection recovery за max_corr статистиката.

Инжектира реалистични BBH сигнали (pycbc IMRPhenomD, проектирани върху
H1/L1 с реални антенни функции и реално междудетекторно закъснение по
случайна позиция на небето) в РЕАЛЕН детекторен шум от валидните
сегменти (reuse на segments_cache.json от global_background), и мери
възстановения max_corr със СЪЩАТА best-of-9 процедура като null-а и
събитията.

Дизайн решения (фиксирани преди пускане, за протокола):
  - Маси: m1, m2 log-uniform в [7, 50] M_sun (GWTC-1-подобен диапазон),
    спинове 0 (опростяване, декларира се в write-up).
  - Небе/ориентация: uniform sky (ra, dec), uniform polarization,
    uniform cos(inclination).
  - Амплитуда: мащабирана до целеви network optimal SNR, log-uniform
    в [5, 30], изчислен срещу PSD-то на СЪЩИЯ шумов сегмент (Welch).
  - Инжекцията е в суровия strain ПРЕДИ whitening (правилният ред).
  - Статистика: max_corr, best-of-9 offsets около времето на мержъра —
    идентично на null-а и събитията. Никакви параметри не се менят
    след поглед към резултатите.

Изход: injections.jsonl (един ред на инжекция: параметри + recovery).

Употреба:
  python injection_recovery.py --n 300 --seed 101
  python injection_recovery.py --n 300 --seed 101 --resume
После:
  python analyze_injections.py --inj injections.jsonl --bg global_background_v3.jsonl

Изисква: pycbc (pip install pycbc), gwpy, segments_cache.json от
global_background_v2/v3 в текущата директория.
"""

import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, ".")
from triaxis_analyzer_v5 import TriAxisAnalyzerV5

SAMPLE_RATE = 4096
SEGMENT_HALF = 10
OFFSET_GRID_MS = [-100, -75, -50, -25, 0, 25, 50, 75, 100]
SEGMENTS_CACHE = "segments_cache.json"

M_MIN, M_MAX = 7.0, 50.0
FIX_MASSES = None   # (m1, m2) за целеви тестове; сетва се от --fix-m1/--fix-m2
SNR_MIN, SNR_MAX = 5.0, 30.0
F_LOWER = 25.0

CONFIG = {
    'sample_rate': SAMPLE_RATE, 'window_size': 0.25,
    'bandpass_low': 30, 'bandpass_high': 500,
    'viterbi_freq_penalty': 50.0, 'skip_time_axis': True
}


def load_segments():
    with open(SEGMENTS_CACHE) as f:
        cached = json.load(f)
    key = "O1+O2" if "O1+O2" in cached else sorted(cached.keys())[0]
    segs = [tuple(s) for s in cached[key]]
    usable = [(a + SEGMENT_HALF, b - SEGMENT_HALF) for a, b in segs
              if (b - a) > 2 * SEGMENT_HALF + 4]
    return usable


def sample_epoch(usable, rng):
    lengths = np.array([hi - lo for lo, hi in usable])
    i = rng.choice(len(usable), p=lengths / lengths.sum())
    lo, hi = usable[i]
    return float(rng.uniform(lo, hi))


def make_injection(rng, gps_geocent, morphology="bbh"):
    """Генерира hp/hc (BBH, sine-Gaussian или white-noise burst) и
    проекции върху H1/L1 с реални антенни функции.

    Burst морфологиите (sg, wnb) са стандартните немоделирани тестови
    сигнали от burst търсенията (напр. cWB валидациите) — целта е да се
    ИЗМЕРИ, а не само да се твърди, че статистиката е template-free."""
    from pycbc.detector import Detector

    inclination = float(np.arccos(rng.uniform(-1, 1)))
    ra = float(rng.uniform(0, 2 * np.pi))
    dec = float(np.arcsin(rng.uniform(-1, 1)))
    pol = float(rng.uniform(0, 2 * np.pi))
    extra = {}

    if morphology == "bbh":
        from pycbc.waveform import get_td_waveform
        if FIX_MASSES is not None:
            m1, m2 = FIX_MASSES
        else:
            m1 = float(np.exp(rng.uniform(np.log(M_MIN), np.log(M_MAX))))
            m2 = float(np.exp(rng.uniform(np.log(M_MIN), np.log(M_MAX))))
            if m2 > m1:
                m1, m2 = m2, m1
        hp, hc = get_td_waveform(approximant="IMRPhenomD",
                                 mass1=m1, mass2=m2,
                                 inclination=inclination,
                                 delta_t=1.0 / SAMPLE_RATE,
                                 f_lower=F_LOWER, distance=400.0)
        hp_a, hc_a = hp.numpy(), hc.numpy()
        extra = {'m1': m1, 'm2': m2, 'mtotal': m1 + m2,
                 'mchirp': (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2}

    elif morphology == "sg":
        # sine-Gaussian: f0 log-uniform в лентата, Q uniform [3, 20]
        f0 = float(np.exp(rng.uniform(np.log(50.0), np.log(300.0))))
        Q = float(rng.uniform(3.0, 20.0))
        tau = Q / (np.sqrt(2.0) * np.pi * f0)
        t = np.arange(-6 * tau, 6 * tau, 1.0 / SAMPLE_RATE)
        envlp = np.exp(-t ** 2 / (2 * tau ** 2))
        hp_a = 1e-22 * envlp * np.cos(2 * np.pi * f0 * t)
        hc_a = 1e-22 * envlp * np.sin(2 * np.pi * f0 * t)
        extra = {'f0': f0, 'Q': Q}

    elif morphology == "wnb":
        # white-noise burst: bandlimited, независими поляризации
        dur = float(rng.uniform(0.01, 0.1))
        flo = float(rng.uniform(40.0, 100.0))
        fhi = flo + float(rng.uniform(50.0, 300.0))
        nsamp = int(dur * SAMPLE_RATE)
        from scipy.signal import butter, filtfilt
        bb, aa = butter(4, [flo / (SAMPLE_RATE / 2),
                            min(fhi, SAMPLE_RATE / 2 * 0.9) / (SAMPLE_RATE / 2)],
                        btype='band')
        win = np.hanning(nsamp)
        hp_a = 1e-22 * filtfilt(bb, aa, rng.normal(0, 1, nsamp)) * win
        hc_a = 1e-22 * filtfilt(bb, aa, rng.normal(0, 1, nsamp)) * win
        extra = {'duration_s': dur, 'f_lo': flo, 'f_hi': fhi}
    else:
        raise ValueError(f"непозната морфология: {morphology}")

    proj = {}
    dets = {d: Detector(d) for d in ("H1", "L1")}
    for name, det in dets.items():
        fp, fx = det.antenna_pattern(ra, dec, pol, gps_geocent)
        dt_det = det.time_delay_from_earth_center(ra, dec, gps_geocent)
        h = fp * hp_a + fx * hc_a
        proj[name] = (h, float(dt_det))
    # Конвенция на analyzer-а (емпирично проверена): delay_ms = H1 - L1
    delay_true_ms = (proj["H1"][1] - proj["L1"][1]) * 1000.0

    params = {'morphology': morphology,
              'ra': ra, 'dec': dec, 'inclination': inclination,
              'polarization': pol, 'delay_true_ms': delay_true_ms}
    params.update(extra)
    # Мержърът се намира от envelope-а (пик на |h|), НЕ от sample_times:
    # при FD апроксиманти (IMRPhenomD) TD епохата не гарантира merger@t=0,
    # и конверсията оставя дълга почти-нулева опашка след мержъра.
    env = np.abs(hp_a) + np.abs(hc_a)
    merger_idx = int(np.argmax(env))

    # КРИТИЧНО: crop до частта, която реално влиза в сегмента, ПРЕДИ
    # SNR калкулацията. За леки системи waveform-ът (при f_lower=25) е
    # по-дълъг от сегмента; np.fft.rfft(h, n<len(h)) мълчаливо реже до
    # първите n семпла (тих ранен inspiral, БЕЗ мержъра) -> optimal_snr
    # се смяташе върху тихата част, а се инжектираше силната ->
    # реален SNR с порядъци над целевия (хваната от smoke теста:
    # max_corr 0.95+ при "SNR 5" за Mtot<20). Отрязваме до
    # [merger - KEEP_BEFORE, merger + KEEP_AFTER] и SNR-ът, и
    # инжекцията ползват ЕДИН И СЪЩ масив. SNR дефиницията става
    # "SNR на инжектираното съдържание" — декларира се в write-up.
    KEEP_BEFORE = int((SEGMENT_HALF - 1.5) * SAMPLE_RATE)   # 8.5 s inspiral
    KEEP_AFTER = int(0.5 * SAMPLE_RATE)                      # 0.5 s ringdown
    lo = max(0, merger_idx - KEEP_BEFORE)
    hi = min(len(env), merger_idx + KEEP_AFTER)
    for name in proj:
        h, dtd = proj[name]
        proj[name] = (h[lo:hi], dtd)
    merger_idx = merger_idx - lo
    return proj, params, merger_idx


def optimal_snr(h, noise, fs):
    """Optimal SNR на инжекцията срещу Welch PSD на реалния шумов сегмент."""
    from scipy.signal import welch
    nper = int(fs * 0.25)
    freqs, psd = welch(noise, fs=fs, nperseg=nper)
    hf = np.fft.rfft(h, n=len(noise)) / fs
    f_h = np.fft.rfftfreq(len(noise), 1 / fs)
    psd_i = np.interp(f_h, freqs, psd)
    band = (f_h >= F_LOWER) & (f_h <= 1000)
    df = f_h[1] - f_h[0]
    return float(np.sqrt(4 * np.sum((np.abs(hf[band]) ** 2 / psd_i[band])) * df))


def process_injection(analyzer, usable, rng, morphology="bbh"):
    from gwpy.timeseries import TimeSeries

    t0 = sample_epoch(usable, rng)
    h1_raw = TimeSeries.fetch_open_data("H1", t0 - SEGMENT_HALF, t0 + SEGMENT_HALF,
                                        sample_rate=SAMPLE_RATE, cache=False)
    l1_raw = TimeSeries.fetch_open_data("L1", t0 - SEGMENT_HALF, t0 + SEGMENT_HALF,
                                        sample_rate=SAMPLE_RATE, cache=False)
    h1v, l1v = h1_raw.value.copy(), l1_raw.value.copy()
    times = h1_raw.times.value

    proj, params, merger_idx = make_injection(rng, t0, morphology)

    # Целеви network SNR -> мащаб
    snr_target = float(np.exp(rng.uniform(np.log(SNR_MIN), np.log(SNR_MAX))))
    s_h1 = optimal_snr(proj['H1'][0], h1v, SAMPLE_RATE)
    s_l1 = optimal_snr(proj['L1'][0], l1v, SAMPLE_RATE)
    snr_net_unit = np.sqrt(s_h1 ** 2 + s_l1 ** 2)
    if snr_net_unit < 1e-30:   # физичните strain амплитуди са ~1e-21
        return None
    scale = snr_target / snr_net_unit

    # Инжектиране: МЕРЖЪРЪТ (waveform индекс merger_idx) ляга точно на
    # center_idx + детекторното закъснение. Директно позициониране, без
    # аритметика през края на waveform-а.
    n = len(h1v)
    center_idx = n // 2
    for det, vec in (("H1", h1v), ("L1", l1v)):
        h, dt_det = proj[det]
        shift = int(round(dt_det * SAMPLE_RATE))
        start_idx = center_idx + shift - merger_idx
        a, b = max(0, start_idx), min(n, start_idx + len(h))
        if b <= a:
            return None
        vec[a:b] += scale * h[a - start_idx: b - start_idx]

    # Идентична обработка като null/събития
    from gwpy.timeseries import TimeSeries as TS
    h1_p = TS(h1v, times=times).whiten(fftlength=0.25, overlap=0.1).bandpass(30, 500)
    l1_p = TS(l1v, times=times).whiten(fftlength=0.25, overlap=0.1).bandpass(30, 500)

    # ВАЖНО: изборът на offset е по V5 score (delay-претегления) — СЪЩОТО
    # правило като null-а (global_background_v3) и събитията
    # (analyze_events_v5). Първоначалната версия избираше по суров
    # max_corr, което даваше на инжекциите по-щедро правило от null-а
    # (леко оптимистичен AUC) и създаваше фалшив "twin дефицит" при
    # сравнение със събитията. Записаният max_corr е този при избрания
    # offset — консистентно с всички останали компоненти.
    best = None
    raw_max = 0.0
    for off in OFFSET_GRID_MS:
        r = analyzer.analyze(h1_p.value, l1_p.value, times, t0 + off / 1000.0)
        if r:
            raw_max = max(raw_max, r['max_corr'])
            if best is None or r['triaxis_score'] > best['triaxis_score']:
                best = r
    if best is None:
        return None

    rec = dict(params)
    rec['max_corr_rawmax'] = raw_max
    rec.update({
        'epoch_gps': t0,
        'snr_net': snr_target,
        'snr_h1': s_h1 * scale, 'snr_l1': s_l1 * scale,
        'max_corr': float(best['max_corr']),
        'delay_ms_measured': float(best['delay_ms']),
        'width_ms': float(best['width_ms']),
        'triaxis_v5': float(best['triaxis_score']),
        'triaxis_v4_legacy': float(best['triaxis_score_v4_legacy']),
    })
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--out", default="injections.jsonl")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--morphology", default="bbh", choices=["bbh", "sg", "wnb"],
                    help="bbh (IMRPhenomD), sg (sine-Gaussian), wnb (white-noise burst)")
    ap.add_argument("--fix-m1", type=float, default=None,
                    help="Фиксира m1 (за целеви тестове, напр. GW151226 twin)")
    ap.add_argument("--fix-m2", type=float, default=None)
    ap.add_argument("--fix-snr", type=float, default=None,
                    help="Фиксира network SNR (±0.5)")
    args = ap.parse_args()

    if args.fix_m1 and args.fix_m2:
        globals()['FIX_MASSES'] = (max(args.fix_m1, args.fix_m2),
                                   min(args.fix_m1, args.fix_m2))
    if args.fix_snr:
        globals()['SNR_MIN'] = args.fix_snr
        globals()['SNR_MAX'] = args.fix_snr + 0.5

    rng = np.random.default_rng(args.seed)
    usable = load_segments()
    print(f"Валидни сегменти: {len(usable)} (от {SEGMENTS_CACHE})")

    n_done = 0
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            n_done = sum(1 for _ in f)
        print(f"Resume: {n_done} инжекции вече в {args.out}")
        # Нов derived stream: [seed, n_done]. Прекъснат run НЕ възпроизвежда
        # байт-по-байт непрекъснат run (декларира се в лога), но остава
        # детерминистичен и не преизползва вече изтеглени случайни числа.
        rng = np.random.default_rng([args.seed, n_done])

    analyzer = TriAxisAnalyzerV5(CONFIG)
    t_start = time.time()
    with open(args.out, 'a') as fout:
        i = n_done
        attempts = 0
        while i < args.n and attempts < args.n * 3:
            attempts += 1
            try:
                rec = process_injection(analyzer, usable, rng,
                                        args.morphology)
                if rec is None:
                    continue
                rec['seed'] = args.seed
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                i += 1
                el = time.time() - t_start
                eta = el / max(i - n_done, 1) * (args.n - i)
                tag = (f"Mtot={rec['mtotal']:.0f}" if 'mtotal' in rec
                       else (f"f0={rec['f0']:.0f}Hz Q={rec['Q']:.0f}"
                             if 'f0' in rec else f"dur={rec['duration_s']*1000:.0f}ms"))
                print(f"  [{i}/{args.n}] {rec['morphology']} SNR={rec['snr_net']:.1f} "
                      f"{tag} -> max_corr={rec['max_corr']:.3f} "
                      f"(ETA ~{eta/60:.0f} min)")
            except Exception as e:
                print(f"  пропусната ({type(e).__name__}: {e})")
    print("Готово.")


if __name__ == "__main__":
    main()
