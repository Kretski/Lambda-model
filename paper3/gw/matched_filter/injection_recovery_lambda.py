#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
injection_recovery_lambda.py
==============================
Λ-инжекционен вариант на injection_recovery.py.

ЦЕЛ: тества дали TriAxis V5 (template-free max_corr детекторът) има
различна detection efficiency за Λ-деформирани сигнали спрямо чист GR
(Λ=0), при еднакъв network SNR. TriAxis V5 e geometric/coherence
statistic — не сравнява с template — затова хипотезата НЕ е "Λ прави
detector-а по-чувствителен", а: "не пропуска ли detector-ът по-често
non-GR (Λ≠0) сигнали, защото геометрията на inspiral chirp-а леко се
променя с Λ".

Използва РЕАЛНИЯ Λ-dispersion waveform модел (waveform.py,
physically-derived от Paper 2 FLRW секцията:
  Psi(f;Lambda) = Psi_GR(f) - (4*pi^3*Lambda*K(z)/c^3) * f^3
), не toy модел.

Разлики спрямо injection_recovery.py:
  - Нова морфология "bbh_lambda": вместо pycbc get_td_waveform (чист
    GR), генерира h(f) през waveform_frequency_domain(..., Lambda, K_z)
    и обратен FFT до времеви domain.
  - hc се извежда от hp чрез 90°-фазов shift (face-on/circular
    поляризационно опростяване — waveform.py не включва inclination-
    зависимост explicit, декларира се тук).
  - Продължителност T на waveform-а: Newtonian time-to-coalescence от
    f_lower (стандартна PN формула), с margin, за да побере целия
    inspiral без wraparound.
  - Всичко останало (antenna projection, SNR scaling срещу реалния
    Welch PSD на шумовия сегмент, инжекция в реален H1/L1 шум,
    best-of-9 TriAxis V5 offset scan, output schema) е ИДЕНТИЧНО на
    injection_recovery.py — за директна сравнимост.

Употреба:
  python injection_recovery_lambda.py --n 60 --seed 201 \
      --lambda-true 0 --fix-snr 10 --out inj_lambda_L0_snr10.jsonl
  python injection_recovery_lambda.py --n 60 --seed 202 \
      --lambda-true 10 --fix-snr 10 --out inj_lambda_L10_snr10.jsonl
  ... (Λ = 0, ±5, ±10, ±20, ±50 при фиксиран SNR)

После: сравни AUC(Λ) срещу global_background_v3.jsonl, аналогично на
analyze_injections_v4_vs_v5.py, но с groupby по lambda_true вместо
V4/V5.
"""

import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, ".")
from triaxis_analyzer_v5 import TriAxisAnalyzerV5
from waveform import waveform_frequency_domain, cosmological_K_factor

SAMPLE_RATE = 4096
SEGMENT_HALF = 10
OFFSET_GRID_MS = [-100, -75, -50, -25, 0, 25, 50, 75, 100]
SEGMENTS_CACHE = "segments_cache.json"

M_MIN, M_MAX = 7.0, 50.0
FIX_MASSES = None
SNR_MIN, SNR_MAX = 5.0, 30.0
F_LOWER = 25.0
DISTANCE_MPC = 400.0
REDSHIFT_Z = 0.09  # приблизителен redshift за D_L=400 Mpc (GW150914-скала); K(z) фиксиран за протокола

CONFIG = {
    'sample_rate': SAMPLE_RATE, 'window_size': 0.25,
    'bandpass_low': 30, 'bandpass_high': 500,
    'viterbi_freq_penalty': 50.0, 'skip_time_axis': True
}

_K_Z_CACHE = {}


def get_K_z(z):
    if z not in _K_Z_CACHE:
        _K_Z_CACHE[z] = cosmological_K_factor(z)
    return _K_Z_CACHE[z]


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


def newtonian_chirp_duration(m1_msun, m2_msun, f_lower, margin=1.3):
    """Newtonian time-to-coalescence от f_lower (стандартна PN формула),
    с margin за безопасност. Определя дължината на времевия масив,
    нужна да побере целия inspiral без wraparound при обратния FFT."""
    G_SI, C_SI, MSUN_SI = 6.674e-11, 2.998e8, 1.989e30
    m1, m2 = m1_msun * MSUN_SI, m2_msun * MSUN_SI
    Mc = (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)
    tau = (5.0 / 256.0) * (G_SI * Mc / C_SI ** 3) ** (-5.0 / 3.0) * \
          (np.pi * f_lower) ** (-8.0 / 3.0)
    return float(tau * margin)


def make_injection_lambda(rng, gps_geocent, lambda_true, K_z):
    """Λ-версия на make_injection(): генерира hp/hc от waveform.py
    (Λ-dispersion GR+Lambda модел) вместо pycbc, проектира върху H1/L1
    със същата antenna-pattern логика."""
    from pycbc.detector import Detector

    inclination = float(np.arccos(rng.uniform(-1, 1)))
    ra = float(rng.uniform(0, 2 * np.pi))
    dec = float(np.arcsin(rng.uniform(-1, 1)))
    pol = float(rng.uniform(0, 2 * np.pi))

    if FIX_MASSES is not None:
        m1, m2 = FIX_MASSES
    else:
        m1 = float(np.exp(rng.uniform(np.log(M_MIN), np.log(M_MAX))))
        m2 = float(np.exp(rng.uniform(np.log(M_MIN), np.log(M_MAX))))
        if m2 > m1:
            m1, m2 = m2, m1

    T = newtonian_chirp_duration(m1, m2, F_LOWER)
    N = int(np.ceil(T * SAMPLE_RATE))
    # заокръгли нагоре до удобна дължина
    N = int(2 ** np.ceil(np.log2(N))) if N > 0 else 4096
    freqs = np.fft.rfftfreq(N, d=1.0 / SAMPLE_RATE)

    # tc=0: мержърът (най-високите честоти) ляга на t=0 -> при irfft
    # обвива се в края на масива (циклично); ще го "разгънем" по-долу.
    hp_fd = waveform_frequency_domain(freqs, m1, m2, lambda_true, K_z,
                                       tc=0.0, phi_c=0.0,
                                       distance_Mpc=DISTANCE_MPC)
    # 90°-фазов shift за hc (face-on/circular поляризационно опростяване
    # — waveform.py не включва inclination-зависима амплитуда explicit;
    # декларира се тук като опростяване спрямо пълния PyCBC BBH случай).
    hc_fd = -1j * hp_fd

    hp_a = np.fft.irfft(hp_fd, n=N)
    hc_a = np.fft.irfft(hc_fd, n=N)

    # "Разгъване": мержърът е близо до index 0 (циклично увит в края).
    # Намери пика на обвивката и roll-ни го да е ясно вътре в масива,
    # с достатъчно inspiral преди и малко ringdown опашка след.
    env = np.abs(hp_a) + np.abs(hc_a)
    merger_idx_raw = int(np.argmax(env))
    # roll-ни така, че мержърът да е близо до края на масива (имитира
    # PyCBC конвенцията, където merger_idx = argmax е близо до края
    # на масива, с дълъг inspiral преди него)
    shift = (N - int(0.5 * SAMPLE_RATE)) - merger_idx_raw
    hp_a = np.roll(hp_a, shift)
    hc_a = np.roll(hc_a, shift)
    env = np.roll(env, shift)
    merger_idx = int(np.argmax(env))

    proj = {}
    dets = {d: Detector(d) for d in ("H1", "L1")}
    for name, det in dets.items():
        fp, fx = det.antenna_pattern(ra, dec, pol, gps_geocent)
        dt_det = det.time_delay_from_earth_center(ra, dec, gps_geocent)
        h = fp * hp_a + fx * hc_a
        proj[name] = (h, float(dt_det))
    delay_true_ms = (proj["H1"][1] - proj["L1"][1]) * 1000.0

    params = {'morphology': 'bbh_lambda', 'lambda_true': lambda_true,
              'ra': ra, 'dec': dec, 'inclination': inclination,
              'polarization': pol, 'delay_true_ms': delay_true_ms,
              'm1': m1, 'm2': m2, 'mtotal': m1 + m2,
              'mchirp': (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2}

    # Същата crop логика като GR baseline-а (make_injection): режем
    # ПРЕДИ SNR калкулацията, за да няма rfft(h, n<len(h)) truncation бъг.
    KEEP_BEFORE = int((SEGMENT_HALF - 1.5) * SAMPLE_RATE)
    KEEP_AFTER = int(0.5 * SAMPLE_RATE)
    lo = max(0, merger_idx - KEEP_BEFORE)
    hi = min(len(env), merger_idx + KEEP_AFTER)
    for name in proj:
        h, dtd = proj[name]
        proj[name] = (h[lo:hi], dtd)
    merger_idx = merger_idx - lo
    return proj, params, merger_idx


def optimal_snr(h, noise, fs):
    from scipy.signal import welch
    nper = int(fs * 0.25)
    freqs, psd = welch(noise, fs=fs, nperseg=nper)
    hf = np.fft.rfft(h, n=len(noise)) / fs
    f_h = np.fft.rfftfreq(len(noise), 1 / fs)
    psd_i = np.interp(f_h, freqs, psd)
    band = (f_h >= F_LOWER) & (f_h <= 1000)
    df = f_h[1] - f_h[0]
    return float(np.sqrt(4 * np.sum((np.abs(hf[band]) ** 2 / psd_i[band])) * df))


def process_injection_lambda(analyzer, usable, rng, lambda_true, K_z,
                              fix_snr=None):
    from gwpy.timeseries import TimeSeries

    t0 = sample_epoch(usable, rng)
    h1_raw = TimeSeries.fetch_open_data("H1", t0 - SEGMENT_HALF, t0 + SEGMENT_HALF,
                                        sample_rate=SAMPLE_RATE, cache=False)
    l1_raw = TimeSeries.fetch_open_data("L1", t0 - SEGMENT_HALF, t0 + SEGMENT_HALF,
                                        sample_rate=SAMPLE_RATE, cache=False)
    h1v, l1v = h1_raw.value.copy(), l1_raw.value.copy()
    times = h1_raw.times.value

    proj, params, merger_idx = make_injection_lambda(rng, t0, lambda_true, K_z)

    if fix_snr is not None:
        snr_target = float(fix_snr + rng.uniform(0, 0.5))
    else:
        snr_target = float(np.exp(rng.uniform(np.log(SNR_MIN), np.log(SNR_MAX))))

    s_h1 = optimal_snr(proj['H1'][0], h1v, SAMPLE_RATE)
    s_l1 = optimal_snr(proj['L1'][0], l1v, SAMPLE_RATE)
    snr_net_unit = np.sqrt(s_h1 ** 2 + s_l1 ** 2)
    if snr_net_unit < 1e-30:
        return None
    scale = snr_target / snr_net_unit

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

    from gwpy.timeseries import TimeSeries as TS
    h1_p = TS(h1v, times=times).whiten(fftlength=0.25, overlap=0.1).bandpass(30, 500)
    l1_p = TS(l1v, times=times).whiten(fftlength=0.25, overlap=0.1).bandpass(30, 500)

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
    global FIX_MASSES
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=201)
    ap.add_argument("--out", default="injections_lambda.jsonl")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--lambda-true", type=float, required=True,
                     help="Фиксирана Λ стойност за целия run (напр. 0, 5, -5, 10, -10, 20, -20, 50, -50)")
    ap.add_argument("--fix-snr", type=float, default=None,
                     help="Фиксира SNR в [fix_snr, fix_snr+0.5] (иначе log-uniform [5,30])")
    ap.add_argument("--fix-m1", type=float, default=None)
    ap.add_argument("--fix-m2", type=float, default=None)
    ap.add_argument("--redshift", type=float, default=REDSHIFT_Z)
    args = ap.parse_args()

    if args.fix_m1 is not None and args.fix_m2 is not None:
        FIX_MASSES = (args.fix_m1, args.fix_m2)

    K_z = get_K_z(args.redshift)
    print(f"K(z={args.redshift}) = {K_z:.6e}")

    usable = load_segments()
    analyzer = TriAxisAnalyzerV5(CONFIG)
    rng = np.random.default_rng(args.seed)

    mode = "a" if args.resume and os.path.exists(args.out) else "w"
    n_done = 0
    if mode == "a":
        with open(args.out) as f:
            n_done = sum(1 for _ in f)
        print(f"Resume: {n_done} вече записани, продължавам до {args.n}")

    t_start = time.time()
    with open(args.out, mode) as fout:
        i = n_done
        while i < args.n:
            rec = process_injection_lambda(analyzer, usable, rng,
                                            args.lambda_true, K_z,
                                            fix_snr=args.fix_snr)
            if rec is None:
                continue
            rec['seed'] = args.seed
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            i += 1
            elapsed = time.time() - t_start
            rate = elapsed / max(i - n_done, 1)
            eta = rate * (args.n - i)
            print(f"  [{i}/{args.n}] Λ={args.lambda_true} SNR={rec['snr_net']:.1f} "
                  f"Mtot={rec['mtotal']:.0f} -> max_corr={rec['max_corr']:.3f} "
                  f"triaxis_v5={rec['triaxis_v5']:.3f} (ETA ~{eta/60:.0f} min)")

    print("Готово.")


if __name__ == "__main__":
    main()
