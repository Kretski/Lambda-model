#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
global_background_v3.py
=========================
V3 = V2 семплиране (разпръснато по целия run, veto, matched max-over-9)
+ V5 score (непрекъснат delay член, score = coherence)
+ ЗАПИС НА СУРОВИТЕ ПРИЗНАЦИ (max_corr, delay_ms, width_ms, legacy скорове),
  така че бъдещи промени на score формулата да се преизчисляват от JSONL-а
  за секунди, без нов 10+ часов fetch+analyze run.
+ skip_time_axis: Viterbi не се смята (не влиза в score-а) -> голямо ускорение.
Reuse-ва segments_cache.json от v2 — timeline не се тегли наново.

Поправя пет проблема на старата версия:
  1. Един epoch (GW170814-100s..-10s)      -> N_EPOCHS независими GPS времена,
                                              равномерно по всички science сегменти.
  2. 10 000 припокрити прозореца от 90 s   -> K прозореца/epoch с мин. отстояние,
                                              ефективният N е реален, не фиктивен.
  3. Няма нестационарност в null-а         -> epochs покриват целия run (PSD drift,
                                              glitch rate, микросеизмика влизат в null-а).
  4. Няма veto                             -> изключени: зони около GWTC-1 събития,
                                              CBC/BURST hardware injections, данни
                                              извън CBC_CAT2 quality.
  5. Look-elsewhere mismatch               -> за всеки bg прозорец се записва И
                                              max-over-9-offsets score (същата
                                              процедура като analyze_event), за да
                                              се сравнява като-с-като.

Възпроизводимост: фиксиран seed + JSONL лог на всеки семплиран GPS.
Устойчивост: incremental запис (crash не губи направеното), resume при рестарт.

Употреба:
  python global_background_v2.py --n-epochs 300 --windows-per-epoch 6 --seed 42
  python global_background_v2.py --runs O2 --n-epochs 200        # само O2
  python global_background_v2.py --resume                        # продължава JSONL

Изисква: gwpy, gwosc, numpy, triaxis_analyzer_v4.py в текущата директория.
"""

import sys, os, json, argparse, time
import numpy as np

sys.path.insert(0, ".")
from triaxis_analyzer_v5 import TriAxisAnalyzerV5

# ==============================================================================
# Конфигурация
# ==============================================================================
SAMPLE_RATE   = 4096
SEGMENT_HALF  = 10          # ±10 s fetch около всеки epoch — СЪЩОТО като analyze_event
EDGE_MARGIN   = 4.0         # s, без прозорци в тази зона (whiten/bandpass edge ringing)
WINDOW_SIZE   = 0.25        # s, същото като analyzer конфига
MIN_SEPARATION = 0.5        # s, мин. отстояние между центрове на прозорци в един epoch
OFFSET_GRID_MS = [-100, -75, -50, -25, 0, 25, 50, 75, 100]  # същият като analyze_event

# GWTC-1 събития (O1+O2) — veto зона около всяко
GWTC1_EVENTS = {
    'GW150914': 1126259462.4, 'GW151012': 1128678900.4, 'GW151226': 1135136350.6,
    'GW170104': 1167559936.6, 'GW170608': 1180922494.5, 'GW170729': 1185389807.3,
    'GW170809': 1186302519.8, 'GW170814': 1186741861.5, 'GW170817': 1187008882.4,
    'GW170818': 1187058327.1, 'GW170823': 1187529256.5,
}
EVENT_VETO_HALF = 64.0      # s от всяка страна на събитие — щедро, за да не тече
                            # inspiral/ringdown/veto-related данни в null-а

RUN_SPANS = {               # GPS граници на наблюдателните кампании
    'O1': (1126051217, 1137254417),
    'O2': (1164556817, 1187733618),
}

ANALYZER_CONFIG = {
    'sample_rate': SAMPLE_RATE,
    'window_size': WINDOW_SIZE,
    'bandpass_low': 30,
    'bandpass_high': 500,
    'viterbi_freq_penalty': 50.0,
    'skip_time_axis': True     # Viterbi не влиза в V5 score -> прескачаме го
}


# ==============================================================================
# Интервална аритметика за сегменти
# ==============================================================================
def intersect_segments(a, b):
    """Сечение на два списъка от (start, end) интервали."""
    out, i, j = [], 0, 0
    a = sorted(a); b = sorted(b)
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo < hi:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def subtract_intervals(segs, vetoes):
    """Изважда veto интервали от списък сегменти."""
    out = list(sorted(segs))
    for v_lo, v_hi in sorted(vetoes):
        nxt = []
        for s_lo, s_hi in out:
            if v_hi <= s_lo or v_lo >= s_hi:
                nxt.append((s_lo, s_hi))
                continue
            if s_lo < v_lo:
                nxt.append((s_lo, v_lo))
            if v_hi < s_hi:
                nxt.append((v_hi, s_hi))
        out = nxt
    return out


def total_livetime(segs):
    return sum(hi - lo for lo, hi in segs)


# ==============================================================================
# Изграждане на валидните сегменти (с veto)
# ==============================================================================
SEGMENTS_CACHE = "segments_cache.json"


def _get_segments_retry(flag, gps_lo, gps_hi, n_retry=4, wait=15):
    """GWOSC timeline с retry — paging-ът от 1000+ страници понякога пада."""
    from gwosc.timeline import get_segments
    last = None
    for attempt in range(1, n_retry + 1):
        try:
            return get_segments(flag, gps_lo, gps_hi)
        except Exception as e:
            last = e
            print(f"    {flag}: опит {attempt}/{n_retry} провален "
                  f"({type(e).__name__}), чакам {wait}s...")
            time.sleep(wait)
    raise last


def build_valid_segments(runs, verbose=True):
    """
    Връща списък (start, end) сегменти, в които:
      - H1 и L1 едновременно имат данни,
      - данните минават CBC_CAT2 quality и на двата детектора,
      - няма CBC/BURST hardware injections,
      - изключени са ±EVENT_VETO_HALF около всички GWTC-1 събития,
      - сегментът е достатъчно дълъг за пълен fetch (2*SEGMENT_HALF + резерв).

    Резултатът се кешира в SEGMENTS_CACHE — timeline метаданните за O1/O2
    са замразени (архивни run-ове), така че кешът е винаги валиден за
    същата комбинация от runs.
    """
    cache_key = "+".join(sorted(runs))
    if os.path.exists(SEGMENTS_CACHE):
        try:
            with open(SEGMENTS_CACHE) as f:
                cached = json.load(f)
            if cache_key in cached:
                segs = [tuple(s) for s in cached[cache_key]]
                if verbose:
                    print(f"  Сегменти от кеш ({SEGMENTS_CACHE}): "
                          f"{len(segs)}, {total_livetime(segs)/3600:.1f} h")
                return segs
        except Exception:
            pass  # повреден кеш -> теглим наново

    all_valid = []
    for run in runs:
        gps_lo, gps_hi = RUN_SPANS[run]
        if verbose:
            print(f"  [{run}] Тегля timeline сегменти от GWOSC...")

        # Quality + injection flags. CBC_CAT2 имплицира наличие на данни.
        # NO_*_HW_INJ = данни БЕЗ хардуерни инжекции.
        flag_lists = []
        for det in ('H1', 'L1'):
            for flag in (f'{det}_CBC_CAT2', f'{det}_NO_CBC_HW_INJ',
                         f'{det}_NO_BURST_HW_INJ'):
                segs = _get_segments_retry(flag, gps_lo, gps_hi)
                flag_lists.append([(float(a), float(b)) for a, b in segs])
                if verbose:
                    print(f"    {flag}: {len(segs)} сегмента, "
                          f"{total_livetime(segs)/3600:.1f} h")

        valid = flag_lists[0]
        for fl in flag_lists[1:]:
            valid = intersect_segments(valid, fl)
        all_valid.extend(valid)

    # Veto около GWTC-1 събития
    vetoes = [(t - EVENT_VETO_HALF, t + EVENT_VETO_HALF)
              for t in GWTC1_EVENTS.values()]
    all_valid = subtract_intervals(all_valid, vetoes)

    # Само сегменти, побиращи пълен fetch + резерв
    min_len = 2 * SEGMENT_HALF + 4.0
    all_valid = [(a, b) for a, b in all_valid if (b - a) >= min_len]

    if verbose:
        print(f"  Валидни сегменти след всички veto: {len(all_valid)}, "
              f"общо {total_livetime(all_valid)/3600:.1f} h coincident livetime")

    # Запис в кеша
    cached = {}
    if os.path.exists(SEGMENTS_CACHE):
        try:
            with open(SEGMENTS_CACHE) as f:
                cached = json.load(f)
        except Exception:
            cached = {}
    cached[cache_key] = [list(s) for s in all_valid]
    with open(SEGMENTS_CACHE, 'w') as f:
        json.dump(cached, f)
    if verbose:
        print(f"  Сегментите са кеширани в {SEGMENTS_CACHE} — "
              f"следващите стартове няма да теглят timeline.")
    return all_valid


# ==============================================================================
# Семплиране на epochs, претеглено по продължителност на сегментите
# ==============================================================================
def sample_epochs(segments, n_epochs, rng):
    """
    Равномерно по livetime: вероятността epoch да падне в сегмент е
    пропорционална на дължината му. Центърът се тегли така, че целият
    fetch прозорец [t0-SEGMENT_HALF, t0+SEGMENT_HALF] да е вътре в сегмента.
    """
    usable = [(a + SEGMENT_HALF, b - SEGMENT_HALF) for a, b in segments]
    lengths = np.array([hi - lo for lo, hi in usable], dtype=float)
    probs = lengths / lengths.sum()

    epochs = []
    idx = rng.choice(len(usable), size=n_epochs, p=probs)
    for i in idx:
        lo, hi = usable[i]
        epochs.append(float(rng.uniform(lo, hi)))
    return sorted(epochs)


def sample_window_centers(t0, k, rng):
    """
    k центъра на прозорци в централната зона на fetch сегмента,
    извън EDGE_MARGIN, с мин. отстояние MIN_SEPARATION.
    Offset grid-ът добавя ±0.1 s, така че центърът трябва да е
    поне EDGE_MARGIN + 0.1 + WINDOW_SIZE/2 от ръба.
    """
    guard = EDGE_MARGIN + 0.1 + WINDOW_SIZE / 2
    lo, hi = t0 - SEGMENT_HALF + guard, t0 + SEGMENT_HALF - guard
    centers = []
    attempts = 0
    while len(centers) < k and attempts < 200:
        c = float(rng.uniform(lo, hi))
        if all(abs(c - x) >= MIN_SEPARATION for x in centers):
            centers.append(c)
        attempts += 1
    return sorted(centers)


# ==============================================================================
# Обработка на един epoch
# ==============================================================================
def process_epoch(analyzer, t0, k_windows, rng, verbose=True):
    """
    Fetch ±SEGMENT_HALF около t0, обработка ИДЕНТИЧНА на analyze_event
    (whiten fftlength=0.25 overlap=0.1, bandpass 30-500), после k прозореца.
    За всеки прозорец записва:
      - single-offset score (класическия),
      - max-over-9-offsets score (matched към процедурата за събития).
    """
    from gwpy.timeseries import TimeSeries

    start, end = t0 - SEGMENT_HALF, t0 + SEGMENT_HALF
    h1_raw = TimeSeries.fetch_open_data("H1", start, end,
                                        sample_rate=SAMPLE_RATE, cache=False)
    l1_raw = TimeSeries.fetch_open_data("L1", start, end,
                                        sample_rate=SAMPLE_RATE, cache=False)

    h1_p = h1_raw.whiten(fftlength=0.25, overlap=0.1).bandpass(30, 500)
    l1_p = l1_raw.whiten(fftlength=0.25, overlap=0.1).bandpass(30, 500)
    h1_strain, l1_strain = h1_p.value, l1_p.value
    times = h1_p.times.value

    records = []
    for c in sample_window_centers(t0, k_windows, rng):
        # (а) единичен offset — за хистограмата на "суровия" score
        r0 = analyzer.analyze(h1_strain, l1_strain, times, c)
        if r0 is None:
            continue

        # (б) max over 9 offsets — СЪЩАТА процедура като analyze_event,
        # за да е валидно сравнението със score-овете на събитията
        best = r0
        for off in OFFSET_GRID_MS:
            if off == 0:
                continue
            r = analyzer.analyze(h1_strain, l1_strain, times, c + off / 1000.0)
            if r and r['triaxis_score'] > best['triaxis_score']:
                best = r

        records.append({
            'epoch_gps': t0,
            'window_gps': c,
            # single-offset: V5 score + СУРОВИ признаци
            'triaxis_score': float(r0['triaxis_score']),
            'max_corr': float(r0['max_corr']),
            'delay_ms': float(r0['delay_ms']),
            'width_ms': float(r0['width_ms']),
            'delay_weight': float(r0['delay_weight']),
            'structure_score': float(r0['structure_score']),
            'triaxis_score_v4_legacy': float(r0['triaxis_score_v4_legacy']),
            # matched (max-over-offsets) — сравнявай СЪБИТИЯТА с ТОВА
            'triaxis_score_maxoff': float(best['triaxis_score']),
            'max_corr_maxoff': float(best['max_corr']),
            'delay_ms_maxoff': float(best['delay_ms']),
            'width_ms_maxoff': float(best['width_ms']),
            'v4_legacy_maxoff': float(best['triaxis_score_v4_legacy']),
        })
    return records


# ==============================================================================
# Основен цикъл с incremental JSONL запис и resume
# ==============================================================================
def run(n_epochs, k_windows, runs, seed, out_jsonl, resume, verbose=True):
    rng = np.random.default_rng(seed)

    print("=" * 60)
    print("GLOBAL BACKGROUND v2 — разпръснато семплиране")
    print("=" * 60)
    print(f"  runs={runs}  n_epochs={n_epochs}  windows/epoch={k_windows}  "
          f"seed={seed}")

    segments = build_valid_segments(runs, verbose=verbose)
    if not segments:
        print("  ❌ Няма валидни сегменти — провери gwosc/мрежата.")
        return

    epochs = sample_epochs(segments, n_epochs, rng)

    # Resume: пропускаме вече обработени epochs
    done = set()
    if resume and os.path.exists(out_jsonl):
        with open(out_jsonl) as f:
            for line in f:
                try:
                    done.add(round(json.loads(line)['epoch_gps'], 3))
                except Exception:
                    pass
        print(f"  Resume: {len(done)} epoch-а вече в {out_jsonl}")

    analyzer = TriAxisAnalyzerV5(ANALYZER_CONFIG)

    n_ok, n_fail, t_start = 0, 0, time.time()
    with open(out_jsonl, 'a') as fout:
        for i, t0 in enumerate(epochs):
            if round(t0, 3) in done:
                continue
            try:
                recs = process_epoch(analyzer, t0, k_windows, rng,
                                     verbose=verbose)
                for r in recs:
                    r['seed'] = seed
                    fout.write(json.dumps(r) + "\n")
                fout.flush()
                n_ok += 1
                if verbose:
                    el = time.time() - t_start
                    eta = el / max(n_ok, 1) * (len(epochs) - i - 1)
                    print(f"  [{i+1}/{len(epochs)}] GPS {t0:.1f}: "
                          f"{len(recs)} прозореца  (ETA ~{eta/60:.0f} min)")
            except Exception as e:
                n_fail += 1
                print(f"  [{i+1}/{len(epochs)}] GPS {t0:.1f}: "
                      f"пропуснат ({type(e).__name__}: {e})")

    print(f"\n  Готово: {n_ok} epoch-а OK, {n_fail} пропуснати.")
    summarize(out_jsonl)


def summarize(jsonl_path):
    """Кратка статистика + ефективен N."""
    scores_1, scores_max, epochs = [], [], set()
    with open(jsonl_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                scores_1.append(r['triaxis_score'])
                scores_max.append(r['triaxis_score_maxoff'])
                epochs.add(round(r['epoch_gps'], 3))
            except Exception:
                pass
    if not scores_1:
        return
    s1, sm = np.array(scores_1), np.array(scores_max)
    print(f"\n  BACKGROUND РЕЗЮМЕ ({jsonl_path})")
    print(f"    Прозорци: {len(s1)}   Независими epochs: {len(epochs)}")
    print(f"    single-offset : median={np.median(s1):.3f}  "
          f"mean={np.mean(s1):.3f}  std={np.std(s1):.3f}  "
          f"p99={np.percentile(s1, 99):.3f}")
    print(f"    max-over-9    : median={np.median(sm):.3f}  "
          f"mean={np.mean(sm):.3f}  std={np.std(sm):.3f}  "
          f"p99={np.percentile(sm, 99):.3f}")
    print(f"    ⚠ Събитията (best-of-9 offsets) се сравняват с "
          f"max-over-9 колоната, НЕ със single-offset.")


# ==============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-epochs", type=int, default=1700)
    ap.add_argument("--windows-per-epoch", type=int, default=6)
    ap.add_argument("--runs", nargs="+", default=["O1", "O2"],
                    choices=["O1", "O2"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="global_background_v3.jsonl")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summary-only", action="store_true",
                    help="Само статистика върху съществуващ JSONL")
    args = ap.parse_args()

    if args.summary_only:
        summarize(args.out)
    else:
        run(args.n_epochs, args.windows_per_epoch, args.runs,
            args.seed, args.out, args.resume)
