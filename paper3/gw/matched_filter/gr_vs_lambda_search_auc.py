#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gr_vs_lambda_search_auc.py
=============================
СТЪПКА 2: GR-only matched filter (Λ фиксиран на 0) СРЕЩУ Λ-search
matched filter (максимум по целия Λ grid), сравнени по detection
efficiency (AUC) за ЧИСТИ GR (Λ=0) инжекции, срещу реален off-source
null.

ВЪПРОС: плащаш ли look-elsewhere цена (по-висок null от max-over-grid),
без реална полза, ако истинските сигнали физически имат Λ=0?

И двата score-a се смятат на СЪЩИТЕ conditioning стъпки (highpass +
tukey + real Welch PSD), валидирани в чат-сесията върху
stage6E3H-R_v2.py (noise≈5.7, injected≈23.5 при target=24 -- реалистично
поведение, потвърдено).

Употреба:
  python gr_vs_lambda_search_auc.py --data H-H1_GWOSC_4KHZ_R1-1126257415-4096.hdf5 \
      --n-null 200 --n-inj-per-snr 40 --out gr_vs_lsearch_results.json
"""

import sys, json, argparse
import numpy as np

sys.path.insert(0, ".")
from pycbc.types import TimeSeries
from pycbc.filter import matched_filter, highpass
from scipy.signal.windows import tukey

# Зарежда всички функции/константи от вече патчнатия stage6E3H-R_v2.py
# (highpass+tukey conditioning вече вградени в неговия main() loop;
# тук реплицираме същото conditioning извън main(), за преизползваемост)
_src = open("stage6E3H-R_v2.py").read().split("def main():")[0]
exec(_src)

LAMBDA_SEARCH_GRID = np.arange(LAMBDA_MIN, LAMBDA_MAX + 0.5 * LAMBDA_STEP, LAMBDA_STEP)


def condition_segment(full_data, gps):
    """Едно и също conditioning навсякъде: extract -> highpass -> tukey."""
    segment = extract_segment(full_data, gps)
    segment = highpass(segment, frequency=F_LOW * 0.9)
    win = tukey(len(segment), alpha=0.1)
    segment = TimeSeries(np.asarray(segment) * win, delta_t=segment.delta_t,
                          epoch=segment.start_time)
    return segment


def score_gr_only(data, gr_template, psd):
    """Score от ЕДИНСТВЕН GR (Λ=0) template."""
    snr = matched_filter(gr_template, data, psd=psd,
                          low_frequency_cutoff=F_LOW, high_frequency_cutoff=F_HIGH)
    return float(np.max(np.abs(np.asarray(snr, dtype=np.complex128))))


def score_lambda_search(data, templates, psd):
    """Score = максимум по целия Λ grid (look-elsewhere search)."""
    best = -np.inf
    for lam, template in templates:
        try:
            snr = matched_filter(template, data, psd=psd,
                                  low_frequency_cutoff=F_LOW, high_frequency_cutoff=F_HIGH)
            s = float(np.max(np.abs(np.asarray(snr, dtype=np.complex128))))
            if np.isfinite(s) and s > best:
                best = s
        except Exception:
            continue
    return best


def auc_rank(pos, neg):
    """AUC чрез Mann-Whitney ранков метод (без параметрични допускания)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    x = np.concatenate([neg, pos])
    ranks = np.argsort(np.argsort(x)) + 1
    r_pos = ranks[len(neg):].sum()
    n1, n0 = len(pos), len(neg)
    return (r_pos - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-null", type=int, default=200,
                     help="Брой независими off-source null реализации")
    ap.add_argument("--n-inj-per-snr", type=int, default=40,
                     help="Брой GR инжекции на всяко SNR ниво")
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--out", default="gr_vs_lsearch_results.json")
    ap.add_argument("--snr-list", default=None,
                     help="Заменя TARGET_SNRS от stage6E3H-R_v2.py с CSV списък "
                          "(напр. '3,4,5,6,7' -- нужен за да се избегне AUC=1.0 "
                          "ceiling effect при SNR>=8, където и двата статистика "
                          "дискриминират перфектно и look-elsewhere цената не се "
                          "вижда в AUC сравнението)")
    args = ap.parse_args()

    global TARGET_SNRS
    if args.snr_list:
        TARGET_SNRS = [float(s) for s in args.snr_list.split(",")]

    rng = np.random.default_rng(args.seed)

    print("[1] Зареждам данни и построявам GR template + Λ-search template bank...")
    full_data, detector = read_losc_hdf5(args.data)
    full_data = full_data.astype(np.float64)
    gr_template = generate_gr_template()

    templates = [(float(lam), apply_lambda_phase(gr_template, float(lam)))
                 for lam in LAMBDA_SEARCH_GRID]
    print(f"    Λ-search templates: {len(templates)} (grid [{LAMBDA_MIN},{LAMBDA_MAX}] step {LAMBDA_STEP})")

    gps_start = float(full_data.start_time)
    duration = len(full_data) * float(full_data.delta_t)
    event_offset = EVENT_GPS - gps_start

    n_needed = args.n_null + args.n_inj_per_snr * len(TARGET_SNRS)
    print(f"[2] Търся {n_needed} независими off-source епохи...")
    gps_times = build_offsource_times(gps_start, duration, event_offset, n_needed)
    if len(gps_times) < n_needed:
        print(f"ГРЕШКА: само {len(gps_times)} налични, нужни {n_needed}.")
        sys.exit(1)
    rng.shuffle(gps_times)

    null_gps = gps_times[:args.n_null]
    inj_gps_pool = gps_times[args.n_null:]

    results = {"null": [], "injections": []}

    # --- OFF-SOURCE NULL ---
    print(f"[3] Off-source null ({args.n_null} реализации)...")
    for i, gps in enumerate(null_gps):
        segment = condition_segment(full_data, gps)
        psd = make_psd(segment)
        s_gr = score_gr_only(segment, gr_template, psd)
        s_ls = score_lambda_search(segment, templates, psd)
        results["null"].append({"gps": gps, "score_gr": s_gr, "score_lsearch": s_ls})
        if (i + 1) % 20 == 0:
            print(f"    null {i+1}/{args.n_null}  score_gr={s_gr:.3f}  score_lsearch={s_ls:.3f}")

    # --- GR (Λ=0) ИНЖЕКЦИИ, различни SNR ---
    idx = 0
    for snr in TARGET_SNRS:
        print(f"[4] GR инжекции SNR={snr} ({args.n_inj_per_snr} реализации)...")
        for i in range(args.n_inj_per_snr):
            gps = inj_gps_pool[idx]
            idx += 1
            segment = condition_segment(full_data, gps)
            psd = make_psd(segment)

            injected_template = apply_lambda_phase(gr_template, 0.0)  # чист GR
            scaled_signal, _, achieved_snr = scale_to_snr(injected_template, psd, snr)
            injected_data = TimeSeries(np.asarray(segment) + np.asarray(scaled_signal),
                                        delta_t=segment.delta_t, epoch=segment.start_time)

            s_gr = score_gr_only(injected_data, gr_template, psd)
            s_ls = score_lambda_search(injected_data, templates, psd)
            results["injections"].append({
                "gps": gps, "target_snr": snr, "achieved_snr": achieved_snr,
                "score_gr": s_gr, "score_lsearch": s_ls,
            })
            if (i + 1) % 10 == 0:
                print(f"    SNR={snr} {i+1}/{args.n_inj_per_snr}  "
                      f"score_gr={s_gr:.3f}  score_lsearch={s_ls:.3f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nЗаписано: {args.out}")

    # --- AUC ПО SNR ---
    null_gr = [r["score_gr"] for r in results["null"]]
    null_ls = [r["score_lsearch"] for r in results["null"]]

    print(f"\n{'SNR':>6}  {'n':>4}  {'AUC GR-only':>12}  {'AUC Λ-search':>14}  {'delta':>8}")
    for snr in TARGET_SNRS:
        inj_gr = [r["score_gr"] for r in results["injections"] if r["target_snr"] == snr]
        inj_ls = [r["score_lsearch"] for r in results["injections"] if r["target_snr"] == snr]
        if not inj_gr:
            continue
        auc_gr = auc_rank(inj_gr, null_gr)
        auc_ls = auc_rank(inj_ls, null_ls)
        print(f"{snr:>6.1f}  {len(inj_gr):>4}  {auc_gr:>12.4f}  {auc_ls:>14.4f}  {auc_ls-auc_gr:>+8.4f}")

    print(f"\nNull ниво (медиана): GR-only={np.median(null_gr):.3f}  "
          f"Λ-search={np.median(null_ls):.3f}  "
          f"(Λ-search look-elsewhere inflation: {np.median(null_ls)-np.median(null_gr):+.3f})")


if __name__ == "__main__":
    main()
