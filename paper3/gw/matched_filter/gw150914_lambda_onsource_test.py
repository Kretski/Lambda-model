#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gw150914_lambda_onsource_test.py
===================================
GW150914 ON-SOURCE Λ тест: правилно калибрирана версия, заменяща
двата предишни НЕВАЛИДНИ опита в тази кодова база:
  - lambda_analysis.py: np.random.normal() fake резултат (Λ=-2.78 "NON-GR")
  - sterile_lambda_analysis.py: boundary-hit artifact (Λ=-30, ръба на grid-а)

Използва ВЕЧЕ ВАЛИДИРАНИЯ pipeline от stage6E3H-R_v2.py (highpass+tukey
conditioning, alias-safe Λ grid, потвърден с injection-recovery
correlation=0.81 в чат-сесията).

Метод:
  1. Off-source null: recover_lambda() на много independent
     off-source сегменти (без инжекция) -> разпределение на
     "естествения" recovered Λ от чист шум.
  2. On-source: recover_lambda() ТОЧНО на GW150914 мержъра.
  3. z-score / percentile на on-source спрямо off-source null.

ВАЖНО: Λ grid-ът е ЕДНАКЪВ и за null-а, и за on-source (alias-safe
[-0.5,0.5], идентично на вече валидираната калибрация) -- ако
on-source recovered Λ падне НА РЪБА на този grid, това е сигнал, че
или (а) обхватът е твърде тесен за реалния ефект (нужен по-широк,
внимателно alias-checked grid), или (б) е artifact, НЕ физически
резултат -- скриптът изрично флагва това, вместо да го подмине.

Употреба:
  python gw150914_lambda_onsource_test.py \
      --data H-H1_GWOSC_4KHZ_R1-1126257415-4096.hdf5 \
      --n-null 60 --out gw150914_lambda_result.json
"""

import sys, json, argparse
import numpy as np

sys.path.insert(0, ".")
from pycbc.types import TimeSeries
from pycbc.filter import matched_filter, highpass
from scipy.signal.windows import tukey

_src = open("stage6E3H-R_v2.py").read().split("def main():")[0]
exec(_src)


def condition_segment(full_data, gps):
    segment = extract_segment(full_data, gps)
    segment = highpass(segment, frequency=F_LOW * 0.9)
    win = tukey(len(segment), alpha=0.1)
    segment = TimeSeries(np.asarray(segment) * win, delta_t=segment.delta_t,
                          epoch=segment.start_time)
    return segment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-null", type=int, default=60)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", default="gw150914_lambda_result.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print("[1] Зареждам данни и построявам Λ-search template bank...")
    full_data, detector = read_losc_hdf5(args.data)
    full_data = full_data.astype(np.float64)
    gr_template = generate_gr_template()

    templates = [(float(lam), apply_lambda_phase(gr_template, float(lam)))
                 for lam in np.arange(LAMBDA_MIN, LAMBDA_MAX + 0.5 * LAMBDA_STEP, LAMBDA_STEP)]
    print(f"    Λ grid: [{LAMBDA_MIN},{LAMBDA_MAX}] step {LAMBDA_STEP}  "
          f"({len(templates)} темплейта)")

    gps_start = float(full_data.start_time)
    duration = len(full_data) * float(full_data.delta_t)
    event_offset = EVENT_GPS - gps_start

    print(f"[2] Търся {args.n_null} off-source епохи (избягвайки GW150914)...")
    null_gps_times = build_offsource_times(gps_start, duration, event_offset, args.n_null)
    if len(null_gps_times) < args.n_null:
        print(f"ГРЕШКА: само {len(null_gps_times)} налични, нужни {args.n_null}.")
        sys.exit(1)

    print("[3] OFF-SOURCE NULL Λ recovery...")
    null_lambdas = []
    null_scores = []
    for i, gps in enumerate(null_gps_times):
        segment = condition_segment(full_data, gps)
        psd = make_psd(segment)
        best_lambda, best_score = recover_lambda(segment, templates, psd)
        if best_lambda is not None:
            null_lambdas.append(best_lambda)
            null_scores.append(best_score)
        if (i + 1) % 15 == 0:
            print(f"    null {i+1}/{args.n_null}  recovered_Λ={best_lambda:+.3f}  score={best_score:.3f}")

    null_lambdas = np.array(null_lambdas)
    null_scores = np.array(null_scores)

    bg_mean = float(np.mean(null_lambdas))
    bg_std = float(np.std(null_lambdas))
    bg_at_edge_frac = float(np.mean(np.isclose(np.abs(null_lambdas), LAMBDA_MAX, atol=LAMBDA_STEP / 2)))

    print(f"\nOff-source null: mean={bg_mean:+.4f}  std={bg_std:.4f}  "
          f"n={len(null_lambdas)}  frac_at_grid_edge={bg_at_edge_frac:.3f}")

    print(f"\n[4] ON-SOURCE GW150914 (GPS={EVENT_GPS})...")
    onsource_segment = condition_segment(full_data, EVENT_GPS)
    onsource_psd = make_psd(onsource_segment)
    on_lambda, on_score = recover_lambda(onsource_segment, templates, onsource_psd)

    at_edge = on_lambda is not None and np.isclose(abs(on_lambda), LAMBDA_MAX, atol=LAMBDA_STEP / 2)

    print(f"    On-source recovered Λ = {on_lambda:+.4f}")
    print(f"    On-source score       = {on_score:.4f}")
    print(f"    На ръба на grid-а?    = {at_edge}")

    if bg_std > 0:
        z_score = (on_lambda - bg_mean) / bg_std
    else:
        z_score = float("nan")

    percentile = float(np.mean(null_lambdas <= on_lambda) * 100)

    print(f"\n{'='*70}")
    print("GW150914 Λ РЕЗУЛТАТ (валидиран pipeline)")
    print(f"{'='*70}")
    print(f"On-source Λ           = {on_lambda:+.4f}")
    print(f"Off-source null: mean = {bg_mean:+.4f}, std = {bg_std:.4f} (n={len(null_lambdas)})")
    print(f"z-score                = {z_score:+.4f}σ")
    print(f"Percentile spрямо null = {percentile:.1f}%")
    print(f"На ръба на grid-а       = {at_edge}")

    if at_edge:
        print("\n⚠️  ВНИМАНИЕ: on-source resultatят е ТОЧНО на ръба на Λ grid-а.")
        print("    Това Е artifact сигнатура (виж diagnостиката за sterile_lambda_analysis.py")
        print("    и injection_recovery_lambda.py в чат-сесията) -- НЕ физически резултат.")
        print("    Не докладвай тази стойност като Λ estimate без разширяване/проверка на grid-а.")
    elif abs(z_score) < 2:
        print("\n-> В рамките на 2σ от off-source null. НЕ Е статистически значима аномалия.")
        print("   Консистентно с GR (Λ=0).")
    else:
        print(f"\n-> {abs(z_score):.1f}σ отклонение от null. Заслужава по-внимателна проверка")
        print("   (по-голям null, look-elsewhere корекция, независима реимплементация)")
        print("   ПРЕДИ да се твърди каквото и да е за физическа значимост.")

    result = {
        "on_source_lambda": on_lambda,
        "on_source_score": on_score,
        "on_source_at_grid_edge": bool(at_edge),
        "null_mean": bg_mean,
        "null_std": bg_std,
        "null_n": len(null_lambdas),
        "null_frac_at_grid_edge": bg_at_edge_frac,
        "z_score": z_score,
        "percentile": percentile,
        "lambda_grid": [LAMBDA_MIN, LAMBDA_MAX, LAMBDA_STEP],
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nЗаписано: {args.out}")


if __name__ == "__main__":
    main()
