#!/usr/bin/env python3
"""
analyze_delay_vs_lambda.py

Задача 1: деградира ли delay reconstruction accuracy на TriAxis V5
при Λ-деформирани инжекции?

Чете наличните inj_lambda_*.jsonl батчове, смята
    delay_error = |delay_ms_measured - delay_true_ms|
и сравнява разпределенията по |Λ|.

Употреба:
    conda activate lal_env
    python analyze_delay_vs_lambda.py \
        --files inj_lambda_L0.jsonl inj_lambda_L5.jsonl inj_lambda_Lm5.jsonl

Ако Λ не е записан във всеки ред, се извлича от името на файла
(L0 -> 0, L5 -> +5, Lm5 -> -5, L10 -> +10, Lm20 -> -20).

Нищо не се пуска паралелно, нищо не се записва — само чете.
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

# --- гъвкаво разпознаване на имена на полета -------------------------------
MEASURED_KEYS = ["delay_ms_measured", "delay_measured_ms", "delay_ms", "measured_delay_ms"]
TRUE_KEYS     = ["delay_true_ms", "true_delay_ms", "delay_ms_true", "injected_delay_ms"]
LAMBDA_KEYS   = ["lambda_true", "lambda", "Lambda", "lambda_injected", "lam"]
SNR_KEYS      = ["snr", "snr_target", "fix_snr", "network_snr", "injected_snr"]
SCORE_KEYS    = ["max_corr", "score", "corr", "detection_score"]
DETECT_KEYS   = ["detected", "is_detected", "recovered"]


def pick(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def lambda_from_filename(path):
    base = os.path.basename(path)
    m = re.search(r"_L(m?)(\d+(?:p\d+)?)", base)
    if not m:
        return None
    sign = -1.0 if m.group(1) == "m" else 1.0
    return sign * float(m.group(2).replace("p", "."))


def load(path):
    """Чете .jsonl толерантно: CRLF, празни редове, счупени редове."""
    rows, bad = [], 0
    lam_file = lambda_from_filename(path)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip().lstrip("\ufeff")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if not isinstance(rec, dict):
                bad += 1
                continue
            rec["_lambda"] = pick(rec, LAMBDA_KEYS)
            if rec["_lambda"] is None:
                rec["_lambda"] = lam_file
            rec["_src"] = os.path.basename(path)
            rows.append(rec)
    return rows, bad


# --- статистика без scipy (за да върви и в гол env) ------------------------
def pct(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def mannwhitney_u(a, b):
    """Двустранен U тест с нормална апроксимация + tie correction."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    merged = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks, i = [0.0] * len(merged), 0
    tie_term = 0.0
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        t = j - i + 1
        tie_term += t ** 3 - t
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    r1 = sum(r for r, (_, g) in zip(ranks, merged) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    n = n1 + n2
    mu = n1 * n2 / 2.0
    var = n1 * n2 * (n ** 3 - n - tie_term) / (12.0 * n * (n - 1))
    if var <= 0:
        return u1, float("nan")
    z = (u1 - mu) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return u1, p


def cliffs_delta(a, b):
    """Ефект-сайз, устойчив на outliers. |d|<0.15 ~ нищожен."""
    if not a or not b:
        return float("nan")
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / float(len(a) * len(b))


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    if len(xs) < 3:
        return float("nan"), float("nan")
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    if den == 0:
        return float("nan"), float("nan")
    rho = num / den
    t = rho * math.sqrt(max(len(xs) - 2, 1) / max(1e-12, 1 - rho ** 2))
    p = math.erfc(abs(t) / math.sqrt(2.0))  # груба апроксимация
    return rho, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--snr-min", type=float, default=None,
                    help="ако батчовете смесват SNR, ограничи до един SNR слой")
    ap.add_argument("--snr-max", type=float, default=None)
    ap.add_argument("--detected-only", action="store_true",
                    help="използвай само редове с detected/recovered=True")
    ap.add_argument("--edge-ms", type=float, default=None,
                    help="абсолютен предел на delay search range (ms); "
                         "проверява натрупване на ръба")
    args = ap.parse_args()

    all_rows, total_bad = [], 0
    for f in args.files:
        if not os.path.exists(f):
            print(f"!! липсва: {f}", file=sys.stderr)
            continue
        rows, bad = load(f)
        total_bad += bad
        print(f"  {os.path.basename(f):<28} n={len(rows):<5} (счупени редове: {bad})")
        all_rows += rows

    if not all_rows:
        sys.exit("Няма прочетени редове.")
    if total_bad:
        print(f"  ВНИМАНИЕ: общо {total_bad} нечетими реда (частичен/прекъснат batch?)")

    # диагностика на схемата — веднъж, за да не гадаем
    sample = all_rows[0]
    print("\nПолета в първия ред:", ", ".join(sorted(sample.keys())))

    groups = defaultdict(list)
    raw_signed = defaultdict(list)
    skipped = 0
    for r in all_rows:
        meas, true = pick(r, MEASURED_KEYS), pick(r, TRUE_KEYS)
        if meas is None or true is None or r["_lambda"] is None:
            skipped += 1
            continue
        if args.detected_only:
            det = pick(r, DETECT_KEYS)
            if det is not None and not det:
                continue
        snr = pick(r, SNR_KEYS)
        if snr is not None:
            if args.snr_min is not None and snr < args.snr_min:
                continue
            if args.snr_max is not None and snr > args.snr_max:
                continue
        err = abs(float(meas) - float(true))
        groups[float(r["_lambda"])].append(err)
        raw_signed[float(r["_lambda"])].append(float(meas) - float(true))

    if skipped:
        print(f"  пропуснати редове (липсващи delay/Λ полета): {skipped}")

    print("\n=== Delay error по Λ ===")
    print(f"{'Λ':>7} {'n':>5} {'median':>8} {'p68':>8} {'p90':>8} {'p95':>8} "
          f"{'<2ms':>7} {'bias':>8}")
    for lam in sorted(groups):
        e = groups[lam]
        s = raw_signed[lam]
        frac2 = 100.0 * sum(1 for x in e if x <= 2.0) / len(e)
        bias = sum(s) / len(s)
        print(f"{lam:>7.1f} {len(e):>5} {pct(e,50):>8.3f} {pct(e,68):>8.3f} "
              f"{pct(e,90):>8.3f} {pct(e,95):>8.3f} {frac2:>6.1f}% {bias:>8.3f}")

    # референтна стойност от Zenodo депозита
    print("\n  (публикувано за GR сигнали: median 0.33 ms, 96% within 2 ms)")

    # --- сравнение Λ=0 vs всяко |Λ|>0 ---
    base = groups.get(0.0, [])
    if base:
        print("\n=== Λ=0 срещу всяко Λ≠0 (Mann-Whitney, двустранен) ===")
        for lam in sorted(groups):
            if lam == 0.0:
                continue
            u, p = mannwhitney_u(base, groups[lam])
            d = cliffs_delta(groups[lam], base)
            dmed = pct(groups[lam], 50) - pct(base, 50)
            flag = "  <-- значимо" if p == p and p < 0.05 else ""
            print(f"  Λ={lam:>5.1f}: Δmedian={dmed:+.3f} ms, "
                  f"Cliff's δ={d:+.3f}, p={p:.3f}{flag}")
        print("  (|δ|<0.15 нищожен, <0.33 малък, <0.47 среден ефект)")
    else:
        print("\n!! Няма Λ=0 batch — няма спрямо какво да се сравнява.")

    # --- монотонност спрямо |Λ| ---
    xs, ys = [], []
    for lam, e in groups.items():
        for v in e:
            xs.append(abs(lam))
            ys.append(v)
    rho, p = spearman(xs, ys)
    print(f"\n=== Монотонност: Spearman(|Λ|, delay_error) "
          f"rho={rho:+.3f}, p={p:.3f}, n={len(xs)} ===")
    if len(set(xs)) < 3:
        print("  ВНИМАНИЕ: само {} различни |Λ| стойности — трендът е "
              "практически неопределим. Пусни ±10, ±20.".format(len(set(xs))))

    # --- sanity: натрупване на ръба на search range ---
    if args.edge_ms is not None:
        print(f"\n=== Edge check (|delay_measured| близо до ±{args.edge_ms} ms) ===")
        for lam in sorted(groups):
            rows = [r for r in all_rows if r["_lambda"] == lam]
            meas = [pick(r, MEASURED_KEYS) for r in rows]
            meas = [float(m) for m in meas if m is not None]
            if not meas:
                continue
            frac = 100.0 * sum(1 for m in meas
                               if abs(abs(m) - args.edge_ms) < 0.05 * args.edge_ms) / len(meas)
            flag = "  <-- артефакт?" if frac > 10 else ""
            print(f"  Λ={lam:>5.1f}: {frac:5.1f}% на ръба{flag}")
    else:
        print("\n(подай --edge-ms <предел на delay search>, за да се провери "
              "натрупване на границата — уроците от Λ-recovery aliasing bug-а)")

    # --- груба оценка на мощността ---
    if base:
        n_min = min(len(v) for v in groups.values())
        print(f"\n=== Мощност ===")
        print(f"  Най-малка група: n={n_min}. При n≈60 vs 60 Mann-Whitney хваща "
              f"надеждно (80% power) едва Cliff's δ≈0.30 —")
        print(f"  т.е. ~35-40% изместване на разпределението. По-фина деградация "
              f"(<20%) НЕ е разграничима с наличните данни.")
        print(f"  Ако резултатът излезе null, това е горната граница на "
          f"твърдението: 'няма ефект > ~35%', не 'няма ефект'.")


if __name__ == "__main__":
    main()
