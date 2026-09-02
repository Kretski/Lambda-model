#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_events_to_null.py
===========================
Сравнява TriAxis score-овете на GWTC-1 събитията (от triaxis_validation.json,
получени с best-of-9 offsets) срещу КОРЕКТНИЯ null — max-over-9 колоната от
global_background_v2.jsonl.

Емпиричен p-value: p = (1 + #{bg >= score}) / (1 + N)   [конservативен]
Без σ — разпределението е ляво-изкривено, гаусова апроксимация не важи.

Употреба:
  python compare_events_to_null.py
  python compare_events_to_null.py --events triaxis_validation.json --bg global_background_v2.jsonl
"""

import json, argparse
import numpy as np


STATS = {
    # име: (ключ в bg JSONL, ключ в events JSON)
    'v5':       ('triaxis_score_maxoff', 'triaxis_score'),
    'max_corr': ('max_corr_maxoff',      'max_corr'),
    'v4':       ('v4_legacy_maxoff',     'v4_legacy'),
}


def load_bg(path, key):
    scores = []
    with open(path) as f:
        for line in f:
            try:
                scores.append(json.loads(line)[key])
            except Exception:
                pass
    return np.array(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="triaxis_validation.json")
    ap.add_argument("--bg", default="global_background_v2.jsonl")
    ap.add_argument("--plot", action="store_true",
                    help="Записва хистограма events_vs_null.png")
    ap.add_argument("--stat", default="v5", choices=list(STATS),
                    help="Статистика за сравнение: v5 (клипнат score), "
                         "max_corr (сурова, без сатурация), v4 (legacy)")
    args = ap.parse_args()

    bg_key, ev_key = STATS[args.stat]
    bg = load_bg(args.bg, bg_key)
    if len(bg) == 0:
        print("Няма background данни — провери пътя до JSONL.")
        return
    N = len(bg)

    with open(args.events) as f:
        results = json.load(f)
    events = results.get('events', results)

    print(f"Null: {N} прозореца (max-over-9, stat={args.stat})  "
          f"median={np.median(bg):.3f}  p95={np.percentile(bg, 95):.3f}  "
          f"p99={np.percentile(bg, 99):.3f}  max={bg.max():.3f}")
    print()
    print(f"{'Събитие':<12} {'Тип':<5} {'Score':>7} {'Перцентил':>10} "
          f"{'p-value':>10} {'p(Holm)':>9}  бележка")
    print("-" * 72)

    rows = []
    for name, e in events.items():
        s = e.get(ev_key, e.get('triaxis_score'))
        pct = 100.0 * np.mean(bg < s)
        p = (1 + np.sum(bg >= s)) / (1 + N)
        rows.append((name, e.get('type', '?'), s, pct, p))

    # Holm–Bonferroni: сортирай по p, коригирай с (m-k) множител, наложи монотонност
    rows.sort(key=lambda r: r[4])
    m = len(rows)
    p_holm, run_max = [], 0.0
    for k, r in enumerate(rows):
        adj = min(1.0, (m - k) * r[4])
        run_max = max(run_max, adj)   # step-down монотонност
        p_holm.append(run_max)

    for (name, typ, s, pct, p), ph in zip(rows, p_holm):
        note = ""
        if p < 1 / (1 + N) + 1e-12:
            note = "p е долна граница (над всички bg)"
        elif p > 0.25:
            note = "неразличимо от шум"
        elif p > 0.05:
            note = "слабо; незначимо"
        sig = "✓" if ph < 0.05 else " "
        print(f"{name:<12} {typ:<5} {s:>7.3f} {pct:>9.1f}% {p:>10.4f} "
              f"{ph:>8.4f}{sig}  {note}")

    print()
    print("Забележки:")
    print(f"  • Минималният постижим p с този null е "
          f"{(1/(1+N)):.4f} — за по-нисък трябва по-голям N.")
    print("  • p-стойностите са per-event, без корекция за 11 теста.")
    print("    За глобално твърдение приложи Holm-Bonferroni.")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.hist(bg, bins=60, alpha=0.6, color='gray',
                label=f'Null max-over-9 (n={N})')
        colors = plt.cm.tab10(np.linspace(0, 1, len(rows)))
        for (name, typ, s, pct, p), c in zip(rows, colors):
            ax.axvline(s, color=c, lw=2,
                       label=f"{name} ({s:.3f}, p={p:.3f})")
        ax.set_xlabel('TriAxis score (best-of-9 offsets)')
        ax.set_ylabel('Count')
        ax.set_title('GWTC-1 събития срещу коректен глобален null')
        ax.legend(fontsize=7, loc='upper left')
        fig.tight_layout()
        fig.savefig('events_vs_null.png', dpi=130)
        print("  Записано: events_vs_null.png")


if __name__ == "__main__":
    main()
