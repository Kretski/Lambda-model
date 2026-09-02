#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_signal.py
================
"Какво вижда статистиката" — илюстративна фигура (Figure 1 за текста):

  (a) Whitened H1 и L1 strain в 0.25 s прозореца на събитието; L1 е
      изместен с ИЗМЕРЕНИЯ delay и умножен по знака на корелационния
      пик (антенната ориентация на детекторите дава относителна
      полярност — за GW150914 L1 е прочуто инвертиран).
  (b) Нормализираната крос-корелация по lag; физичната ±10 ms зона е
      защрихована; пикът (= max_corr, = измереният delay) е маркиран.
  (c)+(d) Същото за шумов прозорец на −30 s от събитието — контрастът.

Употреба:
  python plot_signal.py --event GW150914
  python plot_signal.py --event GW170814 --noise-offset -45
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GW_EVENTS = {
    "GW150914": 1126259462.4, "GW151012": 1128678900.4, "GW151226": 1135136350.6,
    "GW170104": 1167559936.6, "GW170608": 1180922494.5, "GW170729": 1185389807.3,
    "GW170809": 1186302519.8, "GW170814": 1186741861.5, "GW170817": 1187008882.4,
    "GW170818": 1187058327.1, "GW170823": 1187529256.5,
}
SAMPLE_RATE = 4096
WIN = 0.25
OFFSET_GRID_MS = [-100, -75, -50, -25, 0, 25, 50, 75, 100]
PHYS_MS = 10.0


def window(x, times, center):
    i = np.argmin(np.abs(times - center))
    h = int(WIN * SAMPLE_RATE / 2)
    return x[i - h:i + h], times[i - h:i + h] - center


def corr_full(a, b):
    an = (a - a.mean()) / (a.std() + 1e-10)
    bn = (b - b.mean()) / (b.std() + 1e-10)
    c = np.correlate(an, bn, mode='same') / len(a)
    lags_ms = (np.arange(len(c)) - len(c) // 2) / SAMPLE_RATE * 1000.0
    return lags_ms, c


def best_of_9(h1, l1, times, gps):
    best = None
    for off in OFFSET_GRID_MS:
        w1, _ = window(h1, times, gps + off / 1000.0)
        w2, _ = window(l1, times, gps + off / 1000.0)
        lags, c = corr_full(w1, w2)
        i = np.argmax(np.abs(c))
        if best is None or abs(c[i]) > abs(best[0]):
            best = (c[i], lags[i], off, lags, c)
    return best  # (corr_at_peak_signed, delay_ms, offset_ms, lags, c)


def panel_pair(axL, axR, h1, l1, times, center, label, color_sig):
    peak, delay_ms, off, lags, c = best_of_9(h1, l1, times, center)
    ct = center + off / 1000.0
    w1, t1 = window(h1, times, ct)
    w2, _ = window(l1, times, ct)

    # L1: изместен с измерения delay, знак = знака на пика
    shift = int(round(delay_ms / 1000.0 * SAMPLE_RATE))
    w2s = np.roll(w2, -shift) * np.sign(peak)

    axL.plot(t1 * 1000, w1, lw=0.9, color='tab:blue', label='H1')
    axL.plot(t1 * 1000, w2s, lw=0.9, color='tab:orange',
             label=f'L1 (изм. {delay_ms:+.1f} ms'
                   f'{", инверт." if peak < 0 else ""})')
    axL.set_xlabel(f"Време спрямо GPS{off:+d}ms (ms)")
    axL.set_ylabel("Whitened strain (σ)")
    axL.set_title(label)
    axL.legend(fontsize=8, loc='upper left')

    axR.axvspan(-PHYS_MS, PHYS_MS, color=color_sig, alpha=0.12,
                label='физичен ±10 ms')
    axR.plot(lags, c, lw=1.0, color='0.3')
    axR.plot([delay_ms], [peak], 'v', color='tab:red', ms=9,
             label=f'|max_corr|={abs(peak):.3f} @ {delay_ms:+.2f} ms')
    axR.set_xlim(-125, 125)
    axR.set_xlabel("Lag (ms)")
    axR.set_ylabel("Норм. крос-корелация")
    axR.set_title(f"{label}: корелационна функция")
    axR.legend(fontsize=8, loc='upper left')
    return abs(peak), delay_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="GW150914")
    ap.add_argument("--noise-offset", type=float, default=-30.0,
                    help="s спрямо събитието за контрастния шумов прозорец")
    args = ap.parse_args()
    gps = GW_EVENTS[args.event]

    from gwpy.timeseries import TimeSeries
    span = max(40.0, abs(args.noise_offset) + 15.0)
    print(f"Fetch {args.event} ±{span:.0f}s...")
    h1 = TimeSeries.fetch_open_data("H1", gps - span, gps + span,
                                    sample_rate=SAMPLE_RATE, cache=False)
    l1 = TimeSeries.fetch_open_data("L1", gps - span, gps + span,
                                    sample_rate=SAMPLE_RATE, cache=False)
    h1p = h1.whiten(fftlength=0.25, overlap=0.1).bandpass(30, 500)
    l1p = l1.whiten(fftlength=0.25, overlap=0.1).bandpass(30, 500)
    h1v, l1v, times = h1p.value, l1p.value, h1p.times.value

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))
    mc_e, d_e = panel_pair(axes[0, 0], axes[0, 1], h1v, l1v, times, gps,
                           f"(a) {args.event}", 'tab:green')
    mc_n, d_n = panel_pair(axes[1, 0], axes[1, 1], h1v, l1v, times,
                           gps + args.noise_offset,
                           f"(c) Шум ({args.noise_offset:+.0f} s)", 'tab:green')
    axes[1, 0].set_title(f"(c) Шумов прозорец ({args.noise_offset:+.0f} s)")
    axes[1, 1].set_title("(d) Шум: корелационна функция")
    axes[0, 1].set_title(f"(b) {args.event}: корелационна функция")

    fig.suptitle(f"{args.event}: какво вижда max_corr статистиката "
                 f"(събитие {mc_e:.3f} срещу шум {mc_n:.3f})", fontsize=12)
    fig.tight_layout()
    out = f"figure_signal_{args.event}.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Записано: {out} (+ .pdf)")
    print(f"  Събитие: |max_corr|={mc_e:.3f} @ {d_e:+.2f} ms")
    print(f"  Шум:     |max_corr|={mc_n:.3f} @ {d_n:+.2f} ms")


if __name__ == "__main__":
    main()
