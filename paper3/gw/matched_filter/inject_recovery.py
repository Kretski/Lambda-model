#!/usr/bin/env python3
"""
inject_recovery.py -- does Stage B measure B, or return the grid?
=================================================================

REPLACES the earlier version, whose "6/6 recovered" verdict was wrong.
That version tested |B_hat - B_inj| < 2 sigma, but sigma itself was of
the order of the grid half-width, so the test compared an error against
an equally large error and could not fail. Its own numbers already
contradicted its verdict:

    B_inj:  0        4.68e-9   8.19e-9   1.17e-8   1.76e-8
    B_hat: -2.341e-8 -2.185e-8 -1.795e-8 -1.483e-8 -8.58e-9
    diff:  -2.34e-8  -2.65e-8  -2.61e-8  -2.65e-8  -2.61e-8

The difference is nearly constant, so d(B_hat)/d(B_inj) ~ 1: the
injected deformation DOES reach the profiler at full strength. What is
wrong is an offset of about -2.6e-8, slightly more than one grid
half-width. That rules out most failure modes (wrong phase, wrong
exponent, double application, normalisation) and points at one thing:
the scan spans a single alias period, so a maximum that truly sits
outside it wraps back to an edge.

WHAT THIS VERSION DOES
----------------------
  1. Scans +/- N_PERIODS alias periods instead of one.
  2. Records the FULL rho(B) curve and every local maximum, not just
     the global one.
  3. Fits B_hat = a + b * B_inj for both the global maximum and the
     maximum nearest the injected value ("unwrapped").
  4. Reports the offset in units of the alias period, so an integer
     value identifies wrapping rather than bias.
  5. Compares rho at B_inj against rho at 0: does the profile actually
     prefer the injected value?

READING THE RESULT
------------------
  slope ~ 1, offset ~ integer * period, maxima repeat at period
      -> alias wrapping. The method works; B is identifiable only
         modulo the alias period. This is NOT licence to widen the
         physical scan -- it is a statement to make explicitly.

  slope ~ 1, offset not a multiple of the period
      -> a genuine bias in profile_b(). Fix before quoting anything.

  slope far from 1, or rho(B_inj) not above rho(0)
      -> the deformation is not reaching the likelihood at full
         strength after all. Check the injection path.

The physical scan in mdr_search.py stays at one alias period. This
script is diagnostic only and does not change the science config.

Usage:
    python3 inject_recovery.py           # alpha = 4
    python3 inject_recovery.py 0.0       # another alpha
"""

import json
import sys

import numpy as np

import mdr_search as M

# ======================================================================
# CONFIG
# ======================================================================
EVENT_GPS = 1126259462.428          # GW150914 as recovered by the search
H1 = "H-H1_GWOSC_4KHZ_R1-1126257415-4096.hdf5"
TEMPLATE = (36.0, 36.0, 0.0, 0.0)   # the template the search selected

ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0

N_PERIODS = 3           # scan +/- this many alias periods
PTS_PER_PERIOD = 41     # resolution within one period
FRACTIONS = [0.0, 0.10, 0.25, 0.50, 0.75, -0.25, -0.50, -0.75]
N_NULL = 15             # off-source segments for the null spread

OUT_JSON = f"injection_alpha{ALPHA:g}_wide.json"


# ----------------------------------------------------------------------
def profile_curve(block, psd, hp, freqs, alpha, grid, centre, win=0.05):
    """Full rho(B) curve rather than just its argmax."""
    rho = np.full(len(grid), -1.0)
    for i, b in enumerate(grid):
        tm = M.deform(hp, freqs, alpha, b)
        snr = M.matched_filter(tm, block, psd=psd,
                               low_frequency_cutoff=M.F_LOW,
                               high_frequency_cutoff=M.F_HIGH)
        snr = snr.crop(M.OVERLAP / 2, M.OVERLAP / 2)
        t = snr.sample_times.numpy()
        m = np.abs(t - centre) < win
        if m.any():
            rho[i] = np.abs(snr.numpy())[m].max()
    return rho


def local_maxima(grid, rho, min_prominence=0.005):
    """Interior local maxima, filtered by prominence above neighbours."""
    out = []
    for i in range(1, len(rho) - 1):
        if rho[i] > rho[i - 1] and rho[i] >= rho[i + 1]:
            prom = rho[i] - min(rho[i - 1], rho[i + 1])
            if prom >= min_prominence:
                out.append((grid[i], rho[i]))
    return out


def inject(block, alpha, b_inj):
    """Apply the dispersion phase to the conditioned strain itself."""
    stilde = block.to_frequencyseries(delta_f=1.0 / M.BLOCK)
    if b_inj != 0.0:
        f_all = stilde.sample_frequencies.numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            ph = f_all ** (alpha - 1.0)
        ph[~np.isfinite(ph)] = 0.0
        band = (f_all >= M.F_LOW) & (f_all <= M.F_HIGH)
        data = stilde.data.copy()
        data[band] *= np.exp(-1j * b_inj * ph[band])
        stilde.data[:] = data
    out = stilde.to_timeseries()
    out.start_time = block.start_time
    return out


# ======================================================================
def main():
    print("=" * 74)
    print(f"INJECTION RECOVERY -- WIDE SCAN -- alpha = {ALPHA}")
    print("=" * 74)

    ts = M.condition(M.load_gwosc(H1))
    block = ts.time_slice(EVENT_GPS - M.BLOCK / 2, EVENT_GPS + M.BLOCK / 2)
    psd = M.block_psd(block)
    hp = M.fd_template(TEMPLATE, 1.0 / M.BLOCK, len(psd))
    freqs = psd.sample_frequencies.numpy()

    narrow, period = M.b_grid(ALPHA)
    half = narrow[-1]
    span = N_PERIODS * period
    npts = 2 * N_PERIODS * PTS_PER_PERIOD + 1
    wide = np.linspace(-span, span, npts)

    print(f"  alias period   : {period:.6e}")
    print(f"  physical grid  : +/-{half:.4e}  (unchanged in mdr_search)")
    print(f"  diagnostic grid: +/-{span:.4e}  ({npts} points, "
          f"{N_PERIODS} periods each side)")
    print(f"  template       : {TEMPLATE[:2]}   event {EVENT_GPS:.3f}")
    print()

    # ---- null spread on the PHYSICAL grid ----------------------------
    rng = np.random.default_rng(11)
    lo, hi = float(ts.start_time) + M.BLOCK, float(ts.end_time) - M.BLOCK
    centres = []
    while len(centres) < N_NULL:
        c = rng.uniform(lo, hi)
        if abs(c - EVENT_GPS) > M.OFFSOURCE_GAP:
            centres.append(c)

    print(f"  null spread over {N_NULL} off-source segments...")
    null = []
    for c in centres:
        blk = ts.time_slice(c - M.BLOCK / 2, c + M.BLOCK / 2)
        p = M.block_psd(blk)
        h = M.fd_template(TEMPLATE, 1.0 / M.BLOCK, len(p))
        b, _, _ = M.profile_b(blk, p, h, p.sample_frequencies.numpy(),
                              ALPHA, narrow, c)
        null.append(b)
    null = np.array(null)
    sigma = null.std(ddof=1)
    edge_frac = float((np.abs(np.abs(null) - half) < 1e-12).mean())
    print(f"  null sigma     : {sigma:.4e}")
    print(f"  null at edge   : {100*edge_frac:.0f}% of segments")
    if edge_frac > 0.5:
        print("  >> Most null segments land on the edge: on the physical")
        print("     grid this sigma measures the grid, not the noise.")
    print()

    # ---- the ladder ---------------------------------------------------
    print(f"  {'B_inj':>13} {'B_glob':>13} {'B_near':>13} "
          f"{'off/period':>11} {'nmax':>5} {'d_rho2':>8}")
    rows = []
    for frac in FRACTIONS:
        b_inj = frac * half
        blk = inject(block, ALPHA, b_inj)
        rho = profile_curve(blk, psd, hp, freqs, ALPHA, wide, EVENT_GPS)

        b_glob = float(wide[int(np.argmax(rho))])
        maxima = local_maxima(wide, rho)
        if maxima:
            b_near = float(min(maxima, key=lambda m: abs(m[0] - b_inj))[0])
        else:
            b_near = b_glob

        # does the profile prefer the injected value over zero?
        i_inj = int(np.argmin(np.abs(wide - b_inj)))
        i_zero = int(np.argmin(np.abs(wide)))
        d_rho2 = float(rho[i_inj] ** 2 - rho[i_zero] ** 2)

        off = (b_glob - b_inj) / period
        print(f"  {b_inj:>+13.4e} {b_glob:>+13.4e} {b_near:>+13.4e} "
              f"{off:>+11.3f} {len(maxima):>5} {d_rho2:>+8.2f}")

        rows.append(dict(fraction=frac, B_injected=float(b_inj),
                         B_global=b_glob, B_nearest=b_near,
                         offset_periods=float(off), n_maxima=len(maxima),
                         d_rho2_inj_vs_zero=d_rho2,
                         maxima=[[float(x), float(y)] for x, y in maxima]))

    # ---- fits ----------------------------------------------------------
    bi = np.array([r["B_injected"] for r in rows])
    bg = np.array([r["B_global"] for r in rows])
    bn = np.array([r["B_nearest"] for r in rows])
    sg, ag = np.polyfit(bi, bg, 1)
    sn, an = np.polyfit(bi, bn, 1)

    print()
    print("=" * 74)
    print("FITS   B_hat = a + b * B_injected")
    print("=" * 74)
    print(f"  global maximum : b = {sg:+.4f}   a = {ag:+.4e}  "
          f"({ag/period:+.3f} periods)")
    print(f"  nearest maximum: b = {sn:+.4f}   a = {an:+.4e}  "
          f"({an/period:+.3f} periods)")

    offs = np.array([r["offset_periods"] for r in rows])
    near_int = np.abs(offs - np.round(offs))
    nmax = np.array([r["n_maxima"] for r in rows])
    drho = np.array([r["d_rho2_inj_vs_zero"] for r in rows])

    print()
    print("=" * 74)
    print("VERDICT")
    print("=" * 74)

    prefers = (drho[bi != 0] > 0).mean() if (bi != 0).any() else 0.0
    strongest = float(np.max(drho)) if len(drho) else 0.0

    if nmax.max() == 0:
        # No local maxima anywhere in N_PERIODS periods: rho(B) is
        # monotonic. The global maximum is then wherever the curve
        # happened to stop rising, so its offset carries no meaning and
        # the integer-period test below cannot be applied. This is not
        # a bias -- it is the absence of anything to measure.
        print("  No local maxima over "
              f"{2*N_PERIODS} alias periods: the profile is monotonic.")
        print(f"  Largest discrimination is dRho^2 = {strongest:+.2f}, "
              "against ~1")
        print("  expected from one free parameter in noise alone.")
        print()
        print("  => NOT MEASURABLE at this alpha. The unit slope shows the")
        print("     injected phase does reach the likelihood, but t_c and")
        print("     phi_c absorb it, leaving no maximum to locate. The")
        print("     offset printed above is where a monotonic curve was")
        print("     truncated, not a bias. Report this alpha as")
        print("     unconstrained; do not quote its interval as a bound.")
        print("     Run fisher_alpha.py for the analytic reason.")
    elif abs(sn - 1.0) < 0.2 and abs(an) < 0.3 * period:
        print("  Unwrapped recovery is unbiased: the nearest local maximum")
        print("  tracks the injection with unit slope and no offset.")
        if near_int.max() < 0.25 and nmax.max() > 1:
            print("  Global-maximum offsets are near-integer multiples of the")
            print("  alias period, and multiple maxima are present.")
            print()
            print("  => ALIAS WRAPPING, not a bug. B is identifiable only")
            print("     modulo the alias period. Keep the physical scan at")
            print("     one period and state this limitation explicitly:")
            print("     the quoted intervals are modulo-period, and where")
            print("     the profile is flat they reflect the grid, not the")
            print("     data.")
        else:
            print("  => Recovery works on the wide grid; the narrow-grid")
            print("     result was a range problem.")
    elif abs(sg - 1.0) < 0.2 or abs(sn - 1.0) < 0.2:
        print(f"  Slope is ~1 but the offset ({an/period:+.3f} periods) is")
        print("  not an integer multiple of the alias period.")
        print("  => GENUINE BIAS in profile_b(). Do not quote any interval")
        print("     until this is found.")
    else:
        print(f"  Slope is {sn:+.3f}, not ~1: the injected deformation is")
        print("  not reaching the likelihood at full strength.")
        print("  => Check the injection path, the sign convention, and")
        print("     whether the band mask matches between injection and")
        print("     recovery.")

    print()
    print(f"  profile prefers B_inj over 0 in {100*prefers:.0f}% of "
          f"non-zero injections")
    if prefers < 0.75:
        print("  >> The likelihood barely distinguishes the injected value")
        print("     from zero. At this chirp mass and band the f^(a-1)")
        print("     phase is largely absorbed by t_c and phi_c.")

    with open(OUT_JSON, "w") as fh:
        json.dump(dict(alpha=ALPHA, event_gps=EVENT_GPS,
                       template=list(TEMPLATE), alias_period=float(period),
                       physical_half_grid=float(half),
                       diagnostic_span=float(span),
                       null_sigma=float(sigma), null_edge_fraction=edge_frac,
                       slope_global=float(sg), intercept_global=float(ag),
                       slope_nearest=float(sn), intercept_nearest=float(an),
                       ladder=rows), fh, indent=2)
    print(f"\n  written: {OUT_JSON}")


if __name__ == "__main__":
    main()
