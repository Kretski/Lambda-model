#!/usr/bin/env python3
"""
mdr_search.py -- active sliding-window detector with generalised
                 modified-dispersion post-processing
==========================================================================

Search engine version of the Lambda pipeline. Scans continuous strain
rather than a known event time, and generalises the deformation from the
single alpha=4 case (Lambda) to the whole Mirshekari-Yunes-Will family,
which covers graviton mass and Lorentz-violating dispersion with the
same code.

ARCHITECTURE -- two stages, deliberately
----------------------------------------
Stage A searches with GR templates only. Stage B applies the dispersion
scan to surviving candidates.

This is NOT the same as putting the dispersion parameter into the search
bank. A bank extended over B_alpha multiplies the trials factor by the
grid size while adding almost nothing to detection efficiency for real
(GR-like) signals: the deformation is degenerate with chirp mass over
most of the band, so the extra templates mostly buy extra chances to fit
noise. LVK tests of GR are structured this way for that reason.

STAGES
------
  A1  condition H1 and L1, estimate PSD per analysis block
  A2  sliding-window matched filter over the mass/spin bank
  A3  power chi-squared veto -> reweighted SNR
  A4  H1/L1 coincidence: chirp mass within tolerance, |dt| < light travel
  A5  time-slide background -> FAR for each coincidence
  B1  dispersion profiling on surviving candidates, per alpha
  B2  off-source null per alpha -> interval, not label

THE ALPHA FAMILY
----------------
    dPhi_alpha(f) = -B_alpha * f**(alpha - 1)          (alpha != 1)

    alpha = 0    graviton mass          dPhi ~ f^-1
    alpha = 2    generic subluminal
    alpha = 2.5  LIV
    alpha = 3    LIV
    alpha = 3.5  LIV
    alpha = 4    your Lambda            dPhi ~ f^3

alpha = 1 is deliberately excluded: there the phase is constant in f and
is absorbed entirely by phi_c, so it is unmeasurable by matched filter
as a matter of principle, not of implementation.

B_alpha is the phase coefficient measured directly in the band. It is
unambiguous and needs no cosmology. Converting to a physical A_alpha
(Lambda in m^3/s, m_g in eV) needs the distance factor K(z); supply it
in KAPPA_BY_ALPHA where you have it. None means the result is reported
as B_alpha only, which is still a valid bound, just band-referenced.

ON THE BOUNDARY-HIT WARNING
---------------------------
The scan grid spans exactly one aliasing period, so its edges are at
+/- pi of accumulated phase. A maximum pinned at the edge means the
profile is still rising when the phase wraps: B is not localised, and
the value is not a measurement. In practice this happens when the GR
template is a poor match and the deformation is absorbing waveform
error (wrong mass ratio, neglected spin) rather than dispersion. The
fix is a denser bank, not a wider grid.

Requires: numpy, scipy, pycbc, lalsuite, h5py
Usage:    python3 mdr_search.py
"""

import json
import sys
import numpy as np

# ======================================================================
# CONFIG
# ======================================================================
LOCAL = {
    "H1": "H-H1_GWOSC_4KHZ_R1-1126257415-4096.hdf5",
    "L1": "L-L1_GWOSC_4KHZ_R1-1126257415-4096.hdf5",
}

F_LOW = 20.0
F_HIGH = 512.0          # keep merger-ringdown; alpha=4 lives at high f
SAMPLE_RATE = 4096
BLOCK = 64.0            # s, analysis block
OVERLAP = 8.0           # s, block overlap (must exceed template duration)
PSD_SEG = 8             # s, Welch segment

# --- bank ---------------------------------------------------------------
# A production search needs pycbc_geom_nonspinbank (or an aligned-spin
# equivalent) with a stated minimal match, typically 0.97. This grid is
# for pipeline validation. Do not describe results from it as a "search"
# sensitivity in a deposit.
# NOTE: these are DETECTOR-FRAME (redshifted) masses, which is what
# get_fd_waveform expects. A source at z has detector-frame masses
# (1+z) times its source-frame ones. GW150914 is (36, 29) at source
# but roughly (39, 32) as seen, i.e. total ~71 — so a grid that jumps
# 65 -> 90 cannot match it and is forced into a wrong mass ratio.
BANK_MTOTAL = np.array([12.0, 20.0, 30.0, 45.0, 55.0, 65.0, 72.0, 85.0, 100.0])
BANK_Q = np.array([1.0, 1.25, 1.5, 2.0, 3.0, 4.0])
BANK_SPIN = np.array([0.0])     # aligned spin chi_1z = chi_2z; try
                                # [-0.3, 0.0, 0.3] if boundary hits persist
APPROXIMANT = "IMRPhenomD"

SNR_THRESH = 4.5
NEWSNR_THRESH = 5.0
CHISQ_BINS = 16
CLUSTER_WIN = 1.0       # s, peak clustering window
COINC_WIN = 0.015       # s, H1-L1 light travel + timing error
MC_TOL = 0.25           # fractional tolerance in log chirp mass

N_SLIDES = 100          # time slides for background
SLIDE_STEP = 1.0        # s, must exceed COINC_WIN

# --- dispersion scan ---------------------------------------------------
ALPHAS = [0.0, 2.5, 3.0, 3.5, 4.0]
N_B = 61
N_OFFSOURCE = 40
OFFSOURCE_GAP = 40.0

# Calibration from B_alpha to a physical parameter. Fill in what you have.
KAPPA_BY_ALPHA = {
    4.0: 2.02e-7,       # Lambda [m^3/s], validated against your alias period
    0.0: None,          # needs K(z) to convert to m_g
    2.0: None,
    2.5: None,
    3.0: None,
    3.5: None,
}

OUT_JSON = "mdr_search_results.json"

N_CANDIDATES = 1

# --- smoke test --------------------------------------------------------
# True  -> ~5-10 min validation run on a short stretch, alpha=4 only.
# False -> full analysis (hours; run it under nohup).
SMOKE = False
SMOKE_CENTRE = 1126259462.423   # GW150914
SMOKE_DURATION = 600.0          # s of data kept around SMOKE_CENTRE


# ======================================================================
# IMPORTS (surfaced, not swallowed)
# ======================================================================
def _import_pycbc():
    import traceback

    try:
        from pycbc.waveform import get_fd_waveform
        from pycbc.filter import matched_filter, highpass, resample_to_delta_t
        from pycbc.psd import welch, interpolate, inverse_spectrum_truncation
        from pycbc.types import TimeSeries
        from pycbc.vetoes import power_chisq
    except Exception:
        traceback.print_exc()
        print("\n  pip install lalsuite  /  pip install -U pycbc scipy")
        sys.exit(1)
    return dict(
        get_fd_waveform=get_fd_waveform, matched_filter=matched_filter,
        highpass=highpass, resample_to_delta_t=resample_to_delta_t,
        welch=welch, interpolate=interpolate,
        inverse_spectrum_truncation=inverse_spectrum_truncation,
        TimeSeries=TimeSeries, power_chisq=power_chisq,
    )


_P = _import_pycbc()
get_fd_waveform = _P["get_fd_waveform"]
matched_filter = _P["matched_filter"]
highpass = _P["highpass"]
resample_to_delta_t = _P["resample_to_delta_t"]
welch = _P["welch"]
interpolate = _P["interpolate"]
inverse_spectrum_truncation = _P["inverse_spectrum_truncation"]
TimeSeries = _P["TimeSeries"]
power_chisq = _P["power_chisq"]


# ======================================================================
# helpers
# ======================================================================
def load_gwosc(path):
    import h5py

    with h5py.File(path, "r") as fh:
        strain = fh["strain/Strain"][:]
        t0 = float(fh["meta/GPSstart"][()])
        dur = float(fh["meta/Duration"][()])
    return TimeSeries(strain, delta_t=dur / len(strain), epoch=t0)


def condition(ts):
    ts = highpass(ts, 15.0)
    ts = resample_to_delta_t(ts, 1.0 / SAMPLE_RATE)
    return ts.crop(4, 4)


def block_psd(block):
    p = welch(block, seg_len=PSD_SEG * SAMPLE_RATE,
              seg_stride=PSD_SEG * SAMPLE_RATE // 2)
    p = interpolate(p, 1.0 / block.duration)
    return inverse_spectrum_truncation(p, int(4 * SAMPLE_RATE),
                                       low_frequency_cutoff=F_LOW)


def mchirp(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def make_bank():
    """Bank entries are (m1, m2, spin1z, spin2z)."""
    bank = []
    for mt in BANK_MTOTAL:
        for q in BANK_Q:
            m1 = mt * q / (1 + q)
            m2 = mt / (1 + q)
            for s in BANK_SPIN:
                bank.append((float(m1), float(m2), float(s), float(s)))
    return bank


def fd_template(entry, delta_f, length):
    m1, m2, s1, s2 = entry
    hp, _ = get_fd_waveform(approximant=APPROXIMANT, mass1=m1, mass2=m2,
                            spin1z=s1, spin2z=s2,
                            delta_f=delta_f, f_lower=F_LOW)
    hp.resize(length)
    return hp


def deform(hp, freqs, alpha, b):
    """dPhi = -b * f**(alpha-1). Zero-frequency bin guarded."""
    out = hp.copy()
    e = alpha - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ph = freqs ** e
    ph[~np.isfinite(ph)] = 0.0
    out.data[:] = hp.data * np.exp(-1j * b * ph)
    return out


def newsnr(snr, rchisq, q=6.0, n=2.0):
    s = np.atleast_1d(np.asarray(snr, dtype=float)).copy()
    r = np.atleast_1d(np.asarray(rchisq, dtype=float))
    m = r > 1.0
    s[m] = s[m] / ((1.0 + r[m] ** (q / n)) / 2.0) ** (1.0 / q)
    return s


def cluster(times, values, win):
    """Greedy peak clustering: keep local maxima separated by > win."""
    order = np.argsort(values)[::-1]
    keep = []
    for i in order:
        if all(abs(times[i] - times[j]) > win for j in keep):
            keep.append(i)
    return sorted(keep, key=lambda k: times[k])


# ======================================================================
# STAGE A -- sliding-window search
# ======================================================================
def search_detector(ts, bank, label):
    print(f"  [{label}] blocks of {BLOCK:.0f}s, overlap {OVERLAP:.0f}s, "
          f"{len(bank)} templates")
    trig_t, trig_snr, trig_nsnr, trig_tmpl = [], [], [], []

    t0, t1 = float(ts.start_time), float(ts.end_time)
    starts = np.arange(t0, t1 - BLOCK, BLOCK - OVERLAP)
    for bi, bs in enumerate(starts):
        block = ts.time_slice(bs, bs + BLOCK)
        psd = block_psd(block)
        flen = len(psd)

        for ti, entry in enumerate(bank):
            hp = fd_template(entry, 1.0 / BLOCK, flen)
            snr = matched_filter(hp, block, psd=psd,
                                 low_frequency_cutoff=F_LOW,
                                 high_frequency_cutoff=F_HIGH)
            snr = snr.crop(OVERLAP / 2, OVERLAP / 2)
            a = np.abs(snr.numpy())
            hits = np.where(a > SNR_THRESH)[0]
            if hits.size == 0:
                continue

            cs = power_chisq(hp, block, CHISQ_BINS, psd,
                             low_frequency_cutoff=F_LOW,
                             high_frequency_cutoff=F_HIGH)
            cs = cs.crop(OVERLAP / 2, OVERLAP / 2)
            dof = CHISQ_BINS * 2 - 2
            rchisq = cs.numpy() / dof

            ns = newsnr(a[hits], rchisq[hits])
            good = ns > NEWSNR_THRESH
            if not good.any():
                continue
            tt = snr.sample_times.numpy()[hits][good]
            trig_t.extend(tt.tolist())
            trig_snr.extend(a[hits][good].tolist())
            trig_nsnr.extend(ns[good].tolist())
            trig_tmpl.extend([ti] * int(good.sum()))

        if (bi + 1) % 5 == 0:
            print(f"    block {bi+1}/{len(starts)}  triggers so far: {len(trig_t)}")

    if not trig_t:
        return np.array([]), np.array([]), np.array([]), np.array([], dtype=int)

    trig_t = np.array(trig_t)
    trig_nsnr = np.array(trig_nsnr)
    keep = cluster(trig_t, trig_nsnr, CLUSTER_WIN)
    print(f"  [{label}] {len(trig_t)} raw -> {len(keep)} clustered triggers")
    return (trig_t[keep], np.array(trig_snr)[keep], trig_nsnr[keep],
            np.array(trig_tmpl)[keep])


def coincide(t1, n1, k1, t2, n2, k2, bank, shift=0.0):
    """
    Coincidences: chirp masses within MC_TOL and |dt| < COINC_WIN.

    Matching on chirp mass rather than on template index matters: with a
    coarse bank the two detectors routinely pick neighbouring templates
    for the same signal, and an identical-index requirement discards
    real coincidences (it discarded GW150914 in an earlier version).
    """
    if len(t1) == 0 or len(t2) == 0:
        return []
    mc = np.array([mchirp(b[0], b[1]) for b in bank])
    out = []
    for i in range(len(t1)):
        dt = np.abs(t2 + shift - t1[i])
        close = np.abs(np.log(mc[k2] / mc[k1[i]])) < MC_TOL
        m = (dt < COINC_WIN) & close
        if m.any():
            j = int(np.argmax(np.where(m, n2, -np.inf)))
            out.append((t1[i], float(np.sqrt(n1[i] ** 2 + n2[j] ** 2)), int(k1[i])))
    return out


# ======================================================================
# STAGE B -- dispersion profiling on candidates
# ======================================================================
def b_grid(alpha):
    """One aliasing period in the phase coefficient, centred on zero."""
    e = alpha - 1.0
    span = abs(F_HIGH ** e - F_LOW ** e)
    period = 2 * np.pi / max(span, 1e-30)
    return np.linspace(-period / 2, period / 2, N_B), period


def profile_b(block, psd, hp, freqs, alpha, grid, centre, win=0.05):
    best = np.full(len(grid), -1.0)
    for i, b in enumerate(grid):
        tm = deform(hp, freqs, alpha, b)
        snr = matched_filter(tm, block, psd=psd,
                             low_frequency_cutoff=F_LOW,
                             high_frequency_cutoff=F_HIGH)
        snr = snr.crop(OVERLAP / 2, OVERLAP / 2)
        t = snr.sample_times.numpy()
        m = np.abs(t - centre) < win
        if m.any():
            best[i] = np.abs(snr.numpy())[m].max()
    j = int(np.argmax(best))
    j0 = int(np.argmin(np.abs(grid)))
    return grid[j], best[j], best[j0]


# ======================================================================
# MAIN
# ======================================================================
def main():
    global N_SLIDES, ALPHAS, N_OFFSOURCE, N_B, N_CANDIDATES
    if SMOKE:
        N_SLIDES, ALPHAS, N_OFFSOURCE, N_B, N_CANDIDATES = 20, [4.0], 8, 21, 1
        print("*** SMOKE MODE: short data stretch, alpha=4 only. ***")
        print("*** Set SMOKE = False for the full analysis.       ***\n")

    bank = make_bank()
    print("=" * 70)
    print("STAGE A1 -- CONDITIONING")
    print("=" * 70)
    data = {}
    for det, path in LOCAL.items():
        ts = condition(load_gwosc(path))
        if SMOKE:
            lo = max(float(ts.start_time), SMOKE_CENTRE - SMOKE_DURATION / 2)
            hi = min(float(ts.end_time), SMOKE_CENTRE + SMOKE_DURATION / 2)
            ts = ts.time_slice(lo, hi)
        data[det] = ts
        print(f"  {det}: {data[det].duration:.0f}s @ {SAMPLE_RATE} Hz")
    print(f"  bank: {len(bank)} templates "
          f"({len(BANK_MTOTAL)} x {len(BANK_Q)} x {len(BANK_SPIN)})")
    print()

    print("=" * 70)
    print("STAGE A2/A3 -- SLIDING-WINDOW SEARCH + CHI-SQUARED VETO")
    print("=" * 70)
    res = {d: search_detector(data[d], bank, d) for d in data}
    print()

    print("=" * 70)
    print("STAGE A4/A5 -- COINCIDENCE + TIME-SLIDE BACKGROUND")
    print("=" * 70)
    tH, sH, nH, kH = res["H1"]
    tL, sL, nL, kL = res["L1"]
    for lbl, tt, nn, kk in (("H1", tH, nH, kH), ("L1", tL, nL, kL)):
        for a, b, c in zip(tt, nn, kk):
            e = bank[c]
            print(f"  {lbl} trigger  t={a:.3f}  newsnr={b:.2f}  "
                  f"m=({e[0]:.1f},{e[1]:.1f}) s={e[2]:+.2f}  "
                  f"mc={mchirp(e[0], e[1]):.1f}")

    zerolag = coincide(tH, nH, kH, tL, nL, kL, bank, 0.0)
    print(f"  zero-lag coincidences: {len(zerolag)}")

    bg = []
    for s in range(1, N_SLIDES + 1):
        bg += [c[1] for c in coincide(tH, nH, kH, tL, nL, kL, bank,
                                      s * SLIDE_STEP)]
    bg = np.array(bg)
    livetime = float(data["H1"].duration)
    bg_time = livetime * N_SLIDES
    print(f"  background coincidences: {len(bg)} over {bg_time/86400:.2f} days")
    if len(bg) == 0:
        print("  NOTE: empty background. FAR below is a floor, not a")
        print("        significance. Needs hours of data, not minutes.")
    print()

    candidates = []
    for t, stat, k in sorted(zerolag, key=lambda c: -c[1]):
        louder = int((bg >= stat).sum()) if len(bg) else 0
        far = louder / bg_time * 86400.0 * 365.25
        candidates.append(dict(gps=t, stat=stat, tmpl=k, far_per_year=far))
        e = bank[k]
        print(f"  t={t:.3f}  stat={stat:.2f}  m=({e[0]:.1f},{e[1]:.1f})  "
              f"FAR={far:.3e}/yr")
    print()

    print("=" * 70)
    print("STAGE B -- DISPERSION PROFILING (alpha family)")
    print("=" * 70)
    rng = np.random.default_rng(7)
    report = []

    for cand in candidates[:N_CANDIDATES]:
        t = cand["gps"]
        entry = bank[cand["tmpl"]]
        block = data["H1"].time_slice(t - BLOCK / 2, t + BLOCK / 2)
        psd = block_psd(block)
        hp = fd_template(entry, 1.0 / BLOCK, len(psd))
        freqs = psd.sample_frequencies.numpy()

        print(f"\n  candidate t={t:.3f}  template "
              f"({entry[0]:.1f}, {entry[1]:.1f}, s={entry[2]:+.2f})")
        rec = dict(gps=t, mass1=entry[0], mass2=entry[1], spin=entry[2],
                   far_per_year=cand["far_per_year"], alphas={})

        # off-source centres and their PSDs, reused across alphas
        lo = float(data["H1"].start_time) + BLOCK
        hi = float(data["H1"].end_time) - BLOCK
        centres = []
        while len(centres) < N_OFFSOURCE:
            c = rng.uniform(lo, hi)
            if abs(c - t) > OFFSOURCE_GAP:
                centres.append(c)
        off = []
        for c in centres:
            blk = data["H1"].time_slice(c - BLOCK / 2, c + BLOCK / 2)
            p = block_psd(blk)
            off.append((c, blk, p, fd_template(entry, 1.0 / BLOCK, len(p)),
                        p.sample_frequencies.numpy()))

        for alpha in ALPHAS:
            grid, period = b_grid(alpha)
            b_hat, rho_max, rho_0 = profile_b(block, psd, hp, freqs, alpha,
                                              grid, t)
            null = np.array([
                profile_b(blk, p, h, fr, alpha, grid, c)[0]
                for c, blk, p, h, fr in off
            ])
            sd = null.std(ddof=1)
            ci = 1.645 * sd
            z = (b_hat - null.mean()) / sd if sd > 0 else 0.0
            edge = abs(b_hat) >= grid[-1] * 0.98

            kap = KAPPA_BY_ALPHA.get(alpha)
            phys = None if not kap else dict(value=b_hat / kap, ci90=ci / kap)

            rec["alphas"][str(alpha)] = dict(
                B_hat=float(b_hat), ci90=float(ci), z=float(z),
                alias_period=float(period), snr_gr=float(rho_0),
                snr_best=float(rho_max), boundary_hit=bool(edge),
                physical=phys,
            )
            tag = "" if not kap else f"  ->  {b_hat/kap:+.3e} +/- {ci/kap:.3e}"
            print(f"    alpha={alpha:<4}  B = {b_hat:+.4e} +/- {ci:.4e}  "
                  f"z={z:+.2f}{tag}")
            print(f"                  SNR {rho_0:.3f} -> {rho_max:.3f}")
            if edge:
                ts_gain = rho_max**2 - rho_0**2
                if ts_gain < 4.0:
                    print(f"                  FLAT PROFILE (dSNR^2 = {ts_gain:.2f}):")
                    print("                  B is unconstrained within one alias")
                    print("                  period. The edge is where the noise")
                    print("                  points, not a preference. Null result.")
                else:
                    print(f"                  BOUNDARY HIT (dSNR^2 = {ts_gain:.2f}):")
                    print("                  real pull to the edge, not a")
                    print("                  measurement. Densify the bank first.")

        report.append(rec)

    with open(OUT_JSON, "w") as fh:
        json.dump(dict(candidates=candidates, dispersion=report), fh, indent=2)
    print(f"\n  written: {OUT_JSON}")
    print("\n  Report every alpha as an interval. A z below ~3 with a")
    print("  boundary-free maximum means: consistent with GR, and the")
    print("  bound is the 90% CI. It does not mean GR is confirmed.")


if __name__ == "__main__":
    main()
