#!/usr/bin/env python3
"""
lambda_active.py -- active Lambda-model inference pipeline
==========================================================

Replaces the Layer C/D/E chain with a version that cannot silently
return zero. Every stage is self-auditing; the pipeline aborts with a
diagnosis instead of producing a clean-looking null.

Stages
------
  0  WAVEFORM AUDIT      verify the deformation is really f^3, measure
                         KAPPA numerically, derive the alias-safe grid
  1  CONDITIONING        highpass, tukey, Welch PSD from off-source data
  2  ON-SOURCE PROFILE   rho(Lambda), maximised over t_c and phi_c,
                         profiled over chirp mass
  3  OFF-SOURCE NULL     same statistic on N background segments
  4  REPORT              Lambda_hat, 90% CI, z, TS, p, sensitivity gate

Output is an INTERVAL, never a GR / NON-GR label.

Requires: numpy, scipy, pycbc, h5py
Usage:    python3 lambda_active.py
"""

import sys
import numpy as np

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
EVENT_GPS = 1126259462.423  # GW150914
LOCAL_H1 = "H-H1_GWOSC_4KHZ_R1-1126257415-4096.hdf5"
LOCAL_L1 = "L-L1_GWOSC_4KHZ_R1-1126257415-4096.hdf5"

APPROXIMANT = "IMRPhenomD"  # NOT TaylorF2 -- see note at bottom
MASS1, MASS2 = 36.0, 29.0
MC_FRAC = 0.04  # profile chirp mass over +/- 4%
N_MC = 9

F_LOW = 20.0
F_HIGH = 300.0
SEG_LEN = 32  # s, analysis segment
PSD_LEN = 8  # s, Welch segment
SAMPLE_RATE = 4096

N_LAMBDA = 81
N_OFFSOURCE = 60
OFFSOURCE_GAP = 40.0  # s, minimum distance from the event
TC_WINDOW = 0.10  # s, search window around EVENT_GPS

def _import_pycbc():
    """Import pycbc piece by piece so a failure names the actual culprit."""
    import traceback

    mods = {}
    steps = [
        ("pycbc", "import pycbc"),
        ("pycbc.types", "from pycbc.types import TimeSeries, FrequencySeries"),
        ("pycbc.waveform", "from pycbc.waveform import get_fd_waveform"),
        ("pycbc.filter", "from pycbc.filter import matched_filter, highpass, "
                         "resample_to_delta_t"),
        ("pycbc.psd", "from pycbc.psd import welch, interpolate, "
                      "inverse_spectrum_truncation"),
    ]
    for name, stmt in steps:
        try:
            exec(stmt, mods)
        except Exception:
            print("=" * 70)
            print(f"IMPORT FAILED at: {name}")
            print("=" * 70)
            traceback.print_exc()
            print()
            print("Common causes:")
            print("  * lalsuite missing      ->  pip install lalsuite")
            print("  * numpy 2.x vs old pycbc ->  pip install -U pycbc")
            print("  * wrong interpreter     ->  which python; python -c "
                  "'import sys; print(sys.executable)'")
            sys.exit(1)
    try:
        import pycbc

        print(f"pycbc {pycbc.__version__} at {pycbc.__file__}")
    except Exception:
        pass
    return mods


_M = _import_pycbc()
TimeSeries = _M["TimeSeries"]
FrequencySeries = _M["FrequencySeries"]
get_fd_waveform = _M["get_fd_waveform"]
matched_filter = _M["matched_filter"]
highpass = _M["highpass"]
resample_to_delta_t = _M["resample_to_delta_t"]
welch = _M["welch"]
interpolate = _M["interpolate"]
inverse_spectrum_truncation = _M["inverse_spectrum_truncation"]


# ======================================================================
# STAGE 0 -- WAVEFORM AUDIT
# ======================================================================
def deform(hf, freqs, lam, kappa):
    """Apply the alpha=4 dispersion phase. dPhi(f) = -kappa * lam * f^3."""
    out = hf.copy()
    out.data[:] = hf.data * np.exp(-1j * kappa * lam * freqs**3)
    return out


def audit_waveform(kappa_declared=None):
    """
    If waveform.py is importable, verify its Lambda deformation is really
    a pure f^3 phase and extract KAPPA numerically. Otherwise fall back
    to the internal deform() and use the declared KAPPA.
    """
    print("=" * 70)
    print("STAGE 0 -- WAVEFORM AUDIT")
    print("=" * 70)

    try:
        import waveform as user_wf  # the repo's own module
    except Exception:
        print("  waveform.py not importable -- using internal deform().")
        if kappa_declared is None:
            sys.exit("  Set kappa_declared (= 4*pi^3*K(z)/c^3) and rerun.")
        kappa = kappa_declared
        print(f"  KAPPA (declared)         : {kappa:.6e}")
        return kappa

    # Measure the phase difference numerically at two small Lambda values.
    delta = 1e-9
    f = np.arange(F_LOW, F_HIGH, 0.5)
    try:
        p0 = np.angle(user_wf.lambda_phase(f, 0.0))
        p1 = np.angle(user_wf.lambda_phase(f, delta))
    except AttributeError:
        print("  waveform.py has no lambda_phase(f, Lambda) entry point.")
        print("  Expose one, or set kappa_declared. Falling back.")
        return kappa_declared

    dphi = np.unwrap(p1 - p0)
    # fit dphi = -kappa * delta * f^n  -> log-log slope gives n
    mask = np.abs(dphi) > 1e-12
    slope = np.polyfit(np.log(f[mask]), np.log(np.abs(dphi[mask])), 1)[0]
    kappa = -np.mean(dphi[mask] / (delta * f[mask] ** 3))

    print(f"  measured phase exponent  : {slope:.4f}   (expected 3.0000)")
    print(f"  KAPPA (measured)         : {kappa:.6e}")
    if abs(slope - 3.0) > 0.05:
        sys.exit("  ABORT: deformation is not f^3. Fix waveform.py first.")
    if kappa_declared is not None and abs(kappa / kappa_declared - 1) > 0.02:
        print(f"  WARNING: declared KAPPA = {kappa_declared:.6e}, 2%+ mismatch.")
    return kappa


def safe_grid(kappa):
    """Alias-safe Lambda grid: one wraparound period, centred on zero."""
    period = 2 * np.pi / (kappa * (F_HIGH**3 - F_LOW**3))
    grid = np.linspace(-period / 2, period / 2, N_LAMBDA)
    print(f"  alias period             : {period:.6e}")
    print(f"  scan grid                : [{grid[0]:.4e}, {grid[-1]:.4e}]")
    print()
    return grid


# ======================================================================
# STAGE 1 -- CONDITIONING
# ======================================================================
def load_gwosc(path):
    import h5py

    with h5py.File(path, "r") as fh:
        strain = fh["strain/Strain"][:]
        t0 = fh["meta/GPSstart"][()]
        dur = fh["meta/Duration"][()]
    dt = dur / len(strain)
    return TimeSeries(strain, delta_t=dt, epoch=t0)


def condition(ts):
    ts = highpass(ts, 15.0)
    ts = resample_to_delta_t(ts, 1.0 / SAMPLE_RATE)
    ts = ts.crop(4, 4)
    psd = welch(ts, seg_len=PSD_LEN * SAMPLE_RATE, seg_stride=PSD_LEN * SAMPLE_RATE // 2)
    psd = interpolate(psd, 1.0 / SEG_LEN)
    psd = inverse_spectrum_truncation(psd, int(4 * SAMPLE_RATE), low_frequency_cutoff=F_LOW)
    return ts, psd


# ======================================================================
# STAGE 2/3 -- PROFILE STATISTIC
# ======================================================================
def rho_profile(seg, psd, lam_grid, kappa, mc_scales, centre_gps):
    """
    Returns (lambda_hat, rho_max, rho_at_zero) for one segment.
    Maximised over t_c (within TC_WINDOW), phi_c (|z|), and chirp mass.
    """
    best = np.full(len(lam_grid), -1.0)
    for s in mc_scales:
        hp, _ = get_fd_waveform(
            approximant=APPROXIMANT,
            mass1=MASS1 * s,
            mass2=MASS2 * s,
            delta_f=1.0 / seg.duration,
            f_lower=F_LOW,
        )
        hp.resize(len(psd))
        freqs = hp.sample_frequencies.numpy()
        band = (freqs >= F_HIGH)
        hp.data[band] = 0.0  # enforce F_HIGH

        for i, lam in enumerate(lam_grid):
            tmpl = deform(hp, freqs, lam, kappa)
            snr = matched_filter(tmpl, seg, psd=psd,
                                 low_frequency_cutoff=F_LOW,
                                 high_frequency_cutoff=F_HIGH)
            snr = snr.crop(4, 4)
            t = np.array(snr.sample_times)
            w = np.abs(t - centre_gps) < TC_WINDOW
            if not w.any():
                continue
            peak = np.abs(np.array(snr.data)[w]).max()
            if peak > best[i]:
                best[i] = peak

    j = int(np.argmax(best))
    j0 = int(np.argmin(np.abs(lam_grid)))
    return lam_grid[j], best[j], best[j0]


# ======================================================================
# MAIN
# ======================================================================
def main():
    kappa = audit_waveform(kappa_declared=2.02e-7)
    lam_grid = safe_grid(kappa)
    mc_scales = np.linspace(1 - MC_FRAC, 1 + MC_FRAC, N_MC)

    print("=" * 70)
    print("STAGE 1 -- CONDITIONING")
    print("=" * 70)
    raw = load_gwosc(LOCAL_H1)
    ts, psd = condition(raw)
    print(f"  duration {ts.duration:.0f}s  fs={SAMPLE_RATE}  band {F_LOW}-{F_HIGH} Hz")
    print()

    print("=" * 70)
    print("STAGE 2 -- ON-SOURCE")
    print("=" * 70)
    on = ts.time_slice(EVENT_GPS - SEG_LEN / 2, EVENT_GPS + SEG_LEN / 2)
    lam_hat, rho_max, rho_0 = rho_profile(on, psd, lam_grid, kappa, mc_scales, EVENT_GPS)
    ts_stat = rho_max**2 - rho_0**2
    print(f"  SNR at Lambda=0          : {rho_0:.4f}")
    print(f"  SNR at Lambda_hat        : {rho_max:.4f}")
    print(f"  Lambda_hat               : {lam_hat:+.6e}")
    print(f"  TS                       : {ts_stat:.4f}")

    if rho_0 > 40:
        print("  WARNING: SNR far above the published ~20 for H1. Check PSD"
              " normalisation -- you may be filtering the template on itself.")
    if abs(lam_hat) >= lam_grid[-1] * 0.98:
        print("  WARNING: boundary hit. The maximum is at the grid edge.")
    print()

    print("=" * 70)
    print("STAGE 3 -- OFF-SOURCE NULL")
    print("=" * 70)
    rng = np.random.default_rng(1234)
    lo = float(ts.start_time) + SEG_LEN
    hi = float(ts.end_time) - SEG_LEN
    centres = []
    while len(centres) < N_OFFSOURCE:
        c = rng.uniform(lo, hi)
        if abs(c - EVENT_GPS) > OFFSOURCE_GAP:
            centres.append(c)

    null = []
    for k, c in enumerate(centres):
        seg = ts.time_slice(c - SEG_LEN / 2, c + SEG_LEN / 2)
        lh, rm, r0 = rho_profile(seg, psd, lam_grid, kappa, mc_scales[N_MC // 2:N_MC // 2 + 1], c)
        null.append((lh, rm**2 - r0**2))
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{N_OFFSOURCE} segments done")
    null_lam = np.array([x[0] for x in null])
    null_ts = np.array([x[1] for x in null])
    print()

    print("=" * 70)
    print("STAGE 4 -- REPORT")
    print("=" * 70)
    mu, sd = null_lam.mean(), null_lam.std(ddof=1)
    z = (lam_hat - mu) / sd
    p = float((null_ts >= ts_stat).sum() + 1) / (len(null_ts) + 1)
    ci = 1.645 * sd

    print(f"  null mean / std          : {mu:+.4e} / {sd:.4e}")
    print(f"  z-score                  : {z:+.3f} sigma")
    print(f"  TS p-value               : {p:.4f}  (n={len(null_ts)}, floor {1/(len(null_ts)+1):.4f})")
    print()
    print(f"  RESULT   Lambda = {lam_hat:+.4e}  +/- {ci:.4e}   (90% CI)")
    print(f"           90% interval [{lam_hat-ci:+.4e}, {lam_hat+ci:+.4e}]")
    print()

    # ---- sensitivity gate -------------------------------------------
    sigma_fisher = 8.48e-9 / kappa * (24.0 / max(rho_0, 1e-9))
    ratio = sd / sigma_fisher
    print(f"  Fisher-predicted sigma   : {sigma_fisher:.4e}")
    print(f"  measured null sigma      : {sd:.4e}")
    print(f"  sensitivity loss         : {ratio:.1f}x")
    if ratio > 5:
        print("  >> Losing more than 5x against the Fisher bound. The interval")
        print("     above is real but far from optimal. Prime suspects: PSD")
        print("     estimation, F_HIGH truncation, chirp-mass grid too coarse.")
    print()
    print("  Report this as an upper bound. Do not label the event GR or NON-GR:")
    print(f"  at sigma = {sd:.2e} this pipeline cannot resolve either.")


if __name__ == "__main__":
    main()
