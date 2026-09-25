#!/usr/bin/env python3
"""
benchmark_grid_speedup.py
==========================

Honest benchmark of "coarse-to-fine two-stage Lambda grid search"
vs. the existing full-grid brute-force search in stage6E3H-R_v2.py.

WHY THIS SCRIPT EXISTS
-----------------------
A prior message in this thread claimed W-Twin's entropy metric gives
"early rejection" of >95% of MCMC candidates for milliseconds of cost.
That claim has a hole: to compute a deviation-from-signal metric for
a given Lambda, you need *something* cheaper than the full matched-
filter integral, or there's no saving. This script tests the one
concrete, standard, honestly-cheaper proxy available here: evaluating
FEWER grid points first (coarse pass), then refining only around the
surviving candidate(s) (fine pass) -- NOT a magic entropy shortcut
that skips waveform generation entirely.

TWO MODES
---------
--mode synthetic  (default, runs anywhere, no dependencies beyond
                   numpy/scipy -- what this sandbox can actually run)
    Self-contained smoke test:
      - builds the SAME recovery grid as stage6E3H-R_v2.py
        (LAMBDA_MIN/MAX/STEP imported directly from that file)
      - uses the REAL lambda_phase_correction() from waveform.py
      - injects a known Lambda into a synthetic frequency-domain
        signal + colored noise, evaluates a matched-filter-style
        overlap statistic per template (real FFT-sized work, same
        N_freq as the actual 16s/4096Hz analysis band)
      - compares full-grid vs. two-stage: same recovered Lambda?
        how many template evaluations? how much wall-clock time?
    HONEST LIMITS OF THIS MODE: no real LIGO noise, no pycbc
    time-domain matched_filter (no time-of-arrival maximization),
    no real H1 data. This checks whether the coarse-to-fine LOGIC
    is correct and gives a lower-bound, directionally-correct
    speedup from call-count reduction alone. It is NOT the number
    to put in a README.

--mode real  (requires pycbc installed + a GWOSC H1 HDF5 file)
    Imports generate_gr_template, make_psd, apply_lambda_phase,
    extract_segment, calculate_band_snr, scale_to_snr straight from
    stage6E3H-R_v2.py (same file the project already uses) and runs
    two versions of the recovery loop -- full-grid (their existing
    recover_lambda) and two-stage -- on the same injected signal,
    timed with time.time(). THIS is the number that belongs in a
    README, because it uses the real pycbc matched_filter and real
    H1 strain.

USAGE
-----
    python3 benchmark_grid_speedup.py --mode synthetic
    python3 benchmark_grid_speedup.py --mode real --data H-H1_LOSC_4_V2-1126259446-32.hdf5
"""

import argparse
import time
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from waveform import lambda_phase_correction, cosmological_K_factor

# Pull the EXACT grid config the real project uses, so the synthetic
# test isn't quietly using an easier grid than production.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "stage6E3H_R_v2",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage6E3H-R_v2.py"),
)
_stage6 = importlib.util.module_from_spec(_spec)
# stage6E3H-R_v2.py imports pycbc/h5py at module level, so a bare
# exec_module() will fail without those installed. We only need the
# plain-Python constants, so read them out with a tiny regex fallback
# if the real import fails (e.g. in this sandbox, no pycbc/h5py).
try:
    _spec.loader.exec_module(_stage6)
    LAMBDA_MIN = _stage6.LAMBDA_MIN
    LAMBDA_MAX = _stage6.LAMBDA_MAX
    LAMBDA_STEP = _stage6.LAMBDA_STEP
    F_LOW = _stage6.F_LOW
    F_HIGH = _stage6.F_HIGH
    SAMPLE_RATE = _stage6.SAMPLE_RATE
    DURATION = _stage6.DURATION
    REDSHIFT = _stage6.REDSHIFT
    STAGE6_IMPORTED = True
except Exception as e:
    print(f"[note] could not import stage6E3H-R_v2.py directly ({e.__class__.__name__}: {e})")
    print("[note] falling back to reading its constants as text (no pycbc/h5py needed)")
    import re
    src = open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage6E3H-R_v2.py"),
        encoding="utf-8",
    ).read()

    def _grab(name, cast=float):
        m = re.search(rf"^{name}\s*=\s*([-\d.]+)", src, re.MULTILINE)
        return cast(m.group(1))

    LAMBDA_MIN = _grab("LAMBDA_MIN")
    LAMBDA_MAX = _grab("LAMBDA_MAX")
    LAMBDA_STEP = _grab("LAMBDA_STEP")
    F_LOW = _grab("F_LOW")
    F_HIGH = _grab("F_HIGH")
    SAMPLE_RATE = _grab("SAMPLE_RATE")
    DURATION = _grab("DURATION")
    REDSHIFT = _grab("REDSHIFT")
    STAGE6_IMPORTED = False

K_Z = cosmological_K_factor(REDSHIFT)


def build_recovery_grid(lambda_min, lambda_max, lambda_step):
    return np.arange(lambda_min, lambda_max + 0.5 * lambda_step, lambda_step)


# Default grid = exactly what stage6E3H-R_v2.py already uses today.
# Overridden in __main__ if --lambda-min/--lambda-max/--lambda-step are passed.
RECOVERY_GRID = build_recovery_grid(LAMBDA_MIN, LAMBDA_MAX, LAMBDA_STEP)


# ============================================================
# TWO-STAGE SEARCH LOGIC (mode-independent)
# ============================================================

def two_stage_indices(grid, coarse_stride=5, refine_halfwidth=2):
    """
    Returns:
      coarse_idx : indices of grid evaluated in stage 1
      refine_fn  : given the best coarse index, returns stage-2 indices
    coarse_stride=5 on a 101-point grid -> 21 coarse points (~5x fewer).
    refine_halfwidth=2 -> fine pass covers +/- 2 coarse-steps around
    the best coarse candidate, i.e. up to 2*coarse_stride*2+1 points,
    but only if the true optimum could plausibly hide between coarse
    samples (checked empirically below, not assumed).
    """
    coarse_idx = np.arange(0, len(grid), coarse_stride)

    def refine_fn(best_coarse_pos):
        center = coarse_idx[best_coarse_pos]
        lo = max(0, center - coarse_stride * refine_halfwidth)
        hi = min(len(grid), center + coarse_stride * refine_halfwidth + 1)
        return np.arange(lo, hi)

    return coarse_idx, refine_fn


# ============================================================
# SYNTHETIC MODE
# ============================================================

def synthetic_overlap_score(freqs, S_n, d_f, lam):
    """
    |<d|h(lambda)>| overlap statistic, frequency domain, using the
    REAL lambda_phase_correction from waveform.py. Deliberately does
    the full-sized FFT-domain integral every call -- no shortcut --
    so per-call cost is representative of "one grid point = one full
    integral", matching the concern raised about W-Twin's early
    claim.
    """
    phase = lambda_phase_correction(freqs, lam, K_Z)
    h_f = np.exp(1j * phase)  # unit-amplitude template phase model
    integrand = (d_f * np.conj(h_f)) / S_n
    overlap = 4.0 * np.sum(integrand.real) * (freqs[1] - freqs[0])
    return abs(overlap)


def run_synthetic(n_trials, seed):
    rng = np.random.default_rng(seed)

    n_samples = int(round(DURATION * SAMPLE_RATE))
    freqs_full = np.fft.rfftfreq(n_samples, d=1.0 / SAMPLE_RATE)
    band = (freqs_full >= F_LOW) & (freqs_full <= F_HIGH)
    freqs = freqs_full[band]
    n_band = len(freqs)

    # simple rising-with-frequency colored-noise PSD stand-in
    # (NOT aLIGO design curve -- just something non-flat so the
    # weighting in the integral isn't trivial)
    S_n = 1e-46 * (freqs / 100.0) ** (-4.0) + 1e-47

    print(f"Recovery grid: {len(RECOVERY_GRID)} points, "
          f"[{RECOVERY_GRID[0]:.3f}, {RECOVERY_GRID[-1]:.3f}] "
          f"step={RECOVERY_GRID[1]-RECOVERY_GRID[0]:.3f}")
    print(f"Band: {F_LOW}-{F_HIGH} Hz, {n_band} frequency bins per template")
    print(f"stage6E3H-R_v2.py constants imported directly: {STAGE6_IMPORTED}")
    print()

    coarse_idx, refine_fn = two_stage_indices(RECOVERY_GRID, coarse_stride=5,
                                               refine_halfwidth=2)

    full_calls_total = 0
    two_stage_calls_total = 0
    full_time_total = 0.0
    two_stage_time_total = 0.0
    matches = 0
    mismatches = []

    for trial in range(n_trials):
        true_lambda = rng.uniform(LAMBDA_MIN * 0.5, LAMBDA_MAX * 0.5)

        # injected "data": true signal + colored noise, in frequency domain
        signal_phase = lambda_phase_correction(freqs, true_lambda, K_Z)
        signal_f = 5.0 * np.exp(1j * signal_phase)  # arbitrary amplitude
        noise_re = rng.normal(0, 1, n_band) * np.sqrt(S_n / 2)
        noise_im = rng.normal(0, 1, n_band) * np.sqrt(S_n / 2)
        d_f = signal_f + noise_re + 1j * noise_im

        # ---------------- FULL GRID (baseline) ----------------
        t0 = time.time()
        full_scores = np.array([
            synthetic_overlap_score(freqs, S_n, d_f, lam) for lam in RECOVERY_GRID
        ])
        t1 = time.time()
        full_best_idx = int(np.argmax(full_scores))
        full_best_lambda = RECOVERY_GRID[full_best_idx]
        full_time = t1 - t0
        full_calls = len(RECOVERY_GRID)

        # ---------------- TWO-STAGE ----------------
        t0 = time.time()
        coarse_scores = np.array([
            synthetic_overlap_score(freqs, S_n, d_f, RECOVERY_GRID[i])
            for i in coarse_idx
        ])
        best_coarse_pos = int(np.argmax(coarse_scores))
        refine_idx = refine_fn(best_coarse_pos)
        fine_scores = np.array([
            synthetic_overlap_score(freqs, S_n, d_f, RECOVERY_GRID[i])
            for i in refine_idx
        ])
        t1 = time.time()

        all_idx = np.concatenate([coarse_idx, refine_idx])
        all_scores = np.concatenate([coarse_scores, fine_scores])
        two_stage_best_idx = int(all_idx[np.argmax(all_scores)])
        two_stage_best_lambda = RECOVERY_GRID[two_stage_best_idx]
        two_stage_time = t1 - t0
        two_stage_calls = len(coarse_idx) + len(refine_idx)

        full_calls_total += full_calls
        two_stage_calls_total += two_stage_calls
        full_time_total += full_time
        two_stage_time_total += two_stage_time

        match = (two_stage_best_idx == full_best_idx)
        if match:
            matches += 1
        else:
            mismatches.append((trial, true_lambda, full_best_lambda, two_stage_best_lambda))

        print(f"trial {trial+1:2d}/{n_trials}  true={true_lambda:+.3f}  "
              f"full_best={full_best_lambda:+.3f} ({full_calls} calls, {full_time*1e3:.2f} ms)  "
              f"two_stage_best={two_stage_best_lambda:+.3f} ({two_stage_calls} calls, {two_stage_time*1e3:.2f} ms)  "
              f"{'MATCH' if match else 'MISMATCH'}")

    print()
    print("=" * 70)
    print("SUMMARY (synthetic mode -- smoke test only, see header)")
    print("=" * 70)
    print(f"Trials                         = {n_trials}")
    print(f"Full-grid calls/trial          = {len(RECOVERY_GRID)}")
    print(f"Two-stage calls/trial (mean)   = {two_stage_calls_total / n_trials:.1f}")
    print(f"Call-count reduction           = {(1 - two_stage_calls_total / full_calls_total) * 100:.1f}%")
    print(f"Full-grid total time           = {full_time_total*1e3:.2f} ms")
    print(f"Two-stage total time           = {two_stage_time_total*1e3:.2f} ms")
    print(f"Wall-clock speedup             = {full_time_total / two_stage_time_total:.2f}x")
    print(f"Two-stage matched full-grid    = {matches}/{n_trials} "
          f"({100*matches/n_trials:.1f}%)")
    if mismatches:
        print()
        print("MISMATCHES (two-stage missed the true full-grid peak):")
        for trial, true_l, full_l, ts_l in mismatches:
            print(f"  trial {trial}: true={true_l:+.3f} full_best={full_l:+.3f} "
                  f"two_stage_best={ts_l:+.3f}")
        print()
        print("^ If this list is non-empty, coarse_stride=5 is too coarse for this "
              "SNR/noise level -- the speedup is not free, there is a real "
              "accuracy/speed tradeoff to report, not just a speed number.")


# ============================================================
# REAL MODE (pycbc + GWOSC data required)
# ============================================================

def run_real(data_path, n_trials, target_snr, seed):
    if not STAGE6_IMPORTED:
        raise RuntimeError(
            "stage6E3H-R_v2.py could not be imported (pycbc/h5py missing?). "
            "--mode real requires the exact environment the project already runs in."
        )

    np.random.seed(seed)
    s6 = _stage6

    full_data, detector = s6.read_losc_hdf5(data_path)
    full_data = full_data.astype(np.float64)
    gr_template = s6.generate_gr_template()

    gps_start = float(full_data.start_time)
    duration = len(full_data) * float(full_data.delta_t)
    event_offset = s6.EVENT_GPS - gps_start
    gps_times = s6.build_offsource_times(gps_start, duration, event_offset, n_trials)

    recovery_grid = RECOVERY_GRID
    print(f"[real mode] building {len(recovery_grid)} full-resolution templates once "
          f"(shared across trials, same as production code)...")
    templates_full = [(float(lam), s6.apply_lambda_phase(gr_template, float(lam)))
                       for lam in recovery_grid]

    coarse_idx, refine_fn = two_stage_indices(recovery_grid, coarse_stride=5, refine_halfwidth=2)
    templates_coarse = [templates_full[i] for i in coarse_idx]

    def recover_two_stage(data, psd):
        # stage 1
        best_coarse_pos, best_coarse_score = None, -np.inf
        coarse_results = []
        for pos, (lam, template) in enumerate(templates_coarse):
            snr_series = s6.matched_filter(template, data, psd=psd,
                                            low_frequency_cutoff=F_LOW,
                                            high_frequency_cutoff=F_HIGH)
            score = float(np.max(np.abs(np.asarray(snr_series, dtype=np.complex128))))
            coarse_results.append((lam, score))
            if score > best_coarse_score:
                best_coarse_score, best_coarse_pos = score, pos

        # stage 2
        refine_idx = refine_fn(best_coarse_pos)
        best_lambda, best_score = templates_coarse[best_coarse_pos][0], best_coarse_score
        n_refine_calls = 0
        for i in refine_idx:
            lam, template = templates_full[i]
            snr_series = s6.matched_filter(template, data, psd=psd,
                                            low_frequency_cutoff=F_LOW,
                                            high_frequency_cutoff=F_HIGH)
            score = float(np.max(np.abs(np.asarray(snr_series, dtype=np.complex128))))
            n_refine_calls += 1
            if score > best_score:
                best_score, best_lambda = score, lam

        n_calls = len(templates_coarse) + n_refine_calls
        return best_lambda, best_score, n_calls

    full_time_total, two_stage_time_total = 0.0, 0.0
    matches = 0
    for i, gps in enumerate(gps_times):
        segment = s6.extract_segment(full_data, gps)
        segment = s6.highpass(segment, frequency=F_LOW * 0.9)
        from scipy.signal.windows import tukey
        win = tukey(len(segment), alpha=0.1)
        from pycbc.types import TimeSeries
        segment = TimeSeries(np.asarray(segment) * win, delta_t=segment.delta_t,
                              epoch=segment.start_time)
        psd = s6.make_psd(segment)

        injected_lambda = float(np.random.choice(s6.INJECTED_LAMBDAS))
        injected_template = s6.apply_lambda_phase(gr_template, injected_lambda)
        scaled_signal, _, _ = s6.scale_to_snr(injected_template, psd, target_snr)
        injected_data = injected_template.__class__(
            np.asarray(segment) + np.asarray(scaled_signal),
            delta_t=segment.delta_t, epoch=segment.start_time)

        t0 = time.time()
        full_lambda, full_score = s6.recover_lambda(injected_data, templates_full, psd)
        t1 = time.time()
        full_time = t1 - t0

        t0 = time.time()
        ts_lambda, ts_score, ts_calls = recover_two_stage(injected_data, psd)
        t1 = time.time()
        ts_time = t1 - t0

        full_time_total += full_time
        two_stage_time_total += ts_time
        match = abs(full_lambda - ts_lambda) < 1e-9
        matches += int(match)

        print(f"trial {i+1}/{len(gps_times)} inj={injected_lambda:+.3f} "
              f"full_best={full_lambda:+.3f} ({len(templates_full)} calls, {full_time:.3f}s)  "
              f"two_stage_best={ts_lambda:+.3f} ({ts_calls} calls, {ts_time:.3f}s)  "
              f"{'MATCH' if match else 'MISMATCH'}")

    print()
    print("=" * 70)
    print("SUMMARY (real mode -- pycbc + GWOSC H1 data, this IS README-grade)")
    print("=" * 70)
    print(f"Full-grid total time  = {full_time_total:.3f} s")
    print(f"Two-stage total time  = {two_stage_time_total:.3f} s")
    print(f"Wall-clock speedup    = {full_time_total / two_stage_time_total:.2f}x")
    print(f"Two-stage matched full-grid = {matches}/{len(gps_times)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--data", default=None, help="GWOSC HDF5 path (--mode real)")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--target-snr", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-min", type=float, default=None,
                         help="Override recovery grid lower bound "
                              "(default: LAMBDA_MIN from stage6E3H-R_v2.py, currently %.3f)" % LAMBDA_MIN)
    parser.add_argument("--lambda-max", type=float, default=None,
                         help="Override recovery grid upper bound "
                              "(default: LAMBDA_MAX from stage6E3H-R_v2.py, currently %.3f)" % LAMBDA_MAX)
    parser.add_argument("--lambda-step", type=float, default=None,
                         help="Override recovery grid step "
                              "(default: LAMBDA_STEP from stage6E3H-R_v2.py, currently %.3f)" % LAMBDA_STEP)
    args = parser.parse_args()

    if args.lambda_min is not None or args.lambda_max is not None or args.lambda_step is not None:
        new_min = args.lambda_min if args.lambda_min is not None else LAMBDA_MIN
        new_max = args.lambda_max if args.lambda_max is not None else LAMBDA_MAX
        new_step = args.lambda_step if args.lambda_step is not None else LAMBDA_STEP
        RECOVERY_GRID = build_recovery_grid(new_min, new_max, new_step)
        print(f"[override] recovery grid = [{new_min}, {new_max}] step={new_step} "
              f"({len(RECOVERY_GRID)} points) -- default was "
              f"[{LAMBDA_MIN}, {LAMBDA_MAX}] step={LAMBDA_STEP} ({len(build_recovery_grid(LAMBDA_MIN, LAMBDA_MAX, LAMBDA_STEP))} points)")

    if args.mode == "synthetic":
        run_synthetic(args.n_trials, args.seed)
    else:
        if not args.data:
            parser.error("--mode real requires --data <path_to_gwosc_hdf5>")
        run_real(args.data, args.n_trials, args.target_snr, args.seed)
