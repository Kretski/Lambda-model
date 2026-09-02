#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 6E3G — REAL-NOISE INJECTION RECOVERY / INDEPENDENT CALIBRATION

Purpose
-------
Test whether the Stage 6E3F null-bias correction can recover known
Lambda injections placed into independent real off-source H1 data.

Injection values:
    Lambda = -4, -2, 0, +2, +4

The calibration is performed on real off-source data only.

IMPORTANT
---------
This is an estimator-validation experiment.

It does NOT establish a physical Lambda.
The injected Lambda values are artificial calibration signals.

The injection waveform uses the same phenomenological Lambda*f^2
phase deformation as the preceding Stage 6E3 family.

Outputs
-------
stage6E3G_injection_results/
    stage6E3G_recovery.csv
    stage6E3G_summary.csv
    stage6E3G_null_bias.csv
"""

import os
import csv
import argparse
import numpy as np
import h5py

from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries
from pycbc.filter import matched_filter
from pycbc.psd import welch


# ============================================================
# CONFIGURATION
# ============================================================

EVENT_GPS = 1126259462.0

MASS1 = 36.0
MASS2 = 29.0

DISTANCE_MPC = 440.0

FS = 4096.0
DURATION = 16.0

F_LOW = 20.0
F_HIGH = 300.0

LAMBDA_MIN = -20.0
LAMBDA_MAX = 20.0
LAMBDA_STEP = 0.10

INJECTION_LAMBDAS = [
    -4.0,
    -2.0,
     0.0,
     2.0,
     4.0,
]

N_OFFSOURCE = 100
OFFSOURCE_SPACING = 32.0
EVENT_GUARD = 16.0

OUTDIR = "stage6E3G_injection_results"


# ============================================================
# OUTPUT
# ============================================================

def ensure_output_dir():
    os.makedirs(OUTDIR, exist_ok=True)


# ============================================================
# HDF5
# ============================================================

def read_hdf5_strain(path):

    with h5py.File(path, "r") as f:

        if "strain/Strain" not in f:
            raise KeyError(
                "Expected dataset 'strain/Strain' not found."
            )

        data = np.asarray(
            f["strain/Strain"][:],
            dtype=np.float64
        )

        gps_start = None

        if "meta/GPSstart" in f:
            gps_start = float(
                np.asarray(f["meta/GPSstart"])
            )

        if "meta/Duration" in f:
            duration = float(
                np.asarray(f["meta/Duration"])
            )
        else:
            duration = len(data) / FS

        detector = "H1"

        if "meta/Detector" in f:
            try:
                detector = str(
                    np.asarray(
                        f["meta/Detector"]
                    ).astype(str)
                )
            except Exception:
                detector = "H1"

    return data, gps_start, duration, detector


def clean_strain(data):

    x = np.asarray(
        data,
        dtype=np.float64
    ).copy()

    x[~np.isfinite(x)] = 0.0

    mean = np.mean(x)

    x -= mean

    return x, mean


# ============================================================
# SEGMENT EXTRACTION
# ============================================================

def extract_segment(
    raw,
    gps_start,
    center_gps,
    fs=FS,
    duration=DURATION,
):

    if gps_start is None:
        raise RuntimeError(
            "GPS start is unavailable."
        )

    n = int(
        round(duration * fs)
    )

    center_index = int(
        round(
            (center_gps - gps_start) * fs
        )
    )

    start = center_index - n // 2
    end = start + n

    if start < 0 or end > len(raw):
        return None

    segment = raw[start:end]

    if len(segment) != n:
        return None

    return segment.copy()


# ============================================================
# OFF-SOURCE TIMES
# ============================================================

def build_offsource_times(
    gps_start,
    duration,
    event_gps,
    n_requested,
    spacing,
    guard,
):

    half = DURATION / 2.0

    usable = []

    k = 1

    while len(usable) < n_requested:

        t = event_gps - k * spacing

        if (
            t - half >= gps_start
            and
            t + half <= gps_start + duration
            and
            abs(t - event_gps) > guard + half
        ):
            usable.append(t)

        k += 1

        if k > 10000:
            break

    k = 1

    while len(usable) < n_requested:

        t = event_gps + k * spacing

        if (
            t - half >= gps_start
            and
            t + half <= gps_start + duration
            and
            abs(t - event_gps) > guard + half
        ):
            usable.append(t)

        k += 1

        if k > 10000:
            break

    return usable[:n_requested]


# ============================================================
# TEMPLATE
# ============================================================

def generate_gr_template():

    hp, hc = get_td_waveform(
        approximant="SEOBNRv4_opt",
        mass1=MASS1,
        mass2=MASS2,
        delta_t=1.0 / FS,
        f_lower=F_LOW,
        distance=DISTANCE_MPC,
    )

    hp = hp.copy()

    target_n = int(
        round(DURATION * FS)
    )

    if len(hp) > target_n:

        hp = hp[-target_n:]

    elif len(hp) < target_n:

        padded = np.zeros(target_n)

        padded[-len(hp):] = np.asarray(hp)

        hp = TimeSeries(
            padded,
            delta_t=1.0 / FS
        )

    else:

        hp = TimeSeries(
            np.asarray(hp),
            delta_t=1.0 / FS
        )

    arr = np.asarray(hp)

    maxabs = np.max(
        np.abs(arr)
    )

    if maxabs > 0:

        hp /= maxabs

    return hp


# ============================================================
# LAMBDA PHASE MODEL
# ============================================================

def lambda_phase_factor(
    frequencies,
    lam,
):

    f = np.asarray(
        frequencies,
        dtype=np.float64
    )

    phase = lam * (
        f / 100.0
    ) ** 2

    return np.exp(
        1j * phase
    )


def apply_lambda_phase(
    template,
    lam,
):

    n = len(template)

    dt = 1.0 / FS

    freqs = np.fft.rfftfreq(
        n,
        dt
    )

    hf = np.fft.rfft(
        np.asarray(template)
    )

    deformation = lambda_phase_factor(
        freqs,
        lam
    )

    hf_modified = (
        hf * deformation
    )

    modified = np.fft.irfft(
        hf_modified,
        n=n
    )

    return TimeSeries(
        modified,
        delta_t=dt
    )


# ============================================================
# PSD
# ============================================================

def estimate_psd(segment):

    ts = TimeSeries(
        np.asarray(
            segment,
            dtype=np.float64
        ),
        delta_t=1.0 / FS
    )

    seg_len = int(
        4.0 * FS
    )

    psd = welch(
        ts,
        seg_len=seg_len,
        seg_stride=seg_len // 2,
    )

    return psd


def prepare_psd_for_filter(
    psd,
    data_length,
    fs=FS,
):

    expected_delta_f = fs / data_length

    current_delta_f = float(
        psd.delta_f
    )

    if np.isclose(
        current_delta_f,
        expected_delta_f,
        rtol=1e-10,
        atol=1e-12,
    ):
        return psd

    freqs_old = np.arange(
        len(psd)
    ) * current_delta_f

    freqs_new = np.arange(
        data_length // 2 + 1
    ) * expected_delta_f

    values_old = np.asarray(
        psd,
        dtype=np.float64
    )

    values_new = np.interp(
        freqs_new,
        freqs_old,
        values_old,
        left=values_old[0],
        right=values_old[-1],
    )

    return type(psd)(
        values_new,
        delta_f=expected_delta_f
    )


# ============================================================
# MATCHED FILTER
# ============================================================

def matched_score(
    strain_segment,
    template,
    psd,
):

    strain = TimeSeries(
        np.asarray(
            strain_segment,
            dtype=np.float64
        ),
        delta_t=1.0 / FS,
    )

    try:

        mf = matched_filter(
            template,
            strain,
            psd=psd,
            low_frequency_cutoff=F_LOW,
            high_frequency_cutoff=F_HIGH,
        )

        arr = np.asarray(mf)

        if len(arr) == 0:
            return 0.0

        peak = float(
            np.max(
                np.abs(arr)
            )
        )

        return peak

    except Exception as exc:

        raise RuntimeError(
            "matched_filter failed: "
            + repr(exc)
        )


# ============================================================
# PROFILE
# ============================================================

def profile_scan(
    strain_segment,
    templates,
    lambdas,
    psd,
):

    scores = []

    for template in templates:

        score = matched_score(
            strain_segment,
            template,
            psd
        )

        scores.append(score)

    scores = np.asarray(
        scores,
        dtype=np.float64
    )

    zero_index = int(
        np.argmin(
            np.abs(lambdas)
        )
    )

    score0 = float(
        scores[zero_index]
    )

    delta_scores = (
        scores ** 2
        -
        score0 ** 2
    )

    best_index = int(
        np.argmax(delta_scores)
    )

    return {
        "scores": scores,
        "delta_scores": delta_scores,
        "score0": score0,
        "best_index": best_index,
        "best_lambda":
            float(lambdas[best_index]),
        "best_delta":
            float(delta_scores[best_index]),
    }


# ============================================================
# QUADRATIC REFINEMENT
# ============================================================

def local_quadratic_refinement(
    lambdas,
    values,
    index,
):

    if index <= 0:
        return float(lambdas[index])

    if index >= len(lambdas) - 1:
        return float(lambdas[index])

    x1 = float(
        lambdas[index - 1]
    )

    x2 = float(
        lambdas[index]
    )

    x3 = float(
        lambdas[index + 1]
    )

    y1 = float(
        values[index - 1]
    )

    y2 = float(
        values[index]
    )

    y3 = float(
        values[index + 1]
    )

    denom = (
        (x1 - x2)
        *
        (x1 - x3)
    )

    denom2 = (
        (x2 - x1)
        *
        (x2 - x3)
    )

    denom3 = (
        (x3 - x1)
        *
        (x3 - x2)
    )

    if (
        abs(denom) < 1e-15
        or
        abs(denom2) < 1e-15
        or
        abs(denom3) < 1e-15
    ):
        return x2

    A = (
        y1 / denom
        +
        y2 / denom2
        +
        y3 / denom3
    )

    B = -(
        y1 * (x2 + x3) / denom
        +
        y2 * (x1 + x3) / denom2
        +
        y3 * (x1 + x2) / denom3
    )

    if A >= 0:
        return x2

    vertex = -B / (
        2.0 * A
    )

    if (
        vertex < x1
        or
        vertex > x3
    ):
        return x2

    return float(vertex)


# ============================================================
# NULL BIAS
# ============================================================

def estimate_null_bias(
    offsource_profiles,
):

    matrix = np.asarray(
        offsource_profiles,
        dtype=np.float64
    )

    if matrix.ndim != 2:
        raise ValueError(
            "Invalid off-source profile matrix."
        )

    bias_profile = np.median(
        matrix,
        axis=0
    )

    return bias_profile


def correct_profile(
    raw_profile,
    null_bias,
):

    return (
        np.asarray(raw_profile)
        -
        np.asarray(null_bias)
    )


# ============================================================
# CSV
# ============================================================

def save_recovery(records):

    path = os.path.join(
        OUTDIR,
        "stage6E3G_recovery.csv"
    )

    with open(
        path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "offsource_index",
            "gps",
            "lambda_injected",
            "raw_best_lambda",
            "raw_refined_lambda",
            "corrected_best_lambda",
            "corrected_refined_lambda",
            "raw_deltaS",
            "corrected_deltaS",
            "lambda_error",
        ])

        for r in records:

            writer.writerow([
                r["index"],
                f'{r["gps"]:.3f}',
                f'{r["lambda_injected"]:.6f}',
                f'{r["raw_best_lambda"]:.10f}',
                f'{r["raw_refined_lambda"]:.10f}',
                f'{r["corrected_best_lambda"]:.10f}',
                f'{r["corrected_refined_lambda"]:.10f}',
                f'{r["raw_deltaS"]:.12e}',
                f'{r["corrected_deltaS"]:.12e}',
                f'{r["lambda_error"]:.10f}',
            ])

    return path


def save_bias(
    lambdas,
    bias,
):

    path = os.path.join(
        OUTDIR,
        "stage6E3G_null_bias.csv"
    )

    with open(
        path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "lambda",
            "median_null_deltaS"
        ])

        for lam, value in zip(
            lambdas,
            bias
        ):

            writer.writerow([
                f"{lam:.10f}",
                f"{value:.12e}"
            ])

    return path


def save_summary(rows):

    path = os.path.join(
        OUTDIR,
        "stage6E3G_summary.csv"
    )

    with open(
        path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerows(rows)

    return path


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Stage 6E3G real-noise "
            "Lambda injection recovery"
        )
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to LOSC HDF5 file"
    )

    parser.add_argument(
        "--n-offsource",
        type=int,
        default=N_OFFSOURCE
    )

    parser.add_argument(
        "--spacing",
        type=float,
        default=OFFSOURCE_SPACING
    )

    parser.add_argument(
        "--lambda-min",
        type=float,
        default=LAMBDA_MIN
    )

    parser.add_argument(
        "--lambda-max",
        type=float,
        default=LAMBDA_MAX
    )

    parser.add_argument(
        "--lambda-step",
        type=float,
        default=LAMBDA_STEP
    )

    args = parser.parse_args()

    # IMPORTANT:
    # Do NOT use "global" here.
    # Keep command-line configuration local.

    lambda_min = float(
        args.lambda_min
    )

    lambda_max = float(
        args.lambda_max
    )

    lambda_step = float(
        args.lambda_step
    )

    ensure_output_dir()

    print("=" * 80)
    print(
        "STAGE 6E3G — REAL-NOISE "
        "INJECTION RECOVERY"
    )
    print("=" * 80)

    print()
    print(
        f"Data file          = {args.data}"
    )

    print(
        f"Event GPS          = {EVENT_GPS}"
    )

    print(
        f"Lambda scan        = "
        f"[{lambda_min}, {lambda_max}]"
    )

    print(
        f"Lambda step        = "
        f"{lambda_step}"
    )

    print(
        f"Off-source N       = "
        f"{args.n_offsource}"
    )

    print(
        f"Injection Lambdas  = "
        f"{INJECTION_LAMBDAS}"
    )

    print()
    print(
        "Lambda_true        = NONE"
    )

    print(
        "Purpose            = estimator calibration"
    )

    print(
        "Data               = REAL H1 off-source"
    )

    print()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print("[1] LOADING REAL H1 DATA")

    raw, gps_start, duration, detector = \
        read_hdf5_strain(args.data)

    print(
        f"GPS start          = "
        f"{gps_start}"
    )

    print(
        f"Duration           = "
        f"{duration:.3f} s"
    )

    print(
        f"Samples            = "
        f"{len(raw)}"
    )

    print(
        f"Detector           = "
        f"{detector}"
    )

    print(
        f"Event offset       = "
        f"{EVENT_GPS - gps_start:.3f} s"
    )

    raw, mean_removed = clean_strain(
        raw
    )

    print()
    print("[2] STRAIN CLEANING")

    print(
        f"Mean removed       = "
        f"{mean_removed:.6e}"
    )

    # --------------------------------------------------------
    # TEMPLATE
    # --------------------------------------------------------

    print()
    print("[3] GENERATING GR TEMPLATE")

    template_gr = generate_gr_template()

    print(
        f"Template samples   = "
        f"{len(template_gr)}"
    )

    # --------------------------------------------------------
    # GRID
    # --------------------------------------------------------

    print()
    print("[4] BUILDING LAMBDA GRID")

    lambdas = np.arange(
        lambda_min,
        lambda_max
        + 0.5 * lambda_step,
        lambda_step
    )

    print(
        f"Grid points        = "
        f"{len(lambdas)}"
    )

    print(
        f"Range              = "
        f"[{lambdas[0]:.3f}, "
        f"{lambdas[-1]:.3f}]"
    )

    # --------------------------------------------------------
    # TEMPLATE FAMILY
    # --------------------------------------------------------

    print()
    print(
        "[5] BUILDING LAMBDA TEMPLATE FAMILY"
    )

    templates = []

    for lam in lambdas:

        templates.append(
            apply_lambda_phase(
                template_gr,
                float(lam)
            )
        )

    print(
        f"Templates          = "
        f"{len(templates)}"
    )

    # --------------------------------------------------------
    # OFF SOURCE
    # --------------------------------------------------------

    print()
    print(
        "[6] BUILDING OFF-SOURCE SEGMENTS"
    )

    off_times = build_offsource_times(
        gps_start,
        duration,
        EVENT_GPS,
        args.n_offsource,
        args.spacing,
        EVENT_GUARD
    )

    print(
        f"Usable segments    = "
        f"{len(off_times)}"
    )

    if len(off_times) == 0:

        raise RuntimeError(
            "No off-source segments available."
        )

    # --------------------------------------------------------
    # NULL PROFILES
    # --------------------------------------------------------

    print()
    print(
        "[7] COMPUTING REAL-DATA "
        "OFF-SOURCE NULL PROFILES"
    )

    null_profiles = []

    for i, gps in enumerate(
        off_times,
        start=1
    ):

        segment = extract_segment(
            raw,
            gps_start,
            gps
        )

        if segment is None:
            continue

        psd = estimate_psd(
            segment
        )

        psd = prepare_psd_for_filter(
            psd,
            len(segment)
        )

        profile = profile_scan(
            segment,
            templates,
            lambdas,
            psd
        )

        null_profiles.append(
            profile["delta_scores"]
        )

        print(
            f"{i:3d} "
            f"GPS={gps:.1f} "
            f"Lambda="
            f"{profile['best_lambda']:+.3f} "
            f"DeltaS="
            f"{profile['best_delta']:.8e}"
        )

    null_profiles = np.asarray(
        null_profiles,
        dtype=np.float64
    )

    print()
    print(
        f"Null profiles      = "
        f"{len(null_profiles)}"
    )

    if len(null_profiles) < 10:

        raise RuntimeError(
            "Too few null profiles "
            "for calibration."
        )

    # --------------------------------------------------------
    # BIAS
    # --------------------------------------------------------

    print()
    print(
        "[8] ESTIMATING NULL BIAS"
    )

    print(
        f"Null calibration uses "
        f"{len(null_profiles)} "
        f"off-source profiles."
    )

    null_bias = estimate_null_bias(
        null_profiles
    )

    bias_path = save_bias(
        lambdas,
        null_bias
    )

    print(
        f"Saved null bias    = "
        f"{bias_path}"
    )

    # --------------------------------------------------------
    # INJECTION RECOVERY
    # --------------------------------------------------------

    print()
    print(
        "[9] INJECTION RECOVERY"
    )

    records = []

    # Use every off-source segment.
    # Each injection is added independently.
    #
    # Injection amplitude is fixed to the same normalized
    # template amplitude for every real-noise segment.
    #
    # This is intentionally not optimized per segment.

    for i, gps in enumerate(
        off_times,
        start=1
    ):

        segment = extract_segment(
            raw,
            gps_start,
            gps
        )

        if segment is None:
            continue

        psd = estimate_psd(
            segment
        )

        psd = prepare_psd_for_filter(
            psd,
            len(segment)
        )

        # Keep one independent realization of the GR template.
        gr_arr = np.asarray(
            template_gr,
            dtype=np.float64
        )

        for lambda_injected in INJECTION_LAMBDAS:

            injected_template = apply_lambda_phase(
                template_gr,
                lambda_injected
            )

            # Fixed injection amplitude.
            #
            # The template was normalized to max(abs)=1.
            # Use a small but deterministic amplitude.
            #
            # This amplitude is intentionally fixed.
            injection_amplitude = 1.0e-21

            injected_signal = (
                injection_amplitude
                *
                np.asarray(
                    injected_template,
                    dtype=np.float64
                )
            )

            injected_segment = (
                np.asarray(segment)
                +
                injected_signal
            )

            raw_profile = profile_scan(
                injected_segment,
                templates,
                lambdas,
                psd
            )

            raw_best_lambda = (
                raw_profile["best_lambda"]
            )

            raw_refined = \
                local_quadratic_refinement(
                    lambdas,
                    raw_profile["delta_scores"],
                    raw_profile["best_index"]
                )

            corrected_delta = correct_profile(
                raw_profile["delta_scores"],
                null_bias
            )

            corrected_index = int(
                np.argmax(
                    corrected_delta
                )
            )

            corrected_best_lambda = float(
                lambdas[corrected_index]
            )

            corrected_refined = \
                local_quadratic_refinement(
                    lambdas,
                    corrected_delta,
                    corrected_index
                )

            corrected_deltaS = float(
                corrected_delta[
                    corrected_index
                ]
            )

            error = (
                corrected_refined
                -
                lambda_injected
            )

            record = {
                "index": i,
                "gps": gps,
                "lambda_injected":
                    lambda_injected,
                "raw_best_lambda":
                    raw_best_lambda,
                "raw_refined_lambda":
                    raw_refined,
                "corrected_best_lambda":
                    corrected_best_lambda,
                "corrected_refined_lambda":
                    corrected_refined,
                "raw_deltaS":
                    raw_profile["best_delta"],
                "corrected_deltaS":
                    corrected_deltaS,
                "lambda_error":
                    error,
            }

            records.append(
                record
            )

        print(
            f"{i:3d} GPS={gps:.1f} "
            f"completed"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    recovery_path = save_recovery(
        records
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print(
        "[10] CALIBRATION SUMMARY"
    )

    summary_rows = [
        [
            "parameter",
            "value"
        ],
        [
            "event",
            "GW150914"
        ],
        [
            "detector",
            "H1"
        ],
        [
            "offsource_N",
            len(null_profiles)
        ],
        [
            "injection_count",
            len(INJECTION_LAMBDAS)
        ],
        [
            "total_recovery_trials",
            len(records)
        ],
        [
            "injection_amplitude",
            1.0e-21
        ],
    ]

    for inj in INJECTION_LAMBDAS:

        subset = [
            r for r in records
            if np.isclose(
                r["lambda_injected"],
                inj
            )
        ]

        recovered = np.asarray(
            [
                r["corrected_refined_lambda"]
                for r in subset
            ],
            dtype=np.float64
        )

        errors = (
            recovered - inj
        )

        raw_recovered = np.asarray(
            [
                r["raw_refined_lambda"]
                for r in subset
            ],
            dtype=np.float64
        )

        if len(recovered) > 0:

            median_rec = float(
                np.median(recovered)
            )

            mean_rec = float(
                np.mean(recovered)
            )

            bias = float(
                np.mean(errors)
            )

            rmse = float(
                np.sqrt(
                    np.mean(
                        errors ** 2
                    )
                )
            )

            std = float(
                np.std(
                    recovered,
                    ddof=1
                )
            ) if len(recovered) > 1 else 0.0

            q16, q84 = np.percentile(
                recovered,
                [16, 84]
            )

            q025, q975 = np.percentile(
                recovered,
                [2.5, 97.5]
            )

            within_1 = float(
                np.mean(
                    np.abs(errors)
                    <= 1.0
                )
            )

            within_2 = float(
                np.mean(
                    np.abs(errors)
                    <= 2.0
                )
            )

            raw_median = float(
                np.median(raw_recovered)
            )

        else:

            median_rec = np.nan
            mean_rec = np.nan
            bias = np.nan
            rmse = np.nan
            std = np.nan
            q16 = np.nan
            q84 = np.nan
            q025 = np.nan
            q975 = np.nan
            within_1 = np.nan
            within_2 = np.nan
            raw_median = np.nan

        print()
        print(
            f"Lambda injected = "
            f"{inj:+.1f}"
        )

        print(
            f"Raw median      = "
            f"{raw_median:+.6f}"
        )

        print(
            f"Corrected median= "
            f"{median_rec:+.6f}"
        )

        print(
            f"Corrected mean  = "
            f"{mean_rec:+.6f}"
        )

        print(
            f"Bias            = "
            f"{bias:+.6f}"
        )

        print(
            f"RMSE            = "
            f"{rmse:.6f}"
        )

        print(
            f"Std             = "
            f"{std:.6f}"
        )

        print(
            f"68% interval    = "
            f"[{q16:+.6f}, "
            f"{q84:+.6f}]"
        )

        print(
            f"95% interval    = "
            f"[{q025:+.6f}, "
            f"{q975:+.6f}]"
        )

        print(
            f"|error| <= 1    = "
            f"{within_1:.6f}"
        )

        print(
            f"|error| <= 2    = "
            f"{within_2:.6f}"
        )

        summary_rows.extend([
            [
                f"injection_{inj:+.1f}_N",
                len(recovered)
            ],
            [
                f"injection_{inj:+.1f}_raw_median",
                raw_median
            ],
            [
                f"injection_{inj:+.1f}_corrected_median",
                median_rec
            ],
            [
                f"injection_{inj:+.1f}_corrected_mean",
                mean_rec
            ],
            [
                f"injection_{inj:+.1f}_bias",
                bias
            ],
            [
                f"injection_{inj:+.1f}_rmse",
                rmse
            ],
            [
                f"injection_{inj:+.1f}_std",
                std
            ],
            [
                f"injection_{inj:+.1f}_q16",
                q16
            ],
            [
                f"injection_{inj:+.1f}_q84",
                q84
            ],
            [
                f"injection_{inj:+.1f}_q025",
                q025
            ],
            [
                f"injection_{inj:+.1f}_q975",
                q975
            ],
            [
                f"injection_{inj:+.1f}_within_1",
                within_1
            ],
            [
                f"injection_{inj:+.1f}_within_2",
                within_2
            ],
        ])

    summary_rows.extend([
        [
            "physical_lambda_claim",
            "NO"
        ],
        [
            "purpose",
            "estimator_validation_only"
        ],
        [
            "calibration_data",
            "real_offsource_H1"
        ],
        [
            "injection_values",
            str(INJECTION_LAMBDAS)
        ],
    ])

    summary_path = save_summary(
        summary_rows
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "STAGE 6E3G INTERPRETATION"
    )
    print("=" * 80)

    print()
    print(
        "This experiment tests recovery of known "
        "phenomenological Lambda injections "
        "embedded in real off-source H1 data."
    )

    print(
        "The null bias was estimated from "
        "real off-source profiles."
    )

    print(
        "The injected Lambda values were not "
        "used to construct the null bias."
    )

    print()
    print(
        "A useful calibration should show:"
    )

    print(
        "  1. Lambda=0 remains approximately centered."
    )

    print(
        "  2. Negative injections recover negative Lambda."
    )

    print(
        "  3. Positive injections recover positive Lambda."
    )

    print(
        "  4. The recovered median tracks "
        "the injected value."
    )

    print(
        "  5. Bias and RMSE remain acceptably small."
    )

    print()
    print(
        "Failure to recover the injections means "
        "the Stage 6E3F correction should not be "
        "used as a physical Lambda estimator."
    )

    print()
    print(
        "No physical nonzero Lambda is established "
        "by Stage 6E3G."
    )

    print()
    print(
        f"Recovery CSV       = "
        f"{recovery_path}"
    )

    print(
        f"Null bias CSV      = "
        f"{bias_path}"
    )

    print(
        f"Summary CSV        = "
        f"{summary_path}"
    )

    print()
    print("=" * 80)
    print(
        "STAGE 6E3G COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
