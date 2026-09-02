#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 6E3F — NULL-BIAS-CORRECTED PROFILE TEST

Purpose
-------
Estimate the systematic Lambda preference of the real off-source H1
null distribution and remove that bias before evaluating GW150914.

IMPORTANT
---------
1. GW150914 is NEVER used to estimate the null bias.
2. The null calibration is constructed exclusively from off-source data.
3. Lambda remains phenomenological.
4. This is an exploratory statistical test, not a fundamental-physics
   significance calculation.

Method
------
For every Lambda grid point:

    B(lambda) = median_offsource[DeltaS(lambda)]

    sigma(lambda) = robust null scale

Then for GW150914:

    corrected(lambda)
        = DeltaS_on(lambda) - B(lambda)

    Z(lambda)
        = corrected(lambda) / sigma(lambda)

The script also calculates:
    - corrected best Lambda
    - corrected profile RMS
    - maximum |Z|
    - empirical global null p-values
    - bootstrap null envelope
    - sign/boundary diagnostics

Outputs
-------
stage6E3F_profile_results/
    stage6E3F_on_source_corrected_profile.csv
    stage6E3F_off_source_profiles.csv
    stage6E3F_null_bias.csv
    stage6E3F_null_envelope.csv
    stage6E3F_summary.csv
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

N_OFFSOURCE = 100
OFFSOURCE_SPACING = 32.0
EVENT_GUARD = 16.0

OUTDIR = "stage6E3F_profile_results"


# ============================================================
# OUTPUT
# ============================================================

def ensure_output_dir():
    os.makedirs(OUTDIR, exist_ok=True)


# ============================================================
# DATA
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

    mean = float(np.mean(x))

    x -= mean

    return x, mean


def extract_segment(
    raw,
    gps_start,
    center_gps,
    fs=FS,
    duration=DURATION,
):

    if gps_start is None:
        raise RuntimeError(
            "GPS start unavailable."
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


def prepare_psd_for_filter(psd, template):

    expected_df = 1.0 / (
        len(template) * (1.0 / FS)
    )

    current_df = float(
        psd.delta_f
    )

    print(
        f"PSD delta_f       = "
        f"{current_df:.8f} Hz"
    )

    print(
        f"Expected delta_f  = "
        f"{expected_df:.8f} Hz"
    )

    if abs(
        current_df - expected_df
    ) < 1e-10:

        return psd

    # --------------------------------------------------------
    # PyCBC-compatible manual interpolation.
    # FrequencySeries does not always expose .interpolate().
    # --------------------------------------------------------

    old_freqs = (
        np.arange(len(psd))
        * current_df
    )

    new_freqs = (
        np.arange(
            len(template) // 2 + 1
        )
        * expected_df
    )

    old_values = np.asarray(
        psd,
        dtype=np.float64
    )

    new_values = np.interp(
        new_freqs,
        old_freqs,
        old_values,
        left=old_values[0],
        right=old_values[-1],
    )

    from pycbc.types import FrequencySeries

    return FrequencySeries(
        new_values,
        delta_f=expected_df
    )


# ============================================================
# LAMBDA MODEL
# ============================================================

def lambda_phase_factor(
    frequencies,
    lam,
):

    f = np.asarray(
        frequencies,
        dtype=np.float64
    )

    phase = (
        lam
        * (f / 100.0) ** 2
    )

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

    modified = np.fft.irfft(
        hf * deformation,
        n=n
    )

    return TimeSeries(
        modified,
        delta_t=dt
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
        delta_t=1.0 / FS
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
            return 0.0, 0.0

        absarr = np.abs(arr)

        idx = int(
            np.argmax(absarr)
        )

        peak = float(
            absarr[idx]
        )

        return peak, idx / FS

    except Exception as exc:

        print(
            "WARNING: matched filter failed:",
            repr(exc)
        )

        return 0.0, 0.0


# ============================================================
# PROFILE
# ============================================================

def build_lambda_grid():

    return np.arange(
        LAMBDA_MIN,
        LAMBDA_MAX
        + 0.5 * LAMBDA_STEP,
        LAMBDA_STEP,
    )


def profile_scan(
    strain_segment,
    templates,
    lambdas,
    psd,
):

    scores = []
    times = []

    for template in templates:

        score, peak_time = matched_score(
            strain_segment,
            template,
            psd,
        )

        scores.append(score)
        times.append(peak_time)

    scores = np.asarray(
        scores,
        dtype=np.float64
    )

    times = np.asarray(
        times,
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
        - score0 ** 2
    )

    best_index = int(
        np.argmax(delta_scores)
    )

    best_lambda = float(
        lambdas[best_index]
    )

    best_delta = float(
        delta_scores[best_index]
    )

    if best_index == 0:

        location = "LOWER_BOUNDARY"

    elif best_index == len(lambdas) - 1:

        location = "UPPER_BOUNDARY"

    else:

        location = "INTERIOR"

    return {
        "lambdas": lambdas,
        "scores": scores,
        "delta_scores": delta_scores,
        "times": times,
        "score0": score0,
        "best_index": best_index,
        "best_lambda": best_lambda,
        "best_delta": best_delta,
        "location": location,
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
        * (x1 - x3)
    )

    denom2 = (
        (x2 - x1)
        * (x2 - x3)
    )

    denom3 = (
        (x3 - x1)
        * (x3 - x2)
    )

    if (
        abs(denom) < 1e-15
        or abs(denom2) < 1e-15
        or abs(denom3) < 1e-15
    ):
        return x2

    A = (
        y1 / denom
        + y2 / denom2
        + y3 / denom3
    )

    B = -(
        y1 * (x2 + x3) / denom
        + y2 * (x1 + x3) / denom2
        + y3 * (x1 + x2) / denom3
    )

    if A >= 0:
        return x2

    vertex = -B / (
        2.0 * A
    )

    if vertex < x1 or vertex > x3:
        return x2

    return float(vertex)


# ============================================================
# NULL BIAS
# ============================================================

def robust_scale(x):

    x = np.asarray(
        x,
        dtype=np.float64
    )

    med = np.median(x)

    mad = np.median(
        np.abs(x - med)
    )

    sigma = (
        1.4826 * mad
    )

    if (
        not np.isfinite(sigma)
        or sigma <= 0
    ):

        sigma = np.std(
            x,
            ddof=1
        )

    if (
        not np.isfinite(sigma)
        or sigma <= 0
    ):

        sigma = 1.0

    return float(sigma)


def calculate_null_calibration(
    null_profiles,
):

    matrix = np.asarray(
        null_profiles,
        dtype=np.float64
    )

    bias = np.median(
        matrix,
        axis=0
    )

    sigma = np.zeros(
        matrix.shape[1],
        dtype=np.float64
    )

    lower = np.zeros_like(
        sigma
    )

    upper = np.zeros_like(
        sigma
    )

    for j in range(
        matrix.shape[1]
    ):

        column = matrix[:, j]

        sigma[j] = robust_scale(
            column
        )

        lower[j] = np.percentile(
            column,
            2.5
        )

        upper[j] = np.percentile(
            column,
            97.5
        )

    return (
        bias,
        sigma,
        lower,
        upper
    )


# ============================================================
# EMPIRICAL GLOBAL NULL TEST
# ============================================================

def empirical_ge(
    observed,
    null_values,
):

    null_values = np.asarray(
        null_values,
        dtype=np.float64
    )

    if len(null_values) == 0:
        return np.nan

    return float(
        (
            np.sum(
                null_values >= observed
            ) + 1
        )
        / (
            len(null_values) + 1
        )
    )


# ============================================================
# CSV
# ============================================================

def save_on_source_corrected(
    lambdas,
    delta_on,
    bias,
    sigma,
    z,
):

    path = os.path.join(
        OUTDIR,
        "stage6E3F_on_source_corrected_profile.csv"
    )

    with open(
        path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "lambda",
            "deltaS_on",
            "null_bias",
            "null_sigma",
            "bias_corrected_deltaS",
            "Z",
        ])

        for row in zip(
            lambdas,
            delta_on,
            bias,
            sigma,
            delta_on - bias,
            z,
        ):

            writer.writerow([
                f"{row[0]:.10f}",
                f"{row[1]:.12e}",
                f"{row[2]:.12e}",
                f"{row[3]:.12e}",
                f"{row[4]:.12e}",
                f"{row[5]:.12e}",
            ])

    return path


def save_null_bias(
    lambdas,
    bias,
    sigma,
    lower,
    upper,
):

    path = os.path.join(
        OUTDIR,
        "stage6E3F_null_bias.csv"
    )

    with open(
        path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "lambda",
            "null_median_bias",
            "null_robust_sigma",
            "null_lower_2p5",
            "null_upper_97p5",
        ])

        for row in zip(
            lambdas,
            bias,
            sigma,
            lower,
            upper,
        ):

            writer.writerow([
                f"{row[0]:.10f}",
                f"{row[1]:.12e}",
                f"{row[2]:.12e}",
                f"{row[3]:.12e}",
                f"{row[4]:.12e}",
            ])

    return path


def save_null_envelope(
    lambdas,
    bias,
    lower,
    upper,
):

    path = os.path.join(
        OUTDIR,
        "stage6E3F_null_envelope.csv"
    )

    with open(
        path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "lambda",
            "median",
            "lower_2p5",
            "upper_97p5",
        ])

        for row in zip(
            lambdas,
            bias,
            lower,
            upper,
        ):

            writer.writerow([
                f"{row[0]:.10f}",
                f"{row[1]:.12e}",
                f"{row[2]:.12e}",
                f"{row[3]:.12e}",
            ])

    return path


def save_off_source_profiles(
    records
):

    path = os.path.join(
        OUTDIR,
        "stage6E3F_off_source_profiles.csv"
    )

    with open(
        path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "index",
            "gps",
            "best_lambda",
            "refined_lambda",
            "best_deltaS",
            "location",
        ])

        for r in records:

            writer.writerow([
                r["index"],
                f'{r["gps"]:.3f}',
                f'{r["best_lambda"]:.10f}',
                f'{r["refined_lambda"]:.10f}',
                f'{r["best_deltaS"]:.12e}',
                r["location"],
            ])

    return path


def save_summary(
    rows
):

    path = os.path.join(
        OUTDIR,
        "stage6E3F_summary.csv"
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

        t = (
            event_gps
            - k * spacing
        )

        if (
            t - half >= gps_start
            and
            t + half
            <= gps_start + duration
            and
            abs(t - event_gps)
            > guard + half
        ):

            usable.append(t)

        k += 1

        if k > 10000:
            break

    k = 1

    while len(usable) < n_requested:

        t = (
            event_gps
            + k * spacing
        )

        if (
            t - half >= gps_start
            and
            t + half
            <= gps_start + duration
            and
            abs(t - event_gps)
            > guard + half
        ):

            usable.append(t)

        k += 1

        if k > 10000:
            break

    return usable[:n_requested]


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Stage 6E3F "
            "null-bias-corrected "
            "Lambda profile test"
        )
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to LOSC HDF5 file",
    )

    parser.add_argument(
        "--n-offsource",
        type=int,
        default=N_OFFSOURCE,
    )

    parser.add_argument(
        "--spacing",
        type=float,
        default=OFFSOURCE_SPACING,
    )

    parser.add_argument(
        "--lambda-min",
        type=float,
        default=LAMBDA_MIN,
    )

    parser.add_argument(
        "--lambda-max",
        type=float,
        default=LAMBDA_MAX,
    )

    parser.add_argument(
        "--lambda-step",
        type=float,
        default=LAMBDA_STEP,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # IMPORTANT:
    # Do NOT use global declarations here.
    # Keep runtime configuration local.
    # --------------------------------------------------------

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
        "STAGE 6E3F — "
        "NULL-BIAS-CORRECTED PROFILE TEST"
    )
    print("=" * 80)

    print()
    print(
        f"Data file          = "
        f"{args.data}"
    )

    print(
        f"Event GPS          = "
        f"{EVENT_GPS}"
    )

    print(
        f"Lambda range       = "
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
        f"Off-source spacing = "
        f"{args.spacing} s"
    )

    print()
    print(
        "Lambda_true        = NONE"
    )

    print(
        "Detector data      = REAL H1"
    )

    print(
        "Model              = "
        "phenomenological Lambda*f^2 phase"
    )

    print()
    print(
        "NULL CALIBRATION   = OFF-SOURCE ONLY"
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print()
    print(
        "[1] LOADING REAL H1 DATA"
    )

    raw, gps_start, duration, detector = \
        read_hdf5_strain(
            args.data
        )

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

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    raw, mean_removed = clean_strain(
        raw
    )

    print()
    print(
        "[2] STRAIN CLEANING"
    )

    print(
        f"Mean removed       = "
        f"{mean_removed:.6e}"
    )

    # --------------------------------------------------------
    # TEMPLATE
    # --------------------------------------------------------

    print()
    print(
        "[3] GENERATING GR TEMPLATE"
    )

    template_gr = generate_gr_template()

    print(
        f"Template samples   = "
        f"{len(template_gr)}"
    )

    # --------------------------------------------------------
    # GRID
    # --------------------------------------------------------

    print()
    print(
        "[4] BUILDING LAMBDA TEMPLATE FAMILY"
    )

    lambdas = np.arange(
        lambda_min,
        lambda_max
        + 0.5 * lambda_step,
        lambda_step,
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
        f"Lambda templates   = "
        f"{len(templates)}"
    )

    print(
        f"Lambda range       = "
        f"[{lambdas[0]:.3f}, "
        f"{lambdas[-1]:.3f}]"
    )

    # ========================================================
    # ON-SOURCE
    # ========================================================

    print()
    print(
        "[5] ON-SOURCE GW150914"
    )

    on_segment = extract_segment(
        raw,
        gps_start,
        EVENT_GPS,
    )

    if on_segment is None:

        raise RuntimeError(
            "Could not extract on-source segment."
        )

    on_psd = estimate_psd(
        on_segment
    )

    on_psd = prepare_psd_for_filter(
        on_psd,
        template_gr
    )

    on_profile = profile_scan(
        on_segment,
        templates,
        lambdas,
        on_psd,
    )

    on_refined = local_quadratic_refinement(
        on_profile["lambdas"],
        on_profile["delta_scores"],
        on_profile["best_index"],
    )

    print(
        f"Raw best Lambda   = "
        f"{on_profile['best_lambda']:+.6f}"
    )

    print(
        f"Raw refined Lambda = "
        f"{on_refined:+.6f}"
    )

    print(
        f"Raw DeltaS         = "
        f"{on_profile['best_delta']:.12e}"
    )

    print(
        f"Raw location       = "
        f"{on_profile['location']}"
    )

    # ========================================================
    # OFF-SOURCE
    # ========================================================

    print()
    print(
        "[6] BUILDING OFF-SOURCE TIMES"
    )

    off_times = build_offsource_times(
        gps_start,
        duration,
        EVENT_GPS,
        args.n_offsource,
        args.spacing,
        EVENT_GUARD,
    )

    print(
        f"Usable segments    = "
        f"{len(off_times)}"
    )

    # ========================================================
    # OFF-SOURCE PROFILE MATRIX
    # ========================================================

    print()
    print(
        "[7] BUILDING OFF-SOURCE PROFILE MATRIX"
    )

    records = []

    null_profiles = []

    for i, gps in enumerate(
        off_times,
        start=1
    ):

        segment = extract_segment(
            raw,
            gps_start,
            gps,
        )

        if segment is None:
            continue

        psd = estimate_psd(
            segment
        )

        psd = prepare_psd_for_filter(
            psd,
            template_gr
        )

        profile = profile_scan(
            segment,
            templates,
            lambdas,
            psd,
        )

        refined = local_quadratic_refinement(
            profile["lambdas"],
            profile["delta_scores"],
            profile["best_index"],
        )

        null_profiles.append(
            profile["delta_scores"]
        )

        records.append({
            "index": i,
            "gps": gps,
            "best_lambda":
                profile["best_lambda"],
            "refined_lambda":
                refined,
            "best_deltaS":
                profile["best_delta"],
            "location":
                profile["location"],
        })

        print(
            f"{i:3d} "
            f"GPS={gps:.1f} "
            f"Lambda="
            f"{profile['best_lambda']:+.3f} "
            f"refined="
            f"{refined:+.3f} "
            f"DeltaS="
            f"{profile['best_delta']:.8e} "
            f"{profile['location']}"
        )

    if len(null_profiles) < 5:

        raise RuntimeError(
            "Too few valid off-source profiles "
            "for null calibration."
        )

    null_profiles = np.asarray(
        null_profiles,
        dtype=np.float64
    )

    # ========================================================
    # NULL CALIBRATION
    # ========================================================

    print()
    print(
        "[8] ESTIMATING NULL BIAS"
    )

    (
        null_bias,
        null_sigma,
        null_lower,
        null_upper,
    ) = calculate_null_calibration(
        null_profiles
    )

    print(
        "Null calibration uses "
        f"{len(null_profiles)} "
        "off-source profiles."
    )

    # ========================================================
    # ON-SOURCE CORRECTION
    # ========================================================

    print()
    print(
        "[9] APPLYING NULL-BIAS CORRECTION"
    )

    delta_on = np.asarray(
        on_profile["delta_scores"],
        dtype=np.float64
    )

    corrected = (
        delta_on
        - null_bias
    )

    z_profile = (
        corrected
        / null_sigma
    )

    corrected_index = int(
        np.argmax(corrected)
    )

    corrected_lambda = float(
        lambdas[corrected_index]
    )

    corrected_refined = \
        local_quadratic_refinement(
            lambdas,
            corrected,
            corrected_index,
        )

    corrected_value = float(
        corrected[corrected_index]
    )

    corrected_location = "INTERIOR"

    if corrected_index == 0:

        corrected_location = \
            "LOWER_BOUNDARY"

    elif corrected_index == len(lambdas) - 1:

        corrected_location = \
            "UPPER_BOUNDARY"

    # --------------------------------------------------------
    # RMS Z
    # --------------------------------------------------------

    rms_z = float(
        np.sqrt(
            np.mean(
                z_profile ** 2
            )
        )
    )

    max_abs_z = float(
        np.max(
            np.abs(z_profile)
        )
    )

    max_abs_z_index = int(
        np.argmax(
            np.abs(z_profile)
        )
    )

    max_abs_z_lambda = float(
        lambdas[max_abs_z_index]
    )

    # --------------------------------------------------------
    # Null distribution of global RMS/max statistics
    # --------------------------------------------------------

    null_bias_matrix = np.tile(
        null_bias,
        (
            len(null_profiles),
            1
        )
    )

    null_sigma_matrix = np.tile(
        null_sigma,
        (
            len(null_profiles),
            1
        )
    )

    null_z_matrix = (
        (
            null_profiles
            - null_bias_matrix
        )
        / null_sigma_matrix
    )

    null_rms = np.sqrt(
        np.mean(
            null_z_matrix ** 2,
            axis=1
        )
    )

    null_max_abs = np.max(
        np.abs(
            null_z_matrix
        ),
        axis=1
    )

    p_rms = empirical_ge(
        rms_z,
        null_rms
    )

    p_max = empirical_ge(
        max_abs_z,
        null_max_abs
    )

    # ========================================================
    # NULL LAMBDA DIAGNOSTICS
    # ========================================================

    null_best_lambda = np.asarray(
        [
            r["best_lambda"]
            for r in records
        ],
        dtype=np.float64
    )

    positive_fraction = float(
        np.mean(
            null_best_lambda > 0
        )
    )

    negative_fraction = float(
        np.mean(
            null_best_lambda < 0
        )
    )

    zero_fraction = float(
        np.mean(
            null_best_lambda == 0
        )
    )

    upper_count = int(
        np.sum(
            [
                r["location"]
                == "UPPER_BOUNDARY"
                for r in records
            ]
        )
    )

    lower_count = int(
        np.sum(
            [
                r["location"]
                == "LOWER_BOUNDARY"
                for r in records
            ]
        )
    )

    interior_count = int(
        np.sum(
            [
                r["location"]
                == "INTERIOR"
                for r in records
            ]
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    on_path = save_on_source_corrected(
        lambdas,
        delta_on,
        null_bias,
        null_sigma,
        z_profile,
    )

    bias_path = save_null_bias(
        lambdas,
        null_bias,
        null_sigma,
        null_lower,
        null_upper,
    )

    envelope_path = save_null_envelope(
        lambdas,
        null_bias,
        null_lower,
        null_upper,
    )

    off_path = save_off_source_profiles(
        records
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 80)
    print(
        "STAGE 6E3F RESULTS"
    )
    print("=" * 80)

    print()
    print(
        f"Raw on-source Lambda       = "
        f"{on_profile['best_lambda']:+.6f}"
    )

    print(
        f"Raw refined Lambda         = "
        f"{on_refined:+.6f}"
    )

    print(
        f"Raw DeltaS                 = "
        f"{on_profile['best_delta']:.12e}"
    )

    print()
    print(
        "BIAS-CORRECTED ON-SOURCE"
    )

    print(
        f"Corrected best Lambda      = "
        f"{corrected_lambda:+.6f}"
    )

    print(
        f"Corrected refined Lambda   = "
        f"{corrected_refined:+.6f}"
    )

    print(
        f"Corrected DeltaS           = "
        f"{corrected_value:.12e}"
    )

    print(
        f"Corrected location         = "
        f"{corrected_location}"
    )

    print()
    print(
        "PROFILE Z TEST"
    )

    print(
        f"On-source RMS Z            = "
        f"{rms_z:.6f}"
    )

    print(
        f"On-source max |Z|          = "
        f"{max_abs_z:.6f}"
    )

    print(
        f"Max |Z| Lambda             = "
        f"{max_abs_z_lambda:+.6f}"
    )

    print(
        f"Empirical p(RMS Z)         = "
        f"{p_rms:.6f}"
    )

    print(
        f"Empirical p(max |Z|)       = "
        f"{p_max:.6f}"
    )

    print()
    print(
        "NULL LAMBDA DISTRIBUTION"
    )

    print(
        f"Positive fraction          = "
        f"{positive_fraction:.6f}"
    )

    print(
        f"Negative fraction          = "
        f"{negative_fraction:.6f}"
    )

    print(
        f"Zero fraction              = "
        f"{zero_fraction:.6f}"
    )

    print()
    print(
        "BOUNDARY COUNTS"
    )

    print(
        f"Interior                   = "
        f"{interior_count}"
    )

    print(
        f"Upper boundary             = "
        f"{upper_count}"
    )

    print(
        f"Lower boundary             = "
        f"{lower_count}"
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    print()
    print("=" * 80)
    print(
        "STAGE 6E3F INTERPRETATION"
    )
    print("=" * 80)

    print()
    print(
        "The Lambda bias was estimated exclusively "
        "from off-source real H1 profiles."
    )

    print(
        "GW150914 was not used in the null calibration."
    )

    print()

    if p_rms < 0.05:

        print(
            "The on-source RMS profile residual is "
            "unusual relative to the empirical null."
        )

    else:

        print(
            "The on-source RMS profile residual is "
            "compatible with the empirical null."
        )

    if p_max < 0.05:

        print(
            "The maximum profile deviation is "
            "unusual relative to the empirical null."
        )

    else:

        print(
            "The maximum profile deviation is "
            "compatible with the empirical null."
        )

    print()
    print(
        "The bias-corrected Lambda maximum is "
        "a phenomenological estimator."
    )

    print(
        "It is not by itself evidence for a "
        "physical nonzero Lambda."
    )

    print()
    print(
        "The empirical p-values are conditional on "
        "the selected off-source segments and "
        "the tested Lambda grid."
    )

    print(
        "No fundamental-physics significance is "
        "claimed by Stage 6E3F."
    )

    # ========================================================
    # SUMMARY CSV
    # ========================================================

    summary_rows = [

        [
            "parameter",
            "value",
        ],

        [
            "event",
            "GW150914",
        ],

        [
            "detector",
            "H1",
        ],

        [
            "lambda_min",
            lambda_min,
        ],

        [
            "lambda_max",
            lambda_max,
        ],

        [
            "lambda_step",
            lambda_step,
        ],

        [
            "off_source_N",
            len(records),
        ],

        [
            "raw_best_lambda",
            on_profile["best_lambda"],
        ],

        [
            "raw_refined_lambda",
            on_refined,
        ],

        [
            "raw_deltaS",
            on_profile["best_delta"],
        ],

        [
            "corrected_best_lambda",
            corrected_lambda,
        ],

        [
            "corrected_refined_lambda",
            corrected_refined,
        ],

        [
            "corrected_deltaS",
            corrected_value,
        ],

        [
            "corrected_location",
            corrected_location,
        ],

        [
            "rms_Z",
            rms_z,
        ],

        [
            "max_abs_Z",
            max_abs_z,
        ],

        [
            "max_abs_Z_lambda",
            max_abs_z_lambda,
        ],

        [
            "empirical_p_rms_Z",
            p_rms,
        ],

        [
            "empirical_p_max_abs_Z",
            p_max,
        ],

        [
            "null_positive_lambda_fraction",
            positive_fraction,
        ],

        [
            "null_negative_lambda_fraction",
            negative_fraction,
        ],

        [
            "null_zero_lambda_fraction",
            zero_fraction,
        ],

        [
            "null_interior_count",
            interior_count,
        ],

        [
            "null_upper_boundary_count",
            upper_count,
        ],

        [
            "null_lower_boundary_count",
            lower_count,
        ],

        [
            "physical_lambda_claim",
            "NO",
        ],

        [
            "null_calibration_source",
            "OFF_SOURCE_ONLY",
        ],
    ]

    summary_path = save_summary(
        summary_rows
    )

    # ========================================================
    # FINAL PATHS
    # ========================================================

    print()
    print(
        f"On-source corrected = "
        f"{on_path}"
    )

    print(
        f"Off-source profiles = "
        f"{off_path}"
    )

    print(
        f"Null bias           = "
        f"{bias_path}"
    )

    print(
        f"Null envelope       = "
        f"{envelope_path}"
    )

    print(
        f"Summary             = "
        f"{summary_path}"
    )

    print()
    print("=" * 80)
    print(
        "STAGE 6E3F COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
