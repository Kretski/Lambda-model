#!/usr/bin/env python3

import os
import sys
import csv
import argparse
import numpy as np
import h5py

from pycbc.types import TimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.psd import welch, interpolate
from pycbc.filter import matched_filter, highpass
from scipy.signal.windows import tukey

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from waveform import (
    lambda_phase_correction,
    cosmological_K_factor,
)


# ============================================================
# STAGE 6E3H-R v2
# REALISTIC INJECTION / RECOVERY CALIBRATION
#
# PURPOSE
# -------
# Validate whether the Lambda parameter used by the project
# can be recovered from realistic H1 off-source data.
#
# IMPORTANT:
#   This is a calibration/control stage.
#   It is NOT a physical detection claim.
#
# v2 CONTROL POINTS
# -----------------
#   A. Same physical Lambda phase model for injection/recovery
#   B. Same 20-300 Hz analysis band throughout
#   C. Time + phase maximization through max(|complex SNR|)
#   D. Explicit Lambda=0 GR calibration
#   E. No post-hoc Lambda bias correction
#   F. Consistent IMRPhenomD masses
#   G. Explicit frequency/PSD sanity checks
#
# Analysis band:
#       20 Hz <= f <= 300 Hz
#
# Lambda model:
#       h_Lambda(f) = h_GR(f) * exp(i * DeltaPsi(f))
#
# DeltaPsi(f) is supplied by waveform.py.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

EVENT_GPS = 1126259462.0

M1 = 35.6
M2 = 30.6

SAMPLE_RATE = 4096.0
DURATION = 16.0

F_LOW = 20.0
F_HIGH = 300.0

REDSHIFT = 0.09
K_Z = cosmological_K_factor(REDSHIFT)

TARGET_SNRS = [
    8.0,
    12.0,
    20.0,
    24.0,
]

INJECTED_LAMBDAS = [
    -0.4,
    -0.2,
    0.0,
    0.2,
    0.4,
]

LAMBDA_MIN = -3.0
LAMBDA_MAX = 3.0
LAMBDA_STEP = 0.06

N_REALIZATIONS = 100

OFFSOURCE_SPACING = 32.0

OUTPUT_DIR = "stage6E3H_realistic_results_v2"


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Stage 6E3H-R v2 realistic "
            "Lambda injection/recovery calibration"
        )
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to GWOSC/LOSC HDF5 strain file",
    )

    parser.add_argument(
        "--n-realizations",
        type=int,
        default=N_REALIZATIONS,
        help="Number of off-source realizations",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260831,
        help="Random seed",
    )

    parser.add_argument(
        "--target-snr",
        type=float,
        default=None,
        help="Run only one target SNR",
    )

    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Output directory",
    )

    return parser.parse_args()


# ============================================================
# HDF5 LOADER
# ============================================================

def read_losc_hdf5(path):

    with h5py.File(path, "r") as f:

        if "strain/Strain" not in f:

            raise KeyError(
                "Expected strain/Strain dataset in HDF5."
            )

        data = np.asarray(
            f["strain/Strain"][:],
            dtype=np.float64,
        )

        if "meta/GPSstart" not in f:

            raise RuntimeError(
                "HDF5 file has no meta/GPSstart."
            )

        gps_start = float(
            np.asarray(
                f["meta/GPSstart"]
            )
        )

        duration = None

        if "meta/Duration" in f:

            duration = float(
                np.asarray(
                    f["meta/Duration"]
                )
            )

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

    n = len(data)

    if (
        duration is not None
        and duration > 0
    ):

        fs = n / duration

    else:

        fs = SAMPLE_RATE

    delta_t = 1.0 / fs

    ts = TimeSeries(
        data,
        delta_t=delta_t,
        epoch=gps_start,
    )

    print(
        f"  Detector        = {detector}"
    )

    print(
        f"  GPS start       = {gps_start:.6f}"
    )

    print(
        f"  Samples         = {n}"
    )

    print(
        f"  Sample rate     = {fs:.6f} Hz"
    )

    print(
        f"  Duration        = "
        f"{n * delta_t:.6f} s"
    )

    return ts, detector


# ============================================================
# LAMBDA PHASE APPLICATION
# ============================================================

def apply_lambda_phase(
    template,
    lam,
):

    """
    Apply the physical Lambda phase model supplied by waveform.py.

    No ad-hoc Lambda*f^2 model is used.

    The same function is used for:
        - injection
        - recovery template generation
    """

    n = len(template)

    dt = float(
        template.delta_t
    )

    freq = np.fft.rfftfreq(
        n,
        d=dt,
    )

    hf = np.fft.rfft(
        np.asarray(template),
    )

    phase = lambda_phase_correction(
        freq,
        float(lam),
        K_Z,
    )

    deformed_hf = (
        hf *
        np.exp(
            1j * phase
        )
    )

    out = np.fft.irfft(
        deformed_hf,
        n=n,
    )

    result = TimeSeries(
        out,
        delta_t=dt,
        epoch=template.start_time,
    )

    return result


# ============================================================
# GR TEMPLATE
# ============================================================

def generate_gr_template():

    print(
        f"  IMRPhenomD masses = "
        f"{M1:.2f} + {M2:.2f} Msun"
    )

    hp, hc = get_td_waveform(
        approximant="IMRPhenomD",
        mass1=M1,
        mass2=M2,
        delta_t=1.0 / SAMPLE_RATE,
        f_lower=F_LOW,
    )

    template = hp.copy()

    target_n = int(
        round(
            DURATION *
            SAMPLE_RATE
        )
    )

    arr = np.asarray(
        template,
        dtype=np.float64,
    )

    if len(arr) > target_n:

        arr = arr[-target_n:]

    elif len(arr) < target_n:

        padded = np.zeros(
            target_n,
            dtype=np.float64,
        )

        padded[-len(arr):] = arr

        arr = padded

    template = TimeSeries(
        arr,
        delta_t=1.0 / SAMPLE_RATE,
    )

    template = (
        template -
        np.mean(template)
    )

    return template


# ============================================================
# PSD
# ============================================================

def make_psd(data):

    seg_len = int(
        4.0 *
        SAMPLE_RATE
    )

    psd = welch(
        data,
        seg_len=seg_len,
        seg_stride=seg_len // 2,
    )

    target_delta_f = (
        1.0 /
        DURATION
    )

    if abs(
        psd.delta_f -
        target_delta_f
    ) > 1e-12:

        psd = interpolate(
            psd,
            target_delta_f,
        )

    return psd


# ============================================================
# PSD / FREQUENCY SANITY CHECK
# ============================================================

def check_frequency_psd_consistency(
    signal,
    psd,
):

    dt = float(
        signal.delta_t
    )

    n = len(signal)

    freqs = np.fft.rfftfreq(
        n,
        d=dt,
    )

    psd_arr = np.asarray(
        psd[:len(freqs)],
        dtype=np.float64,
    )

    mask = (
        (freqs >= F_LOW)
        &
        (freqs <= F_HIGH)
    )

    if not np.any(mask):

        raise RuntimeError(
            "No frequency bins inside analysis band."
        )

    valid = (
        mask
        &
        np.isfinite(psd_arr)
        &
        (psd_arr > 0)
    )

    if not np.any(valid):

        raise RuntimeError(
            "No valid PSD values in analysis band."
        )

    df_fft = (
        1.0 /
        (n * dt)
    )

    print(
        f"    FFT df       = {df_fft:.8f} Hz"
    )

    print(
        f"    PSD df       = "
        f"{float(psd.delta_f):.8f} Hz"
    )

    print(
        f"    Band bins    = "
        f"{int(np.sum(valid))}"
    )

    print(
        f"    PSD median   = "
        f"{np.median(psd_arr[valid]):.6e}"
    )

    return True


# ============================================================
# BAND-LIMITED OPTIMAL SNR
# ============================================================

def calculate_band_snr(
    signal,
    psd,
):

    """
    Calculate optimal SNR only in F_LOW-F_HIGH.

    This is deliberately identical in band definition
    to the recovery matched filter.
    """

    dt = float(
        signal.delta_t
    )

    n = len(signal)

    freqs = np.fft.rfftfreq(
        n,
        d=dt,
    )

    hf = (
        np.fft.rfft(
            np.asarray(signal)
        )
        *
        dt
    )

    psd_arr = np.asarray(
        psd[:len(hf)],
        dtype=np.float64,
    )

    mask = (
        (freqs >= F_LOW)
        &
        (freqs <= F_HIGH)
        &
        np.isfinite(psd_arr)
        &
        (psd_arr > 0)
    )

    if not np.any(mask):

        raise RuntimeError(
            "No valid PSD bins in "
            f"{F_LOW}-{F_HIGH} Hz."
        )

    df = 1.0 / (
        n * dt
    )

    sigma2 = (
        4.0 *
        df *
        np.sum(
            (
                np.abs(
                    hf[mask]
                ) ** 2
            )
            /
            psd_arr[mask]
        )
    )

    if (
        not np.isfinite(sigma2)
        or sigma2 <= 0
    ):

        raise RuntimeError(
            "Invalid band-limited SNR."
        )

    return float(
        np.sqrt(sigma2)
    )


# ============================================================
# SCALE SIGNAL TO TARGET SNR
# ============================================================

def scale_to_snr(
    signal,
    psd,
    target_snr,
):

    current_snr = calculate_band_snr(
        signal,
        psd,
    )

    if current_snr <= 0:

        raise RuntimeError(
            "Template has zero band-limited SNR."
        )

    factor = (
        float(target_snr) /
        current_snr
    )

    scaled = (
        signal *
        factor
    )

    achieved_snr = calculate_band_snr(
        scaled,
        psd,
    )

    return (
        scaled,
        current_snr,
        achieved_snr,
    )


# ============================================================
# MATCHED-FILTER RECOVERY
# ============================================================

def recover_lambda(
    data,
    templates,
    psd,
):

    """
    Recover Lambda by maximizing the absolute complex SNR
    over the matched-filter time series.

    abs(complex SNR) provides phase maximization.

    max over the time series provides time maximization.

    Therefore:

        score(lambda)
          = max_t |rho_lambda(t)|

    and the selected Lambda is the grid point with
    maximum score.
    """

    best_lambda = None
    best_score = -np.inf

    for lam, template in templates:

        try:

            snr_series = matched_filter(
                template,
                data,
                psd=psd,
                low_frequency_cutoff=F_LOW,
                high_frequency_cutoff=F_HIGH,
            )

            snr_values = np.asarray(
                snr_series,
                dtype=np.complex128,
            )

            finite = np.isfinite(
                snr_values
            )

            if not np.any(finite):

                continue

            score = float(
                np.max(
                    np.abs(
                        snr_values[finite]
                    )
                )
            )

            if (
                np.isfinite(score)
                and score > best_score
            ):

                best_score = score
                best_lambda = float(lam)

        except Exception:

            continue

    return (
        best_lambda,
        best_score,
    )


# ============================================================
# OFFSOURCE TIMES
# ============================================================

def build_offsource_times(
    gps_start,
    duration,
    event_offset,
    n,
):

    event_time = (
        gps_start +
        event_offset
    )

    times = []

    k = 1

    while len(times) < n:

        t = (
            event_time -
            k *
            OFFSOURCE_SPACING
        )

        if (
            t <
            gps_start +
            DURATION
        ):

            break

        if (
            t +
            DURATION >
            gps_start +
            duration
        ):

            k += 1
            continue

        times.append(t)

        k += 1

    k = 1

    while len(times) < n:

        t = (
            event_time +
            k *
            OFFSOURCE_SPACING
        )

        if (
            t +
            DURATION >
            gps_start +
            duration
        ):

            break

        if (
            t <
            gps_start +
            DURATION
        ):

            k += 1
            continue

        times.append(t)

        k += 1

    return times[:n]


# ============================================================
# EXTRACT SEGMENT
# ============================================================

def extract_segment(
    full_data,
    gps,
):

    dt = float(
        full_data.delta_t
    )

    start = int(
        round(
            (
                gps -
                float(
                    full_data.start_time
                )
            )
            /
            dt
        )
    )

    n = int(
        round(
            DURATION /
            dt
        )
    )

    end = start + n

    if (
        start < 0
        or end > len(full_data)
    ):

        raise ValueError(
            "Requested segment outside data."
        )

    segment = (
        full_data[start:end]
        .copy()
    )

    segment = (
        segment -
        np.mean(segment)
    )

    return segment


# ============================================================
# SUMMARY
# ============================================================

def summarize(rows):

    print()
    print("=" * 80)
    print(
        "STAGE 6E3H-R v2 "
        "CALIBRATION SUMMARY"
    )
    print("=" * 80)

    grouped = {}

    for row in rows:

        key = (
            float(row["target_snr"]),
            float(row["lambda_injected"]),
        )

        grouped.setdefault(
            key,
            [],
        ).append(row)

    summary_rows = []

    for (
        snr,
        injected,
    ), values in sorted(
        grouped.items()
    ):

        recovered = np.asarray(
            [
                x["lambda_recovered"]
                for x in values
            ],
            dtype=float,
        )

        errors = (
            recovered -
            injected
        )

        mean = float(
            np.mean(recovered)
        )

        median = float(
            np.median(recovered)
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
            np.std(recovered)
        )

        within1 = float(
            np.mean(
                np.abs(errors) <= 1.0
            )
        )

        within2 = float(
            np.mean(
                np.abs(errors) <= 2.0
            )
        )

        print()
        print(
            f"Target SNR       = {snr:.1f}"
        )

        print(
            f"Lambda injected  = "
            f"{injected:+.3f}"
        )

        print(
            f"Recovered mean   = "
            f"{mean:+.6f}"
        )

        print(
            f"Recovered median = "
            f"{median:+.6f}"
        )

        print(
            f"Bias             = "
            f"{bias:+.6f}"
        )

        print(
            f"RMSE             = "
            f"{rmse:.6f}"
        )

        print(
            f"Std              = "
            f"{std:.6f}"
        )

        print(
            f"|error| <= 1     = "
            f"{within1:.6f}"
        )

        print(
            f"|error| <= 2     = "
            f"{within2:.6f}"
        )

        summary_rows.append({

            "target_snr":
                snr,

            "lambda_injected":
                injected,

            "recovered_mean":
                mean,

            "recovered_median":
                median,

            "bias":
                bias,

            "rmse":
                rmse,

            "std":
                std,

            "fraction_error_le_1":
                within1,

            "fraction_error_le_2":
                within2,

        })

    return summary_rows


# ============================================================
# GLOBAL DIAGNOSTIC
# ============================================================

def global_diagnostic(rows):

    injected = np.asarray(
        [
            r["lambda_injected"]
            for r in rows
        ],
        dtype=float,
    )

    recovered = np.asarray(
        [
            r["lambda_recovered"]
            for r in rows
        ],
        dtype=float,
    )

    if (
        np.std(injected) > 0
        and np.std(recovered) > 0
    ):

        correlation = float(
            np.corrcoef(
                injected,
                recovered,
            )[0, 1]
        )

    else:

        correlation = float("nan")

    global_rmse = float(
        np.sqrt(
            np.mean(
                (
                    recovered -
                    injected
                ) ** 2
            )
        )
    )

    zero_mask = np.isclose(
        injected,
        0.0,
    )

    if np.any(zero_mask):

        gr_recovered = recovered[
            zero_mask
        ]

        gr_bias = float(
            np.mean(
                gr_recovered
            )
        )

        gr_std = float(
            np.std(
                gr_recovered
            )
        )

        gr_within1 = float(
            np.mean(
                np.abs(
                    gr_recovered
                ) <= 1.0
            )
        )

    else:

        gr_bias = float("nan")
        gr_std = float("nan")
        gr_within1 = float("nan")

    print()
    print("=" * 80)
    print(
        "GLOBAL REALISTIC "
        "RECOVERY DIAGNOSTIC"
    )
    print("=" * 80)

    print(
        f"Correlation(injected,recovered) "
        f"= {correlation:+.6f}"
    )

    print(
        f"Global RMSE = "
        f"{global_rmse:.6f}"
    )

    print()
    print(
        "Lambda = 0 GR CONTROL"
    )

    print(
        f"GR recovered mean = "
        f"{gr_bias:+.6f}"
    )

    print(
        f"GR recovered std  = "
        f"{gr_std:.6f}"
    )

    print(
        f"GR fraction |Lambda| <= 1 "
        f"= {gr_within1:.6f}"
    )

    return {
        "global_correlation":
            correlation,

        "global_rmse":
            global_rmse,

        "gr_bias":
            gr_bias,

        "gr_std":
            gr_std,

        "gr_fraction_abs_lambda_le_1":
            gr_within1,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    np.random.seed(
        args.seed
    )

    target_snrs = list(
        TARGET_SNRS
    )

    if (
        args.target_snr
        is not None
    ):

        target_snrs = [
            float(
                args.target_snr
            )
        ]

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        "STAGE 6E3H-R v2"
    )
    print(
        "REALISTIC INJECTION / RECOVERY"
    )
    print("=" * 80)

    print()
    print(
        "Configuration:"
    )

    print(
        f"  M1             = "
        f"{M1:.2f} Msun"
    )

    print(
        f"  M2             = "
        f"{M2:.2f} Msun"
    )

    print(
        f"  Redshift       = "
        f"{REDSHIFT:.3f}"
    )

    print(
        f"  K(z)           = "
        f"{K_Z:.8e}"
    )

    print(
        f"  Sample rate    = "
        f"{SAMPLE_RATE:.1f} Hz"
    )

    print(
        f"  Duration       = "
        f"{DURATION:.1f} s"
    )

    print(
        f"  Analysis band  = "
        f"{F_LOW:.1f}-{F_HIGH:.1f} Hz"
    )

    print(
        f"  Target SNRs    = "
        f"{target_snrs}"
    )

    print(
        f"  Injected Lambda = "
        f"{INJECTED_LAMBDAS}"
    )

    print(
        f"  Recovery grid  = "
        f"[{LAMBDA_MIN}, {LAMBDA_MAX}] "
        f"step={LAMBDA_STEP}"
    )

    print(
        f"  Realizations   = "
        f"{args.n_realizations}"
    )

    print(
        f"  Output         = "
        f"{args.output_dir}"
    )

    print()
    print(
        "Lambda = 0 is the GR control point."
    )

    print(
        "No post-hoc Lambda bias correction."
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print()
    print(
        "[1] LOADING REAL H1 DATA"
    )

    full_data, detector = (
        read_losc_hdf5(
            args.data
        )
    )

    if detector != "H1":

        print(
            "WARNING: detector metadata "
            f"is '{detector}', expected H1."
        )

    full_data = (
        full_data.astype(
            np.float64
        )
    )

    # --------------------------------------------------------
    # TEMPLATE
    # --------------------------------------------------------

    print()
    print(
        "[2] GENERATING GR TEMPLATE"
    )

    gr_template = (
        generate_gr_template()
    )

    print(
        f"Template samples = "
        f"{len(gr_template)}"
    )

    print(
        f"Template duration = "
        f"{len(gr_template) / SAMPLE_RATE:.6f} s"
    )

    # --------------------------------------------------------
    # OFFSOURCE
    # --------------------------------------------------------

    gps_start = float(
        full_data.start_time
    )

    duration = (
        len(full_data) *
        float(
            full_data.delta_t
        )
    )

    event_offset = (
        EVENT_GPS -
        gps_start
    )

    gps_times = (
        build_offsource_times(
            gps_start,
            duration,
            event_offset,
            args.n_realizations,
        )
    )

    if len(gps_times) < (
        args.n_realizations
    ):

        raise RuntimeError(
            f"Only {len(gps_times)} "
            "usable off-source segments found."
        )

    print()
    print(
        "[3] OFFSOURCE SEGMENTS"
    )

    print(
        f"Usable segments = "
        f"{len(gps_times)}"
    )

    # --------------------------------------------------------
    # RECOVERY GRID
    # --------------------------------------------------------

    recovery_grid = np.arange(
        LAMBDA_MIN,
        LAMBDA_MAX +
        0.5 *
        LAMBDA_STEP,
        LAMBDA_STEP,
    )

    print()
    print(
        "[4] BUILDING RECOVERY "
        "TEMPLATE FAMILY"
    )

    templates = []

    for lam in recovery_grid:

        template = (
            apply_lambda_phase(
                gr_template,
                float(lam),
            )
        )

        templates.append(
            (
                float(lam),
                template,
            )
        )

    print(
        f"Templates = "
        f"{len(templates)}"
    )

    # --------------------------------------------------------
    # TEMPLATE SANITY CHECK
    # --------------------------------------------------------

    print()
    print(
        "[5] TEMPLATE / PSD SANITY CHECK"
    )

    test_segment = extract_segment(
        full_data,
        gps_times[0],
    )

    test_psd = make_psd(
        test_segment
    )

    check_frequency_psd_consistency(
        test_segment,
        test_psd,
    )

    test_snr = calculate_band_snr(
        gr_template,
        test_psd,
    )

    print(
        f"GR template band SNR "
        f"before scaling = {test_snr:.6f}"
    )

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    rows = []

    total = (
        len(gps_times)
        *
        len(target_snrs)
        *
        len(INJECTED_LAMBDAS)
    )

    count = 0

    print()
    print(
        "[6] REALISTIC "
        "INJECTION / RECOVERY"
    )

    print(
        f"Total trials = "
        f"{total}"
    )

    for (
        gps_index,
        gps,
    ) in enumerate(
        gps_times
    ):

        segment = extract_segment(
            full_data,
            gps,
        )

        # КРИТИЧНО: highpass ПРЕДИ PSD estimation и matched filtering.
        # Без това, мощното нискочестотно съдържание на реалния strain
        # (seismic wall << F_LOW) доминира segment std() и PSD-то,
        # заглушавайки инжектирания сигнал в matched filter score-а
        # (диагностицирано в чат-сесия: score ~466 независимо от
        # инжекция, synthetic-noise-от-същото-PSD контролен тест
        # показа delta=+24 коректно -- проблемът е реалният unfiltered
        # strain, не кода).
        segment = highpass(
            segment,
            frequency=F_LOW * 0.9,
        )

        # Tukey прозорец ПРЕДИ PSD/matched-filter -- премахва edge
        # discontinuities (сегментна граница + highpass transient),
        # които иначе доминират score-а чрез spectral leakage
        # (диагностицирано в чат-сесия: highpass сам намали noise
        # score от 466 на 80, но не достатъчно -- остатъчният проблем
        # е липса на windowing, аналогично на вече валидирания
        # sterile_lambda_analysis.py, който ползва tukey alpha=0.1).
        _win = tukey(len(segment), alpha=0.1)
        segment = TimeSeries(
            np.asarray(segment) * _win,
            delta_t=segment.delta_t,
            epoch=segment.start_time,
        )

        psd = make_psd(
            segment
        )

        for target_snr in target_snrs:

            for injected_lambda in (
                INJECTED_LAMBDAS
            ):

                count += 1

                # --------------------------------------------
                # INJECTION
                # --------------------------------------------

                injected_template = (
                    apply_lambda_phase(
                        gr_template,
                        injected_lambda,
                    )
                )

                (
                    scaled_signal,
                    intrinsic_snr,
                    achieved_snr,
                ) = scale_to_snr(
                    injected_template,
                    psd,
                    target_snr,
                )

                injected_data = TimeSeries(
                    (
                        np.asarray(segment)
                        +
                        np.asarray(
                            scaled_signal
                        )
                    ),
                    delta_t=segment.delta_t,
                    epoch=segment.start_time,
                )

                # --------------------------------------------
                # RECOVERY
                # --------------------------------------------

                (
                    recovered_lambda,
                    score,
                ) = recover_lambda(
                    injected_data,
                    templates,
                    psd,
                )

                if (
                    recovered_lambda
                    is None
                ):

                    print(
                        f"{count:5d}/{total} "
                        f"SNR={target_snr:4.1f} "
                        f"inj={injected_lambda:+6.3f} "
                        f"RECOVERY FAILED"
                    )

                    continue

                error = (
                    recovered_lambda -
                    injected_lambda
                )

                print(
                    f"{count:5d}/{total} "
                    f"SNR={target_snr:4.1f} "
                    f"inj={injected_lambda:+6.3f} "
                    f"rec={recovered_lambda:+6.3f} "
                    f"err={error:+7.3f} "
                    f"bandSNR={achieved_snr:6.2f} "
                    f"score={score:8.3f}"
                )

                rows.append({

                    "gps":
                        gps,

                    "realization":
                        gps_index + 1,

                    "target_snr":
                        target_snr,

                    "lambda_injected":
                        injected_lambda,

                    "lambda_recovered":
                        recovered_lambda,

                    "lambda_error":
                        error,

                    "matched_filter_score":
                        score,

                    "intrinsic_template_snr":
                        intrinsic_snr,

                    "achieved_injection_snr":
                        achieved_snr,
                })

    if not rows:

        raise RuntimeError(
            "No successful recovery results."
        )

    # --------------------------------------------------------
    # SAVE RAW RESULTS
    # --------------------------------------------------------

    recovery_file = os.path.join(
        args.output_dir,
        "stage6E3H-R_v2_recovery.csv",
    )

    with open(
        recovery_file,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = summarize(
        rows
    )

    summary_file = os.path.join(
        args.output_dir,
        "stage6E3H-R_v2_summary.csv",
    )

    with open(
        summary_file,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summary[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(summary)

    # --------------------------------------------------------
    # GLOBAL DIAGNOSTIC
    # --------------------------------------------------------

    diagnostic = global_diagnostic(
        rows
    )

    diagnostic_file = os.path.join(
        args.output_dir,
        "stage6E3H-R_v2_diagnostic.csv",
    )

    with open(
        diagnostic_file,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                diagnostic.keys()
            ),
        )

        writer.writeheader()
        writer.writerow(
            diagnostic
        )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "STAGE 6E3H-R v2 FINAL REPORT"
    )
    print("=" * 80)

    print()
    print(
        "The calibration uses:"
    )

    print(
        "  [A] Physical Lambda phase model "
        "from waveform.py"
    )

    print(
        "  [B] Same 20-300 Hz band for "
        "injection SNR and recovery"
    )

    print(
        "  [C] Time maximization via "
        "max_t |complex SNR(t)|"
    )

    print(
        "  [D] Phase maximization via "
        "absolute value of complex SNR"
    )

    print(
        "  [E] Explicit Lambda=0 GR control"
    )

    print(
        "  [F] No post-hoc Lambda bias correction"
    )

    print()
    print(
        "Interpretation:"
    )

    print(
        "  A successful calibration requires "
        "systematic recovery of injected Lambda."
    )

    print(
        "  Lambda=0 should provide the "
        "GR calibration/control distribution."
    )

    print(
        "  This stage does NOT establish a "
        "physical non-GR detection."
    )

    print()
    print(
        f"Recovery CSV   = "
        f"{recovery_file}"
    )

    print(
        f"Summary CSV    = "
        f"{summary_file}"
    )

    print(
        f"Diagnostic CSV = "
        f"{diagnostic_file}"
    )

    print()
    print("=" * 80)
    print(
        "STAGE 6E3H-R v2 COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":

    main()
