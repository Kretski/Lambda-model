#!/usr/bin/env python3

import argparse
import csv
import os
import sys

import numpy as np

from pycbc.waveform import get_td_waveform
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.types import TimeSeries
from pycbc.filter import matched_filter


# ============================================================
# STAGE 6E2P
# BLIND LAMBDA INJECTION / RECOVERY
#
# IMPORTANT:
#   Lambda_true is NEVER passed to the estimator.
#
#   It is used only by:
#       1. the synthetic signal generator
#       2. the final post-hoc comparison
#
#   The estimator receives:
#       data
#       PSD
#       candidate Lambda grid
#
#   No real detector data.
#   No physical Lambda constraint.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

FS = 4096.0
DURATION = 16.0

N = int(FS * DURATION)

DT = 1.0 / FS
DF = 1.0 / DURATION

F_LOW = 20.0
F_HIGH = 300.0

M1 = 36.0
M2 = 29.0

DISTANCE = 440.0

INJECTION_TIME = 8.0

TARGET_SNRS = [
    10.0,
    20.0,
]

# ------------------------------------------------------------
# Blind truth set.
#
# IMPORTANT:
# This list is used ONLY by the synthetic injection generator.
# The estimator never sees which value was selected.
# ------------------------------------------------------------

BLIND_LAMBDA_VALUES = [
    -4.0,
    -2.78,
    -1.0,
    0.0,
    1.0,
    2.0,
]

# ------------------------------------------------------------
# Grid convergence
# ------------------------------------------------------------

GRID_STEPS = [
    0.020,
    0.010,
    0.005,
]

GRID_MIN = -6.0
GRID_MAX = 2.0

# ------------------------------------------------------------

BASE_SEED = 20260830

OUTPUT_DIR = "stage6E2P_results"


# ============================================================
# GRID
# ============================================================

def build_lambda_grid(step):
    """
    Construct a deterministic Lambda grid.

    The grid is supplied to the estimator.
    It contains no information about the hidden Lambda_true.
    """

    n = int(round(
        (GRID_MAX - GRID_MIN) / step
    ))

    grid = (
        GRID_MIN
        +
        np.arange(n + 1, dtype=np.float64) * step
    )

    return grid


# ============================================================
# WAVEFORM
# ============================================================

def generate_gr_template():
    """
    Generate the GR reference waveform.

    Lambda=0 corresponds to the GR reference waveform.
    """

    hp, _ = get_td_waveform(
        approximant="IMRPhenomD",
        mass1=M1,
        mass2=M2,
        delta_t=DT,
        f_lower=F_LOW,
        f_final=F_HIGH,
    )

    x = np.asarray(
        hp,
        dtype=np.float64
    )

    if len(x) > N:
        x = x[-N:]

    elif len(x) < N:

        y = np.zeros(
            N,
            dtype=np.float64
        )

        y[:len(x)] = x

        x = y

    return TimeSeries(
        x,
        delta_t=DT,
        epoch=0
    )


# ============================================================
# TEMPLATE PLACEMENT
# ============================================================

def place_template_at_time(
    template,
    requested_time
):

    x = np.asarray(
        template,
        dtype=np.float64
    )

    placement_index = int(
        round(
            requested_time * FS
        )
    )

    peak_index = int(
        np.argmax(
            np.abs(x)
        )
    )

    shift = (
        placement_index
        -
        peak_index
    )

    output = np.zeros(
        N,
        dtype=np.float64
    )

    if shift >= 0:

        src_start = 0

        src_end = min(
            N,
            N - shift
        )

        if src_end > src_start:

            length = (
                src_end
                -
                src_start
            )

            output[
                shift:
                shift + length
            ] = x[
                src_start:
                src_end
            ]

    else:

        src_start = -shift

        src_end = N

        if src_start < src_end:

            length = (
                src_end
                -
                src_start
            )

            output[
                0:
                length
            ] = x[
                src_start:
                src_end
            ]

    return TimeSeries(
        output,
        delta_t=DT,
        epoch=0
    )


# ============================================================
# LAMBDA PHASE MODEL
# ============================================================

def lambda_phase_factor(
    frequencies,
    lambda_value
):
    """
    Simple deterministic dispersive phase model.

    The phase perturbation is proportional to:

        Lambda * f^2

    This is used only to create a controlled synthetic
    injection/recovery problem.

    It is NOT claimed to be a complete astrophysical
    propagation model.
    """

    f = np.asarray(
        frequencies,
        dtype=np.float64
    )

    f_ref = 100.0

    normalized = (
        f / f_ref
    )

    phase = (
        lambda_value
        *
        0.015
        *
        normalized ** 2
    )

    return phase


# ============================================================
# BUILD LAMBDA WAVEFORM
# ============================================================

def build_lambda_waveform(
    gr_template,
    lambda_value
):
    """
    Construct the synthetic waveform corresponding to a
    requested Lambda value.

    This function is part of the injection generator.

    The resulting waveform is returned to the caller as data.

    The Lambda estimator does NOT call this function with its
    hidden truth value.
    """

    x = np.asarray(
        gr_template,
        dtype=np.float64
    )

    freq = np.fft.rfftfreq(
        N,
        d=DT
    )

    spectrum = np.fft.rfft(
        x
    )

    phase = lambda_phase_factor(
        freq,
        lambda_value
    )

    mask = (
        (freq >= F_LOW)
        &
        (freq <= F_HIGH)
    )

    transfer = np.ones_like(
        spectrum,
        dtype=np.complex128
    )

    transfer[mask] = np.exp(
        1j * phase[mask]
    )

    modified = np.fft.irfft(
        spectrum * transfer,
        n=N
    )

    return TimeSeries(
        modified.astype(
            np.float64
        ),
        delta_t=DT,
        epoch=0
    )


# ============================================================
# PSD
# ============================================================

def build_psd():

    return aLIGOZeroDetHighPower(
        N // 2 + 1,
        DF,
        F_LOW
    )


# ============================================================
# MATCHED FILTER
# ============================================================

def matched_filter_value(
    data,
    template,
    psd
):

    result = matched_filter(
        template,
        data,
        psd=psd,
        low_frequency_cutoff=F_LOW,
        high_frequency_cutoff=F_HIGH
    )

    arr = np.asarray(
        result
    )

    idx = int(
        np.argmax(
            np.abs(arr)
        )
    )

    value = float(
        np.abs(
            arr[idx]
        )
    )

    return value, idx


# ============================================================
# SNR NORMALIZATION
# ============================================================

def calculate_self_match(
    template,
    psd
):

    peak, idx = matched_filter_value(
        template,
        template,
        psd
    )

    return peak, idx


# ============================================================
# PSD-CONSISTENT NOISE
# ============================================================

def generate_psd_noise(
    psd,
    seed
):

    rng = np.random.default_rng(
        seed
    )

    psd_array = np.asarray(
        psd,
        dtype=np.float64
    )

    nfreq = len(
        psd_array
    )

    fd = np.zeros(
        nfreq,
        dtype=np.complex128
    )

    if nfreq > 2:

        variance = (
            np.maximum(
                psd_array[1:-1],
                0.0
            )
            *
            FS
            /
            2.0
        )

        sigma = np.sqrt(
            variance
        )

        fd[1:-1] = (
            rng.normal(
                size=nfreq - 2
            )
            +
            1j
            *
            rng.normal(
                size=nfreq - 2
            )
        ) * sigma

    fd[0] = 0.0

    if N % 2 == 0:

        fd[-1] = (
            rng.normal()
            *
            np.sqrt(
                max(
                    psd_array[-1],
                    0.0
                )
                *
                FS
                /
                2.0
            )
        )

    noise = np.fft.irfft(
        fd,
        n=N
    )

    return TimeSeries(
        noise.astype(
            np.float64
        ),
        delta_t=DT,
        epoch=0
    )


# ============================================================
# SIGNAL SNR CALIBRATION
# ============================================================

def calibrate_signal_amplitude(
    signal,
    template,
    psd,
    target_snr
):

    peak, _ = matched_filter_value(
        signal,
        template,
        psd
    )

    if peak <= 0:
        raise RuntimeError(
            "Invalid signal norm during SNR calibration."
        )

    return (
        target_snr
        /
        peak
    )


# ============================================================
# LOCALIZED TIME
# ============================================================

def localized_peak(
    data,
    template,
    psd,
    expected_index,
    half_window
):

    result = matched_filter(
        template,
        data,
        psd=psd,
        low_frequency_cutoff=F_LOW,
        high_frequency_cutoff=F_HIGH
    )

    arr = np.asarray(
        result
    )

    lo = max(
        0,
        expected_index - half_window
    )

    hi = min(
        len(arr),
        expected_index + half_window + 1
    )

    local = np.abs(
        arr[lo:hi]
    )

    local_idx = int(
        np.argmax(local)
    )

    absolute_idx = (
        lo
        +
        local_idx
    )

    return (
        float(local[local_idx]),
        absolute_idx
    )


# ============================================================
# CYCLIC TIME OFFSET
# ============================================================

def calibrate_time_offset(
    template,
    psd
):

    reference = place_template_at_time(
        template,
        INJECTION_TIME
    )

    _, mf_index = matched_filter_value(
        reference,
        template,
        psd
    )

    mf_time = (
        mf_index * DT
    )

    offset = (
        mf_time
        -
        INJECTION_TIME
    ) % DURATION

    return offset


def recover_physical_time(
    mf_index,
    offset
):

    mf_time = (
        mf_index * DT
    )

    return (
        mf_time
        -
        offset
    ) % DURATION


# ============================================================
# BLIND LIKELIHOOD / ESTIMATOR
# ============================================================

def estimate_lambda_blind(
    data,
    gr_template,
    psd,
    lambda_grid
):
    """
    ========================================================
    BLIND ESTIMATOR
    ========================================================

    CRITICAL:

    There is NO lambda_true argument.

    The estimator has access only to:

        data
        GR reference waveform
        PSD
        candidate Lambda grid

    It independently constructs candidate templates for
    every grid point and evaluates the matched-filter score.

    The hidden injection truth is unavailable here.
    ========================================================
    """

    best_lambda = None
    best_score = -np.inf

    scores = []

    for candidate_lambda in lambda_grid:

        candidate_waveform = build_lambda_waveform(
            gr_template,
            candidate_lambda
        )

        candidate = place_template_at_time(
            candidate_waveform,
            INJECTION_TIME
        )

        value, _ = matched_filter_value(
            data,
            candidate,
            psd
        )

        score = (
            value * value
        )

        scores.append(
            score
        )

        if score > best_score:

            best_score = score

            best_lambda = float(
                candidate_lambda
            )

    return (
        best_lambda,
        float(best_score),
        np.asarray(
            scores,
            dtype=np.float64
        )
    )


# ============================================================
# ONE BLIND REALIZATION
# ============================================================

def run_blind_realization(
    hidden_lambda,
    target_snr,
    realization,
    gr_template,
    psd,
    self_peak,
    time_offset,
    grid
):
    """
    ========================================================
    DATA GENERATION SIDE
    ========================================================

    hidden_lambda is allowed here because this function
    generates the synthetic injection.

    After the synthetic data are created, the estimator is
    called without hidden_lambda.
    ========================================================
    """

    # --------------------------------------------------------
    # 1. Generate hidden-Lambda waveform
    # --------------------------------------------------------

    lambda_waveform = build_lambda_waveform(
        gr_template,
        hidden_lambda
    )

    placed_signal = place_template_at_time(
        lambda_waveform,
        INJECTION_TIME
    )

    # --------------------------------------------------------
    # 2. Calibrate amplitude to requested SNR.
    #
    # IMPORTANT:
    # This is signal generation, NOT Lambda estimation.
    # --------------------------------------------------------

    amplitude = calibrate_signal_amplitude(
        placed_signal,
        gr_template,
        psd,
        target_snr
    )

    signal = (
        placed_signal
        *
        amplitude
    )

    # --------------------------------------------------------
    # 3. Generate independent PSD noise
    # --------------------------------------------------------

    seed = (
        BASE_SEED
        +
        realization * 100000
        +
        int(
            target_snr * 100
        )
        +
        int(
            (hidden_lambda + 10.0)
            * 1000
        )
    )

    noise = generate_psd_noise(
        psd,
        seed
    )

    data = (
        signal
        +
        noise
    )

    # --------------------------------------------------------
    # 4. BLIND ESTIMATION
    #
    # NO hidden_lambda passed here.
    # --------------------------------------------------------

    recovered_lambda, best_score, scores = (
        estimate_lambda_blind(
            data=data,
            gr_template=gr_template,
            psd=psd,
            lambda_grid=grid
        )
    )

    # --------------------------------------------------------
    # 5. Recover time independently
    # --------------------------------------------------------

    expected_mf_index = int(
        round(
            (
                (
                    INJECTION_TIME
                    +
                    time_offset
                )
                %
                DURATION
            )
            / DT
        )
    )

    # Use the GR template for the localized timing diagnostic.
    recovered_snr, mf_index = localized_peak(
        data,
        gr_template,
        psd,
        expected_mf_index,
        int(
            round(
                0.250 * FS
            )
        )
    )

    recovered_time = recover_physical_time(
        mf_index,
        time_offset
    )

    timing_error = (
        recovered_time
        -
        INJECTION_TIME
    )

    timing_error = (
        (
            timing_error
            +
            DURATION / 2.0
        )
        %
        DURATION
    ) - DURATION / 2.0

    # --------------------------------------------------------
    # 6. FINAL comparison.
    #
    # This is the FIRST place where hidden_lambda is used
    # after estimation.
    # --------------------------------------------------------

    lambda_error = (
        recovered_lambda
        -
        hidden_lambda
    )

    return {
        "lambda_true": hidden_lambda,
        "target_snr": target_snr,
        "realization": realization,
        "seed": seed,
        "amplitude": amplitude,
        "lambda_recovered": recovered_lambda,
        "lambda_error": lambda_error,
        "best_score": best_score,
        "recovered_snr": recovered_snr,
        "mf_index": mf_index,
        "recovered_time": recovered_time,
        "timing_error": timing_error,
    }


# ============================================================
# GRID CONVERGENCE
# ============================================================

def summarize_grid_results(
    grid_results
):

    grouped = {}

    for row in grid_results:

        key = (
            row["lambda_true"],
            row["target_snr"],
            row["grid_step"]
        )

        grouped.setdefault(
            key,
            []
        ).append(
            row["lambda_recovered"]
        )

    summary = []

    for key, values in grouped.items():

        lambda_true, snr, grid_step = key

        values = np.asarray(
            values,
            dtype=np.float64
        )

        mean_recovered = float(
            np.mean(values)
        )

        std_recovered = float(
            np.std(values)
        )

        bias = (
            mean_recovered
            -
            lambda_true
        )

        summary.append(
            {
                "lambda_true": lambda_true,
                "target_snr": snr,
                "grid_step": grid_step,
                "mean_recovered": mean_recovered,
                "std_recovered": std_recovered,
                "bias": bias,
            }
        )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Stage 6E2P blind Lambda "
            "injection/recovery campaign"
        )
    )

    parser.add_argument(
        "--n-realizations",
        type=int,
        default=50
    )

    args = parser.parse_args()

    if args.n_realizations < 1:

        raise ValueError(
            "--n-realizations must be >= 1"
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 80)
    print(
        "STAGE 6E2P — BLIND LAMBDA "
        "INJECTION / RECOVERY"
    )
    print("=" * 80)

    print(
        f"FS                  = {FS} Hz"
    )

    print(
        f"DURATION            = {DURATION} s"
    )

    print(
        f"N                   = {N}"
    )

    print(
        f"DF                  = {DF:.8f} Hz"
    )

    print(
        f"BAND                = "
        f"{F_LOW} - {F_HIGH} Hz"
    )

    print(
        f"M1                  = {M1} Msun"
    )

    print(
        f"M2                  = {M2} Msun"
    )

    print(
        f"DISTANCE            = "
        f"{DISTANCE:.1f} Mpc"
    )

    print(
        f"Injection time      = "
        f"{INJECTION_TIME:.6f} s"
    )

    print(
        "Hidden Lambda set   = "
        f"{BLIND_LAMBDA_VALUES}"
    )

    print(
        "SNR values          = "
        f"{TARGET_SNRS}"
    )

    print(
        "Grid steps          = "
        f"{GRID_STEPS}"
    )

    print(
        f"Realizations        = "
        f"{args.n_realizations}"
    )

    print()

    print(
        "Lambda estimator    = BLIND INDEPENDENT GRID SEARCH"
    )

    print(
        "Detector data       = NONE"
    )

    print(
        "Noise               = PSD-consistent Gaussian"
    )

    print()

    print(
        "BLINDNESS CONTRACT:"
    )

    print(
        "  Lambda_true -> injection generator ONLY"
    )

    print(
        "  Lambda_true -> estimator NEVER"
    )

    print(
        "  Lambda_true -> final comparison ONLY"
    )

    print()

    # ========================================================
    # TEMPLATE
    # ========================================================

    print(
        "[1] GENERATING GR REFERENCE TEMPLATE"
    )

    gr_template = generate_gr_template()

    print(
        f"template length     = "
        f"{len(gr_template)}"
    )

    peak_idx = int(
        np.argmax(
            np.abs(
                np.asarray(
                    gr_template
                )
            )
        )
    )

    print(
        f"template peak idx   = "
        f"{peak_idx}"
    )

    print(
        f"template peak time  = "
        f"{peak_idx * DT:.12f} s"
    )

    print()

    # ========================================================
    # PSD
    # ========================================================

    print(
        "[2] BUILDING PSD"
    )

    psd = build_psd()

    print(
        f"PSD bins            = "
        f"{len(psd)}"
    )

    print(
        f"PSD delta_f         = "
        f"{psd.delta_f}"
    )

    print()

    # ========================================================
    # SELF MATCH
    # ========================================================

    print(
        "[3] GR SELF MATCH"
    )

    self_peak, self_idx = calculate_self_match(
        gr_template,
        psd
    )

    print(
        f"self-match peak     = "
        f"{self_peak:.12e}"
    )

    print(
        f"self-match index    = "
        f"{self_idx}"
    )

    print()

    # ========================================================
    # TIME OFFSET
    # ========================================================

    print(
        "[4] CYCLIC TIME CALIBRATION"
    )

    time_offset = calibrate_time_offset(
        gr_template,
        psd
    )

    print(
        f"cyclic offset       = "
        f"{time_offset:.12f} s"
    )

    ref_time = recover_physical_time(
        int(
            round(
                (
                    INJECTION_TIME
                    +
                    time_offset
                )
                % DURATION
                / DT
            )
        ),
        time_offset
    )

    print(
        f"reference recovery  = "
        f"{ref_time:.12f} s"
    )

    print()

    # ========================================================
    # GRID INFORMATION
    # ========================================================

    print(
        "[5] BUILDING BLIND GRIDS"
    )

    grids = {}

    for step in GRID_STEPS:

        grid = build_lambda_grid(
            step
        )

        grids[step] = grid

        print(
            f"grid step={step:.5f}  "
            f"points={len(grid):5d}  "
            f"range=[{grid[0]:.3f}, "
            f"{grid[-1]:.3f}]"
        )

    print()

    # ========================================================
    # BLIND CAMPAIGN
    # ========================================================

    print(
        "[6] BLIND INJECTION / RECOVERY"
    )

    print()

    print(
        "The estimator is now isolated from Lambda_true."
    )

    print()

    all_rows = []
    grid_rows = []

    # --------------------------------------------------------
    # The hidden truth is used here ONLY to construct data.
    # --------------------------------------------------------

    for hidden_lambda in BLIND_LAMBDA_VALUES:

        for target_snr in TARGET_SNRS:

            print("-" * 80)

            print(
                f"HIDDEN INJECTION = "
                f"{hidden_lambda:+.3f}"
            )

            print(
                f"TARGET SNR       = "
                f"{target_snr:.1f}"
            )

            print(
                "Estimator input does NOT include hidden Lambda."
            )

            print()

            for realization in range(
                1,
                args.n_realizations + 1
            ):

                # ------------------------------------------------
                # Run the same blind experiment on every grid.
                # ------------------------------------------------

                for grid_step in GRID_STEPS:

                    result = run_blind_realization(
                        hidden_lambda=hidden_lambda,
                        target_snr=target_snr,
                        realization=realization,
                        gr_template=gr_template,
                        psd=psd,
                        self_peak=self_peak,
                        time_offset=time_offset,
                        grid=grids[grid_step]
                    )

                    row = {
                        "lambda_true": result[
                            "lambda_true"
                        ],
                        "target_snr": result[
                            "target_snr"
                        ],
                        "realization": result[
                            "realization"
                        ],
                        "grid_step": grid_step,
                        "seed": result[
                            "seed"
                        ],
                        "amplitude": result[
                            "amplitude"
                        ],
                        "lambda_recovered": result[
                            "lambda_recovered"
                        ],
                        "lambda_error": result[
                            "lambda_error"
                        ],
                        "best_score": result[
                            "best_score"
                        ],
                        "recovered_snr": result[
                            "recovered_snr"
                        ],
                        "mf_index": result[
                            "mf_index"
                        ],
                        "recovered_time": result[
                            "recovered_time"
                        ],
                        "timing_error": result[
                            "timing_error"
                        ],
                    }

                    grid_rows.append(
                        row
                    )

                    if grid_step == GRID_STEPS[0]:

                        all_rows.append(
                            row
                        )

                # ------------------------------------------------
                # Print only primary-grid result.
                # ------------------------------------------------

                primary = [
                    x for x in grid_rows
                    if (
                        x["lambda_true"]
                        == hidden_lambda
                        and
                        x["target_snr"]
                        == target_snr
                        and
                        x["realization"]
                        == realization
                        and
                        x["grid_step"]
                        == GRID_STEPS[0]
                    )
                ][-1]

                print(
                    f"realization={realization:3d}  "
                    f"recovered="
                    f"{primary['lambda_recovered']:+.5f}  "
                    f"error="
                    f"{primary['lambda_error']:+.5f}  "
                    f"SNR="
                    f"{primary['recovered_snr']:.4f}  "
                    f"time="
                    f"{primary['recovered_time']:.6f}s"
                )

            print()

    # ========================================================
    # PRIMARY SUMMARY
    # ========================================================

    print("=" * 80)

    print(
        "STAGE 6E2P PRIMARY BLIND SUMMARY"
    )

    print("=" * 80)

    print()

    print(
        "Primary grid step = "
        f"{GRID_STEPS[0]:.5f}"
    )

    print()

    print(
        "Lambda_true | SNR | recovered mean ± std | "
        "bias | timing RMS"
    )

    print("-" * 80)

    primary_pass = True

    for hidden_lambda in BLIND_LAMBDA_VALUES:

        for target_snr in TARGET_SNRS:

            rows = [
                r for r in all_rows
                if (
                    r["lambda_true"]
                    == hidden_lambda
                    and
                    r["target_snr"]
                    == target_snr
                )
            ]

            recovered = np.asarray(
                [
                    r["lambda_recovered"]
                    for r in rows
                ],
                dtype=np.float64
            )

            timing = np.asarray(
                [
                    r["timing_error"]
                    for r in rows
                ],
                dtype=np.float64
            )

            mean_rec = float(
                np.mean(recovered)
            )

            std_rec = float(
                np.std(recovered)
            )

            bias = (
                mean_rec
                -
                hidden_lambda
            )

            timing_rms = float(
                np.sqrt(
                    np.mean(
                        timing ** 2
                    )
                )
            )

            print(
                f"{hidden_lambda:+10.3f} | "
                f"{target_snr:3.0f} | "
                f"{mean_rec:+10.5f} ± "
                f"{std_rec:.5f} | "
                f"{bias:+.5f} | "
                f"{timing_rms * 1000:.5f} ms"
            )

            # ----------------------------------------------------
            # Controlled numerical acceptance.
            #
            # This is NOT a physical confidence interval.
            # ----------------------------------------------------

            if abs(bias) > 0.20:

                primary_pass = False

            if timing_rms > (
                2.0 * DT
            ):

                primary_pass = False

    # ========================================================
    # GRID CONVERGENCE
    # ========================================================

    print()

    print("=" * 80)

    print(
        "GRID CONVERGENCE TEST"
    )

    print("=" * 80)

    print()

    print(
        "Lambda_true | SNR | grid | recovered mean | "
        "std | bias"
    )

    print("-" * 80)

    summary = summarize_grid_results(
        grid_rows
    )

    convergence_pass = True

    for item in sorted(
        summary,
        key=lambda x: (
            x["lambda_true"],
            x["target_snr"],
            x["grid_step"]
        )
    ):

        print(
            f"{item['lambda_true']:+10.3f} | "
            f"{item['target_snr']:3.0f} | "
            f"{item['grid_step']:.5f} | "
            f"{item['mean_recovered']:+14.6f} | "
            f"{item['std_recovered']:.6f} | "
            f"{item['bias']:+.6f}"
        )

    print()

    # --------------------------------------------------------
    # Compare successive grid refinements.
    # --------------------------------------------------------

    for hidden_lambda in BLIND_LAMBDA_VALUES:

        for target_snr in TARGET_SNRS:

            values = {}

            for item in summary:

                if (
                    item["lambda_true"]
                    == hidden_lambda
                    and
                    item["target_snr"]
                    == target_snr
                ):

                    values[
                        item["grid_step"]
                    ] = item[
                        "mean_recovered"
                    ]

            coarse = values.get(
                0.020
            )

            medium = values.get(
                0.010
            )

            fine = values.get(
                0.005
            )

            if (
                coarse is not None
                and
                medium is not None
                and
                fine is not None
            ):

                coarse_medium = abs(
                    medium
                    -
                    coarse
                )

                medium_fine = abs(
                    fine
                    -
                    medium
                )

                print(
                    f"convergence "
                    f"Lambda={hidden_lambda:+.3f} "
                    f"SNR={target_snr:.0f}: "
                    f"|0.020-0.010|="
                    f"{coarse_medium:.6f}, "
                    f"|0.010-0.005|="
                    f"{medium_fine:.6f}"
                )

                # ------------------------------------------------
                # Numerical convergence criterion.
                # ------------------------------------------------

                if (
                    medium_fine
                    >
                    max(
                        0.05,
                        coarse_medium * 1.5
                    )
                ):

                    convergence_pass = False

    # ========================================================
    # SAVE PRIMARY CSV
    # ========================================================

    primary_csv = os.path.join(
        OUTPUT_DIR,
        "stage6E2P_primary_blind_results.csv"
    )

    with open(
        primary_csv,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lambda_true",
                "target_snr",
                "realization",
                "grid_step",
                "seed",
                "amplitude",
                "lambda_recovered",
                "lambda_error",
                "best_score",
                "recovered_snr",
                "mf_index",
                "recovered_time",
                "timing_error",
            ]
        )

        writer.writeheader()

        for row in all_rows:

            writer.writerow(
                row
            )

    # ========================================================
    # SAVE FULL GRID CSV
    # ========================================================

    grid_csv = os.path.join(
        OUTPUT_DIR,
        "stage6E2P_grid_convergence.csv"
    )

    with open(
        grid_csv,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lambda_true",
                "target_snr",
                "realization",
                "grid_step",
                "seed",
                "amplitude",
                "lambda_recovered",
                "lambda_error",
                "best_score",
                "recovered_snr",
                "mf_index",
                "recovered_time",
                "timing_error",
            ]
        )

        writer.writeheader()

        for row in grid_rows:

            writer.writerow(
                row
            )

    # ========================================================
    # FINAL
    # ========================================================

    print()

    print("=" * 80)

    print(
        "STAGE 6E2P FINAL VALIDATION"
    )

    print("=" * 80)

    print()

    print(
        "Blind estimator          : ENABLED"
    )

    print(
        "Lambda_true to estimator : NEVER"
    )

    print(
        "Synthetic detector data  : NONE"
    )

    print(
        "PSD noise                : YES"
    )

    print(
        "Positive Lambda tests    : YES"
    )

    print(
        "Negative Lambda tests    : YES"
    )

    print(
        "Grid convergence         : "
        +
        (
            "PASS"
            if convergence_pass
            else
            "FAIL"
        )
    )

    print(
        "Injection/recovery       : "
        +
        (
            "PASS"
            if primary_pass
            else
            "FAIL"
        )
    )

    print()

    if (
        primary_pass
        and
        convergence_pass
    ):

        print(
            "RESULT: PASS"
        )

        print()

        print(
            "Blind Lambda injection/"
            "recovery is numerically consistent."
        )

        print(
            "The estimator independently recovers "
            "hidden positive and negative Lambda values."
        )

        print(
            "Recovery was tested across multiple "
            "Lambda-grid resolutions."
        )

    else:

        print(
            "RESULT: FAIL"
        )

        print()

        print(
            "The controlled blind campaign requires "
            "additional numerical investigation."
        )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This is a synthetic controlled test."
    )

    print(
        "It is NOT a measurement using real "
        "gravitational-wave detector data."
    )

    print(
        "It does NOT establish a physical Lambda value."
    )

    print(
        "It does NOT provide a physical Lambda constraint."
    )

    print()

    print(
        f"Primary CSV = {primary_csv}"
    )

    print(
        f"Grid CSV    = {grid_csv}"
    )

    print()

    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()