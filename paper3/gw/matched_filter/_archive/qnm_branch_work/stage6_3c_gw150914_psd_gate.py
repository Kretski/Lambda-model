#!/usr/bin/env python3

"""
Stage 6.3c — GW150914 low-frequency PSD gate

Purpose
-------
Technical data-quality gate before any matched-filter test
of the Root-B QNM.

Root-B frequency:
    approximately 9.6 - 9.8 Hz

This stage ONLY checks whether the selected GWOSC data product
contains usable low-frequency information near 9.7 Hz.

It does NOT:
    - test the Lambda model
    - fit a Lambda template
    - perform matched filtering
    - claim an astrophysical detection
    - establish a Kerr mapping
    - reject the Lambda model

Input files:
    ./gw150914_data/H-H1_LOSC_4_V1-1126259446-32.hdf5
    ./gw150914_data/L-L1_LOSC_4_V1-1126259446-32.hdf5

Outputs:
    stage6_3c_gw150914_psd_gate.csv
    stage6_3c_gw150914_rootB_psd.csv
    stage6_3c_gw150914_psd_gate.png
    stage6_3c_gw150914_psd_low_frequency.png
    stage6_3c_gw150914_psd_gate.json
"""

import os
import json

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import welch


# ============================================================
# CONFIGURATION
# ============================================================

# IMPORTANT:
# The GW150914 files are in the gw150914_data subdirectory.
FILES = {
    "H1": os.path.join(
        "gw150914_data",
        "H-H1_LOSC_4_V1-1126259446-32.hdf5"
    ),
    "L1": os.path.join(
        "gw150914_data",
        "L-L1_LOSC_4_V1-1126259446-32.hdf5"
    ),
}


# Validated Root-B frequency range from Stage 5K
ROOT_B_FREQS = np.array([
    9.596712868,
    9.647287000,
    9.697698000,
    9.747953000,
    9.798060000,
    9.824309473,
])


# Frequencies used for technical comparison
REFERENCE_FREQS = np.array([
    9.7,
    20.0,
    30.0,
    50.0,
    100.0,
    200.0,
])


# Use central part of the 32-second file.
# This avoids file edges.
ANALYSIS_START = 4.0
ANALYSIS_DURATION = 24.0


# Welch PSD parameters
PSD_SEGMENT_SECONDS = 4.0
OVERLAP = 0.5


# Plot range
FMIN = 2.0
FMAX = 500.0


# ============================================================
# LOAD GWOSC HDF5
# ============================================================

def load_gwosc_strain(filename):
    """
    Load strain and metadata from a GWOSC HDF5 file.
    """

    if not os.path.exists(filename):

        raise FileNotFoundError(
            "\nFile not found:\n"
            f"    {os.path.abspath(filename)}\n\n"
            "Expected GW150914 data files under:\n"
            "    ./gw150914_data/\n"
        )


    with h5py.File(filename, "r") as f:

        if "strain" not in f:

            raise RuntimeError(
                f"{filename}: missing /strain group."
            )


        if "Strain" not in f["strain"]:

            raise RuntimeError(
                f"{filename}: missing /strain/Strain dataset."
            )


        dataset = f["strain"]["Strain"]

        attrs = dict(dataset.attrs)

        strain = np.asarray(
            dataset[:],
            dtype=np.float64
        )


        # ----------------------------------------------------
        # Determine sample rate
        # ----------------------------------------------------

        sample_rate = None


        if "Xspacing" in attrs:

            sample_rate = 1.0 / float(
                attrs["Xspacing"]
            )


        elif "sample_rate" in attrs:

            sample_rate = float(
                attrs["sample_rate"]
            )


        elif "SampleRate" in attrs:

            sample_rate = float(
                attrs["SampleRate"]
            )


        # Try metadata groups if necessary
        if sample_rate is None:

            possible_groups = [
                "meta",
                "meta/Observatory",
                "meta/Detector",
            ]


            for group_name in possible_groups:

                if group_name not in f:
                    continue


                group = f[group_name]


                for key in [
                    "SampleRate",
                    "sample_rate",
                    "SamplingRate",
                ]:

                    if key in group.attrs:

                        sample_rate = float(
                            group.attrs[key]
                        )

                        break


                if sample_rate is not None:
                    break


        if sample_rate is None:

            raise RuntimeError(
                f"{filename}: "
                "could not determine sample rate."
            )


        # ----------------------------------------------------
        # GPS start
        # ----------------------------------------------------

        gps_start = None


        for key in [
            "Xstart",
            "XStart",
            "GPSstart",
            "gps_start",
        ]:

            if key in attrs:

                gps_start = float(
                    attrs[key]
                )

                break


    return (
        strain,
        sample_rate,
        gps_start,
        attrs,
    )


# ============================================================
# PSD / ASD
# ============================================================

def compute_psd(strain, fs):
    """
    Compute Welch PSD and ASD.
    """

    nperseg = int(
        PSD_SEGMENT_SECONDS * fs
    )


    if nperseg > len(strain):

        nperseg = len(strain)


    noverlap = int(
        nperseg * OVERLAP
    )


    if noverlap >= nperseg:

        noverlap = nperseg // 2


    freqs, psd = welch(
        strain,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )


    psd = np.maximum(
        psd,
        0.0
    )


    asd = np.sqrt(psd)


    return (
        freqs,
        psd,
        asd
    )


# ============================================================
# NEAREST FREQUENCY
# ============================================================

def nearest_value(freqs, values, target):
    """
    Return the value and actual frequency bin nearest to target.
    """

    idx = np.argmin(
        np.abs(
            freqs - target
        )
    )


    return (
        float(values[idx]),
        float(freqs[idx])
    )


# ============================================================
# DETECTOR ANALYSIS
# ============================================================

def analyze_detector(detector, filename):

    print()
    print("=" * 72)
    print(f"DETECTOR: {detector}")
    print("=" * 72)


    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        strain,
        fs,
        gps_start,
        attrs
    ) = load_gwosc_strain(
        filename
    )


    duration = (
        len(strain) / fs
    )


    print(
        f"File:        {filename}"
    )

    print(
        f"Samples:     {len(strain):,}"
    )

    print(
        f"Sample rate: {fs:.6f} Hz"
    )

    print(
        f"Duration:    {duration:.3f} s"
    )


    if gps_start is not None:

        print(
            f"GPS start:   {gps_start:.3f}"
        )


    # --------------------------------------------------------
    # Analysis window
    # --------------------------------------------------------

    start_index = int(
        ANALYSIS_START * fs
    )


    stop_index = int(
        (
            ANALYSIS_START
            + ANALYSIS_DURATION
        ) * fs
    )


    stop_index = min(
        stop_index,
        len(strain)
    )


    if stop_index <= start_index:

        raise RuntimeError(
            f"{detector}: invalid analysis window."
        )


    x = strain[
        start_index:stop_index
    ]


    actual_start = (
        start_index / fs
    )

    actual_stop = (
        stop_index / fs
    )


    print(
        f"PSD interval: "
        f"{actual_start:.3f} - "
        f"{actual_stop:.3f} s"
    )


    # Remove DC
    x = (
        x
        - np.mean(x)
    )


    # --------------------------------------------------------
    # PSD
    # --------------------------------------------------------

    (
        freqs,
        psd,
        asd
    ) = compute_psd(
        x,
        fs
    )


    # --------------------------------------------------------
    # Reference frequencies
    # --------------------------------------------------------

    reference_rows = []


    for target in REFERENCE_FREQS:

        asd_value, actual_frequency = (
            nearest_value(
                freqs,
                asd,
                target
            )
        )


        psd_value, _ = (
            nearest_value(
                freqs,
                psd,
                target
            )
        )


        reference_rows.append({
            "detector": detector,
            "requested_frequency_hz": float(
                target
            ),
            "actual_frequency_hz": (
                actual_frequency
            ),
            "asd": asd_value,
            "psd": psd_value,
        })


    reference_df = pd.DataFrame(
        reference_rows
    )


    # --------------------------------------------------------
    # Root-B frequencies
    # --------------------------------------------------------

    root_rows = []


    for target in ROOT_B_FREQS:

        asd_value, actual_frequency = (
            nearest_value(
                freqs,
                asd,
                target
            )
        )


        psd_value, _ = (
            nearest_value(
                freqs,
                psd,
                target
            )
        )


        root_rows.append({
            "detector": detector,
            "rootB_frequency_hz": float(
                target
            ),
            "nearest_frequency_hz": (
                actual_frequency
            ),
            "asd": asd_value,
            "psd": psd_value,
        })


    root_df = pd.DataFrame(
        root_rows
    )


    # --------------------------------------------------------
    # 9.7 Hz and 100 Hz comparison
    # --------------------------------------------------------

    asd_9p7, actual_9p7 = (
        nearest_value(
            freqs,
            asd,
            9.7
        )
    )


    asd_100, actual_100 = (
        nearest_value(
            freqs,
            asd,
            100.0
        )
    )


    if asd_100 > 0:

        ratio = (
            asd_9p7
            / asd_100
        )

        ratio_db = (
            20.0
            * np.log10(ratio)
        )

    else:

        ratio = np.inf
        ratio_db = np.inf


    # --------------------------------------------------------
    # 5-15 Hz versus 90-110 Hz
    # --------------------------------------------------------

    low_mask = (
        (freqs >= 5.0)
        & (freqs <= 15.0)
    )


    reference_mask = (
        (freqs >= 90.0)
        & (freqs <= 110.0)
    )


    if np.any(low_mask):

        median_low = float(
            np.median(
                asd[low_mask]
            )
        )

    else:

        median_low = np.nan


    if np.any(reference_mask):

        median_reference = float(
            np.median(
                asd[reference_mask]
            )
        )

    else:

        median_reference = np.nan


    if (
        np.isfinite(median_low)
        and np.isfinite(median_reference)
        and median_reference > 0
    ):

        low_high_ratio = (
            median_low
            / median_reference
        )

    else:

        low_high_ratio = np.nan


    # --------------------------------------------------------
    # Simple cutoff diagnostic
    #
    # IMPORTANT:
    # This does NOT identify the official processing cutoff.
    # It only searches for a sharp spectral transition.
    # --------------------------------------------------------

    diagnostic_freqs = np.arange(
        5.0,
        30.0,
        0.5
    )


    diagnostic_asd = []


    for target in diagnostic_freqs:

        value, _ = (
            nearest_value(
                freqs,
                asd,
                target
            )
        )


        diagnostic_asd.append(
            value
        )


    diagnostic_asd = np.asarray(
        diagnostic_asd
    )


    possible_cutoff_hz = None


    positive = (
        diagnostic_asd
        > np.finfo(float).tiny
    )


    if np.sum(positive) >= 3:

        log_asd = np.log10(
            diagnostic_asd[
                positive
            ]
        )


        derivative = np.abs(
            np.diff(log_asd)
        )


        if len(derivative) > 0:

            idx = int(
                np.argmax(
                    derivative
                )
            )


            if derivative[idx] > 1.0:

                possible_cutoff_hz = float(
                    diagnostic_freqs[idx]
                )


    # --------------------------------------------------------
    # Technical classification
    #
    # NOT a physical detection threshold.
    # --------------------------------------------------------

    if ratio > 100.0:

        classification = (
            "VERY_HIGH_LOW_FREQUENCY_NOISE"
        )

    elif ratio > 10.0:

        classification = (
            "HIGH_LOW_FREQUENCY_NOISE"
        )

    elif ratio > 3.0:

        classification = (
            "ELEVATED_LOW_FREQUENCY_NOISE"
        )

    else:

        classification = (
            "NO_LARGE_ASD_PENALTY"
        )


    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print()
    print("ASD reference values:")
    print()


    for _, row in (
        reference_df.iterrows()
    ):

        print(
            f"  "
            f"{row['requested_frequency_hz']:8.1f} Hz"
            f"   ASD = "
            f"{row['asd']:.6e}"
        )


    print()

    print(
        f"ASD @ 9.7 Hz  = "
        f"{asd_9p7:.6e}"
    )

    print(
        f"ASD @ 100 Hz  = "
        f"{asd_100:.6e}"
    )

    print(
        f"9.7/100 ratio  = "
        f"{ratio:.6e}"
    )

    print(
        f"9.7/100 dB     = "
        f"{ratio_db:.2f} dB"
    )

    print(
        f"Median ASD 5-15 Hz = "
        f"{median_low:.6e}"
    )

    print(
        f"Median ASD 90-110 Hz = "
        f"{median_reference:.6e}"
    )

    print(
        f"Low/high band ratio = "
        f"{low_high_ratio:.6e}"
    )


    if possible_cutoff_hz is not None:

        print(
            "Possible sharp spectral transition "
            f"near {possible_cutoff_hz:.1f} Hz"
        )

    else:

        print(
            "No obvious sharp spectral transition "
            "detected by diagnostic."
        )


    print(
        f"Technical classification: "
        f"{classification}"
    )


    return {
        "detector": detector,
        "filename": filename,
        "sample_rate": float(fs),
        "gps_start": gps_start,
        "duration": float(duration),
        "freqs": freqs,
        "psd": psd,
        "asd": asd,
        "reference": reference_df,
        "rootB": root_df,
        "asd_9p7": float(asd_9p7),
        "asd_100": float(asd_100),
        "ratio_9p7_100": float(ratio),
        "ratio_db_9p7_100": float(
            ratio_db
        ),
        "median_asd_5_15": median_low,
        "median_asd_90_110": median_reference,
        "low_high_band_ratio": low_high_ratio,
        "possible_cutoff_hz": possible_cutoff_hz,
        "classification": classification,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print(
        "STAGE 6.3c — "
        "GW150914 LOW-FREQUENCY PSD GATE"
    )
    print("=" * 72)

    print()

    print(
        "Root-B target: approximately 9.6-9.8 Hz"
    )

    print()

    print(
        "This is a TECHNICAL DATA-QUALITY TEST."
    )

    print(
        "No physical interpretation is performed "
        "in this stage."
    )


    # --------------------------------------------------------
    # Analyze detectors
    # --------------------------------------------------------

    results = {}


    for detector, filename in FILES.items():

        results[detector] = (
            analyze_detector(
                detector,
                filename
            )
        )


    # ========================================================
    # REFERENCE CSV
    # ========================================================

    reference_rows = []


    for detector, result in (
        results.items()
    ):

        for _, row in (
            result["reference"].iterrows()
        ):

            reference_rows.append(
                row.to_dict()
            )


    reference_df = pd.DataFrame(
        reference_rows
    )


    reference_df.to_csv(
        "stage6_3c_gw150914_psd_gate.csv",
        index=False
    )


    # ========================================================
    # ROOT-B CSV
    # ========================================================

    root_rows = []


    for detector, result in (
        results.items()
    ):

        for _, row in (
            result["rootB"].iterrows()
        ):

            root_rows.append(
                row.to_dict()
            )


    root_df = pd.DataFrame(
        root_rows
    )


    root_df.to_csv(
        "stage6_3c_gw150914_rootB_psd.csv",
        index=False
    )


    # ========================================================
    # FULL ASD PLOT
    # ========================================================

    plt.figure(
        figsize=(11, 7)
    )


    for detector, result in (
        results.items()
    ):

        f = result["freqs"]
        a = result["asd"]


        mask = (
            (f >= FMIN)
            & (f <= FMAX)
            & (a > 0)
        )


        plt.loglog(
            f[mask],
            a[mask],
            label=detector
        )


    plt.axvline(
        9.7,
        linestyle="--",
        label="Root-B ~9.7 Hz"
    )


    plt.axvline(
        20.0,
        linestyle=":",
        label="20 Hz"
    )


    plt.xlabel(
        "Frequency [Hz]"
    )

    plt.ylabel(
        "ASD [strain / sqrt(Hz)]"
    )

    plt.title(
        "GW150914 H1/L1 ASD — 2–500 Hz"
    )


    plt.xlim(
        FMIN,
        FMAX
    )


    plt.grid(
        True,
        which="both",
        alpha=0.25
    )


    plt.legend()

    plt.tight_layout()


    plt.savefig(
        "stage6_3c_gw150914_psd_gate.png",
        dpi=180
    )


    plt.close()


    # ========================================================
    # LOW-FREQUENCY DETAIL PLOT
    # ========================================================

    plt.figure(
        figsize=(11, 7)
    )


    for detector, result in (
        results.items()
    ):

        f = result["freqs"]
        a = result["asd"]


        mask = (
            (f >= 2.0)
            & (f <= 40.0)
            & (a > 0)
        )


        plt.semilogy(
            f[mask],
            a[mask],
            label=detector
        )


    plt.axvspan(
        9.6,
        9.83,
        alpha=0.20,
        label="Root-B frequency range"
    )


    plt.axvline(
        20.0,
        linestyle=":",
        label="20 Hz"
    )


    plt.xlabel(
        "Frequency [Hz]"
    )

    plt.ylabel(
        "ASD [strain / sqrt(Hz)]"
    )

    plt.title(
        "GW150914 low-frequency ASD — 2–40 Hz"
    )


    plt.xlim(
        2.0,
        40.0
    )


    plt.grid(
        True,
        which="both",
        alpha=0.25
    )


    plt.legend()

    plt.tight_layout()


    plt.savefig(
        "stage6_3c_gw150914_psd_low_frequency.png",
        dpi=180
    )


    plt.close()


    # ========================================================
    # OVERALL TECHNICAL GATE
    # ========================================================

    ratios = [
        result["ratio_9p7_100"]
        for result in results.values()
    ]


    if all(
        ratio > 100.0
        for ratio in ratios
    ):

        overall_status = (
            "LOW_FREQUENCY_GATE_FAIL"
        )


    elif all(
        ratio > 10.0
        for ratio in ratios
    ):

        overall_status = (
            "LOW_FREQUENCY_GATE_WEAK"
        )


    else:

        overall_status = (
            "LOW_FREQUENCY_INFORMATION_PRESENT"
        )


    # ========================================================
    # JSON SUMMARY
    # ========================================================

    detector_summary = {}


    for detector, result in (
        results.items()
    ):

        detector_summary[detector] = {

            "filename": result[
                "filename"
            ],

            "sample_rate": result[
                "sample_rate"
            ],

            "duration_s": result[
                "duration"
            ],

            "asd_9p7": result[
                "asd_9p7"
            ],

            "asd_100": result[
                "asd_100"
            ],

            "ratio_9p7_to_100": result[
                "ratio_9p7_100"
            ],

            "ratio_db": result[
                "ratio_db_9p7_100"
            ],

            "median_asd_5_15": result[
                "median_asd_5_15"
            ],

            "median_asd_90_110": result[
                "median_asd_90_110"
            ],

            "low_high_band_ratio": result[
                "low_high_band_ratio"
            ],

            "possible_cutoff_hz": result[
                "possible_cutoff_hz"
            ],

            "classification": result[
                "classification"
            ],
        }


    summary = {

        "stage": "6.3c",

        "title": (
            "GW150914 low-frequency PSD gate"
        ),

        "purpose": (
            "Determine whether the selected "
            "GWOSC data product contains "
            "potentially usable information "
            "around the Root-B frequency."
        ),

        "rootB_frequency_min_hz": float(
            np.min(ROOT_B_FREQS)
        ),

        "rootB_frequency_max_hz": float(
            np.max(ROOT_B_FREQS)
        ),

        "analysis_start_s": (
            ANALYSIS_START
        ),

        "analysis_duration_s": (
            ANALYSIS_DURATION
        ),

        "detectors": detector_summary,

        "overall_technical_status": (
            overall_status
        ),

        "scientific_interpretation": (
            "No physical interpretation is "
            "performed in this stage. The result "
            "only determines whether the selected "
            "data product is technically suitable "
            "for a subsequent low-frequency "
            "matched-filter experiment."
        ),

        "matched_filter_run": False,

        "astrophysical_claim": False,
    }


    with open(
        "stage6_3c_gw150914_psd_gate.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )


    # ========================================================
    # FINAL CONSOLE OUTPUT
    # ========================================================

    print()
    print("=" * 72)
    print(
        "STAGE 6.3c FINAL RESULT"
    )
    print("=" * 72)


    print()

    print(
        f"Root-B range: "
        f"{ROOT_B_FREQS.min():.3f} - "
        f"{ROOT_B_FREQS.max():.3f} Hz"
    )


    for detector, result in (
        results.items()
    ):

        print()
        print(
            f"{detector}:"
        )


        print(
            f"  ASD @ 9.7 Hz    = "
            f"{result['asd_9p7']:.6e}"
        )


        print(
            f"  ASD @ 100 Hz    = "
            f"{result['asd_100']:.6e}"
        )


        print(
            f"  ASD ratio       = "
            f"{result['ratio_9p7_100']:.6e}"
        )


        print(
            f"  ratio [dB]      = "
            f"{result['ratio_db_9p7_100']:.2f} dB"
        )


        print(
            f"  classification  = "
            f"{result['classification']}"
        )


        if (
            result["possible_cutoff_hz"]
            is not None
        ):

            print(
                f"  possible cutoff = "
                f"{result['possible_cutoff_hz']:.1f} Hz"
            )


    print()

    print(
        "OVERALL TECHNICAL STATUS: "
        f"{overall_status}"
    )


    print()

    print("Files written:")

    print(
        "  stage6_3c_gw150914_psd_gate.csv"
    )

    print(
        "  stage6_3c_gw150914_rootB_psd.csv"
    )

    print(
        "  stage6_3c_gw150914_psd_gate.png"
    )

    print(
        "  stage6_3c_gw150914_psd_low_frequency.png"
    )

    print(
        "  stage6_3c_gw150914_psd_gate.json"
    )


    print()

    print(
        "NO MATCHED FILTER WAS RUN."
    )

    print(
        "NO ASTROPHYSICAL CONCLUSION WAS MADE."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()