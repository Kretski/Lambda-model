"""
STAGE 6.3b
GW150914 REAL-DATA RINGDOWN STABILITY TEST

Purpose
-------
Determine whether a generic damped ringdown component extracted
from real GW150914 data is stable against the choice of
ringdown start time.

IMPORTANT
---------
This script does NOT fit the Lambda-model.
This script does NOT impose Kerr parameters.
This script does NOT force the data toward Root-B.

It is an observational control experiment.

Model
-----
    h(t) =
        A * exp(-t/tau)
        * cos(2*pi*f*t + phi)
        + offset

Measured quantities
------------------
    f_R
    tau
    Q = pi*f_R*tau
    RMS
    R^2

The tau lower bound is deliberately much lower than the previous
3 ms artificial boundary.
"""

import os
import csv
import h5py

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.optimize import least_squares


# ============================================================
# CONFIGURATION
# ============================================================

EVENT_GPS = 1126259462.422

FS = 4096.0

DATA_DIR = "gw150914_data"

H1_FILE = os.path.join(
    DATA_DIR,
    "H-H1_LOSC_4_V1-1126259446-32.hdf5"
)

L1_FILE = os.path.join(
    DATA_DIR,
    "L-L1_LOSC_4_V1-1126259446-32.hdf5"
)


# ------------------------------------------------------------
# Ringdown start times
#
# milliseconds after merger / waveform peak
# ------------------------------------------------------------

START_MS = [
    3,
    5,
    7,
    10,
    15,
    20,
    25,
    30
]


# ------------------------------------------------------------
# Window lengths
#
# We keep a fixed 100 ms analysis window initially.
# Later we can repeat this with different durations.
# ------------------------------------------------------------

WINDOW_MS = 100


# ------------------------------------------------------------
# Bandpass
# ------------------------------------------------------------

LOW_FREQ = 20.0
HIGH_FREQ = 500.0


# ------------------------------------------------------------
# Fit bounds
#
# IMPORTANT:
# Previous run hit tau = 3 ms boundary.
#
# Here:
#     tau_min = 0.5 ms
#
# This is deliberately below the expected ~4 ms GW150914
# ringdown damping time.
# ------------------------------------------------------------

TAU_MIN = 0.0005
TAU_MAX = 0.100

F_MIN = 50.0
F_MAX = 500.0


# ============================================================
# LOAD GWOSC DATA
# ============================================================

def load_strain(filename):

    with h5py.File(filename, "r") as f:

        strain = np.array(
            f["strain/Strain"]
        )

        gps_start = float(
            f["meta/GPSstart"][()]
        )

        duration = float(
            f["meta/Duration"][()]
        )

    time = (
        gps_start
        + np.arange(len(strain)) / FS
    )

    return strain, time


# ============================================================
# PREPROCESS
# ============================================================

def preprocess(strain):

    x = np.asarray(
        strain,
        dtype=float
    )

    # Remove linear trend
    x = signal.detrend(x)

    # Bandpass
    sos = signal.butter(
        4,
        [LOW_FREQ, HIGH_FREQ],
        btype="bandpass",
        fs=FS,
        output="sos"
    )

    x = signal.sosfiltfilt(
        sos,
        x
    )

    # --------------------------------------------------------
    # Estimate PSD
    # --------------------------------------------------------

    freqs, psd = signal.welch(
        x,
        fs=FS,
        nperseg=min(
            int(4 * FS),
            len(x)
        ),
        scaling="density"
    )

    # --------------------------------------------------------
    # Whitening
    # --------------------------------------------------------

    fft_freqs = np.fft.rfftfreq(
        len(x),
        1.0 / FS
    )

    psd_interp = np.interp(
        fft_freqs,
        freqs,
        psd
    )

    psd_interp = np.maximum(
        psd_interp,
        np.max(psd_interp) * 1e-20
    )

    X = np.fft.rfft(x)

    white = (
        X
        / np.sqrt(psd_interp)
    )

    y = np.fft.irfft(
        white,
        n=len(x)
    )

    # Normalize
    std = np.std(y)

    if std > 0:
        y /= std

    return y


# ============================================================
# RINGDOWN MODEL
# ============================================================

def ringdown_model(
    t,
    A,
    tau,
    f,
    phi,
    offset
):

    return (
        A
        * np.exp(-t / tau)
        * np.cos(
            2.0 * np.pi * f * t
            + phi
        )
        + offset
    )


# ============================================================
# FIT
# ============================================================

def fit_ringdown(t, y):

    # --------------------------------------------------------
    # Initial frequency estimate
    #
    # FFT peak gives a more useful starting point than
    # simply assuming 250 Hz.
    # --------------------------------------------------------

    freqs = np.fft.rfftfreq(
        len(y),
        1.0 / FS
    )

    spectrum = np.abs(
        np.fft.rfft(y)
    )

    band = (
        (freqs >= F_MIN)
        &
        (freqs <= F_MAX)
    )

    if np.any(band):

        f0 = freqs[band][
            np.argmax(
                spectrum[band]
            )
        ]

    else:

        f0 = 250.0


    # --------------------------------------------------------
    # Initial parameters
    # --------------------------------------------------------

    A0 = np.max(
        np.abs(y)
    )

    tau0 = 0.004

    phi0 = 0.0

    offset0 = np.mean(y)

    p0 = [
        A0,
        tau0,
        f0,
        phi0,
        offset0
    ]


    # --------------------------------------------------------
    # Bounds
    # --------------------------------------------------------

    lower = [
        -10.0,
        TAU_MIN,
        F_MIN,
        -2.0 * np.pi,
        -10.0
    ]

    upper = [
        10.0,
        TAU_MAX,
        F_MAX,
        2.0 * np.pi,
        10.0
    ]


    # --------------------------------------------------------
    # Residual
    # --------------------------------------------------------

    def residuals(p):

        return (
            ringdown_model(
                t,
                *p
            )
            - y
        )


    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    result = least_squares(
        residuals,
        p0,
        bounds=(
            lower,
            upper
        ),
        max_nfev=30000
    )


    A, tau, f, phi, offset = (
        result.x
    )


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    fitted = ringdown_model(
        t,
        *result.x
    )

    residual = (
        fitted - y
    )

    rms = np.sqrt(
        np.mean(
            residual ** 2
        )
    )

    ss_res = np.sum(
        residual ** 2
    )

    ss_tot = np.sum(
        (y - np.mean(y)) ** 2
    )

    if ss_tot > 0:

        r2 = (
            1.0
            - ss_res / ss_tot
        )

    else:

        r2 = np.nan


    Q = (
        np.pi
        * f
        * tau
    )


    # --------------------------------------------------------
    # Check whether tau is hitting boundary
    # --------------------------------------------------------

    tau_at_lower = (
        abs(tau - TAU_MIN)
        / TAU_MIN
        < 1e-6
    )

    tau_at_upper = (
        abs(tau - TAU_MAX)
        / TAU_MAX
        < 1e-6
    )


    return {

        "A": A,

        "tau": tau,

        "f_R": f,

        "phi": phi,

        "offset": offset,

        "Q": Q,

        "rms": rms,

        "R2": r2,

        "success": bool(
            result.success
        ),

        "cost": result.cost,

        "tau_at_lower": tau_at_lower,

        "tau_at_upper": tau_at_upper,

        "nfev": result.nfev
    }


# ============================================================
# EXTRACT WINDOW
# ============================================================

def extract_window(
    strain,
    time,
    event_gps,
    start_ms,
    window_ms
):

    t0 = (
        event_gps
        + start_ms / 1000.0
    )

    t1 = (
        t0
        + window_ms / 1000.0
    )

    mask = (
        (time >= t0)
        &
        (time <= t1)
    )

    tw = (
        time[mask]
        - t0
    )

    yw = strain[mask]

    return tw, yw


# ============================================================
# ANALYZE DETECTOR
# ============================================================

def analyze_detector(
    detector,
    strain,
    time
):

    print()
    print("=" * 70)
    print(
        f"{detector} RINGDOWN START-TIME SWEEP"
    )
    print("=" * 70)

    results = []

    for start_ms in START_MS:

        print()
        print(
            f"{detector} "
            f"start = {start_ms} ms"
        )

        t, y = extract_window(
            strain,
            time,
            EVENT_GPS,
            start_ms,
            WINDOW_MS
        )

        result = fit_ringdown(
            t,
            y
        )

        result["detector"] = detector
        result["start_ms"] = start_ms

        results.append(result)

        print(
            f"  success        : "
            f"{result['success']}"
        )

        print(
            f"  f_R            : "
            f"{result['f_R']:.6f} Hz"
        )

        print(
            f"  tau            : "
            f"{result['tau']:.8f} s"
        )

        print(
            f"  Q              : "
            f"{result['Q']:.6f}"
        )

        print(
            f"  R2             : "
            f"{result['R2']:.6f}"
        )

        print(
            f"  RMS            : "
            f"{result['rms']:.6e}"
        )

        print(
            f"  tau lower hit  : "
            f"{result['tau_at_lower']}"
        )

        print(
            f"  tau upper hit  : "
            f"{result['tau_at_upper']}"
        )

    return results


# ============================================================
# SAVE CSV
# ============================================================

def save_results(results):

    filename = (
        "stage6_3b_gw150914_ringdown_stability.csv"
    )

    fieldnames = [
        "detector",
        "start_ms",
        "f_R",
        "tau",
        "Q",
        "R2",
        "rms",
        "success",
        "tau_at_lower",
        "tau_at_upper",
        "nfev"
    ]

    with open(
        filename,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for r in results:

            writer.writerow({
                key: r[key]
                for key in fieldnames
            })

    print()
    print(
        "Saved:",
        filename
    )


# ============================================================
# PLOT
# ============================================================

def plot_results(results):

    detectors = [
        "H1",
        "L1"
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 11)
    )


    # --------------------------------------------------------
    # f_R
    # --------------------------------------------------------

    for detector in detectors:

        r = [
            x
            for x in results
            if x["detector"] == detector
        ]

        x = [
            z["start_ms"]
            for z in r
        ]

        y = [
            z["f_R"]
            for z in r
        ]

        axes[0].plot(
            x,
            y,
            marker="o",
            label=detector
        )

    axes[0].set_ylabel(
        "f_R [Hz]"
    )

    axes[0].set_title(
        "GW150914 ringdown frequency "
        "vs start time"
    )

    axes[0].legend()


    # --------------------------------------------------------
    # tau
    # --------------------------------------------------------

    for detector in detectors:

        r = [
            x
            for x in results
            if x["detector"] == detector
        ]

        x = [
            z["start_ms"]
            for z in r
        ]

        y = [
            1000.0 * z["tau"]
            for z in r
        ]

        axes[1].plot(
            x,
            y,
            marker="o",
            label=detector
        )

    axes[1].set_ylabel(
        "tau [ms]"
    )

    axes[1].set_title(
        "GW150914 damping time "
        "vs start time"
    )

    axes[1].legend()


    # --------------------------------------------------------
    # Q
    # --------------------------------------------------------

    for detector in detectors:

        r = [
            x
            for x in results
            if x["detector"] == detector
        ]

        x = [
            z["start_ms"]
            for z in r
        ]

        y = [
            z["Q"]
            for z in r
        ]

        axes[2].plot(
            x,
            y,
            marker="o",
            label=detector
        )

    axes[2].set_xlabel(
        "ringdown start after merger [ms]"
    )

    axes[2].set_ylabel(
        "Q"
    )

    axes[2].set_title(
        "GW150914 Q vs start time"
    )

    axes[2].legend()


    plt.tight_layout()

    output = (
        "stage6_3b_gw150914_ringdown_stability.png"
    )

    plt.savefig(
        output,
        dpi=160
    )

    plt.close()

    print(
        "Saved:",
        output
    )


# ============================================================
# SUMMARY
# ============================================================

def summary(results):

    print()
    print("=" * 70)
    print(
        "STAGE 6.3b -- SUMMARY"
    )
    print("=" * 70)

    for detector in [
        "H1",
        "L1"
    ]:

        r = [
            x
            for x in results
            if x["detector"] == detector
        ]

        valid = [
            x
            for x in r
            if x["success"]
        ]

        print()
        print(
            detector,
            "valid fits:",
            len(valid),
            "/",
            len(r)
        )

        if not valid:
            continue

        f = np.array([
            x["f_R"]
            for x in valid
        ])

        tau = np.array([
            x["tau"]
            for x in valid
        ])

        Q = np.array([
            x["Q"]
            for x in valid
        ])

        print(
            f"f_R range: "
            f"{np.min(f):.3f} -- "
            f"{np.max(f):.3f} Hz"
        )

        print(
            f"tau range: "
            f"{1000*np.min(tau):.3f} -- "
            f"{1000*np.max(tau):.3f} ms"
        )

        print(
            f"Q range: "
            f"{np.min(Q):.3f} -- "
            f"{np.max(Q):.3f}"
        )

        boundary_hits = sum(
            x["tau_at_lower"]
            for x in valid
        )

        print(
            "tau lower-bound hits:",
            boundary_hits
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "A stable result is NOT defined by "
        "one successful fit."
    )

    print(
        "We require stability across "
        "multiple ringdown start times "
        "and consistency between H1 and L1."
    )

    print(
        "No Lambda-model interpretation "
        "is performed in this stage."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "STAGE 6.3b -- "
        "GW150914 REAL-DATA RINGDOWN STABILITY"
    )
    print("=" * 70)

    print()
    print(
        "This is a generic observational test."
    )

    print(
        "No Lambda-model fitting."
    )

    print(
        "No Kerr parameters imposed."
    )

    print(
        "No event-calibrated Lambda mapping."
    )

    print()

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(H1_FILE):

        raise FileNotFoundError(
            f"Missing H1 file: {H1_FILE}"
        )

    if not os.path.exists(L1_FILE):

        raise FileNotFoundError(
            f"Missing L1 file: {L1_FILE}"
        )


    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print(
        "Loading H1..."
    )

    h1, t1 = load_strain(
        H1_FILE
    )

    print(
        "Loading L1..."
    )

    l1, t2 = load_strain(
        L1_FILE
    )


    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    print()
    print(
        "Preprocessing H1..."
    )

    h1p = preprocess(
        h1
    )

    print(
        "Preprocessing L1..."
    )

    l1p = preprocess(
        l1
    )


    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    h1_results = analyze_detector(
        "H1",
        h1p,
        t1
    )

    l1_results = analyze_detector(
        "L1",
        l1p,
        t2
    )


    results = (
        h1_results
        + l1_results
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        results
    )


    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_results(
        results
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary(
        results
    )


if __name__ == "__main__":
    main()