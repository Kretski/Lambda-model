"""
Stage 6.3
GW150914 real-data ringdown extraction

IMPORTANT:
This script does NOT fit the Lambda-model.
It first extracts a generic damped sinusoid from real GW data.

Model:
    h(t) = A * exp(-(t-t0)/tau) * cos(2*pi*f*(t-t0) + phi)

Outputs:
    f_R
    tau
    Q = pi*f_R*tau

This is an observational extraction / control experiment.
No Kerr or Lambda interpretation is imposed here.
"""

import os
import urllib.request

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

H1_URL = (
    "https://gwosc.org/s/events/GW150914/"
    "H-H1_LOSC_4_V1-1126259446-32.hdf5"
)

L1_URL = (
    "https://gwosc.org/s/events/GW150914/"
    "L-L1_LOSC_4_V1-1126259446-32.hdf5"
)

H1_FILE = os.path.join(
    DATA_DIR,
    "H-H1_LOSC_4_V1-1126259446-32.hdf5"
)

L1_FILE = os.path.join(
    DATA_DIR,
    "L-L1_LOSC_4_V1-1126259446-32.hdf5"
)

# Ringdown start relative to merger.
#
# IMPORTANT:
# We deliberately keep this configurable.
# We are NOT choosing it to make Lambda fit.
#
# Start with a conservative range after the peak.
RINGDOWN_START = 0.010

# Analyze this much time after the chosen start.
RINGDOWN_DURATION = 0.100

# Broad GW band.
LOW_FREQ = 20.0
HIGH_FREQ = 500.0


# ============================================================
# DOWNLOAD
# ============================================================

def download_if_missing(url, filename):

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if os.path.exists(filename):
        print("Already exists:", filename)
        return

    print("Downloading:")
    print(url)

    urllib.request.urlretrieve(url, filename)

    print("Saved:", filename)


# ============================================================
# LOAD GWOSC HDF5
# ============================================================

def load_strain(filename):

    with h5py.File(filename, "r") as f:

        # GWOSC structure:
        # meta/strain
        strain = np.array(f["strain/Strain"])

        gps_start = float(
            f["meta/GPSstart"][()]
        )

        sample_rate = float(
            f["meta/Duration"][()]
        )

        duration = float(
            f["meta/Duration"][()]
        )

        # Some files contain sampling information in metadata.
        # For this GW150914 release we use the documented 4096 Hz.
        fs = FS

    time = gps_start + np.arange(len(strain)) / fs

    return strain, time, fs


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(strain, fs):

    x = np.asarray(strain, dtype=float)

    # Remove DC / slow drift
    x = signal.detrend(x)

    # Butterworth bandpass
    sos = signal.butter(
        4,
        [LOW_FREQ, HIGH_FREQ],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    x = signal.sosfiltfilt(sos, x)

    # Whitening using Welch PSD
    freqs, psd = signal.welch(
        x,
        fs=fs,
        nperseg=min(4 * int(fs), len(x)),
        scaling="density"
    )

    # Interpolate PSD to FFT frequencies
    fft_freqs = np.fft.rfftfreq(len(x), 1.0 / fs)

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

    white = X / np.sqrt(psd_interp)

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
# GENERIC RINGDOWN MODEL
# ============================================================

def ringdown_model(t, A, tau, f, phi, offset):

    return (
        A
        * np.exp(-t / tau)
        * np.cos(2.0 * np.pi * f * t + phi)
        + offset
    )


# ============================================================
# FIT
# ============================================================

def fit_ringdown(t, y):

    # Initial values deliberately broad.
    A0 = np.max(np.abs(y))

    tau0 = 0.03
    f0 = 250.0
    phi0 = 0.0
    offset0 = np.mean(y)

    p0 = [
        A0,
        tau0,
        f0,
        phi0,
        offset0
    ]

    lower = [
        -10.0,
        0.003,
        50.0,
        -2.0 * np.pi,
        -10.0
    ]

    upper = [
        10.0,
        1.0,
        500.0,
        2.0 * np.pi,
        10.0
    ]

    def residuals(p):

        return (
            ringdown_model(t, *p) - y
        )

    result = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        max_nfev=10000
    )

    A, tau, f, phi, offset = result.x

    Q = np.pi * f * tau

    rms = np.sqrt(
        np.mean(result.fun ** 2)
    )

    return {
        "A": A,
        "tau": tau,
        "f_R": f,
        "phi": phi,
        "offset": offset,
        "Q": Q,
        "rms": rms,
        "success": result.success,
        "cost": result.cost
    }


# ============================================================
# EXTRACT WINDOW
# ============================================================

def extract_window(
    strain,
    time,
    event_gps,
    start,
    duration
):

    t0 = event_gps + start
    t1 = t0 + duration

    mask = (
        (time >= t0)
        &
        (time <= t1)
    )

    tw = time[mask] - t0
    yw = strain[mask]

    return tw, yw


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STAGE 6.3 -- GW150914 REAL-DATA RINGDOWN EXTRACTION")
    print("=" * 70)

    print()
    print("IMPORTANT:")
    print("No Lambda-model fitting is performed.")
    print("No Kerr parameters are imposed.")
    print("The first task is purely observational extraction.")
    print()

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    download_if_missing(
        H1_URL,
        H1_FILE
    )

    download_if_missing(
        L1_URL,
        L1_FILE
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    h1, t1, fs1 = load_strain(H1_FILE)
    l1, t2, fs2 = load_strain(L1_FILE)

    print("H1 samples:", len(h1))
    print("L1 samples:", len(l1))
    print("Sampling rate:", fs1, "Hz")

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    print()
    print("Preprocessing H1...")
    h1p = preprocess(h1, fs1)

    print("Preprocessing L1...")
    l1p = preprocess(l1, fs2)

    # --------------------------------------------------------
    # Extract ringdown
    # --------------------------------------------------------

    t_h1, y_h1 = extract_window(
        h1p,
        t1,
        EVENT_GPS,
        RINGDOWN_START,
        RINGDOWN_DURATION
    )

    t_l1, y_l1 = extract_window(
        l1p,
        t2,
        EVENT_GPS,
        RINGDOWN_START,
        RINGDOWN_DURATION
    )

    print()
    print("Ringdown window:")
    print(
        f"[{RINGDOWN_START:.4f}, "
        f"{RINGDOWN_START + RINGDOWN_DURATION:.4f}] s"
    )

    # --------------------------------------------------------
    # Fit H1
    # --------------------------------------------------------

    print()
    print("Fitting H1...")

    result_h1 = fit_ringdown(
        t_h1,
        y_h1
    )

    print()
    print("H1 RESULT")
    print("-" * 40)

    for key, value in result_h1.items():
        print(f"{key:10s}: {value}")

    # --------------------------------------------------------
    # Fit L1
    # --------------------------------------------------------

    print()
    print("Fitting L1...")

    result_l1 = fit_ringdown(
        t_l1,
        y_l1
    )

    print()
    print("L1 RESULT")
    print("-" * 40)

    for key, value in result_l1.items():
        print(f"{key:10s}: {value}")

    # --------------------------------------------------------
    # Internal consistency
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("INTERNAL Q CONSISTENCY")
    print("=" * 70)

    for detector, result in [
        ("H1", result_h1),
        ("L1", result_l1)
    ]:

        Q_direct = result["Q"]

        Q_check = (
            np.pi
            * result["f_R"]
            * result["tau"]
        )

        rel_error = abs(
            Q_direct - Q_check
        ) / max(
            abs(Q_direct),
            1e-30
        )

        print(
            f"{detector}: "
            f"Q={Q_direct:.8g}, "
            f"pi*f*tau={Q_check:.8g}, "
            f"relative error={rel_error:.3e}"
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True
    )

    axes[0].plot(
        t_h1,
        y_h1,
        label="H1 real data"
    )

    axes[0].plot(
        t_h1,
        ringdown_model(
            t_h1,
            result_h1["A"],
            result_h1["tau"],
            result_h1["f_R"],
            result_h1["phi"],
            result_h1["offset"]
        ),
        label="generic damped-sinusoid fit"
    )

    axes[0].set_ylabel("whitened strain")
    axes[0].set_title(
        "GW150914 H1 ringdown extraction"
    )
    axes[0].legend()

    axes[1].plot(
        t_l1,
        y_l1,
        label="L1 real data"
    )

    axes[1].plot(
        t_l1,
        ringdown_model(
            t_l1,
            result_l1["A"],
            result_l1["tau"],
            result_l1["f_R"],
            result_l1["phi"],
            result_l1["offset"]
        ),
        label="generic damped-sinusoid fit"
    )

    axes[1].set_xlabel("time after ringdown start [s]")
    axes[1].set_ylabel("whitened strain")
    axes[1].set_title(
        "GW150914 L1 ringdown extraction"
    )
    axes[1].legend()

    plt.tight_layout()

    output_plot = (
        "stage6_3_gw150914_ringdown_extraction.png"
    )

    plt.savefig(
        output_plot,
        dpi=160
    )

    plt.close()

    print()
    print("Saved:", output_plot)

    # --------------------------------------------------------
    # Final scientific status
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SCIENTIFIC STATUS")
    print("=" * 70)

    print(
        "This run extracts a generic damped mode from real GW data."
    )

    print(
        "It does NOT establish a Lambda-model interpretation."
    )

    print(
        "It does NOT establish a Kerr mapping."
    )

    print(
        "It provides observational parameters for the next comparison."
    )


if __name__ == "__main__":
    main()