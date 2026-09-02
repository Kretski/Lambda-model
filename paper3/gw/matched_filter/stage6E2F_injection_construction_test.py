import argparse
import os
import numpy as np

from pycbc.waveform import get_td_waveform
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.types import TimeSeries
from pycbc.filter import matched_filter


FS = 4096.0
DURATION = 16.0
N = int(FS * DURATION)

F_LOW = 20.0
F_HIGH = 300.0

M1 = 36.0
M2 = 29.0

INJECTION_TIME = 8.0
TARGET_SNRS = [5.0, 10.0, 20.0]

SEED = 20260830

OUTPUT_DIR = "stage6E2F_results"


def generate_template():
    hp, _ = get_td_waveform(
        approximant="IMRPhenomD",
        mass1=M1,
        mass2=M2,
        delta_t=1.0 / FS,
        f_lower=F_LOW,
        f_final=F_HIGH,
    )

    hp = hp.copy()

    # Force exact duration.
    if len(hp) < N:
        x = np.zeros(N)
        x[:len(hp)] = np.asarray(hp)
        hp = TimeSeries(
            x,
            delta_t=1.0 / FS,
            epoch=0
        )
    else:
        hp = TimeSeries(
            np.asarray(hp[:N]),
            delta_t=1.0 / FS,
            epoch=0
        )

    return hp


def build_psd():
    return aLIGOZeroDetHighPower(
        int(N // 2 + 1),
        1.0 / DURATION,
        F_LOW
    )


def shift_to_injection_time(template):
    data = np.zeros(N)

    peak = int(np.argmax(np.abs(np.asarray(template))))

    target = int(round(INJECTION_TIME * FS))

    shift = target - peak

    src_start = max(0, -shift)
    src_end = min(N, N - shift)

    dst_start = max(0, shift)
    dst_end = dst_start + (src_end - src_start)

    if src_end > src_start:
        data[dst_start:dst_end] = np.asarray(template)[src_start:src_end]

    return TimeSeries(
        data,
        delta_t=1.0 / FS,
        epoch=0
    )


def make_psd_consistent_noise(psd, seed):
    rng = np.random.default_rng(seed)

    nfreq = len(psd)

    # Generate complex frequency-domain Gaussian noise.
    re = rng.normal(0.0, 1.0, nfreq)
    im = rng.normal(0.0, 1.0, nfreq)

    freq_noise = re + 1j * im

    # Remove DC.
    freq_noise[0] = 0.0

    # Nyquist must be real.
    if N % 2 == 0:
        freq_noise[-1] = rng.normal(0.0, 1.0)

    # PSD scaling.
    scale = np.sqrt(
        np.maximum(np.asarray(psd), 0.0)
        * FS
        / 2.0
    )

    freq_noise *= scale

    # irfft gives exactly N samples.
    noise = np.fft.irfft(freq_noise, n=N)

    noise = noise.astype(np.float64)

    return TimeSeries(
        noise,
        delta_t=1.0 / FS,
        epoch=0
    )


def mf_peak(data, template, psd):
    result = matched_filter(
        template,
        data,
        psd=psd,
        low_frequency_cutoff=F_LOW,
        high_frequency_cutoff=F_HIGH
    )

    arr = np.asarray(result)

    idx = int(np.argmax(np.abs(arr)))
    peak = float(np.abs(arr[idx]))

    return peak, idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-realizations",
        type=int,
        default=3
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("STAGE 6E2F — INJECTION CONSTRUCTION TEST")
    print("=" * 70)
    print(f"FS                  = {FS} Hz")
    print(f"DURATION            = {DURATION} s")
    print(f"N                   = {N}")
    print(f"DT                  = {1.0 / FS:.15e} s")
    print(f"DF                  = {1.0 / DURATION:.8f} Hz")
    print(f"BAND                = {F_LOW} - {F_HIGH} Hz")
    print(f"M1                  = {M1} Msun")
    print(f"M2                  = {M2} Msun")
    print(f"INJECTION TIME      = {INJECTION_TIME:.6f} s")
    print(f"REALIZATIONS        = {args.n_realizations}")
    print()

    print("[1] GENERATING TEMPLATE")

    template = generate_template()

    print(f"template length     = {len(template)}")
    print(f"template L2         = {np.linalg.norm(np.asarray(template)):.12e}")
    print(f"template std        = {np.std(np.asarray(template)):.12e}")
    print()

    print("[2] BUILDING PSD")

    psd = build_psd()

    print(f"PSD bins            = {len(psd)}")
    print(f"PSD delta_f         = {psd.delta_f}")
    print(
        f"PSD range           = "
        f"{np.min(np.asarray(psd)):.6e} .. "
        f"{np.max(np.asarray(psd)):.6e}"
    )
    print()

    print("[3] SELF MATCH")

    self_peak, self_idx = mf_peak(
        template,
        template,
        psd
    )

    print(f"self-match peak     = {self_peak:.12e}")
    print(f"self-match index    = {self_idx}")
    print()

    print("[4] SHIFTED TEMPLATE")

    shifted = shift_to_injection_time(template)

    shifted_peak_index = int(
        np.argmax(np.abs(np.asarray(shifted)))
    )

    shifted_peak_time = shifted_peak_index / FS

    print(f"shifted peak index  = {shifted_peak_index}")
    print(f"shifted peak time   = {shifted_peak_time:.9f} s")
    print(
        f"requested time      = "
        f"{INJECTION_TIME:.9f} s"
    )
    print(
        f"sample error        = "
        f"{shifted_peak_index - int(round(INJECTION_TIME * FS))}"
    )
    print()

    print("[5] SIGNAL-ONLY TEST")
    print()

    for target in TARGET_SNRS:

        # Since MF(A*h, h) = A * MF(h,h),
        # this amplitude produces the requested target.
        amplitude = target / self_peak

        signal = shifted * amplitude

        recovered, idx = mf_peak(
            signal,
            template,
            psd
        )

        error = recovered - target

        print(
            f"target={target:6.2f}  "
            f"amplitude={amplitude:.12e}  "
            f"recovered={recovered:.9f}  "
            f"error={error:+.6e}  "
            f"index={idx}"
        )

    print()

    print("[6] NOISE-ONLY TEST")

    noise_peaks = []

    for r in range(args.n_realizations):

        noise = make_psd_consistent_noise(
            psd,
            SEED + r
        )

        peak, idx = mf_peak(
            noise,
            template,
            psd
        )

        noise_peaks.append(peak)

        print(
            f"realization={r + 1:3d}  "
            f"noise_std={np.std(np.asarray(noise)):.6e}  "
            f"noise_peak={peak:.6f}  "
            f"index={idx}"
        )

    print()

    print("[7] SIGNAL + NOISE TEST")

    for target in TARGET_SNRS:

        amplitude = target / self_peak

        recovered_values = []

        for r in range(args.n_realizations):

            noise = make_psd_consistent_noise(
                psd,
                SEED + 1000 + r
            )

            signal = shifted * amplitude

            data = signal + noise

            recovered, idx = mf_peak(
                data,
                template,
                psd
            )

            recovered_values.append(recovered)

            print(
                f"target={target:6.2f}  "
                f"realization={r + 1:3d}  "
                f"recovered={recovered:.6f}  "
                f"index={idx}"
            )

        mean = float(np.mean(recovered_values))
        std = float(np.std(recovered_values))

        print(
            f"   mean={mean:.6f}  "
            f"std={std:.6f}"
        )

    print()

    print("[8] PSD-WEIGHTED SIGNAL CHECK")

    signal_norms = []

    for target in TARGET_SNRS:

        amplitude = target / self_peak

        signal = shifted * amplitude

        sig_f = np.fft.rfft(
            np.asarray(signal)
        )

        psd_arr = np.asarray(psd)

        valid = (
            (np.arange(len(psd_arr)) * (FS / N) >= F_LOW) &
            (np.arange(len(psd_arr)) * (FS / N) <= F_HIGH) &
            (psd_arr > 0)
        )

        weighted = np.sum(
            np.abs(sig_f[valid]) ** 2
            / psd_arr[valid]
        )

        print(
            f"target={target:6.2f}  "
            f"weighted_power={weighted:.12e}"
        )

        signal_norms.append(weighted)

    print()

    print("=" * 70)
    print("STAGE 6E2F SUMMARY")
    print("=" * 70)

    print()
    print("Expected behavior:")
    print()
    print("1. Signal-only recovered SNR should equal target.")
    print("2. Noise-only peaks should be O(1- few),")
    print("   not 1e22.")
    print("3. Signal+noise should remain near the injected")
    print("   target with statistical scatter.")
    print()
    print("No Lambda inference.")
    print("No physical Lambda claim.")
    print("This test isolates injection/noise construction.")
    print("=" * 70)


if __name__ == "__main__":
    main()