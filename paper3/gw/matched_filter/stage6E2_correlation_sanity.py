import argparse
import numpy as np
import matplotlib.pyplot as plt

from pycbc.waveform import get_fd_waveform
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.types import FrequencySeries, TimeSeries
from pycbc.filter import matched_filter
from pycbc.filter.matchedfilter import sigmasq

# ============================================================

# PAPER 3 — STAGE 6E2

# CORRELATION / NORMALIZATION SANITY TEST

#

# Purpose:

# Diagnose where unphysical ~1e22-1e28 "SNR" values arise.

#

# Tests:

# 1. Template generation

# 2. PSD

# 3. PyCBC sigmasq()

# 4. Template self-match

# 5. Unit-amplitude injection, no noise

# 6. Unit-amplitude injection + noise

# 7. Explicit target-SNR injection

# 8. PyCBC matched_filter() output inspection

#

# NO Lambda inference.

# NO physical Lambda claim.

# ============================================================

FS = 4096.0
DURATION = 16.0
N = int(FS * DURATION)

F_LOWER = 20.0
F_UPPER = 300.0

MASS1 = 36.0
MASS2 = 29.0

INJECTION_TIME = 8.0
DETECTION_THRESHOLD = 5.0

SEED = 20261330

def banner(text):
print()
print("=" * 70)
print(text)
print("=" * 70)

def stats(name, x):
x = np.asarray(x)

```
finite = np.isfinite(x)

print(f"\n{name}")
print(f"  dtype       = {x.dtype}")
print(f"  shape       = {x.shape}")
print(f"  finite      = {finite.all()}")

if finite.any():
    xf = x[finite]
    print(f"  min         = {np.min(xf):.6e}")
    print(f"  max         = {np.max(xf):.6e}")
    print(f"  mean        = {np.mean(xf):.6e}")
    print(f"  std         = {np.std(xf):.6e}")
    print(f"  abs max     = {np.max(np.abs(xf)):.6e}")
```

def make_template():
hp, hc = get_fd_waveform(
approximant="IMRPhenomD",
mass1=MASS1,
mass2=MASS2,
delta_f=1.0 / DURATION,
f_lower=F_LOWER,
f_final=F_UPPER,
)

```
hp = hp.copy()

# Force expected frequency-domain length.
nfreq = N // 2 + 1

if len(hp) > nfreq:
    hp = hp[:nfreq]

elif len(hp) < nfreq:
    padded = np.zeros(nfreq, dtype=np.complex128)
    padded[:len(hp)] = np.asarray(hp)
    hp = FrequencySeries(
        padded,
        delta_f=1.0 / DURATION,
    )

return hp
```

def build_psd():
psd = aLIGOZeroDetHighPower(
N // 2 + 1,
1.0 / DURATION,
F_LOWER,
)

```
return psd
```

def make_time_series_from_fd(fd):
"""
Convert FD waveform to a real TimeSeries with exactly N samples.
"""
arr = np.asarray(fd)

```
td = np.fft.irfft(arr, n=N)

return TimeSeries(
    td,
    delta_t=1.0 / FS,
    epoch=0,
)
```

def whitened_inner_product_fd(a, b, psd):
"""
Explicit diagnostic only.

```
Computes:
    4 Re sum[a* conj(b) / PSD] df

This is NOT used for the production matched filter.
"""
aa = np.asarray(a)
bb = np.asarray(b)
pp = np.asarray(psd)

n = min(len(aa), len(bb), len(pp))

aa = aa[:n]
bb = bb[:n]
pp = pp[:n]

df = 1.0 / DURATION

mask = np.isfinite(pp) & (pp > 0)

value = 4.0 * np.real(
    np.sum(
        aa[mask] * np.conj(bb[mask]) / pp[mask]
    )
) * df

return float(value)
```

def inspect_sigmasq(template, psd):
banner("DIAGNOSTIC A — PyCBC sigmasq()")

```
try:
    ss = sigmasq(
        template,
        psd=psd,
        low_frequency_cutoff=F_LOWER,
        high_frequency_cutoff=F_UPPER,
    )

    print(f"sigmasq(template) = {ss:.12e}")

    if ss > 1e20:
        print("WARNING: sigmasq itself is enormous.")

    elif ss < 1e-20:
        print("WARNING: sigmasq is extremely small.")

    else:
        print("sigmasq is numerically finite and plausible.")

    return float(ss)

except Exception as e:
    print("sigmasq FAILED:")
    print(repr(e))
    return None
```

def inspect_self_match(template_td, template, psd):
banner("DIAGNOSTIC B — PyCBC SELF MATCH")

```
print("Running matched_filter(template, template)...")

try:
    mf = matched_filter(
        template_td,
        template,
        psd=psd,
        low_frequency_cutoff=F_LOWER,
        high_frequency_cutoff=F_UPPER,
    )

    arr = np.asarray(mf)

    stats("matched_filter(template, template)", arr)

    idx = int(np.argmax(np.abs(arr)))
    peak = arr[idx]

    print(f"\nPeak index = {idx}")
    print(f"Peak value = {peak}")
    print(f"|peak|     = {abs(peak):.12e}")

    return mf

except Exception as e:
    print("SELF MATCH FAILED:")
    print(repr(e))
    return None
```

def make_injection(template_td, target_snr, psd):
"""
Scale a time-domain template so that its theoretical
PyCBC sigma is approximately target_snr.

```
This is a diagnostic construction.
"""

template_fd = FrequencySeries(
    np.fft.rfft(np.asarray(template_td)),
    delta_f=1.0 / DURATION,
)

try:
    ss = sigmasq(
        template_fd,
        psd=psd,
        low_frequency_cutoff=F_LOWER,
        high_frequency_cutoff=F_UPPER,
    )
except Exception:
    ss = None

if ss is None or ss <= 0 or not np.isfinite(ss):
    raise RuntimeError(
        f"Invalid sigmasq during injection scaling: {ss}"
    )

sigma = np.sqrt(ss)

amplitude = target_snr / sigma

injected = template_td.copy()
injected *= amplitude

return injected, amplitude, sigma
```

def shift_to_injection_time(x, injection_time):
"""
Shift waveform so that its nominal waveform location is
centered near the requested injection time.

```
Uses circular shift only for diagnostic purposes.
"""
arr = np.asarray(x).copy()

current_peak = int(np.argmax(np.abs(arr)))

target_sample = int(round(injection_time * FS))

shift = target_sample - current_peak

shifted = np.roll(arr, shift)

return TimeSeries(
    shifted,
    delta_t=1.0 / FS,
    epoch=0,
), current_peak, target_sample, shift
```

def run_matched_filter(strain, template, psd, label):
banner(f"DIAGNOSTIC — MATCHED FILTER: {label}")

```
print("Input strain:")
stats("strain", np.asarray(strain))

print("\nTemplate:")
stats("template", np.asarray(template))

print("\nPSD:")
stats("PSD", np.asarray(psd))

try:
    mf = matched_filter(
        strain,
        template,
        psd=psd,
        low_frequency_cutoff=F_LOWER,
        high_frequency_cutoff=F_UPPER,
    )

    arr = np.asarray(mf)

    stats("matched_filter output", arr)

    abs_arr = np.abs(arr)

    peak_idx = int(np.argmax(abs_arr))
    peak_value = arr[peak_idx]

    recovered = float(abs(peak_value))

    print("\nRESULT")
    print(f"  peak index     = {peak_idx}")
    print(f"  peak value     = {peak_value}")
    print(f"  recovered SNR  = {recovered:.12e}")

    if recovered > 1e10:
        print()
        print("!!! CRITICAL !!!")
        print("Matched-filter output is unphysical (>1e10).")
        print("The huge value appears INSIDE PyCBC matched_filter output.")

    elif recovered > 1000:
        print()
        print("WARNING:")
        print("Matched-filter output is unexpectedly large.")

    else:
        print()
        print("Matched-filter output is in a numerically plausible range.")

    return mf

except Exception as e:
    print("MATCHED FILTER FAILED:")
    print(repr(e))
    return None
```

def add_noise(strain, seed, noise_scale=1.0):
rng = np.random.default_rng(seed)

```
noise = rng.normal(
    0.0,
    noise_scale,
    size=len(strain),
)

return TimeSeries(
    np.asarray(strain) + noise,
    delta_t=1.0 / FS,
    epoch=0,
)
```

def main():

```
parser = argparse.ArgumentParser()

parser.add_argument(
    "--plot",
    action="store_true",
    help="Show diagnostic plots.",
)

args = parser.parse_args()

banner("PAPER 3 — STAGE 6E2 CORRELATION SANITY TEST")

print(f"Sampling rate       = {FS} Hz")
print(f"Duration            = {DURATION} s")
print(f"Samples             = {N}")
print(f"Frequency band      = {F_LOWER} – {F_UPPER} Hz")
print(f"delta_t             = {1.0 / FS:.16f} s")
print(f"delta_f             = {1.0 / DURATION:.8f} Hz")
print(f"Waveform            = IMRPhenomD")
print(f"Mass1               = {MASS1} Msun")
print(f"Mass2               = {MASS2} Msun")
print(f"Injection time      = {INJECTION_TIME:.6f} s")
print()
print("This test diagnoses correlation normalization only.")
print("No Lambda inference is performed.")

# --------------------------------------------------------
# 1. PSD
# --------------------------------------------------------

banner("[1] BUILDING PSD")

psd = build_psd()

print(f"PSD bins = {len(psd)}")
print(f"PSD delta_f = {psd.delta_f:.8f} Hz")

stats("PSD values", np.asarray(psd))

# --------------------------------------------------------
# 2. TEMPLATE
# --------------------------------------------------------

banner("[2] GENERATING TEMPLATE")

template_fd = make_template()

print(f"Template FD bins = {len(template_fd)}")
print(f"Template delta_f  = {template_fd.delta_f:.8f} Hz")

stats(
    "Template FD",
    np.asarray(template_fd),
)

template_td = make_time_series_from_fd(template_fd)

print(f"\nTemplate TD samples = {len(template_td)}")
print(f"Template delta_t    = {template_td.delta_t}")

stats(
    "Template TD",
    np.asarray(template_td),
)

# --------------------------------------------------------
# 3. SIGMASQ
# --------------------------------------------------------

sigma_sq = inspect_sigmasq(
    template_fd,
    psd,
)

if sigma_sq is not None:
    sigma = np.sqrt(sigma_sq)

    print(f"\nsqrt(sigmasq) = {sigma:.12e}")

else:
    sigma = None

# --------------------------------------------------------
# 4. EXPLICIT INNER PRODUCT
# --------------------------------------------------------

banner("DIAGNOSTIC C — EXPLICIT FD INNER PRODUCT")

explicit = whitened_inner_product_fd(
    template_fd,
    template_fd,
    psd,
)

print(
    "4 Re sum(h*conj(h)/PSD) df = "
    f"{explicit:.12e}"
)

if sigma_sq is not None:
    print(
        "PyCBC sigmasq               = "
        f"{sigma_sq:.12e}"
    )

    print(
        "Ratio explicit / sigmasq    = "
        f"{explicit / sigma_sq:.12e}"
    )

# --------------------------------------------------------
# 5. SELF MATCH
# --------------------------------------------------------

self_mf = inspect_self_match(
    template_td,
    template_fd,
    psd,
)

# --------------------------------------------------------
# 6. ZERO-NOISE UNIT TEMPLATE
# --------------------------------------------------------

banner("DIAGNOSTIC D — ZERO-NOISE TEMPLATE")

shifted_template, current_peak, target_sample, shift = \
    shift_to_injection_time(
        template_td,
        INJECTION_TIME,
    )

print(f"Original template peak = {current_peak}")
print(f"Target injection sample = {target_sample}")
print(f"Applied circular shift = {shift}")

unit_mf = run_matched_filter(
    shifted_template,
    template_fd,
    psd,
    "ZERO-NOISE UNIT TEMPLATE",
)

# --------------------------------------------------------
# 7. TARGET SNR INJECTIONS
# --------------------------------------------------------

banner("DIAGNOSTIC E — TARGET-SNR INJECTIONS")

target_snrs = [5.0, 8.0, 10.0, 15.0, 20.0]

results = []

for target in target_snrs:

    print()
    print("-" * 70)
    print(f"TARGET SNR = {target:.1f}")

    try:

        scaled, amplitude, theoretical_sigma = make_injection(
            shifted_template,
            target,
            psd,
        )

        print(
            f"Injection amplitude scale = "
            f"{amplitude:.12e}"
        )

        print(
            f"Template sigma             = "
            f"{theoretical_sigma:.12e}"
        )

        fd_scaled = FrequencySeries(
            np.fft.rfft(np.asarray(scaled)),
            delta_f=1.0 / DURATION,
        )

        scaled_sigma_sq = sigmasq(
            fd_scaled,
            psd=psd,
            low_frequency_cutoff=F_LOWER,
            high_frequency_cutoff=F_UPPER,
        )

        print(
            f"Scaled sigmasq             = "
            f"{scaled_sigma_sq:.12e}"
        )

        print(
            f"Expected sqrt(sigmasq)     = "
            f"{np.sqrt(scaled_sigma_sq):.12e}"
        )

        mf = run_matched_filter(
            scaled,
            template_fd,
            psd,
            f"ZERO-NOISE TARGET SNR {target:.1f}",
        )

        recovered = None

        if mf is not None:
            recovered = float(
                np.max(np.abs(np.asarray(mf)))
            )

        results.append(
            (
                target,
                amplitude,
                recovered,
            )
        )

    except Exception as e:

        print("TARGET-SNR TEST FAILED:")
        print(repr(e))

# --------------------------------------------------------
# 8. NOISE TEST
# --------------------------------------------------------

banner("DIAGNOSTIC F — TARGET-SNR + GAUSSIAN NOISE")

target = 10.0

try:

    scaled, amplitude, theoretical_sigma = make_injection(
        shifted_template,
        target,
        psd,
    )

    noisy = add_noise(
        scaled,
        SEED,
        noise_scale=1.0,
    )

    noise_mf = run_matched_filter(
        noisy,
        template_fd,
        psd,
        f"SNR {target:.1f} + GAUSSIAN NOISE",
    )

except Exception as e:

    print("NOISE TEST FAILED:")
    print(repr(e))

    noise_mf = None

# --------------------------------------------------------
# 9. FINAL DIAGNOSIS
# --------------------------------------------------------

banner("FINAL DIAGNOSIS")

print("\nTarget / recovered comparison:")

for target, amplitude, recovered in results:

    if recovered is None:
        print(
            f"  target={target:5.1f} | "
            f"recovered=FAILED"
        )
    else:
        print(
            f"  target={target:5.1f} | "
            f"recovered={recovered:.12e} | "
            f"ratio={recovered / target:.12e}"
        )

print()
print("Interpretation rules:")
print()
print("1. If sigmasq() is ~1e20 or larger:")
print("   -> problem is already present in PSD/template normalization.")
print()
print("2. If sigmasq() is reasonable but matched_filter()")
print("   returns ~1e20:")
print("   -> problem is in matched_filter input conventions")
print("      or template/strain normalization.")
print()
print("3. If zero-noise target SNR=10 returns ~10:")
print("   -> normalization is correct.")
print()
print("4. If zero-noise target SNR=10 returns ~1e20:")
print("   -> scaling or correlation convention is broken.")
print()
print("5. If zero-noise works but noise case fails:")
print("   -> investigate PSD/noise consistency.")
print()
print("NO LAMBDA INFERENCE.")
print("NO PHYSICAL LAMBDA CLAIM.")

# --------------------------------------------------------
# 10. OPTIONAL PLOT
# --------------------------------------------------------

if args.plot:

    banner("DIAGNOSTIC PLOT")

    plt.figure(figsize=(12, 5))

    if self_mf is not None:
        plt.plot(
            np.abs(np.asarray(self_mf)),
            label="Self-match",
        )

    if unit_mf is not None:
        plt.plot(
            np.abs(np.asarray(unit_mf)),
            label="Zero-noise unit",
        )

    if noise_mf is not None:
        plt.plot(
            np.abs(np.asarray(noise_mf)),
            label="SNR 10 + noise",
        )

    plt.yscale("log")
    plt.xlabel("Sample")
    plt.ylabel("|matched_filter output|")
    plt.title("Stage 6E2 Correlation Sanity Test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "stage6E2_correlation_sanity.png",
        dpi=150,
    )

    print(
        "\nSaved: "
        "stage6E2_correlation_sanity.png"
    )
```

if **name** == "**main**":
main()
