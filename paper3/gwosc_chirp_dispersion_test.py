"""
gwosc_chirp_dispersion_test.py
==================================

REAL DATA TEST: extract the instantaneous chirp frequency evolution from
actual GW150914 (or other event) strain data, and test whether the
arrival-time-vs-frequency relation is consistent with the Lambda-model
FLRW dispersion formula:

    Delta T(omega, z) = -(Lambda / 2c^3) * omega^2 * K(z)

METHOD:

  1. Load H1 and L1 strain around the event GPS time.
  2. Whiten and bandpass the strain (standard LIGO preprocessing).
  3. Extract the instantaneous frequency track of the chirp via the
     analytic signal (Hilbert transform) phase derivative, OR via a
     time-frequency (Q-transform-like) ridge extraction. We use the
     simpler robust approach: track the dominant frequency in
     overlapping short-time windows across the chirp, in both H1 and L1
     independently.
  4. For a genuine dispersion test, compare arrival times of specific
     frequency bins BETWEEN H1 and L1 is NOT what we want (that gives
     sky location / polarization timing, not dispersion -- H1 and L1
     see the SAME dispersed signal, just time-shifted by geometry).

  THE CORRECT TEST: dispersion is intrinsic to propagation, so it
  affects the SHAPE of the frequency-vs-time chirp track itself,
  relative to the GR (non-dispersive) post-Newtonian prediction for a
  binary with the known component masses. We compare:

      f_GR(t)       = standard PN inspiral frequency evolution,
                       computed from the (independently measured) chirp
                       mass of GW150914
      f_observed(t) = frequency track extracted from the actual strain

  A LIV/dispersion signature would show up as a SYSTEMATIC, frequency-
  dependent deviation between these two: delta_t(f) = t_observed(f) -
  t_GR(f), predicted to scale as delta_t(f) ~ Lambda * f^2 * K(z) at
  fixed source (Section on chromatic shadow in Paper 2 uses the same
  E^2 mechanism; here f = omega/2pi plays the role of E).

  5. Fit delta_t(f) vs f^2 by least squares -> extract Lambda, with
     proper error propagation from the frequency-extraction uncertainty.
  6. Report a bound whether or not a nonzero Lambda is detected --- a
     null result (Lambda consistent with zero) is reported explicitly
     as a bound, not hidden.

REQUIREMENTS:
    pip install h5py scipy numpy matplotlib

USAGE:
    python gwosc_chirp_dispersion_test.py \
        --h1 "H-H1_LOSC_4_V1-1126256640-4096.hdf5" \
        --l1 "L-L1_LOSC_4_V1-1126256640-4096.hdf5" \
        --event GW150914

Run with --help for all options. Designed to run on GWOSC strain files
already present on the user's machine (see file paths discussed in
chat); this script does not download anything.
"""

import argparse
import numpy as np
import h5py
from pathlib import Path
from scipy.signal import butter, filtfilt, hilbert, welch, get_window
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# ── Known event parameters (from GWTC-1 catalog, public) ──────────────────
EVENT_CATALOG = {
    "GW150914": dict(gps_merger=1126259462.4, m1=35.6, m2=30.6,
                      distance_Mpc=440, z=0.09),
    "GW170814": dict(gps_merger=1187008882.4, m1=30.5, m2=25.3,
                      distance_Mpc=540, z=0.11),
    "GW170817": dict(gps_merger=1187008882.4, m1=1.46, m2=1.27,
                      distance_Mpc=40, z=0.009),
}

G_SI = 6.674e-11
C_SI = 2.998e8
MSUN_SI = 1.989e30
MPC_SI = 3.0857e22


# ── Load strain ──────────────────────────────────────────────────────────
def load_strain(path, gps_center, half_window=16.0):
    """
    Load a segment of GWOSC strain data centered on gps_center, spanning
    +/- half_window seconds. Returns (time_array, strain_array, fs).
    """
    with h5py.File(path, "r") as f:
        strain = f["strain"]["Strain"]
        attrs = strain.attrs
        fs = 1.0 / attrs["Xspacing"]
        gps_start = attrs["Xstart"]
        n_total = attrs["Npoints"]

        idx_center = int(round((gps_center - gps_start) * fs))
        idx_half = int(round(half_window * fs))
        idx_lo = max(0, idx_center - idx_half)
        idx_hi = min(n_total, idx_center + idx_half)

        data = strain[idx_lo:idx_hi]
        t = gps_start + np.arange(idx_lo, idx_hi) / fs

    return t, np.array(data), fs


# ── Preprocessing: bandpass + whitening ────────────────────────────────────
def bandpass(data, fs, f_lo=20.0, f_hi=400.0, order=4):
    nyq = fs / 2
    b, a = butter(order, [f_lo / nyq, f_hi / nyq], btype="band")
    return filtfilt(b, a, data)


def whiten(data, fs, psd_seglen=4.0):
    """
    Simple whitening: estimate PSD via Welch, divide the FFT of the
    data by sqrt(PSD), transform back. Standard LIGO preprocessing step.
    """
    n = len(data)
    freqs_psd, psd = welch(data, fs=fs, nperseg=int(psd_seglen * fs))
    psd_interp = np.interp(np.fft.rfftfreq(n, d=1 / fs), freqs_psd, psd)
    psd_interp[psd_interp <= 0] = np.min(psd_interp[psd_interp > 0])

    data_fft = np.fft.rfft(data)
    white_fft = data_fft / np.sqrt(psd_interp * fs / 2)
    return np.fft.irfft(white_fft, n=n)


# ── Instantaneous frequency extraction ─────────────────────────────────────
def instantaneous_frequency(data, fs):
    """
    Extract the instantaneous frequency track via the analytic signal
    (Hilbert transform) phase derivative. Standard, robust method for
    chirp signals with reasonably good SNR.
    """
    analytic = hilbert(data)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) / (2 * np.pi) * fs
    t_freq = np.arange(len(inst_freq)) / fs
    return t_freq, inst_freq, np.abs(analytic)


# ── Post-Newtonian GR prediction for f(t) ───────────────────────────────────
def pn_chirp_mass(m1_msun, m2_msun):
    """Chirp mass in solar masses."""
    return (m1_msun * m2_msun) ** (3 / 5) / (m1_msun + m2_msun) ** (1 / 5)


def pn_frequency_of_time(t_before_merger, chirp_mass_msun):
    """
    Leading-order (Newtonian) post-Newtonian inspiral frequency as a
    function of time before merger:

        f(t) = (1/pi) * (5/(256*t))^(3/8) * (G*Mc/c^3)^(-5/8)

    t_before_merger must be POSITIVE (time remaining until merger).
    This is the standard 0PN quadrupole formula; sufficient for
    identifying gross dispersion effects near merger, where SNR peaks.
    """
    Mc_SI = chirp_mass_msun * MSUN_SI
    tau = np.maximum(t_before_merger, 1e-6)  # avoid singularity at t=0
    f = (1 / np.pi) * (5 / (256 * tau)) ** (3 / 8) * \
        (G_SI * Mc_SI / C_SI ** 3) ** (-5 / 8)
    return f


# ── Dispersion fit ──────────────────────────────────────────────────────────
def cosmological_K_factor(z):
    """
    K(z) = integral_0^z (1+z')^2 / H(z') dz'  [seconds]
    Same formula as Paper 2's FLRW extension.
    """
    from scipy.integrate import quad
    H0 = 67.4  # km/s/Mpc
    Om, OL = 0.315, 0.685

    def H_of_zp(zp):
        return H0 * np.sqrt(Om * (1 + zp) ** 3 + OL) * 1e3 / MPC_SI  # s^-1

    def integrand(zp):
        return (1 + zp) ** 2 / H_of_zp(zp)

    val, _ = quad(integrand, 0, z)
    return val  # seconds


def fit_lambda_from_delta_t(f_arr, delta_t_arr, delta_t_err, z):
    """
    Fit delta_t(f) = -(Lambda/2c^3) * (2*pi*f)^2 * K(z) + t0
    for Lambda (and a nuisance offset t0, since absolute alignment
    between observed and PN-predicted chirp has an arbitrary time
    origin degeneracy).
    """
    K = cosmological_K_factor(z)

    def model(f, Lambda, t0):
        omega = 2 * np.pi * f
        return -(Lambda / (2 * C_SI ** 3)) * omega ** 2 * K + t0

    popt, pcov = curve_fit(model, f_arr, delta_t_arr, sigma=delta_t_err,
                            p0=[0.0, 0.0], absolute_sigma=True)
    Lambda_fit, t0_fit = popt
    Lambda_err = np.sqrt(pcov[0, 0])

    return Lambda_fit, Lambda_err, t0_fit, K


# ── Main pipeline ────────────────────────────────────────────────────────────
def run_dispersion_test(h1_path, l1_path, event_name, out_dir, skip_whiten=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    if event_name not in EVENT_CATALOG:
        raise ValueError(f"Unknown event {event_name}, add it to EVENT_CATALOG")

    ev = EVENT_CATALOG[event_name]
    gps_merger = ev["gps_merger"]
    Mc = pn_chirp_mass(ev["m1"], ev["m2"])
    z = ev["z"]

    print("=" * 72)
    print(f"GWOSC CHIRP DISPERSION TEST — {event_name}")
    print("=" * 72)
    print()
    print(f"  GPS merger time: {gps_merger}")
    print(f"  Component masses: m1={ev['m1']} Msun, m2={ev['m2']} Msun")
    print(f"  Chirp mass: {Mc:.3f} Msun")
    print(f"  Redshift: z={z}")
    print()

    # ── Load and preprocess H1 ────────────────────────────────────────────
    print("Loading H1 strain...")
    t_h1, strain_h1, fs = load_strain(h1_path, gps_merger, half_window=16.0)
    print(f"  Loaded {len(strain_h1)} samples at fs={fs} Hz")

    if skip_whiten:
        strain_h1_bp = bandpass(strain_h1, fs, f_lo=20, f_hi=400)
        strain_h1_white = strain_h1_bp
        print("  (whitening skipped: --no-whiten)")
    else:
        # CRITICAL ORDER: whiten BEFORE bandpassing, not after.
        # Whitening divides by the PSD estimate; if applied AFTER
        # bandpass filtering, the PSD in the stopband is near-zero
        # (already filtered out), causing division-by-near-zero and
        # numerical explosion at those frequencies -- which then
        # dominates the "whitened" signal entirely. Standard LIGO
        # preprocessing whitens on the full band first, then bandpasses
        # the already-whitened data.
        strain_h1_white_full = whiten(strain_h1, fs)
        strain_h1_white = bandpass(strain_h1_white_full, fs, f_lo=20, f_hi=400)

    # ── Extract instantaneous frequency near merger ───────────────────────
    # Focus on the last ~0.5s before merger where SNR is highest and PN
    # is most reliable (avoid very early inspiral where SNR is too low
    # for reliable phase tracking in a single-detector, unfiltered
    # analysis like this). Stop safely BEFORE the merger GPS time to
    # avoid edge/ringdown artifacts contaminating the Hilbert transform
    # right where frequency changes fastest.
    idx_center = len(t_h1) // 2
    window_samples = int(0.5 * fs)
    margin_samples = int(0.01 * fs)  # stop 10ms before merger, avoid edge ringing
    idx_lo = max(0, idx_center - window_samples)
    idx_hi = max(idx_lo + 1, idx_center - margin_samples)

    seg = strain_h1_white[idx_lo:idx_hi]
    t_seg = t_h1[idx_lo:idx_hi]

    t_freq, inst_freq, envelope = instantaneous_frequency(seg, fs)
    t_freq_abs = t_seg[0] + t_freq

    # Smooth the raw instantaneous-frequency estimate with a short
    # median filter to suppress single-sample phase-noise spikes before
    # gating (standard practice for Hilbert-transform frequency tracks)
    from scipy.signal import medfilt
    mf_len = 15 if len(inst_freq) > 15 else (len(inst_freq) // 2) * 2 + 1
    if mf_len >= 3:
        inst_freq = medfilt(inst_freq, kernel_size=mf_len)

    # Keep only the physically reasonable inspiral band and where the
    # analytic-signal envelope is above a noise floor (crude SNR gate).
    # Use a LOWER percentile threshold than a matched-filter pipeline
    # would need, since single-detector Hilbert-transform envelope is a
    # much cruder SNR proxy; if this still yields too few points, the
    # printed diagnostic below shows the percentile-vs-yield tradeoff.
    env_threshold = np.percentile(envelope, 40)
    valid = (inst_freq > 20) & (inst_freq < 350) & \
            (envelope[:-1] > env_threshold)

    if valid.sum() < 10:
        print(f"  Diagnostic: only {valid.sum()} points at 40th percentile envelope cut.")
        print(f"  Trying looser threshold...")
        for pctl in [30, 20, 10, 5]:
            env_threshold = np.percentile(envelope, pctl)
            valid = (inst_freq > 20) & (inst_freq < 350) & \
                    (envelope[:-1] > env_threshold)
            print(f"    percentile {pctl}: {valid.sum()} points")
            if valid.sum() >= 10:
                break

    f_obs = inst_freq[valid]
    t_obs = t_freq_abs[valid]

    print(f"  Extracted {len(f_obs)} valid (f, t) points from chirp track")
    print()

    if len(f_obs) < 10:
        print("  WARNING: too few valid points for a reliable fit.")
        print("  This is expected for single-detector, unmatched-filter")
        print("  frequency tracking at moderate SNR -- a full LIGO-grade")
        print("  matched-filter pipeline would recover many more usable")
        print("  points. Proceeding with what is available, but treat")
        print("  the resulting bound as illustrative only.")
        print()

    # ── Compute PN-predicted time-of-arrival for each observed frequency ──
    tau_before_merger_obs = gps_merger - t_obs  # time remaining until merger
    # Invert f(t) numerically for t_PN(f): use a dense PN curve and
    # interpolate.
    tau_dense = np.logspace(-4, np.log10(2.0), 5000)
    f_dense = pn_frequency_of_time(tau_dense, Mc)
    # f_dense is decreasing as tau increases; sort for interpolation
    order = np.argsort(f_dense)
    tau_of_f = np.interp(f_obs, f_dense[order], tau_dense[order])

    t_PN_predicted = gps_merger - tau_of_f
    delta_t = t_obs - t_PN_predicted  # observed minus PN-predicted arrival

    # Crude error estimate: frequency-track jitter propagated through
    # df/dt of the PN model (standard error propagation)
    df_dt = np.gradient(f_dense[order], tau_dense[order])
    df_dt_at_f = np.interp(f_obs, f_dense[order], -df_dt)  # tau decreases as f increases
    f_jitter = np.std(np.diff(f_obs)) if len(f_obs) > 1 else 1.0
    delta_t_err = np.full_like(delta_t, 1.0 / fs) + f_jitter / np.maximum(df_dt_at_f, 1e-3)

    # ── Fit Lambda ──────────────────────────────────────────────────────
    print("Fitting Lambda from delta_t(f) vs f^2...")
    Lambda_fit, Lambda_err, t0_fit, K = fit_lambda_from_delta_t(
        f_obs, delta_t, delta_t_err, z)

    n_sigma = abs(Lambda_fit) / Lambda_err if Lambda_err > 0 else float("nan")

    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)
    print()
    print(f"  K(z={z}) = {K:.4e} s")
    print(f"  Fitted Lambda = {Lambda_fit:.4e} +/- {Lambda_err:.4e} m^2")
    print(f"  Significance vs Lambda=0: {n_sigma:.2f} sigma")
    print()

    if n_sigma < 2:
        print("  => CONSISTENT WITH Lambda=0 (no detected dispersion)")
        print(f"  => 95% upper bound: |Lambda| < {2*Lambda_err:.4e} m^2")
    else:
        print(f"  => {n_sigma:.1f}-sigma deviation from Lambda=0")
        print("  => Before interpreting as a detection, rule out systematic")
        print("     effects: calibration timing, PN order truncation,")
        print("     single-detector frequency-tracking bias.")

    print()
    print("  CAVEATS (read before citing this number):")
    print("  - Single-detector (H1 only), not a matched-filter pipeline")
    print("  - Leading-order (Newtonian) PN model, not full IMRPhenom/SEOBNR")
    print("  - Instantaneous-frequency extraction via Hilbert transform is")
    print("    noise-sensitive; a proper analysis would use a matched-filter")
    print("    or Bayesian time-frequency ridge (e.g. reduced-order quadrature)")
    print("  - This is a DEMONSTRATION of the test methodology on real GWOSC")
    print("    data, not a publication-grade LIGO/Virgo constraint.")

    # ── Plots ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(t_seg - gps_merger, seg, lw=0.5, color="steelblue")
    ax.set_xlabel("Time from merger [s]")
    ax.set_ylabel("Whitened strain (H1)")
    ax.set_title(f"{event_name}: whitened strain near merger")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_obs - gps_merger, f_obs, "o", color="firebrick", markersize=3,
             label="extracted from strain")
    tau_plot = np.logspace(-3, np.log10(0.5), 200)
    ax.plot(-tau_plot, pn_frequency_of_time(tau_plot, Mc), "k-", lw=1.5,
             label="PN prediction")
    ax.set_xlabel("Time from merger [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title("Chirp frequency track vs PN prediction")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.errorbar(f_obs, delta_t * 1e3, yerr=delta_t_err * 1e3, fmt="o",
                color="steelblue", markersize=4, alpha=0.6,
                label="observed - PN")
    f_smooth = np.linspace(f_obs.min(), f_obs.max(), 100)
    omega_smooth = 2 * np.pi * f_smooth
    model_curve = -(Lambda_fit / (2 * C_SI ** 3)) * omega_smooth ** 2 * K + t0_fit
    ax.plot(f_smooth, model_curve * 1e3, "r-", lw=2,
            label=fr"fit: $\Lambda$={Lambda_fit:.2e} m$^2$")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"$\Delta t$ [ms]")
    ax.set_title(r"Dispersion fit: $\Delta t$ vs $f$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Lambda-model dispersion test on real GWOSC data — {event_name}",
                 fontsize=11)
    plt.tight_layout()
    path = out_dir / f"gwosc_dispersion_{event_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved -> {path}")

    return dict(Lambda_fit=Lambda_fit, Lambda_err=Lambda_err,
                n_sigma=n_sigma, event=event_name, K=K)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h1", required=True, help="Path to H1 strain HDF5")
    parser.add_argument("--l1", required=False, default=None,
                         help="Path to L1 strain HDF5 (optional, not yet used)")
    parser.add_argument("--event", required=True, choices=list(EVENT_CATALOG.keys()))
    parser.add_argument("--out", default="gwosc_results", help="Output directory")
    parser.add_argument("--no-whiten", action="store_true",
                         help="Skip PSD whitening (for testing on synthetic/flat-noise data only; "
                              "always use whitening on real GWOSC data)")
    args = parser.parse_args()

    run_dispersion_test(args.h1, args.l1, args.event, args.out,
                         skip_whiten=args.no_whiten)


if __name__ == "__main__":
    main()
