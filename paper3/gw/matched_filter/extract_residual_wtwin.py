import argparse
import os
import numpy as np
import h5py
from scipy.signal import welch
from scipy.signal.windows import tukey

def load_gwosc_hdf5(filepath):
    with h5py.File(filepath, 'r') as f:
        strain = f['strain']['Strain'][:]
        gps_start = f['meta']['GPSstart'][()]
        duration = f['meta']['Duration'][()]
        sample_rate = len(strain) / duration
    return strain, gps_start, sample_rate

def compute_psd(strain, sample_rate, nperseg=4096*4):
    freqs, psd = welch(strain, fs=sample_rate, nperseg=int(nperseg), window='hann')
    return freqs, psd

def build_frequency_mask(freqs, f_low=20.0, f_high=300.0, notch_freqs=None, notch_width=1.0):
    if notch_freqs is None:
        notch_freqs = [60.0, 120.0, 180.0, 240.0, 300.0]
    mask = (freqs >= f_low) & (freqs <= f_high)
    for line in notch_freqs:
        mask &= ~((freqs >= line - notch_width) & (freqs <= line + notch_width))
    return mask

def generate_gr_template_fd(freqs):
    f_safe = np.maximum(freqs, 1.0)
    amp = f_safe ** (-7/6)
    phase = 2.0 * np.pi * f_safe
    return amp * np.exp(-1j * phase)

def process_detector_whitened(filepath, detector_name, target_gps=1126259462.0, seg_duration=16.0, out_dir="wtwin_input"):
    print(f"\n[{detector_name}] Извличане на ИЗБЕЛЕН остатък (Whitened Residual)...")
    strain, gps_start, sample_rate = load_gwosc_hdf5(filepath)
    
    psd_freqs, psd = compute_psd(strain, sample_rate)
    n_points = int(seg_duration * sample_rate)
    freqs = np.fft.rfftfreq(n_points, d=1.0/sample_rate)
    df = freqs[1] - freqs[0]
    
    psd_interp = np.interp(freqs, psd_freqs, psd)
    psd_safe = np.copy(psd_interp)
    psd_safe[psd_safe <= 0] = np.inf
    
    valid_mask = build_frequency_mask(freqs, f_low=20.0, f_high=300.0)
    
    # Извличане на сегмента с Tukey прозорец
    start_idx = int(round((target_gps - seg_duration / 2.0 - gps_start) * sample_rate))
    seg = strain[start_idx:start_idx + n_points].copy()
    seg -= np.mean(seg)
    seg *= tukey(len(seg), alpha=0.1)
    
    seg_fd = np.fft.rfft(seg) / sample_rate
    tmpl_fd = generate_gr_template_fd(freqs)
    
    # Нормиране на шаблона
    sigmasq = 4.0 * df * np.sum((np.abs(tmpl_fd[valid_mask]) ** 2) / psd_safe[valid_mask])
    u_fd = tmpl_fd / np.sqrt(sigmasq)
    
    # Напасване на фаза и амплитуда
    corr_fd = np.zeros_like(seg_fd, dtype=np.complex128)
    corr_fd[valid_mask] = (seg_fd[valid_mask] * np.conj(u_fd[valid_mask])) / psd_safe[valid_mask]
    snr_series = np.fft.irfft(corr_fd, n=n_points) * (4.0 * sample_rate)
    best_idx = np.argmax(np.abs(snr_series))
    matched_z = snr_series[best_idx]
    
    shift_phase = np.exp(-2j * np.pi * freqs * (best_idx / sample_rate))
    aligned_tmpl_fd = u_fd * matched_z * shift_phase
    
    # ИЗБЕЛВАНЕ (WHITENING) на данните и шаблона
    seg_fd_whitened = np.zeros_like(seg_fd, dtype=np.complex128)
    tmpl_fd_whitened = np.zeros_like(tmpl_fd, dtype=np.complex128)
    
    seg_fd_whitened[valid_mask] = seg_fd[valid_mask] / np.sqrt(psd_safe[valid_mask] / 2.0)
    tmpl_fd_whitened[valid_mask] = aligned_tmpl_fd[valid_mask] / np.sqrt(psd_safe[valid_mask] / 2.0)
    
    residual_fd_whitened = seg_fd_whitened - tmpl_fd_whitened
    
    # Връщане във времевия домейн
    ground_truth_time = np.fft.irfft(seg_fd_whitened, n=n_points)
    forecast_time = np.fft.irfft(tmpl_fd_whitened, n=n_points)
    residual_time = np.fft.irfft(residual_fd_whitened, n=n_points)
    
    out_file = os.path.join(out_dir, f"{detector_name}_wtwin_data.npz")
    np.savez(
        out_file,
        time=np.linspace(-seg_duration/2, seg_duration/2, n_points),
        ground_truth=ground_truth_time,
        forecast_gr=forecast_time,
        residual=residual_time,
        sample_rate=sample_rate
    )
    print(f"  Запазен избелен остатък в: {out_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h1-data", type=str, required=True)
    parser.add_argument("--l1-data", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="wtwin_input")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    process_detector_whitened(args.h1_data, "H1", out_dir=args.out_dir)
    process_detector_whitened(args.l1_data, "L1", out_dir=args.out_dir)

if __name__ == "__main__":
    main()