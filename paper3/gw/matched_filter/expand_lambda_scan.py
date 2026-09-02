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

def generate_lambda_template_fd(freqs, lambda_val, phase_scale=1e-6):
    """Генерира фазово модифициран шаблон с дисперсионен член Lambda."""
    f_safe = np.maximum(freqs, 1.0)
    amp = f_safe ** (-7/6)
    # Дисперсионна фазова корекция
    phase_gr = 2.0 * np.pi * f_safe
    phase_lambda = lambda_val * phase_scale * (f_safe ** (-1/3))
    return amp * np.exp(-1j * (phase_gr + phase_lambda))

def evaluate_grid(seg_fd, freqs, psd_safe, valid_mask, df, sample_rate, n_points, lambda_grid):
    best_snr = -1.0
    best_lambda = None
    snr_profile = []

    for l_val in lambda_grid:
        tmpl_fd = generate_lambda_template_fd(freqs, l_val)
        sigmasq = 4.0 * df * np.sum((np.abs(tmpl_fd[valid_mask]) ** 2) / psd_safe[valid_mask])
        u_fd = tmpl_fd / np.sqrt(sigmasq)

        corr_fd = np.zeros_like(seg_fd, dtype=np.complex128)
        corr_fd[valid_mask] = (seg_fd[valid_mask] * np.conj(u_fd[valid_mask])) / psd_safe[valid_mask]
        snr_series = np.abs(np.fft.irfft(corr_fd, n=n_points) * (4.0 * sample_rate))
        
        max_snr = np.max(snr_series)
        snr_profile.append(max_snr)
        
        if max_snr > best_snr:
            best_snr = max_snr
            best_lambda = l_val

    return best_lambda, best_snr, np.array(snr_profile)

def run_adaptive_scan(filepath, detector_name, target_gps=1126259462.0, seg_duration=16.0):
    print(f"\n[{detector_name}] Стартиране на Coarse-to-Fine Profile Likelihood сканиране...")
    strain, gps_start, sample_rate = load_gwosc_hdf5(filepath)
    psd_freqs, psd = compute_psd(strain, sample_rate)
    
    n_points = int(seg_duration * sample_rate)
    freqs = np.fft.rfftfreq(n_points, d=1.0/sample_rate)
    df = freqs[1] - freqs[0]
    
    psd_safe = np.interp(freqs, psd_freqs, psd)
    psd_safe[psd_safe <= 0] = np.inf
    
    valid_mask = (freqs >= 20.0) & (freqs <= 300.0)
    for line in [60.0, 120.0, 180.0, 240.0, 300.0]:
        valid_mask &= ~((freqs >= line - 1.0) & (freqs <= line + 1.0))
        
    start_idx = int(round((target_gps - seg_duration / 2.0 - gps_start) * sample_rate))
    seg = strain[start_idx:start_idx + n_points].copy()
    seg -= np.mean(seg)
    seg *= tukey(len(seg), alpha=0.1)
    seg_fd = np.fft.rfft(seg) / sample_rate

    # 1. GR Базова стойност (Lambda = 0)
    _, snr_gr, _ = evaluate_grid(seg_fd, freqs, psd_safe, valid_mask, df, sample_rate, n_points, [0.0])

    # 2. Груб скан (Coarse Scan: -1000 до +1000, стъпка 50)
    coarse_grid = np.arange(-1000, 1050, 50)
    best_coarse_l, best_coarse_snr, _ = evaluate_grid(seg_fd, freqs, psd_safe, valid_mask, df, sample_rate, n_points, coarse_grid)
    print(f"  [Coarse Scan] Най-добра стойност: Λ = {best_coarse_l} (SNR = {best_coarse_snr:.4f})")

    # 3. Фин скан (Fine Scan около намерения оптимум: +/- 100, стъпка 2)
    fine_grid = np.arange(best_coarse_l - 100, best_coarse_l + 102, 2)
    best_fine_l, best_fine_snr, _ = evaluate_grid(seg_fd, freqs, psd_safe, valid_mask, df, sample_rate, n_points, fine_grid)
    print(f"  [Fine Scan]   Окончателен оптимум: Λ = {best_fine_l} (SNR = {best_fine_snr:.4f})")

    # 4. Статистика на логаритмичното съотношение на вероятностите (Delta Log-Likelihood)
    # Δln L ≈ 0.5 * (SNR_best^2 - SNR_GR^2)
    delta_log_l = 0.5 * (best_fine_snr**2 - snr_gr**2)
    
    print(f"  ------------------------------------------------------------------")
    print(f"  GR SNR (Λ = 0):        {snr_gr:.4f}")
    print(f"  Max SNR (Λ = {best_fine_l}):  {best_fine_snr:.4f}")
    print(f"  Δln L (Log-Likelihood Gain): {delta_log_l:+.4f}")
    
    if np.abs(best_fine_l) in [1000, -1000]:
        print("  ВНИМАНИЕ: Оптимумът все още удря границата! Скалата или фазовият модел са неограничени.")
    else:
        print("  ОК: Намерен е вътрешен екстремум на функцията за вероятност.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h1-data", type=str, required=True)
    parser.add_argument("--l1-data", type=str, required=True)
    args = parser.parse_args()

    if os.path.exists(args.h1_data):
        run_adaptive_scan(args.h1_data, "H1")
    if os.path.exists(args.l1_data):
        run_adaptive_scan(args.l1_data, "L1")

if __name__ == "__main__":
    main()