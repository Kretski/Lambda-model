import os
import numpy as np
from scipy.stats import wasserstein_distance, entropy
from scipy.signal import welch

def load_wtwin_data(npz_path):
    data = np.load(npz_path)
    return (
        data['time'],
        data['ground_truth'],
        data['forecast_gr'],
        data['residual'],
        float(data['sample_rate'])
    )

def compute_spectral_entropy(signal, sample_rate, window_size=1024):
    """Изчислява спектралната ентропия за засичане на не-гаусова кохерентност."""
    freqs, psd = welch(signal, fs=sample_rate, nperseg=window_size)
    psd_norm = psd / np.sum(psd)
    # Нормирана ентропия (1.0 = чист бял шум, < 1.0 = налична структура/кохерентност)
    spec_entropy = entropy(psd_norm) / np.log(len(psd_norm))
    return spec_entropy

def sliding_wtwin_degradation(forecast, ground_truth, residual, sample_rate, win_sec=0.2, step_sec=0.05):
    """Изчислява W-Twin аномалния индекс (Wasserstein дистанция между forecast и real data)."""
    win_len = int(win_sec * sample_rate)
    step_len = int(step_sec * sample_rate)
    n_samples = len(residual)
    
    scores = []
    times = []
    
    for start in range(0, n_samples - win_len, step_len):
        end = start + win_len
        f_win = forecast[start:end]
        g_win = ground_truth[start:end]
        
        # W-Twin метрика: Wasserstein разстояние между разпределението на прогнозата и реалните данни
        w_dist = wasserstein_distance(f_win, g_win)
        scores.append(w_dist)
        times.append((start + win_len / 2) / sample_rate)
        
    return np.array(times), np.array(scores)

def analyze_detector(detector_name, file_path):
    print(f"\n================================================================================")
    print(f"[{detector_name}] W-TWIN ANOMALY & DEGRADATION ANALYSIS")
    print(f"================================================================================")
    
    time, ground_truth, forecast_gr, residual, sample_rate = load_wtwin_data(file_path)
    
    # 1. Спектрална ентропия на остатъка
    spec_ent = compute_spectral_entropy(residual, sample_rate)
    print(f"  Residual Spectral Entropy = {spec_ent:.6f}  (1.000 = Pure White Noise)")
    
    # 2. Плъзгащ се W-Twin аномален индекс
    w_times, w_scores = sliding_wtwin_degradation(forecast_gr, ground_truth, residual, sample_rate)
    
    bg_w_mean = np.mean(w_scores)
    bg_w_std = np.std(w_scores)
    max_w_score = np.max(w_scores)
    
    z_anomaly = (max_w_score - bg_w_mean) / bg_w_std if bg_w_std > 0 else 0.0
    
    print(f"  W-Twin Baseline Mean / Std = {bg_w_mean:.4e} / {bg_w_std:.4e}")
    print(f"  Peak W-Twin Anomaly Score  = {max_w_score:.4e} (Z-Score = {z_anomaly:+.2f}σ)")
    
    return residual, sample_rate, spec_ent, z_anomaly

def main():
    h1_path = os.path.join("wtwin_input", "H1_wtwin_data.npz")
    l1_path = os.path.join("wtwin_input", "L1_wtwin_data.npz")
    
    h1_res, sr, h1_ent, h1_z = analyze_detector("H1", h1_path)
    l1_res, _, l1_ent, l1_z = analyze_detector("L1", l1_path)
    
    # 3. Крос-детекторна кохерентност на остатъка (H1 vs L1)
    norm_h1 = h1_res / np.std(h1_res)
    norm_l1 = l1_res / np.std(l1_res)
    cross_corr = np.max(np.abs(np.correlate(norm_h1, norm_l1, mode='full'))) / len(norm_h1)
    
    print(f"\n================================================================================")
    print(f"CROSS-DETECTOR COHERENCE (H1 vs L1 Residuals)")
    print(f"================================================================================")
    print(f"  Max Cross-Correlation = {cross_corr:.4f}")
    
    if cross_corr > 0.15 and h1_ent < 0.98 and l1_ent < 0.98:
        print("  РЕЗУЛТАТ: Открита е кохерентна аномалия в остатъка между H1 и L1.")
    else:
        print("  РЕЗУЛТАТ: Остатъкът е статистически неразличим от чист инструментален шум.")

if __name__ == "__main__":
    main()