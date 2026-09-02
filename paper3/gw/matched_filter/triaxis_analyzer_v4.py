#!/usr/bin/env python3
"""
TriAxis Analyzer v4 - Прецизен Ridge Tracking
==============================================
Ключови подобрения:
1. Viterbi алгоритъм за глобален ridge tracking
2. Автоматично центриране около inspiral-а
3. Подобрена chirp детекция
"""

import numpy as np
from scipy.signal import hilbert, correlate, spectrogram, find_peaks
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import hann

class TriAxisAnalyzerV4:
    """TriAxis v4 с Viterbi ridge tracking."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.sample_rate = self.config.get('sample_rate', 4096)
        self.window_size = self.config.get('window_size', 0.25)  # По-малък прозорец
        self.bandpass_low = self.config.get('bandpass_low', 30)
        self.bandpass_high = self.config.get('bandpass_high', 500)
        self.viterbi_freq_penalty = self.config.get('viterbi_freq_penalty', 50.0)
    
    def extract_window(self, h1_data, l1_data, times, center_time):
        """Извлича прозорец около center_time."""
        half_window = self.window_size / 2.0
        mask = (times >= center_time - half_window) & (times <= center_time + half_window)
        return h1_data[mask], l1_data[mask], times[mask]
    
    def viterbi_ridge_track(self, spectrogram_power, frequencies, times):
        """
        Viterbi алгоритъм за оптимално проследяване на ridge.
        
        Намира пътя през спектрограмата, който максимизира:
        - Енергията по пътя
        - Минимизира честотните скокове
        """
        n_freq, n_time = spectrogram_power.shape
        
        if n_freq < 2 or n_time < 2:
            return None, 0.0
        
        # Нормализиране на спектрограмата
        spec_norm = spectrogram_power / (np.max(spectrogram_power, axis=0, keepdims=True) + 1e-10)
        
        # Viterbi матрица: [n_freq, n_time]
        V = np.zeros((n_freq, n_time))
        backpointer = np.zeros((n_freq, n_time), dtype=int)
        
        # Инициализация - първа колона
        V[:, 0] = spec_norm[:, 0]
        
        # Попълване на Viterbi матрицата
        for t in range(1, n_time):
            for f in range(n_freq):
                # Преходни вероятности от всички честоти в t-1
                transition_costs = np.zeros(n_freq)
                
                for f_prev in range(n_freq):
                    # Енергия на текущата позиция
                    energy = spec_norm[f, t]
                    
                    # Преходна цена (по-малка за близки честоти)
                    freq_jump = abs(f - f_prev)
                    # ПОПРАВКА: старата формула (freq_jump/n_freq * viterbi_freq_penalty,
                    # с default viterbi_freq_penalty=50.0) даваше penalty от порядъка
                    # на 3-5 за единичен bin скок, докато spec_norm е нормализиран до
                    # макс 1.0 на колона — санкцията математически надвишаваше ВСЯКА
                    # възможна енергийна печалба, гарантирайки напълно плосък ridge
                    # независимо от сигнала (потвърдено емпирично: ridge_min==ridge_max
                    # за GW150914/170809/170814, и при zero-phase, и при causal
                    # preprocessing — бъгът е в Viterbi, не в whitening/bandpass).
                    # Новата скала държи penalty в единици, сравними с [0,1] енергията.
                    transition_penalty = freq_jump * (self.viterbi_freq_penalty / 1000.0)
                    
                    # Обща цена
                    transition_costs[f_prev] = V[f_prev, t-1] + energy - transition_penalty
                
                # Избор на най-добър преход
                best_prev = np.argmax(transition_costs)
                V[f, t] = transition_costs[best_prev]
                backpointer[f, t] = best_prev
        
        # Намиране на най-добрия краен път
        best_final_freq = np.argmax(V[:, -1])
        best_path_score = V[best_final_freq, -1] / n_time
        
        # Реконструиране на пътя
        ridge_indices = np.zeros(n_time, dtype=int)
        ridge_indices[-1] = best_final_freq
        
        for t in range(n_time-2, -1, -1):
            ridge_indices[t] = backpointer[ridge_indices[t+1], t+1]
        
        # Конвертиране до честоти
        ridge_frequencies = frequencies[ridge_indices]
        
        # Изчисляване на качеството на ridge-а
        ridge_quality = np.mean(spec_norm[ridge_indices, np.arange(n_time)])
        
        return ridge_frequencies, ridge_quality, best_path_score
    
    def time_axis_viterbi(self, h1_window):
        """
        Time ос с Viterbi ridge tracking.
        """
        # Спектрограма с добра резолюция
        nperseg = max(32, min(128, len(h1_window) // 3))
        noverlap = nperseg * 7 // 8  # 87.5% overlap
        
        frequencies, times, Sxx = spectrogram(
            h1_window, fs=self.sample_rate,
            nperseg=nperseg, noverlap=noverlap,
            window=hann(nperseg)
        )
        
        # Филтриране в GW обхвата
        freq_mask = (frequencies >= self.bandpass_low) & (frequencies <= self.bandpass_high)
        freq_filtered = frequencies[freq_mask]
        Sxx_filtered = Sxx[freq_mask, :]
        
        if len(freq_filtered) < 3 or Sxx_filtered.shape[1] < 3:
            return self._empty_time_features()
        
        # Изглаждане
        Sxx_smooth = gaussian_filter1d(Sxx_filtered, sigma=0.8, axis=0)
        
        # Viterbi ridge tracking
        result = self.viterbi_ridge_track(Sxx_smooth, freq_filtered, times)
        
        if result is None:
            return self._empty_time_features()
        
        ridge_freq, ridge_quality, viterbi_score = result
        
        # Анализ на ridge-а
        t_norm = times - times[0]
        duration = times[-1] - times[0]
        
        # Проверка за монотонност
        freq_diff = np.diff(ridge_freq)
        is_monotonic_rising = np.all(freq_diff > 0)
        is_monotonic_falling = np.all(freq_diff < 0)
        
        # Полиномни фитове
        if len(ridge_freq) > 3:
            # Линеен фит
            coeffs_lin = np.polyfit(t_norm, ridge_freq, 1)
            df_dt = coeffs_lin[0]
            pred_lin = np.polyval(coeffs_lin, t_norm)
            r2_lin = 1 - np.sum((ridge_freq - pred_lin)**2) / (np.sum((ridge_freq - np.mean(ridge_freq))**2) + 1e-10)
            
            # Квадратичен фит
            coeffs_quad = np.polyfit(t_norm, ridge_freq, 2)
            pred_quad = np.polyval(coeffs_quad, t_norm)
            r2_quad = 1 - np.sum((ridge_freq - pred_quad)**2) / (np.sum((ridge_freq - np.mean(ridge_freq))**2) + 1e-10)
            
            # Ускорение
            if len(coeffs_quad) > 2:
                acceleration = coeffs_quad[0] * 2  # Втора производна на a*t²
            else:
                acceleration = 0.0
        else:
            df_dt = 0.0
            r2_lin = 0.0
            r2_quad = 0.0
            acceleration = 0.0
        
        # Честотен обхват
        freq_start = ridge_freq[0]
        freq_end = ridge_freq[-1]
        # ПОПРАВКА: max-min вместо last-first. При symmetric-wing ringing
        # артефакт (ridge се качва, после пада обратно) last-first дава
        # подвеждащо малък или дори отрицателен span, макар реалната
        # честотна екскурзия да е голяма.
        freq_span = float(np.max(ridge_freq) - np.min(ridge_freq))
        
        # Ridge стабилност
        ridge_std = np.std(np.diff(ridge_freq))
        ridge_stability = 1.0 / (1.0 + ridge_std / 50.0)
        
        # Chirp метрики
        chirp_score = 0.0
        
        # Възходящ chirp (df/dt > 0)
        if is_monotonic_rising and df_dt > 50:
            chirp_score += 0.3
        elif df_dt > 0 and r2_quad > 0.3:
            chirp_score += 0.2
        
        # Реалистичен честотен обхват за GW chirp
        if 50 < freq_span < 400:
            chirp_score += 0.2
        elif 20 < freq_span < 500:
            chirp_score += 0.1
        
        # Продължителност (GW chirps: 50-300ms)
        chirp_duration_ms = duration * 1000
        if 40 < chirp_duration_ms < 400:
            chirp_score += 0.15
        
        # Ускорение (chirp-овете се ускоряват)
        if acceleration > 0 and df_dt > 0:
            chirp_score += 0.15
        
        # Качество на ridge-а
        if ridge_quality > 0.4:
            chirp_score += 0.1
        if viterbi_score > 0.3:
            chirp_score += 0.1
        
        chirp_score = min(1.0, chirp_score)
        has_chirp = chirp_score > 0.5
        
        # Определяне на посоката на chirp-а
        if is_monotonic_rising:
            chirp_direction = "rising"
        elif is_monotonic_falling:
            chirp_direction = "falling"
        else:
            chirp_direction = "mixed"
        
        return {
            'df_dt': df_dt,
            'chirp_duration_ms': chirp_duration_ms,
            'freq_start': freq_start,
            'freq_end': freq_end,
            'freq_span': freq_span,
            'r2_linear': r2_lin,
            'r2_quadratic': r2_quad,
            'acceleration': acceleration,
            'is_monotonic_rising': is_monotonic_rising,
            'ridge_quality': ridge_quality,
            'viterbi_score': viterbi_score,
            'ridge_stability': ridge_stability,
            'chirp_score': chirp_score,
            'has_chirp': has_chirp,
            'chirp_direction': chirp_direction,
            'ridge_frequencies': ridge_freq,
            'ridge_times': times
        }
    
    def _empty_time_features(self):
        """Празни time features."""
        return {
            'df_dt': 0.0,
            'chirp_duration_ms': 0.0,
            'freq_start': 0.0,
            'freq_end': 0.0,
            'freq_span': 0.0,
            'r2_linear': 0.0,
            'r2_quadratic': 0.0,
            'acceleration': 0.0,
            'is_monotonic_rising': False,
            'ridge_quality': 0.0,
            'viterbi_score': 0.0,
            'ridge_stability': 0.0,
            'chirp_score': 0.0,
            'has_chirp': False,
            'chirp_direction': 'none',
            'ridge_frequencies': np.array([]),
            'ridge_times': np.array([])
        }
    
    def structure_axis(self, h1_window, l1_window):
        """Подобрена Structure ос."""
        # Multi-resolution entropy
        entropies = []
        window_sizes = [len(h1_window)//2, len(h1_window)//4, len(h1_window)//8]
        
        for n_fft in window_sizes:
            if n_fft < 16:
                continue
            fft_h1 = np.abs(np.fft.rfft(h1_window, n=n_fft))
            fft_l1 = np.abs(np.fft.rfft(l1_window, n=n_fft))
            
            psd_h1 = fft_h1**2 / (np.sum(fft_h1**2) + 1e-10)
            psd_l1 = fft_l1**2 / (np.sum(fft_l1**2) + 1e-10)
            
            ent_h1 = entropy(psd_h1 + 1e-10)
            ent_l1 = entropy(psd_l1 + 1e-10)
            entropies.append((ent_h1 + ent_l1) / 2.0)
        
        avg_entropy = np.mean(entropies) if entropies else 5.0
        entropy_consistency = 1.0 / (1.0 + np.std(entropies)) if len(entropies) > 1 else 0.5
        
        # Cross-spectral анализ
        from scipy.signal import csd, coherence
        
        nperseg = min(128, len(h1_window)//2)
        if nperseg >= 16:
            f, Pxy = csd(h1_window, l1_window, fs=self.sample_rate,
                        nperseg=nperseg, noverlap=nperseg//2)
            
            phase_xy = np.angle(Pxy)
            phase_unwrapped = np.unwrap(phase_xy)
            
            # Фазова линейност
            if len(phase_unwrapped) > 3:
                coeffs = np.polyfit(f, phase_unwrapped, 1)
                phase_linearity = max(0, 1.0 - np.std(phase_unwrapped - np.polyval(coeffs, f)) / np.pi)
            else:
                phase_linearity = 0.0
            
            # Кохерентност
            f_coh, coh = coherence(h1_window, l1_window, fs=self.sample_rate,
                                  nperseg=nperseg, noverlap=nperseg//2)
            freq_mask = (f_coh >= 30) & (f_coh <= 500)
            mean_coherence = np.mean(coh[freq_mask]) if np.any(freq_mask) else 0.0
        else:
            phase_linearity = 0.0
            mean_coherence = 0.0
        
        # Локална консистентност
        h1_norm = (h1_window - np.mean(h1_window)) / (np.std(h1_window) + 1e-10)
        l1_norm = (l1_window - np.mean(l1_window)) / (np.std(l1_window) + 1e-10)
        
        local_corrs = []
        chunk_size = max(16, len(h1_window) // 6)
        for i in range(0, len(h1_window) - chunk_size, chunk_size//2):
            chunk_h1 = h1_norm[i:i+chunk_size]
            chunk_l1 = l1_norm[i:i+chunk_size]
            if len(chunk_h1) > 10:
                local_corr = np.corrcoef(chunk_h1, chunk_l1)[0, 1]
                if not np.isnan(local_corr):
                    local_corrs.append(abs(local_corr))
        
        mean_local_corr = np.mean(local_corrs) if local_corrs else 0.0
        local_consistency = mean_local_corr * (1.0 - min(np.std(local_corrs) if len(local_corrs) > 1 else 0.0, 1.0))
        
        return {
            'avg_entropy': avg_entropy,
            'entropy_consistency': entropy_consistency,
            'phase_linearity': phase_linearity,
            'mean_coherence': mean_coherence,
            'local_consistency': local_consistency
        }
    
    def coherence_axis(self, h1_window, l1_window):
        """Coherence ос."""
        h1_norm = (h1_window - np.mean(h1_window)) / (np.std(h1_window) + 1e-10)
        l1_norm = (l1_window - np.mean(l1_window)) / (np.std(l1_window) + 1e-10)
        
        correlation = correlate(h1_norm, l1_norm, mode='same')
        correlation /= len(h1_window)
        
        max_corr = np.max(np.abs(correlation))
        max_idx = np.argmax(np.abs(correlation))
        center_idx = len(correlation) // 2
        delay_ms = ((max_idx - center_idx) / self.sample_rate) * 1000
        
        realistic_delay = abs(delay_ms) < 10.0
        
        half_max = max_corr / 2.0
        above_half = np.abs(correlation) > half_max
        width_ms = (np.sum(above_half) / self.sample_rate) * 1000
        
        return {
            'max_corr': max_corr,
            'delay_ms': delay_ms,
            'width_ms': width_ms,
            'realistic_delay': realistic_delay
        }
    
    def compute_scores(self, struct_feat, coh_feat, time_feat):
        """Изчислява финалните скорове."""
        structure_score = (
            # ПОПРАВКА: phase_linearity и mean_coherence са H1-L1
            # CROSS-detector мерки — вече се мерят директно в
            # coherence_axis (45% от triaxis_score). Тук дублираха
            # сигнала и разводняваха единствената истинска
            # single-detector морфологична мярка (ентропия), заради
            # което Structure излизаше под background за реални събития.
            # Новите тежести държат Structure фокусиран само върху
            # ЕДИН детектор: колко концентрирана/структурирана е
            # енергията в самия H1/L1 сигнал, независимо от другия.
            np.clip(1.0 / (1.0 + struct_feat['avg_entropy']/3.0), 0, 1) * 0.55 +
            np.clip(struct_feat['entropy_consistency'], 0, 1) * 0.25 +
            np.clip(struct_feat['local_consistency'], 0, 1) * 0.20
            # phase_linearity, mean_coherence умишлено премахнати оттук —
            # остават достъпни в struct_feat за диагностика, но не влизат
            # в score-а, за да не дублират coherence_axis.
        )
        
        coherence_score = (
            np.clip(coh_feat['max_corr'] * 3.0, 0, 1) * 0.5 +
            (0.5 if coh_feat['realistic_delay'] else 0.0)
        )
        
        time_score = (
            np.clip(time_feat['chirp_score'], 0, 1) * 0.6 +
            np.clip(time_feat['ridge_quality'], 0, 1) * 0.2 +
            np.clip(time_feat['viterbi_score'] * 2.0, 0, 1) * 0.2
        )
        
        triaxis_score = (
            0.30 * structure_score +
            0.45 * coherence_score +
            0.25 * time_score
        )
        
        return {
            'structure_score': structure_score,
            'coherence_score': coherence_score,
            'time_score': time_score,
            'triaxis_score': triaxis_score
        }
    
    def analyze(self, h1_strain, l1_strain, times, center_time):
        """Основен анализ."""
        h1_window, l1_window, window_times = self.extract_window(
            h1_strain, l1_strain, times, center_time
        )
        
        if len(h1_window) < 64:
            return None
        
        struct_features = self.structure_axis(h1_window, l1_window)
        coh_features = self.coherence_axis(h1_window, l1_window)
        time_features = self.time_axis_viterbi(h1_window)
        
        scores = self.compute_scores(struct_features, coh_features, time_features)
        
        all_features = {}
        all_features.update(struct_features)
        all_features.update(coh_features)
        all_features.update(time_features)
        all_features.update(scores)
        all_features['window_size'] = len(h1_window)
        all_features['window_times'] = window_times
        all_features['h1_window'] = h1_window
        all_features['l1_window'] = l1_window
        
        return all_features
    
    def generate_background(self, h1_strain, l1_strain, times, n_samples=300,
                           exclude_center=None, exclude_window=2.0):
        """Генерира background."""
        bg_scores = []
        for _ in range(n_samples):
            while True:
                random_time = np.random.uniform(
                    times[0] + self.window_size,
                    times[-1] - self.window_size
                )
                if exclude_center is None or abs(random_time - exclude_center) > exclude_window:
                    break
            
            result = self.analyze(h1_strain, l1_strain, times, random_time)
            if result:
                bg_scores.append(result['triaxis_score'])
        return bg_scores
    
    def print_summary(self, features, bg_scores=None):
        """Принтира резултати."""
        print("\n" + "="*70)
        print("TRIAXIS v4 ANALYSIS (with Viterbi Ridge Tracking)")
        print("="*70)
        
        print(f"\nПрозорец: {features['window_size']} samples")
        
        print("\n1. STRUCTURE (30%)")
        print("-"*40)
        print(f"  Multi-res entropy:     {features['avg_entropy']:.2f}")
        print(f"  Entropy consistency:   {features['entropy_consistency']:.3f}")
        print(f"  Phase linearity:       {features['phase_linearity']:.3f}")
        print(f"  Mean coherence:        {features['mean_coherence']:.3f}")
        print(f"  Local consistency:     {features['local_consistency']:.3f}")
        print(f"  → Score:              {features['structure_score']:.3f}")
        
        print("\n2. COHERENCE (45%)")
        print("-"*40)
        print(f"  Max correlation:       {features['max_corr']:.3f}")
        print(f"  Delay:                 {features['delay_ms']:.2f} ms")
        print(f"  Width:                 {features['width_ms']:.2f} ms")
        print(f"  Realistic delay:       {features['realistic_delay']}")
        print(f"  → Score:              {features['coherence_score']:.3f}")
        
        print("\n3. TIME (25%) - Viterbi Ridge")
        print("-"*40)
        print(f"  df/dt:                 {features['df_dt']:.1f} Hz/s")
        print(f"  Direction:             {features['chirp_direction']}")
        print(f"  Freq range:            {features['freq_start']:.0f} → {features['freq_end']:.0f} Hz")
        print(f"  Duration:              {features['chirp_duration_ms']:.1f} ms")
        print(f"  R² (linear):           {features['r2_linear']:.3f}")
        print(f"  R² (quadratic):        {features['r2_quadratic']:.3f}")
        print(f"  Acceleration:          {features['acceleration']:.1f} Hz/s²")
        print(f"  Monotonic rising:      {features['is_monotonic_rising']}")
        print(f"  Ridge quality:         {features['ridge_quality']:.3f}")
        print(f"  Viterbi score:         {features['viterbi_score']:.3f}")
        print(f"  Chirp score:           {features['chirp_score']:.3f}")
        print(f"  Has chirp:             {features['has_chirp']}")
        print(f"  → Score:              {features['time_score']:.3f}")
        
        print("\n" + "="*70)
        print(f"TRIAXIS SCORE:          {features['triaxis_score']:.3f}")
        
        if bg_scores:
            percentile = (np.sum(np.array(bg_scores) < features['triaxis_score']) / 
                         len(bg_scores)) * 100
            sigma = ((features['triaxis_score'] - np.mean(bg_scores)) / 
                    (np.std(bg_scores) + 1e-10))
            
            print(f"\nBACKGROUND (n={len(bg_scores)})")
            print(f"  Median:               {np.median(bg_scores):.3f}")
            print(f"  Mean:                 {np.mean(bg_scores):.3f}")
            print(f"  Std:                  {np.std(bg_scores):.3f}")
            print(f"  Percentile:           {percentile:.1f}%")
            print(f"  Significance:         {sigma:.1f}σ")
        
        print("="*70 + "\n")