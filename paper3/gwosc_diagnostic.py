"""
gwosc_diagnostic.py
=======================
Quick diagnostic to see WHY zero points pass the validity gate on real
GWOSC data. Run this before re-running the main test.
"""
import sys
sys.path.insert(0, '.')
from gwosc_chirp_dispersion_test import *
import numpy as np

h1_path = r"C:\Users\Lenovo\Desktop\z\H-H1_LOSC_4_V1-1126256640-4096.hdf5"
gps_merger = 1126259462.4

t_h1, strain_h1, fs = load_strain(h1_path, gps_merger, half_window=16.0)
print(f"Loaded {len(strain_h1)} samples, fs={fs}")
print(f"Raw strain stats: min={strain_h1.min():.3e} max={strain_h1.max():.3e} std={strain_h1.std():.3e}")

strain_h1_bp = bandpass(strain_h1, fs, f_lo=20, f_hi=400)
print(f"After bandpass: std={strain_h1_bp.std():.3e}")

strain_h1_white = whiten(strain_h1_bp, fs)
print(f"After whitening: std={strain_h1_white.std():.3e}")

idx_center = len(t_h1) // 2
window_samples = int(0.5 * fs)
margin_samples = int(0.01 * fs)
idx_lo = max(0, idx_center - window_samples)
idx_hi = max(idx_lo + 1, idx_center - margin_samples)

seg = strain_h1_white[idx_lo:idx_hi]
t_seg = t_h1[idx_lo:idx_hi]
print(f"Segment length: {len(seg)} samples ({len(seg)/fs:.3f} s)")

t_freq, inst_freq, envelope = instantaneous_frequency(seg, fs)
print(f"\ninst_freq (raw, before median filter):")
print(f"  min={inst_freq.min():.1f} max={inst_freq.max():.1f} median={np.median(inst_freq):.1f}")
print(f"  in [20,350] band: {((inst_freq>20)&(inst_freq<350)).sum()} / {len(inst_freq)}")

from scipy.signal import medfilt
mf_len = 15 if len(inst_freq) > 15 else (len(inst_freq)//2)*2+1
inst_freq_smooth = medfilt(inst_freq, kernel_size=mf_len)
print(f"\ninst_freq (after median filter, kernel={mf_len}):")
print(f"  min={inst_freq_smooth.min():.1f} max={inst_freq_smooth.max():.1f} median={np.median(inst_freq_smooth):.1f}")
in_band = (inst_freq_smooth>20)&(inst_freq_smooth<350)
print(f"  in [20,350] band: {in_band.sum()} / {len(inst_freq_smooth)}")

print(f"\nenvelope stats: min={envelope.min():.3e} max={envelope.max():.3e}")
for pctl in [10, 30, 50, 60, 70, 90]:
    thresh = np.percentile(envelope, pctl)
    above = (envelope[:-1] > thresh) & in_band
    print(f"  percentile {pctl}: threshold={thresh:.3e}, points passing (band+env): {above.sum()}")
