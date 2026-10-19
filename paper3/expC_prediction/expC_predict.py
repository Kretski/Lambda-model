#!/usr/bin/env python3
"""EXP C prediction (pre-registered): counter-rotating resonances omega_q of the Smaniotto et al.
model (arXiv:2502.11209, Methods Eqs. 4-17) for the EXP C flow parameters of Table SI
(C = 583 mm^2/s, Omega = 0.10 rad/s, h0 = 28 mm, gamma = 2.25e-6 m^3/s^2, r_B = 37.3 mm).
Uses the frozen EXP B machinery (stage5I_3_resonance_v2.py, stage5I_4_full_m_scan_v6_FIXED.py)
unchanged; only the parameter module differs. No experimental or reference frequencies are used:
the EXP B regression gate and the data-derived frequency window are replaced by a fixed,
model-only window. Output: expC_predictions.csv (to be committed and tagged before any EXP C data)."""
import csv, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage5I_1_2_dispersion_radial_p as P
import stage5I_4_full_m_scan_v6_FIXED as S

F_MIN, F_MAX = 2.0, 14.0                     # Hz, fixed model-only search window
M_VALUES = [int(a) for a in sys.argv[1:]] or list(range(-21, -3))
OUT = HERE / "expC_predictions.csv"

print(f"EXP C parameters: C={P.C_METHODS*1e6:.0f} mm^2/s Omega={P.OMEGA} h0={P.H0*1e3:.0f} mm "
      f"gamma={P.GAMMA:.2e} r_B={P.R_B*1e3:.1f} mm; window [{F_MIN}, {F_MAX}] Hz; m={M_VALUES}")
rows = []
for m in M_VALUES:
    t0 = time.time()
    r_sp, w_lr = P.find_light_ring(m)
    f_lr = w_lr/(2*np.pi) if w_lr is not None else float('nan')
    poles = S.solve_one_m(m, F_MIN, F_MAX)
    for q, p in enumerate(poles, start=1):
        rows.append(dict(m=m, q=q, f_re_hz=p["f_re"], f_im_hz=p["f_im"], abs_res=p["abs_res"],
                         light_ring_hz=f_lr, root_success=p["root_success"]))
        print(f"  m={m:4d} q={q} f={p['f_re']:.6f} {p['f_im']:+.6f}i Hz |Res|={p['abs_res']:.1e}  (f_lr={f_lr:.4f})")
    print(f"  m={m}: {len(poles)} poles, {time.time()-t0:.0f} s", flush=True)
with OUT.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["m"]); w.writeheader(); w.writerows(rows)
print(f"wrote {OUT} ({len(rows)} poles)")
