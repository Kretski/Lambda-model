import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stage5I_4_full_m_scan_v6_FIXED import safe_complex_res

# ============================================================
# STAGE 5I-22 -- DENSE ENDPOINT SCAN
#
# Goal:
#   Track min |R| extremely close to the suspected endpoint
#   and determine whether a root disappears at a sharply
#   localized finite m.
#
# Exact residual:
#   safe_complex_res()
#
# No substitute residual.
# No q label.
# No claim of new physics.
# ============================================================

ROOT_RE = 9.6046425466
ROOT_IM = -0.3031465788
ROOT_M   = -14.7500

# Last known endpoint estimate
MC_EST = -14.79415

# Dense m grid around endpoint
M_VALUES = np.array([
    -14.78000,
    -14.78500,
    -14.79000,
    -14.79200,
    -14.79300,
    -14.79400,
    -14.79405,
    -14.79410,
    -14.79412,
    -14.79414,
    -14.79415,
    -14.79416,
    -14.79418,
    -14.79420,
    -14.79425,
    -14.79430,
])

# Local prediction using the already validated slope near endpoint
DRE_DM = -0.27183
DIM_DM = -0.00670

# Window around predicted branch location
HALF_WINDOW_RE = 0.004
HALF_WINDOW_IM = 0.004

# Dense grid
N = 81

print("=" * 100)
print("STAGE 5I-22 -- DENSE ENDPOINT SCAN")
print("=" * 100)
print()
print("Exact residual: stage5I-4 -> safe_complex_res()")
print(f"Endpoint estimate: m_c ~ {MC_EST:.8f}")
print(f"Grid: {N} x {N}")
print(f"Local window: +/- {HALF_WINDOW_RE:.4f} Hz")
print()
print("NO q LABEL.")
print("NO CLAIM OF NEW PHYSICS.")
print("=" * 100)

results = []

def predicted_position(m):
    dm = m - ROOT_M
    return ROOT_RE + DRE_DM * dm, ROOT_IM + DIM_DM * dm

for m in M_VALUES:

    pre, pim = predicted_position(m)

    re_vals = np.linspace(
        pre - HALF_WINDOW_RE,
        pre + HALF_WINDOW_RE,
        N
    )

    im_vals = np.linspace(
        pim - HALF_WINDOW_IM,
        pim + HALF_WINDOW_IM,
        N
    )

    best = None

    for re in re_vals:
        for im in im_vals:
            try:
                z = safe_complex_res(
                    re, im, m,
                    r_sp=None,
                    omega_lr=None
                )
            except TypeError:
                # Compatibility with versions where extra arguments
                # are inferred internally.
                try:
                    z = safe_complex_res(re, im, m)
                except Exception:
                    continue
            except Exception:
                continue

            if not np.isfinite(np.real(z)) or not np.isfinite(np.imag(z)):
                continue

            val = abs(z)

            if best is None or val < best[0]:
                best = (val, re, im, np.real(z), np.imag(z))

    print("-" * 100)
    print(f"m = {m:.8f}")

    if best is None:
        print("  NO EVALUABLE POINTS")
        results.append((m, np.nan, np.nan, np.nan))
        continue

    val, re, im, rr, ri = best

    print(f"  predicted: Re={pre:.10f}  Im={pim:.10f}")
    print(f"  minimum : Re={re:.10f}  Im={im:.10f}")
    print(f"  min|Res| = {val:.12e}")
    print(f"  Re(R)   = {rr:.12e}")
    print(f"  Im(R)   = {ri:.12e}")

    results.append((m, val, re, im))

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print()
print(f"{'m':>12} {'min|Res|':>18} {'Re':>15} {'Im':>15}")
print("-" * 65)

for m, val, re, im in results:
    if np.isfinite(val):
        print(f"{m:12.8f} {val:18.10e} {re:15.8f} {im:15.8f}")
    else:
        print(f"{m:12.8f} {'NO RESULT':>18}")

print()
print("=" * 100)
print("ENDPOINT CONTRAST")
print("=" * 100)

valid = [(m, v) for m, v, _, _ in results if np.isfinite(v)]

if valid:
    for (m1, v1), (m2, v2) in zip(valid[:-1], valid[1:]):
        if v1 > 0:
            ratio = v2 / v1
        else:
            ratio = np.nan
        print(
            f"{m1:.8f} -> {m2:.8f}   "
            f"min|Res| ratio = {ratio:.6e}"
        )

print()
print("=" * 100)
print("INTERPRETATION")
print("=" * 100)
print()
print("This test asks whether the residual minimum changes sharply")
print("inside the already localized endpoint interval.")
print()
print("A genuine root should produce min|Res| near numerical zero.")
print("A persistent finite minimum indicates that no root is being")
print("resolved in the tested local window.")
print()
print("IMPORTANT:")
print("  - This is a diagnostic, not a proof of the mechanism.")
print("  - The validated root solver remains authoritative.")
print("  - A frequency-window limitation must still be excluded.")
print("  - NO q LABEL.")
print("  - NO CLAIM OF NEW PHYSICS.")
print("=" * 100)
