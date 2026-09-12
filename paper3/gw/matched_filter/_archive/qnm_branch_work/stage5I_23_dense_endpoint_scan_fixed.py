import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stage5I_4_full_m_scan_v6_FIXED import (
    find_light_ring,
    safe_complex_res,
    refine_and_polish_candidate,
)

# ============================================================
# STAGE 5I-23 -- DENSE ENDPOINT SCAN (FIXED)
#
# Goal:
#   Track min |R| extremely close to the suspected endpoint AND
#   confirm/refute with the ORIGINAL validated solver -- not the
#   FAST grid alone.
#
# TWO BUGS FIXED vs the original dense_endpoint_scan draft:
#
#   1. safe_complex_res(f_re, f_im, m, r_sp, omega_lr) requires
#      r_sp and omega_lr as MANDATORY arguments with no internal
#      fallback -- passing None (as in the original draft) causes
#      every single grid-point evaluation to fail silently via the
#      except-Exception branch, producing "NO EVALUABLE POINTS"
#      for every m tested. FIX: call find_light_ring(m) ONCE per m
#      and pass the correct r_sp, omega_lr into every grid call for
#      that m.
#
#   2. Stage 5I-22 (intermediate_root_validation) already
#      demonstrated conclusively that a FAST-grid-only "large
#      min|Res|" result is NOT reliable evidence of absence: three
#      genuine roots (m=-14.3, -14.5, -14.7) were missed by the
#      grid alone (min|Res|~0.01-0.02) but recovered perfectly
#      (|Res|~1e-14) once fed into the ORIGINAL solver
#      (refine_and_polish_candidate). This script inherits that
#      same risk structurally if left as a grid-only diagnostic.
#      FIX: after the dense grid at each m, feed its best few
#      candidates into refine_and_polish_candidate() for genuine
#      validation, exactly as Stage 5I-22 did.
#
# No substitute residual. No q label. No claim of new physics.
# ============================================================

ROOT_RE = 9.6046425466
ROOT_IM = -0.3031465788
ROOT_M = -14.7500

MC_EST = -14.79415

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

DRE_DM = -0.27183
DIM_DM = -0.00670

HALF_WINDOW_RE = 0.004
HALF_WINDOW_IM = 0.004

N = 41  # reduced from 81: this version does REAL solver validation on
        # top candidates per m, which is far more expensive per point
        # than a bare grid evaluation -- N=41 (1681 pts/m) keeps total
        # runtime reasonable across 16 m-values while still being a
        # dense local scan.

N_TOP_CANDIDATES = 5  # how many best grid points per m get fed into
                       # the full solver validation stage

print("=" * 100)
print("STAGE 5I-23 -- DENSE ENDPOINT SCAN (FIXED: correct r_sp/omega_lr,")
print("plus mandatory full-solver validation stage)")
print("=" * 100)
print()
print("Exact residual: stage5I-4 -> safe_complex_res()")
print("Root validation: ORIGINAL refine_and_polish_candidate()")
print(f"Endpoint estimate: m_c ~ {MC_EST:.8f}")
print(f"Grid: {N} x {N}, top {N_TOP_CANDIDATES} candidates validated per m")
print(f"Local window: +/- {HALF_WINDOW_RE:.4f} Hz")
print()
print("NO q LABEL.")
print("NO CLAIM OF NEW PHYSICS.")
print("=" * 100)


def predicted_position(m):
    dm = m - ROOT_M
    return ROOT_RE + DRE_DM * dm, ROOT_IM + DIM_DM * dm


results = []

for m in M_VALUES:

    m = float(m)
    r_sp, omega_lr = find_light_ring(m)

    if r_sp is None:
        print("-" * 100)
        print(f"m = {m:.8f}")
        print("  NO LIGHT RING FOUND")
        results.append((m, np.nan, np.nan, np.nan, 0))
        continue

    pre, pim = predicted_position(m)

    re_vals = np.linspace(pre - HALF_WINDOW_RE, pre + HALF_WINDOW_RE, N)
    im_vals = np.linspace(pim - HALF_WINDOW_IM, pim + HALF_WINDOW_IM, N)

    grid_points = []
    for re in re_vals:
        for im in im_vals:
            z = safe_complex_res(float(re), float(im), m, r_sp, omega_lr)
            if z is None:
                continue
            if not (np.isfinite(z.real) and np.isfinite(z.imag)):
                continue
            grid_points.append((abs(z), float(re), float(im), z.real, z.imag))

    print("-" * 100)
    print(f"m = {m:.8f}")

    if not grid_points:
        print("  NO EVALUABLE POINTS")
        results.append((m, np.nan, np.nan, np.nan, 0))
        continue

    grid_points.sort(key=lambda x: x[0])
    best_val, best_re, best_im, best_rr, best_ri = grid_points[0]

    print(f"  predicted: Re={pre:.10f}  Im={pim:.10f}")
    print(f"  grid minimum : Re={best_re:.10f}  Im={best_im:.10f}")
    print(f"  grid min|Res| = {best_val:.12e}")
    print(f"  Re(R)={best_rr:.6e}  Im(R)={best_ri:.6e}")
    print()

    print(f"  FULL SOLVER VALIDATION of top {N_TOP_CANDIDATES} grid "
          f"candidates:")
    validated_roots = []
    for i, (val, re, im, rr, ri) in enumerate(grid_points[:N_TOP_CANDIDATES]):
        pole = refine_and_polish_candidate(
            f0_re=re, f0_im_hint=im, m=m, r_sp=r_sp, omega_lr=omega_lr,
            real_axis_abs_res=float("nan"), is_complex_seed=True,
        )
        if pole is not None:
            print(f"    candidate {i+1} (grid |Res|={val:.4e}) -> "
                  f"VALIDATED: Re={pole['f_re']:.8f} Im={pole['f_im']:.8f} "
                  f"|Res|={pole['abs_res']:.4e}")
            validated_roots.append(pole)
        else:
            print(f"    candidate {i+1} (grid |Res|={val:.4e}) -> "
                  f"no validated root")

    n_validated = len(validated_roots)
    best_validated_res = (min(p["abs_res"] for p in validated_roots)
                           if validated_roots else np.nan)

    print()
    if n_validated > 0:
        print(f"  RESULT: {n_validated}/{N_TOP_CANDIDATES} candidates "
              f"VALIDATED (best |Res|={best_validated_res:.4e})")
    else:
        print(f"  RESULT: 0/{N_TOP_CANDIDATES} candidates validated -- "
              f"NO ROOT confirmed by the full solver at this m.")
    print()

    results.append((m, best_val, best_re, best_im, n_validated))

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print()
print(f"{'m':>12} {'grid min|Res|':>16} {'validated?':>12}")
print("-" * 45)
for m, val, re, im, n_val in results:
    val_str = f"{val:.4e}" if np.isfinite(val) else "NO GRID"
    status = f"YES ({n_val}/{N_TOP_CANDIDATES})" if n_val > 0 else "NO"
    print(f"{m:12.8f} {val_str:>16} {status:>12}")

print()
print("=" * 100)
print("INTERPRETATION")
print("=" * 100)
print()
print("A m-value is only credible evidence of branch ABSENCE if the")
print("FULL SOLVER (not the grid alone) fails to validate ANY of the")
print("top grid candidates. Per Stage 5I-22's finding, a large grid")
print("min|Res| alone does NOT establish absence -- it may simply")
print("reflect grid centering/resolution, exactly as it did at")
print("m=-14.3, -14.5, -14.7 before those were confirmed as genuine")
print("roots via this same two-stage procedure.")
print()
print("IMPORTANT:")
print("  - The validated root solver remains authoritative.")
print("  - NO q LABEL.")
print("  - NO CLAIM OF NEW PHYSICS.")
print("=" * 100)
