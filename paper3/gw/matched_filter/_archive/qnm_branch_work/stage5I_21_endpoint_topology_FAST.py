import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stage5I_4_full_m_scan_v6_FIXED import safe_complex_res

# ============================================================
# STAGE 5I-21 FAST
# ENDPOINT TOPOLOGY / ZERO-CONTOUR TEST
#
# EXACT residual:
#     safe_complex_res()
#
# FAST version:
#   41 x 41 grid instead of 161 x 161
#   only 5 m-values near endpoint
#
# Goal:
#   Check whether Re(R)=0 and Im(R)=0 contours
#   still have a simultaneous intersection after m_c.
# ============================================================

MC = -14.79415

M_VALUES = [
    -14.7900,
    -14.7940,
    -14.79410,
    -14.79415,
    -14.79420,
]

# Local window around the known endpoint branch.
RE_MIN, RE_MAX = 9.58, 9.63
IM_MIN, IM_MAX = -0.33, -0.28

N = 41

print("=" * 100, flush=True)
print("STAGE 5I-21 FAST -- ENDPOINT TOPOLOGY TEST", flush=True)
print("=" * 100)
print()
print("Exact residual: stage5I-4 -> safe_complex_res()", flush=True)
print(f"Grid: {N} x {N} = {N*N} residual evaluations per m", flush=True)
print(f"Endpoint estimate: m_c ~ {MC}", flush=True)
print()
print("NO q LABEL.")
print("NO CLAIM OF NEW PHYSICS.")
print("=" * 100, flush=True)


def evaluate_grid(m):
    re_values = np.linspace(RE_MIN, RE_MAX, N)
    im_values = np.linspace(IM_MIN, IM_MAX, N)

    best = None
    near_count = 0

    # Store nearest points to the two zero contours.
    re_zero_candidates = []
    im_zero_candidates = []

    for re in re_values:
        for im in im_values:
            try:
                z = safe_complex_res(
                    re,
                    im,
                    m,
                    None,
                    None,
                )
            except TypeError:
                # safe_complex_res requires r_sp and omega_lr.
                return None, "MISSING_LIGHT_RING_ARGUMENTS"

            if not np.isfinite(z.real) or not np.isfinite(z.imag):
                continue

            absz = abs(z)

            if best is None or absz < best["abs_res"]:
                best = {
                    "re": re,
                    "im": im,
                    "res_re": z.real,
                    "res_im": z.imag,
                    "abs_res": absz,
                }

    return best, "OK"


# ------------------------------------------------------------
# IMPORTANT:
# safe_complex_res requires the exact light-ring quantities.
# We therefore obtain them exactly from stage5I-4 for each m.
# ------------------------------------------------------------

from stage5I_4_full_m_scan_v6_FIXED import find_light_ring


def evaluate_grid_exact(m):
    r_sp, omega_lr = find_light_ring(m)

    if r_sp is None:
        return None, "NO_LIGHT_RING"

    re_values = np.linspace(RE_MIN, RE_MAX, N)
    im_values = np.linspace(IM_MIN, IM_MAX, N)

    best = None

    for i, re in enumerate(re_values):
        for im in im_values:
            try:
                z = safe_complex_res(
                    re,
                    im,
                    m,
                    r_sp,
                    omega_lr,
                )
            except Exception:
                continue

            if not np.isfinite(z.real) or not np.isfinite(z.imag):
                continue

            absz = abs(z)

            if best is None or absz < best["abs_res"]:
                best = {
                    "re": re,
                    "im": im,
                    "res_re": z.real,
                    "res_im": z.imag,
                    "abs_res": absz,
                }

    return best, "OK"


results = []

for m in M_VALUES:
    print()
    print("-" * 100, flush=True)
    print(f"m = {m:.5f}", flush=True)
    print("-" * 100, flush=True)

    best, status = evaluate_grid_exact(m)

    if best is None:
        print(f"  RESULT: {status}", flush=True)
        results.append((m, None))
        continue

    print(
        f"  global local minimum |Res| = {best['abs_res']:.6e}",
        flush=True
    )
    print(
        f"  at Re={best['re']:.8f}, Im={best['im']:.8f}",
        flush=True
    )
    print(
        f"  Re(R)={best['res_re']:.6e}, Im(R)={best['res_im']:.6e}",
        flush=True
    )

    results.append((m, best))

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

for m, best in results:
    if best is None:
        print(f"m={m:.5f} -> NO VALID GRID RESULT")
    else:
        print(
            f"m={m:.5f} -> min |Res|={best['abs_res']:.6e} "
            f"at ({best['re']:.6f},{best['im']:.6f})"
        )

print()
print("=" * 100)
print("INTERPRETATION")
print("=" * 100)
print("""
This FAST test is only a topology diagnostic.

If the minimum |Res| remains very small before m_c but
jumps upward immediately after m_c, that supports the
absence of a simultaneous Re(R)=0, Im(R)=0 intersection
in this local window.

It does NOT by itself prove the mechanism of termination.

The exact validated root solver remains the authority
for root confirmation.

NO q LABEL.
NO CLAIM OF NEW PHYSICS.
""")
