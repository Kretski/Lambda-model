import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stage5I_4_full_m_scan_v6_FIXED import (
    find_light_ring,
    safe_complex_res,
)

# ============================================================
# STAGE 5I-21 -- ENDPOINT TOPOLOGY / ZERO-CONTOUR TEST
#
# Uses the EXACT residual wrapper from stage5I-4:
#     safe_complex_res()
#
# No substitute residual is constructed.
#
# We test the simultaneous equations
#
#       Re R(Re(w), Im(w), m) = 0
#       Im R(Re(w), Im(w), m) = 0
#
# in a dense local 3D neighborhood around the endpoint.
#
# Goal:
#   Determine whether the two zero contours cease to intersect
#   after the localized endpoint.
#
# This is a topology diagnostic only.
# NO q LABEL.
# NO CLAIM OF NEW PHYSICS.
# ============================================================

print("=" * 100)
print("STAGE 5I-21 -- ENDPOINT TOPOLOGY / ZERO-CONTOUR TEST")
print("=" * 100)
print()
print("Residual source:")
print("  stage5I-4 -> safe_complex_res()")
print()
print("Testing the localized endpoint near")
print("  m_c ~ -14.79415")
print()
print("NO q LABEL.")
print("NO CLAIM OF NEW PHYSICS.")
print()

# ------------------------------------------------------------
# Endpoint reference
# ------------------------------------------------------------

MC = -14.79415
W_RE = 9.6128
W_IM = -0.30294

# Local m values on both sides of the endpoint.
M_VALUES = [-14.0]

# Dense local frequency window.
#
# The window is deliberately wider than the last validated
# root by several continuation steps.
RE_OFFSETS = np.linspace(-0.08, 0.08, 161)
IM_OFFSETS = np.linspace(-0.08, 0.08, 161)

# A numerical zero threshold.
# We do NOT claim this is an absolute mathematical threshold.
ZERO_TOL = 1.0e-3


def residual_grid(m):
    r_sp, omega_lr = find_light_ring(m)

    if r_sp is None:
        return None

    Re = W_RE + RE_OFFSETS
    Im = W_IM + IM_OFFSETS

    grid = np.empty((len(Im), len(Re)), dtype=float)

    for iy, im in enumerate(Im):
        for ix, re in enumerate(Re):
            try:
                z = safe_complex_res(
                    re,
                    im,
                    m,
                    r_sp,
                    omega_lr,
                )
                grid[iy, ix] = abs(z)
            except Exception:
                grid[iy, ix] = np.inf

    return grid


def nearest_zero(grid):
    idx = np.unravel_index(np.nanargmin(grid), grid.shape)

    iy, ix = idx

    re = W_RE + RE_OFFSETS[ix]
    im = W_IM + IM_OFFSETS[iy]

    return re, im, grid[iy, ix]


def sign_changes_2d(values):
    """
    Count cells in which BOTH Re(R) and Im(R) exhibit
    sign variation.

    Such cells are candidates for a local intersection
    of the two zero contours.

    This is a diagnostic, not a proof of an exact zero.
    """

    real = np.real(values)
    imag = np.imag(values)

    count = 0

    ny, nx = real.shape

    for iy in range(ny - 1):
        for ix in range(nx - 1):

            rcell = real[iy:iy + 2, ix:ix + 2]
            icell = imag[iy:iy + 2, ix:ix + 2]

            rmin = np.min(rcell)
            rmax = np.max(rcell)

            imin = np.min(icell)
            imax = np.max(icell)

            if rmin <= 0.0 <= rmax and imin <= 0.0 <= imax:
                count += 1

    return count


def analyze_m(m):
    r_sp, omega_lr = find_light_ring(m)

    if r_sp is None:
        return {
            "m": m,
            "status": "NO_LIGHT_RING",
        }

    Re = W_RE + RE_OFFSETS
    Im = W_IM + IM_OFFSETS

    Z = np.empty((len(Im), len(Re)), dtype=complex)

    for iy, im in enumerate(Im):
        for ix, re in enumerate(Re):
            try:
                Z[iy, ix] = safe_complex_res(
                    re,
                    im,
                    m,
                    r_sp,
                    omega_lr,
                )
            except Exception:
                Z[iy, ix] = np.nan + 1j * np.nan

    absZ = np.abs(Z)

    finite = np.isfinite(absZ)

    if not np.any(finite):
        return {
            "m": m,
            "status": "NO_FINITE_RESIDUAL",
        }

    masked = np.where(finite, absZ, np.inf)

    iy, ix = np.unravel_index(np.argmin(masked), masked.shape)

    min_re = Re[ix]
    min_im = Im[iy]
    min_res = masked[iy, ix]

    candidates = sign_changes_2d(Z)

    return {
        "m": m,
        "status": "OK",
        "min_re": min_re,
        "min_im": min_im,
        "min_res": min_res,
        "candidate_cells": candidates,
    }


# ============================================================
# MAIN TOPOLOGY SCAN
# ============================================================

print("=" * 100)
print("LOCAL 2D RESIDUAL TOPOLOGY")
print("=" * 100)
print()
print(
    f"Frequency window: Re=[{W_RE-0.08:.3f},{W_RE+0.08:.3f}] "
    f"Im=[{W_IM-0.08:.3f},{W_IM+0.08:.3f}]"
)
print(f"Grid: {len(RE_OFFSETS)} x {len(IM_OFFSETS)}")
print(f"Zero diagnostic threshold: {ZERO_TOL:.1e}")
print()

results = []

for m in M_VALUES:

    print("-" * 100)
    print(f"m = {m:.5f}")

    result = analyze_m(m)
    results.append(result)

    if result["status"] != "OK":
        print(f"  STATUS: {result['status']}")
        continue

    print(
        f"  global local minimum:"
        f"  Re={result['min_re']:.8f}"
        f"  Im={result['min_im']:.8f}"
        f"  |Res|={result['min_res']:.6e}"
    )

    print(
        f"  simultaneous-zero candidate cells:"
        f"  {result['candidate_cells']}"
    )

    if result["min_res"] < ZERO_TOL:
        print("  -> SUB-THRESHOLD MINIMUM PRESENT")
    else:
        print("  -> NO SUB-THRESHOLD MINIMUM")

    if result["candidate_cells"] > 0:
        print("  -> ZERO-CONTOUR INTERSECTION CANDIDATE PRESENT")
    else:
        print("  -> NO ZERO-CONTOUR INTERSECTION CELL")


# ============================================================
# COMPARISON ACROSS ENDPOINT
# ============================================================

print()
print("=" * 100)
print("ENDPOINT CROSSING DIAGNOSTIC")
print("=" * 100)

valid = [r for r in results if r["status"] == "OK"]

before = [
    r for r in valid
    if r["m"] <= -14.79415
]

after = [
    r for r in valid
    if r["m"] > -14.79415
]

print()

if before:
    print("BEFORE / AT ENDPOINT:")
    for r in before:
        print(
            f"  m={r['m']:.5f}"
            f"  min|Res|={r['min_res']:.6e}"
            f"  zero-cells={r['candidate_cells']}"
        )

print()

if after:
    print("AFTER ENDPOINT:")
    for r in after:
        print(
            f"  m={r['m']:.5f}"
            f"  min|Res|={r['min_res']:.6e}"
            f"  zero-cells={r['candidate_cells']}"
        )

# ============================================================
# INTERPRETATION
# ============================================================

print()
print("=" * 100)
print("INTERPRETATION")
print("=" * 100)
print()

before_zero = sum(
    1 for r in before
    if r["candidate_cells"] > 0
)

after_zero = sum(
    1 for r in after
    if r["candidate_cells"] > 0
)

before_low = sum(
    1 for r in before
    if r["min_res"] < ZERO_TOL
)

after_low = sum(
    1 for r in after
    if r["min_res"] < ZERO_TOL
)

print(
    f"Before/at endpoint: "
    f"{before_zero} points with zero-contour candidate cells, "
    f"{before_low} with |Res| < {ZERO_TOL:.1e}"
)

print(
    f"After endpoint: "
    f"{after_zero} points with zero-contour candidate cells, "
    f"{after_low} with |Res| < {ZERO_TOL:.1e}"
)

print()

if before_zero > 0 and after_zero == 0:
    print(">>> TOPOLOGICAL CHANGE CANDIDATE <<<")
    print()
    print(
        "The local residual geometry contains simultaneous-zero "
        "intersection candidates before/at the endpoint, while "
        "none are detected after it in the tested neighborhood."
    )
    print()
    print(
        "This is stronger evidence for a genuine disappearance "
        "of the root branch than a single failed solver call."
    )
    print()
    print(
        "However, this is still NOT a mathematical proof of a "
        "physical endpoint. The result depends on the tested "
        "frequency window and numerical resolution."
    )

elif before_low > 0 and after_low == 0:
    print(">>> LOCAL MINIMUM DISAPPEARANCE CANDIDATE <<<")
    print()
    print(
        "Low-residual structure is present before the endpoint "
        "but disappears after it."
    )
    print(
        "A denser adaptive contour calculation is warranted."
    )

else:
    print(">>> NO DECISIVE TOPOLOGICAL CHANGE DETECTED <<<")
    print()
    print(
        "The present local grid does not yet distinguish "
        "branch termination from a search-domain limitation."
    )
    print(
        "A larger frequency window and/or adaptive contour "
        "tracking should be performed before drawing conclusions."
    )

print()
print("=" * 100)
print("FINAL STATUS")
print("=" * 100)
print()
print("NO q LABEL ASSIGNED.")
print("NO CLAIM OF NEW PHYSICS.")
print()
print(
    "This test addresses the geometry of the EXACT residual "
    "system near the observed finite-m endpoint."
)
print("=" * 100)

