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

print("=" * 90)
print("STAGE 5I-21 FAST -- ENDPOINT TOPOLOGY TEST")
print("=" * 90)
print()
print("Uses EXACT residual: stage5I-4 -> safe_complex_res()")
print("No substitute residual.")
print("NO q LABEL. NO CLAIM OF NEW PHYSICS.")
print()

# Endpoint from previous tests
MC = -14.79415

# Center of the last validated root
W_RE = 9.6128
W_IM = -0.30294

# Only the most informative m values.
M_VALUES = [
    -14.78000,
    -14.79000,
    -14.79300,
    -14.79400,
    -14.79410,
    -14.79415,
    -14.79420,
    -14.79430,
]

# FAST local grid: 81 x 81 instead of 161 x 161
N = 81

RE_OFFSETS = np.linspace(-0.08, 0.08, N)
IM_OFFSETS = np.linspace(-0.08, 0.08, N)

ZERO_TOL = 1.0e-3


def analyze_m(m):

    r_sp, omega_lr = find_light_ring(m)

    if r_sp is None:
        return {
            "m": m,
            "status": "NO_LIGHT_RING"
        }

    min_abs = np.inf
    min_re = np.nan
    min_im = np.nan

    # Count cells where both Re(R) and Im(R)
    # change sign.
    zero_cells = 0

    # Store only the previous row so memory stays tiny.
    previous = None

    for iy, im_offset in enumerate(IM_OFFSETS):

        current = np.empty(N, dtype=complex)

        im = W_IM + im_offset

        for ix, re_offset in enumerate(RE_OFFSETS):

            re = W_RE + re_offset

            try:
                z = safe_complex_res(
                    re,
                    im,
                    m,
                    r_sp,
                    omega_lr,
                )
                current[ix] = z

                az = abs(z)

                if np.isfinite(az) and az < min_abs:
                    min_abs = az
                    min_re = re
                    min_im = im

            except Exception:
                current[ix] = np.nan + 1j*np.nan

        # Check 2x2 cells once the previous row exists.
        if previous is not None:

            for ix in range(N - 1):

                cell = np.array([
                    previous[ix],
                    previous[ix + 1],
                    current[ix],
                    current[ix + 1],
                ])

                if not np.all(np.isfinite(cell)):
                    continue

                real = np.real(cell)
                imag = np.imag(cell)

                if (
                    np.min(real) <= 0.0 <= np.max(real)
                    and
                    np.min(imag) <= 0.0 <= np.max(imag)
                ):
                    zero_cells += 1

        previous = current

    return {
        "m": m,
        "status": "OK",
        "min_abs": min_abs,
        "min_re": min_re,
        "min_im": min_im,
        "zero_cells": zero_cells,
    }


results = []

print("=" * 90)
print("FAST LOCAL TOPOLOGY SCAN")
print("=" * 90)
print()
print(f"Grid: {N} x {N}")
print(
    f"Frequency window: "
    f"Re=[{W_RE-0.08:.3f},{W_RE+0.08:.3f}] "
    f"Im=[{W_IM-0.08:.3f},{W_IM+0.08:.3f}]"
)
print(f"m points: {len(M_VALUES)}")
print()

for m in M_VALUES:

    print("-" * 90)
    print(f"m = {m:.5f}")

    result = analyze_m(m)
    results.append(result)

    if result["status"] != "OK":
        print("  STATUS:", result["status"])
        continue

    print(
        f"  minimum |Res| = {result['min_abs']:.6e}"
    )

    print(
        f"  location: Re={result['min_re']:.8f}, "
        f"Im={result['min_im']:.8f}"
    )

    print(
        f"  simultaneous-zero candidate cells = "
        f"{result['zero_cells']}"
    )

    if result["min_abs"] < ZERO_TOL:
        print("  -> LOW-RESIDUAL STRUCTURE PRESENT")
    else:
        print("  -> NO LOW-RESIDUAL STRUCTURE")

    if result["zero_cells"] > 0:
        print("  -> ZERO-CONTOUR INTERSECTION CANDIDATE")
    else:
        print("  -> NO ZERO-CONTOUR INTERSECTION DETECTED")


print()
print("=" * 90)
print("ENDPOINT COMPARISON")
print("=" * 90)
print()

before = [
    r for r in results
    if r["status"] == "OK" and r["m"] <= MC
]

after = [
    r for r in results
    if r["status"] == "OK" and r["m"] > MC
]

print("BEFORE / AT ENDPOINT")
print()

for r in before:
    print(
        f"m={r['m']:.5f}  "
        f"min|Res|={r['min_abs']:.6e}  "
        f"zero-cells={r['zero_cells']}"
    )

print()
print("AFTER ENDPOINT")
print()

for r in after:
    print(
        f"m={r['m']:.5f}  "
        f"min|Res|={r['min_abs']:.6e}  "
        f"zero-cells={r['zero_cells']}"
    )


before_zero = sum(
    r["zero_cells"] > 0 for r in before
)

after_zero = sum(
    r["zero_cells"] > 0 for r in after
)

before_low = sum(
    r["min_abs"] < ZERO_TOL for r in before
)

after_low = sum(
    r["min_abs"] < ZERO_TOL for r in after
)


print()
print("=" * 90)
print("FINAL DIAGNOSTIC")
print("=" * 90)
print()

print(
    f"Before/at endpoint: "
    f"{before_zero}/{len(before)} "
    f"with zero-contour candidates"
)

print(
    f"After endpoint: "
    f"{after_zero}/{len(after)} "
    f"with zero-contour candidates"
)

print(
    f"Before/at endpoint: "
    f"{before_low}/{len(before)} "
    f"with |Res| < {ZERO_TOL:.1e}"
)

print(
    f"After endpoint: "
    f"{after_low}/{len(after)} "
    f"with |Res| < {ZERO_TOL:.1e}"
)

print()

if before_zero > 0 and after_zero == 0:

    print(">>> TOPOLOGICAL ENDPOINT CANDIDATE <<<")
    print()
    print(
        "Simultaneous-zero contour candidates are detected "
        "before/at the endpoint but disappear after it."
    )
    print()
    print(
        "This strengthens the case that the disappearance "
        "is a property of the residual system rather than "
        "a simple solver failure."
    )
    print()
    print(
        "This is still a numerical diagnostic, not a proof "
        "of the physical mechanism."
    )

elif before_low > 0 and after_low == 0:

    print(">>> LOW-RESIDUAL STRUCTURE DISAPPEARS <<<")
    print()
    print(
        "The branch-like low-residual structure disappears "
        "after the endpoint."
    )
    print(
        "A higher-resolution contour test is recommended."
    )

else:

    print(">>> NO DECISIVE TOPOLOGICAL CHANGE <<<")
    print()
    print(
        "The fast grid does not yet establish whether the "
        "endpoint is a genuine topological disappearance."
    )

print()
print("=" * 90)
print("STATUS")
print("=" * 90)
print("NO q LABEL ASSIGNED.")
print("NO CLAIM OF NEW PHYSICS.")
print("=" * 90)

