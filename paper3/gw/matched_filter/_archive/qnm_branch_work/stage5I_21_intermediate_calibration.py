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
# STAGE 5I-21 -- INTERMEDIATE CALIBRATED TOPOLOGY TEST
#
# EXACT residual:
#     safe_complex_res()
#
# Same 81x81 resolution as the validated m=-14 control.
#
# Known branch points:
#   m=-14   Re=9.3995509552  Im=-0.3086180377
#   m=-13   Re=9.1218868177  Im=-0.3173351972
#
# We linearly interpolate/extrapolate the expected branch
# position and center an identical +/-0.01 Hz window around it.
#
# Test points:
#   m=-14.3
#   m=-14.5
#   m=-14.7
#
# NO q LABEL.
# NO CLAIM OF NEW PHYSICS.
# ============================================================

KNOWN = {
    -14.0: (9.3995509552, -0.3086180377),
    -13.0: (9.1218868177, -0.3173351972),
}

M_VALUES = [-14.3, -14.5, -14.7]

N = 81
HALF_WINDOW = 0.010


def predicted_position(m):
    m1 = -14.0
    m2 = -13.0

    re1, im1 = KNOWN[m1]
    re2, im2 = KNOWN[m2]

    dre_dm = (re2 - re1) / (m2 - m1)
    dim_dm = (im2 - im1) / (m2 - m1)

    re = re1 + dre_dm * (m - m1)
    im = im1 + dim_dm * (m - m1)

    return re, im


def evaluate(m):

    r_sp, omega_lr = find_light_ring(m)

    if r_sp is None:
        print(f"m={m:.5f}: LIGHT RING FAILED")
        return None

    center_re, center_im = predicted_position(m)

    re_values = np.linspace(
        center_re - HALF_WINDOW,
        center_re + HALF_WINDOW,
        N,
    )

    im_values = np.linspace(
        center_im - HALF_WINDOW,
        center_im + HALF_WINDOW,
        N,
    )

    best = None

    for re in re_values:
        for im in im_values:

            z = safe_complex_res(
                re,
                im,
                m,
                r_sp,
                omega_lr,
            )

            val = abs(z)

            if best is None or val < best["abs_res"]:
                best = {
                    "abs_res": val,
                    "re": re,
                    "im": im,
                    "res": z,
                }

    return center_re, center_im, best


def main():

    print("=" * 100)
    print("STAGE 5I-21 -- INTERMEDIATE CALIBRATED TOPOLOGY TEST")
    print("=" * 100)

    print()
    print("Residual source:")
    print("  stage5I-4 -> safe_complex_res()")
    print()
    print("Grid: 81 x 81")
    print("Window: +/- 0.010 Hz around predicted branch position")
    print()
    print("Testing:")
    print("  m=-14.3")
    print("  m=-14.5")
    print("  m=-14.7")
    print()

    results = []

    for m in M_VALUES:

        print("-" * 100)
        print(f"m = {m:.5f}")
        print("-" * 100)

        predicted_re, predicted_im = predicted_position(m)

        print(
            f"predicted center: "
            f"Re={predicted_re:.10f}, "
            f"Im={predicted_im:.10f}"
        )

        result = evaluate(m)

        if result is None:
            continue

        center_re, center_im, best = result

        print(
            f"grid minimum: "
            f"Re={best['re']:.10f}, "
            f"Im={best['im']:.10f}"
        )

        print(
            f"|Res|={best['abs_res']:.12e}"
        )

        print(
            f"Re(R)={best['res'].real:.6e}, "
            f"Im(R)={best['res'].imag:.6e}"
        )

        results.append(
            (
                m,
                best["abs_res"],
                best["re"],
                best["im"],
            )
        )

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    for m, abs_res, re, im in results:
        print(
            f"m={m:8.5f}  "
            f"min|Res|={abs_res:.12e}  "
            f"at ({re:.8f},{im:.8f})"
        )

    print()
    print("=" * 100)
    print("COMPARISON WITH KNOWN m=-14 CONTROL")
    print("=" * 100)

    print(
        "m=-14 control: min|Res| = 3.768402330995e-09"
    )

    if results:
        print()

        for m, abs_res, _, _ in results:

            ratio = abs_res / 3.768402330995e-9

            print(
                f"m={m:8.5f}  "
                f"ratio to m=-14 control = {ratio:.6e}"
            )

    print()
    print("=" * 100)
    print("INTERPRETATION")
    print("=" * 100)

    print("""
These three points are a calibration of the residual topology
diagnostic between the known root at m=-14 and the suspected
endpoint near m=-14.79415.

Important:
  - A small minimum means the local grid is resolving the branch.
  - A large minimum means no root is being resolved inside the
    chosen +/-0.01 Hz window.
  - The test does NOT prove branch termination by itself.
  - The exact validated root solver remains authoritative.

The important question is whether min|Res| evolves smoothly as
m approaches the endpoint, or whether it suddenly becomes large.

NO q LABEL.
NO CLAIM OF NEW PHYSICS.
""")

    print("=" * 100)


if __name__ == "__main__":
    main()
