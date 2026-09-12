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
# STAGE 5I-20
# EXACT JACOBIAN ENDPOINT TEST
#
# Uses the EXACT residual wrapper from stage5I-4:
#     safe_complex_res(...)
#
# No substitute residual is constructed.
#
# Question:
#   Does det(J) or the smallest singular value of J
#   collapse as m approaches the localized endpoint?
#
# A collapse is consistent with a singular endpoint/fold.
# No collapse argues against a simple local fold mechanism.
# ============================================================

ROOTS = [
    (-14.78000, 9.612801, -0.302944),
    (-14.78500, 9.611441, -0.302978),
    (-14.79000, 9.610082, -0.303012),
    (-14.79200, 9.609538, -0.303025),
    (-14.79300, 9.609266, -0.303032),
    (-14.79400, 9.608994, -0.303039),
    (-14.79410, 9.608967, -0.303039),
    (-14.79415, 9.608953, -0.303040),
]

EPS_RE = 1.0e-5
EPS_IM = 1.0e-5


def get_context(m):
    r_sp, omega_lr = find_light_ring(m)

    if r_sp is None:
        raise RuntimeError("find_light_ring failed")

    return r_sp, omega_lr


def R(re, im, m, r_sp, omega_lr):
    """
    EXACT residual used by stage5I-4.
    """
    return safe_complex_res(
        re,
        im,
        m,
        r_sp,
        omega_lr,
    )


def jacobian(re, im, m, r_sp, omega_lr):
    """
    Central finite-difference Jacobian of

        F = (Re R, Im R)

    with respect to

        x = (Re omega, Im omega).
    """

    Rp = R(re + EPS_RE, im, m, r_sp, omega_lr)
    Rm = R(re - EPS_RE, im, m, r_sp, omega_lr)

    Ip = R(re, im + EPS_IM, m, r_sp, omega_lr)
    Im = R(re, im - EPS_IM, m, r_sp, omega_lr)

    dR_dRe = (Rp - Rm) / (2.0 * EPS_RE)
    dR_dIm = (Ip - Im) / (2.0 * EPS_IM)

    J = np.array([
        [dR_dRe.real, dR_dIm.real],
        [dR_dRe.imag, dR_dIm.imag],
    ], dtype=float)

    detJ = np.linalg.det(J)
    svals = np.linalg.svd(J, compute_uv=False)

    return J, detJ, svals


def main():

    print("=" * 100)
    print("STAGE 5I-20 -- EXACT JACOBIAN ENDPOINT TEST")
    print("=" * 100)
    print()
    print("Residual source:")
    print("  stage5I-4 -> safe_complex_res()")
    print("  -> resonance_factor_v2()")
    print()
    print("This is the ORIGINAL residual used by the validated solver.")
    print()
    print("Testing endpoint near m_c ~ -14.79415")
    print()
    print("NO q LABEL.")
    print("NO CLAIM OF NEW PHYSICS.")
    print()

    records = []

    print("=" * 100)
    print("JACOBIAN TRACKING")
    print("=" * 100)

    print(
        f"{'m':>11} "
        f"{'det(J)':>18} "
        f"{'sigma_max':>15} "
        f"{'sigma_min':>15} "
        f"{'cond(J)':>15}"
    )

    for m, re, im in ROOTS:

        try:
            r_sp, omega_lr = get_context(m)

            J, detJ, svals = jacobian(
                re,
                im,
                m,
                r_sp,
                omega_lr,
            )

            sigma_max = float(svals[0])
            sigma_min = float(svals[-1])

            cond = (
                sigma_max / sigma_min
                if sigma_min > 0
                else np.inf
            )

            print(
                f"{m:11.5f} "
                f"{detJ:18.8e} "
                f"{sigma_max:15.6e} "
                f"{sigma_min:15.6e} "
                f"{cond:15.6e}"
            )

            records.append(
                (m, detJ, sigma_max, sigma_min, cond)
            )

        except Exception as exc:
            print(
                f"{m:11.5f} "
                f"{'FAILED':>18} "
                f"{'--':>15} "
                f"{'--':>15} "
                f"{'--':>15}"
            )
            print(f"             {exc}")

    if len(records) < 3:
        print()
        print("INSUFFICIENT DATA FOR ENDPOINT TEST.")
        return

    print()
    print("=" * 100)
    print("LOCAL JACOBIAN BEHAVIOR")
    print("=" * 100)

    abs_det = np.abs(np.array([x[1] for x in records]))
    smin = np.array([x[3] for x in records])

    first_det = max(abs_det[0], 1e-300)
    first_smin = max(smin[0], 1e-300)

    print(f"|det J| first = {abs_det[0]:.8e}")
    print(f"|det J| last  = {abs_det[-1]:.8e}")
    print(f"ratio         = {abs_det[-1] / first_det:.8e}")
    print()
    print(f"sigma_min first = {smin[0]:.8e}")
    print(f"sigma_min last  = {smin[-1]:.8e}")
    print(f"ratio           = {smin[-1] / first_smin:.8e}")

    print()
    print("=" * 100)
    print("DERIVATIVE / CONDITIONING TREND")
    print("=" * 100)

    for i in range(1, len(records)):
        m0, _, _, s0, c0 = records[i - 1]
        m1, _, _, s1, c1 = records[i]

        print(
            f"{m0:9.5f} -> {m1:9.5f}   "
            f"sigma_min ratio={s1/s0:10.6f}   "
            f"cond ratio={c1/c0:10.6f}"
        )

    print()
    print("=" * 100)
    print("FINAL DIAGNOSTIC")
    print("=" * 100)

    det_ratio = abs_det[-1] / first_det
    smin_ratio = smin[-1] / first_smin

    if det_ratio < 1e-2 or smin_ratio < 1e-2:

        print()
        print(">>> JACOBIAN DEGENERACY DETECTED <<<")
        print()
        print("The local Jacobian becomes strongly singular")
        print("as the branch approaches the endpoint.")
        print()
        print("This is CONSISTENT with a critical endpoint")
        print("and could support a fold/saddle-node interpretation.")
        print()
        print("It is NOT mathematical proof of a fold.")

    else:

        print()
        print(">>> NO STRONG JACOBIAN DEGENERACY <<<")
        print()
        print("The root remains locally regular up to the last")
        print("validated point.")
        print()
        print("Therefore the data do NOT support a simple")
        print("Jacobian-singularity/fold explanation.")

    print()
    print("=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    print()
    print("Combined with Stage 5I-14, 15, 16 and the endpoint")
    print("continuation result, this test determines whether")
    print("the disappearance is accompanied by local loss")
    print("of root regularity.")
    print()
    print("NO q LABEL ASSIGNED.")
    print("NO CLAIM OF NEW PHYSICS.")
    print("=" * 100)


if __name__ == "__main__":
    main()
