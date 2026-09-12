
import numpy as np
from scipy.optimize import root

from stage5I_4_full_m_scan_v6_FIXED import (
    find_light_ring,
    refine_and_polish_candidate,
)

# ============================================================
# STAGE 5I-19
# ENDPOINT MECHANISM TEST
#
# Goal:
#   Distinguish a possible ordinary smooth termination from
#   a fold/saddle-node-type endpoint.
#
# We do NOT assign q and do NOT claim new physics.
# ============================================================

M_VALUES = [
    -14.7800,
    -14.7850,
    -14.7900,
    -14.7920,
    -14.7930,
    -14.7940,
    -14.79405,
    -14.79410,
    -14.79415,
    -14.79420,
    -14.79425,
    -14.79430,
]

SEED_RE = 9.6160
SEED_IM = -0.3029

IM_SHIFTS = [
    0.0,
    -0.005,
    -0.010,
    -0.020,
    +0.005,
    +0.010,
]


def try_root(m, seed_re, seed_im):
    r_sp, omega_lr = find_light_ring(m)

    if r_sp is None:
        return None

    try:
        out = refine_and_polish_candidate(
            seed_re,
            seed_im,
            m,
            r_sp,
            omega_lr,
            real_axis_abs_res=float("nan"),
            is_complex_seed=True,
        )
    except Exception:
        return None

    if out is None:
        return None

    if not np.isfinite(out["f_re"]) or not np.isfinite(out["f_im"]):
        return None

    if out["abs_res"] > 1e-8:
        return None

    return (
        float(out["f_re"]),
        float(out["f_im"]),
        float(out["abs_res"]),
    )


def search_m(m, predicted_re, predicted_im):
    seeds = [(predicted_re, predicted_im + d) for d in IM_SHIFTS]

    # Also perturb Re slightly.
    for dre in [0.0, -0.01, +0.01]:
        for sre, sim in seeds:
            ans = try_root(m, sre + dre, sim)
            if ans is not None:
                return ans

    return None


def main():
    print("=" * 100)
    print("STAGE 5I-19 -- ENDPOINT MECHANISM TEST")
    print("=" * 100)
    print()
    print("Testing the localized endpoint near")
    print("m_c ≈ -14.79415")
    print()
    print("The key diagnostic is whether the branch remains smooth")
    print("with finite derivatives, or whether the local continuation")
    print("develops fold-like behavior / derivative growth.")
    print()
    print("NO q LABEL.")
    print("NO CLAIM OF NEW PHYSICS.")
    print()

    results = []

    # Start from known point near the endpoint.
    prev_re = 9.612801
    prev_im = -0.302944

    print("=" * 100)
    print("CONTINUATION TABLE")
    print("=" * 100)
    print(
        f"{'m':>11} {'Re':>14} {'Im':>14} "
        f"{'|Res|':>12} {'status':>12}"
    )

    for m in M_VALUES:
        ans = search_m(m, prev_re, prev_im)

        if ans is None:
            print(f"{m:11.5f} {'--':>14} {'--':>14} {'--':>12} {'NO ROOT':>12}")
            results.append((m, None))
        else:
            re, im, res = ans
            print(
                f"{m:11.5f} {re:14.8f} {im:14.8f} "
                f"{res:12.3e} {'ROOT':>12}"
            )
            results.append((m, ans))
            prev_re = re
            prev_im = im

    valid = [(m, x) for m, x in results if x is not None]

    print()
    print("=" * 100)
    print("LOCAL DERIVATIVE DIAGNOSTIC")
    print("=" * 100)

    if len(valid) >= 3:
        vals = []

        for i in range(1, len(valid)):
            m0, (re0, im0, _) = valid[i - 1]
            m1, (re1, im1, _) = valid[i]

            dm = m1 - m0
            dre = (re1 - re0) / dm
            dim = (im1 - im0) / dm

            vals.append((m1, dre, dim))

        print(f"{'m':>11} {'dRe/dm':>14} {'dIm/dm':>14}")
        for m, dre, dim in vals:
            print(f"{m:11.5f} {dre:14.8f} {dim:14.8f}")

        abs_dre = np.abs([x[1] for x in vals])
        abs_dim = np.abs([x[2] for x in vals])

        print()
        print(f"max |dRe/dm| = {np.max(abs_dre):.8f}")
        print(f"max |dIm/dm| = {np.max(abs_dim):.8f}")

        growth_re = abs_dre[-1] / max(abs_dre[0], 1e-30)
        growth_im = abs_dim[-1] / max(abs_dim[0], 1e-30)

        print(f"derivative growth Re = {growth_re:.4f}x")
        print(f"derivative growth Im = {growth_im:.4f}x")

        if growth_re > 2.0 or growth_im > 2.0:
            print()
            print(">>> POSSIBLE FOLD-LIKE / CRITICAL BEHAVIOR <<<")
            print("Derivative growth detected near the endpoint.")
        else:
            print()
            print(">>> NO STRONG DERIVATIVE DIVERGENCE DETECTED <<<")
            print("The branch remains locally smooth up to the last")
            print("validated point.")

    print()
    print("=" * 100)
    print("ENDPOINT CLASSIFICATION")
    print("=" * 100)

    if valid:
        last_m = valid[-1][0]
        print(f"Last validated point: m = {last_m:.5f}")

    failed = [m for m, x in results if x is None]

    if failed:
        print(f"Failed points: {failed}")
        print()
        print("Interpretation:")
        print("  The branch still terminates inside the tested interval.")
        print("  This test alone cannot establish the physical mechanism.")
    else:
        print("All tested points yielded roots.")
        print("The previously inferred endpoint requires revision.")

    print()
    print("IMPORTANT:")
    print("A finite-m endpoint is a numerical observation.")
    print("A fold/saddle-node interpretation requires an additional")
    print("local two-parameter analysis of the underlying root equations.")
    print()
    print("NO q LABEL ASSIGNED.")
    print("NO CLAIM OF NEW PHYSICS.")
    print("=" * 100)


if __name__ == "__main__":
    main()
