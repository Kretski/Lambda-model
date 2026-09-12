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
# STAGE 5I-22 -- INTERMEDIATE ROOT VALIDATION
#
# Question:
#   Are the relatively large FAST-grid minima at
#   m=-14.3, -14.5, -14.7 caused by grid resolution,
#   or is there genuinely no validated root there?
#
# Procedure:
#   1. Dense local grid using EXACT safe_complex_res()
#   2. Select several best grid candidates
#   3. Feed them into the ORIGINAL validated
#      refine_and_polish_candidate() pipeline
#
# NO substitute residual.
# NO q label.
# NO claim of new physics.
# ============================================================

KNOWN = {
    -14.0: (9.3995509552, -0.3086180377),
    -13.0: (9.1218868177, -0.3173351972),
}

M_VALUES = [-14.3, -14.5, -14.7]

N = 101
HALF_WINDOW = 0.015
N_BEST = 12


def predicted_position(m):
    re1, im1 = KNOWN[-14.0]
    re2, im2 = KNOWN[-13.0]

    dre_dm = re2 - re1
    dim_dm = im2 - im1

    re = re1 + dre_dm * (m + 14.0)
    im = im1 + dim_dm * (m + 14.0)

    return re, im


def grid_candidates(m, r_sp, omega_lr):

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

    candidates = []

    for re in re_values:
        for im in im_values:

            z = safe_complex_res(
                re,
                im,
                m,
                r_sp,
                omega_lr,
            )

            candidates.append(
                (abs(z), re, im)
            )

    candidates.sort(key=lambda x: x[0])

    # Remove nearly identical seeds.
    selected = []

    for candidate in candidates:

        _, re, im = candidate

        too_close = False

        for _, sre, sim in selected:
            if np.hypot(re - sre, im - sim) < 0.001:
                too_close = True
                break

        if not too_close:
            selected.append(candidate)

        if len(selected) >= N_BEST:
            break

    return selected


def main():

    print("=" * 100)
    print("STAGE 5I-22 -- INTERMEDIATE ROOT VALIDATION")
    print("=" * 100)

    print()
    print("Residual source:")
    print("  stage5I-4 -> safe_complex_res()")
    print()
    print("Root validation:")
    print("  ORIGINAL refine_and_polish_candidate()")
    print()
    print(f"Grid: {N} x {N}")
    print(f"Window: +/- {HALF_WINDOW:.3f} Hz")
    print(f"Best independent seeds per m: {N_BEST}")
    print()

    all_results = []

    for m in M_VALUES:

        print()
        print("=" * 100)
        print(f"m = {m:.5f}")
        print("=" * 100)

        r_sp, omega_lr = find_light_ring(m)

        if r_sp is None:
            print("find_light_ring FAILED")
            continue

        pred_re, pred_im = predicted_position(m)

        print(
            f"Predicted branch position: "
            f"Re={pred_re:.10f}, Im={pred_im:.10f}"
        )

        candidates = grid_candidates(
            m,
            r_sp,
            omega_lr,
        )

        print()
        print("BEST GRID CANDIDATES")
        print("-" * 100)

        for i, (val, re, im) in enumerate(candidates, 1):

            print(
                f"{i:2d}: |Res|={val:.8e} "
                f"seed=({re:.8f},{im:.8f})"
            )

        print()
        print("ORIGINAL SOLVER VALIDATION")
        print("-" * 100)

        found = []

        for i, (grid_res, seed_re, seed_im) in enumerate(
            candidates, 1
        ):

            print()
            print(
                f"Seed #{i}: "
                f"Re={seed_re:.10f}, "
                f"Im={seed_im:.10f}"
            )

            polished = refine_and_polish_candidate(
                seed_re,
                seed_im,
                m,
                r_sp,
                omega_lr,
                real_axis_abs_res=float("nan"),
                is_complex_seed=True,
            )

            if polished is None:

                print("  -> NOT VALIDATED")

                continue

            print(
                f"  -> VALIDATED ROOT: "
                f"Re={polished['f_re']:.10f} "
                f"Im={polished['f_im']:.10f} "
                f"|Res|={polished['abs_res']:.4e}"
            )

            found.append(polished)

        # Cluster validated roots.
        unique = []

        for p in found:

            duplicate = False

            for u in unique:

                if np.hypot(
                    p["f_re"] - u["f_re"],
                    p["f_im"] - u["f_im"],
                ) < 1e-5:

                    duplicate = True
                    break

            if not duplicate:
                unique.append(p)

        print()
        print("-" * 100)

        if unique:

            print(
                f"VALIDATED ROOTS FOUND: {len(unique)}"
            )

            for p in unique:

                print(
                    f"  Re={p['f_re']:.10f} "
                    f"Im={p['f_im']:.10f} "
                    f"|Res|={p['abs_res']:.4e}"
                )

        else:

            print(
                "NO VALIDATED ROOTS FOUND from any of "
                f"{len(candidates)} independent seeds."
            )

        all_results.append(
            (
                m,
                candidates[0][0],
                len(unique),
            )
        )

    print()
    print("=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    print()
    print(
        f"{'m':>10} {'best grid |Res|':>20} "
        f"{'validated roots':>18}"
    )
    print("-" * 55)

    for m, best_res, nroots in all_results:

        print(
            f"{m:10.5f} "
            f"{best_res:20.10e} "
            f"{nroots:18d}"
        )

    print()
    print("=" * 100)
    print("INTERPRETATION")
    print("=" * 100)

    print("""
The decisive quantity here is NOT the grid minimum alone.

If the ORIGINAL solver converges from several independent
best-grid seeds to the same low-residual root, then the
intermediate branch exists and the FAST grid was simply not
centered exactly enough.

If NONE of the independent seeds produces a validated root,
despite the dense grid and the unchanged solver, that is
substantially stronger evidence that the branch is absent
at that m.

This test still does not determine the physical mechanism
of any endpoint.

NO q LABEL.
NO CLAIM OF NEW PHYSICS.
""")

    print("=" * 100)


if __name__ == "__main__":
    main()
