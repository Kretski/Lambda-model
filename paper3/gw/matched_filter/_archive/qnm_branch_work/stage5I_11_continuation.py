"""
STAGE 5I-11 -- CONTINUATION TEST OF THE DEEP ROOT BRANCH
=========================================================
Tests whether the deep-decay roots can be continuously followed
in m using the EXACT solver/polishing pipeline from
stage5I_4_full_m_scan_v6_FIXED.py.

No new physics, residual, tolerance, or branch criterion is introduced.

Strategy:
  FORWARD:
      start from the independently validated m=-10 targeted root
      and use each validated root as the seed for the next m.

  BACKWARD:
      start from the independently observed m=-14 deep root
      and continue toward m=-13 ... m=-10.

The purpose is NOT to assign q labels.
The purpose is only to test whether the deep roots form a continuous
solver branch.

A root is accepted only if the existing refine_and_polish_candidate()
accepts it under the same residual validation as Stage 5I-10.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stage5I_4_full_m_scan_v6_FIXED import (
    find_light_ring,
    refine_and_polish_candidate,
)


# Independently validated targeted roots.
# These are used ONLY as starting anchors.
ANCHOR_FORWARD = {
    "m": -10,
    "re": 8.2426577969,
    "im": -0.3571800363,
}

ANCHOR_BACKWARD = {
    "m": -14,
    "re": 9.39955,
    "im": -0.30862,
}


def continue_one_step(seed_re, seed_im, m):
    print("=" * 100)
    print(
        f"continuation @ m={m:3d} "
        f"(seed Re={seed_re:.10f}, Im={seed_im:.10f})"
    )
    print("=" * 100)

    r_sp, omega_lr = find_light_ring(m)

    if r_sp is None:
        print("  find_light_ring FAILED")
        return None

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
        print("  RESULT: NOT FOUND")
        return None

    re = polished["f_re"]
    im = polished["f_im"]
    res = polished["abs_res"]

    print(
        f"  RESULT: FOUND  "
        f"Re={re:.10f} Hz  "
        f"Im={im:.10f} Hz  "
        f"|Res|={res:.3e}"
    )

    return {
        "m": m,
        "re": re,
        "im": im,
        "res": res,
    }


def run_forward():
    print()
    print("#" * 100)
    print("FORWARD CONTINUATION")
    print("#" * 100)

    current = dict(ANCHOR_FORWARD)
    results = []

    print(
        f"\nANCHOR m={current['m']}: "
        f"Re={current['re']:.10f}, Im={current['im']:.10f}"
    )

    results.append({
        "m": current["m"],
        "re": current["re"],
        "im": current["im"],
        "res": 8.059e-14,
        "anchor": True,
    })

    for m in range(current["m"] + 1, -16, -1):
        out = continue_one_step(
            current["re"],
            current["im"],
            m,
        )

        if out is None:
            print()
            print(
                f"  FORWARD CONTINUATION STOPPED at m={m}."
            )
            break

        out["anchor"] = False
        results.append(out)

        current = {
            "m": m,
            "re": out["re"],
            "im": out["im"],
        }

    return results


def run_backward():
    print()
    print("#" * 100)
    print("BACKWARD CONTINUATION")
    print("#" * 100)

    current = dict(ANCHOR_BACKWARD)
    results = []

    print(
        f"\nANCHOR m={current['m']}: "
        f"Re={current['re']:.10f}, Im={current['im']:.10f}"
    )

    results.append({
        "m": current["m"],
        "re": current["re"],
        "im": current["im"],
        "res": float("nan"),
        "anchor": True,
    })

    for m in range(current["m"] - 1, -9):
        out = continue_one_step(
            current["re"],
            current["im"],
            m,
        )

        if out is None:
            print()
            print(
                f"  BACKWARD CONTINUATION STOPPED at m={m}."
            )
            break

        out["anchor"] = False
        results.append(out)

        current = {
            "m": m,
            "re": out["re"],
            "im": out["im"],
        }

    return results


def print_summary(forward, backward):
    print()
    print("#" * 100)
    print("CONTINUATION SUMMARY")
    print("#" * 100)

    print("\nFORWARD:")
    for r in sorted(forward, key=lambda x: x["m"]):
        print(
            f"  m={r['m']:3d}  "
            f"Re={r['re']:.8f}  "
            f"Im={r['im']:.8f}  "
            f"|Res|={r['res']:.3e}"
        )

    print("\nBACKWARD:")
    for r in sorted(backward, key=lambda x: x["m"]):
        print(
            f"  m={r['m']:3d}  "
            f"Re={r['re']:.8f}  "
            f"Im={r['im']:.8f}  "
            f"|Res|={r['res']:.3e}"
        )

    print()
    print("#" * 100)
    print("INTERPRETATION")
    print("#" * 100)

    forward_ms = {r["m"] for r in forward}
    backward_ms = {r["m"] for r in backward}

    print(
        "\nThis test does NOT assign q labels."
    )
    print(
        "It tests only whether the validated deep roots can be "
        "continued smoothly in m."
    )

    if -15 in forward_ms:
        print(
            "\nFORWARD reached m=-15: "
            "strong evidence that the deep branch extends through "
            "the previously missing endpoint."
        )
    else:
        print(
            "\nFORWARD did not reach m=-15: "
            "the branch is not yet demonstrated through m=-15."
        )

    if -10 in backward_ms:
        print(
            "BACKWARD reached m=-10: "
            "independent reverse-direction continuation succeeds."
        )
    else:
        print(
            "BACKWARD did not reach m=-10."
        )


def main():
    forward = run_forward()
    backward = run_backward()
    print_summary(forward, backward)


if __name__ == "__main__":
    main()