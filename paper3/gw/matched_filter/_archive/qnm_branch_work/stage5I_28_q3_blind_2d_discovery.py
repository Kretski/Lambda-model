"""
STAGE 5I-28 — BLIND 2D COMPLEX-PLANE DISCOVERY
================================================

Purpose
-------
Determine whether the model contains ANY mathematical poles in the
intermediate-damping region above the validated q=2 Root-B pole.

This is deliberately NOT a continuation search.

Stage 5I-27 asked:
    "Can a q=3 branch be reached continuously from q=2?"

Stage 5I-28 asks:
    "Are there any roots here at all?"

IMPORTANT
---------
No experimental frequency is imported or used anywhere in the discovery.

The search region is defined ONLY relative to the model's validated
q=2 anchor.

A root is accepted only if:
    |Res| < RES_ACCEPT

All accepted roots are deduplicated and reported.

Known q=2 is identified only AFTER mathematical roots are found.
"""

import sys
import csv
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


# ---------------------------------------------------------------------------
# SAME VALIDATED PIPELINE
# ---------------------------------------------------------------------------

try:
    from stage5I_3_resonance_v2 import find_light_ring
    from stage5I_4_full_m_scan_v6_FIXED import refine_and_polish_candidate
    import stage5I_4_full_m_scan_v6_FIXED as _pipeline_mod

except ImportError as e:
    print(f"[ERROR] Cannot import pipeline modules: {e}")
    print()
    print("Place this script in the same directory as:")
    print("  stage5I_3_resonance_v2.py")
    print("  stage5I_4_full_m_scan_v6_FIXED.py")
    sys.exit(1)


# ===========================================================================
# CONFIG
# ===========================================================================

ANCHOR_M = -14

# Validated model q=2 Root-B anchor.
#
# This is a MODEL quantity, not an experimental quantity.
#
ANCHOR_Q2_RE = 9.5967
ANCHOR_Q2_IM = -0.1358


# ---------------------------------------------------------------------------
# BLIND MODEL-RELATIVE SEARCH REGION
# ---------------------------------------------------------------------------
#
# We deliberately do NOT specify an experimental q=3 frequency.
#
# The region is defined relative to q=2:
#
#   Re: q2 - 0.8 ... q2 + 1.5 Hz
#   Im: 0 ... -2.5 Hz
#
# This covers:
#   - q=2 itself
#   - intermediate damping
#   - the region occupied by the previously observed deep poles
#
# It is therefore a discovery scan rather than a targeted q=3 fit.
#

RE_MIN_OFFSET = -0.80
RE_MAX_OFFSET = +1.50

IM_MAX = 0.0
IM_MIN = -2.5


# Seed density.
#
# Increase if the solver landscape appears complicated.
#
N_RE = 17
N_IM = 26


# Root acceptance.
RES_ACCEPT = 1.0e-8

# Deduplication tolerance in Hz.
DEDUP_TOL = 5.0e-3


# Known q=2 exclusion/identification tolerance.
#
# This is NOT used to prevent discovery.
# It is applied only after roots have been found.
#
Q2_IDENT_TOL_RE = 0.05
Q2_IDENT_TOL_IM = 0.05


sep = "=" * 78


# ===========================================================================
# IM BOUND PATCH
# ===========================================================================

def patch_im_bound(new_min):
    """
    Temporarily expand the complex Im lower bound in the validated pipeline.
    """
    for attr in (
        "COMPLEX_IM_MIN",
        "IM_MIN",
        "F_IM_MIN",
        "COMPLEX_IM_LOWER",
    ):
        if hasattr(_pipeline_mod, attr):
            original = getattr(_pipeline_mod, attr)
            setattr(_pipeline_mod, attr, new_min)
            return attr, original

    print(
        "WARNING: could not find an Im-bound attribute to patch; "
        "pipeline default may clip deep seeds."
    )

    return None, None


def restore_im_bound(attr, original):
    if attr is not None:
        setattr(_pipeline_mod, attr, original)


# ===========================================================================
# SINGLE SEED
# ===========================================================================

def try_seed(m, seed_re, seed_im, r_sp, omega_lr):
    """
    Refine one complex seed using the SAME validated NM -> hybr pipeline.
    """

    if seed_im < IM_MIN:
        return None

    try:

        pole = refine_and_polish_candidate(
            f0_re=seed_re,
            f0_im_hint=seed_im,
            m=m,
            r_sp=r_sp,
            omega_lr=omega_lr,
            real_axis_abs_res=float("nan"),
            is_complex_seed=True,
        )

    except Exception:
        return None

    if pole is None:
        return None

    try:
        fre = float(pole["f_re"])
        fim = float(pole["f_im"])
        abs_res = float(pole.get("abs_res", np.inf))
    except Exception:
        return None

    if not (
        np.isfinite(fre)
        and np.isfinite(fim)
        and np.isfinite(abs_res)
    ):
        return None

    if abs_res >= RES_ACCEPT:
        return None

    # Physical pole must be in lower half plane.
    if fim > 1e-9:
        return None

    return {
        "f_re": fre,
        "f_im": fim,
        "abs_res": abs_res,
    }


# ===========================================================================
# DEDUPLICATION
# ===========================================================================

def is_same_root(p, q):
    return (
        abs(p["f_re"] - q["f_re"]) < DEDUP_TOL
        and
        abs(p["f_im"] - q["f_im"]) < DEDUP_TOL
    )


def deduplicate(roots):
    """
    Keep one representative for each distinct mathematical root.
    """

    unique = []

    # Best residual first.
    roots = sorted(
        roots,
        key=lambda p: p["abs_res"]
    )

    for p in roots:

        duplicate = False

        for q in unique:
            if is_same_root(p, q):
                duplicate = True
                break

        if not duplicate:
            unique.append(p)

    # Sort by Re, then Im for reproducible output.
    unique.sort(
        key=lambda p: (
            p["f_re"],
            p["f_im"],
        )
    )

    return unique


# ===========================================================================
# ROOT CLASSIFICATION
# ===========================================================================

def classify_root(p):
    """
    Classification is descriptive only.

    No experimental information is used.
    """

    dre = p["f_re"] - ANCHOR_Q2_RE
    dim = p["f_im"] - ANCHOR_Q2_IM

    if (
        abs(dre) < Q2_IDENT_TOL_RE
        and
        abs(dim) < Q2_IDENT_TOL_IM
    ):
        return "known q=2 anchor"

    if dre > 0.0 and dim < 0.0:
        return "new root above q=2"

    if dre > 0.0:
        return "new root, higher Re"

    if dre < 0.0:
        return "new root, lower Re"

    return "new root"


# ===========================================================================
# MAIN
# ===========================================================================

def main():

    print(sep)
    print("STAGE 5I-28 — BLIND 2D COMPLEX-PLANE DISCOVERY")
    print(sep)

    print()
    print("EXPERIMENT-BLIND DISCOVERY")
    print("--------------------------")

    print(f"Anchor m              : {ANCHOR_M}")
    print(
        f"Model q=2 Root-B      : "
        f"Re={ANCHOR_Q2_RE:.4f}  Im={ANCHOR_Q2_IM:.4f}"
    )

    print()
    print("2D search region")
    print("----------------")

    print(
        f"Re offset from q=2   : "
        f"[{RE_MIN_OFFSET:+.2f}, {RE_MAX_OFFSET:+.2f}] Hz"
    )

    print(
        f"Absolute Re region    : "
        f"[{ANCHOR_Q2_RE + RE_MIN_OFFSET:.4f}, "
        f"{ANCHOR_Q2_RE + RE_MAX_OFFSET:.4f}] Hz"
    )

    print(
        f"Im region             : "
        f"[{IM_MIN:.2f}, {IM_MAX:.2f}] Hz"
    )

    print(
        f"Seed grid             : "
        f"{N_RE} x {N_IM} = {N_RE * N_IM}"
    )

    print(f"|Res| acceptance      : {RES_ACCEPT:.1e}")
    print(f"Dedup tolerance       : {DEDUP_TOL:.3e} Hz")

    print()
    print("IMPORTANT:")
    print("  No experimental q=3 frequency is imported.")
    print("  No experimental q=3 frequency is used as a seed.")
    print("  No experimental value is used for root selection.")
    print("  Discovery is performed over the model-relative complex plane.")
    print()

    # -----------------------------------------------------------------------
    # LIGHT RING
    # -----------------------------------------------------------------------

    r = find_light_ring(ANCHOR_M)

    if r is None or r[0] is None:
        print("[ERROR] Anchor light ring is None.")
        return

    r_sp, omega_lr = r

    # -----------------------------------------------------------------------
    # PATCH IM BOUND
    # -----------------------------------------------------------------------

    attr, original = patch_im_bound(IM_MIN)

    try:

        # -------------------------------------------------------------------
        # BUILD 2D MODEL-ONLY GRID
        # -------------------------------------------------------------------

        re_values = np.linspace(
            ANCHOR_Q2_RE + RE_MIN_OFFSET,
            ANCHOR_Q2_RE + RE_MAX_OFFSET,
            N_RE,
        )

        im_values = np.linspace(
            IM_MAX,
            IM_MIN,
            N_IM,
        )

        seeds = [
            (float(re0), float(im0))
            for re0 in re_values
            for im0 in im_values
        ]

        print(sep)
        print("RUNNING BLIND 2D DISCOVERY")
        print(sep)

        roots_raw = []

        total = len(seeds)

        for idx, (seed_re, seed_im) in enumerate(seeds, start=1):

            p = try_seed(
                ANCHOR_M,
                seed_re,
                seed_im,
                r_sp,
                omega_lr,
            )

            if p is not None:

                roots_raw.append(p)

                print(
                    f"  seed {idx:4d}/{total}: "
                    f"seed=({seed_re:8.4f}, {seed_im:8.4f}) "
                    f"-> root=({p['f_re']:10.7f}, "
                    f"{p['f_im']:10.7f}) "
                    f"|Res|={p['abs_res']:.3e}"
                )

            elif idx % max(1, total // 10) == 0:

                print(
                    f"  progress: {idx:4d}/{total}"
                )

        # -------------------------------------------------------------------
        # DEDUPLICATE
        # -------------------------------------------------------------------

        roots = deduplicate(roots_raw)

    finally:

        restore_im_bound(attr, original)

    # -----------------------------------------------------------------------
    # RESULTS
    # -----------------------------------------------------------------------

    print()
    print(sep)
    print("BLIND 2D DISCOVERY RESULTS")
    print(sep)

    print(
        f"Accepted solver hits   : {len(roots_raw)}"
    )

    print(
        f"Distinct mathematical roots : {len(roots)}"
    )

    print()

    if not roots:

        print("NO ROOTS FOUND.")
        print()
        print(
            "Interpretation:"
        )
        print(
            "  No mathematical pole satisfying the residual criterion"
        )
        print(
            "  was located anywhere in the scanned model-relative region."
        )
        print()
        print(
            "This is substantially stronger than Stage 5I-27:"
        )
        print(
            "  continuation failure + blind 2D discovery failure."
        )
        print()
        print(
            "It still does NOT prove absence outside the scanned region."
        )

    else:

        print(
            f"{'#':>3} "
            f"{'Re [Hz]':>12} "
            f"{'Im [Hz]':>12} "
            f"{'|Res|':>12} "
            f"{'classification'}"
        )

        print("-" * 78)

        for i, p in enumerate(roots, start=1):

            cls = classify_root(p)

            print(
                f"{i:3d} "
                f"{p['f_re']:12.7f} "
                f"{p['f_im']:12.7f} "
                f"{p['abs_res']:12.3e} "
                f"{cls}"
            )

    # -----------------------------------------------------------------------
    # SAVE CSV
    # -----------------------------------------------------------------------

    out_csv = HERE / "stage5I_28_blind_2d_roots.csv"

    with out_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "m",
                "f_re_Hz",
                "f_im_Hz",
                "abs_res",
                "classification",
                "delta_Re_from_q2_Hz",
                "delta_Im_from_q2_Hz",
            ]
        )

        for p in roots:

            writer.writerow(
                [
                    ANCHOR_M,
                    f"{p['f_re']:.9f}",
                    f"{p['f_im']:.9f}",
                    f"{p['abs_res']:.6e}",
                    classify_root(p),
                    f"{p['f_re'] - ANCHOR_Q2_RE:.9f}",
                    f"{p['f_im'] - ANCHOR_Q2_IM:.9f}",
                ]
            )

    print()
    print(f"Saved roots -> {out_csv}")

    # -----------------------------------------------------------------------
    # FINAL LOGIC
    # -----------------------------------------------------------------------

    print()
    print(sep)
    print("STAGE 5I-28 INTERPRETATION")
    print(sep)

    if not roots:

        print()
        print(
            "RESULT: NO DISTINCT ROOT LOCATED"
        )

        print()
        print(
            "Combined with Stage 5I-27:"
        )

        print(
            "  - continuation could not reach q=3;"
        )

        print(
            "  - blind 2D discovery found no pole in the scanned region."
        )

        print()
        print(
            "The model therefore provides no mathematical q=3 candidate"
        )

        print(
            "in this tested complex-plane region."
        )

        print()
        print(
            "This is a model-domain negative result, not an experimental fit."
        )

    else:

        new_roots = [
            p for p in roots
            if classify_root(p) != "known q=2 anchor"
        ]

        print()
        print(
            f"Distinct roots found: {len(roots)}"
        )

        print(
            f"Roots other than known q=2: {len(new_roots)}"
        )

        if new_roots:

            print()
            print(
                "IMPORTANT:"
            )

            print(
                "  At least one new mathematical pole exists in the"
            )

            print(
                "  scanned model-relative region."
            )

            print()
            print(
                "The next question is branch identity, not existence."
            )

            print(
                "Stage 5I-29 should test whether these roots form a"
            )

            print(
                "smooth family under m continuation."
            )

        else:

            print()
            print(
                "Only the known q=2 anchor was recovered."
            )

            print(
                "No distinct higher branch was discovered in this region."
            )

    print()
    print(sep)


if __name__ == "__main__":
    main()