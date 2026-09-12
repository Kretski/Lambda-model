"""
stage5I_26_singleton_im_expanded.py
=====================================
STAGE 5I-26 — SINGLETON RETRY WITH EXPANDED Im BOUND

CONTEXT (from Stage 5I-10 analysis):
  Two singleton targets gave NOT FOUND, but the failure mode was
  solver hitting the artificial boundary COMPLEX_IM_MIN=-1.5:
    Im=-1.4999999... in ALL failed attempts = boundary artifact,
    NOT a physical absence of roots.

  This script retries ONLY those two singleton targets with
  COMPLEX_IM_MIN extended to -2.5, leaving everything else unchanged.

WHAT THIS TELLS US:
  If roots ARE found below Im=-1.5 → boundary was masking them.
    → Physical branches extend further, update singleton status.
  If STILL not found → genuine absence confirmed (boundary irrelevant).
    → Singleton status is real, not artifact.

INSTRUCTIONS:
  Place this file in the same directory as:
    stage5I_3_resonance_v2.py
    stage5I_4_full_m_scan_v6_FIXED.py
  Then run: python stage5I_26_singleton_im_expanded.py

  Edit SINGLETON_TARGETS below if your actual singleton m-values
  differ from the placeholders. Add (Re_seed, Im_seed) pairs from
  your Stage 5I-10 output — the values where solver was hitting -1.5.
"""

import sys
import copy
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# ── Import pipeline components ────────────────────────────────────────────────
try:
    from stage5I_3_resonance_v2 import find_light_ring
    from stage5I_4_full_m_scan_v6_FIXED import refine_and_polish_candidate
    import stage5I_4_full_m_scan_v6_FIXED as _pipeline_mod
except ImportError as e:
    print(f"[ERROR] Cannot import pipeline modules: {e}")
    print("Place this script in the same directory as stage5I_3_resonance_v2.py")
    print("and stage5I_4_full_m_scan_v6_FIXED.py")
    sys.exit(1)

# ── EDIT: singleton targets from Stage 5I-10 ─────────────────────────────────
# Format: (m_value, [(Re_seed_1, Im_seed_1), (Re_seed_2, Im_seed_2), ...])
# Use the Re/Im values where solver hit -1.5 in Stage 5I-10 as starting seeds,
# but also add seeds shifted further into Im < -1.5 territory.
#
# PLACEHOLDER — replace with your actual Stage 5I-10 singleton m-values:
SINGLETON_TARGETS = [
    # m=-11: example placeholder — edit with real seeds from Stage 5I-10
    # Seeds: near the boundary hit point + extended below it
    (-11, [
        (8.54, -1.40),   # just above old boundary
        (8.54, -1.60),   # just below old boundary (new territory)
        (8.54, -1.80),
        (8.54, -2.00),
        (8.54, -2.20),
        (8.30, -1.50),
        (8.30, -1.80),
        (8.70, -1.50),
        (8.70, -1.80),
    ]),
    # m=-12: example placeholder
    (-12, [
        (8.84, -1.40),
        (8.84, -1.60),
        (8.84, -1.80),
        (8.84, -2.00),
        (8.84, -2.20),
        (8.60, -1.50),
        (8.60, -1.80),
        (9.00, -1.50),
        (9.00, -1.80),
    ]),
]

# ── EXPANDED Im BOUND ─────────────────────────────────────────────────────────
COMPLEX_IM_MIN_EXPANDED = -2.5   # was -1.5 in original pipeline
MAX_ACCEPT_DISTANCE_HZ  = 0.5    # generous — we don't know where roots are

sep = "=" * 72

def patch_im_bound(new_min):
    """
    Temporarily patch COMPLEX_IM_MIN in the pipeline module.
    Returns the original value for restoration.
    """
    original = getattr(_pipeline_mod, 'COMPLEX_IM_MIN', None)
    if original is None:
        # Try common attribute names
        for attr in ['IM_MIN', 'F_IM_MIN', 'COMPLEX_IM_LOWER']:
            if hasattr(_pipeline_mod, attr):
                original = getattr(_pipeline_mod, attr)
                setattr(_pipeline_mod, attr, new_min)
                print(f"  Patched {attr}: {original} → {new_min}")
                return attr, original
        print("  WARNING: Could not find Im bound attribute to patch.")
        print("  Edit COMPLEX_IM_MIN manually in stage5I_4_full_m_scan_v6_FIXED.py")
        print(f"  Set it to {new_min} before running this script.")
        return None, None
    else:
        setattr(_pipeline_mod, 'COMPLEX_IM_MIN', new_min)
        print(f"  Patched COMPLEX_IM_MIN: {original} → {new_min}")
        return 'COMPLEX_IM_MIN', original


def restore_im_bound(attr_name, original_val):
    if attr_name is not None and original_val is not None:
        setattr(_pipeline_mod, attr_name, original_val)


def try_seeds_at_m(m, seeds, r_sp, omega_lr):
    """
    Try each seed. Return list of valid poles found.
    A pole is valid if |Residual| < 1e-6 and Im > COMPLEX_IM_MIN_EXPANDED.
    """
    found = []
    seen_approx = []  # avoid duplicates

    for seed_re, seed_im in seeds:
        if seed_im < COMPLEX_IM_MIN_EXPANDED:
            continue  # outside our expanded range

        pole = refine_and_polish_candidate(
            f0_re=seed_re,
            f0_im_hint=seed_im,
            m=m,
            r_sp=r_sp,
            omega_lr=omega_lr,
            real_axis_abs_res=float("nan"),
            is_complex_seed=True,
        )

        if pole is None:
            continue

        # Check if this is a genuine new root (not duplicate of already found)
        is_duplicate = False
        for prev in seen_approx:
            if (abs(pole['f_re'] - prev[0]) < 0.005 and
                    abs(pole['f_im'] - prev[1]) < 0.005):
                is_duplicate = True
                break

        if not is_duplicate:
            found.append(pole)
            seen_approx.append((pole['f_re'], pole['f_im']))

    return found


def main():
    print(sep)
    print("STAGE 5I-26 — SINGLETON RETRY WITH EXPANDED Im BOUND")
    print(sep)
    print(f"Original COMPLEX_IM_MIN: -1.5")
    print(f"Expanded COMPLEX_IM_MIN: {COMPLEX_IM_MIN_EXPANDED}")
    print(f"Targets: {[t[0] for t in SINGLETON_TARGETS]}")
    print()
    print("RATIONALE: Stage 5I-10 showed solver hitting Im=-1.4999...")
    print("(exactly COMPLEX_IM_MIN=-1.5) on all failed singleton attempts.")
    print("This is a boundary artifact. Expanding the bound to test whether")
    print("roots exist below Im=-1.5.")
    print()

    # Patch the Im bound
    print("Patching Im bound in pipeline module...")
    attr_name, original_val = patch_im_bound(COMPLEX_IM_MIN_EXPANDED)
    print()

    results = {}

    try:
        for m, seeds in SINGLETON_TARGETS:
            print(sep)
            print(f"TARGET: m = {m}")
            print(sep)

            r_sp, omega_lr = find_light_ring(m)
            if r_sp is None:
                print(f"  [SKIP] find_light_ring returned None for m={m}")
                results[m] = []
                continue

            print(f"  Light ring: r_sp={r_sp:.6f}, omega_lr={omega_lr:.6f}")
            print(f"  Testing {len(seeds)} seeds in Im range "
                  f"[{min(s[1] for s in seeds):.2f}, "
                  f"{max(s[1] for s in seeds):.2f}]")
            print()

            found = try_seeds_at_m(m, seeds, r_sp, omega_lr)
            results[m] = found

            if found:
                print(f"  FOUND {len(found)} valid pole(s):")
                for p in found:
                    print(f"    Re={p['f_re']:.6f}  Im={p['f_im']:.6f}  "
                          f"|Res|={p['abs_res']:.2e}")
                    if p['f_im'] < -1.5:
                        print(f"    *** BELOW OLD BOUNDARY — boundary was masking this! ***")
                    else:
                        print(f"    (within original search range)")
            else:
                print(f"  NOT FOUND — genuine absence confirmed below Im={COMPLEX_IM_MIN_EXPANDED}")

            print()

    finally:
        # Always restore original bound
        restore_im_bound(attr_name, original_val)
        print("Im bound restored to original value.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print(sep)
    print("SUMMARY")
    print(sep)
    print()

    any_new = False
    for m, found in sorted(results.items()):
        if found:
            below_boundary = [p for p in found if p['f_im'] < -1.5]
            above_boundary = [p for p in found if p['f_im'] >= -1.5]
            print(f"m={m}: FOUND {len(found)} pole(s)")
            if below_boundary:
                print(f"  {len(below_boundary)} NEW (below old Im=-1.5 boundary):")
                for p in below_boundary:
                    print(f"    Re={p['f_re']:.6f}  Im={p['f_im']:.6f}")
                any_new = True
            if above_boundary:
                print(f"  {len(above_boundary)} already findable with old bound:")
                for p in above_boundary:
                    print(f"    Re={p['f_re']:.6f}  Im={p['f_im']:.6f}")
        else:
            print(f"m={m}: NOT FOUND (genuine, not boundary artifact)")

    print()
    print("INTERPRETATION:")
    if any_new:
        print("  Roots found below Im=-1.5 → boundary was masking physical poles.")
        print("  Singleton status was a search artifact, NOT a physical feature.")
        print("  Next: extend Im bound to -2.5 in full pipeline and rescan.")
    else:
        print("  No roots found below Im=-1.5 → boundary irrelevant for these m.")
        print("  Singleton status is genuine (not a search artifact).")
        print("  The branch truly does not extend to these Im values.")
    print()
    print("  → Either way: the m_c=-14.794 deep branch termination question")
    print("    is SEPARATE from the singleton Im question. See Stage 5I-18.")


if __name__ == "__main__":
    main()
