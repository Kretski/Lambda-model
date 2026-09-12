"""
stage5I_targeted_m15_rescan.py
================================
STAGE 5I -- targeted dense-grid rescan for the SINGLE mode m=-15,
following the exact precedent of the m=-11 targeted test referenced
in generalized_branch_resolution.py's docstring.

WHY THIS SCRIPT EXISTS
-----------------------
The v6_FIXED full m-scan's automatic complex-plane discovery grid for
m=-15 sampled only 8 (Re,Im) starting points, with Im seeds at
[-0.10, -0.20, -0.35, -0.50, -0.70, -0.90, -1.20, -1.20] -- leaving a
gap between -0.10 and -0.35 exactly where the q=2 branch is expected
to sit, by interpolation from its validated neighbors:

    m=-16  q=2:  Re=10.096 Hz  Im=-0.104 Hz  (|Res|=1.08e-15, validated)
    m=-14  q=2:  Re=9.400  Hz  Im=-0.309 Hz  (|Res|=1.37e-14, validated)

The single best candidate the coarse discovery DID find near there
(Re=9.896, Im=-0.10 seed) only polished to |Res|=2.5e-2 -- roughly
13 orders of magnitude worse than a validated pole -- consistent with
the discovery grid having missed the true root, not with the branch
genuinely terminating at m=-15.

WHAT THIS SCRIPT DOES
-----------------------
Runs a much denser local scan restricted to the gap: Im in roughly
[-0.40, -0.05] Hz (step 0.01 Hz, ~35 seeds instead of the 2 that fell
in this range in v6_FIXED), Re seeded near the value v6_FIXED's own
NM step already pulled candidates toward (~9.85-9.90 Hz), then
polishes every candidate through the SAME refine_complex_resonance()
pipeline already validated for every other m in the repo -- no new
method, no new tolerance, just finer coverage of the SAME search
space the v6_FIXED discovery grid under-sampled.

Reports every candidate's |Res| and the best one found, so a genuine
"no pole here" result (all candidates stuck above ~1e-2/1e-3, as in
v6_FIXED's own discovery) is visibly different from a genuine find
(|Res| dropping to ~1e-14/1e-15 as at every other validated m).

Does not touch m_c, does not touch any other m, does not modify
stage5I_3_resonance_v2.py or the v6_FIXED results file.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stage5I_3_resonance_v2 import (
    find_light_ring,
    resonance_factor,
    refine_complex_resonance,
)

M_TARGET = -15

# The gap in v6_FIXED's discovery grid, informed by the two validated
# neighbors (m=-16 Im=-0.104, m=-14 Im=-0.309): search comfortably
# around and between them.
IM_SCAN_MIN = -0.40
IM_SCAN_MAX = -0.05
IM_SCAN_STEP = 0.01

# Re seed: v6_FIXED's own NM polish already pulled every candidate in
# this Im range toward Re ~ 9.85-9.90 Hz, so seed there and let
# refine_complex_resonance's own re_half_width cover nearby drift.
RE_SEED = 9.87
RE_HALF_WIDTH = 0.30


def main():
    print("=" * 100)
    print(f"STAGE 5I -- TARGETED DENSE RESCAN, m={M_TARGET}")
    print("=" * 100)

    r_sp, omega_lr = find_light_ring(M_TARGET)
    if r_sp is None:
        print(f"ERROR: no light ring found for m={M_TARGET}")
        return
    print(f"light ring: r_sp={r_sp * 1e3:.6f} mm  "
          f"f_lr={omega_lr / (2 * np.pi):.10f} Hz")
    print()

    im_seeds = np.arange(IM_SCAN_MIN, IM_SCAN_MAX + 1e-9, IM_SCAN_STEP)
    print(f"scanning {len(im_seeds)} Im-seeds in [{IM_SCAN_MIN}, {IM_SCAN_MAX}] Hz, "
          f"step={IM_SCAN_STEP} Hz, Re seed={RE_SEED} Hz")
    print()

    results = []
    for im0 in im_seeds:
        refined = refine_complex_resonance(
            f0=RE_SEED,
            im0=float(im0),
            resonance_factor=resonance_factor,
            m=M_TARGET,
            re_half_width=RE_HALF_WIDTH,
            im_min=IM_SCAN_MIN - 0.1,
            im_max=IM_SCAN_MAX + 0.1,
        )
        if refined is None:
            print(f"  im0={im0:+.3f}  ->  refinement FAILED (no result object)")
            continue
        results.append(refined)
        print(f"  im0={im0:+.3f}  ->  f_re={refined['f_re']:.6f} Hz  "
              f"f_im={refined['f_im']:.6f} Hz  |Res|={refined['abs_res']:.4e}  "
              f"success={refined['success']}")

    print()
    print("=" * 100)
    if not results:
        print("NO VALIDATED POLE -- every seed in this Im range failed to refine at all.")
        return

    best = min(results, key=lambda r: r["abs_res"])
    print("BEST CANDIDATE FOUND:")
    print(f"    f_re  = {best['f_re']:.10f} Hz")
    print(f"    f_im  = {best['f_im']:.10f} Hz")
    print(f"    |Res| = {best['abs_res']:.6e}")
    print(f"    success = {best['success']}")
    print()

    if best["abs_res"] < 1e-8:
        print("VERDICT: genuine pole found (|Res| at the level of other validated poles).")
        print("This would mean the earlier 'gap at m=-15' was a discovery-grid artifact,")
        print("NOT evidence of branch termination near m_c.")
    else:
        print("VERDICT: best |Res| is still far above the ~1e-14/1e-15 level seen at every")
        print("other validated m. This is now much stronger (denser, targeted) evidence")
        print("that no q=2-type pole exists in this Im range for m=-15 -- worth taking")
        print("seriously as a genuine feature, though a still-denser/wider scan and an")
        print("independent re-implementation (as in forensic_continuation_comparison.py)")
        print("should confirm it before calling it a termination.")


if __name__ == "__main__":
    main()
