"""
stage5I_targeted_m15_rescan_v2.py
===================================
Corrected targeted dense rescan for m=-15, using the EXACT two-stage
pipeline (Nelder-Mead seed -> hybr polish) from
stage5I_4_full_m_scan_v6_FIXED.py -- imported directly, not
reimplemented -- so results are directly comparable to every other
validated m in that file's own output.

WHY v1 (stage5I_targeted_m15_rescan.py) WAS INCOMPLETE
--------------------------------------------------------
v1 used ONLY refine_complex_resonance() from stage5I_3_resonance_v2.py,
which is pure Nelder-Mead minimization of log(1+|Res|) -- it has NO
hybr polish step. That is NOT the same pipeline that produced the
validated |Res|~1e-14/1e-15 poles everywhere else in v6_FIXED's
results: those all went through an ADDITIONAL
scipy.optimize.root(method='hybr') polish (polish_root() in
v6_FIXED), against acceptance threshold MAX_VALIDATED_ABS_RES=1e-8.
v1's NM-only floor of |Res|~1.7e-2 therefore only tested the first of
v6_FIXED's two stages -- not a fair test of whether v6_FIXED's actual
pipeline finds a pole here.

WHAT THIS VERSION DOES
------------------------
Imports complex_minimize_from_seed() and polish_root() directly from
stage5I_4_full_m_scan_v6_FIXED (no reimplementation, no new
tolerances), applies the SAME NM->hybr pipeline to a much denser
Im-seed grid than v6_FIXED's own 8-point discovery grid, across the
[-0.40, -0.05] Hz gap identified from the original log (where the
q=2 branch is expected by interpolation from its validated neighbors
at m=-16 and m=-14), and reports every seed's fate through BOTH
stages, plus the same "VALIDATED" / "REJECTED" language v6_FIXED
itself uses.

Does not touch m_c, does not touch any other m, does not modify
v6_FIXED's own results file.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stage5I_3_resonance_v2 import find_light_ring
from stage5I_4_full_m_scan_v6_FIXED import (
    complex_minimize_from_seed,
    polish_root,
    MAX_VALIDATED_ABS_RES,
)

M_TARGET = -15
RE_SEED = 9.87

IM_SCAN_MIN = -0.40
IM_SCAN_MAX = -0.05
IM_SCAN_STEP = 0.01


def main():
    print("=" * 100)
    print(f"STAGE 5I -- CORRECTED TARGETED DENSE RESCAN (NM + hybr), m={M_TARGET}")
    print("=" * 100)

    r_sp, omega_lr = find_light_ring(M_TARGET)
    if r_sp is None:
        print("ERROR: no light ring found.")
        return
    print(f"light ring: r_sp={r_sp * 1e3:.6f} mm  "
          f"f_lr={omega_lr / (2 * np.pi):.10f} Hz")
    print()

    im_seeds = np.arange(IM_SCAN_MIN, IM_SCAN_MAX + 1e-9, IM_SCAN_STEP)
    print(f"scanning {len(im_seeds)} Im-seeds in [{IM_SCAN_MIN}, {IM_SCAN_MAX}] Hz, "
          f"step={IM_SCAN_STEP} Hz, Re seed={RE_SEED} Hz")
    print(f"acceptance threshold (same as v6_FIXED): |Res| < {MAX_VALIDATED_ABS_RES:.1e}")
    print()

    validated = []
    for im0 in im_seeds:
        nm = complex_minimize_from_seed(RE_SEED, float(im0), M_TARGET, r_sp, omega_lr)
        if nm is None:
            print(f"  im0={im0:+.3f}  NM failed")
            continue

        polished = polish_root(nm["f_re"], nm["f_im"], M_TARGET, r_sp, omega_lr,
                                RE_SEED, nm["abs_res"])
        if polished is None:
            print(f"  im0={im0:+.3f}  NM: f_re={nm['f_re']:.6f} f_im={nm['f_im']:.6f} "
                  f"|Res|={nm['abs_res']:.4e}  ->  hybr: REJECTED")
            continue

        print(f"  im0={im0:+.3f}  NM: f_re={nm['f_re']:.6f} f_im={nm['f_im']:.6f} "
              f"|Res|={nm['abs_res']:.4e}  ->  hybr: f_re={polished['f_re']:.10f} "
              f"f_im={polished['f_im']:.10f} |Res|={polished['abs_res']:.4e}  VALIDATED")
        validated.append(polished)

    print()
    print("=" * 100)
    if validated:
        best = min(validated, key=lambda r: r["abs_res"])
        print(f"VALIDATED POLE(S) FOUND: {len(validated)}")
        print(f"  best: f_re={best['f_re']:.10f} Hz  f_im={best['f_im']:.10f} Hz  "
              f"|Res|={best['abs_res']:.4e}")
        print()
        print("VERDICT: a genuine root of the SAME pipeline used everywhere else in")
        print("v6_FIXED exists here -- the earlier 'gap at m=-15' was a discovery-grid")
        print("Im-seed spacing artifact, NOT evidence for branch termination at m_c.")
    else:
        print("NO VALIDATED POLE across all seeds -- every NM->hybr attempt was REJECTED")
        print(f"(threshold |Res| < {MAX_VALIDATED_ABS_RES:.1e}).")
        print("This is now a much stronger negative result: the SAME two-stage pipeline")
        print("that validates poles at every other m fails uniformly across a dense Im")
        print("grid at m=-15. Worth taking seriously as a candidate branch-termination")
        print("signature, though continuation-from-m=-14 with Jacobian conditioning")
        print("remains the strongest test before calling it that.")


if __name__ == "__main__":
    main()
