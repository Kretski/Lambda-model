"""
STAGE 5I-10 -- TARGETED RESCAN OF UNRESOLVED CANDIDATES
===========================================================
Reuses the EXACT pole-finding pipeline from
stage5I_4_full_m_scan_v6_FIXED.py (find_light_ring +
complex_minimize_from_seed + polish_root, wrapped as
refine_and_polish_candidate) -- no reimplementation, no risk of a
subtly different residual condition. This file must sit in the same
directory as stage5I_4_full_m_scan_v6_FIXED.py and
stage5I_3_resonance_v2.py.

TARGETS -- same rationale as before:
  1/2. Quartet extension (m=-11..-14 unidentified branch) predicted
       at m=-10 and m=-15 via linear fit.
  3/4. m=-16 q_csv=3 singleton (Im=-1.336) -- check neighbors m=-15,-17.
  5/6. m=-14 q_csv=4 singleton (Im=-1.482) -- check neighbors m=-13,-15.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from stage5I_4_full_m_scan_v6_FIXED import (
        find_light_ring,
        refine_and_polish_candidate,
    )
except ImportError as exc:
    raise ImportError(
        "This script must sit next to stage5I_4_full_m_scan_v6_FIXED.py "
        "and stage5I_3_resonance_v2.py (same directory)."
    ) from exc


TARGETS = [
    {"label": "quartet extension @ m=-10", "m": -10, "seed_re": 8.255, "seed_im": -0.353},
    {"label": "quartet extension @ m=-15", "m": -15, "seed_re": 9.680, "seed_im": -0.300},
    {"label": "m=-16 q=3 neighbor @ m=-15", "m": -15, "seed_re": 10.90, "seed_im": -1.30},
    {"label": "m=-16 q=3 neighbor @ m=-17", "m": -17, "seed_re": 11.25, "seed_im": -1.37},
    {"label": "m=-14 q=4 neighbor @ m=-13", "m": -13, "seed_re": 10.40, "seed_im": -1.45},
    {"label": "m=-14 q=4 neighbor @ m=-15", "m": -15, "seed_re": 10.90, "seed_im": -1.50},
]


def run_target(t):
    m = t["m"]
    print("=" * 100)
    print(f"{t['label']}  (m={m}, seed Re={t['seed_re']}, Im={t['seed_im']})")
    print("=" * 100)

    r_sp, omega_lr = find_light_ring(m)
    if r_sp is None:
        print("  find_light_ring FAILED for this m -- cannot search")
        return {**t, "status": "NO_LIGHT_RING"}

    polished = refine_and_polish_candidate(
        t["seed_re"], t["seed_im"], m, r_sp, omega_lr,
        real_axis_abs_res=float("nan"), is_complex_seed=True,
    )

    if polished is None:
        print("  RESULT: NOT FOUND (no seed variant validated)")
        return {**t, "status": "NOT FOUND"}

    print(f"  RESULT: FOUND  Re={polished['f_re']:.7f} Hz  "
          f"Im={polished['f_im']:.7f} Hz  |Res|={polished['abs_res']:.3e}")
    return {**t, "status": "FOUND", "re": polished["f_re"],
            "im": polished["f_im"], "res": polished["abs_res"]}


def main():
    results = [run_target(t) for t in TARGETS]

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for r in results:
        if r["status"] == "FOUND":
            print(f"  {r['label']:35s} -> FOUND      Re={r['re']:.5f}  Im={r['im']:.5f}  |Res|={r['res']:.2e}")
        else:
            print(f"  {r['label']:35s} -> {r['status']}")

    print()
    print("INTERPRETATION")
    print("  - If BOTH quartet-extension targets show FOUND near the predicted")
    print("    seeds: the m=-11..-14 branch is real and continuous; the coarse")
    print("    scan missed it at m=-10/-15 (a search-coverage gap, not physics")
    print("    confined to exactly 4 m-values).")
    print("  - If NOT FOUND at m=-10 and/or m=-15: try widening the seed grid")
    print("    (a few more Im shifts around the prediction) before concluding")
    print("    the branch genuinely terminates -- one seed miss is not proof")
    print("    of absence.")
    print("  - Singleton targets: a FOUND neighbor at similar Im depth is")
    print("    evidence of a real deep-decay family; still needs a 3rd point")
    print("    to test smoothness the way the quartet was tested.")


if __name__ == "__main__":
    main()
