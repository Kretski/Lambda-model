import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stage5I_4_full_m_scan_v6_FIXED import (
    find_light_ring,
    refine_and_polish_candidate,
)

print("=" * 100)
print("STAGE 5I-17 -- CONTINUOUS-m BRANCH TERMINATION SEARCH")
print("=" * 100)
print()
print("Goal:")
print("  Treat m temporarily as a continuous parameter between -14 and -15.")
print("  Track the validated deep branch from m=-14 toward m=-15.")
print("  Look for a critical termination point rather than assuming")
print("  that failure at integer m=-15 is numerical.")
print()

# Validated m=-14 root
re0 = 9.399550955175973
im0 = -0.3086180376787981

# Dense continuation points.
# The existing solver accepts numerical m, so we test fractional values.
m_values = np.arange(-14.0, -15.0001, -0.05)

results = []

prev_re = re0
prev_im = im0

print("=" * 100)
print("CONTINUATION")
print("=" * 100)

for m in m_values:
    print()
    print(f"m={m:8.3f}   seed Re={prev_re:.10f}  Im={prev_im:.10f}")

    try:
        r_sp, omega_lr = find_light_ring(m)
    except Exception as exc:
        print(f"  find_light_ring ERROR: {exc}")
        results.append((m, None, None, None))
        print("  STOP")
        break

    if r_sp is None:
        print("  find_light_ring FAILED")
        results.append((m, None, None, None))
        print("  STOP")
        break

    # Try several nearby imaginary seeds while keeping Re from continuation.
    seeds = [
        (prev_re, prev_im),
        (prev_re, prev_im - 0.01),
        (prev_re, prev_im + 0.01),
        (prev_re, prev_im - 0.03),
        (prev_re, prev_im + 0.03),
    ]

    found = None

    for seed_re, seed_im in seeds:
        try:
            polished = refine_and_polish_candidate(
                seed_re,
                seed_im,
                m,
                r_sp,
                omega_lr,
                real_axis_abs_res=float("nan"),
                is_complex_seed=True,
            )
        except Exception as exc:
            print(f"    seed ({seed_re:.6f},{seed_im:.6f}) ERROR: {exc}")
            continue

        if polished is None:
            print(
                f"    seed ({seed_re:.6f},{seed_im:.6f}) -> rejected"
            )
            continue

        re = polished["f_re"]
        im = polished["f_im"]
        res = polished["abs_res"]

        print(
            f"    FOUND: Re={re:.10f}  Im={im:.10f}  |Res|={res:.3e}"
        )

        found = (re, im, res)
        break

    if found is None:
        print("  NO VALIDATED ROOT")
        results.append((m, None, None, None))
        print()
        print("  >>> BRANCH TERMINATION CANDIDATE <<<")
        break

    re, im, res = found
    results.append((m, re, im, res))

    prev_re = re
    prev_im = im

print()
print("=" * 100)
print("RESULT TABLE")
print("=" * 100)

for m, re, im, res in results:
    if re is None:
        print(f"m={m:8.3f}   NO ROOT")
    else:
        print(
            f"m={m:8.3f}   Re={re:.10f}   Im={im:.10f}   |Res|={res:.3e}"
        )

# Estimate local behavior near termination.
valid = [(m, re, im, res) for m, re, im, res in results if re is not None]

print()
print("=" * 100)
print("TERMINATION DIAGNOSTIC")
print("=" * 100)

if len(valid) >= 3:
    print(f"  Valid continuation points: {len(valid)}")

    ms = np.array([x[0] for x in valid])
    res_re = np.array([x[1] for x in valid])
    res_im = np.array([x[2] for x in valid])

    d_re = np.diff(res_re) / np.diff(ms)
    d_im = np.diff(res_im) / np.diff(ms)

    print()
    print("  Last valid points:")
    for x in valid[-5:]:
        print(
            f"    m={x[0]:.3f}  Re={x[1]:.10f}  Im={x[2]:.10f}"
        )

    print()
    print("  Local dRe/dm:")
    for i, value in enumerate(d_re[-5:]):
        idx = len(d_re) - len(d_re[-5:]) + i
        print(f"    interval ending at m={ms[idx+1]:.3f}: {value:+.8f}")

    print()
    print("  Local dIm/dm:")
    for i, value in enumerate(d_im[-5:]):
        idx = len(d_im) - len(d_im[-5:]) + i
        print(f"    interval ending at m={ms[idx+1]:.3f}: {value:+.8f}")

    print()
    print("  If the branch approaches a finite critical m and then")
    print("  disappears, this is consistent with a genuine finite-m")
    print("  termination. A smooth continuation all the way to -15")
    print("  would instead argue against termination.")

else:
    print("  Too few valid continuation points for a termination diagnosis.")

print()
print("=" * 100)
print("FINAL VERDICT")
print("=" * 100)

if valid:
    last_m = valid[-1][0]

    if last_m > -14.95:
        print(
            f"  Branch followed from m=-14 to m={last_m:.3f}, "
            "then no validated root was found."
        )
        print()
        print("  This localizes a TERMINATION REGION between the last")
        print("  valid point and the first failed point.")
        print()
        print("  IMPORTANT: this is evidence for finite-m termination,")
        print("  not yet proof of a physical branch endpoint."
        )
    else:
        print(
            "  Branch survives very close to m=-15."
        )
        print(
            "  A finer scan is required before claiming termination."
        )
else:
    print("  No continuation established.")

print()
print("NO q LABEL ASSIGNED.")
print("NO CLAIM OF NEW PHYSICS.")
print("=" * 100)
