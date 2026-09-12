import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stage5I_4_full_m_scan_v6_FIXED import safe_complex_res

# ============================================================
# STAGE 5I-21 CONTROL FIXED
#
# Calibration of the FAST topology diagnostic at the known
# validated root m=-14.
#
# IMPORTANT:
# The previous control was invalid because its frequency window
# did NOT contain the known root.
#
# Known validated root:
#   m=-14
#   Re=9.3995509552
#   Im=-0.3086180377
#
# This test uses the EXACT safe_complex_res() wrapper.
# ============================================================

m = -14.0
root_re = 9.3995509552
root_im = -0.3086180377

# Obtain the exact light-ring quantities used by the pipeline.
from stage5I_4_full_m_scan_v6_FIXED import find_light_ring

r_sp, omega_lr = find_light_ring(m)

if r_sp is None:
    raise RuntimeError("find_light_ring failed for m=-14")

# Small local window centered on the known root.
re_values = np.linspace(root_re - 0.010, root_re + 0.010, 81)
im_values = np.linspace(root_im - 0.010, root_im + 0.010, 81)

best = None
zero_cells = 0

for re in re_values:
    for im in im_values:
        z = safe_complex_res(
            re, im, m, r_sp, omega_lr
        )
        val = abs(z)

        if best is None or val < best[0]:
            best = (val, re, im, z)

# Explicit evaluation exactly at the known root.
z_exact = safe_complex_res(
    root_re, root_im, m, r_sp, omega_lr
)

print("=" * 90)
print("STAGE 5I-21 -- FIXED CONTROL CALIBRATION")
print("=" * 90)
print()
print("m =", m)
print(f"Known root: Re={root_re:.10f}, Im={root_im:.10f}")
print(f"Grid: {len(re_values)} x {len(im_values)}")
print(
    f"Window: Re=[{re_values[0]:.6f},{re_values[-1]:.6f}] "
    f"Im=[{im_values[0]:.6f},{im_values[-1]:.6f}]"
)
print()
print("=" * 90)
print("GRID RESULT")
print("=" * 90)
print(
    f"minimum |Res| = {best[0]:.12e} "
    f"at Re={best[1]:.10f}, Im={best[2]:.10f}"
)
print(
    f"Re(R)={best[3].real:.6e}, "
    f"Im(R)={best[3].imag:.6e}"
)

print()
print("=" * 90)
print("EXACT KNOWN-ROOT EVALUATION")
print("=" * 90)
print(
    f"Re={root_re:.10f}, Im={root_im:.10f}"
)
print(
    f"Re(R)={z_exact.real:.12e}, "
    f"Im(R)={z_exact.imag:.12e}"
)
print(f"|Res|={abs(z_exact):.12e}")

print()
print("=" * 90)
print("CALIBRATION VERDICT")
print("=" * 90)

if abs(z_exact) < 1e-8:
    print(">>> EXACT ROOT CONFIRMED BY safe_complex_res() <<<")
    print()
    print("The residual wrapper correctly evaluates the known")
    print("m=-14 root as essentially zero.")
    print()
    print("Therefore the topology diagnostic can now be")
    print("calibrated against a genuinely known root.")
else:
    print(">>> WARNING: KNOWN ROOT IS NOT ZERO UNDER THIS WRAPPER <<<")
    print()
    print("Do NOT interpret the topology test yet.")
    print("The residual normalization/arguments need inspection.")

if best[0] < 1e-3:
    print()
    print("GRID ALSO RESOLVES A SUB-THRESHOLD MINIMUM.")
else:
    print()
    print("GRID DOES NOT RESOLVE |Res| < 1e-3.")
    print("This means the grid is too coarse even though the")
    print("exact point evaluation may still confirm the root.")

print()
print("NO q LABEL.")
print("NO CLAIM OF NEW PHYSICS.")
print("=" * 90)
