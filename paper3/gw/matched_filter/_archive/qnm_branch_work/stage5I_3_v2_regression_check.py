"""
stage5I_3_v2_regression_check.py
====================================

REGRESSION CHECK: confirms that stage5I_3_resonance_v2.py's fixed
radial_action_complex() reproduces the known m=-12 benchmark result
through the FULL pipeline (dense real-axis scan -> Nelder-Mead complex
refinement -> hybr root polish), not just via isolated diagnostic
calls used during development.

BENCHMARK (established during Stage 5I-3b/3c diagnosis):
    f_re  ~ 8.3066307169 Hz
    f_im  ~ -0.0005664350 Hz
    |Res| ~ 2.78e-17  (machine precision, after hybr polish)

This must be run BEFORE the full m-scan (Stage 5I-4), per the agreed
plan: regression check on m=-12 before integrating v2 into the full
m-scan.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import root

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stage5I_3_resonance_v2 import (
    resonance_factor,
    find_light_ring,
    find_resonances,
    refine_complex_resonance,
    DEFAULT_N_COARSE,
    DEFAULT_TOP_N,
    DEFAULT_LOCAL_HALF_WIDTH,
)


BENCHMARK_F_RE = 8.3066307169
BENCHMARK_F_IM = -0.0005664350
BENCHMARK_ABS_RES_MAX = 1.0e-10


def polish_resonance_root(f_re0, f_im0, m, r_sp=None, omega_lr=None,
                            method="hybr", xtol=1e-13):
    """Genuine 2D root-solve for Res(omega)=0, seeded at (f_re0, f_im0)."""

    def F_vector(x):
        f_re, f_im = float(x[0]), float(x[1])
        omega = 2.0 * np.pi * (f_re + 1j * f_im)
        z = resonance_factor(omega=omega, m=m, r_sp=r_sp,
                              omega_lr=omega_lr, complex_action=True)
        if z is None or not (np.isfinite(z.real) and np.isfinite(z.imag)):
            return [1.0e6, 1.0e6]
        return [z.real, z.imag]

    z_seed = resonance_factor(
        omega=2.0 * np.pi * (f_re0 + 1j * f_im0), m=m, r_sp=r_sp,
        omega_lr=omega_lr, complex_action=True)
    abs_res_before = abs(z_seed) if z_seed is not None else np.nan

    sol = root(F_vector, x0=np.array([f_re0, f_im0]), method=method, tol=xtol)

    f_re_final, f_im_final = float(sol.x[0]), float(sol.x[1])
    z_final = resonance_factor(
        omega=2.0 * np.pi * (f_re_final + 1j * f_im_final), m=m,
        r_sp=r_sp, omega_lr=omega_lr, complex_action=True)
    abs_res_after = abs(z_final) if z_final is not None else np.nan

    return dict(f_re=f_re_final, f_im=f_im_final,
                abs_res_before=abs_res_before, abs_res_after=abs_res_after,
                success=bool(sol.success), message=str(sol.message),
                n_fev=int(sol.get("nfev", -1)))


def main():
    print("=" * 78)
    print("STAGE 5I-3 v2 - REGRESSION CHECK (m=-12)")
    print("=" * 78)
    print()
    print(f"  Benchmark target: f_re~{BENCHMARK_F_RE}, "
          f"f_im~{BENCHMARK_F_IM}, |Res| after polish < {BENCHMARK_ABS_RES_MAX:.0e}")
    print()

    m = -12
    f_min, f_max = 7.0, 10.5

    r_sp, omega_lr = find_light_ring(m)
    if r_sp is None:
        print("  FAIL: no light ring found.")
        sys.exit(1)

    f_lr = omega_lr / (2.0 * np.pi)
    print(f"  Light ring: r_sp={r_sp*1e3:.6f} mm, f_lr={f_lr:.10f} Hz")
    print()

    print(f"  Running FULL pipeline: dense real-axis scan "
          f"({DEFAULT_N_COARSE} points)...")
    resonances = find_resonances(
        m=m, f_min=f_min, f_max=f_max, resonance_factor=resonance_factor,
        n_coarse=DEFAULT_N_COARSE, top_n=DEFAULT_TOP_N,
        local_half_width=DEFAULT_LOCAL_HALF_WIDTH)

    if not resonances:
        print("  FAIL: no real-axis candidates found.")
        sys.exit(1)

    print()
    print("  Real-axis candidates:")
    for c in resonances:
        print(f"      f={c['f_real']:.10f} Hz  |Res|={c['abs_res']:.8e}")
    print()

    f0 = resonances[0]["f_real"]

    print("  -> Nelder-Mead complex refinement...")
    nm_result = refine_complex_resonance(
        f0=f0, im0=-0.01, resonance_factor=resonance_factor, m=m,
        re_half_width=DEFAULT_LOCAL_HALF_WIDTH, im_min=-1.5, im_max=0.0)

    if nm_result is None:
        print("  FAIL: Nelder-Mead refinement failed.")
        sys.exit(1)

    print(f"      f_re={nm_result['f_re']:.10f} Hz  "
          f"f_im={nm_result['f_im']:.10f} Hz  |Res|={nm_result['abs_res']:.8e}")
    print()

    print("  -> hybr root polish...")
    polished = polish_resonance_root(
        f_re0=nm_result["f_re"], f_im0=nm_result["f_im"], m=m,
        r_sp=r_sp, omega_lr=omega_lr)

    print(f"      f_re={polished['f_re']:.10f} Hz  "
          f"f_im={polished['f_im']:.10f} Hz")
    print(f"      |Res| before polish = {polished['abs_res_before']:.8e}")
    print(f"      |Res| after polish  = {polished['abs_res_after']:.8e}")
    print(f"      success={polished['success']}, "
          f"message={polished['message']}, n_fev={polished['n_fev']}")
    print()

    print("=" * 78)
    print("REGRESSION VERDICT")
    print("=" * 78)
    print()

    diff_re = abs(polished["f_re"] - BENCHMARK_F_RE)
    diff_im = abs(polished["f_im"] - BENCHMARK_F_IM)
    abs_res_ok = polished["abs_res_after"] < BENCHMARK_ABS_RES_MAX

    print(f"  |f_re - benchmark| = {diff_re:.2e} Hz  "
          f"{'PASS' if diff_re < 1e-6 else 'FAIL'} (threshold 1e-6)")
    print(f"  |f_im - benchmark| = {diff_im:.2e} Hz  "
          f"{'PASS' if diff_im < 1e-6 else 'FAIL'} (threshold 1e-6)")
    print(f"  |Res| after polish  = {polished['abs_res_after']:.2e}  "
          f"{'PASS' if abs_res_ok else 'FAIL'} (threshold {BENCHMARK_ABS_RES_MAX:.0e})")
    print()

    all_pass = diff_re < 1e-6 and diff_im < 1e-6 and abs_res_ok

    if all_pass:
        print("  REGRESSION CHECK PASSED.")
        print("  v2's fixed radial_action_complex() reproduces the")
        print("  established m=-12 benchmark through the FULL pipeline")
        print("  (not just isolated diagnostic calls). Safe to proceed")
        print("  to the full m-scan against the experimental CSV data.")
    else:
        print("  REGRESSION CHECK FAILED.")
        print("  Do NOT proceed to the full m-scan until this is")
        print("  resolved.")
        sys.exit(1)


if __name__ == "__main__":
    main()
