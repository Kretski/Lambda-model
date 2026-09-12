"""
stage5I_3c_radial_action_patch.py
=====================================

STAGE 5I-3c - FIX FOR radial_action_complex's FRAGILE FIRST STEP

DIAGNOSIS (from root-polishing investigation, Stage 5I-3b):

    scipy.optimize.root('hybr') got stuck with ZERO movement when
    attempting to polish the Nelder-Mead resonance estimate
    (f=8.3066307032 - 0.0005696389j Hz) down toward a true Res=0 root.

    Direct testing showed resonance_factor() returns a finite value
    ONLY within an extremely narrow (~1e-9 Hz) neighborhood of the
    Nelder-Mead point; perturbing f_im by just 1e-9 Hz already returns
    None.

    Tracing through radial_action_complex() and solve_p_complex(), the
    root cause: the FIRST continuation step (i=1 in the loop over
    r_path) calls solve_p_complex(omega, m, r_path[1],
    p_initial=p_previous) with p_previous = 0.0j (hardcoded at i=0,
    since p=0 exactly AT the turning point r_minus by definition).
    This seeds Newton's method exactly at a branch point of the
    Hamiltonian H(p,r): at a classical turning point, dH/dp = 0 by
    definition, so the Jacobian of [Re(H), Im(H)]=0 is singular/near-
    singular exactly at the seed point -- a textbook ill-conditioned
    starting point for Newton-type continuation.

    Small perturbations in omega shift r_minus (found by its own
    root-solve), which shifts exactly how close r_path[1] is to the
    true turning point -- for some perturbations the first Newton
    step from p=0 fails to converge, propagating a FloatingPointError
    up through radial_action_complex, caught by resonance_factor's
    except clause, returning None.

FIX: replace the blind p_initial=0 seed for the FIRST step with a
locally-accurate analytic estimate from the standard near-turning-
point Taylor expansion:

    H(p, r) ~= H_r * (r - r_minus) + (1/2) * H_pp * p^2

Setting H=0 and solving for p gives:

    p ~= sqrt( -2 * H_r * (r - r_minus) / H_pp )

used ONLY as a better INITIAL GUESS for the first Newton continuation
step (still refined by hybr afterward) -- standard practice for
continuation methods near branch points.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stage5I_3_resonance import (
    hamiltonian,
    solve_p_complex,
    DEFAULT_COMPLEX_RADIAL_N,
    R_B,
)


def compute_H_r_at_turning_point(omega, m, r_minus, eps=None):
    """dH/dr at (p=0, r=r_minus), finite differences, physically-scaled eps."""
    if eps is None:
        eps = abs(r_minus) * 1e-6 if abs(r_minus) > 0 else 1e-9

    H_plus = hamiltonian(omega, 0.0j, m, r_minus + eps)
    H_minus = hamiltonian(omega, 0.0j, m, r_minus - eps)
    return (H_plus - H_minus) / (2 * eps)


def radial_action_complex_v2(omega, m, r_minus, r_B=R_B,
                               n=DEFAULT_COMPLEX_RADIAL_N):
    """
    FIXED version of radial_action_complex(): the first continuation
    step (i=1) uses an analytically-informed initial guess for p,
    instead of blindly seeding Newton's method at p=0 (singular
    Jacobian point, since r_minus is a turning point). All subsequent
    steps continue to use p_initial=p_previous exactly as in the
    original.
    """
    t = np.linspace(0.0, 1.0, n)
    r_path = r_minus + t * (r_B - r_minus)

    p_values = np.zeros(n, dtype=complex)
    p_previous = 0.0j

    eps_p_probe = 1.0  # 1/m, modest probe step for local H_pp estimate
    H_pp_estimate = (
        hamiltonian(omega, eps_p_probe + 0j, m, r_minus)
        - 2 * hamiltonian(omega, 0j, m, r_minus)
        + hamiltonian(omega, -eps_p_probe + 0j, m, r_minus)
    ) / eps_p_probe ** 2

    H_r = compute_H_r_at_turning_point(omega, m, r_minus)

    for i, rr in enumerate(r_path):
        if i == 0:
            p_values[i] = 0.0j
            continue

        if i == 1:
            delta_r = rr - r_minus
            radicand = -2.0 * H_r * delta_r / H_pp_estimate
            # CORRECT branch: verified by direct comparison against
            # the validated real-axis solve_p() at omega slightly off
            # the real axis (see Stage 5I-3c diagnostic session) --
            # the POSITIVE sqrt root is the physically correct
            # continuation matching the propagating-wave convention
            # used throughout the rest of this codebase (solve_p in
            # stage5I_1_2_dispersion_radial_p.py returns p>=0 for the
            # propagating branch). The negative root was previously
            # tried and found to disagree with the validated real
            # solution by O(1), while the positive root agrees to
            # ~2e-5 (limited by the size of the finite first step).
            p_seed = np.sqrt(radicand)
        else:
            p_seed = p_previous

        p_new = solve_p_complex(omega, m, rr, p_initial=p_seed)

        if p_new is None and i == 1:
            # Fallback only: try the other branch if the primary
            # (positive-root) seed somehow fails to converge
            p_new = solve_p_complex(omega, m, rr, p_initial=-p_seed)

        if p_new is None:
            raise FloatingPointError(
                f"Complex p continuation failed at step {i}, r={rr}."
            )

        if abs(p_new - p_previous) > abs(-p_new - p_previous):
            p_new = -p_new

        p_values[i] = p_new
        p_previous = p_new

    if hasattr(np, "trapezoid"):
        return np.trapezoid(p_values, r_path)
    return np.trapz(p_values, r_path)


def main():
    """Quick standalone verification: does the fixed first step succeed
    where the original failed, at the exact perturbed points that
    triggered None in the diagnostic session?"""
    from stage5I_3_resonance import (
        find_light_ring, find_turning_point_complex,
        find_last_scattering_point, reflection_coefficient,
    )

    print("=" * 78)
    print("STAGE 5I-3c - VERIFY FIX AT PREVIOUSLY-FAILING PERTURBED POINTS")
    print("=" * 78)
    print()

    m = -12
    r_sp, omega_lr = find_light_ring(m)

    f_re0 = 8.3066307032346
    f_im0 = -0.0005696389388967092

    print(f"Base point: f_re={f_re0}, f_im={f_im0}")
    print()

    for delta_im, label in [(0.0, "exact base point"),
                              (1e-9, "+1e-9 Hz perturbation"),
                              (1e-7, "+1e-7 Hz perturbation"),
                              (1e-5, "+1e-5 Hz perturbation")]:
        f_im_test = f_im0 + delta_im
        omega_test = 2.0 * np.pi * (f_re0 + 1j * f_im_test)

        r_minus_real = find_last_scattering_point(omega_test, m, r_sp)
        r_minus_complex = find_turning_point_complex(omega_test, m, r_minus_real)

        if r_minus_complex is None:
            print(f"  {label}: turning point search FAILED")
            continue

        try:
            action = radial_action_complex_v2(omega_test, m, r_minus_complex)
            R = reflection_coefficient(omega_test, m, r_sp, omega_lr)
            Res = R * np.exp(2j * action) - 1.0
            print(f"  {label}: SUCCESS, |Res|={abs(Res):.6e}")
        except FloatingPointError as e:
            print(f"  {label}: STILL FAILS - {e}")
        except Exception as e:
            print(f"  {label}: unexpected error - {type(e).__name__}: {e}")

    print()
    print("If perturbed points now succeed where they previously returned")
    print("None, the analytic first-step seed has fixed the fragility, and")
    print("root-polishing (Stage 5I-3b) can be re-attempted using")
    print("radial_action_complex_v2 in place of the original.")


if __name__ == "__main__":
    main()
