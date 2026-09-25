
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 6.5B-0.8
==============

COMPLEX CONTINUATION PREFLIGHT

Purpose
-------
Test the actual complex radial continuation chain used by the
validated Stage 5I-3c machinery.

This replaces the previous artificial test:

    solve_p_complex(omega, m, arbitrary_complex_r, p_initial=0)

which can legitimately fail simply because the Newton seed is poor.

Here we follow the physical/numerical chain:

    complex omega
        |
        v
    real last-scattering point
        |
        v
    complex turning point
        |
        v
    solve_p_complex near turning point
        |
        v
    radial_action_complex_v2
        |
        v
    reflection coefficient
        |
        v
    resonance residual

IMPORTANT
---------
This is still a PREFLIGHT.

It does NOT perform a resonance root search.

It does NOT vary C.

It does NOT fit Kerr.

It does NOT calibrate against GW data.

It does NOT modify the authoritative solver on disk.

The purpose is simply to establish that the complete complex
resonance evaluation path is callable at the known baseline
benchmark frequency.
"""

from __future__ import annotations

import sys
import inspect
import importlib.util
from pathlib import Path

import numpy as np


# ============================================================================
# PATHS
# ============================================================================

HERE = Path(__file__).resolve().parent

RES_FILE = HERE / "stage5I_3_resonance.py"
PATCH_FILE = HERE / "stage5I_3c_radial_action_patch.py"


# ============================================================================
# BASELINE PARAMETERS
# ============================================================================

M = -12

C0 = 7.690000000000e-4
OMEGA = 1.000000000000e-1
GAMMA = 2.220000000000e-6
H0 = 3.400000000000e-2
G = 9.810000000000e0
R_B = 3.730000000000e-2


# ============================================================================
# BENCHMARK COMPLEX FREQUENCY
# ============================================================================

F_RE = 8.3066307169
F_IM = -0.0005664350

OMEGA_TEST = (
    2.0
    * np.pi
    * complex(F_RE, F_IM)
)


# ============================================================================
# LOCAL COMPATIBILITY VALUE
# ============================================================================

DEFAULT_COMPLEX_RADIAL_N = 20000


# ============================================================================
# LOADER
# ============================================================================

def load_module(path: Path, name: str):

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not create import specification for:\n{path}"
        )

    module = importlib.util.module_from_spec(spec)

    module.__file__ = str(path)
    module.__name__ = name

    sys.modules[name] = module

    spec.loader.exec_module(module)

    return module


# ============================================================================
# FINITE COMPLEX CHECK
# ============================================================================

def finite_complex(z):

    if z is None:
        return False

    try:
        zr = np.real(z)
        zi = np.imag(z)

        return (
            np.isfinite(zr)
            and np.isfinite(zi)
        )

    except Exception:
        return False


# ============================================================================
# HEADER
# ============================================================================

print("=" * 78)
print("STAGE 6.5B-0.8 — COMPLEX CONTINUATION PREFLIGHT")
print("=" * 78)
print()

print("Authoritative solver:")
print(f"  {RES_FILE}")
print()

print("Validated radial-action patch:")
print(f"  {PATCH_FILE}")
print()

print("Mode:")
print(f"  m = {M}")
print()

print("Baseline C:")
print(f"  C = {C0:.12e} m^2/s")
print()

print("This test follows the actual complex continuation chain.")
print()
print("No resonance root search.")
print("No C variation.")
print("No Kerr fit.")
print("No GW calibration.")
print("No frequency rescaling.")
print("No low-k approximation.")
print("No authoritative-file modification.")
print()


# ============================================================================
# LOAD AUTHORITATIVE SOLVER
# ============================================================================

print("-" * 78)
print("LOADING AUTHORITATIVE SOLVER")
print("-" * 78)

solver = load_module(
    RES_FILE,
    "stage5I_3_resonance",
)

print("Loaded successfully.")
print()


# ============================================================================
# API CHECK
# ============================================================================

print("-" * 78)
print("AUTHORITATIVE COMPLEX API")
print("-" * 78)

required = [
    "hamiltonian",
    "solve_p_complex",
    "find_light_ring",
    "find_last_scattering_point",
    "find_turning_point_complex",
    "reflection_coefficient",
]

for name in required:

    if not hasattr(solver, name):

        raise RuntimeError(
            f"Required function missing: {name}"
        )

    print(
        f"  {name:35s}: FOUND"
    )

print()

print("Signatures:")

print(
    "  hamiltonian:",
    inspect.signature(
        solver.hamiltonian
    ),
)

print(
    "  solve_p_complex:",
    inspect.signature(
        solver.solve_p_complex
    ),
)

print(
    "  find_last_scattering_point:",
    inspect.signature(
        solver.find_last_scattering_point
    ),
)

print(
    "  find_turning_point_complex:",
    inspect.signature(
        solver.find_turning_point_complex
    ),
)

print()


# ============================================================================
# PATCH LOAD
# ============================================================================

print("-" * 78)
print("LOADING STAGE 5I-3c PATCH")
print("-" * 78)

print()

if not hasattr(
    solver,
    "DEFAULT_COMPLEX_RADIAL_N",
):

    solver.DEFAULT_COMPLEX_RADIAL_N = (
        DEFAULT_COMPLEX_RADIAL_N
    )

    print(
        "Compatibility shim:"
    )

    print(
        "  solver.DEFAULT_COMPLEX_RADIAL_N = "
        f"{DEFAULT_COMPLEX_RADIAL_N}"
    )

else:

    print(
        "Authoritative DEFAULT_COMPLEX_RADIAL_N found."
    )

print()

patch = load_module(
    PATCH_FILE,
    "stage5I_3c_radial_action_patch",
)

if not hasattr(
    patch,
    "radial_action_complex_v2",
):

    raise RuntimeError(
        "radial_action_complex_v2 missing."
    )

print(
    "Patch loaded successfully."
)

print(
    "  radial_action_complex_v2:",
    inspect.signature(
        patch.radial_action_complex_v2
    ),
)

print()


# ============================================================================
# BASELINE LIGHT RING
# ============================================================================

print("-" * 78)
print("1. BASELINE LIGHT RING")
print("-" * 78)

r_sp, omega_lr = solver.find_light_ring(
    M,
    C=C0,
    Omega=OMEGA,
    gamma=GAMMA,
    h0=H0,
    g=G,
    r_min=1.0e-3,
    r_max=R_B,
)

if r_sp is None:

    raise RuntimeError(
        "Baseline light ring not found."
    )

f_lr = omega_lr / (
    2.0 * np.pi
)

print(
    f"  r_sp     = {r_sp * 1e3:.12f} mm"
)

print(
    f"  omega_lr = {omega_lr:.12e} rad/s"
)

print(
    f"  f_lr     = {f_lr:.12f} Hz"
)

print()


# ============================================================================
# COMPLEX FREQUENCY
# ============================================================================

print("-" * 78)
print("2. COMPLEX BENCHMARK FREQUENCY")
print("-" * 78)

print(
    f"  f = {F_RE:.12f}"
    f" {F_IM:+.12f} i Hz"
)

print(
    f"  omega = {OMEGA_TEST.real:.12e}"
    f" {OMEGA_TEST.imag:+.12e} i rad/s"
)

print()


# ============================================================================
# LAST SCATTERING POINT
# ============================================================================

print("-" * 78)
print("3. LAST-SCATTERING POINT")
print("-" * 78)

try:

    r_minus_real = (
        solver.find_last_scattering_point(
            OMEGA_TEST,
            M,
            r_sp,
        )
    )

    print(
        f"  r_minus_real = {r_minus_real}"
    )

    if not finite_complex(
        r_minus_real
    ):

        raise RuntimeError(
            "Last-scattering point is invalid."
        )

    print(
        "  STATUS: PASS"
    )

except Exception as exc:

    print(
        "  STATUS: FAIL"
    )

    print(
        f"  {type(exc).__name__}: {exc}"
    )

    raise

print()


# ============================================================================
# COMPLEX TURNING POINT
# ============================================================================

print("-" * 78)
print("4. COMPLEX TURNING POINT")
print("-" * 78)

try:

    r_minus_complex = (
        solver.find_turning_point_complex(
            OMEGA_TEST,
            M,
            r_minus_real,
        )
    )

    print(
        f"  r_minus_complex = "
        f"{r_minus_complex}"
    )

    if not finite_complex(
        r_minus_complex
    ):

        raise RuntimeError(
            "Complex turning point is invalid."
        )

    print(
        "  STATUS: PASS"
    )

except Exception as exc:

    print(
        "  STATUS: FAIL"
    )

    print(
        f"  {type(exc).__name__}: {exc}"
    )

    raise

print()


# ============================================================================
# HAMILTONIAN AT TURNING POINT
# ============================================================================

print("-" * 78)
print("5. HAMILTONIAN AT COMPLEX TURNING POINT")
print("-" * 78)

try:

    # At a turning point p should be close to zero.
    H_turn = solver.hamiltonian(
        OMEGA_TEST,
        0.0j,
        M,
        r_minus_complex,
    )

    print(
        f"  H(r_turn, p=0) = {H_turn}"
    )

    if not finite_complex(H_turn):

        raise RuntimeError(
            "Hamiltonian at turning point is non-finite."
        )

    print(
        "  |H| = "
        f"{abs(H_turn):.12e}"
    )

    print(
        "  STATUS: PASS"
    )

except Exception as exc:

    print(
        "  STATUS: FAIL"
    )

    print(
        f"  {type(exc).__name__}: {exc}"
    )

    raise

print()


# ============================================================================
# LOCAL COMPLEX RADIAL MOMENTUM
# ============================================================================

print("-" * 78)
print("6. COMPLEX RADIAL MOMENTUM")
print("-" * 78)

print()
print(
    "Testing solve_p_complex in the immediate"
)
print(
    "neighborhood of the validated complex turning point."
)
print()

# Small complex/real offsets around turning point.
# We deliberately do NOT test exactly at the turning point,
# because p=0 is itself the singular/degenerate Newton location.

offsets = [
    1.0e-7,
    3.0e-7,
    1.0e-6,
    -1.0e-7,
    -3.0e-7,
    -1.0e-6,
]

p_results = []

for dr in offsets:

    r_probe = (
        r_minus_complex
        + complex(dr, 0.0)
    )

    print(
        f"  dr = {dr:+.3e} m"
    )

    try:

        # First attempt with default solver seed.
        p_val = solver.solve_p_complex(
            OMEGA_TEST,
            M,
            r_probe,
        )

        # If the default seed does not converge,
        # try a small analytic-scale complex seed.
        if p_val is None:

            # Estimate a small seed from the local scale.
            seed_mag = np.sqrt(
                max(abs(dr), 1.0e-16)
            )

            seeds = [
                complex(seed_mag, 0.0),
                complex(-seed_mag, 0.0),
                complex(0.0, seed_mag),
                complex(0.0, -seed_mag),
            ]

            for seed in seeds:

                p_try = solver.solve_p_complex(
                    OMEGA_TEST,
                    M,
                    r_probe,
                    p_initial=seed,
                )

                if p_try is not None:

                    p_val = p_try
                    break

        print(
            f"    p = {p_val}"
        )

        if finite_complex(p_val):

            p_results.append(
                (
                    dr,
                    p_val,
                )
            )

            print(
                "    STATUS: PASS"
            )

        else:

            print(
                "    STATUS: NO CONVERGED ROOT"
            )

    except Exception as exc:

        print(
            "    STATUS: EXCEPTION"
        )

        print(
            f"    {type(exc).__name__}: {exc}"
        )


print()

print(
    f"  converged points = "
    f"{len(p_results)}/{len(offsets)}"
)

if len(p_results) < 2:

    raise RuntimeError(
        "Insufficient complex radial momentum "
        "convergence near the turning point."
    )

print(
    "  RADIAL MOMENTUM STATUS: PASS"
)

print()


# ============================================================================
# RADIAL ACTION
# ============================================================================

print("-" * 78)
print("7. COMPLEX RADIAL ACTION")
print("-" * 78)

try:

    S = patch.radial_action_complex_v2(
        OMEGA_TEST,
        M,
        r_minus_complex,
        r_B=R_B,
        n=DEFAULT_COMPLEX_RADIAL_N,
    )

    print(
        f"  S = {S}"
    )

    if not finite_complex(S):

        raise RuntimeError(
            "Complex radial action is non-finite."
        )

    print(
        f"  |S| = {abs(S):.12e}"
    )

    print(
        "  STATUS: PASS"
    )

except Exception as exc:

    print(
        "  STATUS: FAIL"
    )

    print(
        f"  {type(exc).__name__}: {exc}"
    )

    raise

print()


# ============================================================================
# REFLECTION COEFFICIENT
# ============================================================================

print("-" * 78)
print("8. REFLECTION COEFFICIENT")
print("-" * 78)

try:

    R = solver.reflection_coefficient(
        OMEGA_TEST,
        M,
        r_sp,
        omega_lr,
    )

    print(
        f"  R = {R}"
    )

    if not finite_complex(R):

        raise RuntimeError(
            "Reflection coefficient is non-finite."
        )

    print(
        f"  |R| = {abs(R):.12e}"
    )

    print(
        "  STATUS: PASS"
    )

except Exception as exc:

    print(
        "  STATUS: FAIL"
    )

    print(
        f"  {type(exc).__name__}: {exc}"
    )

    raise

print()


# ============================================================================
# RESONANCE RESIDUAL — EVALUATION ONLY
# ============================================================================

print("-" * 78)
print("9. RESONANCE RESIDUAL — EVALUATION ONLY")
print("-" * 78)

try:

    Res = (
        R
        * np.exp(
            2.0j * S
        )
        - 1.0
    )

    print(
        f"  Res = {Res}"
    )

    print(
        f"  |Res| = {abs(Res):.12e}"
    )

    if not finite_complex(Res):

        raise RuntimeError(
            "Res is non-finite."
        )

    print(
        "  STATUS: PASS"
    )

except Exception as exc:

    print(
        "  STATUS: FAIL"
    )

    print(
        f"  {type(exc).__name__}: {exc}"
    )

    raise

print()


# ============================================================================
# FINAL
# ============================================================================

print("=" * 78)
print("STAGE 6.5B-0.8 — FINAL")
print("=" * 78)
print()

print(
    "PREFLIGHT: PASS"
)

print()

print(
    "Validated chain:"
)

print(
    "  [PASS] authoritative solver"
)

print(
    "  [PASS] baseline light ring"
)

print(
    "  [PASS] complex benchmark frequency"
)

print(
    "  [PASS] last-scattering point"
)

print(
    "  [PASS] complex turning point"
)

print(
    "  [PASS] local complex radial momentum"
)

print(
    "  [PASS] Stage 5I-3c radial action"
)

print(
    "  [PASS] reflection coefficient"
)

print(
    "  [PASS] resonance residual evaluation"
)

print()

print(
    "IMPORTANT:"
)

print(
    "This does NOT establish a resonance root."
)

print(
    "It establishes that the complete complex"
)

print(
    "resonance evaluation chain is operational."
)

print()

print(
    "NEXT:"
)

print(
    "  Stage 6.5B-1 — C-causality resonance test"
)

print()

