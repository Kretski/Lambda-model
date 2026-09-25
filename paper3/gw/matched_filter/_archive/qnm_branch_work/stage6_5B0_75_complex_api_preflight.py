#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STAGE 6.5B-0.75
================

COMPLEX RESONANCE API PREFLIGHT

Purpose
-------
Verify that the authoritative complex-resonance machinery and the
validated Stage 5I-3c radial-action patch can be connected correctly.

This is ONLY an API/connectivity preflight.

NO resonance root search is performed.

Rules
-----
* authoritative solver is loaded normally
* no AST rewriting
* no modification of authoritative source
* no Kerr fit
* no GW calibration
* no frequency rescaling
* no low-k approximation
* baseline parameters only
* no physical conclusion is drawn here

Important API fact
------------------
The authoritative complex API uses:

    hamiltonian(omega, p, m, r)

and

    solve_p_complex(omega, m, r, p_initial=0j)

Therefore C, Omega, gamma, h0 and g are NOT passed as keyword
arguments to these complex functions.

The real light-ring API DOES expose C explicitly:

    find_light_ring(m, C=..., Omega=..., ...)

The validated Stage 5I-3c patch imports DEFAULT_COMPLEX_RADIAL_N,
but the authoritative solver does not export that symbol.

We therefore provide DEFAULT_COMPLEX_RADIAL_N only through a
temporary in-memory compatibility shim.

The authoritative file is NOT modified on disk.
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
# BENCHMARK RESONANCE
# ============================================================================

F_BENCH = 8.3066307169
FI_BENCH = -0.0005664350


# ============================================================================
# LOCAL COMPATIBILITY CONSTANT
# ============================================================================

# This constant is required only because the Stage 5I-3c patch imports it.
#
# It is NOT written into stage5I_3_resonance.py.
#
# The patch itself exposes:
#
#     radial_action_complex_v2(..., n=DEFAULT_COMPLEX_RADIAL_N)
#
# and the actual function signature will be inspected below.

DEFAULT_COMPLEX_RADIAL_N = 20000


# ============================================================================
# MODULE LOADER
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

    # Important for modules using __file__.
    module.__file__ = str(path)
    module.__name__ = name

    # Make the module available to normal imports.
    sys.modules[name] = module

    spec.loader.exec_module(module)

    return module


# ============================================================================
# HEADER
# ============================================================================

print("=" * 78)
print("STAGE 6.5B-0.75 — COMPLEX RESONANCE API PREFLIGHT")
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

print("Only API connectivity is tested.")
print("No resonance root search.")
print("No Kerr fit.")
print("No GW calibration.")
print("No frequency rescaling.")
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
# AUTHORITATIVE API CHECK
# ============================================================================

print("-" * 78)
print("AUTHORITATIVE API")
print("-" * 78)

required_solver = [
    "hamiltonian",
    "solve_p_complex",
    "find_light_ring",
    "find_last_scattering_point",
    "find_turning_point_complex",
    "reflection_coefficient",
]

for name in required_solver:

    exists = hasattr(solver, name)

    print(
        f"  {name:35s} : "
        f"{'FOUND' if exists else 'MISSING'}"
    )

    if not exists:
        raise RuntimeError(
            f"Required authoritative function missing: {name}"
        )

print()

print("Signatures:")
print(
    "  hamiltonian:",
    inspect.signature(solver.hamiltonian),
)
print(
    "  solve_p_complex:",
    inspect.signature(solver.solve_p_complex),
)
print(
    "  find_light_ring:",
    inspect.signature(solver.find_light_ring),
)
print()


# ============================================================================
# EXPLICIT API EXPECTATION
# ============================================================================

print("-" * 78)
print("COMPLEX API EXPECTATION")
print("-" * 78)

ham_sig = inspect.signature(solver.hamiltonian)
spc_sig = inspect.signature(solver.solve_p_complex)

expected_hamiltonian = ["omega", "p", "m", "r"]
expected_solve_p = ["omega", "m", "r", "p_initial"]

actual_hamiltonian = list(ham_sig.parameters.keys())
actual_solve_p = list(spc_sig.parameters.keys())

print(
    f"  hamiltonian parameters = {actual_hamiltonian}"
)

print(
    f"  solve_p_complex parameters = {actual_solve_p}"
)

print()

if actual_hamiltonian != expected_hamiltonian:

    raise RuntimeError(
        "Unexpected hamiltonian API.\n"
        f"Expected: {expected_hamiltonian}\n"
        f"Found:    {actual_hamiltonian}"
    )

if actual_solve_p[:3] != expected_solve_p[:3]:

    raise RuntimeError(
        "Unexpected solve_p_complex API.\n"
        f"Expected first parameters: {expected_solve_p[:3]}\n"
        f"Found: {actual_solve_p[:3]}"
    )

print("  API structure: PASS")
print()


# ============================================================================
# PATCH INSPECTION
# ============================================================================

print("-" * 78)
print("STAGE 5I-3c PATCH INSPECTION")
print("-" * 78)

print()

print(
    "Local compatibility value:"
)

print(
    f"  DEFAULT_COMPLEX_RADIAL_N = "
    f"{DEFAULT_COMPLEX_RADIAL_N}"
)

print()

# --------------------------------------------------------------------------
# Compatibility shim
# --------------------------------------------------------------------------
#
# The patch contains:
#
# from stage5I_3_resonance import (
#     hamiltonian,
#     solve_p_complex,
#     DEFAULT_COMPLEX_RADIAL_N,
#     R_B,
# )
#
# The authoritative solver does not export DEFAULT_COMPLEX_RADIAL_N.
#
# Therefore expose this symbol ONLY in memory so that the patch can import.
#
# This does NOT modify the authoritative source file.
#

if not hasattr(
    solver,
    "DEFAULT_COMPLEX_RADIAL_N",
):

    solver.DEFAULT_COMPLEX_RADIAL_N = (
        DEFAULT_COMPLEX_RADIAL_N
    )

    print(
        "Compatibility shim installed:"
    )

    print(
        "  solver.DEFAULT_COMPLEX_RADIAL_N "
        "= local value"
    )

else:

    print(
        "Authoritative solver already provides "
        "DEFAULT_COMPLEX_RADIAL_N."
    )

print()


# ============================================================================
# LOAD PATCH
# ============================================================================

patch = load_module(
    PATCH_FILE,
    "stage5I_3c_radial_action_patch",
)

print(
    "Patch loaded successfully."
)

print()

if not hasattr(
    patch,
    "radial_action_complex_v2",
):

    raise RuntimeError(
        "radial_action_complex_v2 not found in patch."
    )

patch_sig = inspect.signature(
    patch.radial_action_complex_v2
)

print(
    "radial_action_complex_v2:",
    patch_sig,
)

print()


# ============================================================================
# VERIFY PATCH FUNCTION PARAMETERS
# ============================================================================

patch_params = list(
    patch_sig.parameters.keys()
)

expected_patch = [
    "omega",
    "m",
    "r_minus",
    "r_B",
    "n",
]

print(
    "Patch parameters:"
)

print(
    f"  {patch_params}"
)

print()

for required in [
    "omega",
    "m",
    "r_minus",
]:

    if required not in patch_params:

        raise RuntimeError(
            f"Patch function missing required "
            f"parameter: {required}"
        )

print(
    "  Patch API structure: PASS"
)

print()


# ============================================================================
# BASELINE LIGHT RING
# ============================================================================

print("-" * 78)
print("BASELINE LIGHT RING")
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

    print(
        "LIGHT RING: FAIL"
    )

    raise RuntimeError(
        "Baseline light ring unexpectedly missing."
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
# BENCHMARK COMPLEX FREQUENCY
# ============================================================================

omega_test = (
    2.0
    * np.pi
    * complex(
        F_BENCH,
        FI_BENCH,
    )
)

print("-" * 78)
print("COMPLEX FREQUENCY")
print("-" * 78)

print(
    f"  f = {F_BENCH:.12f}"
    f" {FI_BENCH:+.12f} i Hz"
)

print(
    f"  omega = {omega_test.real:.12e}"
    f" {omega_test.imag:+.12e} i rad/s"
)

print()


# ============================================================================
# COMPLEX HAMILTONIAN TEST
# ============================================================================

print("-" * 78)
print("COMPLEX HAMILTONIAN TEST")
print("-" * 78)

# IMPORTANT:
#
# The authoritative signature is:
#
#     hamiltonian(omega, p, m, r)
#
# Therefore C/Omega/gamma/h0/g are NOT supplied here.
#
# The complex solver uses the constants already defined in the
# authoritative module.

r_test = complex(
    r_sp,
    -1.0e-5,
)

p_seed = 0.0j

try:

    H_test = solver.hamiltonian(
        omega_test,
        p_seed,
        M,
        r_test,
    )

    print(
        f"  r_test = {r_test}"
    )

    print(
        f"  p_test = {p_seed}"
    )

    print(
        f"  H = {H_test}"
    )

    if not (
        np.isfinite(
            np.real(H_test)
        )
        and np.isfinite(
            np.imag(H_test)
        )
    ):

        raise RuntimeError(
            "Hamiltonian returned non-finite value."
        )

    print(
        "  STATUS: PASS"
    )

except Exception as exc:

    print(
        "  STATUS: FAIL"
    )

    print(
        f"  Exception: "
        f"{type(exc).__name__}: {exc}"
    )

    raise

print()


# ============================================================================
# COMPLEX RADIAL MOMENTUM TEST
# ============================================================================

print("-" * 78)
print("COMPLEX RADIAL MOMENTUM TEST")
print("-" * 78)

try:

    p_test = solver.solve_p_complex(
        omega_test,
        M,
        r_test,
    )

    print(
        f"  p_complex = {p_test}"
    )

    if p_test is None:

        raise RuntimeError(
            "solve_p_complex returned None."
        )

    if not (
        np.isfinite(
            np.real(p_test)
        )
        and np.isfinite(
            np.imag(p_test)
        )
    ):

        raise RuntimeError(
            "solve_p_complex returned non-finite value."
        )

    print(
        "  STATUS: PASS"
    )

except Exception as exc:

    print(
        "  STATUS: FAIL"
    )

    print(
        f"  Exception: "
        f"{type(exc).__name__}: {exc}"
    )

    raise

print()


# ============================================================================
# COMPLEX TURNING-POINT TEST
# ============================================================================

print("-" * 78)
print("COMPLEX TURNING-POINT TEST")
print("-" * 78)

try:

    r_turn = solver.find_turning_point_complex(
        omega_test,
        M,
        r_sp,
    )

    print(
        f"  r_turn = {r_turn}"
    )

    if r_turn is None:

        raise RuntimeError(
            "find_turning_point_complex returned None."
        )

    if not (
        np.isfinite(
            np.real(r_turn)
        )
        and np.isfinite(
            np.imag(r_turn)
        )
    ):

        raise RuntimeError(
            "Complex turning point is non-finite."
        )

    print(
        "  STATUS: PASS"
    )

except Exception as exc:

    print(
        "  STATUS: FAIL"
    )

    print(
        f"  Exception: "
        f"{type(exc).__name__}: {exc}"
    )

    raise

print()


# ============================================================================
# RADIAL ACTION TEST
# ============================================================================

print("-" * 78)
print("RADIAL ACTION TEST")
print("-" * 78)

# Use explicit n only if the patch supports it.
#
# Current discovered signature:
#
#     radial_action_complex_v2(
#         omega,
#         m,
#         r_minus,
#         r_B=0.0373,
#         n=20000
#     )

try:

    if "n" in patch_params:

        S = patch.radial_action_complex_v2(
            omega_test,
            M,
            r_turn,
            r_B=R_B,
            n=DEFAULT_COMPLEX_RADIAL_N,
        )

    else:

        S = patch.radial_action_complex_v2(
            omega_test,
            M,
            r_turn,
        )

    print(
        f"  S = {S}"
    )

    if not (
        np.isfinite(
            np.real(S)
        )
        and np.isfinite(
            np.imag(S)
        )
    ):

        raise RuntimeError(
            "Radial action returned non-finite value."
        )

    print(
        "  STATUS: PASS"
    )

except Exception as exc:

    print(
        "  STATUS: FAIL"
    )

    print(
        f"  Exception: "
        f"{type(exc).__name__}: {exc}"
    )

    raise

print()


# ============================================================================
# FINAL
# ============================================================================

print("=" * 78)
print("STAGE 6.5B-0.75 — FINAL")
print("=" * 78)
print()

print(
    "PREFLIGHT: PASS"
)

print()

print(
    "Verified:"
)

print(
    "  [PASS] authoritative solver loaded"
)

print(
    "  [PASS] authoritative complex API found"
)

print(
    "  [PASS] hamiltonian API verified"
)

print(
    "  [PASS] solve_p_complex API verified"
)

print(
    "  [PASS] validated Stage 5I-3c patch loaded"
)

print(
    "  [PASS] baseline light ring reproduced"
)

print(
    "  [PASS] complex Hamiltonian callable"
)

print(
    "  [PASS] complex radial momentum callable"
)

print(
    "  [PASS] complex turning point callable"
)

print(
    "  [PASS] radial action callable"
)

print()

print(
    "The missing DEFAULT_COMPLEX_RADIAL_N was supplied"
)

print(
    "only through an in-memory compatibility shim."
)

print(
    "The authoritative solver source was NOT modified."
)

print()

print(
    "NEXT:"
)

print(
    "  Stage 6.5B-1 — resonance-level C-causality test"
)

print()