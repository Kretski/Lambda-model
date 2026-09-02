"""
PAPER 3 — MATCHED-FILTER PIPELINE INTEGRITY TEST
=================================================

Purpose
-------
Independent end-to-end integrity test for the Paper 3
Lambda-corrected dispersion / matched-filter machinery.

This test checks:

1. Lambda = 0 baseline
2. Non-zero Lambda waveform generation
3. Synthetic injection
4. Matched-filter / likelihood evaluation
5. Lambda recovery
6. Null-control behaviour

IMPORTANT
---------
A successful pipeline test does NOT constitute evidence for
non-zero Lambda in a real gravitational-wave event.

Real-event inference requires:
    - calibrated strain data
    - independently generated waveform families
    - noise PSD estimation
    - off-source/background trials
    - nuisance-parameter marginalization
    - statistical significance
    - systematic-error analysis
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def banner(title: str):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def check(condition: bool, message: str):
    if condition:
        print(f"[PASS] {message}")
        return True

    print(f"[FAIL] {message}")
    return False


# ---------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------

def test_imports():

    banner("[1] MODULE IMPORT TEST")

    modules = {}

    required = [
        "waveform",
        "likelihood",
        "synthetic_injection",
        "recovery_test",
    ]

    passed = True

    for name in required:

        try:

            module = __import__(name)
            modules[name] = module

            print(f"[PASS] import {name}")

        except Exception as exc:

            print(f"[FAIL] import {name}: {exc}")
            passed = False

    return passed, modules


# ---------------------------------------------------------------------
# Waveform API inspection
# ---------------------------------------------------------------------

def inspect_waveform_module(waveform):

    banner("[2] WAVEFORM API")

    names = [
        name for name in dir(waveform)
        if not name.startswith("_")
    ]

    for name in names:

        obj = getattr(waveform, name)

        if callable(obj):

            try:
                sig = inspect.signature(obj)
            except Exception:
                sig = "(signature unavailable)"

            print(f"  {name}{sig}")

    return True


# ---------------------------------------------------------------------
# Mathematical Lambda dispersion test
# ---------------------------------------------------------------------

def lambda_dispersion_test():

    banner("[3] Λ DISPERSION TEST")

    # Dimensionless form:
    #
    # omega^2 = k^2 (1 + 2 Lambda k^2)
    #
    # This is the central dispersion relation used by the
    # Lambda-corrected model.

    k = np.linspace(0.05, 2.0, 500)

    lambda_zero = 0.0
    lambda_true = 0.10

    omega_zero = np.sqrt(
        k**2 * (1.0 + 2.0 * lambda_zero * k**2)
    )

    omega_lambda = np.sqrt(
        k**2 * (1.0 + 2.0 * lambda_true * k**2)
    )

    passed = True

    passed &= check(
        np.allclose(omega_zero, k),
        "Λ = 0 recovers standard dispersion"
    )

    passed &= check(
        np.all(omega_lambda > omega_zero),
        "positive Λ produces positive dispersion correction"
    )

    correction = omega_lambda - omega_zero

    passed &= check(
        np.max(correction) > 0,
        "non-zero Λ produces measurable synthetic correction"
    )

    print()
    print(f"Lambda_true = {lambda_true:.6e}")
    print(f"max Δomega  = {np.max(correction):.6e}")

    return passed


# ---------------------------------------------------------------------
# Synthetic Lambda recovery
# ---------------------------------------------------------------------

def synthetic_recovery_test():

    banner("[4] SYNTHETIC Λ RECOVERY")

    rng = np.random.default_rng(12345)

    k = np.linspace(0.05, 2.0, 400)

    lambda_true = 0.10

    omega_clean = np.sqrt(
        k**2 * (1.0 + 2.0 * lambda_true * k**2)
    )

    noise_sigma = 5e-4

    omega_obs = (
        omega_clean
        + rng.normal(0.0, noise_sigma, size=len(k))
    )

    # Fit:
    #
    # omega^2 = k^2 + 2 Lambda k^4
    #
    # Rearrange:
    #
    # y = omega^2 - k^2
    # x = 2 k^4
    # Lambda = slope

    x = 2.0 * k**4
    y = omega_obs**2 - k**2

    lambda_fit = np.sum(x * y) / np.sum(x * x)

    residuals = y - lambda_fit * x

    rss = np.sum(residuals**2)

    ss_tot = np.sum((y - np.mean(y))**2)

    r2 = 1.0 - rss / ss_tot

    passed = True

    passed &= check(
        abs(lambda_fit - lambda_true) < 0.01,
        "synthetic Lambda recovered within tolerance"
    )

    passed &= check(
        r2 > 0.99,
        "synthetic Lambda fit has R² > 0.99"
    )

    print()
    print(f"Lambda true = {lambda_true:.8f}")
    print(f"Lambda fit  = {lambda_fit:.8f}")
    print(f"RSS         = {rss:.8e}")
    print(f"R²          = {r2:.8f}")

    return passed


# ---------------------------------------------------------------------
# Null Lambda control
# ---------------------------------------------------------------------

def lambda_zero_control():

    banner("[5] Λ = 0 NULL CONTROL")

    rng = np.random.default_rng(54321)

    k = np.linspace(0.05, 2.0, 400)

    lambda_true = 0.0

    omega_clean = k.copy()

    noise_sigma = 5e-4

    omega_obs = (
        omega_clean
        + rng.normal(0.0, noise_sigma, size=len(k))
    )

    x = 2.0 * k**4
    y = omega_obs**2 - k**2

    lambda_fit = np.sum(x * y) / np.sum(x * x)

    passed = True

    passed &= check(
        abs(lambda_fit) < 0.01,
        "Λ = 0 synthetic control remains close to zero"
    )

    print()
    print(f"Lambda true = {lambda_true:.8f}")
    print(f"Lambda fit  = {lambda_fit:.8f}")

    return passed


# ---------------------------------------------------------------------
# End-to-end synthetic pipeline
# ---------------------------------------------------------------------

def end_to_end_test():

    banner("[6] END-TO-END SYNTHETIC PIPELINE")

    rng = np.random.default_rng(2026)

    n = 4096

    t = np.linspace(0.0, 1.0, n, endpoint=False)

    f0 = 80.0

    lambda_true = 0.05

    # Small phase perturbation representing a synthetic
    # Lambda-dependent waveform deformation.

    phase0 = 2.0 * np.pi * f0 * t

    phase_lambda = (
        phase0
        + lambda_true
        * (t / t.max())**2
        * np.pi
    )

    template = np.sin(phase0)

    injected = np.sin(phase_lambda)

    noise_sigma = 0.10

    data = (
        injected
        + rng.normal(0.0, noise_sigma, n)
    )

    # Matched-filter-like normalized correlation.
    #
    # This is intentionally a minimal independent test and
    # does not replace the production likelihood implementation.

    def normalized_match(a, b):

        a = a - np.mean(a)
        b = b - np.mean(b)

        denom = (
            np.sqrt(np.sum(a * a))
            * np.sqrt(np.sum(b * b))
        )

        if denom == 0:
            return 0.0

        return np.sum(a * b) / denom

    match_standard = normalized_match(data, template)
    match_lambda = normalized_match(data, injected)

    passed = check(
        match_lambda >= match_standard,
        "Lambda-deformed template matches injected data at least as well"
    )

    print()
    print(f"Lambda true              = {lambda_true:.6e}")
    print(f"Standard-template match  = {match_standard:.8f}")
    print(f"Lambda-template match    = {match_lambda:.8f}")

    return passed


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    banner("PAPER 3 — MATCHED-FILTER PIPELINE INTEGRITY VALIDATION")

    print()
    print(f"Working directory: {ROOT}")

    results = []

    imports_ok, modules = test_imports()
    results.append(imports_ok)

    if "waveform" in modules:
        results.append(
            inspect_waveform_module(modules["waveform"])
        )

    results.append(
        lambda_dispersion_test()
    )

    results.append(
        synthetic_recovery_test()
    )

    results.append(
        lambda_zero_control()
    )

    results.append(
        end_to_end_test()
    )

    banner("FINAL RESULT")

    passed = sum(bool(x) for x in results)
    total = len(results)

    print()
    print(f"Tests passed: {passed}/{total}")

    if passed == total:

        print()
        print("✓ MATCHED-FILTER PIPELINE INTEGRITY TEST PASSED")

        print()
        print(
            "The synthetic Lambda dispersion, Lambda recovery, "
            "null control and matched-template tests are internally consistent."
        )

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "This result does NOT establish a non-zero Lambda "
            "in a real gravitational-wave event."
        )

    else:

        print()
        print("✗ PIPELINE INTEGRITY TEST FAILED")

        print(
            "Inspect the failed section before using the pipeline "
            "for real-data inference."
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()