"""
paper3_Dn4_test.py
=====================

Validates the discrete fourth-order (biharmonic) finite-difference operator
D_n^4, which approximates Laplacian(Laplacian(phi)) on a uniform grid.

This operator implements the quartic term in the real-space PDE

    d^2 phi/dt^2 = c^2 Laplacian(phi) - c^2 Lambda Laplacian^2(phi)

Two independent checks are performed:

  1. SYMBOL CHECK: the Fourier symbol of D_n^4 should approach k^4 as the
     grid spacing h -> 0. We verify this analytically and numerically.

  2. EIGENFUNCTION CHECK: applying D_n^4 to a plane wave exp(i k x) should
     return (k^4 + O(h^2)) * exp(i k x). We measure the O(h^2) error directly.

This is a prerequisite for trusting any finite-difference (as opposed to
spectral) implementation of the Lambda-model wave equation.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def laplacian_1d_matrix(N, h):
    """Standard second-order central-difference Laplacian, periodic BC."""
    D2 = np.zeros((N, N))
    for i in range(N):
        D2[i, i] = -2.0
        D2[i, (i + 1) % N] = 1.0
        D2[i, (i - 1) % N] = 1.0
    return D2 / h**2


def biharmonic_1d_matrix(N, h):
    """
    D_n^4 = D2 @ D2  (composed discrete Laplacian squared).
    This is the standard construction: apply the 3-point Laplacian twice,
    which gives a 5-point stencil [1, -4, 6, -4, 1] / h^4.
    """
    D2 = laplacian_1d_matrix(N, h)
    return D2 @ D2


def fourier_symbol_D2(k, h):
    """Exact Fourier symbol of the discrete Laplacian: -(2/h^2)(1-cos(kh))."""
    return -(2.0 / h**2) * (1 - np.cos(k * h))


def fourier_symbol_D4(k, h):
    """Fourier symbol of D_n^4 = (symbol of D2)^2."""
    return fourier_symbol_D2(k, h) ** 2


def continuum_symbol_D4(k):
    """Continuum limit: Laplacian^2 has Fourier symbol k^4 (in 1D: k^4)."""
    return k**4


def test_symbol_convergence():
    """
    TEST 1 — Symbol convergence.
    For fixed k, as h -> 0, the discrete symbol of D_n^4 should converge
    to k^4 with O(h^2) accuracy (standard 5-point stencil order).
    """
    print("=" * 68)
    print("TEST 1 — FOURIER SYMBOL CONVERGENCE OF D_n^4")
    print("=" * 68)
    print()
    print("  Discrete symbol: [-(2/h^2)(1-cos(kh))]^2")
    print("  Continuum limit: k^4")
    print()

    k_test = 1.0
    h_values = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

    print(f"  {'h':>8}  {'symbol(D_n^4)':>16}  {'k^4 (exact)':>14}  "
          f"{'rel. error':>12}  {'order (log2 ratio)':>18}")
    print("  " + "-" * 76)

    errors = []
    for h in h_values:
        sym_discrete = fourier_symbol_D4(k_test, h)
        sym_exact = continuum_symbol_D4(k_test)
        rel_err = abs(sym_discrete - sym_exact) / sym_exact
        errors.append(rel_err)

        order_str = ""
        if len(errors) > 1:
            order = np.log2(errors[-2] / errors[-1])
            order_str = f"{order:.3f}"

        print(f"  {h:>8.3f}  {sym_discrete:>16.8f}  {sym_exact:>14.8f}  "
              f"{rel_err:>12.2e}  {order_str:>18}")

    print()
    print("  Expected convergence order: 2 (standard 5-point stencil)")
    print(f"  Measured order (last step): {np.log2(errors[-2]/errors[-1]):.3f}")
    print()

    passed = abs(np.log2(errors[-2] / errors[-1]) - 2.0) < 0.3
    print(f"  TEST 1: {'PASS ✓' if passed else 'FAIL ✗'}")
    print()
    return passed


def test_eigenfunction_check():
    """
    TEST 2 — Direct eigenfunction check.
    Build the full D_n^4 matrix on a periodic grid, apply it to a
    discretized plane wave, and compare to the analytic eigenvalue.
    """
    print("=" * 68)
    print("TEST 2 — EIGENFUNCTION CHECK (matrix application)")
    print("=" * 68)
    print()

    N = 128
    L = 2 * np.pi
    h = L / N
    x = np.linspace(0, L, N, endpoint=False)

    # Test with a low mode number so k*h is small
    n_mode = 3
    k = 2 * np.pi * n_mode / L

    phi = np.exp(1j * k * x)

    D4 = biharmonic_1d_matrix(N, h)
    D4_phi = D4 @ phi

    # Expected eigenvalue (discrete symbol) and continuum k^4
    eig_measured = np.real(D4_phi[0] / phi[0])
    eig_discrete_theory = fourier_symbol_D4(k, h)
    eig_continuum = continuum_symbol_D4(k)

    print(f"  Grid: N={N}, L={L:.4f}, h={h:.6f}")
    print(f"  Mode: n={n_mode}, k={k:.6f}")
    print()
    print(f"  Measured eigenvalue (matrix @ phi)  = {eig_measured:.8f}")
    print(f"  Discrete symbol theory              = {eig_discrete_theory:.8f}")
    print(f"  Continuum k^4                       = {eig_continuum:.8f}")
    print()

    err_vs_discrete = abs(eig_measured - eig_discrete_theory)
    err_vs_continuum = abs(eig_measured - eig_continuum) / eig_continuum

    print(f"  |measured - discrete theory|        = {err_vs_discrete:.2e}  "
          f"(should be ~machine precision)")
    print(f"  |measured - continuum| / continuum  = {err_vs_continuum:.2e}  "
          f"(finite-h discretization error)")
    print()

    passed = err_vs_discrete < 1e-8
    print(f"  TEST 2: {'PASS ✓' if passed else 'FAIL ✗'}")
    print()
    return passed


def main():
    out = Path(__file__).parent / "figures"
    out.mkdir(exist_ok=True)

    t1 = test_symbol_convergence()
    t2 = test_eigenfunction_check()

    # ── Plot: discrete vs continuum symbol ────────────────────────────────
    k_arr = np.linspace(0, 3, 200)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    for h, col in zip([0.5, 0.2, 0.1], ["firebrick", "steelblue", "forestgreen"]):
        ax.plot(k_arr, fourier_symbol_D4(k_arr, h), color=col, lw=2,
                 label=f"D_n^4 symbol, h={h}")
    ax.plot(k_arr, continuum_symbol_D4(k_arr), "k--", lw=2, label="k^4 (continuum)")
    ax.set_xlabel("k")
    ax.set_ylabel("Fourier symbol")
    ax.set_title("Discrete D_n^4 symbol vs continuum k^4")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    h_arr = np.logspace(-2.5, -0.3, 30)
    err_arr = [abs(fourier_symbol_D4(1.0, h) - 1.0) for h in h_arr]
    ax.loglog(h_arr, err_arr, "o-", color="steelblue", lw=2, markersize=4,
              label="measured error")
    ax.loglog(h_arr, h_arr**2 * err_arr[-1] / h_arr[-1]**2, "k--", lw=1,
              alpha=0.6, label="O(h^2) reference")
    ax.set_xlabel("h (grid spacing)")
    ax.set_ylabel("relative error at k=1")
    ax.set_title("Convergence order of D_n^4")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    path = out / "Dn4_operator_validation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")

    print()
    print("=" * 68)
    print("SUMMARY")
    print("=" * 68)
    print(f"  Test 1 (symbol convergence):   {'PASS ✓' if t1 else 'FAIL ✗'}")
    print(f"  Test 2 (eigenfunction check):  {'PASS ✓' if t2 else 'FAIL ✗'}")
    print()
    print("  The discrete operator D_n^4 = (D_n^2)^2 converges to the")
    print("  continuum biharmonic operator at O(h^2), as expected for a")
    print("  standard 5-point stencil composition.")


if __name__ == "__main__":
    main()
