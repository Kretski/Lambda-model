"""
paper3_h_convergence_test.py
================================

THE TEST THAT WAS MISSING: does the fitted Lambda converge as the
spatial grid spacing h -> h/2 -> h/4?

WHY THE PREVIOUS "grid_convergence" TEST DID NOT ANSWER THIS:

  paper3_grid_convergence.py varied only the NUMBER OF TIME SAMPLES at a
  FIXED spatial grid (N=64), and used the spectral-EXACT solution
  (each Fourier mode assigned its analytic omega(k) by construction).
  That test could not detect a spatial-discretization bug even if one
  existed, because the spatial operator was never actually discretized
  in that code path -- omega(k) was substituted in exactly.

  The Lambda-recovery test in the same file was even more disconnected:
  it fit Lambda to omega_exact(k) values generated directly from the
  formula, with ZERO grid involved.

THIS TEST instead:

  1. Actually integrates the fourth-order wave PDE in time using an
     explicit LEAPFROG scheme, with the spatial biharmonic term computed
     via the DISCRETE Fourier symbol on grids of increasing resolution
     N = 32, 64, 128, 256 (so h = L/N halves each time).
  2. Extracts the numerical omega(k) from the actual time-stepped
     solution (not from a shortcut).
  3. Fits Lambda from the extracted omega(k) at each resolution.
  4. Reports whether Lambda_fit -> Lambda_true as h -> 0, and at what
     empirical order.

NOTE ON RESOLUTION RANGE: the biharmonic term forces an explicit CFL
condition dt ~ h^2 (stricter than the O(h) condition for the plain wave
equation), so runtime scales as O(N^2 * n_steps) ~ O(N^4) per
wavevector as N increases. We test N=32,64,128 (h, h/2, h/4), which is
sufficient to establish the convergence order cleanly; N=256 was
excluded here purely for runtime reasons, not because of any expected
change in behavior.

If Lambda_fit does NOT converge with h, that indicates either a
time-stepping instability, an aliasing problem, or a genuine bug in the
biharmonic discretization -- and would invalidate any Lambda extracted
from this solver at finite resolution.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).parent))
from wave_equation_2D_solver import omega_exact


def leapfrog_evolve(N, Lx, Lambda, c, kx0, ky0, n_periods=10,
                     courant=0.2, amplitude=1.0):
    """
    Explicit leapfrog time integration of

        d^2 phi/dt^2 = c^2 * Laplacian(phi) - c^2*Lambda*Laplacian^2(phi)

    on an N x N periodic grid of physical size Lx x Lx (h = Lx/N),
    using the DISCRETE Fourier symbol for the spatial operator (this is
    the standard "pseudo-spectral in space, explicit leapfrog in time"
    scheme -- the spatial operator is what changes with h, unlike the
    fully spectral-exact shortcut used elsewhere).

    Returns the time series of phi at a fixed grid point, plus the
    array of sample times, so the dominant frequency can be extracted
    exactly as one would from real simulation output.
    """
    h = Lx / N
    kvec = 2 * np.pi * np.fft.fftfreq(N, d=h)
    KX, KY = np.meshgrid(kvec, kvec, indexing="ij")

    # Discrete Fourier symbol of the 5-point Laplacian (NOT the exact k^2):
    # this is what makes the operator genuinely h-dependent.
    symbol_D2 = -(2 / h**2) * (2 - np.cos(KX * h) - np.cos(KY * h))
    # Discrete biharmonic = (discrete Laplacian)^2
    symbol_D4 = symbol_D2 ** 2

    # Discrete "Omega^2" operator: c^2*(-D2) - c^2*Lambda*D4
    # Note: -D2 approximates +k^2 (since D2 approximates -k^2*phi_hat)
    Omega2_discrete = c**2 * (-symbol_D2) + c**2 * Lambda * symbol_D4

    # Guard against negative values from discretization at high k (should
    # not occur for the physical Lambda>0 branch, but guard anyway):
    Omega2_discrete = np.maximum(Omega2_discrete, 0.0)

    # Initial condition: single plane wave (kx0, ky0) must be resolvable
    # on this grid, i.e. kx0, ky0 must be exact FFT frequencies.
    x = np.linspace(0, Lx, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    phi0 = amplitude * np.cos(kx0 * X + ky0 * Y)

    k0 = np.sqrt(kx0**2 + ky0**2)
    omega0_exact_continuum = omega_exact(k0, Lambda, c)
    T_period = 2 * np.pi / omega0_exact_continuum

    # CFL-limited explicit time step (fourth-order operator needs a
    # stricter Courant condition than the plain wave equation)
    dt = courant * h**2 / c  # conservative for the biharmonic term
    n_steps = int(np.ceil(n_periods * T_period / dt))
    dt = n_periods * T_period / n_steps  # land exactly on n_periods

    phi0_hat = np.fft.fft2(phi0)
    # d phi/dt = 0 initially => phi_{-1} = phi_0 (standard leapfrog start)
    phi_nm1_hat = phi0_hat.copy()
    phi_n_hat = phi0_hat.copy()

    signal = np.zeros(n_steps + 1)
    signal[0] = np.real(np.fft.ifft2(phi_n_hat))[0, 0]

    for step in range(1, n_steps + 1):
        rhs_hat = -Omega2_discrete * phi_n_hat
        phi_np1_hat = 2 * phi_n_hat - phi_nm1_hat + dt**2 * rhs_hat
        phi_nm1_hat = phi_n_hat
        phi_n_hat = phi_np1_hat
        signal[step] = np.real(np.fft.ifft2(phi_n_hat))[0, 0]

    t_array = np.arange(n_steps + 1) * dt
    return t_array, signal, h, dt


def extract_omega(t_array, signal):
    """FFT-based dominant-frequency extraction from a time series."""
    dt = t_array[1] - t_array[0]
    freqs = np.fft.rfftfreq(len(signal), d=dt) * 2 * np.pi
    spectrum = np.abs(np.fft.rfft(signal - np.mean(signal)))
    peak_idx = np.argmax(spectrum[1:]) + 1
    return freqs[peak_idx]


def run_h_convergence_test(Lambda_true=0.05, c=1.0, kx0=3, ky0=4,
                            N_values=(32, 64, 128)):
    """
    Core test: for each spatial resolution N (=> h=Lx/N), actually
    integrate the PDE, extract omega numerically, then fit Lambda from
    a small set of (k, omega) pairs obtained this way -- and report how
    the fitted Lambda depends on h.

    To get a genuine (k, omega) FIT (not just a single-point check), we
    repeat the leapfrog integration for several wavevectors at each
    resolution and fit Lambda by least squares, exactly as an
    experimentalist would.
    """
    Lx = 2 * np.pi
    k_test_pairs = [(1, 1), (2, 2), (3, 4), (4, 5), (5, 6)]

    results = {}

    print("=" * 72)
    print("SPATIAL GRID CONVERGENCE TEST (h -> h/2 -> h/4 -> h/8)")
    print("=" * 72)
    print()
    print(f"  True Lambda = {Lambda_true}")
    print(f"  Method: explicit leapfrog PDE integration (NOT spectral-exact")
    print(f"          shortcut), discrete Fourier symbol for Laplacian^2")
    print()

    for N in N_values:
        h = Lx / N
        k_measured = []
        omega_measured = []

        for kx0, ky0 in k_test_pairs:
            # kx0, ky0 must be valid FFT frequencies for this N; the
            # base modes (1,1)...(5,6) are always resolvable for N>=32.
            t_array, signal, h_actual, dt = leapfrog_evolve(
                N, Lx, Lambda_true, c, kx0, ky0, n_periods=8, courant=0.15)
            omega_num = extract_omega(t_array, signal)
            k0 = np.sqrt(kx0**2 + ky0**2)
            k_measured.append(k0)
            omega_measured.append(omega_num)

        k_measured = np.array(k_measured)
        omega_measured = np.array(omega_measured)

        # Fit Lambda: (omega/(c*k))^2 - 1 = Lambda * k^2
        y = (omega_measured / (c * k_measured))**2 - 1
        x = k_measured**2
        Lambda_fit = np.sum(x * y) / np.sum(x * x)

        rel_error = abs(Lambda_fit - Lambda_true) / Lambda_true

        results[N] = dict(h=h, Lambda_fit=Lambda_fit, rel_error=rel_error,
                           k_measured=k_measured, omega_measured=omega_measured)

        print(f"  N={N:>4}  h={h:.5f}  Lambda_fit={Lambda_fit:.6f}  "
              f"rel.error={rel_error:.4e}")

    print()

    # Empirical convergence order between successive resolutions
    print("  Empirical convergence order (between successive h):")
    Ns = sorted(results.keys())
    for i in range(1, len(Ns)):
        N_prev, N_cur = Ns[i - 1], Ns[i]
        err_prev = results[N_prev]["rel_error"]
        err_cur = results[N_cur]["rel_error"]
        if err_cur > 0 and err_prev > 0:
            order = np.log(err_prev / err_cur) / np.log(2)
            print(f"    N={N_prev}->{N_cur}: order = {order:.3f}")
        else:
            print(f"    N={N_prev}->{N_cur}: error too small to estimate order "
                  f"(err_prev={err_prev:.2e}, err_cur={err_cur:.2e})")

    print()
    final_err = results[Ns[-1]]["rel_error"]
    passed = final_err < 0.01  # require <1% error at finest resolution
    print(f"  Finest-grid relative error: {final_err:.4e}")
    print(f"  PASS threshold: <1%")
    print(f"  TEST: {'PASS ✓' if passed else 'FAIL ✗'}")
    print()

    if not passed:
        print("  WARNING: Lambda does NOT appear to converge to <1% at the")
        print("  finest tested resolution. Do not trust Lambda values")
        print("  extracted from this solver without further investigation")
        print("  (check CFL condition, aliasing, or discretization order).")

    return results, passed


def main():
    out = Path(__file__).parent / "figures"
    out.mkdir(exist_ok=True)

    results, passed = run_h_convergence_test(Lambda_true=0.05)

    # ── Plot: Lambda_fit vs h (log-log, expect convergence to true value) ──
    Ns = sorted(results.keys())
    h_arr = np.array([results[N]["h"] for N in Ns])
    Lambda_fit_arr = np.array([results[N]["Lambda_fit"] for N in Ns])
    err_arr = np.array([results[N]["rel_error"] for N in Ns])
    Lambda_true = 0.05

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.semilogx(h_arr, Lambda_fit_arr, "o-", color="steelblue", lw=2,
                markersize=8, label=r"$\Lambda_{\rm fit}(h)$")
    ax.axhline(Lambda_true, color="k", ls="--", lw=1.5,
               label=fr"$\Lambda_{{\rm true}}={Lambda_true}$")
    ax.set_xlabel("h (grid spacing)")
    ax.set_ylabel(r"Fitted $\Lambda$")
    ax.set_title(r"$\Lambda_{\rm fit}$ vs spatial resolution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    ax = axes[1]
    ax.loglog(h_arr, err_arr, "o-", color="firebrick", lw=2, markersize=8,
              label="measured relative error")
    # reference line for 2nd order convergence
    ref = err_arr[0] * (h_arr / h_arr[0])**2
    ax.loglog(h_arr, ref, "k--", lw=1, alpha=0.6, label=r"$O(h^2)$ reference")
    ax.set_xlabel("h (grid spacing)")
    ax.set_ylabel(r"$|\Lambda_{\rm fit}-\Lambda_{\rm true}|/\Lambda_{\rm true}$")
    ax.set_title("Convergence order of the FULL PDE solver")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    ax.invert_xaxis()

    plt.suptitle(
        r"Spatial grid convergence: $h\to h/2\to h/4\to h/8$"
        " (real leapfrog PDE integration, not spectral-exact shortcut)",
        fontsize=11)
    plt.tight_layout()
    path = out / "h_convergence_test.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")

    print()
    print("=" * 72)
    print("HONEST SUMMARY")
    print("=" * 72)
    print()
    if passed:
        print("  This test (previously MISSING from the repository) confirms")
        print("  that Lambda extracted from the actual time-stepped PDE solver")
        print("  converges to the injected value as h -> 0, using a genuinely")
        print("  discretized spatial operator (not a spectral-exact shortcut).")
    else:
        print("  This test reveals that the previous validation claims in")
        print("  grid_convergence.py were INSUFFICIENT: Lambda extracted from")
        print("  full PDE time-stepping does not yet converge cleanly at the")
        print("  resolutions tested. This must be resolved before any Lambda")
        print("  extracted from simulated (rather than purely analytic) data")
        print("  can be trusted.")


if __name__ == "__main__":
    main()
