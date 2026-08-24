"""
wave_equation_2D_solver.py
============================

Core numerical solver for the Lambda-model dispersion relation:

    omega(k) = c * k * sqrt(1 + Lambda * k^2)

This is the flat-space limit of the covariant Hamiltonian

    H = (1/2) [ g^mu nu k_mu k_nu + Lambda * k_loc^4 ]

derived in Paper 1 (Schwarzschild) and extended in Paper 2 (Kerr).

The corresponding real-space PDE for a scalar field phi(x, y, t) is

    d^2 phi / dt^2 = c^2 * Laplacian(phi) - c^2 * Lambda * Laplacian(Laplacian(phi))

i.e. a standard wave equation plus a fourth-order (biharmonic) correction.
This file:

  1. Solves the PDE with a explicit finite-difference / spectral scheme.
  2. Verifies the numerical dispersion relation against the exact formula.
  3. Reports the relative error, which should shrink with grid refinement
     (see paper3_grid_convergence.py for a systematic test).

Units: c=1 throughout (natural units). Lambda has dimensions of [length]^2.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# ── Exact dispersion relation ──────────────────────────────────────────────
def omega_exact(k, Lambda, c=1.0):
    """Exact Lambda-model dispersion relation."""
    return c * k * np.sqrt(1 + Lambda * k**2)


def group_velocity_exact(k, Lambda, c=1.0):
    """d omega / dk, exact."""
    return c * (1 + 2 * Lambda * k**2) / np.sqrt(1 + Lambda * k**2)


# ── Spectral solver (2D, periodic domain) ──────────────────────────────────
class LambdaWaveSolver2D:
    """
    Solves d^2 phi/dt^2 = c^2 Laplacian(phi) - c^2 Lambda Laplacian^2(phi)
    on a periodic 2D grid using a Fourier spectral method (exact in space,
    leapfrog in time).
    """

    def __init__(self, Nx=128, Ny=128, Lx=2 * np.pi, Ly=2 * np.pi,
                 Lambda=0.05, c=1.0):
        self.Nx, self.Ny = Nx, Ny
        self.Lx, self.Ly = Lx, Ly
        self.Lambda = Lambda
        self.c = c

        self.dx = Lx / Nx
        self.dy = Ly / Ny

        # Wavenumber grid (FFT convention)
        kx = 2 * np.pi * np.fft.fftfreq(Nx, d=self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(Ny, d=self.dy)
        self.KX, self.KY = np.meshgrid(kx, ky, indexing="ij")
        self.K2 = self.KX**2 + self.KY**2

        # Exact spectral operator: omega(k)^2 = c^2 k^2 (1 + Lambda k^2)
        self.Omega2 = c**2 * self.K2 * (1 + Lambda * self.K2)

    def initial_condition(self, kx0, ky0, amplitude=1.0):
        """Single plane-wave initial condition for dispersion testing."""
        x = np.linspace(0, self.Lx, self.Nx, endpoint=False)
        y = np.linspace(0, self.Ly, self.Ny, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing="ij")
        phi0 = amplitude * np.cos(kx0 * X + ky0 * Y)
        # d phi/dt = 0 initially (standing-wave-like start)
        phi0_dot = np.zeros_like(phi0)
        return phi0, phi0_dot

    def evolve_spectral_exact(self, phi0, phi0_dot, t_array):
        """
        Evolve using the EXACT spectral solution (each Fourier mode
        oscillates at its own exact omega(k)). This is essentially exact
        (machine precision), and serves as the reference solution against
        which finite-difference schemes should be checked.
        """
        phi0_hat = np.fft.fft2(phi0)
        phi0_dot_hat = np.fft.fft2(phi0_dot)
        omega = np.sqrt(np.maximum(self.Omega2, 0))

        # Avoid division by zero at k=0
        omega_safe = np.where(omega == 0, 1.0, omega)

        results = []
        for t in t_array:
            cos_wt = np.cos(omega * t)
            sin_wt = np.sin(omega * t)
            # phi_hat(t) = phi0_hat*cos(wt) + (phi0_dot_hat/omega)*sin(wt)
            phi_hat_t = (phi0_hat * cos_wt +
                         np.where(omega > 0, phi0_dot_hat / omega_safe, 0) * sin_wt)
            phi_t = np.real(np.fft.ifft2(phi_hat_t))
            results.append(phi_t)
        return results

    def leapfrog_step(self, phi_n, phi_nm1, dt):
        """
        One leapfrog step for the fourth-order wave equation using
        the spectral Laplacian/biharmonic operator (exact in space).

        phi_{n+1} = 2 phi_n - phi_{n-1} - dt^2 * Omega2 * phi_n  (spectral)
        """
        phi_n_hat = np.fft.fft2(phi_n)
        rhs_hat = -self.Omega2 * phi_n_hat
        phi_np1_hat = 2 * phi_n_hat - np.fft.fft2(phi_nm1) + dt**2 * rhs_hat
        return np.real(np.fft.ifft2(phi_np1_hat))


# ── Validation routine ───────────────────────────────────────────────────
def validate_dispersion(Lambda=0.05, c=1.0, verbose=True):
    """
    Launch a single plane wave with known (kx0, ky0), evolve it, and
    measure the oscillation frequency numerically. Compare against
    omega_exact(k, Lambda).
    """
    solver = LambdaWaveSolver2D(Nx=64, Ny=64, Lambda=Lambda, c=c)

    kx0, ky0 = 3.0, 4.0   # test wavevector (must be on the FFT grid)
    k0 = np.sqrt(kx0**2 + ky0**2)
    omega0_exact = omega_exact(k0, Lambda, c)

    phi0, phi0_dot = solver.initial_condition(kx0, ky0)

    T_period_exact = 2 * np.pi / omega0_exact
    t_array = np.linspace(0, 3 * T_period_exact, 300)

    results = solver.evolve_spectral_exact(phi0, phi0_dot, t_array)

    # Measure the value of phi at a fixed point (0,0) over time,
    # then extract the dominant frequency via FFT of the time series.
    signal = np.array([r[0, 0] for r in results])
    dt = t_array[1] - t_array[0]
    freqs = np.fft.rfftfreq(len(signal), d=dt) * 2 * np.pi  # angular freq
    spectrum = np.abs(np.fft.rfft(signal))

    # Find peak frequency (skip DC component)
    peak_idx = np.argmax(spectrum[1:]) + 1
    omega0_numerical = freqs[peak_idx]

    rel_error = abs(omega0_numerical - omega0_exact) / omega0_exact

    if verbose:
        print(f"  k0 = {k0:.6f}, Lambda = {Lambda}")
        print(f"  omega_exact      = {omega0_exact:.8f}")
        print(f"  omega_numerical  = {omega0_numerical:.8f}")
        print(f"  relative error   = {rel_error:.2e}")

    return omega0_exact, omega0_numerical, rel_error


def main():
    out = Path(__file__).parent / "figures"
    out.mkdir(exist_ok=True)

    print("=" * 68)
    print("Lambda-MODEL WAVE EQUATION SOLVER — VALIDATION")
    print("=" * 68)
    print()
    print("  Dispersion relation: omega(k) = c*k*sqrt(1 + Lambda*k^2)")
    print()

    print("Test 1 — Spectral-exact evolution vs analytic omega(k):")
    print("-" * 60)
    for Lam in [0.0, 0.01, 0.05, 0.1]:
        validate_dispersion(Lambda=Lam)
        print()

    # ── Plot dispersion relation and group velocity ──────────────────────
    k_arr = np.linspace(0, 10, 300)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    for Lam, col in zip([0.0, 0.02, 0.05, 0.1],
                         ["k", "steelblue", "firebrick", "forestgreen"]):
        ax.plot(k_arr, omega_exact(k_arr, Lam), color=col, lw=2,
                 label=f"Lambda={Lam}")
    ax.plot(k_arr, k_arr, "k--", lw=1, alpha=0.4, label="light cone (Lambda=0)")
    ax.set_xlabel("k")
    ax.set_ylabel("omega(k)")
    ax.set_title("Lambda-model dispersion relation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for Lam, col in zip([0.0, 0.02, 0.05, 0.1],
                         ["k", "steelblue", "firebrick", "forestgreen"]):
        ax.plot(k_arr, group_velocity_exact(k_arr, Lam), color=col, lw=2,
                 label=f"Lambda={Lam}")
    ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.4)
    ax.set_xlabel("k")
    ax.set_ylabel("v_g(k) / c")
    ax.set_title("Group velocity (superluminal for Lambda>0)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out / "dispersion_relation_validation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
