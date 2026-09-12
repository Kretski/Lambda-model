"""
response_function.py -- STAGE 5I QNM -> response spectrum
==========================================================

Builds R(omega; m) as a sum of complex-pole (Breit-Wigner) resonances
from the QNM branch data (Re/Im frequencies per m, per q), and derives
the observable spectral density S(omega;m) = |R(omega;m)|^2.

Purpose: turn the internal QNM branch-tracking output into the one
thing an experimentalist can actually measure -- a driven response
spectrum -- so that "branch termination near m_c" becomes a claim
about a measurable quantity (peak width / peak existence), not an
artifact of the solver's own bookkeeping.

Input: a CSV with columns m, q, f_model_hz, f_model_im_hz -- the SAME
format generalized_branch_resolution.py consumes (the model's own
computed complex frequencies, NOT qnms.csv/EXP_B, which only has the
experimental Re(f) and no Im part).

Output:
  - per-m spectral density S(omega; m) sampled over a frequency window
  - per-(m,q) peak diagnostics: FWHM, peak height, peak position
  - a table showing how FWHM/height evolve as m -> m_c, so a
    termination signature (width diverging / height vanishing) can be
    read off directly, instead of inferred from solver failure alone

EXPLICITLY FLAGGED ASSUMPTION (must be stated in any write-up):
Pole amplitudes A_q are set to a fixed "unit residue" (A_q = 1)
because the branch-tracking data as currently delivered carries no
coupling/excitation-amplitude information. This means the lineshape
WIDTHS and peak POSITIONS are physical (they come directly from the
computed complex frequencies), but absolute peak HEIGHTS are not yet
a first-principles prediction -- only relative height/width trends
across m are meaningful until amplitudes are derived from the actual
Hamiltonian (e.g. via the residue of R at each pole, from the
transfer-matrix / radial-action framework in stage5I_3_resonance).
Do not present peak-height numbers as calibrated predictions until
that is done.
"""

import csv
import sys
from collections import defaultdict

import numpy as np


def load_branches(path):
    """Load m -> list of (q, f_re_hz, f_im_hz) from a model-output CSV
    in the same format generalized_branch_resolution.py consumes."""
    by_m = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            m = int(float(row["m"]))
            q = int(float(row["q"]))
            f_re = float(row["f_model_hz"])
            f_im = float(row["f_model_im_hz"])
            by_m[m].append((q, f_re, f_im))
    return by_m


def response(omega, poles, amplitudes=None):
    """R(omega) = sum_q A_q / (omega - omega_q)

    poles: complex angular frequencies omega_q = 2*pi*(f_re + i*f_im)
    amplitudes: optional complex weights A_q (default: all 1, see the
    module docstring for why this is a flagged placeholder, not a
    fitted or derived value)
    """
    poles = np.asarray(poles, dtype=complex)
    if amplitudes is None:
        amplitudes = np.ones(len(poles), dtype=complex)
    amplitudes = np.asarray(amplitudes, dtype=complex)
    omega = np.asarray(omega, dtype=complex)
    return np.sum(amplitudes[None, :] / (omega[..., None] - poles[None, :]), axis=-1)


def spectral_density(omega, poles, amplitudes=None):
    R = response(omega, poles, amplitudes)
    return np.abs(R) ** 2


def peak_diagnostics(freq_grid_hz, S, f_re_guess_hz, search_half_width_hz=0.5):
    """Find the peak nearest f_re_guess_hz in S and estimate its FWHM
    by simple half-max crossing within the search window. Returns None
    if the window contains no data -- never silently substitutes a
    default, per the repo's existing no-silent-fallback convention."""
    mask = np.abs(freq_grid_hz - f_re_guess_hz) <= search_half_width_hz
    if not np.any(mask):
        return None
    freqs_local = freq_grid_hz[mask]
    S_local = S[mask]
    idx_local = np.argmax(S_local)
    peak_freq = freqs_local[idx_local]
    peak_height = S_local[idx_local]

    half = peak_height / 2.0
    above = S_local >= half
    if not np.any(above):
        return {"peak_freq_hz": peak_freq, "peak_height": peak_height, "fwhm_hz": None}
    idxs = np.where(above)[0]
    fwhm = freqs_local[idxs[-1]] - freqs_local[idxs[0]]
    return {"peak_freq_hz": peak_freq, "peak_height": peak_height, "fwhm_hz": fwhm}


def scan_termination(by_m, q_target=2, freq_window_hz=1.0, n_points=4000):
    """For each m that has a q_target branch, build S(omega;m) (using
    ALL branches present at that m as poles, so neighboring resonances
    contribute to the lineshape) and extract FWHM/height for the
    q_target peak. Returns a list of dicts sorted by m -- read this
    directly for a termination signature (diverging FWHM / vanishing
    height as m -> m_c), rather than inferring it from solver
    non-convergence."""
    results = []
    for m in sorted(by_m):
        branches = by_m[m]
        target = [b for b in branches if b[0] == q_target]
        if not target:
            results.append({"m": m, "status": "no_q_target_branch"})
            continue
        _, f_re, f_im = target[0]

        all_poles = [2 * np.pi * (fr + 1j * fi) for (_, fr, fi) in branches]
        f_grid_hz = np.linspace(f_re - freq_window_hz, f_re + freq_window_hz, n_points)
        omega_grid = 2 * np.pi * f_grid_hz
        S = spectral_density(omega_grid, all_poles)

        diag = peak_diagnostics(f_grid_hz, S, f_re, search_half_width_hz=freq_window_hz * 0.5)
        if diag is None:
            results.append({"m": m, "status": "empty_window"})
            continue
        diag["m"] = m
        diag["im_hz"] = f_im
        diag["status"] = "ok"
        results.append(diag)
    return results


def print_report(results):
    print("=" * 100)
    print(f"RESPONSE-SPECTRUM TERMINATION SCAN")
    print("=" * 100)
    header = f"{'m':>5} {'status':>16} {'peak_freq_hz':>14} {'fwhm_hz':>12} {'im_hz':>12}"
    print(header)
    for r in results:
        if r["status"] != "ok":
            print(f"{r['m']:>5} {r['status']:>16}")
            continue
        fwhm = f"{r['fwhm_hz']:.5f}" if r["fwhm_hz"] is not None else "no_half_max"
        print(f"{r['m']:>5} {r['status']:>16} {r['peak_freq_hz']:>14.5f} "
              f"{fwhm:>12} {r['im_hz']:>12.6f}")
    print()
    print("Read this table for the termination signature:")
    print("  - fwhm_hz growing sharply / im_hz growing in magnitude as m -> m_c")
    print("    is the predicted observable (peak broadens toward the background)")
    print("  - 'no_q_target_branch' at some m is a genuine gap in the model output,")
    print("    not evidence by itself of physical termination -- cross-check against")
    print("    generalized_branch_resolution.py's UNIDENTIFIED list for that m")
    print("  - a peak height (not width) trending to ~0 is NOT yet a calibrated result")
    print("    (see amplitude-placeholder caveat in the module docstring)")


def main(path, q_target=2, freq_window_hz=1.0):
    by_m = load_branches(path)
    results = scan_termination(by_m, q_target=q_target, freq_window_hz=freq_window_hz)
    print_report(results)
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python response_function.py <model_results.csv> [q_target] [freq_window_hz]")
        sys.exit(1)
    q_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    fw_arg = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    main(sys.argv[1], q_target=q_arg, freq_window_hz=fw_arg)
