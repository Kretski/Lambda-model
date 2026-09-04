#!/usr/bin/env python3
"""
lambda_diagnostic.py
====================
Self-contained sanity check for the Lambda-model inference pipeline.
No data download, no PyCBC, no GWOSC. Only numpy.

It answers three questions that the current Layers C/D/E cannot answer
about themselves:

  TEST 1 -- How much Lambda information survives after t_c and phi_c are
            projected out? If the answer is ~0%, the pipeline CANNOT
            measure Lambda no matter how clean the data are, and
            Lambda_hat = 0.000000 is a structural artefact, not physics.

  TEST 2 -- What is the phase-aliasing period in Lambda over the chosen
            band, before and after that projection? A scan grid wider
            than the period returns the noise realisation, not Lambda.

  TEST 3 -- Does a full likelihood scan recover an INJECTED nonzero
            Lambda in synthetic Gaussian noise? This is the decisive
            test. If recovery fails here, the bug is in the model code.
            If recovery works here but the real-data run still returns
            0.000000, the bug is in data conditioning / PSD estimation.

Usage:
    python3 lambda_diagnostic.py
"""

import numpy as np

# ----------------------------------------------------------------------
# Configuration -- edit to match your Layer B/C setup
# ----------------------------------------------------------------------
F_LOW = 20.0  # Hz, low frequency cutoff
F_HIGH = 300.0  # Hz, your current F_HIGH
DF = 0.05  # Hz, frequency resolution
MC_DET = 30.9  # detector-frame chirp mass, solar masses (GW150914)
SNR_TARGET = 24.0  # network SNR to normalise the injection to

# dPhi(f) = -KAPPA * Lambda * f**3
# Set KAPPA = 4*pi**3*K(z)/c**3 from your verified alpha=4 mapping.
# KAPPA = 1.0 keeps the diagnostic in dimensionless "Lambda units";
# TEST 1 and TEST 3 are invariant to its value, TEST 2 scales as 1/KAPPA.
KAPPA = 1.0

N_TRIALS = 200  # injection-recovery trials per Lambda value
RNG = np.random.default_rng(20260904)

MSUN_SEC = 4.925490947e-6  # GM_sun/c^3, seconds


# ----------------------------------------------------------------------
# Analytic aLIGO ZERO_DET_high_P PSD (Ajith 2011 fit)
# ----------------------------------------------------------------------
def psd_aligo(f):
    x = f / 215.0
    return 1e-49 * (
        x ** (-4.14)
        - 5.0 * x ** (-2.0)
        + 111.0 * (1.0 - x**2 + 0.5 * x**4) / (1.0 + 0.5 * x**2)
    )


# ----------------------------------------------------------------------
# Leading-order TaylorF2 template
# ----------------------------------------------------------------------
def taylorf2(f, mc_det, lam=0.0, tc=0.0, phic=0.0):
    """Frequency-domain template with the alpha=4 dispersion deformation."""
    mc = mc_det * MSUN_SEC
    amp = f ** (-7.0 / 6.0)
    psi_gr = (3.0 / 128.0) * (np.pi * mc * f) ** (-5.0 / 3.0)
    dphi = -KAPPA * lam * f**3
    return amp * np.exp(1j * (psi_gr + dphi + 2 * np.pi * f * tc - phic))


# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
f = np.arange(F_LOW, F_HIGH + DF, DF)
sn = psd_aligo(f)

h0 = taylorf2(f, MC_DET)
# whitening weight: w = 4|h|^2 df / S_n  -> sum(w) = SNR^2
w = 4.0 * np.abs(h0) ** 2 * DF / sn
w *= SNR_TARGET**2 / w.sum()

hw = np.sqrt(w) * np.exp(1j * np.angle(h0))  # whitened, unit-normalised to SNR


def phase_inner(a, b):
    """Noise- and amplitude-weighted inner product on phase-space functions."""
    return float(np.sum(w * a * b))


# ----------------------------------------------------------------------
# TEST 1 -- information retained after projecting out t_c and phi_c
# ----------------------------------------------------------------------
print("=" * 66)
print("TEST 1 -- Lambda information after t_c / phi_c projection")
print("=" * 66)

v_lam = f**3
basis = [np.ones_like(f), f]  # phi_c and t_c degeneracy directions

v_perp = v_lam.copy()
for b in basis:
    b_n = b / np.sqrt(phase_inner(b, b))
    v_perp -= phase_inner(v_perp, b_n) * b_n

fisher_raw = KAPPA**2 * phase_inner(v_lam, v_lam)
fisher_eff = KAPPA**2 * phase_inner(v_perp, v_perp)
retained = fisher_eff / fisher_raw

sigma_raw = 1.0 / np.sqrt(fisher_raw)
sigma_eff = 1.0 / np.sqrt(fisher_eff)

print(f"  band                     : {F_LOW:.0f}-{F_HIGH:.0f} Hz")
print(f"  Fisher F_LL  (raw)       : {fisher_raw:.6e}")
print(f"  Fisher F_LL  (projected) : {fisher_eff:.6e}")
print(f"  information retained     : {100*retained:.2f} %")
print(f"  sigma_Lambda  raw        : {sigma_raw:.6e}")
print(f"  sigma_Lambda  projected  : {sigma_eff:.6e}")
print(f"  degradation factor       : {sigma_eff/sigma_raw:.2f}x")
print()
if retained < 0.05:
    print("  >> Under 5% retained. t_c and phi_c absorb nearly all of the")
    print("     Lambda phase in this band. A scan that does not project")
    print("     them out is measuring arrival time, not dispersion.")
print()


# ----------------------------------------------------------------------
# TEST 2 -- aliasing period
# ----------------------------------------------------------------------
print("=" * 66)
print("TEST 2 -- phase aliasing period in Lambda")
print("=" * 66)

span_raw = KAPPA * (F_HIGH**3 - F_LOW**3)
alias_raw = 2 * np.pi / span_raw

# post-projection: the residual phase excursion after removing 1 and f
amp_perp = np.sqrt(phase_inner(v_perp, v_perp) / w.sum())
alias_eff = 2 * np.pi / (KAPPA * amp_perp * np.sqrt(2))

print(f"  raw alias period         : {alias_raw:.6e}  Lambda units")
print(f"  effective (post-proj.)   : {alias_eff:.6e}  Lambda units")
print(f"  SAFE SCAN GRID           : [{-alias_eff/2:.4e}, {alias_eff/2:.4e}]")
print()
print("  Any grid wider than this wraps in phase. A wrapped scan returns")
print("  the noise realisation, independent of the true Lambda.")
print()


# ----------------------------------------------------------------------
# TEST 3 -- injection recovery
# ----------------------------------------------------------------------
print("=" * 66)
print("TEST 3 -- injection recovery in synthetic Gaussian noise")
print("=" * 66)

lam_grid = np.linspace(-alias_eff / 2, alias_eff / 2, 201)
tc_grid = np.linspace(-0.01, 0.01, 81)  # seconds

# Precompute whitened templates over the (Lambda, t_c) grid
tmpl = np.empty((len(lam_grid), len(tc_grid), len(f)), dtype=complex)
for i, lam in enumerate(lam_grid):
    base = np.sqrt(w) * np.exp(1j * (np.angle(h0) - KAPPA * lam * f**3))
    for j, tc in enumerate(tc_grid):
        tmpl[i, j] = base * np.exp(2j * np.pi * f * tc)
norms = np.sqrt(np.sum(np.abs(tmpl) ** 2, axis=2))


def recover(data):
    """Max over Lambda, maximised analytically over phi_c (abs) and over t_c."""
    z = np.abs(np.einsum("ijk,k->ij", np.conj(tmpl), data)) / norms
    idx = np.unravel_index(np.argmax(z), z.shape)
    return lam_grid[idx[0]], z[idx]


for lam_true in (0.0, 0.25 * alias_eff / 2, 0.5 * alias_eff / 2):
    inj = np.sqrt(w) * np.exp(1j * (np.angle(h0) - KAPPA * lam_true * f**3))
    rec = np.empty(N_TRIALS)
    for t in range(N_TRIALS):
        noise = (RNG.standard_normal(len(f)) + 1j * RNG.standard_normal(len(f))) / np.sqrt(2)
        rec[t], _ = recover(inj + noise)
    bias = rec.mean() - lam_true
    print(f"  Lambda_inj = {lam_true:+.6e}")
    print(f"    recovered mean = {rec.mean():+.6e}   std = {rec.std():.6e}")
    print(f"    bias           = {bias:+.6e}   ({abs(bias)/max(rec.std(),1e-30):.2f} sigma)")
    print(f"    exact zeros    = {int(np.sum(rec == 0.0))}/{N_TRIALS}")
    print()

print("=" * 66)
print("READING THE RESULT")
print("=" * 66)
print("  - Nonzero injections recovered with bias << std  -> model code OK.")
print("  - Recovered mean pinned at 0 for ALL injections   -> the Lambda")
print("    phase is not reaching the likelihood. Check that the")
print("    deformation is applied BEFORE whitening and is not lost to")
print("    an abs() or a real() somewhere in likelihood.py.")
print("  - 'exact zeros' should be ~1/201 of trials by chance. A count")
print("    near N_TRIALS reproduces the 0.000000 pathology.")
print("  - If this passes but real data still gives 0.000000, the fault")
print("    is in conditioning/PSD, not in the Lambda model.")
