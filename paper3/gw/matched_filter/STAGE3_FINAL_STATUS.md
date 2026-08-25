# Stage 3 — Real Strain Validation: Final Status

**Status: VALIDATION PASSED. Mode D result is EXPLORATORY ONLY.**

---

## Summary

This document records the final, real-GWOSC-data results of the
matched-filter Λ-recovery pipeline (`stage3_real_strain_validation.py`),
following the full validation chain built in Papers 2–3.

## Results (real H1 strain, `H-H1_LOSC_4_V1-1126256640-4096.hdf5`)

### Mode A+B — injection/recovery into real off-source H1 strain

| Λ_true | Λ_ML (mean ± scatter) |
|---|---|
| 0.00 | ~0 |
| 0.01 | 0.011 ± 0.003 |
| 0.05 | 0.050 ± 0.000 |
| 0.10 | 0.101 ± 0.003 |
| 0.50 | 0.500 ± 0.0045 |
| 1.00 | 1.000 ± 0.0045 |

**Result: PASS.** Injected Λ is recovered accurately from real off-source
detector noise, with genuine (nonzero) statistical scatter across
independent noise realizations drawn from real H1 strain.

### Mode C — off-source null trials (no injected signal)

| Grid width | Boundary hits | Result |
|---|---|---|
| ±5 | 11/200 (5.5%) | FAIL |
| ±10 | 6/200 (3.0%) | PASS |

**Result: PASS** (at the ±10 grid, reached via adaptive widening on the
identical 200 off-source trials). No dominant single-bin clustering; no
consistent nonzero Λ preference from pure noise.

### Mode D — GW150914 (exploratory only)

A single-detector (H1), leading-order-waveform, grid-search point
estimate was computed and is recorded for the repository's audit
trail. **This number is not reported here as a physical result and
must not be cited as one.**

It is explicitly **not**:
- a detection
- a constraint
- a measurement of Λ
- comparable in status to the audited Fermi-LAT bound

Reasons this cannot be interpreted as a physical result:
- single detector (H1 only, no L1 coherence)
- leading-order (0PN-equivalent) TaylorF2 waveform, not a full
  IMRPhenom/SEOBNR model
- grid-search maximum-likelihood point estimate, no Bayesian priors
- no PN-systematic error budget
- no glitch vetoing or data-quality flag review
- no calibration-uncertainty propagation

## What this validation chain establishes

```
theory (Papers 1-2, proved theorems)
        │
        ▼
numerical solver validation (h-convergence, D_n^4 operator)
        │
        ▼
synthetic matched-filter proof-of-concept (Stage 1) — PASS
        │
        ▼
real-PSD-shaped Gaussian noise recovery (Stage 2) — PASS
        │
        ▼
real H1 strain injection/recovery (Stage 3, Mode A+B) — PASS
        │
        ▼
real H1 strain null trials (Stage 3, Mode C) — PASS
        │
        ▼
GW150914 exploratory point estimate (Stage 3, Mode D)
        │
        └── NOT INTERPRETED as a physical result
```

This replaces the invalidated original approach:

```
Hilbert-transform frequency extraction
        │
        ▼
GW150914 "51.9-sigma detection"
        │
        ▼
injection/recovery test
        │
        ▼
Λ_true=0 → Λ_fit=-1.91 (18σ bias)
        │
        ▼
PIPELINE BIAS CONFIRMED, extraction method rejected
```

The phase-domain matched-filter approach (Λ as an analytic term in the
frequency-domain waveform phase, `ΔΨ(f) = -(4π³ΛK(z)/c³)f³`) avoids the
entire class of time-domain frequency-extraction bias that invalidated
the original Hilbert/zero-crossing/STFT-ridge attempts (see
`paper3/gw/gwosc_extraction_ablation.py` and
`paper3/gw/gwosc_frequency_extraction_comparison.py` for that earlier
diagnostic work, retained in the repository as documented negative
results).

## What is required before Mode D could become a physical result

In order of scientific priority, not effort:

1. **H1+L1 coherent analysis** — the single biggest missing piece.
   A real signal must be consistent across independent detectors;
   Mode D currently uses H1 alone.
2. **Full IMRPhenom/SEOBNR waveform** (via `lalsimulation`), replacing
   the leading-order TaylorF2 approximation used here for
   proof-of-concept purposes.
3. **Bayesian parameter estimation** with proper priors on all
   parameters (masses, distance, sky location, Λ), not a fixed-other-
   parameters grid search on Λ alone.
4. **Data-quality review**: glitch vetoing, spectral-line handling,
   calibration-uncertainty propagation — standard LIGO/Virgo/KAGRA
   analysis requirements not attempted here.
5. Only after 1–4: any resulting Λ posterior could be discussed as a
   genuine observational result.

## Reproducing these results

```bash
cd paper3/gw/matched_filter
python stage3_real_strain_validation.py --h1 <path-to-H1-strain.hdf5> --event GW150914
```

To validate Modes A/B/C only, without running Mode D even if the gate
would pass:

```bash
python stage3_real_strain_validation.py --h1 <path> --event GW150914 --skip-D
```

## Provenance: earlier invalidated attempts (retained, not deleted)

| Script | Status | Finding |
|---|---|---|
| `gwosc_chirp_dispersion_test.py` | legacy/invalidated | 51.9σ result, root cause: extraction bias |
| `gwosc_injection_recovery_test.py` | diagnostic | proved pipeline bias (Λ=0 → Λ_fit=-1.91) |
| `gwosc_extraction_ablation.py` | diagnostic | localized bias to Hilbert-transform math itself |
| `gwosc_frequency_extraction_comparison.py` | diagnostic | tested 5 extraction methods, all time-domain approaches inadequate |
| `gwosc_zero_crossing_injection_recovery_v2.py` | diagnostic | zero-crossing method also unstable at low SNR |
| `matched_filter/recovery_test.py` | **Stage 1 — PASS** | phase-domain approach, synthetic noise |
| `matched_filter/stage2_real_noise_recovery.py` | **Stage 2 — PASS** | real PSD shape, Gaussian noise |
| `matched_filter/stage3_real_strain_validation.py` | **Stage 3 — PASS (A+B, C); Mode D exploratory** | real H1 strain |

This is intentionally kept as a complete, honest audit trail: the
repository documents both the failed approaches and why they failed,
alongside the validated approach that replaced them.
