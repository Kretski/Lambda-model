# Stage 4 — Coherent H1+L1 Validation: Final Status

**Status: VALIDATION PASSED (Modes 4C, 4D). Modes 4E/4F reveal a**
**dominant PN-systematic; no physical result is claimed.**

---

## Summary

Real H1+L1 coherent matched-filter validation, following the same
staged-gating discipline as Stage 3.

## Results (real strain, GW150914 files)

### Mode 4C — coherent off-source null trials

| Grid width | Boundary hits | Result |
|---|---|---|
| ±5 | 13/200 (6.5%) | FAIL |
| ±10 | 9/200 (4.5%) | PASS |

**PASS** at ±10 (adaptive widening, same 200 trials).

### Mode 4D — coherent injection/recovery

All six injected Λ values (0, 0.01, 0.05, 0.1, 0.5, 1.0) recovered
exactly. **PASS.**

*Caveat:* scatter=0.0000 across realizations confirms the estimator
recovers Λ when injection and recovery share the same 0PN waveform
structure. It does **not** yet establish realistic statistical
uncertainty under waveform-model variation — that is precisely what
Modes 4E/4F expose next.

### Modes 4E/4F — GW150914 exploratory, waveform-systematics check

| Waveform | Λ_ML | Fisher error |
|---|---|---|
| 0PN (Mode 4E) | +6.5700 | 0.1206 |
| 1PN (Mode 4F) | −1.2700 | 0.0758 |
| **Difference** | **7.8400** | — |

## Interpretation

**The PN-systematic (7.84) is roughly 65× larger than either
individual statistical error bar (~0.08–0.12).** This means the
current point estimate is dominated by waveform-model choice, not by
detector noise. The 0PN→1PN shift is not a small correction — it flips
the sign of the inferred Λ entirely.

**No physical result, constraint, or detection is claimed from this
number, at any PN order tested here.**

This is a stronger and more informative diagnostic than the earlier
invalidated 51.9σ Hilbert-transform result: rather than a single
biased number, Stage 4 directly demonstrates *why* single-waveform
point estimates cannot be trusted at this stage — the inferred Λ is
degenerate with unmodeled PN structure.

## What this does NOT mean

This does **not** invalidate the matched-filter architecture itself.
Modes 4C and 4D confirm the joint H1+L1 estimator is statistically
sound: it passes off-source null trials and recovers known injected
Λ accurately when the injection and recovery waveforms match
structurally. The problem is specifically **waveform-model systematic
error**, isolated and quantified for the first time in this pipeline
via Mode 4F — this is the intended purpose of a systematics check, and
it worked.

## Required before Mode 4E/4F could inform a physical claim

1. **Validated GR waveform ensemble** (Stage 5) — IMRPhenom, SEOBNR, or
   other independently validated models via `lalsimulation`, replacing
   the hand-rolled 0PN/1PN prescriptions used here for proof-of-concept
   purposes only.
2. **Cross-waveform injection/recovery**: inject with one waveform
   family, recover with another, at Λ_true=0 and several nonzero
   values, to directly quantify Λ_bias from waveform-family mismatch
   under controlled, known conditions — the correct next test, run
   BEFORE any further GW150914 analysis.
3. Only if cross-waveform bias is shown to be small and well-
   characterized: revisit GW150914 with a validated waveform and a
   proper PN-systematic error budget.
4. Everything listed in Stage 3's requirements (H1+L1 antenna pattern,
   sky-location marginalization, Bayesian priors, glitch vetoing,
   calibration uncertainty) remains outstanding as well.

## Reproducing these results

```bash
cd paper3/gw/matched_filter
python stage4_coherent_h1l1_validation.py --h1 <H1.hdf5> --l1 <L1.hdf5> --event GW150914
```

To validate Modes 4A–4D only, without running 4E/4F even if the gate
would pass:

```bash
python stage4_coherent_h1l1_validation.py --h1 <H1.hdf5> --l1 <L1.hdf5> --event GW150914 --skip-E
```

## Updated provenance table

| Stage/Script | Status | Finding |
|---|---|---|
| Hilbert transform (legacy) | invalidated | 51.9σ, root cause: extraction bias |
| Zero-crossing (legacy) | invalidated | unstable at low SNR even after fixes |
| Stage 1 (synthetic matched filter) | PASS | phase-domain Λ identifiable |
| Stage 2 (real PSD + Gaussian noise) | PASS | robust to real noise spectrum |
| Stage 3 Mode A+B (real H1 injection) | PASS | robust to real noise realizations |
| Stage 3 Mode C (real H1 null) | PASS | no spurious Λ from pure noise |
| Stage 3 Mode D (GW150914, H1 only) | exploratory, not interpreted | Λ=−8.15 (audit trail only) |
| **Stage 4 Mode 4C (coherent null)** | **PASS** | ±10 grid, no spurious coherent Λ |
| **Stage 4 Mode 4D (coherent injection)** | **PASS** | exact recovery, structurally matched waveforms |
| **Stage 4 Mode 4E/4F (GW150914 coherent, PN systematics)** | **PN-systematic dominates** | 0PN=+6.57, 1PN=−1.27, ΔΛ=7.84 — **no claim made** |

**Next required step: Stage 5, cross-waveform-family injection/recovery
using validated waveform models, before any further real-event
analysis.**
