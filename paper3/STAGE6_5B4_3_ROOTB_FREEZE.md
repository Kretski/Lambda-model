# Stage 6.5B4-3 — Root-B Freeze: Combined Evidence and Experimental Comparison

## Purpose

This document freezes the Root-B q=2 branch of the giant-vortex QNM model
as a numerically established object, combining three independent lines of
evidence, and records its post-hoc comparison against the experimental
q=2 resonance frequencies (Švančara et al. EXP B).

The freeze is a statement about **numerical reproducibility and
experimental consistency** of this specific branch. It does not extend to
q=3/q=4 (still unresolved, see `PAPER3_ADDENDUM_BLIND_2D.md`), and it does
not constitute a claim about Kerr black holes, universality of Λ, or any
result outside the giant-vortex laboratory system (see "Permanent
boundaries" in `PAPER3_VALIDATION_STATUS.md`).

## Three independent lines of evidence

### 1. Continuation (stage5I_29 / 30 / 31)

Root-B was continued from a validated anchor at m=-14
(f = 9.596712868 − 0.135838317i Hz) across the range m=-14 to m=-4, seeded
at each step only from the immediately preceding accepted root — never
from experimental q=2 values.

**Coverage is not fully continuous in the permanent record**, due to a
session interruption during the m=-14→-4 run. Recovered segments:

| Range | Points | Status |
|---|---|---|
| m=-14.0 to -13.7 | 4 | full (anchor + continuation) |
| m=-11.5 | 1 | sparse checkpoint |
| m=-10.9, -10.8 | 2 | full |
| m=-7.6 | 1 | sparse checkpoint |
| m=-6.9 to -4.0 | 20 | full |

At every recovered point, residuals are ≤10⁻¹⁴ and no branch jumps are
observed (Re and Im vary smoothly and monotonically between adjacent
points). This is treated as strong but not exhaustive evidence of a
single continuous branch; the un-sampled intervals (m=-13.6 to -11.6,
-11.4 to -10.9, -10.7 to -7.7, -7.5 to -7.0) were not independently
verified point-by-point and are not claimed to have been.

### 2. Independent solver reproduction (stage6_5B4_2)

Seven frozen Root-B points (m = -4, -5, -6, -11, -12, -13, -14) were each
re-solved from two independent local perturbations
(Δf = (+0.01,+0.01) and (−0.01,+0.005) Hz) using a custom damped Newton
solver — central finite-difference Jacobian, explicit 2×2 linear solve,
backtracking line search — built independently of `scipy.optimize.root`
and `scipy.optimize.minimize`, and using no continuation seeding and no
experimental frequencies anywhere.

The solver reuses the same validated physical residual
(`resonance_factor_v2`, `find_light_ring` from `stage5I_3_resonance_v2.py`)
— this is a test of solver-independence, not a re-derivation of the
underlying physics.

**Result: 14/14 independent solves converged to the frozen roots.**
Max |Res| among successful solves: 3.225×10⁻¹¹. Max distance between an
independently-found root and its corresponding frozen point:
5.467×10⁻¹⁰ Hz.

This shows Root-B is not an artifact of the specific `hybr` continuation
solver used to originally locate it.

### 3. Experimental comparison (post-hoc only)

Experimental q=2 values are used here **only** as a final comparison,
after both the continuation and the independent-solver checks above were
completed without reference to them.

| m | Root-B (Hz) | Experimental q=2 (Hz) | rel. error | Source |
|---:|---:|---:|---:|---|
| -19 | 10.82690 | 10.83337 | 0.060% | discovery grid |
| -18 | 10.58474 | 10.58962 | 0.046% | discovery grid |
| -17 | 10.34130 | 10.34296 | 0.016% | discovery grid |
| -16 | 10.09596 | 10.09844 | 0.025% | discovery grid |
| -14 | 9.59671 | 9.59930 | 0.027% | anchor |
| -13 | 9.34113 | 9.34220 | 0.011% | continuation |
| -12 | 9.08027 | 9.08209 | 0.020% | continuation |
| -11 | 8.81297 | 8.81124 | 0.020% | continuation |
| -10 | 8.53792 | 8.53343 | 0.053% | discovery grid |
| -9 | 8.25363 | 8.24601 | 0.092% | discovery grid |
| -8 | 7.95839 | 7.94560 | 0.161% | discovery grid |
| -7 | 7.65039 | 7.63379 | 0.217% | discovery grid |
| -6 | 7.32784 | 7.29834 | 0.404% | continuation (confirmed) |
| -5 | 6.98952 | 6.94859 | 0.589% | continuation (confirmed) |
| -4 | 6.63644 | 6.57004 | 1.011% | continuation (confirmed) |

**Continuation confirmation at m=-6,-5,-4:** the newly-continued Root-B
values at these three points are numerically identical (to the precision
recorded) to the previously-used discovery-grid values. This directly
tests, and does **not** confirm, the hypothesis that the growing
discrepancy toward small |m| (0.05%→1.01%) was a branch-selection artifact
of the discovery grid. The discovery grid had already found the correct
Root-B branch in this region; the growing discrepancy is a genuine feature
of this branch relative to experiment, not a numerical artifact of how it
was located.

## What is established by this freeze

- Root-B is a numerically robust, reproducible mathematical object:
  continuous under continuation (where sampled), independent of the
  specific root-finding algorithm used to locate it, with residuals at
  the 10⁻¹⁰–10⁻¹⁵ level throughout.
- Root-B matches experimental q=2 to within 0.01–0.06% for
  |m| ≥ 10, degrading smoothly to ~1% at m=-4. This degradation is now
  known to be a property of the branch itself, not an artifact of
  incomplete branch-following.

## What is NOT established

- **Not full point-by-point continuity across the entire m=-14 to -4
  range.** Several intervals were not independently sampled in the
  permanent record (see table in §1). The smooth, monotonic behavior at
  all recovered points is consistent with — but does not prove — full
  continuity across the unsampled gaps.
- **Not a q=3/q=4 result.** This freeze concerns only the q=2 branch.
- **Not a Kerr mapping, not evidence of universality, not an
  astrophysical claim.** See `PAPER3_VALIDATION_STATUS.md`,
  "Permanent boundaries."

## Claim boundary

> The Root-B q=2 branch of the giant-vortex QNM model is frozen as a
> numerically reproducible mathematical object, confirmed independent of
> the specific continuation and root-finding algorithms used to locate
> it, and its parameter-free correspondence with the experimental q=2
> resonance is confirmed to degrade smoothly (not discontinuously or
> artifactually) from 0.01% at large |m| to ~1% at m=-4. This is a
> statement about the numerical and experimental behavior of this
> specific branch in this specific laboratory system, not a broader
> physical claim.

## Provenance

- Continuation data: `stage5I_29_rootB_continuation_toward_m11.csv`,
  `stage5I_30_rootB_continuation_toward_m4.py` (partial terminal record),
  `stage5I_31_rootB_resume_results.csv`.
- Independent solver reproduction: `stage6_5B4_2_INDEPENDENT_root_reproduction.py`,
  reviewed line-by-line (confirmed: reuses validated physical residual,
  independent Newton implementation, no scipy root/minimize, no
  experimental data used); raw run output confirms 14/14 PASS,
  max |Res|=3.225×10⁻¹¹, max distance=5.467×10⁻¹⁰ Hz.
- Experimental q=2 values: Švančara et al., EXP B (qnms.csv).
- Gaps in the continuation record are due to a session interruption; they
  are reported explicitly above rather than interpolated or omitted.
