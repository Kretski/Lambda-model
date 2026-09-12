# Stage 5J–5K: QNM Branch Continuation, Classification, and Observable Recovery

**Status: frozen (Root B adopted as primary working branch).** This document
summarizes a completed, self-contained numerical result. No further QNM
root-hunting or continuation work is planned on this branch unless real
observational data (Section 7) motivates reopening it.

## 1. Two candidate q=2 branches — why this document covers both

The original stage5I_4_full_m_scan_v6_FIXED.py discovery grid found a pole
at each of m=-11..-14 that was labeled "q=2" ("**Root A**"). A separate
coverage/sensitivity investigation (`stage5I_24_coverage_sensitivity_finding.py`,
`stage5I_25_expanded_discovery_scan.py`) later found a SECOND genuine root
at each of those same m-values ("**Root B**"), confirmed at machine
precision (|Res| < 1e-8) by both the original solver and an independent
from-scratch reimplementation. Root B:

- matches the experimental q=2 frequency far more closely
  (dq2 ≈ 0.001–0.003 Hz vs Root A's 0.20–0.27 Hz),
- is smoothly continuous with the rest of the validated q=2 family at
  m=-10..-19 (Im(f) increases monotonically from -0.059 Hz at m=-19 through
  -0.136 Hz at m=-14 through -0.422 Hz at m=-4, no discontinuity anywhere
  except the m=-15 gap), whereas Root A's Im=-0.309 Hz at m=-14 is a ~3×
  discontinuous jump from its neighbors.

Neither root is "wrong" as a solution of Res(ω)=0 — both are genuine,
independently validated roots. But only one can be the physical
continuation of the smooth q=2 family, and Root B is much the stronger
candidate for that role. **Root B is therefore adopted as the primary
working branch for all Stage 5K observable/ringdown results below. Root A
is retained as a documented secondary/diagnostic branch — its own
continuation and observables are NOT deleted, just not used as the primary
result.**

## 2. Headline result (Root B, primary)

> The Root-B q=2 QNM branch is continuously and numerically validated from
> m = −14 to m ≈ −14.9055 — substantially farther than Root A's
> m ≈ −14.7725. Continuation then breaks down abruptly: the residual jumps
> from ~1e-14 to ~1e-3 within a single fine-resolution step, while the
> residual-equations' Jacobian smallest singular value shows no collapse
> toward zero beforehand (it in fact grows slightly, from ~9.96 at m=-14 to
> ~10.4 at the last validated point) — no fold signature. An independent
> 41-seed dense search at m = −15, using the same validated NM→hybr solver
> pipeline, finds 0 validated poles.
>
> This is classified as a **candidate critical QNM branch termination** —
> not a demonstrated saddle-node/fold, and not a proven physical branch
> termination.

## 3. Root A result (retained, secondary/diagnostic)

> The Root-A q=2 branch is continuously validated from m = −14 to
> m ≈ −14.7725, breaking down abruptly there without a fold signature
> (σ_min non-monotonic, no collapse). An independent 51-seed dense search
> at m = −15 finds 0 validated poles.
>
> Root A's own observables (f_R ≈ 9.40→9.61 Hz, τ ≈ 0.52 s, Q ≈ 15–16) and
> noise-robustness results are kept in the repo (`stage5K_1_observables.csv`,
> `stage5K_5_noise_robustness_summary.csv`) as a valid diagnostic result on
> a distinct, machine-precision-validated resonance — just not the
> branch used for the primary observable/ringdown claims.

## 4. Stage 5J — branch continuation and classification (both roots)

**Root B files:** `stage5J_rootB_continuation.py`,
`stage5J_rootB_branch_trajectory.csv`, `stage5J_rootB_classification.json`
**Root A files:** `stage5J_branch_continuation.py`,
`stage5J_branch_trajectory.csv`, `stage5J_classification.json`

**Method (identical for both):** starting from the validated pole at m=−14,
continue via `scipy.optimize.root(method="hybr")`, seeded from the previous
step's root. A coarse pass (dm=−0.01) locates the approximate breakdown
region; a fine pass (dm=−5e-4) refines it — for Root B, the fine pass runs
all the way to the target m (not an artificially short window), so its
classification is based on a genuine breakdown, not a truncated healthy
tail. At each step, the 2×2 finite-difference Jacobian is evaluated at
three independent step sizes (1e-4, 1e-5, 1e-6 Hz); classification is
discarded (`JACOBIAN_UNRELIABLE`) if these disagree by more than 10×.

**Classification criteria (fixed before any run, identical for both roots):**

| Class | Criterion |
|---|---|
| `VALID` | reached target m with every step's \|Res\| < 1e-8 |
| `JACOBIAN_UNRELIABLE` | max/min disagreement across the 3 Jacobian step sizes > 10× in the pre-breakdown window — takes priority over the other two |
| `FOLD_CANDIDATE` | σ_min monotonically decreasing over the last 5 trustworthy steps AND final value < 10% of the prior 5-step median |
| `NUMERICAL_BREAKDOWN` | \|Res\| > 1e-3 without the fold pattern above |
| `NO_ROOT_AT_TARGET` | appended if a dense multi-seed search at the target m finds 0 validated poles |

**Results:** both roots return `NUMERICAL_BREAKDOWN + NO_ROOT_AT_TARGET`,
at different critical m (Root A ≈ −14.7725, Root B ≈ −14.9055). Root B's
σ_min window before breakdown is monotonically *increasing*
(9.96→10.4, not collapsing); Root A's showed a single non-monotonic dip
(123.7→17.4→127.8). Neither satisfies the fold criterion.

## 5. Stage 5K.1 — validated observables

**Root B (primary):** `stage5K_1_observable_extraction_rootB.py`,
`stage5K_1_observables_rootB.csv` — 239 validated points, m=−14.0000 to
−14.9050. f_R rises smoothly from 9.5967 Hz to 9.8243 Hz, τ from 1.1717 s
to 1.3139 s, Q from 35.32 to 40.55.

**Root A (diagnostic):** `stage5K_1_observable_extraction.py`,
`stage5K_1_observables.csv` — validated to m=−14.7725. f_R: 9.3996→9.6108 Hz,
τ: 0.5157→0.5253 s, Q: 15.23→15.86.

Both stop strictly at their own last validated point — no extrapolation
into either breakdown region.

## 6. Stage 5K.3 / 5K.5 — recovery and noise robustness

**Root B files:** `stage5K_3_ringdown_injection_recovery_rootB.py`,
`stage5K_5_noise_robustness_rootB.py`, `stage5K_5_noise_robustness_summary_rootB.csv`
**Root A files:** `stage5K_3_ringdown_injection_recovery.py`,
`stage5K_5_noise_robustness.py`, `stage5K_5_noise_robustness_summary.csv`

Noiseless recovery (5K.3) is exact to floating-point precision for both
roots at all tested points — confirms the recovery method itself is
unbiased, prior to any noise test (tautological sanity check, not a
physics result).

SNR sweep (5K.5, SNR=30/20/10/7, 30 realizations each, identical protocol
for both roots): success rate degrades gracefully with SNR for both
(Root B: ≈0.83–0.90 at SNR=30 down to ≈0.20–0.30 at SNR=7; Root A:
≈0.90–0.93 down to ≈0.13–0.23), with no persistent bias in either f_R or τ
as SNR improves, for either root. Root B's narrower resonance (higher Q)
gives a tighter f_R standard deviation at matched SNR, as expected.
Both candidate resonances are, in principle, numerically recoverable
under noise — this A/B comparison does not by itself favor one root's
physical reality over the other; it shows the recovery methodology is
sound for both.

## 7. What this does NOT establish, and what comes next

This pipeline does **not** establish:
- that m ≈ −14.7725 or m ≈ −14.9055 is a genuine mathematical
  saddle-node/fold,
- that either Root A or Root B is the "true" physical q=2 branch (Root B is
  adopted as primary on smoothness/frequency-match grounds, not proven to
  be uniquely correct),
- that any of this is observable in a real astrophysical or laboratory
  system.

It establishes a reproducible, pre-registered numerical result for two
independently-validated candidate branches, with Root B's ~0.095 gap to
the m=−15 target (vs Root A's ~0.23) making its independent 0/41
dense-search result the tighter of the two negative results.

**Deliberately deferred, not abandoned:** the next scientific step is
observational — real GW ringdown data → QNM parameter estimation →
statistical comparison of Λ-model predictions (Root B's f_R, τ, Q as
primary; Root A's as a secondary check) vs. GR. This is intentionally
sequenced *after* freezing this repo state, so the prediction is fixed
before being compared against data. Only a positive result from that
observational comparison would justify upgrading "candidate critical QNM
branch termination" to a claim of an established physical effect.
