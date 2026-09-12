# Stage 5I -- Deep Branch Termination: Mechanism-Hunting FINAL STATUS

**Frozen as of this session. Mechanism-hunting is CLOSED. No further
tests planned unless new evidence emerges.**

## Summary statement (agreed final wording)

> A distinct smooth numerical resonance branch, independently validated
> across multiple continuous m-values, terminates at a finite
> m_c ~ -14.79415, localized by adaptive continuation/bisection. The
> branch remains locally regular up to the last validated point, with
> no evidence of Jacobian degeneracy, and the light-ring radius remains
> finite and smooth near the endpoint. The secondary saddle-point
> mechanism associated with non-monotonic truncated quartic dispersion
> is not applicable to the Svancara capillary-gravity dispersion, whose
> dispersion function was found to be monotonic over the relevant
> range. The physical origin of the finite-m termination therefore
> remains unidentified.

## What is established (positive results)

1. Branch smoothness (Stage 5I-15): quadratic fits to Re(m) and Im(m)
   across the branch give RMSE ~0.0006 Hz (Re) and ~0.0001 Hz (Im) --
   a genuinely smooth sequence, not scattered points.

2. Independent leave-one-out validation (Stage 5I-16): each of the 5
   branch points (m=-10..-14) independently re-derived from a seed
   built ONLY from the other 4 points (never its own value), converging
   to machine precision (|Res| < 1e-8) and matching the original values
   to < 1e-8 Hz.

3. Endpoint localization: m_c ~ -14.79415, localized via adaptive
   coarse-to-fine bisection with dense multi-seed search at each
   tested m.

4. Independent from-scratch re-implementation (SymPy exact
   derivatives, adaptive quadrature, Levenberg-Marquardt instead of
   hybr): confirms genuine roots at machine precision for the branch
   points, via completely different numerical machinery than the
   original solver. (Side effect: this sensitivity probe also revealed
   a coverage gap in Stage 5I-4's original discovery grid -- see
   stage5I_24_coverage_sensitivity_finding.py -- documented separately,
   NOT part of the mechanism-hunting investigation.)

## What was tested and found NOT to explain the termination (negative results)

| Candidate mechanism | Test | Result |
|---|---|---|
| Jacobian degeneracy / fold-saddle-node | sigma_min of the [Re(Res),Im(Res)]=0 Jacobian near the endpoint | sigma_min does NOT approach zero (e.g. ~1.2e2 at m=-14.78 vs ~2.6e5 at m=-14.79415) -- no evidence of Jacobian singularity |
| Light-ring radius collapse (Patrick & Solidoro 2020, l^2>0 mechanism) | Direct computation of r_sp(m) via find_light_ring(m), m=-10 through -20 | r_sp(m) is smooth and finite throughout, including well past m_c (up to m=-20 tested); no vanishing behavior |
| Secondary saddle-point family from non-monotonic dispersion (Patrick & Solidoro 2020, l^2<0 mechanism) | Analytic: checked monotonicity of F(k)=(gk+gamma*k^3)tanh(h0*k) over k in [0.01, 5000] | F(k) is STRICTLY MONOTONIC (0 sign changes in dF/dk) -- this mechanism requires non-monotonic dispersion (an artifact of Patrick & Solidoro's TRUNCATED quartic toy dispersion, explicitly noted in their Appendix A as "spurious behaviour" absent in the full dispersion relation). Svancara's F(k) is the FULL, untruncated capillary-gravity dispersion, so this mechanism is analytically excluded, not merely undetected. |

## Context: relevant external literature

Patrick & Solidoro, "Quasinormal modes in Lorentz violating black hole
analogues" (arXiv:2007.06671) -- studies QNM spectra of a draining
vortex analog black hole under a quartic dispersion modification
Omega^2 = c^2(k^2 + l^2*k^4), directly analogous in structure to a
Lambda-type quartic correction. Their model identifies TWO distinct
termination mechanisms for QNM branches (critical rotation where
light-ring radius -> 0 for l^2>0; inflection-point crossing enabling a
second saddle-point family for l^2<0). Both were checked against the
Svancara system and found inapplicable (see table above). Note their
model uses a different vortex geometry (draining bathtub vortex with
radial drain D and literal horizon r_h) than Svancara's (pure
rotational vortex + residual solid-body rotation Omega, no drain, no
horizon) -- direct numerical formula transfer (Eq. 4.18) was not
possible; the physical MECHANISMS were tested against Svancara's own,
correctly-derived quantities instead.

## Decision

Mechanism-hunting is frozen here. This is documented as an honest,
disciplined negative result: a robust numerical finite-m termination
with unidentified physical mechanism -- not "new physics," not
overclaimed, not forced into a borrowed explanation that the data
does not support.

## Next step

Return to Stage 5I-25 (expanded full-m discovery scan, root B
coverage question) as the active priority. The m_c endpoint
investigation and all negative mechanism tests remain as a separate,
well-documented section of the results, not blocking further branch-
assignment work.
