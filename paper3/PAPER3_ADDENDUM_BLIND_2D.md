# Paper 3 — Blind 2D Discovery Addendum

## Purpose

This addendum documents an additional blind two-dimensional discovery analysis
performed as a supplementary test of the Λ-model framework.

The analysis is not intended to replace the primary Paper 3 validation
pipeline. It is recorded separately so that the provenance, assumptions,
and interpretation of the additional test remain explicit.

## Scope

The purpose of the blind 2D analysis is to determine whether the spectral
structure investigated in Paper 3 can be recovered from the model equations
without directly imposing the final Λ-dependent functional form as the
fitting result.

The test is therefore treated as a blind structural consistency test.

It is complementary to, rather than a replacement for, the experimental
validation of the model against the real laboratory resonance data.

## Relation to the Paper 3 model

The Paper 3 framework is based on the quartic dispersion relation

\[
\omega(k) = c k \sqrt{1 + \Lambda k^2},
\]

equivalently,

\[
\omega^2 = c^2 k^2 + \Lambda c^2 k^4.
\]

The additional \(\Lambda k^4\) term represents a non-classical dispersive
contribution relative to the classical linear dispersion relation

\[
\omega^2 = c^2 k^2.
\]

This modification changes the wave dynamics and consequently the structure
of the allowed resonant spectrum.

The blind 2D analysis tests whether the corresponding spectral structure
can be recovered without directly hard-coding the final Λ-dependent
functional form into the discovery step.

## Experimental validation context

The Λ-model has also been tested against real laboratory resonance data
from the experimental system considered in Paper 3.

Using independently specified physical parameters of that system, the model
reproduces the experimentally observed fundamental and first-overtone
resonance branches (q=1 and q=2) within the reported numerical tolerances,
without fitting the individual resonance frequencies as free parameters.

This provides experimental validation of the model's spectral prediction
for the tested laboratory system.

The result should be interpreted specifically as validation of the
Λ-deformed dispersive model in that physical system. It does not by itself
establish that Λ is a universal fundamental constant or a new fundamental
interaction.

## Blind 2D discovery

The blind 2D analysis searches the complex-frequency plane for mathematical
poles without using the experimental q=3 or q=4 frequencies as target
values.

At \(m=-14\), the search recovered four distinct mathematical roots:

| Re [Hz] | Im [Hz] | Status |
|---:|---:|---|
| 8.8467330 | -0.0002949 | fundamental branch |
| 9.3995510 | -0.3086180 | previously identified Root A |
| 9.5967129 | -0.1358383 | Root B / q=2 anchor |
| 10.6492263 | -1.4816071 | validated deep pole, unassigned |

The recovered roots satisfy the numerical residual criterion to machine
precision. The blind search therefore independently recovers the previously
identified spectral structure and additionally confirms the existence of a
strongly damped mathematical pole in the scanned region.

The deep pole is not assigned to an experimental overtone solely on the
basis of frequency proximity. Its physical branch identity remains
unresolved.

## Interpretation

The blind analysis provides an additional structural consistency check:
the relevant mathematical spectral structure is recoverable without
directly imposing the final Λ-dependent fitting form.

Together with the real-data laboratory comparison, the results support the
following statement:

> The Λ-model is mathematically well-defined, produces a non-classical
> dispersive spectral structure, and its predicted q=1 and q=2 resonance
> branches are experimentally reproduced in the tested laboratory system.

The blind 2D analysis should not be interpreted as an independent
measurement of a fundamental physical constant.

In particular, the present results do not establish:

- a universal non-zero value of Λ;
- a new fundamental interaction;
- a modification of General Relativity;
- an astrophysical QNM interpretation;
- or a unique physical mapping between the laboratory model parameters and
  Kerr black-hole parameters.

## Higher overtones

The present calculations do not provide an unambiguous correspondence
between the model spectrum and the experimental q=3 or q=4 branches.

The absence of an unambiguously identified q=3/q=4 pole in the present
scanned region is therefore reported as a current model limitation rather
than as proof that such poles are absent from the complete mathematical
spectrum.

Further branch continuation or a wider spectral search may be required to
resolve higher-overtone correspondence.

## Relation to the GW analysis

The blind 2D analysis is complementary to the gravitational-wave
matched-filter work documented elsewhere in this repository.

The GW pipeline tests the Λ-dependent propagation model against
gravitational-wave strain data and includes injection/recovery and
off-source validation.

The blind 2D analysis addresses a different question: whether the relevant
low-dimensional spectral structure can be recovered without directly
hard-coding the final Λ-model form into the discovery step.

These analyses therefore provide different forms of validation and should
not be treated as statistically independent measurements unless their data,
preprocessing, and inference procedures are demonstrated to be independent.

## Current status

**Status:** supplementary analysis

**Claim level:** mathematical spectral structure + experimental consistency
for the tested laboratory system

**Experimental result:** q=1 and q=2 reproduced

**Higher-overtone status:** unresolved beyond q=2

**Fundamental physical interpretation:** not independently established

The present evidence supports experimental validation of the Λ-model's
spectral prediction in the tested laboratory system. It should not be
described as a confirmed detection of a new fundamental interaction.

## Supplementary data: full q=1 and q=2 comparison tables

*Added as a supplement to the summary root list above. All values were
produced by the same parameter-free pipeline (physical parameters taken
directly from the experiment's published flow-parameter table, not fitted
to the resonance frequencies) described in the sections above. No line in
this table was used as a seed or selection criterion for any other line —
each \(m\) was solved independently (q=1) or reached by continuation from
a single validated anchor (q=2 Root-B), never from the experimental value
itself.*

### q=1 (fundamental branch), all 18 tested azimuthal numbers

| m | experimental (Hz) | model (Hz) | Δ (Hz) | rel. error |
|---:|---:|---:|---:|---:|
| -21 | 10.53128 | 10.53419 | +0.00291 | 0.028% |
| -20 | 10.30228 | 10.30555 | +0.00327 | 0.032% |
| -19 | 10.07028 | 10.07388 | +0.00359 | 0.036% |
| -18 | 9.84068 | 9.83859 | -0.00209 | 0.021% |
| -17 | 9.60049 | 9.59904 | -0.00145 | 0.015% |
| -16 | 9.35501 | 9.35448 | -0.00054 | 0.006% |
| -15 | 9.10327 | 9.10404 | +0.00077 | 0.008% |
| -14 | 8.84414 | 8.84673 | +0.00259 | 0.029% |
| -13 | 8.58228 | 8.58139 | -0.00089 | 0.010% |
| -12 | 8.30426 | 8.30663 | +0.00237 | 0.029% |
| -11 | 8.02013 | 8.02078 | +0.00065 | 0.008% |
| -10 | 7.72173 | 7.72181 | +0.00008 | 0.001% |
| -9 | 7.40626 | 7.40719 | +0.00093 | 0.013% |
| -8 | 7.07617 | 7.07368 | -0.00249 | 0.035% |
| -7 | 6.71469 | 6.71708 | +0.00239 | 0.036% |
| -6 | 6.33352 | 6.33164 | -0.00189 | 0.030% |
| -5 | 5.91100 | 5.90921 | -0.00179 | 0.030% |
| -4 | 5.43752 | 5.43740 | -0.00012 | 0.002% |

Mean \(|\text{rel. error}|\) = 0.021%, maximum = 0.036%, across all 18
tested \(m\).

### q=2 (first overtone), Root-B branch

The Root-B q=2 branch was originally validated by continuation from the
\(m=-14\) anchor toward \(m=-14.9055\) (239 points, machine-precision
residuals; see the main Paper 3 QNM-branch documentation). It has now been
extended by continuation in the opposite direction, \(m=-14 \to -11\)
(31 points, step 0.1, seeded only from the immediately preceding accepted
root at each step; maximum residual \(1.5\times10^{-14}\) across the whole
range; no branch jumps or discontinuities observed). This closes the
integer-\(m\) gap at \(m=-11,-12,-13\), which the original discovery grid
had only reached via the weaker "Root A" branch.

| m | experimental (Hz) | model Root-B (Hz) | Δ (Hz) | rel. error | branch |
|---:|---:|---:|---:|---:|---|
| -19 | 10.83337 | 10.82690 | -0.00647 | 0.060% | discovery grid |
| -18 | 10.58962 | 10.58474 | -0.00488 | 0.046% | discovery grid |
| -17 | 10.34296 | 10.34130 | -0.00166 | 0.016% | discovery grid |
| -16 | 10.09844 | 10.09596 | -0.00248 | 0.025% | discovery grid |
| -14 | 9.59930 | 9.59671 | -0.00259 | 0.027% | anchor (both directions) |
| -13 | 9.34220 | 9.34113 | -0.00107 | 0.011% | continuation (this update) |
| -12 | 9.08209 | 9.08027 | -0.00182 | 0.020% | continuation (this update) |
| -11 | 8.81124 | 8.81297 | +0.00174 | 0.020% | continuation (this update) |
| -10 | 8.53343 | 8.53792 | +0.00450 | 0.053% | discovery grid |
| -9 | 8.24601 | 8.25363 | +0.00762 | 0.092% | discovery grid |
| -8 | 7.94560 | 7.95839 | +0.01279 | 0.161% | discovery grid |
| -7 | 7.63379 | 7.65039 | +0.01660 | 0.217% | discovery grid |
| -6 | 7.29834 | 7.32784 | +0.02950 | 0.404% | discovery grid |
| -5 | 6.94859 | 6.98952 | +0.04093 | 0.589% | discovery grid |
| -4 | 6.57004 | 6.63644 | +0.06640 | 1.011% | discovery grid |

At the three points newly reached by continuation (\(m=-11,-12,-13\)), the
Root-B branch improves the match from the previously available Root-A
values (\(\sim\)2.4–3.0% relative error) to 0.011–0.020%, i.e. to the same
precision level as the q=1 branch. Root-B is continuous and machine-precision
validated across the full range \(m=-11\) to \(-19\) (plus the previously
documented extension to \(m\approx-14.9055\)); Root-A values are retained
in the repository as a secondary/diagnostic branch (see the main QNM-branch
status document) and are not used in this table.

The precision gradually degrades toward smaller \(|m|\) (\(m=-4\) to
\(-9\)), from 0.02% to 1.0%. This is a smooth trend, not a discontinuity,
and coincides with the region where Root-B continuation from the \(m=-14\)
anchor has not yet been extended; these points still use the original
discovery-grid values. Extending the Root-B continuation further toward
\(m=-4\) is a natural next step and is expected to reduce this residual
trend, consistent with the pattern already observed at \(m=-11,-12,-13\).

## Reproducibility

The numerical values, plots, root lists, and intermediate outputs associated
with this addendum should be retained in the repository together with the
exact analysis configuration used to obtain them.

Where a result is subsequently updated, the previous result should remain
traceable through the repository history rather than being silently
overwritten.

## Relationship to the main validation status

This document supplements, but does not supersede:

`paper3/PAPER3_VALIDATION_STATUS.md`

The main validation-status document remains the authoritative summary of the
validated Paper 3 results.

## Scientific claim boundary

The strongest defensible conclusion from the combined mathematical,
numerical, blind-discovery, and laboratory evidence is:

> The Λ-deformed dispersion relation introduces a specific non-classical
> dispersive contribution to the wave dynamics. The resulting mathematical
> spectral structure is numerically reproducible, and the model predicts the
> experimentally observed q=1 and q=2 resonance branches of the tested
> laboratory system without fitting those individual frequencies as free
> parameters.

This constitutes experimental validation of the model's spectral prediction
within the tested physical system.

It should not be used alone to claim:

- detection of a universal non-zero Λ;
- confirmation of a new fundamental interaction;
- exclusion of General Relativity;
- an independent gravitational-wave constraint;
- or confirmation that the same Λ parameter governs astrophysical
  black-hole physics.

Such claims require additional independent physical and observational
evidence.

## Provenance note

This addendum is intentionally maintained as a separate record so that
future analyses can distinguish the original Paper 3 validation results
from subsequent exploratory or supplementary tests.

The supplementary data tables above were added in a later update than the
original addendum text; the root-list summary in the "Blind 2D discovery"
section above is the earlier, independently-recorded result and is left
unchanged. The q=2 continuation extending to \(m=-11,-12,-13\)
(stage5I_29, this update) is a distinct analysis step from the blind 2D
discovery (stage5I_28) documented above it — the former is a continuation
search seeded only from the previously validated Root-B anchor, the latter
is an unseeded discovery scan; neither result was used to seed or bias the
other.
