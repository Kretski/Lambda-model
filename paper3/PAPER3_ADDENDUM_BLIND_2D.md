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
