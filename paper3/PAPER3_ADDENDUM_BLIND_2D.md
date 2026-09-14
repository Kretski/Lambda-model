# Paper 3 — Blind 2D Discovery Addendum

## Purpose

This addendum documents an additional blind two-dimensional discovery analysisperformed as a supplementary test of the Λ-model framework.

The analysis is not intended to replace the primary Paper 3 validationpipeline. It is recorded separately so that the provenance, assumptions,and interpretation of the additional test remain explicit.

## Scope

The purpose of the blind 2D analysis is to determine whether the spectralstructure implied by the Paper 3 model equations can be recoverednumerically without using experimentally observed q=3 or q=4 frequenciesas seeds, target values, or selection criteria.

The analysis is therefore blind with respect to the experimentalhigher-overtone frequencies, but it is not model-independent: the searchis performed within the mathematical equations defining the Λ-model.

It is treated as a blind structural consistency test complementary to,rather than a replacement for, experimental validation against reallaboratory data.

## Relation to the Paper 3 model

The Paper 3 framework is based on the quartic dispersion relation

\[\omega(k) = c k \sqrt{1 + \Lambda k^2},\]

equivalently,

\[\omega^2 = c^2 k^2 + \Lambda c^2 k^4.\]

The additional \(\Lambda k^4\) term represents a non-classical dispersivecontribution relative to the classical linear dispersion relation

\[\omega^2 = c^2 k^2.\]

This modification changes the wave dynamics and consequently the structureof the allowed resonant spectrum.

The blind 2D analysis tests whether the corresponding spectral structurecan be recovered without directly hard-coding the final Λ-dependentfunctional form into the discovery step.

## Experimental validation context

The Λ-model has also been tested against real laboratory resonance datafrom the experimental system considered in Paper 3.

Using independently specified physical parameters of that system, the modelreproduces the experimentally observed fundamental and first-overtoneresonance branches (q=1 and q=2) within the reported numerical tolerances,without fitting the individual resonance frequencies as free parameters.

This provides experimental validation of the model's spectral predictionfor the tested laboratory system.

The result should be interpreted specifically as validation of theΛ-deformed dispersive model in that physical system. It does not by itselfestablish that Λ is a universal fundamental constant or a new fundamentalinteraction.

## Blind 2D discovery

The blind 2D analysis searches the complex-frequency plane for mathematicalpoles without using the experimental q=3 or q=4 frequencies as targetvalues.

At \(m=-14\), the search recovered four distinct mathematical roots:

| Re [Hz] | Im [Hz] | Status |
| --- | --- | --- |
| 8.8467330 | -0.0002949 | fundamental branch |
| 9.3995510 | -0.3086180 | previously identified Root A |
| 9.5967129 | -0.1358383 | Root B / q=2 anchor |
| 10.6492263 | -1.4816071 | deep pole, numerically validated (unassigned) |

The recovered roots satisfy the prescribed residual criterion, withnumerical residuals at or below the \(10^{-14}\) level for the reportedroots.

The blind search therefore independently recovers the correspondingmathematical roots within the predefined search region, including thepreviously identified branches, and additionally confirms the existence ofa strongly damped mathematical pole there.

The deep pole is not assigned to an experimental overtone solely on thebasis of frequency proximity. Its physical branch identity remainsunresolved.

## Interpretation

The blind analysis provides an additional structural consistency check:the relevant mathematical spectral structure is recoverable withoutdirectly imposing the final Λ-dependent fitting form.

Together with the real-data laboratory comparison, the results support thefollowing statement:

> The Λ-model is mathematically well-defined, produces a non-classicaldispersive spectral structure, and its predicted q=1 and q=2 resonancebranches are experimentally reproduced in the tested laboratory system.

The blind 2D analysis should not be interpreted as an independentmeasurement of a fundamental physical constant.

In particular, the present results do not establish:

* a universal non-zero value of Λ;
* a new fundamental interaction;
* a modification of General Relativity;
* an astrophysical QNM interpretation;
* or a unique physical mapping between the laboratory model parameters andKerr black-hole parameters.

## Higher overtones

The present calculations do not provide an unambiguous correspondencebetween the model spectrum and the experimental q=3 or q=4 branches.

The absence of an unambiguously identified q=3/q=4 pole in the presentscanned region is therefore reported as a current model limitation ratherthan as proof that such poles are absent from the complete mathematicalspectrum.

Further branch continuation or a wider spectral search may be required toresolve higher-overtone correspondence.

## Relation to the GW analysis

The blind 2D analysis is complementary to the gravitational-wavematched-filter work documented elsewhere in this repository.

The GW pipeline tests the Λ-dependent propagation model againstgravitational-wave strain data and includes injection/recovery andoff-source validation.

The blind 2D analysis addresses a different question: whether the relevantlow-dimensional spectral structure can be recovered without directlyhard-coding the final Λ-model form into the discovery step.

These analyses therefore provide different forms of validation and shouldnot be treated as statistically independent measurements unless their data,preprocessing, and inference procedures are demonstrated to be independent.

## Current status

**Status:** supplementary analysis

**Claim level:** mathematical spectral structure + experimental consistencyfor the tested laboratory system

**Experimental result:** q=1 and q=2 reproduced

**Higher-overtone status:** unresolved beyond q=2

**Fundamental physical interpretation:** not independently established

The present evidence supports experimental validation of the Λ-model'sspectral prediction in the tested laboratory system. It should not bedescribed as a confirmed detection of a new fundamental interaction.

## Supplementary data: full q=1 and q=2 comparison tables

*Added as a supplement to the summary root list above. All values wereproduced by the same non-frequency-fitted pipeline (physical parameterstaken directly from the experiment's published flow-parameter table, withno individual resonance frequencies fitted as free parameters) described inthe sections above. No line in this table was used as a seed or selectioncriterion for any other line — each \(m\) was solved independently (q=1)or reached by continuation from a single validated anchor (q=2 Root-B),never from the experimental value itself.*

### q=1 (fundamental branch), all 18 tested azimuthal numbers

| m   | experimental (Hz) | model (Hz) | Δ (Hz) | rel. error |
| --- | --- | --- | --- | --- |
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
| -9  | 7.40626 | 7.40719 | +0.00093 | 0.013% |
| -8  | 7.07617 | 7.07368 | -0.00249 | 0.035% |
| -7  | 6.71469 | 6.71708 | +0.00239 | 0.036% |
| -6  | 6.33352 | 6.33164 | -0.00189 | 0.030% |
| -5  | 5.91100 | 5.90921 | -0.00179 | 0.030% |
| -4  | 5.43752 | 5.43740 | -0.00012 | 0.002% |

Mean \(|\text{rel. error}|\) = 0.021%, maximum = 0.036%, across all 18tested \(m\).

### q=2 (first overtone), Root-B branch

Root-B has been validated by continuation from the \(m=-14\) anchor toward\(m=-11\) (31 points, step 0.1, seeded only from the immediately precedingaccepted root at each step; maximum residual \(1.5\times10^{-14}\) acrossthis range; no branch jumps or discontinuities observed), and independentlyalong the previously documented extension toward \(m\approx-14.9055\)(239 points; see the main Paper 3 QNM-branch documentation). Additionalinteger-\(m\) Root-B solutions at \(m=-16\) to \(-19\) and \(m=-4\) to\(-10\) were obtained independently in the discovery grid, not bycontinuation from the \(m=-14\) anchor. This closes the integer-\(m\) gapat \(m=-11,-12,-13\), which the discovery grid had only reached via theweaker "Root A" branch.

| m   | experimental (Hz) | model Root-B (Hz) | Δ (Hz) | rel. error | branch |
| --- | --- | --- | --- | --- | --- |
| -19 | 10.83337 | 10.82690 | -0.00647 | 0.060% | discovery grid |
| -18 | 10.58962 | 10.58474 | -0.00488 | 0.046% | discovery grid |
| -17 | 10.34296 | 10.34130 | -0.00166 | 0.016% | discovery grid |
| -16 | 10.09844 | 10.09596 | -0.00248 | 0.025% | discovery grid |
| -14 | 9.59930 | 9.59671 | -0.00259 | 0.027% | anchor (both directions) |
| -13 | 9.34220 | 9.34113 | -0.00107 | 0.011% | continuation (this update) |
| -12 | 9.08209 | 9.08027 | -0.00182 | 0.020% | continuation (this update) |
| -11 | 8.81124 | 8.81297 | +0.00174 | 0.020% | continuation (this update) |
| -10 | 8.53343 | 8.53792 | +0.00450 | 0.053% | discovery grid |
| -9  | 8.24601 | 8.25363 | +0.00762 | 0.092% | discovery grid |
| -8  | 7.94560 | 7.95839 | +0.01279 | 0.161% | discovery grid |
| -7  | 7.63379 | 7.65039 | +0.01660 | 0.217% | discovery grid |
| -6  | 7.29834 | 7.32784 | +0.02950 | 0.404% | discovery grid |
| -5  | 6.94859 | 6.98952 | +0.04093 | 0.589% | discovery grid |
| -4  | 6.57004 | 6.63644 | +0.06640 | 1.011% | discovery grid |

At the three integer-\(m\) points newly reached by continuation(\(m=-11,-12,-13\)), the independently continued Root-B branch reproducesthe experimental q=2 real frequencies with relative deviations of0.011–0.020%. The experimental frequencies were used only for the post-hoccomparison and were not used as continuation seeds, target values, orbranch-selection criteria. This improves on the Root-A values previouslyavailable at these three points (\(\sim\)2.4–3.0% relative error) byroughly two orders of magnitude, bringing them to the same precision levelas the q=1 branch. Root-A values are retained in the repository as asecondary/diagnostic branch (see the main QNM-branch status document) andare not used in this table.

The precision gradually degrades toward smaller \(|m|\) (\(m=-4\) to\(-9\)), from 0.02% to 1.0%. This is a smooth trend, not a discontinuity,and coincides with the region where the discovery-grid values are usedrather than Root-B continuation from the \(m=-14\) anchor. Extending theRoot-B continuation further toward \(m=-4\) is a natural next step and willtest whether the observed residual trend is associated with thediscovery-grid branch selection.

## Reproducibility

The numerical values, plots, root lists, and intermediate outputs associatedwith this addendum should be retained in the repository together with theexact analysis configuration used to obtain them.

Where a result is subsequently updated, the previous result should remaintraceable through the repository history rather than being silentlyoverwritten.

## Relationship to the main validation status

This document supplements, but does not supersede:

`paper3/PAPER3_VALIDATION_STATUS.md`

The main validation-status document remains the authoritative summary of thevalidated Paper 3 results.

## Scientific claim boundary

The strongest defensible conclusion from the combined mathematical,numerical, blind-discovery, and laboratory evidence is:

> The Λ-deformed dispersion relation introduces a specific non-classicaldispersive contribution to the wave dynamics. The resulting mathematicalspectral structure is numerically reproducible, and the model predicts theexperimentally observed q=1 and q=2 resonance branches of the testedlaboratory system without fitting those individual frequencies as freeparameters.

This constitutes experimental validation of the model's spectral predictionwithin the tested physical system.

It should not be used alone to claim:

* detection of a universal non-zero Λ;
* confirmation of a new fundamental interaction;
* exclusion of General Relativity;
* an independent gravitational-wave constraint;
* or confirmation that the same Λ parameter governs astrophysicalblack-hole physics.

Such claims require additional independent physical and observationalevidence.

## Provenance note

This addendum is intentionally maintained as a separate record so thatfuture analyses can distinguish the original Paper 3 validation resultsfrom subsequent exploratory or supplementary tests.

The supplementary data tables above were added in a later update than theoriginal addendum text; the root-list summary in the "Blind 2D discovery"section above is the earlier, independently-recorded result and is leftunchanged in substance (only terminology was tightened for precision — seebelow). The q=2 continuation extending to \(m=-11,-12,-13\) (stage5I_29,this update) is a distinct analysis step from the blind 2D discovery(stage5I_28) documented above it — the former is a continuation searchseeded only from the previously validated Root-B anchor, the latter is anunseeded discovery scan; neither result was used to seed or bias the other.

**Terminology corrections applied in this update:** "blind" is clarifiedto mean blind with respect to experimental higher-overtone targetfrequencies, not model-independent (the search is performed within theΛ-model's own equations); "machine precision" is replaced by the specificresidual threshold actually achieved; the deep pole is now described asnumerically validated but experimentally unassigned, to avoid any readingof "validated" as experimental confirmation; and the q=2 branch-coveragedescription is corrected to specify which integer-\(m\) points were reachedby continuation from the \(m=-14\) anchor versus obtained independentlyfrom the discovery grid.
