# Deposit skeleton — independent MDR pipeline on public GWOSC data

Working title:
**An independent detection-and-inference pipeline for modified
gravitational-wave dispersion, applied to four O1/O2 events**

Notes are for you, not for the reader. Delete them as you write.
Every number below is traceable to a committed log or JSON file; the
provenance is given so you can check each one rather than trust it.

**Naming.** Use `B_alpha` throughout, not `Lambda`. The Lambda of
Papers 1–2 has dimension length² (local Hamiltonian dispersion near a
black hole); the MDR amplitude here is a propagation effect with
different units. Same letter, different quantity — say so once,
explicitly, in Section 2, and then never use the letter again in this
work. Also: do not call this Paper 3. The v4 deposit promises Paper 3
as a first-principles dispersive wave-equation derivation, which this
is not.

---

## 1. Abstract

Notes: state four things and stop — what was built, what it recovered,
what the injections showed, what the result is. Resist any phrasing
that implies a new bound. The strongest true sentence available is
that the pipeline recovers events blind and returns no measurement of
B on them.

Do NOT write: "confirms GR", "new constraint", "improves on", "novel
model". Do write: "independent reimplementation", "unconstrained on
these events", "the origin of the α-dependence is not established".

---

## 2. Introduction and scope

Notes: three paragraphs.

**What this tests.** The Mirshekari–Yunes–Will modified dispersion
relation, E² = p²c² + A_α p^α c^α, in the group-velocity convention of
Ezquiaga et al. (2022) as adopted in GWTC-4.0 §3.1 Eq. (12). State
plainly that the parametrisation is standard and not introduced here.
This is the single most important sentence in the paper for
credibility — a reader who suspects you are claiming a new equation
will stop reading.

**What is new.** An independent implementation of the detection and
inference chain, with self-checks that halt the pipeline when it is
measuring the wrong thing, and an injection campaign that establishes
where it can and cannot measure.

**Convention note.** The Lambda/A₄ relation and the particle- vs
group-velocity distinction (constraints differ by a factor 1−α).
Include this: it is a real trap and showing you navigated it correctly
signals competence.

---

## 3. Pipeline

Notes: describe the two-stage architecture and say why it is two-stage,
not one. The reason is not efficiency — it is that a B-extended
template bank multiplies the trials factor while adding nothing to
detection efficiency for GR-like signals, because the deformation is
largely degenerate with chirp mass. This mirrors LVK practice.

Stage A: sliding window, 64 s blocks, 8 s overlap, 54-template bank
(9 total masses × 6 mass ratios, aligned spin fixed at 0), IMRPhenomD,
20–512 Hz, power χ² veto with 16 bins, reweighted SNR, clustering at
1 s, H1/L1 coincidence on chirp mass within 25% and |Δt| < 15 ms,
100 time slides at 1 s.

Stage B: dispersion profiling on the loudest coincident candidate
only, maximised over t_c and φ_c, scan grid set automatically to one
alias period, 40 off-source segments for the null spread.

**Self-checks worth naming individually** (this is a differentiator):
measured phase exponent verified against α−1 before anything runs;
scan grid derived from the alias period rather than fixed; boundary-hit
and flat-profile detection; sensitivity gate against the Fisher bound.

Frozen configuration: git tag `mdr-baseline-v1`. Say that analysis
parameters were fixed before the multi-event run and that only input
paths changed between events.

---

## 4. Detection validation

Notes: this section is solid and should be stated confidently.

Four events recovered blind from raw strain, no event time supplied:

| Event    | zero-lag | stat  | H1/L1 newsnr | Δt (ms) | M_c H1 / L1 | published M_c | N_bg |
|----------|----------|-------|--------------|---------|-------------|---------------|------|
| GW150914 | 2        | 24.04 | 19.97 / 13.56| 7       | 31.3 / 31.3 | ~30.9         | 65   |
| GW170104 | 2        | 12.75 | 8.80 / 9.23  | 3       | 23.8 / 23.8 | ~25.7         | 43   |
| GW170814 | 2        | 14.44 | 9.22 / 11.12 | 8       | 26.4 / 23.8 | ~27.0         | 51   |
| GW170823 | 1        | 11.33 | 6.71 / 9.13  | 3       | 31.1 / 37.0 | ~39.1         | 51   |

T_bg = 4.73 days (100 time slides) for all four. In no case did a
background coincidence exceed the candidate ranking statistic.

Points to make:
- The background threshold is stable across four independent runs
  (loudest background 7.43–7.75), so the noise distribution of the
  ranking statistic is reproducible.
- Δt is inside the H1–L1 light-travel window in every case; 8 ms for
  GW170814 matches the true delay for that event.
- Noise candidates receive measured FARs of order 10²–10³ yr⁻¹
  (e.g. 232 yr⁻¹ and 1544 yr⁻¹ in the GW170104 and GW170814 runs)
  while the events are censored by the finite background.

**FAR reporting — get this wording right.** The pipeline prints
`FAR = 0.000e+00/yr`. That is a zero-count result against a finite
background, not a literal zero false-alarm rate. Write it as:

> No background coincidence exceeded the candidate ranking statistic
> within the 4.73-day time-slide background. The reported FAR is
> therefore a zero-count finite-background result; the corresponding
> conservative scale is 1/T_bg ≈ 77 yr⁻¹.

Do not express the separation between events and noise candidates as
a ratio of FARs — the event FARs are censored, not measured. State
the two facts separately instead.

**Known limitations to state, not hide.**
- The 54-template bank is coarse. GW170104 is recovered at q = 3
  against a true q ≈ 1.4; GW170823's H1 template misses the chirp mass
  by ~20% (31.1 against ~39.1). A production search needs a geometric
  bank with a stated minimal match.
- GW170823 is the weakest recovery of the four, and the two detectors
  disagree by 19% in chirp mass (31.1 vs 37.0) against a coincidence
  tolerance of MC_TOL = 0.25. At a tolerance of 0.15 this coincidence
  would have been missed. Say so: the recovery of this event depends
  on a configuration choice.
- That weakness propagates consistently into Stage B, where GW170823
  gives the flattest profiles of the whole set (Δρ² between 0.00 and
  0.10 at all five α). Worst template, weakest signal, least
  information — the chain is internally coherent, which is worth one
  sentence.

---

## 5. Waveform-systematics control

Notes: **this is your strongest section.** Give it its own heading and
do not bury it. Most null-result papers show only the final null; you
can show a plausible false signal traced to its source.

The sequence, from the logs:

| Configuration | Δρ² |
|---|---|
| TaylorF2, 20–300 Hz, ±4% mass grid | 15.47 |
| IMRPhenomD, 20–512 Hz, coarse bank (72, 18) | 2.46 |
| IMRPhenomD, 20–512 Hz, correct template (36, 36) | 0.56 |

Three controlled changes to the waveform model, monotonic decrease
each time. State the interpretation directly: the apparent preference
for non-zero B was the deformation absorbing template error, not
dispersion. Note the mechanism — a detector-frame mass grid that
skipped ~71 M_sun forced a wrong mass ratio to match the chirp mass.

This is a causal test, not an observation. Say so.

---

## 6. Injection campaign

Notes: the methodological core. Injections applied in the frequency
domain to the conditioned strain, so they traverse the same code path
as a real signal — same PSD, same template, same profiler. Distinguish
this explicitly from the standalone validation
(`lambda_diagnostic.py`), which used an analytic PSD and synthetic
noise and therefore does not certify the full pipeline.

Diagnostic scan: ±3 alias periods, 247 points. The physical scan in
the science configuration stays at one period.

| α   | slope | local maxima | Δρ² at largest injection |
|-----|-------|--------------|--------------------------|
| 0.0 | 0.94  | 2–3          | +24.88                   |
| 2.5 | 1.00  | 0            | +1.68                    |
| 3.0 | 1.02  | 0            | +2.28                    |
| 3.5 | 0.98  | 0            | +1.67                    |
| 4.0 | 0.98  | 0            | −0.06                    |

Two findings, kept separate:

**Transfer works everywhere.** d(B_hat)/d(B_inj) ≈ 1 at every α. The
injected phase reaches the likelihood at full strength. This rules out
a transfer bug and is a positive result about the implementation.

**Discrimination works only at α = 0.** There the profile resolves 2–3
local maxima separated by the alias period and prefers the true value
by Δρ² = 24.9. For α ≥ 2.5 the profile is monotonic over six alias
periods with Δρ² ≤ 2.3, against ~1 expected from one free parameter in
noise alone.

**Unit slope is not demonstrated sensitivity.** Make this explicit —
it is the distinction a careful referee will look for.

**On the offsets.** The apparent B_hat offsets (−0.39P to −0.68P) are
not a bias: with no local maximum the global maximum is wherever a
monotonic curve was truncated. At α = 0, the one case with structure,
the offset is −0.064P, i.e. consistent with zero.

**Unexplained.** Two mechanisms were tested for the α-dependence and
neither is supported. A t_c/φ_c degeneracy argument predicts an
ordering by retained Fisher information; the ordering is not observed
(α = 4 retains 10.85% and does not measure; α = 0 retains 8.34% and
does; r = +0.28 across five α). State this as a negative result with
the number, not as "not fully explained" — a clear negative is
stronger than a hedge.

---

## 7. Results

Notes: short. Twenty profiles, four events × five α. Every one is flat
or at the grid edge. No interior maximum supporting a measurement of B
was found in any of them.

The exact claim, and nothing beyond it:
**B is unconstrained by these four events at all tested α.**

Not "consistent with GR" — with no measurement there is nothing to be
consistent with. Not an upper limit: when the profile is flat the
spread of off-source B_hat values is set by the scan width, so a limit
derived from it would double if the grid did. The injection campaign
demonstrates this directly. Quote grid scales as diagnostics if at
all, clearly labelled.

Report the four-column status per α: measurement / constraint / grid
diagnostic / non-detection. Keep these four categories distinct
throughout the paper.

---

## 8. Limitations

Notes: write this section generously. It is where the work earns
trust, and every item is already established by your own controls.

- σ_B independent of SNR across a factor 2.9 in SNR
  (d log σ / d log ρ = +0.00 against −1.00 expected). The uncertainty
  is not SNR-limited on these events.
- B identifiable only modulo the alias period, where identifiable at
  all.
- Coarse template bank; no spin dimension explored.
- Recovery of GW170823 depends on the coincidence tolerance
  (MC_TOL = 0.25); a tighter tolerance would have rejected it.
- Background limited to 4.73 days; event FARs are censored, not
  measured.
- Four events; sensitivity ~30× wider than the published GWTC-4.0
  interval when it can measure at all.
- Origin of the α-dependence unresolved.
- Only the loudest coincident candidate per event is profiled.

---

## 9. Data and code availability

Notes: GWOSC strain, event names, file GPS starts. Repository with
tags `mdr-baseline-v1`, `mdr-multievent-v1`, `mdr-injections-v1`.
State that the configuration was frozen at the first tag and that
per-event logs and JSON results are committed, including the negative
Fisher test.

Committing a failed hypothesis is unusual and worth one sentence: the
repository contains the tests that did not work as well as those that
did.

---

## What NOT to include

- W-Twin. It produced no number in this work. If mentioned at all,
  one sentence as planned future work — and note the unresolved issue
  that it does not currently distinguish incomplete waveform
  subtraction from new physics.
- Computational benchmarks. A selective-vs-exhaustive comparison is
  predetermined by the grid size and would be seen as such. If you
  want cost numbers, report wall time, CPU time, peak RAM and
  matched-filter count, with no energy figures and no cross-project
  framing.
- Any claim of advantage over LVK analyses.
- The 1.17e-9 "resolving power" from the first injection run. That
  came from a broken criterion and is superseded.

---

## Suggested order of writing

Section 5 first — it is the clearest and will set the tone. Then 6,
then 4, then 7 and 8. Abstract and introduction last, once you know
what the paper actually says.
