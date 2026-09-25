# Section 5 — Waveform-systematics control

Expanded notes for the section to write first. Everything here is
traceable to committed logs; provenance given so you can check rather
than trust.

---

## Why this section comes first

Most null-result papers show only the final null. You can show
something rarer: a plausible signal for new physics, followed to its
source, and eliminated. Three controlled changes to the waveform
model, each reducing the apparent effect, monotonically. That is a
causal test, not an observation — say so explicitly, because a referee
will otherwise read it as a coincidence.

It is also the section where a reader decides whether the rest of the
paper is careful. Write it plainly and let the numbers do the work.

---

## The three configurations

| # | Waveform | Band | Template found | Δρ² |
|---|---|---|---|---|
| 1 | TaylorF2 | 20–300 Hz | ±4% grid around (36, 29) | 15.47 |
| 2 | IMRPhenomD | 20–512 Hz | (72.0, 18.0), M_c = 30.0 | 2.46 |
| 3 | IMRPhenomD | 20–512 Hz | (36.0, 36.0), M_c = 31.3 | 0.56 |

Provenance: config 1 from `lambda_active.log` (SNR 18.2870 → 18.7051);
config 2 from `smoke2.log` (19.645 → 19.659 at the 36-template bank;
the 18-template run gave 16.825 → 16.898); config 3 from `smoke3.log`
and `events/GW150914.log`.

Check the exact numbers against the logs before quoting them. The
Δρ² values above are ρ²_max − ρ²(B=0) at α = 4.

---

## What changed at each step, and why it mattered

**Step 1 → 2: waveform model and band.**
TaylorF2 is inspiral-only and terminates near ISCO, f ≈ 4400/M_tot ≈
63 Hz for this system, while GW150914 peaks around 150–250 Hz. The
entire merger–ringdown was unmodelled. Raising F_HIGH from 300 to 512
Hz matters specifically here because the α = 4 phase goes as f³ and is
therefore concentrated exactly where the inspiral-only template has
nothing.

**Step 2 → 3: detector-frame mass grid.**
This is the instructive one and worth a short paragraph of its own.

The template bank was specified in total mass with values ..., 65, 90,
... The source-frame masses of GW150914 are (36, 29), but at z ≈ 0.09
the detector sees roughly (39, 32), i.e. a total mass near 71 — which
the grid skipped. Matched filtering constrains chirp mass tightly and
mass ratio only loosely, so to reach M_c ≈ 30.9 the search was forced
into (72.0, 18.0): correct chirp mass, mass ratio wrong by a factor of
about three. Adding 55, 72, 85 and 100 to the grid let it find
(36.0, 36.0) with M_c = 31.3, and the recovered SNR rose from 16.8 to
19.6, consistent with the published H1 value of about 20.

Point to make: the deformation was compensating for a template
mismatch that the search itself introduced. Nothing about the data
changed between configurations 2 and 3.

---

## The interpretation, stated carefully

The claim you can defend:

> The apparent preference for non-zero B_α in the initial
> configuration was the dispersion parameter absorbing waveform
> modelling error, not a dispersive effect in the data. Each
> improvement to the waveform model reduced it, and it vanished once
> the template matched the signal.

Scale it for the reader: Δρ² = 0.56 with one free parameter is below
what noise alone supplies (~1 for one degree of freedom). Δρ² = 15.47
would be about 3.9σ if taken at face value — worth saying, because it
shows the initial configuration would have produced a publishable-
looking anomaly.

**Do not overclaim the generality.** This demonstrates the mechanism on
one event with one deformation. It does not establish that every
apparent MDR preference is waveform systematics. One sentence to that
effect protects the section.

---

## Connection to the rest of the paper

Two forward links worth making explicit:

- To Section 6: the injection campaign later shows unit transfer of
  the injected phase at every α, which rules out the alternative
  reading of this section — that the deformation was simply not
  reaching the likelihood. Both facts are needed; neither alone is
  enough.
- To Section 8: the mass-grid failure is a limitation of the coarse
  bank, and the same effect is visible in the other three events,
  where recovered mass ratios remain wrong even when chirp mass is
  close (GW170104 at q = 3 against a true q ≈ 1.4; GW170823's H1
  template off by ~20% in chirp mass).

---

## Framing for a referee

This is the well-known fundamental-bias problem in tests of GR: a
beyond-GR parameter absorbing systematic error in the GR waveform.
Naming it connects the work to existing literature and shows you know
the failure mode has a name. Do not claim to have discovered it.

What is yours is the demonstration — a controlled sequence on real
data with the intermediate states committed, rather than an argument
that the effect could occur in principle.

---

## Length and tone

Aim for one page. Table, three short paragraphs on the configurations,
one paragraph of interpretation, one of caveat. Resist adding a figure
until the text is written; if you do add one, a Δρ² profile over the
scan grid for all three configurations on the same axes would be the
one worth having.

Do not use "confirms", "proves", "robustly demonstrates". Do use
"reduced", "vanished", "consistent with", "attributable to".
