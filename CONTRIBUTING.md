# Contributing

This repository welcomes contributions, but the bar for any physical
claim is high — see `paper3/PAPER3_VALIDATION_STATUS.md` for the
current honest status before proposing new results. **The single most
useful thing you can do is find a mistake, not add a new claim.**

If you're new here, read the [README status
table](README.md#status-of-results--read-before-citing) first — it
tells you what's established, what's speculative, and what's actively
refused (and why).

## The discipline this project follows (please match it)

Every result in this repo has, at some point, been through:

1. **A dimensional audit.** Before identifying two "Λ"-like
   parameters from different domains, check their SI units by direct
   analysis of the formula they appear in — not by the symbol they
   share. (Example: the near-horizon Λ [m²] and the GW-propagation Λ
   [m³/s] in this repo are dimensionally incompatible and are kept
   explicitly separate for exactly this reason.)
2. **A known-baseline check.** Before fitting Λ to real data, confirm
   there's an established, Λ-free model for that system, and that your
   pipeline correctly rejects pure baseline data as "not Λ" (see
   `test_pure_quartic_is_not_lambda_model` for the pattern).
3. **A synthetic sanity check with a known answer**, run *before*
   touching real data — inject a known Λ, confirm you recover it, confirm
   Λ=0 stays at zero.
4. **Independent re-execution**, not just re-reading the code. If a
   result is claimed "validated," re-run it yourself and check the
   actual printed output before citing it. Several early results in
   this repo's history did not survive this step (see the audit trail
   in the Honesty Statement).

A pull request that skips these steps, even for a small change, will
likely get asked to add them before merging.

## Concrete, scoped tasks (good first contributions)

These are deliberately narrow — each is a few hours of work, not a new
research program.

### 1. Dimensional audit of a new analog-gravity platform
Pick one candidate system not yet covered (exciton-polariton
condensates, optical-fiber event horizons, water-tank surface waves,
linear/planar BEC sonic horizons — see the Discussions "Ideas"
category for the current shortlist). For that system:
- Find the published dispersion relation.
- Derive the SI dimension of its leading correction coefficient by
  direct analysis (not analogy).
- State explicitly whether it matches Λ_NH [m²], Λ_GW [m³/s], neither,
  or needs its own symbol.
- Open a PR adding one row to a new `analog_platforms_audit.md` table.

No code required for this one — just careful algebra and a citation.

### 2. Run the validator on your own (k, ω) data
If you have any measured or simulated dispersion data:
```bash
python paper3/lambda_experimental_validator.py --omega your_omega.csv --k your_k.csv
```
Report the fitted Λ, its significance, and — critically — whether your
system has an independently known "Λ-free" baseline to compare
against. A null result (Λ consistent with zero) is exactly as welcome
as a positive one; this repo has more of the former than the latter so
far, and that's by design, not a problem to fix.

### 3. Independent reimplementation of the n=1 regression test
`paper3/gw/matched_filter/A2_chromatic_shadow_generalized.py --n 1`
should reproduce Paper 2's published photon-ring coefficients
(C_pro=−0.6230, C_ret=+11.495) to several significant figures. Write
your own from-scratch implementation of the same photon-ring
condition (H=0, ∂H/∂r=0 at Λ=0) in a different language/library and
confirm you get the same numbers. This is the single highest-value
independent check available right now — it doesn't require any new
data, just a second pair of eyes on the math.

### 4. Extend the n-family exponent measurement to n=5, 6...
The generalized solver (`A2_chromatic_shadow_generalized.py`) predicts
δb ∝ Λ_n·E^(2n), confirmed numerically for n=1–4. Extending to higher
n is mechanical (same code, larger `--n` argument) but worth doing to
see whether the pattern holds or something breaks numerically at
higher order — useful before anyone tries to fit a real n from
observational data.

### 5. Flag anything in the repo that looks unverified
If you find a claimed result with no runnable script producing it, or
a script whose output doesn't match what's written about it in a
README or status file — please open an issue. This has happened
before in this repo's history and was corrected; it will likely happen
again as the codebase grows, and catching it is genuinely valuable
work, not a minor nitpick.

## What NOT to open a PR for

- New Λ mappings to a physical system without the dimensional audit
  above.
- Claims of a "detected" non-zero Λ based on a single fit without a
  null-control comparison and a stated significance level.
- Code that reproduces a result without also including the
  known-baseline / Λ=0 control that shows the pipeline can correctly
  say "no" when the answer is no.

## Questions, not sure where to start?

Use GitHub Discussions (Q&A category) rather than opening an issue —
issues are for concrete bugs or well-defined tasks; Discussions is for
"is this idea worth pursuing" conversations.
