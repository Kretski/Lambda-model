# Pre-registration: the q=2 deviation trend in the giant-vortex spectrum

*Written and committed on 2026-09-24, before any data beyond the EXP B
resonance frequencies (arXiv:2502.11209) have been seen. The commit
timestamp of this file is the record that the rules below were fixed in
advance. This document follows "Before claiming a new result" in
`CONTRIBUTING.md`.*

**Current label: Exploratory.**

## 1. What was seen (discovery data, EXP B)

With the flow parameters of Table SI (arXiv:2308.10773) and no fit to the
resonances, the model reproduces the measured q=1 branch and the q=2
(Root-B) branch. The deviation r(m) = f_model − f_exp behaves differently
for the two branches over m = −10 … −4:

| m | r(m), q=2 [Hz] | r(m), q=1 [Hz] |
|---:|---:|---:|
| −10 | +0.00450 | +0.00008 |
| −9 | +0.00762 | +0.00093 |
| −8 | +0.01279 | −0.00249 |
| −7 | +0.01660 | +0.00239 |
| −6 | +0.02950 | −0.00189 |
| −5 | +0.04093 | −0.00179 |
| −4 | +0.06640 | −0.00012 |

Values from `paper3/PAPER3_ADDENDUM_BLIND_2D.md`.

Reference statistics on these data (least-squares slope of r against m):

- q=2 slope: **S₂ = +0.0096 Hz per unit m**
- q=1 slope: **S₁ = −0.0002 Hz per unit m**
- rms of the q=1 deviation over all 18 values of m: **σ₁ = 0.0020 Hz**

For m ≤ −12 the q=2 deviation is negative and small (−1 to −6 mHz); it
changes sign between m = −12 and m = −11.

These data were used to find the pattern, so they cannot confirm it.

## 2. Hypotheses

- **H1 (physics).** The growth of the q=2 deviation toward small |m| comes
  from physics that the model leaves out, so it is a property of the
  system and appears in other experimental series.
- **H0 (dataset).** The pattern is specific to the EXP B series (for
  example a systematic of that run or of its flow-parameter estimates) and
  does not appear in other series.

No physical mechanism is proposed here, so **no magnitude is predicted**.
The test concerns the sign of the trend and its presence or absence only.

## 3. Frozen analysis procedure for any new series

Applied unchanged to any experimental series other than EXP B (different
temperature, flow rate or geometry):

1. Use the flow parameters reported for that series by the experimenters.
   No parameter is tuned to the resonance frequencies.
2. Use the same solver and the same branch identification (Root-B for
   q=2, the fundamental for q=1). Any value of m where the branch cannot be
   located is reported as missing, not replaced.
3. Compute r(m) = f_model − f_exp for q=1 and q=2 over the overlap of the
   available m values with m = −10 … −4.
4. Fit the slopes S₁ and S₂ by least squares.
   - If the experimenters provide frequency uncertainties, use weighted
     least squares with those uncertainties.
   - Otherwise take the per-point uncertainty to be the rms of the q=1
     deviation over all available m in that series.
5. Compute the slope uncertainties σ(S₁) and σ(S₂) from the same fit.

## 4. Decision rules (fixed now)

| Outcome | Condition |
|---|---|
| **Pattern reproduced (supports H1)** | S₂ > 0 with S₂ / σ(S₂) > 3, **and** \|S₁\| / σ(S₁) < 2 |
| **Pattern absent (supports H0)** | \|S₂\| / σ(S₂) < 2 |
| **Inconclusive** | Anything else, including fewer than 4 usable values of m in the range |

The requirement on S₁ is a built-in control: a slope that appears in both
branches would point to a common offset (for example in the flow
parameters), not to physics specific to the overtone.

## 5. What will not be done

- No change to these rules after seeing new data. Any analysis added
  afterwards is reported as exploratory.
- No reuse of EXP B as confirmation.
- No selection among several new series: every series received is
  analysed and reported, whatever its outcome.

## 6. Label changes

- If a new series gives "Pattern reproduced", the label becomes
  **Reproduced**. Two or more independent series with that outcome make it
  **Robust**.
- If it gives "Pattern absent", the label becomes **Not supported**.
- "Inconclusive" leaves the label at **Exploratory**.
