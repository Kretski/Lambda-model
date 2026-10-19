# EXP C model frequencies (timestamped, computed from fitted parameters)

Counter-rotating resonances ω_q of the Smaniotto et al. model (arXiv:2502.11209, Methods, Eqs. 4–17) for EXP C. The frequencies were computed before any EXP C QNM frequencies were received from the authors and are committed with a tag, so the record shows what was computed and when.

## What this is and is not

- These are **model frequencies**, computed by a reimplementation of the authors' WKB resonance condition. They are not measured peak frequencies.
- The flow parameters (C, Ω, h₀) were **fitted by the authors to the EXP C wave spectra** (Fig. S5). They are not independent of the EXP C data.
- No QNM frequency data were used in the computation. The comparison with measured peaks has not yet been done.
- Therefore agreement with the authors' model values tests the **implementation**; agreement with measured peaks would test the **model**. The current record does not test either against data.

## Inputs

- Flow parameters (Table SI of arXiv:2502.11209): C = 583 mm²/s, Ω = 0.10 rad/s, h₀ = 28 mm, γ = 2.25×10⁻⁶ m³/s².
- r_B = 37.3 mm, **assumed unchanged from EXP B. Not yet confirmed by the authors.**
- Parameter uncertainties (C ± 3 mm²/s, Ω ± 0.01 rad/s) are **not propagated**.

## Code

- The frozen EXP B scripts (`stage5I_3_resonance_v2.py`, `stage5I_4_full_m_scan_v6_FIXED.py`) are unchanged. Only the parameter module (`stage5I_1_2_dispersion_radial_p.py`) carries the EXP C constants.
- The EXP B regression gate and the data-derived frequency window of the EXP B scan are replaced by a fixed, model-only window of 2–14 Hz, chosen before the EXP C run. Results outside this window are not reported.
- Run: `python expC_predict.py` (all m = −21 to −4) → `expC_predictions.csv`.

## Known limitations (stated in advance)

- The scan may not find all four poles at every m. Missing poles are reported as missing, not filled in.
- Parameter uncertainties are not propagated.
- This is a reimplementation of the authors' WKB model, not the authors' code.

## Status

Awaiting from the authors: (1) confirmation that the Table S1 EXP C parameters are the ones to use, (2) the list of m and overtones they plan to include, (3) their measured peak frequencies for EXP C. Until then, this folder contains model output only.
