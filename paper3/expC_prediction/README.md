# EXP C prediction (pre-registered)

Counter-rotating resonances $\omega_q$ of the Smaniotto et al. model (arXiv:2502.11209, Methods,
Eqs. 4–17) for **EXP C**, computed **before** any EXP C frequencies are received, and committed with
a tag so that the later comparison is a prediction, not a fit.

- Flow parameters (Table SI of arXiv:2502.11209): $C = 583\ {\rm mm^2/s}$, $\Omega = 0.10$ rad/s,
  $h_0 = 28$ mm, $\gamma = 2.25\times10^{-6}\ {\rm m^3/s^2}$; $r_B = 37.3$ mm assumed unchanged from EXP B.
- Machinery: the frozen EXP B scripts (`stage5I_3_resonance_v2.py`, `stage5I_4_full_m_scan_v6_FIXED.py`)
  unchanged; only the parameter module (`stage5I_1_2_dispersion_radial_p.py`) has the EXP C constants.
- Independence: no measured or reference frequencies are used. The EXP B regression gate and the
  data-derived frequency window of the EXP B scan are replaced by a fixed model-only window
  (2–14 Hz).
- Run: `python expC_predict.py` (all m = -21..-4) → `expC_predictions.csv`.
- Known limitations, stated in advance: parameter uncertainties (C ± 3 mm²/s, Ω ± 0.01 rad/s) are
  not propagated; the scan may not find all four poles at every m (missing poles are reported, not
  filled in); this is a reimplementation of the authors' WKB model, so agreement with the authors'
  model values tests the implementation, and agreement with measured peaks tests the model.
