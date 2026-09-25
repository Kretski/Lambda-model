# BEC residual test: Steinhauer et al. (2002)

Test of whether the measured excitation spectrum of a trapped вЃёвЃ·Rb BoseвЂ“Einstein
condensate requires an additional kвЃґ term beyond standard Bogoliubov theory.

**Source:** J. Steinhauer, R. Ozeri, N. Katz, N. Davidson,
*Excitation Spectrum of a Bose-Einstein Condensate*, PRL 88, 120407 (2002),
[arXiv:cond-mat/0111438](https://arxiv.org/abs/cond-mat/0111438)

## Result

| Data | N | Hв‚Ђ П‡ВІ/dof | g_extra | 95% interval | empirical p |
|---|---|---|---|---|---|
| all points | 13 | 42.4/13 | +0.031 В± 0.023 | [в€’0.01, +0.07] | 0.20 |
| k в‰Ґ 1.5 ОјmвЃ»В№ | 9 | 10.9/9 | +0.012 В± 0.024 | [в€’0.03, +0.05] | 0.59 |

No statistically significant additional kвЃґ correction is detected. At 95% confidence
the Bogoliubov kвЃґ coefficient is constrained to within about в€’3% вЂ¦ +6% of its standard
value for this condensate. The Hв‚Ђ goodness-of-fit excess is confined to k < 1.5 ОјmвЃ»В№
(phonon regime, near the LDA validity limit) and does not affect g_extra
(see `steinhauer_kmin_scan.csv`).

## What this is вЂ” and what it is not

- **Hв‚Ђ:** LDA Bogoliubov spectrum, Eq. (4) of the paper, with Ој/h = 1.91 В± 0.09 kHz
  (independently measured by the authors) as a Gaussian prior.
- **Hв‚Ѓ:** Hв‚Ђ with the kвЃґ coefficient Д§ВІ/4mВІ в†’ Д§ВІ/4mВІ В· (1 + g_extra).
- g_extra is a **relative correction to the standard quantum-pressure term** of this
  condensate. It is **not** a measurement of, or a constraint on, the fundamental О›
  of the Lambda-model: the BEC quartic term О› = ОѕВІ/4 is standard Bogoliubov physics,
  and no theoretical mapping between the two exists. The `Lambda_extra` value printed
  by the script is unit bookkeeping only.

## Reproduce

```bash
python extract_fig3_from_eps.py arXiv-cond-mat0111438v1.tar.gz
python steinhauer_test_v3.py steinhauer2002_fig3_extracted.csv --kmin-scan
```

Requires numpy, scipy, pandas, matplotlib. Runtime ~1вЂ“2 min (Monte Carlo null tests).

## Data provenance

No machine-readable dataset was published. The 13 measured points were extracted
directly from the vector EPS (`figure3.eps`, OriginLab) in the arXiv source archive вЂ”
not digitized from a raster image.

- Marker centres and 1Пѓ error bars are read from the PostScript drawing commands;
  each panel is calibrated from its own axis ticks.
- Every point appears in three panels (3a, inset, 3b); they agree to ~0.001 kHz.
- The authors' own plotted LDA curve (Ој/h = 1.91 kHz) is reproduced by our Eq. (4)
  implementation to 0.003 kHz вЂ” an independent check of both the calibration and Hв‚Ђ.
- For 4 points the error bar is hidden inside the marker (as stated in the figure
  caption); Пѓ is then set to the marker radius and flagged `sigma_is_upper_bound`.
  This makes П‡ВІ conservative (too small), not too large.

## Statistical checks built into the script

- fits with and without the Ој prior, ОјвЂ“g_extra correlation
- profile likelihood for g_extra
- null test: 2000 pseudo-experiments from Hв‚Ђ в†’ empirical p-value and bias check
- sign-runs test on Hв‚Ђ residuals (detects non-random, smooth-curve-like input)
- low-k cut scan with thresholds fixed in k (not chosen by residuals)

## Note on an earlier input file

An earlier 21-point `steinhauer_fig3a.csv`, produced by an AI assistant as a claimed
вЂњdigitizationвЂќ of Fig. 3a, was **not real data** (uniform k grid, a non-existent point
at 8.8 ОјmвЃ»В№, smooth residuals: 3 sign runs vs ~10 expected). It gave a spurious ~2Пѓ
preference for Hв‚Ѓ. It is not used anywhere and is kept, if at all, only as
`steinhauer_fig3a_FABRICATED_do_not_use.csv` to document what the runs test catches.

## Files

| File | Content |
|---|---|
| `arXiv-cond-mat0111438v1.tar.gz` | original arXiv source (TeX + EPS figures) |
| `extract_fig3_from_eps.py` | vector extraction + consistency checks |
| `steinhauer2002_fig3_extracted.csv` | the 13 measured points (k, f, Пѓ, flags) |
| `steinhauer_test_v3.py` | Hв‚Ђ/Hв‚Ѓ fits, prior, profile, null test, k-cut scan |
| `steinhauer_kmin_scan.csv` | robustness scan table |
| `steinhauer_test_v3.png` | data, pulls, profile О”П‡ВІ, null distribution |
