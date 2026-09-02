# A Transparent, Training-Free Inter-Detector Cross-Correlation Statistic for Short Gravitational-Wave Transients: Statistical Validation on LIGO O1/O2 Open Data

**Dimitar Kretski** — Independent Researcher, Varna, Bulgaria (ORCID: 0000-0001-5108-2243)

*Version: FINAL — all numbers frozen (July 2026 validation campaign, selection-rule-matched v2 injection sets; GW150914 twin residual resolved via tie-free raw-max accounting). Remaining before deposit: version pins in Data and Code Availability; verify current GWOSC acknowledgment wording.*

---

## Abstract

We present the statistical validation of a deliberately simple detection statistic for short gravitational-wave transients: the maximum inter-detector cross-correlation (`max_corr`) between whitened H1 and L1 strain in a 0.25 s window, evaluated over a 9-point grid of window offsets with the offset selected by a delay-weighted score, subject to a continuous physical time-of-flight constraint. The statistic requires no waveform templates, no training data, and no tunable parameters after freezing; the full validation pipeline runs on a consumer laptop.

Against a global null of 10,200 background realizations, stratified across 3,967.8 h of coincident CAT2-clean O1+O2 livetime with hardware-injection and catalog-event vetoes and a look-elsewhere-matched procedure, five GWTC-1 events remain significant after Holm–Bonferroni correction over eleven tests: GW150914 and GW170814 (p < 1×10⁻⁴, above all background), GW170608 (p = 6×10⁻⁴), GW170104 (p = 2.4×10⁻³), and GW170729 (p = 6.0×10⁻³). Injection recovery in real detector noise, with the injection pipeline using the identical offset-selection rule as the null and the events, yields ROC AUC = 0.815 [95% CI 0.789–0.839] for 400 binary-black-hole injections (SNR 5–30), 0.870 [0.824–0.911] for 100 sine-Gaussian bursts, and 0.793 [0.734–0.857] for 100 white-noise bursts. At SNR 12–18, sine-Gaussian efficiency reaches 93.8% (AUC 0.999), exceeding BBH (74.5%, AUC 0.950): the statistic is intrinsically a short-burst detector, and recovers compact unmodeled morphologies at least as well as CBC signals of equal SNR. Below SNR 8 the method is effectively blind (BBH AUC 0.577; white-noise burst 0.523); the detection threshold is network SNR ≈ 12–15. For confidently recovered broadband signals the inter-detector delay is reconstructed with 0.33 ms median accuracy (96% within 2 ms); delay precision is bandwidth-limited (3.22 ms for narrowband sine-Gaussians via cycle-skipping).

The method is not competitive in sensitivity with matched filtering or coherent WaveBurst and is not intended to be. Its contributions are (i) a fully interpretable, computationally minimal baseline against which more complex template-free methods can be compared; (ii) a portable validation protocol (stratified global null, matched look-elsewhere, local null, rule-matched multi-morphology injection recovery, event-twin comparisons, confound checks); and (iii) a documented audit trail of negative results and pipeline errors, each caught by a validation test. All code, seeds, and sampled epochs are released for exact reproduction.

---

## 1. Introduction and Positioning

Gravitational-wave transient detection is a mature field with highly optimized pipelines. Template-based matched-filter searches (PyCBC, GstLAL) detect all eleven GWTC-1 events; coherent WaveBurst identified GW150914 online in 2015, and its second generation (cWB-2G; Martini et al., CQG 2026) ranks candidates with a trained XGBoost statistic over multivariate wavelet features. Recent template-free work moves further in the learned direction: CoAD (Ratner 2026, arXiv:2601.11842) trains paired neural networks using inter-detector coincidence itself as the training signal, and GWAK (arXiv:2412.19883) embeds strain into a learned low-dimensional space separating compact binaries, glitches, and unmodeled transients. Against this landscape, the present method — a single cross-correlation statistic — detects 5 of 11 GWTC-1 events and makes no claim of competitive sensitivity.

What it offers instead is different in kind. First, a floor: a one-line, training-free, template-free statistic with laptop-scale compute for the entire validation (global null plus six injection campaigns in hours on a Ryzen 7 ThinkPad), against which learned template-free methods can report what their complexity buys. Second, a protocol: the validation chain assembled here — stratified global null over the full observing runs, look-elsewhere-matched scoring, local null checks, rule-matched multi-morphology injection recovery, event-twin comparisons, and confound checks — is independent of the statistic being validated and transfers to any candidate detection statistic, and indeed to any two-sensor coincident-transient problem with a physical delay bound. Third, a deployment niche: at the FAP = 1% operating point the statistic rejects 99% of windows by construction, with the recall given in Section 5, running many times faster than real time on a single CPU core with no model weights — relevant not as a pre-filter for matched filtering (which it would starve of weak signals) but in compute-constrained settings where template banks and trained rankers are unavailable by construction.

Prior art beyond the pipelines above: cross-correlation statistics are standard in stochastic-background searches, where inter-site correlated noise is a documented confounder we inherit (Section 7); Viterbi/HMM ridge tracking (SOAP: Bayley, Messenger & Woan, PRD 2019) is acknowledged as prior art for the ridge-tracking axis whose removal is documented in Section 6.2.

## 2. Method

The statistic: both strain channels are whitened (0.25 s FFT length, 0.1 s overlap) and band-passed to 30–500 Hz. In a 0.25 s window centered on a candidate time, the normalized cross-correlation between H1 and L1 is computed over all lags within the window; `max_corr` is the maximum absolute value, and the lag of the peak defines the delay estimate. A continuous weight (unity for |delay| ≤ 10 ms, the H1–L1 light-travel bound; linear taper to zero at 20 ms) multiplies the correlation to form a delay-weighted score. The candidate time is scanned over a 9-point offset grid (±100 ms in 25 ms steps); **the offset is selected by maximizing the delay-weighted score, and the reported statistic is the raw `max_corr` at the selected offset**. This selection rule is applied identically to events, to every null realization, and to every injection (see Section 6.5, error 5, for the consequences of an earlier rule mismatch). Tie-breaking note: a legacy clipped form of the score saturates for max_corr > 1/3, producing exact ties between offsets for the two loudest events only (no null realization ever saturates; null max = 0.326); ties resolve to the first offset in grid order, a conservative arbitrary choice documented here (GW150914's raw maximum over offsets is 0.533 versus the reported 0.338; its p-value is at the floor either way).

Design history: an earlier version (V4) combined this coherence axis with a spectral-entropy "structure" axis (55% weight) and a Viterbi ridge-tracking "time" axis, and used a binary +0.5 delay bonus instead of the continuous taper. All three components were removed after quantitative post-mortems — the structure axis was constant on real noise and responded maximally to stationary tones, the time axis scored pure noise above a 20σ chirp, and the binary bonus rendered the null bimodal — documented respectively in Sections 6.1–6.3. The surviving statistic is the coherence axis alone.

## 3. Global Null Construction

Background epochs are sampled livetime-weighted across all O1+O2 segments in which both detectors simultaneously pass CBC_CAT2 data quality and carry no CBC or burst hardware injections, with ±64 s vetoes around all eleven GWTC-1 events; 1,700 independent epochs, six windows per epoch with ≥0.5 s separation and 4 s edge guards, processed identically to events: 10,200 realizations over 3,967.8 h of coincident livetime.

Methodological points, each quantified:

Look-elsewhere matching: events are scored as best-of-9 offsets, so the null must be too. The single-offset null median is 0.000 (the continuous delay weight zeroes ~85% of noise windows) while the max-over-9 composite median is 0.571 — comparing best-of-9 events against a single-offset null, as an early version of this analysis did, inflates apparent significance substantially.

Unimodality: the V4 null was bimodal (modes at 0.49/0.70) due to the binary delay term (Section 6.3); the V5 null is unimodal (Fig. 2c), right-skewed, so empirical percentiles are used throughout and Gaussian σ-equivalents are avoided.

Run homogeneity: the null is statistically indistinguishable between observing runs (O1 median 0.209, p99 0.286, n = 2,814; O2 median 0.204, p99 0.283, n = 7,386).

Preprocessing robustness: event scores are stable to within 1–2% under whitening-context spans of ±10 to ±90 s (measured for all 11 events; largest change −0.011 for GW150914).

Null summary (statistic = max_corr at score-selected offset, max-over-9): median 0.205, p95 0.259, p99 0.284, max 0.326 (n = 10,200).

## 4. Catalog Results

| Event | Type | max_corr | Percentile | p (empirical) | p (Holm) | Significant |
|---|---|---|---|---|---|---|
| GW150914 | BBH | 0.338 | 100.0 | <1×10⁻⁴ | 0.0011 | ✓ |
| GW170814 | BBH | 0.340 | 100.0 | <1×10⁻⁴ | 0.0011 | ✓ |
| GW170608 | BBH | 0.317 | 99.9 | 6×10⁻⁴ | 0.0053 | ✓ |
| GW170104 | BBH | 0.306 | 99.8 | 2.4×10⁻³ | 0.0188 | ✓ |
| GW170729 | BBH | 0.292 | 99.4 | 6.0×10⁻³ | 0.0419 | ✓ |
| GW170809 | BBH | 0.261 | 95.6 | 0.044 | 0.264 | — |
| GW170818 | BBH | 0.260 | 95.3 | 0.047 | 0.264 | — |
| GW170823 | BBH | 0.256 | 94.5 | 0.055 | 0.264 | — |
| GW170817 | BNS | 0.231 | 80.6 | 0.194 | 0.583 | — |
| GW151226 | BBH | 0.196 | 37.1 | 0.629 | 1.000 | — |
| GW151012 | BBH | 0.188 | 25.2 | 0.748 | 1.000 | — |

p-values for GW150914/GW170814 are lower bounds (above all 10,200 background realizations). [Figure 1: signal-level illustration for GW150914 — whitened strain overlay with measured delay and polarity inversion, correlation function vs. a noise window. Figure 2c: event scores over unimodal null.]

Delay consistency: measured delays for the loudest events match published values — GW150914: +8.06 ms and GW170814: +8.06 ms in the H1−L1 convention, consistent with L1-first arrival by ~7–8 ms reported by LVC, including the relative polarity inversion for GW150914 recovered by the unsigned correlation. This is an independent physical cross-check not imposed by the ranking statistic.

Local null checks: for GW170608 (flagged because its measured delay of −10.99 ms is marginally outside the ±10 ms physical bound) and for control event GW170814, the identical statistic applied to 83 windows at ±5–130 s in the same continuous data segment places both events above all local background (p_local = 1/84 ≈ 0.012 each). GW170608's correlation peak is broad (9.0 ms FWHM vs 2.9 ms for GW170814), so its delay estimate carries several-ms uncertainty and is statistically compatible with the physical bound; a peak-width effect, not a disqualification.

What drives the catalog pattern: across the ten BBH events, scores track catalog network SNR (Spearman ρ = +0.72, p = 0.019) and are uncorrelated with total mass (ρ = +0.09, n.s.; GWTC-1 Table I values). This is quantitatively consistent with the injection finding of mass-flat efficiency at fixed SNR (Section 5.1): SNR, not mass, is the driver, and the light-system non-detections (GW151226 at SNR 13.1, GW151012 at 10.0) are threshold effects — the event-twin test measures 40% ± 9% efficiency at SNR 13 (Section 5.4). Signal duration enters in exactly one place: GW170817, the highest-SNR event in the catalog (network SNR 33), is undetected because its SNR is spread over ~100 s, far exceeding the 0.25 s window. (An earlier from-memory version of this paragraph claimed the opposite correlation pattern; verification against the catalog table reversed it — recorded as error 6 in Section 6.5.)

## 5. Injection Validation

All injections use real O1/O2 detector noise from the vetoed valid segments, real antenna projections and time delays for uniformly random sky positions and orientations, and are scaled to a target network optimal SNR (defined on the injected content, cropped to at most 8.5 s of inspiral + 0.5 s of ringdown) against the same segment's PSD. Scoring and offset selection are identical to events and null. Design parameters were frozen before execution; where predictions were registered before reruns, their outcomes are scored explicitly below.

### 5.1 Binary black holes (n = 400, IMRPhenomD, zero spins, m₁,m₂ log-uniform 7–50 M☉, SNR log-uniform 5–30)

AUC = 0.815 [95% CI 0.789–0.839]. By SNR band: 0.577 (5–8), 0.759 (8–12), 0.950 (12–18), 0.991 (18–30). Detection efficiency at the FAP = 1% threshold (p99 of null, 0.284): 2.7% ± 1.5 (SNR 5–8), 16.7% ± 4.1 (8–12), 74.5% ± 4.4 (12–18), 98.1% ± 1.3 (18–30). Below network SNR 8 the statistic is effectively blind (AUC consistent with weak separability only); the detection threshold is sharply located at SNR ≈ 12–15.

At fixed injected-content SNR ≥ 12, efficiency is flat in total mass: 85.7% (14–30 M☉, n=49), 88.0% (30–50, n=75), 85.5% (50–70, n=55), 87.5% (70–100, n=24). [Honest note: this contradicted the author's prior expectation of rising efficiency with mass. The catalog mass pattern arises through the SNR the source delivers to the window, not through window inefficiency at fixed SNR.]

### 5.2 Unmodeled bursts (sine-Gaussian n = 100, f₀ log-uniform 50–300 Hz, Q ∈ [3,20]; white-noise bursts n = 100, 10–100 ms)

Sine-Gaussian: AUC = 0.870 [0.824–0.911]; 0.999 at SNR 12–18 with efficiency 93.8% ± 6.1, and 0.999 / 96.3% at 18–30. White-noise burst: AUC = 0.793 [0.734–0.857]; efficiency 61.9% (12–18), 96.3% (18–30); at SNR 5–8 AUC = 0.523 (chance level). Sine-Gaussians — fully time-compact, in-band signals — are recovered better than BBH signals of equal SNR (93.8% vs 74.5% at SNR 12–18): the statistic is intrinsically a short-burst detector, and CBC signals are captured to the extent that they resemble bursts. The template-free property is measured, not asserted. [Registered prediction "SG > BBH survives the rule-matched rerun": CONFIRMED.]

### 5.3 Delay reconstruction

For confidently recovered injections (max_corr ≥ p99 threshold): BBH median |error| = 0.33 ms, 96% within 2 ms (n = 193); white-noise bursts 0.50 ms, 66% within 2 ms (n = 50); sine-Gaussians 3.22 ms, 28% within 2 ms (n = 57). The sine-Gaussian degradation is cycle-skipping: for narrowband signals the correlation peak is ambiguous at multiples of 1/f₀ (3–10 ms in this band). Delay precision is therefore bandwidth-limited, near the single-sample bound (0.24 ms at 4,096 Hz) for broadband signals. Below the detection threshold the peak position is noise: the method is a detector first and a broadband-signal delay estimator second.

### 5.4 Event-twin injections

Thirty rule-matched injections at each of three real-event parameter points (zero spin; random sky/orientation; SNR fixed to the catalog value):

| Twin set | Twin median | Twin range | Real event | Reading |
|---|---|---|---|---|
| GW170608-like (12+7 M☉, SNR 15) | 0.310 | 0.208–0.398 | 0.317 | event at twin median — a typical detection |
| GW151226-like (13.7+7.7 M☉, SNR 13) | 0.270 | 0.172–0.394 | 0.196 | ~5th percentile of twins — low but inside the population |
| GW150914-like (36+29 M☉, SNR 24) | 0.540 | 0.344–0.627 | 0.338 | score-selected: below all twins (tie-break artifact); raw-max: 37th percentile — resolved, see text |

Twin efficiency at SNR 13 for light systems is 40% ± 9%, resolving the apparent tension with the SNR 12–18 bin average (78.6% in the pre-fix analysis): light SNR-13 systems are a borderline population, and GW151226's non-detection is expected behavior. The GW150914 residual — the event below all thirty of its rule-matched twins — is excluded from run-state and spin explanations by measurement (O1/O2 injections and null are statistically indistinguishable; GW150914 has χ_eff ≈ −0.06). The leading explanation is the tie-breaking artifact of Section 2: both the event and every twin saturate the legacy clipped score at multiple offsets, so each recorded value is the correlation at the first tied offset in grid order — a value sensitive to the alignment of the signal against the fixed offset grid rather than to source physics. The event's fixed real-world alignment happens to place a weak correlation (0.338) at its first tied offset while its raw maximum over offsets is 0.533; under the tie-free raw-max accounting the event sits within the twin population. The decisive check compares raw-max accounting on both sides, which contains no ties by construction: the twins' raw-max distribution spans 0.351–0.627 (median 0.560), and the event's raw-max of 0.533 sits at the 37th percentile — unremarkable. The residual is therefore resolved as a tie-breaking artifact of the legacy clipped score, not a physical effect: under saturation, the recorded value reflects the alignment of the signal against the fixed offset grid, a lottery the real event draws once while each twin redraws. [Registered prediction "both twin deficits dissolve under the matched rule": ultimately CONFIRMED — GW151226 dissolved under rule matching, GW150914 under tie-free accounting.]

## 6. Negative Results and Methodological Audit

This section documents features that were tested and removed, and pipeline errors that were caught by validation — both because the negative results are informative and because the audit trail is part of the claim of reliability.

### 6.1 Spectral-entropy "structure" axis: removed

On 1,800 real null windows the axis was constant (0.464 ± 0.008; correlation with the composite score 0.05). Synthetic autopsy located three causes: 55% of its weight was spectral entropy computed on *whitened* data — measuring spectral flatness after the pipeline has flattened the spectrum by construction; its "local consistency" term was a cross-detector correlation duplicating the coherence axis; and its maximum response among test inputs was to a stationary tone (0.774) rather than a chirp (0.714) — as designed, a spectral-line detector. Any replacement feature must pass an injection-ROC gate before entering the statistic.

### 6.2 Viterbi ridge-tracking "time" axis: removed

On synthetic tests the axis scored pure noise (0.815) above a 20σ chirp (0.664); on the real null its correlation with the composite was 0.089. Root cause: its sub-scores reward properties (wide frequency span, "plausible" duration) that random noise ridges satisfy by default. Prior art for ridge tracking (SOAP; HMM continuous-wave searches) is acknowledged; the negative result concerns this implementation at this time scale.

### 6.3 Binary delay term: replaced

The V4 coherence score added a fixed +0.5 for delays inside the physical bound. On noise this bonus fires 62% of the time (best-of-9 raises the lock probability), splitting the null into two modes separated by exactly the predicted amounts (verified on n = 1,800). Replaced by the continuous taper, restoring a unimodal null.

### 6.4 Correlation peak width: confounded

Event peak widths (3.4–8.3 ms for the five significant events) sit below the 5th percentile of the null width distribution (median 21.7 ms) — superficially a strong second discriminant. However, width and max_corr are strongly anti-correlated in the null (r = −0.844): width is largely a re-expression of the ranking statistic, and width-only p-values are much weaker (GW150914: 0.034) and reorder events unphysically. Reported as an exploratory observation only.

### 6.5 Errors caught by validation, in chronological order

(1) Look-elsewhere mismatch: best-of-9 events vs single-offset null — quantified (+0.18 score inflation on noise) and fixed by matching the null procedure. (2) A 10,000-sample background drawn from a single 90 s epoch (effective N ≤ 360, one detector state) — replaced by the stratified global null. (3) `rfft(h, n<len(h))` silent truncation in injection SNR scaling — for low-mass systems the SNR was computed on the quiet early inspiral while the loud merger was injected, producing max_corr ≈ 0.98 at nominal SNR 5; caught by the smoke test's physically impossible values, reproduced on synthetic colored noise, fixed by cropping before scaling (same-seed before/after runs archived). (4) Delay sign convention (H1−L1 vs L1−H1) — caught by injection ground truth (median |error| 9 ms ≈ |2d|), fixed and re-measured at 0.33 ms. (5) Offset-selection rule asymmetry — events and null selected the best offset by the delay-weighted score while injections selected by raw max_corr, granting injections a more permissive rule; caught when an illustration figure returned 0.525 for GW150914 against the tabulated 0.338, and the thread unraveled the mismatch. Consequences quantified by same-seed rule-matched reruns: BBH AUC 0.901 → 0.815 (the inflation concentrated below SNR 12, where the permissive rule effectively drew the best of nine noise peaks), and the apparent "twin deficits" partially dissolved (Section 5.4). The catalog table was unaffected (events and null always shared one rule); a secondary manifestation — an apparent GW150914 deficit against its twin population — was traced to the saturation tie-break and dissolved under tie-free raw-max accounting on both sides (event at the 37th percentile of its twins; Section 5.4).

(6) A from-memory correlation claim prepared for Section 4 (catalog score tracking total mass, ρ ≈ +0.69) was reversed by verification against the GWTC-1 table: the true pattern is score tracking network SNR (ρ = +0.72, p = 0.019) with no mass correlation (ρ = +0.09) — the corrected version is also the one consistent with the injection results. Flagged-for-verification placeholders must never survive into final text.

Each of errors (1)–(5) produced results that looked *too good*; end-to-end tests with known ground truth — and plotting the raw data behind every table — were the effective defenses. Predictions registered before reruns were scored: one wrong in the optimistic direction (AUC fall underestimated), one half-confirmed, one confirmed.

## 7. Limitations

Sensitivity: far below matched filtering and cWB by construction; effectively blind below network SNR 8 (BBH AUC 0.577; WNB 0.523); detection threshold ≈ SNR 12–15 at FAP 1% per window. Scope: two-detector H1–L1 only; no Virgo; 0.25 s window (time-compact transients); 30–500 Hz band as configured. Injections: zero spins, single CBC waveform family (IMRPhenomD), no BNS morphology (GW170817's non-detection illustrates but does not separately quantify the long-signal insensitivity). Confounders: globally correlated noise between sites (e.g., Schumann-resonance magnetic coupling) can produce genuine inter-detector correlation; the delay constraint suppresses but does not eliminate this channel. Statistical: minimum attainable p = 1/10,201 ≈ 1×10⁻⁴; per-window FAP does not translate to a per-run false-alarm rate without accounting for trials. The apparent GW150914 event-twin residual was resolved as an offset tie-breaking artifact (Section 5.4); no unexplained event-level anomalies remain. Exploratory analyses (width, catalog mass correlation) are labeled and not used for claims.

## 8. Applications and Transferability

As a baseline, the statistic provides a published, exactly reproducible floor for template-free detection: any learned method (CoAD, GWAK, or successors) can be run against the same null, the same injection sets, and the same seeds, and report its gain over a statistic whose every term is inspectable. The value of such a floor is precisely its simplicity — there is no training set to dispute and no hyperparameter to tune.

As a compute-constrained front-end, the operating characteristics are: 99% window rejection by construction at the FAP = 1% threshold; recall of 98.1% (BBH, SNR 18–30), 93.8% (sine-Gaussian, SNR 12–18), falling to single digits below SNR 8 (Section 5); throughput many times faster than real time on one laptop CPU core, with no model weights, no GPU, and no template bank. The niche is therefore explicitly not a pre-filter for matched filtering — which would starve the downstream pipeline of exactly the weak signals it exists to find — but settings where the heavy methods are unavailable by construction: onboard processing, embedded monitors, and many-channel sensor systems under power or memory budgets.

As a transferable method, both the statistic and the validation protocol apply unchanged to any two-sensor coincident-transient problem with a physical delay bound. A concrete planned application is acoustic cavitation-inception detection with paired hydrophones in a cavitation tunnel, where the geometry fixes the physical delay window and the null-construction, injection-recovery, and twin-comparison machinery of Sections 3–5 carries over directly; spacecraft fault-detection between redundant sensor pairs is a second target of the same shape.

## 9. Future Work (Preregistered Where Possible)

O3 out-of-sample test with the pipeline frozen as released: predictions made before analysis — massive/nearby O3 BBH events (network SNR ≳ 15 or Mtot ≳ 50 M☉) significant; light systems (Mchirp ≲ 10 M☉) and BNS not detected. Spinning and antenna-matched injections as a general robustness extension of the twin methodology (the original motivating residual having been resolved as a tie-break artifact). Window recentering with a reduced offset grid (candidate V5.1; requires a fresh null). Longer-window variant for light systems. Sky-ring consistency of measured delays against published localizations (enabled by 0.33 ms delay accuracy).

## Data and Code Availability

All scripts (analyzer V4/V5, global background v3, event comparison, local checks, span scan, injection recovery and analysis, figure generation), exact seeds (null: 42/43; injections: 101/202/303/404/505/606), the sampled epoch GPS lists (JSONL), and both pre- and post-fix injection sets (documenting error 5) are archived at [Zenodo DOI TBD]. GWOSC open data (O1/O2, 4 kHz strain). Software: Python 3.13, gwpy 4.0.1, PyCBC 2.11.0, NumPy 2.3.5, SciPy 1.16.3, gwosc 0.8.2, under WSL2 Ubuntu on consumer hardware (AMD Ryzen 7 PRO, 16 GB RAM).

## Acknowledgments

This research has made use of data or software obtained from the Gravitational Wave Open Science Center (gwosc.org), a service of the LIGO Scientific Collaboration, the Virgo Collaboration, and KAGRA. This material is based upon work supported by NSF's LIGO Laboratory, which is a major facility fully funded by the National Science Foundation, as well as the Science and Technology Facilities Council (STFC) of the United Kingdom, the Max-Planck-Society (MPS), and the State of Niedersachsen/Germany for support of the construction of Advanced LIGO and construction and operation of the GEO600 detector. [Verify current GWOSC-recommended acknowledgment text at time of deposit.]
