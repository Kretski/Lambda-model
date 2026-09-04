# MDR pipeline — `paper3/gw/matched_filter/`

An independent implementation of the modified-dispersion-relation (MDR)
test for gravitational waves, run blind on public GWOSC strain.

The parametrisation tested here is **not new**. It is the
Mirshekari–Yunes–Will modified dispersion relation,

    E² = p²c² + A_α p^α c^α

in the group-velocity convention of Ezquiaga et al. (2022), as adopted
in GWTC-4.0 §3.1 Eq. (12). What is implemented independently is the
detection and inference chain around it, together with self-checks that
stop the pipeline when it is measuring the wrong thing.

**Note on notation.** The phase amplitude here is written `B_α`, defined
by δΨ(f) = −B_α f^(α−1). It is a propagation effect and is *not* the
Λ of Papers 1–2 in this repository, which has dimension length² and
describes local Hamiltonian dispersion near a black hole. Different
quantity, different units.

---

## Result in one line

Four O1/O2 events are recovered blind from raw strain with a real
time-slide background. **B_α is unconstrained by these four events at
all five tested α.** No interior maximum supporting a measurement was
found in any of the 20 real-data profiles.

This is a non-detection with no bound attached, not a confirmation of
GR: with no measurement there is nothing to be consistent with. See
`deposit_skeleton.md` for what the results do and do not license.

---

## What runs what

| Script | Purpose |
|---|---|
| `mdr_search.py` | The pipeline. Sliding-window search → χ² veto → H1/L1 coincidence → time-slide background → dispersion profiling. Run directly for one event, or via `run_events.py`. |
| `run_events.py` | Runs the pipeline over several events. Overrides **only** input paths and event GPS; every analysis parameter is inherited from `mdr_search.py`, so events stay comparable. |
| `combine_mdr.py` | Combines per-event results. Inverse-variance for profiles with a genuine interior maximum; everything else reported as unconstrained. |
| `inject_recovery.py` | Injects known B_α into real strain and asks whether the profiler finds it. Diagnostic scan over ±3 alias periods. |
| `fisher_alpha.py` | Analytic projection of the MDR phase against the t_c/φ_c subspace. Ran as a candidate explanation for the α-dependence; **the result was negative** and is kept for the record. |
| `lambda_diagnostic.py` | Standalone check of the inference maths with an analytic PSD and synthetic noise. Validates the algorithm, *not* the full pipeline. |
| `lambda_active.py` | Targeted single-event version, superseded by `mdr_search.py` for search work. |

Quick start:

```bash
python mdr_search.py          # SMOKE = True by default: ~5 min sanity run
python run_events.py          # all configured events, hours
python combine_mdr.py         # combine whatever is in events/
```

Strain files are GWOSC 4096 s, 4 kHz HDF5 for H1 and L1. Paths go in
`LOCAL` (single event) or the `EVENTS` table in `run_events.py`.

---

## Tags

| Tag | State it fixes |
|---|---|
| `mdr-baseline-v1` | Frozen configuration + GW150914 result. Thresholds, bank, α grid, background and candidate selection were fixed here and not touched afterwards. |
| `mdr-multievent-v1` | Four events under that frozen configuration, plus the combined analysis and the σ-vs-SNR scaling check. |
| `mdr-injections-v1` | Injection campaign across α, and the negative Fisher projection test. |

Analysis parameters were frozen at the first tag. Later commits change
only input data, reporting, and diagnostics — never the analysis.

---

## What the controls established

**Detection works.** Four events found without being told where to
look, H1/L1 coincidence inside the light-travel window, 43–65
background coincidences per run, loudest background stable at 7.43–7.75
across four independent runs.

**An apparent signal was traced to its source.** Δρ² fell 15.47 → 2.46
→ 0.56 across three controlled improvements to the waveform model. The
early preference for non-zero B was the deformation absorbing template
error, not dispersion. This is a causal test, not an observation.

**Injected phase transfers at full strength.** d(B̂)/d(B_inj) ≈ 1 at
every α, established by injections into real strain through the same
code path a real signal takes.

**But discrimination exists only at α = 0.** There the profile resolves
2–3 local maxima separated by the alias period and prefers the true
value by Δρ² = 24.9. For α ≥ 2.5 the profile is monotonic over six
alias periods with Δρ² ≤ 2.3. Unit slope is not demonstrated
sensitivity — the distinction matters.

**The origin of that α-dependence is not established.** Two mechanisms
were tested and neither is supported. The nuisance-projection argument
predicts an ordering by retained Fisher information that is not
observed: α = 4 retains 10.85% and does not measure, α = 0 retains
8.34% and does (r = +0.28 across five α). Both the test and its
negative result are committed.

---

## Reading the output

Four categories are kept distinct throughout, and should stay distinct
in anything built on this:

- **measurement** — interior maximum, Δρ² above threshold
- **constraint** — an interval derived from a measurement
- **grid diagnostic** — the scan width, which is *not* a bound
- **non-detection** — no candidate at all

When a profile is flat, the spread of off-source B̂ values is set by the
width of the scan grid: a "limit" derived from it would double if the
grid did, without a single datum changing. `combine_mdr.py` therefore
reports these as UNCONSTRAINED and prints the grid scale only as a
labelled diagnostic. Do not quote it as a bound.

Reported FARs of `0.000e+00/yr` are zero-count results against a finite
background, not literal zero false-alarm rates. The conservative scale
is 1/T_bg ≈ 77 yr⁻¹ for T_bg = 4.73 days.

---

## Known limitations

- 54-template bank is coarse; recovered mass ratios are often wrong
  even where chirp mass is close. GW170823's H1 template misses the
  chirp mass by ~20%.
- Recovery of GW170823 depends on the coincidence tolerance
  (`MC_TOL = 0.25`); a tighter tolerance would have rejected it.
- σ_B is independent of SNR across a factor 2.9 in SNR
  (d log σ / d log ρ = +0.00 against −1.00 expected).
- B_α identifiable only modulo the alias period, where identifiable.
- Background limited to 4.73 days; event FARs are censored.
- Only the loudest coincident candidate per event is profiled.
- No spin dimension in the bank.

---

## Not part of this work

W-Twin residual diagnostics and computational benchmarks are not used
here and produced no number in these results. Any future use of W-Twin
needs to resolve first whether it can distinguish incomplete waveform
subtraction from unmodelled physics — at present it cannot, and a
structured residual is the expected outcome of an imperfect template
either way.
