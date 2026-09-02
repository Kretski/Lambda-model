# TriAxis V5 — Validated Cross-Correlation Statistic on LIGO O1/O2 Open Data
## Deposit contents and reproduction guide

**Author:** Dimitar Kretski (ORCID: 0000-0001-5108-2243) · **License:** paper CC-BY-4.0, code MIT, data products CC-BY-4.0

### Paper
- `triaxis_v5_validation.md` — full text (all numbers final; audit trail in Section 6)

### Figures
- `figure_signal_GW150914.png/.pdf` — Fig. 1: what the statistic sees (strain overlay + correlation, event vs noise)
- `figure_validation.png/.pdf` — Fig. 2: ROC / efficiency / null+events / delay accuracy
- `figure_twins.png/.pdf` — Fig. 3: event-twin injections vs real events (100 per event; see Section 5.4)
- `events_vs_null.png` — catalog events over the global null histogram

### Code (Python 3.13; gwpy 4.0.1, PyCBC 2.11.0, NumPy 2.3.5, SciPy 1.16.3, gwosc 0.8.2)
- `triaxis_analyzer_v4.py`, `triaxis_analyzer_v5.py` — the statistic (V5 = coherence-only, continuous delay taper)
- `global_background_v3.py` — stratified global null construction (segments cache, vetoes, matched max-over-9)
- `analyze_events_v5.py`, `compare_events_to_null.py` — catalog scoring and significance table
- `gw170608_local_check.py` — local null test
- `span_scan.py` — whitening-context robustness scan
- `injection_recovery.py` — injections (bbh/sg/wnb, --fix-m1/m2/snr twins; final version: V5-score offset selection + raw-max logging)
- `analyze_injections.py` — ROC/efficiency/delay analysis
- `plot_signal.py`, `make_figures.py` — figure generation

### Data products
- `global_background_v3.jsonl` — 10,200 null realizations (1,700 epochs × 6 windows)
- `segments_cache.json` — GWOSC timeline cache (O1+O2 coincident CAT2, vetoed)
- `events_v5.json` — 11 GWTC-1 event scores
- `injections_v2.jsonl` (400 BBH), `inj_sg_v2.jsonl` (100), `inj_wnb_v2.jsonl` (100) — rule-matched injection sets
- `inj_gw151226_twin_v2.jsonl`, `inj_gw150914_twin_v2.jsonl`, `inj_gw170608_twin_v2.jsonl` (30 each) — original event twins (retained for continuity)
- `inj_gw150914_twin_v3.jsonl` — intermediate twin set with raw-max logging (tie-break resolution, Section 5.4)
- `inj_gw150914_twin_v4.jsonl` — **expanded twin set, n=100, seed=707** (supersedes v2/v3 for analysis)
- `inj_gw151226_twin_v4.jsonl` — **expanded twin set, n=100, seed=808** (supersedes v2 for analysis)
- `inj_gw170608_twin_v4.jsonl` — **expanded twin set, n=100, seed=909** (supersedes v2 for analysis)
- `legacy_prefix/` — pre-fix injection sets (raw-max selection rule), retained as documentation of audit error 5

### Twin set expansion (v4, 2026-08-02)
The original n=30 twin sets (seeds 404/505/606) have been expanded to n=100 per event
(seeds 707/808/909) to provide bootstrap confidence intervals on percentile estimates.
All conclusions from the n=30 analysis are confirmed; key updated figures:

| Event | n=30 result | n=100 result | 95% bootstrap CI |
|---|---|---|---|
| GW170608 raw-max pct | at twin median | 46th percentile | [36th–56th] |
| GW151226 raw-max pct | ~5th percentile | 1st percentile | [0th–3rd] |
| GW150914 raw-max pct | 37th percentile | 47th percentile | [37th–57th] |

GW151226 note: the 1st percentile reflects the event falling in the undetected half of a
borderline population — the twin median (0.285) coincides with the null p99 threshold
(0.284), giving 53% twin efficiency at FAP=1%. One twin (seed 808, index 65) produced
max_corr=0.196, identical to the real event score. The lower percentile estimate is more
precise, not more anomalous. GW150914 raw-max tie-break resolution is strengthened:
47th percentile [37th–57th CI] on n=100 versus 37th on n=30.

### Reproduction
Seeds: null 42/43; injections 101 (bbh), 202 (sg), 303 (wnb); twins v4: 707 (GW150914),
808 (GW151226), 909 (GW170608). All scripts run on a consumer laptop (AMD Ryzen 7 PRO,
16 GB RAM, WSL2 Ubuntu). Order: `global_background_v3.py` → `analyze_events_v5.py` →
`compare_events_to_null.py --stat max_corr` → `injection_recovery.py` (six main sets +
three twin v4 sets) → `analyze_injections.py` → `make_figures.py`.
Data are fetched from GWOSC at runtime (network required); the segments cache avoids
repeated timeline queries. Note: resumed runs are deterministic but not byte-identical to
uninterrupted runs (RNG re-derived from [seed, n_done]).
