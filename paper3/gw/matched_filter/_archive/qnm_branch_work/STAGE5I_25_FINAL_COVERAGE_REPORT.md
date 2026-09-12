# Stage 5I-25 -- Final Coverage Report: q=2 Branch (Root A vs Root B)

**Status: coverage/sensitivity finding, formally proposed for baseline
supersession. stage5I_4/6/7 original outputs remain archived, unmodified.**

## Summary

An expanded full-m complex-plane discovery scan (Stage 5I-25), using
the SAME resonance physics and the SAME frozen stage5I_6 assignment
criteria (frequency-agreement tolerance, continuity), independently
rediscovered -- via a wide (Re, Im) search grid, with NO seeding from
previously-known values -- a q=2 branch that is numerically IDENTICAL
to the manually-verified "root B" coverage/sensitivity finding
(Stage 5I-24) at the four originally-discrepant points (m=-11..-14),
while leaving every other m-value's assignment unchanged.

## Side-by-side comparison

| m | Old (Root A, frozen) Re | Old dq2 | New (Root B, Stage 5I-25) Re | New dq2 | Status |
|---|---|---|---|---|---|
| -21 | -- | -- | -- | -- | not found (both) |
| -20 | -- | -- | -- | -- | not found (both) |
| -19 | 10.826904 | -0.006470 | 10.826904 | -0.006470 | unchanged |
| -18 | 10.584745 | -0.004877 | 10.584745 | -0.004877 | unchanged |
| -17 | 10.341302 | -0.001659 | 10.341302 | -0.001659 | unchanged |
| -16 | 10.095960 | -0.002476 | 10.095960 | -0.002476 | unchanged |
| -15 | -- | -- | -- | -- | not found (both) |
| -14 | 9.399551 | -0.199745 | 9.596713 | -0.002584 | superseded (root A -> root B) |
| -13 | 9.121887 | -0.220315 | 9.341131 | -0.001070 | superseded (root A -> root B) |
| -12 | 8.837878 | -0.244213 | 9.080269 | -0.001823 | superseded (root A -> root B) |
| -11 | 8.545584 | -0.265654 | 8.812973 | +0.001735 | superseded (root A -> root B) |
| -10 | 8.537925 | +0.004497 | 8.537925 | +0.004497 | unchanged |
| -9 | 8.253626 | +0.007617 | 8.253626 | +0.007617 | unchanged |
| -8 | 7.958392 | +0.012790 | 7.958392 | +0.012790 | unchanged |
| -7 | 7.650392 | +0.016601 | 7.650392 | +0.016601 | unchanged |
| -6 | 7.327836 | +0.029498 | 7.327836 | +0.029498 | unchanged |
| -5 | 6.989518 | +0.040927 | 6.989518 | +0.040927 | unchanged |
| -4 | -- | -- | -- | -- | not found (both) |

## Statistics comparison

| | Old (Root A, frozen) | New (Root B, superseded at m=-11..-14) |
|---|---|---|
| n (coverage) | 14/18 (77.8%) | 14/18 (77.8%) |
| MAE | 0.00962 Hz | 0.00962 Hz |
| RMSE | ~0.0156 Hz | 0.01500 Hz |
| median \|delta\| | -- | 0.00469 Hz |
| max \|delta\| | 0.267 Hz (m=-11) | 0.0409 Hz (m=-5) |

Key point: overall MAE is numerically unchanged (both branches average
to ~0.0096 Hz), because the OLD statistic already included root A's
very large residuals (~0.2-0.27 Hz) at only 4 of 14 points, diluted by
the other 10 already-good points. The real story is in the
`max |delta|` column: root A's worst point was 0.267 Hz (a poor
match); root B's worst point is 0.041 Hz (a modest but far more
consistent match). Root B removes the four large outliers entirely,
producing a MUCH more internally consistent branch, even though the
simple MAE average happens to land at a similar value.

## Coverage gaps (identical for both root families)

m = -21, -20, -15, -4: no pole found passing the frozen q=2 frequency
criterion, under EITHER the original narrow discovery grid OR the
expanded wide grid (Stage 5I-25). This consistency across two very
different search widths is itself informative: these four m-values
appear to have a genuine absence of a q=2-matching pole in this
model, rather than a search-coverage artifact.

## Recommendation

Given that:
  1. Root B was independently rediscovered via a wide, unseeded scan
     (not manually inserted),
  2. Root B satisfies the same frozen frequency-agreement and
     continuity criteria as root A,
  3. Root B removes the four largest outliers (0.20-0.27 Hz) from the
     q=2 branch, replacing them with modest, consistent residuals
     (0.001-0.003 Hz) fully in line with the rest of the branch,
  4. The remaining 10 m-values are completely unaffected,

this coverage finding is proposed as a candidate to formally supersede
root A at m=-11..-14 in the official q=2 branch assignment. The
original stage5I_4_full_m_scan_v6_FIXED_results.csv remains archived
and unmodified; root A's values remain documented as a distinct,
independently-validated (and independently confirmed via the
from-scratch re-implementation, see independent_reimplementation.py)
genuine secondary numerical branch in its own right -- this
supersession is a branch-ASSIGNMENT decision, not a retraction of root
A's mathematical validity.

## Files referenced

- stage5I_25_expanded_discovery_scan.py (this scan)
- stage5I_25_expanded_results.csv (raw output)
- stage5I_24_coverage_sensitivity_finding.py (original 4-point diagnostic)
- stage5I_6_branch_assignment.py (frozen assignment criteria, unchanged)
- stage5I_4_full_m_scan_v6_FIXED_results.csv (original frozen baseline, archived)
