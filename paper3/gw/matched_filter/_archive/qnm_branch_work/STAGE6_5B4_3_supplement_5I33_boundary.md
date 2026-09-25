## Supplement: Stage 5I-33 — fine-resolution boundary confirmation

*This supplement documents a follow-up test on the Root-B continuation
boundary near m=-15, run after the freeze above. It is additive — it
does not reopen or modify the freeze's conclusions about the validated
range (m=-19 to -4), and it does not establish anything new about
q=3/q=4 (a separate, already-closed line of inquiry; see the addendum's
q=3/q=4 closure supplement).*

### What was tested

The known Root-B continuation boundary (previously located near
m≈-14.905 to -14.9055 at coarse step resolution) was re-approached with
a 10x finer step (Δm=0.005 vs. the original Δm=0.1-scale), seeded from
the last confirmed Root-B point and walking toward m=-15.000, the
nearest integer m with experimental data. Same discipline throughout:
seeded only from the immediately preceding accepted root; experimental
q2 value used only in a final post-hoc comparison.

### Result

The continuation reached m=-14.95 (10/20 planned fine steps), then
failed: the next step (m=-14.955) returned no validated pole. The
rejected candidate at that step had a large residual (|Res|≈0.48) —
not a near-miss root, a genuine non-convergence.

| Quantity | Value |
|---|---|
| Continuation validated to | m = -14.95 |
| Target (m = -15) reached | No |
| Finer-step (10x) confirmation of the boundary | Yes — same qualitative failure recurs |
| Validated pole at m = -15 | None |
| Residual at failed continuation step | ~0.48 (large; not a near-miss) |
| Closest validated point vs. experimental q2(m=-15) | Re(f)=9.835547 Hz vs. 9.848900 Hz — 0.136% (post-hoc only) |

### Interpretation — explicit claim boundary

**What this establishes:** reproducible boundary behavior of this
specific continuation procedure, confirmed independently at two step
resolutions (coarse and 10x finer). The continuation genuinely cannot
proceed past m≈-14.95 with the tested seeding strategy.

**What this does NOT establish:** that the underlying mathematical QNM
branch itself terminates at this m — only that this continuation
procedure fails there. A branch termination (fold, saddle-node, or
similar) has not been demonstrated; the earlier deep-pole work (Stage
5I-26/28/32) already showed how sensitive this solver's search is to
the Im(f) seeding region, so a procedural continuation failure and a
genuine mathematical termination remain distinct possibilities that
this test does not separate.

**On the 0.136% proximity:** the closest validated point sits within
0.136% of the experimental q2 value at m=-15. This is a post-hoc
observation, not evidence that the termination is physical or that
Root-B "is" q2 at m=-15 — it was not used to seed, guide, or select
the continuation in any way. It is recorded as a noteworthy but
non-evidentiary proximity, flagged for whoever revisits this boundary
in the future, not as a result in itself.

### Status and next steps

This is recorded as sufficient numerical characterization of the Root-B
continuation boundary for now. No further continuation attempts on this
specific boundary are planned — two independent step-resolution tests
agreeing is treated as adequate documentation of the procedure's limits,
not as license to keep narrowing the step further in search of a
different outcome. Any future revisiting of this boundary should use a
genuinely different method (e.g., a fold/bifurcation-specific numerical
test, or a wide blind 2D discovery scan localized to this m, analogous
to Stage 5I-28) rather than further step-size refinement of the same
continuation procedure.
