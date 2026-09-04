#!/usr/bin/env python3
"""
combine_mdr.py -- combine per-event MDR results into a joint constraint
=======================================================================

Reads events/*_results.json produced by run_events.py and produces:

  1. A per-event table for each alpha.
  2. A combined constraint per alpha.
  3. The sigma_B vs SNR scaling check.

WHY THIS IS NOT A PLAIN INVERSE-VARIANCE AVERAGE
------------------------------------------------
Inverse-variance combination assumes each B_hat is a Gaussian point
estimate. That assumption holds only when the profile has an interior
maximum. When the profile is flat and the maximum sits at the grid
edge (boundary_hit), B_hat is not an estimate of anything -- it is
just whichever end of the alias period the noise happened to favour,
and it takes the same value regardless of the signal. Feeding those
into a weighted mean assigns real weight to a meaningless number and
will bias the combined result toward the grid edges.

So this script splits the events:

  interior maxima  -> inverse-variance combination, quoted as B +/- CI
  boundary hits    -> UNCONSTRAINED. No central value and no upper
                      limit either: when the profile is flat, the
                      spread of the off-source B_hat values is set by
                      the width of the scan grid, so a "limit" derived
                      from it would double if the grid did, without a
                      single datum changing. The injection campaign
                      confirmed this directly.

If an alpha has no interior-maximum events at all, it is reported as
unconstrained. That is an honest result, not a failure.

Usage:
    python3 combine_mdr.py
"""

import glob
import json
import os

import numpy as np

IN_GLOB = "events/*_results.json"
OUT_JSON = "mdr_combined.json"
Z90 = 1.645  # CI90 = Z90 * sigma

# A profile only counts as a measurement if the deformation actually
# improved the fit. Position of the maximum is not enough: a completely
# flat profile can land inside the grid by chance rather than on an
# edge, which made GW170823 (dSNR^2 = 0.00 and 0.01) look like the sole
# "interior" event for alpha = 3.0 and 4.0 and carry 100% of the weight
# in an otherwise unconstrained combination. One free parameter buys
# dSNR^2 ~ 1 from noise alone, so require meaningfully more than that.
DSNR2_MIN = 4.0


def load_all():
    recs = []
    for path in sorted(glob.glob(IN_GLOB)):
        with open(path) as fh:
            d = json.load(fh)
        name = d.get("event", os.path.basename(path).split("_")[0])
        if not d.get("dispersion"):
            print(f"  {name}: no dispersion record, skipped")
            continue
        # the pipeline profiles the loudest candidate only
        recs.append((name, d["dispersion"][0]))
    return recs


def main():
    recs = load_all()
    if not recs:
        print(f"No results found matching {IN_GLOB}")
        print("Run: python3 run_events.py")
        return

    print(f"Loaded {len(recs)} event(s): {[r[0] for r in recs]}\n")

    alphas = sorted({a for _, r in recs for a in r["alphas"]}, key=float)
    combined = {}
    scaling = []

    for a in alphas:
        rows = []
        for name, r in recs:
            e = r["alphas"].get(a)
            if e is None:
                continue
            sigma = e["ci90"] / Z90
            rows.append(dict(
                name=name, B=e["B_hat"], sigma=sigma, z=e["z"],
                snr=e["snr_gr"], edge=bool(e["boundary_hit"]),
                dsnr2=e["snr_best"] ** 2 - e["snr_gr"] ** 2,
            ))
            if a == alphas[-1]:
                scaling.append((name, e["snr_gr"], sigma))

        if not rows:
            continue

        print(f"alpha = {a}")
        print(f"  {'event':<12} {'B_hat':>13} {'sigma':>12} {'z':>7} "
              f"{'SNR':>7} {'dSNR2':>7}  profile")
        for w in rows:
            if w["edge"]:
                tag = "edge"
            elif w["dsnr2"] < DSNR2_MIN:
                tag = "flat"
            else:
                tag = "interior"
            print(f"  {w['name']:<12} {w['B']:>+13.4e} {w['sigma']:>12.4e} "
                  f"{w['z']:>+7.2f} {w['snr']:>7.2f} {w['dsnr2']:>7.2f}  {tag}")

        interior = [w for w in rows
                    if not w["edge"] and w["dsnr2"] >= DSNR2_MIN]
        edges = [w for w in rows
                 if w["edge"] or w["dsnr2"] < DSNR2_MIN]

        entry = dict(n_events=len(rows), n_interior=len(interior))

        if interior:
            wts = np.array([1.0 / w["sigma"] ** 2 for w in interior])
            bs = np.array([w["B"] for w in interior])
            sig_c = 1.0 / np.sqrt(wts.sum())
            b_c = float((bs * wts).sum() / wts.sum())
            ci_c = Z90 * sig_c
            entry.update(B_combined=b_c, sigma_combined=float(sig_c),
                         ci90_combined=float(ci_c))
            print(f"  -> combined ({len(interior)} interior): "
                  f"B = {b_c:+.4e} +/- {ci_c:.4e}  (90% CI)")
            if abs(b_c) > ci_c:
                print("     NOTE: zero is OUTSIDE the combined interval. "
                      "Check for a single dominating event before "
                      "interpreting this.")
            # does one event dominate?
            frac = wts / wts.sum()
            k = int(np.argmax(frac))
            if frac[k] > 0.7:
                print(f"     WARNING: {interior[k]['name']} carries "
                      f"{100*frac[k]:.0f}% of the weight. The combined "
                      f"result is essentially that one event.")
        else:
            print("  -> no interior maxima")

        if edges:
            # These are NOT bounds. When the profile is flat the spread
            # of the off-source B_hat values is set by the width of the
            # scan grid, not by the noise: scanning two alias periods
            # instead of one would double the number without a single
            # datum changing. The injection campaign confirmed this --
            # unit transfer slope at every alpha, but no resolvable
            # maximum except at alpha = 0, and only far above anything
            # the real events contain. Report unconstrained; quote the
            # grid width as context, never as a limit.
            widths = [e["sigma"] * Z90 for e in edges]
            entry["grid_scale_not_a_limit"] = float(min(widths))
            print(f"  -> {len(edges)} flat/edge profiles: B UNCONSTRAINED.")
            print(f"     (grid scale {min(widths):.4e} -- reflects the scan")
            print(f"     width, not the data; do not quote as a bound)")
        print()
        combined[a] = entry

    # ---- sigma_B vs SNR ----------------------------------------------
    if len(scaling) >= 3:
        print("=" * 70)
        print("SCALING CHECK -- sigma_B should go as 1/SNR")
        print("=" * 70)
        snr = np.array([s for _, s, _ in scaling])
        sg = np.array([g for _, _, g in scaling])
        slope, icept = np.polyfit(np.log(snr), np.log(sg), 1)
        print(f"  events: {[n for n, _, _ in scaling]}")
        print(f"  fitted d(log sigma)/d(log SNR) = {slope:+.2f}  "
              f"(expected -1.00)")
        if abs(slope + 1.0) > 0.4:
            print("  >> Departs from 1/SNR. Something other than SNR is")
            print("     setting the error: template mismatch, PSD quality,")
            print("     or the alias period capping the profile.")
        combined["_scaling"] = dict(slope=float(slope),
                                    events=[n for n, _, _ in scaling])
    elif scaling:
        print(f"Scaling check needs 3+ events; have {len(scaling)}.")

    with open(OUT_JSON, "w") as fh:
        json.dump(combined, fh, indent=2)
    print(f"\nwritten: {OUT_JSON}")


if __name__ == "__main__":
    main()
