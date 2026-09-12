"""
stage5J_branch_continuation.py
=================================
STAGE 5J -- automated QNM branch continuation + classification.

Consolidates the ad-hoc Stage 5I-7 / 5I-8 continuation scripts into a
reusable tool: given a validated starting pole (m0, f_re0, f_im0),
continues it in m via Newton (hybr) root-tracking, records the full
trajectory (f_re, f_im, |Res|, singular values at three
finite-difference step sizes), auto-refines around any breakdown
point, then classifies the outcome using criteria FIXED BEFORE this
run examines any data -- not an after-the-fact judgment call -- so the
same numbers would be produced by anyone re-running this on any other
branch.

CLASSIFICATION CRITERIA (fixed constants, see below):

  VALID
    reached m_end with every step's |Res| < RES_VALID.

  BREAKDOWN, further split into exactly one of:
    FOLD_CANDIDATE
      sigma_min over the last SIGMA_WINDOW trustworthy steps before
      breakdown is monotonically decreasing, AND the last trustworthy
      value is below FOLD_SIGMA_FRACTION times the median of the
      SIGMA_WINDOW steps before that.
    JACOBIAN_UNRELIABLE
      in the trustworthy-step window immediately before breakdown,
      the three JAC_EPS sigma_min estimates disagreed by more than
      JAC_AGREE_RATIO at any step -- neither FOLD nor NUMERICAL can be
      claimed from this data.
    NUMERICAL_BREAKDOWN
      |Res| exceeded RES_BREAKDOWN without either of the above --
      i.e. the solver lost the root abruptly while the Jacobian
      stayed well-conditioned and non-decaying.

  Independently of the above, NO_ROOT_AT_TARGET is appended if a dense
  multi-seed search (same validated NM->hybr pipeline, safe_complex_res
  + complex_minimize_from_seed + polish_root, all imported unchanged
  from stage5I_4_full_m_scan_v6_FIXED) at the requested target m finds
  zero validated poles.

Writes the full per-step trajectory to CSV plus a JSON classification
summary. No new physics, no new tolerances.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import root

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stage5I_3_resonance_v2 import find_light_ring
from stage5I_4_full_m_scan_v6_FIXED import (
    safe_complex_res,
    complex_minimize_from_seed,
    polish_root,
    MAX_VALIDATED_ABS_RES,
)

# ----------------------------------------------------------------------
# Run configuration -- the specific branch being continued this run
# ----------------------------------------------------------------------
M_START = -14.0
F_RE_START = 9.39955096
F_IM_START = -0.30861804
M_TARGET = -15.0        # where we ultimately want to know: is there a pole?
DM_COARSE = -0.01
DM_FINE = -5.0e-4
FINE_WINDOW_BACK = 0.05  # how far before the coarse breakdown point to
                         # restart the fine scan

ROOT_XTOL = 1.0e-12
ROOT_MAXFEV = 300
JAC_EPS_LIST = [1.0e-4, 1.0e-5, 1.0e-6]

# ----------------------------------------------------------------------
# Classification criteria -- FIXED before inspecting this run's data
# ----------------------------------------------------------------------
RES_VALID = 1.0e-8
RES_BREAKDOWN = 1.0e-3
SIGMA_WINDOW = 5
FOLD_SIGMA_FRACTION = 0.10
JAC_AGREE_RATIO = 10.0

# Dense post-breakdown search settings (same shape as the m=-15 rescan)
DENSE_RE_SEED_OFFSET = 0.0   # relative to last trustworthy f_re
DENSE_IM_MIN_OFFSET = -0.30  # relative to last trustworthy f_im
DENSE_IM_MAX_OFFSET = +0.20
DENSE_IM_STEP = 0.01

OUTPUT_CSV = HERE / "stage5J_branch_trajectory.csv"
OUTPUT_JSON = HERE / "stage5J_classification.json"


def equations(x, m, r_sp, omega_lr):
    f_re, f_im = float(x[0]), float(x[1])
    z = safe_complex_res(f_re, f_im, m, r_sp, omega_lr)
    if z is None:
        return np.array([1.0e3, 1.0e3])
    return np.array([z.real, z.imag])


def numerical_jacobian(x, m, r_sp, omega_lr, eps):
    J = np.zeros((2, 2))
    f0 = equations(x, m, r_sp, omega_lr)
    for j in range(2):
        xp = x.copy()
        xp[j] += eps
        fp = equations(xp, m, r_sp, omega_lr)
        J[:, j] = (fp - f0) / eps
    return J


def sigma_mins_at(x, m, r_sp, omega_lr):
    vals = []
    for eps in JAC_EPS_LIST:
        J = numerical_jacobian(x, m, r_sp, omega_lr, eps)
        try:
            svals = np.linalg.svd(J, compute_uv=False)
            vals.append(float(np.min(svals)))
        except np.linalg.LinAlgError:
            vals.append(np.nan)
    return vals


def continue_branch(m0, f_re0, f_im0, m_end, dm):
    """Newton-continue the root from (f_re0,f_im0) at m0, stepping by dm
    toward m_end. Returns the full trajectory (list of dict) and the
    breakdown m (None if it reached m_end cleanly)."""
    m = float(m0)
    x = np.array([f_re0, f_im0])
    trajectory = []
    breakdown_m = None

    going_down = dm < 0
    while (m >= m_end - 1e-9) if going_down else (m <= m_end + 1e-9):
        r_sp, omega_lr = find_light_ring(m)
        if r_sp is None:
            breakdown_m = m
            break

        try:
            result = root(lambda xx: equations(xx, m, r_sp, omega_lr), x0=x,
                          method="hybr",
                          options={"xtol": ROOT_XTOL, "maxfev": ROOT_MAXFEV})
        except (ArithmeticError, FloatingPointError, OverflowError,
                ValueError, RuntimeError):
            breakdown_m = m
            break

        x_new = np.asarray(result.x, dtype=float)
        z_final = safe_complex_res(x_new[0], x_new[1], m, r_sp, omega_lr)
        abs_res = abs(z_final) if z_final is not None else np.nan
        sig_mins = sigma_mins_at(x_new, m, r_sp, omega_lr)

        finite = [s for s in sig_mins if np.isfinite(s) and s > 0]
        jac_reliable = (len(finite) >= 2 and (max(finite) / min(finite) <= JAC_AGREE_RATIO))

        trajectory.append({
            "m": m, "f_re": x_new[0], "f_im": x_new[1], "abs_res": abs_res,
            "success": bool(result.success),
            "sigma_min_1e-4": sig_mins[0], "sigma_min_1e-5": sig_mins[1],
            "sigma_min_1e-6": sig_mins[2], "jac_reliable": jac_reliable,
        })

        if (not np.isfinite(abs_res)) or (abs_res > RES_BREAKDOWN):
            breakdown_m = m
            break

        x = x_new
        m += dm

    return trajectory, breakdown_m


def classify(trajectory):
    """Apply the fixed criteria to the tail of a trajectory that ended
    in breakdown. Returns (verdict, reasoning_dict)."""
    if len(trajectory) < SIGMA_WINDOW + 1:
        return "INSUFFICIENT_DATA", {"n_steps": len(trajectory)}

    window = trajectory[-SIGMA_WINDOW:]
    prior = trajectory[-2 * SIGMA_WINDOW:-SIGMA_WINDOW] or window

    any_unreliable = any(not row["jac_reliable"] for row in window)
    if any_unreliable:
        return "JACOBIAN_UNRELIABLE", {
            "window_jac_reliable": [row["jac_reliable"] for row in window],
        }

    sig_seq = [row["sigma_min_1e-5"] for row in window]
    monotonic_decreasing = all(sig_seq[i] > sig_seq[i + 1] for i in range(len(sig_seq) - 1))
    prior_median = float(np.median([row["sigma_min_1e-5"] for row in prior]))
    last_val = sig_seq[-1]

    if monotonic_decreasing and (last_val < FOLD_SIGMA_FRACTION * prior_median):
        return "FOLD_CANDIDATE", {
            "sigma_min_window": sig_seq, "prior_median": prior_median,
            "last_val": last_val, "fraction": last_val / prior_median if prior_median else None,
        }

    return "NUMERICAL_BREAKDOWN", {
        "sigma_min_window": sig_seq, "prior_median": prior_median,
        "monotonic_decreasing": monotonic_decreasing,
    }


def dense_search_at_target(m_target, re_seed, im_center):
    r_sp, omega_lr = find_light_ring(m_target)
    if r_sp is None:
        return None, []

    im_seeds = np.arange(im_center + DENSE_IM_MIN_OFFSET,
                          im_center + DENSE_IM_MAX_OFFSET + 1e-9, DENSE_IM_STEP)
    re_seed = re_seed + DENSE_RE_SEED_OFFSET
    validated = []
    for i, im0 in enumerate(im_seeds):
        print(f"    [{i+1}/{len(im_seeds)}] im0={im0:+.3f} ...", flush=True)
        nm = complex_minimize_from_seed(re_seed, float(im0), m_target, r_sp, omega_lr)
        if nm is None:
            print("        NM failed")
            continue
        print(f"        NM: f_re={nm['f_re']:.6f} f_im={nm['f_im']:.6f} |Res|={nm['abs_res']:.3e}")
        polished = polish_root(nm["f_re"], nm["f_im"], m_target, r_sp, omega_lr,
                                re_seed, nm["abs_res"])
        if polished is not None:
            print(f"        hybr: VALIDATED f_re={polished['f_re']:.10f} "
                  f"f_im={polished['f_im']:.10f} |Res|={polished['abs_res']:.3e}")
            validated.append(polished)
        else:
            print("        hybr: REJECTED")
    return len(im_seeds), validated


def write_csv(trajectory, path):
    if not trajectory:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trajectory[0].keys()))
        writer.writeheader()
        for row in trajectory:
            writer.writerow(row)


def main():
    print("=" * 100)
    print("STAGE 5J -- AUTOMATED BRANCH CONTINUATION + CLASSIFICATION")
    print("=" * 100)
    print(f"start: m={M_START}, f_re={F_RE_START}, f_im={F_IM_START}")
    print(f"target: m={M_TARGET}")
    print()

    print("--- coarse phase (dm=%.4f) ---" % DM_COARSE)
    traj_coarse, breakdown_m = continue_branch(M_START, F_RE_START, F_IM_START,
                                                M_TARGET, DM_COARSE)
    for row in traj_coarse:
        print(f"  m={row['m']:+.4f}  f_re={row['f_re']:.6f}  f_im={row['f_im']:.6f}  "
              f"|Res|={row['abs_res']:.3e}  reliable={row['jac_reliable']}")

    full_trajectory = list(traj_coarse)
    verdict = "VALID"
    reasoning = {}

    if breakdown_m is not None and traj_coarse:
        last_good = traj_coarse[-1]
        fine_start_m = min(last_good["m"] - DM_COARSE, last_good["m"] + FINE_WINDOW_BACK * np.sign(DM_COARSE) * -1)
        # restart the fine scan from FINE_WINDOW_BACK before the last good point
        fine_start_m = last_good["m"] + (FINE_WINDOW_BACK if DM_COARSE < 0 else -FINE_WINDOW_BACK)
        print()
        print(f"--- fine phase (dm={DM_FINE}), restarting near m={fine_start_m:+.4f} ---")

        # find nearest earlier coarse point at/after fine_start_m to reseed from
        reseed_candidates = [r for r in traj_coarse if
                              (r["m"] >= fine_start_m if DM_COARSE < 0 else r["m"] <= fine_start_m)]
        reseed_row = reseed_candidates[-1] if reseed_candidates else traj_coarse[0]

        traj_fine, breakdown_m_fine = continue_branch(
            reseed_row["m"], reseed_row["f_re"], reseed_row["f_im"],
            breakdown_m + DM_FINE * 4, DM_FINE)
        for row in traj_fine:
            print(f"  m={row['m']:+.4f}  f_re={row['f_re']:.6f}  f_im={row['f_im']:.6f}  "
                  f"|Res|={row['abs_res']:.3e}  reliable={row['jac_reliable']}")

        full_trajectory = traj_coarse + traj_fine
        classify_source = traj_fine if len(traj_fine) >= SIGMA_WINDOW + 1 else traj_coarse
        verdict, reasoning = classify(classify_source)

    print()
    print("=" * 100)
    print(f"BREAKDOWN CLASSIFICATION: {verdict}")
    print(json.dumps(reasoning, indent=2, default=str))

    last_row = full_trajectory[-1] if full_trajectory else None
    no_root = None
    if last_row is not None and verdict != "VALID":
        print()
        print(f"--- dense search at target m={M_TARGET} ---")
        n_seeds, validated = dense_search_at_target(M_TARGET, last_row["f_re"], last_row["f_im"])
        no_root = (len(validated) == 0)
        print(f"{len(validated)}/{n_seeds if n_seeds else 0} seeds validated at m={M_TARGET}")
        if no_root:
            verdict = verdict + " + NO_ROOT_AT_TARGET"

    write_csv(full_trajectory, OUTPUT_CSV)
    with open(OUTPUT_JSON, "w") as f:
        json.dump({"verdict": verdict, "reasoning": reasoning,
                   "no_root_at_target": no_root}, f, indent=2, default=str)

    print()
    print(f"FINAL VERDICT: {verdict}")
    print(f"trajectory written to {OUTPUT_CSV}")
    print(f"classification written to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
