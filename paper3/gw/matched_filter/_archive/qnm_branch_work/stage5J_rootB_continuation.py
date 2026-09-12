"""
stage5J_rootB_continuation.py
================================
STAGE 5J (Root B) -- branch continuation from the CORRECT q=2 branch
assignment, toward m=-15.

WHY THIS SUPERSEDES stage5J_branch_continuation.py's original run:

The original Stage 5J continuation started from what stage5I_4's
original discovery grid had labeled "q=2" at m=-14
(f_re=9.39955096, f_im=-0.30861804 -- "Root A"). Stage 5I-24/5I-25
(coverage/sensitivity investigation, done independently of this
script) found a SECOND genuine root at each of m=-11..-14 ("Root B"),
confirmed at machine precision (|Res|<1e-8) by BOTH the original
solver and an independent from-scratch reimplementation, which:

  - matches the experimental q=2 frequency far more closely
    (dq2 ~ 0.001-0.003 Hz vs Root A's 0.20-0.27 Hz),
  - is smoothly continuous with the ALREADY-ESTABLISHED q=2 branch at
    m=-10..-19 (Im(f) increases monotonically from -0.059 at m=-19
    through -0.136 at m=-14 through -0.422 at m=-4, with NO
    discontinuity anywhere except the m=-15 gap), whereas Root A's
    Im=-0.309 at m=-14 is a 3x jump inconsistent with that trend.

Root A is not wrong as a root of Res(omega)=0 -- it is a genuine but
almost certainly SEPARATE/secondary resonance, not the physical
continuation of the smooth q=2 family. This script continues the
correct (Root B) branch instead, using the identical method,
tolerances, and pre-registered classification criteria as the
original stage5J_branch_continuation.py, so the two results are
directly comparable.

Root A's continuation result (m=-14 to m~-14.7725,
NUMERICAL_BREAKDOWN + NO_ROOT_AT_TARGET) is NOT retracted -- it remains
a valid, documented exploratory/diagnostic result on a distinct,
independently-validated secondary root. It is simply not the basis for
the Stage 5K observable/ringdown results going forward; those must be
rebuilt on Root B if this script validates cleanly.
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
# Root B starting point (from stage5I_25_expanded_results.csv, m=-14 row
# with passes_freq_criterion=True, dq2=-0.00258, |Res|=8.7e-15)
# ----------------------------------------------------------------------
M_START = -14.0
F_RE_START = 9.596712867697871
F_IM_START = -0.13583831664187473
M_TARGET = -15.0
DM_COARSE = -0.01
DM_FINE = -5.0e-4
FINE_WINDOW_BACK = 0.05

ROOT_XTOL = 1.0e-12
ROOT_MAXFEV = 300
JAC_EPS_LIST = [1.0e-4, 1.0e-5, 1.0e-6]

RES_VALID = 1.0e-8
RES_BREAKDOWN = 1.0e-3
SIGMA_WINDOW = 5
FOLD_SIGMA_FRACTION = 0.10
JAC_AGREE_RATIO = 10.0

DENSE_IM_MIN_OFFSET = -0.20
DENSE_IM_MAX_OFFSET = +0.20
DENSE_IM_STEP = 0.01

OUTPUT_CSV = HERE / "stage5J_rootB_branch_trajectory.csv"
OUTPUT_JSON = HERE / "stage5J_rootB_classification.json"


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
    if len(trajectory) < SIGMA_WINDOW + 1:
        return "INSUFFICIENT_DATA", {"n_steps": len(trajectory)}
    window = trajectory[-SIGMA_WINDOW:]
    prior = trajectory[-2 * SIGMA_WINDOW:-SIGMA_WINDOW] or window

    if any(not row["jac_reliable"] for row in window):
        return "JACOBIAN_UNRELIABLE", {
            "window_jac_reliable": [row["jac_reliable"] for row in window]}

    sig_seq = [row["sigma_min_1e-5"] for row in window]
    monotonic_decreasing = all(sig_seq[i] > sig_seq[i + 1] for i in range(len(sig_seq) - 1))
    prior_median = float(np.median([row["sigma_min_1e-5"] for row in prior]))
    last_val = sig_seq[-1]

    if monotonic_decreasing and (last_val < FOLD_SIGMA_FRACTION * prior_median):
        return "FOLD_CANDIDATE", {"sigma_min_window": sig_seq,
                                   "prior_median": prior_median, "last_val": last_val}
    return "NUMERICAL_BREAKDOWN", {"sigma_min_window": sig_seq,
                                    "prior_median": prior_median,
                                    "monotonic_decreasing": monotonic_decreasing}


def dense_search_at_target(m_target, re_seed, im_center):
    r_sp, omega_lr = find_light_ring(m_target)
    if r_sp is None:
        return None, []
    im_seeds = np.arange(im_center + DENSE_IM_MIN_OFFSET,
                          im_center + DENSE_IM_MAX_OFFSET + 1e-9, DENSE_IM_STEP)
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
    print("STAGE 5J (ROOT B) -- BRANCH CONTINUATION + CLASSIFICATION")
    print("=" * 100)
    print(f"start (Root B): m={M_START}, f_re={F_RE_START}, f_im={F_IM_START}")
    print(f"target: m={M_TARGET}")
    print()

    print("--- coarse phase (dm=%.4f) ---" % DM_COARSE)
    traj_coarse, breakdown_m = continue_branch(M_START, F_RE_START, F_IM_START,
                                                M_TARGET, DM_COARSE)
    for r in traj_coarse:
        print(f"  m={r['m']:+.4f}  f_re={r['f_re']:.6f}  f_im={r['f_im']:.6f}  "
              f"|Res|={r['abs_res']:.3e}  reliable={r['jac_reliable']}")

    full_trajectory = list(traj_coarse)
    verdict, reasoning = "VALID", {}

    if breakdown_m is not None and traj_coarse:
        last_good = traj_coarse[-1]
        fine_start_m = last_good["m"] + (FINE_WINDOW_BACK if DM_COARSE < 0 else -FINE_WINDOW_BACK)
        print()
        print(f"--- fine phase (dm={DM_FINE}), restarting near m={fine_start_m:+.4f} ---")

        reseed_candidates = [r for r in traj_coarse if
                              (r["m"] >= fine_start_m if DM_COARSE < 0 else r["m"] <= fine_start_m)]
        reseed_row = reseed_candidates[-1] if reseed_candidates else traj_coarse[0]

        traj_fine, breakdown_m_fine = continue_branch(
            reseed_row["m"], reseed_row["f_re"], reseed_row["f_im"],
            M_TARGET, DM_FINE)
        for r in traj_fine:
            print(f"  m={r['m']:+.4f}  f_re={r['f_re']:.6f}  f_im={r['f_im']:.6f}  "
                  f"|Res|={r['abs_res']:.3e}  reliable={r['jac_reliable']}")

        full_trajectory = traj_coarse + traj_fine
        if breakdown_m_fine is None:
            # fine phase reached M_TARGET cleanly -- no breakdown at all;
            # do NOT run the breakdown classifier on a healthy tail.
            verdict, reasoning = "VALID", {
                "note": "fine-resolution continuation reached target m "
                         "without breakdown; coarse-phase breakdown was a "
                         "step-size artifact, not a genuine termination"}
        else:
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
