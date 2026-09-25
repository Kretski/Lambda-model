
"""
stage6_5B4_1B_BLIND_rootB_continuation.py
=========================================

BLIND Root-B continuation with durable checkpointing.

Purpose
-------
Test whether Root-B can be followed continuously WITHOUT using any
experimental QNM frequencies for branch selection.

IMPORTANT:
- No experimental q=2 frequencies appear anywhere in this script.
- Root selection is based ONLY on:
    * previous accepted Root-B pole
    * residual
    * jump guards
- Every accepted point is immediately written to an external
  checkpoint file and flushed to disk.
- If the run is interrupted, the next run resumes from the last
  confirmed checkpoint instead of starting again.

This is Test 1:
    BLIND BRANCH SELECTION

It is intentionally separate from experimental comparison.
"""

import sys
import csv
import time
import os
from pathlib import Path
from datetime import datetime

import numpy as np


# ================================================================
# PATH / IMPORTS
# ================================================================

HERE = Path(__file__).resolve().parent

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from stage5I_3_resonance_v2 import find_light_ring
    from stage5I_4_full_m_scan_v6_FIXED import refine_and_polish_candidate
except ImportError as e:
    print(f"[ERROR] Cannot import pipeline modules: {e}")
    sys.exit(1)


# ================================================================
# CONFIGURATION
# ================================================================

# ---- Starting point: last independently confirmed Root-B point ----
START_M = -5.600
START_F_RE = 7.194422146
START_F_IM = -0.334876498

# ---- Blind target ----
TARGET_M = -4.000

# Same continuation step as previous successful run.
DM = 0.100

# ---- Acceptance ----
RES_ACCEPT = 1.0e-8

# Maximum allowed movement from previous root.
MAX_JUMP_RE_HZ = 0.30
MAX_JUMP_IM_HZ = 0.30

# ---- Fallback seeds ----
FALLBACK_RE_OFFSETS = [
    0.0,
    +0.02,
    -0.02,
    +0.05,
    -0.05,
]

FALLBACK_IM_OFFSETS = [
    0.0,
    +0.01,
    -0.01,
    +0.03,
    -0.03,
]

# ---- Early exit ----
# If continuation converges essentially onto the previous branch,
# don't waste time on the other fallback seeds.
EARLY_EXIT_RES = 1.0e-13
EARLY_EXIT_JUMP = 0.01


# ================================================================
# OUTPUT FILES
# ================================================================

# Main durable checkpoint.
CHECKPOINT_CSV = HERE / "stage6_5B4_1B_BLIND_checkpoint.csv"

# Human-readable run log.
RUN_LOG = HERE / "stage6_5B4_1B_BLIND_run.log"

# Final clean output after successful completion.
FINAL_CSV = HERE / "stage6_5B4_1B_BLIND_final.csv"

# Small state file containing the last accepted point.
STATE_FILE = HERE / "stage6_5B4_1B_BLIND_state.txt"


sep = "=" * 78


# ================================================================
# LOGGING
# ================================================================

def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message):
    line = f"[{timestamp()}] {message}"

    print(line, flush=True)

    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


# ================================================================
# DURABLE CSV WRITE
# ================================================================

def initialize_checkpoint():
    """
    Create checkpoint CSV if it does not exist.

    IMPORTANT:
    This happens before the first expensive calculation.
    """

    if CHECKPOINT_CSV.exists():
        return

    with CHECKPOINT_CSV.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "m",
            "f_re_Hz",
            "f_im_Hz",
            "abs_res",
            "jump_re_Hz",
            "jump_im_Hz",
            "elapsed_step_s",
            "status",
        ])

        f.flush()
        os.fsync(f.fileno())


def append_checkpoint(
    m,
    f_re,
    f_im,
    abs_res,
    jump_re,
    jump_im,
    elapsed_step,
    status="PASS",
):
    """
    Append ONE accepted point and force it to disk.
    """

    with CHECKPOINT_CSV.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            timestamp(),
            f"{m:.6f}",
            f"{f_re:.12f}",
            f"{f_im:.12f}",
            f"{abs_res:.6e}",
            f"{jump_re:.12f}",
            f"{jump_im:.12f}",
            f"{elapsed_step:.3f}",
            status,
        ])

        f.flush()
        os.fsync(f.fileno())


def write_state(m, f_re, f_im, abs_res):
    """
    Store the latest accepted state separately.

    This is redundant by design: if CSV recovery is damaged,
    this tiny state file still tells us where the branch stopped.
    """

    tmp = STATE_FILE.with_suffix(".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        f.write(f"m={m:.12f}\n")
        f.write(f"f_re={f_re:.15f}\n")
        f.write(f"f_im={f_im:.15f}\n")
        f.write(f"abs_res={abs_res:.15e}\n")
        f.write(f"timestamp={timestamp()}\n")
        f.flush()
        os.fsync(f.fileno())

    # Atomic replacement.
    tmp.replace(STATE_FILE)


# ================================================================
# RESUME DETECTION
# ================================================================

def load_last_checkpoint():
    """
    Read the last valid PASS row from checkpoint CSV.

    Returns:
        (m, f_re, f_im, abs_res)
    or None if no checkpoint exists.
    """

    if not CHECKPOINT_CSV.exists():
        return None

    last = None

    with CHECKPOINT_CSV.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            try:
                if row["status"] != "PASS":
                    continue

                last = (
                    float(row["m"]),
                    float(row["f_re_Hz"]),
                    float(row["f_im_Hz"]),
                    float(row["abs_res"]),
                )

            except Exception:
                # Ignore incomplete/corrupted final line.
                continue

    return last


# ================================================================
# QNM SOLVER
# ================================================================

def try_seed(m, seed_re, seed_im):

    r = find_light_ring(m)

    if r is None or r[0] is None:
        return None

    r_sp, omega_lr = r

    try:
        pole = refine_and_polish_candidate(
            f0_re=seed_re,
            f0_im_hint=seed_im,
            m=m,
            r_sp=r_sp,
            omega_lr=omega_lr,
            real_axis_abs_res=float("nan"),
            is_complex_seed=True,
        )

    except Exception:
        return None

    if pole is None:
        return None

    try:
        fre = float(pole["f_re"])
        fim = float(pole["f_im"])
        abs_res = float(pole.get("abs_res", np.inf))

    except Exception:
        return None

    if not (
        np.isfinite(fre)
        and np.isfinite(fim)
        and np.isfinite(abs_res)
    ):
        return None

    # Physical QNM damping convention.
    if fim > 1e-9:
        return None

    return {
        "f_re": fre,
        "f_im": fim,
        "abs_res": abs_res,
    }


# ================================================================
# BLIND CONTINUATION
# ================================================================

def continuation_step(m, previous_root):

    seeds = []

    for dre in FALLBACK_RE_OFFSETS:

        for dim in FALLBACK_IM_OFFSETS:

            seed = (
                previous_root[0] + dre,
                previous_root[1] + dim,
            )

            if dre == 0.0 and dim == 0.0:
                seeds.insert(0, seed)
            else:
                seeds.append(seed)

    best = None
    best_score = None

    for seed_index, (sre, sim) in enumerate(seeds):

        pole = try_seed(
            m,
            sre,
            sim,
        )

        if pole is None:
            continue

        if pole["abs_res"] >= RES_ACCEPT:
            continue

        jump_re = abs(
            pole["f_re"] - previous_root[0]
        )

        jump_im = abs(
            pole["f_im"] - previous_root[1]
        )

        if jump_re > MAX_JUMP_RE_HZ:
            continue

        if jump_im > MAX_JUMP_IM_HZ:
            continue

        score = (
            pole["abs_res"]
            + 0.1 * jump_re
            + 0.1 * jump_im
        )

        if (
            best is None
            or score < best_score
        ):
            best = pole
            best_score = score

        # --------------------------------------------------------
        # IMPORTANT:
        # First seed is previous accepted root.
        # If it returns almost exactly to the branch,
        # stop immediately.
        # --------------------------------------------------------

        if (
            best is not None
            and best["abs_res"] < EARLY_EXIT_RES
            and jump_re < EARLY_EXIT_JUMP
            and jump_im < EARLY_EXIT_JUMP
        ):
            break

    if best is None:
        return None, "NO_CONTINUATION_POLE"

    return best, "PASS"


# ================================================================
# MAIN
# ================================================================

def main():

    # ------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------

    initialize_checkpoint()

    log("")
    log(sep)
    log("STAGE 6.5B4-1B — BLIND ROOT-B CONTINUATION")
    log(sep)

    log("BLIND TEST: experimental frequencies are NOT used.")
    log("Branch selection uses ONLY mathematical continuation.")
    log("")

    log(
        f"Configured range: "
        f"m={START_M:.3f} -> m={TARGET_M:.3f}"
    )

    log(f"DM={DM:.3f}")
    log(f"Residual acceptance={RES_ACCEPT:.1e}")
    log(
        f"Jump guard: "
        f"Re={MAX_JUMP_RE_HZ:.3f} Hz, "
        f"Im={MAX_JUMP_IM_HZ:.3f} Hz"
    )

    log("")
    log("Checkpoint:")
    log(f"  {CHECKPOINT_CSV}")

    log("Run log:")
    log(f"  {RUN_LOG}")

    # ------------------------------------------------------------
    # RESUME OR START
    # ------------------------------------------------------------

    checkpoint = load_last_checkpoint()

    if checkpoint is None:

        current_m = START_M
        previous_root = (
            START_F_RE,
            START_F_IM,
        )

        log("")
        log("No previous checkpoint found.")
        log("Starting from confirmed Root-B seed:")
        log(
            f"  m={current_m:+.6f}, "
            f"f={previous_root[0]:+.12f}"
            f"{previous_root[1]:+.12f}i Hz"
        )

        # Write starting point immediately.
        append_checkpoint(
            current_m,
            previous_root[0],
            previous_root[1],
            0.0,
            0.0,
            0.0,
            0.0,
            status="START",
        )

        write_state(
            current_m,
            previous_root[0],
            previous_root[1],
            0.0,
        )

    else:

        current_m, f_re, f_im, abs_res = checkpoint

        previous_root = (
            f_re,
            f_im,
        )

        log("")
        log("RESUMING FROM CHECKPOINT:")
        log(
            f"  m={current_m:+.6f}"
        )
        log(
            f"  f={f_re:+.12f}"
            f"{f_im:+.12f}i Hz"
        )
        log(
            f"  |Res|={abs_res:.6e}"
        )

    # ------------------------------------------------------------
    # DETERMINE NEXT POINT
    # ------------------------------------------------------------

    remaining = TARGET_M - current_m

    if remaining <= 1e-12:

        log("")
        log("Target already reached.")
        log("Nothing to compute.")

        return

    n_steps = int(
        round(
            remaining / DM
        )
    )

    log("")
    log(
        f"Remaining continuation steps: "
        f"{n_steps}"
    )

    # ------------------------------------------------------------
    # CONTINUATION LOOP
    # ------------------------------------------------------------

    chain_broke_at = None

    total_start = time.perf_counter()

    for step in range(
        1,
        n_steps + 1
    ):

        m = round(
            current_m + DM,
            6,
        )

        # Avoid overshooting target.
        if m > TARGET_M:
            m = TARGET_M

        step_start = time.perf_counter()

        log("")
        log(
            f"START STEP: "
            f"m={m:+.6f}"
        )

        log(
            f"  seed = "
            f"{previous_root[0]:+.12f}"
            f"{previous_root[1]:+.12f}i Hz"
        )

        pole, status = continuation_step(
            m,
            previous_root,
        )

        elapsed = (
            time.perf_counter()
            - step_start
        )

        # --------------------------------------------------------
        # FAILURE
        # --------------------------------------------------------

        if status != "PASS":

            log(
                f"CHAIN STOP at m={m:+.6f}"
            )

            log(
                f"Reason: {status}"
            )

            log(
                f"Step elapsed: "
                f"{elapsed:.2f} s"
            )

            chain_broke_at = m
            break

        # --------------------------------------------------------
        # SUCCESS
        # --------------------------------------------------------

        jump_re = abs(
            pole["f_re"]
            - previous_root[0]
        )

        jump_im = abs(
            pole["f_im"]
            - previous_root[1]
        )

        log(
            f"PASS m={m:+.6f}"
        )

        log(
            f"  f = "
            f"{pole['f_re']:+.12f}"
            f"{pole['f_im']:+.12f}i Hz"
        )

        log(
            f"  |Res| = "
            f"{pole['abs_res']:.6e}"
        )

        log(
            f"  jump = "
            f"Re:{jump_re:.6e} Hz "
            f"Im:{jump_im:.6e} Hz"
        )

        log(
            f"  step time = "
            f"{elapsed:.2f} s"
        )

        # --------------------------------------------------------
        # CRITICAL:
        # SAVE IMMEDIATELY AFTER EVERY PASS
        # --------------------------------------------------------

        append_checkpoint(
            m,
            pole["f_re"],
            pole["f_im"],
            pole["abs_res"],
            jump_re,
            jump_im,
            elapsed,
            status="PASS",
        )

        write_state(
            m,
            pole["f_re"],
            pole["f_im"],
            pole["abs_res"],
        )

        # Update continuation state only AFTER durable save.
        current_m = m

        previous_root = (
            pole["f_re"],
            pole["f_im"],
        )

        total_elapsed = (
            time.perf_counter()
            - total_start
        )

        log(
            f"  CHECKPOINT SAVED "
            f"(total elapsed {total_elapsed:.2f} s)"
        )

        # --------------------------------------------------------
        # TARGET
        # --------------------------------------------------------

        if abs(
            current_m - TARGET_M
        ) < 1e-9:

            log("")
            log("TARGET REACHED.")
            break

    # ============================================================
    # FINALIZATION
    # ============================================================

    log("")
    log(sep)

    if chain_broke_at is None:

        log("BLIND CONTINUATION COMPLETE")
        log(
            f"Final m={current_m:+.6f}"
        )

    else:

        log("BLIND CONTINUATION STOPPED")
        log(
            f"Chain broke at "
            f"m={chain_broke_at:+.6f}"
        )

    log(sep)

    # ------------------------------------------------------------
    # CREATE FINAL CSV FROM CHECKPOINT
    # ------------------------------------------------------------

    if CHECKPOINT_CSV.exists():

        try:

            with CHECKPOINT_CSV.open(
                "r",
                newline="",
                encoding="utf-8"
            ) as src:

                rows = list(
                    csv.DictReader(src)
                )

            with FINAL_CSV.open(
                "w",
                newline="",
                encoding="utf-8"
            ) as dst:

                fieldnames = [
                    "timestamp",
                    "m",
                    "f_re_Hz",
                    "f_im_Hz",
                    "abs_res",
                    "jump_re_Hz",
                    "jump_im_Hz",
                    "elapsed_step_s",
                    "status",
                ]

                writer = csv.DictWriter(
                    dst,
                    fieldnames=fieldnames,
                )

                writer.writeheader()

                for row in rows:

                    if row["status"] in (
                        "START",
                        "PASS",
                    ):

                        writer.writerow(row)

                dst.flush()
                os.fsync(dst.fileno())

            log(
                f"Final CSV -> "
                f"{FINAL_CSV}"
            )

        except Exception as e:

            log(
                f"WARNING: could not create "
                f"final CSV: {e}"
            )

    log("")
    log("BLIND TEST FINISHED.")
    log(
        "Experimental QNM frequencies were "
        "NOT used by this script."
    )
    log("")


if __name__ == "__main__":
    main()

