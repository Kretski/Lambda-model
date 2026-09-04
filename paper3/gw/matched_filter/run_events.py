#!/usr/bin/env python3
"""
run_events.py -- run the frozen MDR pipeline over several events
=================================================================

Imports mdr_search and overrides ONLY the input data (file paths and
event GPS) plus the output filename. Every analysis parameter --
thresholds, template bank, alpha grid, background slides, off-source
count -- is inherited unchanged from mdr_search.py, so all events are
analysed identically.

If you find yourself editing anything below except the EVENTS table,
stop: that would break the frozen-configuration guarantee that makes
the events comparable.

Usage:
    python3 run_events.py                # all events in the table
    python3 run_events.py GW170104       # one event by name
"""

import json
import os
import sys
import time
import traceback

import mdr_search as M

# ======================================================================
# EVENTS -- the only thing you edit
# ======================================================================
# Download the 4096 s, 4 kHz HDF5 strain files for each event from the
# GWOSC event page (gwosc.org) into this directory, then fill in the
# exact filenames below. Leave an entry commented out until its files
# are present.
EVENTS = [
    dict(
        name="GW150914",
        gps=1126259462.423,
        h1="H-H1_GWOSC_4KHZ_R1-1126257415-4096.hdf5",
        l1="L-L1_GWOSC_4KHZ_R1-1126257415-4096.hdf5",
    ),
    dict(
        name="GW170104",
        gps=1167559936.600,
        h1="H-H1_GWOSC_O2_4KHZ_R1-1167556608-4096.hdf5",
        l1="L-L1_GWOSC_O2_4KHZ_R1-1167556608-4096.hdf5",
    ),
    dict(
        name="GW170814",
        gps=1186741861.500,
        h1="H-H1_GWOSC_O2_4KHZ_R1-1186738176-4096.hdf5",
        l1="L-L1_GWOSC_O2_4KHZ_R1-1186738176-4096.hdf5",
    ),
    dict(
        name="GW170823",
        gps=1187529256.500,
        h1="H-H1_GWOSC_O2_4KHZ_R1-1187528704-4096.hdf5",
        l1="L-L1_GWOSC_O2_4KHZ_R1-1187528704-4096.hdf5",
    ),
]

OUT_DIR = "events"


def run_one(ev):
    name = ev["name"]
    if not ev["h1"] or not ev["l1"]:
        print(f"[{name}] SKIP: strain files not configured")
        return False
    for p in (ev["h1"], ev["l1"]):
        if not os.path.exists(p):
            print(f"[{name}] SKIP: missing file {p}")
            return False

    out = os.path.join(OUT_DIR, f"{name}_results.json")
    log = os.path.join(OUT_DIR, f"{name}.log")

    # --- the ONLY overrides: input data and output path ---------------
    M.LOCAL = {"H1": ev["h1"], "L1": ev["l1"]}
    M.SMOKE_CENTRE = ev["gps"]
    M.OUT_JSON = out
    M.SMOKE = False

    print(f"\n{'='*70}\n[{name}] starting  GPS={ev['gps']:.3f}\n{'='*70}")
    t0 = time.time()

    # Tee stdout to a per-event log so each run keeps its own evidence.
    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                st.write(s)

        def flush(self):
            for st in self.streams:
                st.flush()

    real_stdout = sys.stdout
    with open(log, "w") as fh:
        sys.stdout = Tee(real_stdout, fh)
        try:
            M.main()
            ok = True
        except Exception:
            traceback.print_exc()
            ok = False
        finally:
            sys.stdout = real_stdout

    dt = time.time() - t0
    print(f"[{name}] {'done' if ok else 'FAILED'} in {dt/60:.1f} min -> {out}")

    if ok and os.path.exists(out):
        # stamp the event name into the JSON so combine_mdr.py can label it
        with open(out) as fh:
            d = json.load(fh)
        d["event"] = name
        d["event_gps"] = ev["gps"]
        d["runtime_sec"] = dt
        with open(out, "w") as fh:
            json.dump(d, fh, indent=2)
    return ok


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    wanted = sys.argv[1:] if len(sys.argv) > 1 else None

    print("Frozen configuration inherited from mdr_search.py:")
    print(f"  SNR_THRESH    = {M.SNR_THRESH}")
    print(f"  NEWSNR_THRESH = {M.NEWSNR_THRESH}")
    print(f"  F_LOW/F_HIGH  = {M.F_LOW} / {M.F_HIGH}")
    print(f"  ALPHAS        = {M.ALPHAS}")
    print(f"  N_SLIDES      = {M.N_SLIDES}   N_OFFSOURCE = {M.N_OFFSOURCE}")
    print(f"  bank          = {len(M.make_bank())} templates")

    done = []
    for ev in EVENTS:
        if wanted and ev["name"] not in wanted:
            continue
        if run_one(ev):
            done.append(ev["name"])

    print(f"\ncompleted: {done}")
    print(f"next: python3 combine_mdr.py")


if __name__ == "__main__":
    main()
