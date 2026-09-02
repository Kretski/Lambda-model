"""
check_metadata.py
==================

Quick helper: print HDF5 structure/attributes for the H1 strain file,
so we can see whether it has real GPS start-time metadata (Xstart) or
not. Run this from the same folder as stage3_real_strain_validation.py.

Usage:
    python check_metadata.py --h1 "C:\\path\\to\\H1_GW150914_4096s.hdf5"
"""

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from stage3_real_strain_validation import inspect_hdf5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h1", required=True)
    args = parser.parse_args()

    inspect_hdf5(args.h1)


if __name__ == "__main__":
    main()
