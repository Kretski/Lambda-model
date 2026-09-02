"""
apply_fix.py
============

Automatically patches stage3_real_strain_validation.py in place:
replaces the old get_attribute() and load_real_segment() functions
with the fixed versions that also recognize "x0"/"dx" attribute
names (not just "Xstart"/"Xspacing").

Makes a backup copy first: stage3_real_strain_validation.py.bak

Usage (run from the paper3\\gw\\matched_filter folder):
    python apply_fix.py
"""

from pathlib import Path
import shutil
import sys

TARGET = Path("stage3_real_strain_validation.py")

OLD_GET_ATTRIBUTE = '''def get_attribute(obj, name, default=None):
    """
    Safely retrieve an HDF5 attribute.

    Handles numpy scalar values.
    """

    if name not in obj.attrs:
        return default

    value = obj.attrs[name]

    try:
        return value.item()
    except Exception:
        return value'''

NEW_GET_ATTRIBUTE = '''def get_attribute(obj, names, default=None):
    """
    Safely retrieve an HDF5 attribute, trying several possible
    attribute names in order (different HDF5 strain files use
    different naming conventions -- e.g. LIGO-style "Xstart"/
    "Xspacing" vs. "x0"/"dx").

    `names` can be a single string or a list/tuple of candidate
    names; the first one found in obj.attrs is used.
    """

    if isinstance(names, str):
        names = [names]

    for name in names:
        if name in obj.attrs:
            value = obj.attrs[name]
            try:
                return value.item()
            except Exception:
                return value

    return default'''

OLD_LOAD_SEGMENT = '''def load_real_segment(
    h1_path,
    gps_center,
    half_window,
    fs_override=None,
):
    """
    Load a real strain segment.

    Supports both HDF5 layouts:

        /strain/Strain

    and:

        /Strain

    Metadata priority:

        Xspacing
        Xstart
        Npoints

    If metadata is missing:

        fs_override is required.

    For a simple HDF5 file with no GPS metadata, gps_center is interpreted
    relative to the beginning of the file.
    """

    with h5py.File(h1_path, "r") as f:

        strain = find_strain_dataset(f)

        # --------------------------------------------------------------
        # Number of samples
        # --------------------------------------------------------------

        n_total_attr = get_attribute(strain, "Npoints", None)

        if n_total_attr is None:
            n_total = strain.shape[0]
        else:
            n_total = int(n_total_attr)

        # --------------------------------------------------------------
        # Sampling frequency
        # --------------------------------------------------------------

        xspacing = get_attribute(strain, "Xspacing", None)

        if xspacing is not None:
            fs = 1.0 / float(xspacing)
        elif fs_override is not None:
            fs = float(fs_override)
        else:
            raise ValueError(
                "\\nSampling frequency is missing from the HDF5 file.\\n"
                "Use --fs, for example:\\n\\n"
                "    --fs 4096\\n"
            )

        # --------------------------------------------------------------
        # GPS start
        # --------------------------------------------------------------

        gps_start = get_attribute(strain, "Xstart", None)

        if gps_start is None:
            gps_start = get_attribute(f, "Xstart", None)

        # --------------------------------------------------------------
        # Simple HDF5 files without GPS metadata
        #
        # In that case we treat the file as a time series starting at
        # t=0 and gps_center as a relative time coordinate.
        # --------------------------------------------------------------

        if gps_start is None:

            # If the requested center looks like a GPS timestamp,
            # default to the center of the available file.
            if abs(float(gps_center)) > 1e6:
                idx_center = n_total // 2
            else:
                idx_center = int(round(float(gps_center) * fs))

            effective_gps_start = 0.0

        else:

            gps_start = float(gps_start)

            idx_center = int(
                round((float(gps_center) - gps_start) * fs)
            )

            effective_gps_start = gps_start

        # --------------------------------------------------------------
        # Requested window
        # --------------------------------------------------------------

        idx_half = int(round(half_window * fs))

        idx_lo = max(0, idx_center - idx_half)
        idx_hi = min(n_total, idx_center + idx_half)

        if idx_hi <= idx_lo:
            raise ValueError(
                "\\nRequested segment lies outside the available data.\\n"
                f"n_total={n_total}\\n"
                f"fs={fs}\\n"
                f"idx_center={idx_center}\\n"
                f"idx_lo={idx_lo}\\n"
                f"idx_hi={idx_hi}\\n"
            )

        data = np.asarray(
            strain[idx_lo:idx_hi],
            dtype=np.float64,
        )

        seg_gps_start = (
            effective_gps_start + idx_lo / fs
        )

    print(
        f"  Loaded HDF5 dataset successfully: "
        f"{len(data)} samples, fs={fs:.3f} Hz"
    )

    return data, fs, seg_gps_start'''

NEW_LOAD_SEGMENT = '''def load_real_segment(
    h1_path,
    gps_center,
    half_window,
    fs_override=None,
):
    """
    Load a real strain segment.

    Supports both HDF5 layouts:

        /strain/Strain

    and:

        /Strain

    Metadata priority (tries multiple naming conventions):

        GPS start:       Xstart  or  x0
        Sample spacing:  Xspacing  or  dx
        Sample count:    Npoints  (else falls back to dataset shape)

    If no GPS-start metadata is found under ANY known name, fs_override
    is required and gps_center is interpreted relative to the beginning
    of the file (last-resort fallback only).
    """

    with h5py.File(h1_path, "r") as f:

        strain = find_strain_dataset(f)

        # --------------------------------------------------------------
        # Number of samples
        # --------------------------------------------------------------

        n_total_attr = get_attribute(strain, ["Npoints"], None)

        if n_total_attr is None:
            n_total = strain.shape[0]
        else:
            n_total = int(n_total_attr)

        # --------------------------------------------------------------
        # Sampling frequency (dx = sample spacing in seconds)
        # --------------------------------------------------------------

        xspacing = get_attribute(strain, ["Xspacing", "dx"], None)

        if xspacing is not None:
            fs = 1.0 / float(xspacing)
        elif fs_override is not None:
            fs = float(fs_override)
        else:
            raise ValueError(
                "\\nSampling frequency is missing from the HDF5 file.\\n"
                "Use --fs, for example:\\n\\n"
                "    --fs 4096\\n"
            )

        # --------------------------------------------------------------
        # GPS start (x0 = start time in seconds, GPS convention)
        # --------------------------------------------------------------

        gps_start = get_attribute(strain, ["Xstart", "x0"], None)

        if gps_start is None:
            gps_start = get_attribute(f, ["Xstart", "x0"], None)

        # --------------------------------------------------------------
        # Files with real GPS-start metadata: locate the requested GPS
        # center exactly. Only files with genuinely NO start-time
        # metadata under any known name fall back to "middle of file".
        # --------------------------------------------------------------

        if gps_start is None:

            if abs(float(gps_center)) > 1e6:
                idx_center = n_total // 2
                print(
                    "  WARNING: no GPS start-time metadata found under "
                    "any known attribute name (Xstart/x0). Falling back "
                    "to the middle of the file as the analysis center. "
                    "This is likely WRONG for a real-event analysis -- "
                    "verify the file's true GPS start time."
                )
            else:
                idx_center = int(round(float(gps_center) * fs))

            effective_gps_start = 0.0

        else:

            gps_start = float(gps_start)

            idx_center = int(
                round((float(gps_center) - gps_start) * fs)
            )

            effective_gps_start = gps_start

        # --------------------------------------------------------------
        # Requested window
        # --------------------------------------------------------------

        idx_half = int(round(half_window * fs))

        idx_lo = max(0, idx_center - idx_half)
        idx_hi = min(n_total, idx_center + idx_half)

        if idx_hi <= idx_lo:
            raise ValueError(
                "\\nRequested segment lies outside the available data.\\n"
                f"n_total={n_total}\\n"
                f"fs={fs}\\n"
                f"idx_center={idx_center}\\n"
                f"idx_lo={idx_lo}\\n"
                f"idx_hi={idx_hi}\\n"
            )

        data = np.asarray(
            strain[idx_lo:idx_hi],
            dtype=np.float64,
        )

        seg_gps_start = (
            effective_gps_start + idx_lo / fs
        )

    print(
        f"  Loaded HDF5 dataset successfully: "
        f"{len(data)} samples, fs={fs:.3f} Hz"
    )

    if gps_start is not None:
        merger_offset_in_segment = (
            float(gps_center) - seg_gps_start
        )
        print(
            f"  GPS start used: {gps_start:.3f}  "
            f"-> requested center at t={merger_offset_in_segment:.4f}s "
            f"into the loaded segment"
        )

    return data, fs, seg_gps_start'''


def main():
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found in the current folder.")
        print("Run this script from paper3\\gw\\matched_filter\\")
        sys.exit(1)

    original = TARGET.read_text(encoding="utf-8")

    if NEW_GET_ATTRIBUTE in original and NEW_LOAD_SEGMENT in original:
        print("Already patched -- nothing to do.")
        return

    if OLD_GET_ATTRIBUTE not in original:
        print("ERROR: could not find the original get_attribute() text.")
        print("The file may already be modified differently than expected.")
        print("No changes made.")
        sys.exit(1)

    if OLD_LOAD_SEGMENT not in original:
        print("ERROR: could not find the original load_real_segment() text.")
        print("The file may already be modified differently than expected.")
        print("No changes made.")
        sys.exit(1)

    backup = TARGET.with_suffix(".py.bak")
    shutil.copy(TARGET, backup)
    print(f"Backup saved -> {backup}")

    patched = original.replace(OLD_GET_ATTRIBUTE, NEW_GET_ATTRIBUTE)
    patched = patched.replace(OLD_LOAD_SEGMENT, NEW_LOAD_SEGMENT)

    TARGET.write_text(patched, encoding="utf-8")

    print(f"Patched -> {TARGET}")
    print()
    print("Done. Now re-run stage6B_tc_diagnostic.py.")


if __name__ == "__main__":
    main()
