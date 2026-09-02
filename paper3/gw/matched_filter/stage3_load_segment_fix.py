"""
FIX for stage3_real_strain_validation.py
=========================================

The HDF5 file uses attribute names "x0" (GPS start) and "dx" (sample
spacing), but load_real_segment() was only checking for "Xstart" and
"Xspacing". Because of that mismatch, gps_start was always None, and
the code silently fell back to "middle of the whole file" instead of
using the real GPS start time -- which meant the analysis window was
NOT reliably centered on the true GW150914 merger.

Replace get_attribute() and load_real_segment() in
stage3_real_strain_validation.py with the versions below. Everything
else in the file (imports, EVENT_CATALOG, to_frequency_domain,
estimate_psd_from_segment, etc.) stays exactly as it is.
"""

import numpy as np
import h5py


def get_attribute(obj, names, default=None):
    """
    Safely retrieve an HDF5 attribute, trying several possible
    attribute names in order (different HDF5 strain files use
    different naming conventions -- e.g. LIGO-style "Xstart"/
    "Xspacing" vs. this file's "x0"/"dx").

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

    return default


def load_real_segment(
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

        GPS start:      Xstart  or  x0
        Sample spacing:  Xspacing  or  dx
        Sample count:    Npoints  (else falls back to dataset shape)

    If no GPS-start metadata is found under ANY known name, fs_override
    is required and gps_center is interpreted relative to the beginning
    of the file (this fallback is now a last resort, not the default
    path for files like this one that DO have real metadata, just
    under different attribute names).
    """

    # local import to avoid depending on this fix file's location
    from stage3_real_strain_validation import find_strain_dataset

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
                "\nSampling frequency is missing from the HDF5 file.\n"
                "Use --fs, for example:\n\n"
                "    --fs 4096\n"
            )

        # --------------------------------------------------------------
        # GPS start (x0 = start time in seconds, GPS convention)
        # --------------------------------------------------------------

        gps_start = get_attribute(strain, ["Xstart", "x0"], None)

        if gps_start is None:
            gps_start = get_attribute(f, ["Xstart", "x0"], None)

        # --------------------------------------------------------------
        # Files with real GPS-start metadata (now correctly detected):
        # locate the requested GPS center exactly.
        #
        # Only files with genuinely NO start-time metadata under any
        # known name fall back to "middle of file" / relative-time
        # interpretation.
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
                "\nRequested segment lies outside the available data.\n"
                f"n_total={n_total}\n"
                f"fs={fs}\n"
                f"idx_center={idx_center}\n"
                f"idx_lo={idx_lo}\n"
                f"idx_hi={idx_hi}\n"
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

    return data, fs, seg_gps_start
