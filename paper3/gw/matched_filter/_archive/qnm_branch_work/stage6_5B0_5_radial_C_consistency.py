from pathlib import Path
import sys
import importlib.util
import inspect
import csv
import json
import math


# =============================================================================
# STAGE 6.5B-0.5 — RADIAL C-CONSISTENCY PREFLIGHT
# =============================================================================
#
# PURPOSE
# -------
# Verify that explicit C injection propagates consistently through the
# authoritative radial/QNM machinery BEFORE any complex resonance root search.
#
# This is NOT a resonance test.
#
# We test only:
#   1. find_light_ring(..., C=...)
#   2. light-ring frequency
#   3. solve_p(...) at/near the light ring
#   4. real radial structure around the light ring
#
# SCIENTIFIC RULES
# ----------------
# * authoritative solver is loaded normally
# * no AST rewriting
# * no modification of authoritative globals
# * no modification of function defaults
# * only explicit C argument is varied
# * no complex root search
# * no Kerr fit
# * no GW calibration
# * no frequency rescaling
# * no low-k approximation
#
# EXPECTATION
# -----------
# C=0:
#   no interior light ring -> radial resonance machinery is NOT tested
#
# C=C0/2, C=C0, C=2C0:
#   interior light ring exists
#   explicit C must affect the real radial structure
#
# The test must fail if the same radial structure is returned for different C.
#
# =============================================================================


HERE = Path(__file__).resolve().parent

RES_FILE = HERE / "stage5I_3_resonance.py"

OUTPUT_DIR = HERE / "stage6_5B0_5_radial_C_consistency"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = OUTPUT_DIR / "stage6_5B0_5_radial_results.csv"
JSON_FILE = OUTPUT_DIR / "stage6_5B0_5_radial_results.json"


# =============================================================================
# FIXED AUTHORITATIVE BASELINE
# =============================================================================

BASE_C = 7.690000000000e-04
M = -12

OMEGA = 1.000000000000e-01
GAMMA = 2.220000000000e-06
H0 = 3.400000000000e-02
G_GRAV = 9.810000000000e+00
R_B = 3.730000000000e-02


CASES = [
    ("C_ZERO", 0.0),
    ("C_HALF", 0.5 * BASE_C),
    ("BASELINE", BASE_C),
    ("C_DOUBLE", 2.0 * BASE_C),
]


# =============================================================================
# SMALL RADIAL OFFSETS
# =============================================================================
#
# These are deliberately small.
#
# We do NOT try to infer a resonance.
# We only check whether the real radial structure is internally consistent
# around the light ring.
#

OFFSETS_MM = [
    -2.0,
    -1.0,
    -0.25,
    0.0,
    +0.25,
    +1.0,
    +2.0,
]


# =============================================================================
# LOAD AUTHORITATIVE SOLVER NORMALLY
# =============================================================================

def load_authoritative_solver():
    if not RES_FILE.exists():
        raise FileNotFoundError(
            f"Authoritative solver not found:\n{RES_FILE}"
        )

    module_name = "_stage5I_3_resonance_radial_preflight"

    # Remove only our temporary import if it exists.
    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(
        module_name,
        str(RES_FILE),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not create import spec for:\n{RES_FILE}"
        )

    module = importlib.util.module_from_spec(spec)

    # Normal import execution.
    spec.loader.exec_module(module)

    return module


# =============================================================================
# FIND FUNCTION
# =============================================================================

def find_callable(mod, name):
    fn = getattr(mod, name, None)

    if fn is None:
        raise AttributeError(
            f"Authoritative solver does not expose required function: {name}"
        )

    if not callable(fn):
        raise TypeError(
            f"Authoritative attribute is not callable: {name}"
        )

    return fn


# =============================================================================
# LIGHT-RING CALL
# =============================================================================

def call_light_ring(mod, C_value):
    fn = find_callable(mod, "find_light_ring")

    sig = inspect.signature(fn)

    if "C" not in sig.parameters:
        raise RuntimeError(
            "Authoritative find_light_ring() does not expose explicit C argument."
        )

    kwargs = {
        "C": float(C_value),
    }

    # Pass authoritative parameters explicitly when supported.
    #
    # This avoids accidentally depending on changed defaults while still
    # leaving the authoritative module untouched.
    optional_authoritative = {
        "Omega": OMEGA,
        "gamma": GAMMA,
        "h0": H0,
        "g": G_GRAV,
        "r_min": 1.0e-3,
        "r_max": R_B,
    }

    for key, value in optional_authoritative.items():
        if key in sig.parameters:
            kwargs[key] = value

    r_sp, omega_lr = fn(M, **kwargs)

    if r_sp is None or omega_lr is None:
        return None, None

    return float(r_sp), float(omega_lr)


# =============================================================================
# SOLVE_P
# =============================================================================

def call_solve_p(mod, omega, r, C_value):
    fn = find_callable(mod, "solve_p")
    sig = inspect.signature(fn)

    kwargs = {}

    # solve_p is expected to expose these according to the validated
    # radial module API. We inspect first rather than guessing.
    if "C" in sig.parameters:
        kwargs["C"] = float(C_value)

    if "Omega" in sig.parameters:
        kwargs["Omega"] = OMEGA

    if "gamma" in sig.parameters:
        kwargs["gamma"] = GAMMA

    if "h0" in sig.parameters:
        kwargs["h0"] = H0

    if "g" in sig.parameters:
        kwargs["g"] = G_GRAV

    try:
        value = fn(
            float(omega),
            M,
            float(r),
            **kwargs,
        )
    except TypeError:
        # Some authoritative versions may expose a different positional
        # ordering. Do NOT silently invent an alternative API.
        raise RuntimeError(
            "Authoritative solve_p() signature is incompatible with this "
            "preflight.\n"
            f"Detected signature: {sig}"
        )

    return value


# =============================================================================
# NUMERIC NORMALIZATION
# =============================================================================

def finite_real(value):
    """
    Return a finite real float if possible.

    solve_p may return:
      - a real scalar
      - complex with negligible imaginary part
      - NaN / inf
    """

    if value is None:
        return None

    try:
        z = complex(value)
    except Exception:
        try:
            x = float(value)
        except Exception:
            return None

        return x if math.isfinite(x) else None

    if not (
        math.isfinite(z.real)
        and math.isfinite(z.imag)
    ):
        return None

    # Keep the real component when the imaginary part is only numerical noise.
    if abs(z.imag) < 1.0e-10 * max(1.0, abs(z.real)):
        return float(z.real)

    return None


# =============================================================================
# SINGLE CASE
# =============================================================================

def run_case(mod, label, C_value):
    print()
    print("-" * 78)
    print(f"{label}: C={C_value:.12e} m^2/s")
    print("-" * 78)

    result = {
        "case": label,
        "C": float(C_value),
        "light_ring": False,
        "r_sp_m": None,
        "r_sp_mm": None,
        "omega_lr": None,
        "f_lr_hz": None,
        "radial_points": [],
        "status": "UNKNOWN",
        "error": None,
    }

    try:
        r_sp, omega_lr = call_light_ring(mod, C_value)
    except Exception as exc:
        result["status"] = "LIGHT_RING_CALL_FAILED"
        result["error"] = repr(exc)

        print("  ERROR calling find_light_ring():")
        print(f"    {exc}")

        return result

    if r_sp is None or omega_lr is None:
        print("  LIGHT RING: NONE")
        print("  radial structure test: SKIPPED")

        result["status"] = "NO_LIGHT_RING"
        return result

    f_lr = omega_lr / (2.0 * math.pi)

    result["light_ring"] = True
    result["r_sp_m"] = r_sp
    result["r_sp_mm"] = 1000.0 * r_sp
    result["omega_lr"] = omega_lr
    result["f_lr_hz"] = f_lr

    print(f"  LIGHT RING: r_sp={1000.0*r_sp:.12f} mm")
    print(f"  omega_lr  = {omega_lr:.12e} rad/s")
    print(f"  f_lr      = {f_lr:.12f} Hz")

    # -------------------------------------------------------------------------
    # Radial consistency around light ring
    # -------------------------------------------------------------------------

    print()
    print("  REAL RADIAL STRUCTURE AROUND LIGHT RING")
    print()
    print("    offset [mm]       r [mm]          solve_p")

    valid_count = 0

    for offset_mm in OFFSETS_MM:

        r = r_sp + offset_mm * 1.0e-3

        # Stay strictly inside the authoritative radial interval.
        if r <= 1.0e-3 or r >= R_B:
            continue

        try:
            p_value = call_solve_p(
                mod,
                omega_lr,
                r,
                C_value,
            )

            p_real = finite_real(p_value)

            if p_real is not None:
                valid_count += 1

            point = {
                "offset_mm": float(offset_mm),
                "r_m": float(r),
                "r_mm": float(1000.0 * r),
                "solve_p_raw": repr(p_value),
                "solve_p_real": p_real,
                "finite_real": p_real is not None,
            }

            result["radial_points"].append(point)

            if p_real is None:
                display = str(p_value)
            else:
                display = f"{p_real:.12e}"

            print(
                f"    {offset_mm:+10.3f}"
                f"   {1000.0*r:14.9f}"
                f"   {display}"
            )

        except Exception as exc:

            point = {
                "offset_mm": float(offset_mm),
                "r_m": float(r),
                "r_mm": float(1000.0 * r),
                "solve_p_raw": None,
                "solve_p_real": None,
                "finite_real": False,
                "error": repr(exc),
            }

            result["radial_points"].append(point)

            print(
                f"    {offset_mm:+10.3f}"
                f"   {1000.0*r:14.9f}"
                f"   ERROR: {exc}"
            )

    result["valid_radial_points"] = valid_count

    # At the exact light ring solve_p may legitimately approach zero.
    # Therefore we require only that the surrounding real radial machinery
    # returns finite real values somewhere around the light ring.
    if valid_count >= 3:
        result["status"] = "PASS"
    else:
        result["status"] = "RADIAL_STRUCTURE_FAILED"

    print()
    print(
        f"  finite real radial points = "
        f"{valid_count}/{len(result['radial_points'])}"
    )

    print(f"  CASE STATUS: {result['status']}")

    return result


# =============================================================================
# CROSS-CASE VALIDATION
# =============================================================================

def validate(results):
    failures = []

    by_case = {
        item["case"]: item
        for item in results
    }

    # -------------------------------------------------------------------------
    # C=0 must have no interior light ring.
    # -------------------------------------------------------------------------

    zero = by_case["C_ZERO"]

    if zero["status"] != "NO_LIGHT_RING":
        failures.append(
            "C=0 did not produce the expected NO_LIGHT_RING result."
        )

    # -------------------------------------------------------------------------
    # Nonzero cases must have light rings.
    # -------------------------------------------------------------------------

    for name in ("C_HALF", "BASELINE", "C_DOUBLE"):

        item = by_case[name]

        if not item["light_ring"]:
            failures.append(
                f"{name} has no light ring."
            )

        if item["status"] != "PASS":
            failures.append(
                f"{name} radial consistency status = {item['status']}"
            )

    # -------------------------------------------------------------------------
    # Baseline benchmark.
    # -------------------------------------------------------------------------

    baseline = by_case["BASELINE"]

    if baseline["light_ring"]:

        expected_r = 20.087461380721
        expected_f = 8.835443157444

        dr_mm = abs(
            baseline["r_sp_mm"] - expected_r
        )

        df_hz = abs(
            baseline["f_lr_hz"] - expected_f
        )

        print()
        print("-" * 78)
        print("BASELINE BENCHMARK")
        print("-" * 78)
        print(f"  Delta r = {dr_mm:.6e} mm")
        print(f"  Delta f = {df_hz:.6e} Hz")

        if dr_mm > 1.0e-6:
            failures.append(
                f"Baseline r_sp mismatch: {dr_mm:.6e} mm"
            )

        if df_hz > 1.0e-9:
            failures.append(
                f"Baseline f_lr mismatch: {df_hz:.6e} Hz"
            )

    # -------------------------------------------------------------------------
    # C-dependence.
    # -------------------------------------------------------------------------

    nonzero = [
        by_case["C_HALF"],
        by_case["BASELINE"],
        by_case["C_DOUBLE"],
    ]

    r_values = [
        item["r_sp_mm"]
        for item in nonzero
        if item["r_sp_mm"] is not None
    ]

    f_values = [
        item["f_lr_hz"]
        for item in nonzero
        if item["f_lr_hz"] is not None
    ]

    if len(r_values) == 3:

        r_spread = max(r_values) - min(r_values)

        print()
        print("C-DEPENDENCE:")
        print(f"  r_sp spread = {r_spread:.12f} mm")

        if r_spread < 1.0e-3:
            failures.append(
                "Light-ring radius does not respond measurably to C."
            )

    if len(f_values) == 3:

        f_spread = max(f_values) - min(f_values)

        print(f"  f_lr spread = {f_spread:.12f} Hz")

        if f_spread < 1.0e-4:
            failures.append(
                "Light-ring frequency does not respond measurably to C."
            )

    # -------------------------------------------------------------------------
    # Monotonic trend expected from the established Stage 6.5 control.
    # -------------------------------------------------------------------------

    if all(item["r_sp_mm"] is not None for item in nonzero):

        r_half = by_case["C_HALF"]["r_sp_mm"]
        r_base = by_case["BASELINE"]["r_sp_mm"]
        r_double = by_case["C_DOUBLE"]["r_sp_mm"]

        if not (r_half < r_base < r_double):
            failures.append(
                "Expected monotonic r_sp(C) trend failed."
            )

    if all(item["f_lr_hz"] is not None for item in nonzero):

        f_half = by_case["C_HALF"]["f_lr_hz"]
        f_base = by_case["BASELINE"]["f_lr_hz"]
        f_double = by_case["C_DOUBLE"]["f_lr_hz"]

        if not (f_half > f_base > f_double):
            failures.append(
                "Expected monotonic f_lr(C) trend failed."
            )

    return failures


# =============================================================================
# SAVE CSV
# =============================================================================

def save_csv(results):

    rows = []

    for item in results:

        rows.append({
            "case": item["case"],
            "C": item["C"],
            "light_ring": item["light_ring"],
            "r_sp_m": item["r_sp_m"],
            "r_sp_mm": item["r_sp_mm"],
            "omega_lr": item["omega_lr"],
            "f_lr_hz": item["f_lr_hz"],
            "valid_radial_points": item.get(
                "valid_radial_points",
                0
            ),
            "status": item["status"],
            "error": item["error"],
        })

    with CSV_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# SAVE JSON
# =============================================================================

def save_json(results, failures):

    payload = {
        "stage": "6.5B-0.5",
        "description": (
            "Radial C-consistency preflight before complex resonance search"
        ),
        "authoritative_solver": str(RES_FILE),
        "mode_m": M,
        "baseline_C": BASE_C,
        "constants": {
            "Omega": OMEGA,
            "gamma": GAMMA,
            "h0": H0,
            "g": G_GRAV,
            "r_B": R_B,
        },
        "rules": [
            "authoritative solver loaded normally",
            "no AST rewriting",
            "no modification of globals",
            "no modification of function defaults",
            "only explicit C argument is varied",
            "no complex resonance root search",
            "no Kerr fit",
            "no GW calibration",
            "no frequency rescaling",
            "no low-k approximation",
        ],
        "results": results,
        "failures": failures,
        "overall": "PASS" if not failures else "FAIL",
    }

    JSON_FILE.write_text(
        json.dumps(
            payload,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 78)
    print("STAGE 6.5B-0.5 — RADIAL C-CONSISTENCY PREFLIGHT")
    print("=" * 78)

    print()
    print("Authoritative solver:")
    print(f"  {RES_FILE}")

    print()
    print("Mode:")
    print(f"  m = {M}")

    print()
    print("Baseline circulation:")
    print(f"  C0 = {BASE_C:.12e} m^2/s")

    print()
    print("Controlled cases:")

    for label, C_value in CASES:
        print(
            f"  {label:>10}: C = {C_value:.12e}"
        )

    print()
    print("Rules:")
    print("  * authoritative solver loaded normally")
    print("  * no AST rewriting")
    print("  * no modification of globals")
    print("  * no modification of function defaults")
    print("  * only explicit C argument is varied")
    print("  * no complex resonance root search")
    print("  * no Kerr fit")
    print("  * no GW calibration")
    print("  * no frequency rescaling")
    print("  * no low-k approximation")

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    print()
    print("-" * 78)
    print("LOADING AUTHORITATIVE SOLVER")
    print("-" * 78)

    try:
        mod = load_authoritative_solver()
    except Exception as exc:

        print()
        print("FATAL: authoritative solver could not be loaded.")
        print()
        print(exc)

        raise

    print()
    print("Loaded:")
    print(f"  {RES_FILE}")

    # -------------------------------------------------------------------------
    # API audit
    # -------------------------------------------------------------------------

    print()
    print("-" * 78)
    print("AUTHORITATIVE API CHECK")
    print("-" * 78)

    find_lr = find_callable(mod, "find_light_ring")

    print()
    print(
        "find_light_ring signature:"
    )
    print(
        f"  {inspect.signature(find_lr)}"
    )

    solve_p = find_callable(mod, "solve_p")

    print()
    print(
        "solve_p signature:"
    )
    print(
        f"  {inspect.signature(solve_p)}"
    )

    if "C" not in inspect.signature(find_lr).parameters:
        raise RuntimeError(
            "FAIL: find_light_ring does not expose explicit C."
        )

    # -------------------------------------------------------------------------
    # Run cases
    # -------------------------------------------------------------------------

    results = []

    for label, C_value in CASES:

        result = run_case(
            mod,
            label,
            C_value,
        )

        results.append(result)

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    failures = validate(results)

    print()
    print("=" * 78)
    print("STAGE 6.5B-0.5 — PREFLIGHT SUMMARY")
    print("=" * 78)

    print()
    print(
        f"{'case':>12} "
        f"{'C':>16} "
        f"{'LR':>6} "
        f"{'r_sp [mm]':>18} "
        f"{'f_lr [Hz]':>18} "
        f"{'status':>24}"
    )

    print("-" * 78)

    for item in results:

        r_text = (
            f"{item['r_sp_mm']:.12f}"
            if item["r_sp_mm"] is not None
            else "NONE"
        )

        f_text = (
            f"{item['f_lr_hz']:.12f}"
            if item["f_lr_hz"] is not None
            else "NONE"
        )

        print(
            f"{item['case']:>12} "
            f"{item['C']:16.9e} "
            f"{str(item['light_ring']):>6} "
            f"{r_text:>18} "
            f"{f_text:>18} "
            f"{item['status']:>24}"
        )

    print()
    print("=" * 78)

    if failures:

        print("PREFLIGHT: FAIL")
        print()
        print("Failures:")

        for failure in failures:
            print(f"  - {failure}")

        print()
        print(
            "DO NOT RUN THE COMPLEX RESONANCE TEST."
        )

    else:

        print("PREFLIGHT: PASS")
        print()
        print(
            "The explicit-C radial structure is internally consistent."
        )
        print()
        print(
            "The system is ready for the complex resonance-level "
            "C-causality test."
        )

    print()
    print("No authoritative solver file was modified.")

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    save_csv(results)
    save_json(results, failures)

    print()
    print("=" * 78)
    print("OUTPUT FILES")
    print("=" * 78)
    print()
    print(f"CSV  -> {CSV_FILE}")
    print(f"JSON -> {JSON_FILE}")
    print()

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()