"""
Stage 6.1 -- Lambda / Giant-Vortex -> Kerr QNM Mapping Audit

Purpose
-------
Audit whether the Root-B giant-vortex QNM observables provide
dimensionless spectral features that could potentially support
a mapping onto the Kerr QNM parameter space.

IMPORTANT
---------
This script deliberately does NOT:

  - use real GW strain data
  - fit GW150914
  - optimize a mapping against an observed event
  - identify m_gv with Kerr m
  - identify circulation with a/M
  - identify analog frequency scale with astrophysical frequency scale
  - claim that a Kerr mapping exists

This is a mapping audit only.

Input
-----
stage5K_1_observables_rootB.csv

Actual Root-B CSV schema:
    m
    f_re
    f_im
    f_R_hz
    tau_s
    Q
    abs_res

Outputs
-------
stage6_1_mapping_audit.csv
stage6_1_mapping_audit.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# Paths
# =====================================================================

HERE = Path(__file__).resolve().parent

INPUT = HERE / "stage5K_1_observables_rootB.csv"

OUTPUT_CSV = HERE / "stage6_1_mapping_audit.csv"
OUTPUT_JSON = HERE / "stage6_1_mapping_audit.json"


# =====================================================================
# Helper functions
# =====================================================================

def safe_div(a, b):
    """
    Safe element-wise division.
    Invalid / infinite values become NaN.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b

    out[~np.isfinite(out)] = np.nan

    return out


def relative_span(x):
    """
    (max-min)/mean(abs(x))
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    mean_abs = np.mean(np.abs(x))

    if mean_abs == 0:
        return np.nan

    return (np.max(x) - np.min(x)) / mean_abs


def monotonic_fraction(x):
    """
    Fraction of successive differences that are >= 0.

    1.0 means monotonically non-decreasing.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan

    dx = np.diff(x)

    return float(np.mean(dx >= 0))


def unique_value_count(x, decimals=10):
    """
    Number of numerically distinct values after rounding.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return 0

    return int(len(np.unique(np.round(x, decimals))))


def correlation_with_m(m, x):
    """
    Pearson correlation between m and feature x.
    """
    m = np.asarray(m, dtype=float)
    x = np.asarray(x, dtype=float)

    mask = np.isfinite(m) & np.isfinite(x)

    if np.sum(mask) < 2:
        return np.nan

    return float(np.corrcoef(m[mask], x[mask])[0, 1])


# =====================================================================
# Start
# =====================================================================

print("=" * 100)
print("STAGE 6.1 -- Λ / GIANT-VORTEX -> KERR QNM MAPPING AUDIT")
print("=" * 100)


# =====================================================================
# Load input
# =====================================================================

if not INPUT.exists():
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT}\n\n"
        "Run Stage 5K.1 Root B first."
    )


df = pd.read_csv(INPUT)

print(f"\nInput file: {INPUT.name}")

print("\nDetected columns:")
print(list(df.columns))


# =====================================================================
# Normalize possible old column names
# =====================================================================

# The current Root-B file uses:
#
#   f_R_hz
#   tau_s
#   abs_res
#
# Older scripts may have used:
#
#   f_R (Hz)
#   tau (s)
#   |Res|
#
# Support both formats without changing the underlying data.

COLUMN_ALIASES = {
    "f_R (Hz)": "f_R_hz",
    "tau (s)": "tau_s",
    "|Res|": "abs_res",
}

df = df.rename(columns=COLUMN_ALIASES)


# =====================================================================
# Validate schema
# =====================================================================

required_cols = [
    "m",
    "f_re",
    "f_im",
    "f_R_hz",
    "tau_s",
    "Q",
    "abs_res",
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(
        "\nMissing required columns:\n"
        f"{missing}\n\n"
        "Available columns:\n"
        f"{list(df.columns)}"
    )


# =====================================================================
# Select relevant columns
# =====================================================================

df = df[required_cols].copy()


# =====================================================================
# Numeric conversion
# =====================================================================

for column in required_cols:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# Remove invalid rows
df = df.dropna().reset_index(drop=True)


# =====================================================================
# Basic validation
# =====================================================================

if len(df) == 0:
    raise ValueError(
        "No valid numerical rows remain after input cleaning."
    )


# Root-B Stage 5K.1 already contains only validated points.
#
# We nevertheless retain the residual column and report it explicitly.

print(f"\nValidated Root-B points: {len(df)}")

print(
    f"m range: "
    f"[{df['m'].min():.6f}, {df['m'].max():.6f}]"
)

print(
    f"Residual range: "
    f"[{df['abs_res'].min():.3e}, "
    f"{df['abs_res'].max():.3e}]"
)


# =====================================================================
# Construct spectral quantities
# =====================================================================

m_gv = df["m"].to_numpy(dtype=float)

f_R = df["f_R_hz"].to_numpy(dtype=float)

tau = df["tau_s"].to_numpy(dtype=float)

Q_analog = df["Q"].to_numpy(dtype=float)

abs_res = df["abs_res"].to_numpy(dtype=float)


# ---------------------------------------------------------------------
# Convert to angular-frequency quantities
# ---------------------------------------------------------------------

# omega_R = 2*pi*f_R
#
# For a damped mode:
#
# omega_I = -1/tau
#
# Therefore:
#
# omega_R / |omega_I|
#       = 2*pi*f_R*tau
#
# and:
#
# Q = omega_R/(2|omega_I|)
#   = pi*f_R*tau

omega_R = 2.0 * np.pi * f_R

omega_I_abs = 1.0 / tau


# =====================================================================
# Dimensionless spectral features
# =====================================================================

df["omega_R"] = omega_R

df["omega_I_abs"] = omega_I_abs


df["omegaR_over_abs_omegaI"] = safe_div(
    omega_R,
    omega_I_abs
)


df["Q_from_f_tau"] = (
    np.pi * f_R * tau
)


df["damping_ratio"] = safe_div(
    omega_I_abs,
    omega_R
)


df["inverse_damping_ratio"] = safe_div(
    omega_R,
    omega_I_abs
)


# =====================================================================
# Q consistency check
# =====================================================================

df["Q_consistency_error"] = (
    df["Q_from_f_tau"] - Q_analog
)


df["Q_consistency_relative_error"] = safe_div(
    df["Q_consistency_error"],
    Q_analog
)


# =====================================================================
# Relative spectral features
# =====================================================================

# These remove the absolute analog frequency scale.
#
# They are useful because a physically meaningful mapping should
# ideally distinguish dimensionless spectral structure from arbitrary
# unit choices.

f0 = f_R[0]

tau0 = tau[0]

Q0 = Q_analog[0]

ratio0 = df["omegaR_over_abs_omegaI"].iloc[0]


df["fR_relative_to_start"] = (
    f_R / f0
)


df["tau_relative_to_start"] = (
    tau / tau0
)


df["Q_relative_to_start"] = (
    Q_analog / Q0
)


df["spectral_ratio_relative_to_start"] = (
    df["omegaR_over_abs_omegaI"] / ratio0
)


# =====================================================================
# Candidate feature definitions
# =====================================================================

features = {

    # Analog mode number.
    #
    # IMPORTANT:
    # Not dimensionless in the sense relevant to this audit,
    # and NOT identified with Kerr m.
    "m_gv": m_gv,

    # Dimensionless.
    "Q": Q_analog,

    # Dimensionless.
    "omegaR_over_abs_omegaI":
        df["omegaR_over_abs_omegaI"].to_numpy(),

    # Dimensionless.
    "damping_ratio":
        df["damping_ratio"].to_numpy(),

    # Dimensionless ratios.
    "fR_relative_to_start":
        df["fR_relative_to_start"].to_numpy(),

    "tau_relative_to_start":
        df["tau_relative_to_start"].to_numpy(),

    "Q_relative_to_start":
        df["Q_relative_to_start"].to_numpy(),

    "spectral_ratio_relative_to_start":
        df["spectral_ratio_relative_to_start"].to_numpy(),
}


# =====================================================================
# Feature audit
# =====================================================================

audit_rows = []


for name, values in features.items():

    values = np.asarray(values, dtype=float)

    finite = values[np.isfinite(values)]

    if len(finite) == 0:
        continue

    dimensionless = (
        name != "m_gv"
    )

    audit_rows.append({

        "feature": name,

        "dimensionless":
            dimensionless,

        "n":
            int(len(finite)),

        "min":
            float(np.min(finite)),

        "max":
            float(np.max(finite)),

        "mean":
            float(np.mean(finite)),

        "std":
            float(np.std(finite)),

        "relative_span":
            float(relative_span(finite)),

        "monotonic_fraction_increasing":
            float(monotonic_fraction(finite)),

        "unique_values":
            int(unique_value_count(finite)),

        "correlation_with_m":
            correlation_with_m(m_gv, finite),
    })


audit = pd.DataFrame(audit_rows)


# =====================================================================
# Print dimensionless feature audit
# =====================================================================

print("\n" + "-" * 100)
print("DIMENSIONLESS FEATURE AUDIT")
print("-" * 100)


for _, row in audit.iterrows():

    print(
        f"{row['feature']:40s} "
        f"dimensionless="
        f"{str(bool(row['dimensionless'])):5s} "
        f"range=["
        f"{row['min']:.8g}, "
        f"{row['max']:.8g}] "
        f"span="
        f"{row['relative_span']:.6g} "
        f"mono="
        f"{row['monotonic_fraction_increasing']:.6f} "
        f"corr(m)="
        f"{row['correlation_with_m']:.6f}"
    )


# =====================================================================
# Internal spectral consistency
# =====================================================================

q_relative_error = (
    df["Q_consistency_relative_error"]
    .to_numpy()
)

finite_q_error = q_relative_error[
    np.isfinite(q_relative_error)
]


if len(finite_q_error) > 0:

    max_q_error = float(
        np.max(np.abs(finite_q_error))
    )

else:

    max_q_error = np.nan


print("\n" + "-" * 100)
print("INTERNAL SPECTRAL CONSISTENCY")
print("-" * 100)

print(
    "Relation tested:"
)

print(
    "    Q = pi * f_R * tau"
)

print(
    f"\nMaximum relative consistency error: "
    f"{max_q_error:.3e}"
)


if np.isfinite(max_q_error) and max_q_error < 1e-8:

    q_consistent = True

    print(
        "PASS: supplied Q is internally consistent "
        "with f_R and tau."
    )

else:

    q_consistent = False

    print(
        "WARNING: Q consistency exceeds "
        "the numerical tolerance."
    )


# =====================================================================
# Candidate uniqueness / monotonicity
# =====================================================================

q_values = df["Q"].to_numpy(
    dtype=float
)

ratio_values = df[
    "omegaR_over_abs_omegaI"
].to_numpy(
    dtype=float
)


q_mono = (
    monotonic_fraction(q_values)
    >= 0.999
)


ratio_mono = (
    monotonic_fraction(ratio_values)
    >= 0.999
)


q_unique = (
    unique_value_count(
        q_values,
        decimals=10
    )
    == len(q_values)
)


ratio_unique = (
    unique_value_count(
        ratio_values,
        decimals=10
    )
    == len(ratio_values)
)


candidate_features = []


if q_mono and q_unique:

    candidate_features.append(
        "Q"
    )


if ratio_mono and ratio_unique:

    candidate_features.append(
        "omegaR_over_abs_omegaI"
    )


# =====================================================================
# Scientific non-assumptions
# =====================================================================

print("\n" + "-" * 100)
print("SCIENTIFIC NON-ASSUMPTIONS")
print("-" * 100)

print(
    "1. m_gv is NOT identified with Kerr m."
)

print(
    "2. Giant-vortex circulation is NOT identified with Kerr a/M."
)

print(
    "3. Analog frequency scale is NOT identified with "
    "astrophysical frequency scale."
)

print(
    "4. No real GW event is used."
)

print(
    "5. No event-specific/post-hoc mapping is fitted."
)

print(
    "6. A dimensionless spectral similarity is NOT by itself "
    "a physical equivalence."
)


# =====================================================================
# Mapping interpretation
# =====================================================================

print("\n" + "-" * 100)
print("PRELIMINARY MAPPING INTERPRETATION")
print("-" * 100)


if candidate_features:

    print(
        "Dimensionless features with monotonic + injective "
        "behavior on the tested Root-B branch:"
    )

    for feature in candidate_features:

        print(
            f"    - {feature}"
        )

else:

    print(
        "No dimensionless feature satisfies the preliminary "
        "monotonic + injective criterion."
    )


print(
    "\nIMPORTANT:"
)

print(
    "This does NOT establish a Kerr mapping."
)

print(
    "It only establishes that the analog branch contains "
    "well-defined dimensionless spectral features."
)


# =====================================================================
# Save enriched observable table
# =====================================================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# =====================================================================
# JSON report
# =====================================================================

report = {

    "stage":
        "6.1",

    "title":
        "Lambda / Giant-Vortex to Kerr QNM Mapping Audit",

    "input":
        str(INPUT),

    "output_csv":
        str(OUTPUT_CSV),

    "output_json":
        str(OUTPUT_JSON),

    "n_validated_points":
        int(len(df)),

    "m_range":
        [
            float(df["m"].min()),
            float(df["m"].max())
        ],

    "last_validated_m":
        float(df["m"].min()),

    "residual_range":
        [
            float(df["abs_res"].min()),
            float(df["abs_res"].max())
        ],

    "q_internal_consistency":
        {
            "relation":
                "Q = pi * f_R * tau",

            "max_relative_error":
                float(max_q_error)
                if np.isfinite(max_q_error)
                else None,

            "pass":
                bool(q_consistent),
        },

    "candidate_dimensionless_features":
        candidate_features,

    "candidate_status":
        {

            "Q":
                {
                    "dimensionless":
                        True,

                    "injective_on_tested_branch":
                        bool(q_unique),

                    "monotonic_on_tested_branch":
                        bool(q_mono),

                    "physical_Kerr_mapping_established":
                        False,
                },

            "omegaR_over_abs_omegaI":
                {
                    "dimensionless":
                        True,

                    "injective_on_tested_branch":
                        bool(ratio_unique),

                    "monotonic_on_tested_branch":
                        bool(ratio_mono),

                    "physical_Kerr_mapping_established":
                        False,
                },
        },

    "explicit_non_assumptions":
        [

            "m_gv is not identified with Kerr m",

            "giant-vortex circulation is not identified "
            "with Kerr a/M",

            "analog frequency scale is not identified "
            "with astrophysical frequency scale",

            "no real GW event is used",

            "no post-hoc event-specific mapping is fitted",

            "dimensionless spectral similarity is not "
            "treated as physical equivalence",
        ],

    "scientific_conclusion":
        (
            "The Root-B branch provides well-defined "
            "dimensionless spectral features, including "
            "Q and omega_R/|omega_I|, that are internally "
            "consistent and monotonic over the validated "
            "branch. However, this audit does not establish "
            "a physical or unique mapping onto Kerr QNM "
            "parameter space. A physical correspondence "
            "requires an independently justified mapping "
            "between analog mode/rotation variables and "
            "Kerr dimensionless parameters (a/M, l, m, n), "
            "together with an independently justified "
            "frequency scale."
        ),

    "next_stage":
        (
            "Derive or falsify an independent physical "
            "mapping between giant-vortex variables and "
            "Kerr QNM parameters before using real GW "
            "strain data."
        ),
}


# =====================================================================
# Write JSON
# =====================================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


# =====================================================================
# Final verdict
# =====================================================================

print("\n" + "=" * 100)
print("STAGE 6.1 FINAL VERDICT")
print("=" * 100)

print(
    f"Validated Root-B points: {len(df)}"
)

print(
    f"Validated m range: "
    f"{df['m'].min():.6f} -> "
    f"{df['m'].max():.6f}"
)

print(
    f"Q internal consistency: "
    f"{'PASS' if q_consistent else 'FAIL'}"
)

print(
    "Candidate dimensionless features: "
    +
    (
        ", ".join(candidate_features)
        if candidate_features
        else "NONE"
    )
)

print()

print(
    "PHYSICAL KERR MAPPING: NOT ESTABLISHED"
)

print(
    "No unsupported identification of analog m, "
    "circulation, or frequency scale has been made."
)

print()

print(
    "This stage is an audit, not a Kerr-fit."
)

print()

print(
    f"CSV  -> {OUTPUT_CSV}"
)

print(
    f"JSON -> {OUTPUT_JSON}"
)

print("=" * 100)