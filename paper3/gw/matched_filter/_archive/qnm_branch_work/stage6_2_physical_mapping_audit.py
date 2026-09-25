"""
Stage 6.2 -- Physical Mapping Audit
Lambda / Giant-Vortex QNM -> Kerr QNM

Conservative physical-mapping audit.

This script DOES NOT:
- use real GW strain data
- fit Kerr parameters
- fit a/M from analog QNM frequencies
- identify m_gv with Kerr m
- identify circulation with a/M
- calibrate a frequency scale to a GW event
- accept numerical similarity as physical evidence

A physical mapping is accepted only if it is independently derived
from equations, symmetry, conservation laws, boundary conditions,
dimensional analysis, or another explicit physical argument.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

HERE = Path(__file__).resolve().parent

INPUT = HERE / "stage5K_1_observables_rootB.csv"

OUTPUT_CSV = HERE / "stage6_2_physical_mapping_audit.csv"
OUTPUT_JSON = HERE / "stage6_2_physical_mapping_audit.json"


# ============================================================================
# PHYSICAL MAPPING EVIDENCE
# ============================================================================
#
# IMPORTANT:
#
# These are deliberately FALSE.
#
# They may only be changed after an independent physical derivation
# has been established and documented.
#
# Numerical similarity or post-hoc fitting does NOT qualify.
# ============================================================================


@dataclass
class PhysicalMappingEvidence:

    # 1. Mode mapping
    mode_mapping_derived: bool = False

    mode_mapping_source: str = (
        "NOT PROVIDED: no independent derivation connecting "
        "giant-vortex mode number to Kerr (l,m,n)"
    )

    mode_mapping_equation: str = ""

    # 2. Rotation mapping
    rotation_mapping_derived: bool = False

    rotation_mapping_source: str = (
        "NOT PROVIDED: no independent derivation connecting "
        "giant-vortex circulation/rotation to Kerr a/M"
    )

    rotation_mapping_equation: str = ""

    # 3. Frequency scale
    frequency_scale_derived: bool = False

    frequency_scale_source: str = (
        "NOT PROVIDED: no independent derivation of "
        "analog-to-Kerr frequency scale"
    )

    frequency_scale_equation: str = ""

    # 4. Overtone / QNM index
    overtone_mapping_derived: bool = False

    overtone_mapping_source: str = (
        "NOT PROVIDED: no independent derivation of "
        "analog QNM index -> Kerr overtone n"
    )

    overtone_mapping_equation: str = ""

    # 5. Multi-feature relation
    multifature_relation_derived: bool = False

    multifature_relation_source: str = (
        "NOT PROVIDED: no independently derived multi-feature "
        "relation between analog and Kerr QNM observables"
    )

    multifature_relation_equation: str = ""

    # 6. Normalization independence
    normalization_independent: bool = False

    normalization_source: str = (
        "NOT PROVIDED: normalization independence has not "
        "been independently demonstrated"
    )

    # 7. Uniqueness
    unique_mapping_derived: bool = False

    uniqueness_source: str = (
        "NOT PROVIDED: uniqueness of the analog -> Kerr mapping "
        "has not been independently demonstrated"
    )


# ============================================================================
# HEADER
# ============================================================================

print("=" * 100)
print("STAGE 6.2 -- PHYSICAL MAPPING AUDIT")
print("Λ / GIANT-VORTEX -> KERR QNM")
print("=" * 100)


# ============================================================================
# INPUT
# ============================================================================

if not INPUT.exists():
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT}\n\n"
        "Run Stage 5K.1 Root B first."
    )

df = pd.read_csv(INPUT)

print(f"\nInput file: {INPUT.name}")

print("\nDetected columns:")
print(list(df.columns))


# ============================================================================
# COLUMN NORMALIZATION
# ============================================================================

COLUMN_ALIASES = {
    "f_R (Hz)": "f_R_hz",
    "tau (s)": "tau_s",
    "|Res|": "abs_res",
}

df = df.rename(columns=COLUMN_ALIASES)


required = [
    "m",
    "f_re",
    "f_im",
    "f_R_hz",
    "tau_s",
    "Q",
    "abs_res",
]

missing = [
    column
    for column in required
    if column not in df.columns
]

if missing:
    raise ValueError(
        "\nMissing required columns:\n"
        f"{missing}\n\n"
        "Available columns:\n"
        f"{list(df.columns)}"
    )


df = df[required].copy()

for column in required:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.dropna().reset_index(drop=True)

if len(df) == 0:
    raise ValueError(
        "No valid numerical Root-B rows remain."
    )


# ============================================================================
# ROOT-B NUMERICAL VALIDATION
# ============================================================================

m_gv = df["m"].to_numpy(dtype=float)

f_R = df["f_R_hz"].to_numpy(dtype=float)

tau = df["tau_s"].to_numpy(dtype=float)

Q = df["Q"].to_numpy(dtype=float)

abs_res = df["abs_res"].to_numpy(dtype=float)


omega_R = 2.0 * np.pi * f_R

omega_I_abs = 1.0 / tau

omega_ratio = omega_R / omega_I_abs

Q_from_ftau = np.pi * f_R * tau

Q_relative_error = np.abs(
    (Q_from_ftau - Q) / Q
)


print("\n" + "-" * 100)
print("ROOT-B NUMERICAL INPUT VALIDATION")
print("-" * 100)

print(
    f"Validated points: {len(df)}"
)

print(
    f"m range: "
    f"[{np.min(m_gv):.6f}, {np.max(m_gv):.6f}]"
)

print(
    f"f_R range: "
    f"[{np.min(f_R):.9f}, {np.max(f_R):.9f}] Hz"
)

print(
    f"tau range: "
    f"[{np.min(tau):.9f}, {np.max(tau):.9f}] s"
)

print(
    f"Q range: "
    f"[{np.min(Q):.9f}, {np.max(Q):.9f}]"
)

print(
    f"Residual range: "
    f"[{np.min(abs_res):.3e}, "
    f"{np.max(abs_res):.3e}]"
)

print(
    f"Max Q consistency error: "
    f"{np.max(Q_relative_error):.3e}"
)


q_consistent = bool(
    np.max(Q_relative_error) < 1e-10
)


# ============================================================================
# DIMENSIONLESS OBSERVABLES
# ============================================================================

f_relative = f_R / f_R[0]

tau_relative = tau / tau[0]

Q_relative = Q / Q[0]

omega_ratio_relative = (
    omega_ratio / omega_ratio[0]
)

damping_ratio = (
    omega_I_abs / omega_R
)


feature_data = {

    "Q": Q,

    "omegaR_over_abs_omegaI":
        omega_ratio,

    "damping_ratio":
        damping_ratio,

    "fR_relative":
        f_relative,

    "tau_relative":
        tau_relative,

    "Q_relative":
        Q_relative,

    "omega_ratio_relative":
        omega_ratio_relative,
}


# ============================================================================
# FEATURE SUMMARY
# ============================================================================

feature_rows = []

for name, values in feature_data.items():

    values = np.asarray(
        values,
        dtype=float
    )

    finite = values[
        np.isfinite(values)
    ]

    if len(finite) == 0:
        continue

    feature_rows.append({

        "feature": name,

        "dimensionless": True,

        "n": int(len(finite)),

        "min": float(np.min(finite)),

        "max": float(np.max(finite)),

        "mean": float(np.mean(finite)),

        "std": float(np.std(finite)),

        "range":
            float(
                np.max(finite)
                -
                np.min(finite)
            ),

    })


features_df = pd.DataFrame(
    feature_rows
)


print("\n" + "-" * 100)
print("AVAILABLE DIMENSIONLESS ANALOG FEATURES")
print("-" * 100)

for _, row in features_df.iterrows():

    print(
        f"{row['feature']:35s} "
        f"range=["
        f"{row['min']:.9g}, "
        f"{row['max']:.9g}] "
        f"span="
        f"{row['range']:.9g}"
    )


# ============================================================================
# PHYSICAL EVIDENCE
# ============================================================================

evidence = PhysicalMappingEvidence()


# ============================================================================
# CRITERIA
# ============================================================================

print("\n" + "-" * 100)
print("INDEPENDENT PHYSICAL EVIDENCE AUDIT")
print("-" * 100)


criteria = {

    "mode_mapping":
        evidence.mode_mapping_derived,

    "rotation_mapping":
        evidence.rotation_mapping_derived,

    "frequency_scale":
        evidence.frequency_scale_derived,

    "overtone_mapping":
        evidence.overtone_mapping_derived,

    "multi_feature_relation":
        evidence.multifature_relation_derived,

    "normalization_independence":
        evidence.normalization_independent,

    "unique_mapping":
        evidence.unique_mapping_derived,

}


for name, status in criteria.items():

    print(
        f"{name:30s}: "
        f"{'PASS' if status else 'NOT ESTABLISHED'}"
    )


# ============================================================================
# REJECT POST-HOC NUMERICAL MAPPING
# ============================================================================

print("\n" + "-" * 100)
print("NUMERICAL-SIMILARITY SAFETY CHECK")
print("-" * 100)

print(
    "A numerical match between analog QNM observables and Kerr "
    "QNM observables is NOT counted as physical evidence."
)

print(
    "A best-fit (a/M,l,m,n) obtained from the analog QNM values "
    "would be classified as POST-HOC CALIBRATION."
)

print(
    "A frequency scale chosen to match a GW event would be "
    "classified as EVENT-CALIBRATED and rejected."
)

print(
    "Identification m_gv = Kerr m is rejected unless independently derived."
)

print(
    "Identification circulation = a/M is rejected unless independently derived."
)


# ============================================================================
# DECISION LOGIC
# ============================================================================

all_core_physics = all([
    evidence.mode_mapping_derived,
    evidence.rotation_mapping_derived,
    evidence.frequency_scale_derived,
    evidence.overtone_mapping_derived,
])


all_validation = all([
    evidence.multifature_relation_derived,
    evidence.normalization_independent,
    evidence.unique_mapping_derived,
])


if all_core_physics and all_validation:

    final_status = (
        "PHYSICAL_MAPPING_ESTABLISHED"
    )

elif any(criteria.values()):

    final_status = (
        "PARTIAL_MAPPING_ONLY"
    )

else:

    final_status = (
        "NO_PHYSICAL_MAPPING"
    )


# ============================================================================
# AUDIT TABLE
# ============================================================================

audit_rows = [

    {
        "criterion":
            "Mode mapping m_gv -> Kerr (l,m,n)",

        "status":
            evidence.mode_mapping_derived,

        "physical_evidence":
            evidence.mode_mapping_source,

        "equation":
            evidence.mode_mapping_equation,
    },

    {
        "criterion":
            "Rotation mapping -> Kerr a/M",

        "status":
            evidence.rotation_mapping_derived,

        "physical_evidence":
            evidence.rotation_mapping_source,

        "equation":
            evidence.rotation_mapping_equation,
    },

    {
        "criterion":
            "Independent frequency scale",

        "status":
            evidence.frequency_scale_derived,

        "physical_evidence":
            evidence.frequency_scale_source,

        "equation":
            evidence.frequency_scale_equation,
    },

    {
        "criterion":
            "QNM overtone/index mapping",

        "status":
            evidence.overtone_mapping_derived,

        "physical_evidence":
            evidence.overtone_mapping_source,

        "equation":
            evidence.overtone_mapping_equation,
    },

    {
        "criterion":
            "Multi-feature physical relation",

        "status":
            evidence.multifature_relation_derived,

        "physical_evidence":
            evidence.multifature_relation_source,

        "equation":
            evidence.multifature_relation_equation,
    },

    {
        "criterion":
            "Normalization independent",

        "status":
            evidence.normalization_independent,

        "physical_evidence":
            evidence.normalization_source,

        "equation":
            "",
    },

    {
        "criterion":
            "Unique mapping",

        "status":
            evidence.unique_mapping_derived,

        "physical_evidence":
            evidence.uniqueness_source,

        "equation":
            "",
    },

]


audit_df = pd.DataFrame(
    audit_rows
)


# ============================================================================
# SAVE CSV
# ============================================================================

audit_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================================
# JSON REPORT
# ============================================================================

report = {

    "stage":
        "6.2",

    "title":
        "Physical Mapping Audit: "
        "Lambda / Giant-Vortex -> Kerr QNM",

    "input":
        str(INPUT),

    "output_csv":
        str(OUTPUT_CSV),

    "output_json":
        str(OUTPUT_JSON),

    "rootB_validation": {

        "n_points":
            int(len(df)),

        "m_range": [
            float(np.min(m_gv)),
            float(np.max(m_gv)),
        ],

        "f_R_range_Hz": [
            float(np.min(f_R)),
            float(np.max(f_R)),
        ],

        "tau_range_s": [
            float(np.min(tau)),
            float(np.max(tau)),
        ],

        "Q_range": [
            float(np.min(Q)),
            float(np.max(Q)),
        ],

        "residual_range": [
            float(np.min(abs_res)),
            float(np.max(abs_res)),
        ],

        "Q_consistency_max_relative_error":
            float(
                np.max(Q_relative_error)
            ),

        "Q_consistency_pass":
            q_consistent,
    },

    "dimensionless_features": {
        name: [
            float(x)
            for x in values
        ]
        for name, values
        in feature_data.items()
    },

    "physical_evidence":
        asdict(evidence),

    "decision_criteria":
        criteria,

    "final_status":
        final_status,

    "rejected_shortcuts": [

        "m_gv = Kerr m",

        "giant-vortex circulation = Kerr a/M",

        "analog frequency scale = astrophysical Kerr scale",

        "best-fit Kerr parameters from analog QNM values",

        "event-calibrated frequency mapping",

        "post-hoc GW-event fitting",

        "dimensionless spectral similarity as proof of physical equivalence",
    ],

    "scientific_conclusion": (
        "The Root-B branch is numerically validated and possesses "
        "well-defined dimensionless spectral observables. A physical "
        "mapping to Kerr QNM parameter space is accepted only when "
        "independently derived from model equations, symmetry, "
        "conservation laws, boundary conditions, dimensional analysis, "
        "or another explicit physical argument. Numerical similarity "
        "or post-hoc fitting is not accepted as evidence."
    ),

    "stage6_policy": {

        "real_GW_fit_allowed":
            final_status ==
            "PHYSICAL_MAPPING_ESTABLISHED",

        "post_hoc_mapping_allowed":
            False,

        "event_calibration_allowed":
            False,

        "unsupported_mode_identification_allowed":
            False,
    },

    "final_verdict_text": (
        "NO_PHYSICAL_MAPPING: no independently derived physical "
        "correspondence between the giant-vortex Root-B variables "
        "and Kerr parameters has been established."
        if final_status == "NO_PHYSICAL_MAPPING"
        else
        f"{final_status}: see detailed criteria in this report."
    ),
}


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================================
# FINAL REPORT
# ============================================================================

print("\n" + "=" * 100)
print("STAGE 6.2 FINAL VERDICT")
print("=" * 100)

print(
    f"\nRoot-B points: {len(df)}"
)

print(
    f"Q internal consistency: "
    f"{'PASS' if q_consistent else 'FAIL'}"
)

print("\nIndependent physical mapping:")

print(
    f"  Mode mapping: "
    f"{'ESTABLISHED' if evidence.mode_mapping_derived else 'NOT ESTABLISHED'}"
)

print(
    f"  Rotation mapping: "
    f"{'ESTABLISHED' if evidence.rotation_mapping_derived else 'NOT ESTABLISHED'}"
)

print(
    f"  Frequency scale: "
    f"{'ESTABLISHED' if evidence.frequency_scale_derived else 'NOT ESTABLISHED'}"
)

print(
    f"  Overtone mapping: "
    f"{'ESTABLISHED' if evidence.overtone_mapping_derived else 'NOT ESTABLISHED'}"
)

print(
    f"  Multi-feature relation: "
    f"{'ESTABLISHED' if evidence.multifature_relation_derived else 'NOT ESTABLISHED'}"
)

print(
    f"  Normalization independence: "
    f"{'ESTABLISHED' if evidence.normalization_independent else 'NOT ESTABLISHED'}"
)

print(
    f"  Uniqueness: "
    f"{'ESTABLISHED' if evidence.unique_mapping_derived else 'NOT ESTABLISHED'}"
)


print("\n" + "-" * 100)

print(
    f"FINAL STATUS: {final_status}"
)


if final_status == "NO_PHYSICAL_MAPPING":

    print(
        "\nNo independently justified physical Kerr mapping "
        "has been established."
    )

    print(
        "The Root-B QNM branch remains a valid numerical result "
        "within the Lambda-model."
    )

    print(
        "No real-GW interpretation is authorized by this stage."
    )


elif final_status == "PARTIAL_MAPPING_ONLY":

    print(
        "\nSome physical correspondences are documented, "
        "but the complete mapping is not established."
    )

    print(
        "Do NOT proceed to event-level GW fitting."
    )


elif final_status == "PHYSICAL_MAPPING_ESTABLISHED":

    print(
        "\nAll required physical mapping criteria are marked "
        "as independently established."
    )

    print(
        "A separate Stage 6.3 Kerr-QNM comparison may now be considered."
    )


print("\n" + "-" * 100)

print("IMPORTANT SCIENTIFIC RULE:")

print(
    "A mapping discovered only because it makes the analog QNM "
    "numbers agree with Kerr is NOT accepted."
)

print(
    "A mapping calibrated on a real GW event is NOT accepted."
)

print(
    "A mapping must be independently derived before observational testing."
)


print("\n" + "-" * 100)

print(f"CSV  -> {OUTPUT_CSV}")

print(f"JSON -> {OUTPUT_JSON}")

print("=" * 100)