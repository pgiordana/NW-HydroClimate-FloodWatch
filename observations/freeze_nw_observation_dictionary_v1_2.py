#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
freeze_nw_observation_dictionary_v1_2.py

Congela il dizionario canonico delle variabili osservate dopo il probe
dei 7 codici OTHER__* residui.

Input:
nw_observations_standardized_v1_1/
  observation_series_registry_v1_1.csv
  station_receptor_relations_v1_1.csv
  variable_dictionary_v1_1.csv
  observation_registry_audit_v1_1.json
  other_variable_probe_v1_0/other_variable_probe_v1_0.json

Mappature residue:
OTHER__bagnatura_fogliare
    -> LEAF_WETNESS_DURATION_S
OTHER__radiazione_riflessa
    -> REFLECTED_SOLAR_RAD_W_M2
OTHER__bagnatura_foglia_inferiore
    -> LEAF_WETNESS_LOWER_PCT
OTHER__bagnatura_foglia_superiore
    -> LEAF_WETNESS_UPPER_PCT

Le quattro variabili sono classificate ANCILLARY: vengono preservate nel
dataset osservativo, ma non sono considerate predittori core del modello
idroclimatico salvo analisi successive.

Output:
nw_observations_standardized_v1_2/
  observation_series_registry_v1_2.csv
  station_receptor_relations_v1_2.csv
  variable_dictionary_v1_2.csv
  observation_dictionary_audit_v1_2.json
  observation_dictionary_audit_v1_2.txt

Non modifica alcun dato sorgente.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd


EXPECTED_ROWS = 1709

REMAP = {
    "OTHER__bagnatura_fogliare": "LEAF_WETNESS_DURATION_S",
    "OTHER__radiazione_riflessa": "REFLECTED_SOLAR_RAD_W_M2",
    "OTHER__bagnatura_foglia_inferiore": "LEAF_WETNESS_LOWER_PCT",
    "OTHER__bagnatura_foglia_superiore": "LEAF_WETNESS_UPPER_PCT",
}

ANCILLARY_CODES = set(REMAP.values())

CORE_CODES = {
    "PRECIP_MM",
    "RIVER_STAGE_M",
    "DISCHARGE_M3_S",
    "DISCHARGE_MIN_M3_S",
    "DISCHARGE_MAX_M3_S",
    "AIR_TEMP_C",
    "REL_HUMIDITY_PCT",
    "WIND_SPEED_M_S",
    "WIND_DIR_DEG",
    "AIR_PRESSURE_HPA",
    "SNOW_DEPTH_CM",
    "SOLAR_RAD_W_M2",
    "SUNSHINE_DURATION_MIN",
    "SOIL_MOISTURE_SOURCE_UNIT",
    "EVAPORATION_SOURCE_UNIT",
}


def remap_codes(value):
    parts = [
        p.strip()
        for p in str(value or "").split("|")
        if p.strip()
    ]
    out = []
    for p in parts:
        q = REMAP.get(p, p)
        if q not in out:
            out.append(q)
    return "|".join(out)


def code_role(code):
    if code in ANCILLARY_CODES:
        return "ANCILLARY"
    if code in CORE_CODES:
        return "CORE"
    if code.startswith("OTHER__"):
        return "UNRESOLVED"
    return "SECONDARY"


def main():
    root = Path(__file__).resolve().parent

    src = root / "nw_observations_standardized_v1_1"
    out = root / "nw_observations_standardized_v1_2"
    out.mkdir(parents=True, exist_ok=True)

    reg_p = src / "observation_series_registry_v1_1.csv"
    rel_p = src / "station_receptor_relations_v1_1.csv"
    var_p = src / "variable_dictionary_v1_1.csv"
    audit_p = src / "observation_registry_audit_v1_1.json"
    probe_p = (
        src / "other_variable_probe_v1_0"
        / "other_variable_probe_v1_0.json"
    )

    print("=" * 128)
    print("NW OBSERVATIONS — FREEZE VARIABLE DICTIONARY v1.2")
    print("=" * 128)

    for p in [reg_p, rel_p, var_p, audit_p, probe_p]:
        if not p.exists():
            raise SystemExit(f"Manca: {p}")

    old_audit = json.loads(audit_p.read_text(encoding="utf-8"))
    probe = json.loads(probe_p.read_text(encoding="utf-8"))

    if old_audit.get("overall_status") != "PASS":
        raise SystemExit("Registro v1.1 non PASS.")

    if probe.get("overall_status") != "PASS":
        raise SystemExit("Probe OTHER v1.0 non PASS.")

    if int(probe.get("other_rows", -1)) != 7:
        raise SystemExit(
            f"Numero OTHER inatteso: {probe.get('other_rows')}"
        )

    reg = pd.read_csv(reg_p, low_memory=False)
    rel = pd.read_csv(rel_p, low_memory=False)
    vardict = pd.read_csv(var_p, low_memory=False)

    reasons = []

    if len(reg) != EXPECTED_ROWS:
        reasons.append(
            f"registry_rows={len(reg)} expected={EXPECTED_ROWS}"
        )

    before_other = int(
        reg["variable_codes"]
        .astype(str)
        .str.contains(r"(?:^|\|)OTHER__", regex=True, na=False)
        .sum()
    )

    reg["variable_codes"] = reg["variable_codes"].map(remap_codes)

    after_other = int(
        reg["variable_codes"]
        .astype(str)
        .str.contains(r"(?:^|\|)OTHER__", regex=True, na=False)
        .sum()
    )

    if before_other != 7:
        reasons.append(f"before_other_rows={before_other} expected=7")
    if after_other != 0:
        reasons.append(f"after_other_rows={after_other} expected=0")

    # Exploded variable dictionary from the final registry.
    exploded = []
    for _, r in reg.iterrows():
        for code in str(r["variable_codes"]).split("|"):
            code = code.strip()
            if not code:
                continue
            exploded.append({
                "provider": r["provider"],
                "variable_code": code,
                "parameter_source": r["parameter_source"],
                "unit_source": r["unit_source"],
                "kind_source": r["kind_source"],
                "model_role": code_role(code),
            })

    newdict = (
        pd.DataFrame(exploded)
        .drop_duplicates()
        .sort_values(
            [
                "model_role",
                "variable_code",
                "provider",
                "parameter_source",
            ]
        )
        .reset_index(drop=True)
    )

    unresolved_dict = int(
        newdict["variable_code"]
        .astype(str)
        .str.startswith("OTHER__", na=False)
        .sum()
    )
    if unresolved_dict:
        reasons.append(
            f"unresolved_dictionary_codes={unresolved_dict}"
        )

    # Add a row-level model role summary.
    def roles_for_codes(s):
        roles = []
        for c in str(s).split("|"):
            c = c.strip()
            if not c:
                continue
            r = code_role(c)
            if r not in roles:
                roles.append(r)
        return "|".join(roles)

    reg["model_roles"] = reg["variable_codes"].map(roles_for_codes)

    # Preserve relation table unchanged semantically.
    reg_out = out / "observation_series_registry_v1_2.csv"
    rel_out = out / "station_receptor_relations_v1_2.csv"
    dict_out = out / "variable_dictionary_v1_2.csv"

    reg.to_csv(reg_out, index=False)
    rel.to_csv(rel_out, index=False)
    newdict.to_csv(dict_out, index=False)

    # Audit ancillary rows.
    ancillary_rows = int(
        reg["variable_codes"]
        .astype(str)
        .apply(
            lambda s: any(
                c.strip() in ANCILLARY_CODES
                for c in s.split("|")
            )
        )
        .sum()
    )

    core_rows = int(
        reg["model_roles"]
        .astype(str)
        .str.contains(r"(?:^|\|)CORE(?:$|\|)", regex=True, na=False)
        .sum()
    )

    provider_counts = {
        str(k): int(v)
        for k, v in reg["provider"].value_counts().to_dict().items()
    }

    overall = "PASS" if not reasons else "REVIEW"

    report = {
        "version": "1.2",
        "overall_status": overall,
        "registry_rows": int(len(reg)),
        "provider_counts": provider_counts,
        "other_rows_before": before_other,
        "other_rows_after": after_other,
        "ancillary_rows": ancillary_rows,
        "core_rows": core_rows,
        "variable_dictionary_rows": int(len(newdict)),
        "remap": REMAP,
        "model_role_policy": {
            "CORE": "candidate hydroclimate core variable",
            "ANCILLARY": (
                "preserved in standardized observations but not part of "
                "the default hydrological predictor set"
            ),
            "SECONDARY": (
                "standardized but not yet assigned to default core predictor set"
            ),
        },
        "timezone_policy": {
            "Piemonte": "DAILY_SOURCE_DATE",
            "Valle_d_Aosta": "UNRESOLVED_SOURCE_TIME_CONVENTION",
            "Liguria": "PORTAL_DECLARED_UTC_PRESERVE_PROVENANCE",
        },
        "next_step": (
            "Probe representative DATA_OK source-file schemas for Piemonte, "
            "Valle d'Aosta and Liguria, then build provider-aware long-value ingestion."
        ),
        "reasons": reasons,
        "raw_modified": False,
    }

    (out / "observation_dictionary_audit_v1_2.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "=" * 128,
        "NW OBSERVATIONS — FREEZE VARIABLE DICTIONARY v1.2",
        "=" * 128,
        f"OVERALL STATUS       : {overall}",
        f"Registry rows        : {len(reg)} / {EXPECTED_ROWS}",
        f"OTHER rows before    : {before_other}",
        f"OTHER rows after     : {after_other}",
        f"Ancillary rows       : {ancillary_rows}",
        f"Core rows            : {core_rows}",
        f"Variable dictionary  : {len(newdict)}",
        "",
        "RESIDUAL REMAP",
        "Bagnatura fogliare            -> LEAF_WETNESS_DURATION_S",
        "Radiazione riflessa           -> REFLECTED_SOLAR_RAD_W_M2",
        "Bagnatura foglia inferiore    -> LEAF_WETNESS_LOWER_PCT",
        "Bagnatura foglia superiore    -> LEAF_WETNESS_UPPER_PCT",
        "",
        "MODEL POLICY",
        "Le 4 variabili sopra sono ANCILLARY: preservate, non core per default.",
        "",
        f"Registry : {reg_out}",
        f"Relations: {rel_out}",
        f"Dict     : {dict_out}",
    ]

    (out / "observation_dictionary_audit_v1_2.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines[3:]))
    print("\n" + "=" * 128)
    print(f"OVERALL STATUS : {overall}")
    print(f"Output         : {out}")

    if reasons:
        print("REASONS:")
        for r in reasons:
            print(f"  - {r}")

    print("=" * 128)

    if overall != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
