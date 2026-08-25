#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
freeze_nw_observation_registry_v1_3.py

Corregge e congela la semantica canonica del registro osservativo dopo
l'error-probe del builder valori v1.0.

CORREZIONI SCIENTIFICHE:
1) ARPAL:
   i target *_level sono LIVELLO IDROMETRICO, non portata.
   La classificazione precedente era stata falsata dal testo sorgente
   "PORTATA - LIVELLO MEDIO DEL TORRENTE (m)", intercettato prima sulla
   parola "portata".
   Regola canonica:
       *_precip -> PRECIP_MM
       *_level  -> RIVER_STAGE_M

2) ARPA Piemonte daily_hydro:
   il file contiene sempre più colonne candidate
   (livellomedio, livellomedio1, livfreamedio, portatamedia, ...),
   ma alcune sono completamente vuote per una data stazione.
   La v1.2 marcava genericamente entrambe le famiglie.
   Qui si determinano, per ciascuna serie, le colonne REALMENTE numeriche
   nel periodo Sep-Dec 1987-2025 e si congela:
       active_level_columns
       active_discharge_columns
       variable_codes effettivi

NON modifica alcun dato sorgente.
NON modifica nw_observations_standardized_v1_2.

Output:
nw_observations_standardized_v1_3/
  observation_series_registry_v1_3.csv
  station_receptor_relations_v1_3.csv
  variable_dictionary_v1_3.csv
  observation_registry_audit_v1_3.json
  observation_registry_audit_v1_3.txt
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_ROWS = 1709
EXPECTED_DATA_OK = 1312
TARGET_MONTHS = {9, 10, 11, 12}

ANCILLARY_CODES = {
    "LEAF_WETNESS_DURATION_S",
    "REFLECTED_SOLAR_RAD_W_M2",
    "LEAF_WETNESS_LOWER_PCT",
    "LEAF_WETNESS_UPPER_PCT",
}

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


def parse_qc_column_list(value, actual_columns):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    raw = str(value).strip()
    if not raw:
        return []

    candidates = []

    if raw.startswith("[") and raw.endswith("]"):
        try:
            obj = ast.literal_eval(raw)
            if isinstance(obj, (list, tuple)):
                candidates.extend(str(x).strip() for x in obj)
        except Exception:
            pass

    if not candidates:
        cleaned = raw.strip("[](){}")
        parts = re.split(r"[|;,]+", cleaned)
        candidates.extend(
            p.strip().strip("'\"")
            for p in parts
            if p.strip()
        )

    actual = set(map(str, actual_columns))
    out = []

    for c in candidates:
        if c in actual and c not in out:
            out.append(c)

    if not out and raw in actual:
        out.append(raw)

    return out


def active_numeric_columns(df, date_mask, columns):
    out = []
    counts = {}

    for c in columns:
        vals = pd.to_numeric(df[c], errors="coerce")
        n = int((date_mask & vals.notna()).sum())
        counts[c] = n
        if n > 0:
            out.append(c)

    return out, counts


def code_role(code):
    if code in ANCILLARY_CODES:
        return "ANCILLARY"
    if code in CORE_CODES:
        return "CORE"
    if str(code).startswith("OTHER__"):
        return "UNRESOLVED"
    return "SECONDARY"


def roles_for_codes(s):
    roles = []
    for c in str(s or "").split("|"):
        c = c.strip()
        if not c:
            continue
        r = code_role(c)
        if r not in roles:
            roles.append(r)
    return "|".join(roles)


def main():
    root = Path(__file__).resolve().parent

    src = root / "nw_observations_standardized_v1_2"
    out = root / "nw_observations_standardized_v1_3"
    out.mkdir(parents=True, exist_ok=True)

    reg_p = src / "observation_series_registry_v1_2.csv"
    rel_p = src / "station_receptor_relations_v1_2.csv"
    audit_p = src / "observation_dictionary_audit_v1_2.json"

    pie_qc_p = (
        root / "observations_nw" / "piemonte"
        / "qc_final_v1_0" / "file_qc_final_v1_0.csv"
    )

    err_probe_p = (
        root / "nw_observations_values_v1_0"
        / "error_probe_v1_0"
        / "error_probe_v1_0.json"
    )

    print("=" * 132)
    print("NW OBSERVATIONS — FREEZE REGISTRY SEMANTICS v1.3")
    print("=" * 132)

    for p in [reg_p, rel_p, audit_p, pie_qc_p, err_probe_p]:
        if not p.exists():
            raise SystemExit(f"Manca: {p}")

    audit = json.loads(audit_p.read_text(encoding="utf-8"))
    err_probe = json.loads(err_probe_p.read_text(encoding="utf-8"))

    if audit.get("overall_status") != "PASS":
        raise SystemExit("Registro/dizionario v1.2 non PASS.")
    if err_probe.get("overall_status") != "PASS":
        raise SystemExit("Error probe v1.0 non PASS.")

    reg = pd.read_csv(reg_p, low_memory=False)
    rel = pd.read_csv(rel_p, low_memory=False)
    pie_qc = pd.read_csv(pie_qc_p, low_memory=False)

    reasons = []

    if len(reg) != EXPECTED_ROWS:
        reasons.append(
            f"REGISTRY_ROWS={len(reg)} expected={EXPECTED_ROWS}"
        )

    if int(reg["scientific_status"].astype(str).str.upper().eq("DATA_OK").sum()) != EXPECTED_DATA_OK:
        reasons.append("DATA_OK_TOTAL_MISMATCH")

    # New explicit columns.
    reg["active_level_columns"] = ""
    reg["active_discharge_columns"] = ""
    reg["semantic_correction_v1_3"] = ""

    # ------------------------------------------------------------------
    # ARPAL: target suffix is canonical semantic discriminator.
    # ------------------------------------------------------------------
    arpal = reg["provider"].eq("ARPAL")

    precip_mask = arpal & reg["target"].astype(str).str.endswith("_precip")
    level_mask = arpal & reg["target"].astype(str).str.endswith("_level")
    unknown_mask = arpal & ~(precip_mask | level_mask)

    if int(unknown_mask.sum()) != 0:
        reasons.append(
            f"ARPAL_UNKNOWN_TARGET_SUFFIX={int(unknown_mask.sum())}"
        )

    reg.loc[precip_mask, "variable_codes"] = "PRECIP_MM"
    reg.loc[level_mask, "variable_codes"] = "RIVER_STAGE_M"

    reg.loc[
        precip_mask,
        "semantic_correction_v1_3"
    ] = "arpal_target_suffix_precip"
    reg.loc[
        level_mask,
        "semantic_correction_v1_3"
    ] = "arpal_target_suffix_level"

    # Scientific check on DATA_OK ARPAL known from builder/error probe.
    arpal_ok = arpal & reg["scientific_status"].astype(str).str.upper().eq("DATA_OK")

    arpal_ok_precip = int(
        (arpal_ok & reg["variable_codes"].eq("PRECIP_MM")).sum()
    )
    arpal_ok_level = int(
        (arpal_ok & reg["variable_codes"].eq("RIVER_STAGE_M")).sum()
    )

    if arpal_ok_precip != 278:
        reasons.append(
            f"ARPAL_DATA_OK_PRECIP={arpal_ok_precip} expected=278"
        )
    if arpal_ok_level != 133:
        reasons.append(
            f"ARPAL_DATA_OK_LEVEL={arpal_ok_level} expected=133"
        )

    # ------------------------------------------------------------------
    # Piemonte hydro: derive actual active columns from numeric data.
    # ------------------------------------------------------------------
    pie_lookup = {}
    for _, q in pie_qc.iterrows():
        key = (
            str(q["kind"]).strip(),
            str(q["station_id"]).strip(),
        )
        pie_lookup[key] = q.to_dict()

    pie_hydro_mask = (
        reg["provider"].eq("ARPA_PIEMONTE")
        & reg["kind_source"].eq("daily_hydro")
    )

    hydro_class_counts = {
        "stage_and_discharge": 0,
        "stage_only": 0,
        "discharge_only": 0,
        "neither": 0,
    }

    hydro_details = []

    for idx in reg.index[pie_hydro_mask]:
        r = reg.loc[idx]

        key = (
            str(r["kind_source"]).strip(),
            str(r["station_id"]).strip(),
        )
        q = pie_lookup.get(key)

        if q is None:
            reasons.append(
                f"PIEMONTE_QC_NOT_FOUND:{key}"
            )
            continue

        source_path = Path(str(r["source_data_path"]))
        if not source_path.exists():
            reasons.append(
                f"PIEMONTE_SOURCE_MISSING:{source_path}"
            )
            continue

        df = pd.read_csv(source_path, low_memory=False)

        if "data" not in df.columns:
            reasons.append(
                f"PIEMONTE_DATA_COLUMN_MISSING:{key}"
            )
            continue

        dates = pd.to_datetime(
            df["data"],
            errors="coerce",
        )

        date_mask = (
            dates.notna()
            & dates.dt.year.between(1987, 2025)
            & dates.dt.month.isin(TARGET_MONTHS)
        )

        level_candidates = parse_qc_column_list(
            q.get("level_columns"),
            df.columns,
        )
        discharge_candidates = parse_qc_column_list(
            q.get("discharge_columns"),
            df.columns,
        )

        active_level, level_counts = active_numeric_columns(
            df,
            date_mask,
            level_candidates,
        )
        active_discharge, discharge_counts = active_numeric_columns(
            df,
            date_mask,
            discharge_candidates,
        )

        codes = []
        if active_level:
            codes.append("RIVER_STAGE_M")
        if active_discharge:
            codes.append("DISCHARGE_M3_S")

        if active_level and active_discharge:
            cls = "stage_and_discharge"
        elif active_level:
            cls = "stage_only"
        elif active_discharge:
            cls = "discharge_only"
        else:
            cls = "neither"

        hydro_class_counts[cls] += 1

        if not codes:
            reasons.append(
                f"PIEMONTE_HYDRO_NO_ACTIVE_CORE:{key}"
            )

        reg.at[idx, "variable_codes"] = "|".join(codes)
        reg.at[idx, "active_level_columns"] = "|".join(active_level)
        reg.at[idx, "active_discharge_columns"] = "|".join(active_discharge)
        reg.at[
            idx,
            "semantic_correction_v1_3"
        ] = "piemonte_numeric_active_core_columns_sepdec_1987_2025"

        hydro_details.append({
            "station_id": str(r["station_id"]),
            "station_name": str(r.get("station_name", "") or ""),
            "class": cls,
            "active_level_columns": active_level,
            "active_discharge_columns": active_discharge,
            "level_numeric_counts": level_counts,
            "discharge_numeric_counts": discharge_counts,
        })

    if sum(hydro_class_counts.values()) != int(pie_hydro_mask.sum()):
        reasons.append("PIEMONTE_HYDRO_CLASS_COUNT_MISMATCH")

    # DATA_OK rows may not have empty codes.
    empty_ok = int(
        (
            reg["scientific_status"].astype(str).str.upper().eq("DATA_OK")
            & reg["variable_codes"].fillna("").astype(str).str.strip().eq("")
        ).sum()
    )
    if empty_ok:
        reasons.append(
            f"DATA_OK_EMPTY_VARIABLE_CODES={empty_ok}"
        )

    # No OTHER allowed.
    other_rows = int(
        reg["variable_codes"]
        .fillna("")
        .astype(str)
        .str.contains(r"(?:^|\|)OTHER__", regex=True)
        .sum()
    )
    if other_rows:
        reasons.append(
            f"OTHER_ROWS={other_rows}"
        )

    reg["model_roles"] = reg["variable_codes"].map(roles_for_codes)

    # Rebuild dictionary from final registry.
    dict_rows = []
    for _, r in reg.iterrows():
        for code in str(r["variable_codes"] or "").split("|"):
            code = code.strip()
            if not code:
                continue

            dict_rows.append({
                "provider": r["provider"],
                "variable_code": code,
                "parameter_source": r["parameter_source"],
                "unit_source": r["unit_source"],
                "kind_source": r["kind_source"],
                "model_role": code_role(code),
            })

    vardict = (
        pd.DataFrame(dict_rows)
        .drop_duplicates()
        .sort_values(
            ["model_role", "variable_code", "provider", "parameter_source"]
        )
        .reset_index(drop=True)
    )

    unresolved = int(
        vardict["variable_code"].astype(str).str.startswith("OTHER__").sum()
    )
    if unresolved:
        reasons.append(
            f"UNRESOLVED_DICTIONARY_CODES={unresolved}"
        )

    # Output.
    reg_out = out / "observation_series_registry_v1_3.csv"
    rel_out = out / "station_receptor_relations_v1_3.csv"
    dict_out = out / "variable_dictionary_v1_3.csv"

    reg.to_csv(reg_out, index=False)
    rel.to_csv(rel_out, index=False)
    vardict.to_csv(dict_out, index=False)

    overall = "PASS" if not reasons else "REVIEW"

    report = {
        "version": "1.3",
        "overall_status": overall,
        "registry_rows": int(len(reg)),
        "data_ok_rows": int(
            reg["scientific_status"].astype(str).str.upper().eq("DATA_OK").sum()
        ),
        "arpal_data_ok": {
            "PRECIP_MM": arpal_ok_precip,
            "RIVER_STAGE_M": arpal_ok_level,
        },
        "piemonte_hydro_rows": int(pie_hydro_mask.sum()),
        "piemonte_hydro_semantic_classes": hydro_class_counts,
        "piemonte_hydro_details": hydro_details,
        "other_rows": other_rows,
        "variable_dictionary_rows": int(len(vardict)),
        "corrections": {
            "ARPAL": (
                "*_precip -> PRECIP_MM; *_level -> RIVER_STAGE_M. "
                "The old DISCHARGE classification for *_level is invalid."
            ),
            "ARPA_PIEMONTE_daily_hydro": (
                "variable_codes and active source columns are derived from "
                "actual numeric Sep-Dec 1987-2025 values, not from header presence."
            ),
        },
        "raw_modified": False,
        "reasons": reasons,
        "next_step": (
            "Run standardized values builder v1.1, which reuses all compatible "
            "PASS outputs from v1.0 and processes only the previously failed series."
        ),
    }

    (out / "observation_registry_audit_v1_3.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "=" * 132,
        "NW OBSERVATIONS — FREEZE REGISTRY SEMANTICS v1.3",
        "=" * 132,
        f"OVERALL STATUS                : {overall}",
        f"Registry rows                 : {len(reg)} / {EXPECTED_ROWS}",
        f"DATA_OK rows                  : {int(reg['scientific_status'].astype(str).str.upper().eq('DATA_OK').sum())} / {EXPECTED_DATA_OK}",
        f"OTHER rows                    : {other_rows}",
        "",
        "ARPAL DATA_OK SEMANTICS",
        f"PRECIP_MM                     : {arpal_ok_precip} / 278",
        f"RIVER_STAGE_M                 : {arpal_ok_level} / 133",
        "",
        "PIEMONTE DAILY_HYDRO SEMANTICS",
        f"total                         : {int(pie_hydro_mask.sum())}",
        f"stage + discharge             : {hydro_class_counts['stage_and_discharge']}",
        f"stage only                    : {hydro_class_counts['stage_only']}",
        f"discharge only                : {hydro_class_counts['discharge_only']}",
        f"neither                       : {hydro_class_counts['neither']}",
        "",
        f"Registry : {reg_out}",
        f"Relations: {rel_out}",
        f"Dict     : {dict_out}",
    ]

    (out / "observation_registry_audit_v1_3.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines[3:]))
    print("\n" + "=" * 132)
    print(f"OVERALL STATUS : {overall}")
    print(f"Output         : {out}")

    if reasons:
        print("REASONS:")
        for r in reasons:
            print(f"  - {r}")

    print("=" * 132)

    if overall != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
