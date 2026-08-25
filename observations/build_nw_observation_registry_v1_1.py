#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_nw_observation_registry_v1_1.py

Registro canonico corretto delle osservazioni regionali per nw_hydroclimate.

Correzione principale rispetto a v1.0:
- Piemonte NON possiede una colonna "parameter" nel file QC finale.
- Il file QC finale descrive una serie/file tramite:
    kind
    station_id
    scientific_status
    source_csv
    ptot_columns
    level_columns
    discharge_columns
    receptor_ids
- La semantica variabile Piemonte viene quindi ricostruita ESPLICITAMENTE:
    daily_meteo + ptot_columns       -> PRECIP_MM
    daily_hydro + level_columns      -> RIVER_STAGE_M
    daily_hydro + discharge_columns  -> DISCHARGE_M3_S
  Una singola serie idrologica può contenere livello e portata insieme;
  il registro resta a 1 riga per serie/file e usa `variable_codes`
  separati da |.

Righe attese:
  Piemonte      407
  Valle d'Aosta 522
  Liguria       780
  TOTALE       1709

Output:
nw_observations_standardized_v1_1/
  observation_series_registry_v1_1.csv
  station_receptor_relations_v1_1.csv
  variable_dictionary_v1_1.csv
  observation_registry_audit_v1_1.json
  observation_registry_audit_v1_1.txt

Non modifica alcun dato sorgente.
Non converte timestamp.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED = {
    "PIE_ROWS": 407,
    "PIE_OK": 397,
    "PIE_NO": 10,
    "VDA_ROWS": 522,
    "VDA_OK": 504,
    "VDA_NO": 18,
    "LIG_ROWS": 780,
    "LIG_OK": 411,
    "LIG_NO": 369,
}

LIG_PREFIX_TO_RECEPTOR = {
    "bisagno": "LIG_BISAGNO",
    "polcevera": "LIG_POLCEVERA",
    "entella": "LIG_ENTELLA",
    "magra": "LIG_MAGRA",
    "centa": "LIG_CENTA",
}


def ascii_norm(x):
    s = str(x or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s


def slug(x):
    s = ascii_norm(x)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "unknown"


def norm_status(x):
    return str(x or "").strip().upper()


def find_col(df, candidates, contains=()):
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    for token in contains:
        for c in df.columns:
            if token.lower() in str(c).lower():
                return c
    return None


def find_status_col(df):
    return find_col(
        df,
        ["scientific_status", "final_status", "status"],
        contains=["status"],
    )


def detect_path_col(df):
    candidates = []
    for c in df.columns:
        cl = str(c).lower()
        if any(tok in cl for tok in ["normalized_output", "source_csv", "output", "path", "file"]):
            vals = df[c].dropna().astype(str)
            if len(vals) == 0:
                continue
            score = vals.str.contains(
                r"\.(?:csv|csv\.gz|gz|txt)$",
                case=False,
                regex=True,
            ).mean()
            candidates.append((float(score), c))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1] if candidates[0][0] > 0 else None


def resolve_source_path(root: Path, value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = str(value).strip()
    if not s:
        return ""
    p = Path(s)
    if p.is_absolute():
        return str(p)
    return str(root / p)


def parse_receptor_ids(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    if not s:
        return []
    # Prima usa separatori espliciti: il QC Piemonte scrive " | ".
    parts = [p.strip() for p in re.split(r"[|,;]+", s) if p.strip()]
    out = []
    for p in parts:
        if p not in out:
            out.append(p)
    return out


def classify_variable(parameter, unit="", extra_text=""):
    t = " ".join(
        ascii_norm(x)
        for x in [parameter, unit, extra_text]
        if str(x or "").strip()
    )

    if "portata" in t or "discharge" in t:
        if "max" in t:
            return "DISCHARGE_MAX_M3_S"
        if "min" in t:
            return "DISCHARGE_MIN_M3_S"
        return "DISCHARGE_M3_S"

    if any(k in t for k in [
        "livello idrometrico",
        "altezza idrometrica",
        "livello medio del torrente",
        "river level",
        "stage",
    ]):
        return "RIVER_STAGE_M"

    if any(k in t for k in [
        "precipitazione",
        "precipitation",
        "pioggia",
        "rain",
    ]):
        return "PRECIP_MM"

    if "temperatura" in t or "temperature" in t:
        return "AIR_TEMP_C"

    if any(k in t for k in [
        "umidita relativa",
        "relative humidity",
    ]):
        return "REL_HUMIDITY_PCT"

    if any(k in t for k in [
        "velocita vento",
        "wind speed",
    ]):
        return "WIND_SPEED_M_S"

    if any(k in t for k in [
        "direzione vento",
        "wind direction",
    ]):
        return "WIND_DIR_DEG"

    if "pressione" in t or "pressure" in t:
        return "AIR_PRESSURE_HPA"

    if any(k in t for k in [
        "altezza neve",
        "snow depth",
        "neve al suolo",
    ]):
        return "SNOW_DEPTH_CM"

    if any(k in t for k in [
        "radiazione totale",
        "radiazione globale",
        "solar radiation",
    ]):
        return "SOLAR_RAD_W_M2"

    if "insolazione" in t or "sunshine" in t:
        return "SUNSHINE_DURATION_MIN"

    if "evapor" in t:
        return "EVAPORATION_SOURCE_UNIT"

    if any(k in t for k in [
        "umidita suolo",
        "soil moisture",
    ]):
        return "SOIL_MOISTURE_SOURCE_UNIT"

    return "OTHER__" + slug(parameter or extra_text)


def piemontese_variable_codes(row):
    codes = []

    ptot_cols = str(row.get("ptot_columns", "") or "").strip()
    level_cols = str(row.get("level_columns", "") or "").strip()
    discharge_cols = str(row.get("discharge_columns", "") or "").strip()
    kind = str(row.get("kind", "") or "").strip()

    if kind == "daily_meteo" and ptot_cols:
        codes.append("PRECIP_MM")

    if kind == "daily_hydro":
        if level_cols:
            codes.append("RIVER_STAGE_M")
        if discharge_cols:
            codes.append("DISCHARGE_M3_S")

    # Il QC finale garantisce che DATA_OK abbia almeno una variabile core.
    # NO_DATA_SOURCE può comunque avere colonne core ma nessun valore numerico.
    return list(dict.fromkeys(codes))


def main():
    root = Path(__file__).resolve().parent

    preflight = (
        root / "nw_hydroclimate_preflight_v1_2"
        / "nw_hydroclimate_preflight_v1_2.json"
    )

    if not preflight.exists():
        raise SystemExit(f"Preflight v1.2 non trovato: {preflight}")

    pf = json.loads(preflight.read_text(encoding="utf-8"))

    if pf.get("overall_status") != "PASS":
        raise SystemExit(
            f"Preflight osservativo v1.2 non PASS: {pf.get('overall_status')}"
        )

    obs = root / "observations_nw"
    out_dir = root / "nw_observations_standardized_v1_1"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 132)
    print("NW OBSERVATIONS — CANONICAL SERIES REGISTRY v1.1")
    print("=" * 132)

    reasons = []
    registry = []
    relations = []
    variable_rows = []

    # ==================================================================
    # PIEMONTE — schema esplicito dal QC finale
    # ==================================================================
    pie_qc_path = (
        obs / "piemonte" / "qc_final_v1_0"
        / "file_qc_final_v1_0.csv"
    )
    pie_map_path = obs / "station_basin_map.csv"

    pie = pd.read_csv(pie_qc_path, low_memory=False)
    pmap = pd.read_csv(pie_map_path, low_memory=False)

    required_pie_cols = {
        "kind",
        "station_id",
        "station_name",
        "scientific_status",
        "source_csv",
        "ptot_columns",
        "level_columns",
        "discharge_columns",
        "receptor_ids",
    }

    missing = sorted(required_pie_cols - set(pie.columns))
    if missing:
        reasons.append(
            "PIEMONTE_REQUIRED_COLUMNS_MISSING:" + ",".join(missing)
        )

    # station map key must include kind because a station ID may appear
    # both in meteo and hydro.
    pmap_lookup = {}
    if {"kind", "station_id"}.issubset(pmap.columns):
        for _, r in pmap.iterrows():
            key = (
                str(r["kind"]).strip(),
                str(r["station_id"]).strip(),
            )
            pmap_lookup[key] = r.to_dict()
    else:
        reasons.append("PIEMONTE_MAP_KIND_STATION_COLUMNS_MISSING")

    if not missing:
        for _, r in pie.iterrows():
            kind = str(r["kind"]).strip()
            sid = str(r["station_id"]).strip()
            station_name = str(r["station_name"] or "").strip()
            status = norm_status(r["scientific_status"])

            codes = piemontese_variable_codes(r)

            if not codes:
                reasons.append(
                    f"PIEMONTE_VARIABLE_CODE_EMPTY:{kind}:{sid}"
                )

            receptors = parse_receptor_ids(r["receptor_ids"])

            md = pmap_lookup.get(
                (
                    "meteo" if kind == "daily_meteo" else "hydro",
                    sid,
                ),
                {},
            )

            nearest = str(
                md.get("nearest_receptor", "") or ""
            ).strip()

            source_path = resolve_source_path(
                root,
                r["source_csv"],
            )

            source_series_id = f"PIE::{kind}::{sid}"

            registry.append({
                "provider": "ARPA_PIEMONTE",
                "region": "Piemonte",
                "source_series_id": source_series_id,
                "station_id": sid,
                "station_name": station_name,
                "target": "",
                "parameter_source": (
                    "precipitazione_totale_giornaliera"
                    if kind == "daily_meteo"
                    else "livello_e_o_portata_giornalieri"
                ),
                "unit_source": (
                    "mm"
                    if kind == "daily_meteo"
                    else "m | m3/s"
                ),
                "kind_source": kind,
                "variable_codes": "|".join(codes),
                "source_year": "",
                "scientific_status": status,
                "source_data_path": source_path,
                "timezone_status": "DAILY_SOURCE_DATE",
                "primary_receptor_id": (
                    receptors[0]
                    if len(receptors) == 1
                    else ""
                ),
                "receptor_ids_source": "|".join(receptors),
                "nearest_receptor": nearest,
                "mapping_method": (
                    "polygon_membership"
                    if receptors
                    else (
                        "nearest_receptor_fallback_available"
                        if nearest
                        else "unmapped"
                    )
                ),
            })

            for code in codes:
                variable_rows.append({
                    "provider": "ARPA_PIEMONTE",
                    "variable_code": code,
                    "parameter_source": (
                        "ptot_columns"
                        if code == "PRECIP_MM"
                        else (
                            "level_columns"
                            if code == "RIVER_STAGE_M"
                            else "discharge_columns"
                        )
                    ),
                    "unit_source": (
                        "mm"
                        if code == "PRECIP_MM"
                        else (
                            "m"
                            if code == "RIVER_STAGE_M"
                            else "m3/s"
                        )
                    ),
                    "source_scope": kind,
                })

            if receptors:
                for rid in receptors:
                    relations.append({
                        "provider": "ARPA_PIEMONTE",
                        "station_or_target_id": sid,
                        "kind_source": kind,
                        "receptor_id": rid,
                        "relation_type": "POLYGON_MEMBERSHIP",
                        "is_primary": len(receptors) == 1,
                    })
            elif nearest:
                relations.append({
                    "provider": "ARPA_PIEMONTE",
                    "station_or_target_id": sid,
                    "kind_source": kind,
                    "receptor_id": nearest,
                    "relation_type": "NEAREST_RECEPTOR_FALLBACK",
                    "is_primary": True,
                })

    # ==================================================================
    # VALLE D'AOSTA
    # ==================================================================
    vda_root = obs / "valle_d_aosta" / "final_v1_0"
    vda_qc_path = vda_root / "qc" / "vda_file_qc.csv"
    vda_catalog_path = vda_root / "catalog" / "vda_station_catalog.csv"

    vda = pd.read_csv(vda_qc_path, low_memory=False)
    vcat = pd.read_csv(vda_catalog_path, low_memory=False)

    vcat_lookup = {
        str(r["station_code"]).strip(): r.to_dict()
        for _, r in vcat.iterrows()
    }

    for _, r in vda.iterrows():
        sid = str(r["station_code"]).strip()
        parameter = str(r["parameter"]).strip()
        unit = str(r["unit"]).strip()

        numeric = int(
            pd.to_numeric(
                pd.Series([r["numeric_sepdec_1996_2025"]]),
                errors="coerce",
            ).fillna(0).iloc[0]
        )

        status = (
            "DATA_OK"
            if numeric > 0
            else "NO_DATA_SOURCE"
        )

        cat = vcat_lookup.get(sid, {})

        station_name = str(
            r.get("station_name_metadata")
            or cat.get("station_name")
            or ""
        ).strip()

        code = classify_variable(
            parameter,
            unit,
        )

        source_path = resolve_source_path(
            root,
            r.get("normalized_output", ""),
        )

        registry.append({
            "provider": "CENTRO_FUNZIONALE_RAVDA",
            "region": "Valle_d_Aosta",
            "source_series_id": (
                f"VDA::{sid}::{slug(parameter)}"
            ),
            "station_id": sid,
            "station_name": station_name,
            "target": "",
            "parameter_source": parameter,
            "unit_source": unit,
            "kind_source": "",
            "variable_codes": code,
            "source_year": "",
            "scientific_status": status,
            "source_data_path": source_path,
            "timezone_status": "UNRESOLVED_SOURCE_TIME_CONVENTION",
            "primary_receptor_id": "NW_DORA_BALTEA",
            "receptor_ids_source": "NW_DORA_BALTEA",
            "nearest_receptor": "",
            "mapping_method": (
                "model_scope_all_vda_stations_to_current_single_vda_receptor"
            ),
        })

        variable_rows.append({
            "provider": "CENTRO_FUNZIONALE_RAVDA",
            "variable_code": code,
            "parameter_source": parameter,
            "unit_source": unit,
            "source_scope": "hourly_sepdec_1996_2025",
        })

        relations.append({
            "provider": "CENTRO_FUNZIONALE_RAVDA",
            "station_or_target_id": sid,
            "kind_source": "",
            "receptor_id": "NW_DORA_BALTEA",
            "relation_type": "MODEL_SCOPE_MAPPING",
            "is_primary": True,
        })

    # ==================================================================
    # LIGURIA
    # ==================================================================
    lig_root = obs / "liguria_groundtruth_v1_3"
    lig_qc_path = (
        lig_root / "qc_final_v1_1"
        / "file_qc_final_v1_1.csv"
    )

    lig = pd.read_csv(lig_qc_path, low_memory=False)

    l_status_col = find_status_col(lig)
    l_target_col = find_col(
        lig,
        ["target"],
        contains=["target"],
    )
    l_year_col = find_col(
        lig,
        ["year", "anno"],
        contains=["year"],
    )
    l_param_col = find_col(
        lig,
        ["parameter", "parametro"],
        contains=["parameter"],
    )
    l_unit_col = find_col(
        lig,
        ["unit", "unita", "unità"],
        contains=["unit"],
    )
    l_station_col = find_col(
        lig,
        ["station", "station_name", "stazione"],
        contains=["station"],
    )
    l_path_col = detect_path_col(lig)

    if l_status_col is None:
        reasons.append("LIGURIA_STATUS_COL_NOT_FOUND")
    if l_target_col is None:
        reasons.append("LIGURIA_TARGET_COL_NOT_FOUND")
    if l_year_col is None:
        reasons.append("LIGURIA_YEAR_COL_NOT_FOUND")

    if l_status_col and l_target_col and l_year_col:
        for _, r in lig.iterrows():
            target = str(r[l_target_col]).strip()
            year = int(r[l_year_col])
            status = norm_status(r[l_status_col])

            prefix = target.split("_", 1)[0].lower()
            rid = LIG_PREFIX_TO_RECEPTOR.get(
                prefix,
                "",
            )

            if not rid:
                reasons.append(
                    f"LIGURIA_TARGET_UNMAPPED:{target}"
                )

            parameter = (
                str(r[l_param_col]).strip()
                if l_param_col is not None
                and pd.notna(r[l_param_col])
                else target
            )

            unit = (
                str(r[l_unit_col]).strip()
                if l_unit_col is not None
                and pd.notna(r[l_unit_col])
                else ""
            )

            station_name = (
                str(r[l_station_col]).strip()
                if l_station_col is not None
                and pd.notna(r[l_station_col])
                else target
            )

            source_path = (
                resolve_source_path(
                    root,
                    r[l_path_col],
                )
                if l_path_col is not None
                else ""
            )

            code = classify_variable(
                parameter,
                unit,
                target,
            )

            registry.append({
                "provider": "ARPAL",
                "region": "Liguria",
                "source_series_id": (
                    f"LIG::{target}::{year}"
                ),
                "station_id": target,
                "station_name": station_name,
                "target": target,
                "parameter_source": parameter,
                "unit_source": unit,
                "kind_source": "",
                "variable_codes": code,
                "source_year": year,
                "scientific_status": status,
                "source_data_path": source_path,
                "timezone_status": (
                    "PORTAL_DECLARED_UTC_PRESERVE_PROVENANCE"
                ),
                "primary_receptor_id": rid,
                "receptor_ids_source": rid,
                "nearest_receptor": "",
                "mapping_method": "canonical_target_slug_prefix",
            })

            variable_rows.append({
                "provider": "ARPAL",
                "variable_code": code,
                "parameter_source": parameter,
                "unit_source": unit,
                "source_scope": target,
            })

            relations.append({
                "provider": "ARPAL",
                "station_or_target_id": target,
                "kind_source": "",
                "receptor_id": rid,
                "relation_type": "CANONICAL_TARGET_BASIN_MAPPING",
                "is_primary": True,
            })

    # ==================================================================
    # AUDIT
    # ==================================================================
    reg = pd.DataFrame(registry)
    rel = pd.DataFrame(relations).drop_duplicates().reset_index(drop=True)
    vardict = (
        pd.DataFrame(variable_rows)
        .drop_duplicates()
        .sort_values(
            [
                "variable_code",
                "provider",
                "parameter_source",
            ]
        )
        .reset_index(drop=True)
    )

    expected_rows = (
        EXPECTED["PIE_ROWS"]
        + EXPECTED["VDA_ROWS"]
        + EXPECTED["LIG_ROWS"]
    )

    if len(reg) != expected_rows:
        reasons.append(
            f"REGISTRY_ROWS={len(reg)} expected={expected_rows}"
        )

    if reg["source_series_id"].duplicated().any():
        reasons.append(
            f"DUPLICATE_SOURCE_SERIES_ID="
            f"{int(reg['source_series_id'].duplicated().sum())}"
        )

    status_tab = (
        reg.groupby(
            ["provider", "scientific_status"]
        )
        .size()
        .unstack(fill_value=0)
    )

    def count(provider, status):
        if provider not in status_tab.index:
            return 0
        return int(
            status_tab.loc[provider].get(
                status,
                0,
            )
        )

    checks = [
        (
            count("ARPA_PIEMONTE", "DATA_OK"),
            EXPECTED["PIE_OK"],
            "PIEMONTE_DATA_OK",
        ),
        (
            count("ARPA_PIEMONTE", "NO_DATA_SOURCE"),
            EXPECTED["PIE_NO"],
            "PIEMONTE_NO_DATA_SOURCE",
        ),
        (
            count("CENTRO_FUNZIONALE_RAVDA", "DATA_OK"),
            EXPECTED["VDA_OK"],
            "VDA_DATA_OK",
        ),
        (
            count("CENTRO_FUNZIONALE_RAVDA", "NO_DATA_SOURCE"),
            EXPECTED["VDA_NO"],
            "VDA_NO_DATA_SOURCE",
        ),
        (
            count("ARPAL", "DATA_OK"),
            EXPECTED["LIG_OK"],
            "LIGURIA_DATA_OK",
        ),
        (
            count("ARPAL", "NO_DATA_CONFIRMED"),
            EXPECTED["LIG_NO"],
            "LIGURIA_NO_DATA_CONFIRMED",
        ),
    ]

    for actual, expected, label in checks:
        if actual != expected:
            reasons.append(
                f"{label}={actual} expected={expected}"
            )

    provider_rows = (
        reg.groupby("provider")
        .size()
        .to_dict()
    )

    if provider_rows.get("ARPA_PIEMONTE", 0) != EXPECTED["PIE_ROWS"]:
        reasons.append(
            f"PIEMONTE_REGISTRY_ROWS="
            f"{provider_rows.get('ARPA_PIEMONTE', 0)}"
        )

    # Paths.
    reg["source_path_present"] = (
        reg["source_data_path"]
        .astype(str)
        .map(
            lambda s: bool(s)
            and Path(s).exists()
        )
    )

    path_stats = (
        reg.groupby("provider")["source_path_present"]
        .agg(["sum", "count"])
        .reset_index()
    )

    # All DATA_OK must point to an existing source file.
    bad_ok_paths = int(
        (
            reg["scientific_status"].eq("DATA_OK")
            & ~reg["source_path_present"]
        ).sum()
    )

    if bad_ok_paths:
        reasons.append(
            f"DATA_OK_WITHOUT_SOURCE_PATH={bad_ok_paths}"
        )

    # Piemonte DATA_OK must have at least one variable code.
    bad_pie_codes = int(
        (
            reg["provider"].eq("ARPA_PIEMONTE")
            & reg["scientific_status"].eq("DATA_OK")
            & reg["variable_codes"].astype(str).str.strip().eq("")
        ).sum()
    )

    if bad_pie_codes:
        reasons.append(
            f"PIEMONTE_DATA_OK_EMPTY_VARIABLE_CODES={bad_pie_codes}"
        )

    other_rows = int(
        reg["variable_codes"]
        .astype(str)
        .str.contains(r"(?:^|\|)OTHER__", regex=True)
        .sum()
    )

    reg_path = (
        out_dir
        / "observation_series_registry_v1_1.csv"
    )
    rel_path = (
        out_dir
        / "station_receptor_relations_v1_1.csv"
    )
    var_path = (
        out_dir
        / "variable_dictionary_v1_1.csv"
    )

    reg.to_csv(
        reg_path,
        index=False,
    )
    rel.to_csv(
        rel_path,
        index=False,
    )
    vardict.to_csv(
        var_path,
        index=False,
    )

    overall = (
        "PASS"
        if not reasons
        else "REVIEW"
    )

    report = {
        "version": "1.1",
        "overall_status": overall,
        "registry_rows": int(len(reg)),
        "expected_registry_rows": expected_rows,
        "provider_rows": {
            str(k): int(v)
            for k, v in provider_rows.items()
        },
        "relation_rows": int(len(rel)),
        "variable_dictionary_rows": int(len(vardict)),
        "rows_with_other_variable_code": other_rows,
        "status_counts": {
            provider: {
                str(k): int(v)
                for k, v in row[
                    row > 0
                ].to_dict().items()
            }
            for provider, row
            in status_tab.iterrows()
        },
        "source_path_stats": (
            path_stats.to_dict(
                orient="records"
            )
        ),
        "data_ok_without_existing_source_path": bad_ok_paths,
        "piemonte_semantics": {
            "daily_meteo": "ptot_columns -> PRECIP_MM",
            "daily_hydro": (
                "level_columns -> RIVER_STAGE_M; "
                "discharge_columns -> DISCHARGE_M3_S"
            ),
            "registry_granularity": (
                "one row per source file/station-kind; "
                "variable_codes may contain multiple codes"
            ),
        },
        "timezone_policy": {
            "Piemonte": "DAILY_SOURCE_DATE",
            "Valle_d_Aosta": "UNRESOLVED_SOURCE_TIME_CONVENTION",
            "Liguria": "PORTAL_DECLARED_UTC_PRESERVE_PROVENANCE",
        },
        "next_step": (
            "Inspect only remaining OTHER variable mappings, then read DATA_OK "
            "source files into standardized long observation tables."
        ),
        "reasons": reasons,
        "raw_modified": False,
    }

    json_path = (
        out_dir
        / "observation_registry_audit_v1_1.json"
    )
    json_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "=" * 132,
        "NW OBSERVATIONS — CANONICAL SERIES REGISTRY v1.1",
        "=" * 132,
        f"OVERALL STATUS          : {overall}",
        f"Registry rows           : {len(reg)} / {expected_rows}",
        f"Piemonte rows           : {provider_rows.get('ARPA_PIEMONTE', 0)} / {EXPECTED['PIE_ROWS']}",
        f"VdA rows                : {provider_rows.get('CENTRO_FUNZIONALE_RAVDA', 0)} / {EXPECTED['VDA_ROWS']}",
        f"Liguria rows            : {provider_rows.get('ARPAL', 0)} / {EXPECTED['LIG_ROWS']}",
        f"Station-receptor rel.   : {len(rel)}",
        f"Variable dictionary     : {len(vardict)}",
        f"Rows with OTHER code    : {other_rows}",
        f"DATA_OK source missing  : {bad_ok_paths}",
        "",
        "STATUS BY PROVIDER",
        status_tab.to_string(),
        "",
        "SOURCE PATHS PRESENT",
        path_stats.to_string(index=False),
        "",
        "PIEMONTE VARIABLE SEMANTICS",
        "daily_meteo: ptot_columns -> PRECIP_MM",
        "daily_hydro: level_columns -> RIVER_STAGE_M",
        "daily_hydro: discharge_columns -> DISCHARGE_M3_S",
        "",
        "TIMEZONE POLICY",
        "Piemonte      : daily source dates",
        "Valle d'Aosta : UNRESOLVED_SOURCE_TIME_CONVENTION",
        "Liguria       : portal-declared UTC, provenance preserved",
        "",
        f"Registry       : {reg_path}",
        f"Relations      : {rel_path}",
        f"Variable dict. : {var_path}",
    ]

    txt_path = (
        out_dir
        / "observation_registry_audit_v1_1.txt"
    )
    txt_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines[3:]))
    print("\n" + "=" * 132)
    print(f"OVERALL STATUS : {overall}")
    print(f"Output         : {out_dir}")

    if reasons:
        print("REASONS:")
        for r in reasons:
            print(f"  - {r}")

    print("=" * 132)

    if overall != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
