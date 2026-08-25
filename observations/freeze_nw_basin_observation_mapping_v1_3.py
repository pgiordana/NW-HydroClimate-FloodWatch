#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
freeze_nw_basin_observation_mapping_v1_3.py

Freeze canonico definitivo del mapping osservazioni -> 21 recettori NW.

Evidenza dal probe v1.0:
- 122/122 serie incoerenti sono ARPA Piemonte;
- tutte hanno registry.primary_receptor_id vuoto;
- tutte hanno nearest_receptor presente e contenuto nelle relazioni;
- tutte hanno receptor_ids_source che coincide esattamente con l'insieme
  delle relazioni effettive;
- 114 serie hanno 2 relazioni, 8 serie ne hanno 3.

Regola canonica v1.3:
1) se la relation table ha esattamente un is_primary=True, mantenere quello;
2) se una serie ha una sola relazione, quella relazione è primaria;
3) se una serie ha più relazioni e nessun primario nella relation table,
   usare registry.nearest_receptor SOLO SE:
   - nearest_receptor è tra le relazioni;
   - receptor_ids_source coincide con l'insieme delle relazioni.
4) qualsiasi incoerenza -> REVIEW.

Nota scientifica:
- la multi-membership resta preservata;
- canonical_is_primary serve solo come relazione principale/model-scope;
- per le feature meteorologiche di bacino si potranno usare tutte le relazioni
  eleggibili, non soltanto quella primaria;
- stage/discharge restano separati per stazione/sezione.

Output:
nw_observations_basin_mapping_v1_3/
  canonical_series_receptor_relations_v1_3.csv
  receptor_variable_coverage_v1_3.csv
  hydrological_candidate_series_v1_3.csv
  meteorological_candidate_series_v1_3.csv
  receptor_summary_v1_3.csv
  basin_observation_mapping_audit_v1_3.json
  basin_observation_mapping_audit_v1_3.txt
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pandas as pd


EXPECTED_RECEPTORS = [
    "NW_DORA_RIPARIA",
    "NW_STURA_LANZO",
    "NW_STURA_DEMONTE",
    "NW_TANARO_ALTO",
    "NW_TANARO_MEDIO_BASSO",
    "NW_BORMIDA",
    "NW_ORBA",
    "NW_SCRIVIA",
    "NW_DORA_BALTEA",
    "NW_ORCO",
    "NW_PELLICE",
    "NW_CHISONE",
    "NW_MAIRA",
    "NW_VARAITA",
    "NW_SESIA",
    "NW_TOCE",
    "LIG_BISAGNO",
    "LIG_POLCEVERA",
    "LIG_ENTELLA",
    "LIG_MAGRA",
    "LIG_CENTA",
]

HYDRO_CODES = {
    "RIVER_STAGE_M",
    "DISCHARGE_M3_S",
    "DISCHARGE_MIN_M3_S",
    "DISCHARGE_MAX_M3_S",
}

METEO_CODES = {
    "PRECIP_MM",
    "AIR_TEMP_C",
    "REL_HUMIDITY_PCT",
    "WIND_SPEED_M_S",
    "WIND_DIR_DEG",
    "AIR_PRESSURE_HPA",
    "SNOW_DEPTH_CM",
    "SOLAR_RAD_W_M2",
    "SUNSHINE_DURATION_MIN",
}


def normalize_bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    try:
        if pd.isna(v):
            return False
    except Exception:
        pass
    return str(v).strip().lower() in {
        "true", "1", "yes", "y", "si", "sì"
    }


def split_pipe(v):
    if v is None:
        return []
    try:
        if pd.isna(v):
            return []
    except Exception:
        pass
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return []
    return [x.strip() for x in s.split("|") if x.strip()]


def split_codes(v):
    return split_pipe(v)


def parse_dict(v):
    if isinstance(v, dict):
        return v
    if v is None:
        return {}
    try:
        if pd.isna(v):
            return {}
    except Exception:
        pass

    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null", "{}"}:
        return {}

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    return {}


def provider_aware_join(data_ok, relations):
    outs = []

    # Piemonte: provider + station_id + kind_source
    rg = data_ok[data_ok["provider"].eq("ARPA_PIEMONTE")].copy()
    rl = relations[relations["provider"].eq("ARPA_PIEMONTE")].copy()

    rg["station_id"] = rg["station_id"].astype(str)
    rl["station_or_target_id"] = rl["station_or_target_id"].astype(str)

    outs.append(
        rg.merge(
            rl,
            left_on=["station_id", "kind_source"],
            right_on=["station_or_target_id", "kind_source"],
            how="left",
            suffixes=("", "_relation"),
        )
    )

    # Valle d'Aosta: provider + station_id
    rg = data_ok[
        data_ok["provider"].eq("CENTRO_FUNZIONALE_RAVDA")
    ].copy()
    rl = relations[
        relations["provider"].eq("CENTRO_FUNZIONALE_RAVDA")
    ].copy()

    rg["station_id"] = rg["station_id"].astype(str)
    rl["station_or_target_id"] = rl["station_or_target_id"].astype(str)

    outs.append(
        rg.merge(
            rl,
            left_on=["station_id"],
            right_on=["station_or_target_id"],
            how="left",
            suffixes=("", "_relation"),
        )
    )

    # Liguria: provider + target
    rg = data_ok[data_ok["provider"].eq("ARPAL")].copy()
    rl = relations[relations["provider"].eq("ARPAL")].copy()

    rg["target"] = rg["target"].astype(str)
    rl["station_or_target_id"] = rl["station_or_target_id"].astype(str)

    outs.append(
        rg.merge(
            rl,
            left_on=["target"],
            right_on=["station_or_target_id"],
            how="left",
            suffixes=("", "_relation"),
        )
    )

    return pd.concat(outs, ignore_index=True, sort=False)


def main():
    root = Path(__file__).resolve().parent

    std_root = root / "nw_observations_standardized_v1_4"
    daily_root = root / "nw_observations_daily_v1_0"

    reg_p = std_root / "observation_series_registry_v1_4.csv"
    rel_p = std_root / "station_receptor_relations_v1_4.csv"
    time_audit_p = std_root / "observation_time_audit_v1_0.json"

    daily_man_p = daily_root / "daily_series_manifest_v1_0.csv"
    daily_audit_p = daily_root / "daily_station_layer_audit_v1_0.json"

    out_root = root / "nw_observations_basin_mapping_v1_3"
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 144)
    print("NW OBSERVATIONS — FREEZE BASIN MAPPING v1.3")
    print("=" * 144)

    for p in [
        reg_p,
        rel_p,
        time_audit_p,
        daily_man_p,
        daily_audit_p,
    ]:
        if not p.exists():
            raise SystemExit(f"Manca: {p}")

    time_audit = json.loads(
        time_audit_p.read_text(encoding="utf-8")
    )
    daily_audit = json.loads(
        daily_audit_p.read_text(encoding="utf-8")
    )

    if time_audit.get("overall_status") != "PASS":
        raise SystemExit("Time audit non PASS.")
    if daily_audit.get("overall_status") != "PASS":
        raise SystemExit("Daily station layer non PASS.")

    registry = pd.read_csv(reg_p, low_memory=False)
    relations = pd.read_csv(rel_p, low_memory=False)
    daily_man = pd.read_csv(daily_man_p, low_memory=False)

    data_ok = registry[
        registry["scientific_status"]
        .astype(str)
        .str.upper()
        .eq("DATA_OK")
    ].copy()

    if len(data_ok) != 1312:
        raise SystemExit(
            f"DATA_OK={len(data_ok)}, atteso=1312"
        )

    joined = provider_aware_join(data_ok, relations)

    mapped_ids = set(
        joined.loc[
            joined["receptor_id"].notna(),
            "source_series_id",
        ].astype(str)
    )
    all_ids = set(data_ok["source_series_id"].astype(str))

    reasons = []

    unmapped = sorted(all_ids - mapped_ids)
    if unmapped:
        reasons.append(
            f"UNMAPPED_DATA_OK_SERIES={len(unmapped)}"
        )

    joined = joined[
        joined["receptor_id"].notna()
    ].copy()

    joined["relation_is_primary"] = (
        joined["is_primary"].map(normalize_bool)
    )

    canonical_rows = []

    resolution_counts = {
        "RELATION_TABLE": 0,
        "NEAREST_RECEPTOR_VALIDATED": 0,
        "SINGLE_RELATION_FALLBACK": 0,
    }

    incoherent = []
    multiple_true = []

    for sid, g in joined.groupby(
        "source_series_id",
        sort=False,
    ):
        g = g.copy()

        receptors = list(
            dict.fromkeys(
                g["receptor_id"].astype(str).tolist()
            )
        )

        true_receptors = list(
            dict.fromkeys(
                g.loc[
                    g["relation_is_primary"],
                    "receptor_id",
                ].astype(str).tolist()
            )
        )

        r0 = g.iloc[0]

        nearest = str(
            r0.get("nearest_receptor", "")
            if pd.notna(r0.get("nearest_receptor"))
            else ""
        ).strip()

        source_receptors = split_pipe(
            r0.get("receptor_ids_source")
        )

        if len(true_receptors) > 1:
            multiple_true.append({
                "source_series_id": sid,
                "true_receptors": true_receptors,
                "relations": receptors,
            })
            continue

        if len(true_receptors) == 1:
            primary = true_receptors[0]
            resolution = "RELATION_TABLE"

        elif len(receptors) == 1:
            primary = receptors[0]
            resolution = "SINGLE_RELATION_FALLBACK"

        else:
            # Validated by the probe: nearest must be one of the actual
            # relations and receptor_ids_source must match the relation set.
            if not nearest:
                incoherent.append({
                    "source_series_id": sid,
                    "reason": "MULTI_RELATION_NEAREST_EMPTY",
                    "relations": receptors,
                })
                continue

            if nearest not in receptors:
                incoherent.append({
                    "source_series_id": sid,
                    "reason": "NEAREST_NOT_IN_RELATIONS",
                    "nearest": nearest,
                    "relations": receptors,
                })
                continue

            if not source_receptors:
                incoherent.append({
                    "source_series_id": sid,
                    "reason": "RECEPTOR_IDS_SOURCE_EMPTY",
                    "nearest": nearest,
                    "relations": receptors,
                })
                continue

            if set(source_receptors) != set(receptors):
                incoherent.append({
                    "source_series_id": sid,
                    "reason": "SOURCE_RELATION_SET_MISMATCH",
                    "nearest": nearest,
                    "source_receptors": source_receptors,
                    "relations": receptors,
                })
                continue

            primary = nearest
            resolution = "NEAREST_RECEPTOR_VALIDATED"

        resolution_counts[resolution] += 1

        for _, r in g.iterrows():
            rec = str(r["receptor_id"])

            canonical_rows.append({
                "provider": str(r["provider"]),
                "source_series_id": str(r["source_series_id"]),
                "station_id": str(r["station_id"]),
                "station_name": str(
                    r.get("station_name", "") or ""
                ),
                "target": str(
                    r.get("target", "") or ""
                ),
                "kind_source": str(
                    r.get("kind_source", "") or ""
                ),
                "station_or_target_id": str(
                    r.get("station_or_target_id", "") or ""
                ),
                "receptor_id": rec,
                "relation_type": str(
                    r.get("relation_type", "") or ""
                ),
                "relation_is_primary_source": bool(
                    r["relation_is_primary"]
                ),
                "nearest_receptor_registry": nearest,
                "receptor_ids_source_registry": "|".join(
                    source_receptors
                ),
                "canonical_is_primary": rec == primary,
                "primary_resolution_source": resolution,
                "variable_codes": str(
                    r.get("variable_codes", "") or ""
                ),
                "time_basis_canonical": str(
                    r.get("time_basis_canonical", "") or ""
                ),
            })

    if multiple_true:
        reasons.append(
            f"MULTIPLE_TRUE_PRIMARY_SERIES={len(multiple_true)}"
        )

    if incoherent:
        reasons.append(
            f"PRIMARY_MAPPING_INCOHERENT_SERIES={len(incoherent)}"
        )

    canonical = pd.DataFrame(canonical_rows)

    canonical_ids = set(
        canonical["source_series_id"].astype(str)
    ) if len(canonical) else set()

    if len(canonical_ids) != 1312:
        reasons.append(
            f"CANONICAL_MAPPED_SERIES={len(canonical_ids)} "
            f"expected=1312"
        )

    # Exactly one primary relation per source series.
    if len(canonical):
        primary_check = (
            canonical.groupby("source_series_id")
            .agg(
                relation_count=("receptor_id", "size"),
                primary_count=("canonical_is_primary", "sum"),
            )
            .reset_index()
        )

        bad_primary_count = primary_check[
            primary_check["primary_count"] != 1
        ]

        if len(bad_primary_count):
            reasons.append(
                "SERIES_WITH_CANONICAL_PRIMARY_COUNT_NOT_1="
                + str(len(bad_primary_count))
            )
    else:
        reasons.append("CANONICAL_MAPPING_EMPTY")

    receptors_seen = sorted(
        set(canonical["receptor_id"].astype(str))
    ) if len(canonical) else []

    missing_receptors = [
        r for r in EXPECTED_RECEPTORS
        if r not in receptors_seen
    ]

    unexpected_receptors = [
        r for r in receptors_seen
        if r not in EXPECTED_RECEPTORS
    ]

    if missing_receptors:
        reasons.append(
            "MISSING_RECEPTORS="
            + "|".join(missing_receptors)
        )

    if unexpected_receptors:
        reasons.append(
            "UNEXPECTED_RECEPTORS="
            + "|".join(unexpected_receptors)
        )

    # Attach daily manifest.
    man_small = daily_man[
        [
            "source_series_id",
            "daily_rows",
            "date_min",
            "date_max",
            "variable_counts",
            "output_path",
        ]
    ].copy()

    canonical = canonical.merge(
        man_small,
        on="source_series_id",
        how="left",
    )

    missing_daily = int(
        canonical["daily_rows"].isna().sum()
    )
    if missing_daily:
        reasons.append(
            f"CANONICAL_RELATIONS_WITHOUT_DAILY_MANIFEST={missing_daily}"
        )

    # Expand variables.
    cand_rows = []

    for _, r in canonical.iterrows():
        vc = parse_dict(r["variable_counts"])

        for code in split_codes(r["variable_codes"]):
            cand_rows.append({
                "receptor_id": r["receptor_id"],
                "canonical_is_primary": bool(
                    r["canonical_is_primary"]
                ),
                "primary_resolution_source": r[
                    "primary_resolution_source"
                ],
                "provider": r["provider"],
                "source_series_id": r["source_series_id"],
                "station_id": r["station_id"],
                "station_name": r["station_name"],
                "target": r["target"],
                "kind_source": r["kind_source"],
                "variable_code": code,
                "daily_rows_variable": int(
                    vc.get(code, 0)
                ),
                "date_min": r["date_min"],
                "date_max": r["date_max"],
                "daily_output_path": r["output_path"],
                "time_basis_canonical": r[
                    "time_basis_canonical"
                ],
            })

    cand = pd.DataFrame(cand_rows)

    hydro = cand[
        cand["variable_code"].isin(HYDRO_CODES)
    ].copy()

    meteo = cand[
        cand["variable_code"].isin(METEO_CODES)
    ].copy()

    coverage = (
        cand.groupby(
            ["receptor_id", "variable_code"],
            dropna=False,
        )
        .agg(
            providers=(
                "provider",
                lambda s: "|".join(
                    sorted(set(map(str, s)))
                ),
            ),
            series_count=("source_series_id", "nunique"),
            station_count=("station_id", "nunique"),
            primary_series_count=(
                "canonical_is_primary",
                "sum",
            ),
            daily_rows_total=(
                "daily_rows_variable",
                "sum",
            ),
            date_min=("date_min", "min"),
            date_max=("date_max", "max"),
        )
        .reset_index()
        .sort_values(
            ["receptor_id", "variable_code"]
        )
    )

    hydro_receptors = set(hydro["receptor_id"])
    hydro_missing = [
        r for r in EXPECTED_RECEPTORS
        if r not in hydro_receptors
    ]

    precip_receptors = set(
        meteo.loc[
            meteo["variable_code"].eq("PRECIP_MM"),
            "receptor_id",
        ]
    )

    precip_missing = [
        r for r in EXPECTED_RECEPTORS
        if r not in precip_receptors
    ]

    # Lack of observed precip is treated as a mapping/coverage issue.
    if precip_missing:
        reasons.append(
            "RECEPTORS_WITHOUT_OBS_PRECIP="
            + "|".join(precip_missing)
        )

    rec_rows = []

    for rec in EXPECTED_RECEPTORS:
        h = hydro[hydro["receptor_id"].eq(rec)]
        m = meteo[meteo["receptor_id"].eq(rec)]

        rec_rows.append({
            "receptor_id": rec,
            "meteo_series": int(
                m["source_series_id"].nunique()
            ),
            "precip_stations": int(
                m.loc[
                    m["variable_code"].eq("PRECIP_MM"),
                    "station_id",
                ].nunique()
            ),
            "hydro_series": int(
                h["source_series_id"].nunique()
            ),
            "stage_stations": int(
                h.loc[
                    h["variable_code"].eq("RIVER_STAGE_M"),
                    "station_id",
                ].nunique()
            ),
            "discharge_stations": int(
                h.loc[
                    h["variable_code"].isin({
                        "DISCHARGE_M3_S",
                        "DISCHARGE_MIN_M3_S",
                        "DISCHARGE_MAX_M3_S",
                    }),
                    "station_id",
                ].nunique()
            ),
            "primary_hydro_series": int(
                h.loc[
                    h["canonical_is_primary"],
                    "source_series_id",
                ].nunique()
            ),
        })

    rec_summary = pd.DataFrame(rec_rows)

    # Outputs.
    canonical_out = (
        out_root
        / "canonical_series_receptor_relations_v1_3.csv"
    )
    coverage_out = (
        out_root
        / "receptor_variable_coverage_v1_3.csv"
    )
    hydro_out = (
        out_root
        / "hydrological_candidate_series_v1_3.csv"
    )
    meteo_out = (
        out_root
        / "meteorological_candidate_series_v1_3.csv"
    )
    rec_out = (
        out_root
        / "receptor_summary_v1_3.csv"
    )

    canonical.to_csv(canonical_out, index=False)
    coverage.to_csv(coverage_out, index=False)
    hydro.to_csv(hydro_out, index=False)
    meteo.to_csv(meteo_out, index=False)
    rec_summary.to_csv(rec_out, index=False)

    overall = "PASS" if not reasons else "REVIEW"

    report = {
        "version": "1.3",
        "overall_status": overall,
        "data_ok_series": 1312,
        "canonical_mapped_series": int(len(canonical_ids)),
        "canonical_relation_rows": int(len(canonical)),
        "receptors_seen": receptors_seen,
        "primary_resolution_counts": resolution_counts,
        "multiple_true_primary_series": multiple_true,
        "primary_mapping_incoherent_series": incoherent,
        "receptors_without_hydrological_candidates": hydro_missing,
        "receptors_without_observed_precipitation": precip_missing,
        "hydrology_gap_is_fatal": False,
        "scientific_note": (
            "Multi-membership is preserved. canonical_is_primary is a "
            "principal model-scope relation. For meteorological basin "
            "aggregation all eligible relations can be retained; observed "
            "hydrological series remain separate by station/section."
        ),
        "raw_modified": False,
        "reasons": reasons,
        "next_step": (
            "If PASS, build observed meteorological daily basin features "
            "and separately rank/select hydrological target/control series."
        ),
    }

    audit_json = (
        out_root
        / "basin_observation_mapping_audit_v1_3.json"
    )
    audit_txt = (
        out_root
        / "basin_observation_mapping_audit_v1_3.txt"
    )

    audit_json.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "=" * 144,
        "NW OBSERVATIONS — FREEZE BASIN MAPPING v1.3",
        "=" * 144,
        f"OVERALL STATUS                       : {overall}",
        f"DATA_OK series                       : 1312 / 1312",
        f"Canonical mapped series              : {len(canonical_ids)} / 1312",
        f"Canonical relation rows              : {len(canonical)}",
        f"Receptors seen                       : {len(receptors_seen)} / 21",
        f"Primary from relation table          : {resolution_counts['RELATION_TABLE']}",
        f"Primary from validated nearest       : {resolution_counts['NEAREST_RECEPTOR_VALIDATED']}",
        f"Single-relation primary fallback     : {resolution_counts['SINGLE_RELATION_FALLBACK']}",
        f"Multiple true primary series         : {len(multiple_true)}",
        f"Incoherent primary mappings          : {len(incoherent)}",
        f"Receptors without hydro              : {len(hydro_missing)}",
        f"Receptors without observed precip    : {len(precip_missing)}",
        "",
        "RECEPTOR SUMMARY",
        rec_summary.to_string(index=False),
        "",
        "HYDROLOGICAL COVERAGE GAPS",
        "|".join(hydro_missing) if hydro_missing else "NONE",
        "",
        "OBSERVED PRECIPITATION GAPS",
        "|".join(precip_missing) if precip_missing else "NONE",
        "",
        f"Canonical relations: {canonical_out}",
        f"Coverage           : {coverage_out}",
        f"Hydro candidates   : {hydro_out}",
        f"Meteo candidates   : {meteo_out}",
        f"Receptor summary   : {rec_out}",
    ]

    audit_txt.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines[3:]))
    print("\n" + "=" * 144)
    print(f"OVERALL STATUS : {overall}")
    print(f"Output         : {out_root}")

    if reasons:
        print("REASONS:")
        for r in reasons:
            print(f"  - {r}")

    print("=" * 144)

    if overall != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
