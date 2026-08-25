#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
repair_nw_standardized_observation_values_v1_2.py

Repair mirato finale delle sole 5 serie Piemonte rimaste ERROR dopo v1.1.

Causa corretta:
le 5 serie sono le 5 `stage only` identificate dal freeze v1.3.
Nel CSV del registro, `active_discharge_columns` vuoto viene riletto da pandas
come NaN; la funzione split_cols della v1.1 trasformava erroneamente NaN
nella stringa "nan" e tentava quindi di leggere una colonna inesistente.

Questa v1.2:
- riusa integralmente le 1307 serie PASS del manifest v1.1;
- ripara soltanto le 5 serie ERROR;
- tratta correttamente stringhe vuote/NaN;
- verifica che siano ARPA_PIEMONTE / daily_hydro / RIVER_STAGE_M;
- usa solo `active_level_columns` congelate nel registro v1.3;
- produce manifest, summary e audit v1.2.

NON modifica dati sorgente.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_TOTAL = 1312
EXPECTED_ERRORS_IN = 5
TARGET_MONTHS = {9, 10, 11, 12}


def safe_name(s):
    s = str(s or "")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    return s or "series"


def split_cols_safe(value):
    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    s = str(value).strip()

    if not s or s.lower() in {"nan", "none", "null"}:
        return []

    return [
        x.strip()
        for x in s.split("|")
        if x.strip()
        and x.strip().lower() not in {"nan", "none", "null"}
    ]


def atomic_write_csv_gz(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, compression="gzip")
    tmp.replace(path)


def atomic_write_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def source_fingerprint(path):
    st = path.stat()
    return {
        "source_size_bytes": int(st.st_size),
        "source_mtime_ns": int(st.st_mtime_ns),
    }


def build_stage_only(reg_row, source_path):
    df = pd.read_csv(source_path, low_memory=False)

    if "data" not in df.columns:
        raise ValueError("colonna `data` mancante")

    dates = pd.to_datetime(df["data"], errors="coerce")

    date_mask = (
        dates.notna()
        & dates.dt.year.between(1987, 2025)
        & dates.dt.month.isin(TARGET_MONTHS)
    )

    level_cols = split_cols_safe(
        reg_row.get("active_level_columns")
    )
    discharge_cols = split_cols_safe(
        reg_row.get("active_discharge_columns")
    )

    if not level_cols:
        raise ValueError(
            "stage-only senza active_level_columns"
        )

    if discharge_cols:
        raise ValueError(
            f"serie attesa stage-only ma active_discharge_columns={discharge_cols}"
        )

    codes = split_cols_safe(
        reg_row.get("variable_codes")
    )

    if codes != ["RIVER_STAGE_M"]:
        raise ValueError(
            f"variable_codes inatteso: {codes}"
        )

    parts = []
    selected = []

    for source_col in level_cols:
        if source_col not in df.columns:
            raise ValueError(
                f"active level column mancante: {source_col}"
            )

        vals = pd.to_numeric(
            df[source_col],
            errors="coerce",
        )

        mask = date_mask & vals.notna()

        if not mask.any():
            raise ValueError(
                f"active level column senza numerici: {source_col}"
            )

        d = dates.loc[mask]
        v = vals.loc[mask]

        part = pd.DataFrame({
            "source_series_id": [
                str(reg_row["source_series_id"])
            ] * len(v),
            "provider": ["ARPA_PIEMONTE"] * len(v),
            "station_id": [
                str(reg_row["station_id"])
            ] * len(v),
            "target": [""] * len(v),
            "receptor_ids_source": [
                str(
                    reg_row.get(
                        "receptor_ids_source", ""
                    )
                    or ""
                )
            ] * len(v),
            "variable_code": [
                "RIVER_STAGE_M"
            ] * len(v),
            "source_column": [
                source_col
            ] * len(v),
            "unit_source": ["m"] * len(v),
            "unit_canonical": ["m"] * len(v),
            "time_resolution": ["daily"] * len(v),
            "timestamp_source": (
                d.dt.strftime("%Y-%m-%d").tolist()
            ),
            "interval_end_source": [""] * len(v),
            "date_source": (
                d.dt.strftime("%Y-%m-%d").tolist()
            ),
            "timestamp_utc": [""] * len(v),
            "value_numeric": (
                v.astype(float).tolist()
            ),
            "value_raw": (
                df.loc[mask, source_col]
                .astype(str)
                .tolist()
            ),
            "value_semantics": [
                "daily_mean_stage"
            ] * len(v),
            "timezone_status": [
                "DAILY_SOURCE_DATE"
            ] * len(v),
        })

        parts.append(part)
        selected.append(source_col)

    out = pd.concat(
        parts,
        ignore_index=True,
    )

    dup = int(
        out.duplicated(
            [
                "timestamp_source",
                "variable_code",
                "source_column",
            ]
        ).sum()
    )

    if dup:
        raise ValueError(
            f"duplicate long keys={dup}"
        )

    return out, selected


def main():
    root = Path(__file__).resolve().parent

    values_root = (
        root / "nw_observations_values_v1_0"
    )

    manifest_in = (
        values_root
        / "standardized_series_manifest_v1_1.csv"
    )

    registry_path = (
        root
        / "nw_observations_standardized_v1_3"
        / "observation_series_registry_v1_3.csv"
    )

    registry_audit = (
        root
        / "nw_observations_standardized_v1_3"
        / "observation_registry_audit_v1_3.json"
    )

    series_root = values_root / "series"
    meta_root = values_root / "metadata"

    print("=" * 136)
    print(
        "NW OBSERVATIONS — FINAL STAGE-ONLY REPAIR v1.2"
    )
    print("=" * 136)

    for p in [
        manifest_in,
        registry_path,
        registry_audit,
    ]:
        if not p.exists():
            raise SystemExit(f"Manca: {p}")

    audit = json.loads(
        registry_audit.read_text(
            encoding="utf-8"
        )
    )

    if audit.get("overall_status") != "PASS":
        raise SystemExit(
            "Registro semantico v1.3 non PASS."
        )

    man = pd.read_csv(
        manifest_in,
        low_memory=False,
    )
    reg = pd.read_csv(
        registry_path,
        low_memory=False,
    )

    if len(man) != EXPECTED_TOTAL:
        raise SystemExit(
            f"Manifest v1.1 rows={len(man)}, "
            f"atteso={EXPECTED_TOTAL}"
        )

    err = man[
        ~man["status"]
        .astype(str)
        .str.upper()
        .eq("PASS")
    ].copy()

    if len(err) != EXPECTED_ERRORS_IN:
        raise SystemExit(
            f"Errori v1.1={len(err)}, "
            f"attesi={EXPECTED_ERRORS_IN}"
        )

    reg_lookup = (
        reg.set_index(
            "source_series_id",
            drop=False,
        )
    )

    final_records = []
    repaired = 0
    errors = []

    # Mantieni i 1307 PASS esattamente come sono.
    for _, r in man[
        man["status"]
        .astype(str)
        .str.upper()
        .eq("PASS")
    ].iterrows():
        final_records.append(
            r.to_dict()
        )

    # Ripara soltanto le 5 ERROR.
    for _, old in err.iterrows():
        sid = str(old["source_series_id"])

        if sid not in reg_lookup.index:
            errors.append(
                f"{sid}: non trovato nel registry v1.3"
            )
            continue

        rr = reg_lookup.loc[sid]

        # Se per qualche motivo ci fossero duplicati.
        if isinstance(rr, pd.DataFrame):
            rr = rr.iloc[0]

        provider = str(rr["provider"])
        kind = str(rr["kind_source"])
        codes = split_cols_safe(
            rr["variable_codes"]
        )

        if provider != "ARPA_PIEMONTE":
            errors.append(
                f"{sid}: provider inatteso {provider}"
            )
            continue

        if kind != "daily_hydro":
            errors.append(
                f"{sid}: kind inatteso {kind}"
            )
            continue

        if codes != ["RIVER_STAGE_M"]:
            errors.append(
                f"{sid}: attesa stage-only, codes={codes}"
            )
            continue

        source_path = Path(
            str(rr["source_data_path"])
        )

        try:
            out_df, selected = build_stage_only(
                rr,
                source_path,
            )

            fname = (
                safe_name(sid)
                + ".csv.gz"
            )

            out_path = (
                series_root
                / "ARPA_PIEMONTE"
                / fname
            )

            meta_path = (
                meta_root
                / "ARPA_PIEMONTE"
                / (fname + ".meta.json")
            )

            atomic_write_csv_gz(
                out_df,
                out_path,
            )

            fp = source_fingerprint(
                source_path
            )

            meta = {
                "status": "PASS",
                "provider": "ARPA_PIEMONTE",
                "source_series_id": sid,
                "station_id": str(
                    rr["station_id"]
                ),
                "target": "",
                "source_data_path": str(
                    source_path
                ),
                "output_path": str(
                    out_path
                ),
                "rows": int(len(out_df)),
                "date_min": str(
                    out_df[
                        "date_source"
                    ].min()
                ),
                "date_max": str(
                    out_df[
                        "date_source"
                    ].max()
                ),
                "variable_counts": {
                    "RIVER_STAGE_M":
                    int(len(out_df))
                },
                "variable_codes_registry":
                    "RIVER_STAGE_M",
                "timezone_status":
                    "DAILY_SOURCE_DATE",
                "source_size_bytes":
                    fp["source_size_bytes"],
                "source_mtime_ns":
                    fp["source_mtime_ns"],
                "selected_source_columns":
                    selected,
                "builder_version":
                    "1.2_stage_only_nan_fix",
            }

            atomic_write_json(
                meta,
                meta_path,
            )

            final_records.append(meta)
            repaired += 1

            print(
                f"PASS {repaired}/5 | "
                f"{sid} | columns={selected} "
                f"| rows={len(out_df)}"
            )

        except Exception as exc:
            errors.append(
                f"{sid}: {repr(exc)}"
            )

    final = pd.DataFrame(
        final_records
    )

    # Ordina secondo il registry DATA_OK.
    order = {
        sid: i
        for i, sid in enumerate(
            reg[
                reg["scientific_status"]
                .astype(str)
                .str.upper()
                .eq("DATA_OK")
            ]["source_series_id"]
        )
    }

    final["_order"] = (
        final["source_series_id"]
        .map(order)
    )

    final = (
        final.sort_values(
            "_order"
        )
        .drop(
            columns=["_order"]
        )
        .reset_index(drop=True)
    )

    reasons = []

    if errors:
        reasons.append(
            f"REPAIR_ERRORS={len(errors)}"
        )

    if len(final) != EXPECTED_TOTAL:
        reasons.append(
            f"FINAL_MANIFEST_ROWS={len(final)} "
            f"expected={EXPECTED_TOTAL}"
        )

    pass_count = int(
        final["status"]
        .astype(str)
        .str.upper()
        .eq("PASS")
        .sum()
    )

    if pass_count != EXPECTED_TOTAL:
        reasons.append(
            f"PASS_SERIES={pass_count} "
            f"expected={EXPECTED_TOTAL}"
        )

    provider_counts = (
        final["provider"]
        .value_counts()
        .to_dict()
    )

    expected_provider = {
        "ARPA_PIEMONTE": 397,
        "CENTRO_FUNZIONALE_RAVDA": 504,
        "ARPAL": 411,
    }

    for p, n in expected_provider.items():
        got = int(
            provider_counts.get(p, 0)
        )
        if got != n:
            reasons.append(
                f"{p}_PASS={got} expected={n}"
            )

    missing_outputs = int(
        (
            ~final[
                "output_path"
            ].map(
                lambda s:
                Path(str(s)).exists()
            )
        ).sum()
    )

    if missing_outputs:
        reasons.append(
            f"MISSING_OUTPUT_FILES="
            f"{missing_outputs}"
        )

    # Summary finale.
    summary_rows = []

    for _, r in final.iterrows():
        vc = r.get(
            "variable_counts",
            {}
        )

        if isinstance(vc, str):
            try:
                vc = json.loads(
                    vc.replace("'", '"')
                )
            except Exception:
                try:
                    import ast
                    vc = ast.literal_eval(vc)
                except Exception:
                    vc = {}

        if not isinstance(
            vc,
            dict,
        ):
            vc = {}

        for code, n in vc.items():
            summary_rows.append({
                "provider":
                    r["provider"],
                "variable_code":
                    str(code),
                "rows":
                    int(n),
                "series":
                    1,
            })

    summary = pd.DataFrame(
        summary_rows
    )

    summary = (
        summary.groupby(
            [
                "provider",
                "variable_code",
            ]
        )
        .agg(
            rows=("rows", "sum"),
            series=("series", "sum"),
        )
        .reset_index()
        .sort_values(
            [
                "provider",
                "variable_code",
            ]
        )
    )

    total_rows = int(
        pd.to_numeric(
            final["rows"],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    overall = (
        "PASS"
        if not reasons
        else "REVIEW"
    )

    manifest_out = (
        values_root
        / "standardized_series_manifest_v1_2.csv"
    )
    summary_out = (
        values_root
        / "standardized_value_summary_v1_2.csv"
    )

    final.to_csv(
        manifest_out,
        index=False,
    )
    summary.to_csv(
        summary_out,
        index=False,
    )

    report = {
        "version": "1.2",
        "overall_status": overall,
        "expected_data_ok_series":
            EXPECTED_TOTAL,
        "manifest_rows":
            int(len(final)),
        "pass_series":
            pass_count,
        "error_series":
            int(EXPECTED_TOTAL - pass_count),
        "reused_pass_v1_1":
            1307,
        "repaired_stage_only":
            repaired,
        "total_standardized_numeric_rows":
            total_rows,
        "provider_pass_counts": {
            str(k): int(v)
            for k, v
            in provider_counts.items()
        },
        "missing_output_files":
            missing_outputs,
        "fix": (
            "NaN-safe handling of empty "
            "active_discharge_columns for "
            "the 5 Piemonte stage-only series."
        ),
        "raw_modified":
            False,
        "reasons":
            reasons,
        "repair_errors":
            errors,
    }

    atomic_write_json(
        report,
        values_root
        / "standardized_values_audit_v1_2.json",
    )

    lines = [
        "=" * 136,
        "NW OBSERVATIONS — FINAL STAGE-ONLY REPAIR v1.2",
        "=" * 136,
        f"OVERALL STATUS                  : {overall}",
        f"DATA_OK series expected         : {EXPECTED_TOTAL}",
        f"Manifest rows                   : {len(final)}",
        f"PASS series                     : {pass_count}",
        f"ERROR series                    : {EXPECTED_TOTAL - pass_count}",
        f"Reused PASS v1.1                : 1307",
        f"Repaired stage-only             : {repaired}",
        f"Total standardized numeric rows : {total_rows}",
        f"Missing output files            : {missing_outputs}",
        "",
        "PASS SERIES BY PROVIDER",
        f"ARPA_PIEMONTE                  : {int(provider_counts.get('ARPA_PIEMONTE', 0))}",
        f"CENTRO_FUNZIONALE_RAVDA       : {int(provider_counts.get('CENTRO_FUNZIONALE_RAVDA', 0))}",
        f"ARPAL                          : {int(provider_counts.get('ARPAL', 0))}",
        "",
        "VALUE SUMMARY",
        summary.to_string(index=False),
        "",
        f"Manifest: {manifest_out}",
        f"Summary : {summary_out}",
    ]

    (
        values_root
        / "standardized_values_audit_v1_2.txt"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 136)
    print(
        "\n".join(
            lines[3:]
        )
    )
    print(
        "\n" + "=" * 136
    )
    print(
        f"OVERALL STATUS : {overall}"
    )
    print(
        f"Output         : {values_root}"
    )

    if errors:
        print("REPAIR ERRORS:")
        for e in errors:
            print(f"  - {e}")

    if reasons:
        print("REASONS:")
        for r in reasons:
            print(f"  - {r}")

    print("=" * 136)

    if overall != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
