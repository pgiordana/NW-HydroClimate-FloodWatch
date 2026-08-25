#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_nw_standardized_observation_values_v1_0.py

Costruisce i VALORI OSSERVATI STANDARDIZZATI delle 1.312 serie DATA_OK
(Piemonte + Valle d'Aosta + Liguria) in file long compressi, uno per serie.

Prerequisiti canonici:
- nw_observations_standardized_v1_2/
    observation_series_registry_v1_2.csv
    observation_dictionary_audit_v1_2.json
- ARPAL native parser probe v1.0 = PASS
- QC finali regionali già PASS

Parser espliciti:
PIEMONTE
    daily_meteo: colonne elencate da ptot_columns -> PRECIP_MM
    daily_hydro: level_columns -> RIVER_STAGE_M
                 discharge_columns -> DISCHARGE_M3_S
    date source: colonna `data`, settembre-dicembre 1987-2025

VALLE D'AOSTA
    schema normalizzato:
      station_code, parameter, unit, timestamp_source,
      value_numeric, value_raw, source_zip, source_entry
    righe numeriche settembre-dicembre 1996-2025
    timestamp NON convertito in UTC:
      UNRESOLVED_SOURCE_TIME_CONVENTION

LIGURIA / ARPAL
    parser nativo validato:
      cp1252, CSV a virgola, 5 campi,
      campo 0 = inizio intervallo,
      campo 1 = fine intervallo,
      campo 2 = valore.
    Il portale dichiara i timestamp UTC; la provenienza viene conservata.

Output:
nw_observations_values_v1_0/
  series/
    ARPA_PIEMONTE/*.csv.gz
    CENTRO_FUNZIONALE_RAVDA/*.csv.gz
    ARPAL/*.csv.gz
  metadata/
    <same>.meta.json
  standardized_series_manifest_v1_0.csv
  standardized_value_summary_v1_0.csv
  standardized_values_audit_v1_0.json
  standardized_values_audit_v1_0.txt

Proprietà:
- restart-safe: una serie già PASS viene saltata se il file sorgente
  non è cambiato (size + mtime_ns);
- scrittura atomica dei CSV.gz;
- nessuna modifica dei sorgenti;
- nessuna imputazione;
- vengono scritte soltanto osservazioni numeriche;
- le varianti di colonna Piemonte restano distinte tramite source_column.
"""

from __future__ import annotations

import ast
import csv
import gzip
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_DATA_OK = 1312
TARGET_MONTHS = {9, 10, 11, 12}

DATE_PATTERNS = [
    re.compile(
        r"^\s*(?P<d>\d{1,2})/(?P<m>\d{1,2})/(?P<y>\d{4})"
        r"(?:[ T]+(?P<h>\d{1,2}):(?P<mi>\d{2}))?\s*$"
    ),
    re.compile(
        r"^\s*(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})"
        r"(?:[ T]+(?P<h>\d{1,2}):(?P<mi>\d{2}))?\s*$"
    ),
]


CANONICAL_UNITS = {
    "PRECIP_MM": "mm",
    "RIVER_STAGE_M": "m",
    "DISCHARGE_M3_S": "m3/s",
    "DISCHARGE_MIN_M3_S": "m3/s",
    "DISCHARGE_MAX_M3_S": "m3/s",
    "AIR_TEMP_C": "degC",
    "REL_HUMIDITY_PCT": "%",
    "WIND_SPEED_M_S": "m/s",
    "WIND_DIR_DEG": "deg",
    "AIR_PRESSURE_HPA": "hPa",
    "SNOW_DEPTH_CM": "cm",
    "SOLAR_RAD_W_M2": "W/m2",
    "REFLECTED_SOLAR_RAD_W_M2": "W/m2",
    "SUNSHINE_DURATION_MIN": "min",
    "LEAF_WETNESS_DURATION_S": "s",
    "LEAF_WETNESS_LOWER_PCT": "%",
    "LEAF_WETNESS_UPPER_PCT": "%",
}


def safe_name(s):
    s = str(s or "")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    return s or "series"


def source_fingerprint(path: Path):
    st = path.stat()
    return {
        "source_size_bytes": int(st.st_size),
        "source_mtime_ns": int(st.st_mtime_ns),
    }


def canonical_unit(code, source_unit):
    return CANONICAL_UNITS.get(
        str(code),
        str(source_unit or "").strip(),
    )


def value_semantics(code, provider):
    code = str(code)

    if code == "PRECIP_MM":
        if provider == "ARPAL":
            return "interval_accumulation"
        if provider == "ARPA_PIEMONTE":
            return "daily_accumulation"
        return "source_parameter_precipitation"

    if code == "RIVER_STAGE_M":
        if provider == "ARPAL":
            return "interval_mean_stage"
        if provider == "ARPA_PIEMONTE":
            return "daily_mean_stage"
        return "source_parameter_stage"

    if code == "DISCHARGE_M3_S":
        if provider == "ARPA_PIEMONTE":
            return "daily_mean_discharge"
        return "source_parameter_discharge"

    if code == "DISCHARGE_MIN_M3_S":
        return "source_parameter_min_discharge"

    if code == "DISCHARGE_MAX_M3_S":
        return "source_parameter_max_discharge"

    return "source_parameter_value"


def parse_qc_column_list(value, actual_columns):
    """
    Risolve una colonna QC come ptot_columns/level_columns/discharge_columns
    contro l'header reale del file. Non accetta nomi inventati.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    raw = str(value).strip()
    if not raw:
        return []

    candidates = []

    # Eventuale rappresentazione Python/JSON di lista.
    if raw.startswith("[") and raw.endswith("]"):
        try:
            obj = ast.literal_eval(raw)
            if isinstance(obj, (list, tuple)):
                candidates.extend(str(x).strip() for x in obj)
        except Exception:
            pass

    # Separazioni comuni.
    if not candidates:
        cleaned = raw.strip("[](){}")
        parts = re.split(r"[|;,]+", cleaned)
        candidates.extend(
            p.strip().strip("'\"")
            for p in parts
            if p.strip()
        )

    actual = {str(c): str(c) for c in actual_columns}

    resolved = []
    for c in candidates:
        if c in actual and c not in resolved:
            resolved.append(c)

    # Se il QC contiene un singolo nome esatto.
    if not resolved and raw in actual:
        resolved.append(raw)

    return resolved


def atomic_write_csv_gz(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(
        tmp,
        index=False,
        compression="gzip",
    )
    tmp.replace(path)


def atomic_write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def build_common_frame(
    *,
    source_series_id,
    provider,
    station_id,
    target,
    receptor_ids_source,
    variable_code,
    source_column,
    unit_source,
    time_resolution,
    timestamp_source,
    interval_end_source,
    date_source,
    timestamp_utc,
    value_numeric,
    value_raw,
    timezone_status,
):
    n = len(value_numeric)

    return pd.DataFrame({
        "source_series_id": [source_series_id] * n,
        "provider": [provider] * n,
        "station_id": [station_id] * n,
        "target": [target] * n,
        "receptor_ids_source": [receptor_ids_source] * n,
        "variable_code": [variable_code] * n,
        "source_column": [source_column] * n,
        "unit_source": [unit_source] * n,
        "unit_canonical": [
            canonical_unit(variable_code, unit_source)
        ] * n,
        "time_resolution": [time_resolution] * n,
        "timestamp_source": timestamp_source,
        "interval_end_source": interval_end_source,
        "date_source": date_source,
        "timestamp_utc": timestamp_utc,
        "value_numeric": value_numeric,
        "value_raw": value_raw,
        "value_semantics": [
            value_semantics(variable_code, provider)
        ] * n,
        "timezone_status": [timezone_status] * n,
    })


# ---------------------------------------------------------------------
# ARPAL native parser — same validated semantics as probe v1.0
# ---------------------------------------------------------------------

def looks_utf16(raw: bytes):
    sample = raw[:20000]
    if len(sample) < 20:
        return None

    even = sample[0::2]
    odd = sample[1::2]
    even_zero = even.count(0) / max(1, len(even))
    odd_zero = odd.count(0) / max(1, len(odd))

    if odd_zero > 0.25 and even_zero < 0.05:
        return "utf-16-le"
    if even_zero > 0.25 and odd_zero < 0.05:
        return "utf-16-be"
    return None


def decode_arpal_bytes(raw: bytes):
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig"), "utf-8-sig"
    if raw.startswith(b"\xff\xfe"):
        return raw.decode("utf-16-le"), "utf-16-le-bom"
    if raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16-be"), "utf-16-be-bom"

    enc16 = looks_utf16(raw)
    if enc16:
        try:
            return raw.decode(enc16), enc16
        except Exception:
            pass

    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass

    try:
        return raw.decode("cp1252"), "cp1252"
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace"), "latin-1"


def parse_timestamp_token(token: str):
    s = str(token or "").strip().strip('"').strip("'")

    for pat in DATE_PATTERNS:
        m = pat.match(s)
        if not m:
            continue
        try:
            return datetime(
                int(m.group("y")),
                int(m.group("m")),
                int(m.group("d")),
                int(m.group("h") or 0),
                int(m.group("mi") or 0),
            )
        except Exception:
            return None

    return None


def parse_numeric_token(token: str):
    s = str(token or "").strip().strip('"').strip("'").replace("\xa0", " ")
    if not s:
        return None

    s2 = s.replace(" ", "").replace(",", ".")

    if not re.fullmatch(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)",
        s2,
    ):
        return None

    try:
        return float(s2)
    except Exception:
        return None


def parse_arpal_data_line(line: str):
    try:
        fields = next(csv.reader([line], delimiter=","))
    except Exception:
        return None

    fields = [str(x).strip() for x in fields]

    # Probe v1.0 ha dimostrato su 411/411 file DATA_OK:
    # 5 campi e valore all'indice 2.
    if len(fields) != 5:
        return None

    start = parse_timestamp_token(fields[0])
    end = parse_timestamp_token(fields[1])

    if start is None or end is None:
        return None

    value = parse_numeric_token(fields[2])
    if value is None:
        return None

    return {
        "start": start,
        "end": end,
        "value": value,
        "value_raw": fields[2],
    }


# ---------------------------------------------------------------------
# Provider builders
# ---------------------------------------------------------------------

def build_piemonte_series(
    reg_row,
    qc_row,
    source_path: Path,
):
    df = pd.read_csv(source_path, low_memory=False)

    if "data" not in df.columns:
        raise ValueError("PIEMONTE: colonna `data` mancante")

    dates = pd.to_datetime(
        df["data"],
        errors="coerce",
        dayfirst=False,
    )

    base_mask = (
        dates.notna()
        & dates.dt.year.between(1987, 2025)
        & dates.dt.month.isin(TARGET_MONTHS)
    )

    kind = str(reg_row["kind_source"])
    outputs = []
    selected_columns = []

    mappings = []

    if kind == "daily_meteo":
        cols = parse_qc_column_list(
            qc_row.get("ptot_columns"),
            df.columns,
        )
        if not cols:
            raise ValueError(
                "PIEMONTE daily_meteo: ptot_columns non risolte"
            )
        for c in cols:
            mappings.append((c, "PRECIP_MM"))

    elif kind == "daily_hydro":
        lcols = parse_qc_column_list(
            qc_row.get("level_columns"),
            df.columns,
        )
        qcols = parse_qc_column_list(
            qc_row.get("discharge_columns"),
            df.columns,
        )

        if "RIVER_STAGE_M" in str(reg_row["variable_codes"]).split("|"):
            if not lcols:
                raise ValueError(
                    "PIEMONTE daily_hydro: level_columns non risolte"
                )
            for c in lcols:
                mappings.append((c, "RIVER_STAGE_M"))

        if "DISCHARGE_M3_S" in str(reg_row["variable_codes"]).split("|"):
            if not qcols:
                raise ValueError(
                    "PIEMONTE daily_hydro: discharge_columns non risolte"
                )
            for c in qcols:
                mappings.append((c, "DISCHARGE_M3_S"))

    else:
        raise ValueError(f"PIEMONTE kind inatteso: {kind}")

    for source_col, code in mappings:
        vals = pd.to_numeric(
            df[source_col],
            errors="coerce",
        )
        mask = base_mask & vals.notna()

        if not mask.any():
            # È una serie DATA_OK: ogni variabile dichiarata deve produrre
            # almeno un dato numerico.
            raise ValueError(
                f"PIEMONTE {source_col}/{code}: zero valori numerici"
            )

        d = dates.loc[mask]
        v = vals.loc[mask]

        part = build_common_frame(
            source_series_id=reg_row["source_series_id"],
            provider="ARPA_PIEMONTE",
            station_id=str(reg_row["station_id"]),
            target="",
            receptor_ids_source=str(
                reg_row.get("receptor_ids_source", "") or ""
            ),
            variable_code=code,
            source_column=source_col,
            unit_source=canonical_unit(code, ""),
            time_resolution="daily",
            timestamp_source=d.dt.strftime("%Y-%m-%d").tolist(),
            interval_end_source=[""] * len(v),
            date_source=d.dt.strftime("%Y-%m-%d").tolist(),
            timestamp_utc=[""] * len(v),
            value_numeric=v.astype(float).tolist(),
            value_raw=df.loc[mask, source_col].astype(str).tolist(),
            timezone_status="DAILY_SOURCE_DATE",
        )

        outputs.append(part)
        selected_columns.append(source_col)

    out = pd.concat(outputs, ignore_index=True)

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
            f"PIEMONTE duplicate keys nel long output: {dup}"
        )

    return out, {
        "selected_source_columns": selected_columns,
        "expected_rows_qc": None,
    }


def build_vda_series(
    reg_row,
    qc_row,
    source_path: Path,
):
    usecols = [
        "station_code",
        "parameter",
        "unit",
        "timestamp_source",
        "value_numeric",
        "value_raw",
        "source_zip",
        "source_entry",
    ]

    df = pd.read_csv(
        source_path,
        usecols=usecols,
        low_memory=False,
    )

    ts = pd.to_datetime(
        df["timestamp_source"],
        errors="coerce",
    )
    vals = pd.to_numeric(
        df["value_numeric"],
        errors="coerce",
    )

    mask = (
        ts.notna()
        & vals.notna()
        & ts.dt.year.between(1996, 2025)
        & ts.dt.month.isin(TARGET_MONTHS)
    )

    expected = int(qc_row["numeric_sepdec_1996_2025"])
    actual = int(mask.sum())

    if actual != expected:
        raise ValueError(
            f"VDA row count {actual} != QC {expected}"
        )

    if actual <= 0:
        raise ValueError("VDA DATA_OK con zero righe numeriche")

    d = ts.loc[mask]
    v = vals.loc[mask]

    codes = [
        c.strip()
        for c in str(reg_row["variable_codes"]).split("|")
        if c.strip()
    ]
    if len(codes) != 1:
        raise ValueError(
            f"VDA variable_codes inatteso: {codes}"
        )
    code = codes[0]

    out = build_common_frame(
        source_series_id=reg_row["source_series_id"],
        provider="CENTRO_FUNZIONALE_RAVDA",
        station_id=str(reg_row["station_id"]),
        target="",
        receptor_ids_source=str(
            reg_row.get("receptor_ids_source", "") or ""
        ),
        variable_code=code,
        source_column="value_numeric",
        unit_source=str(reg_row.get("unit_source", "") or ""),
        time_resolution="source_hourly_or_parameter_native",
        timestamp_source=d.dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        ).tolist(),
        interval_end_source=[""] * actual,
        date_source=d.dt.strftime("%Y-%m-%d").tolist(),
        timestamp_utc=[""] * actual,
        value_numeric=v.astype(float).tolist(),
        value_raw=df.loc[mask, "value_raw"].astype(str).tolist(),
        timezone_status="UNRESOLVED_SOURCE_TIME_CONVENTION",
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
            f"VDA duplicate keys nel long output: {dup}"
        )

    return out, {
        "selected_source_columns": ["value_numeric"],
        "expected_rows_qc": expected,
    }


def build_arpal_series(
    reg_row,
    qc_row,
    source_path: Path,
):
    raw = source_path.read_bytes()
    text, enc = decode_arpal_bytes(raw)

    if enc != "cp1252":
        raise ValueError(
            f"ARPAL encoding inatteso {enc}; probe validato cp1252"
        )

    rows = []

    for line in text.splitlines():
        if not line.strip():
            continue
        rec = parse_arpal_data_line(line)
        if rec is None:
            continue
        rows.append(rec)

    expected = int(qc_row["observation_rows"])

    if len(rows) != expected:
        raise ValueError(
            f"ARPAL parsed rows {len(rows)} != QC {expected}"
        )

    year = int(qc_row["year"])

    starts = [r["start"] for r in rows]
    ends = [r["end"] for r in rows]

    if len(set(starts)) != len(starts):
        raise ValueError("ARPAL timestamp iniziali duplicati")

    if any(
        dt.year != year or dt.month not in TARGET_MONTHS
        for dt in starts
    ):
        raise ValueError("ARPAL righe fuori anno/stagione")

    codes = [
        c.strip()
        for c in str(reg_row["variable_codes"]).split("|")
        if c.strip()
    ]

    if len(codes) != 1:
        raise ValueError(
            f"ARPAL variable_codes inatteso: {codes}"
        )
    code = codes[0]

    if code not in {"PRECIP_MM", "RIVER_STAGE_M"}:
        raise ValueError(
            f"ARPAL variable_code inatteso: {code}"
        )

    # Tutte le righe validate sono intervalli start/end.
    interval_seconds = [
        int((e - s).total_seconds())
        for s, e in zip(starts, ends)
    ]

    if any(x <= 0 for x in interval_seconds):
        raise ValueError("ARPAL intervallo temporale non positivo")

    out = build_common_frame(
        source_series_id=reg_row["source_series_id"],
        provider="ARPAL",
        station_id=str(reg_row["station_id"]),
        target=str(reg_row.get("target", "") or ""),
        receptor_ids_source=str(
            reg_row.get("receptor_ids_source", "") or ""
        ),
        variable_code=code,
        source_column="field_2",
        unit_source=str(reg_row.get("unit_source", "") or ""),
        time_resolution="interval_hourly_source",
        timestamp_source=[
            d.strftime("%Y-%m-%d %H:%M:%S")
            for d in starts
        ],
        interval_end_source=[
            d.strftime("%Y-%m-%d %H:%M:%S")
            for d in ends
        ],
        date_source=[
            d.strftime("%Y-%m-%d")
            for d in starts
        ],
        timestamp_utc=[
            d.strftime("%Y-%m-%dT%H:%M:%SZ")
            for d in starts
        ],
        value_numeric=[float(r["value"]) for r in rows],
        value_raw=[str(r["value_raw"]) for r in rows],
        timezone_status="PORTAL_DECLARED_UTC_PRESERVE_PROVENANCE",
    )

    out["interval_seconds"] = interval_seconds

    return out, {
        "selected_source_columns": ["field_2"],
        "expected_rows_qc": expected,
        "parser_encoding": enc,
        "interval_seconds_unique": sorted(
            set(interval_seconds)
        ),
    }


def main():
    root = Path(__file__).resolve().parent

    canonical_root = root / "nw_observations_standardized_v1_2"
    reg_path = canonical_root / "observation_series_registry_v1_2.csv"
    dict_audit_path = canonical_root / "observation_dictionary_audit_v1_2.json"
    arpal_probe_path = (
        canonical_root
        / "arpal_native_parser_probe_v1_0"
        / "arpal_native_parser_probe_v1_0.json"
    )

    pie_qc_path = (
        root
        / "observations_nw"
        / "piemonte"
        / "qc_final_v1_0"
        / "file_qc_final_v1_0.csv"
    )
    vda_qc_path = (
        root
        / "observations_nw"
        / "valle_d_aosta"
        / "final_v1_0"
        / "qc"
        / "vda_file_qc.csv"
    )
    lig_qc_path = (
        root
        / "observations_nw"
        / "liguria_groundtruth_v1_3"
        / "qc_final_v1_1"
        / "file_qc_final_v1_1.csv"
    )

    out_root = root / "nw_observations_values_v1_0"
    series_root = out_root / "series"
    meta_root = out_root / "metadata"
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 136)
    print("NW OBSERVATIONS — STANDARDIZED DATA_OK VALUES BUILDER v1.0")
    print("=" * 136)

    for p in [
        reg_path,
        dict_audit_path,
        arpal_probe_path,
        pie_qc_path,
        vda_qc_path,
        lig_qc_path,
    ]:
        if not p.exists():
            raise SystemExit(f"Manca: {p}")

    dict_audit = json.loads(
        dict_audit_path.read_text(encoding="utf-8")
    )
    arpal_probe = json.loads(
        arpal_probe_path.read_text(encoding="utf-8")
    )

    if dict_audit.get("overall_status") != "PASS":
        raise SystemExit("Dizionario canonico v1.2 non PASS.")

    if arpal_probe.get("overall_status") != "PASS":
        raise SystemExit("ARPAL native parser probe v1.0 non PASS.")

    reg = pd.read_csv(reg_path, low_memory=False)
    data_ok = reg[
        reg["scientific_status"]
        .astype(str)
        .str.upper()
        .eq("DATA_OK")
    ].copy()

    if len(data_ok) != EXPECTED_DATA_OK:
        raise SystemExit(
            f"DATA_OK={len(data_ok)}, atteso={EXPECTED_DATA_OK}"
        )

    pie_qc = pd.read_csv(pie_qc_path, low_memory=False)
    vda_qc = pd.read_csv(vda_qc_path, low_memory=False)
    lig_qc = pd.read_csv(lig_qc_path, low_memory=False)

    # Lookups ----------------------------------------------------------
    pie_lookup = {}
    for _, r in pie_qc.iterrows():
        key = (
            str(r["kind"]).strip(),
            str(r["station_id"]).strip(),
        )
        pie_lookup[key] = r.to_dict()

    vda_lookup = {}
    for _, r in vda_qc.iterrows():
        key = (
            str(r["station_code"]).strip(),
            str(r["parameter"]).strip(),
        )
        vda_lookup[key] = r.to_dict()

    lig_lookup = {}
    for _, r in lig_qc.iterrows():
        key = (
            str(r["target"]).strip(),
            int(r["year"]),
        )
        lig_lookup[key] = r.to_dict()

    manifest = []
    processed = 0
    skipped = 0
    errors = 0

    for seq, (_, rr) in enumerate(data_ok.iterrows(), 1):
        provider = str(rr["provider"])
        source_series_id = str(rr["source_series_id"])
        source_path = Path(str(rr["source_data_path"]))

        provider_dir = series_root / safe_name(provider)
        provider_meta_dir = meta_root / safe_name(provider)

        fname = safe_name(source_series_id) + ".csv.gz"
        out_path = provider_dir / fname
        meta_path = provider_meta_dir / (fname + ".meta.json")

        fp = source_fingerprint(source_path)

        # Restart-safe skip --------------------------------------------
        if out_path.exists() and meta_path.exists():
            try:
                old = json.loads(
                    meta_path.read_text(encoding="utf-8")
                )
                if (
                    old.get("status") == "PASS"
                    and old.get("source_size_bytes")
                    == fp["source_size_bytes"]
                    and old.get("source_mtime_ns")
                    == fp["source_mtime_ns"]
                ):
                    manifest.append(old)
                    skipped += 1
                    if seq % 100 == 0:
                        print(
                            f"{seq}/{EXPECTED_DATA_OK} | "
                            f"processed={processed} skipped={skipped} errors={errors}"
                        )
                    continue
            except Exception:
                pass

        try:
            if provider == "ARPA_PIEMONTE":
                key = (
                    str(rr["kind_source"]).strip(),
                    str(rr["station_id"]).strip(),
                )
                qc_row = pie_lookup.get(key)
                if qc_row is None:
                    raise ValueError(
                        f"QC Piemonte non trovato: {key}"
                    )

                out_df, extra = build_piemonte_series(
                    rr,
                    qc_row,
                    source_path,
                )

            elif provider == "CENTRO_FUNZIONALE_RAVDA":
                key = (
                    str(rr["station_id"]).strip(),
                    str(rr["parameter_source"]).strip(),
                )
                qc_row = vda_lookup.get(key)
                if qc_row is None:
                    raise ValueError(
                        f"QC VdA non trovato: {key}"
                    )

                out_df, extra = build_vda_series(
                    rr,
                    qc_row,
                    source_path,
                )

            elif provider == "ARPAL":
                key = (
                    str(rr["target"]).strip(),
                    int(float(rr["source_year"])),
                )
                qc_row = lig_lookup.get(key)
                if qc_row is None:
                    raise ValueError(
                        f"QC ARPAL non trovato: {key}"
                    )

                out_df, extra = build_arpal_series(
                    rr,
                    qc_row,
                    source_path,
                )

            else:
                raise ValueError(
                    f"Provider inatteso: {provider}"
                )

            if len(out_df) <= 0:
                raise ValueError("Output standardizzato vuoto")

            # Common checks.
            if out_df["value_numeric"].isna().any():
                raise ValueError(
                    "NaN in value_numeric dopo standardizzazione"
                )

            date_min = str(out_df["date_source"].min())
            date_max = str(out_df["date_source"].max())

            variable_counts = {
                str(k): int(v)
                for k, v in (
                    out_df["variable_code"]
                    .value_counts()
                    .to_dict()
                    .items()
                )
            }

            atomic_write_csv_gz(
                out_df,
                out_path,
            )

            meta = {
                "status": "PASS",
                "provider": provider,
                "source_series_id": source_series_id,
                "station_id": str(rr["station_id"]),
                "target": str(rr.get("target", "") or ""),
                "source_data_path": str(source_path),
                "output_path": str(out_path),
                "rows": int(len(out_df)),
                "date_min": date_min,
                "date_max": date_max,
                "variable_counts": variable_counts,
                "variable_codes_registry": str(
                    rr["variable_codes"]
                ),
                "timezone_status": str(
                    rr["timezone_status"]
                ),
                "source_size_bytes": fp[
                    "source_size_bytes"
                ],
                "source_mtime_ns": fp[
                    "source_mtime_ns"
                ],
                **extra,
            }

            atomic_write_json(
                meta,
                meta_path,
            )

            manifest.append(meta)
            processed += 1

        except Exception as exc:
            errors += 1

            meta = {
                "status": "ERROR",
                "provider": provider,
                "source_series_id": source_series_id,
                "station_id": str(rr["station_id"]),
                "target": str(rr.get("target", "") or ""),
                "source_data_path": str(source_path),
                "output_path": str(out_path),
                "rows": 0,
                "date_min": "",
                "date_max": "",
                "variable_counts": {},
                "variable_codes_registry": str(
                    rr["variable_codes"]
                ),
                "timezone_status": str(
                    rr["timezone_status"]
                ),
                "source_size_bytes": fp[
                    "source_size_bytes"
                ],
                "source_mtime_ns": fp[
                    "source_mtime_ns"
                ],
                "error": repr(exc),
            }

            atomic_write_json(
                meta,
                meta_path,
            )
            manifest.append(meta)

        if seq % 100 == 0:
            print(
                f"{seq}/{EXPECTED_DATA_OK} | "
                f"processed={processed} skipped={skipped} errors={errors}"
            )

    man = pd.DataFrame(manifest)

    # Audit ------------------------------------------------------------
    pass_rows = man[
        man["status"].eq("PASS")
    ].copy()
    error_rows = man[
        ~man["status"].eq("PASS")
    ].copy()

    reasons = []

    if len(man) != EXPECTED_DATA_OK:
        reasons.append(
            f"MANIFEST_ROWS={len(man)} expected={EXPECTED_DATA_OK}"
        )

    if len(pass_rows) != EXPECTED_DATA_OK:
        reasons.append(
            f"PASS_SERIES={len(pass_rows)} expected={EXPECTED_DATA_OK}"
        )

    if len(error_rows):
        reasons.append(
            f"ERROR_SERIES={len(error_rows)}"
        )

    # Provider series counts.
    provider_counts = (
        pass_rows["provider"]
        .value_counts()
        .to_dict()
    )

    expected_provider = {
        "ARPA_PIEMONTE": 397,
        "CENTRO_FUNZIONALE_RAVDA": 504,
        "ARPAL": 411,
    }

    for p, n in expected_provider.items():
        got = int(provider_counts.get(p, 0))
        if got != n:
            reasons.append(
                f"{p}_PASS={got} expected={n}"
            )

    # All output files must exist.
    missing_outputs = int(
        (~pass_rows["output_path"].map(
            lambda s: Path(str(s)).exists()
        )).sum()
    )
    if missing_outputs:
        reasons.append(
            f"MISSING_OUTPUT_FILES={missing_outputs}"
        )

    # Expand variable counts for summary.
    summary_rows = []
    for _, r in pass_rows.iterrows():
        vc = r["variable_counts"]
        if isinstance(vc, str):
            try:
                vc = ast.literal_eval(vc)
            except Exception:
                vc = {}

        if not isinstance(vc, dict):
            vc = {}

        for code, n in vc.items():
            summary_rows.append({
                "provider": r["provider"],
                "variable_code": str(code),
                "rows": int(n),
                "series": 1,
            })

    summary = pd.DataFrame(summary_rows)

    if len(summary):
        summary = (
            summary.groupby(
                ["provider", "variable_code"]
            )
            .agg(
                rows=("rows", "sum"),
                series=("series", "sum"),
            )
            .reset_index()
            .sort_values(
                ["provider", "variable_code"]
            )
        )
    else:
        summary = pd.DataFrame(
            columns=[
                "provider",
                "variable_code",
                "rows",
                "series",
            ]
        )

    manifest_path = (
        out_root
        / "standardized_series_manifest_v1_0.csv"
    )
    summary_path = (
        out_root
        / "standardized_value_summary_v1_0.csv"
    )

    man.to_csv(
        manifest_path,
        index=False,
    )
    summary.to_csv(
        summary_path,
        index=False,
    )

    total_rows = int(
        pd.to_numeric(
            pass_rows["rows"],
            errors="coerce",
        ).fillna(0).sum()
    )

    overall = (
        "PASS"
        if not reasons
        else "REVIEW"
    )

    report = {
        "version": "1.0",
        "overall_status": overall,
        "expected_data_ok_series": EXPECTED_DATA_OK,
        "manifest_rows": int(len(man)),
        "pass_series": int(len(pass_rows)),
        "error_series": int(len(error_rows)),
        "processed_this_run": processed,
        "skipped_restart_safe": skipped,
        "total_standardized_numeric_rows": total_rows,
        "provider_pass_counts": {
            str(k): int(v)
            for k, v in provider_counts.items()
        },
        "missing_output_files": missing_outputs,
        "time_policy": {
            "ARPA_PIEMONTE": (
                "daily source date; no subdaily timezone inference"
            ),
            "CENTRO_FUNZIONALE_RAVDA": (
                "timestamp_source preserved; UTC conversion prohibited "
                "until source time convention is resolved"
            ),
            "ARPAL": (
                "portal-declared UTC; start and end of source interval preserved"
            ),
        },
        "piemonte_policy": (
            "All QC-declared core source columns are preserved as separate "
            "source_column variants; no silent collapse of level/discharge variants."
        ),
        "no_imputation": True,
        "raw_modified": False,
        "reasons": reasons,
    }

    audit_json = (
        out_root
        / "standardized_values_audit_v1_0.json"
    )
    atomic_write_json(
        report,
        audit_json,
    )

    lines = [
        "=" * 136,
        "NW OBSERVATIONS — STANDARDIZED DATA_OK VALUES BUILDER v1.0",
        "=" * 136,
        f"OVERALL STATUS                  : {overall}",
        f"DATA_OK series expected         : {EXPECTED_DATA_OK}",
        f"Manifest rows                   : {len(man)}",
        f"PASS series                     : {len(pass_rows)}",
        f"ERROR series                    : {len(error_rows)}",
        f"Processed this run              : {processed}",
        f"Restart-safe skipped            : {skipped}",
        f"Total standardized numeric rows : {total_rows}",
        f"Missing output files            : {missing_outputs}",
        "",
        "PASS SERIES BY PROVIDER",
    ]

    for p in [
        "ARPA_PIEMONTE",
        "CENTRO_FUNZIONALE_RAVDA",
        "ARPAL",
    ]:
        lines.append(
            f"{p:<30}: {int(provider_counts.get(p, 0))}"
        )

    lines += [
        "",
        "VALUE SUMMARY",
        summary.to_string(index=False),
        "",
        "TIME POLICY",
        "Piemonte      : daily source date",
        "Valle d'Aosta : timestamp_source preserved; UTC unresolved",
        "Liguria       : portal-declared UTC; interval start/end preserved",
        "",
        f"Manifest: {manifest_path}",
        f"Summary : {summary_path}",
    ]

    audit_txt = (
        out_root
        / "standardized_values_audit_v1_0.txt"
    )
    audit_txt.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 136)
    print("\n".join(lines[3:]))
    print("\n" + "=" * 136)
    print(f"OVERALL STATUS : {overall}")
    print(f"Output         : {out_root}")

    if reasons:
        print("REASONS:")
        for r in reasons:
            print(f"  - {r}")

    print("=" * 136)

    if overall != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
