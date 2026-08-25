#!/usr/bin/env python3
"""
ARPA PIEMONTE — AUDIT SCIENTIFICO FINALE v1.0
==============================================

Scopo
-----
Esaminare in sola lettura il blocco osservativo ARPA Piemonte già scaricato
da download_observations_nw_v1_1.py e produrre un QC conclusivo indipendente.

Input attesi:
  observations_nw/piemonte/daily_meteo/*_autumn_1987_2025.csv
  observations_nw/piemonte/daily_hydro/*_autumn_1987_2025.csv
  observations_nw/piemonte/daily_meteo/*_raw.json
  observations_nw/piemonte/daily_hydro/*_raw.json
  observations_nw/piemonte/daily_meteo/*_metadata.json
  observations_nw/piemonte/daily_hydro/*_metadata.json
  observations_nw/station_basin_map.csv

Variabili scientifiche principali:
  METEO : ptot / precipitazione totale giornaliera
  HYDRO : livellomedio / livello
          portatamedia / portata

Periodo:
  settembre-dicembre 1987-2025

Controlli:
- integrità e leggibilità CSV/JSON;
- corrispondenza CSV <-> raw.json <-> metadata.json;
- date valide, solo settembre-dicembre, range 1987-2025;
- duplicati giornalieri e ordine temporale;
- individuazione robusta delle colonne ptot/livello/portata;
- conteggio valori numerici, missing e valori testuali non numerici;
- precipitazioni negative (anomalia fisica);
- copertura stazione x anno rispetto ai 122 giorni Sep-Dec;
- sintesi per bacino usando observations_nw/station_basin_map.csv.

Classificazione:
  DATA_OK
      struttura valida + almeno un valore utile per la variabile core;
  NO_DATA_SOURCE
      struttura valida ma nessun valore utile per la variabile core;
  REVIEW
      anomalia tecnica/scientifica residua (duplicati, date invalide,
      colonna core mancante, valori testuali anomali, ptot negativa, ecc.).

IMPORTANTE
----------
Lo script NON modifica, NON riscarica e NON sovrascrive i dati raw.
Produce solo file QC in:
  observations_nw/piemonte/qc_final_v1_0/

Nota scientifica:
la banca dati automatizzata ARPA Piemonte qui usata è giornaliera.
Livello medio e portata media sono adatti allo screening storico delle piene,
ma possono attenuare i picchi rapidi. I dati orari/suborari saranno acquisiti
solo per gli eventi critici selezionati.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent

OBS = ROOT / "observations_nw"
PIE = OBS / "piemonte"
METEO = PIE / "daily_meteo"
HYDRO = PIE / "daily_hydro"
STATION_MAP = OBS / "station_basin_map.csv"

OUT = PIE / "qc_final_v1_0"

START_YEAR = 1987
END_YEAR = 2025
AUTUMN_MONTHS = {9, 10, 11, 12}
EXPECTED_AUTUMN_DAYS = 122

# Sono valori testuali che consideriamo missing, non errori.
MISSING_TEXT = {
    "", "NAN", "NONE", "NULL", "N/A", "NA", "ND", "N.D.", "-", "--"
}


def norm(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.replace("\xa0", " ").strip().upper()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_date(v: Any):
    if v in (None, ""):
        return None
    s = str(v).strip()
    # L'API storica è giornaliera; accettiamo comunque timestamp ISO.
    candidates = [
        s,
        s[:19],
        s[:10],
    ]
    for x in candidates:
        try:
            return datetime.fromisoformat(x)
        except Exception:
            pass
    for fmt in (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    return None


def parse_number(v: Any):
    """
    Ritorna:
      ("missing", None)
      ("numeric", float)
      ("invalid", raw_string)
    Lo zero è sempre numerico valido.
    """
    if v is None:
        return "missing", None

    if isinstance(v, bool):
        return "invalid", str(v)

    if isinstance(v, (int, float)):
        if isinstance(v, float) and not math.isfinite(v):
            return "missing", None
        return "numeric", float(v)

    s = str(v).strip()
    if norm(s) in MISSING_TEXT:
        return "missing", None

    # Gestione virgola/punto decimale.
    t = s.replace(" ", "")
    if "," in t and "." in t:
        if t.rfind(",") > t.rfind("."):
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(",", "")
    elif "," in t:
        t = t.replace(",", ".")

    try:
        x = float(t)
    except Exception:
        return "invalid", s

    if not math.isfinite(x):
        return "missing", None

    return "numeric", x


def read_csv_records(path: Path):
    if not path.exists():
        return [], [], "MISSING_FILE"

    if path.stat().st_size == 0:
        return [], [], "EMPTY_FILE"

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            if fields == ["no_records"]:
                return [], fields, "NO_RECORDS"
            rows = list(reader)
            return rows, fields, "OK"
    except UnicodeDecodeError:
        try:
            with path.open("r", encoding="cp1252", newline="") as f:
                reader = csv.DictReader(f)
                fields = reader.fieldnames or []
                if fields == ["no_records"]:
                    return [], fields, "NO_RECORDS"
                rows = list(reader)
                return rows, fields, "OK_CP1252"
        except Exception as exc:
            return [], [], f"CSV_ERROR:{exc}"
    except Exception as exc:
        return [], [], f"CSV_ERROR:{exc}"


def json_status(path: Path):
    if not path.exists():
        return "MISSING"
    try:
        with path.open("r", encoding="utf-8") as f:
            json.load(f)
        return "OK"
    except UnicodeDecodeError:
        try:
            with path.open("r", encoding="cp1252") as f:
                json.load(f)
            return "OK_CP1252"
        except Exception as exc:
            return f"ERROR:{exc}"
    except Exception as exc:
        return f"ERROR:{exc}"


def date_column(fields):
    preferred = ["data", "date", "giorno", "DATA"]
    for p in preferred:
        for f in fields:
            if f == p:
                return f
    nf = {f: norm(f) for f in fields}
    for f, n in nf.items():
        if n in {"DATA", "DATE", "GIORNO"}:
            return f
    return None


def metric_columns(fields, metric):
    nf = {f: norm(f) for f in fields}

    if metric == "ptot":
        return sorted(
            f for f, n in nf.items()
            if n == "PTOT" or "PRECIPITAZIONE" in n
        )

    if metric == "level":
        return sorted(
            f for f, n in nf.items()
            if n.startswith("LIVELLOMEDIO")
            or n.startswith("LIVFREAMEDIO")
            or ("LIVELLO" in n and "CLASSE" not in n)
        )

    if metric == "discharge":
        return sorted(
            f for f, n in nf.items()
            if n.startswith("PORTATAMEDIA")
            or ("PORTATA" in n and "CLASSE" not in n)
        )

    return []


def row_metric(row, columns):
    """
    Cerca il primo valore non-missing tra le colonne candidate.
    Ritorna status, value, column.
    """
    if not columns:
        return "no_column", None, ""

    saw_invalid = None
    for c in columns:
        st, val = parse_number(row.get(c))
        if st == "numeric":
            return "numeric", val, c
        if st == "invalid" and saw_invalid is None:
            saw_invalid = (val, c)

    if saw_invalid is not None:
        return "invalid", saw_invalid[0], saw_invalid[1]

    return "missing", None, ""


def station_from_filename(path: Path):
    # esempio: PIE-004028-900_BOVES_autumn_1987_2025.csv
    stem = path.name
    suffix = f"_autumn_{START_YEAR}_{END_YEAR}.csv"
    if stem.endswith(suffix):
        stem = stem[:-len(suffix)]
    if "_" in stem:
        sid, safe_name = stem.split("_", 1)
    else:
        sid, safe_name = stem, ""
    return sid, safe_name.replace("_", " ")


def load_station_map():
    out = defaultdict(set)
    names = {}
    if not STATION_MAP.exists():
        return out, names

    with STATION_MAP.open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if norm(r.get("provider")) != "ARPA_PIEMONTE":
                continue
            kind = str(r.get("kind", "")).strip()
            sid = str(r.get("station_id", "")).strip()
            if not sid:
                continue
            names[(kind, sid)] = str(r.get("station_name", "")).strip()
            for rid in str(r.get("receptor_ids", "")).split("|"):
                rid = rid.strip()
                if rid:
                    out[(kind, sid)].add(rid)
    return out, names


def percentile(values, p):
    if not values:
        return ""
    x = sorted(values)
    if len(x) == 1:
        return x[0]
    k = (len(x) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return x[int(k)]
    return x[f] * (c - k) + x[c] * (k - f)


def audit_file(path: Path, kind: str, station_map, station_names):
    sid, fallback_name = station_from_filename(path)
    map_kind = "meteo" if kind == "daily_meteo" else "hydro"
    station_name = station_names.get((map_kind, sid), fallback_name)

    raw = path.with_name(path.name.replace(
        f"_autumn_{START_YEAR}_{END_YEAR}.csv", "_raw.json"
    ))
    meta = path.with_name(path.name.replace(
        f"_autumn_{START_YEAR}_{END_YEAR}.csv", "_metadata.json"
    ))

    rows, fields, csv_status = read_csv_records(path)
    raw_status = json_status(raw)
    meta_status = json_status(meta)

    dcol = date_column(fields)

    ptot_cols = metric_columns(fields, "ptot")
    level_cols = metric_columns(fields, "level")
    discharge_cols = metric_columns(fields, "discharge")

    if kind == "daily_meteo":
        core_metrics = [("ptot", ptot_cols)]
    else:
        # per idrologia una serie è scientificamente utile se possiede
        # livello e/o portata.
        core_metrics = [
            ("level", level_cols),
            ("discharge", discharge_cols),
        ]

    invalid_dates = 0
    out_of_period_dates = 0
    out_of_autumn_dates = 0
    duplicate_dates = 0
    out_of_order = 0
    dated_rows = 0

    seen_dates = set()
    prev_dt = None

    metric_stats = {
        "ptot": Counter(),
        "level": Counter(),
        "discharge": Counter(),
    }
    metric_values = {
        "ptot": [],
        "level": [],
        "discharge": [],
    }
    metric_by_year = defaultdict(lambda: {
        "rows": 0,
        "unique_dates": set(),
        "ptot_numeric": 0,
        "ptot_missing": 0,
        "ptot_invalid": 0,
        "level_numeric": 0,
        "level_missing": 0,
        "level_invalid": 0,
        "discharge_numeric": 0,
        "discharge_missing": 0,
        "discharge_invalid": 0,
        "ptot_negative": 0,
    })

    ptot_negative = 0
    ptot_gt_1000 = 0

    for row in rows:
        if not dcol:
            invalid_dates += 1
            continue

        dt = parse_date(row.get(dcol))
        if dt is None:
            invalid_dates += 1
            continue

        dated_rows += 1

        if not (START_YEAR <= dt.year <= END_YEAR):
            out_of_period_dates += 1

        if dt.month not in AUTUMN_MONTHS:
            out_of_autumn_dates += 1

        date_key = dt.date().isoformat()
        if date_key in seen_dates:
            duplicate_dates += 1
        else:
            seen_dates.add(date_key)

        if prev_dt is not None and dt < prev_dt:
            out_of_order += 1
        prev_dt = dt

        y = metric_by_year[dt.year]
        y["rows"] += 1
        y["unique_dates"].add(date_key)

        for metric, cols in (
            ("ptot", ptot_cols),
            ("level", level_cols),
            ("discharge", discharge_cols),
        ):
            st, val, _ = row_metric(row, cols)
            metric_stats[metric][st] += 1

            if st == "numeric":
                metric_values[metric].append(val)
                y[f"{metric}_numeric"] += 1

                if metric == "ptot":
                    if val < 0:
                        ptot_negative += 1
                        y["ptot_negative"] += 1
                    if val > 1000:
                        ptot_gt_1000 += 1

            elif st == "missing":
                y[f"{metric}_missing"] += 1
            elif st == "invalid":
                y[f"{metric}_invalid"] += 1

    # Determine relevant numeric coverage
    if kind == "daily_meteo":
        core_numeric = metric_stats["ptot"]["numeric"]
        missing_core_column = not ptot_cols
        core_invalid = metric_stats["ptot"]["invalid"]
    else:
        core_numeric = (
            metric_stats["level"]["numeric"]
            + metric_stats["discharge"]["numeric"]
        )
        # hydro può avere solo livello oppure solo portata: va bene.
        missing_core_column = not (level_cols or discharge_cols)
        core_invalid = (
            metric_stats["level"]["invalid"]
            + metric_stats["discharge"]["invalid"]
        )

    review_reasons = []

    if not csv_status.startswith("OK") and csv_status != "NO_RECORDS":
        review_reasons.append(csv_status)

    if raw_status.startswith("ERROR") or raw_status == "MISSING":
        review_reasons.append(f"RAW_JSON_{raw_status}")

    if meta_status.startswith("ERROR") or meta_status == "MISSING":
        review_reasons.append(f"METADATA_JSON_{meta_status}")

    if rows and not dcol:
        review_reasons.append("DATE_COLUMN_MISSING")

    if invalid_dates:
        review_reasons.append(f"INVALID_DATES={invalid_dates}")

    if out_of_period_dates:
        review_reasons.append(f"OUT_OF_PERIOD={out_of_period_dates}")

    if out_of_autumn_dates:
        review_reasons.append(f"OUTSIDE_SEPDEC={out_of_autumn_dates}")

    if duplicate_dates:
        review_reasons.append(f"DUPLICATE_DATES={duplicate_dates}")

    if out_of_order:
        review_reasons.append(f"OUT_OF_ORDER={out_of_order}")

    if missing_core_column and rows:
        review_reasons.append("CORE_COLUMN_MISSING")

    if core_invalid:
        review_reasons.append(f"INVALID_CORE_VALUES={core_invalid}")

    if ptot_negative:
        review_reasons.append(f"NEGATIVE_PTOT={ptot_negative}")

    # >1000 mm/day: segnaliamo ma non facciamo fallire automaticamente.
    warning = ""
    if ptot_gt_1000:
        warning = f"PTOT_GT_1000_MM_DAY={ptot_gt_1000}"

    if review_reasons:
        scientific_status = "REVIEW"
    elif core_numeric > 0:
        scientific_status = "DATA_OK"
    else:
        scientific_status = "NO_DATA_SOURCE"

    receptors = sorted(station_map.get((map_kind, sid), set()))

    file_row = {
        "kind": kind,
        "station_id": sid,
        "station_name": station_name,
        "scientific_status": scientific_status,
        "review_reasons": " | ".join(review_reasons),
        "warning": warning,
        "source_csv": str(path.relative_to(ROOT)),
        "raw_json_status": raw_status,
        "metadata_json_status": meta_status,
        "csv_status": csv_status,
        "csv_rows": len(rows),
        "dated_rows": dated_rows,
        "unique_dates": len(seen_dates),
        "invalid_dates": invalid_dates,
        "duplicate_dates": duplicate_dates,
        "out_of_order_rows": out_of_order,
        "out_of_period_dates": out_of_period_dates,
        "outside_sepdec_dates": out_of_autumn_dates,
        "first_date": min(seen_dates) if seen_dates else "",
        "last_date": max(seen_dates) if seen_dates else "",
        "years_with_any_rows": len({
            y for y, q in metric_by_year.items() if q["rows"] > 0
        }),
        "ptot_columns": " | ".join(ptot_cols),
        "level_columns": " | ".join(level_cols),
        "discharge_columns": " | ".join(discharge_cols),
        "ptot_numeric_days": metric_stats["ptot"]["numeric"],
        "ptot_missing_days": metric_stats["ptot"]["missing"],
        "ptot_invalid_days": metric_stats["ptot"]["invalid"],
        "ptot_negative_days": ptot_negative,
        "ptot_gt_1000_days": ptot_gt_1000,
        "level_numeric_days": metric_stats["level"]["numeric"],
        "level_missing_days": metric_stats["level"]["missing"],
        "level_invalid_days": metric_stats["level"]["invalid"],
        "discharge_numeric_days": metric_stats["discharge"]["numeric"],
        "discharge_missing_days": metric_stats["discharge"]["missing"],
        "discharge_invalid_days": metric_stats["discharge"]["invalid"],
        "ptot_min": min(metric_values["ptot"]) if metric_values["ptot"] else "",
        "ptot_max": max(metric_values["ptot"]) if metric_values["ptot"] else "",
        "level_min": min(metric_values["level"]) if metric_values["level"] else "",
        "level_max": max(metric_values["level"]) if metric_values["level"] else "",
        "discharge_min": (
            min(metric_values["discharge"]) if metric_values["discharge"] else ""
        ),
        "discharge_max": (
            max(metric_values["discharge"]) if metric_values["discharge"] else ""
        ),
        "receptor_ids": " | ".join(receptors),
    }

    year_rows = []
    for year in range(START_YEAR, END_YEAR + 1):
        y = metric_by_year[year]
        unique = len(y["unique_dates"])
        year_rows.append({
            "kind": kind,
            "station_id": sid,
            "station_name": station_name,
            "year": year,
            "rows": y["rows"],
            "unique_dates": unique,
            "coverage_pct_122_days": (
                round(100.0 * unique / EXPECTED_AUTUMN_DAYS, 3)
                if unique else 0.0
            ),
            "ptot_numeric_days": y["ptot_numeric"],
            "ptot_missing_days": y["ptot_missing"],
            "ptot_invalid_days": y["ptot_invalid"],
            "ptot_negative_days": y["ptot_negative"],
            "level_numeric_days": y["level_numeric"],
            "level_missing_days": y["level_missing"],
            "level_invalid_days": y["level_invalid"],
            "discharge_numeric_days": y["discharge_numeric"],
            "discharge_missing_days": y["discharge_missing"],
            "discharge_invalid_days": y["discharge_invalid"],
            "receptor_ids": " | ".join(receptors),
        })

    return file_row, year_rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def median(values):
    return statistics.median(values) if values else 0.0


def build_basin_summary(year_rows):
    # Una stazione può essere associata a più recettori (buffer prudenziale).
    agg = defaultdict(lambda: {
        "meteo_any": set(),
        "meteo_ptot": set(),
        "hydro_any": set(),
        "hydro_level": set(),
        "hydro_discharge": set(),
        "ptot_valid_station_days": 0,
        "level_valid_station_days": 0,
        "discharge_valid_station_days": 0,
    })

    receptors_seen = set()

    for r in year_rows:
        year = int(r["year"])
        sid = r["station_id"]
        receptors = [
            x.strip() for x in str(r.get("receptor_ids", "")).split("|")
            if x.strip()
        ]

        for rid in receptors:
            receptors_seen.add(rid)
            a = agg[(rid, year)]

            if r["kind"] == "daily_meteo":
                if int(r["rows"]) > 0:
                    a["meteo_any"].add(sid)
                if int(r["ptot_numeric_days"]) > 0:
                    a["meteo_ptot"].add(sid)
                a["ptot_valid_station_days"] += int(r["ptot_numeric_days"])

            else:
                if int(r["rows"]) > 0:
                    a["hydro_any"].add(sid)
                if int(r["level_numeric_days"]) > 0:
                    a["hydro_level"].add(sid)
                if int(r["discharge_numeric_days"]) > 0:
                    a["hydro_discharge"].add(sid)
                a["level_valid_station_days"] += int(r["level_numeric_days"])
                a["discharge_valid_station_days"] += int(r["discharge_numeric_days"])

    basin_year_rows = []
    for rid in sorted(receptors_seen):
        for year in range(START_YEAR, END_YEAR + 1):
            a = agg[(rid, year)]
            basin_year_rows.append({
                "receptor_id": rid,
                "year": year,
                "meteo_stations_with_any_data": len(a["meteo_any"]),
                "meteo_stations_with_valid_ptot": len(a["meteo_ptot"]),
                "hydro_stations_with_any_data": len(a["hydro_any"]),
                "hydro_stations_with_valid_level": len(a["hydro_level"]),
                "hydro_stations_with_valid_discharge": len(a["hydro_discharge"]),
                "ptot_valid_station_days": a["ptot_valid_station_days"],
                "level_valid_station_days": a["level_valid_station_days"],
                "discharge_valid_station_days": a["discharge_valid_station_days"],
            })

    summary = []
    for rid in sorted(receptors_seen):
        rr = [x for x in basin_year_rows if x["receptor_id"] == rid]
        pt = [x["meteo_stations_with_valid_ptot"] for x in rr]
        lv = [x["hydro_stations_with_valid_level"] for x in rr]
        qd = [x["hydro_stations_with_valid_discharge"] for x in rr]

        summary.append({
            "receptor_id": rid,
            "years_total": len(rr),
            "years_with_ptot": sum(v > 0 for v in pt),
            "years_with_level": sum(v > 0 for v in lv),
            "years_with_discharge": sum(v > 0 for v in qd),
            "median_ptot_stations_per_year": median(pt),
            "max_ptot_stations_per_year": max(pt) if pt else 0,
            "median_level_stations_per_year": median(lv),
            "max_level_stations_per_year": max(lv) if lv else 0,
            "median_discharge_stations_per_year": median(qd),
            "max_discharge_stations_per_year": max(qd) if qd else 0,
        })

    return basin_year_rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-root",
        default=str(PIE),
        help="Cartella observations_nw/piemonte",
    )
    args = ap.parse_args()

    pie_dir = Path(args.input_root).expanduser().resolve()
    meteo_dir = pie_dir / "daily_meteo"
    hydro_dir = pie_dir / "daily_hydro"
    out_dir = pie_dir / "qc_final_v1_0"

    if not pie_dir.exists():
        raise SystemExit(f"Cartella Piemonte non trovata:\n{pie_dir}")

    if not meteo_dir.exists() or not hydro_dir.exists():
        raise SystemExit(
            "Mancano daily_meteo e/o daily_hydro in:\n"
            f"{pie_dir}"
        )

    meteo_files = sorted(meteo_dir.glob(
        f"*_autumn_{START_YEAR}_{END_YEAR}.csv"
    ))
    hydro_files = sorted(hydro_dir.glob(
        f"*_autumn_{START_YEAR}_{END_YEAR}.csv"
    ))

    station_map, station_names = load_station_map()

    file_rows = []
    year_rows = []

    total = len(meteo_files) + len(hydro_files)
    done = 0

    print("=" * 108)
    print("ARPA PIEMONTE — AUDIT SCIENTIFICO FINALE v1.0")
    print("=" * 108)
    print(f"METEO CSV trovati : {len(meteo_files)}")
    print(f"HYDRO CSV trovati : {len(hydro_files)}")
    print(f"Totale            : {total}")
    print()

    for kind, files in (
        ("daily_meteo", meteo_files),
        ("daily_hydro", hydro_files),
    ):
        for path in files:
            done += 1
            print(f"[{done:03d}/{total:03d}] {kind} | {path.name}")
            fr, yr = audit_file(
                path,
                kind,
                station_map,
                station_names,
            )
            file_rows.append(fr)
            year_rows.extend(yr)

    basin_year_rows, basin_summary = build_basin_summary(year_rows)

    # Sintesi per tipologia
    met = [r for r in file_rows if r["kind"] == "daily_meteo"]
    hyd = [r for r in file_rows if r["kind"] == "daily_hydro"]

    statuses = Counter(r["scientific_status"] for r in file_rows)
    met_statuses = Counter(r["scientific_status"] for r in met)
    hyd_statuses = Counter(r["scientific_status"] for r in hyd)

    ptot_data = sum(int(r["ptot_numeric_days"]) > 0 for r in met)
    level_data = sum(int(r["level_numeric_days"]) > 0 for r in hyd)
    discharge_data = sum(int(r["discharge_numeric_days"]) > 0 for r in hyd)

    raw_ok = sum(str(r["raw_json_status"]).startswith("OK") for r in file_rows)
    meta_ok = sum(str(r["metadata_json_status"]).startswith("OK") for r in file_rows)

    duplicates = sum(int(r["duplicate_dates"]) for r in file_rows)
    out_order = sum(int(r["out_of_order_rows"]) for r in file_rows)
    invalid_dates = sum(int(r["invalid_dates"]) for r in file_rows)
    negative_ptot = sum(int(r["ptot_negative_days"]) for r in file_rows)

    # File review separato
    review_rows = [
        r for r in file_rows if r["scientific_status"] == "REVIEW"
    ]
    nodata_rows = [
        r for r in file_rows if r["scientific_status"] == "NO_DATA_SOURCE"
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "file_qc_final_v1_0.csv", file_rows)
    write_csv(out_dir / "station_year_qc_final_v1_0.csv", year_rows)
    write_csv(out_dir / "basin_year_qc_final_v1_0.csv", basin_year_rows)
    write_csv(out_dir / "basin_summary_final_v1_0.csv", basin_summary)
    write_csv(out_dir / "review_cases_final_v1_0.csv", review_rows)
    write_csv(out_dir / "no_data_source_final_v1_0.csv", nodata_rows)

    # Report finale
    report = [
        "=" * 108,
        "ARPA PIEMONTE — AUDIT SCIENTIFICO FINALE v1.0",
        "=" * 108,
        f"Periodo controllato                    : settembre-dicembre {START_YEAR}-{END_YEAR}",
        f"Serie CSV complessive                  : {len(file_rows)}",
        f"  METEO                                : {len(met)}",
        f"  HYDRO                                : {len(hyd)}",
        "",
        f"DATA_OK                                : {statuses['DATA_OK']}",
        f"NO_DATA_SOURCE                         : {statuses['NO_DATA_SOURCE']}",
        f"REVIEW                                 : {statuses['REVIEW']}",
        "",
        f"METEO con ptot numerica                : {ptot_data}/{len(met)}",
        f"HYDRO con livello numerico             : {level_data}/{len(hyd)}",
        f"HYDRO con portata numerica             : {discharge_data}/{len(hyd)}",
        "",
        f"RAW JSON leggibili                     : {raw_ok}/{len(file_rows)}",
        f"Metadata JSON leggibili                : {meta_ok}/{len(file_rows)}",
        f"Date invalide                          : {invalid_dates}",
        f"Date duplicate                         : {duplicates}",
        f"Righe fuori ordine                     : {out_order}",
        f"Precipitazioni negative                : {negative_ptot}",
        "",
        "CLASSIFICAZIONE PER TIPO:",
        f"  METEO DATA_OK={met_statuses['DATA_OK']} "
        f"NO_DATA_SOURCE={met_statuses['NO_DATA_SOURCE']} "
        f"REVIEW={met_statuses['REVIEW']}",
        f"  HYDRO DATA_OK={hyd_statuses['DATA_OK']} "
        f"NO_DATA_SOURCE={hyd_statuses['NO_DATA_SOURCE']} "
        f"REVIEW={hyd_statuses['REVIEW']}",
        "",
        "COPERTURA PER BACINO:",
        "receptor_id                    anni_ptot anni_livello anni_portata",
    ]

    for r in basin_summary:
        report.append(
            f"{r['receptor_id']:<30s} "
            f"{int(r['years_with_ptot']):>9d} "
            f"{int(r['years_with_level']):>11d} "
            f"{int(r['years_with_discharge']):>12d}"
        )

    report += [
        "",
        "INTERPRETAZIONE:",
        "- ptot giornaliera osservata è la ground truth principale per Y_rain a 24/48 h;",
        "- livellomedio/portatamedia sono ground truth idrologiche giornaliere per screening Y_flood;",
        "- un valore 0 di precipitazione è valido e NON è missing;",
        "- una serie strutturalmente valida senza valori core è NO_DATA_SOURCE, non errore di download;",
        "- i massimi rapidi possono essere attenuati dalle medie giornaliere;",
        "- per eventi critici selezionati serviranno dati ARPA Piemonte orari/suborari ufficiali.",
        "",
        "OUTPUT:",
        f"  {out_dir / 'file_qc_final_v1_0.csv'}",
        f"  {out_dir / 'station_year_qc_final_v1_0.csv'}",
        f"  {out_dir / 'basin_year_qc_final_v1_0.csv'}",
        f"  {out_dir / 'basin_summary_final_v1_0.csv'}",
        f"  {out_dir / 'review_cases_final_v1_0.csv'}",
        f"  {out_dir / 'no_data_source_final_v1_0.csv'}",
        "",
        "ESITO AUTOMATICO:",
        "  PASS — nessun caso REVIEW."
        if statuses["REVIEW"] == 0 else
        f"  REVIEW — {statuses['REVIEW']} serie richiedono controllo mirato.",
        "=" * 108,
    ]

    text = "\n".join(report) + "\n"
    (out_dir / "piemonte_final_audit_v1_0.txt").write_text(
        text, encoding="utf-8"
    )

    print()
    print(text)


if __name__ == "__main__":
    main()
