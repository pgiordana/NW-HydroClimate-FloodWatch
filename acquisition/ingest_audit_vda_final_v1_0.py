#!/usr/bin/env python3
"""
VALLE D'AOSTA — INGESTIONE E AUDIT DEFINITIVI v1.0
===================================================

Scopo
-----
Leggere in SOLA LETTURA i 93 ZIP scaricati manualmente dal portale
regionale, senza estrarli o modificarli, e costruire il dataset storico
normalizzato settembre-dicembre utile al modello regionale.

La struttura osservata dei CSV è:
- encoding contenuto: cp1252;
- righe iniziali di disclaimer/metadati;
- nessun header tabellare esplicito;
- dati a 2 colonne separati da ";":
      YYYY-MM-DD HH:MM:SS ; valore
- frequenza nominale: oraria;
- campo vuoto dopo ";" = MISSING, NON zero.

Periodo prodotto per il modello storico:
- settembre-dicembre 1996-2025, limitatamente alla reale disponibilità
  di ciascuna stazione/parametro.
- NON vengono inventati dati 1987-1995.
- I raw 2026 restano negli ZIP originali ma non entrano nel training storico.

Output
------
observations_nw/valle_d_aosta/final_v1_0/
├── hourly_sepdec_1996_2025/
│   └── <station_code>/
│       └── <station_code>__<parameter_slug>.csv.gz
├── catalog/
│   └── vda_station_catalog.csv
├── qc/
│   ├── vda_file_qc.csv
│   ├── vda_year_coverage.csv
│   ├── vda_parameter_summary.csv
│   └── vda_final_report.txt
└── vda_ingest_manifest.jsonl

Note scientifiche
-----------------
- I timestamp vengono conservati COME PUBBLICATI dalla fonte
  ("timestamp_source"); questa versione NON assume né converte il fuso.
  La normalizzazione UTC va fatta solo dopo verifica documentale del timezone.
- I valori mancanti restano mancanti.
- Il parser accetta sia virgola sia punto decimale.
- Eventuali sentinel numerici sospetti (-999, -9999, 9999, ecc.) vengono
  SEGNALATI ma non modificati automaticamente.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "VDA dati" / "Singole stazioni dal 1996 al 2026"
OUT = ROOT / "observations_nw" / "valle_d_aosta" / "final_v1_0"

START_YEAR = 1996
END_YEAR = 2025
MONTHS = {9, 10, 11, 12}
NOMINAL_HOURS_SEPDEC = 122 * 24  # 2928; solo riferimento, timezone non ancora normalizzato.

NAME_RE = re.compile(r"^Dati_(?P<station>\d+)-(?P<parameter>.+?)\.csv$", re.I)
TS_RE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?)\s*$")

SUSPECT_SENTINELS = {-999999.0, -99999.0, -9999.0, -999.0, 999.0, 9999.0, 99999.0, 999999.0}


def repair_zip_name(s: str) -> str:
    """Ripara eventuale nome interno ZIP UTF-8 interpretato come cp437."""
    try:
        candidate = s.encode("cp437").decode("utf-8")
        bad = "├┤┬┼"
        if sum(ch in candidate for ch in bad) < sum(ch in s for ch in bad):
            return candidate
    except Exception:
        pass
    return s


def norm_key(s: str) -> str:
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def slugify(s: str) -> str:
    x = norm_key(s)
    return x or "parametro"


def parse_timestamp(s: str):
    s = s.strip().strip('"')
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def parse_number(raw: str):
    """
    Ritorna float oppure None.
    Non trasforma i blank in 0.
    """
    s = raw.strip().strip('"').replace("\u00a0", " ").replace(" ", "")
    if s == "":
        return None

    low = s.lower()
    if low in {"na", "nan", "null", "none", "n.d.", "nd", "-", "--"}:
        return None

    # gestione robusta virgola/punto
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")

    try:
        v = float(s)
    except ValueError:
        return "INVALID"

    if not math.isfinite(v):
        return None
    return v


def decode_line(raw: bytes) -> str:
    # struttura verificata: cp1252
    return raw.decode("cp1252", errors="replace").rstrip("\r\n")


def parse_metadata(lines_before_data):
    """
    Legge genericamente righe 'Chiave: valore'.
    Conserva anche eventuali righe non key:value.
    """
    meta = {}
    free = []
    for line in lines_before_data:
        s = line.strip()
        if not s:
            continue
        if ":" in s:
            k, v = s.split(":", 1)
            k = k.strip()
            v = v.strip()
            if k:
                meta[k] = v
        else:
            free.append(s)
    if free:
        meta["_free_lines"] = free
    return meta


def meta_get(meta: dict, *keys):
    nk = {norm_key(k): v for k, v in meta.items() if not k.startswith("_")}
    for key in keys:
        v = nk.get(norm_key(key))
        if v not in (None, ""):
            return v
    return ""


def pct(n, d):
    if not d:
        return ""
    return f"{100.0*n/d:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    args = ap.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    if not input_dir.exists():
        raise SystemExit(f"Cartella non trovata:\n{input_dir}")

    zips = sorted(input_dir.glob("*.zip"))
    if not zips:
        raise SystemExit(f"Nessun ZIP trovato in:\n{input_dir}")

    hourly_dir = OUT / "hourly_sepdec_1996_2025"
    catalog_dir = OUT / "catalog"
    qc_dir = OUT / "qc"
    for d in (hourly_dir, catalog_dir, qc_dir):
        d.mkdir(parents=True, exist_ok=True)

    file_qc_rows = []
    coverage_rows = []
    manifest_rows = []

    # Catalogo stazioni costruito aggregando i metadata di tutti i parametri
    station_meta_values = defaultdict(lambda: defaultdict(set))
    station_params = defaultdict(set)
    station_zip_names = defaultdict(set)

    parameter_files = Counter()
    parameter_stations = defaultdict(set)
    parameter_units = defaultdict(set)
    parameter_selected_rows = Counter()
    parameter_valid_values = Counter()
    parameter_missing_values = Counter()

    total_selected_rows = 0
    total_valid_values = 0
    total_missing_values = 0
    total_invalid_values = 0
    total_duplicate_ts = 0
    total_out_of_order = 0
    total_sentinel_suspects = 0

    processed_files = 0
    skipped_nonmatching = 0

    manifest_path = OUT / "vda_ingest_manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    for zip_index, zp in enumerate(zips, 1):
        print(f"[ZIP {zip_index:02d}/{len(zips)}] {zp.name}")

        with zipfile.ZipFile(zp, "r") as zf:
            bad = zf.testzip()
            if bad:
                raise RuntimeError(f"CRC non valido in {zp.name}: {bad}")

            for info in zf.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".csv"):
                    continue

                entry = repair_zip_name(Path(info.filename).name)
                m = NAME_RE.match(entry)
                if not m:
                    skipped_nonmatching += 1
                    continue

                processed_files += 1
                station = m.group("station")
                parameter = re.sub(r"\s+", " ", m.group("parameter").strip())
                pslug = slugify(parameter)

                station_params[station].add(parameter)
                station_zip_names[station].add(zp.name)
                parameter_files[parameter] += 1
                parameter_stations[parameter].add(station)

                # ---------- prima passata logica sul singolo file ----------
                # Leggiamo streaming; conserviamo solo metadata pre-dati e statistiche.
                metadata_lines = []
                metadata = {}
                source_first_ts = None
                source_last_ts = None
                first_numeric_ts = None
                last_numeric_ts = None
                rows_total = 0
                rows_with_timestamp = 0
                invalid_timestamp_rows = 0
                blank_value_total = 0
                numeric_value_total = 0
                invalid_value_total = 0
                sentinel_suspect_total = 0
                selected_rows = 0
                selected_blank = 0
                selected_numeric = 0
                selected_invalid = 0
                selected_duplicates = 0
                selected_out_of_order = 0
                selected_sentinel = 0
                seen_selected = set()
                prev_selected_ts = None

                by_year = defaultdict(lambda: {
                    "rows": 0,
                    "unique_ts": set(),
                    "numeric": 0,
                    "blank": 0,
                    "invalid": 0,
                    "duplicates": 0,
                    "sentinel": 0,
                    "first_ts": None,
                    "last_ts": None,
                })

                out_station = hourly_dir / station
                out_station.mkdir(parents=True, exist_ok=True)
                out_path = out_station / f"{station}__{pslug}.csv.gz"

                with zf.open(info, "r") as src, gzip.open(
                    out_path, "wt", encoding="utf-8", newline=""
                ) as dst:
                    writer = csv.writer(dst)
                    writer.writerow([
                        "station_code",
                        "parameter",
                        "unit",
                        "timestamp_source",
                        "value_numeric",
                        "value_raw",
                        "source_zip",
                        "source_entry",
                    ])

                    data_started = False

                    for raw_line in src:
                        line = decode_line(raw_line)
                        if not line.strip():
                            if not data_started:
                                metadata_lines.append(line)
                            continue

                        row = next(csv.reader([line], delimiter=";"))

                        ts = parse_timestamp(row[0]) if row else None

                        if ts is None:
                            if not data_started:
                                metadata_lines.append(line)
                            else:
                                # dopo l'inizio dati una riga senza timestamp è anomala
                                invalid_timestamp_rows += 1
                            continue

                        if not data_started:
                            data_started = True
                            metadata = parse_metadata(metadata_lines)

                        rows_total += 1
                        rows_with_timestamp += 1

                        if source_first_ts is None:
                            source_first_ts = ts
                        source_last_ts = ts

                        raw_value = row[1] if len(row) > 1 else ""
                        num = parse_number(raw_value)

                        if num is None:
                            blank_value_total += 1
                        elif num == "INVALID":
                            invalid_value_total += 1
                        else:
                            numeric_value_total += 1
                            if num in SUSPECT_SENTINELS:
                                sentinel_suspect_total += 1
                            if first_numeric_ts is None:
                                first_numeric_ts = ts
                            last_numeric_ts = ts

                        # storico modello: Sep-Dec 1996-2025
                        if START_YEAR <= ts.year <= END_YEAR and ts.month in MONTHS:
                            selected_rows += 1
                            total_selected_rows += 1
                            parameter_selected_rows[parameter] += 1

                            y = by_year[ts.year]
                            y["rows"] += 1

                            ts_key = ts.strftime("%Y-%m-%d %H:%M:%S")
                            if ts_key in seen_selected:
                                selected_duplicates += 1
                                total_duplicate_ts += 1
                                y["duplicates"] += 1
                            else:
                                seen_selected.add(ts_key)
                                y["unique_ts"].add(ts_key)

                            if prev_selected_ts is not None and ts < prev_selected_ts:
                                selected_out_of_order += 1
                                total_out_of_order += 1
                            prev_selected_ts = ts

                            if y["first_ts"] is None:
                                y["first_ts"] = ts
                            y["last_ts"] = ts

                            if num is None:
                                selected_blank += 1
                                total_missing_values += 1
                                parameter_missing_values[parameter] += 1
                                y["blank"] += 1
                                val_num = ""
                            elif num == "INVALID":
                                selected_invalid += 1
                                total_invalid_values += 1
                                y["invalid"] += 1
                                val_num = ""
                            else:
                                selected_numeric += 1
                                total_valid_values += 1
                                parameter_valid_values[parameter] += 1
                                y["numeric"] += 1
                                val_num = format(num, ".12g")
                                if num in SUSPECT_SENTINELS:
                                    selected_sentinel += 1
                                    total_sentinel_suspects += 1
                                    y["sentinel"] += 1

                            # unità non è nota fino al parsing metadata; la aggiorniamo
                            # dopo? Per non fare seconda passata, usiamo metadata già letto:
                            unit_now = meta_get(metadata, "Unità misura", "Unita misura", "Unità", "Unita")

                            writer.writerow([
                                station,
                                parameter,
                                unit_now,
                                ts_key,
                                val_num,
                                raw_value.strip(),
                                zp.name,
                                entry,
                            ])

                unit = meta_get(metadata, "Unità misura", "Unita misura", "Unità", "Unita")
                altitude = meta_get(metadata, "Quota mslm", "Quota", "Altitudine", "Altitude")
                station_name = meta_get(
                    metadata,
                    "Stazione", "Nome stazione", "Nome", "Località", "Localita"
                )
                municipality = meta_get(metadata, "Comune", "Municipio")
                latitude = meta_get(metadata, "Latitudine", "Latitude", "Lat")
                longitude = meta_get(metadata, "Longitudine", "Longitude", "Lon", "Long")

                parameter_units[parameter].add(unit)

                # catalogo: conserva tutti i metadati, senza perdere campi non previsti
                for k, v in metadata.items():
                    if k.startswith("_"):
                        continue
                    station_meta_values[station][k].add(v)

                # QC per anno: includiamo anche anni con zero righe per rendere espliciti i buchi.
                for year in range(START_YEAR, END_YEAR + 1):
                    y = by_year[year]
                    uniq = len(y["unique_ts"])
                    coverage_rows.append({
                        "station_code": station,
                        "parameter": parameter,
                        "unit": unit,
                        "year": year,
                        "nominal_expected_hours_sepdec": NOMINAL_HOURS_SEPDEC,
                        "rows_sepdec": y["rows"],
                        "unique_timestamps_sepdec": uniq,
                        "row_coverage_pct_nominal": pct(uniq, NOMINAL_HOURS_SEPDEC),
                        "numeric_values": y["numeric"],
                        "missing_blank_values": y["blank"],
                        "invalid_values": y["invalid"],
                        "duplicate_timestamps": y["duplicates"],
                        "sentinel_suspects": y["sentinel"],
                        "numeric_value_pct_of_unique_rows": pct(y["numeric"], uniq),
                        "first_timestamp": (
                            y["first_ts"].strftime("%Y-%m-%d %H:%M:%S")
                            if y["first_ts"] else ""
                        ),
                        "last_timestamp": (
                            y["last_ts"].strftime("%Y-%m-%d %H:%M:%S")
                            if y["last_ts"] else ""
                        ),
                    })

                file_qc_rows.append({
                    "station_code": station,
                    "parameter": parameter,
                    "unit": unit,
                    "altitude_mslm": altitude,
                    "station_name_metadata": station_name,
                    "municipality_metadata": municipality,
                    "latitude_metadata": latitude,
                    "longitude_metadata": longitude,
                    "source_zip": zp.name,
                    "source_entry": entry,
                    "source_size_bytes": info.file_size,
                    "metadata_data_inizio": meta_get(metadata, "Data inizio"),
                    "metadata_data_fine": meta_get(metadata, "Data fine"),
                    "source_first_timestamp": (
                        source_first_ts.strftime("%Y-%m-%d %H:%M:%S")
                        if source_first_ts else ""
                    ),
                    "source_last_timestamp": (
                        source_last_ts.strftime("%Y-%m-%d %H:%M:%S")
                        if source_last_ts else ""
                    ),
                    "first_numeric_timestamp": (
                        first_numeric_ts.strftime("%Y-%m-%d %H:%M:%S")
                        if first_numeric_ts else ""
                    ),
                    "last_numeric_timestamp": (
                        last_numeric_ts.strftime("%Y-%m-%d %H:%M:%S")
                        if last_numeric_ts else ""
                    ),
                    "rows_with_timestamp_all_period": rows_with_timestamp,
                    "numeric_values_all_period": numeric_value_total,
                    "blank_values_all_period": blank_value_total,
                    "invalid_values_all_period": invalid_value_total,
                    "sentinel_suspects_all_period": sentinel_suspect_total,
                    "rows_sepdec_1996_2025": selected_rows,
                    "numeric_sepdec_1996_2025": selected_numeric,
                    "blank_sepdec_1996_2025": selected_blank,
                    "invalid_sepdec_1996_2025": selected_invalid,
                    "duplicate_ts_sepdec_1996_2025": selected_duplicates,
                    "out_of_order_sepdec_1996_2025": selected_out_of_order,
                    "sentinel_suspects_sepdec_1996_2025": selected_sentinel,
                    "normalized_output": str(out_path.relative_to(ROOT)),
                    "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                })

                manifest = {
                    "station_code": station,
                    "parameter": parameter,
                    "unit": unit,
                    "source_zip": str(zp.relative_to(ROOT)),
                    "source_entry": info.filename,
                    "normalized_output": str(out_path.relative_to(ROOT)),
                    "status": "ok",
                    "selected_period": f"Sep-Dec {START_YEAR}-{END_YEAR}",
                    "rows_selected": selected_rows,
                    "numeric_selected": selected_numeric,
                    "missing_selected": selected_blank,
                    "invalid_selected": selected_invalid,
                    "duplicates_selected": selected_duplicates,
                    "timestamp_policy": "source timestamp preserved; timezone not converted",
                }
                with manifest_path.open("a", encoding="utf-8") as mf:
                    mf.write(json.dumps(manifest, ensure_ascii=False) + "\n")

    # ---------- catalogo stazioni ----------
    catalog_rows = []
    preferred_keys = [
        "Stazione", "Nome stazione", "Nome", "Località", "Localita",
        "Comune", "Quota mslm", "Quota", "Altitudine",
        "Latitudine", "Longitudine"
    ]

    for station in sorted(station_params, key=int):
        metas = station_meta_values[station]

        def first_meta(*names):
            normalized = {norm_key(k): vals for k, vals in metas.items()}
            for name in names:
                vals = normalized.get(norm_key(name))
                if vals:
                    return " | ".join(sorted(vals))
            return ""

        all_meta = {k: sorted(v) for k, v in sorted(metas.items())}

        catalog_rows.append({
            "station_code": station,
            "station_name": first_meta("Stazione", "Nome stazione", "Nome", "Località", "Localita"),
            "municipality": first_meta("Comune"),
            "altitude_mslm": first_meta("Quota mslm", "Quota", "Altitudine"),
            "latitude": first_meta("Latitudine", "Latitude", "Lat"),
            "longitude": first_meta("Longitudine", "Longitude", "Lon", "Long"),
            "parameter_count": len(station_params[station]),
            "parameters": " | ".join(sorted(station_params[station])),
            "zip_count": len(station_zip_names[station]),
            "zip_names": " | ".join(sorted(station_zip_names[station])),
            "metadata_json": json.dumps(all_meta, ensure_ascii=False, sort_keys=True),
        })

    # ---------- summary parametri ----------
    parameter_summary_rows = []
    for p in sorted(parameter_files, key=lambda x: (-len(parameter_stations[x]), x.lower())):
        units = sorted(u for u in parameter_units[p] if u != "")
        parameter_summary_rows.append({
            "parameter": p,
            "stations": len(parameter_stations[p]),
            "files": parameter_files[p],
            "units": " | ".join(units),
            "rows_sepdec_1996_2025": parameter_selected_rows[p],
            "numeric_values_sepdec_1996_2025": parameter_valid_values[p],
            "missing_blank_sepdec_1996_2025": parameter_missing_values[p],
            "numeric_pct_of_rows": pct(
                parameter_valid_values[p],
                parameter_selected_rows[p]
            ),
        })

    # ---------- scrittura CSV ----------
    def write_csv(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv(catalog_dir / "vda_station_catalog.csv", catalog_rows)
    write_csv(qc_dir / "vda_file_qc.csv", file_qc_rows)
    write_csv(qc_dir / "vda_year_coverage.csv", coverage_rows)
    write_csv(qc_dir / "vda_parameter_summary.csv", parameter_summary_rows)

    # ---------- report ----------
    unit_variants = {
        p: sorted(u for u in parameter_units[p] if u != "")
        for p in parameter_units
    }
    params_with_multiple_units = {p: u for p, u in unit_variants.items() if len(u) > 1}

    files_no_numeric_selected = sum(
        1 for r in file_qc_rows if int(r["numeric_sepdec_1996_2025"]) == 0
    )
    files_with_duplicates = sum(
        1 for r in file_qc_rows if int(r["duplicate_ts_sepdec_1996_2025"]) > 0
    )
    files_with_invalid_values = sum(
        1 for r in file_qc_rows if int(r["invalid_sepdec_1996_2025"]) > 0
    )
    files_with_sentinel = sum(
        1 for r in file_qc_rows if int(r["sentinel_suspects_sepdec_1996_2025"]) > 0
    )

    report = [
        "=" * 110,
        "VALLE D'AOSTA — INGESTIONE E AUDIT DEFINITIVI v1.0",
        "=" * 110,
        f"Cartella RAW ZIP                         : {input_dir}",
        f"ZIP validi analizzati                   : {len(zips)}",
        f"CSV stazione-parametro processati       : {processed_files}",
        f"File non conformi al naming atteso      : {skipped_nonmatching}",
        f"Stazioni distinte                       : {len(station_params)}",
        f"Parametri distinti                      : {len(parameter_files)}",
        "",
        f"Periodo normalizzato                    : settembre-dicembre {START_YEAR}-{END_YEAR}",
        "Timestamp                              : conservato come pubblicato; timezone NON convertito",
        f"Righe selezionate                       : {total_selected_rows}",
        f"Valori numerici validamente parsati     : {total_valid_values}",
        f"Valori blank/missing                    : {total_missing_values}",
        f"Valori testuali/non numerici invalidi   : {total_invalid_values}",
        f"Timestamp duplicati (raw source time)   : {total_duplicate_ts}",
        f"Righe fuori ordine temporale            : {total_out_of_order}",
        f"Sentinel numerici sospetti              : {total_sentinel_suspects}",
        "",
        f"File senza valori numerici Sep-Dec      : {files_no_numeric_selected}/{processed_files}",
        f"File con timestamp duplicati            : {files_with_duplicates}",
        f"File con valori invalidi                : {files_with_invalid_values}",
        f"File con sentinel sospetti              : {files_with_sentinel}",
        f"Parametri con >1 unità dichiarata       : {len(params_with_multiple_units)}",
        "",
        "PARAMETRI:",
    ]

    for r in parameter_summary_rows:
        report.append(
            f"  {r['parameter']:<42.42s} "
            f"stazioni={int(r['stations']):3d}  "
            f"unità={r['units'] or '[non dichiarata]':<16.16s}  "
            f"numeric%={r['numeric_pct_of_rows'] or 'n.d.':>7s}"
        )

    if params_with_multiple_units:
        report += ["", "ATTENZIONE — parametri con unità multiple:"]
        for p, units in sorted(params_with_multiple_units.items()):
            report.append(f"  {p}: {' | '.join(units)}")

    report += [
        "",
        "CRITERI DI INTERPRETAZIONE:",
        "- un campo vuoto dopo ';' è MISSING e NON viene trasformato in zero;",
        "- il 1996-2025 valdostano viene usato solo dove realmente osservato;",
        "- il periodo 1987-1995 resta esplicitamente privo di osservazioni VdA;",
        "- la copertura annuale è confrontata con 2928 ore nominali Sep-Dec solo come riferimento;",
        "- prima della fusione con ERA5 va verificato documentalmente il timezone dei timestamp;",
        "- eventuali duplicati attorno al cambio ora NON vanno eliminati automaticamente prima di tale verifica.",
        "",
        "OUTPUT PRINCIPALI:",
        f"  {hourly_dir}",
        f"  {catalog_dir / 'vda_station_catalog.csv'}",
        f"  {qc_dir / 'vda_file_qc.csv'}",
        f"  {qc_dir / 'vda_year_coverage.csv'}",
        f"  {qc_dir / 'vda_parameter_summary.csv'}",
        f"  {manifest_path}",
        "",
        "PROSSIMA VERIFICA:",
        "- leggere questo report;",
        "- controllare eventuali unità multiple, invalidi, sentinel e duplicati;",
        "- poi associare le stazioni alle geometrie/bacini e definire la normalizzazione temporale UTC.",
        "=" * 110,
    ]

    report_text = "\n".join(report) + "\n"
    (qc_dir / "vda_final_report.txt").write_text(report_text, encoding="utf-8")
    print("\n" + report_text)


if __name__ == "__main__":
    main()
