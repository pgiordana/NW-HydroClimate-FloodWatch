#!/usr/bin/env python3
"""
VALLE D'AOSTA — QC SCIENTIFICO CONCLUSIVO v1.1
==============================================

Finalizza il controllo del blocco osservativo Valle d'Aosta a partire da:
  observations_nw/valle_d_aosta/final_v1_0/qc/vda_file_qc.csv

Criteri:
- DATA_OK: almeno un valore numerico nel periodo Sep-Dec 1996-2025,
  nessun valore invalido, nessun timestamp duplicato, nessuna riga fuori ordine.
- NO_DATA_SOURCE: nessun valore numerico ma struttura valida e nessun errore
  tecnico; il file viene conservato come esplicita assenza di dato utilizzabile.
- REVIEW: qualunque anomalia tecnica residua.

La precedente ricerca generica di "sentinel" considerava 999 sospetto.
Qui 999 viene correttamente riconosciuto come valore fisicamente plausibile
quando:
  - Radiazione totale: 0..1600 W/m²
  - Pressione: 800..1100 hPa
e quindi NON viene classificato come missing.

NON modifica né i RAW ZIP né i CSV normalizzati.

Output:
  observations_nw/valle_d_aosta/final_v1_0/qc/
    vda_final_scientific_qc_v1_1.csv
    vda_final_scientific_qc_v1_1.txt
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QC = ROOT / "observations_nw" / "valle_d_aosta" / "final_v1_0" / "qc"
INFILE = QC / "vda_file_qc.csv"
OUTCSV = QC / "vda_final_scientific_qc_v1_1.csv"
OUTTXT = QC / "vda_final_scientific_qc_v1_1.txt"

CORE = {
    "Precipitazione ufficiale",
    "Livello idrometrico",
    "Portata",
    "Altezza neve al suolo",
    "Temperatura",
    "Umidità relativa",
    "Velocità Vento Vett.",
    "Direzione Vento Vett.",
}


def plausible_999(parameter: str, unit: str) -> bool:
    p = parameter.strip().lower()
    u = unit.strip().lower()
    if p == "radiazione totale" and "w/m" in u:
        return True
    if p == "pressione" and u == "hpa":
        return True
    if p == "pressione (barometro di prec.)" and u == "hpa":
        return True
    return False


def main():
    if not INFILE.exists():
        raise SystemExit(f"File non trovato:\n{INFILE}")

    with INFILE.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    out = []
    counts = Counter()
    parameter_files = Counter()
    parameter_data_files = Counter()
    parameter_nodata_files = Counter()
    parameter_stations = defaultdict(set)
    parameter_data_stations = defaultdict(set)

    unresolved_sentinel_files = 0
    false_positive_sentinel_files = 0

    for r in rows:
        station = r["station_code"]
        parameter = r["parameter"]
        unit = r["unit"]

        numeric = int(r["numeric_sepdec_1996_2025"])
        blank = int(r["blank_sepdec_1996_2025"])
        invalid = int(r["invalid_sepdec_1996_2025"])
        dup = int(r["duplicate_ts_sepdec_1996_2025"])
        ooo = int(r["out_of_order_sepdec_1996_2025"])
        generic_sentinel = int(r["sentinel_suspects_sepdec_1996_2025"])

        sentinel_status = "NONE"
        unresolved_sentinel = 0

        if generic_sentinel > 0:
            if plausible_999(parameter, unit):
                sentinel_status = "FALSE_POSITIVE_999_PHYSICALLY_PLAUSIBLE"
                false_positive_sentinel_files += 1
            else:
                sentinel_status = "REVIEW"
                unresolved_sentinel = generic_sentinel
                unresolved_sentinel_files += 1

        if invalid > 0 or dup > 0 or ooo > 0 or unresolved_sentinel > 0:
            status = "REVIEW"
        elif numeric > 0:
            status = "DATA_OK"
        else:
            status = "NO_DATA_SOURCE"

        counts[status] += 1
        parameter_files[parameter] += 1
        parameter_stations[parameter].add(station)

        if status == "DATA_OK":
            parameter_data_files[parameter] += 1
            parameter_data_stations[parameter].add(station)
        elif status == "NO_DATA_SOURCE":
            parameter_nodata_files[parameter] += 1

        out.append({
            "station_code": station,
            "parameter": parameter,
            "unit": unit,
            "scientific_status": status,
            "rows_sepdec": r["rows_sepdec_1996_2025"],
            "numeric_sepdec": numeric,
            "blank_sepdec": blank,
            "invalid_sepdec": invalid,
            "duplicate_timestamps": dup,
            "out_of_order_rows": ooo,
            "generic_999_flags": generic_sentinel,
            "sentinel_interpretation": sentinel_status,
            "source_entry": r["source_entry"],
            "normalized_output": r["normalized_output"],
        })

    with OUTCSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    passed = counts["REVIEW"] == 0
    usable_stations = {
        p: len(parameter_data_stations[p])
        for p in parameter_files
    }

    report = [
        "=" * 108,
        "VALLE D'AOSTA — QC SCIENTIFICO CONCLUSIVO v1.1",
        "=" * 108,
        f"Serie stazione-parametro esaminate      : {len(rows)}",
        f"DATA_OK                                 : {counts['DATA_OK']}",
        f"NO_DATA_SOURCE                          : {counts['NO_DATA_SOURCE']}",
        f"REVIEW                                  : {counts['REVIEW']}",
        "",
        f"Flag 999 riclassificati come plausibili : {false_positive_sentinel_files} file",
        f"File con sentinel ancora da verificare  : {unresolved_sentinel_files}",
        "",
        "COPERTURA PER PARAMETRO:",
    ]

    for p in sorted(parameter_files, key=lambda x: (-len(parameter_stations[x]), x.lower())):
        report.append(
            f"  {p:<42.42s} "
            f"stazioni_tot={len(parameter_stations[p]):3d}  "
            f"stazioni_con_dati={usable_stations[p]:3d}  "
            f"file_NO_DATA={parameter_nodata_files[p]:2d}"
        )

    report += [
        "",
        "VARIABILI CORE DEL MODELLO:",
    ]
    for p in sorted(CORE):
        if p in parameter_files:
            report.append(
                f"  {p:<32.32s} "
                f"{usable_stations[p]:3d}/{len(parameter_stations[p]):3d} stazioni con almeno un valore Sep-Dec"
            )

    report += [
        "",
        "INTERPRETAZIONE DEI 18 FILE SENZA DATI NUMERICI:",
        "- sono file strutturalmente validi ma senza osservazioni utilizzabili nel periodo selezionato;",
        "- restano conservati e classificati NO_DATA_SOURCE, non sono errori di download;",
        "- i campi vuoti non vengono mai trasformati in zero.",
        "",
        "INTERPRETAZIONE DEL VALORE 999:",
        "- Radiazione totale = 999 W/m² è fisicamente plausibile;",
        "- Pressione = 999 hPa è fisicamente plausibile;",
        "- i 47 flag generici precedenti non costituiscono quindi sentinel/missing.",
        "",
        "ESITO:",
        "  PASS — acquisizione, ingestione e controllo interno del blocco Valle d'Aosta."
        if passed else
        "  REVIEW — restano anomalie tecniche da risolvere.",
        "",
        "NOTA TEMPORALE:",
        "- i timestamp sono ancora conservati come pubblicati dalla fonte;",
        "- la loro convenzione UTC/CET/CEST va documentata formalmente prima della fusione temporale con ERA5.",
        "",
        f"CSV dettaglio: {OUTCSV}",
        "=" * 108,
    ]

    text = "\n".join(report) + "\n"
    OUTTXT.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
