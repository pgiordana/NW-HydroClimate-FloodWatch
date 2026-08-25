#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_tanaro_arroscia_glofas_v1_2.py

Audit finale/canonico del blocco GloFAS Historical Tanaro–Arroscia.

Stato atteso del dataset:
- 156 file mensili, settembre-dicembre 1987-2025
- GloFAS Historical v4 / LISFLOOD / consolidated
- variabile: average river discharge in the last 24 hours
- GRIB: glofas_discharge_YYYY_MM.grib
- 154 mesi devono superare un audit STRICT PASS
- due eccezioni di sorgente, dimostrate con riscarico e confronto byte-per-byte:
    * 2024-12: assente 2024-12-31; il riscarico è byte-identico al primo download
    * 2025-12: assente 2025-12-31; il server ha restituito due copie byte-identiche
      dello stesso blocco 1-30 dicembre; il file corrente è old+old

Esito canonico previsto:
    PASS_WITH_SOURCE_EXCEPTIONS
    DATASET STATUS: CLOSED

Principi:
- NON modifica mai i GRIB raw.
- NON deduplica il raw 2025-12.
- NON interpola i due giorni mancanti.
- Le eccezioni sono accettate SOLO se corrispondono esattamente alla firma già
  dimostrata e se le copie di quarantena consentono la verifica byte-per-byte.
- Qualunque altra anomalia resta FAIL.

Ambiente:
    source ../.venv_glofas_audit/bin/activate

Uso:
    python audit_tanaro_arroscia_glofas_v1_2.py
"""

from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SCRIPT_VERSION = "1.2"

START_YEAR = 1987
END_YEAR = 2025
MONTHS = (9, 10, 11, 12)
EXPECTED_TOTAL_FILES = (END_YEAR - START_YEAR + 1) * len(MONTHS)

EXPECTED_GRID = {
    "gridType": "regular_ll",
    "Ni": 35,
    "Nj": 21,
    "Nx": 35,
    "Ny": 21,
    "numberOfPoints": 735,
    "latitudeOfFirstGridPointInDegrees": 44.475,
    "longitudeOfFirstGridPointInDegrees": 7.025,
    "latitudeOfLastGridPointInDegrees": 43.475,
    "longitudeOfLastGridPointInDegrees": 8.725,
    "iDirectionIncrementInDegrees": 0.05,
    "jDirectionIncrementInDegrees": 0.05,
    "iScansNegatively": 0,
    "jScansPositively": 0,
}

EXPECTED_VARIABLE = {
    "shortName": "avg_dis",
    "name": "Time-mean discharge from rivers or streams",
    "units": "m**3 s**-1",
    "paramId": 235270,
    "typeOfStatisticalProcessing": 0,
    "stepType": "avg",
    "stepRange": "0-24",
    "startStep": 0,
    "endStep": 24,
}

SPECIAL_2024 = (2024, 12)
SPECIAL_2025 = (2025, 12)


@dataclass
class Issue:
    severity: str   # ERROR | WARN | INFO
    code: str
    file: str
    detail: str


def expected_month_keys() -> List[Tuple[int, int]]:
    return [(y, m) for y in range(START_YEAR, END_YEAR + 1) for m in MONTHS]


def expected_filename(year: int, month: int) -> str:
    return f"glofas_discharge_{year:04d}_{month:02d}.grib"


def expected_dates(year: int, month: int) -> List[date]:
    ndays = calendar.monthrange(year, month)[1]
    return [date(year, month, d) for d in range(1, ndays + 1)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def import_eccodes():
    try:
        import eccodes  # type: ignore
        return eccodes
    except Exception as exc:
        print(f"ERRORE: impossibile importare eccodes: {exc}", file=sys.stderr)
        raise SystemExit(3)


def get_safe(ec, gid, key: str, default=None):
    try:
        if hasattr(ec, "codes_is_defined") and not ec.codes_is_defined(gid, key):
            return default
        return ec.codes_get(gid, key)
    except Exception:
        return default


def parse_grib_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    try:
        s = f"{int(value):08d}"
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except Exception:
        return None


def normalized_grid(ec, gid) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in EXPECTED_GRID:
        value = get_safe(ec, gid, key, None)
        if isinstance(value, float) and math.isfinite(value):
            value = round(value, 10)
        out[key] = value
    return out


def values_equal(a: Any, b: Any, tol: float = 1e-9) -> bool:
    if isinstance(b, float):
        try:
            return abs(float(a) - b) <= tol
        except Exception:
            return False
    return a == b


def compare_dict(actual: Dict[str, Any], expected: Dict[str, Any]) -> List[str]:
    bad = []
    for k, exp in expected.items():
        act = actual.get(k)
        if not values_equal(act, exp):
            bad.append(f"{k}: got={act!r}, expected={exp!r}")
    return bad


def manifest_mentions(path: Path, filename: str) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return any(filename in line for line in f)
    except Exception:
        return False


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def inspect_grib(ec, path: Path, scan_values: bool = True) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Issue]]:
    """
    Restituisce:
      summary: metadati aggregati del file
      messages: metadati per messaggio + hash raw
      issues: problemi strutturali intrinseci (non ancora calendario mese-specifico)
    """
    issues: List[Issue] = []
    messages: List[Dict[str, Any]] = []

    summary: Dict[str, Any] = {
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() else "",
        "message_count": 0,
        "finite_values": 0,
        "missing_or_nonfinite_values": 0,
        "negative_values": 0,
        "value_min": "",
        "value_max": "",
        "grid": None,
        "variable": None,
    }

    if not path.exists():
        issues.append(Issue("ERROR", "MISSING_FILE", path.name, "File atteso non presente."))
        return summary, messages, issues

    if path.stat().st_size <= 0:
        issues.append(Issue("ERROR", "EMPTY_FILE", path.name, "File di dimensione zero."))
        return summary, messages, issues

    first_grid = None
    first_variable = None

    finite_n = 0
    bad_n = 0
    negative_n = 0
    vmin = None
    vmax = None

    try:
        with path.open("rb") as f:
            idx = 0
            while True:
                gid = ec.codes_grib_new_from_file(f)
                if gid is None:
                    break
                idx += 1
                try:
                    raw = ec.codes_get_message(gid)

                    variable = {
                        "shortName": get_safe(ec, gid, "shortName"),
                        "name": get_safe(ec, gid, "name"),
                        "units": get_safe(ec, gid, "units"),
                        "paramId": get_safe(ec, gid, "paramId"),
                        "typeOfStatisticalProcessing": get_safe(ec, gid, "typeOfStatisticalProcessing"),
                        "stepType": get_safe(ec, gid, "stepType"),
                        "stepRange": get_safe(ec, gid, "stepRange"),
                        "startStep": get_safe(ec, gid, "startStep"),
                        "endStep": get_safe(ec, gid, "endStep"),
                    }

                    grid = normalized_grid(ec, gid)

                    if first_variable is None:
                        first_variable = variable
                    elif variable != first_variable:
                        issues.append(Issue(
                            "ERROR",
                            "VARIABLE_CHANGES_WITHIN_FILE",
                            path.name,
                            f"Firma variabile diversa nel messaggio {idx}."
                        ))

                    if first_grid is None:
                        first_grid = grid
                    elif grid != first_grid:
                        issues.append(Issue(
                            "ERROR",
                            "GRID_CHANGES_WITHIN_FILE",
                            path.name,
                            f"Griglia diversa nel messaggio {idx}."
                        ))

                    dd = parse_grib_date(get_safe(ec, gid, "dataDate"))
                    vd = parse_grib_date(get_safe(ec, gid, "validityDate"))

                    msg = {
                        "message_index": idx,
                        "message_sha256": sha256_bytes(raw),
                        "message_bytes": len(raw),
                        "dataDate": dd,
                        "validityDate": vd,
                        "dataTime": get_safe(ec, gid, "dataTime"),
                        "validityTime": get_safe(ec, gid, "validityTime"),
                    }
                    messages.append(msg)

                    if scan_values:
                        vals = ec.codes_get_values(gid)
                        missing_value = get_safe(ec, gid, "missingValue", None)
                        for raw_v in vals:
                            try:
                                v = float(raw_v)
                            except Exception:
                                bad_n += 1
                                continue

                            if not math.isfinite(v):
                                bad_n += 1
                                continue

                            if missing_value is not None:
                                try:
                                    if v == float(missing_value):
                                        bad_n += 1
                                        continue
                                except Exception:
                                    pass

                            finite_n += 1
                            if v < -1e-6:
                                negative_n += 1
                            vmin = v if vmin is None else min(vmin, v)
                            vmax = v if vmax is None else max(vmax, v)

                finally:
                    ec.codes_release(gid)

    except Exception as exc:
        issues.append(Issue("ERROR", "GRIB_READ_FAILED", path.name, repr(exc)))

    summary["message_count"] = len(messages)
    summary["finite_values"] = finite_n
    summary["missing_or_nonfinite_values"] = bad_n
    summary["negative_values"] = negative_n
    summary["value_min"] = "" if vmin is None else vmin
    summary["value_max"] = "" if vmax is None else vmax
    summary["grid"] = first_grid
    summary["variable"] = first_variable

    if len(messages) == 0:
        issues.append(Issue("ERROR", "NO_GRIB_MESSAGES", path.name, "Nessun messaggio GRIB leggibile."))

    if first_grid is not None:
        grid_bad = compare_dict(first_grid, EXPECTED_GRID)
        if grid_bad:
            issues.append(Issue(
                "ERROR",
                "UNEXPECTED_GRID",
                path.name,
                "; ".join(grid_bad)
            ))

    if first_variable is not None:
        var_bad = compare_dict(first_variable, EXPECTED_VARIABLE)
        if var_bad:
            issues.append(Issue(
                "ERROR",
                "UNEXPECTED_VARIABLE_SIGNATURE",
                path.name,
                "; ".join(var_bad)
            ))

    if scan_values:
        if finite_n == 0:
            issues.append(Issue("ERROR", "NO_FINITE_VALUES", path.name, "Nessun valore numerico finito."))
        if negative_n > 0:
            issues.append(Issue(
                "ERROR",
                "NEGATIVE_DISCHARGE",
                path.name,
                f"Trovati {negative_n} valori di portata < -1e-6."
            ))

    return summary, messages, issues


def calendar_signature(messages: List[Dict[str, Any]], year: int, month: int) -> Dict[str, Any]:
    exp = expected_dates(year, month)
    exp_set = set(exp)

    data_dates = [m["dataDate"] for m in messages if m["dataDate"] is not None]
    validity_dates = [m["validityDate"] for m in messages if m["validityDate"] is not None]

    counts = Counter(data_dates)
    source_set = set(data_dates)

    offsets = []
    for m in messages:
        dd = m["dataDate"]
        vd = m["validityDate"]
        if dd is not None and vd is not None:
            offsets.append((vd - dd).days)

    return {
        "expected_days": len(exp),
        "unique_source_dates": len(source_set),
        "missing_source_dates": sorted(exp_set - source_set),
        "outside_source_dates": sorted(source_set - exp_set),
        "duplicate_source_dates": sorted(d for d, n in counts.items() if n > 1),
        "source_counts": counts,
        "first_source_date": min(data_dates) if data_dates else None,
        "last_source_date": max(data_dates) if data_dates else None,
        "first_validity_date": min(validity_dates) if validity_dates else None,
        "last_validity_date": max(validity_dates) if validity_dates else None,
        "validity_offsets": sorted(set(offsets)),
        "validity_times": sorted(set(m["validityTime"] for m in messages if m["validityTime"] is not None), key=str),
    }


def strict_month_checks(path: Path, messages: List[Dict[str, Any]], year: int, month: int) -> List[Issue]:
    issues: List[Issue] = []
    cal = calendar_signature(messages, year, month)

    if len(messages) != cal["expected_days"]:
        issues.append(Issue(
            "ERROR", "UNEXPECTED_MESSAGE_COUNT", path.name,
            f"Messaggi={len(messages)}, giorni attesi={cal['expected_days']}."
        ))

    if cal["missing_source_dates"]:
        issues.append(Issue(
            "ERROR", "MISSING_SOURCE_DAYS", path.name,
            "Mancano: " + ", ".join(d.isoformat() for d in cal["missing_source_dates"])
        ))

    if cal["outside_source_dates"]:
        issues.append(Issue(
            "ERROR", "SOURCE_DATES_OUTSIDE_MONTH", path.name,
            "Fuori mese: " + ", ".join(d.isoformat() for d in cal["outside_source_dates"])
        ))

    if cal["duplicate_source_dates"]:
        issues.append(Issue(
            "ERROR", "DUPLICATE_SOURCE_DATES", path.name,
            "Duplicate: " + ", ".join(d.isoformat() for d in cal["duplicate_source_dates"])
        ))

    if cal["validity_offsets"] != [1]:
        issues.append(Issue(
            "ERROR", "UNEXPECTED_VALIDITY_OFFSET", path.name,
            f"Offset validityDate-dataDate={cal['validity_offsets']}; atteso [1]."
        ))

    return issues


def validate_2024_exception(
    current: Path,
    quarantine: Path,
    messages: List[Dict[str, Any]],
) -> Tuple[bool, List[Issue], Dict[str, Any]]:
    issues: List[Issue] = []
    cal = calendar_signature(messages, 2024, 12)
    target_missing = [date(2024, 12, 31)]

    proof = {
        "exception": "MISSING_SOURCE_DAY",
        "accepted_missing_date": "2024-12-31",
        "current_equals_quarantine": False,
        "message_count": len(messages),
        "unique_source_dates": cal["unique_source_dates"],
        "validity_offsets": cal["validity_offsets"],
    }

    if len(messages) != 30:
        issues.append(Issue("ERROR", "2024_EXCEPTION_BAD_COUNT", current.name, f"Attesi 30 messaggi, trovati {len(messages)}."))

    if cal["missing_source_dates"] != target_missing:
        issues.append(Issue(
            "ERROR", "2024_EXCEPTION_BAD_MISSING_SET", current.name,
            f"Missing osservati: {[d.isoformat() for d in cal['missing_source_dates']]}"
        ))

    if cal["outside_source_dates"]:
        issues.append(Issue("ERROR", "2024_EXCEPTION_OUTSIDE_DATES", current.name, str(cal["outside_source_dates"])))

    if cal["duplicate_source_dates"]:
        issues.append(Issue("ERROR", "2024_EXCEPTION_DUPLICATES", current.name, str(cal["duplicate_source_dates"])))

    if cal["validity_offsets"] != [1]:
        issues.append(Issue(
            "ERROR", "2024_EXCEPTION_BAD_OFFSET", current.name,
            f"Offset={cal['validity_offsets']}; atteso [1]."
        ))

    if not quarantine.exists():
        issues.append(Issue(
            "ERROR", "2024_QUARANTINE_PROOF_MISSING", current.name,
            f"Copia di quarantena non trovata: {quarantine}"
        ))
    else:
        same = current.read_bytes() == quarantine.read_bytes()
        proof["current_equals_quarantine"] = same
        if not same:
            issues.append(Issue(
                "ERROR", "2024_REDOWNLOAD_NOT_BYTE_IDENTICAL", current.name,
                "Il riscarico non è byte-identico alla copia precedente."
            ))

    accepted = not any(i.severity == "ERROR" for i in issues)

    if accepted:
        issues.append(Issue(
            "INFO",
            "SOURCE_EXCEPTION_ACCEPTED",
            current.name,
            "31/12/2024 assente dalla sorgente; riscarico byte-identico al primo download. Nessuna imputazione."
        ))

    return accepted, issues, proof


def validate_2025_exception(
    current: Path,
    quarantine: Path,
    messages: List[Dict[str, Any]],
) -> Tuple[bool, List[Issue], Dict[str, Any]]:
    issues: List[Issue] = []
    cal = calendar_signature(messages, 2025, 12)
    target_missing = [date(2025, 12, 31)]

    groups = defaultdict(list)
    for m in messages:
        if m["dataDate"] is not None:
            groups[m["dataDate"]].append(m)

    duplicate_pairs_identical = True
    duplicate_date_count = 0

    for d, group in groups.items():
        if len(group) == 2:
            duplicate_date_count += 1
            if group[0]["message_sha256"] != group[1]["message_sha256"]:
                duplicate_pairs_identical = False
        else:
            duplicate_pairs_identical = False

    current_bytes = current.read_bytes() if current.exists() else b""
    old_bytes = quarantine.read_bytes() if quarantine.exists() else b""

    first_half_equals_second_half = False
    current_equals_old_twice = False
    current_half_equals_old = False

    if current_bytes and len(current_bytes) % 2 == 0:
        half = len(current_bytes) // 2
        first = current_bytes[:half]
        second = current_bytes[half:]
        first_half_equals_second_half = (first == second)
        if old_bytes:
            current_equals_old_twice = (current_bytes == old_bytes + old_bytes)
            current_half_equals_old = (first == old_bytes and second == old_bytes)

    proof = {
        "exception": "DUPLICATED_SOURCE_BLOCK_AND_MISSING_SOURCE_DAY",
        "accepted_missing_date": "2025-12-31",
        "message_count": len(messages),
        "unique_source_dates": cal["unique_source_dates"],
        "duplicate_date_count": duplicate_date_count,
        "all_duplicate_pairs_byte_identical": duplicate_pairs_identical,
        "first_half_equals_second_half": first_half_equals_second_half,
        "current_equals_quarantine_twice": current_equals_old_twice,
        "both_halves_equal_quarantine": current_half_equals_old,
        "validity_offsets": cal["validity_offsets"],
    }

    if len(messages) != 60:
        issues.append(Issue(
            "ERROR", "2025_EXCEPTION_BAD_COUNT", current.name,
            f"Attesi 60 messaggi nella risposta duplicata, trovati {len(messages)}."
        ))

    if cal["missing_source_dates"] != target_missing:
        issues.append(Issue(
            "ERROR", "2025_EXCEPTION_BAD_MISSING_SET", current.name,
            f"Missing osservati: {[d.isoformat() for d in cal['missing_source_dates']]}"
        ))

    if cal["outside_source_dates"]:
        issues.append(Issue("ERROR", "2025_EXCEPTION_OUTSIDE_DATES", current.name, str(cal["outside_source_dates"])))

    if duplicate_date_count != 30:
        issues.append(Issue(
            "ERROR", "2025_EXCEPTION_BAD_DUPLICATE_COUNT", current.name,
            f"Date duplicate attese=30, osservate={duplicate_date_count}."
        ))

    if not duplicate_pairs_identical:
        issues.append(Issue(
            "ERROR", "2025_DUPLICATES_NOT_BYTE_IDENTICAL", current.name,
            "Almeno una coppia con la stessa data non è byte-identica."
        ))

    if cal["validity_offsets"] != [1]:
        issues.append(Issue(
            "ERROR", "2025_EXCEPTION_BAD_OFFSET", current.name,
            f"Offset={cal['validity_offsets']}; atteso [1]."
        ))

    if not quarantine.exists():
        issues.append(Issue(
            "ERROR", "2025_QUARANTINE_PROOF_MISSING", current.name,
            f"Copia di quarantena non trovata: {quarantine}"
        ))
    else:
        if not first_half_equals_second_half:
            issues.append(Issue(
                "ERROR", "2025_HALVES_NOT_IDENTICAL", current.name,
                "Le due metà del file corrente non sono byte-identiche."
            ))
        if not current_equals_old_twice:
            issues.append(Issue(
                "ERROR", "2025_CURRENT_NOT_OLD_TWICE", current.name,
                "Il file corrente non è esattamente quarantine+quarantine."
            ))
        if not current_half_equals_old:
            issues.append(Issue(
                "ERROR", "2025_HALVES_NOT_EQUAL_OLD", current.name,
                "Una o entrambe le metà non coincidono con la copia precedente."
            ))

    accepted = not any(i.severity == "ERROR" for i in issues)

    if accepted:
        issues.append(Issue(
            "INFO",
            "SOURCE_EXCEPTION_ACCEPTED",
            current.name,
            "31/12/2025 assente dalla sorgente; EWDS ha restituito due copie byte-identiche del blocco 01-30/12. "
            "Il raw resta invariato; deduplicazione solo nel prodotto normalizzato."
        ))

    return accepted, issues, proof


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit finale GloFAS Historical Tanaro–Arroscia v1.2")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-values", action="store_true")
    args = parser.parse_args()

    root = args.project_root.expanduser().resolve()
    data_dir = (
        args.data_dir.expanduser().resolve()
        if args.data_dir is not None
        else root / "tanaro_arroscia" / "glofas_historical"
    )
    out_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "tanaro_arroscia" / "audit" / "glofas_v1_2"
    )

    quarantine = data_dir / "_quarantine_missing_dec31"
    manifest = data_dir / "download_manifest.jsonl"

    print("=" * 104)
    print("TANARO–ARROSCIA | GLoFAS HISTORICAL AUDIT v1.2 — FINAL/CANONICAL")
    print(f"Data dir    : {data_dir}")
    print(f"Output dir  : {out_dir}")
    print(f"Quarantine  : {quarantine}")
    print(f"Periodo     : {START_YEAR}-09 .. {END_YEAR}-12")
    print(f"File attesi : {EXPECTED_TOTAL_FILES}")
    print("Policy      : 154 STRICT PASS + 2 eccezioni di sorgente documentate")
    print("=" * 104)

    if not data_dir.exists():
        print(f"ERRORE: cartella non trovata: {data_dir}", file=sys.stderr)
        return 3

    ec = import_eccodes()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_issues: List[Issue] = []
    file_rows: List[Dict[str, Any]] = []
    exception_rows: List[Dict[str, Any]] = []

    expected_names = {expected_filename(y, m) for y, m in expected_month_keys()}
    actual_names = {p.name for p in data_dir.glob("*.grib") if p.is_file()}

    for n in sorted(expected_names - actual_names):
        all_issues.append(Issue("ERROR", "MISSING_FILE", n, "File mensile atteso non presente."))

    for n in sorted(actual_names - expected_names):
        all_issues.append(Issue("WARN", "UNEXPECTED_GRIB_FILE", n, "GRIB inatteso nella cartella principale."))

    strict_pass_count = 0
    source_exception_count = 0
    fail_count = 0

    for idx, (year, month) in enumerate(expected_month_keys(), 1):
        fname = expected_filename(year, month)
        path = data_dir / fname

        print(f"[{idx:03d}/{EXPECTED_TOTAL_FILES}] {fname}", flush=True)

        summary, messages, intrinsic_issues = inspect_grib(
            ec, path, scan_values=not args.skip_values
        )

        local_issues = list(intrinsic_issues)

        manifest_ref = manifest_mentions(manifest, fname)
        if path.exists() and manifest.exists() and not manifest_ref:
            local_issues.append(Issue(
                "WARN", "NOT_FOUND_IN_MANIFEST", fname,
                "Nome file non trovato nel download_manifest.jsonl."
            ))

        cal = calendar_signature(messages, year, month) if messages else {
            "expected_days": calendar.monthrange(year, month)[1],
            "unique_source_dates": 0,
            "missing_source_dates": expected_dates(year, month),
            "outside_source_dates": [],
            "duplicate_source_dates": [],
            "first_source_date": None,
            "last_source_date": None,
            "first_validity_date": None,
            "last_validity_date": None,
            "validity_offsets": [],
            "validity_times": [],
        }

        status = "FAIL"
        exception_code = ""
        exception_proof = {}

        # Se ci sono errori intrinseci (GRIB illeggibile, griglia/variabile errata,
        # valori negativi...), l'eccezione non può sanare il file.
        intrinsic_errors = [i for i in local_issues if i.severity == "ERROR"]

        if not intrinsic_errors:
            if (year, month) == SPECIAL_2024:
                old = quarantine / fname
                accepted, exc_issues, proof = validate_2024_exception(path, old, messages)
                local_issues.extend(exc_issues)
                exception_proof = proof
                if accepted:
                    status = "SOURCE_EXCEPTION_ACCEPTED"
                    exception_code = "MISSING_2024-12-31"
                    source_exception_count += 1
                else:
                    fail_count += 1

            elif (year, month) == SPECIAL_2025:
                old = quarantine / fname
                accepted, exc_issues, proof = validate_2025_exception(path, old, messages)
                local_issues.extend(exc_issues)
                exception_proof = proof
                if accepted:
                    status = "SOURCE_EXCEPTION_ACCEPTED"
                    exception_code = "MISSING_2025-12-31_AND_DUPLICATED_BLOCK"
                    source_exception_count += 1
                else:
                    fail_count += 1

            else:
                strict_issues = strict_month_checks(path, messages, year, month)
                local_issues.extend(strict_issues)
                if any(i.severity == "ERROR" for i in strict_issues):
                    fail_count += 1
                else:
                    status = "STRICT_PASS"
                    strict_pass_count += 1
        else:
            fail_count += 1

        # Se sono comparsi errori durante la validazione eccezione, forza FAIL.
        if any(i.severity == "ERROR" for i in local_issues):
            if status != "FAIL":
                if status == "STRICT_PASS":
                    strict_pass_count -= 1
                elif status == "SOURCE_EXCEPTION_ACCEPTED":
                    source_exception_count -= 1
                fail_count += 1
            status = "FAIL"

        row = {
            "file": fname,
            "year": year,
            "month": month,
            "status": status,
            "exception_code": exception_code,
            "exists": summary["exists"],
            "size_bytes": summary["size_bytes"],
            "sha256": summary["sha256"],
            "message_count": summary["message_count"],
            "expected_days": cal["expected_days"],
            "unique_source_dates": cal["unique_source_dates"],
            "missing_source_dates": "|".join(d.isoformat() for d in cal["missing_source_dates"]),
            "outside_source_dates": "|".join(d.isoformat() for d in cal["outside_source_dates"]),
            "duplicate_source_dates": "|".join(d.isoformat() for d in cal["duplicate_source_dates"]),
            "first_source_date": "" if cal["first_source_date"] is None else cal["first_source_date"].isoformat(),
            "last_source_date": "" if cal["last_source_date"] is None else cal["last_source_date"].isoformat(),
            "first_validity_date": "" if cal["first_validity_date"] is None else cal["first_validity_date"].isoformat(),
            "last_validity_date": "" if cal["last_validity_date"] is None else cal["last_validity_date"].isoformat(),
            "validity_offsets": "|".join(str(x) for x in cal["validity_offsets"]),
            "finite_values": summary["finite_values"],
            "missing_or_nonfinite_values": summary["missing_or_nonfinite_values"],
            "negative_values": summary["negative_values"],
            "value_min": summary["value_min"],
            "value_max": summary["value_max"],
            "manifest_ref": manifest_ref,
            "grid": json.dumps(summary["grid"], ensure_ascii=False, sort_keys=True),
            "variable": json.dumps(summary["variable"], ensure_ascii=False, sort_keys=True),
        }
        file_rows.append(row)

        if exception_proof:
            erow = {
                "file": fname,
                "status": status,
                "exception_code": exception_code,
                "proof_json": json.dumps(exception_proof, ensure_ascii=False, sort_keys=True),
            }
            exception_rows.append(erow)

        all_issues.extend(local_issues)

    if not manifest.exists():
        all_issues.append(Issue(
            "WARN", "MANIFEST_MISSING", "download_manifest.jsonl",
            "Manifest non trovato."
        ))

    error_count = sum(i.severity == "ERROR" for i in all_issues)
    warn_count = sum(i.severity == "WARN" for i in all_issues)
    info_count = sum(i.severity == "INFO" for i in all_issues)

    overall = (
        "PASS_WITH_SOURCE_EXCEPTIONS"
        if (
            error_count == 0
            and fail_count == 0
            and strict_pass_count == 154
            and source_exception_count == 2
            and len(file_rows) == EXPECTED_TOTAL_FILES
        )
        else "FAIL"
    )

    dataset_status = "CLOSED" if overall == "PASS_WITH_SOURCE_EXCEPTIONS" else "OPEN_REVIEW_REQUIRED"

    file_fields = [
        "file", "year", "month", "status", "exception_code",
        "exists", "size_bytes", "sha256",
        "message_count", "expected_days", "unique_source_dates",
        "missing_source_dates", "outside_source_dates", "duplicate_source_dates",
        "first_source_date", "last_source_date",
        "first_validity_date", "last_validity_date", "validity_offsets",
        "finite_values", "missing_or_nonfinite_values", "negative_values",
        "value_min", "value_max", "manifest_ref", "grid", "variable",
    ]
    write_csv(out_dir / "glofas_audit_files_v1_2.csv", file_rows, file_fields)

    write_csv(
        out_dir / "glofas_audit_issues_v1_2.csv",
        [asdict(i) for i in all_issues],
        ["severity", "code", "file", "detail"],
    )

    write_csv(
        out_dir / "glofas_source_exceptions_v1_2.csv",
        exception_rows,
        ["file", "status", "exception_code", "proof_json"],
    )

    summary_json = {
        "audit_version": SCRIPT_VERSION,
        "overall_status": overall,
        "dataset_status": dataset_status,
        "period": {
            "start": "1987-09-01",
            "end_nominal": "2025-12-31",
            "months": [9, 10, 11, 12],
        },
        "counts": {
            "expected_monthly_files": EXPECTED_TOTAL_FILES,
            "strict_pass_files": strict_pass_count,
            "source_exception_files": source_exception_count,
            "failed_files": fail_count,
            "errors": error_count,
            "warnings": warn_count,
            "info": info_count,
            "nominal_days": sum(
                calendar.monthrange(y, m)[1]
                for y, m in expected_month_keys()
            ),
            "documented_missing_source_days": 2 if overall == "PASS_WITH_SOURCE_EXCEPTIONS" else None,
        },
        "accepted_source_exceptions": [
            {
                "date": "2024-12-31",
                "reason": "Not returned by EWDS/GloFAS v4 in repeated download; current file byte-identical to quarantined first download.",
                "treatment": "MISSING_SOURCE; no interpolation.",
            },
            {
                "date": "2025-12-31",
                "reason": "Not returned by EWDS/GloFAS v4; repeated download returned the 01-30 Dec block twice, byte-identically.",
                "treatment": "MISSING_SOURCE; deduplicate only in derived normalized product; raw preserved unchanged.",
            },
        ],
        "canonical_policy": {
            "raw_immutable": True,
            "interpolation_of_missing_source_days": False,
            "deduplicate_raw_2025_12": False,
            "deduplicate_only_derived_normalized_product": True,
        },
        "paths": {
            "data_dir": str(data_dir),
            "quarantine": str(quarantine),
            "manifest": str(manifest),
            "output_dir": str(out_dir),
        },
    }

    (out_dir / "glofas_audit_summary_v1_2.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    nominal_days = summary_json["counts"]["nominal_days"]
    missing_pct = 100.0 * 2 / nominal_days

    lines = [
        "TANARO–ARROSCIA | GLoFAS HISTORICAL AUDIT v1.2 — FINAL/CANONICAL",
        "=" * 88,
        f"OVERALL STATUS : {overall}",
        f"DATASET STATUS : {dataset_status}",
        "",
        f"File mensili attesi              : {EXPECTED_TOTAL_FILES}",
        f"STRICT PASS                      : {strict_pass_count}",
        f"SOURCE_EXCEPTION_ACCEPTED        : {source_exception_count}",
        f"FAIL                             : {fail_count}",
        f"ERROR non spiegati               : {error_count}",
        f"WARN                             : {warn_count}",
        f"Giorni nominali Sep-Dec 1987-2025: {nominal_days}",
        f"Giorni mancanti di sorgente      : 2 ({missing_pct:.6f}%)",
        "",
        "ECCEZIONI DI SORGENTE ACCETTATE",
        "1) 2024-12-31",
        "   - non restituito dalla sorgente",
        "   - il secondo download è byte-identico al primo",
        "   - trattamento: MISSING_SOURCE, nessuna interpolazione",
        "",
        "2) 2025-12-31",
        "   - non restituito dalla sorgente",
        "   - il secondo download contiene due copie byte-identiche del blocco 01-30 dicembre",
        "   - raw preservato invariato",
        "   - deduplicazione soltanto nel prodotto normalizzato derivato",
        "   - trattamento del 31/12: MISSING_SOURCE, nessuna interpolazione",
        "",
        "POLICY DI CONSERVAZIONE",
        "- Non modificare i GRIB raw.",
        "- Conservare la cartella _quarantine_missing_dec31.",
        "- Conservare i report diagnostici v1.0/v1.1 e la diagnostica dicembre.",
        "- Per le elaborazioni future usare la v1.2 come audit canonico.",
        "",
    ]

    (out_dir / "glofas_audit_summary_v1_2.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 104)
    print(f"RISULTATO AUDIT : {overall}")
    print(f"DATASET STATUS  : {dataset_status}")
    print(f"STRICT PASS     : {strict_pass_count}")
    print(f"SOURCE EXCEPT.  : {source_exception_count}")
    print(f"FAIL            : {fail_count}")
    print(f"ERROR           : {error_count} | WARN: {warn_count}")
    print(f"Report          : {out_dir}")
    print("=" * 104)

    if overall == "PASS_WITH_SOURCE_EXCEPTIONS":
        print("GLoFAS HISTORICAL: CLOSED — 154 mesi strict PASS + 2 eccezioni di sorgente documentate.")
        return 0

    print("GLoFAS HISTORICAL: FAIL — il blocco NON è chiuso; leggere glofas_audit_issues_v1_2.csv.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
