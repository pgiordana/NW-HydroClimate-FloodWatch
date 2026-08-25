#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audit_tanaro_arroscia_observed_control_series_v1_0.py

Audit mirato delle DUE serie osservative canoniche usate come sezioni di controllo:

TANARO
------
ARPA Piemonte — GARESSIO TANARO
station id: PIE-004095-900
cartella canonica:
  tanaro_arroscia/observations/arpa_piemonte/daily_hydro/

ARROSCIA
--------
ARPAL/OMIRL — PIEVE DI TECO (IDRO)
station code: ME00342
slug canonico: pieve_teco_level
cartella canonica:
  tanaro_arroscia/observations/arpal_omirl/hourly/pieve_teco_level/

Scopo
-----
- trovare i file realmente presenti;
- distinguere dati veri da file vuoti/header-only;
- ricostruire colonne, copertura temporale e passo;
- NON interpolare;
- NON confondere le copie observations_nw/* con il ramo canonico tanaro_arroscia/*;
- dichiarare se una serie osservata ORARIA congiunta è già costruibile.

Output:
  tanaro_arroscia/hydrology/observed_control_series_audit_v1_0.json
  tanaro_arroscia/hydrology/observed_control_series_audit_v1_0.txt
  tanaro_arroscia/hydrology/observed_control_series_files_v1_0.csv

Non modifica alcun dato sorgente.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from statistics import median

import pandas as pd
import numpy as np


VERSION = "1.0"

TANARO_ID = "PIE-004095-900"
TANARO_NAME = "GARESSIO_TANARO"

ARROSCIA_CODE = "ME00342"
ARROSCIA_SLUG = "pieve_teco_level"


def safe_read_csv(path: Path):
    attempts = [
        dict(sep=None, engine="python"),
        dict(sep=","),
        dict(sep=";"),
        dict(sep="\t"),
    ]
    last = None
    for kw in attempts:
        try:
            df = pd.read_csv(path, **kw)
            return df, None
        except Exception as exc:
            last = repr(exc)
    return None, last


def detect_datetime(df: pd.DataFrame):
    if df is None or len(df) == 0:
        return None, None

    preferred = []
    for c in df.columns:
        cl = str(c).lower()
        if any(x in cl for x in (
            "datetime", "timestamp", "date_time", "dataora",
            "data_ora", "time", "date", "data", "ora"
        )):
            preferred.append(c)

    candidates = preferred + [c for c in df.columns if c not in preferred]

    best = None
    for c in candidates:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            continue
        try:
            dt = pd.to_datetime(s, errors="coerce", utc=False)
        except Exception:
            continue
        n = int(dt.notna().sum())
        if n < max(2, int(0.5 * len(df))):
            continue
        score = n
        if best is None or score > best[0]:
            best = (score, c, dt)

    if best is None:
        # prova combinazioni date + hour
        date_cols = [
            c for c in df.columns
            if any(x in str(c).lower() for x in ("date", "data"))
        ]
        hour_cols = [
            c for c in df.columns
            if any(x in str(c).lower() for x in ("hour", "ora", "time"))
        ]
        for dc in date_cols:
            for hc in hour_cols:
                if dc == hc:
                    continue
                try:
                    comb = (
                        df[dc].astype(str).str.strip()
                        + " "
                        + df[hc].astype(str).str.strip()
                    )
                    dt = pd.to_datetime(comb, errors="coerce", utc=False)
                except Exception:
                    continue
                n = int(dt.notna().sum())
                if n >= max(2, int(0.5 * len(df))):
                    return f"{dc}+{hc}", dt

        return None, None

    return str(best[1]), best[2]


def time_stats(dt):
    if dt is None:
        return {
            "valid_timestamps": 0,
            "start": None,
            "end": None,
            "median_step_hours": None,
            "duplicate_timestamps": None,
        }

    d = pd.Series(dt).dropna().sort_values()
    if len(d) == 0:
        return {
            "valid_timestamps": 0,
            "start": None,
            "end": None,
            "median_step_hours": None,
            "duplicate_timestamps": 0,
        }

    dup = int(d.duplicated().sum())
    u = d.drop_duplicates()
    step = None
    if len(u) >= 2:
        diffs = u.diff().dropna().dt.total_seconds() / 3600.0
        finite = diffs[np.isfinite(diffs)]
        if len(finite):
            step = float(finite.median())

    return {
        "valid_timestamps": int(len(d)),
        "start": str(d.iloc[0]),
        "end": str(d.iloc[-1]),
        "median_step_hours": step,
        "duplicate_timestamps": dup,
    }


def inspect_csv(path: Path, role: str):
    size = path.stat().st_size
    df, err = safe_read_csv(path)

    rec = {
        "role": role,
        "path": str(path),
        "name": path.name,
        "size_bytes": size,
        "read_error": err,
        "rows": None,
        "columns": None,
        "datetime_column": None,
        "valid_timestamps": 0,
        "start": None,
        "end": None,
        "median_step_hours": None,
        "duplicate_timestamps": None,
        "numeric_columns": [],
        "status": None,
    }

    if df is None:
        rec["status"] = "READ_ERROR"
        return rec

    rec["rows"] = int(len(df))
    rec["columns"] = [str(x) for x in df.columns]

    if len(df) == 0:
        rec["status"] = "EMPTY_OR_HEADER_ONLY"
        return rec

    dt_col, dt = detect_datetime(df)
    rec["datetime_column"] = dt_col
    rec.update(time_stats(dt))

    nums = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        n = int(s.notna().sum())
        if n >= max(1, int(0.2 * len(df))):
            nums.append({
                "column": str(c),
                "valid_numeric": n,
                "min": float(s.min()) if n else None,
                "max": float(s.max()) if n else None,
            })
    rec["numeric_columns"] = nums

    if rec["valid_timestamps"] == 0:
        rec["status"] = "DATA_PRESENT_TIME_UNRESOLVED"
    else:
        rec["status"] = "DATA_OK"

    return rec


def matching_manifest_lines(path: Path, terms):
    out = []
    if not path.exists():
        return out
    try:
        for i, line in enumerate(path.read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines(), 1):
            u = line.upper()
            if any(t.upper() in u for t in terms):
                out.append({"line": i, "text": line[:2000]})
    except Exception:
        pass
    return out


def infer_year_from_name(name: str):
    m = re.search(r"(19|20)\d{2}", name)
    return int(m.group(0)) if m else None


def main():
    root = Path(__file__).resolve().parent
    hyd = root / "tanaro_arroscia" / "hydrology"
    hyd.mkdir(parents=True, exist_ok=True)

    tan_dir = (
        root / "tanaro_arroscia" / "observations"
        / "arpa_piemonte" / "daily_hydro"
    )
    arr_dir = (
        root / "tanaro_arroscia" / "observations"
        / "arpal_omirl" / "hourly" / ARROSCIA_SLUG
    )

    print("=" * 104)
    print("TANARO–ARROSCIA | OBSERVED CONTROL SERIES AUDIT v1.0")
    print("=" * 104)

    # ------------------------------------------------------------------
    # TANARO
    # ------------------------------------------------------------------
    print("\n[TANARO] GARESSIO TANARO")
    print(f"Directory: {tan_dir}")

    tan_files = []
    if tan_dir.exists():
        tan_files = sorted(
            p for p in tan_dir.iterdir()
            if p.is_file()
            and (
                TANARO_ID.upper() in p.name.upper()
                or TANARO_NAME.upper() in p.name.upper()
                or "GARESSIO" in p.name.upper()
            )
        )

    tan_csv = []
    tan_other = []
    for p in tan_files:
        if p.suffix.lower() == ".csv":
            rec = inspect_csv(p, "tanaro_garessio_daily")
            tan_csv.append(rec)
        else:
            tan_other.append({
                "path": str(p),
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "suffix": p.suffix.lower(),
            })

    for r in tan_csv:
        print(
            f"  CSV {r['name']} | rows={r['rows']} | "
            f"status={r['status']} | {r['start']} -> {r['end']} | "
            f"step_h={r['median_step_hours']}"
        )

    if not tan_csv:
        print("  NESSUN CSV GARESSIO TROVATO nella cartella canonica.")
    for r in tan_other:
        print(f"  {r['suffix']} {r['name']} | {r['size_bytes']} bytes")

    # ------------------------------------------------------------------
    # ARROSCIA
    # ------------------------------------------------------------------
    print("\n[ARROSCIA] PIEVE DI TECO (IDRO)")
    print(f"Directory: {arr_dir}")

    arr_files = sorted(arr_dir.glob("*.csv")) if arr_dir.exists() else []
    arr_csv = [inspect_csv(p, "arroscia_pieve_teco_hourly") for p in arr_files]

    # Se la cartella canonica non c'è, cerca SOLO nel ramo canonico,
    # senza usare observations_nw come sostituto.
    fallback_paths = []
    if not arr_files:
        base = root / "tanaro_arroscia" / "observations" / "arpal_omirl"
        if base.exists():
            fallback_paths = sorted(base.glob("**/*pieve_teco_level*.csv"))
            arr_csv = [
                inspect_csv(p, "arroscia_pieve_teco_hourly_fallback")
                for p in fallback_paths
            ]

    years_data = []
    years_empty = []
    years_error = []
    for r in arr_csv:
        y = infer_year_from_name(r["name"])
        if r["status"] == "DATA_OK":
            if y is not None:
                years_data.append(y)
        elif r["status"] == "EMPTY_OR_HEADER_ONLY":
            if y is not None:
                years_empty.append(y)
        else:
            if y is not None:
                years_error.append(y)

    print(f"  file CSV trovati: {len(arr_csv)}")
    print(f"  anni DATA_OK    : {sorted(set(years_data))}")
    print(f"  anni EMPTY      : {sorted(set(years_empty))}")
    print(f"  anni ERROR      : {sorted(set(years_error))}")

    good_arr = [r for r in arr_csv if r["status"] == "DATA_OK"]
    if good_arr:
        starts = [r["start"] for r in good_arr if r["start"]]
        ends = [r["end"] for r in good_arr if r["end"]]
        steps = [
            r["median_step_hours"] for r in good_arr
            if r["median_step_hours"] is not None
        ]
        print(
            f"  coverage data   : "
            f"{min(starts) if starts else None} -> "
            f"{max(ends) if ends else None}"
        )
        print(
            f"  median step h   : "
            f"{median(steps) if steps else None}"
        )

    # ------------------------------------------------------------------
    # MANIFEST
    # ------------------------------------------------------------------
    tan_manifest = (
        root / "tanaro_arroscia" / "observations"
        / "arpa_piemonte" / "arpa_download_manifest_v1_1.jsonl"
    )
    arr_manifest = (
        root / "tanaro_arroscia" / "observations"
        / "arpal_omirl" / "arpal_download_manifest_v5_4.jsonl"
    )

    tan_manifest_hits = matching_manifest_lines(
        tan_manifest, [TANARO_ID, "GARESSIO"]
    )
    arr_manifest_hits = matching_manifest_lines(
        arr_manifest, [ARROSCIA_CODE, ARROSCIA_SLUG, "PIEVE DI TECO"]
    )

    # ------------------------------------------------------------------
    # READINESS
    # ------------------------------------------------------------------
    tan_good = [r for r in tan_csv if r["status"] == "DATA_OK"]
    tan_hourly = any(
        r["median_step_hours"] is not None
        and r["median_step_hours"] <= 1.5
        for r in tan_good
    )
    tan_daily = any(
        r["median_step_hours"] is not None
        and 20 <= r["median_step_hours"] <= 28
        for r in tan_good
    )

    arr_hourly = any(
        r["median_step_hours"] is not None
        and r["median_step_hours"] <= 1.5
        for r in good_arr
    )

    joint_hourly_ready = bool(tan_hourly and arr_hourly)

    if joint_hourly_ready:
        readiness = "READY_FOR_OBSERVED_HOURLY_JOINT_HYDROGRAPHS"
    elif tan_daily and arr_hourly:
        readiness = (
            "NOT_READY_FOR_HOURLY_PEAK_LAG: "
            "TANARO_DAILY_ONLY_ARROSCIA_HOURLY"
        )
    elif tan_good or good_arr:
        readiness = "PARTIAL_OBSERVED_COVERAGE"
    else:
        readiness = "OBSERVED_CONTROL_SERIES_NOT_RESOLVED"

    report = {
        "version": VERSION,
        "status": "PASS",
        "canonical_only": True,
        "tanaro": {
            "station": "GARESSIO TANARO",
            "station_id": TANARO_ID,
            "directory": str(tan_dir),
            "csv_files": tan_csv,
            "other_files": tan_other,
            "manifest": str(tan_manifest),
            "manifest_matches": tan_manifest_hits,
            "hourly_available": tan_hourly,
            "daily_available": tan_daily,
        },
        "arroscia": {
            "station": "PIEVE DI TECO (IDRO)",
            "station_code": ARROSCIA_CODE,
            "slug": ARROSCIA_SLUG,
            "directory": str(arr_dir),
            "csv_files": arr_csv,
            "years_data_ok": sorted(set(years_data)),
            "years_empty": sorted(set(years_empty)),
            "years_error": sorted(set(years_error)),
            "manifest": str(arr_manifest),
            "manifest_matches": arr_manifest_hits,
            "hourly_available": arr_hourly,
        },
        "joint_hourly_ready": joint_hourly_ready,
        "readiness": readiness,
        "rule": (
            "No temporal upsampling/interpolation of daily Tanaro data "
            "is permitted for peak-lag estimation."
        ),
        "raw_modified": False,
    }

    json_p = hyd / "observed_control_series_audit_v1_0.json"
    json_p.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_p = hyd / "observed_control_series_files_v1_0.csv"
    flat = []
    for r in tan_csv + arr_csv:
        flat.append({
            "role": r["role"],
            "path": r["path"],
            "name": r["name"],
            "size_bytes": r["size_bytes"],
            "rows": r["rows"],
            "status": r["status"],
            "datetime_column": r["datetime_column"],
            "start": r["start"],
            "end": r["end"],
            "median_step_hours": r["median_step_hours"],
            "duplicate_timestamps": r["duplicate_timestamps"],
        })
    with csv_p.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "role", "path", "name", "size_bytes", "rows", "status",
            "datetime_column", "start", "end", "median_step_hours",
            "duplicate_timestamps",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(flat)

    txt_p = hyd / "observed_control_series_audit_v1_0.txt"
    lines = [
        "TANARO–ARROSCIA | OBSERVED CONTROL SERIES AUDIT v1.0",
        "=" * 104,
        "STATUS                         : PASS",
        f"READINESS                      : {readiness}",
        f"JOINT HOURLY OBSERVED READY    : {joint_hourly_ready}",
        "",
        "TANARO — GARESSIO",
        f"  CSV canonici                 : {len(tan_csv)}",
        f"  Daily available              : {tan_daily}",
        f"  Hourly available             : {tan_hourly}",
    ]
    for r in tan_csv:
        lines.append(
            f"  - {r['name']} | {r['status']} | rows={r['rows']} | "
            f"{r['start']} -> {r['end']} | step_h={r['median_step_hours']}"
        )

    lines += [
        "",
        "ARROSCIA — PIEVE DI TECO IDRO",
        f"  CSV canonici                 : {len(arr_csv)}",
        f"  Hourly available             : {arr_hourly}",
        f"  DATA_OK years                : {sorted(set(years_data))}",
        f"  EMPTY years                  : {sorted(set(years_empty))}",
        f"  ERROR years                  : {sorted(set(years_error))}",
        "",
        "REGOLA METODOLOGICA",
        "  Non effettuare upsampling/interpolazione della serie giornaliera",
        "  del Tanaro per stimare orari di picco o Δt sub-giornalieri.",
        "",
        "OUTPUT",
        f"  {json_p}",
        f"  {csv_p}",
    ]

    txt_p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 104)
    print(f"READINESS                   : {readiness}")
    print(f"JOINT HOURLY OBSERVED READY : {joint_hourly_ready}")
    print(f"JSON                        : {json_p}")
    print(f"CSV                         : {csv_p}")
    print(f"TXT                         : {txt_p}")
    print("=" * 104)


if __name__ == "__main__":
    main()
