#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
probe_nw_operational_providers_current_v1_0.py

FASE 13 — PROBE REALE DELLE SORGENTI OPERATIVE CORRENTI.

Questo script effettua un test MINIMO ma reale di accesso alle due sorgenti
candidate del futuro NW FloodWatch:

A) ECMWF Open Data / IFS
   - identifica il ciclo più recente disponibile;
   - scarica un piccolo set di campi single-level sicuramente documentati:
       msl, tcwv, tp
   - scarica q/u/v/t ai livelli 925/850/700 hPa;
   - testa separatamente alcuni parametri che il precedente preflight aveva
     ipotizzato ma che NON devono essere considerati disponibili finché il
     probe non lo dimostra:
       cape, sd, swvl1, swvl2, swvl3
   - nessun parametro opzionale fallito fa fallire l'intero provider.

B) Copernicus Marine
   - prova i due spelling dataset candidati:
       cmems_mod_med_phy-tem_ancf_4.2km_P1D-m
       cmems_mod_med_phy-tem_anfc_4.2km_P1D-m
   - seleziona automaticamente quello realmente presente nel catalogo;
   - legge la copertura temporale disponibile;
   - scarica un micro-subset thetao nella Ligurian Sea dall'ultimo giorno
     disponibile/recente.

SCOPO
-----
Questo NON costruisce ancora le 83 feature e NON esegue il modello.
Serve a sapere se:
- le API funzionano davvero dalla macchina dell'utente;
- i pacchetti sono installati;
- i dataset/parametri candidati esistono;
- qual è il ciclo/data realmente disponibile;
- quali gap del preflight sono reali.

OUTPUT
------
nw_operational_provider_probe_current_v1_0/
  ecmwf/
  copernicus_marine/
  ecmwf_probe_registry_v1_0.csv
  copernicus_marine_probe_registry_v1_0.csv
  provider_probe_summary_v1_0.csv
  provider_probe_audit_v1_0.json
  provider_probe_audit_v1_0.txt
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


ECMWF_SAFE_SURFACE_PARAMS = ["msl", "tcwv", "tp"]
ECMWF_SAFE_PL_PARAMS = ["q", "u", "v", "t"]
ECMWF_SAFE_LEVELS = [925, 850, 700]

# These are intentionally tested, not assumed.
ECMWF_OPTIONAL_CANDIDATES = [
    "cape",
    "sd",
    "swvl1",
    "swvl2",
    "swvl3",
]

CMEMS_DATASET_CANDIDATES = [
    "cmems_mod_med_phy-tem_ancf_4.2km_P1D-m",
    "cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
]

CMEMS_VARIABLE = "thetao"

# Tiny Ligurian Sea box, intentionally offshore.
CMEMS_BBOX = {
    "minimum_longitude": 8.00,
    "maximum_longitude": 8.10,
    "minimum_latitude": 43.50,
    "maximum_latitude": 43.60,
    "minimum_depth": 0.0,
    "maximum_depth": 10.0,
}


def fmt_seconds(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def progress(phase, done, total, started, current=""):
    elapsed = max(time.time() - started, 1e-9)
    pct = 100.0 * done / total if total else 100.0
    rate = done / elapsed if done else 0.0
    eta = (total - done) / rate if rate > 0 else float("inf")

    msg = (
        f"\r{phase} | {done}/{total} | {pct:6.2f}% "
        f"| elapsed {fmt_seconds(elapsed)} "
        f"| rate {rate:7.2f}/s | ETA {fmt_seconds(eta)}"
    )
    if current:
        msg += f" | {str(current)[:135]}"

    print(msg.ljust(285), end="", flush=True)
    if done >= total:
        print(flush=True)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def short_error(exc):
    return f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')[:500]}"


def module_available(name):
    return importlib.util.find_spec(name) is not None


def to_jsonable(obj):
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except TypeError:
            return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    return json.loads(json.dumps(obj, default=str))


def parse_datetime_loose(value):
    if value is None:
        return None

    # Already datetime-like.
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)

    # Numerical timestamps: try common units.
    if isinstance(value, (int, float)):
        x = float(value)
        for divisor in [1.0, 1e3, 1e6, 1e9]:
            try:
                dt = datetime.fromtimestamp(x / divisor, tz=timezone.utc)
                if 1990 <= dt.year <= 2100:
                    return dt
            except Exception:
                pass
        return None

    s = str(value).strip()

    # numeric as string
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        try:
            return parse_datetime_loose(float(s))
        except Exception:
            pass

    s = s.replace("Z", "+00:00")

    for candidate in [s, s[:19]]:
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass

    return None


def recursively_find_time_max(obj, path="root"):
    """
    Generic extractor for Copernicus Marine catalogue structures.
    Returns list of candidate (datetime, path, raw_value).
    """
    found = []

    if isinstance(obj, dict):
        text = " ".join(
            f"{k}={v}"
            for k, v in obj.items()
            if isinstance(v, (str, int, float))
        ).lower()

        if "maximum_value" in obj:
            dt = parse_datetime_loose(obj.get("maximum_value"))
            if dt is not None:
                score_text = text + " " + path.lower()
                if "time" in score_text:
                    found.append(
                        (dt, path, obj.get("maximum_value"))
                    )

        for k, v in obj.items():
            found.extend(
                recursively_find_time_max(v, f"{path}.{k}")
            )

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found.extend(
                recursively_find_time_max(v, f"{path}[{i}]")
            )

    return found


def find_dataset_ids(obj):
    ids = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "dataset_id" and isinstance(v, str):
                ids.append(v)
            else:
                ids.extend(find_dataset_ids(v))

    elif isinstance(obj, list):
        for v in obj:
            ids.extend(find_dataset_ids(v))

    return ids


def run_ecmwf_probe(out_dir):
    rows = []
    files = []

    if not module_available("ecmwf.opendata"):
        rows.append(
            {
                "probe_id": "ECMWF_PACKAGE",
                "probe_type": "PACKAGE",
                "status": "FAIL_PACKAGE_MISSING",
                "request": "import ecmwf.opendata",
                "available_datetime_utc": "",
                "file": "",
                "bytes": 0,
                "sha256": "",
                "error": (
                    "Install with: python -m pip install -U ecmwf-opendata"
                ),
            }
        )
        return rows, files, False

    from ecmwf.opendata import Client

    client = Client(
        source="ecmwf",
        model="ifs",
        resol="0p25",
    )

    # Latest safe availability without download.
    try:
        latest = client.latest(
            stream="oper",
            type="fc",
            step=3,
            param="msl",
        )

        rows.append(
            {
                "probe_id": "ECMWF_LATEST",
                "probe_type": "LATEST",
                "status": "PASS",
                "request": "oper/fc step=3 param=msl",
                "available_datetime_utc": str(latest),
                "file": "",
                "bytes": 0,
                "sha256": "",
                "error": "",
            }
        )
    except Exception as exc:
        rows.append(
            {
                "probe_id": "ECMWF_LATEST",
                "probe_type": "LATEST",
                "status": "FAIL",
                "request": "oper/fc step=3 param=msl",
                "available_datetime_utc": "",
                "file": "",
                "bytes": 0,
                "sha256": "",
                "error": short_error(exc),
            }
        )
        latest = None

    # Safe documented surface fields.
    surface_p = out_dir / "ecmwf_surface_safe_step3.grib2"

    try:
        result = client.retrieve(
            stream="oper",
            type="fc",
            step=3,
            param=ECMWF_SAFE_SURFACE_PARAMS,
            target=str(surface_p),
        )

        ok = surface_p.exists() and surface_p.stat().st_size > 0

        rows.append(
            {
                "probe_id": "ECMWF_SAFE_SURFACE",
                "probe_type": "DOWNLOAD",
                "status": "PASS" if ok else "FAIL_EMPTY_FILE",
                "request": (
                    "oper/fc step=3 param="
                    + ",".join(ECMWF_SAFE_SURFACE_PARAMS)
                ),
                "available_datetime_utc": str(
                    getattr(result, "datetime", latest or "")
                ),
                "file": str(surface_p),
                "bytes": (
                    surface_p.stat().st_size
                    if surface_p.exists()
                    else 0
                ),
                "sha256": (
                    sha256(surface_p)
                    if ok
                    else ""
                ),
                "error": "",
            }
        )

        if ok:
            files.append(surface_p)

    except Exception as exc:
        rows.append(
            {
                "probe_id": "ECMWF_SAFE_SURFACE",
                "probe_type": "DOWNLOAD",
                "status": "FAIL",
                "request": (
                    "oper/fc step=3 param="
                    + ",".join(ECMWF_SAFE_SURFACE_PARAMS)
                ),
                "available_datetime_utc": "",
                "file": str(surface_p),
                "bytes": 0,
                "sha256": "",
                "error": short_error(exc),
            }
        )

    # Safe documented pressure-level fields.
    pl_p = out_dir / "ecmwf_pressure_safe_925_850_700_step3.grib2"

    try:
        result = client.retrieve(
            stream="oper",
            type="fc",
            step=3,
            param=ECMWF_SAFE_PL_PARAMS,
            levelist=ECMWF_SAFE_LEVELS,
            target=str(pl_p),
        )

        ok = pl_p.exists() and pl_p.stat().st_size > 0

        rows.append(
            {
                "probe_id": "ECMWF_SAFE_PRESSURE",
                "probe_type": "DOWNLOAD",
                "status": "PASS" if ok else "FAIL_EMPTY_FILE",
                "request": (
                    "oper/fc step=3 param="
                    + ",".join(ECMWF_SAFE_PL_PARAMS)
                    + " levelist="
                    + ",".join(str(x) for x in ECMWF_SAFE_LEVELS)
                ),
                "available_datetime_utc": str(
                    getattr(result, "datetime", latest or "")
                ),
                "file": str(pl_p),
                "bytes": (
                    pl_p.stat().st_size
                    if pl_p.exists()
                    else 0
                ),
                "sha256": (
                    sha256(pl_p)
                    if ok
                    else ""
                ),
                "error": "",
            }
        )

        if ok:
            files.append(pl_p)

    except Exception as exc:
        rows.append(
            {
                "probe_id": "ECMWF_SAFE_PRESSURE",
                "probe_type": "DOWNLOAD",
                "status": "FAIL",
                "request": (
                    "oper/fc step=3 param="
                    + ",".join(ECMWF_SAFE_PL_PARAMS)
                    + " levelist="
                    + ",".join(str(x) for x in ECMWF_SAFE_LEVELS)
                ),
                "available_datetime_utc": "",
                "file": str(pl_p),
                "bytes": 0,
                "sha256": "",
                "error": short_error(exc),
            }
        )

    # Optional availability checks. Do not download.
    for param in ECMWF_OPTIONAL_CANDIDATES:
        try:
            dt = client.latest(
                stream="oper",
                type="fc",
                step=3,
                param=param,
            )

            rows.append(
                {
                    "probe_id": f"ECMWF_OPTIONAL_{param.upper()}",
                    "probe_type": "OPTIONAL_AVAILABILITY",
                    "status": "PASS_AVAILABLE",
                    "request": f"oper/fc step=3 param={param}",
                    "available_datetime_utc": str(dt),
                    "file": "",
                    "bytes": 0,
                    "sha256": "",
                    "error": "",
                }
            )

        except Exception as exc:
            rows.append(
                {
                    "probe_id": f"ECMWF_OPTIONAL_{param.upper()}",
                    "probe_type": "OPTIONAL_AVAILABILITY",
                    "status": "NOT_AVAILABLE_OR_UNSUPPORTED",
                    "request": f"oper/fc step=3 param={param}",
                    "available_datetime_utc": "",
                    "file": "",
                    "bytes": 0,
                    "sha256": "",
                    "error": short_error(exc),
                }
            )

    critical = {
        r["probe_id"]: r["status"]
        for r in rows
    }

    provider_pass = (
        critical.get("ECMWF_SAFE_SURFACE") == "PASS"
        and critical.get("ECMWF_SAFE_PRESSURE") == "PASS"
    )

    return rows, files, provider_pass


def run_cmems_probe(out_dir):
    rows = []
    files = []

    if not module_available("copernicusmarine"):
        rows.append(
            {
                "probe_id": "CMEMS_PACKAGE",
                "probe_type": "PACKAGE",
                "status": "FAIL_PACKAGE_MISSING",
                "dataset_id": "",
                "available_datetime_utc": "",
                "sample_date_utc": "",
                "file": "",
                "bytes": 0,
                "sha256": "",
                "error": (
                    "Install with: python -m pip install -U copernicusmarine"
                ),
            }
        )
        return rows, files, False

    import copernicusmarine

    selected_dataset = None
    selected_catalogue = None
    selected_max_dt = None

    for dataset_id in CMEMS_DATASET_CANDIDATES:
        try:
            cat = copernicusmarine.describe(
                dataset_id=dataset_id,
                disable_progress_bar=True,
            )

            dumped = to_jsonable(cat)
            ids = find_dataset_ids(dumped)

            if dataset_id not in ids and not ids:
                raise RuntimeError(
                    "Catalogue response contains no dataset_id."
                )

            candidates = recursively_find_time_max(dumped)

            max_dt = (
                max(x[0] for x in candidates)
                if candidates
                else None
            )

            rows.append(
                {
                    "probe_id": "CMEMS_DATASET_DESCRIBE",
                    "probe_type": "CATALOGUE",
                    "status": "PASS",
                    "dataset_id": dataset_id,
                    "available_datetime_utc": (
                        max_dt.isoformat()
                        if max_dt
                        else ""
                    ),
                    "sample_date_utc": "",
                    "file": "",
                    "bytes": 0,
                    "sha256": "",
                    "error": "",
                }
            )

            selected_dataset = dataset_id
            selected_catalogue = dumped
            selected_max_dt = max_dt
            break

        except Exception as exc:
            rows.append(
                {
                    "probe_id": "CMEMS_DATASET_DESCRIBE",
                    "probe_type": "CATALOGUE",
                    "status": "FAIL",
                    "dataset_id": dataset_id,
                    "available_datetime_utc": "",
                    "sample_date_utc": "",
                    "file": "",
                    "bytes": 0,
                    "sha256": "",
                    "error": short_error(exc),
                }
            )

    if selected_dataset is None:
        return rows, files, False

    # Save catalogue snapshot for reproducibility.
    catalog_p = out_dir / "cmems_selected_dataset_catalogue.json"
    catalog_p.write_text(
        json.dumps(
            selected_catalogue,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    files.append(catalog_p)

    # Start from metadata max date when available, otherwise yesterday.
    if selected_max_dt is not None:
        initial_date = selected_max_dt.date()
    else:
        initial_date = (
            datetime.now(timezone.utc).date()
            - timedelta(days=1)
        )

    subset_success = False

    for back_days in range(0, 11):
        sample_date = initial_date - timedelta(days=back_days)

        filename = (
            f"cmems_thetao_probe_{sample_date.isoformat()}.nc"
        )

        try:
            result = copernicusmarine.subset(
                dataset_id=selected_dataset,
                variables=[CMEMS_VARIABLE],
                start_datetime=f"{sample_date.isoformat()}T00:00:00",
                end_datetime=f"{sample_date.isoformat()}T23:59:59",
                minimum_longitude=CMEMS_BBOX["minimum_longitude"],
                maximum_longitude=CMEMS_BBOX["maximum_longitude"],
                minimum_latitude=CMEMS_BBOX["minimum_latitude"],
                maximum_latitude=CMEMS_BBOX["maximum_latitude"],
                minimum_depth=CMEMS_BBOX["minimum_depth"],
                maximum_depth=CMEMS_BBOX["maximum_depth"],
                output_directory=str(out_dir),
                output_filename=filename,
                overwrite=True,
                disable_progress_bar=True,
            )

            candidate_paths = [
                out_dir / filename,
            ]

            result_path = getattr(result, "file_path", None)

            if result_path:
                candidate_paths.insert(
                    0,
                    Path(str(result_path))
                )

            sample_p = next(
                (
                    p
                    for p in candidate_paths
                    if p.exists()
                    and p.stat().st_size > 0
                ),
                None,
            )

            if sample_p is None:
                raise RuntimeError(
                    "subset() returned but no non-empty output file was found."
                )

            rows.append(
                {
                    "probe_id": "CMEMS_THETAO_MICRO_SUBSET",
                    "probe_type": "DOWNLOAD",
                    "status": "PASS",
                    "dataset_id": selected_dataset,
                    "available_datetime_utc": (
                        selected_max_dt.isoformat()
                        if selected_max_dt
                        else ""
                    ),
                    "sample_date_utc": sample_date.isoformat(),
                    "file": str(sample_p),
                    "bytes": sample_p.stat().st_size,
                    "sha256": sha256(sample_p),
                    "error": "",
                }
            )

            files.append(sample_p)
            subset_success = True
            break

        except Exception as exc:
            rows.append(
                {
                    "probe_id": "CMEMS_THETAO_MICRO_SUBSET_ATTEMPT",
                    "probe_type": "DOWNLOAD_ATTEMPT",
                    "status": "FAIL",
                    "dataset_id": selected_dataset,
                    "available_datetime_utc": (
                        selected_max_dt.isoformat()
                        if selected_max_dt
                        else ""
                    ),
                    "sample_date_utc": sample_date.isoformat(),
                    "file": str(out_dir / filename),
                    "bytes": 0,
                    "sha256": "",
                    "error": short_error(exc),
                }
            )

    return rows, files, subset_success


def main():
    root = Path(__file__).resolve().parent

    out = (
        root
        / "nw_operational_provider_probe_current_v1_0"
    )
    ecmwf_dir = out / "ecmwf"
    cmems_dir = out / "copernicus_marine"

    ecmwf_dir.mkdir(parents=True, exist_ok=True)
    cmems_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 220)
    print("NW HYDROCLIMATE — CURRENT OPERATIONAL PROVIDER PROBE v1.0")
    print("=" * 220)

    print("\nPHASE 1/4 — environment/package check")
    start = time.time()

    package_rows = [
        {
            "package": "ecmwf-opendata",
            "import_name": "ecmwf.opendata",
            "available": module_available("ecmwf.opendata"),
        },
        {
            "package": "copernicusmarine",
            "import_name": "copernicusmarine",
            "available": module_available("copernicusmarine"),
        },
    ]

    progress(
        "PHASE 1/4",
        1,
        1,
        start,
        " | ".join(
            f"{x['package']}={x['available']}"
            for x in package_rows
        ),
    )

    print("\nPHASE 2/4 — ECMWF Open Data real access/download probe")
    start = time.time()

    ecmwf_rows, ecmwf_files, ecmwf_pass = run_ecmwf_probe(
        ecmwf_dir
    )

    progress(
        "PHASE 2/4",
        1,
        1,
        start,
        f"provider_pass={ecmwf_pass} | files={len(ecmwf_files)}",
    )

    print("\nPHASE 3/4 — Copernicus Marine real catalogue/download probe")
    start = time.time()

    cmems_rows, cmems_files, cmems_pass = run_cmems_probe(
        cmems_dir
    )

    progress(
        "PHASE 3/4",
        1,
        1,
        start,
        f"provider_pass={cmems_pass} | files={len(cmems_files)}",
    )

    print("\nPHASE 4/4 — write provider audit")
    start = time.time()

    ecmwf_df = pd.DataFrame(ecmwf_rows)
    cmems_df = pd.DataFrame(cmems_rows)

    ecmwf_p = out / "ecmwf_probe_registry_v1_0.csv"
    cmems_p = out / "copernicus_marine_probe_registry_v1_0.csv"
    summary_p = out / "provider_probe_summary_v1_0.csv"
    audit_json_p = out / "provider_probe_audit_v1_0.json"
    audit_txt_p = out / "provider_probe_audit_v1_0.txt"

    ecmwf_df.to_csv(
        ecmwf_p,
        index=False,
    )
    cmems_df.to_csv(
        cmems_p,
        index=False,
    )

    optional_available = []
    optional_unavailable = []

    for r in ecmwf_rows:
        if r["probe_type"] == "OPTIONAL_AVAILABILITY":
            param = r["request"].split("param=")[-1]
            if r["status"] == "PASS_AVAILABLE":
                optional_available.append(param)
            else:
                optional_unavailable.append(param)

    selected_dataset = ""

    passing_catalog = [
        r
        for r in cmems_rows
        if r["probe_id"] == "CMEMS_DATASET_DESCRIBE"
        and r["status"] == "PASS"
    ]

    if passing_catalog:
        selected_dataset = passing_catalog[0]["dataset_id"]

    summary = pd.DataFrame(
        [
            {
                "provider": "ECMWF_OPEN_DATA_IFS",
                "critical_probe_pass": ecmwf_pass,
                "selected_dataset_or_cycle": next(
                    (
                        str(r["available_datetime_utc"])
                        for r in ecmwf_rows
                        if r["probe_id"] == "ECMWF_LATEST"
                        and r["status"] == "PASS"
                    ),
                    "",
                ),
                "notes": (
                    "Safe documented core fields tested. "
                    f"Optional available={optional_available}; "
                    f"optional unavailable={optional_unavailable}"
                ),
            },
            {
                "provider": "COPERNICUS_MARINE_MEDSEA",
                "critical_probe_pass": cmems_pass,
                "selected_dataset_or_cycle": selected_dataset,
                "notes": (
                    "Catalogue + real micro-subset thetao tested."
                ),
            },
        ]
    )

    summary.to_csv(
        summary_p,
        index=False,
    )

    if ecmwf_pass and cmems_pass:
        overall = (
            "PASS_BOTH_PROVIDERS_REAL_ACCESS__FEATURE_ENGINE_NEXT"
        )
    elif not module_available("ecmwf.opendata"):
        overall = (
            "SETUP_REQUIRED_ECMWF_OPENDATA_PACKAGE"
        )
    elif not module_available("copernicusmarine"):
        overall = (
            "SETUP_REQUIRED_COPERNICUSMARINE_PACKAGE"
        )
    else:
        overall = (
            "PARTIAL_PROVIDER_PROBE__REVIEW_FAILURES"
        )

    audit = {
        "version": "1.0",
        "overall_status": overall,
        "probe_time_utc":
            datetime.now(timezone.utc).isoformat(),
        "ecmwf_critical_pass": ecmwf_pass,
        "cmems_critical_pass": cmems_pass,
        "ecmwf_optional_available":
            optional_available,
        "ecmwf_optional_unavailable_or_unsupported":
            optional_unavailable,
        "cmems_selected_dataset":
            selected_dataset,
        "model_run_performed":
            False,
        "feature_engine_run_performed":
            False,
        "next_step":
            (
                "If both providers PASS, build the operational raw-field "
                "acquisition/cache engine and quantify ERA5-vs-IFS proxy "
                "compatibility for the high-priority features before the "
                "first beta bulletin."
            ),
    }

    audit_json_p.write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    ecmwf_display_cols = [
        "probe_id",
        "probe_type",
        "status",
        "request",
        "available_datetime_utc",
        "bytes",
        "error",
    ]

    cmems_display_cols = [
        "probe_id",
        "probe_type",
        "status",
        "dataset_id",
        "available_datetime_utc",
        "sample_date_utc",
        "bytes",
        "error",
    ]

    lines = [
        "=" * 220,
        "NW HYDROCLIMATE — CURRENT OPERATIONAL PROVIDER PROBE v1.0",
        "=" * 220,
        f"OVERALL STATUS        : {overall}",
        f"ECMWF critical PASS   : {ecmwf_pass}",
        f"CMEMS critical PASS   : {cmems_pass}",
        f"CMEMS selected dataset: {selected_dataset or 'NONE'}",
        f"ECMWF optional available: {optional_available}",
        f"ECMWF optional unavailable/unsupported: {optional_unavailable}",
        "",
        "ECMWF PROBES",
        (
            ecmwf_df[
                [c for c in ecmwf_display_cols if c in ecmwf_df.columns]
            ].to_string(index=False)
            if len(ecmwf_df)
            else "NONE"
        ),
        "",
        "COPERNICUS MARINE PROBES",
        (
            cmems_df[
                [c for c in cmems_display_cols if c in cmems_df.columns]
            ].to_string(index=False)
            if len(cmems_df)
            else "NONE"
        ),
        "",
        "IMPORTANT",
        "A PASS here proves real provider access, not numerical equivalence with ERA5.",
        "Optional ECMWF parameter failures are expected to be diagnostic, not fatal.",
        "No model prediction is made by this script.",
        "",
        "NEXT STEP",
        (
            "If both critical providers PASS: build the operational raw-field "
            "cache and high-priority feature compatibility audit."
        ),
        "",
        f"ECMWF registry : {ecmwf_p}",
        f"CMEMS registry : {cmems_p}",
        f"Summary        : {summary_p}",
        f"Output         : {out}",
    ]

    audit_txt_p.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    progress(
        "PHASE 4/4",
        1,
        1,
        start,
        f"status={overall}",
    )

    print("\n" + "=" * 220)
    print("\n".join(lines[3:]))
    print("=" * 220)
    print(f"OVERALL STATUS : {overall}")
    print(f"Output         : {out}")
    print("=" * 220)


if __name__ == "__main__":
    main()
