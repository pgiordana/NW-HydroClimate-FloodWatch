#!/usr/bin/env python3
"""
Tanaro–Arroscia | GloFAS Historical Downloader v2.1
===================================================

Correzioni rispetto alla v2.0
-----------------------------
1. Endpoint corretto: CEMS Early Warning Data Store (EWDS)
       https://ewds.climate.copernicus.eu/api
2. timespan verificato sperimentalmente il 21/08/2026:
       time_mean
3. Salvataggio eseguito con la forma ufficiale:
       client.retrieve(DATASET, request, TARGET)
   invece di retrieve(...).download(...).
4. Nessun probe multiplo: il valore time_mean è ormai noto.
5. Test minimo: un solo giorno (03/10/2020).
6. ~/.cdsapirc NON viene modificato; EWDS usa ~/.ewdsapirc.

Uso
---
    python download_tanaro_arroscia_glofas_v2_1.py --test

Se il test termina con TEST OK:
    python download_tanaro_arroscia_glofas_v2_1.py

Prerequisito:
    pip install -U "cdsapi>=0.7.7"
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cdsapi


# ---------------------------------------------------------------------------
# CONFIGURAZIONE
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
STUDY_ROOT = ROOT / "tanaro_arroscia"
OUT = STUDY_ROOT / "glofas_historical"
MANIFEST = OUT / "download_manifest.jsonl"

EWDS_URL = "https://ewds.climate.copernicus.eu/api"
EWDS_CONFIG = Path.home() / ".ewdsapirc"

DATASET = "cems-glofas-historical"
SYSTEM_VERSION = "version_4_0"
HYDROLOGICAL_MODEL = "lisflood"
PRODUCT_TYPE = "consolidated"

# Verificato direttamente contro EWDS il 21/08/2026:
TIMESPAN = "time_mean"

VARIABLE = "average_river_discharge_in_the_last_24_hours"

DEFAULT_START_YEAR = 1987
DEFAULT_END_YEAR = 2025
MONTHS = [9, 10, 11, 12]

# [North, West, South, East]
AREA = [44.50, 7.00, 43.45, 8.75]

# GloFAS è giornaliero. GRIB2 è il formato stabile/non sperimentale.
DATA_FORMAT = "grib2"
DOWNLOAD_FORMAT = "unarchived"

MIN_VALID_BYTES = 500
MIN_FREE_GB = 2.0


# ---------------------------------------------------------------------------
# ARGOMENTI / CARTELLE
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Scarica GloFAS Historical da EWDS per Tanaro–Arroscia."
    )
    p.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    p.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    p.add_argument(
        "--test",
        action="store_true",
        help="Scarica solo il 03/10/2020 per verificare API e salvataggio.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Riscarica anche file già presenti e validi.",
    )
    return p.parse_args()


def ensure_dirs():
    for p in [
        STUDY_ROOT,
        OUT,
        STUDY_ROOT / "observations" / "arpa_piemonte",
        STUDY_ROOT / "observations" / "arpal_omirl",
        STUDY_ROOT / "basins",
        STUDY_ROOT / "terrain",
        STUDY_ROOT / "thresholds",
        STUDY_ROOT / "events",
        STUDY_ROOT / "model",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def check_disk():
    free_gb = shutil.disk_usage(STUDY_ROOT).free / 1024**3
    print(f"Spazio libero sul disco: {free_gb:.1f} GB")
    if free_gb < MIN_FREE_GB:
        raise RuntimeError(
            f"Spazio libero insufficiente: {free_gb:.1f} GB."
        )


def file_ok(path: Path):
    return path.exists() and path.is_file() and path.stat().st_size >= MIN_VALID_BYTES


def log_manifest(**record):
    record["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    with MANIFEST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CREDENZIALI EWDS
# ---------------------------------------------------------------------------

def read_ewds_key():
    env_key = os.environ.get("EWDS_API_KEY", "").strip()
    if env_key:
        print("Credenziali EWDS: variabile EWDS_API_KEY")
        return env_key

    if not EWDS_CONFIG.exists():
        raise RuntimeError(
            f"Non trovo {EWDS_CONFIG}.\n"
            "La v2.0 dovrebbe averlo già creato. In alternativa crea il file con:\n"
            f"url: {EWDS_URL}\n"
            "key: <PERSONAL-ACCESS-TOKEN>"
        )

    key = None
    for raw in EWDS_CONFIG.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("key:"):
            key = line.split(":", 1)[1].strip()
            break

    if not key:
        raise RuntimeError(f"Il file {EWDS_CONFIG} non contiene una riga 'key:' valida.")

    print(f"Credenziali EWDS: {EWDS_CONFIG}")
    return key


def make_client():
    key = read_ewds_key()
    print(f"Endpoint API: {EWDS_URL}")
    return cdsapi.Client(url=EWDS_URL, key=key)


# ---------------------------------------------------------------------------
# REQUEST
# ---------------------------------------------------------------------------

def make_request(year: int, month: int, days: list[str]):
    return {
        "system_version": [SYSTEM_VERSION],
        "hydrological_model": [HYDROLOGICAL_MODEL],
        "product_type": [PRODUCT_TYPE],
        "timespan": [TIMESPAN],
        "variable": [VARIABLE],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": days,
        "area": AREA,
        "data_format": DATA_FORMAT,
        "download_format": DOWNLOAD_FORMAT,
    }


def retrieve_to_target(client, request: dict, target: Path):
    """
    Forma ufficiale CDSAPI/EWDS:
        client.retrieve(dataset, request, target)

    Il target viene affidato direttamente a cdsapi.
    """
    if target.exists():
        target.unlink()

    print(f"Target locale: {target}")

    try:
        client.retrieve(DATASET, request, str(target))
    except Exception:
        # Non lasciamo un frammento minuscolo scambiabile per file valido.
        if target.exists() and target.stat().st_size < MIN_VALID_BYTES:
            target.unlink()
        raise

    if not target.exists():
        raise RuntimeError(
            "EWDS ha completato la richiesta ma cdsapi non ha creato il target.\n"
            f"Target atteso: {target}"
        )

    size = target.stat().st_size
    if size < MIN_VALID_BYTES:
        raise RuntimeError(
            f"Il target esiste ma è troppo piccolo ({size} byte): {target}"
        )

    return size


# ---------------------------------------------------------------------------
# TEST
# ---------------------------------------------------------------------------

def run_test(client):
    test_dir = OUT / "_probe"
    test_dir.mkdir(exist_ok=True)

    target = test_dir / "glofas_TEST_2020-10-03.grib"
    request = make_request(2020, 10, ["03"])

    print()
    print("=" * 88)
    print("TEST EWDS / GLOFAS")
    print("Data      : 03/10/2020")
    print(f"Timespan  : {TIMESPAN}")
    print(f"Variabile : {VARIABLE}")
    print(f"Area      : {AREA}")
    print("=" * 88)

    t0 = time.time()
    size = retrieve_to_target(client, request, target)
    elapsed = time.time() - t0

    print()
    print("=" * 88)
    print("TEST OK")
    print(f"File creato : {target}")
    print(f"Dimensione  : {size:,} byte")
    print(f"Tempo       : {elapsed:.1f} s")
    print("=" * 88)

    log_manifest(
        status="test_ok",
        date="2020-10-03",
        dataset=DATASET,
        system_version=SYSTEM_VERSION,
        timespan=TIMESPAN,
        variable=VARIABLE,
        area=AREA,
        path=str(target),
        bytes=size,
        seconds=round(elapsed, 1),
    )


# ---------------------------------------------------------------------------
# DOWNLOAD STORICO
# ---------------------------------------------------------------------------

def month_days(year: int, month: int):
    n = calendar.monthrange(year, month)[1]
    return [f"{d:02d}" for d in range(1, n + 1)]


def download_month(client, year: int, month: int, force=False):
    target = OUT / f"glofas_discharge_{year}_{month:02d}.grib"

    if file_ok(target) and not force:
        print(
            f"SKIP {year}-{month:02d}: già presente "
            f"({target.stat().st_size/1e6:.3f} MB)"
        )
        return "skipped"

    request = make_request(year, month, month_days(year, month))

    print()
    print("-" * 88)
    print(f"DOWNLOAD {year}-{month:02d}")
    print("-" * 88)

    t0 = time.time()

    try:
        size = retrieve_to_target(client, request, target)
    except Exception as exc:
        log_manifest(
            status="error",
            year=year,
            month=month,
            dataset=DATASET,
            system_version=SYSTEM_VERSION,
            timespan=TIMESPAN,
            variable=VARIABLE,
            area=AREA,
            error=str(exc),
        )
        raise

    elapsed = time.time() - t0

    print(
        f"OK {year}-{month:02d}: "
        f"{size/1e6:.3f} MB in {elapsed/60:.2f} min"
    )

    log_manifest(
        status="ok",
        year=year,
        month=month,
        dataset=DATASET,
        system_version=SYSTEM_VERSION,
        timespan=TIMESPAN,
        variable=VARIABLE,
        area=AREA,
        path=str(target),
        bytes=size,
        seconds=round(elapsed, 1),
    )

    return "ok"


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.start_year > args.end_year:
        raise ValueError("--start-year deve essere <= --end-year")

    ensure_dirs()
    check_disk()

    print("=" * 88)
    print("TANARO–ARROSCIA | GLOFAS HISTORICAL DOWNLOADER v2.1")
    print(f"Dataset : {DATASET}")
    print("Store   : EWDS")
    print(f"Endpoint: {EWDS_URL}")
    print(f"Version : {SYSTEM_VERSION}")
    print(f"Timespan: {TIMESPAN}  [VERIFICATO]")
    print(f"Variable: {VARIABLE}")
    print(f"Area    : {AREA}")
    print("=" * 88)

    client = make_client()

    if args.test:
        run_test(client)
        print()
        print("Se vedi TEST OK, il prossimo comando è:")
        print("python download_tanaro_arroscia_glofas_v2_1.py")
        return

    total = (args.end_year - args.start_year + 1) * len(MONTHS)
    counter = 0

    for year in range(args.start_year, args.end_year + 1):
        for month in MONTHS:
            download_month(client, year, month, force=args.force)
            counter += 1
            print(f"PROGRESSO: {counter}/{total} mesi")

        check_disk()

    print()
    print("=" * 88)
    print("DOWNLOAD GLOFAS COMPLETATO")
    print(f"Output  : {OUT}")
    print(f"Manifest: {MANIFEST}")
    print("=" * 88)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrotto dall'utente. I file completi restano validi.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nFATAL: {exc}", file=sys.stderr)
        sys.exit(1)
