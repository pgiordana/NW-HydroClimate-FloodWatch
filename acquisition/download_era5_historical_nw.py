#!/usr/bin/env python3
"""
Download ERA5 historical predictors for the NW Italy probabilistic
hydro-meteorological model.

Default period: Sep-Dec 1987-2025 (39 autumn seasons).
The start date is aligned with the Mediterranean Sea Physics Reanalysis,
which begins in 1987.

Design:
- broad SOURCE domain for atmospheric transport/source-receptor analysis;
- compact TARGET domain for precipitation and land state;
- 3-hourly synoptic/pressure predictors;
- hourly precipitation;
- daily soil/snow state;
- one file per month/family for robust resume and CDS cost limits.

The script is restart-safe: existing non-trivial files are skipped.
"""

from __future__ import annotations

import argparse
import calendar
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cdsapi

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "era5_historical_nw"
MANIFEST = OUT / "download_manifest.jsonl"

DEFAULT_START_YEAR = 1987
DEFAULT_END_YEAR = 2025
MONTHS = [9, 10, 11, 12]

# Broad atmospheric source corridor:
# N, W, S, E. Includes W Mediterranean / Balearic-Gulf of Lion sector,
# Ligurian/Tyrrhenian sector and all NW Italy receptors.
SOURCE_AREA = [48.0, -4.0, 36.0, 13.0]

# Compact target domain containing Piemonte, Valle d'Aosta and Liguria.
# Used only for hourly precipitation and land-state variables.
TARGET_AREA = [47.0, 6.0, 43.5, 11.0]

TIMES_3H = [f"{h:02d}:00" for h in range(0, 24, 3)]
TIMES_HOURLY = [f"{h:02d}:00" for h in range(24)]
TIMES_DAILY = ["00:00"]

# Native ERA5 vertically integrated moisture transport + broad thermodynamic state.
SOURCE_SINGLE_3H = [
    "vertical_integral_of_eastward_water_vapour_flux",
    "vertical_integral_of_northward_water_vapour_flux",
    "total_column_water_vapour",
    "convective_available_potential_energy",
    "mean_sea_level_pressure",
]

# Only the pressure-level fields needed for circulation/orographic diagnostics.
PRESSURE_3H = [
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "temperature",
]
PRESSURE_LEVELS = ["925", "850", "700"]

# Hourly precipitation is kept hourly so 1/3/6/12/24/48 h accumulations can
# later be reconstructed without losing two out of every three hours.
TARGET_PRECIP_HOURLY = ["total_precipitation"]

# Antecedent land state: one daily snapshot is sufficient at this stage.
TARGET_STATE_DAILY = [
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "snow_depth",
]

MIN_VALID_BYTES = 10_000
MAX_ATTEMPTS = 8
MIN_FREE_GB = 8.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    p.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    p.add_argument(
        "--only-family",
        choices=["source3h", "pressure3h", "precip1h", "state1d"],
        help="Optional: download only one family.",
    )
    p.add_argument(
        "--test",
        action="store_true",
        help="Download only October 2000 to test the pipeline.",
    )
    return p.parse_args()


def days_for(year: int, month: int):
    n = calendar.monthrange(year, month)[1]
    return [f"{d:02d}" for d in range(1, n + 1)]


def file_ok(path: Path) -> bool:
    return path.exists() and path.stat().st_size >= MIN_VALID_BYTES


def log_manifest(**record):
    OUT.mkdir(parents=True, exist_ok=True)
    record["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    with MANIFEST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def check_disk():
    OUT.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(OUT)
    free_gb = usage.free / 1024**3
    print(f"Spazio libero sul disco: {free_gb:.1f} GB")
    if free_gb < MIN_FREE_GB:
        raise RuntimeError(
            f"Spazio libero insufficiente ({free_gb:.1f} GB). "
            f"Servono almeno {MIN_FREE_GB:.0f} GB liberi per proseguire in sicurezza."
        )


def retrieve(client, dataset: str, request: dict, target: Path, family: str,
             year: int, month: int):
    target.parent.mkdir(parents=True, exist_ok=True)

    if file_ok(target):
        size_mb = target.stat().st_size / 1e6
        print(f"  SKIP {family}: già presente {target.name} ({size_mb:.1f} MB)")
        return

    # Remove tiny/failed remnants.
    if target.exists():
        target.unlink()

    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        t0 = time.time()
        try:
            print(f"  {family}: richiesta {attempt}/{MAX_ATTEMPTS} -> {target.name}")
            client.retrieve(dataset, request, str(target))

            if not file_ok(target):
                raise RuntimeError(
                    f"File scaricato troppo piccolo o assente: {target}"
                )

            elapsed = time.time() - t0
            size = target.stat().st_size
            print(
                f"  OK {family}: {size/1e6:.1f} MB "
                f"in {elapsed/60:.1f} min"
            )
            log_manifest(
                status="ok",
                family=family,
                dataset=dataset,
                year=year,
                month=month,
                path=str(target),
                bytes=size,
                seconds=round(elapsed, 1),
            )
            return

        except KeyboardInterrupt:
            print("\nInterrotto dall'utente. I file già completati restano validi.")
            raise

        except Exception as exc:
            last_error = exc
            elapsed = time.time() - t0
            msg = str(exc)
            print(f"  ERRORE {family}: {msg}")
            log_manifest(
                status="error",
                family=family,
                dataset=dataset,
                year=year,
                month=month,
                path=str(target),
                seconds=round(elapsed, 1),
                error=msg,
            )
            if target.exists() and target.stat().st_size < MIN_VALID_BYTES:
                target.unlink()

            if attempt < MAX_ATTEMPTS:
                wait = min(60 * (2 ** (attempt - 1)), 600)
                print(f"  Attendo {wait} s e riprovo...")
                time.sleep(wait)

    raise RuntimeError(
        f"Download fallito dopo {MAX_ATTEMPTS} tentativi: "
        f"{family} {year}-{month:02d}\nUltimo errore: {last_error}"
    )


def common_request(year, month, times, area):
    return {
        "product_type": ["reanalysis"],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": days_for(year, month),
        "time": times,
        "area": area,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def download_month(client, year: int, month: int, only_family=None):
    print("=" * 88)
    print(f"{year}-{month:02d}")

    if only_family in (None, "source3h"):
        req = common_request(year, month, TIMES_3H, SOURCE_AREA)
        req["variable"] = SOURCE_SINGLE_3H
        target = (
            OUT / "source_single_3h" / str(year)
            / f"era5_source_single_3h_{year}{month:02d}.nc"
        )
        retrieve(
            client, "reanalysis-era5-single-levels",
            req, target, "source3h", year, month
        )

    if only_family in (None, "pressure3h"):
        req = common_request(year, month, TIMES_3H, SOURCE_AREA)
        req["variable"] = PRESSURE_3H
        req["pressure_level"] = PRESSURE_LEVELS
        target = (
            OUT / "pressure_3h" / str(year)
            / f"era5_pressure_3h_{year}{month:02d}.nc"
        )
        retrieve(
            client, "reanalysis-era5-pressure-levels",
            req, target, "pressure3h", year, month
        )

    if only_family in (None, "precip1h"):
        req = common_request(year, month, TIMES_HOURLY, TARGET_AREA)
        req["variable"] = TARGET_PRECIP_HOURLY
        target = (
            OUT / "target_precip_hourly" / str(year)
            / f"era5_target_precip_1h_{year}{month:02d}.nc"
        )
        retrieve(
            client, "reanalysis-era5-single-levels",
            req, target, "precip1h", year, month
        )

    if only_family in (None, "state1d"):
        req = common_request(year, month, TIMES_DAILY, TARGET_AREA)
        req["variable"] = TARGET_STATE_DAILY
        target = (
            OUT / "target_state_daily" / str(year)
            / f"era5_target_state_1d_{year}{month:02d}.nc"
        )
        retrieve(
            client, "reanalysis-era5-single-levels",
            req, target, "state1d", year, month
        )


def print_plan(start_year, end_year, test, only_family):
    years = [2000] if test else list(range(start_year, end_year + 1))
    months = [10] if test else MONTHS

    n_months = len(years) * len(months)
    n_families = 1 if only_family else 4
    n_requests = n_months * n_families

    print("ERA5 HISTORICAL NW — DOWNLOAD PLAN")
    print(f"Periodo: {'2000-10 TEST' if test else f'{start_year}-09 -> {end_year}-12, Sep-Dec'}")
    print(f"Stagioni: {len(years) if not test else 1}")
    print(f"Mesi totali: {n_months}")
    print(f"Famiglie: {only_family or '4 (source3h, pressure3h, precip1h, state1d)'}")
    print(f"Richieste CDS previste: {n_requests}")
    print(f"SOURCE area [N,W,S,E]: {SOURCE_AREA}")
    print(f"TARGET area [N,W,S,E]: {TARGET_AREA}")
    print("Temporalità: atmosfera 3h; precipitazione 1h; suolo/neve 1 snapshot/giorno.")
    print("Resume: automatico; i file già completati vengono saltati.")
    print()


def main():
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("--start-year deve essere <= --end-year")

    check_disk()
    print_plan(args.start_year, args.end_year, args.test, args.only_family)

    if args.test:
        years = [2000]
        months = [10]
    else:
        years = range(args.start_year, args.end_year + 1)
        months = MONTHS

    client = cdsapi.Client()

    completed = 0
    total = len(list(years)) * len(months)

    for year in years:
        for month in months:
            download_month(client, year, month, args.only_family)
            completed += 1
            print(f"PROGRESSO MESI: {completed}/{total}")
            check_disk()

    print()
    print("ERA5 HISTORICAL DOWNLOAD: COMPLETE")
    print(f"Output: {OUT}")
    print(f"Manifest: {MANIFEST}")
    print("Il prossimo script trasformerà questi NetCDF in feature giornaliere × bacino.")


if __name__ == "__main__":
    main()
