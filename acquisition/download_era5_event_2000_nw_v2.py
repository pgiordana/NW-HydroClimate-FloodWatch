#!/usr/bin/env python3
from pathlib import Path
import sys
import time

try:
    import cdsapi
except ImportError:
    print('ERRORE: cdsapi non installato.')
    print('Esegui: pip install "cdsapi>=0.7.7"')
    sys.exit(2)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "era5" / "event_2000_10_nw_daily"
P_OUT = OUT / "pressure"
S_OUT = OUT / "single"
P_OUT.mkdir(parents=True, exist_ok=True)
S_OUT.mkdir(parents=True, exist_ok=True)

client = cdsapi.Client()

times = [f"{h:02d}:00" for h in range(24)]
days = [f"{d:02d}" for d in range(10, 19)]

pressure_levels = [
    "1000","975","950","925","900","875","850","825","800","775",
    "750","700","650","600","550","500","450","400","350","300"
]

area = [48, 3, 36, 13]  # North, West, South, East

pressure_variables = [
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "temperature",
    "geopotential",
]

single_variables = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "surface_pressure",
    "total_column_water_vapour",
    "convective_available_potential_energy",
    "total_precipitation",
    "evaporation",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
]

def retrieve_with_retry(dataset, request, target, attempts=3):
    target = Path(target)
    if target.exists() and target.stat().st_size > 10_000:
        print(f"SKIP già presente: {target.name} ({target.stat().st_size/1e6:.1f} MB)")
        return

    last = None
    for n in range(1, attempts + 1):
        try:
            print(f"  richiesta {n}/{attempts}: {target.name}")
            client.retrieve(dataset, request, str(target))
            print(f"  OK: {target.name} ({target.stat().st_size/1e6:.1f} MB)")
            return
        except Exception as e:
            last = e
            print(f"  ERRORE tentativo {n}: {e}")
            if n < attempts:
                print("  attendo 20 s e riprovo...")
                time.sleep(20)

    raise last

print("ERA5 EVENTO 10–18 OTTOBRE 2000 — DOWNLOAD A BLOCCHI GIORNALIERI")
print(f"Output: {OUT}")
print("Dominio: 48N–36N, 3E–13E")
print()

for idx, day in enumerate(days, 1):
    print("=" * 80)
    print(f"GIORNO {idx}/9: 2000-10-{day}")

    pressure_target = P_OUT / f"era5_pressure_200010{day}_nw.nc"
    pressure_request = {
        "product_type": ["reanalysis"],
        "variable": pressure_variables,
        "year": ["2000"],
        "month": ["10"],
        "day": [day],
        "time": times,
        "pressure_level": pressure_levels,
        "area": area,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    print("Pressure levels...")
    retrieve_with_retry(
        "reanalysis-era5-pressure-levels",
        pressure_request,
        pressure_target,
    )

    single_target = S_OUT / f"era5_single_200010{day}_nw.nc"
    single_request = {
        "product_type": ["reanalysis"],
        "variable": single_variables,
        "year": ["2000"],
        "month": ["10"],
        "day": [day],
        "time": times,
        "area": area,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    print("Single levels...")
    retrieve_with_retry(
        "reanalysis-era5-single-levels",
        single_request,
        single_target,
    )

print()
print("ERA5 DAILY DOWNLOAD: COMPLETE")
print(f"Pressure files: {len(list(P_OUT.glob('*.nc')))}")
print(f"Single files:   {len(list(S_OUT.glob('*.nc')))}")
print("I file restano separati per giorno: il prossimo script li leggerà e concatenerà localmente.")
