#!/usr/bin/env python3
from pathlib import Path
import zipfile
import shutil
import numpy as np
import pandas as pd
import xarray as xr
import cdsapi

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "era5_analysis" / "event_2000_10_nw"
OUT.mkdir(parents=True, exist_ok=True)

REF = OUT / "ivt_fields_20001010_18.nc"
RAW = OUT / "era5_integrated_water_vapour_flux_20001010_18.nc"
REPORT = OUT / "ivt_validation_against_era5_integrated.txt"
HOURLY = OUT / "ivt_validation_hourly.csv"

AREA = [48, 3, 36, 13]
TIMES = [f"{h:02d}:00" for h in range(24)]
DAYS = [f"{d:02d}" for d in range(10, 19)]

def normalize(ds):
    ren = {}
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ren["valid_time"] = "time"
    if "lat" in ds.coords and "latitude" not in ds.coords:
        ren["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords:
        ren["lon"] = "longitude"
    if ren:
        ds = ds.rename(ren)
    for c in ("time", "latitude", "longitude"):
        if c in ds.coords:
            ds = ds.sortby(c)
    return ds

def open_maybe_zip(path):
    if not zipfile.is_zipfile(path):
        return normalize(xr.open_dataset(path))
    dest = OUT / "integrated_ivt_unpacked"
    dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".nc")]
        for m in members:
            target = dest / Path(m).name
            if not target.exists():
                with zf.open(m) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    parts = [normalize(xr.open_dataset(p)) for p in sorted(dest.glob("*.nc"))]
    return normalize(xr.merge(parts, compat="override", join="outer"))

def pick(ds, names):
    for n in names:
        if n in ds:
            return ds[n]
    raise KeyError(f"Variabile non trovata fra {names}. Disponibili: {list(ds.data_vars)}")

def from_direction(u, v):
    return (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0

def angular_difference(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)

if not REF.exists():
    raise FileNotFoundError(
        f"Manca {REF}. Esegui prima analyse_era5_event_2000_nw_v2.py"
    )

if not RAW.exists() or RAW.stat().st_size < 10000:
    print("Scarico i due flussi di vapore verticalmente integrati ERA5...")
    req = {
        "product_type": ["reanalysis"],
        "variable": [
            "vertical_integral_of_eastward_water_vapour_flux",
            "vertical_integral_of_northward_water_vapour_flux",
        ],
        "year": ["2000"],
        "month": ["10"],
        "day": DAYS,
        "time": TIMES,
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    cdsapi.Client().retrieve("reanalysis-era5-single-levels", req, str(RAW))
    print(f"OK: {RAW}")
else:
    print(f"SKIP download, già presente: {RAW}")

print("Apro IVT ricostruito e flussi integrati ufficiali ERA5...")
ref = normalize(xr.open_dataset(REF))
era = open_maybe_zip(RAW)

east = pick(era, ["viwve", "vertical_integral_of_eastward_water_vapour_flux"])
north = pick(era, ["viwvn", "vertical_integral_of_northward_water_vapour_flux"])

east = east.transpose("time", "latitude", "longitude")
north = north.transpose("time", "latitude", "longitude")

# Align exact common coordinates/times.
ref, era2 = xr.align(ref, xr.Dataset({"east": east, "north": north}), join="inner")

rx = ref["ivtx_300_surface"].values.astype(float)
ry = ref["ivty_300_surface"].values.astype(float)
rm = np.sqrt(rx**2 + ry**2)
rd = from_direction(rx, ry)

ex = era2["east"].values.astype(float)
ey = era2["north"].values.astype(float)
em = np.sqrt(ex**2 + ey**2)
ed = from_direction(ex, ey)

mask = np.isfinite(rm) & np.isfinite(em)
if mask.sum() == 0:
    raise RuntimeError("Nessun punto comune valido.")

rflat = rm[mask]
eflat = em[mask]
corr = float(np.corrcoef(rflat, eflat)[0, 1])
bias = float(np.mean(rflat - eflat))
mae = float(np.mean(np.abs(rflat - eflat)))
rmse = float(np.sqrt(np.mean((rflat - eflat)**2)))
ratio = float(np.median(np.divide(
    rflat, eflat, out=np.full_like(rflat, np.nan), where=eflat > 1.0
)))
adiff = angular_difference(rd[mask], ed[mask])
dir_med = float(np.nanmedian(adiff))
dir_p90 = float(np.nanpercentile(adiff, 90))

# Domain mean/max by hour for easy inspection.
times = pd.to_datetime(ref.time.values)
rows = []
for i, ts in enumerate(times):
    m = np.isfinite(rm[i]) & np.isfinite(em[i])
    rows.append({
        "time": ts.isoformat(),
        "partial_ivt_domain_mean": float(np.nanmean(rm[i][m])),
        "era5_integrated_ivt_domain_mean": float(np.nanmean(em[i][m])),
        "partial_ivt_domain_max": float(np.nanmax(rm[i][m])),
        "era5_integrated_ivt_domain_max": float(np.nanmax(em[i][m])),
        "median_direction_difference_deg": float(
            np.nanmedian(angular_difference(rd[i][m], ed[i][m]))
        ),
    })
pd.DataFrame(rows).to_csv(HOURLY, index=False)

lines = [
    "IVT VALIDATION — OCTOBER 2000",
    "",
    "Reference A:",
    "- locally reconstructed partial-column IVT from 300 hPa to local surface",
    "",
    "Reference B:",
    "- ERA5 native vertically integrated eastward/northward water-vapour flux",
    "- units kg m-1 s-1",
    "",
    f"Common valid grid-time points: {int(mask.sum())}",
    f"Magnitude Pearson correlation: {corr:.6f}",
    f"Mean partial-minus-native bias: {bias:+.3f} kg m-1 s-1",
    f"Magnitude MAE: {mae:.3f} kg m-1 s-1",
    f"Magnitude RMSE: {rmse:.3f} kg m-1 s-1",
    f"Median partial/native magnitude ratio (native >1): {ratio:.4f}",
    f"Median vector-direction difference: {dir_med:.3f} deg",
    f"90th percentile direction difference: {dir_p90:.3f} deg",
    "",
    "INTERPRETATION:",
    "- The native ERA5 integrated flux is preferable for the long historical model.",
    "- It avoids downloading ~20 pressure levels solely to reconstruct IVT.",
    "- Pressure-level winds/humidity can then be retained only at selected levels for circulation diagnostics.",
    "- This comparison checks whether the existing 300-hPa-to-surface reconstruction is consistent enough to validate the event pipeline.",
]
REPORT.write_text("\n".join(lines), encoding="utf-8")

print("\nVALIDAZIONE IVT COMPLETATA")
print(f"Correlation: {corr:.4f}")
print(f"Median direction difference: {dir_med:.2f} deg")
print(f"Report: {REPORT}")
print(f"Hourly: {HOURLY}")
