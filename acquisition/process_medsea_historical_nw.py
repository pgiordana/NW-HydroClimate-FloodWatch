#!/usr/bin/env python3
"""
Elaborazione Copernicus Marine storico NW (Sep-Dic 1987-2025)

Input atteso:
  medsea_historical_nw/
    daily_sst/medsea_daily_sst_YYYY_SepDec.nc
    monthly_temp_0_110m/medsea_monthly_temp_0_110m_YYYY_SepDec.nc
    static/medsea_my_bathy_mask_source_domain.nc
    static/medsea_my_grid_metrics_0_110m_source_domain.nc

Output:
  medsea_historical_analysis/
    climatology/
      sst_daily_climatology_1991_2020_SepDec.nc
      ohc_monthly_climatology_1991_2020_SepDec.nc
    daily_sst_anomaly/
      medsea_sst_anomaly_YYYY_SepDec.nc
    monthly_ohc_anomaly/
      medsea_ohc_anomaly_YYYY_SepDec.nc
    medsea_daily_sst_summary_1987_2025.csv
    medsea_monthly_ohc_summary_1987_2025.csv
    medsea_historical_analysis_qc.txt

Metodo:
- Climatologia SST: media per giorno calendario (MM-DD), 1991-2020.
- OHC: rho*cp*integrale verticale T dz, 0-100 m per unita' di superficie.
- Climatologia OHC: media mensile Sep/Oct/Nov/Dec, 1991-2020.
- Anomalia OHC: differenza rispetto alla climatologia del mese corrispondente.
- Geometria verticale: e3t + mask; ogni colonna viene troncata esattamente a 100 m.
- Le energie totali nel CSV sono integrali spaziali dell'anomalia OHC [J].
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent
INROOT = ROOT / "medsea_historical_nw"
OUTROOT = ROOT / "medsea_historical_analysis"
CLIM_DIR = OUTROOT / "climatology"
SST_OUT = OUTROOT / "daily_sst_anomaly"
OHC_OUT = OUTROOT / "monthly_ohc_anomaly"
TMP_DIR = OUTROOT / "_tmp"

BASE_START = 1991
BASE_END = 2020
YEARS = list(range(1987, 2026))
RHO = 1025.0       # kg m-3
CP = 3990.0        # J kg-1 K-1
TARGET_DEPTH = 100.0

SST_CLIM_FILE = CLIM_DIR / "sst_daily_climatology_1991_2020_SepDec.nc"
OHC_CLIM_FILE = CLIM_DIR / "ohc_monthly_climatology_1991_2020_SepDec.nc"

ENC_FLOAT = {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--test",
        action="store_true",
        help="Calcola le climatologie complete, ma produce le anomalie solo per il 2000.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Riscrive anche i file annuali gia' presenti.",
    )
    return p.parse_args()


def ensure_dirs():
    for d in [OUTROOT, CLIM_DIR, SST_OUT, OHC_OUT, TMP_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def daily_file(year):
    return INROOT / "daily_sst" / f"medsea_daily_sst_{year}_SepDec.nc"


def monthly_file(year):
    return INROOT / "monthly_temp_0_110m" / f"medsea_monthly_temp_0_110m_{year}_SepDec.nc"


def check_inputs():
    missing = []
    for y in YEARS:
        if not daily_file(y).exists():
            missing.append(str(daily_file(y)))
        if not monthly_file(y).exists():
            missing.append(str(monthly_file(y)))
    static1 = INROOT / "static" / "medsea_my_bathy_mask_source_domain.nc"
    static2 = INROOT / "static" / "medsea_my_grid_metrics_0_110m_source_domain.nc"
    for p in [static1, static2]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError("Input mancanti:\n" + "\n".join(missing[:20]))
    return static1, static2


def load_static(static_bathy, static_metrics):
    with xr.open_dataset(static_bathy) as dsb:
        lat = dsb["latitude"].values.astype(np.float64)
        lon = dsb["longitude"].values.astype(np.float64)
        deptho = dsb["deptho"].values.astype(np.float64)
        mask_da = dsb["mask"].isel(depth=slice(0, 25))
        mask = mask_da.values.astype(np.float64)
        static_depth = mask_da["depth"].values.astype(np.float64)

    with xr.open_dataset(static_metrics) as dsm:
        depth_m = dsm["depth"].values.astype(np.float64)
        if len(depth_m) != 25:
            raise ValueError(f"Attesi 25 livelli statici, trovati {len(depth_m)}")
        if not np.allclose(depth_m, static_depth, atol=1e-5):
            raise ValueError("Profondita' mask e metriche non coincidenti.")

        e3t = dsm["e3t"].values.astype(np.float64)

        e1 = dsm["e1t"]
        e2 = dsm["e2t"]
        if "depth" in e1.dims:
            e1 = e1.isel(depth=0)
        if "depth" in e2.dims:
            e2 = e2.isel(depth=0)
        area = (e1.values.astype(np.float64) * e2.values.astype(np.float64))

    # e3t puo' avere dimensioni depth,lat,lon oppure varianti compatibili.
    if e3t.shape != mask.shape:
        raise ValueError(f"Shape e3t {e3t.shape} != mask {mask.shape}")

    sea3d = np.isfinite(mask) & (mask > 0.5) & np.isfinite(e3t) & (e3t > 0)
    h = np.where(sea3d, e3t, 0.0)

    # Spessore effettivo: integra dall'alto e tronca la colonna a 100 m.
    eff = np.zeros_like(h, dtype=np.float64)
    remaining = np.full(h.shape[1:], TARGET_DEPTH, dtype=np.float64)
    for k in range(h.shape[0]):
        hk = np.minimum(h[k], remaining)
        hk = np.where(sea3d[k], hk, 0.0)
        eff[k] = hk
        remaining = np.maximum(remaining - hk, 0.0)

    eff_total = eff.sum(axis=0)
    target_total = np.where(
        np.isfinite(deptho) & (deptho > 0),
        np.minimum(deptho, TARGET_DEPTH),
        np.nan,
    )
    valid_col = np.isfinite(target_total) & (target_total > 0)

    closure_err = np.where(valid_col, eff_total - target_total, np.nan)

    # Area valida superficiale.
    sea_surface = valid_col & np.isfinite(area) & (area > 0)
    area = np.where(sea_surface, area, np.nan)

    qc = {
        "grid_nlat": len(lat),
        "grid_nlon": len(lon),
        "depth_levels": len(depth_m),
        "depth_min_m": float(depth_m.min()),
        "depth_max_m": float(depth_m.max()),
        "vertical_closure_mean_abs_m": float(np.nanmean(np.abs(closure_err))),
        "vertical_closure_max_abs_m": float(np.nanmax(np.abs(closure_err))),
        "sea_area_km2": float(np.nansum(area) / 1e6),
    }
    return lat, lon, depth_m, deptho, area, eff, eff_total, qc


def file_good(path: Path, var: str, ntime: int):
    if not path.exists() or path.stat().st_size < 10_000:
        return False
    try:
        with xr.open_dataset(path) as ds:
            return var in ds and ds.sizes.get("time") == ntime
    except Exception:
        return False


def build_sst_climatology(lat, lon):
    ref_dates = pd.date_range("2001-09-01", "2001-12-31", freq="D")
    md_to_idx = {d.strftime("%m-%d"): i for i, d in enumerate(ref_dates)}
    shape = (len(ref_dates), len(lat), len(lon))

    sum_path = TMP_DIR / "sst_clim_sum_f64.dat"
    cnt_path = TMP_DIR / "sst_clim_count_u8.dat"
    for p in [sum_path, cnt_path]:
        if p.exists():
            p.unlink()

    sums = np.memmap(sum_path, mode="w+", dtype="float64", shape=shape)
    counts = np.memmap(cnt_path, mode="w+", dtype="uint8", shape=shape)
    sums[:] = 0.0
    counts[:] = 0

    print("\n[1/4] Climatologia SST giornaliera 1991-2020")
    for y in range(BASE_START, BASE_END + 1):
        with xr.open_dataset(daily_file(y)) as ds:
            t = pd.DatetimeIndex(ds["time"].values)
            a = ds["thetao"].isel(depth=0).values.astype(np.float32)
        for j, dt in enumerate(t):
            idx = md_to_idx[dt.strftime("%m-%d")]
            v = a[j]
            ok = np.isfinite(v)
            sums[idx][ok] += v[ok]
            counts[idx][ok] += 1
        print(f"  baseline SST {y}: OK")

    clim = np.full(shape, np.nan, dtype=np.float32)
    np.divide(sums, counts, out=clim, where=counts > 0)

    dsout = xr.Dataset(
        data_vars={"sst_climatology": (("time", "latitude", "longitude"), clim)},
        coords={
            "time": ref_dates.values,
            "latitude": lat,
            "longitude": lon,
        },
        attrs={
            "baseline": "1991-2020",
            "period": "September-December",
            "method": "calendar-day mean (MM-DD)",
            "source": "Copernicus Marine Mediterranean Sea Physics Reanalysis",
        },
    )
    dsout["sst_climatology"].attrs["units"] = "degrees_C"
    dsout.to_netcdf(
        SST_CLIM_FILE,
        encoding={"sst_climatology": ENC_FLOAT},
    )
    dsout.close()

    count_min = int(np.min(counts[counts > 0]))
    count_max = int(np.max(counts))
    # Salva i conteggi come attributi per poter riusare la climatologia senza ricalcolarla.
    with xr.open_dataset(SST_CLIM_FILE) as _ds:
        _loaded = _ds.load()
    _loaded.attrs["contributors_min"] = count_min
    _loaded.attrs["contributors_max"] = count_max
    _loaded.to_netcdf(
        SST_CLIM_FILE,
        mode="w",
        encoding={"sst_climatology": ENC_FLOAT},
    )
    _loaded.close()
    print(f"  SST climatology: {SST_CLIM_FILE.name}")
    print(f"  contributi validi/cella-giorno: min={count_min}, max={count_max}")

    del sums, counts
    gc.collect()
    for p in [sum_path, cnt_path]:
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    return clim, ref_dates, md_to_idx, count_min, count_max


def compute_hc_for_file(path, eff, eff_total):
    with xr.open_dataset(path) as ds:
        times = pd.DatetimeIndex(ds["time"].values)
        temp = ds["thetao"].values.astype(np.float64)

    if temp.shape[1] != eff.shape[0]:
        raise ValueError(f"{path.name}: livelli thetao={temp.shape[1]}, eff={eff.shape[0]}")

    out_hc = np.full((temp.shape[0],) + temp.shape[2:], np.nan, dtype=np.float64)
    out_tm = np.full_like(out_hc, np.nan)

    for j in range(temp.shape[0]):
        tj = temp[j]
        valid = np.isfinite(tj) & (eff > 0)
        covered = np.sum(np.where(valid, eff, 0.0), axis=0)
        integral_t = np.sum(np.where(valid, tj * eff, 0.0), axis=0)

        # Richiediamo copertura verticale praticamente completa della colonna disponibile.
        good = (eff_total > 0) & (covered >= eff_total - 1e-3)
        hc = RHO * CP * integral_t
        tm = np.divide(
            integral_t,
            eff_total,
            out=np.full_like(integral_t, np.nan),
            where=good,
        )
        hc[~good] = np.nan
        tm[~good] = np.nan
        out_hc[j] = hc
        out_tm[j] = tm

    return times, out_hc, out_tm


def build_ohc_climatology(lat, lon, eff, eff_total):
    sums = np.zeros((4, len(lat), len(lon)), dtype=np.float64)
    counts = np.zeros((4, len(lat), len(lon)), dtype=np.uint8)
    tmean_sums = np.zeros_like(sums)

    print("\n[2/4] Climatologia OHC mensile 1991-2020")
    for y in range(BASE_START, BASE_END + 1):
        times, hc, tm = compute_hc_for_file(monthly_file(y), eff, eff_total)
        if len(times) != 4:
            raise ValueError(f"{y}: attesi 4 mesi, trovati {len(times)}")
        for j, dt in enumerate(times):
            idx = dt.month - 9
            if idx not in range(4):
                raise ValueError(f"{y}: mese inatteso {dt.month}")
            ok = np.isfinite(hc[j])
            sums[idx][ok] += hc[j][ok]
            tmean_sums[idx][ok] += tm[j][ok]
            counts[idx][ok] += 1
        print(f"  baseline OHC {y}: OK")

    hc_clim = np.full_like(sums, np.nan, dtype=np.float64)
    tm_clim = np.full_like(sums, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=hc_clim, where=counts > 0)
    np.divide(tmean_sums, counts, out=tm_clim, where=counts > 0)

    ref_times = pd.to_datetime(["2001-09-01", "2001-10-01", "2001-11-01", "2001-12-01"])
    dsout = xr.Dataset(
        data_vars={
            "ohc_0_100_climatology": (
                ("time", "latitude", "longitude"),
                hc_clim.astype(np.float32),
            ),
            "tmean_0_100_climatology": (
                ("time", "latitude", "longitude"),
                tm_clim.astype(np.float32),
            ),
        },
        coords={"time": ref_times.values, "latitude": lat, "longitude": lon},
        attrs={
            "baseline": "1991-2020",
            "period": "September-December",
            "rho_kg_m3": RHO,
            "cp_j_kg_k": CP,
            "target_depth_m": TARGET_DEPTH,
            "method": "monthly mean heat content per unit area, integrated with e3t and mask",
        },
    )
    dsout["ohc_0_100_climatology"].attrs["units"] = "J m-2"
    dsout["tmean_0_100_climatology"].attrs["units"] = "degrees_C"
    dsout.to_netcdf(
        OHC_CLIM_FILE,
        encoding={
            "ohc_0_100_climatology": ENC_FLOAT,
            "tmean_0_100_climatology": ENC_FLOAT,
        },
    )
    dsout.close()

    count_min = int(np.min(counts[counts > 0]))
    count_max = int(np.max(counts))
    with xr.open_dataset(OHC_CLIM_FILE) as _ds:
        _loaded = _ds.load()
    _loaded.attrs["contributors_min"] = count_min
    _loaded.attrs["contributors_max"] = count_max
    _loaded.to_netcdf(
        OHC_CLIM_FILE,
        mode="w",
        encoding={
            "ohc_0_100_climatology": ENC_FLOAT,
            "tmean_0_100_climatology": ENC_FLOAT,
        },
    )
    _loaded.close()
    print(f"  OHC climatology: {OHC_CLIM_FILE.name}")
    print(f"  contributi validi/cella-mese: min={count_min}, max={count_max}")
    return hc_clim, tm_clim, count_min, count_max


def load_sst_climatology_if_available(lat, lon, force=False):
    if force or not SST_CLIM_FILE.exists():
        return None
    try:
        with xr.open_dataset(SST_CLIM_FILE) as ds:
            if "sst_climatology" not in ds or ds.sizes.get("time") != 122:
                return None
            if ds.sizes.get("latitude") != len(lat) or ds.sizes.get("longitude") != len(lon):
                return None
            clim = ds["sst_climatology"].values.astype(np.float32)
            ref_dates = pd.DatetimeIndex(ds["time"].values)
            cmin = int(ds.attrs.get("contributors_min", BASE_END - BASE_START + 1))
            cmax = int(ds.attrs.get("contributors_max", BASE_END - BASE_START + 1))
        md_to_idx = {d.strftime("%m-%d"): i for i, d in enumerate(ref_dates)}
        print("\n[1/4] Climatologia SST giornaliera 1991-2020")
        print(f"  REUSE {SST_CLIM_FILE.name}")
        return clim, ref_dates, md_to_idx, cmin, cmax
    except Exception:
        return None


def load_ohc_climatology_if_available(lat, lon, force=False):
    if force or not OHC_CLIM_FILE.exists():
        return None
    try:
        with xr.open_dataset(OHC_CLIM_FILE) as ds:
            if "ohc_0_100_climatology" not in ds or ds.sizes.get("time") != 4:
                return None
            if ds.sizes.get("latitude") != len(lat) or ds.sizes.get("longitude") != len(lon):
                return None
            hc_clim = ds["ohc_0_100_climatology"].values.astype(np.float64)
            tm_clim = ds["tmean_0_100_climatology"].values.astype(np.float64)
            cmin = int(ds.attrs.get("contributors_min", BASE_END - BASE_START + 1))
            cmax = int(ds.attrs.get("contributors_max", BASE_END - BASE_START + 1))
        print("\n[2/4] Climatologia OHC mensile 1991-2020")
        print(f"  REUSE {OHC_CLIM_FILE.name}")
        return hc_clim, tm_clim, cmin, cmax
    except Exception:
        return None


def area_weighted_mean(field, area):
    ok = np.isfinite(field) & np.isfinite(area)
    if not np.any(ok):
        return np.nan
    return float(np.sum(field[ok] * area[ok]) / np.sum(area[ok]))


def process_sst_years(years, lat, lon, area, clim, md_to_idx, force=False):
    print("\n[3/4] Anomalie SST giornaliere")
    rows = []

    for y in years:
        target = SST_OUT / f"medsea_sst_anomaly_{y}_SepDec.nc"

        with xr.open_dataset(daily_file(y)) as ds:
            times = pd.DatetimeIndex(ds["time"].values)
            sst = ds["thetao"].isel(depth=0).values.astype(np.float32)

        idxs = np.array([md_to_idx[t.strftime("%m-%d")] for t in times], dtype=int)
        anom = sst - clim[idxs]

        if force or not file_good(target, "sst_anomaly", len(times)):
            dsout = xr.Dataset(
                data_vars={"sst_anomaly": (("time", "latitude", "longitude"), anom)},
                coords={"time": times.values, "latitude": lat, "longitude": lon},
                attrs={
                    "baseline": "1991-2020",
                    "definition": "daily SST minus 1991-2020 calendar-day climatology",
                },
            )
            dsout["sst_anomaly"].attrs["units"] = "degrees_C"
            dsout.to_netcdf(target, encoding={"sst_anomaly": ENC_FLOAT})
            dsout.close()

        for j, t in enumerate(times):
            valid_area = np.nansum(np.where(np.isfinite(anom[j]), area, np.nan))
            rows.append(
                {
                    "date": t.strftime("%Y-%m-%d"),
                    "year": y,
                    "month": t.month,
                    "day": t.day,
                    "sst_mean_c": area_weighted_mean(sst[j], area),
                    "sst_anomaly_mean_c": area_weighted_mean(anom[j], area),
                    "valid_area_km2": float(valid_area / 1e6),
                }
            )
        print(f"  SST {y}: OK")

    return pd.DataFrame(rows)


def process_ohc_years(years, lat, lon, area, eff, eff_total, hc_clim, tm_clim, force=False):
    print("\n[4/4] Anomalie OHC mensili")
    rows = []

    for y in years:
        times, hc, tm = compute_hc_for_file(monthly_file(y), eff, eff_total)
        idxs = np.array([t.month - 9 for t in times], dtype=int)
        anom = hc - hc_clim[idxs]
        tm_anom = tm - tm_clim[idxs]

        target = OHC_OUT / f"medsea_ohc_anomaly_{y}_SepDec.nc"
        if force or not file_good(target, "ohc_anomaly_0_100", len(times)):
            dsout = xr.Dataset(
                data_vars={
                    "ohc_0_100": (
                        ("time", "latitude", "longitude"),
                        hc.astype(np.float32),
                    ),
                    "ohc_anomaly_0_100": (
                        ("time", "latitude", "longitude"),
                        anom.astype(np.float32),
                    ),
                    "tmean_0_100": (
                        ("time", "latitude", "longitude"),
                        tm.astype(np.float32),
                    ),
                    "tmean_anomaly_0_100": (
                        ("time", "latitude", "longitude"),
                        tm_anom.astype(np.float32),
                    ),
                },
                coords={"time": times.values, "latitude": lat, "longitude": lon},
                attrs={
                    "baseline": "1991-2020",
                    "rho_kg_m3": RHO,
                    "cp_j_kg_k": CP,
                    "target_depth_m": TARGET_DEPTH,
                },
            )
            dsout["ohc_0_100"].attrs["units"] = "J m-2"
            dsout["ohc_anomaly_0_100"].attrs["units"] = "J m-2"
            dsout["tmean_0_100"].attrs["units"] = "degrees_C"
            dsout["tmean_anomaly_0_100"].attrs["units"] = "degrees_C"
            dsout.to_netcdf(
                target,
                encoding={
                    "ohc_0_100": ENC_FLOAT,
                    "ohc_anomaly_0_100": ENC_FLOAT,
                    "tmean_0_100": ENC_FLOAT,
                    "tmean_anomaly_0_100": ENC_FLOAT,
                },
            )
            dsout.close()

        for j, t in enumerate(times):
            ok = np.isfinite(anom[j]) & np.isfinite(area)
            signed_j = float(np.sum(anom[j][ok] * area[ok])) if np.any(ok) else np.nan
            pos_j = (
                float(np.sum(np.maximum(anom[j][ok], 0.0) * area[ok]))
                if np.any(ok)
                else np.nan
            )
            neg_j = (
                float(np.sum(np.minimum(anom[j][ok], 0.0) * area[ok]))
                if np.any(ok)
                else np.nan
            )
            rows.append(
                {
                    "date": t.strftime("%Y-%m-%d"),
                    "year": y,
                    "month": t.month,
                    "ohc_anomaly_mean_j_m2": area_weighted_mean(anom[j], area),
                    "tmean_0_100_c": area_weighted_mean(tm[j], area),
                    "tmean_anomaly_0_100_c": area_weighted_mean(tm_anom[j], area),
                    "ohc_anomaly_signed_j": signed_j,
                    "ohc_anomaly_positive_j": pos_j,
                    "ohc_anomaly_negative_j": neg_j,
                    "valid_area_km2": float(np.nansum(np.where(ok, area, np.nan)) / 1e6),
                }
            )
        print(f"  OHC {y}: OK")

    return pd.DataFrame(rows)


def write_qc(static_qc, sst_cmin, sst_cmax, ohc_cmin, ohc_cmax, years, sst_df, ohc_df):
    path = OUTROOT / "medsea_historical_analysis_qc.txt"
    lines = [
        "COPERNICUS MARINE HISTORICAL ANALYSIS QC",
        "=" * 72,
        f"Years processed: {years[0]}-{years[-1]}",
        f"Climatology baseline: {BASE_START}-{BASE_END}",
        f"Grid: {static_qc['grid_nlat']} x {static_qc['grid_nlon']}",
        f"Depth levels: {static_qc['depth_levels']}",
        f"Depth range: {static_qc['depth_min_m']:.6f} -> {static_qc['depth_max_m']:.6f} m",
        f"Target integration depth: {TARGET_DEPTH:.1f} m",
        f"Sea area: {static_qc['sea_area_km2']:.3f} km2",
        f"Vertical closure mean abs error: {static_qc['vertical_closure_mean_abs_m']:.6f} m",
        f"Vertical closure max abs error: {static_qc['vertical_closure_max_abs_m']:.6f} m",
        f"SST climatology contributors min/max: {sst_cmin}/{sst_cmax}",
        f"OHC climatology contributors min/max: {ohc_cmin}/{ohc_cmax}",
        f"SST output rows: {len(sst_df)}",
        f"OHC output rows: {len(ohc_df)}",
        "",
        "Definitions:",
        "SST anomaly = daily SST - 1991-2020 mean for same calendar day.",
        "OHC anomaly = monthly OHC(0-100 m) - 1991-2020 mean for same month.",
        "OHC per unit area = rho*cp*sum(thetao*effective_layer_thickness).",
        "Signed/positive/negative energies in CSV are spatial integrals of OHC anomaly.",
        "",
        "NOTE: OHC is a state/reservoir metric, not atmospheric heat release.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    args = parse_args()
    ensure_dirs()
    static_bathy, static_metrics = check_inputs()

    print("MEDSEA HISTORICAL ANALYSIS")
    print(f"Input : {INROOT}")
    print(f"Output: {OUTROOT}")
    print(f"Baseline: {BASE_START}-{BASE_END}")
    print(f"Mode: {'TEST (output 2000 only)' if args.test else 'FULL 1987-2025'}")

    lat, lon, depth_m, deptho, area, eff, eff_total, static_qc = load_static(
        static_bathy, static_metrics
    )

    print("\nSTATIC QC")
    for k, v in static_qc.items():
        print(f"  {k}: {v}")

    sst_loaded = load_sst_climatology_if_available(lat, lon, force=args.force)
    if sst_loaded is None:
        clim, ref_dates, md_to_idx, sst_cmin, sst_cmax = build_sst_climatology(lat, lon)
    else:
        clim, ref_dates, md_to_idx, sst_cmin, sst_cmax = sst_loaded

    ohc_loaded = load_ohc_climatology_if_available(lat, lon, force=args.force)
    if ohc_loaded is None:
        hc_clim, tm_clim, ohc_cmin, ohc_cmax = build_ohc_climatology(
            lat, lon, eff, eff_total
        )
    else:
        hc_clim, tm_clim, ohc_cmin, ohc_cmax = ohc_loaded

    years = [2000] if args.test else YEARS

    sst_df = process_sst_years(
        years, lat, lon, area, clim, md_to_idx, force=args.force
    )
    ohc_df = process_ohc_years(
        years, lat, lon, area, eff, eff_total, hc_clim, tm_clim, force=args.force
    )

    if args.test:
        sst_csv = OUTROOT / "medsea_daily_sst_summary_TEST_2000.csv"
        ohc_csv = OUTROOT / "medsea_monthly_ohc_summary_TEST_2000.csv"
    else:
        sst_csv = OUTROOT / "medsea_daily_sst_summary_1987_2025.csv"
        ohc_csv = OUTROOT / "medsea_monthly_ohc_summary_1987_2025.csv"

    sst_df.to_csv(sst_csv, index=False)
    ohc_df.to_csv(ohc_csv, index=False)

    qc_path = write_qc(
        static_qc, sst_cmin, sst_cmax, ohc_cmin, ohc_cmax, years, sst_df, ohc_df
    )

    print("\n" + "=" * 88)
    print("MEDSEA HISTORICAL ANALYSIS: COMPLETE")
    print(f"SST climatology : {SST_CLIM_FILE}")
    print(f"OHC climatology : {OHC_CLIM_FILE}")
    print(f"SST summary     : {sst_csv}")
    print(f"OHC summary     : {ohc_csv}")
    print(f"QC report       : {qc_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
