#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset, date2num

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "ocean_final"
OUT.mkdir(exist_ok=True)

CURRENT_FILE = DATA / "italy_seas_temp_2026_0_110m.nc"
CLIM_FILE = DATA / "italy_seas_climatology_MJJAS_0_110m_v2.nc"
BATHY_FILE = DATA / "italy_seas_bathy_mask_0_110m.nc"
METRICS_FILE = DATA / "italy_seas_grid_metrics_0_110m.nc"

DAILY_NC = OUT / "italy_seas_daily_anomaly_final.nc"
DAILY_CSV = OUT / "italy_seas_daily_summary_final.csv"
MEAN_NC = OUT / "italy_seas_period_mean_final.nc"
CELLS_CSV = OUT / "italy_seas_cells_final.csv"
CELLS_GEOJSON = OUT / "italy_seas_cells_final.geojson"
QC_TXT = OUT / "quality_control_report_final.txt"

RHO = 1025.0
CP = 3991.86795711963
ZMAX = 100.0
CELL_DEG = 0.5

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"Setting the shape on a NumPy array has been deprecated.*",
)


def normalize(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    aliases = {
        "longitude": ["lon", "longitude", "x"],
        "latitude": ["lat", "latitude", "y"],
        "depth": ["depth", "deptht", "lev", "z"],
        "time": ["time", "time_counter"],
    }
    for canonical, names in aliases.items():
        if canonical in ds.coords or canonical in ds.dims:
            continue
        for name in names:
            if name in ds.coords or name in ds.dims:
                rename[name] = canonical
                break

    if rename:
        ds = ds.rename(rename)

    sort_coords = [c for c in ("time", "depth", "latitude", "longitude")
                   if c in ds.coords or c in ds.dims]
    if sort_coords:
        ds = ds.sortby(sort_coords)
    return ds


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Manca il file richiesto: {path}")


def align_nearest(
    da: xr.DataArray,
    target_lon: xr.DataArray,
    target_lat: xr.DataArray,
    target_depth: xr.DataArray | None = None,
) -> xr.DataArray:
    """
    Allinea prodotti della stessa griglia NEMO senza interpolazione verticale.
    Usa nearest solo per assorbire piccolissime differenze floating-point/ritaglio.
    """
    da = da.sortby([c for c in ("depth", "latitude", "longitude") if c in da.dims])

    if "latitude" in da.dims:
        da = da.reindex(latitude=target_lat.values, method="nearest", tolerance=0.03)
        da = da.assign_coords(latitude=target_lat.values)

    if "longitude" in da.dims:
        da = da.reindex(longitude=target_lon.values, method="nearest", tolerance=0.03)
        da = da.assign_coords(longitude=target_lon.values)

    if target_depth is not None and "depth" in da.dims:
        da = da.reindex(depth=target_depth.values, method="nearest", tolerance=0.20)
        da = da.assign_coords(depth=target_depth.values)

    return da


def climatology_anchors(
    clim_t: xr.DataArray,
    target_template: xr.DataArray,
) -> dict[int, xr.DataArray]:
    """
    Campi mensili maggio-settembre allineati alla griglia 2026.
    Nessuna interpolazione verticale: nearest con tolleranza stretta.
    """
    out = {}
    months_available = [int(m) for m in clim_t.time.dt.month.values]

    for month in (5, 6, 7, 8, 9):
        if month not in months_available:
            raise ValueError(f"Manca il mese climatologico {month}.")

        da = clim_t.sel(
            time=clim_t.time[clim_t.time.dt.month == month]
        ).isel(time=0)

        da = align_nearest(
            da,
            target_template.longitude,
            target_template.latitude,
            target_template.depth,
        )

        out[month] = da

    return out


def daily_climatology(
    ts: pd.Timestamp,
    anchors: dict[int, xr.DataArray],
):
    """
    Climatologia giornaliera continua:
    interpolazione lineare fra medie mensili ancorate al giorno 15.
    """
    year = int(ts.year)
    t_mid = pd.Timestamp(year=year, month=ts.month, day=15)

    if ts >= t_mid:
        m0 = ts.month
        m1 = ts.month + 1
        t0 = t_mid
        t1 = pd.Timestamp(year=year, month=m1, day=15)
    else:
        m1 = ts.month
        m0 = ts.month - 1
        t0 = pd.Timestamp(year=year, month=m0, day=15)
        t1 = t_mid

    if m0 not in anchors or m1 not in anchors:
        raise ValueError(f"Climatologia insufficiente per {ts.date()}: servono {m0},{m1}.")

    w = float((ts - t0).total_seconds() / (t1 - t0).total_seconds())
    w = float(np.clip(w, 0.0, 1.0))
    return (1.0 - w) * anchors[m0] + w * anchors[m1], m0, m1, w


def area_weighted_mean(field: np.ndarray, area: np.ndarray, wet2d: np.ndarray) -> float:
    m = np.isfinite(field) & np.isfinite(area) & wet2d
    if not np.any(m):
        return float("nan")
    return float(np.sum(field[m] * area[m]) / np.sum(area[m]))


def prepare_static(
    cur_t: xr.DataArray,
    bathy: xr.Dataset,
    metrics: xr.Dataset,
):
    lon = cur_t.longitude
    lat = cur_t.latitude
    depth = cur_t.depth

    for v in ("deptho", "mask"):
        if v not in bathy:
            raise KeyError(f"Nel file bathy manca '{v}'.")
    for v in ("e1t", "e2t", "e3t"):
        if v not in metrics:
            raise KeyError(f"Nel file metrics manca '{v}'.")

    deptho = align_nearest(bathy["deptho"], lon, lat)
    mask = align_nearest(bathy["mask"], lon, lat, depth)
    e1t = align_nearest(metrics["e1t"], lon, lat)
    e2t = align_nearest(metrics["e2t"], lon, lat)
    e3t = align_nearest(metrics["e3t"], lon, lat, depth)

    # Squeeze di eventuali dimensioni singleton non geografiche.
    for name, da in [("deptho", deptho), ("e1t", e1t), ("e2t", e2t)]:
        extra = [d for d in da.dims if d not in ("latitude", "longitude")]
        if extra:
            da = da.squeeze(extra, drop=True)
        if name == "deptho":
            deptho = da
        elif name == "e1t":
            e1t = da
        else:
            e2t = da

    # mask/e3t devono avere depth, lat, lon.
    extra = [d for d in mask.dims if d not in ("depth", "latitude", "longitude")]
    if extra:
        mask = mask.squeeze(extra, drop=True)
    extra = [d for d in e3t.dims if d not in ("depth", "latitude", "longitude")]
    if extra:
        e3t = e3t.squeeze(extra, drop=True)

    # Ordine esplicito.
    deptho = deptho.transpose("latitude", "longitude")
    e1t = e1t.transpose("latitude", "longitude")
    e2t = e2t.transpose("latitude", "longitude")
    mask = mask.transpose("depth", "latitude", "longitude")
    e3t = e3t.transpose("depth", "latitude", "longitude")

    return deptho, mask, e1t, e2t, e3t


def effective_thickness_0_100(
    deptho: np.ndarray,
    mask3d: np.ndarray,
    e3t3d: np.ndarray,
    zmax: float = 100.0,
):
    """
    Calcola lo spessore effettivo di ciascun livello entro 0-min(zmax, fondale).

    e3t viene trattato come spessore verticale reale del livello.
    La maschera elimina i livelli non bagnati.
    deptho impone il clipping esatto al fondale e a 100 m.
    """
    wet = np.asarray(mask3d, dtype=float) > 0.5
    thick = np.where(wet & np.isfinite(e3t3d), e3t3d, 0.0).astype(np.float64)

    # Posizione top/bottom costruita cumulando gli spessori effettivi dei livelli bagnati.
    top = np.cumsum(thick, axis=0) - thick
    bottom = top + thick

    target = np.minimum(np.asarray(deptho, dtype=np.float64), float(zmax))
    target = np.where(np.isfinite(target) & (target > 0), target, 0.0)

    eff = np.minimum(bottom, target[None, :, :]) - top
    eff = np.clip(eff, 0.0, None)
    eff = np.where(wet, eff, 0.0)

    used_depth = np.sum(eff, axis=0)
    wet2d = target > 0

    closure_error = np.where(wet2d, used_depth - target, np.nan)
    return eff, used_depth, target, wet2d, closure_error


def main():
    for p in (CURRENT_FILE, CLIM_FILE, BATHY_FILE, METRICS_FILE):
        require_file(p)

    print("Apro i quattro dataset...")
    cur = normalize(xr.open_dataset(CURRENT_FILE))
    clim = normalize(xr.open_dataset(CLIM_FILE))
    bathy = normalize(xr.open_dataset(BATHY_FILE))
    metrics = normalize(xr.open_dataset(METRICS_FILE))

    if "thetao" not in cur:
        raise KeyError("Nel file 2026 manca 'thetao'.")
    if "thetao_avg" not in clim:
        raise KeyError("Nel file climatologico manca 'thetao_avg'.")

    cur_t = cur["thetao"].sortby("depth")
    clim_t = clim["thetao_avg"].sortby("depth")

    current_first_depth = float(cur_t.depth.values[0])
    clim_first_depth = float(clim_t.depth.values[0])
    print(f"Primo livello 2026: {current_first_depth:.6f} m")
    print(f"Primo livello climatologia: {clim_first_depth:.6f} m")

    if abs(current_first_depth - clim_first_depth) > 0.20:
        raise ValueError(
            "Il primo livello verticale della climatologia non coincide con quello 2026. "
            f"2026={current_first_depth:.6f} m; climatologia={clim_first_depth:.6f} m. "
            "La climatologia deve includere il livello superficiale ~1.018236 m."
        )

    template = cur_t.isel(time=0)

    print("Allineo climatologia senza interpolazione verticale...")
    anchors = climatology_anchors(clim_t, template)

    for month, da in anchors.items():
        surface = np.asarray(da.isel(depth=0).values, dtype=float)
        finite_fraction = float(np.isfinite(surface).mean())
        print(f"Mese {month}: frazione finita al primo livello = {finite_fraction:.3%}")
        if finite_fraction == 0.0:
            raise ValueError(
                f"Il mese climatologico {month} non contiene dati al primo livello. "
                "Interrompo prima del calcolo per evitare SST/OHC corrotti."
            )

    print("Allineo batimetria, maschera e metriche della griglia...")
    deptho_da, mask_da, e1t_da, e2t_da, e3t_da = prepare_static(cur_t, bathy, metrics)

    lat = np.asarray(cur_t.latitude.values, dtype=float)
    lon = np.asarray(cur_t.longitude.values, dtype=float)
    depth = np.asarray(cur_t.depth.values, dtype=float)
    times = pd.to_datetime(cur_t.time.values)

    deptho = np.asarray(deptho_da.values, dtype=np.float64)
    mask3d = np.asarray(mask_da.values, dtype=np.float64)
    e1t = np.asarray(e1t_da.values, dtype=np.float64)
    e2t = np.asarray(e2t_da.values, dtype=np.float64)
    e3t = np.asarray(e3t_da.values, dtype=np.float64)

    area = e1t * e2t
    eff_dz, used_depth, target_depth, wet2d, closure_error = effective_thickness_0_100(
        deptho, mask3d, e3t, ZMAX
    )

    # Volume totale rappresentato entro min(100,H).
    volume3d = eff_dz * area[None, :, :]

    nt = len(times)
    nz, ny, nx = mask3d.shape
    print(f"Giorni={nt}; griglia={ny}x{nx}; livelli statici={nz}")

    # QC di geometria.
    wet_count = int(np.sum(wet2d))
    ocean_area_km2 = float(np.nansum(np.where(wet2d, area, 0.0)) / 1e6)
    closure_abs = np.abs(closure_error[np.isfinite(closure_error)])
    closure_max = float(np.max(closure_abs)) if closure_abs.size else np.nan
    closure_mean = float(np.mean(closure_abs)) if closure_abs.size else np.nan

    # Output giornaliero 2D.
    nc = Dataset(DAILY_NC, "w", format="NETCDF4")
    nc.createDimension("time", None)
    nc.createDimension("latitude", ny)
    nc.createDimension("longitude", nx)

    vtime = nc.createVariable("time", "f8", ("time",))
    vlat = nc.createVariable("latitude", "f4", ("latitude",))
    vlon = nc.createVariable("longitude", "f4", ("longitude",))
    vsst = nc.createVariable(
        "near_surface_temperature_anomaly",
        "f4", ("time", "latitude", "longitude"),
        zlib=True, complevel=4, fill_value=np.float32(np.nan)
    )
    vohc = nc.createVariable(
        "ohc_anomaly_0_min100H",
        "f4", ("time", "latitude", "longitude"),
        zlib=True, complevel=4, fill_value=np.float32(np.nan)
    )
    vpos = nc.createVariable(
        "positive_ohc_anomaly_0_min100H",
        "f4", ("time", "latitude", "longitude"),
        zlib=True, complevel=4, fill_value=np.float32(np.nan)
    )
    vdepth = nc.createVariable(
        "integrated_water_depth",
        "f4", ("latitude", "longitude"),
        zlib=True, complevel=4, fill_value=np.float32(np.nan)
    )
    vbathy = nc.createVariable(
        "sea_floor_depth",
        "f4", ("latitude", "longitude"),
        zlib=True, complevel=4, fill_value=np.float32(np.nan)
    )

    vlat[:] = lat.astype("f4")
    vlon[:] = lon.astype("f4")
    vdepth[:, :] = np.where(wet2d, target_depth, np.nan).astype("f4")
    vbathy[:, :] = np.where(wet2d, deptho, np.nan).astype("f4")

    vtime.units = "days since 1970-01-01 00:00:00"
    vtime.calendar = "proleptic_gregorian"
    vlat.units = "degrees_north"
    vlon.units = "degrees_east"
    vsst.units = "degrees_C"
    vohc.units = "J m-2"
    vpos.units = "J m-2"
    vdepth.units = "m"
    vbathy.units = "m"

    nc.title = "Final ocean thermal anomaly dataset for Italy-relevant Mediterranean seas, summer 2026"
    nc.reference_climatology = (
        "Copernicus monthly thetao_avg May-September; daily reference obtained by "
        "linear interpolation between monthly fields anchored on day 15."
    )
    nc.grid_method = (
        "Copernicus e1t*e2t horizontal area and e3t vertical thickness; "
        "Copernicus mask and deptho; integration clipped to min(100 m, local depth)."
    )
    nc.rho_kg_m3 = RHO
    nc.cp_J_kg_K = CP

    sst_sum = np.zeros((ny, nx), dtype=np.float64)
    sst_count = np.zeros((ny, nx), dtype=np.int32)
    ohc_sum = np.zeros((ny, nx), dtype=np.float64)
    ohc_count = np.zeros((ny, nx), dtype=np.int32)
    pos_sum = np.zeros((ny, nx), dtype=np.float64)
    pos_count = np.zeros((ny, nx), dtype=np.int32)

    rows = []
    total_surface_missing = 0
    total_surface_wet = 0

    print("Calcolo il dataset finale...")
    for i, ts in enumerate(times):
        clim_day, m0, m1, w = daily_climatology(ts, anchors)
        day = cur_t.isel(time=i)

        anom = day - clim_day
        a3 = np.asarray(anom.values, dtype=np.float64)

        # Maschera esplicita mare/terra.
        a3 = np.where(mask3d > 0.5, a3, np.nan)

        # Primo livello del modello (~1.018 m): proxy near-surface.
        sst = a3[0, :, :]
        sst = np.where(wet2d, sst, np.nan)

        # OHC per unità di area usando spessori effettivi.
        valid = np.isfinite(a3) & (eff_dz > 0)
        weighted_dTdz = np.where(valid, a3 * eff_dz, 0.0)
        int_dTdz = np.sum(weighted_dTdz, axis=0)

        # Una cella è valida se ha almeno un livello valido con spessore >0.
        valid_col = np.any(valid, axis=0) & wet2d
        int_dTdz = np.where(valid_col, int_dTdz, np.nan)

        ohc = RHO * CP * int_dTdz
        pos = np.where(np.isfinite(ohc), np.maximum(ohc, 0.0), np.nan)

        vtime[i] = date2num(ts.to_pydatetime(), vtime.units, vtime.calendar)
        vsst[i, :, :] = np.ascontiguousarray(sst, dtype=np.float32)
        vohc[i, :, :] = np.ascontiguousarray(ohc, dtype=np.float32)
        vpos[i, :, :] = np.ascontiguousarray(pos, dtype=np.float32)

        m = np.isfinite(sst)
        sst_sum[m] += sst[m]
        sst_count[m] += 1

        m = np.isfinite(ohc)
        ohc_sum[m] += ohc[m]
        ohc_count[m] += 1

        m = np.isfinite(pos)
        pos_sum[m] += pos[m]
        pos_count[m] += 1

        total_surface_missing += int(np.sum(wet2d & ~np.isfinite(sst)))
        total_surface_wet += int(np.sum(wet2d))

        # Energia totale: OHC per m² * area reale e1t*e2t.
        signed_total_J = float(np.nansum(ohc * area))
        positive_total_J = float(np.nansum(pos * area))

        rows.append({
            "date": ts.strftime("%Y-%m-%d"),
            "clim_month_0": m0,
            "clim_month_1": m1,
            "clim_weight_to_month_1": w,
            "area_weighted_near_surface_anomaly_C":
                area_weighted_mean(sst, area, wet2d),
            "signed_heat_anomaly_PJ": signed_total_J / 1e15,
            "positive_heat_anomaly_PJ": positive_total_J / 1e15,
        })

        if i == 0 or (i + 1) % 5 == 0 or i == nt - 1:
            print(
                f"[{i+1:02d}/{nt}] {ts.date()} "
                f"clim={m0}->{m1} w={w:.3f} "
                f"heat={signed_total_J/1e18:+.3f} EJ"
            )

    nc.close()

    daily = pd.DataFrame(rows)
    daily.to_csv(DAILY_CSV, index=False)

    mean_sst = np.divide(
        sst_sum, sst_count, out=np.full_like(sst_sum, np.nan), where=sst_count > 0
    )
    mean_ohc = np.divide(
        ohc_sum, ohc_count, out=np.full_like(ohc_sum, np.nan), where=ohc_count > 0
    )
    mean_pos = np.divide(
        pos_sum, pos_count, out=np.full_like(pos_sum, np.nan), where=pos_count > 0
    )

    mean_ds = xr.Dataset(
        data_vars={
            "mean_near_surface_temperature_anomaly":
                (("latitude", "longitude"), mean_sst.astype("f4")),
            "mean_ohc_anomaly_0_min100H":
                (("latitude", "longitude"), mean_ohc.astype("f4")),
            "mean_positive_ohc_anomaly_0_min100H":
                (("latitude", "longitude"), mean_pos.astype("f4")),
            "integrated_water_depth":
                (("latitude", "longitude"), np.where(wet2d, target_depth, np.nan).astype("f4")),
            "sea_floor_depth":
                (("latitude", "longitude"), np.where(wet2d, deptho, np.nan).astype("f4")),
            "cell_area":
                (("latitude", "longitude"), np.where(wet2d, area, np.nan).astype("f8")),
        },
        coords={
            "latitude": lat.astype("f4"),
            "longitude": lon.astype("f4"),
        },
        attrs={
            "study_period": f"{times[0].date()} to {times[-1].date()}",
            "rho_kg_m3": RHO,
            "cp_J_kg_K": CP,
            "climatology_method":
                "Linear interpolation between monthly Copernicus climatologies anchored at day 15.",
            "vertical_method":
                "Copernicus e3t and mask; clipped to min(100 m, deptho).",
            "horizontal_area_method":
                "Copernicus e1t*e2t.",
        },
    )
    mean_ds["mean_near_surface_temperature_anomaly"].attrs["units"] = "degrees_C"
    mean_ds["mean_ohc_anomaly_0_min100H"].attrs["units"] = "J m-2"
    mean_ds["mean_positive_ohc_anomaly_0_min100H"].attrs["units"] = "J m-2"
    mean_ds["integrated_water_depth"].attrs["units"] = "m"
    mean_ds["sea_floor_depth"].attrs["units"] = "m"
    mean_ds["cell_area"].attrs["units"] = "m2"
    mean_ds.to_netcdf(MEAN_NC)

    # Aggregazione in celle 0.5°.
    print("Aggrego in celle 0.5° x 0.5°...")
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    flat = pd.DataFrame({
        "lat": lat2.ravel(),
        "lon": lon2.ravel(),
        "area_m2": area.ravel(),
        "wet": wet2d.ravel(),
        "deptho_m": deptho.ravel(),
        "used_depth_m": target_depth.ravel(),
        "sst": mean_sst.ravel(),
        "ohc": mean_ohc.ravel(),
        "pos": mean_pos.ravel(),
    })
    flat = flat[flat["wet"]].copy()
    flat["west"] = np.floor(flat["lon"] / CELL_DEG) * CELL_DEG
    flat["south"] = np.floor(flat["lat"] / CELL_DEG) * CELL_DEG

    records = []
    features = []

    for (west, south), g in flat.groupby(["west", "south"], sort=True):
        east = west + CELL_DEG
        north = south + CELL_DEG

        valid_ohc = g[np.isfinite(g["ohc"])]
        if valid_ohc.empty:
            continue

        sea_area = float(g["area_m2"].sum())

        def wmean(col):
            gg = g[np.isfinite(g[col])]
            if gg.empty:
                return np.nan
            return float(np.sum(gg[col] * gg["area_m2"]) / np.sum(gg["area_m2"]))

        signed_J = float(np.nansum(g["ohc"] * g["area_m2"]))
        positive_J = float(np.nansum(g["pos"] * g["area_m2"]))

        rec = {
            "west": west,
            "east": east,
            "south": south,
            "north": north,
            "centroid_lon": (west + east) / 2.0,
            "centroid_lat": (south + north) / 2.0,
            "sea_area_km2": sea_area / 1e6,
            "mean_sea_floor_depth_m": wmean("deptho_m"),
            "mean_integrated_depth_m": wmean("used_depth_m"),
            "mean_near_surface_anomaly_C": wmean("sst"),
            "mean_ohc_anomaly_MJ_m2": wmean("ohc") / 1e6,
            "mean_positive_ohc_MJ_m2": wmean("pos") / 1e6,
            "signed_heat_anomaly_PJ": signed_J / 1e15,
            "positive_heat_anomaly_PJ": positive_J / 1e15,
        }
        records.append(rec)

        features.append({
            "type": "Feature",
            "properties": rec,
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [west, south], [east, south], [east, north],
                    [west, north], [west, south]
                ]]
            }
        })

    pd.DataFrame(records).to_csv(CELLS_CSV, index=False)
    CELLS_GEOJSON.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False),
        encoding="utf-8"
    )

    # Quality control finale.
    heat_ej = daily["signed_heat_anomaly_PJ"].to_numpy(float) / 1000.0
    sst_series = daily["area_weighted_near_surface_anomaly_C"].to_numpy(float)

    d_heat = np.diff(heat_ej)
    d_sst = np.diff(sst_series)

    def idx_of_date(d):
        hit = daily.index[daily["date"] == d].tolist()
        return hit[0] if hit else None

    def delta_between(d0, d1, col, divisor=1.0):
        i0, i1 = idx_of_date(d0), idx_of_date(d1)
        if i0 is None or i1 is None:
            return np.nan
        return (float(daily.loc[i1, col]) - float(daily.loc[i0, col])) / divisor

    junjul_ej = delta_between("2026-06-30", "2026-07-01", "signed_heat_anomaly_PJ", 1000)
    julaug_ej = delta_between("2026-07-31", "2026-08-01", "signed_heat_anomaly_PJ", 1000)
    junjul_sst = delta_between("2026-06-30", "2026-07-01", "area_weighted_near_surface_anomaly_C")
    julaug_sst = delta_between("2026-07-31", "2026-08-01", "area_weighted_near_surface_anomaly_C")

    max_heat_i = int(np.nanargmax(np.abs(d_heat))) if np.isfinite(d_heat).any() else None
    max_sst_i = int(np.nanargmax(np.abs(d_sst))) if np.isfinite(d_sst).any() else None

    surface_missing_fraction = (
        total_surface_missing / total_surface_wet if total_surface_wet else np.nan
    )

    mean_signed_ej = float(np.nanmean(heat_ej))
    mean_positive_ej = float(np.nanmean(daily["positive_heat_anomaly_PJ"].to_numpy(float) / 1000))
    mean_sst_domain = area_weighted_mean(mean_sst, area, wet2d)

    qc_pass_surface = np.isfinite(mean_sst_domain) and surface_missing_fraction < 1e-4
    qc_pass_closure = np.isfinite(closure_max) and closure_max < 0.10

    heat_change_text = (
        f'{daily.loc[max_heat_i, "date"]} -> {daily.loc[max_heat_i+1, "date"]} = {d_heat[max_heat_i]:+.3f} EJ'
        if max_heat_i is not None else "NON DISPONIBILE"
    )
    sst_change_text = (
        f'{daily.loc[max_sst_i, "date"]} -> {daily.loc[max_sst_i+1, "date"]} = {d_sst[max_sst_i]:+.3f} °C'
        if max_sst_i is not None else "NON DISPONIBILE"
    )
    qc_pass_boundaries = (
        np.isfinite(junjul_ej) and np.isfinite(julaug_ej)
        and abs(junjul_ej) < 20 and abs(julaug_ej) < 20
    )

    overall = qc_pass_surface and qc_pass_closure and qc_pass_boundaries

    qc = f"""QUALITY CONTROL — FINAL OCEAN DATASET
Study period: {times[0].date()} to {times[-1].date()}
Domain: lon {lon.min():.3f} to {lon.max():.3f}; lat {lat.min():.3f} to {lat.max():.3f}

INPUT/METHOD
- 2026 daily thetao
- Copernicus monthly thetao_avg May-September
- Daily climatology interpolated continuously between month-centre anchors
- Copernicus deptho, mask, e1t, e2t, e3t
- Horizontal area = e1t*e2t
- Vertical integration = e3t clipped to min(100 m, local deptho)

GEOMETRY QC
Wet surface cells: {wet_count}
Represented ocean area: {ocean_area_km2:,.1f} km2
Mean absolute vertical closure error: {closure_mean:.6f} m
Maximum absolute vertical closure error: {closure_max:.6f} m
Vertical geometry QC: {"PASS" if qc_pass_closure else "CHECK"}

SURFACE QC
Area-weighted mean near-surface anomaly: {mean_sst_domain:+.3f} °C
Missing near-surface values over wet cells/days: {surface_missing_fraction:.8%}
Surface QC: {"PASS" if qc_pass_surface else "CHECK"}

PERIOD OHC RESULTS
Mean signed heat anomaly: {mean_signed_ej:+.3f} EJ
Mean positive heat anomaly: {mean_positive_ej:.3f} EJ

MONTH-BOUNDARY CONTINUITY
30 Jun -> 1 Jul heat change: {junjul_ej:+.3f} EJ
31 Jul -> 1 Aug heat change: {julaug_ej:+.3f} EJ
30 Jun -> 1 Jul near-surface change: {junjul_sst:+.3f} °C
31 Jul -> 1 Aug near-surface change: {julaug_sst:+.3f} °C
Boundary continuity QC: {"PASS" if qc_pass_boundaries else "CHECK"}

LARGEST SINGLE-DAY CHANGE
Heat: {heat_change_text}
Near-surface anomaly: {sst_change_text}

OVERALL OCEAN QC: {"PASS" if overall else "CHECK REQUIRED"}

INTERPRETATION LIMITS
- This is anomalous ocean heat content, not heat released to the atmosphere.
- Daily OHC states must not be summed through time.
- Warm ocean water is a potential heat/moisture source, not a rainfall or flood forecast.
- Atmospheric transport, evaporation, humidity, convergence, instability and orography must be analysed separately.
- Flood potential additionally requires basin rainfall, antecedent wetness and hydrological response.
"""
    QC_TXT.write_text(qc, encoding="utf-8")

    print("\nFATTO — DATASET OCEANICO FINALE.")
    for p in (DAILY_NC, DAILY_CSV, MEAN_NC, CELLS_CSV, CELLS_GEOJSON, QC_TXT):
        print(f"  {p}")
    print("\n" + qc)


if __name__ == "__main__":
    main()
