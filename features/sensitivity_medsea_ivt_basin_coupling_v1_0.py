#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sensitivity_medsea_ivt_basin_coupling_v1_0.py

Sensitivity test metodologico del coupling:
Mediterraneo × corridoio IVT -> 21 bacini.

Obiettivi:
1) verificare la robustezza della CLASSIFICAZIONE DI SUPPORTO marino su
   tutte le 99,918 righe al variare di ampiezza angolare e distanza;
2) verificare la stabilità NUMERICA di SST' e OHC' corridor-weighted al
   variare anche di sigma angolare e scala di decadimento;
3) NON modificare il prodotto canonico v1.0;
4) NON scegliere automaticamente "il parametro migliore": produce metriche
   per congelare il metodo solo dopo lettura critica.

Scenari one-at-a-time intorno al baseline:
BASE            sigma=22.5 cutoff=45 L=700 Dmax=1600
ANGLE_NARROW    sigma=15   cutoff=30 L=700 Dmax=1600
ANGLE_WIDE      sigma=30   cutoff=60 L=700 Dmax=1600
DIST_SHORT      sigma=22.5 cutoff=45 L=500 Dmax=1200
DIST_LONG       sigma=22.5 cutoff=45 L=900 Dmax=2000
SIGMA_NARROW    sigma=15   cutoff=45 L=700 Dmax=1600
SIGMA_WIDE      sigma=30   cutoff=45 L=700 Dmax=1600
SCALE_SHORT     sigma=22.5 cutoff=45 L=500 Dmax=1600
SCALE_LONG      sigma=22.5 cutoff=45 L=900 Dmax=1600

Support sensitivity:
- usa tutte le 99,918 basin-days;
- usa la maschera marina valida del prodotto di riferimento;
- il baseline geometrico DEVE riprodurre esattamente lo stato di supporto
  del coupling v1.0. Se non lo fa, lo script si ferma con REVIEW.

Numeric sensitivity:
- campione stratificato deterministico:
  a) tutti i basin-days nel top 10% IVT di ciascun recettore;
  b) background ogni 14° giorno stagionale;
  c) unione dei due insiemi;
- ricalcola SST' e OHC' corridor-weighted per ogni scenario;
- confronta con il baseline canonico su righe con supporto comune.

Metriche:
- support_pct su tutto il record;
- Jaccard support vs baseline;
- variazione supporto in punti percentuali;
- Pearson r vs baseline;
- MAE;
- bias medio;
- RMSE normalizzato sulla deviazione standard baseline;
- metriche anche per recettore.

Output:
medsea_historical_analysis/basin_coupling_historical_v1_0/
  sensitivity_v1_0/
    scenario_summary_v1_0.csv
    scenario_by_receptor_v1_0.csv
    sensitivity_audit_v1_0.json
    sensitivity_audit_v1_0.txt

NOTA:
Questo resta un proxy Euleriano/geometrico, non una retrotraiettoria
lagrangiana.
"""

from __future__ import annotations

import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse


START_YEAR = 1987
END_YEAR = 2025
MONTHS = [9, 10, 11, 12]
EXPECTED_ROWS = 99918
EXPECTED_RECEPTORS = 21

N_SECTORS = 16
SECTOR_STEP_DEG = 22.5
SECTOR_CENTERS = np.arange(0.0, 360.0, SECTOR_STEP_DEG)

EARTH_RADIUS_KM = 6371.0088
EARTH_RADIUS_M = EARTH_RADIUS_KM * 1000.0

SCENARIOS = [
    dict(name="BASE",         sigma=22.5, cutoff=45.0, scale=700.0, dmax=1600.0),
    dict(name="ANGLE_NARROW", sigma=15.0, cutoff=30.0, scale=700.0, dmax=1600.0),
    dict(name="ANGLE_WIDE",   sigma=30.0, cutoff=60.0, scale=700.0, dmax=1600.0),
    dict(name="DIST_SHORT",   sigma=22.5, cutoff=45.0, scale=500.0, dmax=1200.0),
    dict(name="DIST_LONG",    sigma=22.5, cutoff=45.0, scale=900.0, dmax=2000.0),
    dict(name="SIGMA_NARROW", sigma=15.0, cutoff=45.0, scale=700.0, dmax=1600.0),
    dict(name="SIGMA_WIDE",   sigma=30.0, cutoff=45.0, scale=700.0, dmax=1600.0),
    dict(name="SCALE_SHORT",  sigma=22.5, cutoff=45.0, scale=500.0, dmax=1600.0),
    dict(name="SCALE_LONG",   sigma=22.5, cutoff=45.0, scale=900.0, dmax=1600.0),
]


def angular_diff_deg(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def cell_edges(centers):
    c = np.asarray(centers, dtype=float)
    mids = (c[:-1] + c[1:]) / 2.0
    first = c[0] + (c[0] - mids[0])
    last = c[-1] + (c[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def spherical_cell_areas(lat, lon):
    lat_e = np.deg2rad(cell_edges(lat))
    lon_e = np.deg2rad(cell_edges(lon))
    lat_factor = np.abs(np.sin(lat_e[1:]) - np.sin(lat_e[:-1]))
    lon_width = np.abs(lon_e[1:] - lon_e[:-1])
    return EARTH_RADIUS_M**2 * lat_factor[:, None] * lon_width[None, :]


def basin_to_grid_distance_bearing(basin_lat, basin_lon, grid_lat, grid_lon):
    lat2d, lon2d = np.meshgrid(
        np.asarray(grid_lat, dtype=float),
        np.asarray(grid_lon, dtype=float),
        indexing="ij",
    )
    lat1 = np.deg2rad(float(basin_lat))
    lon1 = np.deg2rad(float(basin_lon))
    lat2 = np.deg2rad(lat2d)
    lon2 = np.deg2rad(lon2d)
    dlat = lat2 - lat1
    dlon = (lon2 - lon1 + np.pi) % (2*np.pi) - np.pi

    a = (
        np.sin(dlat/2.0)**2
        + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    )
    a = np.clip(a, 0.0, 1.0)
    dist = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

    y = np.sin(dlon) * np.cos(lat2)
    x = (
        np.cos(lat1)*np.sin(lat2)
        - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    )
    bearing = (np.rad2deg(np.arctan2(y, x)) + 360.0) % 360.0

    return dist.ravel(), bearing.ravel()


def build_weight_matrix(centroids, lat, lon, sigma, cutoff, scale, dmax):
    areas = spherical_cell_areas(lat, lon).ravel()
    ncell = len(lat) * len(lon)

    rows, cols, data = [], [], []
    col_idx = 0

    for rec in centroids.itertuples(index=False):
        dist, bearing = basin_to_grid_distance_bearing(
            rec.centroid_lat,
            rec.centroid_lon,
            lat,
            lon,
        )

        for center in SECTOR_CENTERS:
            diff = angular_diff_deg(bearing, center)
            mask = (
                np.isfinite(dist)
                & np.isfinite(bearing)
                & (dist <= dmax)
                & (diff <= cutoff)
            )
            idx = np.flatnonzero(mask)

            if len(idx):
                w = (
                    areas[idx]
                    * np.exp(-dist[idx] / scale)
                    * np.exp(-0.5 * (diff[idx] / sigma) ** 2)
                )
                good = np.isfinite(w) & (w > 0)
                idx = idx[good]
                w = w[good]

                rows.extend(idx.tolist())
                cols.extend([col_idx] * len(idx))
                data.extend(w.astype(float).tolist())

            col_idx += 1

    return sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(ncell, len(centroids) * N_SECTORS),
        dtype=np.float64,
    )


def coord_name(ds, names):
    for c in names:
        if c in ds.coords or c in ds.variables:
            return c
    return None


def time_name(ds):
    for c in ["time", "valid_time", "date"]:
        if c in ds.coords or c in ds.variables:
            try:
                if np.issubdtype(ds[c].dtype, np.datetime64):
                    return c
            except Exception:
                pass
    for c in ds.coords:
        try:
            if np.issubdtype(ds[c].dtype, np.datetime64):
                return c
        except Exception:
            pass
    return None


def grid_signature(ds):
    latn = coord_name(ds, ["latitude", "lat"])
    lonn = coord_name(ds, ["longitude", "lon"])
    if latn is None or lonn is None:
        raise RuntimeError("lat/lon non trovate")
    return (
        latn,
        lonn,
        np.asarray(ds[latn].values, dtype=float),
        np.asarray(ds[lonn].values, dtype=float),
    )


def field_flat(ds, var, latn, lonn, timen, time_indices=None):
    if var not in ds.data_vars:
        raise RuntimeError(f"Variabile assente: {var}")

    da = ds[var]
    extra = [d for d in da.dims if d not in (timen, latn, lonn)]
    for d in extra:
        if da.sizes[d] != 1:
            raise RuntimeError(f"{var}: dimensione extra {d}={da.sizes[d]}")
    if extra:
        da = da.squeeze(extra)

    da = da.transpose(timen, latn, lonn)

    if time_indices is not None:
        da = da.isel({timen: time_indices})

    arr = da.values.astype(np.float64)
    return arr.reshape(arr.shape[0], -1)


def sector_num_den(field, W):
    finite = np.isfinite(field)
    values = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)
    num = (W.T.dot(values.T)).T
    den = (W.T.dot(finite.astype(np.float64).T)).T
    return np.asarray(num, dtype=float), np.asarray(den, dtype=float)


def sector_pair(source_bearing):
    b = np.asarray(source_bearing, dtype=float) % 360.0
    pos = b / SECTOR_STEP_DEG
    low = np.floor(pos).astype(int) % N_SECTORS
    frac = pos - np.floor(pos)
    high = (low + 1) % N_SECTORS
    return low, high, frac


def interpolated_value(num_row, den_row, receptor_idx, low, high, frac):
    c0 = receptor_idx * N_SECTORS + int(low)
    c1 = receptor_idx * N_SECTORS + int(high)
    a0 = 1.0 - float(frac)
    a1 = float(frac)
    num = a0 * num_row[c0] + a1 * num_row[c1]
    den = a0 * den_row[c0] + a1 * den_row[c1]
    if not np.isfinite(den) or den <= 0:
        return np.nan, 0.0
    return float(num / den), float(den)


def interpolated_support(den_vector, receptor_idx, low, high, frac):
    c0 = receptor_idx * N_SECTORS + int(low)
    c1 = receptor_idx * N_SECTORS + int(high)
    den = (1.0-float(frac))*den_vector[c0] + float(frac)*den_vector[c1]
    return bool(np.isfinite(den) and den > 0)


def pearson_safe(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    aa, bb = a[m], b[m]
    if np.std(aa) == 0 or np.std(bb) == 0:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def metric_block(base, test):
    base = np.asarray(base, dtype=float)
    test = np.asarray(test, dtype=float)
    m = np.isfinite(base) & np.isfinite(test)
    n = int(m.sum())
    if n == 0:
        return dict(common_n=0, corr=np.nan, mae=np.nan, bias=np.nan, nrmse=np.nan)

    b = base[m]
    t = test[m]
    diff = t - b
    rmse = float(np.sqrt(np.mean(diff**2)))
    sd = float(np.std(b))
    return dict(
        common_n=n,
        corr=pearson_safe(b, t),
        mae=float(np.mean(np.abs(diff))),
        bias=float(np.mean(diff)),
        nrmse=(rmse / sd if sd > 0 else np.nan),
    )


def main():
    root = Path(__file__).resolve().parent

    base_dir = (
        root / "medsea_historical_analysis"
        / "basin_coupling_historical_v1_0"
    )
    coupling_csv = base_dir / "medsea_ivt_basin_coupling_daily_1987_2025_v1_0.csv"
    coupling_audit = base_dir / "medsea_ivt_basin_coupling_audit_v1_0.json"
    support_audit = base_dir / "support_audit_v1_1" / "support_audit_v1_1.json"

    era5_csv = (
        root / "era5_historical_nw"
        / "basin_features_derived_v1_0"
        / "era5_basin_features_daily_derived_1987_2025_v1_0.csv"
    )

    centroid_csv = (
        root / "medsea_historical_analysis"
        / "basin_coupling_preflight_v1_0"
        / "receptor_centroids_v1_0.csv"
    )

    out_dir = base_dir / "sensitivity_v1_0"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 132)
    print("MEDSEA × IVT -> 21 BACINI — SENSITIVITY TEST v1.0")
    print("=" * 132)

    for p in [coupling_csv, coupling_audit, support_audit, era5_csv, centroid_csv]:
        if not p.exists():
            raise SystemExit(f"File richiesto mancante: {p}")

    ca = json.loads(coupling_audit.read_text(encoding="utf-8"))
    sa = json.loads(support_audit.read_text(encoding="utf-8"))
    if ca.get("overall_status") != "PASS" or sa.get("overall_status") != "PASS":
        raise SystemExit("Coupling/support audit non PASS.")

    coupling = pd.read_csv(coupling_csv)
    coupling["date"] = pd.to_datetime(coupling["date"], format="%Y-%m-%d")

    if len(coupling) != EXPECTED_ROWS:
        raise SystemExit(f"Coupling rows={len(coupling)}")

    cent = pd.read_csv(centroid_csv)
    receptor_ids = cent["receptor_id"].astype(str).tolist()
    rec_index = {rid: i for i, rid in enumerate(receptor_ids)}

    # ERA5: solo colonne per selezionare il campione.
    era = pd.read_csv(
        era5_csv,
        usecols=["date", "receptor_id", "season_day", "ivt_mag_mean_kg_m1_s1"],
    )
    era["date"] = pd.to_datetime(era["date"], format="%Y-%m-%d")

    # Merge chiavi + baseline.
    base = coupling[
        [
            "date",
            "receptor_id",
            "marine_source_bearing_deg",
            "medsea_sst_anom_corridor_c",
            "medsea_ohc_anom_corridor_j_m2",
            "sst_corridor_support_weight",
            "ohc_corridor_support_weight",
        ]
    ].merge(
        era,
        on=["date", "receptor_id"],
        how="left",
        validate="one_to_one",
    )

    if base["season_day"].isna().any():
        raise SystemExit("Merge ERA5/coupling incompleto.")

    # Top 10% IVT per recettore + background deterministico ogni 14 giorni.
    q90 = base.groupby("receptor_id")["ivt_mag_mean_kg_m1_s1"].transform(
        lambda s: s.quantile(0.90)
    )
    sample_mask = (
        (base["ivt_mag_mean_kg_m1_s1"] >= q90)
        | (((base["season_day"].astype(int) - 1) % 14) == 0)
    )
    sample = base.loc[sample_mask].copy().reset_index(drop=True)

    print(f"Righe full record         : {len(base)}")
    print(f"Righe sensitivity sample  : {len(sample)}")
    print(
        f"Sample high-IVT rows      : "
        f"{int((base.loc[sample_mask, 'ivt_mag_mean_kg_m1_s1'] >= q90[sample_mask]).sum())}"
    )

    # Reference grids and reference marine masks.
    sst_ref_path = (
        root / "medsea_historical_analysis"
        / "daily_sst_anomaly"
        / "medsea_sst_anomaly_2025_SepDec.nc"
    )
    ohc_ref_path = (
        root / "medsea_historical_analysis"
        / "monthly_ohc_anomaly"
        / "medsea_ohc_anomaly_2025_SepDec.nc"
    )

    with xr.open_dataset(sst_ref_path, decode_times=True) as ds:
        sst_latn, sst_lonn, lat, lon = grid_signature(ds)
        stn = time_name(ds)
        sst_ref_first = field_flat(
            ds, "sst_anomaly", sst_latn, sst_lonn, stn, [0]
        )[0]
        ref_lat, ref_lon = lat.copy(), lon.copy()

    with xr.open_dataset(ohc_ref_path, decode_times=True) as ds:
        ohc_latn, ohc_lonn, lat2, lon2 = grid_signature(ds)
        otn = time_name(ds)
        ohc_ref_first = field_flat(
            ds, "ohc_anomaly_0_100", ohc_latn, ohc_lonn, otn, [0]
        )[0]

    if (
        lat2.shape != ref_lat.shape
        or lon2.shape != ref_lon.shape
        or not np.allclose(lat2, ref_lat, atol=1e-10, rtol=0)
        or not np.allclose(lon2, ref_lon, atol=1e-10, rtol=0)
    ):
        raise SystemExit("SST e OHC non condividono la stessa griglia.")

    ref_mask = np.isfinite(sst_ref_first) & np.isfinite(ohc_ref_first)

    # Baseline actual full support.
    actual_base_support = (
        (pd.to_numeric(base["sst_corridor_support_weight"], errors="coerce") > 0)
        & (pd.to_numeric(base["ohc_corridor_support_weight"], errors="coerce") > 0)
        & base["medsea_sst_anom_corridor_c"].notna()
        & base["medsea_ohc_anom_corridor_j_m2"].notna()
    ).to_numpy()

    source_bearing_full = pd.to_numeric(
        base["marine_source_bearing_deg"], errors="coerce"
    ).to_numpy()
    low_full, high_full, frac_full = sector_pair(source_bearing_full)
    receptor_index_full = base["receptor_id"].map(rec_index).to_numpy()

    # Baseline sample values for numeric comparisons.
    sample_keys = sample[["date", "receptor_id"]].copy()
    baseline_sample = sample[
        [
            "date",
            "receptor_id",
            "medsea_sst_anom_corridor_c",
            "medsea_ohc_anom_corridor_j_m2",
        ]
    ].rename(
        columns={
            "medsea_sst_anom_corridor_c": "base_sst",
            "medsea_ohc_anom_corridor_j_m2": "base_ohc",
        }
    )

    summary_rows = []
    receptor_rows = []

    baseline_geom_support = None

    for si, sc in enumerate(SCENARIOS, start=1):
        name = sc["name"]
        print("\n" + "-" * 132)
        print(
            f"{si}/{len(SCENARIOS)} {name}: "
            f"sigma={sc['sigma']} cutoff={sc['cutoff']} "
            f"L={sc['scale']} Dmax={sc['dmax']}"
        )

        print("  costruzione matrice pesi...")
        W = build_weight_matrix(
            cent,
            ref_lat,
            ref_lon,
            sc["sigma"],
            sc["cutoff"],
            sc["scale"],
            sc["dmax"],
        )

        # Full-record support geometry using static valid marine mask.
        den_ref = np.asarray(
            W.T.dot(ref_mask.astype(np.float64))
        ).ravel()

        geom_support = np.zeros(len(base), dtype=bool)
        for i in range(len(base)):
            geom_support[i] = interpolated_support(
                den_ref,
                int(receptor_index_full[i]),
                int(low_full[i]),
                int(high_full[i]),
                float(frac_full[i]),
            )

        if name == "BASE":
            baseline_geom_support = geom_support.copy()
            mismatch = int((baseline_geom_support != actual_base_support).sum())
            print(f"  baseline geometry vs actual mismatch: {mismatch}")
            if mismatch != 0:
                # Do not continue: support sensitivity would not be valid.
                report = {
                    "version": "1.0",
                    "overall_status": "REVIEW",
                    "reason": (
                        "Baseline static-mask geometry does not reproduce "
                        "actual v1.0 support classification."
                    ),
                    "mismatch_rows": mismatch,
                }
                (out_dir / "sensitivity_audit_v1_0.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                raise SystemExit(2)

        inter = int((geom_support & baseline_geom_support).sum())
        union = int((geom_support | baseline_geom_support).sum())
        jaccard = inter / union if union else np.nan
        support_n = int(geom_support.sum())
        support_pct = 100.0 * support_n / len(base)
        baseline_support_pct = 100.0 * int(baseline_geom_support.sum()) / len(base)

        print(
            f"  full support: {support_n}/{len(base)} "
            f"({support_pct:.3f}%), "
            f"Jaccard vs BASE={jaccard:.5f}"
        )

        # BASE numeric metrics are identity; no recomputation required.
        if name == "BASE":
            sst_metrics = dict(common_n=int(baseline_sample["base_sst"].notna().sum()),
                               corr=1.0, mae=0.0, bias=0.0, nrmse=0.0)
            ohc_metrics = dict(common_n=int(baseline_sample["base_ohc"].notna().sum()),
                               corr=1.0, mae=0.0, bias=0.0, nrmse=0.0)
            scenario_sample = baseline_sample.copy()
            scenario_sample["test_sst"] = scenario_sample["base_sst"]
            scenario_sample["test_ohc"] = scenario_sample["base_ohc"]
        else:
            # Recompute only the deterministic sensitivity sample.
            scenario_parts = []

            for year in range(START_YEAR, END_YEAR + 1):
                sy = sample[sample["date"].dt.year.eq(year)].copy()
                if sy.empty:
                    continue

                sst_path = (
                    root / "medsea_historical_analysis"
                    / "daily_sst_anomaly"
                    / f"medsea_sst_anomaly_{year}_SepDec.nc"
                )
                ohc_path = (
                    root / "medsea_historical_analysis"
                    / "monthly_ohc_anomaly"
                    / f"medsea_ohc_anomaly_{year}_SepDec.nc"
                )

                # SST selected daily indices
                with xr.open_dataset(sst_path, decode_times=True) as ds:
                    latn, lonn, yy_lat, yy_lon = grid_signature(ds)
                    if not (
                        yy_lat.shape == ref_lat.shape
                        and yy_lon.shape == ref_lon.shape
                        and np.allclose(yy_lat, ref_lat, atol=1e-10, rtol=0)
                        and np.allclose(yy_lon, ref_lon, atol=1e-10, rtol=0)
                    ):
                        raise RuntimeError(f"SST grid changed in {year}")

                    tn = time_name(ds)
                    tt = pd.DatetimeIndex(pd.to_datetime(ds[tn].values)).normalize()
                    day_map = {pd.Timestamp(d): i for i, d in enumerate(tt)}
                    unique_dates = pd.DatetimeIndex(sy["date"].unique()).sort_values()
                    idxs = [day_map[pd.Timestamp(d)] for d in unique_dates]

                    fld = field_flat(
                        ds, "sst_anomaly", latn, lonn, tn, idxs
                    )
                    sst_num, sst_den = sector_num_den(fld, W)
                    sst_row_map = {pd.Timestamp(d): i for i, d in enumerate(unique_dates)}

                # OHC all 4 months
                with xr.open_dataset(ohc_path, decode_times=True) as ds:
                    latn, lonn, yy_lat, yy_lon = grid_signature(ds)
                    tn = time_name(ds)
                    ot = pd.DatetimeIndex(pd.to_datetime(ds[tn].values))
                    month_map = {int(d.month): i for i, d in enumerate(ot)}

                    fld = field_flat(
                        ds, "ohc_anomaly_0_100", latn, lonn, tn, None
                    )
                    ohc_num, ohc_den = sector_num_den(fld, W)

                out_rows = []
                for r in sy.itertuples(index=False):
                    rid = str(r.receptor_id)
                    bi = rec_index[rid]
                    sb = float(r.marine_source_bearing_deg)
                    low, high, frac = sector_pair([sb])
                    low = int(low[0]); high = int(high[0]); frac = float(frac[0])

                    sidx = sst_row_map[pd.Timestamp(r.date)]
                    midx = month_map[int(pd.Timestamp(r.date).month)]

                    sst_v, _ = interpolated_value(
                        sst_num[sidx], sst_den[sidx], bi, low, high, frac
                    )
                    ohc_v, _ = interpolated_value(
                        ohc_num[midx], ohc_den[midx], bi, low, high, frac
                    )

                    out_rows.append({
                        "date": pd.Timestamp(r.date),
                        "receptor_id": rid,
                        "test_sst": sst_v,
                        "test_ohc": ohc_v,
                    })

                scenario_parts.append(pd.DataFrame(out_rows))

            scenario_sample = pd.concat(
                scenario_parts, ignore_index=True
            ).merge(
                baseline_sample,
                on=["date", "receptor_id"],
                how="left",
                validate="one_to_one",
            )

            sst_metrics = metric_block(
                scenario_sample["base_sst"].to_numpy(),
                scenario_sample["test_sst"].to_numpy(),
            )
            ohc_metrics = metric_block(
                scenario_sample["base_ohc"].to_numpy(),
                scenario_sample["test_ohc"].to_numpy(),
            )

        print(
            f"  SST sample: n={sst_metrics['common_n']} "
            f"r={sst_metrics['corr']:.5f} "
            f"MAE={sst_metrics['mae']:.5f} C "
            f"NRMSE={sst_metrics['nrmse']:.5f}"
        )
        print(
            f"  OHC sample: n={ohc_metrics['common_n']} "
            f"r={ohc_metrics['corr']:.5f} "
            f"MAE={ohc_metrics['mae']:.3e} J/m2 "
            f"NRMSE={ohc_metrics['nrmse']:.5f}"
        )

        summary_rows.append({
            "scenario": name,
            "sigma_deg": sc["sigma"],
            "cutoff_deg": sc["cutoff"],
            "distance_scale_km": sc["scale"],
            "max_distance_km": sc["dmax"],
            "full_support_rows": support_n,
            "full_support_pct": support_pct,
            "support_delta_pct_points_vs_base": support_pct - baseline_support_pct,
            "support_jaccard_vs_base": jaccard,
            "sample_rows": len(sample),
            "sst_common_n": sst_metrics["common_n"],
            "sst_corr_vs_base": sst_metrics["corr"],
            "sst_mae_c": sst_metrics["mae"],
            "sst_bias_c": sst_metrics["bias"],
            "sst_nrmse_vs_base_sd": sst_metrics["nrmse"],
            "ohc_common_n": ohc_metrics["common_n"],
            "ohc_corr_vs_base": ohc_metrics["corr"],
            "ohc_mae_j_m2": ohc_metrics["mae"],
            "ohc_bias_j_m2": ohc_metrics["bias"],
            "ohc_nrmse_vs_base_sd": ohc_metrics["nrmse"],
        })

        # By receptor: support on full record + numeric sample metrics.
        for rid in receptor_ids:
            rid_mask = base["receptor_id"].eq(rid).to_numpy()
            n_rid = int(rid_mask.sum())
            supp_rid = int(geom_support[rid_mask].sum())
            base_supp_rid = int(baseline_geom_support[rid_mask].sum())

            ss = scenario_sample[scenario_sample["receptor_id"].eq(rid)]

            sm = metric_block(
                ss["base_sst"].to_numpy(),
                ss["test_sst"].to_numpy(),
            )
            om = metric_block(
                ss["base_ohc"].to_numpy(),
                ss["test_ohc"].to_numpy(),
            )

            inter_r = int(
                (geom_support[rid_mask] & baseline_geom_support[rid_mask]).sum()
            )
            union_r = int(
                (geom_support[rid_mask] | baseline_geom_support[rid_mask]).sum()
            )

            receptor_rows.append({
                "scenario": name,
                "receptor_id": rid,
                "rows": n_rid,
                "support_rows": supp_rid,
                "support_pct": 100.0 * supp_rid / n_rid,
                "support_delta_rows_vs_base": supp_rid - base_supp_rid,
                "support_jaccard_vs_base": (
                    inter_r / union_r if union_r else np.nan
                ),
                "sample_rows": len(ss),
                "sst_common_n": sm["common_n"],
                "sst_corr_vs_base": sm["corr"],
                "sst_mae_c": sm["mae"],
                "sst_nrmse_vs_base_sd": sm["nrmse"],
                "ohc_common_n": om["common_n"],
                "ohc_corr_vs_base": om["corr"],
                "ohc_mae_j_m2": om["mae"],
                "ohc_nrmse_vs_base_sd": om["nrmse"],
            })

        del W, geom_support, scenario_sample
        gc.collect()

    summary = pd.DataFrame(summary_rows)
    by_rec = pd.DataFrame(receptor_rows)

    summary_csv = out_dir / "scenario_summary_v1_0.csv"
    by_rec_csv = out_dir / "scenario_by_receptor_v1_0.csv"
    summary.to_csv(summary_csv, index=False)
    by_rec.to_csv(by_rec_csv, index=False)

    # Internal-consistency PASS only; parameter choice is not automated.
    reasons = []

    base_row = summary[summary["scenario"] == "BASE"].iloc[0]
    actual_support_n = int(actual_base_support.sum())

    if int(base_row["full_support_rows"]) != actual_support_n:
        reasons.append(
            f"baseline_support_geometry={int(base_row['full_support_rows'])} "
            f"actual={actual_support_n}"
        )

    if len(summary) != len(SCENARIOS):
        reasons.append(
            f"scenario_rows={len(summary)} expected={len(SCENARIOS)}"
        )

    if summary[
        ["sst_corr_vs_base", "ohc_corr_vs_base"]
    ].isna().all(axis=None):
        reasons.append("all_numeric_correlations_nan")

    overall = "PASS" if not reasons else "REVIEW"

    report = {
        "version": "1.0",
        "overall_status": overall,
        "full_rows": int(len(base)),
        "sensitivity_sample_rows": int(len(sample)),
        "actual_baseline_supported_rows": actual_support_n,
        "actual_baseline_supported_pct": 100.0 * actual_support_n / len(base),
        "scenarios": SCENARIOS,
        "method_note": (
            "Full-record support sensitivity plus numeric SST/OHC sensitivity "
            "on deterministic high-IVT + background sample."
        ),
        "interpretation_note": (
            "PASS means the sensitivity computation is internally coherent. "
            "It does NOT automatically mean the baseline parameter set is "
            "scientifically optimal. Parameter choice must be based on the "
            "reported stability metrics and later event/model validation."
        ),
        "reasons": reasons,
    }

    json_p = out_dir / "sensitivity_audit_v1_0.json"
    json_p.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "=" * 132,
        "MEDSEA × IVT -> 21 BACINI — SENSITIVITY TEST v1.0",
        "=" * 132,
        f"OVERALL STATUS        : {overall}",
        f"Full record rows      : {len(base)}",
        f"Sensitivity sample    : {len(sample)}",
        f"Baseline supported    : {actual_support_n} "
        f"({100.0*actual_support_n/len(base):.3f}%)",
        "",
        "SCENARIO SUMMARY",
    ]

    show_cols = [
        "scenario",
        "full_support_pct",
        "support_delta_pct_points_vs_base",
        "support_jaccard_vs_base",
        "sst_corr_vs_base",
        "sst_mae_c",
        "sst_nrmse_vs_base_sd",
        "ohc_corr_vs_base",
        "ohc_nrmse_vs_base_sd",
    ]

    lines.append(summary[show_cols].to_string(index=False))
    lines += [
        "",
        "NOTA",
        "PASS = calcolo coerente; non seleziona automaticamente il baseline.",
        "Il coupling resta un proxy Euleriano/geometrico, non una retrotraiettoria.",
        "",
        f"Summary CSV: {summary_csv}",
        f"By receptor: {by_rec_csv}",
    ]

    txt_p = out_dir / "sensitivity_audit_v1_0.txt"
    txt_p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 132)
    print(summary[show_cols].to_string(index=False))
    print("=" * 132)
    print(f"OVERALL STATUS : {overall}")
    print(f"Output         : {out_dir}")

    if reasons:
        print("REASONS:")
        for r in reasons:
            print(f"  - {r}")

    if overall != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
