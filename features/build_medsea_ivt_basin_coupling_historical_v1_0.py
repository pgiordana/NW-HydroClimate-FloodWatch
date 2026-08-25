#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_medsea_ivt_basin_coupling_historical_v1_0.py

Costruisce feature marine "transport-conditioned" per i 21 recettori:
Mediterraneo storico -> corridoio di provenienza -> bacino.

INPUT:
- ERA5 derived basin features (99,918 righe, PASS)
- SST anomaly giornaliera spaziale, Sep-Dic 1987-2025
- OHC/Tmean 0-100 m mensile spaziale, Sep-Dic 1987-2025
- centroidi dei 21 recettori dal preflight PASS

METODO:
- il vettore IVT ERA5 indica la direzione VERSO CUI il vapore viene trasportato;
- la direzione di provenienza ("source bearing") è quindi IVT_direction + 180°;
- per ogni bacino e direzione sorgente si costruisce un corridoio marino
  con pesi continui area × distanza × deviazione angolare;
- 16 settori direzionali statici (22.5°) vengono precomputati;
- per ogni giorno si interpola tra i due settori adiacenti sulla base del
  source bearing reale, evitando salti artificiali di settore;
- nessun nearest-cell fallback;
- se il corridoio non intercetta celle marine valide, la feature resta NaN;
- SST è giornaliera; OHC/Tmean sono mensili e vengono associati ai giorni
  dello stesso mese senza interpolazione temporale;
- il prodotto è un PROXY di sorgente marina condizionato dal trasporto,
  non una retrotraiettoria lagrangiana.

PESI GEOMETRICI:
- area sferica della cella;
- decadimento con distanza exp(-d / 700 km);
- kernel angolare gaussian sigma 22.5°;
- cutoff angolare ±45°;
- distanza massima 1600 km.

FEATURE:
- medsea_sst_anom_corridor_c
- medsea_ohc_anom_corridor_j_m2
- medsea_ohc_corridor_j_m2
- medsea_tmean_0_100_corridor_c
- medsea_tmean_anom_0_100_corridor_c
- source_bearing_deg
- source_sector_low_deg / high_deg / interpolation_fraction
- support denominators SST/OHC
- interaction proxies SST'×IVT e OHC'×IVT

OUTPUT:
medsea_historical_analysis/basin_coupling_historical_v1_0/
  annual/YYYY/medsea_ivt_basin_coupling_YYYY_v1_0.csv
  medsea_ivt_basin_coupling_daily_1987_2025_v1_0.csv
  medsea_ivt_basin_coupling_manifest_v1_0.csv
  medsea_ivt_basin_coupling_audit_v1_0.json
  medsea_ivt_basin_coupling_audit_v1_0.txt

RIGHE ATTESE: 4,758 giorni × 21 bacini = 99,918.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

try:
    from scipy import sparse
except Exception as exc:
    raise SystemExit(
        "scipy non disponibile nella .venv principale: "
        f"{exc!r}"
    )


START_YEAR = 1987
END_YEAR = 2025
MONTHS = [9, 10, 11, 12]
EXPECTED_RECEPTORS = 21
EXPECTED_DAYS = 4758
EXPECTED_ROWS = EXPECTED_DAYS * EXPECTED_RECEPTORS

N_SECTORS = 16
SECTOR_STEP_DEG = 360.0 / N_SECTORS
SECTOR_CENTERS = np.arange(0.0, 360.0, SECTOR_STEP_DEG)

ANGULAR_SIGMA_DEG = 22.5
ANGULAR_CUTOFF_DEG = 45.0
DISTANCE_SCALE_KM = 700.0
MAX_DISTANCE_KM = 1600.0

EARTH_RADIUS_KM = 6371.0088
EARTH_RADIUS_M = EARTH_RADIUS_KM * 1000.0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--force",
        action="store_true",
        help="Ricalcola anche gli anni già prodotti e validi.",
    )
    ap.add_argument(
        "--start-year",
        type=int,
        default=START_YEAR,
    )
    ap.add_argument(
        "--end-year",
        type=int,
        default=END_YEAR,
    )
    return ap.parse_args()


def angular_diff_deg(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def coord_name(ds, candidates):
    for c in candidates:
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


def cell_edges(centers):
    c = np.asarray(centers, dtype=float)

    if c.ndim != 1 or len(c) < 2:
        raise ValueError("Coordinate griglia non 1D o troppo corte.")

    mids = (c[:-1] + c[1:]) / 2.0
    first = c[0] + (c[0] - mids[0])
    last = c[-1] + (c[-1] - mids[-1])

    return np.concatenate([[first], mids, [last]])


def spherical_cell_areas(lat, lon):
    """
    Area sferica esatta per celle lat/lon rettangolari.
    Restituisce matrice [nlat, nlon] in m2.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    lat_e = np.deg2rad(cell_edges(lat))
    lon_e = np.deg2rad(cell_edges(lon))

    lat_factor = np.abs(
        np.sin(lat_e[1:]) - np.sin(lat_e[:-1])
    )
    lon_width = np.abs(
        lon_e[1:] - lon_e[:-1]
    )

    return (
        EARTH_RADIUS_M ** 2
        * lat_factor[:, None]
        * lon_width[None, :]
    )


def basin_to_grid_distance_bearing(
    basin_lat_deg,
    basin_lon_deg,
    grid_lat,
    grid_lon,
):
    """
    Distanza haversine e bearing iniziale dal bacino verso ogni centro cella.
    """
    lat2d, lon2d = np.meshgrid(
        np.asarray(grid_lat, dtype=float),
        np.asarray(grid_lon, dtype=float),
        indexing="ij",
    )

    lat1 = np.deg2rad(float(basin_lat_deg))
    lon1 = np.deg2rad(float(basin_lon_deg))
    lat2 = np.deg2rad(lat2d)
    lon2 = np.deg2rad(lon2d)

    dlat = lat2 - lat1
    dlon = (lon2 - lon1 + np.pi) % (2 * np.pi) - np.pi

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)

    dist = (
        2.0 * EARTH_RADIUS_KM
        * np.arcsin(np.sqrt(a))
    )

    y = np.sin(dlon) * np.cos(lat2)
    x = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )

    bearing = (
        np.rad2deg(np.arctan2(y, x)) + 360.0
    ) % 360.0

    return dist, bearing


def build_sector_weight_matrix(
    centroid_df,
    lat,
    lon,
):
    """
    Sparse matrix:
      rows = flattened marine-grid cells
      cols = receptor × 16 sector centers
    Weight = spherical cell area × exp(-d/L) × angular gaussian,
    with distance/angular cutoffs.

    Land is NOT hard-coded here: the finite-value mask of each marine field
    determines actual support at runtime.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    areas = spherical_cell_areas(lat, lon).ravel()
    ncell = len(lat) * len(lon)

    rows = []
    cols = []
    data = []

    column_meta = []

    col_idx = 0

    for rec in centroid_df.itertuples(index=False):
        dist, bearing = basin_to_grid_distance_bearing(
            rec.centroid_lat,
            rec.centroid_lon,
            lat,
            lon,
        )

        dist_f = dist.ravel()
        bear_f = bearing.ravel()

        for center in SECTOR_CENTERS:
            diff = angular_diff_deg(
                bear_f,
                center,
            )

            mask = (
                np.isfinite(dist_f)
                & np.isfinite(bear_f)
                & (dist_f <= MAX_DISTANCE_KM)
                & (diff <= ANGULAR_CUTOFF_DEG)
            )

            idx = np.flatnonzero(mask)

            if len(idx):
                d = dist_f[idx]
                a = diff[idx]

                geom_w = (
                    areas[idx]
                    * np.exp(-d / DISTANCE_SCALE_KM)
                    * np.exp(
                        -0.5
                        * (a / ANGULAR_SIGMA_DEG) ** 2
                    )
                )

                positive = (
                    np.isfinite(geom_w)
                    & (geom_w > 0)
                )

                idx = idx[positive]
                geom_w = geom_w[positive]

                rows.extend(idx.tolist())
                cols.extend(
                    [col_idx] * len(idx)
                )
                data.extend(
                    geom_w.astype(float).tolist()
                )

            column_meta.append({
                "column_idx": col_idx,
                "receptor_id": str(rec.receptor_id),
                "sector_center_deg": float(center),
            })

            col_idx += 1

    W = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(
            ncell,
            len(centroid_df) * N_SECTORS,
        ),
        dtype=np.float64,
    )

    meta = pd.DataFrame(column_meta)

    return W, meta


def grid_signature(ds):
    latn = coord_name(ds, ["latitude", "lat"])
    lonn = coord_name(ds, ["longitude", "lon"])

    if latn is None or lonn is None:
        raise RuntimeError("Coordinate lat/lon non trovate.")

    return (
        latn,
        lonn,
        np.asarray(ds[latn].values, dtype=float),
        np.asarray(ds[lonn].values, dtype=float),
    )


def assert_grid_equal(
    lat,
    lon,
    ref_lat,
    ref_lon,
    label,
):
    if lat.shape != ref_lat.shape or lon.shape != ref_lon.shape:
        raise RuntimeError(
            f"{label}: shape griglia cambiata."
        )

    if not np.allclose(
        lat,
        ref_lat,
        atol=1e-10,
        rtol=0,
    ):
        raise RuntimeError(
            f"{label}: latitudine griglia cambiata."
        )

    if not np.allclose(
        lon,
        ref_lon,
        atol=1e-10,
        rtol=0,
    ):
        raise RuntimeError(
            f"{label}: longitudine griglia cambiata."
        )


def field_2d_time(
    ds,
    var,
    latn,
    lonn,
    timen,
):
    if var not in ds.data_vars:
        raise RuntimeError(
            f"Variabile assente: {var}"
        )

    da = ds[var]

    needed = {timen, latn, lonn}

    if not needed.issubset(
        set(da.dims)
    ):
        raise RuntimeError(
            f"{var}: dims={da.dims}, attese {needed}"
        )

    extra = [
        d for d in da.dims
        if d not in (timen, latn, lonn)
    ]

    if extra:
        # consentiamo solo dimensioni singleton
        for d in extra:
            if da.sizes[d] != 1:
                raise RuntimeError(
                    f"{var}: dimensione extra non singleton {d}={da.sizes[d]}"
                )
        da = da.squeeze(extra)

    arr = da.transpose(
        timen,
        latn,
        lonn,
    ).values.astype(np.float64)

    return arr.reshape(
        arr.shape[0],
        arr.shape[1] * arr.shape[2],
    )


def sector_num_den(
    field_flat,
    W,
):
    """
    field_flat: [ntime, ncell]
    W: [ncell, n_receptor*16]

    Return numerator, denominator.
    """
    finite = np.isfinite(field_flat)

    values = np.nan_to_num(
        field_flat,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # sparse.T @ dense.T -> [ncols, ntime]
    num = (
        W.T.dot(values.T)
    ).T
    den = (
        W.T.dot(
            finite.astype(np.float64).T
        )
    ).T

    return (
        np.asarray(num, dtype=float),
        np.asarray(den, dtype=float),
    )


def sector_pair(
    source_bearing_deg,
):
    b = (
        np.asarray(
            source_bearing_deg,
            dtype=float,
        )
        % 360.0
    )

    pos = b / SECTOR_STEP_DEG
    low_idx = np.floor(pos).astype(int) % N_SECTORS
    frac = pos - np.floor(pos)
    high_idx = (
        low_idx + 1
    ) % N_SECTORS

    low_deg = low_idx * SECTOR_STEP_DEG
    high_deg = high_idx * SECTOR_STEP_DEG

    return (
        low_idx,
        high_idx,
        frac,
        low_deg,
        high_deg,
    )


def interpolate_corridor(
    num_row,
    den_row,
    receptor_index,
    low_sector_idx,
    high_sector_idx,
    frac,
):
    """
    Interpola NUMERATORE e DENOMINATORE tra i due settori adiacenti,
    poi normalizza. Non fa fallback nearest.
    """
    c0 = receptor_index * N_SECTORS + int(low_sector_idx)
    c1 = receptor_index * N_SECTORS + int(high_sector_idx)

    a0 = 1.0 - float(frac)
    a1 = float(frac)

    num = (
        a0 * num_row[c0]
        + a1 * num_row[c1]
    )
    den = (
        a0 * den_row[c0]
        + a1 * den_row[c1]
    )

    if (
        not np.isfinite(den)
        or den <= 0
    ):
        return np.nan, 0.0

    return float(num / den), float(den)


def load_year_era5(
    era5_all,
    year,
):
    y = era5_all[
        era5_all["date"].dt.year.eq(year)
    ].copy()

    expected = 122 * EXPECTED_RECEPTORS

    if len(y) != expected:
        raise RuntimeError(
            f"ERA5 {year}: righe={len(y)}, attese={expected}"
        )

    return y.sort_values(
        ["date", "receptor_id"]
    ).reset_index(drop=True)


def process_year(
    year,
    era5_year,
    receptor_ids,
    sst_path,
    ohc_path,
    W_sst,
    W_ohc,
    ref_sst_lat,
    ref_sst_lon,
    ref_ohc_lat,
    ref_ohc_lon,
):
    # ----------------------------------------------------------
    # SST daily
    # ----------------------------------------------------------
    with xr.open_dataset(
        sst_path,
        decode_times=True,
    ) as ds_sst:
        sst_latn, sst_lonn, lat, lon = grid_signature(ds_sst)

        assert_grid_equal(
            lat,
            lon,
            ref_sst_lat,
            ref_sst_lon,
            f"SST {year}",
        )

        tn = time_name(ds_sst)
        if tn is None:
            raise RuntimeError(
                f"SST {year}: time coord assente."
            )

        sst_t = pd.DatetimeIndex(
            pd.to_datetime(
                ds_sst[tn].values
            )
        )

        sst_field = field_2d_time(
            ds_sst,
            "sst_anomaly",
            sst_latn,
            sst_lonn,
            tn,
        )

        sst_num, sst_den = sector_num_den(
            sst_field,
            W_sst,
        )

    # ----------------------------------------------------------
    # OHC monthly
    # ----------------------------------------------------------
    with xr.open_dataset(
        ohc_path,
        decode_times=True,
    ) as ds_ohc:
        ohc_latn, ohc_lonn, lat, lon = grid_signature(ds_ohc)

        assert_grid_equal(
            lat,
            lon,
            ref_ohc_lat,
            ref_ohc_lon,
            f"OHC {year}",
        )

        tn = time_name(ds_ohc)
        if tn is None:
            raise RuntimeError(
                f"OHC {year}: time coord assente."
            )

        ohc_t = pd.DatetimeIndex(
            pd.to_datetime(
                ds_ohc[tn].values
            )
        )

        fields = {}

        for var in [
            "ohc_0_100",
            "ohc_anomaly_0_100",
            "tmean_0_100",
            "tmean_anomaly_0_100",
        ]:
            fld = field_2d_time(
                ds_ohc,
                var,
                ohc_latn,
                ohc_lonn,
                tn,
            )
            fields[var] = sector_num_den(
                fld,
                W_ohc,
            )

    # Exact expected calendars.
    expected_dates = pd.date_range(
        f"{year}-09-01",
        f"{year}-12-31",
        freq="D",
    )

    if not sst_t.normalize().equals(
        expected_dates
    ):
        raise RuntimeError(
            f"SST {year}: calendario non coincide con Sep-Dic."
        )

    if len(ohc_t) != 4:
        raise RuntimeError(
            f"OHC {year}: timestamp mensili={len(ohc_t)}, attesi=4."
        )

    ohc_month_index = {
        int(ts.month): i
        for i, ts in enumerate(ohc_t)
    }

    if sorted(
        ohc_month_index.keys()
    ) != MONTHS:
        raise RuntimeError(
            f"OHC {year}: mesi={sorted(ohc_month_index.keys())}"
        )

    sst_day_index = {
        pd.Timestamp(ts).normalize(): i
        for i, ts in enumerate(sst_t)
    }

    rec_idx = {
        rid: i
        for i, rid in enumerate(receptor_ids)
    }

    rows = []

    for r in era5_year.itertuples(index=False):
        rid = str(r.receptor_id)
        date = pd.Timestamp(r.date).normalize()
        b = rec_idx[rid]

        ivt_e = float(
            r.ivt_e_mean_kg_m1_s1
        )
        ivt_n = float(
            r.ivt_n_mean_kg_m1_s1
        )
        ivt_mag = float(
            r.ivt_mag_mean_kg_m1_s1
        )

        # Direction TOWARD which the moisture flux points.
        transport_bearing = (
            math.degrees(
                math.atan2(ivt_e, ivt_n)
            )
            + 360.0
        ) % 360.0

        # Direction FROM which the moisture is arriving.
        source_bearing = (
            transport_bearing + 180.0
        ) % 360.0

        (
            low_idx,
            high_idx,
            frac,
            low_deg,
            high_deg,
        ) = sector_pair(
            [source_bearing]
        )

        low_idx = int(low_idx[0])
        high_idx = int(high_idx[0])
        frac = float(frac[0])
        low_deg = float(low_deg[0])
        high_deg = float(high_deg[0])

        si = sst_day_index[date]

        sst_val, sst_support = interpolate_corridor(
            sst_num[si],
            sst_den[si],
            b,
            low_idx,
            high_idx,
            frac,
        )

        mi = ohc_month_index[int(date.month)]

        ohc_abs, ohc_abs_support = interpolate_corridor(
            fields["ohc_0_100"][0][mi],
            fields["ohc_0_100"][1][mi],
            b,
            low_idx,
            high_idx,
            frac,
        )

        ohc_anom, ohc_anom_support = interpolate_corridor(
            fields["ohc_anomaly_0_100"][0][mi],
            fields["ohc_anomaly_0_100"][1][mi],
            b,
            low_idx,
            high_idx,
            frac,
        )

        tmean_abs, tmean_abs_support = interpolate_corridor(
            fields["tmean_0_100"][0][mi],
            fields["tmean_0_100"][1][mi],
            b,
            low_idx,
            high_idx,
            frac,
        )

        tmean_anom, tmean_anom_support = interpolate_corridor(
            fields["tmean_anomaly_0_100"][0][mi],
            fields["tmean_anomaly_0_100"][1][mi],
            b,
            low_idx,
            high_idx,
            frac,
        )

        rows.append({
            "date": date,
            "receptor_id": rid,
            "ivt_transport_bearing_deg": transport_bearing,
            "marine_source_bearing_deg": source_bearing,
            "marine_sector_low_deg": low_deg,
            "marine_sector_high_deg": high_deg,
            "marine_sector_interp_fraction": frac,
            "medsea_sst_anom_corridor_c": sst_val,
            "medsea_ohc_0_100_corridor_j_m2": ohc_abs,
            "medsea_ohc_anom_corridor_j_m2": ohc_anom,
            "medsea_tmean_0_100_corridor_c": tmean_abs,
            "medsea_tmean_anom_0_100_corridor_c": tmean_anom,
            "sst_corridor_support_weight": sst_support,
            "ohc_corridor_support_weight": ohc_anom_support,
            "ohc_abs_corridor_support_weight": ohc_abs_support,
            "tmean_corridor_support_weight": tmean_abs_support,
            "tmean_anom_corridor_support_weight": tmean_anom_support,
            "sst_anom_x_ivt_proxy": (
                sst_val * ivt_mag
                if np.isfinite(sst_val)
                else np.nan
            ),
            "ohc_anom_x_ivt_proxy": (
                ohc_anom * ivt_mag
                if np.isfinite(ohc_anom)
                else np.nan
            ),
            "ivt_mag_mean_kg_m1_s1": ivt_mag,
        })

    out = pd.DataFrame(rows)

    return out


def audit_year(df, year):
    reasons = []

    expected = 122 * EXPECTED_RECEPTORS

    if len(df) != expected:
        reasons.append(
            f"rows={len(df)} expected={expected}"
        )

    dup = int(
        df.duplicated(
            ["date", "receptor_id"]
        ).sum()
    )

    if dup:
        reasons.append(
            f"duplicate_keys={dup}"
        )

    if df["receptor_id"].nunique() != EXPECTED_RECEPTORS:
        reasons.append(
            f"receptors={df['receptor_id'].nunique()}"
        )

    if not pd.to_datetime(
        df["date"]
    ).dt.month.isin(MONTHS).all():
        reasons.append(
            "months_outside_sep_dec"
        )

    core = [
        "marine_source_bearing_deg",
        "medsea_sst_anom_corridor_c",
        "medsea_ohc_anom_corridor_j_m2",
    ]

    all_nan = [
        c for c in core
        if df[c].isna().all()
    ]

    if all_nan:
        reasons.append(
            f"all_nan={all_nan}"
        )

    return {
        "status": (
            "PASS"
            if not reasons
            else "REVIEW"
        ),
        "reasons": reasons,
        "rows": len(df),
        "duplicate_keys": dup,
        "sst_nan": int(
            df["medsea_sst_anom_corridor_c"].isna().sum()
        ),
        "ohc_nan": int(
            df["medsea_ohc_anom_corridor_j_m2"].isna().sum()
        ),
    }


def main():
    args = parse_args()

    if (
        args.start_year < START_YEAR
        or args.end_year > END_YEAR
        or args.start_year > args.end_year
    ):
        raise SystemExit(
            f"Intervallo consentito: {START_YEAR}-{END_YEAR}"
        )

    root = Path(__file__).resolve().parent

    preflight = (
        root / "medsea_historical_analysis"
        / "basin_coupling_preflight_v1_0"
        / "medsea_basin_coupling_preflight_v1_0.json"
    )

    if not preflight.exists():
        raise SystemExit(
            "Preflight MedSea-basin v1.0 non trovato."
        )

    pf = json.loads(
        preflight.read_text(
            encoding="utf-8"
        )
    )

    if pf.get("overall_status") != "PASS":
        raise SystemExit(
            f"Preflight non PASS: {pf.get('overall_status')}"
        )

    era5_audit = (
        root / "era5_historical_nw"
        / "basin_features_derived_v1_0"
        / "era5_basin_features_derived_audit_v1_0.json"
    )

    if not era5_audit.exists():
        raise SystemExit(
            "Audit ERA5 derived non trovato."
        )

    ea = json.loads(
        era5_audit.read_text(
            encoding="utf-8"
        )
    )

    if ea.get("overall_status") != "PASS":
        raise SystemExit(
            f"ERA5 derived non PASS: {ea.get('overall_status')}"
        )

    era5_path = (
        root / "era5_historical_nw"
        / "basin_features_derived_v1_0"
        / "era5_basin_features_daily_derived_1987_2025_v1_0.csv"
    )

    usecols = [
        "date",
        "receptor_id",
        "ivt_e_mean_kg_m1_s1",
        "ivt_n_mean_kg_m1_s1",
        "ivt_mag_mean_kg_m1_s1",
    ]

    era5 = pd.read_csv(
        era5_path,
        usecols=usecols,
    )

    era5["date"] = pd.to_datetime(
        era5["date"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    if era5["date"].isna().any():
        raise SystemExit(
            f"ERA5 date invalide: {int(era5['date'].isna().sum())}"
        )

    centroid_path = (
        root / "medsea_historical_analysis"
        / "basin_coupling_preflight_v1_0"
        / "receptor_centroids_v1_0.csv"
    )

    cent = pd.read_csv(
        centroid_path
    )

    receptor_ids = (
        cent["receptor_id"]
        .astype(str)
        .tolist()
    )

    if len(receptor_ids) != EXPECTED_RECEPTORS:
        raise SystemExit(
            f"Centroidi recettori={len(receptor_ids)}"
        )

    # ------------------------------------------------------
    # Reference grids from 2025
    # ------------------------------------------------------
    ref_sst_path = (
        root / "medsea_historical_analysis"
        / "daily_sst_anomaly"
        / "medsea_sst_anomaly_2025_SepDec.nc"
    )

    ref_ohc_path = (
        root / "medsea_historical_analysis"
        / "monthly_ohc_anomaly"
        / "medsea_ohc_anomaly_2025_SepDec.nc"
    )

    with xr.open_dataset(
        ref_sst_path,
        decode_times=True,
    ) as ds:
        _, _, ref_sst_lat, ref_sst_lon = grid_signature(ds)

    with xr.open_dataset(
        ref_ohc_path,
        decode_times=True,
    ) as ds:
        _, _, ref_ohc_lat, ref_ohc_lon = grid_signature(ds)

    print("=" * 128)
    print("MEDSEA × IVT -> 21 BACINI — COUPLING STORICO v1.0")
    print("=" * 128)
    print(
        f"Periodo     : Sep-Dic {args.start_year}-{args.end_year}"
    )
    print(
        f"Settori     : {N_SECTORS} ({SECTOR_STEP_DEG:g}°)"
    )
    print(
        f"Kernel      : sigma={ANGULAR_SIGMA_DEG:g}°, "
        f"cutoff=±{ANGULAR_CUTOFF_DEG:g}°, "
        f"L={DISTANCE_SCALE_KM:g} km, "
        f"Dmax={MAX_DISTANCE_KM:g} km"
    )
    print(
        "Metodo      : transport-conditioned corridor; no nearest fallback"
    )
    print("=" * 128)

    print("Costruzione matrice corridoi SST...")
    W_sst, meta_sst = build_sector_weight_matrix(
        cent,
        ref_sst_lat,
        ref_sst_lon,
    )

    print(
        f"  SST grid cells={W_sst.shape[0]}, "
        f"columns={W_sst.shape[1]}, "
        f"nnz={W_sst.nnz}"
    )

    print("Costruzione matrice corridoi OHC...")
    W_ohc, meta_ohc = build_sector_weight_matrix(
        cent,
        ref_ohc_lat,
        ref_ohc_lon,
    )

    print(
        f"  OHC grid cells={W_ohc.shape[0]}, "
        f"columns={W_ohc.shape[1]}, "
        f"nnz={W_ohc.nnz}"
    )

    out_dir = (
        root / "medsea_historical_analysis"
        / "basin_coupling_historical_v1_0"
    )
    annual_dir = out_dir / "annual"
    annual_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    meta_sst.to_csv(
        out_dir
        / "sst_corridor_sector_columns_v1_0.csv",
        index=False,
    )

    meta_ohc.to_csv(
        out_dir
        / "ohc_corridor_sector_columns_v1_0.csv",
        index=False,
    )

    manifest_rows = []

    for year in range(
        args.start_year,
        args.end_year + 1,
    ):
        ydir = annual_dir / str(year)
        ydir.mkdir(
            parents=True,
            exist_ok=True,
        )

        out_y = (
            ydir
            / f"medsea_ivt_basin_coupling_{year}_v1_0.csv"
        )

        if out_y.exists() and not args.force:
            try:
                old = pd.read_csv(
                    out_y,
                    parse_dates=["date"],
                )
                ay = audit_year(
                    old,
                    year,
                )

                if ay["status"] == "PASS":
                    print(
                        f"{year} | SKIP PASS | "
                        f"rows={ay['rows']} "
                        f"| SST NaN={ay['sst_nan']} "
                        f"| OHC NaN={ay['ohc_nan']}"
                    )
                    manifest_rows.append({
                        "year": year,
                        "status": "PASS_EXISTING",
                        "rows": ay["rows"],
                        "sst_nan": ay["sst_nan"],
                        "ohc_nan": ay["ohc_nan"],
                        "path": str(out_y),
                        "reasons": "",
                    })
                    continue
            except Exception as exc:
                print(
                    f"{year} | prodotto esistente non valido, ricalcolo: {exc!r}"
                )

        print(
            f"{year} | elaborazione..."
        )

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

        if not sst_path.exists():
            raise FileNotFoundError(
                sst_path
            )
        if not ohc_path.exists():
            raise FileNotFoundError(
                ohc_path
            )

        era_y = load_year_era5(
            era5,
            year,
        )

        try:
            ydf = process_year(
                year,
                era_y,
                receptor_ids,
                sst_path,
                ohc_path,
                W_sst,
                W_ohc,
                ref_sst_lat,
                ref_sst_lon,
                ref_ohc_lat,
                ref_ohc_lon,
            )

            ay = audit_year(
                ydf,
                year,
            )

            if ay["status"] != "PASS":
                raise RuntimeError(
                    f"Audit anno REVIEW: {ay['reasons']}"
                )

            tmp = out_y.with_suffix(
                ".csv.tmp"
            )
            ydf.to_csv(
                tmp,
                index=False,
            )
            tmp.replace(
                out_y
            )

            manifest_rows.append({
                "year": year,
                "status": "PASS",
                "rows": ay["rows"],
                "sst_nan": ay["sst_nan"],
                "ohc_nan": ay["ohc_nan"],
                "path": str(out_y),
                "reasons": "",
            })

            print(
                f"{year} | PASS | rows={ay['rows']} "
                f"| SST NaN={ay['sst_nan']} "
                f"| OHC NaN={ay['ohc_nan']}"
            )

        except Exception as exc:
            manifest_rows.append({
                "year": year,
                "status": "FAIL",
                "rows": None,
                "sst_nan": None,
                "ohc_nan": None,
                "path": str(out_y),
                "reasons": repr(exc),
            })

            pd.DataFrame(
                manifest_rows
            ).to_csv(
                out_dir
                / "medsea_ivt_basin_coupling_manifest_v1_0.csv",
                index=False,
            )

            print(
                f"{year} | FAIL | {exc!r}"
            )
            raise

    # ------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------
    print("\nCONSOLIDAMENTO STORICO...")

    parts = []

    for year in range(
        args.start_year,
        args.end_year + 1,
    ):
        p = (
            annual_dir
            / str(year)
            / f"medsea_ivt_basin_coupling_{year}_v1_0.csv"
        )

        if not p.exists():
            raise RuntimeError(
                f"Manca annuale: {p}"
            )

        x = pd.read_csv(
            p,
            parse_dates=["date"],
        )

        ay = audit_year(
            x,
            year,
        )

        if ay["status"] != "PASS":
            raise RuntimeError(
                f"Annuale {year} non PASS: {ay['reasons']}"
            )

        parts.append(
            x
        )

    full = pd.concat(
        parts,
        ignore_index=True,
    ).sort_values(
        ["date", "receptor_id"]
    ).reset_index(drop=True)

    expected_rows_run = (
        (args.end_year - args.start_year + 1)
        * 122
        * EXPECTED_RECEPTORS
    )

    reasons = []

    if len(full) != expected_rows_run:
        reasons.append(
            f"rows={len(full)} expected={expected_rows_run}"
        )

    dup = int(
        full.duplicated(
            ["date", "receptor_id"]
        ).sum()
    )

    if dup:
        reasons.append(
            f"duplicate_keys={dup}"
        )

    if full["receptor_id"].nunique() != EXPECTED_RECEPTORS:
        reasons.append(
            f"receptors={full['receptor_id'].nunique()}"
        )

    # Missing corridor features are allowed ONLY when support=0.
    support_consistency = {}

    for value_col, support_col in [
        (
            "medsea_sst_anom_corridor_c",
            "sst_corridor_support_weight",
        ),
        (
            "medsea_ohc_anom_corridor_j_m2",
            "ohc_corridor_support_weight",
        ),
    ]:
        bad_missing = int(
            (
                full[value_col].isna()
                & (full[support_col] > 0)
            ).sum()
        )
        bad_present = int(
            (
                full[value_col].notna()
                & ~(full[support_col] > 0)
            ).sum()
        )

        support_consistency[value_col] = {
            "nan_with_positive_support": bad_missing,
            "value_without_positive_support": bad_present,
        }

        if bad_missing or bad_present:
            reasons.append(
                f"{value_col}_support_inconsistent="
                f"{bad_missing}/{bad_present}"
            )

    overall = (
        "PASS"
        if not reasons
        else "REVIEW"
    )

    final_csv = (
        out_dir
        / f"medsea_ivt_basin_coupling_daily_{args.start_year}_{args.end_year}_v1_0.csv"
    )

    tmp = final_csv.with_suffix(
        ".csv.tmp"
    )
    full.to_csv(
        tmp,
        index=False,
    )
    tmp.replace(
        final_csv
    )

    manifest_path = (
        out_dir
        / "medsea_ivt_basin_coupling_manifest_v1_0.csv"
    )

    pd.DataFrame(
        manifest_rows
    ).to_csv(
        manifest_path,
        index=False,
    )

    report = {
        "version": "1.0",
        "overall_status": overall,
        "period": f"Sep-Dec {args.start_year}-{args.end_year}",
        "expected_rows": expected_rows_run,
        "actual_rows": int(len(full)),
        "receptors": int(full["receptor_id"].nunique()),
        "duplicate_keys": dup,
        "sst_nan_rows": int(
            full["medsea_sst_anom_corridor_c"].isna().sum()
        ),
        "ohc_nan_rows": int(
            full["medsea_ohc_anom_corridor_j_m2"].isna().sum()
        ),
        "support_consistency": support_consistency,
        "parameters": {
            "n_sectors": N_SECTORS,
            "sector_step_deg": SECTOR_STEP_DEG,
            "angular_sigma_deg": ANGULAR_SIGMA_DEG,
            "angular_cutoff_deg": ANGULAR_CUTOFF_DEG,
            "distance_scale_km": DISTANCE_SCALE_KM,
            "max_distance_km": MAX_DISTANCE_KM,
        },
        "method_note": (
            "Transport-conditioned Mediterranean source proxy. "
            "Source bearing = ERA5 IVT transport bearing + 180 degrees. "
            "The method is Eulerian/geometric, not a Lagrangian back-trajectory."
        ),
        "sensitivity_note": (
            "Kernel width and distance parameters are methodological choices "
            "and should later be sensitivity-tested before final scientific claims."
        ),
        "reasons": reasons,
        "raw_modified": False,
    }

    json_p = (
        out_dir
        / "medsea_ivt_basin_coupling_audit_v1_0.json"
    )
    json_p.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    txt_p = (
        out_dir
        / "medsea_ivt_basin_coupling_audit_v1_0.txt"
    )

    lines = [
        "=" * 128,
        "MEDSEA × IVT -> 21 BACINI — AUDIT COUPLING STORICO v1.0",
        "=" * 128,
        f"OVERALL STATUS          : {overall}",
        f"Periodo                 : Sep-Dic {args.start_year}-{args.end_year}",
        f"Righe attese            : {expected_rows_run}",
        f"Righe prodotte          : {len(full)}",
        f"Recettori               : {full['receptor_id'].nunique()}",
        f"Chiavi duplicate        : {dup}",
        f"SST corridor NaN        : {int(full['medsea_sst_anom_corridor_c'].isna().sum())}",
        f"OHC corridor NaN        : {int(full['medsea_ohc_anom_corridor_j_m2'].isna().sum())}",
        "",
        "PARAMETRI CORRIDOIO",
        f"Settori                 : {N_SECTORS}",
        f"Passo settore           : {SECTOR_STEP_DEG:g} deg",
        f"Sigma angolare          : {ANGULAR_SIGMA_DEG:g} deg",
        f"Cutoff angolare         : +/- {ANGULAR_CUTOFF_DEG:g} deg",
        f"Scala distanza          : {DISTANCE_SCALE_KM:g} km",
        f"Distanza massima        : {MAX_DISTANCE_KM:g} km",
        "",
        "NOTA",
        "Proxy geometrico condizionato da IVT; non retrotraiettoria lagrangiana.",
        "Nessun nearest-cell fallback.",
        "",
        f"Output                  : {final_csv}",
    ]

    txt_p.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 128)
    for line in lines[3:]:
        print(line)
    print("=" * 128)

    if reasons:
        print("REASONS:")
        for r in reasons:
            print(f"  - {r}")

    if overall != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
