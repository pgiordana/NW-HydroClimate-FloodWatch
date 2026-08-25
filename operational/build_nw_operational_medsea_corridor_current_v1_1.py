#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_nw_operational_medsea_corridor_current_v1_1.py

FASE 16 — MEDSEA × IVT OPERATIVO, COERENTE CON IL COUPLING CANONICO v1.2.

Prerequisiti
------------
1) nw_operational_feature_snapshot/<RUN_ID>/
     operational_dynamic_features_v1_1.parquet
     operational_full_97_predictors_v1_1.parquet

2) nw_operational_raw_cache/<RUN_ID>/
     raw_cache_audit_v1_1.json
     copernicus_marine/

3) artefatti canonici storici:
   medsea_historical_analysis/basin_coupling_preflight_v1_0/
     receptor_centroids_v1_0.csv
   medsea_historical_analysis/climatology/
     sst_daily_climatology_1991_2020_SepDec.nc
     ohc_monthly_climatology_1991_2020_SepDec.nc
   medsea_historical_analysis/daily_sst_anomaly/
     medsea_sst_anomaly_2025_SepDec.nc
   medsea_historical_analysis/monthly_ohc_anomaly/
     medsea_ohc_anomaly_2025_SepDec.nc
   medsea_historical_nw/static/
     medsea_my_bathy_mask_source_domain.nc
     medsea_my_grid_metrics_0_110m_source_domain.nc

Metodo congelato di coupling
----------------------------
- 16 settori da 22.5°;
- source bearing = bearing IVT di trasporto + 180°;
- area sferica cella;
- exp(-d / 700 km);
- kernel angolare gaussiano sigma=22.5°;
- baseline cutoff ±45°;
- Dmax=1600 km;
- interpolazione tra i due settori adiacenti;
- support robustness ±30° / ±45° / ±60°;
- nessun nearest-cell fallback per un corridoio privo di supporto;
- no-support NON viene trasformato in zero per le variabili fisiche.

Correzione verticale automatica
-------------------------------
Il raw cache v1.0 aveva chiesto thetao fino a 100 m. Per ricostruire
l'integrazione storica 0-100 m con clipping esatto serve anche il livello
con centro ~104.944 m. Se il file corrente non contiene tutti i livelli
canonici, questo script scarica automaticamente un micro-top-up thetao
fino a 110 m sul SOLO dominio storico canonico.

Importante: differenza temporale OHC/Tmean
------------------------------------------
Nel training:
- SST = giornaliera, anomalia vs climatologia giornaliera 1991-2020;
- OHC/Tmean 0-100 m = stato MENSILE, ripetuto sui giorni del mese;
- anomalie OHC/Tmean = vs climatologia mensile 1991-2020.

Operativamente usiamo thetao giornaliero analysis/forecast:
- SST mantiene una semantica temporale molto vicina a quella storica;
- OHC/Tmean sono DAILY-STATE PROXY della variabile mensile del training.
  Non vengono dichiarati equivalenti: il range/compatibility audit è
  obbligatorio prima di una lettura scientifica del beta.

Fuori stagione
--------------
Il CORE è settembre-dicembre. Le climatologie canoniche disponibili sono
solo Sep-Dec. Se RUN_ID cade fuori da Sep-Dec (es. 25 agosto):
- geometria, support weights, support robustness e valori ASSOLUTI OHC/Tmean
  vengono comunque costruiti come smoke test;
- SST anomaly, OHC anomaly, Tmean anomaly e interazioni anomalia×IVT
  restano NaN per scelta metodologica;
- NON si extrapola una climatologia di agosto inesistente.

OUTPUT
------
nw_operational_feature_snapshot/<RUN_ID>/
  operational_medsea_corridor_v1_0.parquet
  operational_medsea_corridor_diagnostics_v1_0.csv
  operational_dynamic_features_v1_2.parquet
  operational_full_97_predictors_v1_2.parquet
  operational_feature_build_registry_v1_2.csv
  operational_feature_coverage_v1_2.csv
  operational_medsea_audit_v1_1.json
  operational_medsea_audit_v1_1.txt
"""

from __future__ import annotations

import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse


# ---------------------------------------------------------------------------
# Frozen physical/methodological constants
# ---------------------------------------------------------------------------

RHO = 1025.0
CP = 3990.0
TARGET_DEPTH_M = 100.0

N_SECTORS = 16
SECTOR_STEP_DEG = 22.5
SECTOR_CENTERS = np.arange(0.0, 360.0, SECTOR_STEP_DEG)

SIGMA_DEG = 22.5
SCALE_KM = 700.0
DMAX_KM = 1600.0

CUTOFFS = {
    "narrow": 30.0,
    "baseline": 45.0,
    "wide": 60.0,
}

EARTH_RADIUS_KM = 6371.0088
EARTH_RADIUS_M = EARTH_RADIUS_KM * 1000.0

TARGET_RECEPTORS = 20
EXPECTED_DYNAMIC = 83
EXPECTED_STATIC = 14
EXPECTED_TOTAL = 97

CMEMS_DATASET_ID = "cmems_mod_med_phy-tem_anfc_4.2km_P1D-m"

MEDSEA_CANONICAL_FEATURES = [
    "medsea_ivt__medsea_sst_anom_corridor_c",
    "medsea_ivt__medsea_ohc_0_100_corridor_j_m2",
    "medsea_ivt__medsea_ohc_anom_corridor_j_m2",
    "medsea_ivt__medsea_tmean_0_100_corridor_c",
    "medsea_ivt__medsea_tmean_anom_0_100_corridor_c",
    "medsea_ivt__sst_corridor_support_weight",
    "medsea_ivt__ohc_corridor_support_weight",
    "medsea_ivt__ohc_abs_corridor_support_weight",
    "medsea_ivt__tmean_corridor_support_weight",
    "medsea_ivt__tmean_anom_corridor_support_weight",
    "medsea_ivt__sst_anom_x_ivt_proxy",
    "medsea_ivt__ohc_anom_x_ivt_proxy",
    "medsea_ivt__medsea_support_robust_core",
    "medsea_ivt__medsea_support_baseline",
    "medsea_ivt__medsea_support_angle_wide",
]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

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
        f"| rate {rate:7.3f}/s | ETA {fmt_seconds(eta)}"
    )

    if current:
        msg += f" | {str(current)[:145]}"

    print(msg.ljust(300), end="", flush=True)

    if done >= total:
        print(flush=True)


def module_available(name):
    return importlib.util.find_spec(name) is not None


def latest_snapshot_run(root):
    base = root / "nw_operational_feature_snapshot"

    if not base.exists():
        raise SystemExit(f"Manca: {base}")

    runs = sorted(
        [
            p
            for p in base.iterdir()
            if p.is_dir()
            and (
                p
                / "operational_full_97_predictors_v1_1.parquet"
            ).exists()
        ],
        key=lambda p: p.name,
    )

    if not runs:
        raise SystemExit(
            "Nessun operational feature snapshot v1.1 trovato."
        )

    return runs[-1]


def normalize_coords(ds):
    ren = {}

    for old, new in [
        ("lat", "latitude"),
        ("lon", "longitude"),
        ("deptht", "depth"),
        ("depthu", "depth"),
        ("depthv", "depth"),
    ]:
        if old in ds.dims or old in ds.coords:
            if new not in ds.dims and new not in ds.coords:
                ren[old] = new

    if ren:
        ds = ds.rename(ren)

    return ds


def require_path(path, label):
    if not path.exists():
        raise SystemExit(f"Manca {label}: {path}")


# ---------------------------------------------------------------------------
# Canonical geometry helpers — same structure as historical v1.0/v1.2
# ---------------------------------------------------------------------------

def angular_diff_deg(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def cell_edges(centers):
    c = np.asarray(centers, dtype=float)

    if c.ndim != 1 or len(c) < 2:
        raise ValueError(
            "Coordinate grid non-1D o troppo corte."
        )

    mids = (c[:-1] + c[1:]) / 2.0
    first = c[0] + (c[0] - mids[0])
    last = c[-1] + (c[-1] - mids[-1])

    return np.concatenate(
        [[first], mids, [last]]
    )


def spherical_cell_areas(lat, lon):
    lat_e = np.deg2rad(
        cell_edges(lat)
    )
    lon_e = np.deg2rad(
        cell_edges(lon)
    )

    lat_factor = np.abs(
        np.sin(lat_e[1:])
        - np.sin(lat_e[:-1])
    )
    lon_width = np.abs(
        lon_e[1:]
        - lon_e[:-1]
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
    lat2d, lon2d = np.meshgrid(
        np.asarray(grid_lat, dtype=float),
        np.asarray(grid_lon, dtype=float),
        indexing="ij",
    )

    lat1 = np.deg2rad(
        float(basin_lat_deg)
    )
    lon1 = np.deg2rad(
        float(basin_lon_deg)
    )

    lat2 = np.deg2rad(
        lat2d
    )
    lon2 = np.deg2rad(
        lon2d
    )

    dlat = lat2 - lat1
    dlon = (
        lon2 - lon1 + np.pi
    ) % (2 * np.pi) - np.pi

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    a = np.clip(
        a,
        0.0,
        1.0,
    )

    dist = (
        2.0
        * EARTH_RADIUS_KM
        * np.arcsin(
            np.sqrt(a)
        )
    )

    y = (
        np.sin(dlon)
        * np.cos(lat2)
    )

    x = (
        np.cos(lat1)
        * np.sin(lat2)
        - np.sin(lat1)
        * np.cos(lat2)
        * np.cos(dlon)
    )

    bearing = (
        np.rad2deg(
            np.arctan2(y, x)
        )
        + 360.0
    ) % 360.0

    return (
        dist.ravel(),
        bearing.ravel(),
    )


def build_weight_matrix(
    centroids,
    lat,
    lon,
    cutoff_deg,
):
    areas = spherical_cell_areas(
        lat,
        lon,
    ).ravel()

    ncell = (
        len(lat)
        * len(lon)
    )

    rows = []
    cols = []
    data = []

    col_idx = 0

    for rec in centroids.itertuples(
        index=False
    ):
        dist, bearing = (
            basin_to_grid_distance_bearing(
                rec.centroid_lat,
                rec.centroid_lon,
                lat,
                lon,
            )
        )

        for center in SECTOR_CENTERS:
            diff = angular_diff_deg(
                bearing,
                center,
            )

            mask = (
                np.isfinite(dist)
                & np.isfinite(bearing)
                & (
                    dist
                    <= DMAX_KM
                )
                & (
                    diff
                    <= cutoff_deg
                )
            )

            idx = np.flatnonzero(
                mask
            )

            if len(idx):
                w = (
                    areas[idx]
                    * np.exp(
                        -dist[idx]
                        / SCALE_KM
                    )
                    * np.exp(
                        -0.5
                        * (
                            diff[idx]
                            / SIGMA_DEG
                        ) ** 2
                    )
                )

                good = (
                    np.isfinite(w)
                    & (w > 0)
                )

                idx = idx[good]
                w = w[good]

                rows.extend(
                    idx.tolist()
                )
                cols.extend(
                    [col_idx]
                    * len(idx)
                )
                data.extend(
                    w.astype(
                        float
                    ).tolist()
                )

            col_idx += 1

    return sparse.csr_matrix(
        (
            data,
            (
                rows,
                cols,
            ),
        ),
        shape=(
            ncell,
            len(centroids)
            * N_SECTORS,
        ),
        dtype=np.float64,
    )


def sector_pair(source_bearing):
    b = (
        np.asarray(
            source_bearing,
            dtype=float,
        )
        % 360.0
    )

    pos = (
        b
        / SECTOR_STEP_DEG
    )

    low = (
        np.floor(pos)
        .astype(int)
        % N_SECTORS
    )

    frac = (
        pos
        - np.floor(pos)
    )

    high = (
        low + 1
    ) % N_SECTORS

    return (
        low,
        high,
        frac,
    )


def sector_num_den(field_2d, W):
    field = np.asarray(
        field_2d,
        dtype=float,
    ).reshape(-1)

    finite = np.isfinite(
        field
    )

    values = np.nan_to_num(
        field,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    num = np.asarray(
        W.T.dot(
            values
        )
    ).ravel()

    den = np.asarray(
        W.T.dot(
            finite.astype(
                np.float64
            )
        )
    ).ravel()

    return (
        num,
        den,
    )


def interpolate_corridor(
    num,
    den,
    receptor_index,
    low_sector,
    high_sector,
    frac,
):
    c0 = (
        int(receptor_index)
        * N_SECTORS
        + int(low_sector)
    )

    c1 = (
        int(receptor_index)
        * N_SECTORS
        + int(high_sector)
    )

    a0 = (
        1.0
        - float(frac)
    )
    a1 = float(
        frac
    )

    n = (
        a0 * num[c0]
        + a1 * num[c1]
    )

    d = (
        a0 * den[c0]
        + a1 * den[c1]
    )

    if (
        not np.isfinite(d)
        or d <= 0
    ):
        return (
            np.nan,
            0.0,
        )

    return (
        float(n / d),
        float(d),
    )


def interpolated_support(
    den,
    receptor_index,
    low_sector,
    high_sector,
    frac,
):
    c0 = (
        int(receptor_index)
        * N_SECTORS
        + int(low_sector)
    )

    c1 = (
        int(receptor_index)
        * N_SECTORS
        + int(high_sector)
    )

    d = (
        (
            1.0
            - float(frac)
        )
        * den[c0]
        + float(frac)
        * den[c1]
    )

    return bool(
        np.isfinite(d)
        and d > 0
    )


# ---------------------------------------------------------------------------
# Marine vertical integration / grid alignment
# ---------------------------------------------------------------------------

def effective_thickness_0_100(
    deptho,
    mask3d,
    e3t3d,
):
    wet = (
        np.asarray(
            mask3d,
            dtype=float,
        )
        > 0.5
    )

    thick = np.where(
        wet
        & np.isfinite(
            e3t3d
        ),
        e3t3d,
        0.0,
    ).astype(
        np.float64
    )

    top = (
        np.cumsum(
            thick,
            axis=0,
        )
        - thick
    )

    bottom = (
        top
        + thick
    )

    target = np.minimum(
        np.asarray(
            deptho,
            dtype=np.float64,
        ),
        TARGET_DEPTH_M,
    )

    target = np.where(
        np.isfinite(target)
        & (target > 0),
        target,
        0.0,
    )

    eff = (
        np.minimum(
            bottom,
            target[
                None,
                :,
                :,
            ],
        )
        - top
    )

    eff = np.clip(
        eff,
        0.0,
        None,
    )

    eff = np.where(
        wet,
        eff,
        0.0,
    )

    eff_total = np.sum(
        eff,
        axis=0,
    )

    closure = np.where(
        target > 0,
        eff_total - target,
        np.nan,
    )

    return (
        eff,
        eff_total,
        closure,
    )


def compute_ohc_tmean(
    temp3d,
    eff,
    eff_total,
):
    temp = np.asarray(
        temp3d,
        dtype=float,
    )

    valid = (
        np.isfinite(temp)
        & (eff > 0)
    )

    covered = np.sum(
        np.where(
            valid,
            eff,
            0.0,
        ),
        axis=0,
    )

    integral_t = np.sum(
        np.where(
            valid,
            temp * eff,
            0.0,
        ),
        axis=0,
    )

    good = (
        (eff_total > 0)
        & (
            covered
            >= eff_total - 1e-3
        )
    )

    ohc = (
        RHO
        * CP
        * integral_t
    )

    tmean = np.divide(
        integral_t,
        eff_total,
        out=np.full_like(
            integral_t,
            np.nan,
        ),
        where=good,
    )

    ohc[
        ~good
    ] = np.nan

    tmean[
        ~good
    ] = np.nan

    return (
        ohc,
        tmean,
        good,
    )


def load_static_grid(
    bathy_path,
    metrics_path,
):
    with xr.open_dataset(
        bathy_path
    ) as ds:
        ds = normalize_coords(
            ds
        ).load()

    with xr.open_dataset(
        metrics_path
    ) as dm:
        dm = normalize_coords(
            dm
        ).load()

    lat = np.asarray(
        ds["latitude"].values,
        dtype=float,
    )

    lon = np.asarray(
        ds["longitude"].values,
        dtype=float,
    )

    depth_coord = np.asarray(
        ds["mask"]["depth"].values,
        dtype=float,
    )

    # Historical pipeline uses first 25 levels.
    idx = np.flatnonzero(
        depth_coord
        <= 110.0
    )

    if len(idx) < 25:
        raise RuntimeError(
            f"Static depth levels <=110m={len(idx)}, expected >=25."
        )

    idx = idx[:25]

    depth = depth_coord[
        idx
    ]

    deptho = (
        ds["deptho"]
        .squeeze(drop=True)
        .transpose(
            "latitude",
            "longitude",
        )
        .values.astype(
            np.float64
        )
    )

    mask = (
        ds["mask"]
        .isel(
            depth=idx
        )
        .squeeze(drop=True)
        .transpose(
            "depth",
            "latitude",
            "longitude",
        )
        .values.astype(
            np.float64
        )
    )

    e3t = (
        dm["e3t"]
        .isel(
            depth=idx
        )
        .squeeze(drop=True)
        .transpose(
            "depth",
            "latitude",
            "longitude",
        )
        .values.astype(
            np.float64
        )
    )

    return {
        "lat": lat,
        "lon": lon,
        "depth": depth,
        "deptho": deptho,
        "mask": mask,
        "e3t": e3t,
    }


def current_file_has_canonical_depths(
    path,
    canonical_depth,
):
    try:
        with xr.open_dataset(
            path
        ) as ds:
            ds = normalize_coords(
                ds
            )

            if "depth" not in ds.coords:
                return False

            dep = np.asarray(
                ds["depth"].values,
                dtype=float,
            )

        if len(dep) < len(
            canonical_depth
        ):
            return False

        nearest = np.asarray(
            [
                dep[
                    np.argmin(
                        np.abs(
                            dep - d
                        )
                    )
                ]
                for d
                in canonical_depth
            ]
        )

        return bool(
            np.max(
                np.abs(
                    nearest
                    - canonical_depth
                )
            )
            < 1e-3
        )

    except Exception:
        return False


def ensure_current_canonical_marine_file(
    run_dir,
    issue_date,
    static_grid,
):
    cmems_dir = (
        run_dir
        / "copernicus_marine"
    )

    candidates = sorted(
        cmems_dir.glob(
            "*.nc"
        )
    )

    for p in candidates:
        if current_file_has_canonical_depths(
            p,
            static_grid["depth"],
        ):
            return (
                p,
                "REUSED_EXISTING_WITH_25_CANONICAL_LEVELS",
            )

    if not module_available(
        "copernicusmarine"
    ):
        raise RuntimeError(
            "Serve top-up marine 0-110 m ma copernicusmarine non è installato."
        )

    import copernicusmarine

    target_name = (
        "cmems_thetao_canonical_domain_0_110m_"
        f"{issue_date}.nc"
    )

    target = (
        cmems_dir
        / target_name
    )

    min_lon = float(
        np.min(
            static_grid["lon"]
        )
    )
    max_lon = float(
        np.max(
            static_grid["lon"]
        )
    )
    min_lat = float(
        np.min(
            static_grid["lat"]
        )
    )
    max_lat = float(
        np.max(
            static_grid["lat"]
        )
    )

    print(
        "\nTop-up CMEMS necessario:"
    )
    print(
        f"  domain lon {min_lon:.5f}..{max_lon:.5f}"
    )
    print(
        f"  domain lat {min_lat:.5f}..{max_lat:.5f}"
    )
    print(
        "  depth 0..110 m"
    )

    result = copernicusmarine.subset(
        dataset_id=CMEMS_DATASET_ID,
        variables=["thetao"],
        start_datetime=(
            f"{issue_date}T00:00:00"
        ),
        end_datetime=(
            f"{issue_date}T23:59:59"
        ),
        minimum_longitude=min_lon,
        maximum_longitude=max_lon,
        minimum_latitude=min_lat,
        maximum_latitude=max_lat,
        minimum_depth=0.0,
        maximum_depth=110.0,
        output_directory=str(
            cmems_dir
        ),
        output_filename=target_name,
        overwrite=True,
        disable_progress_bar=False,
    )

    result_path = getattr(
        result,
        "file_path",
        None,
    )

    possible = [
        Path(
            str(
                result_path
            )
        )
        if result_path
        else None,
        target,
    ]

    selected = next(
        (
            p
            for p in possible
            if p is not None
            and p.exists()
            and p.stat().st_size > 0
        ),
        None,
    )

    if selected is None:
        raise RuntimeError(
            "CMEMS top-up terminato senza file valido."
        )

    if not current_file_has_canonical_depths(
        selected,
        static_grid["depth"],
    ):
        raise RuntimeError(
            "CMEMS top-up non contiene i 25 livelli canonici necessari."
        )

    return (
        selected,
        "DOWNLOADED_CANONICAL_0_110M_TOPUP",
    )


def align_current_to_static_grid(
    current_path,
    static_grid,
):
    with xr.open_dataset(
        current_path
    ) as ds:
        ds = normalize_coords(
            ds
        )

        if "thetao" not in ds:
            raise RuntimeError(
                "thetao assente nel file corrente."
            )

        da = ds["thetao"]

        if "time" in da.dims:
            da = da.isel(
                time=0
            )

        # Select exactly/nearest the canonical static nodes.
        da = da.sel(
            latitude=xr.DataArray(
                static_grid["lat"],
                dims="latitude",
            ),
            longitude=xr.DataArray(
                static_grid["lon"],
                dims="longitude",
            ),
            depth=xr.DataArray(
                static_grid["depth"],
                dims="depth",
            ),
            method="nearest",
        )

        lat_selected = np.asarray(
            da["latitude"].values,
            dtype=float,
        )
        lon_selected = np.asarray(
            da["longitude"].values,
            dtype=float,
        )
        dep_selected = np.asarray(
            da["depth"].values,
            dtype=float,
        )

        lat_diff = float(
            np.max(
                np.abs(
                    lat_selected
                    - static_grid["lat"]
                )
            )
        )

        lon_diff = float(
            np.max(
                np.abs(
                    lon_selected
                    - static_grid["lon"]
                )
            )
        )

        dep_diff = float(
            np.max(
                np.abs(
                    dep_selected
                    - static_grid["depth"]
                )
            )
        )

        # We do not allow a true regridding step here.
        if (
            lat_diff > 1e-4
            or lon_diff > 1e-4
            or dep_diff > 1e-3
        ):
            raise RuntimeError(
                "Operational ANFC grid does not coincide with historical "
                f"canonical grid within tolerance: "
                f"lat={lat_diff}, lon={lon_diff}, depth={dep_diff}"
            )

        temp = (
            da.transpose(
                "depth",
                "latitude",
                "longitude",
            )
            .values.astype(
                np.float64
            )
        )

    return {
        "temp": temp,
        "max_lat_diff": lat_diff,
        "max_lon_diff": lon_diff,
        "max_depth_diff": dep_diff,
    }


# ---------------------------------------------------------------------------
# Canonical climatology handling
# ---------------------------------------------------------------------------

def load_in_season_climatology(
    issue_date,
    static_grid,
    sst_clim_path,
    ohc_clim_path,
):
    issue_ts = pd.Timestamp(
        issue_date
    )

    if issue_ts.month not in {
        9,
        10,
        11,
        12,
    }:
        return {
            "in_season": False,
            "sst_clim": None,
            "ohc_clim": None,
            "tmean_clim": None,
            "reason": (
                "OUT_OF_SEASON__CANONICAL_CLIMATOLOGY_ONLY_SEP_DEC"
            ),
        }

    with xr.open_dataset(
        sst_clim_path
    ) as ds:
        ds = normalize_coords(
            ds
        )

        times = pd.DatetimeIndex(
            pd.to_datetime(
                ds["time"].values
            )
        )

        md = issue_ts.strftime(
            "%m-%d"
        )

        matches = [
            i
            for i, t
            in enumerate(times)
            if t.strftime(
                "%m-%d"
            )
            == md
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"SST climatology match for {md}: {matches}"
            )

        sst = (
            ds["sst_climatology"]
            .isel(
                time=matches[0]
            )
            .transpose(
                "latitude",
                "longitude",
            )
            .values.astype(
                np.float64
            )
        )

        lat = np.asarray(
            ds["latitude"].values,
            dtype=float,
        )
        lon = np.asarray(
            ds["longitude"].values,
            dtype=float,
        )

    with xr.open_dataset(
        ohc_clim_path
    ) as ds:
        ds = normalize_coords(
            ds
        )

        times = pd.DatetimeIndex(
            pd.to_datetime(
                ds["time"].values
            )
        )

        matches = [
            i
            for i, t
            in enumerate(times)
            if int(
                t.month
            )
            == int(
                issue_ts.month
            )
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"OHC climatology match month={issue_ts.month}: {matches}"
            )

        ohc = (
            ds[
                "ohc_0_100_climatology"
            ]
            .isel(
                time=matches[0]
            )
            .transpose(
                "latitude",
                "longitude",
            )
            .values.astype(
                np.float64
            )
        )

        tm = (
            ds[
                "tmean_0_100_climatology"
            ]
            .isel(
                time=matches[0]
            )
            .transpose(
                "latitude",
                "longitude",
            )
            .values.astype(
                np.float64
            )
        )

        lat2 = np.asarray(
            ds["latitude"].values,
            dtype=float,
        )
        lon2 = np.asarray(
            ds["longitude"].values,
            dtype=float,
        )

    for name, a, b in [
        (
            "sst_lat",
            lat,
            static_grid["lat"],
        ),
        (
            "sst_lon",
            lon,
            static_grid["lon"],
        ),
        (
            "ohc_lat",
            lat2,
            static_grid["lat"],
        ),
        (
            "ohc_lon",
            lon2,
            static_grid["lon"],
        ),
    ]:
        if (
            a.shape != b.shape
            or not np.allclose(
                a,
                b,
                atol=1e-10,
                rtol=0,
            )
        ):
            raise RuntimeError(
                f"Canonical climatology grid mismatch: {name}"
            )

    return {
        "in_season": True,
        "sst_clim": sst,
        "ohc_clim": ohc,
        "tmean_clim": tm,
        "reason": "IN_SEASON_CANONICAL_CLIMATOLOGY_AVAILABLE",
    }


def load_reference_support_mask(
    root,
    static_grid,
):
    sst_ref = (
        root
        / "medsea_historical_analysis"
        / "daily_sst_anomaly"
        / "medsea_sst_anomaly_2025_SepDec.nc"
    )

    ohc_ref = (
        root
        / "medsea_historical_analysis"
        / "monthly_ohc_anomaly"
        / "medsea_ohc_anomaly_2025_SepDec.nc"
    )

    require_path(
        sst_ref,
        "SST reference 2025",
    )
    require_path(
        ohc_ref,
        "OHC reference 2025",
    )

    with xr.open_dataset(
        sst_ref
    ) as ds:
        ds = normalize_coords(
            ds
        )

        sst = (
            ds["sst_anomaly"]
            .isel(
                time=0
            )
            .transpose(
                "latitude",
                "longitude",
            )
            .values.astype(
                np.float64
            )
        )

        lat = np.asarray(
            ds["latitude"].values,
            dtype=float,
        )
        lon = np.asarray(
            ds["longitude"].values,
            dtype=float,
        )

    with xr.open_dataset(
        ohc_ref
    ) as ds:
        ds = normalize_coords(
            ds
        )

        ohc = (
            ds[
                "ohc_anomaly_0_100"
            ]
            .isel(
                time=0
            )
            .transpose(
                "latitude",
                "longitude",
            )
            .values.astype(
                np.float64
            )
        )

        lat2 = np.asarray(
            ds["latitude"].values,
            dtype=float,
        )
        lon2 = np.asarray(
            ds["longitude"].values,
            dtype=float,
        )

    for a, b, label in [
        (
            lat,
            static_grid["lat"],
            "sst lat",
        ),
        (
            lon,
            static_grid["lon"],
            "sst lon",
        ),
        (
            lat2,
            static_grid["lat"],
            "ohc lat",
        ),
        (
            lon2,
            static_grid["lon"],
            "ohc lon",
        ),
    ]:
        if (
            a.shape != b.shape
            or not np.allclose(
                a,
                b,
                atol=1e-10,
                rtol=0,
            )
        ):
            raise RuntimeError(
                f"Reference support mask grid mismatch: {label}"
            )

    return (
        np.isfinite(sst)
        & np.isfinite(ohc)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    root = Path(__file__).resolve().parent
    snapshot_dir = latest_snapshot_run(
        root
    )

    run_id = snapshot_dir.name

    raw_run = (
        root
        / "nw_operational_raw_cache"
        / run_id
    )

    raw_audit_p = (
        raw_run
        / "raw_cache_audit_v1_1.json"
    )

    require_path(
        raw_audit_p,
        "raw cache audit v1.1",
    )

    raw_audit = json.loads(
        raw_audit_p.read_text(
            encoding="utf-8"
        )
    )

    if (
        raw_audit.get(
            "overall_status"
        )
        != "PASS_RAW_CACHE_SURFACE_REPAIRED_V1_1__FEATURE_ENGINE_READY"
    ):
        raise SystemExit(
            "Raw cache v1.1 non pronto: "
            + str(
                raw_audit.get(
                    "overall_status"
                )
            )
        )

    issue_date = pd.Timestamp(
        raw_audit[
            "issue_cycle_utc"
        ]
    ).date()

    dynamic_v11_p = (
        snapshot_dir
        / "operational_dynamic_features_v1_1.parquet"
    )

    full_v11_p = (
        snapshot_dir
        / "operational_full_97_predictors_v1_1.parquet"
    )

    registry_v11_p = (
        snapshot_dir
        / "operational_feature_build_registry_v1_1.csv"
    )

    for p in [
        dynamic_v11_p,
        full_v11_p,
        registry_v11_p,
    ]:
        require_path(
            p,
            p.name,
        )

    centroids_p = (
        root
        / "medsea_historical_analysis"
        / "basin_coupling_preflight_v1_0"
        / "receptor_centroids_v1_0.csv"
    )

    sst_clim_p = (
        root
        / "medsea_historical_analysis"
        / "climatology"
        / "sst_daily_climatology_1991_2020_SepDec.nc"
    )

    ohc_clim_p = (
        root
        / "medsea_historical_analysis"
        / "climatology"
        / "ohc_monthly_climatology_1991_2020_SepDec.nc"
    )

    bathy_p = (
        root
        / "medsea_historical_nw"
        / "static"
        / "medsea_my_bathy_mask_source_domain.nc"
    )

    metrics_p = (
        root
        / "medsea_historical_nw"
        / "static"
        / "medsea_my_grid_metrics_0_110m_source_domain.nc"
    )

    dictionary_p = (
        root
        / "nw_hydroclimate_core_release_v1_0"
        / "metadata"
        / "nw_hydroclimate_master_predictor_dictionary_v1_0.csv"
    )

    dynamic_whitelist_p = (
        root
        / "nw_hydroclimate_core_release_v1_0"
        / "metadata"
        / "primary_dynamic_feature_whitelist_canonical_v1_3.csv"
    )

    for p, label in [
        (
            centroids_p,
            "canonical centroids",
        ),
        (
            sst_clim_p,
            "SST canonical climatology",
        ),
        (
            ohc_clim_p,
            "OHC canonical climatology",
        ),
        (
            bathy_p,
            "canonical marine bathymetry/mask",
        ),
        (
            metrics_p,
            "canonical marine grid metrics",
        ),
        (
            dictionary_p,
            "97-predictor dictionary",
        ),
        (
            dynamic_whitelist_p,
            "83-feature whitelist",
        ),
    ]:
        require_path(
            p,
            label,
        )

    print("=" * 220)
    print("NW HYDROCLIMATE — OPERATIONAL MEDSEA × IVT CORRIDOR v1.1")
    print("=" * 220)
    print(
        f"Run ID     : {run_id}"
    )
    print(
        f"Issue date : {issue_date}"
    )

    # ------------------------------------------------------------------
    # PHASE 1/7 — load canonical grids + current snapshot
    # ------------------------------------------------------------------
    print(
        "\nPHASE 1/7 — load canonical marine grid, centroids and 97-column snapshot"
    )
    start = time.time()

    static_grid = load_static_grid(
        bathy_p,
        metrics_p,
    )

    dynamic_v11 = pd.read_parquet(
        dynamic_v11_p
    )

    full_v11 = pd.read_parquet(
        full_v11_p
    )

    dictionary = pd.read_csv(
        dictionary_p,
        low_memory=False,
    )

    whitelist = pd.read_csv(
        dynamic_whitelist_p,
        low_memory=False,
    )

    predictor_order = (
        dictionary[
            "predictor"
        ]
        .astype(str)
        .tolist()
    )

    dynamic_order = (
        whitelist[
            "canonical_feature_name"
        ]
        .astype(str)
        .tolist()
    )

    if (
        len(dynamic_v11)
        != TARGET_RECEPTORS
    ):
        raise SystemExit(
            f"Snapshot rows={len(dynamic_v11)}, expected=20"
        )

    if (
        len(predictor_order)
        != EXPECTED_TOTAL
    ):
        raise SystemExit(
            f"Predictors={len(predictor_order)}, expected=97"
        )

    if (
        len(dynamic_order)
        != EXPECTED_DYNAMIC
    ):
        raise SystemExit(
            f"Dynamic whitelist={len(dynamic_order)}, expected=83"
        )

    missing_medsea = [
        f
        for f in MEDSEA_CANONICAL_FEATURES
        if f not in dynamic_order
    ]

    if missing_medsea:
        raise SystemExit(
            "Canonical MedSea feature list mismatch: "
            + ", ".join(
                missing_medsea
            )
        )

    centroids = pd.read_csv(
        centroids_p
    )

    target_ids = (
        dynamic_v11[
            "receptor_id"
        ]
        .astype(str)
        .tolist()
    )

    centroids = (
        centroids[
            centroids[
                "receptor_id"
            ]
            .astype(str)
            .isin(target_ids)
        ]
        .copy()
    )

    centroids[
        "receptor_id"
    ] = centroids[
        "receptor_id"
    ].astype(str)

    centroids = (
        centroids.set_index(
            "receptor_id"
        )
        .loc[target_ids]
        .reset_index()
    )

    if len(
        centroids
    ) != TARGET_RECEPTORS:
        raise SystemExit(
            f"Target centroids={len(centroids)}, expected=20"
        )

    progress(
        "PHASE 1/7",
        1,
        1,
        start,
        (
            f"grid={len(static_grid['lat'])}x{len(static_grid['lon'])} "
            f"| depth_levels={len(static_grid['depth'])} "
            f"| receptors={len(centroids)}"
        ),
    )

    # ------------------------------------------------------------------
    # PHASE 2/7 — ensure 0-110m current file and exact grid alignment
    # ------------------------------------------------------------------
    print(
        "\nPHASE 2/7 — ensure current thetao has canonical 25 levels and exact historical grid"
    )
    start = time.time()

    marine_file, marine_file_mode = (
        ensure_current_canonical_marine_file(
            raw_run,
            issue_date,
            static_grid,
        )
    )

    aligned = align_current_to_static_grid(
        marine_file,
        static_grid,
    )

    temp3d = aligned[
        "temp"
    ]

    progress(
        "PHASE 2/7",
        1,
        1,
        start,
        (
            f"{marine_file_mode} | "
            f"lat_diff={aligned['max_lat_diff']:.3e} "
            f"lon_diff={aligned['max_lon_diff']:.3e} "
            f"depth_diff={aligned['max_depth_diff']:.3e}"
        ),
    )

    # ------------------------------------------------------------------
    # PHASE 3/7 — physical marine state 0-100m
    # ------------------------------------------------------------------
    print(
        "\nPHASE 3/7 — reconstruct current SST / OHC / Tmean 0-100m"
    )
    start = time.time()

    eff, eff_total, closure = (
        effective_thickness_0_100(
            static_grid["deptho"],
            static_grid["mask"],
            static_grid["e3t"],
        )
    )

    closure_mean_abs = float(
        np.nanmean(
            np.abs(
                closure
            )
        )
    )

    closure_max_abs = float(
        np.nanmax(
            np.abs(
                closure
            )
        )
    )

    if (
        closure_max_abs
        > 1e-3
    ):
        raise RuntimeError(
            f"Vertical closure too large: {closure_max_abs} m"
        )

    ohc_abs, tmean_abs, good_col = (
        compute_ohc_tmean(
            temp3d,
            eff,
            eff_total,
        )
    )

    sst_abs = (
        temp3d[
            0,
            :,
            :,
        ]
    )

    clim = load_in_season_climatology(
        issue_date,
        static_grid,
        sst_clim_p,
        ohc_clim_p,
    )

    if clim[
        "in_season"
    ]:
        sst_anom = (
            sst_abs
            - clim[
                "sst_clim"
            ]
        )

        ohc_anom = (
            ohc_abs
            - clim[
                "ohc_clim"
            ]
        )

        tmean_anom = (
            tmean_abs
            - clim[
                "tmean_clim"
            ]
        )
    else:
        sst_anom = np.full_like(
            sst_abs,
            np.nan,
            dtype=float,
        )

        ohc_anom = np.full_like(
            ohc_abs,
            np.nan,
            dtype=float,
        )

        tmean_anom = np.full_like(
            tmean_abs,
            np.nan,
            dtype=float,
        )

    progress(
        "PHASE 3/7",
        1,
        1,
        start,
        (
            f"vertical_closure_max={closure_max_abs:.3e} m "
            f"| in_season={clim['in_season']} "
            f"| climatology={clim['reason']}"
        ),
    )

    # ------------------------------------------------------------------
    # PHASE 4/7 — build frozen corridor weight matrices
    # ------------------------------------------------------------------
    print(
        "\nPHASE 4/7 — build frozen 16-sector corridor weights and support robustness"
    )
    start = time.time()

    W = {}

    for i, (
        name,
        cutoff,
    ) in enumerate(
        CUTOFFS.items(),
        1,
    ):
        W[
            name
        ] = build_weight_matrix(
            centroids,
            static_grid["lat"],
            static_grid["lon"],
            cutoff,
        )

        progress(
            "PHASE 4/7",
            i,
            len(
                CUTOFFS
            ),
            start,
            (
                f"{name} cutoff=±{cutoff:g}° "
                f"| nnz={W[name].nnz}"
            ),
        )

    reference_mask = (
        load_reference_support_mask(
            root,
            static_grid,
        )
    )

    reference_support_vector = (
        reference_mask.astype(
            np.float64
        ).ravel()
    )

    expected_ncell = (
        len(static_grid["lat"])
        * len(static_grid["lon"])
    )

    if reference_support_vector.size != expected_ncell:
        raise RuntimeError(
            "Reference support mask size mismatch: "
            f"mask={reference_support_vector.size}, "
            f"expected={expected_ncell} "
            f"({len(static_grid['lat'])}x{len(static_grid['lon'])})."
        )

    for name in CUTOFFS:
        if W[name].shape[0] != expected_ncell:
            raise RuntimeError(
                f"Weight-matrix row mismatch for {name}: "
                f"{W[name].shape[0]} != {expected_ncell}"
            )

    ref_support_den = {
        name: np.asarray(
            W[name].T.dot(
                reference_support_vector
            )
        ).ravel()
        for name
        in CUTOFFS
    }

    # Numeric current fields always use baseline geometry.
    baseline_W = W[
        "baseline"
    ]

    numeric_fields = {
        "sst_anom": sst_anom,
        "ohc_abs": ohc_abs,
        "ohc_anom": ohc_anom,
        "tmean_abs": tmean_abs,
        "tmean_anom": tmean_anom,
    }

    num_den = {
        name: sector_num_den(
            field,
            baseline_W,
        )
        for name, field
        in numeric_fields.items()
    }

    # When anomaly is unavailable out-of-season, support denominators still
    # use the corresponding physical current field mask, because the historical
    # anomaly and absolute products share the same wet-grid support.
    if not clim[
        "in_season"
    ]:
        sst_support_den = sector_num_den(
            sst_abs,
            baseline_W,
        )[1]

        ohc_support_den = sector_num_den(
            ohc_abs,
            baseline_W,
        )[1]

        tmean_support_den = sector_num_den(
            tmean_abs,
            baseline_W,
        )[1]
    else:
        sst_support_den = num_den[
            "sst_anom"
        ][1]

        ohc_support_den = num_den[
            "ohc_anom"
        ][1]

        tmean_support_den = num_den[
            "tmean_anom"
        ][1]

    # ------------------------------------------------------------------
    # PHASE 5/7 — calculate 15 canonical MedSea predictors
    # ------------------------------------------------------------------
    print(
        "\nPHASE 5/7 — calculate 15 canonical MedSea predictors for 20 receptors"
    )
    start = time.time()

    rec_map = {
        rid: i
        for i, rid
        in enumerate(
            centroids[
                "receptor_id"
            ].astype(
                str
            )
        )
    }

    medsea_rows = []
    diag_rows = []

    for i, r in enumerate(
        dynamic_v11.itertuples(
            index=False
        ),
        1,
    ):
        rid = str(
            r.receptor_id
        )

        b = rec_map[
            rid
        ]

        ivt_e = float(
            getattr(
                r,
                "era5__ivt_e_mean_kg_m1_s1",
            )
        )

        ivt_n = float(
            getattr(
                r,
                "era5__ivt_n_mean_kg_m1_s1",
            )
        )

        ivt_mag = float(
            getattr(
                r,
                "era5__ivt_mag_mean_kg_m1_s1",
            )
        )

        if not all(
            np.isfinite(
                [
                    ivt_e,
                    ivt_n,
                    ivt_mag,
                ]
            )
        ):
            raise RuntimeError(
                f"{rid}: IVT operational proxy missing."
            )

        transport_bearing = (
            math.degrees(
                math.atan2(
                    ivt_e,
                    ivt_n,
                )
            )
            + 360.0
        ) % 360.0

        source_bearing = (
            transport_bearing
            + 180.0
        ) % 360.0

        (
            low,
            high,
            frac,
        ) = sector_pair(
            [
                source_bearing
            ]
        )

        low = int(
            low[0]
        )
        high = int(
            high[0]
        )
        frac = float(
            frac[0]
        )

        def value(field_key):
            num, den = num_den[
                field_key
            ]

            return interpolate_corridor(
                num,
                den,
                b,
                low,
                high,
                frac,
            )

        sst_val, _ = value(
            "sst_anom"
        )

        ohc_abs_val, ohc_abs_support = value(
            "ohc_abs"
        )

        ohc_anom_val, _ = value(
            "ohc_anom"
        )

        tmean_abs_val, tmean_abs_support = value(
            "tmean_abs"
        )

        tmean_anom_val, _ = value(
            "tmean_anom"
        )

        # Explicit support weights with current wet mask.
        _, sst_support = interpolate_corridor(
            np.zeros_like(
                sst_support_den
            ),
            sst_support_den,
            b,
            low,
            high,
            frac,
        )

        _, ohc_support = interpolate_corridor(
            np.zeros_like(
                ohc_support_den
            ),
            ohc_support_den,
            b,
            low,
            high,
            frac,
        )

        _, tmean_support = interpolate_corridor(
            np.zeros_like(
                tmean_support_den
            ),
            tmean_support_den,
            b,
            low,
            high,
            frac,
        )

        narrow = interpolated_support(
            ref_support_den[
                "narrow"
            ],
            b,
            low,
            high,
            frac,
        )

        baseline = interpolated_support(
            ref_support_den[
                "baseline"
            ],
            b,
            low,
            high,
            frac,
        )

        wide = interpolated_support(
            ref_support_den[
                "wide"
            ],
            b,
            low,
            high,
            frac,
        )

        if (
            narrow
            and not baseline
        ) or (
            baseline
            and not wide
        ):
            raise RuntimeError(
                f"{rid}: non-monotonic support narrow/baseline/wide."
            )

        outrow = {
            "receptor_id": rid,
            "issue_date": str(
                issue_date
            ),
            "run_id": run_id,
            "medsea_ivt__medsea_sst_anom_corridor_c":
                sst_val,
            "medsea_ivt__medsea_ohc_0_100_corridor_j_m2":
                ohc_abs_val,
            "medsea_ivt__medsea_ohc_anom_corridor_j_m2":
                ohc_anom_val,
            "medsea_ivt__medsea_tmean_0_100_corridor_c":
                tmean_abs_val,
            "medsea_ivt__medsea_tmean_anom_0_100_corridor_c":
                tmean_anom_val,
            "medsea_ivt__sst_corridor_support_weight":
                sst_support,
            "medsea_ivt__ohc_corridor_support_weight":
                ohc_support,
            "medsea_ivt__ohc_abs_corridor_support_weight":
                ohc_abs_support,
            "medsea_ivt__tmean_corridor_support_weight":
                tmean_abs_support,
            "medsea_ivt__tmean_anom_corridor_support_weight":
                tmean_support,
            "medsea_ivt__sst_anom_x_ivt_proxy":
                (
                    sst_val
                    * ivt_mag
                    if np.isfinite(
                        sst_val
                    )
                    else np.nan
                ),
            "medsea_ivt__ohc_anom_x_ivt_proxy":
                (
                    ohc_anom_val
                    * ivt_mag
                    if np.isfinite(
                        ohc_anom_val
                    )
                    else np.nan
                ),
            "medsea_ivt__medsea_support_robust_core":
                float(
                    narrow
                ),
            "medsea_ivt__medsea_support_baseline":
                float(
                    baseline
                ),
            "medsea_ivt__medsea_support_angle_wide":
                float(
                    wide
                ),
        }

        medsea_rows.append(
            outrow
        )

        diag_rows.append(
            {
                "receptor_id": rid,
                "ivt_transport_bearing_deg":
                    transport_bearing,
                "marine_source_bearing_deg":
                    source_bearing,
                "sector_low_deg":
                    low
                    * SECTOR_STEP_DEG,
                "sector_high_deg":
                    high
                    * SECTOR_STEP_DEG,
                "sector_interp_fraction":
                    frac,
                "support_narrow":
                    narrow,
                "support_baseline":
                    baseline,
                "support_wide":
                    wide,
                "ohc_abs_corridor_j_m2":
                    ohc_abs_val,
                "tmean_abs_corridor_c":
                    tmean_abs_val,
                "sst_anom_corridor_c":
                    sst_val,
                "ohc_anom_corridor_j_m2":
                    ohc_anom_val,
                "climatology_state":
                    clim[
                        "reason"
                    ],
            }
        )

        progress(
            "PHASE 5/7",
            i,
            TARGET_RECEPTORS,
            start,
            (
                f"{rid} | source={source_bearing:.1f}° "
                f"| support={int(narrow)}/{int(baseline)}/{int(wide)} "
                f"| T0-100={tmean_abs_val:.2f}C"
            ),
        )

    medsea = pd.DataFrame(
        medsea_rows
    )

    diagnostics = pd.DataFrame(
        diag_rows
    )

    # ------------------------------------------------------------------
    # PHASE 6/7 — update 83 dynamic + frozen 97 predictor skeleton
    # ------------------------------------------------------------------
    print(
        "\nPHASE 6/7 — merge MedSea predictors into dynamic/97-column operational snapshot"
    )
    start = time.time()

    dynamic_v12 = (
        dynamic_v11.drop(
            columns=[
                c
                for c
                in MEDSEA_CANONICAL_FEATURES
                if c
                in dynamic_v11.columns
            ]
        )
        .merge(
            medsea[
                [
                    "receptor_id",
                    *MEDSEA_CANONICAL_FEATURES,
                ]
            ],
            on="receptor_id",
            how="left",
            validate="one_to_one",
        )
    )

    # Restore strict canonical order.
    dynamic_v12 = dynamic_v12[
        [
            "receptor_id",
            "issue_date",
            "run_id",
            *dynamic_order,
        ]
    ]

    # Keep static predictors from v1.1 and replace all dynamic predictors.
    static_predictors = [
        p
        for p in predictor_order
        if p.startswith(
            "static__"
        )
    ]

    if len(
        static_predictors
    ) != EXPECTED_STATIC:
        raise RuntimeError(
            f"Static predictors in dictionary={len(static_predictors)}, expected=14."
        )

    static_block = full_v11[
        [
            "receptor_id",
            *static_predictors,
        ]
    ].copy()

    full_v12 = (
        dynamic_v12.merge(
            static_block,
            on="receptor_id",
            how="left",
            validate="one_to_one",
        )
    )

    full_v12 = full_v12[
        [
            "receptor_id",
            "issue_date",
            "run_id",
            *predictor_order,
        ]
    ]

    if len(
        full_v12
    ) != TARGET_RECEPTORS:
        raise RuntimeError(
            f"Full v1.2 rows={len(full_v12)}, expected=20."
        )

    static_missing = int(
        full_v12[
            static_predictors
        ]
        .isna()
        .sum()
        .sum()
    )

    if static_missing:
        raise RuntimeError(
            f"Static missing cells after merge={static_missing}"
        )

    progress(
        "PHASE 6/7",
        1,
        1,
        start,
        (
            f"rows={len(full_v12)} | predictors={len(predictor_order)} "
            f"| static_missing={static_missing}"
        ),
    )

    # ------------------------------------------------------------------
    # PHASE 7/7 — coverage registry + audit
    # ------------------------------------------------------------------
    print(
        "\nPHASE 7/7 — freeze MedSea coverage and operational feature audit"
    )
    start = time.time()

    registry_v11 = pd.read_csv(
        registry_v11_p,
        low_memory=False,
    )

    registry_v12 = (
        registry_v11.copy()
    )

    for feature in MEDSEA_CANONICAL_FEATURES:
        nonmissing = int(
            dynamic_v12[
                feature
            ]
            .notna()
            .sum()
        )

        if feature in {
            "medsea_ivt__medsea_sst_anom_corridor_c",
            "medsea_ivt__medsea_ohc_anom_corridor_j_m2",
            "medsea_ivt__medsea_tmean_anom_0_100_corridor_c",
            "medsea_ivt__sst_anom_x_ivt_proxy",
            "medsea_ivt__ohc_anom_x_ivt_proxy",
        } and not clim[
            "in_season"
        ]:
            build_state = (
                "OUT_OF_SEASON_CANONICAL_ANOMALY_UNAVAILABLE"
            )
            semantics = (
                "Canonical climatology exists only Sep-Dec; "
                "feature intentionally left NaN."
            )

        elif feature in {
            "medsea_ivt__medsea_ohc_0_100_corridor_j_m2",
            "medsea_ivt__medsea_tmean_0_100_corridor_c",
        }:
            build_state = (
                "BUILT_DAILY_MARINE_STATE_PROXY_FOR_MONTHLY_TRAINING_FEATURE"
            )
            semantics = (
                "Current daily ANFC thetao integrated 0-100 m; "
                "training OHC/Tmean were monthly states. "
                "Compatibility audit required."
            )

        elif "support_weight" in feature:
            build_state = (
                "BUILT_CANONICAL_GEOMETRY_CURRENT_FIELD_SUPPORT"
            )
            semantics = (
                "Canonical baseline corridor geometry; "
                "denominator from current field finite mask."
            )

        elif "medsea_support_" in feature:
            build_state = (
                "BUILT_CANONICAL_STATIC_REFERENCE_SUPPORT_ROBUSTNESS"
            )
            semantics = (
                "Canonical ±30/±45/±60 support geometry using frozen 2025 "
                "reference marine mask."
            )

        else:
            build_state = (
                "BUILT_OPERATIONAL_MEDSEA_PROXY"
                if nonmissing
                else "MISSING"
            )
            semantics = (
                "Operational MedSea proxy; compatibility audit required."
            )

        mask = (
            registry_v12[
                "canonical_feature_name"
            ]
            .astype(str)
            .eq(feature)
        )

        if not mask.any():
            raise RuntimeError(
                f"Feature absent from build registry: {feature}"
            )

        registry_v12.loc[
            mask,
            "nonmissing_receptors",
        ] = nonmissing

        registry_v12.loc[
            mask,
            "total_receptors",
        ] = TARGET_RECEPTORS

        registry_v12.loc[
            mask,
            "coverage_fraction",
        ] = (
            nonmissing
            / TARGET_RECEPTORS
        )

        registry_v12.loc[
            mask,
            "build_state",
        ] = build_state

        registry_v12.loc[
            mask,
            "operational_semantics",
        ] = semantics

    coverage_v12 = (
        registry_v12.groupby(
            "build_state",
            as_index=False,
        )
        .agg(
            features=(
                "canonical_feature_name",
                "count",
            ),
            mean_receptor_coverage=(
                "coverage_fraction",
                "mean",
            ),
        )
        .sort_values(
            "features",
            ascending=False,
        )
    )

    complete_dynamic = int(
        (
            registry_v12[
                "nonmissing_receptors"
            ]
            == TARGET_RECEPTORS
        ).sum()
    )

    zero_dynamic = int(
        (
            registry_v12[
                "nonmissing_receptors"
            ]
            == 0
        ).sum()
    )

    medsea_complete = int(
        sum(
            dynamic_v12[
                f
            ]
            .notna()
            .all()
            for f
            in MEDSEA_CANONICAL_FEATURES
        )
    )

    medsea_zero = int(
        sum(
            dynamic_v12[
                f
            ]
            .isna()
            .all()
            for f
            in MEDSEA_CANONICAL_FEATURES
        )
    )

    if clim[
        "in_season"
    ]:
        overall = (
            "PASS_MEDSEA_CORRIDOR_ENGINE__COMPATIBILITY_AUDIT_REQUIRED"
        )
    else:
        overall = (
            "PASS_MEDSEA_CORRIDOR_GEOMETRY__OUT_OF_SEASON_ANOMALIES_INTENTIONALLY_NAN"
        )

    medsea_p = (
        snapshot_dir
        / "operational_medsea_corridor_v1_0.parquet"
    )

    diag_p = (
        snapshot_dir
        / "operational_medsea_corridor_diagnostics_v1_0.csv"
    )

    dynamic_p = (
        snapshot_dir
        / "operational_dynamic_features_v1_2.parquet"
    )

    full_p = (
        snapshot_dir
        / "operational_full_97_predictors_v1_2.parquet"
    )

    registry_p = (
        snapshot_dir
        / "operational_feature_build_registry_v1_2.csv"
    )

    coverage_p = (
        snapshot_dir
        / "operational_feature_coverage_v1_2.csv"
    )

    audit_json_p = (
        snapshot_dir
        / "operational_medsea_audit_v1_1.json"
    )

    audit_txt_p = (
        snapshot_dir
        / "operational_medsea_audit_v1_1.txt"
    )

    medsea.to_parquet(
        medsea_p,
        index=False,
    )

    diagnostics.to_csv(
        diag_p,
        index=False,
    )

    dynamic_v12.to_parquet(
        dynamic_p,
        index=False,
    )

    full_v12.to_parquet(
        full_p,
        index=False,
    )

    registry_v12.to_csv(
        registry_p,
        index=False,
    )

    coverage_v12.to_csv(
        coverage_p,
        index=False,
    )

    audit = {
        "version": "1.1",
        "overall_status": overall,
        "run_id": run_id,
        "issue_date": str(
            issue_date
        ),
        "in_core_season":
            bool(
                clim[
                    "in_season"
                ]
            ),
        "climatology_state":
            clim[
                "reason"
            ],
        "marine_file":
            str(
                marine_file
            ),
        "marine_file_mode":
            marine_file_mode,
        "current_product":
            CMEMS_DATASET_ID,
        "historical_training_marine_source":
            "Copernicus Marine Mediterranean Physics MY/reanalysis",
        "grid_exact_within_tolerance":
            True,
        "max_lat_grid_difference_deg":
            aligned[
                "max_lat_diff"
            ],
        "max_lon_grid_difference_deg":
            aligned[
                "max_lon_diff"
            ],
        "max_depth_grid_difference_m":
            aligned[
                "max_depth_diff"
            ],
        "vertical_closure_mean_abs_m":
            closure_mean_abs,
        "vertical_closure_max_abs_m":
            closure_max_abs,
        "medsea_features_total":
            len(
                MEDSEA_CANONICAL_FEATURES
            ),
        "medsea_features_complete_all_receptors":
            medsea_complete,
        "medsea_features_zero_coverage":
            medsea_zero,
        "dynamic_features_complete_all_receptors":
            complete_dynamic,
        "dynamic_features_zero_coverage":
            zero_dynamic,
        "zero_imputation_used":
            False,
        "v1_1_repair":
            (
                "Reference marine support mask is flattened to the canonical "
                "239x409 grid vector before sparse matrix multiplication; "
                "strict ncell assertions added."
            ),
        "support_method":
            {
                "n_sectors": N_SECTORS,
                "sector_step_deg":
                    SECTOR_STEP_DEG,
                "sigma_deg":
                    SIGMA_DEG,
                "scale_km":
                    SCALE_KM,
                "dmax_km":
                    DMAX_KM,
                "cutoffs_deg":
                    CUTOFFS,
            },
        "critical_semantic_caveat":
            (
                "Operational OHC/Tmean use daily ANFC thetao state while "
                "training OHC/Tmean were monthly MY/reanalysis states."
            ),
        "model_prediction_performed":
            False,
        "next_step":
            (
                "Run strict operational-vs-training feature range/distribution "
                "compatibility audit. Do not infer scientific flood probabilities "
                "until high-priority proxy shifts are quantified."
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

    display = diagnostics[
        [
            "receptor_id",
            "ivt_transport_bearing_deg",
            "marine_source_bearing_deg",
            "support_narrow",
            "support_baseline",
            "support_wide",
            "ohc_abs_corridor_j_m2",
            "tmean_abs_corridor_c",
            "sst_anom_corridor_c",
            "ohc_anom_corridor_j_m2",
            "climatology_state",
        ]
    ]

    lines = [
        "=" * 220,
        "NW HYDROCLIMATE — OPERATIONAL MEDSEA × IVT CORRIDOR v1.1",
        "=" * 220,
        f"OVERALL STATUS                         : {overall}",
        f"Run ID                                 : {run_id}",
        f"Issue date                             : {issue_date}",
        f"In CORE season Sep-Dec                 : {clim['in_season']}",
        f"Climatology state                      : {clim['reason']}",
        f"Marine file mode                       : {marine_file_mode}",
        f"Marine grid max lat diff               : {aligned['max_lat_diff']:.3e} deg",
        f"Marine grid max lon diff               : {aligned['max_lon_diff']:.3e} deg",
        f"Marine grid max depth diff             : {aligned['max_depth_diff']:.3e} m",
        f"Vertical closure max abs               : {closure_max_abs:.3e} m",
        f"MedSea canonical features              : {len(MEDSEA_CANONICAL_FEATURES)}",
        f"MedSea complete all receptors          : {medsea_complete}/{len(MEDSEA_CANONICAL_FEATURES)}",
        f"MedSea zero coverage                   : {medsea_zero}/{len(MEDSEA_CANONICAL_FEATURES)}",
        f"Dynamic complete all receptors         : {complete_dynamic}/{EXPECTED_DYNAMIC}",
        f"Dynamic zero coverage                  : {zero_dynamic}/{EXPECTED_DYNAMIC}",
        "Model prediction performed             : False",
        "Zero imputation used                   : False",
        "",
        "MEDSEA CORRIDOR DIAGNOSTICS",
        display.to_string(index=False),
        "",
        "FEATURE BUILD-STATE COUNTS",
        coverage_v12.to_string(index=False),
        "",
        "IMPORTANT",
        "The corridor geometry reproduces the frozen 16-sector Eulerian method.",
        "v1.1 fixes the reference-mask sparse-matrix dimension handling by flattening the canonical 2-D marine mask before multiplication.",
        "Source bearing is IVT transport bearing + 180 degrees.",
        "Support robustness uses the frozen ±30/±45/±60 degree policy.",
        "No-support physical values remain NaN; they are never zero-imputed.",
        "Operational OHC/Tmean are daily-state proxies for monthly training features.",
        "If the run is outside Sep-Dec, anomaly-dependent MedSea predictors are intentionally NaN.",
        "This script does not execute the flood model.",
        "",
        "NEXT STEP",
        "Run strict range/distribution compatibility audit versus historical F3-FIT/F3-VALIDATION before any beta inference.",
        "",
        f"MedSea predictors : {medsea_p}",
        f"Diagnostics       : {diag_p}",
        f"Dynamic v1.2      : {dynamic_p}",
        f"Full 97 v1.2      : {full_p}",
        f"Build registry    : {registry_p}",
        f"Coverage          : {coverage_p}",
        f"Audit             : {audit_json_p}",
        f"Output            : {snapshot_dir}",
    ]

    audit_txt_p.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    progress(
        "PHASE 7/7",
        1,
        1,
        start,
        f"status={overall}",
    )

    print(
        "\n"
        + "=" * 220
    )
    print(
        "\n".join(
            lines[
                3:
            ]
        )
    )
    print(
        "=" * 220
    )
    print(
        f"OVERALL STATUS : {overall}"
    )
    print(
        f"Output         : {snapshot_dir}"
    )
    print(
        "=" * 220
    )


if __name__ == "__main__":
    main()
