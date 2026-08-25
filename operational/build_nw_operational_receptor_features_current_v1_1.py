#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_nw_operational_receptor_features_current_v1_1.py

FASE 15 — PRIMO FEATURE ENGINE OPERATIVO A LIVELLO DI RECETTORE.

Prerequisito:
  nw_operational_raw_cache/<RUN_ID>/raw_cache_audit_v1_1.json
con:
  PASS_RAW_CACHE_SURFACE_REPAIRED_V1_1__FEATURE_ENGINE_READY

COSA FA
-------
1) legge il raw cache IFS/CMEMS già acquisito;
2) apre i GRIB2 ECMWF con cfgrib;
3) ritaglia l'area Piemonte-Liguria-Valle d'Aosta;
4) aggrega i campi atmosferici sui 20 recettori supervisionati;
5) costruisce tutte le feature ERA5->IFS che sono realmente calcolabili
   nel primo run;
6) costruisce precip_3d_incl_today usando t + t-1 + t-2;
7) costruisce il proxy IVT IFS su 1000..300 hPa;
8) unisce le 14 feature statiche canoniche;
9) produce uno skeleton ESATTAMENTE a 97 predictor nell'ordine del modello;
10) lascia NaN, senza zero-imputation, per:
    - rolling/lag non ancora disponibili durante warm-up;
    - feature MedSea-corridor non ancora ricostruite;
    - feature che richiedono un proxy non ancora scientificamente congelato.

IMPORTANTE
----------
Questo script NON esegue ancora il modello.

La finalità è verificare che il futuro runner possa produrre una riga
receptor-day strutturalmente compatibile con i modelli congelati.

SEMANTICA OPERATIVA
-------------------
- IFS 00Z + step 0..24 = "forecast-filled day t proxy".
- precip_max_1h è un proxy ottenuto dal massimo incremento 3h / 3;
  viene marcato NON_EXACT e NON viene utilizzato per fingere equivalenza.
- mucape è proxy della CAPE ERA5.
- vsw livelli 1-3 sono proxy dei soil-water ERA5.
- IVT è integrato sui livelli IFS Open Data disponibili:
  1000,925,850,700,600,500,400,300 hPa.
- tutte le feature MedSea/IVT-corridor restano NaN in questa v1.0:
  saranno costruite nella fase successiva con la geometria canonica.
- nessun NaN è sostituito con zero.

DIPENDENZE
----------
python -m pip install -U cfgrib eccodes xarray geopandas shapely pyarrow

OUTPUT
------
nw_operational_feature_snapshot/<RUN_ID>/
  operational_dynamic_features_v1_1.parquet
  operational_full_97_predictors_v1_1.parquet
  operational_feature_build_registry_v1_1.csv
  operational_feature_coverage_v1_1.csv
  operational_receptor_field_summary_v1_1.csv
  operational_feature_audit_v1_1.json
  operational_feature_audit_v1_1.txt
"""

from __future__ import annotations

import importlib.util
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


G = 9.80665
EXPECTED_DYNAMIC = 83
EXPECTED_STATIC = 14
EXPECTED_TOTAL = 97
TARGET_RECEPTORS = 20

ATM_BBOX = {
    "min_lon": 6.5,
    "max_lon": 9.6,
    "min_lat": 43.0,
    "max_lat": 46.6,
}

IVT_LEVELS_EXPECTED = [1000, 925, 850, 700, 600, 500, 400, 300]

HORIZONS = [24, 48, 72]


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


def require_modules():
    required = [
        ("xarray", "xarray"),
        ("cfgrib", "cfgrib"),
        ("geopandas", "geopandas"),
        ("shapely", "shapely"),
        ("pyarrow", "pyarrow"),
    ]
    missing = [pip for imp, pip in required if not module_available(imp)]
    if missing:
        raise SystemExit(
            "Dipendenze mancanti: "
            + ", ".join(missing)
            + "\nInstalla con:\n"
            + "python -m pip install -U "
            + " ".join(missing)
            + " eccodes"
        )


def latest_repaired_run(root):
    cache_root = root / "nw_operational_raw_cache"
    runs = sorted(
        [
            p for p in cache_root.iterdir()
            if p.is_dir()
            and (p / "raw_cache_audit_v1_1.json").exists()
        ],
        key=lambda p: p.name,
    )
    if not runs:
        raise SystemExit("Nessun raw-cache v1.1 riparato trovato.")
    return runs[-1]


def find_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    raise SystemExit(
        "Nessun file candidato trovato:\n"
        + "\n".join(str(p) for p in paths)
    )


def read_canonical_inputs(root):
    release = root / "nw_hydroclimate_core_release_v1_0"

    dynamic_p = find_first_existing(
        [
            release
            / "metadata"
            / "primary_dynamic_feature_whitelist_canonical_v1_3.csv",
            root
            / "nw_dynamic_causal_feature_whitelist_canonical_v1_3"
            / "primary_dynamic_feature_whitelist_canonical_v1_3.csv",
        ]
    )

    static_whitelist_p = find_first_existing(
        [
            release
            / "metadata"
            / "static_receptor_descriptor_whitelist_canonical_v1_1.csv",
            root
            / "nw_static_receptor_descriptor_whitelist_canonical_v1_1"
            / "static_receptor_descriptor_whitelist_canonical_v1_1.csv",
        ]
    )

    static_values_p = find_first_existing(
        [
            release
            / "metadata"
            / "static_receptor_descriptor_values_canonical_v1_1.csv",
            root
            / "nw_static_receptor_descriptor_whitelist_canonical_v1_1"
            / "static_receptor_descriptor_values_canonical_v1_1.csv",
        ]
    )

    dictionary_p = find_first_existing(
        [
            release
            / "metadata"
            / "nw_hydroclimate_master_predictor_dictionary_v1_0.csv",
            root
            / "nw_hydroclimate_foldwise_master_core_canonical_v1_0"
            / "nw_hydroclimate_master_predictor_dictionary_v1_0.csv",
        ]
    )

    receptors_p = find_first_existing(
        [
            release / "metadata" / "nw_receptors_final.geojson",
            root / "basins_final" / "nw_receptors_final.geojson",
        ]
    )

    return {
        "dynamic": pd.read_csv(dynamic_p, low_memory=False),
        "static_whitelist": pd.read_csv(static_whitelist_p, low_memory=False),
        "static_values": pd.read_csv(static_values_p, low_memory=False),
        "dictionary": pd.read_csv(dictionary_p, low_memory=False),
        "receptors_path": receptors_p,
    }


def open_grib_datasets(path):
    import cfgrib

    return cfgrib.open_datasets(
        str(path),
        backend_kwargs={"indexpath": ""},
    )


def find_var_dataset(datasets, var):
    for ds in datasets:
        if var in ds.data_vars:
            return ds
    return None


def coord_name(ds, candidates):
    for c in candidates:
        if c in ds.coords:
            return c
    for c in candidates:
        if c in ds.dims:
            return c
    return None


def crop_da(da):
    lat_name = coord_name(
        da,
        ["latitude", "lat"],
    )
    lon_name = coord_name(
        da,
        ["longitude", "lon"],
    )

    if lat_name is None or lon_name is None:
        raise RuntimeError(
            f"Lat/lon coords not found for {da.name}"
        )

    lat = da[lat_name]
    lon = da[lon_name]

    if float(lat[0]) > float(lat[-1]):
        da = da.sel(
            {
                lat_name: slice(
                    ATM_BBOX["max_lat"],
                    ATM_BBOX["min_lat"],
                )
            }
        )
    else:
        da = da.sel(
            {
                lat_name: slice(
                    ATM_BBOX["min_lat"],
                    ATM_BBOX["max_lat"],
                )
            }
        )

    if float(lon[0]) > float(lon[-1]):
        da = da.sel(
            {
                lon_name: slice(
                    ATM_BBOX["max_lon"],
                    ATM_BBOX["min_lon"],
                )
            }
        )
    else:
        da = da.sel(
            {
                lon_name: slice(
                    ATM_BBOX["min_lon"],
                    ATM_BBOX["max_lon"],
                )
            }
        )

    return da


def normalize_longitudes(lon):
    a = np.asarray(lon, dtype=float)
    return np.where(a > 180.0, a - 360.0, a)


def grid_mask_for_geom(da, geom):
    from shapely import contains_xy

    lat_name = coord_name(da, ["latitude", "lat"])
    lon_name = coord_name(da, ["longitude", "lon"])

    lats = np.asarray(da[lat_name].values, dtype=float)
    lons = normalize_longitudes(
        da[lon_name].values
    )

    xx, yy = np.meshgrid(lons, lats)

    mask = contains_xy(
        geom,
        xx,
        yy,
    )

    # Fallback for tiny basins with no grid-cell center inside:
    # use nearest cell to representative point.
    if not np.any(mask):
        p = geom.representative_point()
        d2 = (
            (xx - float(p.x)) ** 2
            + (yy - float(p.y)) ** 2
        )
        iy, ix = np.unravel_index(
            np.argmin(d2),
            d2.shape,
        )
        mask[iy, ix] = True

    return mask, lat_name, lon_name


def reduce_over_time_and_space(
    da,
    mask,
    spatial_stat="mean",
    temporal_stat="mean",
):
    """
    da dims may include step/time + lat + lon.
    """
    da = crop_da(da)

    lat_name = coord_name(da, ["latitude", "lat"])
    lon_name = coord_name(da, ["longitude", "lon"])

    spatial_dims = [lat_name, lon_name]

    mask_da = None
    import xarray as xr

    mask_da = xr.DataArray(
        mask,
        coords={
            lat_name: da[lat_name],
            lon_name: da[lon_name],
        },
        dims=(lat_name, lon_name),
    )

    x = da.where(mask_da)

    if spatial_stat == "mean":
        x = x.mean(
            dim=spatial_dims,
            skipna=True,
        )
    elif spatial_stat == "min":
        x = x.min(
            dim=spatial_dims,
            skipna=True,
        )
    elif spatial_stat == "max":
        x = x.max(
            dim=spatial_dims,
            skipna=True,
        )
    else:
        raise ValueError(spatial_stat)

    remaining = list(x.dims)

    # Remove vertical coordinate if singleton; pressure-level selection
    # is handled before calling.
    if temporal_stat == "mean":
        if remaining:
            x = x.mean(
                dim=remaining,
                skipna=True,
            )
    elif temporal_stat == "min":
        if remaining:
            x = x.min(
                dim=remaining,
                skipna=True,
            )
    elif temporal_stat == "max":
        if remaining:
            x = x.max(
                dim=remaining,
                skipna=True,
            )
    elif temporal_stat == "last":
        if remaining:
            # Prefer step.
            dim = "step" if "step" in remaining else remaining[0]
            x = x.isel({dim: -1})
            for d in list(x.dims):
                x = x.mean(dim=d, skipna=True)
    else:
        raise ValueError(temporal_stat)

    return float(x.values)


def select_level(da, level):
    level_name = coord_name(
        da,
        [
            "isobaricInhPa",
            "isobaricInPa",
            "level",
            "hybrid",
            "soilLayer",
        ],
    )

    if level_name is None:
        # Some files opened one level per dataset.
        return da

    values = np.asarray(
        da[level_name].values
    ).astype(float)

    if level_name == "isobaricInPa":
        target = level * 100.0
    else:
        target = level

    idx = int(
        np.argmin(
            np.abs(values - target)
        )
    )

    return da.isel(
        {level_name: idx}
    )


def extract_grib_vars(path, wanted):
    dsets = open_grib_datasets(path)
    out = {}
    for v in wanted:
        ds = find_var_dataset(dsets, v)
        if ds is not None:
            out[v] = crop_da(ds[v])
    return out


def find_grib_by_role(manifest, run_dir, role):
    rows = manifest[
        manifest["role"].astype(str).eq(role)
    ]
    if not len(rows):
        return None

    # Prefer v1.1 correction.
    if "manifest_version" in rows.columns:
        corr = rows[
            rows["manifest_version"]
            .astype(str)
            .str.contains("v1.1", regex=False)
        ]
        if len(corr):
            rows = corr

    rows = rows[
        rows["status"].astype(str).eq("PASS")
    ]

    if not len(rows):
        return None

    p = Path(str(rows.iloc[-1]["file"]))
    if p.exists():
        return p

    # If absolute path moved, reconstruct from basename.
    candidate = run_dir / "ecmwf" / p.name
    return candidate if candidate.exists() else None


def infer_receptor_id_column(gdf):
    for c in [
        "receptor_id",
        "receptor",
        "id",
        "name",
        "basin_id",
    ]:
        if c in gdf.columns:
            return c
    raise RuntimeError(
        "Cannot identify receptor id column in GeoJSON."
    )


def build_receptor_masks(receptors, reference_da):
    masks = {}
    for _, r in receptors.iterrows():
        rid = str(r["receptor_id"])
        masks[rid] = grid_mask_for_geom(
            reference_da,
            r.geometry,
        )[0]
    return masks


def basin_reduce_field(
    da,
    mask,
    *,
    spatial="mean",
    temporal="mean",
):
    return reduce_over_time_and_space(
        da,
        mask,
        spatial_stat=spatial,
        temporal_stat=temporal,
    )


def tp_step24_basin_mm(tp_da, mask):
    """
    ECMWF tp accumulated from initialization, unit m.
    Use last step = full current-day forecast accumulation.
    """
    return 1000.0 * basin_reduce_field(
        tp_da,
        mask,
        spatial="mean",
        temporal="last",
    )


def tp_max3h_rate_proxy_mm_h(tp_da, mask):
    """
    Cumulative tp -> increments -> max 3h mean intensity / 3.
    NON-EXACT proxy for ERA5 1-hour maximum.
    """
    da = crop_da(tp_da)

    step_dim = "step" if "step" in da.dims else None
    if step_dim is None:
        return np.nan

    lat_name = coord_name(da, ["latitude", "lat"])
    lon_name = coord_name(da, ["longitude", "lon"])

    import xarray as xr

    mask_da = xr.DataArray(
        mask,
        coords={
            lat_name: da[lat_name],
            lon_name: da[lon_name],
        },
        dims=(lat_name, lon_name),
    )

    basin = (
        da.where(mask_da)
        .mean(
            dim=[lat_name, lon_name],
            skipna=True,
        )
    )

    vals = np.asarray(
        basin.values,
        dtype=float,
    ).reshape(-1)

    if len(vals) < 2:
        return np.nan

    # tp requests start at step 3; first increment is step3 accumulated
    increments = np.diff(
        np.concatenate(
            [[0.0], vals]
        )
    )
    increments = np.maximum(
        increments,
        0.0,
    )

    return float(
        np.nanmax(increments) * 1000.0 / 3.0
    )


def ivt_components(ivt_vars, mask):
    """
    Returns daily basin:
      ivt_e_mean
      ivt_n_mean
      ivt_mag_mean
      ivt_mag_max
      dir_sin
      dir_cos

    Integration:
       IVT_x = 1/g * integral(q*u dp)
       IVT_y = 1/g * integral(q*v dp)
    using available IFS pressure levels.
    """
    import xarray as xr

    q = ivt_vars["q"]
    u = ivt_vars["u"]
    v = ivt_vars["v"]

    level_name = coord_name(
        q,
        ["isobaricInhPa", "isobaricInPa"],
    )
    if level_name is None:
        raise RuntimeError(
            "IVT pressure-level coordinate not found."
        )

    levels = np.asarray(
        q[level_name].values,
        dtype=float,
    )

    if level_name == "isobaricInhPa":
        p_pa = levels * 100.0
    else:
        p_pa = levels

    order = np.argsort(p_pa)
    p_sorted = p_pa[order]

    q = q.isel({level_name: order})
    u = u.isel({level_name: order})
    v = v.isel({level_name: order})

    # xarray integrate expects coordinate values.
    q = q.assign_coords(
        {level_name: p_sorted}
    )
    u = u.assign_coords(
        {level_name: p_sorted}
    )
    v = v.assign_coords(
        {level_name: p_sorted}
    )

    ivt_e = (
        (q * u).integrate(
            coord=level_name
        )
        / G
    )
    ivt_n = (
        (q * v).integrate(
            coord=level_name
        )
        / G
    )

    mag = np.sqrt(
        ivt_e ** 2
        + ivt_n ** 2
    )

    lat_name = coord_name(ivt_e, ["latitude", "lat"])
    lon_name = coord_name(ivt_e, ["longitude", "lon"])

    mask_da = xr.DataArray(
        mask,
        coords={
            lat_name: ivt_e[lat_name],
            lon_name: ivt_e[lon_name],
        },
        dims=(lat_name, lon_name),
    )

    def basin_series(x):
        return x.where(mask_da).mean(
            dim=[lat_name, lon_name],
            skipna=True,
        )

    e = basin_series(ivt_e)
    n = basin_series(ivt_n)
    m = basin_series(mag)

    e_mean = float(
        e.mean(dim=list(e.dims), skipna=True).values
        if e.dims else e.values
    )
    n_mean = float(
        n.mean(dim=list(n.dims), skipna=True).values
        if n.dims else n.values
    )
    mag_mean = float(
        m.mean(dim=list(m.dims), skipna=True).values
        if m.dims else m.values
    )
    mag_max = float(
        m.max(dim=list(m.dims), skipna=True).values
        if m.dims else m.values
    )

    dir_mag = math.hypot(
        e_mean,
        n_mean,
    )

    if dir_mag > 0:
        dir_cos = e_mean / dir_mag
        dir_sin = n_mean / dir_mag
    else:
        dir_cos = np.nan
        dir_sin = np.nan

    return {
        "era5__ivt_e_mean_kg_m1_s1": e_mean,
        "era5__ivt_n_mean_kg_m1_s1": n_mean,
        "era5__ivt_mag_mean_kg_m1_s1": mag_mean,
        "era5__ivt_mag_max_kg_m1_s1": mag_max,
        "era5__ivt_dir_cos": dir_cos,
        "era5__ivt_dir_sin": dir_sin,
    }


def season_day(issue_date):
    """
    Sep 1 = 1, ..., Dec 31.
    For outside Sep-Dec beta tests, continue as calendar-day offset from Sep 1
    of the issue year. This is explicitly diagnostic outside model season.
    """
    d = pd.Timestamp(issue_date)
    start = pd.Timestamp(
        year=d.year,
        month=9,
        day=1,
    )
    return int(
        (d.normalize() - start).days + 1
    )


def map_static_columns(static_whitelist):
    # canonical_feature_name may exist; otherwise infer from whitelist columns.
    for c in [
        "canonical_feature_name",
        "predictor",
        "feature_name",
        "static_feature_name",
    ]:
        if c in static_whitelist.columns:
            vals = (
                static_whitelist[c]
                .astype(str)
                .tolist()
            )
            if len(vals) == EXPECTED_STATIC:
                return vals
    raise RuntimeError(
        "Cannot infer the 14 static whitelist feature names."
    )


def normalize_static_value_columns(
    static_values,
    static_names,
    static_whitelist,
):
    """
    The canonical model predictor names are prefixed with ``static__``.
    The descriptor-value table can legitimately retain the original/raw
    descriptor names without that prefix.

    Build a deterministic one-to-one mapping:
        raw descriptor column -> canonical static__ predictor

    We first use exact canonical names, then exact unprefixed names, then
    explicit source/value-column fields from the whitelist when present,
    and finally a unique case-insensitive suffix match.

    No fuzzy many-to-one mapping is allowed.
    """
    df = static_values.copy()
    mapping_rows = []
    rename_map = {}

    whitelist_name_col = None
    for c in [
        "canonical_feature_name",
        "predictor",
        "feature_name",
        "static_feature_name",
    ]:
        if c in static_whitelist.columns:
            whitelist_name_col = c
            break

    source_cols = [
        c
        for c in [
            "source_column",
            "descriptor_column",
            "value_column",
            "raw_feature_name",
            "source_feature_name",
            "descriptor",
        ]
        if c in static_whitelist.columns
    ]

    lower_lookup = {
        str(c).lower(): str(c)
        for c in df.columns
    }

    for canonical in static_names:
        if canonical in df.columns:
            mapping_rows.append(
                {
                    "canonical_static_predictor": canonical,
                    "source_column": canonical,
                    "mapping_method": "EXACT_CANONICAL",
                }
            )
            continue

        raw = (
            canonical[len("static__"):]
            if canonical.startswith("static__")
            else canonical
        )

        candidates = []

        if raw in df.columns:
            candidates.append(
                (raw, "EXACT_UNPREFIXED")
            )

        raw_lower = raw.lower()
        if (
            raw_lower in lower_lookup
            and lower_lookup[raw_lower] != raw
        ):
            candidates.append(
                (
                    lower_lookup[raw_lower],
                    "CASE_INSENSITIVE_UNPREFIXED",
                )
            )

        if whitelist_name_col is not None and source_cols:
            wr = static_whitelist[
                static_whitelist[whitelist_name_col]
                .astype(str)
                .eq(canonical)
            ]

            if len(wr) == 1:
                for sc in source_cols:
                    value = wr.iloc[0].get(sc)
                    if pd.notna(value):
                        value = str(value)
                        if value in df.columns:
                            candidates.append(
                                (
                                    value,
                                    f"WHITELIST_{sc.upper()}",
                                )
                            )

        suffix_matches = [
            str(c)
            for c in df.columns
            if str(c).lower().endswith(raw_lower)
        ]

        if len(suffix_matches) == 1:
            candidates.append(
                (
                    suffix_matches[0],
                    "UNIQUE_SUFFIX_MATCH",
                )
            )

        # Deduplicate preserving order.
        dedup = []
        seen = set()
        for col, method in candidates:
            if col not in seen:
                seen.add(col)
                dedup.append((col, method))

        if len(dedup) == 0:
            mapping_rows.append(
                {
                    "canonical_static_predictor": canonical,
                    "source_column": "",
                    "mapping_method": "UNRESOLVED",
                }
            )
            continue

        if len(dedup) > 1:
            cols = {x[0] for x in dedup}
            if len(cols) > 1:
                raise RuntimeError(
                    "Ambiguous static mapping for "
                    f"{canonical}: {dedup}"
                )

        source_col, method = dedup[0]

        if source_col in rename_map and rename_map[source_col] != canonical:
            raise RuntimeError(
                "Static source column mapped twice: "
                f"{source_col} -> {rename_map[source_col]} and {canonical}"
            )

        rename_map[source_col] = canonical

        mapping_rows.append(
            {
                "canonical_static_predictor": canonical,
                "source_column": source_col,
                "mapping_method": method,
            }
        )

    df = df.rename(
        columns=rename_map
    )

    mapping = pd.DataFrame(
        mapping_rows
    )

    unresolved = mapping[
        mapping["mapping_method"].eq("UNRESOLVED")
    ]["canonical_static_predictor"].tolist()

    return df, mapping, unresolved


def main():
    require_modules()

    import geopandas as gpd

    root = Path(__file__).resolve().parent
    run_dir = latest_repaired_run(root)
    run_id = run_dir.name

    audit_p = run_dir / "raw_cache_audit_v1_1.json"
    audit = json.loads(
        audit_p.read_text(encoding="utf-8")
    )

    if (
        audit.get("overall_status")
        != "PASS_RAW_CACHE_SURFACE_REPAIRED_V1_1__FEATURE_ENGINE_READY"
    ):
        raise SystemExit(
            "Raw cache v1.1 non pronto:\n"
            + str(audit.get("overall_status"))
        )

    issue_date = pd.Timestamp(
        audit["issue_cycle_utc"]
    ).date()

    canonical = read_canonical_inputs(root)

    dynamic_whitelist = canonical["dynamic"]
    static_whitelist = canonical["static_whitelist"]
    static_values = canonical["static_values"]
    dictionary = canonical["dictionary"]

    if len(dynamic_whitelist) != EXPECTED_DYNAMIC:
        raise SystemExit(
            f"Dynamic whitelist={len(dynamic_whitelist)}, expected=83"
        )
    if len(dictionary) != EXPECTED_TOTAL:
        raise SystemExit(
            f"Predictor dictionary={len(dictionary)}, expected=97"
        )

    out = (
        root
        / "nw_operational_feature_snapshot"
        / run_id
    )
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 220)
    print("NW HYDROCLIMATE — OPERATIONAL RECEPTOR FEATURE ENGINE v1.1")
    print("=" * 220)
    print(f"Run ID     : {run_id}")
    print(f"Issue date : {issue_date}")

    # ------------------------------------------------------------------
    # PHASE 1/6 — load corrected raw manifest and locate field files
    # ------------------------------------------------------------------
    print("\nPHASE 1/6 — load corrected raw manifest and canonical receptor geometry")
    start = time.time()

    manifest = pd.read_csv(
        run_dir / "raw_cache_manifest_v1_1.csv",
        low_memory=False,
    )

    receptors = gpd.read_file(
        canonical["receptors_path"]
    ).to_crs(4326)

    rid_col = infer_receptor_id_column(
        receptors
    )
    receptors = receptors.rename(
        columns={rid_col: "receptor_id"}
    )

    # Model has 20 target receptors; LIG_ENTELLA is intentionally static-only.
    receptors = receptors[
        ~receptors["receptor_id"]
        .astype(str)
        .eq("LIG_ENTELLA")
    ].copy()

    if len(receptors) != TARGET_RECEPTORS:
        raise SystemExit(
            f"Target receptor polygons={len(receptors)}, expected=20"
        )

    # Derive the crop from the actual receptor geometries instead of relying
    # on a hard-coded eastern limit. This avoids truncating eastern receptors
    # such as LIG_MAGRA.
    global ATM_BBOX
    minx, miny, maxx, maxy = receptors.total_bounds
    pad = 0.40
    ATM_BBOX = {
        "min_lon": float(minx - pad),
        "max_lon": float(maxx + pad),
        "min_lat": float(miny - pad),
        "max_lat": float(maxy + pad),
    }

    files = {
        "safe_surface":
            find_grib_by_role(
                manifest,
                run_dir,
                "SURFACE_SAFE_MSL_TCWV_SD",
            ),
        "mucape":
            find_grib_by_role(
                manifest,
                run_dir,
                "SURFACE_MUCAPE_PROXY",
            ),
        "vsw1":
            find_grib_by_role(
                manifest,
                run_dir,
                "SOIL_VSW_LAYER1_PROXY",
            ),
        "vsw2":
            find_grib_by_role(
                manifest,
                run_dir,
                "SOIL_VSW_LAYER2_PROXY",
            ),
        "vsw3":
            find_grib_by_role(
                manifest,
                run_dir,
                "SOIL_VSW_LAYER3_PROXY",
            ),
        "tp_current":
            find_grib_by_role(
                manifest,
                run_dir,
                "CURRENT_DAY_PRECIP_ACCUM",
            ),
        "low_levels":
            find_grib_by_role(
                manifest,
                run_dir,
                "CURRENT_DAY_LOW_PRESSURE_LEVELS",
            ),
        "ivt_inputs":
            find_grib_by_role(
                manifest,
                run_dir,
                "CURRENT_DAY_IVT_INPUTS",
            ),
        "tp_lag1":
            find_grib_by_role(
                manifest,
                run_dir,
                "PRIOR_DAY_TP_BOOTSTRAP_LAG1D",
            ),
        "tp_lag2":
            find_grib_by_role(
                manifest,
                run_dir,
                "PRIOR_DAY_TP_BOOTSTRAP_LAG2D",
            ),
    }

    critical_missing = [
        k for k in [
            "safe_surface",
            "mucape",
            "vsw1",
            "vsw2",
            "vsw3",
            "tp_current",
            "low_levels",
            "ivt_inputs",
        ]
        if files[k] is None
    ]

    if critical_missing:
        raise SystemExit(
            "Missing critical raw files: "
            + ", ".join(critical_missing)
        )

    progress(
        "PHASE 1/6",
        1,
        1,
        start,
        f"receptors={len(receptors)} | bbox={ATM_BBOX} | critical raw files PASS",
    )

    # ------------------------------------------------------------------
    # PHASE 2/6 — open GRIB fields and construct receptor masks
    # ------------------------------------------------------------------
    print("\nPHASE 2/6 — decode GRIB2 and construct receptor masks")
    start = time.time()

    safe = extract_grib_vars(
        files["safe_surface"],
        ["msl", "tcwv", "sd"],
    )
    mucape = extract_grib_vars(
        files["mucape"],
        ["mucape"],
    )
    vsw1 = extract_grib_vars(
        files["vsw1"],
        ["vsw"],
    )
    vsw2 = extract_grib_vars(
        files["vsw2"],
        ["vsw"],
    )
    vsw3 = extract_grib_vars(
        files["vsw3"],
        ["vsw"],
    )
    tp = extract_grib_vars(
        files["tp_current"],
        ["tp"],
    )
    low = extract_grib_vars(
        files["low_levels"],
        ["q", "u", "v", "t"],
    )
    ivt = extract_grib_vars(
        files["ivt_inputs"],
        ["q", "u", "v"],
    )

    if "msl" not in safe:
        raise SystemExit(
            "Cannot decode msl from corrected GRIB."
        )

    masks = build_receptor_masks(
        receptors,
        safe["msl"],
    )

    progress(
        "PHASE 2/6",
        1,
        1,
        start,
        (
            f"decoded safe={sorted(safe)} mucape={sorted(mucape)} "
            f"tp={sorted(tp)} low={sorted(low)} ivt={sorted(ivt)}"
        ),
    )

    # ------------------------------------------------------------------
    # PHASE 3/6 — current atmospheric receptor-day features
    # ------------------------------------------------------------------
    print("\nPHASE 3/6 — build current atmospheric receptor-day features")
    start = time.time()

    rows = []
    field_summary_rows = []

    for i, (_, rr) in enumerate(
        receptors.sort_values("receptor_id").iterrows(),
        1,
    ):
        rid = str(rr["receptor_id"])
        mask = masks[rid]

        values = {
            "receptor_id": rid,
            "issue_date": str(issue_date),
            "run_id": run_id,
        }

        # Calendar.
        values["era5__season_day"] = season_day(
            issue_date
        )

        # Surface pressure / tcwv.
        values["era5__mslp_mean_pa"] = basin_reduce_field(
            safe["msl"],
            mask,
            spatial="mean",
            temporal="mean",
        )
        values["era5__mslp_min_pa"] = basin_reduce_field(
            safe["msl"],
            mask,
            spatial="min",
            temporal="min",
        )
        values["era5__tcwv_mean_kg_m2"] = basin_reduce_field(
            safe["tcwv"],
            mask,
            spatial="mean",
            temporal="mean",
        )
        values["era5__tcwv_max_kg_m2"] = basin_reduce_field(
            safe["tcwv"],
            mask,
            spatial="max",
            temporal="max",
        )

        # mucape proxy.
        if "mucape" in mucape:
            values["era5__cape_mean_j_kg"] = basin_reduce_field(
                mucape["mucape"],
                mask,
                spatial="mean",
                temporal="mean",
            )
            values["era5__cape_max_j_kg"] = basin_reduce_field(
                mucape["mucape"],
                mask,
                spatial="max",
                temporal="max",
            )

        # Snow.
        values["era5__snow_depth_mwe"] = basin_reduce_field(
            safe["sd"],
            mask,
            spatial="mean",
            temporal="mean",
        )

        # Soil-water proxies.
        soil_vals = []
        for layer, source in [
            (1, vsw1),
            (2, vsw2),
            (3, vsw3),
        ]:
            if "vsw" in source:
                v = basin_reduce_field(
                    source["vsw"],
                    mask,
                    spatial="mean",
                    temporal="mean",
                )
            else:
                v = np.nan

            values[
                f"era5__soil_water_l{layer}_m3_m3"
            ] = v
            soil_vals.append(v)

        values[
            "era5__soil_profile_mean_m3_m3"
        ] = (
            float(np.nanmean(soil_vals))
            if np.isfinite(soil_vals).any()
            else np.nan
        )

        # Current precipitation.
        values["era5__precip_sum_mm"] = tp_step24_basin_mm(
            tp["tp"],
            mask,
        )
        values[
            "era5__precip_max_1h_mm"
        ] = tp_max3h_rate_proxy_mm_h(
            tp["tp"],
            mask,
        )

        # Low pressure levels.
        for level in [700, 850, 925]:
            q_l = select_level(
                low["q"],
                level,
            )
            u_l = select_level(
                low["u"],
                level,
            )
            v_l = select_level(
                low["v"],
                level,
            )
            t_l = select_level(
                low["t"],
                level,
            )

            qmean = basin_reduce_field(
                q_l,
                mask,
                spatial="mean",
                temporal="mean",
            )
            umean = basin_reduce_field(
                u_l,
                mask,
                spatial="mean",
                temporal="mean",
            )
            vmean = basin_reduce_field(
                v_l,
                mask,
                spatial="mean",
                temporal="mean",
            )
            tmean = basin_reduce_field(
                t_l,
                mask,
                spatial="mean",
                temporal="mean",
            )

            # Wind magnitude is computed before reduction.
            wind_l = np.sqrt(
                u_l ** 2
                + v_l ** 2
            )

            windmean = basin_reduce_field(
                wind_l,
                mask,
                spatial="mean",
                temporal="mean",
            )
            windmax = basin_reduce_field(
                wind_l,
                mask,
                spatial="max",
                temporal="max",
            )

            qwind = basin_reduce_field(
                q_l * wind_l,
                mask,
                spatial="mean",
                temporal="mean",
            )

            values[
                f"era5__q{level}_mean_kg_kg"
            ] = qmean
            values[
                f"era5__u{level}_mean_m_s"
            ] = umean
            values[
                f"era5__v{level}_mean_m_s"
            ] = vmean
            values[
                f"era5__t{level}_mean_k"
            ] = tmean
            values[
                f"era5__wind{level}_mean_m_s"
            ] = windmean
            values[
                f"era5__wind{level}_max_m_s"
            ] = windmax
            values[
                f"era5__qwind{level}_proxy"
            ] = qwind

        # IVT operational proxy.
        ivt_values = ivt_components(
            ivt,
            mask,
        )
        values.update(
            ivt_values
        )

        rows.append(values)

        field_summary_rows.append(
            {
                "receptor_id": rid,
                "precip_sum_mm":
                    values["era5__precip_sum_mm"],
                "mslp_mean_pa":
                    values["era5__mslp_mean_pa"],
                "tcwv_mean_kg_m2":
                    values["era5__tcwv_mean_kg_m2"],
                "ivt_mag_mean_kg_m1_s1":
                    values["era5__ivt_mag_mean_kg_m1_s1"],
                "cape_proxy_mean_j_kg":
                    values.get(
                        "era5__cape_mean_j_kg",
                        np.nan,
                    ),
            }
        )

        progress(
            "PHASE 3/6",
            i,
            len(receptors),
            start,
            (
                f"{rid} | P={values['era5__precip_sum_mm']:.2f} mm "
                f"| IVT={values['era5__ivt_mag_mean_kg_m1_s1']:.1f}"
            ),
        )

    current = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # PHASE 4/6 — bootstrap precip lags and build 83-column dynamic skeleton
    # ------------------------------------------------------------------
    print("\nPHASE 4/6 — bootstrap precipitation antecedents and freeze 83-column skeleton")
    start = time.time()

    for lag_name, file_key in [
        ("lag1", "tp_lag1"),
        ("lag2", "tp_lag2"),
    ]:
        if files[file_key] is None:
            current[
                f"_bootstrap_tp_{lag_name}_mm"
            ] = np.nan
            continue

        obj = extract_grib_vars(
            files[file_key],
            ["tp"],
        )

        if "tp" not in obj:
            current[
                f"_bootstrap_tp_{lag_name}_mm"
            ] = np.nan
            continue

        vals = []

        for _, r in current.iterrows():
            rid = str(r["receptor_id"])
            vals.append(
                tp_step24_basin_mm(
                    obj["tp"],
                    masks[rid],
                )
            )

        current[
            f"_bootstrap_tp_{lag_name}_mm"
        ] = vals

    # t-1.
    current[
        "era5__precip_prev1d_mm"
    ] = current[
        "_bootstrap_tp_lag1_mm"
    ]

    # 3d including today = t + t-1 + t-2.
    current[
        "era5__precip_3d_incl_today_mm"
    ] = (
        current["era5__precip_sum_mm"]
        + current["_bootstrap_tp_lag1_mm"]
        + current["_bootstrap_tp_lag2_mm"]
    )

    # Cannot construct t-1..t-3 with only two retained prior cycles.
    current[
        "era5__precip_prev3d_mm"
    ] = np.nan
    current[
        "era5__precip_prev7d_mm"
    ] = np.nan
    current[
        "era5__precip_prev14d_mm"
    ] = np.nan
    current[
        "era5__precip_7d_incl_today_mm"
    ] = np.nan

    # Hourly-max antecedents cannot be recovered from step24 totals.
    current[
        "era5__precip_max1h_prev1d_mm"
    ] = np.nan
    current[
        "era5__precip_max1h_prev3d_max_mm"
    ] = np.nan

    dynamic_names = (
        dynamic_whitelist[
            "canonical_feature_name"
        ]
        .astype(str)
        .tolist()
    )

    # Ensure every canonical feature exists, but never fabricate.
    for f in dynamic_names:
        if f not in current.columns:
            current[f] = np.nan

    dynamic = current[
        [
            "receptor_id",
            "issue_date",
            "run_id",
            *dynamic_names,
        ]
    ].copy()

    progress(
        "PHASE 4/6",
        1,
        1,
        start,
        (
            f"dynamic features={len(dynamic_names)} "
            f"| precip3d_ready={dynamic['era5__precip_3d_incl_today_mm'].notna().all()}"
        ),
    )

    # ------------------------------------------------------------------
    # PHASE 5/6 — attach static descriptors and enforce 97 predictor order
    # ------------------------------------------------------------------
    print("\nPHASE 5/6 — attach 14 static descriptors and enforce frozen 97-predictor order")
    start = time.time()

    static_names = map_static_columns(
        static_whitelist
    )

    # Normalize receptor id.
    if "receptor_id" not in static_values.columns:
        for c in [
            "receptor",
            "id",
            "basin_id",
        ]:
            if c in static_values.columns:
                static_values = static_values.rename(
                    columns={c: "receptor_id"}
                )
                break

    if "receptor_id" not in static_values.columns:
        raise SystemExit(
            "Static values missing receptor_id."
        )

    static_values, static_mapping, unresolved_static = (
        normalize_static_value_columns(
            static_values,
            static_names,
            static_whitelist,
        )
    )

    static_mapping_p = (
        out
        / "operational_static_column_mapping_v1_1.csv"
    )
    static_mapping.to_csv(
        static_mapping_p,
        index=False,
    )

    missing_static_cols = [
        c for c in static_names
        if c not in static_values.columns
    ]

    if unresolved_static or missing_static_cols:
        print(
            "\nSTATIC VALUE COLUMNS ACTUALLY FOUND:\n"
            + "\n".join(
                f"  - {c}"
                for c in static_values.columns
            ),
            flush=True,
        )
        print(
            "\nSTATIC MAPPING ATTEMPT:\n"
            + static_mapping.to_string(index=False),
            flush=True,
        )
        raise SystemExit(
            "Static mapping unresolved. Missing canonical columns: "
            + ", ".join(
                sorted(
                    set(unresolved_static + missing_static_cols)
                )
            )
        )

    full = dynamic.merge(
        static_values[
            [
                "receptor_id",
                *static_names,
            ]
        ],
        on="receptor_id",
        how="left",
        validate="one_to_one",
    )

    predictor_order = (
        dictionary["predictor"]
        .astype(str)
        .tolist()
    )

    for p in predictor_order:
        if p not in full.columns:
            full[p] = np.nan

    full97 = full[
        [
            "receptor_id",
            "issue_date",
            "run_id",
            *predictor_order,
        ]
    ].copy()

    if len(predictor_order) != EXPECTED_TOTAL:
        raise SystemExit(
            f"Predictor order={len(predictor_order)}, expected=97"
        )

    static_missing = int(
        full97[static_names]
        .isna()
        .sum()
        .sum()
    )

    progress(
        "PHASE 5/6",
        1,
        1,
        start,
        (
            f"rows={len(full97)} predictors={len(predictor_order)} "
            f"| static missing cells={static_missing}"
        ),
    )

    # ------------------------------------------------------------------
    # PHASE 6/6 — coverage/build registry/audit
    # ------------------------------------------------------------------
    print("\nPHASE 6/6 — feature coverage, build registry and audit")
    start = time.time()

    feature_rows = []

    for f in dynamic_names:
        n_nonmissing = int(
            dynamic[f].notna().sum()
        )

        if f.startswith("medsea_ivt__"):
            build_state = "PENDING_MEDSEA_CORRIDOR_ENGINE"
            semantics = (
                "Not built in operational feature engine v1.0; "
                "must remain NaN until canonical marine corridor proxy is implemented."
            )
        elif f == "era5__precip_max_1h_mm":
            build_state = (
                "BUILT_NON_EXACT_3H_RATE_PROXY"
                if n_nonmissing
                else "MISSING"
            )
            semantics = (
                "Max 3h accumulation divided by 3; not exact ERA5 hourly maximum."
            )
        elif "cape" in f:
            build_state = (
                "BUILT_MUCAPE_PROXY"
                if n_nonmissing
                else "MISSING"
            )
            semantics = (
                "IFS mucape proxy; not identical to ERA5 CAPE."
            )
        elif "soil_" in f and n_nonmissing:
            build_state = "BUILT_IFS_VSW_PROXY"
            semantics = (
                "IFS volumetric soil water proxy; distribution validation pending."
            )
        elif "ivt" in f and n_nonmissing:
            build_state = "BUILT_IFS_REDUCED_LEVEL_IVT_PROXY"
            semantics = (
                "IFS Open Data pressure-level integration; compatibility validation pending."
            )
        elif n_nonmissing == TARGET_RECEPTORS:
            build_state = "BUILT_ALL_RECEPTORS"
            semantics = "Operational current-day/recent proxy constructed."
        elif n_nonmissing > 0:
            build_state = "PARTIAL"
            semantics = "Available for only a subset of receptors."
        else:
            build_state = "MISSING_WARMUP_OR_NOT_IMPLEMENTED"
            semantics = (
                "Left NaN by design; never zero-filled."
            )

        feature_rows.append(
            {
                "canonical_feature_name": f,
                "nonmissing_receptors":
                    n_nonmissing,
                "total_receptors":
                    TARGET_RECEPTORS,
                "coverage_fraction":
                    n_nonmissing / TARGET_RECEPTORS,
                "build_state":
                    build_state,
                "operational_semantics":
                    semantics,
            }
        )

    build_registry = pd.DataFrame(
        feature_rows
    )

    coverage = (
        build_registry.groupby(
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

    p1_p = (
        root
        / "nw_operational_feature_equivalence_preflight_v1_0"
        / "operational_priority_features_v1_0.csv"
    )

    p1_coverage = None

    if p1_p.exists():
        priority = pd.read_csv(
            p1_p,
            low_memory=False,
        )

        p1 = priority[
            priority["operational_priority"]
            .astype(str)
            .eq("P1_TOP10_ANY_HORIZON")
        ][
            ["canonical_feature_name"]
        ].drop_duplicates()

        p1_coverage = p1.merge(
            build_registry,
            on="canonical_feature_name",
            how="left",
        )

    dynamic_p = (
        out
        / "operational_dynamic_features_v1_1.parquet"
    )
    full_p = (
        out
        / "operational_full_97_predictors_v1_1.parquet"
    )
    registry_p = (
        out
        / "operational_feature_build_registry_v1_1.csv"
    )
    coverage_p = (
        out
        / "operational_feature_coverage_v1_1.csv"
    )
    field_summary_p = (
        out
        / "operational_receptor_field_summary_v1_1.csv"
    )
    audit_json_p = (
        out
        / "operational_feature_audit_v1_1.json"
    )
    audit_txt_p = (
        out
        / "operational_feature_audit_v1_1.txt"
    )

    dynamic.to_parquet(
        dynamic_p,
        index=False,
    )
    full97.to_parquet(
        full_p,
        index=False,
    )
    build_registry.to_csv(
        registry_p,
        index=False,
    )
    coverage.to_csv(
        coverage_p,
        index=False,
    )
    pd.DataFrame(
        field_summary_rows
    ).to_csv(
        field_summary_p,
        index=False,
    )

    complete_dynamic = int(
        (
            build_registry["nonmissing_receptors"]
            == TARGET_RECEPTORS
        ).sum()
    )
    zero_dynamic = int(
        (
            build_registry["nonmissing_receptors"]
            == 0
        ).sum()
    )

    p1_ready = None
    if p1_coverage is not None:
        p1_ready = int(
            (
                p1_coverage["nonmissing_receptors"]
                == TARGET_RECEPTORS
            ).sum()
        )

    static_all_ready = (
        static_missing == 0
    )

    if (
        len(full97) == TARGET_RECEPTORS
        and len(predictor_order) == EXPECTED_TOTAL
        and static_all_ready
    ):
        overall = (
            "PASS_97_COLUMN_OPERATIONAL_SKELETON__BETA_DEGRADED_WARMUP"
        )
    else:
        overall = (
            "FAIL_OPERATIONAL_FEATURE_STRUCTURE"
        )

    audit_out = {
        "version": "1.1",
        "overall_status": overall,
        "run_id": run_id,
        "issue_date": str(issue_date),
        "receptors": int(len(full97)),
        "dynamic_features": EXPECTED_DYNAMIC,
        "static_features": EXPECTED_STATIC,
        "total_predictors": EXPECTED_TOTAL,
        "dynamic_features_complete_all_receptors":
            complete_dynamic,
        "dynamic_features_zero_coverage":
            zero_dynamic,
        "static_missing_cells":
            static_missing,
        "p1_features_ready_all_receptors":
            p1_ready,
        "model_prediction_performed":
            False,
        "medsea_corridor_engine_performed":
            False,
        "cache_warmup_complete":
            False,
        "zero_imputation_used":
            False,
        "static_value_column_mapping":
            static_mapping.to_dict(orient="records"),
        "atmospheric_crop_bbox":
            ATM_BBOX,
        "current_day_semantics":
            "IFS_00Z_FORECAST_FILLED_DAY_T_PROXY",
        "next_step": (
            "Implement MedSea corridor/support operational engine, then run "
            "strict feature-compatibility and range audit before first beta inference."
        ),
    }

    audit_json_p.write_text(
        json.dumps(
            audit_out,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    p1_text = (
        p1_coverage.to_string(index=False)
        if p1_coverage is not None
        else "P1 registry not found."
    )

    lines = [
        "=" * 220,
        "NW HYDROCLIMATE — OPERATIONAL RECEPTOR FEATURE ENGINE v1.1",
        "=" * 220,
        f"OVERALL STATUS                         : {overall}",
        f"Run ID                                 : {run_id}",
        f"Receptors                              : {len(full97)}",
        f"Dynamic features                       : {EXPECTED_DYNAMIC}",
        f"Static features                        : {EXPECTED_STATIC}",
        f"Total predictors                       : {len(predictor_order)}",
        f"Dynamic complete on all receptors      : {complete_dynamic}/{EXPECTED_DYNAMIC}",
        f"Dynamic zero coverage                  : {zero_dynamic}/{EXPECTED_DYNAMIC}",
        f"Static missing cells                   : {static_missing}",
        f"P1 ready on all receptors              : {p1_ready if p1_ready is not None else 'N/A'}",
        "MedSea corridor engine                 : NOT YET",
        "Cache warm-up complete                 : False",
        "Model prediction performed             : False",
        "Zero imputation used                   : False",
        "",
        "FEATURE BUILD-STATE COUNTS",
        coverage.to_string(index=False),
        "",
        "P1 FEATURE COVERAGE",
        p1_text,
        "",
        "RECEPTOR CURRENT FIELD SUMMARY",
        pd.DataFrame(field_summary_rows).to_string(index=False),
        "",
        "IMPORTANT",
        "The output is structurally compatible with the frozen 97-predictor model input.",
        "Static descriptor raw columns are deterministically mapped to canonical static__ predictor names and audited.",
        "Atmospheric crop bounds are derived from the 20 receptor geometries, avoiding hard-coded eastern truncation.",
        "It is NOT yet a full-equivalence operational feature vector.",
        "mucape, vsw and reduced-level IFS IVT remain explicit proxies.",
        "MedSea corridor variables remain NaN until the next phase.",
        "7d/14d and other unavailable antecedent lags remain NaN during warm-up.",
        "No NaN has been replaced with zero.",
        "",
        "NEXT STEP",
        "Build the canonical MedSea corridor/support proxy and then perform a strict range/compatibility audit before beta inference.",
        "",
        f"Dynamic snapshot : {dynamic_p}",
        f"Full 97          : {full_p}",
        f"Build registry   : {registry_p}",
        f"Static mapping   : {static_mapping_p}",
        f"Coverage         : {coverage_p}",
        f"Audit            : {audit_json_p}",
        f"Output           : {out}",
    ]

    audit_txt_p.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    progress(
        "PHASE 6/6",
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
