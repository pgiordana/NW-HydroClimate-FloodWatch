#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_era5_basin_features_historical_v1_0.py

Produzione storica ERA5 -> feature giornaliere x 21 bacini
Sep-Dic 1987-2025.

PREREQUISITI:
- ERA5 historical audit v1.0 = PASS (624/624);
- probe ERA5 -> basin features 2025-12 v1.0 = PASS;
- pesi spaziali validati disponibili in:
    era5_historical_nw/basin_features_probe_v1_0/
      source_grid_weights_v1_0.csv
      target_grid_weights_v1_0.csv

METODO:
- overlap frazionale cella-bacino;
- pesi area-normalizzati;
- nessun nearest-cell fallback;
- elaborazione mese per mese, restart-safe;
- raw NetCDF non modificati.

PER OGNI GIORNO x BACINO:
SOURCE 3h
- IVT est/nord medio;
- IVT magnitudine media e massima;
- direzione del vettore IVT medio;
- TCWV medio/max;
- CAPE medio/max;
- MSLP medio/min.

PRESSURE 3h
- u, v, velocità vento media/max;
- umidità specifica q media;
- temperatura media;
  per 925, 850, 700 hPa.

PRECIP 1h
- precipitazione totale giornaliera [mm];
- massimo orario [mm].

STATE 1d
- soil water layers 1-3;
- snow depth water equivalent.

OUTPUT:
era5_historical_nw/basin_features_historical_v1_0/
  monthly/YYYY/era5_basin_features_YYYYMM_v1_0.csv
  era5_basin_features_daily_1987_2025_v1_0.csv
  era5_basin_features_manifest_v1_0.csv
  era5_basin_features_audit_v1_0.json
  era5_basin_features_audit_v1_0.txt

RIGHE FINALI ATTESE:
39 anni x 122 giorni x 21 bacini = 99,918.

Nota:
questa versione costruisce le feature ERA5 "base".
Accumuli antecedenti, anomalie climatologiche, persistenza IVT e altre feature
derivate verranno costruiti in uno step successivo, evitando di collegare
artificialmente dicembre con settembre dell'anno seguente.
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


START_YEAR = 1987
END_YEAR = 2025
MONTHS = [9, 10, 11, 12]
RECEPTORS_EXPECTED = 21
DAYS_PER_SEASON = 122
EXPECTED_FINAL_ROWS = (END_YEAR - START_YEAR + 1) * DAYS_PER_SEASON * RECEPTORS_EXPECTED

LEVELS = [925, 850, 700]

SOURCE_VARS = ["viwve", "viwvn", "tcwv", "cape", "msl"]
PRESSURE_VARS = ["u", "v", "q", "t"]
PRECIP_VARS = ["tp"]
STATE_VARS = ["swvl1", "swvl2", "swvl3", "sd"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--force",
        action="store_true",
        help="Ricalcola anche i mesi già prodotti.",
    )
    ap.add_argument(
        "--start-year",
        type=int,
        default=START_YEAR,
        help=f"Anno iniziale (default {START_YEAR}).",
    )
    ap.add_argument(
        "--end-year",
        type=int,
        default=END_YEAR,
        help=f"Anno finale (default {END_YEAR}).",
    )
    return ap.parse_args()


def weight_matrix(weights: pd.DataFrame, receptor_ids, nlat, nlon):
    W = np.zeros((nlat * nlon, len(receptor_ids)), dtype=np.float64)
    bmap = {rid: i for i, rid in enumerate(receptor_ids)}

    for r in weights.itertuples(index=False):
        rid = str(r.receptor_id)
        if rid not in bmap:
            continue
        W[int(r.cell_flat_idx), bmap[rid]] = float(r.normalized_weight)

    sums = W.sum(axis=0)
    if not np.allclose(sums, 1.0, atol=1e-10):
        raise RuntimeError(f"Pesi non normalizzati: {sums}")

    return W


def spatial_weighted(da: xr.DataArray, W: np.ndarray):
    if "latitude" not in da.dims or "longitude" not in da.dims:
        raise ValueError(
            f"{da.name}: dimensioni spaziali mancanti: {da.dims}"
        )

    nonspatial = [
        d for d in da.dims
        if d not in ("latitude", "longitude")
    ]

    arr = da.transpose(
        *nonspatial, "latitude", "longitude"
    ).values

    lead_shape = arr.shape[:-2]
    flat = arr.reshape(
        (-1, arr.shape[-2] * arr.shape[-1])
    ).astype(np.float64)

    finite = np.isfinite(flat)
    numerator = np.nan_to_num(flat, nan=0.0) @ W
    denominator = finite.astype(np.float64) @ W

    with np.errstate(divide="ignore", invalid="ignore"):
        out = numerator / denominator

    out[denominator <= 0] = np.nan

    return out.reshape((*lead_shape, W.shape[1]))


def require_vars(ds, names, family):
    missing = [v for v in names if v not in ds.data_vars]
    if missing:
        raise RuntimeError(
            f"{family}: variabili mancanti: {missing}"
        )


def expected_days(year, month):
    start = pd.Timestamp(year, month, 1)
    if month == 12:
        end = pd.Timestamp(year + 1, 1, 1)
    else:
        end = pd.Timestamp(year, month + 1, 1)
    return pd.date_range(start, end, freq="D", inclusive="left")


def check_time_axis(ds, year, month, expected_samples_per_day, family):
    if "valid_time" not in ds.coords:
        raise RuntimeError(f"{family}: valid_time assente.")

    t = pd.DatetimeIndex(pd.to_datetime(ds["valid_time"].values))
    exp_days = expected_days(year, month)
    expected_n = len(exp_days) * expected_samples_per_day

    if len(t) != expected_n:
        raise RuntimeError(
            f"{family}: timestamp={len(t)}, attesi={expected_n}"
        )

    if t.duplicated().any():
        raise RuntimeError(f"{family}: timestamp duplicati.")

    if not t.is_monotonic_increasing:
        raise RuntimeError(f"{family}: timestamp fuori ordine.")

    if not ((t.year == year) & (t.month == month)).all():
        raise RuntimeError(f"{family}: timestamp fuori mese.")

    return t


def daily_source(ds, W, receptor_ids, year, month):
    require_vars(ds, SOURCE_VARS, "source3h")
    t = check_time_axis(ds, year, month, 8, "source3h")

    viwve = spatial_weighted(ds["viwve"], W)
    viwvn = spatial_weighted(ds["viwvn"], W)
    ivtmag = spatial_weighted(
        np.sqrt(ds["viwve"] ** 2 + ds["viwvn"] ** 2),
        W,
    )
    tcwv = spatial_weighted(ds["tcwv"], W)
    cape = spatial_weighted(ds["cape"], W)
    msl = spatial_weighted(ds["msl"], W)

    rows = []
    normdates = t.normalize()

    for d in expected_days(year, month):
        mask = normdates == d
        if int(mask.sum()) != 8:
            raise RuntimeError(
                f"source3h {d.date()}: campioni={int(mask.sum())}, attesi=8"
            )

        for b, rid in enumerate(receptor_ids):
            e_mean = float(np.nanmean(viwve[mask, b]))
            n_mean = float(np.nanmean(viwvn[mask, b]))
            direction = (
                math.degrees(math.atan2(e_mean, n_mean)) + 360.0
            ) % 360.0

            rows.append({
                "date": d,
                "receptor_id": rid,
                "ivt_e_mean_kg_m1_s1": e_mean,
                "ivt_n_mean_kg_m1_s1": n_mean,
                "ivt_mag_mean_kg_m1_s1": float(np.nanmean(ivtmag[mask, b])),
                "ivt_mag_max_kg_m1_s1": float(np.nanmax(ivtmag[mask, b])),
                "ivt_vector_dir_deg_from_north": direction,
                "tcwv_mean_kg_m2": float(np.nanmean(tcwv[mask, b])),
                "tcwv_max_kg_m2": float(np.nanmax(tcwv[mask, b])),
                "cape_mean_j_kg": float(np.nanmean(cape[mask, b])),
                "cape_max_j_kg": float(np.nanmax(cape[mask, b])),
                "mslp_mean_pa": float(np.nanmean(msl[mask, b])),
                "mslp_min_pa": float(np.nanmin(msl[mask, b])),
                "source3h_samples": 8,
            })

    return pd.DataFrame(rows)


def daily_pressure(ds, W, receptor_ids, year, month):
    require_vars(ds, PRESSURE_VARS, "pressure3h")
    t = check_time_axis(ds, year, month, 8, "pressure3h")

    if "pressure_level" not in ds.coords:
        raise RuntimeError("pressure3h: pressure_level assente.")

    actual_levels = [
        int(round(float(x)))
        for x in np.asarray(ds["pressure_level"].values).ravel()
    ]
    if sorted(actual_levels) != sorted(LEVELS):
        raise RuntimeError(
            f"pressure3h: livelli {actual_levels}, attesi {LEVELS}"
        )

    by_level = {}

    for lev in LEVELS:
        sub = ds.sel(pressure_level=lev)

        u = spatial_weighted(sub["u"], W)
        v = spatial_weighted(sub["v"], W)
        q = spatial_weighted(sub["q"], W)
        temp = spatial_weighted(sub["t"], W)

        wind = spatial_weighted(
            np.sqrt(sub["u"] ** 2 + sub["v"] ** 2),
            W,
        )

        by_level[lev] = (u, v, q, temp, wind)

    rows = []
    normdates = t.normalize()

    for d in expected_days(year, month):
        mask = normdates == d
        if int(mask.sum()) != 8:
            raise RuntimeError(
                f"pressure3h {d.date()}: campioni={int(mask.sum())}, attesi=8"
            )

        for b, rid in enumerate(receptor_ids):
            rec = {
                "date": d,
                "receptor_id": rid,
                "pressure3h_samples": 8,
            }

            for lev in LEVELS:
                u, v, q, temp, wind = by_level[lev]

                rec[f"u{lev}_mean_m_s"] = float(
                    np.nanmean(u[mask, b])
                )
                rec[f"v{lev}_mean_m_s"] = float(
                    np.nanmean(v[mask, b])
                )
                rec[f"wind{lev}_mean_m_s"] = float(
                    np.nanmean(wind[mask, b])
                )
                rec[f"wind{lev}_max_m_s"] = float(
                    np.nanmax(wind[mask, b])
                )
                rec[f"q{lev}_mean_kg_kg"] = float(
                    np.nanmean(q[mask, b])
                )
                rec[f"t{lev}_mean_k"] = float(
                    np.nanmean(temp[mask, b])
                )

            rows.append(rec)

    return pd.DataFrame(rows)


def daily_precip(ds, W, receptor_ids, year, month):
    require_vars(ds, PRECIP_VARS, "precip1h")
    t = check_time_axis(ds, year, month, 24, "precip1h")

    tp_mm = spatial_weighted(ds["tp"], W) * 1000.0

    rows = []
    normdates = t.normalize()

    for d in expected_days(year, month):
        mask = normdates == d
        if int(mask.sum()) != 24:
            raise RuntimeError(
                f"precip1h {d.date()}: campioni={int(mask.sum())}, attesi=24"
            )

        for b, rid in enumerate(receptor_ids):
            vals = tp_mm[mask, b]

            rows.append({
                "date": d,
                "receptor_id": rid,
                "precip_sum_mm": float(np.nansum(vals)),
                "precip_max_1h_mm": float(np.nanmax(vals)),
                "precip_hourly_samples": 24,
            })

    return pd.DataFrame(rows)


def daily_state(ds, W, receptor_ids, year, month):
    require_vars(ds, STATE_VARS, "state1d")
    t = check_time_axis(ds, year, month, 1, "state1d")

    sw1 = spatial_weighted(ds["swvl1"], W)
    sw2 = spatial_weighted(ds["swvl2"], W)
    sw3 = spatial_weighted(ds["swvl3"], W)
    sd = spatial_weighted(ds["sd"], W)

    rows = []

    for i, dt in enumerate(t):
        d = dt.normalize()

        for b, rid in enumerate(receptor_ids):
            rows.append({
                "date": d,
                "receptor_id": rid,
                "soil_water_l1_m3_m3": float(sw1[i, b]),
                "soil_water_l2_m3_m3": float(sw2[i, b]),
                "soil_water_l3_m3_m3": float(sw3[i, b]),
                "snow_depth_mwe": float(sd[i, b]),
                "state_daily_samples": 1,
            })

    return pd.DataFrame(rows)


def audit_month(df, year, month):
    days = len(expected_days(year, month))
    expected_rows = days * RECEPTORS_EXPECTED

    reasons = []

    if len(df) != expected_rows:
        reasons.append(f"rows={len(df)} expected={expected_rows}")

    keys = ["date", "receptor_id"]

    dup = int(df.duplicated(keys).sum())
    if dup:
        reasons.append(f"duplicate_keys={dup}")

    if df["receptor_id"].nunique() != RECEPTORS_EXPECTED:
        reasons.append(
            f"receptors={df['receptor_id'].nunique()}"
        )

    sample_expect = {
        "source3h_samples": 8,
        "pressure3h_samples": 8,
        "precip_hourly_samples": 24,
        "state_daily_samples": 1,
    }

    for col, expected in sample_expect.items():
        bad = int((df[col] != expected).sum())
        if bad:
            reasons.append(f"{col}_bad={bad}")

    nonfeature = {
        "date", "receptor_id", "label", "region", "priority",
        *sample_expect.keys(),
    }

    feature_cols = [
        c for c in df.columns
        if c not in nonfeature
    ]

    all_nan = [
        c for c in feature_cols
        if df[c].isna().all()
    ]
    if all_nan:
        reasons.append(f"all_nan={all_nan}")

    return {
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "rows": len(df),
        "expected_rows": expected_rows,
        "duplicates": dup,
    }


def process_month(
    root,
    year,
    month,
    Wsource,
    Wtarget,
    receptor_ids,
    meta,
):
    ym = f"{year}{month:02d}"

    source_path = (
        root / "era5_historical_nw" / "source_single_3h"
        / str(year) / f"era5_source_single_3h_{ym}.nc"
    )
    pressure_path = (
        root / "era5_historical_nw" / "pressure_3h"
        / str(year) / f"era5_pressure_3h_{ym}.nc"
    )
    precip_path = (
        root / "era5_historical_nw" / "target_precip_hourly"
        / str(year) / f"era5_target_precip_1h_{ym}.nc"
    )
    state_path = (
        root / "era5_historical_nw" / "target_state_daily"
        / str(year) / f"era5_target_state_1d_{ym}.nc"
    )

    for p in [source_path, pressure_path, precip_path, state_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    with xr.open_dataset(source_path, decode_times=True) as ds:
        a = daily_source(
            ds, Wsource, receptor_ids, year, month
        )

    with xr.open_dataset(pressure_path, decode_times=True) as ds:
        b = daily_pressure(
            ds, Wsource, receptor_ids, year, month
        )

    with xr.open_dataset(precip_path, decode_times=True) as ds:
        c = daily_precip(
            ds, Wtarget, receptor_ids, year, month
        )

    with xr.open_dataset(state_path, decode_times=True) as ds:
        d = daily_state(
            ds, Wtarget, receptor_ids, year, month
        )

    keys = ["date", "receptor_id"]

    out = (
        a.merge(b, on=keys, how="outer", validate="one_to_one")
        .merge(c, on=keys, how="outer", validate="one_to_one")
        .merge(d, on=keys, how="outer", validate="one_to_one")
        .merge(meta, on="receptor_id", how="left", validate="many_to_one")
        .sort_values(keys)
        .reset_index(drop=True)
    )

    return out


def main():
    args = parse_args()

    if args.start_year < START_YEAR or args.end_year > END_YEAR:
        raise SystemExit(
            f"Intervallo consentito: {START_YEAR}-{END_YEAR}"
        )
    if args.start_year > args.end_year:
        raise SystemExit("start-year > end-year.")

    root = Path(__file__).resolve().parent

    audit_json = (
        root / "era5_historical_nw" / "audit" / "era5_v1_0"
        / "era5_audit_v1_0.json"
    )

    if not audit_json.exists():
        raise SystemExit(
            "Audit ERA5 v1.0 non trovato."
        )

    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    if audit.get("overall_status") != "PASS":
        raise SystemExit(
            f"Audit ERA5 non PASS: {audit.get('overall_status')}"
        )

    probe_dir = (
        root / "era5_historical_nw" / "basin_features_probe_v1_0"
    )
    probe_json = probe_dir / "probe_report_v1_0.json"

    if not probe_json.exists():
        raise SystemExit("Probe basin features v1.0 non trovato.")

    probe = json.loads(probe_json.read_text(encoding="utf-8"))
    if probe.get("status") != "PASS":
        raise SystemExit(
            f"Probe basin features non PASS: {probe.get('status')}"
        )

    weights_dir = probe_dir

    src_weights_path = weights_dir / "source_grid_weights_v1_0.csv"
    tgt_weights_path = weights_dir / "target_grid_weights_v1_0.csv"

    if not src_weights_path.exists() or not tgt_weights_path.exists():
        raise SystemExit(
            "Pesi spaziali validati non trovati nel probe."
        )

    src_weights = pd.read_csv(src_weights_path)
    tgt_weights = pd.read_csv(tgt_weights_path)

    basins_path = (
        root / "basins_final" / "nw_receptors_final.geojson"
    )

    import geopandas as gpd

    basins = gpd.read_file(basins_path)

    receptor_ids = basins["receptor_id"].astype(str).tolist()

    if len(receptor_ids) != RECEPTORS_EXPECTED:
        raise SystemExit(
            f"Attesi {RECEPTORS_EXPECTED} recettori, trovati {len(receptor_ids)}"
        )

    meta = basins[
        ["receptor_id", "label", "region", "priority"]
    ].copy()

    # Ricaviamo dimensioni griglie dai pesi validati.
    nlat_source = int(src_weights["lat_idx"].max()) + 1
    nlon_source = int(src_weights["lon_idx"].max()) + 1
    nlat_target = int(tgt_weights["lat_idx"].max()) + 1
    nlon_target = int(tgt_weights["lon_idx"].max()) + 1

    # I CSV pesi contengono solo celle intersecanti, quindi max index può
    # non coincidere con dimensione griglia completa. Leggiamo un campione.
    sample_source = (
        root / "era5_historical_nw" / "source_single_3h"
        / "2025" / "era5_source_single_3h_202512.nc"
    )
    sample_target = (
        root / "era5_historical_nw" / "target_precip_hourly"
        / "2025" / "era5_target_precip_1h_202512.nc"
    )

    with xr.open_dataset(sample_source) as ds:
        nlat_source = int(ds.sizes["latitude"])
        nlon_source = int(ds.sizes["longitude"])

    with xr.open_dataset(sample_target) as ds:
        nlat_target = int(ds.sizes["latitude"])
        nlon_target = int(ds.sizes["longitude"])

    Wsource = weight_matrix(
        src_weights, receptor_ids, nlat_source, nlon_source
    )
    Wtarget = weight_matrix(
        tgt_weights, receptor_ids, nlat_target, nlon_target
    )

    out_dir = (
        root / "era5_historical_nw"
        / "basin_features_historical_v1_0"
    )
    monthly_dir = out_dir / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    total_months = (
        (args.end_year - args.start_year + 1) * len(MONTHS)
    )
    done = 0

    print("=" * 126)
    print("ERA5 -> 21 BACINI — PRODUZIONE STORICA v1.0")
    print("=" * 126)
    print(
        f"Periodo: Sep-Dic {args.start_year}-{args.end_year}"
    )
    print(f"Mesi: {total_months}")
    print("Metodo: fractional overlap, no nearest fallback")
    print("=" * 126)

    for year in range(args.start_year, args.end_year + 1):
        ydir = monthly_dir / str(year)
        ydir.mkdir(parents=True, exist_ok=True)

        for month in MONTHS:
            ym = f"{year}{month:02d}"
            out_month = (
                ydir / f"era5_basin_features_{ym}_v1_0.csv"
            )

            done += 1

            if out_month.exists() and not args.force:
                try:
                    old = pd.read_csv(
                        out_month,
                        parse_dates=["date"],
                    )
                    audit_m = audit_month(
                        old, year, month
                    )

                    if audit_m["status"] == "PASS":
                        print(
                            f"{ym} | SKIP PASS | "
                            f"{audit_m['rows']} righe "
                            f"| progresso {done}/{total_months}"
                        )
                        manifest_rows.append({
                            "year": year,
                            "month": month,
                            "ym": ym,
                            "status": "PASS_EXISTING",
                            "rows": audit_m["rows"],
                            "path": str(out_month),
                            "reasons": "",
                        })
                        continue

                    print(
                        f"{ym} | file esistente non valido, ricalcolo: "
                        f"{audit_m['reasons']}"
                    )

                except Exception as exc:
                    print(
                        f"{ym} | file esistente illeggibile, ricalcolo: "
                        f"{exc!r}"
                    )

            print(
                f"{ym} | elaborazione | "
                f"progresso {done}/{total_months}"
            )

            try:
                month_df = process_month(
                    root,
                    year,
                    month,
                    Wsource,
                    Wtarget,
                    receptor_ids,
                    meta,
                )

                audit_m = audit_month(
                    month_df, year, month
                )

                if audit_m["status"] != "PASS":
                    raise RuntimeError(
                        f"Audit mese FAIL: {audit_m['reasons']}"
                    )

                tmp = out_month.with_suffix(".csv.tmp")
                month_df.to_csv(tmp, index=False)
                tmp.replace(out_month)

                manifest_rows.append({
                    "year": year,
                    "month": month,
                    "ym": ym,
                    "status": "PASS",
                    "rows": audit_m["rows"],
                    "path": str(out_month),
                    "reasons": "",
                })

                print(
                    f"{ym} | PASS | {audit_m['rows']} righe"
                )

            except Exception as exc:
                manifest_rows.append({
                    "year": year,
                    "month": month,
                    "ym": ym,
                    "status": "FAIL",
                    "rows": None,
                    "path": str(out_month),
                    "reasons": repr(exc),
                })

                manifest_path = (
                    out_dir / "era5_basin_features_manifest_v1_0.csv"
                )
                pd.DataFrame(manifest_rows).to_csv(
                    manifest_path, index=False
                )

                print(f"{ym} | FAIL | {exc!r}")
                raise

    # ------------------------------------------------------------------
    # CONSOLIDAMENTO
    # ------------------------------------------------------------------
    print("\nCONSOLIDAMENTO STORICO...")

    all_parts = []

    for year in range(args.start_year, args.end_year + 1):
        for month in MONTHS:
            ym = f"{year}{month:02d}"
            p = (
                monthly_dir / str(year)
                / f"era5_basin_features_{ym}_v1_0.csv"
            )
            if not p.exists():
                raise RuntimeError(
                    f"Manca prodotto mensile: {p}"
                )

            part = pd.read_csv(
                p,
                parse_dates=["date"],
            )

            audit_m = audit_month(
                part, year, month
            )
            if audit_m["status"] != "PASS":
                raise RuntimeError(
                    f"Prodotto mensile non PASS {ym}: "
                    f"{audit_m['reasons']}"
                )

            all_parts.append(part)

    full = pd.concat(
        all_parts,
        ignore_index=True,
    ).sort_values(
        ["date", "receptor_id"]
    ).reset_index(drop=True)

    expected_rows_run = (
        (args.end_year - args.start_year + 1)
        * DAYS_PER_SEASON
        * RECEPTORS_EXPECTED
    )

    final_reasons = []

    if len(full) != expected_rows_run:
        final_reasons.append(
            f"rows={len(full)} expected={expected_rows_run}"
        )

    dup = int(
        full.duplicated(
            ["date", "receptor_id"]
        ).sum()
    )
    if dup:
        final_reasons.append(
            f"duplicate_keys={dup}"
        )

    if full["receptor_id"].nunique() != RECEPTORS_EXPECTED:
        final_reasons.append(
            f"receptors={full['receptor_id'].nunique()}"
        )

    # Controllo calendario: ogni stagione deve avere 122 giorni per bacino.
    counts = (
        full.assign(year=full["date"].dt.year)
        .groupby(["year", "receptor_id"])
        .size()
    )
    bad_counts = counts[counts != DAYS_PER_SEASON]
    if len(bad_counts):
        final_reasons.append(
            f"bad_year_receptor_counts={len(bad_counts)}"
        )

    # Nessuna feature può essere interamente NaN.
    nonfeature = {
        "date", "receptor_id", "label", "region", "priority",
        "source3h_samples", "pressure3h_samples",
        "precip_hourly_samples", "state_daily_samples",
    }
    feature_cols = [
        c for c in full.columns
        if c not in nonfeature
    ]
    all_nan_features = [
        c for c in feature_cols
        if full[c].isna().all()
    ]
    if all_nan_features:
        final_reasons.append(
            f"all_nan_features={all_nan_features}"
        )

    overall = (
        "PASS"
        if not final_reasons
        else "REVIEW"
    )

    final_csv = (
        out_dir
        / f"era5_basin_features_daily_{args.start_year}_{args.end_year}_v1_0.csv"
    )

    tmp_final = final_csv.with_suffix(".csv.tmp")
    full.to_csv(tmp_final, index=False)
    tmp_final.replace(final_csv)

    manifest_path = (
        out_dir / "era5_basin_features_manifest_v1_0.csv"
    )
    pd.DataFrame(manifest_rows).to_csv(
        manifest_path, index=False
    )

    report = {
        "version": "1.0",
        "overall_status": overall,
        "period": f"Sep-Dec {args.start_year}-{args.end_year}",
        "months": total_months,
        "receptors": RECEPTORS_EXPECTED,
        "expected_rows": expected_rows_run,
        "actual_rows": int(len(full)),
        "duplicate_keys": dup,
        "bad_year_receptor_counts": int(len(bad_counts)),
        "all_nan_features": all_nan_features,
        "feature_columns": feature_cols,
        "reasons": final_reasons,
        "method": (
            "validated fractional cell-basin overlap weights; "
            "no nearest-cell fallback"
        ),
        "raw_modified": False,
    }

    report_json = (
        out_dir / "era5_basin_features_audit_v1_0.json"
    )
    report_json.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_txt = (
        out_dir / "era5_basin_features_audit_v1_0.txt"
    )

    lines = [
        "=" * 126,
        "ERA5 -> 21 BACINI — AUDIT PRODUZIONE STORICA v1.0",
        "=" * 126,
        f"OVERALL STATUS       : {overall}",
        f"Periodo              : Sep-Dic {args.start_year}-{args.end_year}",
        f"Mesi                 : {total_months}",
        f"Recettori            : {RECEPTORS_EXPECTED}",
        f"Righe attese         : {expected_rows_run}",
        f"Righe prodotte       : {len(full)}",
        f"Chiavi duplicate     : {dup}",
        f"Year-basin bad count : {len(bad_counts)}",
        f"Feature tutte NaN    : {all_nan_features}",
        f"Metodo               : fractional overlap, no nearest fallback",
        f"Output               : {final_csv}",
    ]

    report_txt.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 126)
    for line in lines[3:]:
        print(line)
    print("=" * 126)

    if overall != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
