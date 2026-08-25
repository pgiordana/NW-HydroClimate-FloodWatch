#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
freeze_medsea_ivt_basin_coupling_v1_2.py

Congela il coupling Mediterraneo × IVT -> 21 bacini dopo sensitivity test.

Decisione metodologica:
- baseline mantenuto: sigma=22.5°, cutoff=±45°, L=700 km, Dmax=1600 km;
- il sensitivity test mostra che distanza/scala/sigma sono molto stabili;
- il supporto è invece sensibile all'apertura angolare;
- quindi NON trattiamo il supporto baseline come verità binaria assoluta:
  aggiungiamo una classificazione di robustezza rispetto a ±30°, ±45°, ±60°.

Classi:
ROBUST_MEDSEA_SUPPORT
    supporto presente anche con cutoff stretto ±30°.
BASELINE_ANGLE_SENSITIVE_SUPPORT
    supporto presente a ±45° e ±60°, assente a ±30°.
WIDE_ONLY_MEDSEA_SUPPORT
    supporto solo con apertura ampia ±60°.
NO_MEDSEA_SUPPORT_ALL
    nessun supporto neppure con ±60°.
INCONSISTENT
    configurazione non monotona / incoerente (deve essere 0 righe).

Il file canonico mantiene i valori numerici del baseline v1.0 e aggiunge:
- medsea_support_baseline
- medsea_support_angle_narrow
- medsea_support_angle_wide
- medsea_support_robustness_class
- medsea_support_robust_core

Non modifica alcun prodotto precedente.

Output:
medsea_historical_analysis/basin_coupling_canonical_v1_2/
  medsea_ivt_basin_coupling_daily_1987_2025_canonical_v1_2.csv
  medsea_support_robustness_by_receptor_v1_2.csv
  medsea_support_robustness_audit_v1_2.json
  medsea_support_robustness_audit_v1_2.txt
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse


EXPECTED_ROWS = 99918
EXPECTED_RECEPTORS = 21

N_SECTORS = 16
STEP = 22.5
SECTOR_CENTERS = np.arange(0.0, 360.0, STEP)

EARTH_RADIUS_KM = 6371.0088
EARTH_RADIUS_M = EARTH_RADIUS_KM * 1000.0

SIGMA = 22.5
SCALE_KM = 700.0
DMAX_KM = 1600.0

CUTOFFS = {
    "narrow": 30.0,
    "baseline": 45.0,
    "wide": 60.0,
}


def angular_diff_deg(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def cell_edges(c):
    c = np.asarray(c, dtype=float)
    mids = (c[:-1] + c[1:]) / 2.0
    first = c[0] + (c[0] - mids[0])
    last = c[-1] + (c[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def spherical_cell_areas(lat, lon):
    le = np.deg2rad(cell_edges(lat))
    oe = np.deg2rad(cell_edges(lon))
    lf = np.abs(np.sin(le[1:]) - np.sin(le[:-1]))
    ow = np.abs(oe[1:] - oe[:-1])
    return EARTH_RADIUS_M**2 * lf[:, None] * ow[None, :]


def distance_bearing(basin_lat, basin_lon, grid_lat, grid_lon):
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
        np.sin(dlat/2)**2
        + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    )
    a = np.clip(a, 0, 1)
    dist = 2*EARTH_RADIUS_KM*np.arcsin(np.sqrt(a))

    y = np.sin(dlon)*np.cos(lat2)
    x = (
        np.cos(lat1)*np.sin(lat2)
        - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    )
    bearing = (np.rad2deg(np.arctan2(y, x)) + 360) % 360

    return dist.ravel(), bearing.ravel()


def build_weight_matrix(cent, lat, lon, cutoff):
    areas = spherical_cell_areas(lat, lon).ravel()
    ncell = len(lat) * len(lon)

    rows, cols, data = [], [], []
    col = 0

    for rec in cent.itertuples(index=False):
        dist, bear = distance_bearing(
            rec.centroid_lat,
            rec.centroid_lon,
            lat,
            lon,
        )

        for center in SECTOR_CENTERS:
            diff = angular_diff_deg(bear, center)

            mask = (
                np.isfinite(dist)
                & np.isfinite(bear)
                & (dist <= DMAX_KM)
                & (diff <= cutoff)
            )

            idx = np.flatnonzero(mask)

            if len(idx):
                w = (
                    areas[idx]
                    * np.exp(-dist[idx] / SCALE_KM)
                    * np.exp(-0.5*(diff[idx]/SIGMA)**2)
                )

                good = np.isfinite(w) & (w > 0)
                idx = idx[good]
                w = w[good]

                rows.extend(idx.tolist())
                cols.extend([col]*len(idx))
                data.extend(w.astype(float).tolist())

            col += 1

    return sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(ncell, len(cent)*N_SECTORS),
        dtype=np.float64,
    )


def sector_pair(bearing):
    b = np.asarray(bearing, dtype=float) % 360.0
    pos = b / STEP
    low = np.floor(pos).astype(int) % N_SECTORS
    frac = pos - np.floor(pos)
    high = (low + 1) % N_SECTORS
    return low, high, frac


def support_vector(W, marine_mask, receptor_index, low, high, frac):
    den = np.asarray(
        W.T.dot(marine_mask.astype(np.float64))
    ).ravel()

    out = np.zeros(len(receptor_index), dtype=bool)

    for i in range(len(out)):
        c0 = int(receptor_index[i])*N_SECTORS + int(low[i])
        c1 = int(receptor_index[i])*N_SECTORS + int(high[i])

        d = (
            (1.0-float(frac[i]))*den[c0]
            + float(frac[i])*den[c1]
        )
        out[i] = bool(np.isfinite(d) and d > 0)

    return out


def main():
    root = Path(__file__).resolve().parent

    base_dir = (
        root / "medsea_historical_analysis"
        / "basin_coupling_historical_v1_0"
    )

    coupling_csv = (
        base_dir
        / "medsea_ivt_basin_coupling_daily_1987_2025_v1_0.csv"
    )
    sensitivity_json = (
        base_dir
        / "sensitivity_v1_0"
        / "sensitivity_audit_v1_0.json"
    )
    support_json = (
        base_dir
        / "support_audit_v1_1"
        / "support_audit_v1_1.json"
    )

    cent_csv = (
        root / "medsea_historical_analysis"
        / "basin_coupling_preflight_v1_0"
        / "receptor_centroids_v1_0.csv"
    )

    sst_ref = (
        root / "medsea_historical_analysis"
        / "daily_sst_anomaly"
        / "medsea_sst_anomaly_2025_SepDec.nc"
    )
    ohc_ref = (
        root / "medsea_historical_analysis"
        / "monthly_ohc_anomaly"
        / "medsea_ohc_anomaly_2025_SepDec.nc"
    )

    out_dir = (
        root / "medsea_historical_analysis"
        / "basin_coupling_canonical_v1_2"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("="*126)
    print("MEDSEA × IVT -> 21 BACINI — FREEZE CANONICO v1.2")
    print("="*126)

    for p in [
        coupling_csv,
        sensitivity_json,
        support_json,
        cent_csv,
        sst_ref,
        ohc_ref,
    ]:
        if not p.exists():
            raise SystemExit(f"Manca: {p}")

    sens = json.loads(
        sensitivity_json.read_text(encoding="utf-8")
    )
    supp = json.loads(
        support_json.read_text(encoding="utf-8")
    )

    if sens.get("overall_status") != "PASS":
        raise SystemExit("Sensitivity non PASS.")
    if supp.get("overall_status") != "PASS":
        raise SystemExit("Support audit non PASS.")

    df = pd.read_csv(coupling_csv)
    df["date"] = pd.to_datetime(
        df["date"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    if len(df) != EXPECTED_ROWS:
        raise SystemExit(
            f"Righe={len(df)} attese={EXPECTED_ROWS}"
        )

    if df["date"].isna().any():
        raise SystemExit("Date invalide.")

    if df["receptor_id"].nunique() != EXPECTED_RECEPTORS:
        raise SystemExit("Numero recettori errato.")

    cent = pd.read_csv(cent_csv)
    receptor_ids = cent["receptor_id"].astype(str).tolist()
    rmap = {rid: i for i, rid in enumerate(receptor_ids)}

    receptor_index = df["receptor_id"].map(rmap)
    if receptor_index.isna().any():
        raise SystemExit("Recettore non mappato.")
    receptor_index = receptor_index.astype(int).to_numpy()

    source_bearing = pd.to_numeric(
        df["marine_source_bearing_deg"],
        errors="coerce",
    ).to_numpy()

    if not np.isfinite(source_bearing).all():
        raise SystemExit("Source bearing non finiti.")

    low, high, frac = sector_pair(source_bearing)

    # Reference grid + marine mask SST/OHC intersection.
    with xr.open_dataset(sst_ref, decode_times=True) as ds:
        latn = "latitude" if "latitude" in ds.coords else "lat"
        lonn = "longitude" if "longitude" in ds.coords else "lon"
        lat = np.asarray(ds[latn].values, dtype=float)
        lon = np.asarray(ds[lonn].values, dtype=float)
        sst0 = np.asarray(
            ds["sst_anomaly"].isel(
                {list(ds["sst_anomaly"].dims)[0]: 0}
            ).squeeze().values,
            dtype=float,
        ).reshape(-1)

    with xr.open_dataset(ohc_ref, decode_times=True) as ds:
        latn2 = "latitude" if "latitude" in ds.coords else "lat"
        lonn2 = "longitude" if "longitude" in ds.coords else "lon"
        lat2 = np.asarray(ds[latn2].values, dtype=float)
        lon2 = np.asarray(ds[lonn2].values, dtype=float)
        ohc0 = np.asarray(
            ds["ohc_anomaly_0_100"].isel(
                {list(ds["ohc_anomaly_0_100"].dims)[0]: 0}
            ).squeeze().values,
            dtype=float,
        ).reshape(-1)

    if (
        lat.shape != lat2.shape
        or lon.shape != lon2.shape
        or not np.allclose(lat, lat2, atol=1e-10, rtol=0)
        or not np.allclose(lon, lon2, atol=1e-10, rtol=0)
    ):
        raise SystemExit("SST/OHC griglie diverse.")

    marine_mask = np.isfinite(sst0) & np.isfinite(ohc0)

    supports = {}

    for name, cutoff in CUTOFFS.items():
        print(f"Costruzione supporto {name}: cutoff=±{cutoff:g}°...")
        W = build_weight_matrix(
            cent,
            lat,
            lon,
            cutoff,
        )
        supports[name] = support_vector(
            W,
            marine_mask,
            receptor_index,
            low,
            high,
            frac,
        )

    narrow = supports["narrow"]
    baseline = supports["baseline"]
    wide = supports["wide"]

    # Baseline deve coincidere esattamente col prodotto v1.0.
    baseline_actual = (
        (pd.to_numeric(
            df["sst_corridor_support_weight"],
            errors="coerce",
        ) > 0)
        & (pd.to_numeric(
            df["ohc_corridor_support_weight"],
            errors="coerce",
        ) > 0)
        & df["medsea_sst_anom_corridor_c"].notna()
        & df["medsea_ohc_anom_corridor_j_m2"].notna()
    ).to_numpy()

    baseline_mismatch = int(
        (baseline != baseline_actual).sum()
    )

    # Monotonicità attesa:
    # narrow ⊆ baseline ⊆ wide.
    monotonic_bad = int(
        ((narrow & ~baseline) | (baseline & ~wide)).sum()
    )

    conditions = [
        narrow & baseline & wide,
        (~narrow) & baseline & wide,
        (~narrow) & (~baseline) & wide,
        (~narrow) & (~baseline) & (~wide),
    ]
    choices = [
        "ROBUST_MEDSEA_SUPPORT",
        "BASELINE_ANGLE_SENSITIVE_SUPPORT",
        "WIDE_ONLY_MEDSEA_SUPPORT",
        "NO_MEDSEA_SUPPORT_ALL",
    ]

    classes = np.select(
        conditions,
        choices,
        default="INCONSISTENT",
    )

    inconsistent = int((classes == "INCONSISTENT").sum())

    df["medsea_support_angle_narrow"] = narrow
    df["medsea_support_baseline"] = baseline
    df["medsea_support_angle_wide"] = wide
    df["medsea_support_robust_core"] = narrow
    df["medsea_support_robustness_class"] = classes

    # Riordino temporale.
    df = df.sort_values(
        ["date", "receptor_id"]
    ).reset_index(drop=True)

    # Summary by receptor.
    tmp = df[
        [
            "receptor_id",
            "medsea_support_robustness_class",
        ]
    ].copy()

    summary = (
        tmp.groupby(
            ["receptor_id", "medsea_support_robustness_class"]
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for c in choices + ["INCONSISTENT"]:
        if c not in summary.columns:
            summary[c] = 0

    summary["rows"] = summary[
        choices + ["INCONSISTENT"]
    ].sum(axis=1)

    summary["robust_support_pct"] = (
        100.0
        * summary["ROBUST_MEDSEA_SUPPORT"]
        / summary["rows"]
    )
    summary["baseline_supported_pct"] = (
        100.0
        * (
            summary["ROBUST_MEDSEA_SUPPORT"]
            + summary["BASELINE_ANGLE_SENSITIVE_SUPPORT"]
        )
        / summary["rows"]
    )
    summary["wide_supported_pct"] = (
        100.0
        * (
            summary["ROBUST_MEDSEA_SUPPORT"]
            + summary["BASELINE_ANGLE_SENSITIVE_SUPPORT"]
            + summary["WIDE_ONLY_MEDSEA_SUPPORT"]
        )
        / summary["rows"]
    )

    counts = pd.Series(classes).value_counts().to_dict()

    reasons = []
    if baseline_mismatch:
        reasons.append(
            f"baseline_mismatch={baseline_mismatch}"
        )
    if monotonic_bad:
        reasons.append(
            f"monotonic_support_violations={monotonic_bad}"
        )
    if inconsistent:
        reasons.append(
            f"inconsistent_classes={inconsistent}"
        )
    if len(df) != EXPECTED_ROWS:
        reasons.append("row_count_changed")

    overall = "PASS" if not reasons else "REVIEW"

    out_csv = (
        out_dir
        / "medsea_ivt_basin_coupling_daily_1987_2025_canonical_v1_2.csv"
    )
    tmp_out = out_csv.with_suffix(".csv.tmp")
    df.to_csv(tmp_out, index=False)
    tmp_out.replace(out_csv)

    summary_csv = (
        out_dir
        / "medsea_support_robustness_by_receptor_v1_2.csv"
    )
    summary.to_csv(summary_csv, index=False)

    report = {
        "version": "1.2",
        "overall_status": overall,
        "rows": int(len(df)),
        "receptors": int(df["receptor_id"].nunique()),
        "baseline_parameters": {
            "sigma_deg": SIGMA,
            "cutoff_deg": CUTOFFS["baseline"],
            "distance_scale_km": SCALE_KM,
            "max_distance_km": DMAX_KM,
        },
        "angular_sensitivity_cutoffs_deg": {
            "narrow": CUTOFFS["narrow"],
            "baseline": CUTOFFS["baseline"],
            "wide": CUTOFFS["wide"],
        },
        "class_counts": {
            str(k): int(v)
            for k, v in counts.items()
        },
        "baseline_mismatch_rows": baseline_mismatch,
        "monotonic_support_violations": monotonic_bad,
        "inconsistent_rows": inconsistent,
        "decision": (
            "Retain ±45° as canonical central parameterization, "
            "but preserve support robustness across ±30°/±45°/±60° "
            "as an explicit model feature and interpretation flag."
        ),
        "scientific_note": (
            "The marine coupling remains a transport-conditioned Eulerian/geometric proxy, "
            "not a Lagrangian source-attribution product."
        ),
        "reasons": reasons,
        "raw_modified": False,
    }

    json_p = (
        out_dir
        / "medsea_support_robustness_audit_v1_2.json"
    )
    json_p.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    txt_p = (
        out_dir
        / "medsea_support_robustness_audit_v1_2.txt"
    )

    lines = [
        "="*126,
        "MEDSEA × IVT -> 21 BACINI — FREEZE CANONICO v1.2",
        "="*126,
        f"OVERALL STATUS                         : {overall}",
        f"Righe                                  : {len(df)}",
        f"Recettori                              : {df['receptor_id'].nunique()}",
        f"Baseline mismatch                      : {baseline_mismatch}",
        f"Violazioni monotonicità narrow⊆base⊆wide: {monotonic_bad}",
        f"INCONSISTENT                           : {inconsistent}",
        "",
        "CLASS COUNTS",
    ]

    for c in choices + ["INCONSISTENT"]:
        n = int(counts.get(c, 0))
        lines.append(
            f"{c:<40}: {n:6d} ({100.0*n/len(df):6.3f}%)"
        )

    lines += [
        "",
        "DECISIONE",
        "Baseline canonico: sigma 22.5°, cutoff ±45°, L=700 km, Dmax=1600 km.",
        "Il supporto angolare viene conservato come robustezza ±30°/±45°/±60°.",
        "I valori numerici SST/OHC restano quelli del baseline v1.0.",
        "",
        "NOTA",
        "Proxy Euleriano/geometrico condizionato da IVT; non retrotraiettoria lagrangiana.",
        "",
        f"Output: {out_csv}",
    ]

    txt_p.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"Righe                                  : {len(df)}")
    print(f"Recettori                              : {df['receptor_id'].nunique()}")
    print(f"Baseline mismatch                      : {baseline_mismatch}")
    print(
        f"Violazioni monotonicità narrow⊆base⊆wide: {monotonic_bad}"
    )
    print(f"INCONSISTENT                           : {inconsistent}")

    print("\nCLASS COUNTS")
    for c in choices + ["INCONSISTENT"]:
        n = int(counts.get(c, 0))
        print(
            f"{c:<40}: {n:6d} ({100.0*n/len(df):6.3f}%)"
        )

    print("\n" + "="*126)
    print(f"OVERALL STATUS : {overall}")
    print(f"Output         : {out_dir}")
    if reasons:
        print("REASONS:")
        for r in reasons:
            print(f"  - {r}")
    print("="*126)

    if overall != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
