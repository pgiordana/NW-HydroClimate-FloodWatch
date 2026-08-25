#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_era5_basin_features_derived_v1_0.py

Costruisce feature DERIVATE causali a partire dalla base ERA5 giornaliera
x 21 bacini già validata.

Input:
era5_historical_nw/basin_features_historical_v1_0/
  era5_basin_features_daily_1987_2025_v1_0.csv
  era5_basin_features_audit_v1_0.json

Output:
era5_historical_nw/basin_features_derived_v1_0/
  era5_basin_features_daily_derived_1987_2025_v1_0.csv
  era5_basin_features_derived_audit_v1_0.json
  era5_basin_features_derived_audit_v1_0.txt

Principi:
- NON collega dicembre con settembre dell'anno successivo;
- rolling/lag calcolati entro (receptor_id, season_year);
- antecedenti "prev" usano SOLO giorni precedenti;
- nessuna climatologia/anomalia calcolata qui, per evitare leakage:
  percentili/anomalie saranno stimati nei soli fold di training;
- i primi giorni di settembre hanno NaN fisiologici per finestre antecedenti
  che richiedono dati di agosto, non presenti nel dataset storico corrente;
- nessuna modifica del CSV base.

Feature principali:
- indici calendario: season_year, season_day, month, day_of_year;
- direzione IVT in sin/cos;
- variazioni 24h di IVT, TCWV, CAPE, MSLP;
- IVT antecedente 1d e media prev3d/prev7d;
- precipitazione antecedente prev1d, prev3d, prev7d, prev14d;
- precipitazione rolling inclusa oggi 3d/7d (descrittiva, distinta dagli antecedenti);
- stato antecedente prev1d di soil water layers e snow;
- media profilo suolo prev1d;
- proxy moisture-wind q*wind a 925/850/700 hPa;
- controllo completezza delle finestre antecedenti.

Righe attese: 99,918.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_ROWS = 99918
EXPECTED_RECEPTORS = 21
EXPECTED_DAYS_PER_SEASON = 122
START_YEAR = 1987
END_YEAR = 2025


def prev_rolling(grouped, col, window, how="sum"):
    if how == "sum":
        return grouped[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=window).sum()
        )
    if how == "mean":
        return grouped[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=window).mean()
        )
    if how == "max":
        return grouped[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=window).max()
        )
    raise ValueError(how)


def incl_rolling(grouped, col, window, how="sum"):
    if how == "sum":
        return grouped[col].transform(
            lambda s: s.rolling(window, min_periods=window).sum()
        )
    if how == "mean":
        return grouped[col].transform(
            lambda s: s.rolling(window, min_periods=window).mean()
        )
    raise ValueError(how)


def lag(grouped, col, n=1):
    return grouped[col].shift(n)


def main():
    root = Path(__file__).resolve().parent

    base_dir = (
        root / "era5_historical_nw"
        / "basin_features_historical_v1_0"
    )
    base_csv = (
        base_dir
        / "era5_basin_features_daily_1987_2025_v1_0.csv"
    )
    base_audit = (
        base_dir
        / "era5_basin_features_audit_v1_0.json"
    )

    out_dir = (
        root / "era5_historical_nw"
        / "basin_features_derived_v1_0"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 126)
    print("ERA5 -> 21 BACINI — FEATURE DERIVATE CAUSALI v1.0")
    print("=" * 126)

    if not base_csv.exists():
        raise SystemExit(f"CSV base non trovato: {base_csv}")

    if not base_audit.exists():
        raise SystemExit(f"Audit base non trovato: {base_audit}")

    audit = json.loads(base_audit.read_text(encoding="utf-8"))
    if audit.get("overall_status") != "PASS":
        raise SystemExit(
            f"Base ERA5 non PASS: {audit.get('overall_status')}"
        )

    df = pd.read_csv(base_csv)

    if "date" not in df.columns:
        raise SystemExit("Colonna 'date' assente.")

    df["date"] = pd.to_datetime(
        df["date"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    invalid_dates = int(df["date"].isna().sum())
    if invalid_dates:
        raise SystemExit(f"Date non parsabili: {invalid_dates}")

    if len(df) != EXPECTED_ROWS:
        raise SystemExit(
            f"Righe base={len(df)}, attese={EXPECTED_ROWS}"
        )

    if df["receptor_id"].nunique() != EXPECTED_RECEPTORS:
        raise SystemExit(
            f"Recettori={df['receptor_id'].nunique()}, "
            f"attesi={EXPECTED_RECEPTORS}"
        )

    dup = int(df.duplicated(["date", "receptor_id"]).sum())
    if dup:
        raise SystemExit(f"Chiavi duplicate nella base: {dup}")

    # Solo Sep-Dic.
    if not df["date"].dt.month.isin([9, 10, 11, 12]).all():
        raise SystemExit("Trovati mesi fuori Sep-Dic.")

    # ------------------------------------------------------------------
    # Calendario
    # ------------------------------------------------------------------
    df["season_year"] = df["date"].dt.year.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear.astype(int)

    sep1 = pd.to_datetime(
        df["season_year"].astype(str) + "-09-01",
        format="%Y-%m-%d",
    )
    df["season_day"] = (
        (df["date"] - sep1).dt.days + 1
    ).astype(int)

    df = df.sort_values(
        ["receptor_id", "season_year", "date"]
    ).reset_index(drop=True)

    grouped = df.groupby(
        ["receptor_id", "season_year"],
        sort=False,
        group_keys=False,
    )

    # Controllo 122 giorni per anno-bacino.
    counts = grouped.size()
    bad_counts = counts[counts != EXPECTED_DAYS_PER_SEASON]
    if len(bad_counts):
        raise SystemExit(
            f"Gruppi anno-bacino con !=122 giorni: {len(bad_counts)}"
        )

    # ------------------------------------------------------------------
    # IVT direzione: trasformazione circolare
    # ------------------------------------------------------------------
    dir_rad = np.deg2rad(
        pd.to_numeric(
            df["ivt_vector_dir_deg_from_north"],
            errors="coerce",
        )
    )
    df["ivt_dir_sin"] = np.sin(dir_rad)
    df["ivt_dir_cos"] = np.cos(dir_rad)

    # ------------------------------------------------------------------
    # Cambiamenti 24h: current - previous day
    # ------------------------------------------------------------------
    for col, outname in [
        ("ivt_mag_mean_kg_m1_s1", "ivt_mag_change_24h"),
        ("tcwv_mean_kg_m2", "tcwv_change_24h"),
        ("cape_mean_j_kg", "cape_change_24h"),
        ("mslp_mean_pa", "mslp_change_24h_pa"),
    ]:
        current = pd.to_numeric(df[col], errors="coerce")
        previous = lag(grouped, col, 1)
        df[outname] = current - previous

    # ------------------------------------------------------------------
    # IVT antecedente
    # ------------------------------------------------------------------
    df["ivt_mag_prev1d"] = lag(
        grouped, "ivt_mag_mean_kg_m1_s1", 1
    )
    df["ivt_mag_prev3d_mean"] = prev_rolling(
        grouped, "ivt_mag_mean_kg_m1_s1", 3, "mean"
    )
    df["ivt_mag_prev7d_mean"] = prev_rolling(
        grouped, "ivt_mag_mean_kg_m1_s1", 7, "mean"
    )
    df["ivt_mag_prev3d_max"] = prev_rolling(
        grouped, "ivt_mag_max_kg_m1_s1", 3, "max"
    )

    # ------------------------------------------------------------------
    # Precipitazione antecedente: SOLO giorni precedenti
    # ------------------------------------------------------------------
    df["precip_prev1d_mm"] = lag(
        grouped, "precip_sum_mm", 1
    )
    for w in [3, 7, 14]:
        df[f"precip_prev{w}d_mm"] = prev_rolling(
            grouped, "precip_sum_mm", w, "sum"
        )

    # Descrittive incluse oggi, mantenute separate e nominate esplicitamente.
    df["precip_3d_incl_today_mm"] = incl_rolling(
        grouped, "precip_sum_mm", 3, "sum"
    )
    df["precip_7d_incl_today_mm"] = incl_rolling(
        grouped, "precip_sum_mm", 7, "sum"
    )
    df["precip_max1h_prev1d_mm"] = lag(
        grouped, "precip_max_1h_mm", 1
    )
    df["precip_max1h_prev3d_max_mm"] = prev_rolling(
        grouped, "precip_max_1h_mm", 3, "max"
    )

    # ------------------------------------------------------------------
    # Stato antecedente del bacino
    # ------------------------------------------------------------------
    state_cols = [
        "soil_water_l1_m3_m3",
        "soil_water_l2_m3_m3",
        "soil_water_l3_m3_m3",
        "snow_depth_mwe",
    ]

    for col in state_cols:
        df[f"{col}_prev1d"] = lag(
            grouped, col, 1
        )

    df["soil_profile_mean_m3_m3"] = df[
        [
            "soil_water_l1_m3_m3",
            "soil_water_l2_m3_m3",
            "soil_water_l3_m3_m3",
        ]
    ].mean(axis=1)

    # Serve rifare grouped perché abbiamo aggiunto la colonna.
    grouped = df.groupby(
        ["receptor_id", "season_year"],
        sort=False,
        group_keys=False,
    )

    df["soil_profile_mean_prev1d_m3_m3"] = lag(
        grouped, "soil_profile_mean_m3_m3", 1
    )

    # ------------------------------------------------------------------
    # Proxy dinamici umidità x vento
    # ------------------------------------------------------------------
    for lev in [925, 850, 700]:
        qcol = f"q{lev}_mean_kg_kg"
        wcol = f"wind{lev}_mean_m_s"

        df[f"qwind{lev}_proxy"] = (
            pd.to_numeric(df[qcol], errors="coerce")
            * pd.to_numeric(df[wcol], errors="coerce")
        )

        df[f"q{lev}_prev1d_kg_kg"] = lag(
            grouped, qcol, 1
        )
        df[f"wind{lev}_prev1d_m_s"] = lag(
            grouped, wcol, 1
        )

    # ------------------------------------------------------------------
    # Flag completezza antecedenti
    # ------------------------------------------------------------------
    df["antecedent_prev1d_complete"] = (
        df["precip_prev1d_mm"].notna()
        & df["soil_profile_mean_prev1d_m3_m3"].notna()
    )

    df["antecedent_prev3d_complete"] = (
        df["precip_prev3d_mm"].notna()
        & df["ivt_mag_prev3d_mean"].notna()
    )

    df["antecedent_prev7d_complete"] = (
        df["precip_prev7d_mm"].notna()
        & df["ivt_mag_prev7d_mean"].notna()
    )

    df["antecedent_prev14d_complete"] = (
        df["precip_prev14d_mm"].notna()
    )

    # ------------------------------------------------------------------
    # AUDIT
    # ------------------------------------------------------------------
    reasons = []

    if len(df) != EXPECTED_ROWS:
        reasons.append(
            f"rows={len(df)} expected={EXPECTED_ROWS}"
        )

    dup2 = int(
        df.duplicated(["date", "receptor_id"]).sum()
    )
    if dup2:
        reasons.append(f"duplicate_keys={dup2}")

    if df["season_day"].min() != 1 or df["season_day"].max() != 122:
        reasons.append(
            f"season_day_range={df['season_day'].min()}..{df['season_day'].max()}"
        )

    # I NaN attesi ai bordi devono avere conteggi esatti:
    # per ogni 39*21 gruppi: first 1/3/7/14 giorni.
    groups = (END_YEAR - START_YEAR + 1) * EXPECTED_RECEPTORS

    expected_nan_counts = {
        "precip_prev1d_mm": groups * 1,
        "precip_prev3d_mm": groups * 3,
        "precip_prev7d_mm": groups * 7,
        "precip_prev14d_mm": groups * 14,
        "ivt_mag_prev3d_mean": groups * 3,
        "ivt_mag_prev7d_mean": groups * 7,
        "soil_profile_mean_prev1d_m3_m3": groups * 1,
    }

    actual_nan_counts = {
        c: int(df[c].isna().sum())
        for c in expected_nan_counts
    }

    for c, expected in expected_nan_counts.items():
        actual = actual_nan_counts[c]
        if actual != expected:
            reasons.append(
                f"{c}_nan={actual} expected={expected}"
            )

    # Nessuna derived fondamentale deve essere tutta NaN.
    core_derived = [
        "ivt_dir_sin",
        "ivt_dir_cos",
        "precip_prev1d_mm",
        "precip_prev3d_mm",
        "precip_prev7d_mm",
        "soil_profile_mean_prev1d_m3_m3",
        "qwind925_proxy",
        "qwind850_proxy",
        "qwind700_proxy",
    ]
    all_nan = [
        c for c in core_derived
        if df[c].isna().all()
    ]
    if all_nan:
        reasons.append(f"all_nan={all_nan}")

    overall = "PASS" if not reasons else "REVIEW"

    # ------------------------------------------------------------------
    # WRITE
    # ------------------------------------------------------------------
    out_csv = (
        out_dir
        / "era5_basin_features_daily_derived_1987_2025_v1_0.csv"
    )

    tmp = out_csv.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out_csv)

    report = {
        "version": "1.0",
        "overall_status": overall,
        "input_rows": EXPECTED_ROWS,
        "output_rows": int(len(df)),
        "receptors": int(df["receptor_id"].nunique()),
        "season_year_min": int(df["season_year"].min()),
        "season_year_max": int(df["season_year"].max()),
        "season_day_min": int(df["season_day"].min()),
        "season_day_max": int(df["season_day"].max()),
        "duplicate_keys": dup2,
        "expected_edge_nan_counts": expected_nan_counts,
        "actual_edge_nan_counts": actual_nan_counts,
        "core_all_nan": all_nan,
        "reasons": reasons,
        "causality_note": (
            "All columns named prev* use only previous days within the "
            "same Sep-Dec season. No Dec->Sep bridging."
        ),
        "leakage_note": (
            "No historical climatology, percentile or z-score is computed "
            "here. Those must be fitted using training data only in each "
            "validation fold."
        ),
        "september_note": (
            "Early-September antecedent windows are NaN because August ERA5 "
            "was not downloaded in the historical dataset."
        ),
        "raw_modified": False,
    }

    json_p = (
        out_dir
        / "era5_basin_features_derived_audit_v1_0.json"
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
        / "era5_basin_features_derived_audit_v1_0.txt"
    )

    lines = [
        "=" * 126,
        "ERA5 -> 21 BACINI — FEATURE DERIVATE CAUSALI v1.0",
        "=" * 126,
        f"OVERALL STATUS          : {overall}",
        f"Righe input             : {EXPECTED_ROWS}",
        f"Righe output            : {len(df)}",
        f"Recettori               : {df['receptor_id'].nunique()}",
        f"Season years            : {df['season_year'].min()}-{df['season_year'].max()}",
        f"Season day              : {df['season_day'].min()}-{df['season_day'].max()}",
        f"Chiavi duplicate        : {dup2}",
        f"Core derived tutte NaN  : {all_nan}",
        "",
        "EDGE NaN ATTESI / OSSERVATI",
    ]

    for c, expected in expected_nan_counts.items():
        lines.append(
            f"{c:<40}: {actual_nan_counts[c]} / {expected}"
        )

    lines += [
        "",
        "NOTE",
        "- prev* = solo giorni precedenti, stessa stagione.",
        "- nessun collegamento dicembre -> settembre successivo.",
        "- nessuna climatologia/normalizzazione globale: evitare leakage.",
        "- primi giorni di settembre: antecedenti incompleti perché agosto non disponibile.",
        "",
        f"Output: {out_csv}",
    ]

    txt_p.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"Righe input              : {EXPECTED_ROWS}")
    print(f"Righe output             : {len(df)}")
    print(f"Recettori                : {df['receptor_id'].nunique()}")
    print(f"Chiavi duplicate         : {dup2}")
    print(f"Core derived tutte NaN   : {all_nan}")

    print("\nEDGE NaN ATTESI / OSSERVATI")
    for c, expected in expected_nan_counts.items():
        print(
            f"{c:<40}: {actual_nan_counts[c]} / {expected}"
        )

    print("\n" + "=" * 126)
    print(f"OVERALL STATUS : {overall}")
    print(f"Output         : {out_dir}")
    if reasons:
        print("REASONS:")
        for r in reasons:
            print(f"  - {r}")
    print("=" * 126)

    if overall != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
