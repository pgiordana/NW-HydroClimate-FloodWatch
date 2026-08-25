#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audit_medsea_ivt_basin_coupling_support_v1_1.py

Audit semantico del supporto marino per:
medsea_ivt_basin_coupling_daily_1987_2025_v1_0.csv

Scopo:
- distinguere i NaN "tecnici" da assenza reale di supporto marino lungo
  il corridoio di provenienza IVT;
- quantificare il supporto per recettore, mese, anno e settore di provenienza;
- verificare che SST e OHC abbiano identico stato di supporto;
- verificare coerenza valore/support_weight;
- NON modifica il prodotto v1.0.

Classificazione:
MEDSEA_SUPPORTED:
  support SST > 0 e support OHC > 0, valori presenti.
NO_MEDSEA_SOURCE_SUPPORT:
  support SST == 0 e support OHC == 0, valori NaN.
INCONSISTENT:
  ogni altro caso.

Output:
medsea_historical_analysis/basin_coupling_historical_v1_0/support_audit_v1_1/
  support_summary_by_receptor_v1_1.csv
  support_summary_by_month_v1_1.csv
  support_summary_by_year_v1_1.csv
  support_summary_by_source_sector_v1_1.csv
  support_summary_by_receptor_month_v1_1.csv
  support_audit_v1_1.json
  support_audit_v1_1.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_ROWS = 99918
EXPECTED_RECEPTORS = 21


def pct(n, d):
    return 100.0 * n / d if d else np.nan


def main():
    root = Path(__file__).resolve().parent

    base_dir = (
        root / "medsea_historical_analysis"
        / "basin_coupling_historical_v1_0"
    )
    csv_path = (
        base_dir
        / "medsea_ivt_basin_coupling_daily_1987_2025_v1_0.csv"
    )
    audit_path = (
        base_dir
        / "medsea_ivt_basin_coupling_audit_v1_0.json"
    )

    out_dir = base_dir / "support_audit_v1_1"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 124)
    print("MEDSEA × IVT -> 21 BACINI — SUPPORT AUDIT v1.1")
    print("=" * 124)

    if not csv_path.exists():
        raise SystemExit(f"CSV coupling non trovato: {csv_path}")

    if not audit_path.exists():
        raise SystemExit(f"Audit v1.0 non trovato: {audit_path}")

    old_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if old_audit.get("overall_status") != "PASS":
        raise SystemExit(
            f"Coupling v1.0 non PASS: {old_audit.get('overall_status')}"
        )

    df = pd.read_csv(csv_path)

    required = [
        "date",
        "receptor_id",
        "marine_source_bearing_deg",
        "medsea_sst_anom_corridor_c",
        "medsea_ohc_anom_corridor_j_m2",
        "sst_corridor_support_weight",
        "ohc_corridor_support_weight",
        "ivt_mag_mean_kg_m1_s1",
    ]

    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"Colonne mancanti: {missing_cols}")

    df["date"] = pd.to_datetime(
        df["date"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    reasons = []

    invalid_dates = int(df["date"].isna().sum())
    if invalid_dates:
        reasons.append(f"invalid_dates={invalid_dates}")

    if len(df) != EXPECTED_ROWS:
        reasons.append(f"rows={len(df)} expected={EXPECTED_ROWS}")

    if df["receptor_id"].nunique() != EXPECTED_RECEPTORS:
        reasons.append(
            f"receptors={df['receptor_id'].nunique()} expected={EXPECTED_RECEPTORS}"
        )

    dup = int(df.duplicated(["date", "receptor_id"]).sum())
    if dup:
        reasons.append(f"duplicate_keys={dup}")

    sst_support = pd.to_numeric(
        df["sst_corridor_support_weight"], errors="coerce"
    )
    ohc_support = pd.to_numeric(
        df["ohc_corridor_support_weight"], errors="coerce"
    )
    sst_value = pd.to_numeric(
        df["medsea_sst_anom_corridor_c"], errors="coerce"
    )
    ohc_value = pd.to_numeric(
        df["medsea_ohc_anom_corridor_j_m2"], errors="coerce"
    )

    sst_supported = sst_support > 0
    ohc_supported = ohc_support > 0
    sst_present = sst_value.notna()
    ohc_present = ohc_value.notna()

    supported = (
        sst_supported
        & ohc_supported
        & sst_present
        & ohc_present
    )

    no_support = (
        (sst_support.fillna(0) <= 0)
        & (ohc_support.fillna(0) <= 0)
        & sst_value.isna()
        & ohc_value.isna()
    )

    inconsistent = ~(supported | no_support)

    df["medsea_support_class"] = np.select(
        [supported, no_support, inconsistent],
        [
            "MEDSEA_SUPPORTED",
            "NO_MEDSEA_SOURCE_SUPPORT",
            "INCONSISTENT",
        ],
        default="INCONSISTENT",
    )

    inconsistent_n = int(inconsistent.sum())
    if inconsistent_n:
        reasons.append(f"inconsistent_support_rows={inconsistent_n}")

    # SST/OHC state must match exactly.
    state_mismatch = int(
        (
            (sst_supported != ohc_supported)
            | (sst_present != ohc_present)
        ).sum()
    )
    if state_mismatch:
        reasons.append(f"sst_ohc_state_mismatch={state_mismatch}")

    # Bearing bins centered every 22.5°, same family as coupling.
    bearing = pd.to_numeric(
        df["marine_source_bearing_deg"], errors="coerce"
    ) % 360.0

    df["source_sector_22p5_deg"] = (
        np.round(bearing / 22.5) * 22.5
    ) % 360.0

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    def summarize(group):
        n = len(group)
        supp = int((group["medsea_support_class"] == "MEDSEA_SUPPORTED").sum())
        nos = int(
            (group["medsea_support_class"] == "NO_MEDSEA_SOURCE_SUPPORT").sum()
        )
        inc = int((group["medsea_support_class"] == "INCONSISTENT").sum())

        ivt_supported = pd.to_numeric(
            group.loc[
                group["medsea_support_class"] == "MEDSEA_SUPPORTED",
                "ivt_mag_mean_kg_m1_s1",
            ],
            errors="coerce",
        )

        ivt_nos = pd.to_numeric(
            group.loc[
                group["medsea_support_class"] == "NO_MEDSEA_SOURCE_SUPPORT",
                "ivt_mag_mean_kg_m1_s1",
            ],
            errors="coerce",
        )

        return pd.Series({
            "rows": n,
            "supported_rows": supp,
            "no_medsea_support_rows": nos,
            "inconsistent_rows": inc,
            "supported_pct": pct(supp, n),
            "no_medsea_support_pct": pct(nos, n),
            "ivt_mean_supported": (
                float(ivt_supported.mean())
                if ivt_supported.notna().any()
                else np.nan
            ),
            "ivt_mean_no_medsea_support": (
                float(ivt_nos.mean())
                if ivt_nos.notna().any()
                else np.nan
            ),
        })

    by_receptor = (
        df.groupby("receptor_id", sort=True, dropna=False)
        .apply(summarize, include_groups=False)
        .reset_index()
    )

    by_month = (
        df.groupby("month", sort=True, dropna=False)
        .apply(summarize, include_groups=False)
        .reset_index()
    )

    by_year = (
        df.groupby("year", sort=True, dropna=False)
        .apply(summarize, include_groups=False)
        .reset_index()
    )

    by_sector = (
        df.groupby("source_sector_22p5_deg", sort=True, dropna=False)
        .apply(summarize, include_groups=False)
        .reset_index()
    )

    by_receptor_month = (
        df.groupby(
            ["receptor_id", "month"],
            sort=True,
            dropna=False,
        )
        .apply(summarize, include_groups=False)
        .reset_index()
    )

    by_receptor.to_csv(
        out_dir / "support_summary_by_receptor_v1_1.csv",
        index=False,
    )
    by_month.to_csv(
        out_dir / "support_summary_by_month_v1_1.csv",
        index=False,
    )
    by_year.to_csv(
        out_dir / "support_summary_by_year_v1_1.csv",
        index=False,
    )
    by_sector.to_csv(
        out_dir / "support_summary_by_source_sector_v1_1.csv",
        index=False,
    )
    by_receptor_month.to_csv(
        out_dir / "support_summary_by_receptor_month_v1_1.csv",
        index=False,
    )

    supported_n = int(supported.sum())
    no_support_n = int(no_support.sum())

    overall = "PASS" if not reasons else "REVIEW"

    report = {
        "version": "1.1",
        "overall_status": overall,
        "rows": int(len(df)),
        "receptors": int(df["receptor_id"].nunique()),
        "supported_rows": supported_n,
        "supported_pct": pct(supported_n, len(df)),
        "no_medsea_source_support_rows": no_support_n,
        "no_medsea_source_support_pct": pct(no_support_n, len(df)),
        "inconsistent_rows": inconsistent_n,
        "sst_ohc_state_mismatch": state_mismatch,
        "interpretation": (
            "NO_MEDSEA_SOURCE_SUPPORT is a physical/methodological state: "
            "for that basin-day the IVT-conditioned backward source corridor "
            "does not intersect a valid Mediterranean marine cell under the "
            "current kernel/cutoff/distance settings. It is not automatically "
            "a missing-data error and must not be imputed as zero."
        ),
        "next_step": (
            "Run kernel-parameter sensitivity before freezing the marine coupling "
            "for scientific inference."
        ),
        "reasons": reasons,
    }

    (out_dir / "support_audit_v1_1.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "=" * 124,
        "MEDSEA × IVT -> 21 BACINI — SUPPORT AUDIT v1.1",
        "=" * 124,
        f"OVERALL STATUS                : {overall}",
        f"Righe                         : {len(df)}",
        f"Recettori                     : {df['receptor_id'].nunique()}",
        f"MEDSEA_SUPPORTED              : {supported_n} ({pct(supported_n, len(df)):.3f}%)",
        f"NO_MEDSEA_SOURCE_SUPPORT      : {no_support_n} ({pct(no_support_n, len(df)):.3f}%)",
        f"INCONSISTENT                  : {inconsistent_n}",
        f"SST/OHC state mismatch        : {state_mismatch}",
        "",
        "INTERPRETAZIONE",
        "NO_MEDSEA_SOURCE_SUPPORT non è automaticamente un dato mancante:",
        "indica che il corridoio retrogrado condizionato da IVT non trova",
        "supporto marino mediterraneo valido con i parametri correnti.",
        "Non imputare a zero.",
        "",
        "NEXT",
        "Sensitivity test dei parametri del kernel prima di congelare il coupling.",
    ]

    (out_dir / "support_audit_v1_1.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"Righe                         : {len(df)}")
    print(f"Recettori                     : {df['receptor_id'].nunique()}")
    print(
        f"MEDSEA_SUPPORTED              : {supported_n} "
        f"({pct(supported_n, len(df)):.3f}%)"
    )
    print(
        f"NO_MEDSEA_SOURCE_SUPPORT      : {no_support_n} "
        f"({pct(no_support_n, len(df)):.3f}%)"
    )
    print(f"INCONSISTENT                  : {inconsistent_n}")
    print(f"SST/OHC state mismatch        : {state_mismatch}")

    print("\nSUPPORTO PER RECETTORE")
    print(
        by_receptor[
            [
                "receptor_id",
                "rows",
                "supported_rows",
                "no_medsea_support_rows",
                "supported_pct",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 124)
    print(f"OVERALL STATUS : {overall}")
    print(f"Output         : {out_dir}")
    if reasons:
        print("REASONS:")
        for r in reasons:
            print(f"  - {r}")
    print("=" * 124)

    if overall != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
