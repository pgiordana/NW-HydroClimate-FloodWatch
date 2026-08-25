#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_tanaro_arroscia_joint_daily_screen_v1_0.py

Screening congiunto preliminare Tanaro–Arroscia sui soli anni in cui:
- Pieve di Teco ha livello orario osservato (2021-2025);
- Garessio ha dati giornalieri osservati.

IMPORTANTE:
- NON stima ritardi di colmo orari.
- NON interpola Garessio giornaliero.
- NON converte ancora i timestamp Pieve in UTC.
- Usa la portata media giornaliera di Garessio per screening/volume a scala giornaliera.
- Aggrega Pieve di Teco a livello massimo/medio giornaliero per un confronto alla stessa scala.

Input canonici:
1) tanaro_arroscia/hydrology/pieve_teco_level_audit_v1_3/
   pieve_teco_level_canonical_local_v1_3.csv
2) tanaro_arroscia/hydrology/garessio_tanaro_daily_audit_v1_1/
   garessio_tanaro_canonical_daily_v1_1.csv

Output:
tanaro_arroscia/hydrology/joint_daily_screen_v1_0/
  tanaro_arroscia_joint_daily_v1_0.csv
  tanaro_arroscia_candidate_events_v1_0.csv
  tanaro_arroscia_overlap_summary_v1_0.json
  tanaro_arroscia_overlap_summary_v1_0.txt

Criterio candidato esplorativo:
- soglia percentile configurabile (default 90° percentile) su:
    Garessio portata media giornaliera
    oppure
    Pieve livello massimo giornaliero
- giorni consecutivi o separati da <= 1 giorno sono raggruppati nello stesso evento.

La soglia è solo di screening, non una soglia di piena ufficiale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def percentile_rank(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True) * 100.0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--percentile",
        type=float,
        default=90.0,
        help="Percentile esplorativo per selezione giorni candidati (default 90).",
    )
    ap.add_argument(
        "--merge-gap-days",
        type=int,
        default=1,
        help="Giorni massimi di separazione da fondere nello stesso evento (default 1).",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    if not (0 < args.percentile < 100):
        raise SystemExit("--percentile deve essere compreso tra 0 e 100.")
    if args.merge_gap_days < 0:
        raise SystemExit("--merge-gap-days deve essere >= 0.")

    root = Path(__file__).resolve().parent

    pieve_path = (
        root / "tanaro_arroscia" / "hydrology"
        / "pieve_teco_level_audit_v1_3"
        / "pieve_teco_level_canonical_local_v1_3.csv"
    )

    garessio_path = (
        root / "tanaro_arroscia" / "hydrology"
        / "garessio_tanaro_daily_audit_v1_1"
        / "garessio_tanaro_canonical_daily_v1_1.csv"
    )

    out_dir = (
        root / "tanaro_arroscia" / "hydrology"
        / "joint_daily_screen_v1_0"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 124)
    print("TANARO–ARROSCIA — SCREENING CONGIUNTO GIORNALIERO v1.0")
    print("=" * 124)

    if not pieve_path.exists():
        raise SystemExit(f"File Pieve non trovato: {pieve_path}")
    if not garessio_path.exists():
        raise SystemExit(f"File Garessio non trovato: {garessio_path}")

    # PIEVE: orario -> giornaliero
    p = pd.read_csv(pieve_path)

    required_p = {"start_local", "level_m"}
    miss = required_p - set(p.columns)
    if miss:
        raise SystemExit(f"Colonne Pieve mancanti: {sorted(miss)}")

    p["start_local"] = pd.to_datetime(
        p["start_local"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    p["level_m"] = pd.to_numeric(p["level_m"], errors="coerce")
    p = p[p["start_local"].notna() & p["level_m"].notna()].copy()
    p["date"] = p["start_local"].dt.normalize()

    idxmax = p.groupby("date")["level_m"].idxmax()
    p_peak_time = (
        p.loc[idxmax, ["date", "start_local"]]
        .rename(columns={"start_local": "pieve_peak_time_local"})
    )

    p_daily = (
        p.groupby("date")
        .agg(
            pieve_level_max_m=("level_m", "max"),
            pieve_level_mean_m=("level_m", "mean"),
            pieve_obs_hours=("level_m", "count"),
        )
        .reset_index()
        .merge(p_peak_time, on="date", how="left")
    )
    p_daily["pieve_day_complete_24h"] = p_daily["pieve_obs_hours"].eq(24)

    # GARESSIO: giornaliero
    g = pd.read_csv(garessio_path)

    required_g = {"date", "level_mean_m", "discharge_mean_m3s"}
    miss = required_g - set(g.columns)
    if miss:
        raise SystemExit(f"Colonne Garessio mancanti: {sorted(miss)}")

    g["date"] = pd.to_datetime(g["date"], format="%Y-%m-%d", errors="coerce")
    g["level_mean_m"] = pd.to_numeric(g["level_mean_m"], errors="coerce")
    g["discharge_mean_m3s"] = pd.to_numeric(
        g["discharge_mean_m3s"], errors="coerce"
    )
    g = g[g["date"].notna()].copy()
    g = g.rename(
        columns={
            "level_mean_m": "garessio_level_mean_m",
            "discharge_mean_m3s": "garessio_discharge_mean_m3s",
        }
    )

    keep_g = [
        c for c in [
            "date",
            "garessio_level_mean_m",
            "garessio_discharge_mean_m3s",
            "level_class",
            "discharge_class",
        ]
        if c in g.columns
    ]
    g = g[keep_g].copy()

    # OVERLAP 2021-2025, Sep-Dec
    joint = pd.merge(
        p_daily, g, on="date", how="outer", validate="one_to_one"
    ).sort_values("date")

    joint = joint[
        joint["date"].dt.year.between(2021, 2025)
        & joint["date"].dt.month.isin([9, 10, 11, 12])
    ].copy()

    joint["year"] = joint["date"].dt.year
    joint["month"] = joint["date"].dt.month

    joint["garessio_q_percentile"] = percentile_rank(
        joint["garessio_discharge_mean_m3s"]
    )
    joint["pieve_hmax_percentile"] = percentile_rank(
        joint["pieve_level_max_m"]
    )

    joint["joint_min_percentile"] = joint[
        ["garessio_q_percentile", "pieve_hmax_percentile"]
    ].min(axis=1, skipna=False)

    threshold = float(args.percentile)

    joint["candidate_by_garessio_q"] = (
        joint["garessio_q_percentile"] >= threshold
    )
    joint["candidate_by_pieve_hmax"] = (
        joint["pieve_hmax_percentile"] >= threshold
    )
    joint["candidate_day"] = (
        joint["candidate_by_garessio_q"]
        | joint["candidate_by_pieve_hmax"]
    )

    # EVENTI CANDIDATI
    cand = joint[joint["candidate_day"]].copy()
    event_rows = []

    if len(cand):
        cand = cand.sort_values("date").copy()
        gap = cand["date"].diff().dt.days
        new_event = gap.isna() | (gap > args.merge_gap_days + 1)
        cand["event_id"] = new_event.cumsum().astype(int)

        for event_id, e in cand.groupby("event_id"):
            start = e["date"].min()
            end = e["date"].max()
            window = joint[joint["date"].between(start, end)].copy()

            q = window["garessio_discharge_mean_m3s"]
            h = window["pieve_level_max_m"]

            q_peak_date = (
                window.loc[q.idxmax(), "date"] if q.notna().any() else pd.NaT
            )
            h_peak_date = (
                window.loc[h.idxmax(), "date"] if h.notna().any() else pd.NaT
            )

            event_rows.append({
                "event_id": int(event_id),
                "start_date": str(start.date()),
                "end_date": str(end.date()),
                "duration_calendar_days": int((end - start).days + 1),
                "candidate_days": int(len(e)),
                "garessio_q_peak_m3s_daily_mean": (
                    float(q.max()) if q.notna().any() else None
                ),
                "garessio_q_peak_date": (
                    str(q_peak_date.date()) if pd.notna(q_peak_date) else None
                ),
                "pieve_h_peak_m": (
                    float(h.max()) if h.notna().any() else None
                ),
                "pieve_h_peak_date": (
                    str(h_peak_date.date()) if pd.notna(h_peak_date) else None
                ),
                "max_garessio_q_percentile": (
                    float(window["garessio_q_percentile"].max())
                    if window["garessio_q_percentile"].notna().any()
                    else None
                ),
                "max_pieve_hmax_percentile": (
                    float(window["pieve_hmax_percentile"].max())
                    if window["pieve_hmax_percentile"].notna().any()
                    else None
                ),
                "pieve_incomplete_days_in_window": int(
                    (
                        window["pieve_level_max_m"].notna()
                        & ~window["pieve_day_complete_24h"].fillna(False)
                    ).sum()
                ),
                "garessio_q_missing_days_in_window": int(
                    window["garessio_discharge_mean_m3s"].isna().sum()
                ),
                "timing_note": (
                    "Daily screening only; not usable for hour-scale peak lag."
                ),
            })

    events = pd.DataFrame(event_rows)

    # COPERTURA PER ANNO
    year_summary = []
    for year, y in joint.groupby("year"):
        year_summary.append({
            "year": int(year),
            "calendar_days_in_overlap": int(len(y)),
            "pieve_days_present": int(y["pieve_level_max_m"].notna().sum()),
            "pieve_complete_24h_days": int(
                (
                    y["pieve_level_max_m"].notna()
                    & y["pieve_day_complete_24h"].fillna(False)
                ).sum()
            ),
            "garessio_q_days_present": int(
                y["garessio_discharge_mean_m3s"].notna().sum()
            ),
            "garessio_level_days_present": int(
                y["garessio_level_mean_m"].notna().sum()
            ),
            "candidate_days": int(y["candidate_day"].sum()),
        })

    joint_csv = out_dir / "tanaro_arroscia_joint_daily_v1_0.csv"
    joint.to_csv(joint_csv, index=False)

    events_csv = out_dir / "tanaro_arroscia_candidate_events_v1_0.csv"
    events.to_csv(events_csv, index=False)

    summary = {
        "version": "1.0",
        "screening_scale": "daily",
        "period": "Sep-Dec 2021-2025",
        "threshold_percentile": threshold,
        "merge_gap_days": int(args.merge_gap_days),
        "candidate_rule": (
            "Garessio daily mean discharge percentile >= threshold OR "
            "Pieve daily maximum stage percentile >= threshold"
        ),
        "not_a_flood_threshold": True,
        "not_for_hourly_peak_lag": True,
        "pieve_timezone": "local_naive_preserved",
        "joint_rows": int(len(joint)),
        "candidate_days": int(joint["candidate_day"].sum()),
        "candidate_events": int(len(events)),
        "year_summary": year_summary,
        "pieve_incomplete_dates": [
            str(d.date())
            for d in joint.loc[
                joint["pieve_level_max_m"].notna()
                & ~joint["pieve_day_complete_24h"].fillna(False),
                "date",
            ]
        ],
        "garessio_q_missing_dates": [
            str(d.date())
            for d in joint.loc[
                joint["garessio_discharge_mean_m3s"].isna(),
                "date",
            ]
        ],
        "method_note": (
            "Exploratory event screening only. Daily mean discharge at "
            "Garessio is aligned with daily max/mean stage at Pieve di Teco. "
            "No interpolation or hourly lag inference is performed."
        ),
    }

    json_p = out_dir / "tanaro_arroscia_overlap_summary_v1_0.json"
    json_p.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    txt_p = out_dir / "tanaro_arroscia_overlap_summary_v1_0.txt"
    lines = [
        "=" * 124,
        "TANARO–ARROSCIA — SCREENING CONGIUNTO GIORNALIERO v1.0",
        "=" * 124,
        "Periodo                         : Sep-Dec 2021-2025",
        f"Soglia screening               : percentile >= {threshold:g}",
        f"Righe giornaliere              : {len(joint)}",
        f"Giorni candidati               : {int(joint['candidate_day'].sum())}",
        f"Eventi candidati               : {len(events)}",
        "",
        "COPERTURA PER ANNO",
    ]

    for r in year_summary:
        lines.append(
            f"{r['year']} | "
            f"Pieve={r['pieve_days_present']}/122 "
            f"(24h complete={r['pieve_complete_24h_days']}) | "
            f"Garessio Q={r['garessio_q_days_present']}/122 | "
            f"candidate={r['candidate_days']}"
        )

    lines += [
        "",
        "LIMITAZIONE METODOLOGICA",
        "Questo prodotto NON stima ritardi di colmo orari.",
        "Garessio resta a risoluzione giornaliera; Pieve è stato aggregato",
        "alla stessa scala solo per lo screening degli eventi da approfondire.",
        "",
        "OUTPUT",
        f"  {joint_csv}",
        f"  {events_csv}",
        f"  {json_p}",
    ]

    txt_p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nCOPERTURA OVERLAP")
    for r in year_summary:
        print(
            f"{r['year']} | "
            f"Pieve={r['pieve_days_present']:3d}/122 "
            f"| Pieve24h={r['pieve_complete_24h_days']:3d} "
            f"| GaressioQ={r['garessio_q_days_present']:3d}/122 "
            f"| candidate={r['candidate_days']:3d}"
        )

    print("\n" + "=" * 124)
    print(f"Threshold percentile : {threshold:g}")
    print(f"Giorni candidati     : {int(joint['candidate_day'].sum())}")
    print(f"Eventi candidati     : {len(events)}")
    print(f"Output               : {out_dir}")
    print("=" * 124)


if __name__ == "__main__":
    main()
