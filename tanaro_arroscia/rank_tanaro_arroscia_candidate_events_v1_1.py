#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rank_tanaro_arroscia_candidate_events_v1_1.py

Secondo livello dello screening Tanaro–Arroscia.

Legge:
  tanaro_arroscia/hydrology/joint_daily_screen_v1_0/
    tanaro_arroscia_joint_daily_v1_0.csv
    tanaro_arroscia_candidate_events_v1_0.csv

Produce:
  tanaro_arroscia/hydrology/joint_daily_screen_v1_1/
    tanaro_arroscia_candidate_events_ranked_v1_1.csv
    tanaro_arroscia_garessio_hourly_retrieval_windows_v1_1.csv
    tanaro_arroscia_candidate_events_ranked_v1_1.txt
    tanaro_arroscia_candidate_events_ranked_v1_1.json

Obiettivo:
- ordinare i 17 eventi candidati;
- distinguere eventi con entrambi i bacini alti, Tanaro-dominanti,
  Arroscia-dominanti e controlli con picchi nello stesso giorno;
- calcolare SOLO uno sfasamento a scala giornaliera;
- preparare finestre +/- 3 giorni per il successivo reperimento
  dei dati sub-giornalieri/orari di Garessio.

ATTENZIONE:
- NON è una prova di asincronia oraria.
- NON è una prova di beneficio del collegamento idraulico.
- NON interpola dati mancanti.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SCREEN_THRESHOLD = 90.0
RETRIEVAL_PAD_DAYS = 3


def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def main():
    root = Path(__file__).resolve().parent

    in_dir = (
        root / "tanaro_arroscia" / "hydrology"
        / "joint_daily_screen_v1_0"
    )
    joint_path = in_dir / "tanaro_arroscia_joint_daily_v1_0.csv"
    events_path = in_dir / "tanaro_arroscia_candidate_events_v1_0.csv"

    out_dir = (
        root / "tanaro_arroscia" / "hydrology"
        / "joint_daily_screen_v1_1"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 132)
    print("TANARO–ARROSCIA — RANKING EVENTI CANDIDATI v1.1")
    print("=" * 132)

    if not joint_path.exists():
        raise SystemExit(f"File non trovato: {joint_path}")
    if not events_path.exists():
        raise SystemExit(f"File non trovato: {events_path}")

    joint = pd.read_csv(joint_path)
    events = pd.read_csv(events_path)

    joint["date"] = pd.to_datetime(joint["date"], errors="coerce")
    for c in [
        "garessio_discharge_mean_m3s",
        "garessio_q_percentile",
        "pieve_level_max_m",
        "pieve_hmax_percentile",
        "joint_min_percentile",
    ]:
        if c in joint.columns:
            joint[c] = pd.to_numeric(joint[c], errors="coerce")

    rows = []

    for _, ev in events.iterrows():
        event_id = int(ev["event_id"])
        start = pd.Timestamp(ev["start_date"])
        end = pd.Timestamp(ev["end_date"])

        w = joint[joint["date"].between(start, end)].copy()

        q = w["garessio_discharge_mean_m3s"]
        h = w["pieve_level_max_m"]
        qp = w["garessio_q_percentile"]
        hp = w["pieve_hmax_percentile"]

        both_observed = qp.notna() & hp.notna()
        both_high = both_observed & (qp >= SCREEN_THRESHOLD) & (hp >= SCREEN_THRESHOLD)

        if both_observed.any():
            daily_joint = pd.concat([qp, hp], axis=1).min(axis=1)
            daily_joint = daily_joint.where(both_observed)
            max_joint = safe_float(daily_joint.max())

            if daily_joint.notna().any():
                idx_joint = daily_joint.idxmax()
                joint_best_date = w.loc[idx_joint, "date"]
            else:
                joint_best_date = pd.NaT
        else:
            max_joint = None
            joint_best_date = pd.NaT

        q_peak_date = (
            pd.Timestamp(ev["garessio_q_peak_date"])
            if pd.notna(ev.get("garessio_q_peak_date"))
            else pd.NaT
        )
        h_peak_date = (
            pd.Timestamp(ev["pieve_h_peak_date"])
            if pd.notna(ev.get("pieve_h_peak_date"))
            else pd.NaT
        )

        if pd.notna(q_peak_date) and pd.notna(h_peak_date):
            peak_offset_days = int((h_peak_date - q_peak_date).days)
        else:
            peak_offset_days = None

        max_qp = safe_float(qp.max())
        max_hp = safe_float(hp.max())

        q_high = max_qp is not None and max_qp >= SCREEN_THRESHOLD
        h_high = max_hp is not None and max_hp >= SCREEN_THRESHOLD
        has_same_day_both_high = bool(both_high.any())

        if q_high and h_high:
            if peak_offset_days == 0:
                category = "BOTH_HIGH_SAME_DAY_PEAK_CONTROL"
            elif peak_offset_days is not None and peak_offset_days > 0:
                category = "BOTH_HIGH_PIEVE_PEAK_LATER_DAILY"
            elif peak_offset_days is not None and peak_offset_days < 0:
                category = "BOTH_HIGH_PIEVE_PEAK_EARLIER_DAILY"
            else:
                category = "BOTH_HIGH_DAILY_TIMING_UNRESOLVED"
        elif q_high and not h_high:
            category = "TANARO_DOMINANT"
        elif h_high and not q_high:
            category = "ARROSCIA_DOMINANT"
        else:
            category = "MIXED_BELOW_EVENT_MAX_THRESHOLD"

        # Priorità per recupero sub-giornaliero:
        # 1) entrambi alti ma picchi in giorni diversi;
        # 2) entrambi alti nello stesso giorno come controllo;
        # 3) eventi dominanti di uno dei due bacini.
        if category in {
            "BOTH_HIGH_PIEVE_PEAK_LATER_DAILY",
            "BOTH_HIGH_PIEVE_PEAK_EARLIER_DAILY",
        }:
            retrieval_priority = 1
        elif category == "BOTH_HIGH_SAME_DAY_PEAK_CONTROL":
            retrieval_priority = 2
        else:
            retrieval_priority = 3

        # Score di ordinamento puramente esplorativo.
        # Privilegia la contemporanea elevazione dei due sistemi.
        concurrence = max_joint if max_joint is not None else -1.0
        maxside = max(
            [x for x in [max_qp, max_hp] if x is not None],
            default=-1.0,
        )
        ranking_score = concurrence * 1000.0 + maxside

        rows.append({
            "event_id": event_id,
            "start_date": str(start.date()),
            "end_date": str(end.date()),
            "duration_calendar_days": int((end - start).days + 1),
            "category_daily": category,
            "retrieval_priority": retrieval_priority,
            "peak_offset_days_pieve_minus_garessio": peak_offset_days,
            "garessio_q_peak_date": (
                str(q_peak_date.date()) if pd.notna(q_peak_date) else None
            ),
            "garessio_q_peak_m3s_daily_mean": safe_float(q.max()),
            "pieve_h_peak_date": (
                str(h_peak_date.date()) if pd.notna(h_peak_date) else None
            ),
            "pieve_h_peak_m": safe_float(h.max()),
            "max_garessio_q_percentile": max_qp,
            "max_pieve_hmax_percentile": max_hp,
            "max_same_day_joint_min_percentile": max_joint,
            "best_joint_date": (
                str(joint_best_date.date()) if pd.notna(joint_best_date) else None
            ),
            "same_day_both_ge90_present": has_same_day_both_high,
            "garessio_q_missing_days": int(q.isna().sum()),
            "pieve_h_missing_days": int(h.isna().sum()),
            "pieve_incomplete_24h_days": int(
                (
                    w["pieve_level_max_m"].notna()
                    & ~w["pieve_day_complete_24h"].fillna(False)
                ).sum()
            ),
            "ranking_score_internal": ranking_score,
            "interpretation_limit": (
                "Daily-scale screening only; hourly/sub-daily Garessio data "
                "required before any peak-lag inference."
            ),
        })

    ranked = pd.DataFrame(rows)

    ranked = ranked.sort_values(
        [
            "retrieval_priority",
            "ranking_score_internal",
            "event_id",
        ],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    ranked.insert(0, "rank", range(1, len(ranked) + 1))

    # Finestre da reperire: +/- 3 giorni attorno all'intero evento.
    windows = ranked.copy()
    windows["retrieval_start_date"] = (
        pd.to_datetime(windows["start_date"]) - pd.Timedelta(days=RETRIEVAL_PAD_DAYS)
    ).dt.strftime("%Y-%m-%d")
    windows["retrieval_end_date"] = (
        pd.to_datetime(windows["end_date"]) + pd.Timedelta(days=RETRIEVAL_PAD_DAYS)
    ).dt.strftime("%Y-%m-%d")

    windows = windows[
        [
            "rank",
            "event_id",
            "retrieval_priority",
            "category_daily",
            "retrieval_start_date",
            "retrieval_end_date",
            "garessio_q_peak_date",
            "pieve_h_peak_date",
            "peak_offset_days_pieve_minus_garessio",
            "max_garessio_q_percentile",
            "max_pieve_hmax_percentile",
            "max_same_day_joint_min_percentile",
        ]
    ].copy()

    ranked_csv = out_dir / "tanaro_arroscia_candidate_events_ranked_v1_1.csv"
    windows_csv = out_dir / "tanaro_arroscia_garessio_hourly_retrieval_windows_v1_1.csv"

    ranked.drop(columns=["ranking_score_internal"]).to_csv(ranked_csv, index=False)
    windows.to_csv(windows_csv, index=False)

    categories = ranked["category_daily"].value_counts().to_dict()

    report = {
        "version": "1.1",
        "events_total": int(len(ranked)),
        "screen_threshold_percentile": SCREEN_THRESHOLD,
        "retrieval_padding_days": RETRIEVAL_PAD_DAYS,
        "category_counts": categories,
        "method_note": (
            "Peak offset is calculated from DAILY peak dates only. It must not "
            "be interpreted as hour-scale hydrologic lag. Retrieval windows are "
            "designed for a subsequent official sub-daily Garessio data search."
        ),
        "ranked_events": ranked.drop(
            columns=["ranking_score_internal"]
        ).to_dict(orient="records"),
    }

    json_p = out_dir / "tanaro_arroscia_candidate_events_ranked_v1_1.json"
    json_p.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    txt_p = out_dir / "tanaro_arroscia_candidate_events_ranked_v1_1.txt"
    lines = [
        "=" * 132,
        "TANARO–ARROSCIA — RANKING EVENTI CANDIDATI v1.1",
        "=" * 132,
        f"Eventi totali              : {len(ranked)}",
        f"Soglia screening           : percentile >= {SCREEN_THRESHOLD:g}",
        f"Padding finestre retrieval : +/- {RETRIEVAL_PAD_DAYS} giorni",
        "",
        "CATEGORIE",
    ]

    for k, v in categories.items():
        lines.append(f"  {k}: {v}")

    lines += [
        "",
        "RANKING",
        (
            "rank | event | dates | category | offset giorni "
            "(Pieve-Garessio) | Qmax | Hmax | joint%"
        ),
    ]

    for _, r in ranked.iterrows():
        lines.append(
            f"{int(r['rank']):02d} | "
            f"E{int(r['event_id']):02d} | "
            f"{r['start_date']}..{r['end_date']} | "
            f"{r['category_daily']} | "
            f"offset={r['peak_offset_days_pieve_minus_garessio']} | "
            f"Q={r['garessio_q_peak_m3s_daily_mean']} | "
            f"H={r['pieve_h_peak_m']} | "
            f"joint={r['max_same_day_joint_min_percentile']}"
        )

    lines += [
        "",
        "LIMITAZIONE",
        "Lo sfasamento qui espresso è soltanto in giorni.",
        "Nessuna conclusione sull'asincronia oraria o sulla convenienza",
        "del collegamento idraulico può essere tratta da questo prodotto.",
        "",
        "PROSSIMO PASSO",
        "Usare le finestre generate per cercare il dato ufficiale",
        "sub-giornaliero/orario di Garessio nei soli eventi prioritari.",
        "",
        "OUTPUT",
        f"  {ranked_csv}",
        f"  {windows_csv}",
        f"  {json_p}",
    ]

    txt_p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nRANKING EVENTI")
    display_cols = [
        "rank",
        "event_id",
        "start_date",
        "end_date",
        "category_daily",
        "peak_offset_days_pieve_minus_garessio",
        "garessio_q_peak_m3s_daily_mean",
        "pieve_h_peak_m",
        "max_same_day_joint_min_percentile",
    ]
    print(ranked[display_cols].to_string(index=False))

    print("\n" + "=" * 132)
    print(f"Eventi totali : {len(ranked)}")
    print("Categorie:")
    for k, v in categories.items():
        print(f"  {k}: {v}")
    print(f"Output        : {out_dir}")
    print("=" * 132)


if __name__ == "__main__":
    main()
