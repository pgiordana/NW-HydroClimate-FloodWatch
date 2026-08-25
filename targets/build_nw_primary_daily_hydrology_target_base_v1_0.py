#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_nw_primary_daily_hydrology_target_base_v1_0.py

COSTRUISCE LA BASE OSSERVATIVA GIORNALIERA DEI TARGET IDROLOGICI PRIMARI.

MOTIVAZIONE SCIENTIFICA
-----------------------
Il ramo di verifica delle soglie operative ha chiarito che:
- le soglie ufficiali sono preziose come riferimento operativo esterno;
- il catalogo ARPA Piemonte "Portate al colmo" contiene massimi annuali ma
  non la data dell'evento;
- il form storico idrologico orario fornisce livello idrometrico, non una
  serie storica oraria di portata direttamente confrontabile con le soglie
  FLOW senza una curva di deflusso ufficiale;
- le soglie operative correnti non devono essere assunte implicitamente come
  storicamente invarianti dal 1987 al 2025.

Per evitare:
- centinaia di richieste manuali;
- conversioni livello-portata non autorizzate;
- uso retroattivo non verificato di soglie operative correnti;

questa versione costruisce il TARGET BASE giornaliero osservato, senza ancora
creare etichette di piena.

POLICY
------
1) Liguria:
   sorgente oraria -> usa `daily_max` del livello quando disponibile.
2) Piemonte:
   sorgente giornaliera -> usa `daily_value` della variabile canonica
   (portata o livello secondo il primary), senza promuoverla a colmo.
3) Le soglie ufficiali v1.1 vengono riportate soltanto come
   CURRENT_OPERATIONAL_REFERENCE.
4) Nessuna soglia statistica viene calcolata qui.
5) Le future soglie/statistiche per il modello devono essere stimate
   ESCLUSIVAMENTE nel training fold.
6) Nessuna conversione STAGE <-> FLOW.
7) Nessuna imputazione.

OUTPUT
------
nw_primary_daily_hydrology_target_base_v1_0/
  nw_primary_daily_hydrology_target_base_1987_2025_v1_0.csv
  nw_primary_daily_hydrology_target_coverage_v1_0.csv
  nw_primary_daily_hydrology_target_policy_v1_0.csv
  nw_primary_daily_hydrology_target_audit_v1_0.json
  nw_primary_daily_hydrology_target_audit_v1_0.txt
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pandas as pd


START_DATE = pd.Timestamp("1987-09-01")
END_DATE = pd.Timestamp("2025-12-31")
TARGET_MONTHS = {9, 10, 11, 12}


def fmt_seconds(seconds: float) -> str:
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
        f"| rate {rate:7.2f}/s | ETA {fmt_seconds(eta)}"
    )

    if current:
        msg += f" | {str(current)[:105]}"

    print(msg.ljust(230), end="", flush=True)
    if done >= total:
        print(flush=True)


def resolve_path(root: Path, raw):
    s = str(raw or "").strip()

    if not s or s.lower() in {"nan", "none", "null"}:
        return None

    p = Path(s).expanduser()

    if p.is_absolute():
        return p

    return (root / p).resolve()


def selected_alignment_rows(candidates: pd.DataFrame):
    if "selected_for_alignment" not in candidates.columns:
        raise RuntimeError(
            "Manca selected_for_alignment nel mapping v1.1."
        )

    mask = (
        candidates["selected_for_alignment"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )

    sel = candidates[mask].copy()

    if "candidate_rank" in sel.columns:
        sel["_source_priority"] = pd.to_numeric(
            sel["candidate_rank"],
            errors="coerce",
        ).fillna(999999)
    else:
        sel["_source_priority"] = 999999

    return sel


def read_selected_daily_file(
    path: Path,
    variable_code: str,
    source_series_id: str,
    source_priority: float,
):
    if path is None or not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(
            path,
            compression="infer",
            low_memory=False,
        )
    except Exception:
        return pd.DataFrame()

    if "variable_code" not in df.columns:
        return pd.DataFrame()

    d = df[
        df["variable_code"].astype(str).eq(variable_code)
    ].copy()

    if not len(d):
        return pd.DataFrame()

    if "date_model" in d.columns:
        date_col = "date_model"
    elif "date" in d.columns:
        date_col = "date"
    else:
        return pd.DataFrame()

    d["_date"] = pd.to_datetime(
        d[date_col],
        errors="coerce",
    )

    d = d[
        d["_date"].notna()
        & d["_date"].between(START_DATE, END_DATE)
        & d["_date"].dt.month.isin(TARGET_MONTHS)
    ].copy()

    if not len(d):
        return pd.DataFrame()

    d["_daily_value"] = (
        pd.to_numeric(
            d["daily_value"],
            errors="coerce",
        )
        if "daily_value" in d.columns
        else float("nan")
    )

    d["_daily_max"] = (
        pd.to_numeric(
            d["daily_max"],
            errors="coerce",
        )
        if "daily_max" in d.columns
        else float("nan")
    )

    d["_expected_count_per_day"] = (
        pd.to_numeric(
            d["expected_count_per_day"],
            errors="coerce",
        )
        if "expected_count_per_day" in d.columns
        else float("nan")
    )

    d["_time_resolution_source"] = (
        d["time_resolution_source"].astype(str)
        if "time_resolution_source" in d.columns
        else ""
    )

    d["_daily_primary_statistic"] = (
        d["daily_primary_statistic"].astype(str)
        if "daily_primary_statistic" in d.columns
        else ""
    )

    d["_source_series_id"] = str(source_series_id)
    d["_source_priority"] = float(source_priority)
    d["_source_path"] = str(path)

    keep = [
        "_date",
        "_daily_value",
        "_daily_max",
        "_expected_count_per_day",
        "_time_resolution_source",
        "_daily_primary_statistic",
        "_source_series_id",
        "_source_priority",
        "_source_path",
    ]

    return d[keep].copy()


def choose_observation_semantics(provider, frames):
    """
    Return merged daily observations and target value semantics.

    No averaging across overlapping source-series epochs:
    lowest source priority wins on an overlapping date.
    """
    if not frames:
        return pd.DataFrame(), "", ""

    all_d = pd.concat(
        frames,
        ignore_index=True,
    )

    all_d = all_d.sort_values(
        [
            "_date",
            "_source_priority",
            "_source_series_id",
        ]
    )

    # Deterministic canonical preference on overlaps.
    dedup = all_d.drop_duplicates(
        subset=["_date"],
        keep="first",
    ).copy()

    provider = str(provider)

    if provider == "ARPAL":
        # ARPAL canonical series in this target set derive from hourly source.
        # Use a real daily maximum where available.
        target = dedup["_daily_max"].copy()

        fallback = target.isna()
        target.loc[fallback] = dedup.loc[
            fallback, "_daily_value"
        ]

        dedup["_target_observed_value"] = target
        dedup["_target_observation_statistic"] = (
            "DAILY_MAX_FROM_HOURLY_SOURCE"
        )
        dedup["_target_temporal_semantics"] = (
            "DAILY_MAX_LEVEL"
        )
    else:
        # Piemonte canonical source is daily here.
        dedup["_target_observed_value"] = (
            dedup["_daily_value"]
        )
        dedup["_target_observation_statistic"] = (
            "CANONICAL_DAILY_PRIMARY_VALUE"
        )
        dedup["_target_temporal_semantics"] = (
            "DAILY_SOURCE_VALUE_NOT_INSTANTANEOUS_PEAK"
        )

    return (
        dedup,
        str(
            dedup["_target_observation_statistic"]
            .iloc[0]
        )
        if len(dedup)
        else "",
        str(
            dedup["_target_temporal_semantics"]
            .iloc[0]
        )
        if len(dedup)
        else "",
    )


def build_date_index():
    all_days = pd.date_range(
        START_DATE,
        END_DATE,
        freq="D",
    )

    return all_days[
        all_days.month.isin(TARGET_MONTHS)
    ]


def main():
    root = Path(__file__).resolve().parent

    threshold_p = (
        root
        / "nw_official_primary_threshold_registry_v1_1"
        / "official_primary_threshold_registry_v1_1.csv"
    )

    alignment_candidates_p = (
        root
        / "nw_primary_threshold_observation_alignment_v1_1"
        / "primary_threshold_observation_alignment_candidates_v1_1.csv"
    )

    out_root = (
        root
        / "nw_primary_daily_hydrology_target_base_v1_0"
    )
    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 196)
    print("NW HYDROLOGY — PRIMARY DAILY HYDROLOGY TARGET BASE v1.0")
    print("=" * 196)

    for p in (
        threshold_p,
        alignment_candidates_p,
    ):
        if not p.exists():
            raise SystemExit(f"Manca: {p}")

    threshold = pd.read_csv(
        threshold_p,
        low_memory=False,
    )

    candidates = pd.read_csv(
        alignment_candidates_p,
        low_memory=False,
    )

    selected = selected_alignment_rows(
        candidates
    )

    if len(threshold) != 20:
        raise SystemExit(
            f"Attesi 20 primary, trovati {len(threshold)}."
        )

    print(f"Primary controls                  : {len(threshold)}")
    print(f"Selected alignment source rows    : {len(selected)}")

    date_index = build_date_index()

    # ------------------------------------------------------------------
    # PHASE 1/3 — assemble one canonical observed series per primary
    # ------------------------------------------------------------------
    print("\nPHASE 1/3 — assemble daily observed target series")
    start1 = time.time()

    long_rows = []
    coverage_rows = []

    total = len(threshold)

    for i, (_, t) in enumerate(
        threshold.sort_values("receptor_id").iterrows(),
        1,
    ):
        receptor = str(t["receptor_id"])
        provider = str(t["provider"])

        sel = selected[
            selected["receptor_id"]
            .astype(str)
            .eq(receptor)
        ].copy()

        if not len(sel):
            frames = []
            variable_code = ""
        else:
            # All selected rows for a receptor should share variable code.
            variable_codes = sorted(
                set(
                    sel["variable_code"]
                    .dropna()
                    .astype(str)
                )
            )

            if len(variable_codes) != 1:
                raise SystemExit(
                    f"{receptor}: selected variable codes non univoci: "
                    f"{variable_codes}"
                )

            variable_code = variable_codes[0]
            frames = []

            for _, r in sel.sort_values(
                "_source_priority"
            ).iterrows():
                path = resolve_path(
                    root,
                    r["daily_output_path"],
                )

                d = read_selected_daily_file(
                    path=path,
                    variable_code=variable_code,
                    source_series_id=str(
                        r["source_series_id"]
                    ),
                    source_priority=float(
                        r["_source_priority"]
                    ),
                )

                if len(d):
                    frames.append(d)

        merged, statistic, temporal_semantics = (
            choose_observation_semantics(
                provider,
                frames,
            )
        )

        base = pd.DataFrame(
            {
                "date": date_index,
            }
        )

        if len(merged):
            obs = merged.rename(
                columns={
                    "_date": "date",
                    "_target_observed_value":
                        "target_observed_value",
                    "_source_series_id":
                        "selected_source_series_id",
                    "_source_path":
                        "selected_source_path",
                    "_time_resolution_source":
                        "source_time_resolution",
                    "_expected_count_per_day":
                        "expected_count_per_day",
                    "_daily_primary_statistic":
                        "source_daily_primary_statistic",
                }
            )[
                [
                    "date",
                    "target_observed_value",
                    "selected_source_series_id",
                    "selected_source_path",
                    "source_time_resolution",
                    "expected_count_per_day",
                    "source_daily_primary_statistic",
                ]
            ].copy()

            base = base.merge(
                obs,
                on="date",
                how="left",
                validate="one_to_one",
            )
        else:
            base["target_observed_value"] = None
            base["selected_source_series_id"] = ""
            base["selected_source_path"] = ""
            base["source_time_resolution"] = ""
            base["expected_count_per_day"] = None
            base["source_daily_primary_statistic"] = ""

        base.insert(
            0,
            "receptor_id",
            receptor,
        )

        base["station_name"] = t["station_name"]
        base["station_id"] = t.get("station_id", "")
        base["provider"] = provider
        base["primary_role"] = t["primary_role"]
        base["observed_variable_code"] = variable_code
        base["threshold_family_reference"] = (
            t["threshold_family"]
        )
        base["threshold_unit_reference"] = (
            t["threshold_unit"]
        )

        base["official_threshold_1_name"] = (
            t.get("threshold_1_name", "")
        )
        base["official_threshold_1_value"] = (
            t.get("threshold_1_value", None)
        )
        base["official_threshold_2_name"] = (
            t.get("threshold_2_name", "")
        )
        base["official_threshold_2_value"] = (
            t.get("threshold_2_value", None)
        )
        base["official_threshold_3_name"] = (
            t.get("threshold_3_name", "")
        )
        base["official_threshold_3_value"] = (
            t.get("threshold_3_value", None)
        )

        base["target_observation_statistic"] = statistic
        base["target_temporal_semantics"] = temporal_semantics

        # Critical policy: no retrospective operational-threshold label.
        base["official_threshold_role"] = (
            "CURRENT_OPERATIONAL_REFERENCE_ONLY"
        )
        base[
            "official_threshold_historical_invariance_verified"
        ] = False
        base[
            "official_threshold_exceedance_label_created"
        ] = False

        base["model_statistical_label_status"] = (
            "DEFERRED__TRAINING_FOLD_ONLY"
        )

        base["centa_proxy_flag"] = (
            receptor == "LIG_CENTA"
        )

        long_rows.append(base)

        numeric = pd.to_numeric(
            base["target_observed_value"],
            errors="coerce",
        )

        valid_mask = numeric.notna()

        if valid_mask.any():
            first_valid = (
                base.loc[valid_mask, "date"]
                .min()
                .date()
                .isoformat()
            )

            last_valid = (
                base.loc[valid_mask, "date"]
                .max()
                .date()
                .isoformat()
            )

            years = int(
                base.loc[valid_mask, "date"]
                .dt.year
                .nunique()
            )
        else:
            first_valid = ""
            last_valid = ""
            years = 0

        coverage_rows.append(
            {
                "receptor_id": receptor,
                "station_name": t["station_name"],
                "provider": provider,
                "primary_role": t["primary_role"],
                "observed_variable_code": variable_code,
                "target_observation_statistic": statistic,
                "target_temporal_semantics":
                    temporal_semantics,
                "expected_grid_days": int(
                    len(date_index)
                ),
                "numeric_days": int(
                    valid_mask.sum()
                ),
                "coverage_fraction": (
                    float(valid_mask.mean())
                ),
                "years_with_numeric_values": years,
                "first_valid_date": first_valid,
                "last_valid_date": last_valid,
                "source_series_count": int(
                    len(sel)
                ),
                "centa_proxy_flag":
                    receptor == "LIG_CENTA",
            }
        )

        progress(
            "PHASE 1/3",
            i,
            total,
            start1,
            f"{receptor} | {t['station_name']} | {variable_code}",
        )

    target_base = pd.concat(
        long_rows,
        ignore_index=True,
    )

    coverage = pd.DataFrame(
        coverage_rows
    )

    # ------------------------------------------------------------------
    # PHASE 2/3 — structural audit
    # ------------------------------------------------------------------
    print("\nPHASE 2/3 — structural and causal-policy audit")
    start2 = time.time()

    expected_rows = (
        len(threshold) * len(date_index)
    )

    duplicates = int(
        target_base.duplicated(
            subset=[
                "receptor_id",
                "date",
            ]
        ).sum()
    )

    receptors = int(
        target_base["receptor_id"].nunique()
    )

    days_per_receptor = (
        target_base.groupby("receptor_id")["date"]
        .nunique()
    )

    bad_day_counts = int(
        (~days_per_receptor.eq(len(date_index))).sum()
    )

    numeric_total = int(
        pd.to_numeric(
            target_base["target_observed_value"],
            errors="coerce",
        )
        .notna()
        .sum()
    )

    if (
        len(target_base) != expected_rows
        or duplicates != 0
        or receptors != 20
        or bad_day_counts != 0
    ):
        overall = "FAIL"
    else:
        overall = "PASS"

    progress(
        "PHASE 2/3",
        1,
        1,
        start2,
        f"rows={len(target_base)} duplicates={duplicates}",
    )

    # ------------------------------------------------------------------
    # PHASE 3/3 — outputs
    # ------------------------------------------------------------------
    print("\nPHASE 3/3 — write canonical target-base outputs")
    start3 = time.time()

    target_out = (
        out_root
        / "nw_primary_daily_hydrology_target_base_1987_2025_v1_0.csv"
    )

    coverage_out = (
        out_root
        / "nw_primary_daily_hydrology_target_coverage_v1_0.csv"
    )

    policy_out = (
        out_root
        / "nw_primary_daily_hydrology_target_policy_v1_0.csv"
    )

    audit_json = (
        out_root
        / "nw_primary_daily_hydrology_target_audit_v1_0.json"
    )

    audit_txt = (
        out_root
        / "nw_primary_daily_hydrology_target_audit_v1_0.txt"
    )

    target_base.to_csv(
        target_out,
        index=False,
    )

    coverage.to_csv(
        coverage_out,
        index=False,
    )

    policy = pd.DataFrame(
        [
            {
                "policy_id": "P1",
                "rule":
                    "Official thresholds are current operational references, "
                    "not automatically retrospective historical labels.",
            },
            {
                "policy_id": "P2",
                "rule":
                    "No stage-flow conversion without an official "
                    "station-specific rating curve.",
            },
            {
                "policy_id": "P3",
                "rule":
                    "No imputation in the target-base layer.",
            },
            {
                "policy_id": "P4",
                "rule":
                    "Liguria uses daily maximum level from hourly source "
                    "where available.",
            },
            {
                "policy_id": "P5",
                "rule":
                    "Piemonte uses the canonical daily source value and does "
                    "not call it an instantaneous flood peak.",
            },
            {
                "policy_id": "P6",
                "rule":
                    "Any statistical extreme threshold used by the predictive "
                    "model must be estimated only inside each training fold.",
            },
            {
                "policy_id": "P7",
                "rule":
                    "LIG_CENTA remains a Neva-at-Cisano tributary proxy.",
            },
        ]
    )

    policy.to_csv(
        policy_out,
        index=False,
    )

    report = {
        "version": "1.0",
        "overall_status": overall,
        "receptors": receptors,
        "date_grid_days_per_receptor":
            int(len(date_index)),
        "expected_rows": int(expected_rows),
        "actual_rows": int(len(target_base)),
        "duplicate_receptor_date_rows": duplicates,
        "receptors_with_wrong_day_count":
            bad_day_counts,
        "numeric_target_values": numeric_total,
        "official_threshold_exceedance_labels_created":
            False,
        "statistical_thresholds_created": False,
        "stage_flow_conversion_performed": False,
        "imputation_performed": False,
        "official_threshold_temporal_scope":
            "CURRENT_OPERATIONAL_REFERENCE_ONLY",
        "next_step": (
            "Profile daily hydrological extreme distributions and define "
            "training-fold-only event-label rules, then build 24/48/72 h "
            "forecast labels without leakage."
        ),
    }

    audit_json.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    compact_cols = [
        "receptor_id",
        "station_name",
        "provider",
        "primary_role",
        "observed_variable_code",
        "target_observation_statistic",
        "expected_grid_days",
        "numeric_days",
        "coverage_fraction",
        "years_with_numeric_values",
    ]

    lines = [
        "=" * 196,
        "NW HYDROLOGY — PRIMARY DAILY HYDROLOGY TARGET BASE v1.0",
        "=" * 196,
        f"OVERALL STATUS                           : {overall}",
        f"Receptors                                : {receptors}",
        f"Grid days / receptor                     : {len(date_index)}",
        f"Expected rows                            : {expected_rows}",
        f"Actual rows                              : {len(target_base)}",
        f"Duplicate receptor-date rows             : {duplicates}",
        f"Numeric target values                    : {numeric_total}",
        "",
        "COVERAGE SUMMARY",
        coverage[compact_cols].to_string(index=False),
        "",
        "IMPORTANT",
        "The 584 retrieval candidates from the previous screening are NOT required to build this daily target base.",
        "Official thresholds are retained as operational reference metadata only.",
        "No retrospective official-threshold label is created.",
        "No statistical label is created globally; future statistical thresholds are training-fold-only.",
        "No stage-flow conversion and no imputation are performed.",
        "",
        f"Target base : {target_out}",
        f"Coverage    : {coverage_out}",
        f"Policy      : {policy_out}",
    ]

    audit_txt.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    progress(
        "PHASE 3/3",
        1,
        1,
        start3,
        "target base written",
    )

    print("\n" + "=" * 196)
    print("\n".join(lines[3:]))
    print("=" * 196)
    print(f"OVERALL STATUS : {overall}")
    print(f"Output         : {out_root}")
    print("=" * 196)


if __name__ == "__main__":
    main()
