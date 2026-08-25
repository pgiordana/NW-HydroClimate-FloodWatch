#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
update_nw_operational_antecedent_cache_current_v1_0.py

FASE 19 — CACHE OPERATIVA PERSISTENTE + LAG/ROLLING CANONICI.

OBIETTIVO
---------
Trasformare i run giornalieri indipendenti in una vera pipeline operativa
con memoria temporale, ricostruendo ESATTAMENTE le feature derivate ERA5
usate nel training storico.

La logica riproduce:
    build_era5_basin_features_derived_v1_0.py

Principi congelati:
- rolling/lag entro (receptor_id, season_year);
- SOLO settembre-dicembre;
- nessun bridging agosto -> settembre;
- nessun bridging dicembre -> settembre dell'anno successivo;
- prev* usa solo giorni precedenti;
- incl_today include il giorno corrente;
- se manca un run giornaliero intermedio, la finestra resta NaN;
- nessuna zero-imputation.

IMPORTANTE PER IL BETA
----------------------
Il fatto che il cache inizi il 25 agosto NON deve riempire artificiosamente
le finestre di inizio settembre.

Nel training:
- 1 settembre: prev1d / prev3d / prev7d / prev14d = NaN;
- 1-3 settembre: prev3d incompleto;
- 1-7 settembre: prev7d incompleto;
- 1-14 settembre: prev14d incompleto;
- precip_7d_incl_today diventa disponibile solo dal 7 settembre.

Questa missingness è STRUTTURALE E ATTESA dal modello.

INPUT CORRENTE
--------------
nw_operational_feature_snapshot/<RUN_ID>/
    operational_dynamic_features_v1_2.parquet
    operational_full_97_predictors_v1_2.parquet

CACHE PERSISTENTE
-----------------
nw_operational_daily_feature_cache_v1_0/
    operational_dynamic_daily_cache_v1_0.parquet
    operational_cache_manifest_v1_0.csv
    operational_cache_audit_v1_0.json
    operational_cache_audit_v1_0.txt

OUTPUT RUN CORRENTE
-------------------
nw_operational_feature_snapshot/<RUN_ID>/
    operational_dynamic_features_v1_3.parquet
    operational_full_97_predictors_v1_3.parquet
    operational_cache_derived_feature_registry_v1_0.csv
    operational_cache_overlay_audit_v1_0.json
    operational_cache_overlay_audit_v1_0.txt

NON esegue il modello.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_RECEPTORS = 20
EXPECTED_DYNAMIC = 83
EXPECTED_STATIC = 14
EXPECTED_TOTAL = 97
CORE_MONTHS = {9, 10, 11, 12}


# ---------------------------------------------------------------------------
# Frozen derived-feature mapping from historical builder
# ---------------------------------------------------------------------------

CHANGE_24H = {
    "era5__ivt_mag_change_24h":
        "era5__ivt_mag_mean_kg_m1_s1",
    "era5__tcwv_change_24h":
        "era5__tcwv_mean_kg_m2",
    "era5__cape_change_24h":
        "era5__cape_mean_j_kg",
    "era5__mslp_change_24h_pa":
        "era5__mslp_mean_pa",
}

LAG1 = {
    "era5__ivt_mag_prev1d":
        "era5__ivt_mag_mean_kg_m1_s1",

    "era5__precip_prev1d_mm":
        "era5__precip_sum_mm",

    "era5__precip_max1h_prev1d_mm":
        "era5__precip_max_1h_mm",

    "era5__soil_water_l1_m3_m3_prev1d":
        "era5__soil_water_l1_m3_m3",
    "era5__soil_water_l2_m3_m3_prev1d":
        "era5__soil_water_l2_m3_m3",
    "era5__soil_water_l3_m3_m3_prev1d":
        "era5__soil_water_l3_m3_m3",
    "era5__snow_depth_mwe_prev1d":
        "era5__snow_depth_mwe",
    "era5__soil_profile_mean_prev1d_m3_m3":
        "era5__soil_profile_mean_m3_m3",

    "era5__q925_prev1d_kg_kg":
        "era5__q925_mean_kg_kg",
    "era5__q850_prev1d_kg_kg":
        "era5__q850_mean_kg_kg",
    "era5__q700_prev1d_kg_kg":
        "era5__q700_mean_kg_kg",

    "era5__wind925_prev1d_m_s":
        "era5__wind925_mean_m_s",
    "era5__wind850_prev1d_m_s":
        "era5__wind850_mean_m_s",
    "era5__wind700_prev1d_m_s":
        "era5__wind700_mean_m_s",
}

PREV_ROLLING = {
    "era5__ivt_mag_prev3d_mean":
        ("era5__ivt_mag_mean_kg_m1_s1", 3, "mean"),
    "era5__ivt_mag_prev7d_mean":
        ("era5__ivt_mag_mean_kg_m1_s1", 7, "mean"),
    "era5__ivt_mag_prev3d_max":
        ("era5__ivt_mag_max_kg_m1_s1", 3, "max"),

    "era5__precip_prev3d_mm":
        ("era5__precip_sum_mm", 3, "sum"),
    "era5__precip_prev7d_mm":
        ("era5__precip_sum_mm", 7, "sum"),
    "era5__precip_prev14d_mm":
        ("era5__precip_sum_mm", 14, "sum"),

    "era5__precip_max1h_prev3d_max_mm":
        ("era5__precip_max_1h_mm", 3, "max"),
}

INCL_ROLLING = {
    "era5__precip_3d_incl_today_mm":
        ("era5__precip_sum_mm", 3, "sum"),
    "era5__precip_7d_incl_today_mm":
        ("era5__precip_sum_mm", 7, "sum"),
}

QWIND = {
    "era5__qwind925_proxy":
        ("era5__q925_mean_kg_kg", "era5__wind925_mean_m_s"),
    "era5__qwind850_proxy":
        ("era5__q850_mean_kg_kg", "era5__wind850_mean_m_s"),
    "era5__qwind700_proxy":
        ("era5__q700_mean_kg_kg", "era5__wind700_mean_m_s"),
}

SOIL_PROFILE = "era5__soil_profile_mean_m3_m3"

DERIVED_TARGETS = (
    list(CHANGE_24H)
    + list(LAG1)
    + list(PREV_ROLLING)
    + list(INCL_ROLLING)
    + list(QWIND)
)


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


def latest_v12_snapshot(root):
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
                / "operational_dynamic_features_v1_2.parquet"
            ).exists()
            and (
                p
                / "operational_full_97_predictors_v1_2.parquet"
            ).exists()
        ],
        key=lambda p: p.name,
    )

    if not runs:
        raise SystemExit(
            "Nessun operational snapshot v1.2 trovato."
        )

    return runs[-1]


def find_first_existing(paths, label):
    for p in paths:
        if p.exists():
            return p

    raise SystemExit(
        f"Manca {label}. Cercato in:\n"
        + "\n".join(str(p) for p in paths)
    )


def season_day(ts):
    d = pd.Timestamp(ts).normalize()
    sep1 = pd.Timestamp(
        year=d.year,
        month=9,
        day=1,
    )
    return int(
        (d - sep1).days + 1
    )


def exact_daily_grid_for_season(cache, receptor_ids, year, end_date):
    """
    Build a complete daily grid Sep 1 -> min(end_date, Dec 31).
    Missing execution days become explicit NaN rows, so rolling windows
    cannot jump over a missing run.
    """
    start = pd.Timestamp(
        year=year,
        month=9,
        day=1,
    )

    end = min(
        pd.Timestamp(end_date).normalize(),
        pd.Timestamp(
            year=year,
            month=12,
            day=31,
        ),
    )

    if end < start:
        return pd.DataFrame()

    dates = pd.date_range(
        start,
        end,
        freq="D",
    )

    grid = pd.MultiIndex.from_product(
        [
            sorted(receptor_ids),
            dates,
        ],
        names=[
            "receptor_id",
            "issue_date",
        ],
    ).to_frame(
        index=False
    )

    season = cache[
        cache["issue_date"]
        .dt.year
        .eq(year)
        & cache["issue_date"]
        .dt.month
        .isin(CORE_MONTHS)
    ].copy()

    merged = grid.merge(
        season,
        on=[
            "receptor_id",
            "issue_date",
        ],
        how="left",
        validate="one_to_one",
    )

    merged["season_year"] = year
    merged["season_day"] = (
        merged["issue_date"]
        - start
    ).dt.days + 1

    return merged


def derive_one_season(season, dynamic_names):
    """
    Replicates the historical derived-feature formulas on a complete
    receptor-day grid.
    """
    if len(season) == 0:
        return season

    s = season.sort_values(
        [
            "receptor_id",
            "issue_date",
        ]
    ).copy()

    grouped = s.groupby(
        [
            "receptor_id",
            "season_year",
        ],
        sort=False,
        group_keys=False,
    )

    # ------------------------------------------------------------------
    # Soil profile current state — historical builder uses simple mean
    # across L1/L2/L3, skipna=True by pandas default.
    # ------------------------------------------------------------------
    soil_cols = [
        "era5__soil_water_l1_m3_m3",
        "era5__soil_water_l2_m3_m3",
        "era5__soil_water_l3_m3_m3",
    ]

    if all(c in s.columns for c in soil_cols):
        s[SOIL_PROFILE] = (
            s[soil_cols]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .mean(
                axis=1
            )
        )

    # regroup after modifying current feature
    grouped = s.groupby(
        [
            "receptor_id",
            "season_year",
        ],
        sort=False,
        group_keys=False,
    )

    # ------------------------------------------------------------------
    # 24h changes = current - previous day
    # ------------------------------------------------------------------
    for out, source in CHANGE_24H.items():
        if (
            out in dynamic_names
            and source in s.columns
        ):
            current = pd.to_numeric(
                s[source],
                errors="coerce",
            )
            previous = grouped[source].shift(1)
            s[out] = current - previous

    # ------------------------------------------------------------------
    # 1-day lags
    # ------------------------------------------------------------------
    for out, source in LAG1.items():
        if (
            out in dynamic_names
            and source in s.columns
        ):
            s[out] = grouped[source].shift(1)

    # ------------------------------------------------------------------
    # previous-only rolling
    # ------------------------------------------------------------------
    for out, (source, window, how) in PREV_ROLLING.items():
        if (
            out not in dynamic_names
            or source not in s.columns
        ):
            continue

        def calc(series):
            shifted = series.shift(1)
            roll = shifted.rolling(
                window,
                min_periods=window,
            )

            if how == "sum":
                return roll.sum()
            if how == "mean":
                return roll.mean()
            if how == "max":
                return roll.max()

            raise ValueError(how)

        s[out] = grouped[source].transform(
            calc
        )

    # ------------------------------------------------------------------
    # rolling including today
    # ------------------------------------------------------------------
    for out, (source, window, how) in INCL_ROLLING.items():
        if (
            out not in dynamic_names
            or source not in s.columns
        ):
            continue

        def calc_incl(series):
            roll = series.rolling(
                window,
                min_periods=window,
            )

            if how == "sum":
                return roll.sum()

            raise ValueError(how)

        s[out] = grouped[source].transform(
            calc_incl
        )

    # ------------------------------------------------------------------
    # q * wind — same-day current proxy
    # ------------------------------------------------------------------
    for out, (qcol, wcol) in QWIND.items():
        if (
            out in dynamic_names
            and qcol in s.columns
            and wcol in s.columns
        ):
            s[out] = (
                pd.to_numeric(
                    s[qcol],
                    errors="coerce",
                )
                * pd.to_numeric(
                    s[wcol],
                    errors="coerce",
                )
            )

    # Calendar canonical.
    if "era5__season_day" in dynamic_names:
        s["era5__season_day"] = (
            s["season_day"]
            .astype(float)
        )

    return s


def main():
    root = Path(__file__).resolve().parent
    snapshot = latest_v12_snapshot(root)
    run_id = snapshot.name

    dynamic_p = (
        snapshot
        / "operational_dynamic_features_v1_2.parquet"
    )

    full_p = (
        snapshot
        / "operational_full_97_predictors_v1_2.parquet"
    )

    dictionary_p = find_first_existing(
        [
            root
            / "nw_hydroclimate_core_release_v1_0"
            / "metadata"
            / "nw_hydroclimate_master_predictor_dictionary_v1_0.csv",
            root
            / "nw_hydroclimate_foldwise_master_core_canonical_v1_0"
            / "nw_hydroclimate_master_predictor_dictionary_v1_0.csv",
        ],
        "predictor dictionary",
    )

    whitelist_p = find_first_existing(
        [
            root
            / "nw_hydroclimate_core_release_v1_0"
            / "metadata"
            / "primary_dynamic_feature_whitelist_canonical_v1_3.csv",
        ],
        "dynamic whitelist",
    )

    dynamic = pd.read_parquet(
        dynamic_p
    )

    full = pd.read_parquet(
        full_p
    )

    dictionary = pd.read_csv(
        dictionary_p,
        low_memory=False,
    )

    whitelist = pd.read_csv(
        whitelist_p,
        low_memory=False,
    )

    dynamic_names = (
        whitelist[
            "canonical_feature_name"
        ]
        .astype(str)
        .tolist()
    )

    predictor_order = (
        dictionary[
            "predictor"
        ]
        .astype(str)
        .tolist()
    )

    static_names = [
        p
        for p in predictor_order
        if p.startswith(
            "static__"
        )
    ]

    if len(dynamic_names) != EXPECTED_DYNAMIC:
        raise SystemExit(
            f"Dynamic whitelist={len(dynamic_names)}, expected=83"
        )

    if len(predictor_order) != EXPECTED_TOTAL:
        raise SystemExit(
            f"Predictor dictionary={len(predictor_order)}, expected=97"
        )

    if len(static_names) != EXPECTED_STATIC:
        raise SystemExit(
            f"Static predictors={len(static_names)}, expected=14"
        )

    if len(dynamic) != EXPECTED_RECEPTORS:
        raise SystemExit(
            f"Current dynamic rows={len(dynamic)}, expected=20"
        )

    issue_dates = pd.to_datetime(
        dynamic["issue_date"],
        errors="coerce",
    )

    if issue_dates.isna().any():
        raise SystemExit(
            "Current snapshot has invalid issue_date."
        )

    if issue_dates.nunique() != 1:
        raise SystemExit(
            "Current snapshot contains multiple issue dates."
        )

    issue_date = pd.Timestamp(
        issue_dates.iloc[0]
    ).normalize()

    in_core_season = (
        issue_date.month
        in CORE_MONTHS
    )

    print("=" * 220)
    print("NW HYDROCLIMATE — OPERATIONAL ANTECEDENT CACHE v1.0")
    print("=" * 220)
    print(f"Run ID         : {run_id}")
    print(f"Issue date     : {issue_date.date()}")
    print(f"In CORE season : {in_core_season}")

    # ------------------------------------------------------------------
    # PHASE 1/5 — validate derived target names against whitelist
    # ------------------------------------------------------------------
    print(
        "\nPHASE 1/5 — validate historical-derived feature mapping against frozen whitelist"
    )
    start = time.time()

    relevant_targets = [
        f
        for f in DERIVED_TARGETS
        if f in dynamic_names
    ]

    unknown_targets = [
        f
        for f in DERIVED_TARGETS
        if f not in dynamic_names
    ]

    # Unknown here is not necessarily an error: some completeness flags were
    # excluded from the CORE. We only derive targets present in the whitelist.
    progress(
        "PHASE 1/5",
        1,
        1,
        start,
        (
            f"canonical derived targets present={len(relevant_targets)} "
            f"| mapping-only extras={len(unknown_targets)}"
        ),
    )

    # ------------------------------------------------------------------
    # PHASE 2/5 — append/replace current day in persistent cache
    # ------------------------------------------------------------------
    print(
        "\nPHASE 2/5 — append current receptor-day snapshot to persistent cache"
    )
    start = time.time()

    cache_root = (
        root
        / "nw_operational_daily_feature_cache_v1_0"
    )

    cache_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_p = (
        cache_root
        / "operational_dynamic_daily_cache_v1_0.parquet"
    )

    # Cache all 83 dynamic columns. Derived values may later be overwritten
    # by the exact local rolling engine, but keeping them preserves provenance.
    current_cache = dynamic[
        [
            "receptor_id",
            "issue_date",
            "run_id",
            *dynamic_names,
        ]
    ].copy()

    current_cache["issue_date"] = pd.to_datetime(
        current_cache["issue_date"],
        errors="raise",
    ).dt.normalize()

    if cache_p.exists():
        cache = pd.read_parquet(
            cache_p
        )

        cache["issue_date"] = pd.to_datetime(
            cache["issue_date"],
            errors="raise",
        ).dt.normalize()

        # Align schema to current canonical 83 dynamic features.
        for c in [
            "receptor_id",
            "issue_date",
            "run_id",
            *dynamic_names,
        ]:
            if c not in cache.columns:
                cache[c] = np.nan

        cache = cache[
            [
                "receptor_id",
                "issue_date",
                "run_id",
                *dynamic_names,
            ]
        ]

        cache = cache[
            ~(
                cache["issue_date"].eq(
                    issue_date
                )
                & cache["receptor_id"]
                .astype(str)
                .isin(
                    current_cache[
                        "receptor_id"
                    ].astype(str)
                )
            )
        ].copy()

        cache = pd.concat(
            [
                cache,
                current_cache,
            ],
            ignore_index=True,
        )

    else:
        cache = current_cache.copy()

    cache["receptor_id"] = (
        cache["receptor_id"]
        .astype(str)
    )

    cache = cache.sort_values(
        [
            "issue_date",
            "receptor_id",
        ]
    ).reset_index(
        drop=True
    )

    dup = int(
        cache.duplicated(
            [
                "receptor_id",
                "issue_date",
            ]
        ).sum()
    )

    if dup:
        raise SystemExit(
            f"Persistent cache duplicate keys={dup}"
        )

    cache.to_parquet(
        cache_p,
        index=False,
    )

    progress(
        "PHASE 2/5",
        1,
        1,
        start,
        (
            f"cache rows={len(cache)} "
            f"| dates={cache['issue_date'].nunique()} "
            f"| receptors={cache['receptor_id'].nunique()}"
        ),
    )

    # ------------------------------------------------------------------
    # PHASE 3/5 — exact rolling derivation within Sep-Dec season only
    # ------------------------------------------------------------------
    print(
        "\nPHASE 3/5 — reproduce historical lag/rolling formulas with strict season boundaries"
    )
    start = time.time()

    overlay_dynamic = dynamic.copy()

    derived_registry_rows = []

    if in_core_season:
        receptor_ids = sorted(
            dynamic[
                "receptor_id"
            ].astype(str)
        )

        season_grid = exact_daily_grid_for_season(
            cache=cache,
            receptor_ids=receptor_ids,
            year=issue_date.year,
            end_date=issue_date,
        )

        derived = derive_one_season(
            season_grid,
            dynamic_names,
        )

        today = derived[
            derived["issue_date"]
            .eq(issue_date)
        ].copy()

        if len(today) != EXPECTED_RECEPTORS:
            raise SystemExit(
                f"Derived current rows={len(today)}, expected=20"
            )

        today = (
            today.set_index(
                "receptor_id"
            )
        )

        overlay_dynamic = (
            overlay_dynamic.set_index(
                "receptor_id"
            )
        )

        for feature in relevant_targets:
            if feature not in today.columns:
                continue

            before = pd.to_numeric(
                overlay_dynamic[
                    feature
                ],
                errors="coerce",
            )

            new = pd.to_numeric(
                today[
                    feature
                ],
                errors="coerce",
            )

            overlay_dynamic[
                feature
            ] = new

            derived_registry_rows.append(
                {
                    "feature": feature,
                    "current_nonmissing_before":
                        int(
                            before.notna().sum()
                        ),
                    "current_nonmissing_after":
                        int(
                            new.notna().sum()
                        ),
                    "overwritten_with_exact_cache_formula":
                        True,
                    "season_boundary_policy":
                        "STRICT_SEP_DEC_NO_AUGUST_BRIDGE",
                }
            )

        overlay_dynamic = (
            overlay_dynamic.reset_index()
        )

        # Explicit canonical season-day.
        if "era5__season_day" in overlay_dynamic.columns:
            overlay_dynamic[
                "era5__season_day"
            ] = float(
                season_day(
                    issue_date
                )
            )

        canonical_status = (
            "IN_SEASON_CANONICAL_ROLLING_APPLIED"
        )

    else:
        for feature in relevant_targets:
            derived_registry_rows.append(
                {
                    "feature": feature,
                    "current_nonmissing_before":
                        int(
                            pd.to_numeric(
                                overlay_dynamic[
                                    feature
                                ],
                                errors="coerce",
                            )
                            .notna()
                            .sum()
                        ),
                    "current_nonmissing_after":
                        int(
                            pd.to_numeric(
                                overlay_dynamic[
                                    feature
                                ],
                                errors="coerce",
                            )
                            .notna()
                            .sum()
                        ),
                    "overwritten_with_exact_cache_formula":
                        False,
                    "season_boundary_policy":
                        "OUT_OF_SEASON_NO_CANONICAL_ROLLING",
                }
            )

        canonical_status = (
            "OUT_OF_SEASON_ARCHIVED_ONLY"
        )

    registry = pd.DataFrame(
        derived_registry_rows
    )

    progress(
        "PHASE 3/5",
        1,
        1,
        start,
        canonical_status,
    )

    # ------------------------------------------------------------------
    # PHASE 4/5 — rebuild exact 97-predictor row with static block unchanged
    # ------------------------------------------------------------------
    print(
        "\nPHASE 4/5 — rebuild current 97-predictor snapshot with cache-derived overlay"
    )
    start = time.time()

    static_block = full[
        [
            "receptor_id",
            *static_names,
        ]
    ].copy()

    full_v13 = overlay_dynamic.merge(
        static_block,
        on="receptor_id",
        how="left",
        validate="one_to_one",
    )

    for p in predictor_order:
        if p not in full_v13.columns:
            full_v13[p] = np.nan

    full_v13 = full_v13[
        [
            "receptor_id",
            "issue_date",
            "run_id",
            *predictor_order,
        ]
    ].copy()

    dynamic_v13 = overlay_dynamic[
        [
            "receptor_id",
            "issue_date",
            "run_id",
            *dynamic_names,
        ]
    ].copy()

    static_missing = int(
        full_v13[
            static_names
        ].isna().sum().sum()
    )

    if static_missing:
        raise SystemExit(
            f"Static missing after cache overlay={static_missing}"
        )

    dynamic_complete = int(
        sum(
            dynamic_v13[
                c
            ].notna().all()
            for c in dynamic_names
        )
    )

    dynamic_zero = int(
        sum(
            dynamic_v13[
                c
            ].isna().all()
            for c in dynamic_names
        )
    )

    dynamic_v13_p = (
        snapshot
        / "operational_dynamic_features_v1_3.parquet"
    )

    full_v13_p = (
        snapshot
        / "operational_full_97_predictors_v1_3.parquet"
    )

    registry_p = (
        snapshot
        / "operational_cache_derived_feature_registry_v1_0.csv"
    )

    dynamic_v13.to_parquet(
        dynamic_v13_p,
        index=False,
    )

    full_v13.to_parquet(
        full_v13_p,
        index=False,
    )

    registry.to_csv(
        registry_p,
        index=False,
    )

    progress(
        "PHASE 4/5",
        1,
        1,
        start,
        (
            f"rows={len(full_v13)} predictors=97 "
            f"| dynamic complete={dynamic_complete}/83 "
            f"| zero={dynamic_zero}/83"
        ),
    )

    # ------------------------------------------------------------------
    # PHASE 5/5 — persistent cache manifest/audit
    # ------------------------------------------------------------------
    print(
        "\nPHASE 5/5 — freeze cache manifest and season-edge audit"
    )
    start = time.time()

    cache_dates = (
        cache[
            "issue_date"
        ]
        .drop_duplicates()
        .sort_values()
    )

    manifest = pd.DataFrame(
        [
            {
                "cache_version":
                    "v1.0",
                "cache_rows":
                    int(
                        len(cache)
                    ),
                "unique_dates":
                    int(
                        cache["issue_date"]
                        .nunique()
                    ),
                "receptors":
                    int(
                        cache["receptor_id"]
                        .nunique()
                    ),
                "date_min":
                    (
                        cache_dates.min().date().isoformat()
                        if len(cache_dates)
                        else ""
                    ),
                "date_max":
                    (
                        cache_dates.max().date().isoformat()
                        if len(cache_dates)
                        else ""
                    ),
                "latest_run_id":
                    run_id,
                "latest_issue_date":
                    issue_date.date().isoformat(),
                "latest_in_core_season":
                    in_core_season,
                "season_boundary_policy":
                    "NO_AUG_SEP_BRIDGE__NO_DEC_NEXT_SEP_BRIDGE",
            }
        ]
    )

    manifest_p = (
        cache_root
        / "operational_cache_manifest_v1_0.csv"
    )

    cache_audit_json_p = (
        cache_root
        / "operational_cache_audit_v1_0.json"
    )

    cache_audit_txt_p = (
        cache_root
        / "operational_cache_audit_v1_0.txt"
    )

    overlay_audit_json_p = (
        snapshot
        / "operational_cache_overlay_audit_v1_0.json"
    )

    overlay_audit_txt_p = (
        snapshot
        / "operational_cache_overlay_audit_v1_0.txt"
    )

    manifest.to_csv(
        manifest_p,
        index=False,
    )

    # Current season warm-up state.
    if in_core_season:
        sd = season_day(
            issue_date
        )

        expected_states = {
            "prev1d_expected_available":
                sd >= 2,
            "prev3d_expected_available":
                sd >= 4,
            "prev7d_expected_available":
                sd >= 8,
            "prev14d_expected_available":
                sd >= 15,
            "precip_3d_incl_today_expected_available":
                sd >= 3,
            "precip_7d_incl_today_expected_available":
                sd >= 7,
        }
    else:
        sd = None
        expected_states = {
            "prev1d_expected_available":
                False,
            "prev3d_expected_available":
                False,
            "prev7d_expected_available":
                False,
            "prev14d_expected_available":
                False,
            "precip_3d_incl_today_expected_available":
                False,
            "precip_7d_incl_today_expected_available":
                False,
        }

    cache_audit = {
        "version": "1.0",
        "overall_status":
            "PASS_OPERATIONAL_DAILY_CACHE_UPDATED",
        "latest_run_id":
            run_id,
        "latest_issue_date":
            issue_date.date().isoformat(),
        "cache_rows":
            int(
                len(cache)
            ),
        "cache_unique_dates":
            int(
                cache["issue_date"]
                .nunique()
            ),
        "cache_receptors":
            int(
                cache["receptor_id"]
                .nunique()
            ),
        "duplicate_receptor_date_keys":
            dup,
        "core_season_months":
            [9, 10, 11, 12],
        "august_to_september_bridging":
            False,
        "december_to_next_september_bridging":
            False,
        "missing_execution_day_is_explicit_nan":
            True,
        "zero_imputation_used":
            False,
        "historical_formula_source":
            "build_era5_basin_features_derived_v1_0.py",
    }

    cache_audit_json_p.write_text(
        json.dumps(
            cache_audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    overlay_audit = {
        "version": "1.0",
        "overall_status": (
            "PASS_CACHE_OVERLAY__CANONICAL_IN_SEASON"
            if in_core_season
            else "PASS_CACHE_ARCHIVE__OUT_OF_SEASON"
        ),
        "run_id":
            run_id,
        "issue_date":
            issue_date.date().isoformat(),
        "in_core_season":
            in_core_season,
        "season_day":
            sd,
        "canonical_rolling_status":
            canonical_status,
        "dynamic_features_complete_all_receptors":
            dynamic_complete,
        "dynamic_features_zero_coverage":
            dynamic_zero,
        "static_missing_cells":
            static_missing,
        "expected_structural_warmup":
            expected_states,
        "zero_imputation_used":
            False,
        "model_prediction_performed":
            False,
        "next_step":
            (
                "Integrate this cache updater into the single daily runner. "
                "During Sep-Dec, compatibility gating must treat early-season "
                "structural NaNs as expected rather than as operational failure."
            ),
    }

    overlay_audit_json_p.write_text(
        json.dumps(
            overlay_audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    cache_lines = [
        "=" * 170,
        "NW HYDROCLIMATE — PERSISTENT OPERATIONAL DAILY CACHE v1.0",
        "=" * 170,
        "OVERALL STATUS : PASS_OPERATIONAL_DAILY_CACHE_UPDATED",
        f"Latest run      : {run_id}",
        f"Latest date     : {issue_date.date()}",
        f"Cache rows      : {len(cache)}",
        f"Unique dates    : {cache['issue_date'].nunique()}",
        f"Receptors       : {cache['receptor_id'].nunique()}",
        "Aug->Sep bridge : False",
        "Dec->Sep bridge : False",
        "Zero imputation : False",
        "",
        f"Cache    : {cache_p}",
        f"Manifest : {manifest_p}",
    ]

    cache_audit_txt_p.write_text(
        "\n".join(
            cache_lines
        ) + "\n",
        encoding="utf-8",
    )

    overlay_lines = [
        "=" * 190,
        "NW HYDROCLIMATE — CURRENT CACHE-DERIVED OVERLAY v1.0",
        "=" * 190,
        f"OVERALL STATUS                    : {overlay_audit['overall_status']}",
        f"Run ID                            : {run_id}",
        f"Issue date                        : {issue_date.date()}",
        f"In CORE season                    : {in_core_season}",
        f"Season day                        : {sd}",
        f"Canonical rolling status          : {canonical_status}",
        f"Dynamic complete all receptors    : {dynamic_complete}/83",
        f"Dynamic zero coverage             : {dynamic_zero}/83",
        f"Static missing                    : {static_missing}",
        "August -> September bridging     : False",
        "December -> next September bridge: False",
        "Zero imputation                  : False",
        "",
        "EXPECTED STRUCTURAL WARM-UP",
        json.dumps(
            expected_states,
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "DERIVED FEATURE REGISTRY",
        registry.to_string(
            index=False
        ),
        "",
        "IMPORTANT",
        "Early-September NaNs are intentionally preserved exactly as in historical training.",
        "August cache rows are archival/technical only and are never used to fill September antecedent windows.",
        "A missed daily execution creates a genuine gap; rolling windows do not jump over it.",
        "",
        f"Dynamic v1.3 : {dynamic_v13_p}",
        f"Full 97 v1.3 : {full_v13_p}",
        f"Registry     : {registry_p}",
        f"Audit        : {overlay_audit_json_p}",
    ]

    overlay_audit_txt_p.write_text(
        "\n".join(
            overlay_lines
        ) + "\n",
        encoding="utf-8",
    )

    progress(
        "PHASE 5/5",
        1,
        1,
        start,
        f"status={overlay_audit['overall_status']}",
    )

    print("\n" + "=" * 220)
    print(
        f"OVERALL STATUS : {overlay_audit['overall_status']}"
    )
    print(
        f"Persistent cache status : {cache_audit['overall_status']}"
    )
    print(
        f"Run ID                  : {run_id}"
    )
    print(
        f"Issue date              : {issue_date.date()}"
    )
    print(
        f"In CORE season          : {in_core_season}"
    )
    print(
        f"Season day              : {sd}"
    )
    print(
        f"Cache unique dates      : {cache['issue_date'].nunique()}"
    )
    print(
        f"Dynamic complete        : {dynamic_complete}/83"
    )
    print(
        f"Dynamic zero coverage   : {dynamic_zero}/83"
    )
    print(
        "Aug->Sep bridging       : False"
    )
    print(
        "Dec->Sep bridging       : False"
    )
    print(
        "Zero imputation         : False"
    )
    print()
    print(
        f"Cache       : {cache_p}"
    )
    print(
        f"Dynamic v1.3: {dynamic_v13_p}"
    )
    print(
        f"Full 97 v1.3: {full_v13_p}"
    )
    print(
        f"Audit        : {overlay_audit_json_p}"
    )
    print("=" * 220)


if __name__ == "__main__":
    main()
