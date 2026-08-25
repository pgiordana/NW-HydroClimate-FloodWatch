#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audit_nw_operational_beta_gate_current_v1_0.py

FASE 20 — GATE CANONICO PER LA BETA PROSPETTICA.

OBIETTIVO
---------
Distinguere rigorosamente:

1) NaN STRUTTURALI ATTESI dal training:
   - inizio stagione Sep-Dec;
   - no-support fisico MedSea;

2) NaN OPERATIVI INATTESI:
   - downloader/provider fallito;
   - cache interrotta;
   - feature che dovrebbe essere disponibile ma non lo è;

3) drift/range review:
   - valore operativo fuori dal supporto storico F3-FIT;
   - NON viene automaticamente classificato come errore meteorologico.

Il gate NON pretende che IFS/CMEMS siano già dimostrati distributivamente
equivalenti a ERA5/MedSea. Quella verifica richiede shadow verification
prospettica.

POLITICA BETA
-------------
La beta prospettica può partire in Sep-Dec se:

- schema 97 predictor esatto;
- statiche identiche al training;
- nessun NaN operativo inatteso;
- tutti i NaN presenti sono:
    a) strutturali di inizio stagione, oppure
    b) conditional MedSea no-support;
- modelli restano congelati.

Quindi, per esempio, al 1 settembre:
- prev1d è correttamente NaN;
- prev3d/7d/14d sono correttamente NaN;
- precip_3d_incl_today è NaN fino al 3 settembre;
- precip_7d_incl_today è NaN fino al 7 settembre.

Questi NaN NON devono bloccare la beta perché il modello è stato addestrato
con la stessa struttura di missingness stagionale.

FUORI STAGIONE
--------------
Il 25 agosto il gate deve restituire:
    PASS_BETA_GATE_LOGIC__OUT_OF_SEASON_BLOCKED

e:
    prospective_beta_allowed = False

OUTPUT
------
nw_operational_feature_snapshot/<RUN_ID>/
  operational_beta_gate_cells_v1_0.csv
  operational_beta_gate_features_v1_0.csv
  operational_beta_gate_p1_v1_0.csv
  operational_beta_gate_audit_v1_0.json
  operational_beta_gate_audit_v1_0.txt

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
# Exact season-edge rules inherited from historical derived builder
# ---------------------------------------------------------------------------

CHANGE_24H = {
    "era5__ivt_mag_change_24h",
    "era5__tcwv_change_24h",
    "era5__cape_change_24h",
    "era5__mslp_change_24h_pa",
}

LAG1 = {
    "era5__ivt_mag_prev1d",
    "era5__precip_prev1d_mm",
    "era5__precip_max1h_prev1d_mm",
    "era5__soil_water_l1_m3_m3_prev1d",
    "era5__soil_water_l2_m3_m3_prev1d",
    "era5__soil_water_l3_m3_m3_prev1d",
    "era5__snow_depth_mwe_prev1d",
    "era5__soil_profile_mean_prev1d_m3_m3",
    "era5__q925_prev1d_kg_kg",
    "era5__q850_prev1d_kg_kg",
    "era5__q700_prev1d_kg_kg",
    "era5__wind925_prev1d_m_s",
    "era5__wind850_prev1d_m_s",
    "era5__wind700_prev1d_m_s",
}

PREV3D = {
    "era5__ivt_mag_prev3d_mean",
    "era5__ivt_mag_prev3d_max",
    "era5__precip_prev3d_mm",
    "era5__precip_max1h_prev3d_max_mm",
}

PREV7D = {
    "era5__ivt_mag_prev7d_mean",
    "era5__precip_prev7d_mm",
}

PREV14D = {
    "era5__precip_prev14d_mm",
}

INCL3D = {
    "era5__precip_3d_incl_today_mm",
}

INCL7D = {
    "era5__precip_7d_incl_today_mm",
}

# Historical conditional MedSea physical variables: NaN is a valid physical /
# methodological state when there is no baseline marine support.
MEDSEA_CONDITIONAL = {
    "medsea_ivt__medsea_sst_anom_corridor_c",
    "medsea_ivt__medsea_ohc_0_100_corridor_j_m2",
    "medsea_ivt__medsea_ohc_anom_corridor_j_m2",
    "medsea_ivt__medsea_tmean_0_100_corridor_c",
    "medsea_ivt__medsea_tmean_anom_0_100_corridor_c",
    "medsea_ivt__sst_anom_x_ivt_proxy",
    "medsea_ivt__ohc_anom_x_ivt_proxy",
}

MEDSEA_SUPPORT_BASELINE = (
    "medsea_ivt__medsea_support_baseline"
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


def latest_v13_snapshot(root):
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
                / "operational_full_97_predictors_v1_3.parquet"
            ).exists()
            and (
                p
                / "operational_cache_overlay_audit_v1_0.json"
            ).exists()
        ],
        key=lambda p: p.name,
    )

    if not runs:
        raise SystemExit(
            "Nessun operational snapshot v1.3 trovato."
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


def season_day(issue_date):
    d = pd.Timestamp(issue_date).normalize()
    sep1 = pd.Timestamp(
        year=d.year,
        month=9,
        day=1,
    )
    return int(
        (d - sep1).days + 1
    )


def historical_edge_reason(feature, sd):
    if feature in CHANGE_24H and sd < 2:
        return "EXPECTED_SEASON_START_CHANGE24H"

    if feature in LAG1 and sd < 2:
        return "EXPECTED_SEASON_START_PREV1D"

    if feature in PREV3D and sd < 4:
        return "EXPECTED_SEASON_START_PREV3D"

    if feature in PREV7D and sd < 8:
        return "EXPECTED_SEASON_START_PREV7D"

    if feature in PREV14D and sd < 15:
        return "EXPECTED_SEASON_START_PREV14D"

    if feature in INCL3D and sd < 3:
        return "EXPECTED_SEASON_START_INCL3D"

    if feature in INCL7D and sd < 7:
        return "EXPECTED_SEASON_START_INCL7D"

    return ""


def fit_range(values):
    x = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).dropna().to_numpy(dtype=float)

    if len(x) == 0:
        return None

    return {
        "n": int(len(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "sorted": np.sort(x),
    }


def empirical_percentile(sorted_values, x):
    if (
        sorted_values is None
        or len(sorted_values) == 0
        or not np.isfinite(x)
    ):
        return np.nan

    left = np.searchsorted(
        sorted_values,
        x,
        side="left",
    )
    right = np.searchsorted(
        sorted_values,
        x,
        side="right",
    )

    return float(
        (left + right)
        / (2.0 * len(sorted_values))
    )


def main():
    root = Path(__file__).resolve().parent
    snapshot = latest_v13_snapshot(root)
    run_id = snapshot.name

    full_p = (
        snapshot
        / "operational_full_97_predictors_v1_3.parquet"
    )

    cache_overlay_p = (
        snapshot
        / "operational_cache_overlay_audit_v1_0.json"
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

    master_p = find_first_existing(
        [
            root
            / "nw_hydroclimate_foldwise_master_core_canonical_v1_0"
            / "nw_hydroclimate_foldwise_master_core_v1_0.parquet",
        ],
        "canonical master",
    )

    priority_p = find_first_existing(
        [
            root
            / "nw_operational_feature_equivalence_preflight_v1_0"
            / "operational_priority_features_v1_0.csv",
        ],
        "operational priority registry",
    )

    operational = pd.read_parquet(
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

    priority_df = pd.read_csv(
        priority_p,
        low_memory=False,
    )

    cache_overlay = json.loads(
        cache_overlay_p.read_text(
            encoding="utf-8"
        )
    )

    predictor_order = (
        dictionary["predictor"]
        .astype(str)
        .tolist()
    )

    dynamic_names = (
        whitelist["canonical_feature_name"]
        .astype(str)
        .tolist()
    )

    static_names = [
        p
        for p in predictor_order
        if p.startswith("static__")
    ]

    if len(operational) != EXPECTED_RECEPTORS:
        raise SystemExit(
            f"Operational rows={len(operational)}, expected=20"
        )

    actual_predictors = [
        c
        for c in operational.columns
        if c not in {
            "receptor_id",
            "issue_date",
            "run_id",
        }
    ]

    exact_order = (
        actual_predictors
        == predictor_order
    )

    if not exact_order:
        raise SystemExit(
            "Operational predictor order differs from frozen dictionary."
        )

    issue_date = pd.to_datetime(
        operational["issue_date"].iloc[0]
    ).normalize()

    in_core_season = (
        issue_date.month
        in CORE_MONTHS
    )

    sd = (
        season_day(issue_date)
        if in_core_season
        else None
    )

    priority = dict(
        zip(
            priority_df[
                "canonical_feature_name"
            ].astype(str),
            priority_df[
                "operational_priority"
            ].astype(str),
        )
    )

    p1_names = sorted(
        set(
            priority_df.loc[
                priority_df[
                    "operational_priority"
                ]
                .astype(str)
                .eq(
                    "P1_TOP10_ANY_HORIZON"
                ),
                "canonical_feature_name",
            ]
            .astype(str)
        )
    )

    print("=" * 220)
    print("NW HYDROCLIMATE — PROSPECTIVE BETA GATE v1.0")
    print("=" * 220)
    print(f"Run ID         : {run_id}")
    print(f"Issue date     : {issue_date.date()}")
    print(f"In CORE season : {in_core_season}")
    print(f"Season day     : {sd}")

    # ------------------------------------------------------------------
    # PHASE 1/5 — static identity against frozen F3 FIT
    # ------------------------------------------------------------------
    print(
        "\nPHASE 1/5 — verify 97-schema and static identity against F3 FIT"
    )
    start = time.time()

    needed = [
        "fold_id",
        "partition",
        "receptor_id",
        *static_names,
        *dynamic_names,
    ]

    master = pd.read_parquet(
        master_p,
        columns=needed,
    )

    fit = master[
        master["fold_id"]
        .astype(str)
        .eq("F3")
        & master["partition"]
        .astype(str)
        .eq("FIT")
    ].copy()

    static_failures = []

    for rid in operational[
        "receptor_id"
    ].astype(str):
        op = operational[
            operational["receptor_id"]
            .astype(str)
            .eq(rid)
        ].iloc[0]

        hist = fit[
            fit["receptor_id"]
            .astype(str)
            .eq(rid)
        ]

        for feature in static_names:
            hv = (
                pd.to_numeric(
                    hist[feature],
                    errors="coerce",
                )
                .dropna()
                .unique()
            )

            if len(hv) != 1:
                static_failures.append(
                    f"{rid}:{feature}:historical_unique={len(hv)}"
                )
                continue

            ov = float(
                pd.to_numeric(
                    pd.Series(
                        [op[feature]]
                    ),
                    errors="coerce",
                ).iloc[0]
            )

            if not np.isclose(
                ov,
                float(hv[0]),
                atol=1e-10,
                rtol=1e-10,
            ):
                static_failures.append(
                    f"{rid}:{feature}:mismatch"
                )

    static_pass = (
        len(static_failures)
        == 0
    )

    progress(
        "PHASE 1/5",
        1,
        1,
        start,
        (
            f"exact_order={exact_order} "
            f"| static_identity={static_pass}"
        ),
    )

    # ------------------------------------------------------------------
    # PHASE 2/5 — classify every current dynamic missing cell
    # ------------------------------------------------------------------
    print(
        "\nPHASE 2/5 — classify expected structural vs unexpected operational NaN"
    )
    start = time.time()

    cell_rows = []

    total = (
        len(dynamic_names)
        * EXPECTED_RECEPTORS
    )
    done = 0

    for _, r in operational.iterrows():
        rid = str(
            r["receptor_id"]
        )

        support_baseline = pd.to_numeric(
            pd.Series(
                [
                    r.get(
                        MEDSEA_SUPPORT_BASELINE,
                        np.nan,
                    )
                ]
            ),
            errors="coerce",
        ).iloc[0]

        for feature in dynamic_names:
            done += 1

            value = pd.to_numeric(
                pd.Series(
                    [r[feature]]
                ),
                errors="coerce",
            ).iloc[0]

            missing = bool(
                pd.isna(value)
            )

            reason = ""
            missing_class = "PRESENT"

            if missing:
                if not in_core_season:
                    missing_class = (
                        "EXPECTED_OUT_OF_SEASON"
                    )
                    reason = (
                        "CORE_APPLIES_SEP_DEC_ONLY"
                    )

                else:
                    edge = historical_edge_reason(
                        feature,
                        sd,
                    )

                    if edge:
                        missing_class = (
                            "EXPECTED_STRUCTURAL_SEASON_EDGE"
                        )
                        reason = edge

                    elif (
                        feature
                        in MEDSEA_CONDITIONAL
                        and np.isfinite(
                            support_baseline
                        )
                        and float(
                            support_baseline
                        )
                        <= 0.0
                    ):
                        missing_class = (
                            "EXPECTED_CONDITIONAL_MEDSEA_NO_SUPPORT"
                        )
                        reason = (
                            "BASELINE_MEDSEA_SUPPORT_FALSE"
                        )

                    else:
                        missing_class = (
                            "UNEXPECTED_OPERATIONAL_MISSING"
                        )
                        reason = (
                            "NOT_EXPLAINED_BY_FROZEN_MISSINGNESS_POLICY"
                        )

            cell_rows.append(
                {
                    "receptor_id":
                        rid,
                    "feature":
                        feature,
                    "priority":
                        priority.get(
                            feature,
                            "P3_OTHER_CORE",
                        ),
                    "value":
                        value,
                    "is_missing":
                        missing,
                    "missing_class":
                        missing_class,
                    "missing_reason":
                        reason,
                }
            )

            progress(
                "PHASE 2/5",
                done,
                total,
                start,
                (
                    f"{rid} | {feature} "
                    f"| {missing_class}"
                ),
            )

    cells = pd.DataFrame(
        cell_rows
    )

    unexpected_cells = int(
        cells[
            "missing_class"
        ]
        .eq(
            "UNEXPECTED_OPERATIONAL_MISSING"
        )
        .sum()
    )

    unexpected_p1_cells = int(
        (
            cells[
                "missing_class"
            ]
            .eq(
                "UNEXPECTED_OPERATIONAL_MISSING"
            )
            & cells[
                "priority"
            ]
            .eq(
                "P1_TOP10_ANY_HORIZON"
            )
        ).sum()
    )

    # ------------------------------------------------------------------
    # PHASE 3/5 — P1/P2 F3-FIT range review
    # ------------------------------------------------------------------
    print(
        "\nPHASE 3/5 — P1/P2 range review against actual frozen training support"
    )
    start = time.time()

    review_features = sorted(
        set(
            priority_df[
                "canonical_feature_name"
            ].astype(str)
        )
    )

    range_rows = []

    total = (
        len(review_features)
        * EXPECTED_RECEPTORS
    )
    done = 0

    for feature in review_features:
        if feature not in operational.columns:
            continue

        for rid in operational[
            "receptor_id"
        ].astype(str):
            done += 1

            ov = pd.to_numeric(
                operational.loc[
                    operational[
                        "receptor_id"
                    ]
                    .astype(str)
                    .eq(rid),
                    feature,
                ],
                errors="coerce",
            ).iloc[0]

            hist = pd.to_numeric(
                fit.loc[
                    fit[
                        "receptor_id"
                    ]
                    .astype(str)
                    .eq(rid),
                    feature,
                ],
                errors="coerce",
            ).dropna().to_numpy(
                dtype=float
            )

            if (
                pd.isna(ov)
                or len(hist) == 0
            ):
                range_state = (
                    "NOT_EVALUABLE"
                )
                pct = np.nan
                hmin = (
                    float(
                        np.min(hist)
                    )
                    if len(hist)
                    else np.nan
                )
                hmax = (
                    float(
                        np.max(hist)
                    )
                    if len(hist)
                    else np.nan
                )
            else:
                hmin = float(
                    np.min(hist)
                )
                hmax = float(
                    np.max(hist)
                )

                sorted_hist = np.sort(
                    hist
                )

                pct = empirical_percentile(
                    sorted_hist,
                    float(ov),
                )

                if ov < hmin:
                    range_state = (
                        "BELOW_F3_FIT_MIN"
                    )
                elif ov > hmax:
                    range_state = (
                        "ABOVE_F3_FIT_MAX"
                    )
                elif (
                    pct < 0.005
                    or pct > 0.995
                ):
                    range_state = (
                        "WITHIN_RANGE_EXTREME_TAIL"
                    )
                else:
                    range_state = (
                        "WITHIN_F3_FIT_RANGE"
                    )

            range_rows.append(
                {
                    "receptor_id":
                        rid,
                    "feature":
                        feature,
                    "priority":
                        priority.get(
                            feature,
                            "P3_OTHER_CORE",
                        ),
                    "operational_value":
                        ov,
                    "f3_fit_min":
                        hmin,
                    "f3_fit_max":
                        hmax,
                    "empirical_percentile":
                        pct,
                    "range_state":
                        range_state,
                }
            )

            progress(
                "PHASE 3/5",
                done,
                total,
                start,
                f"{feature} | {rid} | {range_state}",
            )

    range_df = pd.DataFrame(
        range_rows
    )

    # ------------------------------------------------------------------
    # PHASE 4/5 — summarize feature/P1 gate
    # ------------------------------------------------------------------
    print(
        "\nPHASE 4/5 — summarize beta gate by feature and P1 critical set"
    )
    start = time.time()

    feature_rows = []

    for feature in dynamic_names:
        x = cells[
            cells[
                "feature"
            ].eq(feature)
        ]

        xr = range_df[
            range_df[
                "feature"
            ].eq(feature)
        ]

        feature_rows.append(
            {
                "feature":
                    feature,
                "priority":
                    priority.get(
                        feature,
                        "P3_OTHER_CORE",
                    ),
                "present_receptors":
                    int(
                        x[
                            "missing_class"
                        ]
                        .eq(
                            "PRESENT"
                        )
                        .sum()
                    ),
                "expected_out_of_season":
                    int(
                        x[
                            "missing_class"
                        ]
                        .eq(
                            "EXPECTED_OUT_OF_SEASON"
                        )
                        .sum()
                    ),
                "expected_structural_season_edge":
                    int(
                        x[
                            "missing_class"
                        ]
                        .eq(
                            "EXPECTED_STRUCTURAL_SEASON_EDGE"
                        )
                        .sum()
                    ),
                "expected_conditional_medsea_no_support":
                    int(
                        x[
                            "missing_class"
                        ]
                        .eq(
                            "EXPECTED_CONDITIONAL_MEDSEA_NO_SUPPORT"
                        )
                        .sum()
                    ),
                "unexpected_operational_missing":
                    int(
                        x[
                            "missing_class"
                        ]
                        .eq(
                            "UNEXPECTED_OPERATIONAL_MISSING"
                        )
                        .sum()
                    ),
                "outside_f3_fit_minmax_receptors":
                    int(
                        xr[
                            "range_state"
                        ]
                        .isin(
                            [
                                "BELOW_F3_FIT_MIN",
                                "ABOVE_F3_FIT_MAX",
                            ]
                        )
                        .sum()
                    ),
                "extreme_tail_receptors":
                    int(
                        xr[
                            "range_state"
                        ]
                        .eq(
                            "WITHIN_RANGE_EXTREME_TAIL"
                        )
                        .sum()
                    ),
            }
        )

    feature_gate = pd.DataFrame(
        feature_rows
    )

    p1_gate = feature_gate[
        feature_gate[
            "feature"
        ].isin(
            p1_names
        )
    ].copy()

    p1_gate[
        "beta_gate_state"
    ] = np.select(
        [
            p1_gate[
                "unexpected_operational_missing"
            ].gt(0),
            p1_gate[
                "expected_structural_season_edge"
            ].gt(0),
            p1_gate[
                "expected_conditional_medsea_no_support"
            ].gt(0),
            p1_gate[
                "expected_out_of_season"
            ].gt(0),
            p1_gate[
                "outside_f3_fit_minmax_receptors"
            ].gt(0),
        ],
        [
            "BLOCK_UNEXPECTED_MISSING",
            "ALLOW_EXPECTED_SEASON_EDGE_NAN",
            "ALLOW_CONDITIONAL_MEDSEA_NAN",
            "BLOCK_OUT_OF_SEASON",
            "ALLOW_BETA_WITH_RANGE_REVIEW",
        ],
        default="ALLOW_BETA_PRESENT",
    )

    p1_features_unexpected = int(
        p1_gate[
            "unexpected_operational_missing"
        ]
        .gt(0)
        .sum()
    )

    p1_features_range_review = int(
        p1_gate[
            "outside_f3_fit_minmax_receptors"
        ]
        .gt(0)
        .sum()
    )

    progress(
        "PHASE 4/5",
        1,
        1,
        start,
        (
            f"unexpected missing cells={unexpected_cells} "
            f"| unexpected P1 cells={unexpected_p1_cells} "
            f"| P1 range review features={p1_features_range_review}"
        ),
    )

    # ------------------------------------------------------------------
    # PHASE 5/5 — final beta gate
    # ------------------------------------------------------------------
    print(
        "\nPHASE 5/5 — freeze prospective-beta decision"
    )
    start = time.time()

    structural_pass = bool(
        exact_order
        and static_pass
    )

    if not structural_pass:
        overall = (
            "FAIL_BETA_GATE_STRUCTURE_OR_STATIC"
        )
        beta_allowed = False

    elif not in_core_season:
        overall = (
            "PASS_BETA_GATE_LOGIC__OUT_OF_SEASON_BLOCKED"
        )
        beta_allowed = False

    elif unexpected_cells > 0:
        overall = (
            "PASS_BETA_GATE__UNEXPECTED_OPERATIONAL_MISSING_BLOCKED"
        )
        beta_allowed = False

    else:
        overall = (
            "PASS_PROSPECTIVE_BETA_ALLOWED__STRUCTURAL_NAN_ACCEPTED"
        )
        beta_allowed = True

    # Distributional equivalence is intentionally a separate concept.
    distribution_equivalence_confirmed = False

    cells_p = (
        snapshot
        / "operational_beta_gate_cells_v1_0.csv"
    )

    features_p = (
        snapshot
        / "operational_beta_gate_features_v1_0.csv"
    )

    p1_p = (
        snapshot
        / "operational_beta_gate_p1_v1_0.csv"
    )

    audit_json_p = (
        snapshot
        / "operational_beta_gate_audit_v1_0.json"
    )

    audit_txt_p = (
        snapshot
        / "operational_beta_gate_audit_v1_0.txt"
    )

    cells.to_csv(
        cells_p,
        index=False,
    )

    feature_gate.to_csv(
        features_p,
        index=False,
    )

    p1_gate.to_csv(
        p1_p,
        index=False,
    )

    audit = {
        "version": "1.0",
        "overall_status":
            overall,
        "run_id":
            run_id,
        "issue_date":
            issue_date.date().isoformat(),
        "in_core_season_sep_dec":
            bool(
                in_core_season
            ),
        "season_day":
            sd,
        "exact_frozen_97_order":
            bool(
                exact_order
            ),
        "static_identity_pass":
            bool(
                static_pass
            ),
        "unexpected_operational_missing_cells":
            unexpected_cells,
        "unexpected_p1_missing_cells":
            unexpected_p1_cells,
        "p1_features_with_unexpected_missing":
            p1_features_unexpected,
        "p1_features_with_range_review":
            p1_features_range_review,
        "prospective_beta_allowed":
            beta_allowed,
        "distribution_equivalence_confirmed":
            distribution_equivalence_confirmed,
        "official_warning_or_civil_protection_use_allowed":
            False,
        "model_prediction_performed":
            False,
        "structural_nan_policy":
            {
                "prev1d_from_season_day":
                    2,
                "prev3d_from_season_day":
                    4,
                "prev7d_from_season_day":
                    8,
                "prev14d_from_season_day":
                    15,
                "precip_3d_incl_today_from_season_day":
                    3,
                "precip_7d_incl_today_from_season_day":
                    7,
                "august_to_september_bridge":
                    False,
                "conditional_medsea_no_support_allowed":
                    True,
            },
        "critical_interpretation":
            (
                "Prospective beta permission means the operational feature vector "
                "respects the frozen structural missingness policy. It does NOT mean "
                "IFS/CMEMS distributional equivalence with ERA5/MedSea has already "
                "been proven."
            ),
        "next_step":
            (
                "Integrate this gate into the single daily nw_flood_watch.py runner. "
                "When prospective_beta_allowed=True, issue an EXPERIMENTAL "
                "PROSPECTIVE BETA bulletin, never an official alert."
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

    missing_counts = (
        cells[
            "missing_class"
        ]
        .value_counts()
        .rename_axis(
            "missing_class"
        )
        .reset_index(
            name="receptor_feature_cells"
        )
    )

    lines = [
        "=" * 220,
        "NW HYDROCLIMATE — PROSPECTIVE BETA GATE v1.0",
        "=" * 220,
        f"OVERALL STATUS                         : {overall}",
        f"Run ID                                 : {run_id}",
        f"Issue date                             : {issue_date.date()}",
        f"In CORE season Sep-Dec                 : {in_core_season}",
        f"Season day                             : {sd}",
        f"Exact frozen 97-predictor order        : {exact_order}",
        f"Static identity PASS                   : {static_pass}",
        f"Unexpected operational missing cells   : {unexpected_cells}",
        f"Unexpected P1 missing cells            : {unexpected_p1_cells}",
        f"P1 features with range review          : {p1_features_range_review}",
        f"Prospective beta allowed               : {beta_allowed}",
        "Distribution equivalence confirmed     : False",
        "Official warning use allowed           : False",
        "Model prediction performed             : False",
        "",
        "MISSINGNESS CLASS COUNTS",
        missing_counts.to_string(
            index=False
        ),
        "",
        "P1 BETA GATE",
        p1_gate.to_string(
            index=False
        ),
        "",
        "IMPORTANT",
        "Early-season structural NaNs are accepted because they reproduce training semantics.",
        "August data are never used to fill September antecedent windows.",
        "Conditional MedSea no-support NaNs are accepted when baseline support is false.",
        "Any unexpected operational NaN blocks the prospective beta.",
        "Range review does not automatically block beta because a real extreme can exceed historical support.",
        "Prospective beta is experimental and never an official civil-protection warning.",
        "",
        "NEXT STEP",
        "Integrate acquisition, feature engine, MedSea, cache, gate and inference into one nw_flood_watch.py command.",
        "",
        f"Cells    : {cells_p}",
        f"Features : {features_p}",
        f"P1 gate  : {p1_p}",
        f"Audit    : {audit_json_p}",
        f"Output   : {snapshot}",
    ]

    audit_txt_p.write_text(
        "\n".join(
            lines
        ) + "\n",
        encoding="utf-8",
    )

    progress(
        "PHASE 5/5",
        1,
        1,
        start,
        f"status={overall}",
    )

    print("\n" + "=" * 220)
    print("\n".join(lines[3:]))
    print("=" * 220)
    print(
        f"OVERALL STATUS           : {overall}"
    )
    print(
        f"Prospective beta allowed : {beta_allowed}"
    )
    print(
        f"Output                   : {snapshot}"
    )
    print("=" * 220)


if __name__ == "__main__":
    main()
