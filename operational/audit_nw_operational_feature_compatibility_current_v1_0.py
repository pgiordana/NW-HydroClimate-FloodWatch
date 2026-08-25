#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audit_nw_operational_feature_compatibility_current_v1_0.py

FASE 17 — STRICT RANGE / COMPATIBILITY AUDIT
OPERATIVO (IFS/CMEMS) vs TRAINING STORICO DEL CORE.

SCOPO
-----
Prima di dare al modello congelato un input operativo, questo script verifica:

1) struttura esatta:
   - 20 recettori;
   - 97 predictor;
   - stesso ordine del predictor dictionary canonico;
   - nessuna feature inattesa.

2) identità delle 14 feature statiche:
   - il valore operativo deve coincidere con il valore storico/canonico
     dello stesso recettore.

3) compatibilità di range delle feature dinamiche:
   - riferimento PRIMARIO = F3 FIT (<=2019), cioè il training effettivo
     del base model congelato;
   - riferimento SECONDARIO = F3 VALIDATION (2020-2022), utile per leggere
     drift temporale ma NON usato per riaddestrare nulla;
   - per ogni feature e recettore: min/max, quantili, empirical percentile,
     robust-z e stato rispetto al supporto storico.

4) missingness operativa:
   - confronta la frazione di NaN corrente con quella storica;
   - identifica P1/P2 mancanti;
   - distingue warm-up / out-of-season / proxy semantici.

5) semantica dei proxy:
   - mucape != ERA5 CAPE;
   - IFS vsw != automaticamente ERA5 soil-water;
   - IVT IFS su livelli Open Data != IVT ERA5 storico;
   - OHC/Tmean giornalieri ANFC != stato mensile MY/reanalysis del training;
   - precip_max_1h operativo è un proxy 3h/3, non un vero massimo orario.

COSA NON FA
-----------
- NON riaddestra;
- NON modifica soglie/calibratori;
- NON esegue il modello;
- NON dichiara "equivalenza distributiva" da una sola giornata;
- NON trasforma un valore estremo meteorologico in "errore" solo perché è raro.

INTERPRETAZIONE
---------------
Questo è un RANGE / PLAUSIBILITY audit di una singola giornata.
Un punto operativo dentro il range storico è necessario ma NON sufficiente
per dimostrare che IFS/CMEMS e ERA5/MedSea abbiano la stessa distribuzione.

Il confronto quantitativo paired IFS-vs-ERA5 dei giorni prospettici sarà
un audit separato ("shadow verification") quando ERA5/ERA5T dello stesso
giorno sarà disponibile.

FUORI STAGIONE
--------------
Il CORE è Sep-Dec. Un run del 25 agosto può superare il controllo strutturale
e di range, ma NON viene autorizzato come previsione scientifica CORE.
In particolare season_day è fuori dal dominio del training e le anomalie
MedSea canoniche sono intenzionalmente NaN.

OUTPUT
------
nw_operational_feature_snapshot/<RUN_ID>/
  operational_compatibility_by_receptor_v1_0.csv
  operational_compatibility_feature_summary_v1_0.csv
  operational_static_identity_check_v1_0.csv
  operational_proxy_semantic_summary_v1_0.csv
  operational_p1_p2_gate_v1_0.csv
  operational_compatibility_audit_v1_0.json
  operational_compatibility_audit_v1_0.txt
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_RECEPTORS = 20
EXPECTED_PREDICTORS = 97
EXPECTED_DYNAMIC = 83
EXPECTED_STATIC = 14

FIT_PARTITION = "FIT"
VAL_PARTITION = "VALIDATION"
REFERENCE_FOLD = "F3"

TAIL_Q = 0.01
EXTREME_Q = 0.005

STATIC_ATOL = 1e-10
STATIC_RTOL = 1e-10


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
            p for p in base.iterdir()
            if p.is_dir()
            and (
                p / "operational_full_97_predictors_v1_2.parquet"
            ).exists()
            and (
                p / "operational_feature_build_registry_v1_2.csv"
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


def empirical_percentile(sorted_values, x):
    if not np.isfinite(x) or len(sorted_values) == 0:
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

    # mid-rank empirical CDF
    return float(
        (left + right)
        / (2.0 * len(sorted_values))
    )


def distribution_stats(values):
    x = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).dropna().to_numpy(dtype=float)

    if len(x) == 0:
        return None

    qs = np.quantile(
        x,
        [
            0.001,
            0.005,
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
            0.995,
            0.999,
        ],
    )

    q001, q005, q01, q05, q25, q50, q75, q95, q99, q995, q999 = qs

    iqr = q75 - q25
    robust_scale = iqr / 1.349 if iqr > 0 else np.nan

    return {
        "n": int(len(x)),
        "min": float(np.min(x)),
        "q001": float(q001),
        "q005": float(q005),
        "q01": float(q01),
        "q05": float(q05),
        "q25": float(q25),
        "median": float(q50),
        "q75": float(q75),
        "q95": float(q95),
        "q99": float(q99),
        "q995": float(q995),
        "q999": float(q999),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "iqr": float(iqr),
        "robust_scale": (
            float(robust_scale)
            if np.isfinite(robust_scale)
            else np.nan
        ),
        "sorted": np.sort(x),
    }


def current_range_state(x, stats):
    if not np.isfinite(x):
        return "CURRENT_MISSING"

    if stats is None or stats["n"] == 0:
        return "NO_HISTORICAL_REFERENCE"

    if x < stats["min"]:
        return "BELOW_HISTORICAL_MIN"

    if x > stats["max"]:
        return "ABOVE_HISTORICAL_MAX"

    pct = empirical_percentile(
        stats["sorted"],
        x,
    )

    if (
        pct < EXTREME_Q
        or pct > 1.0 - EXTREME_Q
    ):
        return "WITHIN_RANGE_EXTREME_TAIL"

    if (
        pct < TAIL_Q
        or pct > 1.0 - TAIL_Q
    ):
        return "WITHIN_RANGE_1PCT_TAIL"

    return "WITHIN_HISTORICAL_RANGE"


def robust_z(x, stats):
    if (
        not np.isfinite(x)
        or stats is None
        or not np.isfinite(stats["robust_scale"])
        or stats["robust_scale"] <= 0
    ):
        return np.nan

    return float(
        (x - stats["median"])
        / stats["robust_scale"]
    )


def priority_map(root):
    p = (
        root
        / "nw_operational_feature_equivalence_preflight_v1_0"
        / "operational_priority_features_v1_0.csv"
    )

    if not p.exists():
        return {}

    df = pd.read_csv(
        p,
        low_memory=False,
    )

    return dict(
        zip(
            df["canonical_feature_name"].astype(str),
            df["operational_priority"].astype(str),
        )
    )


def semantic_map(snapshot_dir):
    p = (
        snapshot_dir
        / "operational_feature_build_registry_v1_2.csv"
    )

    df = pd.read_csv(
        p,
        low_memory=False,
    )

    return df.set_index(
        "canonical_feature_name"
    ).to_dict(
        orient="index"
    )


def equivalence_map(root):
    p = (
        root
        / "nw_operational_feature_equivalence_preflight_v1_0"
        / "operational_feature_equivalence_v1_0.csv"
    )

    if not p.exists():
        return {}

    df = pd.read_csv(
        p,
        low_memory=False,
    )

    return df.set_index(
        "canonical_feature_name"
    ).to_dict(
        orient="index"
    )


def main():
    root = Path(__file__).resolve().parent
    snapshot_dir = latest_v12_snapshot(root)
    run_id = snapshot_dir.name

    full_p = (
        snapshot_dir
        / "operational_full_97_predictors_v1_2.parquet"
    )

    build_registry_p = (
        snapshot_dir
        / "operational_feature_build_registry_v1_2.csv"
    )

    medsea_audit_p = find_first_existing(
        [
            snapshot_dir / "operational_medsea_audit_v1_1.json",
            snapshot_dir / "operational_medsea_audit_v1_0.json",
        ],
        "MedSea operational audit",
    )

    master_p = find_first_existing(
        [
            root
            / "nw_hydroclimate_foldwise_master_core_canonical_v1_0"
            / "nw_hydroclimate_foldwise_master_core_v1_0.parquet",
        ],
        "canonical master",
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

    dynamic_whitelist_p = find_first_existing(
        [
            root
            / "nw_hydroclimate_core_release_v1_0"
            / "metadata"
            / "primary_dynamic_feature_whitelist_canonical_v1_3.csv",
        ],
        "dynamic whitelist",
    )

    print("=" * 220)
    print("NW HYDROCLIMATE — OPERATIONAL FEATURE COMPATIBILITY AUDIT v1.0")
    print("=" * 220)
    print(f"Run ID : {run_id}")

    # ------------------------------------------------------------------
    # PHASE 1/6 — structural freeze checks
    # ------------------------------------------------------------------
    print("\nPHASE 1/6 — verify exact frozen 97-predictor structure")
    start = time.time()

    operational = pd.read_parquet(
        full_p
    )

    dictionary = pd.read_csv(
        dictionary_p,
        low_memory=False,
    )

    whitelist = pd.read_csv(
        dynamic_whitelist_p,
        low_memory=False,
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

    if len(predictor_order) != EXPECTED_PREDICTORS:
        raise SystemExit(
            f"Dictionary predictors={len(predictor_order)}, expected=97"
        )

    if len(dynamic_names) != EXPECTED_DYNAMIC:
        raise SystemExit(
            f"Dynamic features={len(dynamic_names)}, expected=83"
        )

    if len(static_names) != EXPECTED_STATIC:
        raise SystemExit(
            f"Static features={len(static_names)}, expected=14"
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
    )

    in_core_season = int(
        issue_date.month
    ) in {
        9,
        10,
        11,
        12,
    }

    progress(
        "PHASE 1/6",
        1,
        1,
        start,
        (
            f"rows=20 predictors=97 exact_order={exact_order} "
            f"| issue_date={issue_date.date()} in_core_season={in_core_season}"
        ),
    )

    # ------------------------------------------------------------------
    # PHASE 2/6 — load historical F3 FIT and VALIDATION reference
    # ------------------------------------------------------------------
    print("\nPHASE 2/6 — load F3 FIT training reference and F3 VALIDATION drift reference")
    start = time.time()

    needed_cols = [
        "fold_id",
        "partition",
        "receptor_id",
        "issue_date",
        *predictor_order,
    ]

    master = pd.read_parquet(
        master_p,
        columns=needed_cols,
    )

    ref = master[
        master["fold_id"]
        .astype(str)
        .eq(REFERENCE_FOLD)
    ].copy()

    fit = ref[
        ref["partition"]
        .astype(str)
        .eq(FIT_PARTITION)
    ].copy()

    val = ref[
        ref["partition"]
        .astype(str)
        .eq(VAL_PARTITION)
    ].copy()

    if len(fit) == 0 or len(val) == 0:
        raise SystemExit(
            f"F3 reference missing: FIT={len(fit)} VAL={len(val)}"
        )

    progress(
        "PHASE 2/6",
        1,
        1,
        start,
        (
            f"F3 FIT rows={len(fit)} | F3 VAL rows={len(val)} "
            f"| FIT years={pd.to_datetime(fit['issue_date']).dt.year.min()}-"
            f"{pd.to_datetime(fit['issue_date']).dt.year.max()}"
        ),
    )

    # ------------------------------------------------------------------
    # PHASE 3/6 — static identity check
    # ------------------------------------------------------------------
    print("\nPHASE 3/6 — verify static descriptors are identical to training")
    start = time.time()

    static_rows = []

    for rid in operational["receptor_id"].astype(str):
        op_row = operational[
            operational["receptor_id"]
            .astype(str)
            .eq(rid)
        ].iloc[0]

        hist = fit[
            fit["receptor_id"]
            .astype(str)
            .eq(rid)
        ]

        if len(hist) == 0:
            raise SystemExit(
                f"Receptor absent from F3 FIT: {rid}"
            )

        for feature in static_names:
            op = float(
                pd.to_numeric(
                    pd.Series(
                        [op_row[feature]]
                    ),
                    errors="coerce",
                ).iloc[0]
            )

            hist_values = (
                pd.to_numeric(
                    hist[feature],
                    errors="coerce",
                )
                .dropna()
                .unique()
            )

            if len(hist_values) != 1:
                status = (
                    "FAIL_HISTORICAL_STATIC_NOT_CONSTANT"
                )
                hist_value = (
                    float(hist_values[0])
                    if len(hist_values)
                    else np.nan
                )
            else:
                hist_value = float(
                    hist_values[0]
                )

                status = (
                    "PASS_STATIC_IDENTITY"
                    if np.isclose(
                        op,
                        hist_value,
                        atol=STATIC_ATOL,
                        rtol=STATIC_RTOL,
                        equal_nan=False,
                    )
                    else "FAIL_STATIC_VALUE_MISMATCH"
                )

            static_rows.append(
                {
                    "receptor_id": rid,
                    "feature": feature,
                    "operational_value": op,
                    "historical_f3_fit_value": hist_value,
                    "absolute_difference": (
                        abs(op - hist_value)
                        if np.isfinite(op)
                        and np.isfinite(hist_value)
                        else np.nan
                    ),
                    "status": status,
                }
            )

    static_check = pd.DataFrame(
        static_rows
    )

    static_pass = bool(
        static_check["status"]
        .eq("PASS_STATIC_IDENTITY")
        .all()
    )

    progress(
        "PHASE 3/6",
        1,
        1,
        start,
        (
            f"static cells={len(static_check)} "
            f"| exact identity pass={static_pass}"
        ),
    )

    if not static_pass:
        raise SystemExit(
            "Static descriptor identity check FAILED."
        )

    # ------------------------------------------------------------------
    # PHASE 4/6 — receptor-specific dynamic range audit
    # ------------------------------------------------------------------
    print("\nPHASE 4/6 — receptor-specific F3-FIT range / percentile audit")
    start = time.time()

    priority = priority_map(root)
    semantic = semantic_map(snapshot_dir)
    equivalence = equivalence_map(root)

    rows = []
    total_tasks = len(dynamic_names) * EXPECTED_RECEPTORS
    done = 0

    for feature in dynamic_names:
        sem = semantic.get(
            feature,
            {},
        )
        eq = equivalence.get(
            feature,
            {},
        )

        for rid in operational["receptor_id"].astype(str):
            done += 1

            op_series = operational.loc[
                operational["receptor_id"]
                .astype(str)
                .eq(rid),
                feature,
            ]

            op_value = pd.to_numeric(
                op_series,
                errors="coerce",
            ).iloc[0]

            fit_values = pd.to_numeric(
                fit.loc[
                    fit["receptor_id"]
                    .astype(str)
                    .eq(rid),
                    feature,
                ],
                errors="coerce",
            )

            val_values = pd.to_numeric(
                val.loc[
                    val["receptor_id"]
                    .astype(str)
                    .eq(rid),
                    feature,
                ],
                errors="coerce",
            )

            fit_stats = distribution_stats(
                fit_values
            )
            val_stats = distribution_stats(
                val_values
            )

            fit_pct = (
                empirical_percentile(
                    fit_stats["sorted"],
                    float(op_value),
                )
                if fit_stats is not None
                and np.isfinite(op_value)
                else np.nan
            )

            val_pct = (
                empirical_percentile(
                    val_stats["sorted"],
                    float(op_value),
                )
                if val_stats is not None
                and np.isfinite(op_value)
                else np.nan
            )

            state = current_range_state(
                float(op_value)
                if np.isfinite(op_value)
                else np.nan,
                fit_stats,
            )

            rows.append(
                {
                    "receptor_id": rid,
                    "feature": feature,
                    "priority": priority.get(
                        feature,
                        "P3_OTHER_CORE",
                    ),
                    "build_state": sem.get(
                        "build_state",
                        "",
                    ),
                    "operational_semantics": sem.get(
                        "operational_semantics",
                        "",
                    ),
                    "equivalence_class": eq.get(
                        "equivalence_class",
                        "",
                    ),
                    "operational_value": (
                        float(op_value)
                        if np.isfinite(op_value)
                        else np.nan
                    ),
                    "f3_fit_n": (
                        fit_stats["n"]
                        if fit_stats
                        else 0
                    ),
                    "f3_fit_min": (
                        fit_stats["min"]
                        if fit_stats
                        else np.nan
                    ),
                    "f3_fit_q01": (
                        fit_stats["q01"]
                        if fit_stats
                        else np.nan
                    ),
                    "f3_fit_median": (
                        fit_stats["median"]
                        if fit_stats
                        else np.nan
                    ),
                    "f3_fit_q99": (
                        fit_stats["q99"]
                        if fit_stats
                        else np.nan
                    ),
                    "f3_fit_max": (
                        fit_stats["max"]
                        if fit_stats
                        else np.nan
                    ),
                    "f3_fit_empirical_percentile":
                        fit_pct,
                    "f3_fit_robust_z":
                        robust_z(
                            float(op_value)
                            if np.isfinite(op_value)
                            else np.nan,
                            fit_stats,
                        ),
                    "f3_validation_n": (
                        val_stats["n"]
                        if val_stats
                        else 0
                    ),
                    "f3_validation_empirical_percentile":
                        val_pct,
                    "range_state":
                        state,
                }
            )

            progress(
                "PHASE 4/6",
                done,
                total_tasks,
                start,
                f"{feature} | {rid} | {state}",
            )

    by_receptor = pd.DataFrame(
        rows
    )

    # ------------------------------------------------------------------
    # PHASE 5/6 — summarize by feature, missingness, P1/P2 gates
    # ------------------------------------------------------------------
    print("\nPHASE 5/6 — summarize feature compatibility and P1/P2 operational gates")
    start = time.time()

    summary_rows = []

    for feature in dynamic_names:
        x = by_receptor[
            by_receptor["feature"]
            .eq(feature)
        ].copy()

        op_missing = int(
            x["operational_value"]
            .isna()
            .sum()
        )

        outside = int(
            x["range_state"]
            .isin(
                [
                    "BELOW_HISTORICAL_MIN",
                    "ABOVE_HISTORICAL_MAX",
                ]
            )
            .sum()
        )

        extreme_tail = int(
            x["range_state"]
            .eq(
                "WITHIN_RANGE_EXTREME_TAIL"
            )
            .sum()
        )

        onepct_tail = int(
            x["range_state"]
            .eq(
                "WITHIN_RANGE_1PCT_TAIL"
            )
            .sum()
        )

        fit_feature = pd.to_numeric(
            fit[feature],
            errors="coerce",
        )

        val_feature = pd.to_numeric(
            val[feature],
            errors="coerce",
        )

        fit_missing_frac = float(
            fit_feature.isna().mean()
        )

        val_missing_frac = float(
            val_feature.isna().mean()
        )

        op_missing_frac = (
            op_missing
            / EXPECTED_RECEPTORS
        )

        sem = semantic.get(
            feature,
            {},
        )
        eq = equivalence.get(
            feature,
            {},
        )

        if op_missing == EXPECTED_RECEPTORS:
            compatibility_state = (
                "CURRENTLY_UNAVAILABLE"
            )
        elif outside > 0:
            compatibility_state = (
                "REVIEW_OUTSIDE_TRAINING_SUPPORT"
            )
        elif (
            extreme_tail > 0
            or onepct_tail > 0
        ):
            compatibility_state = (
                "WITHIN_SUPPORT_BUT_TAIL_VALUES_PRESENT"
            )
        else:
            compatibility_state = (
                "WITHIN_F3_FIT_SUPPORT"
            )

        summary_rows.append(
            {
                "feature": feature,
                "priority": priority.get(
                    feature,
                    "P3_OTHER_CORE",
                ),
                "build_state": sem.get(
                    "build_state",
                    "",
                ),
                "equivalence_class": eq.get(
                    "equivalence_class",
                    "",
                ),
                "current_nonmissing_receptors":
                    EXPECTED_RECEPTORS - op_missing,
                "current_missing_receptors":
                    op_missing,
                "current_missing_fraction":
                    op_missing_frac,
                "f3_fit_missing_fraction":
                    fit_missing_frac,
                "f3_validation_missing_fraction":
                    val_missing_frac,
                "missingness_shift_vs_fit":
                    op_missing_frac
                    - fit_missing_frac,
                "outside_f3_fit_minmax_receptors":
                    outside,
                "extreme_tail_receptors":
                    extreme_tail,
                "onepct_tail_receptors":
                    onepct_tail,
                "median_fit_empirical_percentile":
                    float(
                        x[
                            "f3_fit_empirical_percentile"
                        ]
                        .median(
                            skipna=True
                        )
                    ),
                "max_abs_fit_robust_z":
                    float(
                        x[
                            "f3_fit_robust_z"
                        ]
                        .abs()
                        .max(
                            skipna=True
                        )
                    )
                    if x[
                        "f3_fit_robust_z"
                    ].notna().any()
                    else np.nan,
                "compatibility_state":
                    compatibility_state,
                "operational_semantics":
                    sem.get(
                        "operational_semantics",
                        "",
                    ),
            }
        )

    feature_summary = pd.DataFrame(
        summary_rows
    )

    p1p2 = feature_summary[
        feature_summary["priority"]
        .isin(
            [
                "P1_TOP10_ANY_HORIZON",
                "P2_TOP25_ANY_HORIZON",
            ]
        )
    ].copy()

    p1p2["gate_state"] = np.select(
        [
            p1p2["current_missing_receptors"]
            .eq(EXPECTED_RECEPTORS),
            p1p2[
                "outside_f3_fit_minmax_receptors"
            ].gt(0),
            p1p2[
                "equivalence_class"
            ].astype(str)
            .str.contains(
                "PROXY",
                na=False,
            ),
        ],
        [
            "BLOCKED_MISSING_CURRENT",
            "REVIEW_OUTSIDE_TRAINING_SUPPORT",
            "PROXY_RANGE_PLAUSIBLE_SEMANTIC_VALIDATION_PENDING",
        ],
        default="READY_RANGE_CHECK",
    )

    p1 = p1p2[
        p1p2["priority"]
        .eq("P1_TOP10_ANY_HORIZON")
    ]

    p1_total = int(
        len(p1)
    )

    p1_complete = int(
        p1[
            "current_missing_receptors"
        ]
        .eq(0)
        .sum()
    )

    p1_missing = int(
        p1_total
        - p1_complete
    )

    p1_outside = int(
        p1[
            "outside_f3_fit_minmax_receptors"
        ]
        .gt(0)
        .sum()
    )

    dynamic_zero = int(
        feature_summary[
            "current_nonmissing_receptors"
        ]
        .eq(0)
        .sum()
    )

    dynamic_complete = int(
        feature_summary[
            "current_nonmissing_receptors"
        ]
        .eq(EXPECTED_RECEPTORS)
        .sum()
    )

    outside_feature_count = int(
        feature_summary[
            "outside_f3_fit_minmax_receptors"
        ]
        .gt(0)
        .sum()
    )

    proxy_features = int(
        feature_summary[
            "equivalence_class"
        ]
        .astype(str)
        .str.contains(
            "PROXY",
            na=False,
        )
        .sum()
    )

    progress(
        "PHASE 5/6",
        1,
        1,
        start,
        (
            f"P1 complete={p1_complete}/{p1_total} "
            f"| P1 outside support features={p1_outside} "
            f"| dynamic complete={dynamic_complete}/83"
        ),
    )

    # ------------------------------------------------------------------
    # PHASE 6/6 — freeze audit / hard scientific gate
    # ------------------------------------------------------------------
    print("\nPHASE 6/6 — freeze compatibility audit and scientific inference gate")
    start = time.time()

    build_registry = pd.read_csv(
        build_registry_p,
        low_memory=False,
    )

    medsea_audit = json.loads(
        medsea_audit_p.read_text(
            encoding="utf-8"
        )
    )

    static_p = (
        snapshot_dir
        / "operational_static_identity_check_v1_0.csv"
    )
    by_rec_p = (
        snapshot_dir
        / "operational_compatibility_by_receptor_v1_0.csv"
    )
    summary_p = (
        snapshot_dir
        / "operational_compatibility_feature_summary_v1_0.csv"
    )
    p1p2_p = (
        snapshot_dir
        / "operational_p1_p2_gate_v1_0.csv"
    )
    semantic_p = (
        snapshot_dir
        / "operational_proxy_semantic_summary_v1_0.csv"
    )
    audit_json_p = (
        snapshot_dir
        / "operational_compatibility_audit_v1_0.json"
    )
    audit_txt_p = (
        snapshot_dir
        / "operational_compatibility_audit_v1_0.txt"
    )

    static_check.to_csv(
        static_p,
        index=False,
    )
    by_receptor.to_csv(
        by_rec_p,
        index=False,
    )
    feature_summary.to_csv(
        summary_p,
        index=False,
    )
    p1p2.to_csv(
        p1p2_p,
        index=False,
    )

    semantic_summary = (
        feature_summary[
            [
                "feature",
                "priority",
                "build_state",
                "equivalence_class",
                "operational_semantics",
                "compatibility_state",
            ]
        ]
        .copy()
    )

    semantic_summary.to_csv(
        semantic_p,
        index=False,
    )

    # Scientific gate:
    # - structural/static must pass;
    # - must be Sep-Dec;
    # - all P1 available;
    # - semantic proxy validation remains pending in this first-day audit.
    structural_ready = bool(
        exact_order
        and static_pass
    )

    scientific_inference_allowed = bool(
        structural_ready
        and in_core_season
        and p1_missing == 0
        and proxy_features == 0
    )

    # The previous condition is intentionally strict. In the first operational
    # generation proxy_features will be >0, so scientific equivalence cannot
    # yet be claimed from a one-day audit.
    technical_smoke_inference_allowed = bool(
        structural_ready
    )

    if not structural_ready:
        overall = (
            "FAIL_COMPATIBILITY_STRUCTURE_OR_STATIC_IDENTITY"
        )
    elif not in_core_season:
        overall = (
            "PASS_RANGE_AUDIT__OUT_OF_SEASON__SCIENTIFIC_INFERENCE_BLOCKED"
        )
    elif p1_missing > 0:
        overall = (
            "PASS_RANGE_AUDIT__P1_WARMUP_MISSING__SCIENTIFIC_INFERENCE_BLOCKED"
        )
    else:
        overall = (
            "PASS_RANGE_AUDIT__PROXY_SEMANTIC_VALIDATION_PENDING"
        )

    audit = {
        "version": "1.0",
        "overall_status": overall,
        "run_id": run_id,
        "issue_date": str(
            issue_date.date()
        ),
        "in_core_season_sep_dec":
            bool(
                in_core_season
            ),
        "structural_97_predictor_order_pass":
            bool(
                exact_order
            ),
        "static_identity_pass":
            bool(
                static_pass
            ),
        "dynamic_features_complete_all_receptors":
            dynamic_complete,
        "dynamic_features_zero_coverage":
            dynamic_zero,
        "p1_features_total":
            p1_total,
        "p1_features_complete_all_receptors":
            p1_complete,
        "p1_features_missing":
            p1_missing,
        "p1_features_with_values_outside_f3_fit_minmax":
            p1_outside,
        "dynamic_features_with_any_value_outside_f3_fit_minmax":
            outside_feature_count,
        "proxy_features":
            proxy_features,
        "scientific_inference_allowed":
            scientific_inference_allowed,
        "technical_smoke_inference_allowed":
            technical_smoke_inference_allowed,
        "model_prediction_performed":
            False,
        "range_audit_is_distribution_equivalence_proof":
            False,
        "primary_reference":
            "F3_FIT__BASE_MODEL_TRAINING__THROUGH_2019",
        "secondary_reference":
            "F3_VALIDATION__2020_2022__DRIFT_DIAGNOSTIC_ONLY",
        "medsea_state":
            medsea_audit.get(
                "overall_status",
                "",
            ),
        "critical_interpretation":
            (
                "A single operational day inside historical ranges does not prove "
                "IFS/CMEMS distributional equivalence with ERA5/MedSea training data. "
                "Prospective paired shadow verification remains required."
            ),
        "next_step":
            (
                "If structural audit passes, a TECHNICAL smoke inference may be "
                "performed with an explicit NON-SCIENTIFIC / OUT-OF-SEASON or "
                "WARM-UP label. Scientific beta interpretation begins only in "
                "Sep-Dec and remains subject to proxy shadow verification."
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

    state_counts = (
        feature_summary[
            "compatibility_state"
        ]
        .value_counts()
        .rename_axis(
            "compatibility_state"
        )
        .reset_index(
            name="features"
        )
    )

    range_counts = (
        by_receptor[
            "range_state"
        ]
        .value_counts()
        .rename_axis(
            "range_state"
        )
        .reset_index(
            name="receptor_feature_cells"
        )
    )

    review_features = (
        feature_summary[
            (
                feature_summary[
                    "outside_f3_fit_minmax_receptors"
                ].gt(0)
            )
            | (
                feature_summary[
                    "priority"
                ].isin(
                    [
                        "P1_TOP10_ANY_HORIZON",
                        "P2_TOP25_ANY_HORIZON",
                    ]
                )
            )
        ][
            [
                "feature",
                "priority",
                "current_nonmissing_receptors",
                "current_missing_receptors",
                "outside_f3_fit_minmax_receptors",
                "extreme_tail_receptors",
                "median_fit_empirical_percentile",
                "max_abs_fit_robust_z",
                "equivalence_class",
                "compatibility_state",
            ]
        ]
        .sort_values(
            [
                "priority",
                "outside_f3_fit_minmax_receptors",
                "max_abs_fit_robust_z",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
    )

    lines = [
        "=" * 220,
        "NW HYDROCLIMATE — OPERATIONAL FEATURE COMPATIBILITY AUDIT v1.0",
        "=" * 220,
        f"OVERALL STATUS                              : {overall}",
        f"Run ID                                      : {run_id}",
        f"Issue date                                  : {issue_date.date()}",
        f"In CORE season Sep-Dec                      : {in_core_season}",
        f"Exact frozen 97-predictor order             : {exact_order}",
        f"Static identity PASS                        : {static_pass}",
        f"Dynamic complete all receptors              : {dynamic_complete}/{EXPECTED_DYNAMIC}",
        f"Dynamic zero coverage                       : {dynamic_zero}/{EXPECTED_DYNAMIC}",
        f"P1 complete all receptors                   : {p1_complete}/{p1_total}",
        f"P1 missing                                  : {p1_missing}",
        f"P1 with any value outside F3-FIT min/max    : {p1_outside}",
        f"Dynamic features outside F3-FIT support     : {outside_feature_count}/{EXPECTED_DYNAMIC}",
        f"Proxy features                              : {proxy_features}/{EXPECTED_DYNAMIC}",
        f"Technical smoke inference allowed           : {technical_smoke_inference_allowed}",
        f"Scientific inference allowed                : {scientific_inference_allowed}",
        "Model prediction performed                    : False",
        "",
        "FEATURE COMPATIBILITY STATE COUNTS",
        state_counts.to_string(index=False),
        "",
        "RECEPTOR-FEATURE RANGE STATE COUNTS",
        range_counts.to_string(index=False),
        "",
        "P1 / P2 GATE",
        p1p2[
            [
                "feature",
                "priority",
                "current_nonmissing_receptors",
                "outside_f3_fit_minmax_receptors",
                "max_abs_fit_robust_z",
                "equivalence_class",
                "compatibility_state",
                "gate_state",
            ]
        ].to_string(index=False),
        "",
        "PRIORITY / REVIEW FEATURES",
        review_features.to_string(index=False),
        "",
        "IMPORTANT",
        "Primary reference is F3 FIT because the frozen base models were fit through 2019.",
        "F3 VALIDATION is a secondary drift diagnostic only.",
        "A meteorological value outside historical min/max is a REVIEW signal, not automatically a data error.",
        "One operational day cannot prove distributional equivalence between IFS/CMEMS and ERA5/MedSea.",
        "Static descriptors must be identical; they are not allowed to drift.",
        "No model prediction is executed by this audit.",
        "",
        "NEXT STEP",
        "If this audit passes structurally, build the first explicitly TECHNICAL smoke inference/bulletin. "
        "For 25 August it must be labelled OUT-OF-SEASON and NON-SCIENTIFIC. "
        "From 1 September the daily prospective beta can begin, initially with cache-warmup/proxy caveats.",
        "",
        f"By receptor      : {by_rec_p}",
        f"Feature summary  : {summary_p}",
        f"Static identity  : {static_p}",
        f"P1/P2 gate       : {p1p2_p}",
        f"Semantic summary : {semantic_p}",
        f"Audit            : {audit_json_p}",
        f"Output           : {snapshot_dir}",
    ]

    audit_txt_p.write_text(
        "\n".join(
            lines
        ) + "\n",
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
    print(f"Output         : {snapshot_dir}")
    print("=" * 220)


if __name__ == "__main__":
    main()
