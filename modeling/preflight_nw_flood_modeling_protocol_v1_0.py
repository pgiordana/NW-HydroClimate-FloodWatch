#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preflight_nw_flood_modeling_protocol_v1_0.py

PREFLIGHT DEL PROTOCOLLO DI MODELLAZIONE SUL MASTER CANONICO v1.0.

INPUT
-----
nw_hydroclimate_foldwise_master_core_canonical_v1_0/
  nw_hydroclimate_foldwise_master_core_v1_0.parquet
  oppure fallback:
  nw_hydroclimate_foldwise_master_core_v1_0.csv.gz
  nw_hydroclimate_master_predictor_dictionary_v1_0.csv
  nw_hydroclimate_master_fold_registry_v1_0.csv

SCOPO
-----
Verificare che il master sia pronto per il fitting e congelare un protocollo
di modeling SENZA addestrare o valutare modelli sul TEST.

REGOLE
------
- 3 classificatori separati: 24 h, 48 h, 72 h.
- Modello regionale pooled sui 20 recettori.
- FIT = training.
- VALIDATION = selezione modello/iperparametri, calibrazione e soglia operativa.
- TEST = sigillato; nessuna metrica di modello viene calcolata in questo script.
- Nessun random split.
- Nessun preprocessing globale.
- Eventuale imputazione/scaling di modelli che lo richiedono: FIT-only.
- Modelli con gestione nativa NaN sono preferibili per il CORE.
- Q95/label/target non sono predictor.
- Le metriche giornaliere non sono considerate osservazioni indipendenti:
  l'incertezza finale dovrà usare bootstrap per stagione/episodio.
- Class imbalance: audit FIT-only.
- Differente disponibilità idrologica per receptor: audit FIT-only e confronto
  futuro tra row-weighting naturale e receptor-balanced weighting, deciso su
  VALIDATION soltanto.

CANDIDATI PREDEFINITI
---------------------
Baseline:
1) GLOBAL_FIT_CLIMATOLOGY
2) RECEPTOR_SMOOTHED_FIT_CLIMATOLOGY

Modelli:
3) LOGISTIC_ELASTICNET
   - median imputation + missing indicators + scaling FIT-only
   - baseline interpretabile

4) HIST_GRADIENT_BOOSTING
   - gestione nativa dei NaN
   - candidato nonlinear CORE

5) XGBOOST (solo se installato)
   - candidato nonlinear sensitivity

Il preflight NON sceglie il vincitore.

METRICHE DI SELEZIONE SU VALIDATION
-----------------------------------
Primary:
- Average Precision / PR-AUC

Secondary:
- ROC-AUC
- Brier score
- Log loss
- calibration slope/intercept

Metriche thresholded:
- CSI / F1
- precision
- recall
La soglia viene scelta SOLO su VALIDATION.

VALUTAZIONE FINALE
------------------
Dopo il freeze di modello + iperparametri + calibrazione + threshold:
- una sola valutazione TEST per fold/horizon;
- metriche day-level + event-cluster level;
- CI con bootstrap per season-year/event cluster, non row-wise.

OUTPUT
------
nw_flood_modeling_protocol_preflight_v1_0/
  modeling_partition_support_v1_0.csv
  modeling_fit_receptor_support_v1_0.csv
  modeling_fit_predictor_audit_v1_0.csv
  modeling_candidate_registry_v1_0.csv
  modeling_weighting_strategy_registry_v1_0.csv
  modeling_metric_registry_v1_0.csv
  modeling_environment_audit_v1_0.csv
  modeling_protocol_preflight_audit_v1_0.json
  modeling_protocol_preflight_audit_v1_0.txt
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_ROWS = 263520
EXPECTED_RECEPTORS = 20
EXPECTED_FOLDS = 3
EXPECTED_PREDICTORS = 97

HORIZONS = [24, 48, 72]

LABEL_COLS = {
    24: "label_extreme_within_24h",
    48: "label_extreme_within_48h",
    72: "label_extreme_within_72h",
}

FORBIDDEN_PATTERNS = [
    r"(^|__)label($|__|_)",
    r"(^|__)target($|__|_)",
    r"future",
    r"q95",
    r"q975",
    r"threshold",
    r"current_target",
    r"official",
    r"season_year",
]


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
        msg += f" | {str(current)[:125]}"

    print(msg.ljust(265), end="", flush=True)
    if done >= total:
        print(flush=True)


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def forbidden_name(name: str):
    lc = str(name).lower()
    return [p for p in FORBIDDEN_PATTERNS if re.search(p, lc)]


def read_master(parquet_p: Path, csv_p: Path):
    if parquet_p.exists():
        try:
            return pd.read_parquet(parquet_p), "PARQUET"
        except Exception:
            pass

    if csv_p.exists():
        return pd.read_csv(csv_p, low_memory=False), "CSV_GZIP"

    raise SystemExit(
        f"Manca sia il Parquet sia il CSV master:\n{parquet_p}\n{csv_p}"
    )


def main():
    root = Path(__file__).resolve().parent

    master_root = (
        root / "nw_hydroclimate_foldwise_master_core_canonical_v1_0"
    )

    parquet_p = (
        master_root / "nw_hydroclimate_foldwise_master_core_v1_0.parquet"
    )
    csv_p = (
        master_root / "nw_hydroclimate_foldwise_master_core_v1_0.csv.gz"
    )
    dictionary_p = (
        master_root / "nw_hydroclimate_master_predictor_dictionary_v1_0.csv"
    )
    fold_registry_p = (
        master_root / "nw_hydroclimate_master_fold_registry_v1_0.csv"
    )

    out = root / "nw_flood_modeling_protocol_preflight_v1_0"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 216)
    print("NW HYDROCLIMATE — FLOOD MODELING PROTOCOL PREFLIGHT v1.0")
    print("=" * 216)

    for p in (dictionary_p, fold_registry_p):
        if not p.exists():
            raise SystemExit(f"Manca: {p}")

    # ------------------------------------------------------------------
    # PHASE 1/5 — master integrity
    # ------------------------------------------------------------------
    print("\nPHASE 1/5 — load and validate canonical master")
    start1 = time.time()

    master, master_format = read_master(parquet_p, csv_p)
    dictionary = pd.read_csv(dictionary_p, low_memory=False)
    folds = pd.read_csv(fold_registry_p, low_memory=False)

    master["issue_date"] = pd.to_datetime(
        master["issue_date"],
        errors="coerce",
    )

    errors = []
    warnings = []

    if len(master) != EXPECTED_ROWS:
        errors.append(f"MASTER_ROWS={len(master)}")

    if master["receptor_id"].nunique() != EXPECTED_RECEPTORS:
        errors.append(
            f"RECEPTORS={master['receptor_id'].nunique()}"
        )

    if master["fold_id"].nunique() != EXPECTED_FOLDS:
        errors.append(
            f"FOLDS={master['fold_id'].nunique()}"
        )

    if len(dictionary) != EXPECTED_PREDICTORS:
        errors.append(f"PREDICTORS={len(dictionary)}")

    dup = int(
        master.duplicated(
            ["fold_id", "receptor_id", "issue_date"]
        ).sum()
    )
    if dup:
        errors.append(f"DUPLICATE_MASTER_ROWS={dup}")

    predictor_cols = dictionary["predictor"].astype(str).tolist()

    missing_predictors = sorted(
        set(predictor_cols) - set(master.columns)
    )
    if missing_predictors:
        errors.append(
            "MISSING_PREDICTORS="
            + ",".join(missing_predictors)
        )

    forbidden_rows = []
    for col in predictor_cols:
        hits = forbidden_name(col)
        if hits:
            forbidden_rows.append(
                {
                    "predictor": col,
                    "patterns": "|".join(hits),
                }
            )

    if forbidden_rows:
        errors.append(
            f"FORBIDDEN_PREDICTORS={len(forbidden_rows)}"
        )

    missing_labels = [
        col for col in LABEL_COLS.values()
        if col not in master.columns
    ]
    if missing_labels:
        errors.append(
            "MISSING_LABELS=" + ",".join(missing_labels)
        )

    if errors:
        progress(
            "PHASE 1/5",
            1,
            1,
            start1,
            f"errors={len(errors)}",
        )
        print("\nPREFLIGHT ABORTED")
        for e in errors:
            print(f" - {e}")
        raise SystemExit(2)

    progress(
        "PHASE 1/5",
        1,
        1,
        start1,
        f"format={master_format} rows={len(master)} predictors={len(predictor_cols)}",
    )

    # ------------------------------------------------------------------
    # PHASE 2/5 — partition/class support
    # ------------------------------------------------------------------
    print("\nPHASE 2/5 — audit fold/horizon eligibility and class imbalance")
    start2 = time.time()

    partition_rows = []
    receptor_rows = []

    total = EXPECTED_FOLDS * len(HORIZONS)
    done = 0

    for fold_id in sorted(master["fold_id"].astype(str).unique()):
        fold = master[
            master["fold_id"].astype(str).eq(fold_id)
        ]

        for h in HORIZONS:
            done += 1
            label_col = LABEL_COLS[h]

            for partition in ("FIT", "VALIDATION", "TEST"):
                sub = fold[
                    fold["partition"].astype(str).eq(partition)
                ].copy()

                y = pd.to_numeric(
                    sub[label_col],
                    errors="coerce",
                )

                eligible = y.notna()
                pos = int(y.eq(1).sum())
                neg = int(y.eq(0).sum())
                n = int(eligible.sum())

                partition_rows.append(
                    {
                        "fold_id": fold_id,
                        "horizon_hours": h,
                        "partition": partition,
                        "issue_rows": int(len(sub)),
                        "eligible_rows": n,
                        "unknown_rows": int(y.isna().sum()),
                        "positive_rows": pos,
                        "negative_rows": neg,
                        "positive_fraction":
                            pos / n if n else np.nan,
                        "fit_pos_weight_neg_over_pos":
                            neg / pos
                            if partition == "FIT" and pos
                            else np.nan,
                        "test_sealed_for_model_scoring":
                            partition == "TEST",
                    }
                )

            fit = fold[
                fold["partition"].astype(str).eq("FIT")
            ].copy()

            fy = pd.to_numeric(
                fit[label_col],
                errors="coerce",
            )
            fit = fit.loc[fy.notna()].copy()
            fit["_y"] = fy.loc[fy.notna()].astype(int).to_numpy()

            total_fit = len(fit)

            for receptor, rg in fit.groupby("receptor_id"):
                pos = int(rg["_y"].eq(1).sum())
                neg = int(rg["_y"].eq(0).sum())
                n = int(len(rg))

                receptor_rows.append(
                    {
                        "fold_id": fold_id,
                        "horizon_hours": h,
                        "receptor_id": receptor,
                        "eligible_fit_rows": n,
                        "fit_row_share":
                            n / total_fit if total_fit else np.nan,
                        "positive_fit_rows": pos,
                        "negative_fit_rows": neg,
                        "positive_fraction":
                            pos / n if n else np.nan,
                        "receptor_balance_weight":
                            total_fit
                            / (EXPECTED_RECEPTORS * n)
                            if n else np.nan,
                    }
                )

            progress(
                "PHASE 2/5",
                done,
                total,
                start2,
                f"{fold_id} | {h}h",
            )

    partition_support = pd.DataFrame(partition_rows)
    receptor_support = pd.DataFrame(receptor_rows)

    fit_zero_pos_receptor_rows = int(
        receptor_support["positive_fit_rows"].eq(0).sum()
    )
    if fit_zero_pos_receptor_rows:
        warnings.append(
            f"{fit_zero_pos_receptor_rows} receptor/fold/horizon FIT rows have zero positives."
        )

    # ------------------------------------------------------------------
    # PHASE 3/5 — FIT-only predictor audit
    # ------------------------------------------------------------------
    print("\nPHASE 3/5 — FIT-only predictor missingness and variability audit")
    start3 = time.time()

    predictor_audit_rows = []

    total = EXPECTED_FOLDS * len(predictor_cols)
    done = 0

    dict_idx = dictionary.set_index("predictor")

    for fold_id in sorted(master["fold_id"].astype(str).unique()):
        fit = master[
            master["fold_id"].astype(str).eq(fold_id)
            & master["partition"].astype(str).eq("FIT")
        ]

        for col in predictor_cols:
            done += 1

            x = pd.to_numeric(
                fit[col],
                errors="coerce",
            )

            nonnull = x.dropna()
            n_unique = int(nonnull.nunique(dropna=True))

            std = (
                float(nonnull.std(ddof=0))
                if len(nonnull)
                else np.nan
            )

            family = str(dict_idx.loc[col, "family"])
            source = str(dict_idx.loc[col, "source"])
            role = str(dict_idx.loc[col, "model_role"])
            miss_sem = str(
                dict_idx.loc[col, "missingness_semantics"]
            )

            predictor_audit_rows.append(
                {
                    "fold_id": fold_id,
                    "predictor": col,
                    "family": family,
                    "source": source,
                    "model_role": role,
                    "missingness_semantics": miss_sem,
                    "fit_rows": int(len(fit)),
                    "non_null_rows": int(x.notna().sum()),
                    "missing_rows": int(x.isna().sum()),
                    "missing_fraction": float(x.isna().mean()),
                    "unique_non_null_values": n_unique,
                    "std_non_null": std,
                    "constant_in_fit": n_unique <= 1,
                }
            )

            progress(
                "PHASE 3/5",
                done,
                total,
                start3,
                f"{fold_id} | {col}",
            )

    predictor_audit = pd.DataFrame(
        predictor_audit_rows
    )

    constant_rows = predictor_audit[
        predictor_audit["constant_in_fit"]
    ].copy()

    if len(constant_rows):
        warnings.append(
            f"{len(constant_rows)} predictor/fold rows are constant in FIT."
        )

    # ------------------------------------------------------------------
    # PHASE 4/5 — environment and frozen candidate protocol
    # ------------------------------------------------------------------
    print("\nPHASE 4/5 — audit model environment and define candidate registry")
    start4 = time.time()

    env_rows = [
        {
            "component": "python",
            "available": True,
            "version_or_note": sys.version.split()[0],
            "required_for_primary_protocol": True,
        },
        {
            "component": "pandas",
            "available": True,
            "version_or_note": pd.__version__,
            "required_for_primary_protocol": True,
        },
        {
            "component": "numpy",
            "available": True,
            "version_or_note": np.__version__,
            "required_for_primary_protocol": True,
        },
    ]

    sklearn_available = module_available("sklearn")
    sklearn_version = ""

    if sklearn_available:
        try:
            import sklearn
            sklearn_version = sklearn.__version__
        except Exception as exc:
            sklearn_version = f"IMPORT_ERROR:{exc}"

    env_rows.append(
        {
            "component": "scikit-learn",
            "available": sklearn_available,
            "version_or_note": sklearn_version,
            "required_for_primary_protocol": True,
        }
    )

    for mod in ("xgboost", "lightgbm", "catboost"):
        env_rows.append(
            {
                "component": mod,
                "available": module_available(mod),
                "version_or_note": "optional sensitivity candidate",
                "required_for_primary_protocol": False,
            }
        )

    environment = pd.DataFrame(env_rows)

    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "B1",
                "model_name": "GLOBAL_FIT_CLIMATOLOGY",
                "model_class": "BASELINE",
                "nan_handling": "NOT_APPLICABLE",
                "preprocessing": "NONE",
                "selection_partition": "VALIDATION",
                "test_policy": "SEALED_UNTIL_FINAL_FREEZE",
            },
            {
                "candidate_id": "B2",
                "model_name": "RECEPTOR_SMOOTHED_FIT_CLIMATOLOGY",
                "model_class": "BASELINE",
                "nan_handling": "NOT_APPLICABLE",
                "preprocessing": "FIT_ONLY_BETA_BINOMIAL_SHRINKAGE",
                "selection_partition": "VALIDATION",
                "test_policy": "SEALED_UNTIL_FINAL_FREEZE",
            },
            {
                "candidate_id": "M1",
                "model_name": "LOGISTIC_ELASTICNET",
                "model_class": "INTERPRETABLE_LINEAR",
                "nan_handling": "FIT_ONLY_MEDIAN_IMPUTATION_PLUS_MISSING_INDICATORS",
                "preprocessing": "FIT_ONLY_SCALING",
                "selection_partition": "VALIDATION",
                "test_policy": "SEALED_UNTIL_FINAL_FREEZE",
            },
            {
                "candidate_id": "M2",
                "model_name": "HIST_GRADIENT_BOOSTING",
                "model_class": "PRIMARY_NONLINEAR",
                "nan_handling": "NATIVE_NAN",
                "preprocessing": "NO_SCALING_REQUIRED",
                "selection_partition": "VALIDATION",
                "test_policy": "SEALED_UNTIL_FINAL_FREEZE",
            },
            {
                "candidate_id": "M3",
                "model_name": "XGBOOST",
                "model_class": "OPTIONAL_NONLINEAR_SENSITIVITY",
                "nan_handling": "NATIVE_NAN",
                "preprocessing": "NO_SCALING_REQUIRED",
                "selection_partition": "VALIDATION_IF_PACKAGE_AVAILABLE",
                "test_policy": "SEALED_UNTIL_FINAL_FREEZE",
            },
        ]
    )

    weighting = pd.DataFrame(
        [
            {
                "strategy_id": "W1",
                "strategy": "NATURAL_ROW_WEIGHTING",
                "definition":
                    "Every eligible FIT receptor-day has unit weight.",
                "selection_rule":
                    "Compare against W2 on VALIDATION only.",
            },
            {
                "strategy_id": "W2",
                "strategy": "RECEPTOR_BALANCED_WEIGHTING",
                "definition":
                    "Within each fold/horizon, total FIT weight is equalized across receptors; "
                    "class imbalance may then be handled by model class weights.",
                "selection_rule":
                    "Compare against W1 on VALIDATION only.",
            },
        ]
    )

    metrics = pd.DataFrame(
        [
            {
                "metric": "AVERAGE_PRECISION",
                "role": "PRIMARY_VALIDATION_SELECTION",
                "threshold_free": True,
            },
            {
                "metric": "ROC_AUC",
                "role": "SECONDARY",
                "threshold_free": True,
            },
            {
                "metric": "BRIER_SCORE",
                "role": "SECONDARY_CALIBRATION",
                "threshold_free": True,
            },
            {
                "metric": "LOG_LOSS",
                "role": "SECONDARY_CALIBRATION",
                "threshold_free": True,
            },
            {
                "metric": "CALIBRATION_SLOPE_INTERCEPT",
                "role": "SECONDARY_CALIBRATION",
                "threshold_free": True,
            },
            {
                "metric": "CSI_F1_PRECISION_RECALL",
                "role": "VALIDATION_THRESHOLD_SELECTION_AND_FINAL_TEST",
                "threshold_free": False,
            },
            {
                "metric": "EVENT_CLUSTER_HIT_RATE_FALSE_ALARM",
                "role": "FINAL_TEST_EVENT_LEVEL",
                "threshold_free": False,
            },
        ]
    )

    progress(
        "PHASE 4/5",
        1,
        1,
        start4,
        f"sklearn={sklearn_available} optional_xgboost={module_available('xgboost')}",
    )

    # ------------------------------------------------------------------
    # PHASE 5/5 — write audit
    # ------------------------------------------------------------------
    print("\nPHASE 5/5 — write modeling protocol preflight artifacts")
    start5 = time.time()

    partition_p = out / "modeling_partition_support_v1_0.csv"
    receptor_p = out / "modeling_fit_receptor_support_v1_0.csv"
    predictor_p = out / "modeling_fit_predictor_audit_v1_0.csv"
    candidates_p = out / "modeling_candidate_registry_v1_0.csv"
    weighting_p = out / "modeling_weighting_strategy_registry_v1_0.csv"
    metrics_p = out / "modeling_metric_registry_v1_0.csv"
    env_p = out / "modeling_environment_audit_v1_0.csv"
    audit_json = out / "modeling_protocol_preflight_audit_v1_0.json"
    audit_txt = out / "modeling_protocol_preflight_audit_v1_0.txt"

    partition_support.to_csv(partition_p, index=False)
    receptor_support.to_csv(receptor_p, index=False)
    predictor_audit.to_csv(predictor_p, index=False)
    candidates.to_csv(candidates_p, index=False)
    weighting.to_csv(weighting_p, index=False)
    metrics.to_csv(metrics_p, index=False)
    environment.to_csv(env_p, index=False)

    sklearn_required_missing = not sklearn_available

    if sklearn_required_missing:
        overall = "PASS_WITH_MODEL_ENVIRONMENT_SETUP_REQUIRED"
    elif len(constant_rows):
        overall = "PASS_WITH_CONSTANT_FIT_PREDICTOR_REVIEW"
    else:
        overall = "PASS"

    fit_support_compact = partition_support[
        partition_support["partition"].eq("FIT")
    ][
        [
            "fold_id",
            "horizon_hours",
            "eligible_rows",
            "positive_rows",
            "negative_rows",
            "positive_fraction",
            "fit_pos_weight_neg_over_pos",
        ]
    ].copy()

    receptor_imbalance = (
        receptor_support.groupby(
            ["fold_id", "horizon_hours"],
            as_index=False,
        )
        .agg(
            min_receptor_fit_rows=("eligible_fit_rows", "min"),
            max_receptor_fit_rows=("eligible_fit_rows", "max"),
            min_receptor_positive_rows=("positive_fit_rows", "min"),
            max_receptor_positive_rows=("positive_fit_rows", "max"),
            min_receptor_balance_weight=("receptor_balance_weight", "min"),
            max_receptor_balance_weight=("receptor_balance_weight", "max"),
        )
    )

    report = {
        "version": "1.0",
        "overall_status": overall,
        "master_format": master_format,
        "master_rows": int(len(master)),
        "predictors": int(len(predictor_cols)),
        "folds": int(master["fold_id"].nunique()),
        "receptors": int(master["receptor_id"].nunique()),
        "horizons_hours": HORIZONS,
        "forbidden_predictor_rows": int(len(forbidden_rows)),
        "fit_zero_positive_receptor_rows": fit_zero_pos_receptor_rows,
        "constant_predictor_fold_rows": int(len(constant_rows)),
        "scikit_learn_available": sklearn_available,
        "xgboost_available": module_available("xgboost"),
        "lightgbm_available": module_available("lightgbm"),
        "catboost_available": module_available("catboost"),
        "test_model_scoring_performed": False,
        "test_used_for_model_selection": False,
        "random_split_allowed": False,
        "global_preprocessing_allowed": False,
        "primary_validation_metric": "AVERAGE_PRECISION",
        "final_uncertainty_policy":
            "SEASON_YEAR_OR_EVENT_CLUSTER_BOOTSTRAP__NOT_ROW_WISE",
        "next_step":
            "If PASS, implement foldwise baseline + candidate-model training using FIT only, "
            "select/calibrate on VALIDATION, freeze each horizon protocol, then score TEST once.",
        "warnings": warnings,
    }

    audit_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "=" * 216,
        "NW HYDROCLIMATE — FLOOD MODELING PROTOCOL PREFLIGHT v1.0",
        "=" * 216,
        f"OVERALL STATUS                         : {overall}",
        f"Master format                          : {master_format}",
        f"Master rows                            : {len(master)}",
        f"Predictors                             : {len(predictor_cols)}",
        f"Folds                                  : {master['fold_id'].nunique()}",
        f"Receptors                              : {master['receptor_id'].nunique()}",
        f"Forbidden predictor rows               : {len(forbidden_rows)}",
        f"FIT receptor/fold/horizon zero-positive: {fit_zero_pos_receptor_rows}",
        f"Constant predictor/fold rows           : {len(constant_rows)}",
        f"scikit-learn available                 : {sklearn_available}",
        f"xgboost available                      : {module_available('xgboost')}",
        "TEST model scoring performed           : False",
        "",
        "FIT CLASS SUPPORT",
        fit_support_compact.to_string(index=False),
        "",
        "FIT RECEPTOR AVAILABILITY RANGE",
        receptor_imbalance.to_string(index=False),
        "",
        "CONSTANT FIT PREDICTORS",
        (
            constant_rows[
                [
                    "fold_id",
                    "predictor",
                    "family",
                    "source",
                    "missing_fraction",
                ]
            ].to_string(index=False)
            if len(constant_rows)
            else "NONE"
        ),
        "",
        "MODEL ENVIRONMENT",
        environment.to_string(index=False),
        "",
        "CANDIDATE MODELS",
        candidates.to_string(index=False),
        "",
        "IMPORTANT",
        "No candidate model has been trained or scored on TEST.",
        "Feature preprocessing, if required, must be fitted inside FIT only.",
        "Model/weighting/hyperparameter choice is made on VALIDATION only.",
        "Final TEST evaluation is performed only after the protocol is frozen.",
        "Final confidence intervals must respect season/event clustering.",
        "",
        f"Output : {out}",
    ]

    audit_txt.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    progress(
        "PHASE 5/5",
        1,
        1,
        start5,
        f"status={overall}",
    )

    print("\n" + "=" * 216)
    print("\n".join(lines[3:]))
    print("=" * 216)
    print(f"OVERALL STATUS : {overall}")
    print(f"Output         : {out}")
    print("=" * 216)


if __name__ == "__main__":
    main()
