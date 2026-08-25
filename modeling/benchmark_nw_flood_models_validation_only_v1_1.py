#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmark_nw_flood_models_validation_only_v1_1.py

PRIMO VERO BENCHMARK DEI MODELLI, MA SOLO SU FIT + VALIDATION.

CORREZIONE METODOLOGICA IMPORTANTE
----------------------------------
I tre rolling fold hanno finestre concatenate:

F1  FIT <=2013 | VAL 2014-2016 | TEST 2017-2019
F2  FIT <=2016 | VAL 2017-2019 | TEST 2020-2022
F3  FIT <=2019 | VAL 2020-2022 | TEST 2023-2025

Quindi:
- il TEST F1 (2017-2019) coincide temporalmente con la VALIDATION F2;
- il TEST F2 (2020-2022) coincide temporalmente con la VALIDATION F3.

Perciò F1/F2 TEST NON possono essere presentati come holdout finali indipendenti
se le validation F2/F3 vengono usate per scegliere il modello.

POLICY DI SVILUPPO v1.1
-----------------------
- F1/F2/F3 FIT: training dei candidati.
- F1/F2/F3 VALIDATION: selezione di famiglia, iperparametri e weighting.
- F1/F2 TEST: NON vengono valutati in questo script.
- F3 TEST 2023-2025: unico FINAL HOLDOUT realmente intatto.
- Nessuna riga TEST viene usata per metriche o selezione.
- Dopo la selezione: freeze del candidato per ogni orizzonte.
- Solo nello step successivo:
    * fit finale su F3 FIT;
    * calibrazione / soglia su F3 VALIDATION;
    * una sola apertura di F3 TEST.

CANDIDATI
---------
B1 GLOBAL_FIT_CLIMATOLOGY
B2 RECEPTOR_SMOOTHED_FIT_CLIMATOLOGY
M1 LOGISTIC_ELASTICNET
M2 HIST_GRADIENT_BOOSTING

XGBoost resta sensitivity opzionale successiva; non serve per questo benchmark.

WEIGHTING DI TRAINING
---------------------
W1 NATURAL_ROW_WEIGHTING
W2 RECEPTOR_BALANCED_WEIGHTING

Non viene applicato class balancing in questa prima benchmark:
i FIT hanno già ~1,400-3,600 positivi pooled a seconda di fold/orizzonte.
Evitiamo di alterare artificialmente la calibrazione probabilistica prima di
aver verificato il comportamento base.

METRICA PRIMARIA DI SELEZIONE
-----------------------------
Average Precision / PR-AUC pooled sulla VALIDATION.

Diagnostiche:
- ROC-AUC
- Brier score
- log loss
- macro average precision per receptor quando il receptor ha entrambe le classi

La scelta automatica preliminare usa:
1) mean validation Average Precision sui 3 fold;
2) tie-break: Brier medio più basso.

NON è ancora il freeze finale del modello.

OUTPUT
------
nw_flood_model_validation_benchmark_v1_1/
  development_holdout_policy_v1_1.csv
  candidate_configuration_registry_v1_1.csv
  validation_candidate_metrics_by_fold_v1_1.csv
  validation_candidate_metrics_aggregate_v1_1.csv
  preliminary_selected_candidate_by_horizon_v1_1.csv
  fit_training_weight_audit_v1_1.csv
  benchmark_environment_v1_1.csv
  benchmark_audit_v1_1.json
  benchmark_audit_v1_1.txt
"""

from __future__ import annotations

import itertools
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn import __version__ as sklearn_version
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 20260825

HORIZONS = [24, 48, 72]
LABEL_COLS = {
    24: "label_extreme_within_24h",
    48: "label_extreme_within_48h",
    72: "label_extreme_within_72h",
}

EXPECTED_FOLDS = ["F1", "F2", "F3"]
EXPECTED_PREDICTORS = 97
EXPECTED_RECEPTORS = 20

B2_PRIOR_STRENGTHS = [50.0, 200.0]

LOGISTIC_GRID = [
    {"C": 0.1, "l1_ratio": 0.0},
    {"C": 0.1, "l1_ratio": 0.5},
    {"C": 1.0, "l1_ratio": 0.0},
    {"C": 1.0, "l1_ratio": 0.5},
]

HGB_GRID = [
    {
        "learning_rate": 0.05,
        "max_leaf_nodes": 15,
        "l2_regularization": 1.0,
    },
    {
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
    },
    {
        "learning_rate": 0.10,
        "max_leaf_nodes": 15,
        "l2_regularization": 1.0,
    },
    {
        "learning_rate": 0.10,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
    },
]

WEIGHTING_STRATEGIES = [
    "NATURAL_ROW_WEIGHTING",
    "RECEPTOR_BALANCED_WEIGHTING",
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
        f"| rate {rate:7.3f}/s | ETA {fmt_seconds(eta)}"
    )
    if current:
        msg += f" | {str(current)[:135]}"

    print(msg.ljust(285), end="", flush=True)
    if done >= total:
        print(flush=True)


def read_master(root: Path):
    parquet_p = (
        root
        / "nw_hydroclimate_foldwise_master_core_canonical_v1_0"
        / "nw_hydroclimate_foldwise_master_core_v1_0.parquet"
    )
    csv_p = (
        root
        / "nw_hydroclimate_foldwise_master_core_canonical_v1_0"
        / "nw_hydroclimate_foldwise_master_core_v1_0.csv.gz"
    )

    if parquet_p.exists():
        return pd.read_parquet(parquet_p), "PARQUET", parquet_p

    if csv_p.exists():
        return pd.read_csv(csv_p, low_memory=False), "CSV_GZIP", csv_p

    raise SystemExit("Master canonico v1.0 non trovato.")


def safe_metrics(y_true, p):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=float)

    p = np.clip(p, 1e-8, 1 - 1e-8)

    out = {
        "average_precision": np.nan,
        "roc_auc": np.nan,
        "brier_score": np.nan,
        "log_loss": np.nan,
    }

    if len(y) == 0:
        return out

    out["average_precision"] = float(
        average_precision_score(y, p)
    )
    out["brier_score"] = float(
        brier_score_loss(y, p)
    )
    out["log_loss"] = float(
        log_loss(y, p, labels=[0, 1])
    )

    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(
            roc_auc_score(y, p)
        )

    return out


def macro_receptor_ap(val_df, y, p):
    work = pd.DataFrame(
        {
            "receptor_id": val_df["receptor_id"].astype(str).to_numpy(),
            "y": np.asarray(y, dtype=int),
            "p": np.asarray(p, dtype=float),
        }
    )

    aps = []

    for _, g in work.groupby("receptor_id"):
        if g["y"].nunique() < 2:
            continue
        aps.append(
            float(
                average_precision_score(
                    g["y"].to_numpy(),
                    g["p"].to_numpy(),
                )
            )
        )

    return (
        float(np.mean(aps)) if aps else np.nan,
        int(len(aps)),
    )


def build_fit_weights(fit_df, strategy):
    n = len(fit_df)

    if strategy == "NATURAL_ROW_WEIGHTING":
        return np.ones(n, dtype=float)

    if strategy == "RECEPTOR_BALANCED_WEIGHTING":
        counts = (
            fit_df["receptor_id"]
            .astype(str)
            .value_counts()
            .to_dict()
        )

        r = len(counts)

        return np.asarray(
            [
                n / (r * counts[str(rec)])
                for rec in fit_df["receptor_id"].astype(str)
            ],
            dtype=float,
        )

    raise ValueError(strategy)


def global_climatology_predict(y_fit, n_val):
    p = float(np.mean(y_fit))
    return np.full(n_val, p, dtype=float)


def receptor_smoothed_predict(
    fit_df,
    y_fit,
    val_df,
    prior_strength,
):
    fit_meta = pd.DataFrame(
        {
            "receptor_id":
                fit_df["receptor_id"].astype(str).to_numpy(),
            "y": np.asarray(y_fit, dtype=int),
        }
    )

    global_p = float(fit_meta["y"].mean())
    alpha = global_p * prior_strength
    beta = (1.0 - global_p) * prior_strength

    stats = (
        fit_meta.groupby("receptor_id")["y"]
        .agg(["sum", "count"])
    )

    probs = {}

    for rid, row in stats.iterrows():
        probs[str(rid)] = (
            (float(row["sum"]) + alpha)
            / (float(row["count"]) + alpha + beta)
        )

    return np.asarray(
        [
            probs.get(str(rid), global_p)
            for rid in val_df["receptor_id"].astype(str)
        ],
        dtype=float,
    )


def build_logistic(C, l1_ratio):
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    C=float(C),
                    l1_ratio=float(l1_ratio),
                    max_iter=2000,
                    tol=1e-4,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_hgb(cfg):
    return HistGradientBoostingClassifier(
        learning_rate=float(cfg["learning_rate"]),
        max_leaf_nodes=int(cfg["max_leaf_nodes"]),
        max_iter=200,
        min_samples_leaf=20,
        l2_regularization=float(cfg["l2_regularization"]),
        early_stopping=False,
        random_state=RANDOM_STATE,
    )


def build_candidate_registry():
    rows = []

    rows.append(
        {
            "config_id": "B1_GLOBAL",
            "model_name": "GLOBAL_FIT_CLIMATOLOGY",
            "model_family": "BASELINE",
            "weighting_strategy": "NOT_APPLICABLE",
            "hyperparameters_json": "{}",
        }
    )

    for k in B2_PRIOR_STRENGTHS:
        rows.append(
            {
                "config_id": f"B2_RECEPTOR_K{int(k)}",
                "model_name": "RECEPTOR_SMOOTHED_FIT_CLIMATOLOGY",
                "model_family": "BASELINE",
                "weighting_strategy": "NOT_APPLICABLE",
                "hyperparameters_json":
                    json.dumps({"prior_strength": k}),
            }
        )

    for weighting in WEIGHTING_STRATEGIES:
        wcode = (
            "W1"
            if weighting == "NATURAL_ROW_WEIGHTING"
            else "W2"
        )

        for cfg in LOGISTIC_GRID:
            cid = (
                f"M1_LOGIT_{wcode}"
                f"_C{str(cfg['C']).replace('.', 'p')}"
                f"_L1R{str(cfg['l1_ratio']).replace('.', 'p')}"
            )
            rows.append(
                {
                    "config_id": cid,
                    "model_name": "LOGISTIC_ELASTICNET",
                    "model_family": "INTERPRETABLE_LINEAR",
                    "weighting_strategy": weighting,
                    "hyperparameters_json": json.dumps(cfg),
                }
            )

        for cfg in HGB_GRID:
            cid = (
                f"M2_HGB_{wcode}"
                f"_LR{str(cfg['learning_rate']).replace('.', 'p')}"
                f"_LEAF{cfg['max_leaf_nodes']}"
                f"_L2{str(cfg['l2_regularization']).replace('.', 'p')}"
            )
            rows.append(
                {
                    "config_id": cid,
                    "model_name": "HIST_GRADIENT_BOOSTING",
                    "model_family": "PRIMARY_NONLINEAR",
                    "weighting_strategy": weighting,
                    "hyperparameters_json": json.dumps(cfg),
                }
            )

    return pd.DataFrame(rows)


def main():
    root = Path(__file__).resolve().parent

    master, master_format, master_path = read_master(root)

    dictionary_p = (
        root
        / "nw_hydroclimate_foldwise_master_core_canonical_v1_0"
        / "nw_hydroclimate_master_predictor_dictionary_v1_0.csv"
    )

    if not dictionary_p.exists():
        raise SystemExit(f"Manca: {dictionary_p}")

    dictionary = pd.read_csv(dictionary_p, low_memory=False)
    predictor_cols = dictionary["predictor"].astype(str).tolist()

    if len(predictor_cols) != EXPECTED_PREDICTORS:
        raise SystemExit(
            f"Predictor count={len(predictor_cols)}, "
            f"expected={EXPECTED_PREDICTORS}"
        )

    master["issue_date"] = pd.to_datetime(
        master["issue_date"],
        errors="coerce",
    )

    out = root / "nw_flood_model_validation_benchmark_v1_1"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 220)
    print("NW HYDROCLIMATE — VALIDATION-ONLY FLOOD MODEL BENCHMARK v1.1")
    print("=" * 220)
    print(
        "FINAL HOLDOUT POLICY: only F3 TEST 2023-2025 is a true untouched final holdout.",
        flush=True,
    )
    print(
        "This script does NOT score any TEST partition.",
        flush=True,
    )

    # ------------------------------------------------------------------
    # PHASE 1/5 — integrity and holdout policy
    # ------------------------------------------------------------------
    print("\nPHASE 1/5 — master integrity and final-holdout policy")
    start1 = time.time()

    errors = []

    if master["fold_id"].nunique() != 3:
        errors.append("UNEXPECTED_FOLD_COUNT")

    if master["receptor_id"].nunique() != EXPECTED_RECEPTORS:
        errors.append("UNEXPECTED_RECEPTOR_COUNT")

    if master.duplicated(
        ["fold_id", "receptor_id", "issue_date"]
    ).any():
        errors.append("DUPLICATE_MASTER_KEYS")

    if errors:
        raise SystemExit(" | ".join(errors))

    holdout_policy = pd.DataFrame(
        [
            {
                "period": "2014-2016",
                "development_role": "VALIDATION_F1",
                "final_holdout": False,
                "reason": "MODEL_SELECTION",
            },
            {
                "period": "2017-2019",
                "development_role": "VALIDATION_F2",
                "final_holdout": False,
                "reason":
                    "COINCIDES_WITH_F1_TEST__THEREFORE_NOT_INDEPENDENT_FINAL_TEST",
            },
            {
                "period": "2020-2022",
                "development_role": "VALIDATION_F3",
                "final_holdout": False,
                "reason":
                    "COINCIDES_WITH_F2_TEST__THEREFORE_NOT_INDEPENDENT_FINAL_TEST",
            },
            {
                "period": "2023-2025",
                "development_role": "SEALED_FINAL_HOLDOUT",
                "final_holdout": True,
                "reason": "F3_TEST__NEVER_USED_FOR_MODEL_SELECTION",
            },
        ]
    )

    progress(
        "PHASE 1/5",
        1,
        1,
        start1,
        f"format={master_format} predictors={len(predictor_cols)}",
    )

    # ------------------------------------------------------------------
    # PHASE 2/5 — candidate registry
    # ------------------------------------------------------------------
    print("\nPHASE 2/5 — freeze development candidate grid before fitting")
    start2 = time.time()

    registry = build_candidate_registry()

    progress(
        "PHASE 2/5",
        1,
        1,
        start2,
        f"candidate configurations={len(registry)}",
    )

    # ------------------------------------------------------------------
    # PHASE 3/5 — validation-only benchmark
    # ------------------------------------------------------------------
    print("\nPHASE 3/5 — FIT candidates and score VALIDATION only")
    start3 = time.time()

    tasks = list(
        itertools.product(
            EXPECTED_FOLDS,
            HORIZONS,
            registry["config_id"].astype(str).tolist(),
        )
    )

    results = []
    weight_audit_rows = []

    warnings.filterwarnings(
        "ignore",
        category=ConvergenceWarning,
    )

    for task_idx, (fold_id, horizon, config_id) in enumerate(
        tasks,
        1,
    ):
        cfg_row = registry[
            registry["config_id"].astype(str).eq(config_id)
        ].iloc[0]

        label_col = LABEL_COLS[horizon]

        fold = master[
            master["fold_id"].astype(str).eq(fold_id)
        ]

        fit = fold[
            fold["partition"].astype(str).eq("FIT")
        ].copy()

        val = fold[
            fold["partition"].astype(str).eq("VALIDATION")
        ].copy()

        # TEST rows are deliberately ignored.
        y_fit_all = pd.to_numeric(
            fit[label_col],
            errors="coerce",
        )
        y_val_all = pd.to_numeric(
            val[label_col],
            errors="coerce",
        )

        fit = fit.loc[y_fit_all.notna()].copy()
        val = val.loc[y_val_all.notna()].copy()

        y_fit = (
            y_fit_all.loc[y_fit_all.notna()]
            .astype(int)
            .to_numpy()
        )
        y_val = (
            y_val_all.loc[y_val_all.notna()]
            .astype(int)
            .to_numpy()
        )

        model_name = str(cfg_row["model_name"])
        weighting = str(cfg_row["weighting_strategy"])
        hyper = json.loads(
            str(cfg_row["hyperparameters_json"])
        )

        status = "PASS"
        error = ""

        try:
            if model_name == "GLOBAL_FIT_CLIMATOLOGY":
                p_val = global_climatology_predict(
                    y_fit,
                    len(val),
                )

            elif model_name == "RECEPTOR_SMOOTHED_FIT_CLIMATOLOGY":
                p_val = receptor_smoothed_predict(
                    fit,
                    y_fit,
                    val,
                    prior_strength=float(
                        hyper["prior_strength"]
                    ),
                )

            elif model_name == "LOGISTIC_ELASTICNET":
                weights = build_fit_weights(
                    fit,
                    weighting,
                )

                weight_audit_rows.append(
                    {
                        "fold_id": fold_id,
                        "horizon_hours": horizon,
                        "config_id": config_id,
                        "weighting_strategy": weighting,
                        "fit_rows": int(len(fit)),
                        "weight_sum": float(weights.sum()),
                        "weight_min": float(weights.min()),
                        "weight_max": float(weights.max()),
                    }
                )

                model = build_logistic(
                    C=hyper["C"],
                    l1_ratio=hyper["l1_ratio"],
                )

                X_fit = fit[predictor_cols]
                X_val = val[predictor_cols]

                model.fit(
                    X_fit,
                    y_fit,
                    clf__sample_weight=weights,
                )

                p_val = model.predict_proba(X_val)[:, 1]

            elif model_name == "HIST_GRADIENT_BOOSTING":
                weights = build_fit_weights(
                    fit,
                    weighting,
                )

                weight_audit_rows.append(
                    {
                        "fold_id": fold_id,
                        "horizon_hours": horizon,
                        "config_id": config_id,
                        "weighting_strategy": weighting,
                        "fit_rows": int(len(fit)),
                        "weight_sum": float(weights.sum()),
                        "weight_min": float(weights.min()),
                        "weight_max": float(weights.max()),
                    }
                )

                model = build_hgb(hyper)

                X_fit = fit[predictor_cols].apply(
                    pd.to_numeric,
                    errors="coerce",
                )
                X_val = val[predictor_cols].apply(
                    pd.to_numeric,
                    errors="coerce",
                )

                model.fit(
                    X_fit,
                    y_fit,
                    sample_weight=weights,
                )

                p_val = model.predict_proba(X_val)[:, 1]

            else:
                raise RuntimeError(
                    f"Unknown model_name={model_name}"
                )

            metrics = safe_metrics(
                y_val,
                p_val,
            )

            macro_ap, macro_n = macro_receptor_ap(
                val,
                y_val,
                p_val,
            )

        except Exception as exc:
            status = "FAIL"
            error = f"{type(exc).__name__}: {exc}"
            metrics = {
                "average_precision": np.nan,
                "roc_auc": np.nan,
                "brier_score": np.nan,
                "log_loss": np.nan,
            }
            macro_ap = np.nan
            macro_n = 0

        results.append(
            {
                "fold_id": fold_id,
                "horizon_hours": horizon,
                "config_id": config_id,
                "model_name": model_name,
                "model_family":
                    str(cfg_row["model_family"]),
                "weighting_strategy": weighting,
                "fit_rows": int(len(fit)),
                "fit_positive_rows":
                    int(np.sum(y_fit == 1)),
                "validation_rows": int(len(val)),
                "validation_positive_rows":
                    int(np.sum(y_val == 1)),
                **metrics,
                "macro_receptor_average_precision":
                    macro_ap,
                "macro_receptor_ap_receptors":
                    macro_n,
                "status": status,
                "error": error,
                "test_rows_scored": 0,
            }
        )

        progress(
            "PHASE 3/5",
            task_idx,
            len(tasks),
            start3,
            (
                f"{fold_id} | {horizon}h | {config_id} "
                f"| AP={metrics['average_precision']:.4f}"
                if np.isfinite(metrics["average_precision"])
                else f"{fold_id} | {horizon}h | {config_id} | {status}"
            ),
        )

    metrics_by_fold = pd.DataFrame(results)
    weight_audit = pd.DataFrame(weight_audit_rows)

    # ------------------------------------------------------------------
    # PHASE 4/5 — aggregate validation ranking
    # ------------------------------------------------------------------
    print("\nPHASE 4/5 — aggregate VALIDATION ranking across folds")
    start4 = time.time()

    passing = metrics_by_fold[
        metrics_by_fold["status"].eq("PASS")
    ].copy()

    aggregate = (
        passing.groupby(
            [
                "horizon_hours",
                "config_id",
                "model_name",
                "model_family",
                "weighting_strategy",
            ],
            as_index=False,
        )
        .agg(
            validation_folds=("fold_id", "nunique"),
            mean_average_precision=(
                "average_precision",
                "mean",
            ),
            min_average_precision=(
                "average_precision",
                "min",
            ),
            mean_macro_receptor_average_precision=(
                "macro_receptor_average_precision",
                "mean",
            ),
            mean_roc_auc=("roc_auc", "mean"),
            mean_brier_score=("brier_score", "mean"),
            mean_log_loss=("log_loss", "mean"),
        )
    )

    aggregate["eligible_for_selection"] = (
        aggregate["validation_folds"].eq(3)
    )

    aggregate["selection_rank"] = np.nan

    selected_rows = []

    for h in HORIZONS:
        cand = aggregate[
            aggregate["horizon_hours"].eq(h)
            & aggregate["eligible_for_selection"]
        ].copy()

        cand = cand.sort_values(
            [
                "mean_average_precision",
                "mean_brier_score",
            ],
            ascending=[False, True],
        ).reset_index()

        if len(cand):
            for rank, idx in enumerate(
                cand["index"],
                1,
            ):
                aggregate.loc[
                    idx,
                    "selection_rank",
                ] = rank

            best = cand.iloc[0]

            selected_rows.append(
                {
                    "horizon_hours": h,
                    "preliminary_selected_config_id":
                        best["config_id"],
                    "model_name": best["model_name"],
                    "weighting_strategy":
                        best["weighting_strategy"],
                    "mean_validation_average_precision":
                        best["mean_average_precision"],
                    "mean_validation_macro_receptor_ap":
                        best[
                            "mean_macro_receptor_average_precision"
                        ],
                    "mean_validation_brier_score":
                        best["mean_brier_score"],
                    "selection_status":
                        "PRELIMINARY_VALIDATION_ONLY__NOT_FINAL_FREEZE",
                }
            )

    selected = pd.DataFrame(selected_rows)

    progress(
        "PHASE 4/5",
        1,
        1,
        start4,
        f"selected horizons={len(selected)}",
    )

    # ------------------------------------------------------------------
    # PHASE 5/5 — write outputs and audit
    # ------------------------------------------------------------------
    print("\nPHASE 5/5 — write validation-only benchmark artifacts")
    start5 = time.time()

    holdout_p = (
        out / "development_holdout_policy_v1_1.csv"
    )
    registry_p = (
        out / "candidate_configuration_registry_v1_1.csv"
    )
    fold_metrics_p = (
        out / "validation_candidate_metrics_by_fold_v1_1.csv"
    )
    aggregate_p = (
        out / "validation_candidate_metrics_aggregate_v1_1.csv"
    )
    selected_p = (
        out / "preliminary_selected_candidate_by_horizon_v1_1.csv"
    )
    weight_p = (
        out / "fit_training_weight_audit_v1_1.csv"
    )
    environment_p = (
        out / "benchmark_environment_v1_1.csv"
    )
    audit_json = (
        out / "benchmark_audit_v1_1.json"
    )
    audit_txt = (
        out / "benchmark_audit_v1_1.txt"
    )

    holdout_policy.to_csv(
        holdout_p,
        index=False,
    )
    registry.to_csv(
        registry_p,
        index=False,
    )
    metrics_by_fold.to_csv(
        fold_metrics_p,
        index=False,
    )
    aggregate.sort_values(
        ["horizon_hours", "selection_rank"],
        na_position="last",
    ).to_csv(
        aggregate_p,
        index=False,
    )
    selected.to_csv(
        selected_p,
        index=False,
    )
    weight_audit.to_csv(
        weight_p,
        index=False,
    )

    environment = pd.DataFrame(
        [
            {
                "component": "master_format",
                "value": master_format,
            },
            {
                "component": "master_path",
                "value": str(master_path),
            },
            {
                "component": "scikit_learn_version",
                "value": sklearn_version,
            },
            {
                "component": "random_state",
                "value": RANDOM_STATE,
            },
            {
                "component": "test_scoring_performed",
                "value": False,
            },
            {
                "component": "true_final_holdout",
                "value": "F3_TEST_2023_2025",
            },
        ]
    )
    environment.to_csv(
        environment_p,
        index=False,
    )

    failed_tasks = int(
        metrics_by_fold["status"].eq("FAIL").sum()
    )

    if failed_tasks:
        overall = "PASS_WITH_CANDIDATE_FIT_FAILURES_REVIEW"
    elif len(selected) != len(HORIZONS):
        overall = "FAIL_SELECTION_INCOMPLETE"
    else:
        overall = "PASS_VALIDATION_ONLY__TEST_UNTOUCHED"

    audit = {
        "version": "1.1",
        "overall_status": overall,
        "master_format": master_format,
        "candidate_configurations": int(len(registry)),
        "benchmark_tasks": int(len(tasks)),
        "candidate_fit_failures": failed_tasks,
        "horizons_selected": int(len(selected)),
        "test_model_scoring_performed": False,
        "f1_test_used_for_final_claims": False,
        "f2_test_used_for_final_claims": False,
        "true_final_holdout":
            "F3 TEST 2023-2025",
        "primary_selection_metric":
            "MEAN_VALIDATION_AVERAGE_PRECISION_ACROSS_F1_F2_F3",
        "selection_tiebreak":
            "LOWER_MEAN_VALIDATION_BRIER_SCORE",
        "class_weighting_used": False,
        "training_weighting_compared": WEIGHTING_STRATEGIES,
        "global_preprocessing_performed": False,
        "final_model_frozen": False,
        "next_step":
            (
                "Review validation rankings. Then freeze one candidate per horizon. "
                "Fit the frozen candidate on F3 FIT, calibrate/select decision threshold "
                "on F3 VALIDATION, and only then score F3 TEST 2023-2025 once."
            ),
    }

    audit_json.write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    top3 = (
        aggregate[
            aggregate["eligible_for_selection"]
        ]
        .sort_values(
            [
                "horizon_hours",
                "selection_rank",
            ]
        )
        .groupby(
            "horizon_hours",
            as_index=False,
        )
        .head(3)
    )

    lines = [
        "=" * 220,
        "NW HYDROCLIMATE — VALIDATION-ONLY FLOOD MODEL BENCHMARK v1.1",
        "=" * 220,
        f"OVERALL STATUS                   : {overall}",
        f"Candidate configurations         : {len(registry)}",
        f"Benchmark tasks                  : {len(tasks)}",
        f"Candidate fit failures           : {failed_tasks}",
        "TEST model scoring performed    : False",
        "True final holdout              : F3 TEST 2023-2025",
        "",
        "PRELIMINARY SELECTED CANDIDATE BY HORIZON",
        (
            selected.to_string(index=False)
            if len(selected)
            else "NONE"
        ),
        "",
        "TOP 3 VALIDATION CANDIDATES PER HORIZON",
        (
            top3[
                [
                    "horizon_hours",
                    "selection_rank",
                    "config_id",
                    "model_name",
                    "weighting_strategy",
                    "mean_average_precision",
                    "mean_macro_receptor_average_precision",
                    "mean_roc_auc",
                    "mean_brier_score",
                    "mean_log_loss",
                ]
            ].to_string(index=False)
            if len(top3)
            else "NONE"
        ),
        "",
        "IMPORTANT",
        "F1/F2 TEST are not used for final claims because their periods reappear as later VALIDATION windows.",
        "Only F3 TEST 2023-2025 remains a true final holdout.",
        "No TEST metrics are calculated in this benchmark.",
        "The selected rows are preliminary development choices, not yet final frozen models.",
        "",
        f"Metrics by fold : {fold_metrics_p}",
        f"Aggregate       : {aggregate_p}",
        f"Selected        : {selected_p}",
        f"Holdout policy  : {holdout_p}",
        f"Output          : {out}",
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

    print("\n" + "=" * 220)
    print("\n".join(lines[3:]))
    print("=" * 220)
    print(f"OVERALL STATUS : {overall}")
    print(f"Output         : {out}")
    print("=" * 220)


if __name__ == "__main__":
    main()
