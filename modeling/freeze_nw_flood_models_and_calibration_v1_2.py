#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
freeze_nw_flood_models_and_calibration_v1_2.py

Congela i candidati selezionati dal benchmark validation-only v1.1,
rigenera predizioni SOLO sulle VALIDATION F1/F2/F3, stima calibrazione
Platt pooled 2014-2022, sceglie soglie di sviluppo e rifitta i modelli
base definitivi sul SOLO F3 FIT (<=2019).

NON genera né valuta predizioni sul FINAL HOLDOUT F3 TEST 2023-2025.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
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

EXPECTED_SELECTED = {
    24: "M2_HGB_W2_LR0p1_LEAF31_L21p0",
    48: "M2_HGB_W2_LR0p1_LEAF31_L21p0",
    72: "M2_HGB_W1_LR0p05_LEAF15_L21p0",
}


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
        msg += f" | {str(current)[:135]}"
    print(msg.ljust(285), end="", flush=True)
    if done >= total:
        print(flush=True)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_master(root):
    p = (
        root
        / "nw_hydroclimate_foldwise_master_core_canonical_v1_0"
        / "nw_hydroclimate_foldwise_master_core_v1_0.parquet"
    )
    if not p.exists():
        raise SystemExit(f"Manca master Parquet: {p}")
    return pd.read_parquet(p), p


def build_fit_weights(fit_df, strategy):
    n = len(fit_df)
    if strategy == "NATURAL_ROW_WEIGHTING":
        return np.ones(n, dtype=float)

    if strategy == "RECEPTOR_BALANCED_WEIGHTING":
        counts = fit_df["receptor_id"].astype(str).value_counts().to_dict()
        r = len(counts)
        return np.asarray(
            [n / (r * counts[str(rec)]) for rec in fit_df["receptor_id"].astype(str)],
            dtype=float,
        )

    raise ValueError(f"Unknown weighting: {strategy}")


def build_model(model_name, hyper):
    if model_name == "HIST_GRADIENT_BOOSTING":
        return HistGradientBoostingClassifier(
            learning_rate=float(hyper["learning_rate"]),
            max_leaf_nodes=int(hyper["max_leaf_nodes"]),
            max_iter=200,
            min_samples_leaf=20,
            l2_regularization=float(hyper["l2_regularization"]),
            early_stopping=False,
            random_state=RANDOM_STATE,
        )

    if model_name == "LOGISTIC_ELASTICNET":
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
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        solver="saga",
                        C=float(hyper["C"]),
                        l1_ratio=float(hyper["l1_ratio"]),
                        max_iter=2000,
                        tol=1e-4,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unsupported selected model: {model_name}")


def fit_model(model, X, y, weights):
    if isinstance(model, Pipeline):
        model.fit(X, y, clf__sample_weight=weights)
    else:
        model.fit(X, y, sample_weight=weights)
    return model


def predict_prob(model, X):
    return model.predict_proba(X)[:, 1]


def safe_metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    return {
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p)).reshape(-1, 1)


def fit_platt(raw_prob, y):
    cal = LogisticRegression(
        solver="lbfgs",
        C=1e6,
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    cal.fit(logit(raw_prob), np.asarray(y, dtype=int))
    return cal


def apply_platt(cal, raw_prob):
    return cal.predict_proba(logit(raw_prob))[:, 1]


def threshold_table(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    grid = np.linspace(0.001, 0.999, 999)
    q = np.quantile(p, np.linspace(0.01, 0.99, 99))
    thresholds = np.unique(np.clip(np.concatenate([grid, q]), 0.0, 1.0))

    rows = []

    for t in thresholds:
        pred = (p >= t).astype(int)
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        tn = int(np.sum((pred == 0) & (y == 0)))

        csi_denom = tp + fp + fn
        csi = tp / csi_denom if csi_denom else np.nan
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan

        rows.append(
            {
                "threshold": float(t),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "csi": csi,
                "precision": precision,
                "recall": recall,
            }
        )

    return pd.DataFrame(rows)


def choose_thresholds(tbl):
    valid = tbl[np.isfinite(tbl["csi"])].copy()

    max_csi = (
        valid.sort_values(
            ["csi", "recall", "precision"],
            ascending=[False, False, False],
        )
        .iloc[0]
    )

    recall80 = valid[
        valid["recall"].ge(0.80) & np.isfinite(valid["precision"])
    ].copy()

    if len(recall80):
        recall80 = (
            recall80.sort_values(
                ["precision", "csi", "threshold"],
                ascending=[False, False, False],
            )
            .iloc[0]
        )
    else:
        recall80 = max_csi

    return max_csi, recall80


def main():
    root = Path(__file__).resolve().parent

    bench = root / "nw_flood_model_validation_benchmark_v1_1"

    audit_p = bench / "benchmark_audit_v1_1.json"
    registry_p = bench / "candidate_configuration_registry_v1_1.csv"
    selected_p = bench / "preliminary_selected_candidate_by_horizon_v1_1.csv"
    holdout_p = bench / "development_holdout_policy_v1_1.csv"

    dictionary_p = (
        root
        / "nw_hydroclimate_foldwise_master_core_canonical_v1_0"
        / "nw_hydroclimate_master_predictor_dictionary_v1_0.csv"
    )

    for p in [audit_p, registry_p, selected_p, holdout_p, dictionary_p]:
        if not p.exists():
            raise SystemExit(f"Manca: {p}")

    out = root / "nw_flood_models_frozen_development_v1_2"
    models_dir = out / "models"
    out.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 220)
    print("NW HYDROCLIMATE — FREEZE FLOOD MODELS + DEVELOPMENT CALIBRATION v1.2")
    print("=" * 220)
    print("F3 TEST 2023-2025 remains sealed. No TEST prediction is produced.", flush=True)

    print("\nPHASE 1/6 — validate benchmark selection and holdout policy")
    start = time.time()

    audit = json.loads(audit_p.read_text(encoding="utf-8"))
    registry = pd.read_csv(registry_p, low_memory=False)
    selected = pd.read_csv(selected_p, low_memory=False)
    holdout = pd.read_csv(holdout_p, low_memory=False)
    dictionary = pd.read_csv(dictionary_p, low_memory=False)

    if audit.get("overall_status") != "PASS_VALIDATION_ONLY__TEST_UNTOUCHED":
        raise SystemExit(f"Benchmark non congelabile: {audit.get('overall_status')}")

    if bool(audit.get("test_model_scoring_performed")):
        raise SystemExit("Benchmark unexpectedly scored TEST.")

    selected_map = {
        int(r["horizon_hours"]): str(r["preliminary_selected_config_id"])
        for _, r in selected.iterrows()
    }

    if selected_map != EXPECTED_SELECTED:
        raise SystemExit(
            "Selected configs differ from benchmark expected values.\n"
            f"Observed: {selected_map}\nExpected: {EXPECTED_SELECTED}"
        )

    true_holdout = holdout[
        holdout["final_holdout"].astype(str).str.lower().isin({"true", "1"})
    ]

    if len(true_holdout) != 1 or str(true_holdout.iloc[0]["period"]) != "2023-2025":
        raise SystemExit("Final holdout policy is not uniquely 2023-2025.")

    predictor_cols = dictionary["predictor"].astype(str).tolist()

    progress("PHASE 1/6", 1, 1, start, "benchmark PASS | configs frozen")

    print("\nPHASE 2/6 — load master and isolate FIT/VALIDATION")
    start = time.time()

    master, master_p = read_master(root)
    master["issue_date"] = pd.to_datetime(master["issue_date"], errors="coerce")

    development = master[
        master["partition"].astype(str).isin(["FIT", "VALIDATION"])
    ].copy()

    test_rows_present = int(
        master["partition"].astype(str).eq("TEST").sum()
    )

    progress(
        "PHASE 2/6",
        1,
        1,
        start,
        f"development_rows={len(development)} | TEST rows not scored={test_rows_present}",
    )

    print("\nPHASE 3/6 — refit frozen configs and predict VALIDATION only")
    start = time.time()

    oof_rows = []
    raw_metric_rows = []
    registry_idx = registry.set_index("config_id")

    total = 9
    done = 0

    for horizon in HORIZONS:
        config_id = selected_map[horizon]
        cfg = registry_idx.loc[config_id]

        model_name = str(cfg["model_name"])
        weighting = str(cfg["weighting_strategy"])
        hyper = json.loads(str(cfg["hyperparameters_json"]))

        for fold_id in ["F1", "F2", "F3"]:
            done += 1

            fold = development[
                development["fold_id"].astype(str).eq(fold_id)
            ]

            fit = fold[fold["partition"].astype(str).eq("FIT")].copy()
            val = fold[fold["partition"].astype(str).eq("VALIDATION")].copy()

            label_col = LABEL_COLS[horizon]

            y_fit_all = pd.to_numeric(fit[label_col], errors="coerce")
            y_val_all = pd.to_numeric(val[label_col], errors="coerce")

            fit = fit.loc[y_fit_all.notna()].copy()
            val = val.loc[y_val_all.notna()].copy()

            y_fit = y_fit_all.loc[y_fit_all.notna()].astype(int).to_numpy()
            y_val = y_val_all.loc[y_val_all.notna()].astype(int).to_numpy()

            weights = build_fit_weights(fit, weighting)

            model = build_model(model_name, hyper)

            X_fit = fit[predictor_cols].apply(pd.to_numeric, errors="coerce")
            X_val = val[predictor_cols].apply(pd.to_numeric, errors="coerce")

            model = fit_model(model, X_fit, y_fit, weights)
            p_val = predict_prob(model, X_val)

            m = safe_metrics(y_val, p_val)

            raw_metric_rows.append(
                {
                    "horizon_hours": horizon,
                    "fold_id": fold_id,
                    "config_id": config_id,
                    "model_name": model_name,
                    "weighting_strategy": weighting,
                    "validation_rows": int(len(val)),
                    "validation_positive_rows": int(np.sum(y_val == 1)),
                    **m,
                }
            )

            temp = pd.DataFrame(
                {
                    "horizon_hours": horizon,
                    "fold_id": fold_id,
                    "receptor_id": val["receptor_id"].astype(str).to_numpy(),
                    "issue_date": val["issue_date"].to_numpy(),
                    "y_true": y_val,
                    "raw_probability": p_val,
                }
            )
            oof_rows.append(temp)

            progress(
                "PHASE 3/6",
                done,
                total,
                start,
                f"{horizon}h | {fold_id} | AP={m['average_precision']:.4f}",
            )

    oof = pd.concat(oof_rows, ignore_index=True)
    raw_metrics = pd.DataFrame(raw_metric_rows)

    print("\nPHASE 4/6 — fit pooled OOF Platt calibration and thresholds")
    start = time.time()

    calibration_rows = []
    threshold_rows = []
    calibrators = {}

    for i, horizon in enumerate(HORIZONS, 1):
        h = oof[oof["horizon_hours"].eq(horizon)].copy()

        y = h["y_true"].astype(int).to_numpy()
        raw_p = h["raw_probability"].astype(float).to_numpy()

        cal = fit_platt(raw_p, y)
        calibrated_p = apply_platt(cal, raw_p)

        calibrators[horizon] = cal

        oof.loc[
            oof["horizon_hours"].eq(horizon),
            "calibrated_probability",
        ] = calibrated_p

        raw_m = safe_metrics(y, raw_p)
        cal_m = safe_metrics(y, calibrated_p)

        calibration_rows.append(
            {
                "horizon_hours": horizon,
                "development_validation_rows": int(len(h)),
                "development_positive_rows": int(np.sum(y == 1)),
                "raw_average_precision": raw_m["average_precision"],
                "calibrated_average_precision": cal_m["average_precision"],
                "raw_brier_score": raw_m["brier_score"],
                "calibrated_brier_score": cal_m["brier_score"],
                "raw_log_loss": raw_m["log_loss"],
                "calibrated_log_loss": cal_m["log_loss"],
                "platt_intercept": float(cal.intercept_[0]),
                "platt_coefficient": float(cal.coef_[0, 0]),
                "diagnostic_scope":
                    "DEVELOPMENT_ONLY__CALIBRATOR_FIT_ON_SAME_OOF_VALIDATION_POOL",
            }
        )

        tbl = threshold_table(y, calibrated_p)
        max_csi, recall80 = choose_thresholds(tbl)

        for policy_name, row in [
            ("MAX_CSI", max_csi),
            ("RECALL80_MAX_PRECISION", recall80),
        ]:
            threshold_rows.append(
                {
                    "horizon_hours": horizon,
                    "threshold_policy": policy_name,
                    "threshold": float(row["threshold"]),
                    "development_csi": float(row["csi"]),
                    "development_precision": float(row["precision"]),
                    "development_recall": float(row["recall"]),
                    "development_tp": int(row["tp"]),
                    "development_fp": int(row["fp"]),
                    "development_fn": int(row["fn"]),
                    "development_tn": int(row["tn"]),
                    "metric_scope":
                        "DEVELOPMENT_VALIDATION_ONLY__NOT_FINAL_PERFORMANCE",
                }
            )

        progress(
            "PHASE 4/6",
            i,
            3,
            start,
            f"{horizon}h | Brier raw={raw_m['brier_score']:.4f} cal={cal_m['brier_score']:.4f}",
        )

    calibration_diag = pd.DataFrame(calibration_rows)
    thresholds = pd.DataFrame(threshold_rows)

    print("\nPHASE 5/6 — fit final base models on F3 FIT only")
    start = time.time()

    frozen_rows = []
    model_files = []
    calibrator_files = []

    for i, horizon in enumerate(HORIZONS, 1):
        config_id = selected_map[horizon]
        cfg = registry_idx.loc[config_id]

        model_name = str(cfg["model_name"])
        weighting = str(cfg["weighting_strategy"])
        hyper = json.loads(str(cfg["hyperparameters_json"]))

        f3fit = development[
            development["fold_id"].astype(str).eq("F3")
            & development["partition"].astype(str).eq("FIT")
        ].copy()

        label_col = LABEL_COLS[horizon]
        y_all = pd.to_numeric(f3fit[label_col], errors="coerce")
        f3fit = f3fit.loc[y_all.notna()].copy()
        y = y_all.loc[y_all.notna()].astype(int).to_numpy()

        weights = build_fit_weights(f3fit, weighting)

        X = f3fit[predictor_cols].apply(pd.to_numeric, errors="coerce")

        base_model = build_model(model_name, hyper)
        base_model = fit_model(base_model, X, y, weights)

        model_p = models_dir / f"horizon_{horizon}h_base_model.joblib"
        cal_p = models_dir / f"horizon_{horizon}h_platt_calibrator.joblib"

        joblib.dump(base_model, model_p)
        joblib.dump(calibrators[horizon], cal_p)

        model_files.append(model_p)
        calibrator_files.append(cal_p)

        th = thresholds[thresholds["horizon_hours"].eq(horizon)]

        max_csi_t = float(
            th.loc[th["threshold_policy"].eq("MAX_CSI"), "threshold"].iloc[0]
        )
        recall80_t = float(
            th.loc[
                th["threshold_policy"].eq("RECALL80_MAX_PRECISION"),
                "threshold",
            ].iloc[0]
        )

        frozen_rows.append(
            {
                "horizon_hours": horizon,
                "frozen_config_id": config_id,
                "model_name": model_name,
                "weighting_strategy": weighting,
                "hyperparameters_json": json.dumps(hyper, sort_keys=True),
                "base_model_training_partition": "F3_FIT_ONLY__THROUGH_2019",
                "calibration_method": "PLATT_ON_POOLED_OOF_VALIDATION_2014_2022",
                "threshold_max_csi": max_csi_t,
                "threshold_recall80_max_precision": recall80_t,
                "final_holdout": "F3_TEST_2023_2025__SEALED",
                "test_predictions_generated": False,
            }
        )

        progress(
            "PHASE 5/6",
            i,
            3,
            start,
            f"{horizon}h | {config_id} | saved",
        )

    frozen = pd.DataFrame(frozen_rows)

    print("\nPHASE 6/6 — write frozen artifacts and audit")
    start = time.time()

    frozen_p = out / "frozen_model_protocol_v1_2.csv"
    oof_p = out / "oof_validation_predictions_v1_2.parquet"
    raw_metrics_p = out / "oof_validation_metrics_raw_v1_2.csv"
    cal_diag_p = out / "oof_validation_calibration_diagnostics_v1_2.csv"
    thresholds_p = out / "development_threshold_registry_v1_2.csv"
    audit_json = out / "freeze_audit_v1_2.json"
    audit_txt = out / "freeze_audit_v1_2.txt"

    frozen.to_csv(frozen_p, index=False)
    oof.to_parquet(oof_p, index=False)
    raw_metrics.to_csv(raw_metrics_p, index=False)
    calibration_diag.to_csv(cal_diag_p, index=False)
    thresholds.to_csv(thresholds_p, index=False)

    if pd.to_datetime(oof["issue_date"]).dt.year.max() > 2022:
        raise SystemExit("OOF validation contains dates after 2022.")

    files_for_hash = [
        frozen_p,
        oof_p,
        raw_metrics_p,
        cal_diag_p,
        thresholds_p,
        *model_files,
        *calibrator_files,
    ]

    checksums = pd.DataFrame(
        [
            {
                "file": str(p.relative_to(out)),
                "sha256": sha256(p),
                "bytes": p.stat().st_size,
            }
            for p in files_for_hash
        ]
    )
    checksums_p = out / "checksums_sha256_v1_2.csv"
    checksums.to_csv(checksums_p, index=False)

    overall = "PASS_FROZEN_DEVELOPMENT__FINAL_TEST_SEALED"

    audit_out = {
        "version": "1.2",
        "overall_status": overall,
        "selected_configs": selected_map,
        "oof_validation_rows": int(len(oof)),
        "oof_validation_max_year":
            int(pd.to_datetime(oof["issue_date"]).dt.year.max()),
        "calibration_method":
            "PLATT_ON_POOLED_OOF_VALIDATION_2014_2022",
        "threshold_policies": [
            "MAX_CSI",
            "RECALL80_MAX_PRECISION",
        ],
        "final_base_model_training":
            "F3_FIT_ONLY_THROUGH_2019",
        "f3_test_predictions_generated": False,
        "f3_test_scoring_performed": False,
        "f1_f2_test_used": False,
        "final_holdout": "F3_TEST_2023_2025",
        "development_models_state": "CLOSED_FROZEN_V1_2",
        "next_step":
            "Run a separate one-shot final holdout evaluation on F3 TEST 2023-2025 only.",
    }

    audit_json.write_text(
        json.dumps(audit_out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "=" * 220,
        "NW HYDROCLIMATE — FROZEN DEVELOPMENT FLOOD MODELS v1.2",
        "=" * 220,
        f"OVERALL STATUS                     : {overall}",
        f"OOF validation rows                : {len(oof)}",
        f"OOF validation max year            : {pd.to_datetime(oof['issue_date']).dt.year.max()}",
        "F3 TEST predictions generated       : False",
        "F3 TEST scoring performed            : False",
        "Final holdout                       : F3 TEST 2023-2025",
        "",
        "FROZEN MODEL PROTOCOL",
        frozen.to_string(index=False),
        "",
        "CALIBRATION DEVELOPMENT DIAGNOSTICS",
        calibration_diag.to_string(index=False),
        "",
        "DEVELOPMENT THRESHOLDS",
        thresholds.to_string(index=False),
        "",
        "IMPORTANT",
        "Benchmark-selected configurations are now frozen.",
        "Calibration and thresholds use development validation only.",
        "Post-calibration metrics are development diagnostics, not final unbiased performance.",
        "No TEST prediction has been generated.",
        "",
        f"Frozen protocol : {frozen_p}",
        f"OOF predictions : {oof_p}",
        f"Thresholds      : {thresholds_p}",
        f"Models          : {models_dir}",
        f"Output          : {out}",
    ]

    audit_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    progress("PHASE 6/6", 1, 1, start, f"status={overall}")

    print("\n" + "=" * 220)
    print("\n".join(lines[3:]))
    print("=" * 220)
    print(f"OVERALL STATUS : {overall}")
    print(f"Output         : {out}")
    print("=" * 220)


if __name__ == "__main__":
    main()
