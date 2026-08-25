#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_nw_flood_models_final_holdout_v1_0.py

ONE-SHOT FINAL HOLDOUT EVALUATION.

APRE PER LA PRIMA VOLTA IL SOLO:
    F3 TEST = 2023-2025

e valuta ESCLUSIVAMENTE i modelli già congelati in:
    nw_flood_models_frozen_development_v1_2/

NON modifica:
- modello;
- iperparametri;
- weighting;
- calibratore;
- soglie operative.

DOPO QUESTO SCRIPT IL FINAL HOLDOUT È CONSIDERATO APERTO.
Qualunque modifica successiva al protocollo deve essere dichiarata come
nuova analisi post-hoc e NON può riutilizzare 2023-2025 come holdout intatto.

VALUTAZIONI
-----------
Per 24/48/72 h:
- probabilità raw;
- probabilità Platt calibrata;
- baseline climatologica F3-FIT;
- Average Precision;
- ROC-AUC;
- Brier score;
- Brier Skill Score vs climatologia FIT;
- log loss;
- calibration intercept/slope sul TEST (diagnostico finale);
- metriche thresholded per le due soglie congelate:
    MAX_CSI
    RECALL80_MAX_PRECISION
- metriche per receptor;
- metriche per anno;
- cluster di issue-date positive/predette;
- CI bootstrap a blocchi receptor × season-year.

BOOTSTRAP
---------
Il bootstrap NON campiona righe giornaliere indipendentemente.
Campiona blocchi receptor_id × year, così preserva la dipendenza temporale
intra-stagionale e quella dovuta alle finestre 24/48/72 h sovrapposte.

OUTPUT
------
nw_flood_model_final_holdout_evaluation_v1_0/
  FINAL_HOLDOUT_OPENED_v1_0.json
  final_holdout_predictions_v1_0.parquet
  final_holdout_probability_metrics_v1_0.csv
  final_holdout_threshold_metrics_v1_0.csv
  final_holdout_receptor_metrics_v1_0.csv
  final_holdout_year_metrics_v1_0.csv
  final_holdout_issue_cluster_metrics_v1_0.csv
  final_holdout_bootstrap_ci_v1_0.csv
  final_holdout_baseline_metrics_v1_0.csv
  checksums_sha256_v1_0.csv
  final_holdout_audit_v1_0.json
  final_holdout_audit_v1_0.txt
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

RANDOM_STATE = 20260825
BOOTSTRAP_REPS = 2000

HORIZONS = [24, 48, 72]
LABEL_COLS = {
    24: "label_extreme_within_24h",
    48: "label_extreme_within_48h",
    72: "label_extreme_within_72h",
}

EXPECTED_FINAL_YEARS = {2023, 2024, 2025}


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
        f"| rate {rate:7.2f}/s | ETA {fmt_seconds(eta)}"
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


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p)).reshape(-1, 1)


def apply_platt(calibrator, raw_prob):
    return calibrator.predict_proba(logit(raw_prob))[:, 1]


def safe_probability_metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)

    out = {
        "average_precision": np.nan,
        "roc_auc": np.nan,
        "brier_score": np.nan,
        "log_loss": np.nan,
    }

    if len(y) == 0:
        return out

    if np.sum(y == 1) > 0:
        out["average_precision"] = float(
            average_precision_score(y, p)
        )

    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(
            roc_auc_score(y, p)
        )

    out["brier_score"] = float(
        brier_score_loss(y, p)
    )

    out["log_loss"] = float(
        log_loss(y, p, labels=[0, 1])
    )

    return out


def calibration_slope_intercept(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    if len(np.unique(y)) < 2:
        return np.nan, np.nan

    model = LogisticRegression(
        solver="lbfgs",
        C=1e6,
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    model.fit(logit(p), y)

    return (
        float(model.intercept_[0]),
        float(model.coef_[0, 0]),
    )


def threshold_metrics(y, p, threshold):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)

    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))

    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan

    csi_denom = tp + fp + fn
    csi = tp / csi_denom if csi_denom else np.nan

    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision)
        and np.isfinite(recall)
        and (precision + recall) > 0
        else np.nan
    )

    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "csi": csi,
        "f1": f1,
        "specificity": specificity,
    }


def issue_run_clusters(df, boolean_col):
    """
    Cluster consecutive positive issue dates within receptor.
    This is an issue-date cluster, NOT a reconstructed hydrological event.
    """
    clusters = []

    for receptor, g in df.groupby("receptor_id"):
        pos = g[g[boolean_col].astype(bool)].copy()
        if not len(pos):
            continue

        pos = pos.sort_values("issue_date")
        dates = pd.to_datetime(pos["issue_date"]).tolist()

        cluster_id = 0
        start = dates[0]
        prev = dates[0]

        for d in dates[1:]:
            if (d - prev).days > 1:
                clusters.append(
                    {
                        "receptor_id": receptor,
                        "cluster_id": cluster_id,
                        "start_date": start,
                        "end_date": prev,
                    }
                )
                cluster_id += 1
                start = d
            prev = d

        clusters.append(
            {
                "receptor_id": receptor,
                "cluster_id": cluster_id,
                "start_date": start,
                "end_date": prev,
            }
        )

    return pd.DataFrame(
        clusters,
        columns=[
            "receptor_id",
            "cluster_id",
            "start_date",
            "end_date",
        ],
    )


def cluster_metrics(df, threshold):
    work = df.copy()
    work["true_positive_issue"] = work["y_true"].eq(1)
    work["pred_positive_issue"] = work["calibrated_probability"].ge(threshold)

    true_clusters = issue_run_clusters(
        work,
        "true_positive_issue",
    )
    pred_clusters = issue_run_clusters(
        work,
        "pred_positive_issue",
    )

    true_hits = 0

    for _, tc in true_clusters.iterrows():
        receptor = tc["receptor_id"]
        start = pd.Timestamp(tc["start_date"])
        end = pd.Timestamp(tc["end_date"])

        g = work[
            work["receptor_id"].astype(str).eq(str(receptor))
            & work["issue_date"].between(start, end)
        ]

        if g["pred_positive_issue"].any():
            true_hits += 1

    false_alert_clusters = 0

    for _, pc in pred_clusters.iterrows():
        receptor = pc["receptor_id"]
        start = pd.Timestamp(pc["start_date"])
        end = pd.Timestamp(pc["end_date"])

        g = work[
            work["receptor_id"].astype(str).eq(str(receptor))
            & work["issue_date"].between(start, end)
        ]

        if not g["true_positive_issue"].any():
            false_alert_clusters += 1

    true_count = len(true_clusters)
    pred_count = len(pred_clusters)

    return {
        "true_issue_clusters": int(true_count),
        "hit_true_issue_clusters": int(true_hits),
        "issue_cluster_hit_rate":
            float(true_hits / true_count)
            if true_count else np.nan,
        "predicted_alert_clusters": int(pred_count),
        "false_alert_clusters": int(false_alert_clusters),
        "false_alert_cluster_fraction":
            float(false_alert_clusters / pred_count)
            if pred_count else np.nan,
    }


def bootstrap_block_ci(df, reps, rng):
    """
    Sample receptor × season-year blocks with replacement.
    Returns bootstrap samples for AP, ROC-AUC, Brier, LogLoss.
    """
    work = df.copy()
    work["season_year"] = pd.to_datetime(
        work["issue_date"]
    ).dt.year.astype(int)

    block_keys = (
        work[["receptor_id", "season_year"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    blocks = {}

    for _, r in block_keys.iterrows():
        key = (str(r["receptor_id"]), int(r["season_year"]))
        blocks[key] = work[
            work["receptor_id"].astype(str).eq(key[0])
            & work["season_year"].eq(key[1])
        ]

    keys = list(blocks.keys())
    n_blocks = len(keys)

    rows = []

    for b in range(reps):
        sampled_idx = rng.integers(
            low=0,
            high=n_blocks,
            size=n_blocks,
        )

        sampled = pd.concat(
            [
                blocks[keys[int(i)]]
                for i in sampled_idx
            ],
            ignore_index=True,
        )

        y = sampled["y_true"].astype(int).to_numpy()
        p = sampled["calibrated_probability"].astype(float).to_numpy()

        m = safe_probability_metrics(y, p)

        rows.append(
            {
                "replicate": b + 1,
                **m,
            }
        )

    return pd.DataFrame(rows), n_blocks


def ci_summary(boot, metric):
    x = pd.to_numeric(
        boot[metric],
        errors="coerce",
    ).dropna()

    if not len(x):
        return np.nan, np.nan, np.nan

    return (
        float(x.mean()),
        float(np.quantile(x, 0.025)),
        float(np.quantile(x, 0.975)),
    )


def main():
    root = Path(__file__).resolve().parent

    frozen_root = (
        root / "nw_flood_models_frozen_development_v1_2"
    )

    protocol_p = frozen_root / "frozen_model_protocol_v1_2.csv"
    checksums_p = frozen_root / "checksums_sha256_v1_2.csv"

    master_root = (
        root
        / "nw_hydroclimate_foldwise_master_core_canonical_v1_0"
    )

    master_p = (
        master_root
        / "nw_hydroclimate_foldwise_master_core_v1_0.parquet"
    )

    dictionary_p = (
        master_root
        / "nw_hydroclimate_master_predictor_dictionary_v1_0.csv"
    )

    for p in [
        protocol_p,
        checksums_p,
        master_p,
        dictionary_p,
    ]:
        if not p.exists():
            raise SystemExit(f"Manca: {p}")

    out = (
        root / "nw_flood_model_final_holdout_evaluation_v1_0"
    )

    sentinel = out / "FINAL_HOLDOUT_OPENED_v1_0.json"

    if sentinel.exists():
        raise SystemExit(
            "\nFINAL HOLDOUT già aperto in precedenza.\n"
            f"Sentinel: {sentinel}\n"
            "Lo script one-shot non viene rieseguito automaticamente."
        )

    out.mkdir(parents=True, exist_ok=True)

    print("=" * 220)
    print("NW HYDROCLIMATE — ONE-SHOT FINAL HOLDOUT EVALUATION v1.0")
    print("=" * 220)
    print(
        "This run opens F3 TEST 2023-2025 for FINAL evaluation.",
        flush=True,
    )
    print(
        "Frozen models, calibration and thresholds will NOT be changed.",
        flush=True,
    )

    # ------------------------------------------------------------------
    # PHASE 1/7 — verify frozen artifacts and protocol
    # ------------------------------------------------------------------
    print("\nPHASE 1/7 — verify frozen artifacts and protocol integrity")
    start = time.time()

    protocol = pd.read_csv(
        protocol_p,
        low_memory=False,
    )

    checksums = pd.read_csv(
        checksums_p,
        low_memory=False,
    )

    bad_hashes = []

    for _, r in checksums.iterrows():
        rel = Path(str(r["file"]))
        p = frozen_root / rel

        if not p.exists():
            bad_hashes.append(
                f"MISSING:{rel}"
            )
            continue

        observed = sha256(p)
        expected = str(r["sha256"])

        if observed != expected:
            bad_hashes.append(
                f"HASH_MISMATCH:{rel}"
            )

    if bad_hashes:
        raise SystemExit(
            "Frozen artifact integrity failure:\n"
            + "\n".join(bad_hashes)
        )

    if len(protocol) != 3:
        raise SystemExit(
            f"Frozen protocol rows={len(protocol)}, expected=3"
        )

    if protocol["test_predictions_generated"].astype(str).str.lower().isin(
        {"true", "1"}
    ).any():
        raise SystemExit(
            "Frozen protocol says TEST predictions already generated."
        )

    if not protocol["final_holdout"].astype(str).str.contains(
        "F3_TEST_2023_2025"
    ).all():
        raise SystemExit(
            "Frozen protocol final holdout mismatch."
        )

    progress(
        "PHASE 1/7",
        1,
        1,
        start,
        "checksums PASS | protocol frozen",
    )

    # ------------------------------------------------------------------
    # PHASE 2/7 — load master, isolate F3 TEST, seal sentinel
    # ------------------------------------------------------------------
    print("\nPHASE 2/7 — isolate F3 TEST 2023-2025 and mark holdout opened")
    start = time.time()

    master = pd.read_parquet(
        master_p
    )

    master["issue_date"] = pd.to_datetime(
        master["issue_date"],
        errors="coerce",
    )

    test = master[
        master["fold_id"].astype(str).eq("F3")
        & master["partition"].astype(str).eq("TEST")
    ].copy()

    years = set(
        test["issue_date"].dt.year.dropna().astype(int).unique()
    )

    if years != EXPECTED_FINAL_YEARS:
        raise SystemExit(
            f"F3 TEST years={sorted(years)}, "
            f"expected={sorted(EXPECTED_FINAL_YEARS)}"
        )

    if test.duplicated(
        ["receptor_id", "issue_date"]
    ).any():
        raise SystemExit(
            "Duplicate receptor/date rows in final holdout."
        )

    sentinel_payload = {
        "opened_at_local_script_runtime":
            pd.Timestamp.now().isoformat(),
        "final_holdout":
            "F3 TEST 2023-2025",
        "status":
            "OPENED_FOR_ONE_SHOT_FINAL_EVALUATION",
        "frozen_protocol":
            str(protocol_p),
        "master":
            str(master_p),
        "rerun_policy":
            "ABORT_IF_SENTINEL_EXISTS",
    }

    sentinel.write_text(
        json.dumps(
            sentinel_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    progress(
        "PHASE 2/7",
        1,
        1,
        start,
        f"raw F3 TEST rows={len(test)} | years={sorted(years)}",
    )

    # ------------------------------------------------------------------
    # PHASE 3/7 — load frozen models, predict TEST once
    # ------------------------------------------------------------------
    print("\nPHASE 3/7 — generate one-shot final TEST predictions")
    start = time.time()

    dictionary = pd.read_csv(
        dictionary_p,
        low_memory=False,
    )
    predictor_cols = dictionary["predictor"].astype(str).tolist()

    pred_rows = []

    baseline_rows = []

    for i, horizon in enumerate(HORIZONS, 1):
        row = protocol[
            protocol["horizon_hours"].eq(horizon)
        ].iloc[0]

        model_p = (
            frozen_root
            / "models"
            / f"horizon_{horizon}h_base_model.joblib"
        )

        cal_p = (
            frozen_root
            / "models"
            / f"horizon_{horizon}h_platt_calibrator.joblib"
        )

        model = joblib.load(
            model_p
        )
        calibrator = joblib.load(
            cal_p
        )

        label_col = LABEL_COLS[horizon]

        y_all = pd.to_numeric(
            test[label_col],
            errors="coerce",
        )

        htest = test.loc[
            y_all.notna()
        ].copy()

        y = (
            y_all.loc[y_all.notna()]
            .astype(int)
            .to_numpy()
        )

        X = htest[predictor_cols].apply(
            pd.to_numeric,
            errors="coerce",
        )

        raw_p = model.predict_proba(
            X
        )[:, 1]

        cal_pv = apply_platt(
            calibrator,
            raw_p,
        )

        # Fixed baseline from F3 FIT only.
        f3fit = master[
            master["fold_id"].astype(str).eq("F3")
            & master["partition"].astype(str).eq("FIT")
        ].copy()

        y_fit = pd.to_numeric(
            f3fit[label_col],
            errors="coerce",
        ).dropna().astype(int)

        fit_climatology = float(
            y_fit.mean()
        )

        baseline_prob = np.full(
            len(htest),
            fit_climatology,
            dtype=float,
        )

        temp = pd.DataFrame(
            {
                "horizon_hours": horizon,
                "receptor_id":
                    htest["receptor_id"].astype(str).to_numpy(),
                "issue_date":
                    htest["issue_date"].to_numpy(),
                "y_true": y,
                "raw_probability": raw_p,
                "calibrated_probability": cal_pv,
                "fit_climatology_probability":
                    baseline_prob,
                "threshold_max_csi":
                    float(row["threshold_max_csi"]),
                "threshold_recall80_max_precision":
                    float(
                        row[
                            "threshold_recall80_max_precision"
                        ]
                    ),
            }
        )

        pred_rows.append(
            temp
        )

        baseline_m = safe_probability_metrics(
            y,
            baseline_prob,
        )

        baseline_rows.append(
            {
                "horizon_hours": horizon,
                "baseline":
                    "GLOBAL_F3_FIT_CLIMATOLOGY",
                "f3_fit_climatology_probability":
                    fit_climatology,
                **baseline_m,
            }
        )

        progress(
            "PHASE 3/7",
            i,
            3,
            start,
            (
                f"{horizon}h | eligible={len(htest)} "
                f"| positives={int(np.sum(y==1))}"
            ),
        )

    predictions = pd.concat(
        pred_rows,
        ignore_index=True,
    )

    baseline_metrics = pd.DataFrame(
        baseline_rows
    )

    # ------------------------------------------------------------------
    # PHASE 4/7 — final probability and threshold metrics
    # ------------------------------------------------------------------
    print("\nPHASE 4/7 — calculate final probability and threshold metrics")
    start = time.time()

    probability_rows = []
    threshold_rows = []

    for i, horizon in enumerate(HORIZONS, 1):
        h = predictions[
            predictions["horizon_hours"].eq(horizon)
        ].copy()

        y = h["y_true"].astype(int).to_numpy()
        raw_p = h["raw_probability"].astype(float).to_numpy()
        cal_p = h["calibrated_probability"].astype(float).to_numpy()
        base_p = h["fit_climatology_probability"].astype(float).to_numpy()

        raw_m = safe_probability_metrics(
            y,
            raw_p,
        )
        cal_m = safe_probability_metrics(
            y,
            cal_p,
        )
        base_m = safe_probability_metrics(
            y,
            base_p,
        )

        cal_intercept, cal_slope = calibration_slope_intercept(
            y,
            cal_p,
        )

        bss = (
            1.0
            - cal_m["brier_score"] / base_m["brier_score"]
            if base_m["brier_score"] > 0
            else np.nan
        )

        probability_rows.append(
            {
                "horizon_hours": horizon,
                "eligible_test_rows": int(len(h)),
                "positive_test_rows":
                    int(np.sum(y == 1)),
                "positive_fraction":
                    float(np.mean(y)),
                "raw_average_precision":
                    raw_m["average_precision"],
                "calibrated_average_precision":
                    cal_m["average_precision"],
                "calibrated_roc_auc":
                    cal_m["roc_auc"],
                "raw_brier_score":
                    raw_m["brier_score"],
                "calibrated_brier_score":
                    cal_m["brier_score"],
                "baseline_brier_score":
                    base_m["brier_score"],
                "brier_skill_score_vs_fit_climatology":
                    bss,
                "raw_log_loss":
                    raw_m["log_loss"],
                "calibrated_log_loss":
                    cal_m["log_loss"],
                "calibration_intercept_test":
                    cal_intercept,
                "calibration_slope_test":
                    cal_slope,
                "evaluation_scope":
                    "FINAL_HOLDOUT_F3_TEST_2023_2025",
            }
        )

        for policy, col in [
            (
                "MAX_CSI",
                "threshold_max_csi",
            ),
            (
                "RECALL80_MAX_PRECISION",
                "threshold_recall80_max_precision",
            ),
        ]:
            threshold = float(
                h[col].iloc[0]
            )

            tm = threshold_metrics(
                y,
                cal_p,
                threshold,
            )

            threshold_rows.append(
                {
                    "horizon_hours": horizon,
                    "threshold_policy": policy,
                    "frozen_threshold": threshold,
                    **tm,
                    "evaluation_scope":
                        "FINAL_HOLDOUT_F3_TEST_2023_2025",
                }
            )

        progress(
            "PHASE 4/7",
            i,
            3,
            start,
            (
                f"{horizon}h | AP={cal_m['average_precision']:.4f} "
                f"| ROC={cal_m['roc_auc']:.4f} "
                f"| BSS={bss:.4f}"
            ),
        )

    probability_metrics = pd.DataFrame(
        probability_rows
    )
    threshold_metrics_df = pd.DataFrame(
        threshold_rows
    )

    # ------------------------------------------------------------------
    # PHASE 5/7 — receptor/year/issue-cluster diagnostics
    # ------------------------------------------------------------------
    print("\nPHASE 5/7 — receptor, year, and issue-cluster diagnostics")
    start = time.time()

    receptor_rows = []
    year_rows = []
    cluster_rows = []

    for horizon in HORIZONS:
        h = predictions[
            predictions["horizon_hours"].eq(horizon)
        ].copy()

        for receptor, g in h.groupby("receptor_id"):
            y = g["y_true"].astype(int).to_numpy()
            p = g["calibrated_probability"].astype(float).to_numpy()

            m = safe_probability_metrics(
                y,
                p,
            )

            receptor_rows.append(
                {
                    "horizon_hours": horizon,
                    "receptor_id": receptor,
                    "eligible_rows": int(len(g)),
                    "positive_rows": int(np.sum(y == 1)),
                    "positive_fraction":
                        float(np.mean(y)) if len(y) else np.nan,
                    **m,
                }
            )

        h["year"] = pd.to_datetime(
            h["issue_date"]
        ).dt.year.astype(int)

        for year, g in h.groupby("year"):
            y = g["y_true"].astype(int).to_numpy()
            p = g["calibrated_probability"].astype(float).to_numpy()

            m = safe_probability_metrics(
                y,
                p,
            )

            year_rows.append(
                {
                    "horizon_hours": horizon,
                    "year": int(year),
                    "eligible_rows": int(len(g)),
                    "positive_rows": int(np.sum(y == 1)),
                    "positive_fraction":
                        float(np.mean(y)) if len(y) else np.nan,
                    **m,
                }
            )

        for policy, col in [
            (
                "MAX_CSI",
                "threshold_max_csi",
            ),
            (
                "RECALL80_MAX_PRECISION",
                "threshold_recall80_max_precision",
            ),
        ]:
            threshold = float(
                h[col].iloc[0]
            )

            cm = cluster_metrics(
                h,
                threshold,
            )

            cluster_rows.append(
                {
                    "horizon_hours": horizon,
                    "threshold_policy": policy,
                    "frozen_threshold": threshold,
                    **cm,
                    "cluster_definition":
                        "CONSECUTIVE_POSITIVE_ISSUE_DATES_WITHIN_RECEPTOR",
                }
            )

    receptor_metrics = pd.DataFrame(
        receptor_rows
    )
    year_metrics = pd.DataFrame(
        year_rows
    )
    cluster_metrics_df = pd.DataFrame(
        cluster_rows
    )

    progress(
        "PHASE 5/7",
        1,
        1,
        start,
        (
            f"receptor rows={len(receptor_metrics)} "
            f"| year rows={len(year_metrics)} "
            f"| cluster rows={len(cluster_metrics_df)}"
        ),
    )

    # ------------------------------------------------------------------
    # PHASE 6/7 — clustered bootstrap CI
    # ------------------------------------------------------------------
    print("\nPHASE 6/7 — clustered bootstrap CI by receptor × season-year")
    start = time.time()

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    ci_rows = []

    for i, horizon in enumerate(HORIZONS, 1):
        h = predictions[
            predictions["horizon_hours"].eq(horizon)
        ].copy()

        boot, n_blocks = bootstrap_block_ci(
            h,
            BOOTSTRAP_REPS,
            rng,
        )

        for metric in [
            "average_precision",
            "roc_auc",
            "brier_score",
            "log_loss",
        ]:
            mean, lo, hi = ci_summary(
                boot,
                metric,
            )

            point = safe_probability_metrics(
                h["y_true"].astype(int).to_numpy(),
                h["calibrated_probability"].astype(float).to_numpy(),
            )[metric]

            ci_rows.append(
                {
                    "horizon_hours": horizon,
                    "metric": metric,
                    "point_estimate": point,
                    "bootstrap_mean": mean,
                    "ci_2_5pct": lo,
                    "ci_97_5pct": hi,
                    "bootstrap_replicates": BOOTSTRAP_REPS,
                    "bootstrap_blocks": n_blocks,
                    "block_definition":
                        "RECEPTOR_ID_X_SEASON_YEAR",
                }
            )

        progress(
            "PHASE 6/7",
            i,
            3,
            start,
            f"{horizon}h | blocks={n_blocks} reps={BOOTSTRAP_REPS}",
        )

    bootstrap_ci = pd.DataFrame(
        ci_rows
    )

    # ------------------------------------------------------------------
    # PHASE 7/7 — write final artifacts, audit, checksums
    # ------------------------------------------------------------------
    print("\nPHASE 7/7 — write final holdout artifacts and close evaluation")
    start = time.time()

    predictions_p = (
        out / "final_holdout_predictions_v1_0.parquet"
    )
    probability_p = (
        out / "final_holdout_probability_metrics_v1_0.csv"
    )
    threshold_p = (
        out / "final_holdout_threshold_metrics_v1_0.csv"
    )
    receptor_p = (
        out / "final_holdout_receptor_metrics_v1_0.csv"
    )
    year_p = (
        out / "final_holdout_year_metrics_v1_0.csv"
    )
    cluster_p = (
        out / "final_holdout_issue_cluster_metrics_v1_0.csv"
    )
    ci_p = (
        out / "final_holdout_bootstrap_ci_v1_0.csv"
    )
    baseline_p = (
        out / "final_holdout_baseline_metrics_v1_0.csv"
    )
    audit_json = (
        out / "final_holdout_audit_v1_0.json"
    )
    audit_txt = (
        out / "final_holdout_audit_v1_0.txt"
    )

    predictions.to_parquet(
        predictions_p,
        index=False,
    )
    probability_metrics.to_csv(
        probability_p,
        index=False,
    )
    threshold_metrics_df.to_csv(
        threshold_p,
        index=False,
    )
    receptor_metrics.to_csv(
        receptor_p,
        index=False,
    )
    year_metrics.to_csv(
        year_p,
        index=False,
    )
    cluster_metrics_df.to_csv(
        cluster_p,
        index=False,
    )
    bootstrap_ci.to_csv(
        ci_p,
        index=False,
    )
    baseline_metrics.to_csv(
        baseline_p,
        index=False,
    )

    overall = "PASS_FINAL_HOLDOUT_EVALUATED__NO_FURTHER_TUNING_ALLOWED"

    audit = {
        "version": "1.0",
        "overall_status": overall,
        "final_holdout":
            "F3 TEST 2023-2025",
        "holdout_opened": True,
        "frozen_models_modified": False,
        "frozen_calibrators_modified": False,
        "frozen_thresholds_modified": False,
        "test_predictions_rows":
            int(len(predictions)),
        "horizons_hours":
            HORIZONS,
        "bootstrap_replicates":
            BOOTSTRAP_REPS,
        "bootstrap_block_definition":
            "RECEPTOR_ID_X_SEASON_YEAR",
        "post_test_tuning_allowed":
            False,
        "interpretation_rule":
            (
                "Any subsequent model or protocol change is post-hoc and "
                "2023-2025 can no longer serve as untouched final holdout."
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

    files_for_hash = [
        sentinel,
        predictions_p,
        probability_p,
        threshold_p,
        receptor_p,
        year_p,
        cluster_p,
        ci_p,
        baseline_p,
        audit_json,
    ]

    checksums_out = pd.DataFrame(
        [
            {
                "file": p.name,
                "sha256": sha256(p),
                "bytes": p.stat().st_size,
            }
            for p in files_for_hash
        ]
    )

    checksums_out_p = (
        out / "checksums_sha256_v1_0.csv"
    )
    checksums_out.to_csv(
        checksums_out_p,
        index=False,
    )

    lines = [
        "=" * 220,
        "NW HYDROCLIMATE — FINAL HOLDOUT EVALUATION v1.0",
        "=" * 220,
        f"OVERALL STATUS : {overall}",
        "Final holdout  : F3 TEST 2023-2025",
        "Frozen protocol changed : False",
        "Further tuning allowed  : False",
        "",
        "FINAL PROBABILITY METRICS",
        probability_metrics.to_string(index=False),
        "",
        "FINAL THRESHOLD METRICS",
        threshold_metrics_df.to_string(index=False),
        "",
        "FINAL ISSUE-CLUSTER METRICS",
        cluster_metrics_df.to_string(index=False),
        "",
        "BOOTSTRAP 95% CI",
        bootstrap_ci.to_string(index=False),
        "",
        "IMPORTANT",
        "The final holdout has now been opened and evaluated.",
        "Do not change model family, hyperparameters, weighting, calibration or thresholds on the basis of these results.",
        "Any later model change is a new post-hoc development cycle and requires a new independent future holdout for unbiased confirmation.",
        "",
        f"Predictions : {predictions_p}",
        f"Metrics     : {probability_p}",
        f"Thresholds  : {threshold_p}",
        f"CI          : {ci_p}",
        f"Audit       : {audit_json}",
        f"Output      : {out}",
    ]

    audit_txt.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    progress(
        "PHASE 7/7",
        1,
        1,
        start,
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
