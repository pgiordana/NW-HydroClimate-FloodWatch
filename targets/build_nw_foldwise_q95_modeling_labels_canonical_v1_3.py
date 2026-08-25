#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_nw_foldwise_q95_modeling_labels_canonical_v1_3.py

ESTENDE LE LABEL CANONICHE v1.2 A FIT + VALIDATION + TEST.

PERCHÉ SERVE
------------
La policy Q95 v1.2 è corretta e chiusa, ma il file canonico delle label v1.2
contiene soltanto VALIDATION e TEST. Per addestrare un modello supervisionato
servono anche le label del FIT di ciascun fold.

Questa v1.3 NON cambia la policy.
Usa esattamente:
- Q95 receptor-specific stimato sul FIT;
- Q97.5 severity diagnostic;
- logica positiva > missing;
- negativo solo con finestra futura completa;
- nessun dicembre -> settembre;
- nessuna imputazione.

CONTROLLO CRITICO
-----------------
Le label VALIDATION/TEST rigenerate devono essere IDENTICHE a quelle canoniche
v1.2. Se anche una sola differisce, lo script si ferma e NON congela la v1.3.

FOLD
----
F1 FIT <= 2013 | VAL 2014-2016 | TEST 2017-2019
F2 FIT <= 2016 | VAL 2017-2019 | TEST 2020-2022
F3 FIT <= 2019 | VAL 2020-2022 | TEST 2023-2025

NOTA SUL FIT
------------
La soglia Q95 del fold è una proprietà stimata sull'intero training set e viene
usata per definire le label supervisionate interne al training set. Questo è
coerente con la validazione fold-wise: nessuna osservazione di validation/test
contribuisce alla soglia.

INPUT
-----
nw_primary_daily_hydrology_target_base_v1_0/
  nw_primary_daily_hydrology_target_base_1987_2025_v1_0.csv

nw_foldwise_q95_forecast_labels_canonical_v1_2/
  foldwise_q95_thresholds_canonical_v1_2.csv
  foldwise_q95_forecast_labels_canonical_v1_2.csv
  fold_registry_canonical_v1_2.csv
  forecast_label_policy_canonical_v1_2.csv

OUTPUT
------
nw_foldwise_q95_modeling_labels_canonical_v1_3/
  foldwise_q95_thresholds_canonical_v1_3.csv
  foldwise_q95_modeling_labels_canonical_v1_3.csv
  foldwise_q95_modeling_label_support_by_receptor_v1_3.csv
  foldwise_q95_modeling_label_support_pooled_v1_3.csv
  fold_registry_canonical_v1_3.csv
  forecast_label_policy_canonical_v1_3.csv
  validation_test_reproduction_audit_v1_3.csv
  checksums_sha256_canonical_v1_3.csv
  modeling_label_audit_v1_3.json
  modeling_label_audit_v1_3.txt
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd


HORIZONS = [1, 2, 3]
LABEL_COLS = [
    "label_extreme_within_24h",
    "label_extreme_within_48h",
    "label_extreme_within_72h",
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
        msg += f" | {str(current)[:120]}"

    print(msg.ljust(255), end="", flush=True)
    if done >= total:
        print(flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def same_autumn(a: pd.Timestamp, b: pd.Timestamp) -> bool:
    return (
        a.year == b.year
        and a.month in {9, 10, 11, 12}
        and b.month in {9, 10, 11, 12}
    )


def future_label(date, value_by_date, threshold, horizon_days):
    """
    1   if any observed future value >= threshold.
    0   if all required future values are observed and all < threshold.
    NaN if no positive is observed but at least one required future day
        is missing or outside the same Sep-Dec season.
    """
    missing = False

    for step in range(1, horizon_days + 1):
        fd = date + pd.Timedelta(days=step)

        if not same_autumn(date, fd):
            missing = True
            continue

        value = value_by_date.get(fd, np.nan)

        if pd.isna(value):
            missing = True
            continue

        if float(value) >= float(threshold):
            return 1

    if missing:
        return np.nan

    return 0


def partition_for_year(year: int, fold_row: pd.Series):
    if year <= int(fold_row["fit_end_year"]):
        return "FIT"

    if (
        int(fold_row["validation_start_year"])
        <= year
        <= int(fold_row["validation_end_year"])
    ):
        return "VALIDATION"

    if (
        int(fold_row["test_start_year"])
        <= year
        <= int(fold_row["test_end_year"])
    ):
        return "TEST"

    return "OUTSIDE"


def values_equal_with_nan(a, b):
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")

    both_nan = aa.isna() & bb.isna()
    same = aa.eq(bb)

    return both_nan | same


def positive_clusters(dates: pd.Series) -> int:
    ds = sorted(pd.Timestamp(x).normalize() for x in dates)

    if not ds:
        return 0

    clusters = 1

    for prev, cur in zip(ds[:-1], ds[1:]):
        if (cur - prev).days > 1:
            clusters += 1

    return clusters


def main():
    root = Path(__file__).resolve().parent

    target_p = (
        root
        / "nw_primary_daily_hydrology_target_base_v1_0"
        / "nw_primary_daily_hydrology_target_base_1987_2025_v1_0.csv"
    )

    src = root / "nw_foldwise_q95_forecast_labels_canonical_v1_2"

    thresholds_p = src / "foldwise_q95_thresholds_canonical_v1_2.csv"
    labels_v12_p = src / "foldwise_q95_forecast_labels_canonical_v1_2.csv"
    folds_p = src / "fold_registry_canonical_v1_2.csv"
    policy_p = src / "forecast_label_policy_canonical_v1_2.csv"

    out = root / "nw_foldwise_q95_modeling_labels_canonical_v1_3"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 210)
    print("NW HYDROLOGY — BUILD CANONICAL FIT/VALIDATION/TEST MODELING LABELS v1.3")
    print("=" * 210)

    for p in (target_p, thresholds_p, labels_v12_p, folds_p, policy_p):
        if not p.exists():
            raise SystemExit(f"Manca: {p}")

    target = pd.read_csv(target_p, low_memory=False)
    thresholds = pd.read_csv(thresholds_p, low_memory=False)
    old_labels = pd.read_csv(labels_v12_p, low_memory=False)
    folds = pd.read_csv(folds_p, low_memory=False)
    policy = pd.read_csv(policy_p, low_memory=False)

    target["date"] = pd.to_datetime(target["date"], errors="coerce")
    target["target_observed_value"] = pd.to_numeric(
        target["target_observed_value"],
        errors="coerce",
    )

    old_labels["issue_date"] = pd.to_datetime(
        old_labels["issue_date"],
        errors="coerce",
    )

    receptors = sorted(target["receptor_id"].astype(str).unique())

    if len(receptors) != 20:
        raise SystemExit(f"Attesi 20 target receptors, trovati {len(receptors)}")

    if len(folds) != 3:
        raise SystemExit(f"Attesi 3 fold, trovati {len(folds)}")

    nonpass = thresholds[
        ~thresholds["threshold_fit_status"].astype(str).eq("PASS")
    ]

    if len(nonpass):
        raise SystemExit(
            f"Threshold canoniche non-PASS: {len(nonpass)}"
        )

    # ------------------------------------------------------------------
    # PHASE 1/4 — build all FIT/VAL/TEST labels
    # ------------------------------------------------------------------
    print("\nPHASE 1/4 — generate FIT + VALIDATION + TEST labels")
    start1 = time.time()

    cache = {}

    for receptor in receptors:
        sub = target[
            target["receptor_id"].astype(str).eq(receptor)
        ][["date", "target_observed_value"]].copy()

        cache[receptor] = {
            pd.Timestamp(r["date"]): r["target_observed_value"]
            for _, r in sub.iterrows()
        }

    rows = []

    total = len(folds) * len(receptors)
    done = 0

    min_target_year = int(target["date"].dt.year.min())

    for _, fold in folds.sort_values("fold_id").iterrows():
        fold_id = str(fold["fold_id"])
        test_end_year = int(fold["test_end_year"])

        for receptor in receptors:
            done += 1

            tr = thresholds[
                thresholds["fold_id"].astype(str).eq(fold_id)
                & thresholds["receptor_id"].astype(str).eq(receptor)
            ]

            if len(tr) != 1:
                raise SystemExit(
                    f"{fold_id}/{receptor}: threshold rows={len(tr)}"
                )

            q95 = float(tr.iloc[0]["q95_fit_threshold"])
            q975 = float(tr.iloc[0]["q975_fit_severity_threshold"])

            issue = target[
                target["receptor_id"].astype(str).eq(receptor)
                & target["date"].dt.year.between(
                    min_target_year,
                    test_end_year,
                )
            ].copy()

            value_by_date = cache[receptor]

            for _, r in issue.iterrows():
                date = pd.Timestamp(r["date"])
                year = int(date.year)
                partition = partition_for_year(year, fold)

                if partition == "OUTSIDE":
                    continue

                rec = {
                    "fold_id": fold_id,
                    "receptor_id": receptor,
                    "issue_date": date,
                    "partition": partition,
                    "q95_fit_threshold": q95,
                    "q975_fit_severity_threshold": q975,
                    "current_target_observed_value":
                        r["target_observed_value"],
                }

                for h in HORIZONS:
                    rec[
                        f"label_extreme_within_{h * 24}h"
                    ] = future_label(
                        date=date,
                        value_by_date=value_by_date,
                        threshold=q95,
                        horizon_days=h,
                    )

                rows.append(rec)

            progress(
                "PHASE 1/4",
                done,
                total,
                start1,
                f"{fold_id} | {receptor}",
            )

    labels = pd.DataFrame(rows)

    dup = int(
        labels.duplicated(
            ["fold_id", "receptor_id", "issue_date"]
        ).sum()
    )

    if dup:
        raise SystemExit(f"Duplicate modeling-label rows={dup}")

    # ------------------------------------------------------------------
    # PHASE 2/4 — reproduce frozen v1.2 validation/test exactly
    # ------------------------------------------------------------------
    print("\nPHASE 2/4 — reproduce frozen v1.2 VALIDATION/TEST labels exactly")
    start2 = time.time()

    generated_eval = labels[
        labels["partition"].isin(["VALIDATION", "TEST"])
    ].copy()

    compare_cols = [
        "fold_id",
        "receptor_id",
        "issue_date",
        "partition",
        *LABEL_COLS,
    ]

    old = old_labels[compare_cols].copy()
    new = generated_eval[compare_cols].copy()

    comp = old.merge(
        new,
        on=["fold_id", "receptor_id", "issue_date", "partition"],
        how="outer",
        suffixes=("_v12", "_v13"),
        indicator=True,
    )

    comp["key_status"] = comp["_merge"].astype(str)

    mismatch_cols = []

    for col in LABEL_COLS:
        equal = values_equal_with_nan(
            comp[f"{col}_v12"],
            comp[f"{col}_v13"],
        )

        comp[f"{col}_match"] = equal

        if (~equal & comp["_merge"].eq("both")).any():
            mismatch_cols.append(col)

    comp["all_labels_match"] = (
        comp["_merge"].eq("both")
        & comp[
            [f"{c}_match" for c in LABEL_COLS]
        ].all(axis=1)
    )

    key_nonboth = int((~comp["_merge"].eq("both")).sum())
    label_mismatch_rows = int(
        (
            comp["_merge"].eq("both")
            & ~comp["all_labels_match"]
        ).sum()
    )

    if key_nonboth or label_mismatch_rows:
        reproduction_status = "FAIL"
    else:
        reproduction_status = "PASS"

    progress(
        "PHASE 2/4",
        1,
        1,
        start2,
        (
            f"keys_nonboth={key_nonboth} "
            f"label_mismatch_rows={label_mismatch_rows}"
        ),
    )

    if reproduction_status != "PASS":
        audit_fail_p = (
            out / "validation_test_reproduction_audit_FAILED_v1_3.csv"
        )
        comp.to_csv(audit_fail_p, index=False)

        raise SystemExit(
            "VALIDATION/TEST reproduction failed; canonical freeze aborted. "
            f"Audit: {audit_fail_p}"
        )

    # ------------------------------------------------------------------
    # PHASE 3/4 — support audit including FIT
    # ------------------------------------------------------------------
    print("\nPHASE 3/4 — audit FIT/VALIDATION/TEST label support")
    start3 = time.time()

    support_rows = []

    for fold_id in sorted(labels["fold_id"].unique()):
        for receptor in receptors:
            for partition in ("FIT", "VALIDATION", "TEST"):
                sub = labels[
                    labels["fold_id"].astype(str).eq(fold_id)
                    & labels["receptor_id"].astype(str).eq(receptor)
                    & labels["partition"].eq(partition)
                ].copy()

                for h in (24, 48, 72):
                    col = f"label_extreme_within_{h}h"
                    x = pd.to_numeric(sub[col], errors="coerce")

                    eligible = x.notna()
                    positive = x.eq(1)
                    negative = x.eq(0)

                    pos_dates = sub.loc[positive, "issue_date"]

                    support_rows.append(
                        {
                            "fold_id": fold_id,
                            "receptor_id": receptor,
                            "partition": partition,
                            "horizon_hours": h,
                            "issue_rows": int(len(sub)),
                            "eligible_issue_dates": int(eligible.sum()),
                            "positive_labels": int(positive.sum()),
                            "negative_labels": int(negative.sum()),
                            "unknown_labels": int(x.isna().sum()),
                            "positive_fraction":
                                float(positive.sum() / eligible.sum())
                                if eligible.sum()
                                else np.nan,
                            "positive_issue_date_clusters":
                                positive_clusters(pos_dates),
                            "support_flag":
                                "SPARSE_POSITIVES"
                                if positive.sum() < 3
                                else "OK",
                        }
                    )

    support = pd.DataFrame(support_rows)

    pooled = (
        support.groupby(
            ["fold_id", "partition", "horizon_hours"],
            as_index=False,
        )
        .agg(
            issue_rows=("issue_rows", "sum"),
            eligible_issue_dates=("eligible_issue_dates", "sum"),
            positive_labels=("positive_labels", "sum"),
            negative_labels=("negative_labels", "sum"),
            unknown_labels=("unknown_labels", "sum"),
            positive_issue_date_clusters=(
                "positive_issue_date_clusters",
                "sum",
            ),
            receptors_sparse_positive_support=(
                "support_flag",
                lambda x: int(
                    sum(v == "SPARSE_POSITIVES" for v in x)
                ),
            ),
        )
    )

    pooled["positive_fraction"] = (
        pooled["positive_labels"]
        / pooled["eligible_issue_dates"].replace(0, np.nan)
    )

    fit_rows = pooled[pooled["partition"].eq("FIT")]

    fit_zero_positive = int(
        fit_rows["positive_labels"].eq(0).sum()
    )

    progress(
        "PHASE 3/4",
        1,
        1,
        start3,
        (
            f"modeling rows={len(labels)} "
            f"FIT pooled zero-positive rows={fit_zero_positive}"
        ),
    )

    if fit_zero_positive:
        raise SystemExit(
            f"FIT pooled zero-positive combinations={fit_zero_positive}"
        )

    # ------------------------------------------------------------------
    # PHASE 4/4 — canonical outputs/checksums
    # ------------------------------------------------------------------
    print("\nPHASE 4/4 — write canonical v1.3 modeling-label artifacts")
    start4 = time.time()

    thresholds_out = out / "foldwise_q95_thresholds_canonical_v1_3.csv"
    labels_out = out / "foldwise_q95_modeling_labels_canonical_v1_3.csv"
    support_out = (
        out
        / "foldwise_q95_modeling_label_support_by_receptor_v1_3.csv"
    )
    pooled_out = (
        out
        / "foldwise_q95_modeling_label_support_pooled_v1_3.csv"
    )
    folds_out = out / "fold_registry_canonical_v1_3.csv"
    policy_out = out / "forecast_label_policy_canonical_v1_3.csv"
    reproduction_out = (
        out / "validation_test_reproduction_audit_v1_3.csv"
    )

    shutil.copy2(thresholds_p, thresholds_out)
    labels.to_csv(labels_out, index=False)
    support.to_csv(support_out, index=False)
    pooled.to_csv(pooled_out, index=False)
    shutil.copy2(folds_p, folds_out)

    policy_v13 = policy.copy()

    extra_policy = pd.DataFrame(
        [
            {
                "policy_id": "L14",
                "rule":
                    "Canonical modeling-label artifact includes FIT, "
                    "VALIDATION and TEST rows for every fold.",
            },
            {
                "policy_id": "L15",
                "rule":
                    "FIT labels use the Q95 threshold estimated from the "
                    "same fold FIT partition; validation/test never "
                    "contribute to threshold estimation.",
            },
        ]
    )

    policy_v13 = pd.concat(
        [policy_v13, extra_policy],
        ignore_index=True,
    )

    policy_v13.to_csv(policy_out, index=False)

    comp.drop(columns=["_merge"]).to_csv(
        reproduction_out,
        index=False,
    )

    checksum_targets = [
        thresholds_out,
        labels_out,
        support_out,
        pooled_out,
        folds_out,
        policy_out,
        reproduction_out,
    ]

    checksums = pd.DataFrame(
        [
            {
                "file": p.name,
                "sha256": sha256(p),
                "bytes": p.stat().st_size,
            }
            for p in checksum_targets
        ]
    )

    checksums_out = (
        out / "checksums_sha256_canonical_v1_3.csv"
    )

    checksums.to_csv(checksums_out, index=False)

    sparse_rows = int(
        support["support_flag"].eq("SPARSE_POSITIVES").sum()
    )

    report = {
        "version": "1.3",
        "overall_status":
            "PASS_WITH_PER_RECEPTOR_SPARSE_FOLDS",
        "canonical_policy_changed": False,
        "canonical_policy_source_version": "v1.2",
        "modeling_partitions":
            ["FIT", "VALIDATION", "TEST"],
        "folds": int(labels["fold_id"].nunique()),
        "receptors": int(labels["receptor_id"].nunique()),
        "modeling_label_rows": int(len(labels)),
        "duplicate_fold_receptor_issue_date_rows": dup,
        "validation_test_reproduction_status":
            reproduction_status,
        "validation_test_key_nonmatch_rows":
            key_nonboth,
        "validation_test_label_mismatch_rows":
            label_mismatch_rows,
        "fit_pooled_zero_positive_rows":
            fit_zero_positive,
        "per_receptor_sparse_support_rows":
            sparse_rows,
        "target_imputation_performed": False,
        "global_q95_used": False,
        "official_thresholds_used_retroactively": False,
        "canonical_state":
            "CLOSED_CANONICAL_MODELING_LABELS",
        "next_step":
            "Re-run causal master-matrix preflight using v1.3 "
            "FIT/VALIDATION/TEST labels, then freeze causal feature whitelist.",
    }

    audit_json = out / "modeling_label_audit_v1_3.json"
    audit_txt = out / "modeling_label_audit_v1_3.txt"

    audit_json.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "=" * 210,
        "NW HYDROLOGY — CANONICAL FIT/VALIDATION/TEST MODELING LABELS v1.3",
        "=" * 210,
        "OVERALL STATUS                              : PASS_WITH_PER_RECEPTOR_SPARSE_FOLDS",
        "Policy                                     : unchanged from canonical v1.2",
        f"Modeling label rows                        : {len(labels)}",
        f"Folds                                      : {labels['fold_id'].nunique()}",
        f"Receptors                                  : {labels['receptor_id'].nunique()}",
        f"Duplicate fold/receptor/issue-date rows    : {dup}",
        f"Validation/Test reproduction               : {reproduction_status}",
        f"Validation/Test key nonmatch rows          : {key_nonboth}",
        f"Validation/Test label mismatch rows        : {label_mismatch_rows}",
        f"FIT pooled zero-positive rows              : {fit_zero_positive}",
        f"Per-receptor sparse support rows           : {sparse_rows}",
        "",
        "POOLED MODELING-LABEL SUPPORT",
        pooled.to_string(index=False),
        "",
        "IMPORTANT",
        "The v1.2 policy is not changed; v1.3 only adds the FIT labels required for supervised training.",
        "Frozen v1.2 VALIDATION/TEST labels are reproduced exactly.",
        "No target imputation, no global Q95, and no retrospective official threshold are used.",
        "",
        f"Labels  : {labels_out}",
        f"Support : {support_out}",
        f"Pooled  : {pooled_out}",
        f"Output  : {out}",
    ]

    audit_txt.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    progress(
        "PHASE 4/4",
        1,
        1,
        start4,
        "canonical modeling labels v1.3 written",
    )

    print("\n" + "=" * 210)
    print("\n".join(lines[3:]))
    print("=" * 210)
    print("OVERALL STATUS : PASS_WITH_PER_RECEPTOR_SPARSE_FOLDS")
    print(f"Output         : {out}")
    print("=" * 210)


if __name__ == "__main__":
    main()
