#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_nw_hydroclimate_foldwise_master_core_v1_0.py

COSTRUISCE IL MASTER FOLD-WISE PRIMARIO DEL MODELLO REGIONALE.

BLOCCHI CANONICI IN INPUT
-------------------------
1) Dynamic causal whitelist:
   nw_dynamic_causal_feature_whitelist_canonical_v1_3/
     primary_dynamic_feature_whitelist_canonical_v1_3.csv

2) Static receptor whitelist:
   nw_static_receptor_descriptor_whitelist_canonical_v1_1/
     static_receptor_descriptor_whitelist_canonical_v1_1.csv
     static_receptor_descriptor_values_canonical_v1_1.csv

3) Modeling labels:
   nw_foldwise_q95_modeling_labels_canonical_v1_3/
     foldwise_q95_modeling_labels_canonical_v1_3.csv
     foldwise_q95_modeling_label_support_pooled_v1_3.csv
     fold_registry_canonical_v1_3.csv

4) Dynamic source tables:
   ERA5 derived v1.0
   MedSea × IVT canonical v1.2

MASTER PRIMARIO
---------------
Una riga = fold_id + receptor_id + issue_date.

Contiene:
- 83 predictor dinamici canonici;
- 14 predictor statici canonici;
- 3 label Q95: 24/48/72 h;
- metadata fold/partition.

NON contiene:
- era5.label;
- current_target_observed_value;
- q95/q97.5 thresholds;
- official thresholds;
- observed precipitation augmentation;
- QA/network metadata;
- season_year;
- target hydrology corrente come predictor.

MISSINGNESS
-----------
Nessuna imputazione.

Sono preservati:
- NaN strutturali dei lag antecedenti a inizio settembre;
- NaN MedSea quando non esiste supporto marino secondo la metodologia
  geometrico-euleriana congelata;
- NaN delle label quando la finestra futura non è osservabile completamente
  e non esiste evidenza positiva.

Il master è pensato per una pipeline di modeling che:
- usa modelli capaci di gestire missingness nativa, OPPURE
- applica una trasformazione/imputazione stimata SOLO sul FIT del fold.
Mai imputazione globale.

VALIDAZIONE
-----------
- join dinamico receptor/date = 100%;
- join statico receptor = 100%;
- 0 duplicati fold/receptor/date;
- 83 + 14 = 97 predictor esatti;
- nessun predictor con nome leakage;
- label support post-join identico al registry canonico v1.3;
- scaling NON applicato qui.

OUTPUT
------
nw_hydroclimate_foldwise_master_core_canonical_v1_0/
  nw_hydroclimate_foldwise_master_core_v1_0.csv.gz
  nw_hydroclimate_foldwise_master_core_v1_0.parquet   [se engine disponibile]
  nw_hydroclimate_master_predictor_dictionary_v1_0.csv
  nw_hydroclimate_master_missingness_v1_0.csv
  nw_hydroclimate_master_label_support_v1_0.csv
  nw_hydroclimate_master_source_manifest_v1_0.csv
  nw_hydroclimate_master_fold_registry_v1_0.csv
  checksums_sha256_v1_0.csv
  nw_hydroclimate_master_audit_v1_0.json
  nw_hydroclimate_master_audit_v1_0.txt
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_DYNAMIC_FEATURES = 83
EXPECTED_STATIC_FEATURES = 14
EXPECTED_TOTAL_PREDICTORS = 97
EXPECTED_LABEL_ROWS = 263520
EXPECTED_MODELING_RECEPTORS = 20
EXPECTED_FOLDS = 3

LABEL_COLS = [
    "label_extreme_within_24h",
    "label_extreme_within_48h",
    "label_extreme_within_72h",
]

FORBIDDEN_PREDICTOR_PATTERNS = [
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def forbidden_name(name: str):
    lc = str(name).lower()
    hits = [
        pat
        for pat in FORBIDDEN_PREDICTOR_PATTERNS
        if re.search(pat, lc)
    ]
    return hits


def normalize_date(df, col):
    out = df.copy()
    out[col] = pd.to_datetime(
        out[col],
        errors="coerce",
    )
    return out


def recompute_label_support(master: pd.DataFrame):
    rows = []

    for fold_id in sorted(
        master["fold_id"].astype(str).unique()
    ):
        for partition in ("FIT", "VALIDATION", "TEST"):
            sub = master[
                master["fold_id"].astype(str).eq(fold_id)
                & master["partition"].astype(str).eq(partition)
            ]

            for h in (24, 48, 72):
                col = f"label_extreme_within_{h}h"
                x = pd.to_numeric(
                    sub[col],
                    errors="coerce",
                )

                rows.append(
                    {
                        "fold_id": fold_id,
                        "partition": partition,
                        "horizon_hours": h,
                        "issue_rows": int(len(sub)),
                        "eligible_issue_dates":
                            int(x.notna().sum()),
                        "positive_labels":
                            int(x.eq(1).sum()),
                        "negative_labels":
                            int(x.eq(0).sum()),
                        "unknown_labels":
                            int(x.isna().sum()),
                        "positive_fraction":
                            float(
                                x.eq(1).sum()
                                / x.notna().sum()
                            )
                            if x.notna().sum()
                            else np.nan,
                    }
                )

    return pd.DataFrame(rows)


def compare_label_support(recomputed, canonical):
    keys = [
        "fold_id",
        "partition",
        "horizon_hours",
    ]

    compare_cols = [
        "issue_rows",
        "eligible_issue_dates",
        "positive_labels",
        "negative_labels",
        "unknown_labels",
    ]

    c = canonical[
        keys + compare_cols
    ].copy()

    r = recomputed[
        keys + compare_cols
    ].copy()

    merged = c.merge(
        r,
        on=keys,
        how="outer",
        suffixes=("_canonical", "_master"),
        indicator=True,
    )

    merged["all_counts_match"] = merged["_merge"].eq("both")

    for col in compare_cols:
        merged["all_counts_match"] &= (
            pd.to_numeric(
                merged[f"{col}_canonical"],
                errors="coerce",
            )
            .eq(
                pd.to_numeric(
                    merged[f"{col}_master"],
                    errors="coerce",
                )
            )
        )

    return merged


def main():
    root = Path(__file__).resolve().parent

    dynamic_root = (
        root
        / "nw_dynamic_causal_feature_whitelist_canonical_v1_3"
    )

    dynamic_whitelist_p = (
        dynamic_root
        / "primary_dynamic_feature_whitelist_canonical_v1_3.csv"
    )

    static_root = (
        root
        / "nw_static_receptor_descriptor_whitelist_canonical_v1_1"
    )

    static_whitelist_p = (
        static_root
        / "static_receptor_descriptor_whitelist_canonical_v1_1.csv"
    )

    static_values_p = (
        static_root
        / "static_receptor_descriptor_values_canonical_v1_1.csv"
    )

    labels_root = (
        root
        / "nw_foldwise_q95_modeling_labels_canonical_v1_3"
    )

    labels_p = (
        labels_root
        / "foldwise_q95_modeling_labels_canonical_v1_3.csv"
    )

    canonical_support_p = (
        labels_root
        / "foldwise_q95_modeling_label_support_pooled_v1_3.csv"
    )

    fold_registry_p = (
        labels_root
        / "fold_registry_canonical_v1_3.csv"
    )

    source_paths = {
        "era5": (
            root
            / "era5_historical_nw"
            / "basin_features_derived_v1_0"
            / "era5_basin_features_daily_derived_1987_2025_v1_0.csv"
        ),
        "medsea_ivt": (
            root
            / "medsea_historical_analysis"
            / "basin_coupling_canonical_v1_2"
            / "medsea_ivt_basin_coupling_daily_1987_2025_canonical_v1_2.csv"
        ),
    }

    out = (
        root
        / "nw_hydroclimate_foldwise_master_core_canonical_v1_0"
    )
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 216)
    print("NW HYDROCLIMATE — BUILD FOLDWISE MASTER CORE CANONICAL v1.0")
    print("=" * 216)

    required = [
        dynamic_whitelist_p,
        static_whitelist_p,
        static_values_p,
        labels_p,
        canonical_support_p,
        fold_registry_p,
        *source_paths.values(),
    ]

    for p in required:
        if not p.exists():
            raise SystemExit(f"Manca: {p}")

    # ------------------------------------------------------------------
    # PHASE 1/6 — validate canonical whitelists
    # ------------------------------------------------------------------
    print("\nPHASE 1/6 — validate canonical dynamic/static whitelists")
    start1 = time.time()

    dyn_w = pd.read_csv(
        dynamic_whitelist_p,
        low_memory=False,
    )

    sta_w = pd.read_csv(
        static_whitelist_p,
        low_memory=False,
    )

    sta_values = pd.read_csv(
        static_values_p,
        low_memory=False,
    )

    errors = []
    warnings = []

    if len(dyn_w) != EXPECTED_DYNAMIC_FEATURES:
        errors.append(
            f"DYNAMIC_FEATURES={len(dyn_w)}"
        )

    if len(sta_w) != EXPECTED_STATIC_FEATURES:
        errors.append(
            f"STATIC_FEATURES={len(sta_w)}"
        )

    if dyn_w["canonical_feature_name"].duplicated().any():
        errors.append(
            "DUPLICATE_DYNAMIC_CANONICAL_NAMES"
        )

    if sta_w["canonical_feature_name"].duplicated().any():
        errors.append(
            "DUPLICATE_STATIC_CANONICAL_NAMES"
        )

    all_predictor_names = (
        dyn_w["canonical_feature_name"].astype(str).tolist()
        + sta_w["canonical_feature_name"].astype(str).tolist()
    )

    forbidden_rows = []

    for name in all_predictor_names:
        hits = forbidden_name(name)
        if hits:
            forbidden_rows.append(
                {
                    "predictor": name,
                    "matched_patterns":
                        "|".join(hits),
                }
            )

    if forbidden_rows:
        errors.append(
            f"FORBIDDEN_PREDICTOR_NAMES={len(forbidden_rows)}"
        )

    if errors:
        progress(
            "PHASE 1/6",
            1,
            1,
            start1,
            f"errors={len(errors)}",
        )

        print("\nMASTER BUILD ABORTED")
        for e in errors:
            print(f" - {e}")

        raise SystemExit(2)

    progress(
        "PHASE 1/6",
        1,
        1,
        start1,
        (
            f"dynamic={len(dyn_w)} static={len(sta_w)} "
            f"total={len(all_predictor_names)}"
        ),
    )

    # ------------------------------------------------------------------
    # PHASE 2/6 — build canonical dynamic daily table
    # ------------------------------------------------------------------
    print("\nPHASE 2/6 — extract and join canonical dynamic predictors")
    start2 = time.time()

    dynamic_frames = []

    total = len(source_paths)

    for i, (source, path) in enumerate(
        source_paths.items(),
        1,
    ):
        src_w = dyn_w[
            dyn_w["source"].astype(str).eq(source)
        ].copy()

        raw = pd.read_csv(
            path,
            low_memory=False,
        )

        if "receptor_id" not in raw.columns or "date" not in raw.columns:
            raise SystemExit(
                f"{source}: receptor_id/date missing"
            )

        raw["receptor_id"] = (
            raw["receptor_id"].astype(str)
        )

        raw["date"] = pd.to_datetime(
            raw["date"],
            errors="coerce",
        )

        if raw.duplicated(
            ["receptor_id", "date"]
        ).any():
            raise SystemExit(
                f"{source}: duplicate receptor/date"
            )

        missing_source_cols = sorted(
            set(
                src_w["feature_column"]
                .astype(str)
            )
            - set(raw.columns)
        )

        if missing_source_cols:
            raise SystemExit(
                f"{source}: whitelist columns missing: "
                + ", ".join(missing_source_cols)
            )

        keep = raw[
            ["receptor_id", "date"]
            + src_w["feature_column"]
            .astype(str)
            .tolist()
        ].copy()

        rename = dict(
            zip(
                src_w["feature_column"].astype(str),
                src_w["canonical_feature_name"].astype(str),
            )
        )

        keep = keep.rename(
            columns=rename
        )

        dynamic_frames.append(
            (source, keep)
        )

        progress(
            "PHASE 2/6",
            i,
            total,
            start2,
            f"{source} | predictors={len(src_w)} rows={len(keep)}",
        )

    dynamic = dynamic_frames[0][1]

    for source, frame in dynamic_frames[1:]:
        dynamic = dynamic.merge(
            frame,
            on=["receptor_id", "date"],
            how="inner",
            validate="one_to_one",
        )

    dynamic_predictors = (
        dyn_w["canonical_feature_name"]
        .astype(str)
        .tolist()
    )

    if len(dynamic_predictors) != EXPECTED_DYNAMIC_FEATURES:
        raise SystemExit(
            "Dynamic predictor count changed unexpectedly."
        )

    # ------------------------------------------------------------------
    # PHASE 3/6 — attach labels + static descriptors
    # ------------------------------------------------------------------
    print("\nPHASE 3/6 — attach canonical modeling labels and static descriptors")
    start3 = time.time()

    labels = pd.read_csv(
        labels_p,
        low_memory=False,
    )

    labels["issue_date"] = pd.to_datetime(
        labels["issue_date"],
        errors="coerce",
    )

    if len(labels) != EXPECTED_LABEL_ROWS:
        raise SystemExit(
            f"Label rows={len(labels)}, "
            f"expected={EXPECTED_LABEL_ROWS}"
        )

    if labels["receptor_id"].nunique() != EXPECTED_MODELING_RECEPTORS:
        raise SystemExit(
            "Unexpected modeling receptor count."
        )

    if labels["fold_id"].nunique() != EXPECTED_FOLDS:
        raise SystemExit(
            "Unexpected fold count."
        )

    if labels.duplicated(
        ["fold_id", "receptor_id", "issue_date"]
    ).any():
        raise SystemExit(
            "Duplicate fold/receptor/issue_date in labels."
        )

    base_cols = [
        "fold_id",
        "receptor_id",
        "issue_date",
        "partition",
        *LABEL_COLS,
    ]

    missing_label_cols = sorted(
        set(base_cols) - set(labels.columns)
    )

    if missing_label_cols:
        raise SystemExit(
            "Missing label columns: "
            + ", ".join(missing_label_cols)
        )

    master = labels[base_cols].copy()

    master = master.merge(
        dynamic.rename(
            columns={"date": "issue_date"}
        ),
        on=["receptor_id", "issue_date"],
        how="left",
        validate="many_to_one",
        indicator="_dynamic_join",
    )

    dynamic_join_fraction = float(
        master["_dynamic_join"].eq("both").mean()
    )

    sta_feature_cols = (
        sta_w["feature_column"]
        .astype(str)
        .tolist()
    )

    missing_static_value_cols = sorted(
        set(
            ["receptor_id"] + sta_feature_cols
        )
        - set(sta_values.columns)
    )

    if missing_static_value_cols:
        raise SystemExit(
            "Missing static value columns: "
            + ", ".join(missing_static_value_cols)
        )

    static_keep = sta_values[
        ["receptor_id"] + sta_feature_cols
    ].copy()

    static_rename = dict(
        zip(
            sta_w["feature_column"].astype(str),
            sta_w["canonical_feature_name"].astype(str),
        )
    )

    static_keep = static_keep.rename(
        columns=static_rename
    )

    master = master.merge(
        static_keep,
        on="receptor_id",
        how="left",
        validate="many_to_one",
        indicator="_static_join",
    )

    static_join_fraction = float(
        master["_static_join"].eq("both").mean()
    )

    master = master.drop(
        columns=[
            "_dynamic_join",
            "_static_join",
        ]
    )

    static_predictors = (
        sta_w["canonical_feature_name"]
        .astype(str)
        .tolist()
    )

    predictor_cols = (
        dynamic_predictors
        + static_predictors
    )

    if len(predictor_cols) != EXPECTED_TOTAL_PREDICTORS:
        raise SystemExit(
            f"Predictor count={len(predictor_cols)} "
            f"expected={EXPECTED_TOTAL_PREDICTORS}"
        )

    progress(
        "PHASE 3/6",
        1,
        1,
        start3,
        (
            f"rows={len(master)} "
            f"dynamic_join={dynamic_join_fraction:.6f} "
            f"static_join={static_join_fraction:.6f}"
        ),
    )

    # ------------------------------------------------------------------
    # PHASE 4/6 — leakage, missingness, label reproduction audit
    # ------------------------------------------------------------------
    print("\nPHASE 4/6 — final leakage, missingness, and label-support audit")
    start4 = time.time()

    dup_master = int(
        master.duplicated(
            ["fold_id", "receptor_id", "issue_date"]
        ).sum()
    )

    missingness_rows = []

    dyn_meta = dyn_w.set_index(
        "canonical_feature_name"
    )

    sta_meta = sta_w.set_index(
        "canonical_feature_name"
    )

    for col in predictor_cols:
        x = pd.to_numeric(
            master[col],
            errors="coerce",
        )

        if col in dyn_meta.index:
            source = str(
                dyn_meta.loc[col, "source"]
            )
            role = str(
                dyn_meta.loc[col, "model_role"]
            )
            miss_sem = str(
                dyn_meta.loc[
                    col,
                    "missingness_semantics",
                ]
            )
            family = "DYNAMIC"
        else:
            source = "static_receptor"
            role = "PRIMARY_STATIC_DESCRIPTOR"
            miss_sem = "NO_MISSING_EXPECTED"
            family = "STATIC"

        missingness_rows.append(
            {
                "predictor": col,
                "family": family,
                "source": source,
                "model_role": role,
                "missingness_semantics": miss_sem,
                "rows": int(len(master)),
                "non_null_rows":
                    int(x.notna().sum()),
                "missing_rows":
                    int(x.isna().sum()),
                "missing_fraction":
                    float(x.isna().mean()),
            }
        )

    missingness = pd.DataFrame(
        missingness_rows
    )

    static_missing_rows = int(
        missingness.loc[
            missingness["family"].eq("STATIC"),
            "missing_rows",
        ].sum()
    )

    forbidden_after_build = []

    for col in predictor_cols:
        hits = forbidden_name(col)
        if hits:
            forbidden_after_build.append(
                {
                    "predictor": col,
                    "matched_patterns":
                        "|".join(hits),
                }
            )

    support = recompute_label_support(
        master
    )

    canonical_support = pd.read_csv(
        canonical_support_p,
        low_memory=False,
    )

    support_compare = compare_label_support(
        support,
        canonical_support,
    )

    support_mismatch_rows = int(
        (~support_compare["all_counts_match"]).sum()
    )

    progress(
        "PHASE 4/6",
        1,
        1,
        start4,
        (
            f"duplicates={dup_master} "
            f"static_missing={static_missing_rows} "
            f"support_mismatch={support_mismatch_rows}"
        ),
    )

    # ------------------------------------------------------------------
    # PHASE 5/6 — write master + data dictionary
    # ------------------------------------------------------------------
    print("\nPHASE 5/6 — write canonical master dataset and dictionary")
    start5 = time.time()

    # Order columns explicitly.
    final_cols = [
        "fold_id",
        "receptor_id",
        "issue_date",
        "partition",
        *predictor_cols,
        *LABEL_COLS,
    ]

    master = master[final_cols].copy()

    master_csv = (
        out
        / "nw_hydroclimate_foldwise_master_core_v1_0.csv.gz"
    )

    master.to_csv(
        master_csv,
        index=False,
        compression="gzip",
    )

    parquet_status = "NOT_ATTEMPTED"
    master_parquet = (
        out
        / "nw_hydroclimate_foldwise_master_core_v1_0.parquet"
    )

    try:
        master.to_parquet(
            master_parquet,
            index=False,
        )
        parquet_status = "WRITTEN"
    except Exception as exc:
        parquet_status = (
            "SKIPPED_NO_PARQUET_ENGINE_OR_WRITE_ERROR: "
            + str(exc).splitlines()[0][:180]
        )

    dictionary_rows = []

    for order, col in enumerate(
        predictor_cols,
        1,
    ):
        if col in dyn_meta.index:
            r = dyn_meta.loc[col]
            dictionary_rows.append(
                {
                    "predictor_order": order,
                    "predictor": col,
                    "family": "DYNAMIC",
                    "source": str(r["source"]),
                    "source_column":
                        str(r["feature_column"]),
                    "model_role":
                        str(r["model_role"]),
                    "missingness_semantics":
                        str(r["missingness_semantics"]),
                    "issue_time_semantics":
                        str(r["issue_time_semantics"]),
                }
            )
        else:
            r = sta_meta.loc[col]
            dictionary_rows.append(
                {
                    "predictor_order": order,
                    "predictor": col,
                    "family": "STATIC",
                    "source": "static_receptor",
                    "source_column":
                        str(r["feature_column"]),
                    "model_role":
                        str(r["model_role"]),
                    "missingness_semantics":
                        "NO_MISSING_EXPECTED",
                    "issue_time_semantics":
                        "TIME_INVARIANT",
                }
            )

    dictionary = pd.DataFrame(
        dictionary_rows
    )

    dictionary_p = (
        out
        / "nw_hydroclimate_master_predictor_dictionary_v1_0.csv"
    )

    missingness_p = (
        out
        / "nw_hydroclimate_master_missingness_v1_0.csv"
    )

    support_p = (
        out
        / "nw_hydroclimate_master_label_support_v1_0.csv"
    )

    support_compare_p = (
        out
        / "nw_hydroclimate_master_label_support_reproduction_v1_0.csv"
    )

    source_manifest_p = (
        out
        / "nw_hydroclimate_master_source_manifest_v1_0.csv"
    )

    fold_registry_out = (
        out
        / "nw_hydroclimate_master_fold_registry_v1_0.csv"
    )

    dictionary.to_csv(
        dictionary_p,
        index=False,
    )

    missingness.to_csv(
        missingness_p,
        index=False,
    )

    support.to_csv(
        support_p,
        index=False,
    )

    support_compare.drop(
        columns=["_merge"]
    ).to_csv(
        support_compare_p,
        index=False,
    )

    source_manifest = pd.DataFrame(
        [
            {
                "component": "DYNAMIC_WHITELIST",
                "path": str(dynamic_whitelist_p),
                "canonical_version": "v1.3",
            },
            {
                "component": "STATIC_WHITELIST",
                "path": str(static_whitelist_p),
                "canonical_version": "v1.1",
            },
            {
                "component": "STATIC_VALUES",
                "path": str(static_values_p),
                "canonical_version": "v1.1",
            },
            {
                "component": "MODELING_LABELS",
                "path": str(labels_p),
                "canonical_version": "v1.3",
            },
            {
                "component": "ERA5_DERIVED",
                "path": str(source_paths["era5"]),
                "canonical_version": "v1.0",
            },
            {
                "component": "MEDSEA_IVT",
                "path": str(source_paths["medsea_ivt"]),
                "canonical_version": "v1.2",
            },
        ]
    )

    source_manifest.to_csv(
        source_manifest_p,
        index=False,
    )

    shutil.copy2(
        fold_registry_p,
        fold_registry_out,
    )

    progress(
        "PHASE 5/6",
        1,
        1,
        start5,
        f"CSV written | parquet={parquet_status[:70]}",
    )

    # ------------------------------------------------------------------
    # PHASE 6/6 — final canonical audit + checksums
    # ------------------------------------------------------------------
    print("\nPHASE 6/6 — final canonical audit and checksums")
    start6 = time.time()

    if (
        len(master) != EXPECTED_LABEL_ROWS
        or dup_master
        or dynamic_join_fraction != 1.0
        or static_join_fraction != 1.0
        or len(predictor_cols)
        != EXPECTED_TOTAL_PREDICTORS
        or len(forbidden_after_build)
        or static_missing_rows
        or support_mismatch_rows
    ):
        overall = "FAIL"
    else:
        overall = (
            "PASS_WITH_EXPECTED_DYNAMIC_MISSINGNESS"
            if int(
                missingness.loc[
                    missingness["family"].eq("DYNAMIC"),
                    "missing_rows",
                ].sum()
            ) > 0
            else "PASS"
        )

    files_for_hash = [
        master_csv,
        dictionary_p,
        missingness_p,
        support_p,
        support_compare_p,
        source_manifest_p,
        fold_registry_out,
    ]

    if master_parquet.exists():
        files_for_hash.append(
            master_parquet
        )

    checksums = pd.DataFrame(
        [
            {
                "file": p.name,
                "sha256": sha256(p),
                "bytes": p.stat().st_size,
            }
            for p in files_for_hash
        ]
    )

    checksums_p = (
        out / "checksums_sha256_v1_0.csv"
    )

    checksums.to_csv(
        checksums_p,
        index=False,
    )

    dynamic_missing = (
        missingness[
            missingness["family"].eq("DYNAMIC")
            & missingness["missing_rows"].gt(0)
        ]
        .sort_values(
            "missing_fraction",
            ascending=False,
        )
    )

    audit = {
        "version": "1.0",
        "overall_status": overall,
        "master_rows": int(len(master)),
        "receptors": int(master["receptor_id"].nunique()),
        "folds": int(master["fold_id"].nunique()),
        "partitions":
            sorted(master["partition"].astype(str).unique()),
        "dynamic_predictors": EXPECTED_DYNAMIC_FEATURES,
        "static_predictors": EXPECTED_STATIC_FEATURES,
        "total_predictors": EXPECTED_TOTAL_PREDICTORS,
        "label_columns": LABEL_COLS,
        "duplicate_fold_receptor_issue_date_rows":
            dup_master,
        "dynamic_join_fraction":
            dynamic_join_fraction,
        "static_join_fraction":
            static_join_fraction,
        "forbidden_predictor_name_rows":
            int(len(forbidden_after_build)),
        "static_missing_value_count":
            static_missing_rows,
        "dynamic_predictors_with_missing_values":
            int(len(dynamic_missing)),
        "label_support_reproduction_mismatch_rows":
            support_mismatch_rows,
        "feature_imputation_performed":
            False,
        "feature_scaling_performed":
            False,
        "observed_precipitation_in_primary_master":
            False,
        "current_hydrology_value_in_predictors":
            False,
        "q95_threshold_in_predictors":
            False,
        "parquet_status":
            parquet_status,
        "master_state":
            (
                "CLOSED_CANONICAL_V1_0"
                if overall.startswith("PASS")
                else "NOT_CANONICAL"
            ),
        "next_step":
            (
                "If PASS, run a modeling-preflight on the canonical master: "
                "define model families, foldwise preprocessing, imbalance "
                "handling, calibration, metrics and baselines without touching "
                "TEST for model selection."
            ),
    }

    audit_json = (
        out
        / "nw_hydroclimate_master_audit_v1_0.json"
    )

    audit_txt = (
        out
        / "nw_hydroclimate_master_audit_v1_0.txt"
    )

    audit_json.write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    top_missing = (
        dynamic_missing[
            [
                "predictor",
                "source",
                "model_role",
                "missingness_semantics",
                "missing_fraction",
            ]
        ].head(30)
    )

    lines = [
        "=" * 216,
        "NW HYDROCLIMATE — FOLDWISE MASTER CORE CANONICAL v1.0",
        "=" * 216,
        f"OVERALL STATUS                           : {overall}",
        f"Master rows                              : {len(master)}",
        f"Receptors                                : {master['receptor_id'].nunique()}",
        f"Folds                                    : {master['fold_id'].nunique()}",
        f"Dynamic predictors                       : {EXPECTED_DYNAMIC_FEATURES}",
        f"Static predictors                        : {EXPECTED_STATIC_FEATURES}",
        f"Total predictors                         : {EXPECTED_TOTAL_PREDICTORS}",
        f"Duplicate fold/receptor/issue-date rows  : {dup_master}",
        f"Dynamic join fraction                    : {dynamic_join_fraction:.6f}",
        f"Static join fraction                     : {static_join_fraction:.6f}",
        f"Forbidden predictor names                : {len(forbidden_after_build)}",
        f"Static missing values                    : {static_missing_rows}",
        f"Dynamic predictors with missing values   : {len(dynamic_missing)}",
        f"Label-support reproduction mismatches    : {support_mismatch_rows}",
        f"Parquet                                   : {parquet_status}",
        "",
        "LABEL SUPPORT",
        support.to_string(index=False),
        "",
        "TOP DYNAMIC MISSINGNESS",
        (
            top_missing.to_string(index=False)
            if len(top_missing)
            else "NONE"
        ),
        "",
        "IMPORTANT",
        "The primary master contains no observed-precip augmentation.",
        "Dynamic NaNs are preserved; no global imputation or scaling has been applied.",
        "Any preprocessing required by a model must be fitted inside FIT only.",
        "Validation is for model selection/calibration; TEST is untouched until final evaluation.",
        "",
        f"Master CSV : {master_csv}",
        (
            f"Master Parquet: {master_parquet}"
            if master_parquet.exists()
            else "Master Parquet: not written"
        ),
        f"Dictionary : {dictionary_p}",
        f"Missingness: {missingness_p}",
        f"Audit      : {audit_json}",
        f"Output     : {out}",
    ]

    audit_txt.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    progress(
        "PHASE 6/6",
        1,
        1,
        start6,
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
