#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
closeout_nw_core_release_and_interpretation_v1_0.py

FASE SUCCESSIVA ALLA CHIUSURA MASTER/MODELLO.

SCOPO
-----
1) Costruire un pacchetto canonico/release del CORE congelato.
2) Calcolare interpretazione del modello SENZA usare il final TEST per
   selezione o tuning:
   - permutation importance su F3 VALIDATION 2020-2022;
   - importanza per feature e per famiglia.
3) Consolidare le diagnostiche finali già prodotte per recettore e anno.
4) Generare un audit di readiness per la futura pipeline operativa.

IMPORTANTISSIMO
---------------
- Nessun modello viene riaddestrato.
- Nessun iperparametro, soglia o calibratore viene modificato.
- La feature importance usa F3 VALIDATION, NON F3 TEST.
- Le metriche finali per recettore/anno sono soltanto una lettura degli
  artefatti finali già prodotti.
- La permutation importance NON è causal attribution.
- Feature correlate possono condividere/ridurre l'importanza apparente.

INPUT CANONICI
--------------
nw_hydroclimate_foldwise_master_core_canonical_v1_0/
nw_dynamic_causal_feature_whitelist_canonical_v1_3/
nw_static_receptor_descriptor_whitelist_canonical_v1_1/
nw_flood_models_frozen_development_v1_2/
nw_flood_model_final_holdout_evaluation_v1_0/
basins_final/nw_receptors_final.geojson

OUTPUT
------
nw_hydroclimate_core_release_v1_0/
  runtime/
  models/
  metadata/
  validation_interpretation/
  final_diagnostics/
  release_manifest_sha256_v1_0.csv
  environment_versions_v1_0.csv
  requirements_core_v1_0.txt
  release_audit_v1_0.json
  release_audit_v1_0.txt
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import platform
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


RANDOM_STATE = 20260825
PERMUTATION_REPEATS = 5
HORIZONS = [24, 48, 72]

LABEL_COLS = {
    24: "label_extreme_within_24h",
    48: "label_extreme_within_48h",
    72: "label_extreme_within_72h",
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def pkg_version(name):
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "UNKNOWN")
    except Exception as exc:
        return f"NOT_AVAILABLE:{type(exc).__name__}"


def predict_prob(model, X):
    return model.predict_proba(X)[:, 1]


def receptor_block_permute(series, receptor_ids, rng):
    """
    Permuta i valori STATICI come mapping receptor -> valore,
    preservando il fatto che una feature statica sia costante nel recettore.
    """
    temp = pd.DataFrame(
        {
            "receptor_id": receptor_ids.astype(str).to_numpy(),
            "value": pd.to_numeric(series, errors="coerce").to_numpy(),
        }
    )

    mapping = (
        temp.groupby("receptor_id", as_index=True)["value"]
        .first()
    )

    keys = mapping.index.to_numpy()
    vals = mapping.to_numpy().copy()
    rng.shuffle(vals)

    shuffled_map = dict(zip(keys, vals))

    return np.asarray(
        [shuffled_map[str(r)] for r in receptor_ids.astype(str)],
        dtype=float,
    )


def summarize_final_receptors(df):
    """
    Aggiunge ranking per AP/Brier quando definibili.
    Non cambia le metriche finali.
    """
    out = df.copy()

    out["ap_rank_within_horizon"] = (
        out.groupby("horizon_hours")["average_precision"]
        .rank(method="min", ascending=False)
    )

    out["brier_rank_within_horizon"] = (
        out.groupby("horizon_hours")["brier_score"]
        .rank(method="min", ascending=True)
    )

    return out


def summarize_final_years(df):
    out = df.copy()

    out["ap_rank_within_horizon"] = (
        out.groupby("horizon_hours")["average_precision"]
        .rank(method="min", ascending=False)
    )

    out["brier_rank_within_horizon"] = (
        out.groupby("horizon_hours")["brier_score"]
        .rank(method="min", ascending=True)
    )

    return out


def main():
    root = Path(__file__).resolve().parent

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

    dyn_root = (
        root
        / "nw_dynamic_causal_feature_whitelist_canonical_v1_3"
    )
    dyn_whitelist_p = (
        dyn_root
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
    static_scope_p = (
        static_root
        / "receptor_static_model_scope_registry_v1_1.csv"
    )

    frozen_root = (
        root
        / "nw_flood_models_frozen_development_v1_2"
    )
    frozen_protocol_p = (
        frozen_root
        / "frozen_model_protocol_v1_2.csv"
    )
    frozen_thresholds_p = (
        frozen_root
        / "development_threshold_registry_v1_2.csv"
    )
    frozen_audit_p = (
        frozen_root
        / "freeze_audit_v1_2.json"
    )

    final_root = (
        root
        / "nw_flood_model_final_holdout_evaluation_v1_0"
    )
    final_audit_p = (
        final_root
        / "final_holdout_audit_v1_0.json"
    )
    final_prob_p = (
        final_root
        / "final_holdout_probability_metrics_v1_0.csv"
    )
    final_threshold_p = (
        final_root
        / "final_holdout_threshold_metrics_v1_0.csv"
    )
    final_ci_p = (
        final_root
        / "final_holdout_bootstrap_ci_v1_0.csv"
    )
    final_receptor_p = (
        final_root
        / "final_holdout_receptor_metrics_v1_0.csv"
    )
    final_year_p = (
        final_root
        / "final_holdout_year_metrics_v1_0.csv"
    )
    final_cluster_p = (
        final_root
        / "final_holdout_issue_cluster_metrics_v1_0.csv"
    )

    receptors_p = (
        root
        / "basins_final"
        / "nw_receptors_final.geojson"
    )

    required = [
        master_p,
        dictionary_p,
        dyn_whitelist_p,
        static_whitelist_p,
        static_values_p,
        static_scope_p,
        frozen_protocol_p,
        frozen_thresholds_p,
        frozen_audit_p,
        final_audit_p,
        final_prob_p,
        final_threshold_p,
        final_ci_p,
        final_receptor_p,
        final_year_p,
        final_cluster_p,
        receptors_p,
    ]

    for p in required:
        if not p.exists():
            raise SystemExit(f"Manca: {p}")

    out = root / "nw_hydroclimate_core_release_v1_0"

    runtime_dir = out / "runtime"
    models_dir = out / "models"
    metadata_dir = out / "metadata"
    interp_dir = out / "validation_interpretation"
    diag_dir = out / "final_diagnostics"

    for d in [
        runtime_dir,
        models_dir,
        metadata_dir,
        interp_dir,
        diag_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 220)
    print("NW HYDROCLIMATE — CORE RELEASE + INTERPRETATION v1.0")
    print("=" * 220)

    # ------------------------------------------------------------------
    # PHASE 1/6 — integrity
    # ------------------------------------------------------------------
    print("\nPHASE 1/6 — verify canonical master/frozen/final states")
    start = time.time()

    frozen_audit = json.loads(
        frozen_audit_p.read_text(encoding="utf-8")
    )
    final_audit = json.loads(
        final_audit_p.read_text(encoding="utf-8")
    )

    if (
        frozen_audit.get("overall_status")
        != "PASS_FROZEN_DEVELOPMENT__FINAL_TEST_SEALED"
    ):
        raise SystemExit(
            "Frozen development state unexpected: "
            + str(frozen_audit.get("overall_status"))
        )

    if (
        final_audit.get("overall_status")
        != "PASS_FINAL_HOLDOUT_EVALUATED__NO_FURTHER_TUNING_ALLOWED"
    ):
        raise SystemExit(
            "Final holdout state unexpected: "
            + str(final_audit.get("overall_status"))
        )

    dictionary = pd.read_csv(
        dictionary_p,
        low_memory=False,
    )

    if len(dictionary) != 97:
        raise SystemExit(
            f"Predictor dictionary rows={len(dictionary)}, expected=97"
        )

    progress(
        "PHASE 1/6",
        1,
        1,
        start,
        "canonical master + frozen model + final holdout states PASS",
    )

    # ------------------------------------------------------------------
    # PHASE 2/6 — build release bundle
    # ------------------------------------------------------------------
    print("\nPHASE 2/6 — assemble immutable CORE release bundle")
    start = time.time()

    copied = []

    metadata_sources = [
        (dictionary_p, metadata_dir / dictionary_p.name),
        (dyn_whitelist_p, metadata_dir / dyn_whitelist_p.name),
        (static_whitelist_p, metadata_dir / static_whitelist_p.name),
        (static_values_p, metadata_dir / static_values_p.name),
        (static_scope_p, metadata_dir / static_scope_p.name),
        (receptors_p, metadata_dir / receptors_p.name),
        (frozen_protocol_p, metadata_dir / frozen_protocol_p.name),
        (frozen_thresholds_p, metadata_dir / frozen_thresholds_p.name),
        (final_prob_p, diag_dir / final_prob_p.name),
        (final_threshold_p, diag_dir / final_threshold_p.name),
        (final_ci_p, diag_dir / final_ci_p.name),
        (final_cluster_p, diag_dir / final_cluster_p.name),
        (final_audit_p, diag_dir / final_audit_p.name),
    ]

    model_sources = []

    for h in HORIZONS:
        model_sources.extend(
            [
                (
                    frozen_root / "models" / f"horizon_{h}h_base_model.joblib",
                    models_dir / f"horizon_{h}h_base_model.joblib",
                ),
                (
                    frozen_root / "models" / f"horizon_{h}h_platt_calibrator.joblib",
                    models_dir / f"horizon_{h}h_platt_calibrator.joblib",
                ),
            ]
        )

    all_sources = metadata_sources + model_sources

    for i, (src, dst) in enumerate(all_sources, 1):
        if not src.exists():
            raise SystemExit(f"Manca release artifact: {src}")

        copy_file(src, dst)
        copied.append(dst)

        progress(
            "PHASE 2/6",
            i,
            len(all_sources),
            start,
            dst.name,
        )

    # ------------------------------------------------------------------
    # PHASE 3/6 — environment + requirements
    # ------------------------------------------------------------------
    print("\nPHASE 3/6 — freeze runtime environment metadata")
    start = time.time()

    env_rows = [
        {
            "component": "python",
            "version": platform.python_version(),
        },
        {
            "component": "platform",
            "version": platform.platform(),
        },
        {
            "component": "numpy",
            "version": pkg_version("numpy"),
        },
        {
            "component": "pandas",
            "version": pkg_version("pandas"),
        },
        {
            "component": "scikit-learn",
            "version": pkg_version("sklearn"),
        },
        {
            "component": "joblib",
            "version": pkg_version("joblib"),
        },
        {
            "component": "pyarrow",
            "version": pkg_version("pyarrow"),
        },
    ]

    environment = pd.DataFrame(env_rows)

    env_p = (
        runtime_dir
        / "environment_versions_v1_0.csv"
    )
    environment.to_csv(
        env_p,
        index=False,
    )
    copied.append(env_p)

    requirements_lines = []

    package_map = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "scikit-learn",
        "joblib": "joblib",
        "pyarrow": "pyarrow",
    }

    for component, package in package_map.items():
        version = environment.loc[
            environment["component"].eq(component),
            "version",
        ].iloc[0]

        if not str(version).startswith("NOT_AVAILABLE"):
            requirements_lines.append(
                f"{package}=={version}"
            )

    requirements_p = (
        runtime_dir
        / "requirements_core_v1_0.txt"
    )
    requirements_p.write_text(
        "\n".join(requirements_lines) + "\n",
        encoding="utf-8",
    )
    copied.append(requirements_p)

    progress(
        "PHASE 3/6",
        1,
        1,
        start,
        f"python={platform.python_version()} sklearn={pkg_version('sklearn')}",
    )

    # ------------------------------------------------------------------
    # PHASE 4/6 — permutation importance on F3 VALIDATION only
    # ------------------------------------------------------------------
    print("\nPHASE 4/6 — model interpretation on F3 VALIDATION only")
    start = time.time()

    master = pd.read_parquet(
        master_p
    )

    master["issue_date"] = pd.to_datetime(
        master["issue_date"],
        errors="coerce",
    )

    val = master[
        master["fold_id"].astype(str).eq("F3")
        & master["partition"].astype(str).eq("VALIDATION")
    ].copy()

    predictor_cols = (
        dictionary["predictor"]
        .astype(str)
        .tolist()
    )

    dict_idx = dictionary.set_index(
        "predictor"
    )

    total_tasks = len(HORIZONS) * len(predictor_cols) * PERMUTATION_REPEATS
    task = 0

    importance_rows = []

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    for horizon in HORIZONS:
        label_col = LABEL_COLS[horizon]

        y_all = pd.to_numeric(
            val[label_col],
            errors="coerce",
        )

        hval = val.loc[
            y_all.notna()
        ].copy()

        y = (
            y_all.loc[y_all.notna()]
            .astype(int)
            .to_numpy()
        )

        X = hval[
            predictor_cols
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        model_p = (
            models_dir
            / f"horizon_{horizon}h_base_model.joblib"
        )

        model = joblib.load(
            model_p
        )

        base_p = predict_prob(
            model,
            X,
        )

        base_ap = float(
            average_precision_score(
                y,
                base_p,
            )
        )

        for feature in predictor_cols:
            family = str(
                dict_idx.loc[
                    feature,
                    "family",
                ]
            )
            source = str(
                dict_idx.loc[
                    feature,
                    "source",
                ]
            )
            model_role = str(
                dict_idx.loc[
                    feature,
                    "model_role",
                ]
            )

            drops = []

            for repeat in range(
                1,
                PERMUTATION_REPEATS + 1,
            ):
                task += 1

                Xp = X.copy()

                if family == "STATIC":
                    Xp[feature] = receptor_block_permute(
                        X[feature],
                        hval["receptor_id"],
                        rng,
                    )
                    permutation_mode = (
                        "RECEPTOR_BLOCK_MAPPING_PERMUTATION"
                    )
                else:
                    vals = (
                        X[feature]
                        .to_numpy()
                        .copy()
                    )
                    rng.shuffle(vals)
                    Xp[feature] = vals
                    permutation_mode = (
                        "ROW_PERMUTATION_WITHIN_F3_VALIDATION"
                    )

                pp = predict_prob(
                    model,
                    Xp,
                )

                ap = float(
                    average_precision_score(
                        y,
                        pp,
                    )
                )

                drops.append(
                    base_ap - ap
                )

                progress(
                    "PHASE 4/6",
                    task,
                    total_tasks,
                    start,
                    (
                        f"{horizon}h | {feature} "
                        f"| repeat={repeat}/{PERMUTATION_REPEATS}"
                    ),
                )

            importance_rows.append(
                {
                    "horizon_hours": horizon,
                    "predictor": feature,
                    "family": family,
                    "source": source,
                    "model_role": model_role,
                    "permutation_mode":
                        permutation_mode,
                    "validation_rows":
                        int(len(hval)),
                    "validation_positive_rows":
                        int(np.sum(y == 1)),
                    "baseline_validation_average_precision":
                        base_ap,
                    "mean_ap_drop":
                        float(np.mean(drops)),
                    "std_ap_drop":
                        float(np.std(drops)),
                    "min_ap_drop":
                        float(np.min(drops)),
                    "max_ap_drop":
                        float(np.max(drops)),
                    "repeats":
                        PERMUTATION_REPEATS,
                }
            )

    importance = pd.DataFrame(
        importance_rows
    )

    importance["importance_rank"] = (
        importance.groupby(
            "horizon_hours"
        )["mean_ap_drop"]
        .rank(
            method="min",
            ascending=False,
        )
    )

    importance_p = (
        interp_dir
        / "feature_permutation_importance_f3_validation_v1_0.csv"
    )

    importance.to_csv(
        importance_p,
        index=False,
    )
    copied.append(importance_p)

    family_importance = (
        importance.groupby(
            [
                "horizon_hours",
                "family",
                "source",
            ],
            as_index=False,
        )
        .agg(
            feature_count=(
                "predictor",
                "nunique",
            ),
            summed_mean_ap_drop=(
                "mean_ap_drop",
                "sum",
            ),
            mean_feature_ap_drop=(
                "mean_ap_drop",
                "mean",
            ),
            max_feature_ap_drop=(
                "mean_ap_drop",
                "max",
            ),
        )
    )

    family_p = (
        interp_dir
        / "feature_family_importance_f3_validation_v1_0.csv"
    )

    family_importance.to_csv(
        family_p,
        index=False,
    )
    copied.append(family_p)

    # ------------------------------------------------------------------
    # PHASE 5/6 — consolidate final receptor/year diagnostics
    # ------------------------------------------------------------------
    print("\nPHASE 5/6 — consolidate final receptor/year diagnostics")
    start = time.time()

    final_receptor = pd.read_csv(
        final_receptor_p,
        low_memory=False,
    )

    final_year = pd.read_csv(
        final_year_p,
        low_memory=False,
    )

    receptor_summary = summarize_final_receptors(
        final_receptor
    )

    year_summary = summarize_final_years(
        final_year
    )

    receptor_summary_p = (
        diag_dir
        / "final_receptor_skill_summary_v1_0.csv"
    )

    year_summary_p = (
        diag_dir
        / "final_year_skill_summary_v1_0.csv"
    )

    receptor_summary.to_csv(
        receptor_summary_p,
        index=False,
    )

    year_summary.to_csv(
        year_summary_p,
        index=False,
    )

    copied.extend(
        [
            receptor_summary_p,
            year_summary_p,
        ]
    )

    top_bottom_rows = []

    for horizon in HORIZONS:
        h = receptor_summary[
            receptor_summary["horizon_hours"].eq(horizon)
        ].copy()

        eligible_ap = h[
            pd.to_numeric(
                h["average_precision"],
                errors="coerce",
            ).notna()
        ].sort_values(
            "average_precision",
            ascending=False,
        )

        for label, part in [
            ("TOP_AP", eligible_ap.head(5)),
            ("BOTTOM_AP", eligible_ap.tail(5)),
        ]:
            for _, r in part.iterrows():
                top_bottom_rows.append(
                    {
                        "horizon_hours": horizon,
                        "ranking_group": label,
                        "receptor_id": r["receptor_id"],
                        "eligible_rows":
                            r["eligible_rows"],
                        "positive_rows":
                            r["positive_rows"],
                        "average_precision":
                            r["average_precision"],
                        "roc_auc":
                            r["roc_auc"],
                        "brier_score":
                            r["brier_score"],
                    }
                )

    top_bottom = pd.DataFrame(
        top_bottom_rows
    )

    top_bottom_p = (
        diag_dir
        / "final_receptor_top_bottom_ap_v1_0.csv"
    )

    top_bottom.to_csv(
        top_bottom_p,
        index=False,
    )
    copied.append(top_bottom_p)

    progress(
        "PHASE 5/6",
        1,
        1,
        start,
        (
            f"receptor rows={len(receptor_summary)} "
            f"| year rows={len(year_summary)}"
        ),
    )

    # ------------------------------------------------------------------
    # PHASE 6/6 — release manifest + operational readiness
    # ------------------------------------------------------------------
    print("\nPHASE 6/6 — release manifest and operational readiness audit")
    start = time.time()

    manifest_rows = []

    for p in sorted(
        set(copied),
        key=lambda x: str(x),
    ):
        manifest_rows.append(
            {
                "relative_path":
                    str(p.relative_to(out)),
                "sha256":
                    sha256(p),
                "bytes":
                    p.stat().st_size,
            }
        )

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest_p = (
        out
        / "release_manifest_sha256_v1_0.csv"
    )

    manifest.to_csv(
        manifest_p,
        index=False,
    )

    top_features = (
        importance.sort_values(
            [
                "horizon_hours",
                "importance_rank",
            ]
        )
        .groupby(
            "horizon_hours",
            as_index=False,
        )
        .head(10)
    )

    top_features_p = (
        interp_dir
        / "top10_features_by_horizon_v1_0.csv"
    )

    top_features.to_csv(
        top_features_p,
        index=False,
    )

    operational_readiness = {
        "core_release_state":
            "CLOSED_IMMUTABLE_RELEASE_V1_0",
        "master_state":
            "CLOSED_CANONICAL_V1_0",
        "model_state":
            "CLOSED_FROZEN_V1_2",
        "final_holdout_state":
            "EVALUATED_NO_FURTHER_TUNING",
        "feature_interpretation_scope":
            "F3_VALIDATION_2020_2022_ONLY",
        "final_receptor_year_diagnostics_scope":
            "READ_ONLY_FROM_EXISTING_FINAL_HOLDOUT_ARTIFACTS",
        "operational_forecast_ready":
            False,
        "blocking_next_step":
            (
                "Build and validate an operational feature-equivalence registry "
                "for all 83 dynamic predictors: real-time/forecast source, "
                "latency, unit, temporal semantics and exact transformation."
            ),
        "model_retraining_required_before_beta":
            False,
        "note":
            (
                "The first beta should reproduce the frozen CORE feature semantics "
                "using data available at issue time. Direct future-NWP predictors "
                "not present in training must not be injected into CORE v1.0."
            ),
    }

    readiness_p = (
        runtime_dir
        / "operational_readiness_v1_0.json"
    )

    readiness_p.write_text(
        json.dumps(
            operational_readiness,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    release_audit = {
        "version": "1.0",
        "overall_status":
            "PASS_CORE_RELEASE_AND_INTERPRETATION",
        "release_artifacts_hashed":
            int(len(manifest)),
        "predictors":
            int(len(dictionary)),
        "permutation_repeats":
            PERMUTATION_REPEATS,
        "feature_importance_rows":
            int(len(importance)),
        "feature_importance_uses_final_test":
            False,
        "feature_importance_scope":
            "F3_VALIDATION_2020_2022",
        "model_modified":
            False,
        "thresholds_modified":
            False,
        "calibrator_modified":
            False,
        "final_holdout_retuned":
            False,
        "operational_forecast_ready":
            False,
        "next_step":
            (
                "Construct operational feature equivalence for the 83 dynamic "
                "features, then implement provider adapters and beta daily runner."
            ),
    }

    audit_json_p = (
        out
        / "release_audit_v1_0.json"
    )

    audit_txt_p = (
        out
        / "release_audit_v1_0.txt"
    )

    release_audit["top10_features_by_horizon"] = {
        str(h): (
            top_features[
                top_features["horizon_hours"].eq(h)
            ][
                [
                    "predictor",
                    "mean_ap_drop",
                    "family",
                    "source",
                ]
            ]
            .to_dict(
                orient="records"
            )
        )
        for h in HORIZONS
    }

    audit_json_p.write_text(
        json.dumps(
            release_audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    top_display = top_features[
        [
            "horizon_hours",
            "importance_rank",
            "predictor",
            "family",
            "source",
            "mean_ap_drop",
        ]
    ]

    lines = [
        "=" * 220,
        "NW HYDROCLIMATE — CORE RELEASE + INTERPRETATION v1.0",
        "=" * 220,
        "OVERALL STATUS : PASS_CORE_RELEASE_AND_INTERPRETATION",
        f"Release artifacts hashed : {len(manifest)}",
        f"Predictors               : {len(dictionary)}",
        f"Permutation repeats      : {PERMUTATION_REPEATS}",
        "Feature importance scope : F3 VALIDATION 2020-2022",
        "Feature importance uses final TEST : False",
        "Model modified           : False",
        "Thresholds modified      : False",
        "Calibrator modified      : False",
        "",
        "TOP 10 FEATURES BY HORIZON — DEVELOPMENT INTERPRETATION",
        top_display.to_string(index=False),
        "",
        "IMPORTANT",
        "Permutation importance is predictive/model-relative, not causal attribution.",
        "Correlated predictors may share or mask importance.",
        "Static descriptors are permuted by receptor mapping, not row-by-row.",
        "Final receptor/year diagnostics are read-only summaries of the already-opened final holdout.",
        "",
        "NEXT STEP",
        "Freeze the operational feature-equivalence matrix for all 83 dynamic predictors.",
        "Only after that should we write the autonomous daily beta runner.",
        "",
        f"Release bundle : {out}",
        f"Manifest       : {manifest_p}",
        f"Importance     : {importance_p}",
        f"Top features   : {top_features_p}",
        f"Readiness      : {readiness_p}",
    ]

    audit_txt_p.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    progress(
        "PHASE 6/6",
        1,
        1,
        start,
        "status=PASS_CORE_RELEASE_AND_INTERPRETATION",
    )

    print("\n" + "=" * 220)
    print("\n".join(lines[3:]))
    print("=" * 220)
    print("OVERALL STATUS : PASS_CORE_RELEASE_AND_INTERPRETATION")
    print(f"Output         : {out}")
    print("=" * 220)


if __name__ == "__main__":
    main()
