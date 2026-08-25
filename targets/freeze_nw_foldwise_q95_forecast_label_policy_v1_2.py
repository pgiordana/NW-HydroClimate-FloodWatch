#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
freeze_nw_foldwise_q95_forecast_label_policy_v1_2.py

CONGELA COME CANONICA LA POLICY DI LABELING IDROLOGICO REGIONALE.

BASE SCIENTIFICA
----------------
Il preflight v1.1 ha mostrato:
- Q95 come percentile primario comune;
- Q97.5 come severity diagnostic;
- 3 fold rolling-origin support-aware;
- 0 threshold-fit non-PASS;
- 0 pooled low-support rows;
- warning soltanto per alcuni fold receptor-specifici con pochi positivi;
- supporto pooled sufficiente in tutte le combinazioni fold/partition/horizon.

POLICY CANONICA
---------------
1) Q95 viene stimato SEPARATAMENTE per receptor e fold, SOLO sul FIT.
2) Q97.5 viene stimato allo stesso modo ed è severity diagnostic.
3) H = 24/48/72 h.
4) label_H(t) = 1 se almeno un valore osservato in t+1...t+H supera Q95_fit.
5) label_H(t) = 0 soltanto se tutti i giorni futuri necessari sono osservati
   e nessuno supera Q95_fit.
6) label_H(t) = NaN se non c'è superamento osservato ma almeno un giorno
   necessario è mancante.
7) Non si attraversa dicembre -> settembre dell'anno successivo.
8) Le soglie ufficiali restano metadata operativi e NON sono applicate
   retrospettivamente.
9) Nessuna imputazione del target.
10) Modello regionale pooled/hierarchical; la scarsità receptor-specifica è
    mantenuta come warning.
11) Le metriche finali NON devono usare random row split; il TEST è costituito
    soltanto dai test fold rolling-origin non sovrapposti.
12) Per inferenza statistica/CI, trattare la dipendenza tra issue-date vicine:
    usare aggregazione/event clustering o bootstrap per anno/evento, non
    assumere righe giornaliere indipendenti.

FOLD CANONICI
-------------
F1 FIT <= 2013 | VAL 2014-2016 | TEST 2017-2019
F2 FIT <= 2016 | VAL 2017-2019 | TEST 2020-2022
F3 FIT <= 2019 | VAL 2020-2022 | TEST 2023-2025

INPUT
-----
nw_foldwise_q95_forecast_label_preflight_v1_1/
  foldwise_q95_thresholds_v1_1.csv
  foldwise_q95_daily_state_v1_1.csv
  foldwise_q95_forecast_labels_v1_1.csv
  foldwise_q95_label_support_by_receptor_v1_1.csv
  foldwise_q95_label_support_pooled_v1_1.csv

OUTPUT
------
nw_foldwise_q95_forecast_labels_canonical_v1_2/
  foldwise_q95_thresholds_canonical_v1_2.csv
  foldwise_q95_daily_state_canonical_v1_2.csv
  foldwise_q95_forecast_labels_canonical_v1_2.csv
  foldwise_q95_label_support_by_receptor_canonical_v1_2.csv
  foldwise_q95_label_support_pooled_canonical_v1_2.csv
  fold_registry_canonical_v1_2.csv
  forecast_label_policy_canonical_v1_2.csv
  forecast_label_policy_audit_v1_2.json
  forecast_label_policy_audit_v1_2.txt
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import pandas as pd


EXPECTED_FOLDS = {
    "F1": {
        "fit_end_year": 2013,
        "validation_start_year": 2014,
        "validation_end_year": 2016,
        "test_start_year": 2017,
        "test_end_year": 2019,
    },
    "F2": {
        "fit_end_year": 2016,
        "validation_start_year": 2017,
        "validation_end_year": 2019,
        "test_start_year": 2020,
        "test_end_year": 2022,
    },
    "F3": {
        "fit_end_year": 2019,
        "validation_start_year": 2020,
        "validation_end_year": 2022,
        "test_start_year": 2023,
        "test_end_year": 2025,
    },
}

EXPECTED_HORIZONS = {24, 48, 72}
EXPECTED_RECEPTORS = 20


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
        msg += f" | {str(current)[:115]}"

    print(msg.ljust(245), end="", flush=True)
    if done >= total:
        print(flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parent

    src = root / "nw_foldwise_q95_forecast_label_preflight_v1_1"

    files = {
        "thresholds": src / "foldwise_q95_thresholds_v1_1.csv",
        "states": src / "foldwise_q95_daily_state_v1_1.csv",
        "labels": src / "foldwise_q95_forecast_labels_v1_1.csv",
        "support": src / "foldwise_q95_label_support_by_receptor_v1_1.csv",
        "pooled": src / "foldwise_q95_label_support_pooled_v1_1.csv",
    }

    out = root / "nw_foldwise_q95_forecast_labels_canonical_v1_2"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 204)
    print("NW HYDROLOGY — FREEZE FOLDWISE Q95 FORECAST LABEL POLICY v1.2")
    print("=" * 204)

    for p in files.values():
        if not p.exists():
            raise SystemExit(f"Manca: {p}")

    # ------------------------------------------------------------------
    # PHASE 1/3 — scientific audit
    # ------------------------------------------------------------------
    print("\nPHASE 1/3 — audit preflight outputs before canonical freeze")
    start1 = time.time()

    thresholds = pd.read_csv(files["thresholds"], low_memory=False)
    states = pd.read_csv(files["states"], low_memory=False)
    labels = pd.read_csv(files["labels"], low_memory=False)
    support = pd.read_csv(files["support"], low_memory=False)
    pooled = pd.read_csv(files["pooled"], low_memory=False)

    errors = []
    warnings = []

    fold_ids = set(thresholds["fold_id"].astype(str))
    if fold_ids != set(EXPECTED_FOLDS):
        errors.append(f"Fold ids inattesi: {sorted(fold_ids)}")

    if thresholds["receptor_id"].nunique() != EXPECTED_RECEPTORS:
        errors.append(
            f"Threshold receptors={thresholds['receptor_id'].nunique()} "
            f"(attesi {EXPECTED_RECEPTORS})"
        )

    if len(thresholds) != EXPECTED_RECEPTORS * len(EXPECTED_FOLDS):
        errors.append(
            f"Threshold rows={len(thresholds)} "
            f"(attese {EXPECTED_RECEPTORS * len(EXPECTED_FOLDS)})"
        )

    nonpass = thresholds[
        ~thresholds["threshold_fit_status"].astype(str).eq("PASS")
    ]
    if len(nonpass):
        errors.append(f"Threshold fit non-PASS rows={len(nonpass)}")

    if "pooled_support_flag" not in pooled.columns:
        errors.append("Manca pooled_support_flag")
    else:
        pooled_bad = pooled[
            ~pooled["pooled_support_flag"].astype(str).eq("OK")
        ]
        if len(pooled_bad):
            errors.append(f"Pooled support non-OK rows={len(pooled_bad)}")

    horizons = set(
        pd.to_numeric(pooled["horizon_hours"], errors="coerce")
        .dropna()
        .astype(int)
    )
    if horizons != EXPECTED_HORIZONS:
        errors.append(f"Horizons inattesi: {sorted(horizons)}")

    label_cols = {
        "label_extreme_within_24h",
        "label_extreme_within_48h",
        "label_extreme_within_72h",
    }
    missing_label_cols = sorted(label_cols - set(labels.columns))
    if missing_label_cols:
        errors.append(
            "Mancano label columns: " + ", ".join(missing_label_cols)
        )

    dup_labels = int(
        labels.duplicated(
            subset=["fold_id", "receptor_id", "issue_date"]
        ).sum()
    )
    if dup_labels:
        errors.append(f"Duplicate fold/receptor/issue_date labels={dup_labels}")

    sparse = int(
        support["support_flag"].astype(str).eq("SPARSE_POSITIVES").sum()
    )
    if sparse:
        warnings.append(
            f"{sparse} receptor/partition/horizon rows con positivi scarsi; "
            "warning coerente con modello pooled/hierarchical."
        )

    # Verify fold year metadata in thresholds.
    for fold_id, spec in EXPECTED_FOLDS.items():
        f = thresholds[thresholds["fold_id"].astype(str).eq(fold_id)]
        if not len(f):
            continue

        for col, expected in spec.items():
            vals = set(
                pd.to_numeric(f[col], errors="coerce")
                .dropna()
                .astype(int)
            )
            if vals != {expected}:
                errors.append(
                    f"{fold_id} {col}: {sorted(vals)} != {expected}"
                )

    progress(
        "PHASE 1/3",
        1,
        1,
        start1,
        f"errors={len(errors)} warnings={len(warnings)}",
    )

    if errors:
        print("\nAUDIT ERRORS")
        for e in errors:
            print(f" - {e}")
        raise SystemExit("CANONICAL FREEZE ABORTED")

    # ------------------------------------------------------------------
    # PHASE 2/3 — write canonical artifacts
    # ------------------------------------------------------------------
    print("\nPHASE 2/3 — freeze canonical policy and artifacts")
    start2 = time.time()

    out_files = {
        "thresholds":
            out / "foldwise_q95_thresholds_canonical_v1_2.csv",
        "states":
            out / "foldwise_q95_daily_state_canonical_v1_2.csv",
        "labels":
            out / "foldwise_q95_forecast_labels_canonical_v1_2.csv",
        "support":
            out / "foldwise_q95_label_support_by_receptor_canonical_v1_2.csv",
        "pooled":
            out / "foldwise_q95_label_support_pooled_canonical_v1_2.csv",
    }

    total_copy = len(out_files)
    for i, key in enumerate(out_files, 1):
        shutil.copy2(files[key], out_files[key])
        progress(
            "PHASE 2/3",
            i,
            total_copy,
            start2,
            out_files[key].name,
        )

    fold_registry = pd.DataFrame(
        [
            {
                "fold_id": fold_id,
                **spec,
                "threshold_fit_scope": "FIT_ONLY",
                "validation_role": "MODEL_SELECTION_ONLY",
                "test_role": "FINAL_OUT_OF_SAMPLE_EVALUATION",
            }
            for fold_id, spec in EXPECTED_FOLDS.items()
        ]
    )

    fold_registry_p = out / "fold_registry_canonical_v1_2.csv"
    fold_registry.to_csv(fold_registry_p, index=False)

    policy = pd.DataFrame(
        [
            {
                "policy_id": "L1",
                "rule":
                    "Primary extreme threshold is receptor-specific Q95 "
                    "estimated independently inside each FIT partition.",
            },
            {
                "policy_id": "L2",
                "rule":
                    "Q97.5 is a secondary severity diagnostic, also FIT-only.",
            },
            {
                "policy_id": "L3",
                "rule":
                    "Forecast horizons are 24, 48 and 72 hours, interpreted "
                    "as t+1, t+1..t+2 and t+1..t+3 daily target windows.",
            },
            {
                "policy_id": "L4",
                "rule":
                    "Positive evidence overrides missingness: any observed "
                    "future Q95 exceedance makes the window positive.",
            },
            {
                "policy_id": "L5",
                "rule":
                    "A negative label requires a complete future target "
                    "window with no Q95 exceedance.",
            },
            {
                "policy_id": "L6",
                "rule":
                    "If no positive is observed and any required future day "
                    "is missing, the label remains NaN.",
            },
            {
                "policy_id": "L7",
                "rule":
                    "No December-to-next-September bridging is allowed.",
            },
            {
                "policy_id": "L8",
                "rule":
                    "Official hydrological thresholds are retained as "
                    "operational metadata and are not retroactively applied.",
            },
            {
                "policy_id": "L9",
                "rule":
                    "No target imputation and no stage-flow conversion.",
            },
            {
                "policy_id": "L10",
                "rule":
                    "Model scope is regional pooled/hierarchical; sparse "
                    "receptor-specific folds remain explicit warnings.",
            },
            {
                "policy_id": "L11",
                "rule":
                    "No random row split. Final evaluation uses TEST periods "
                    "of the rolling-origin folds only.",
            },
            {
                "policy_id": "L12",
                "rule":
                    "Uncertainty/CI must account for clustered issue dates "
                    "within flood episodes; do not assume daily rows are "
                    "independent Bernoulli trials.",
            },
            {
                "policy_id": "L13",
                "rule":
                    "LIG_CENTA remains a Neva-at-Cisano tributary proxy and "
                    "must retain that caveat in modeling/reporting.",
            },
        ]
    )

    policy_p = out / "forecast_label_policy_canonical_v1_2.csv"
    policy.to_csv(policy_p, index=False)

    # ------------------------------------------------------------------
    # PHASE 3/3 — checksums and final audit
    # ------------------------------------------------------------------
    print("\nPHASE 3/3 — checksums and final canonical audit")
    start3 = time.time()

    checksum_rows = []
    checksum_targets = list(out_files.values()) + [
        fold_registry_p,
        policy_p,
    ]

    for p in checksum_targets:
        checksum_rows.append(
            {
                "file": p.name,
                "sha256": sha256(p),
                "bytes": p.stat().st_size,
            }
        )

    checksums = pd.DataFrame(checksum_rows)
    checksums_p = out / "checksums_sha256_canonical_v1_2.csv"
    checksums.to_csv(checksums_p, index=False)

    pooled_positive_min = int(
        pd.to_numeric(
            pooled["positive_labels"], errors="coerce"
        ).min()
    )

    pooled_clusters_min = int(
        pd.to_numeric(
            pooled["positive_issue_date_clusters"], errors="coerce"
        ).min()
    )

    report = {
        "version": "1.2",
        "overall_status":
            "PASS_WITH_PER_RECEPTOR_SPARSE_FOLDS",
        "canonical_label_policy": "Q95_FIT_ONLY",
        "severity_diagnostic": "Q97.5_FIT_ONLY",
        "folds": len(EXPECTED_FOLDS),
        "receptors": EXPECTED_RECEPTORS,
        "threshold_rows": int(len(thresholds)),
        "threshold_nonpass_rows": 0,
        "pooled_support_non_ok_rows": 0,
        "per_receptor_sparse_support_rows": sparse,
        "minimum_pooled_positive_labels":
            pooled_positive_min,
        "minimum_pooled_positive_issue_date_clusters":
            pooled_clusters_min,
        "duplicate_fold_receptor_issue_date_labels":
            dup_labels,
        "official_thresholds_applied_retroactively":
            False,
        "target_imputation_performed": False,
        "stage_flow_conversion_performed": False,
        "random_split_allowed": False,
        "centa_proxy_exception": True,
        "canonical_state":
            "CLOSED_CANONICAL_WITH_DOCUMENTED_SPARSE_RECEPTOR_WARNINGS",
        "next_step":
            "Causal feature/master-matrix preflight and leakage audit.",
        "warnings": warnings,
    }

    audit_json = out / "forecast_label_policy_audit_v1_2.json"
    audit_txt = out / "forecast_label_policy_audit_v1_2.txt"

    audit_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "=" * 204,
        "NW HYDROLOGY — CANONICAL FOLDWISE Q95 FORECAST LABEL POLICY v1.2",
        "=" * 204,
        "OVERALL STATUS                            : PASS_WITH_PER_RECEPTOR_SPARSE_FOLDS",
        "Canonical label policy                    : Q95 FIT-ONLY",
        "Severity diagnostic                       : Q97.5 FIT-ONLY",
        f"Folds                                     : {len(EXPECTED_FOLDS)}",
        f"Receptors                                 : {EXPECTED_RECEPTORS}",
        f"Threshold fit non-PASS rows               : 0",
        f"Pooled support non-OK rows                : 0",
        f"Per-receptor sparse support rows          : {sparse}",
        f"Minimum pooled positive labels            : {pooled_positive_min}",
        f"Minimum pooled positive issue-date clusters: {pooled_clusters_min}",
        f"Duplicate fold/receptor/issue-date rows   : {dup_labels}",
        "",
        "CANONICAL FOLDS",
        fold_registry.to_string(index=False),
        "",
        "POLICY",
        policy.to_string(index=False),
        "",
        "IMPORTANT",
        "This closes the label-policy branch; sparse receptor-specific folds remain documented warnings.",
        "Validation is for model selection; final performance claims use TEST partitions only.",
        "No global Q95, no retrospective official threshold, no target imputation, no random split.",
        "",
        f"Output : {out}",
    ]

    audit_txt.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    progress(
        "PHASE 3/3",
        1,
        1,
        start3,
        "canonical policy frozen",
    )

    print("\n" + "=" * 204)
    print("\n".join(lines[3:]))
    print("=" * 204)
    print("OVERALL STATUS : PASS_WITH_PER_RECEPTOR_SPARSE_FOLDS")
    print(f"Output         : {out}")
    print("=" * 204)


if __name__ == "__main__":
    main()
