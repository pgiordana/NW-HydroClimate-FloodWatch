#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_nw_floodwatch_technical_smoke_current_v1_0.py

FASE 18 — PRIMA INFERENZA TECNICA END-TO-END DEL CORE CONGELATO.

QUESTO È UN TEST TECNICO, NON UN BOLLETTINO SCIENTIFICO.

Prerequisito
------------
L'ultimo snapshot operativo deve avere:
    operational_compatibility_audit_v1_0.json

con:
    technical_smoke_inference_allowed = true

Lo script:
1) verifica l'integrità dei modelli/calibratori congelati;
2) carica esattamente i 97 predictor nello stesso ordine del training;
3) esegue i tre modelli congelati 24/48/72 h;
4) applica la calibrazione Platt esattamente come nel freeze v1.2:
       logit(raw_probability) -> frozen LogisticRegression calibrator
5) applica SOLO come diagnostica le soglie congelate di sviluppo:
       MAX_CSI
       RECALL80_MAX_PRECISION
6) produce una tabella per tutti i 20 recettori;
7) produce TXT, HTML, CSV e JSON;
8) NON interpreta automaticamente le probabilità come rischio/allerta
   se il compatibility audit non autorizza l'inferenza scientifica.

IMPORTANTE
----------
Per il run 2026-08-25:
- è fuori stagione CORE Sep-Dec;
- 30/83 feature dinamiche sono ancora mancanti;
- 2/14 P1 sono mancanti;
- le anomalie MedSea stagionali sono NaN per scelta metodologica;
- il modello HistGradientBoosting sa tecnicamente gestire NaN, ma questo
  NON rende scientificamente valida l'inferenza.

Quindi l'output deve avere:
    TECHNICAL_SMOKE_TEST__NON_SCIENTIFIC

e NON deve essere letto come:
    "probabilità reale di piena oggi".

Nessun modello, soglia o calibratore viene modificato.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


HORIZONS = [24, 48, 72]
EXPECTED_RECEPTORS = 20
EXPECTED_PREDICTORS = 97

MAX_CSI = "MAX_CSI"
RECALL80 = "RECALL80_MAX_PRECISION"


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


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def logit(p):
    p = np.clip(
        np.asarray(p, dtype=float),
        1e-6,
        1.0 - 1e-6,
    )
    return np.log(
        p / (1.0 - p)
    ).reshape(-1, 1)


def apply_platt(calibrator, raw_prob):
    return calibrator.predict_proba(
        logit(raw_prob)
    )[:, 1]


def latest_compatible_snapshot(root):
    base = (
        root
        / "nw_operational_feature_snapshot"
    )

    if not base.exists():
        raise SystemExit(
            f"Manca: {base}"
        )

    candidates = sorted(
        [
            p
            for p in base.iterdir()
            if p.is_dir()
            and (
                p
                / "operational_compatibility_audit_v1_0.json"
            ).exists()
            and (
                p
                / "operational_full_97_predictors_v1_2.parquet"
            ).exists()
        ],
        key=lambda p: p.name,
    )

    if not candidates:
        raise SystemExit(
            "Nessuno snapshot operativo con compatibility audit trovato."
        )

    return candidates[-1]


def find_first_existing(paths, label):
    for p in paths:
        if p.exists():
            return p
    raise SystemExit(
        f"Manca {label}. Cercato in:\n"
        + "\n".join(
            str(p)
            for p in paths
        )
    )


def verify_frozen_checksums(frozen_root):
    checksums_p = (
        frozen_root
        / "checksums_sha256_v1_2.csv"
    )

    if not checksums_p.exists():
        raise SystemExit(
            f"Manca frozen checksum registry: {checksums_p}"
        )

    checksums = pd.read_csv(
        checksums_p,
        low_memory=False,
    )

    failures = []

    for _, r in checksums.iterrows():
        rel = Path(
            str(r["file"])
        )
        p = (
            frozen_root
            / rel
        )

        if not p.exists():
            failures.append(
                f"MISSING:{rel}"
            )
            continue

        observed = sha256(
            p
        )
        expected = str(
            r["sha256"]
        )

        if observed != expected:
            failures.append(
                f"HASH_MISMATCH:{rel}"
            )

    return (
        len(failures) == 0,
        failures,
        checksums_p,
    )


def get_thresholds(thresholds):
    out = {}

    for h in HORIZONS:
        hh = thresholds[
            thresholds[
                "horizon_hours"
            ].astype(int).eq(h)
        ].copy()

        for policy in [
            MAX_CSI,
            RECALL80,
        ]:
            x = hh[
                hh[
                    "threshold_policy"
                ]
                .astype(str)
                .eq(policy)
            ]

            if len(x) != 1:
                raise RuntimeError(
                    f"Threshold {h}h/{policy}: rows={len(x)}"
                )

            out[
                (
                    h,
                    policy,
                )
            ] = float(
                x.iloc[0][
                    "threshold"
                ]
            )

    return out


def model_feature_check(model, predictor_order):
    if hasattr(
        model,
        "feature_names_in_",
    ):
        observed = list(
            map(
                str,
                model.feature_names_in_,
            )
        )

        return (
            observed
            == predictor_order
        )

    if hasattr(
        model,
        "n_features_in_",
    ):
        return int(
            model.n_features_in_
        ) == len(
            predictor_order
        )

    return True


def probability_bucket_for_smoke(p):
    """
    Purely descriptive numeric bucket for the technical output.
    It is NOT a hydrological risk class and MUST NOT be used in scientific mode.
    """
    if not np.isfinite(p):
        return "NA"

    if p < 0.05:
        return "<5%"
    if p < 0.15:
        return "5-15%"
    if p < 0.30:
        return "15-30%"
    if p < 0.50:
        return "30-50%"
    return ">=50%"


def main():
    root = Path(__file__).resolve().parent
    snapshot = latest_compatible_snapshot(
        root
    )
    run_id = snapshot.name

    compatibility_p = (
        snapshot
        / "operational_compatibility_audit_v1_0.json"
    )

    compatibility = json.loads(
        compatibility_p.read_text(
            encoding="utf-8"
        )
    )

    technical_allowed = bool(
        compatibility.get(
            "technical_smoke_inference_allowed",
            False,
        )
    )

    scientific_allowed = bool(
        compatibility.get(
            "scientific_inference_allowed",
            False,
        )
    )

    if not technical_allowed:
        raise SystemExit(
            "Compatibility audit NON autorizza neppure lo smoke inference."
        )

    full_p = (
        snapshot
        / "operational_full_97_predictors_v1_2.parquet"
    )

    feature_summary_p = (
        snapshot
        / "operational_compatibility_feature_summary_v1_0.csv"
    )

    p1_gate_p = (
        snapshot
        / "operational_p1_p2_gate_v1_0.csv"
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

    frozen_root = (
        root
        / "nw_flood_models_frozen_development_v1_2"
    )

    threshold_p = (
        frozen_root
        / "development_threshold_registry_v1_2.csv"
    )

    protocol_p = (
        frozen_root
        / "frozen_model_protocol_v1_2.csv"
    )

    for p in [
        full_p,
        feature_summary_p,
        p1_gate_p,
        threshold_p,
        protocol_p,
    ]:
        if not p.exists():
            raise SystemExit(
                f"Manca: {p}"
            )

    print("=" * 220)
    print("NW HYDROCLIMATE — TECHNICAL SMOKE INFERENCE v1.0")
    print("=" * 220)
    print(
        "ATTENZIONE: questo run è un test tecnico end-to-end.",
        flush=True,
    )
    print(
        f"Scientific inference allowed by compatibility gate: {scientific_allowed}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # PHASE 1/5 — frozen integrity + exact structure
    # ------------------------------------------------------------------
    print(
        "\nPHASE 1/5 — verify frozen artifacts, thresholds and exact 97-predictor structure"
    )
    start = time.time()

    frozen_ok, hash_failures, checksums_p = (
        verify_frozen_checksums(
            frozen_root
        )
    )

    if not frozen_ok:
        raise SystemExit(
            "Frozen integrity failure:\n"
            + "\n".join(
                hash_failures
            )
        )

    operational = pd.read_parquet(
        full_p
    )

    dictionary = pd.read_csv(
        dictionary_p,
        low_memory=False,
    )

    predictor_order = (
        dictionary[
            "predictor"
        ]
        .astype(str)
        .tolist()
    )

    if len(
        operational
    ) != EXPECTED_RECEPTORS:
        raise SystemExit(
            f"Operational rows={len(operational)}, expected=20"
        )

    if len(
        predictor_order
    ) != EXPECTED_PREDICTORS:
        raise SystemExit(
            f"Predictors={len(predictor_order)}, expected=97"
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

    if actual_predictors != predictor_order:
        raise SystemExit(
            "Operational predictor order != frozen predictor dictionary."
        )

    threshold_registry = pd.read_csv(
        threshold_p,
        low_memory=False,
    )

    thresholds = get_thresholds(
        threshold_registry
    )

    protocol = pd.read_csv(
        protocol_p,
        low_memory=False,
    )

    if len(protocol) != 3:
        raise SystemExit(
            f"Frozen protocol rows={len(protocol)}, expected=3."
        )

    progress(
        "PHASE 1/5",
        1,
        1,
        start,
        (
            f"frozen checksums PASS | rows=20 predictors=97 "
            f"| scientific_allowed={scientific_allowed}"
        ),
    )

    # ------------------------------------------------------------------
    # PHASE 2/5 — load models/calibrators and infer
    # ------------------------------------------------------------------
    print(
        "\nPHASE 2/5 — run frozen base models and frozen Platt calibrators"
    )
    start = time.time()

    X = (
        operational[
            predictor_order
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    missing_cells = int(
        X.isna()
        .sum()
        .sum()
    )

    prediction = operational[
        [
            "receptor_id",
            "issue_date",
            "run_id",
        ]
    ].copy()

    loaded_rows = []

    for i, h in enumerate(
        HORIZONS,
        1,
    ):
        model_p = (
            frozen_root
            / "models"
            / f"horizon_{h}h_base_model.joblib"
        )

        calibrator_p = (
            frozen_root
            / "models"
            / f"horizon_{h}h_platt_calibrator.joblib"
        )

        for p in [
            model_p,
            calibrator_p,
        ]:
            if not p.exists():
                raise SystemExit(
                    f"Manca frozen artifact: {p}"
                )

        model = joblib.load(
            model_p
        )

        calibrator = joblib.load(
            calibrator_p
        )

        feature_ok = model_feature_check(
            model,
            predictor_order,
        )

        if not feature_ok:
            raise SystemExit(
                f"{h}h model feature schema mismatch."
            )

        raw_p = model.predict_proba(
            X
        )[:, 1]

        calibrated_p = apply_platt(
            calibrator,
            raw_p,
        )

        if not (
            np.isfinite(
                raw_p
            ).all()
            and np.isfinite(
                calibrated_p
            ).all()
        ):
            raise RuntimeError(
                f"{h}h returned non-finite probabilities."
            )

        if not (
            (
                raw_p
                >= 0
            ).all()
            and (
                raw_p
                <= 1
            ).all()
            and (
                calibrated_p
                >= 0
            ).all()
            and (
                calibrated_p
                <= 1
            ).all()
        ):
            raise RuntimeError(
                f"{h}h probability outside [0,1]."
            )

        prediction[
            f"raw_probability_{h}h"
        ] = raw_p

        prediction[
            f"calibrated_probability_{h}h"
        ] = calibrated_p

        t_csi = thresholds[
            (
                h,
                MAX_CSI,
            )
        ]

        t_r80 = thresholds[
            (
                h,
                RECALL80,
            )
        ]

        prediction[
            f"flag_max_csi_{h}h"
        ] = (
            calibrated_p
            >= t_csi
        )

        prediction[
            f"flag_recall80_{h}h"
        ] = (
            calibrated_p
            >= t_r80
        )

        prediction[
            f"technical_probability_bucket_{h}h"
        ] = [
            probability_bucket_for_smoke(
                p
            )
            for p in calibrated_p
        ]

        loaded_rows.append(
            {
                "horizon_hours": h,
                "model_file":
                    str(
                        model_p
                    ),
                "calibrator_file":
                    str(
                        calibrator_p
                    ),
                "feature_schema_pass":
                    feature_ok,
                "threshold_max_csi":
                    t_csi,
                "threshold_recall80":
                    t_r80,
                "raw_probability_min":
                    float(
                        np.min(
                            raw_p
                        )
                    ),
                "raw_probability_max":
                    float(
                        np.max(
                            raw_p
                        )
                    ),
                "calibrated_probability_min":
                    float(
                        np.min(
                            calibrated_p
                        )
                    ),
                "calibrated_probability_max":
                    float(
                        np.max(
                            calibrated_p
                        )
                    ),
            }
        )

        progress(
            "PHASE 2/5",
            i,
            len(
                HORIZONS
            ),
            start,
            (
                f"{h}h | calibrated range "
                f"{np.min(calibrated_p):.4f}..{np.max(calibrated_p):.4f}"
            ),
        )

    # ------------------------------------------------------------------
    # PHASE 3/5 — attach data-quality context
    # ------------------------------------------------------------------
    print(
        "\nPHASE 3/5 — attach warm-up / compatibility context per receptor"
    )
    start = time.time()

    dynamic_predictors = [
        p
        for p in predictor_order
        if not p.startswith(
            "static__"
        )
    ]

    prediction[
        "missing_dynamic_features"
    ] = (
        X[
            dynamic_predictors
        ]
        .isna()
        .sum(
            axis=1
        )
        .astype(int)
        .to_numpy()
    )

    p1_gate = pd.read_csv(
        p1_gate_p,
        low_memory=False,
    )

    p1_names = (
        p1_gate[
            p1_gate[
                "priority"
            ]
            .astype(str)
            .eq(
                "P1_TOP10_ANY_HORIZON"
            )
        ][
            "feature"
        ]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    p1_existing = [
        p
        for p in p1_names
        if p in X.columns
    ]

    prediction[
        "missing_p1_features"
    ] = (
        X[
            p1_existing
        ]
        .isna()
        .sum(
            axis=1
        )
        .astype(int)
        .to_numpy()
    )

    feature_summary = pd.read_csv(
        feature_summary_p,
        low_memory=False,
    )

    review_feature_names = set(
        feature_summary.loc[
            feature_summary[
                "compatibility_state"
            ]
            .astype(str)
            .eq(
                "REVIEW_OUTSIDE_TRAINING_SUPPORT"
            ),
            "feature",
        ]
        .astype(str)
    )

    review_existing = [
        f
        for f in review_feature_names
        if f in X.columns
    ]

    if review_existing:
        # Count current finite values belonging to a feature that has at least
        # one receptor outside training support. Exact receptor-specific state
        # remains in the compatibility audit CSV.
        prediction[
            "review_feature_values_present"
        ] = (
            X[
                review_existing
            ]
            .notna()
            .sum(
                axis=1
            )
            .astype(int)
            .to_numpy()
        )
    else:
        prediction[
            "review_feature_values_present"
        ] = 0

    if scientific_allowed:
        bulletin_mode = (
            "SCIENTIFIC_BETA"
        )
        interpretation_allowed = True
    else:
        bulletin_mode = (
            "TECHNICAL_SMOKE_TEST__NON_SCIENTIFIC"
        )
        interpretation_allowed = False

    prediction[
        "bulletin_mode"
    ] = bulletin_mode

    prediction[
        "scientific_interpretation_allowed"
    ] = interpretation_allowed

    progress(
        "PHASE 3/5",
        1,
        1,
        start,
        (
            f"dynamic missing cells={missing_cells} "
            f"| P1 count={len(p1_existing)} "
            f"| mode={bulletin_mode}"
        ),
    )

    # ------------------------------------------------------------------
    # PHASE 4/5 — technical bulletin rendering
    # ------------------------------------------------------------------
    print(
        "\nPHASE 4/5 — render technical TXT / HTML / CSV / JSON bulletin"
    )
    start = time.time()

    out = (
        root
        / "nw_floodwatch_output"
        / run_id
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_p = (
        out
        / "technical_smoke_predictions_v1_0.csv"
    )

    json_p = (
        out
        / "technical_smoke_predictions_v1_0.json"
    )

    txt_p = (
        out
        / "NW_FloodWatch_TECHNICAL_SMOKE_v1_0.txt"
    )

    html_p = (
        out
        / "NW_FloodWatch_TECHNICAL_SMOKE_v1_0.html"
    )

    model_registry_p = (
        out
        / "technical_smoke_model_registry_v1_0.csv"
    )

    audit_json_p = (
        out
        / "technical_smoke_audit_v1_0.json"
    )

    prediction = prediction.sort_values(
        "receptor_id"
    ).reset_index(
        drop=True
    )

    prediction.to_csv(
        predictions_p,
        index=False,
    )

    json_p.write_text(
        json.dumps(
            prediction.to_dict(
                orient="records"
            ),
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        loaded_rows
    ).to_csv(
        model_registry_p,
        index=False,
    )

    display = prediction[
        [
            "receptor_id",
            "calibrated_probability_24h",
            "calibrated_probability_48h",
            "calibrated_probability_72h",
            "flag_max_csi_24h",
            "flag_max_csi_48h",
            "flag_max_csi_72h",
            "flag_recall80_24h",
            "flag_recall80_48h",
            "flag_recall80_72h",
            "missing_dynamic_features",
            "missing_p1_features",
        ]
    ].copy()

    for h in HORIZONS:
        display[
            f"calibrated_probability_{h}h"
        ] = (
            100.0
            * display[
                f"calibrated_probability_{h}h"
            ]
        ).round(2)

    max_rows = []

    for h in HORIZONS:
        idx = prediction[
            f"calibrated_probability_{h}h"
        ].idxmax()

        r = prediction.loc[
            idx
        ]

        max_rows.append(
            (
                h,
                str(
                    r[
                        "receptor_id"
                    ]
                ),
                float(
                    r[
                        f"calibrated_probability_{h}h"
                    ]
                ),
            )
        )

    warning_lines = [
        "ATTENZIONE — TEST TECNICO END-TO-END.",
        "QUESTO DOCUMENTO NON È UN BOLLETTINO DI ALLERTA E NON DEVE ESSERE USATO PER DECISIONI OPERATIVE.",
    ]

    if not scientific_allowed:
        warning_lines.extend(
            [
                "La compatibilità scientifica del run NON è autorizzata.",
                (
                    "Motivo gate: "
                    + str(
                        compatibility.get(
                            "overall_status",
                            "UNKNOWN",
                        )
                    )
                ),
                (
                    f"Data: {compatibility.get('issue_date', '')}; "
                    f"in stagione CORE Sep-Dec={compatibility.get('in_core_season_sep_dec', False)}."
                ),
                (
                    f"P1 complete={compatibility.get('p1_features_complete_all_receptors', '?')}/"
                    f"{compatibility.get('p1_features_total', '?')}; "
                    f"dynamic complete={compatibility.get('dynamic_features_complete_all_receptors', '?')}/83."
                ),
                "Le probabilità seguenti servono esclusivamente a verificare il funzionamento software della catena.",
            ]
        )

    txt_lines = [
        "=" * 150,
        "NW FLOODWATCH — TECHNICAL SMOKE TEST v1.0",
        "=" * 150,
        *warning_lines,
        "",
        f"Run ID               : {run_id}",
        f"Issue date            : {prediction['issue_date'].iloc[0]}",
        f"Bulletin mode         : {bulletin_mode}",
        f"Scientific use        : {scientific_allowed}",
        f"Receptors             : {len(prediction)}",
        f"Predictors            : {len(predictor_order)}",
        f"Total missing cells   : {missing_cells}",
        "",
        "MASSIMO NUMERICO CALIBRATO PER ORIZZONTE — SOLO DIAGNOSTICA SOFTWARE",
    ]

    for h, rid, p in max_rows:
        txt_lines.append(
            f"  {h:>2} h : {rid:<24} {100*p:6.2f}%"
        )

    txt_lines.extend(
        [
            "",
            "TABELLA TECNICA — PROBABILITÀ IN PERCENTUALE",
            display.to_string(
                index=False
            ),
            "",
            "NOTE",
            "MAX_CSI e RECALL80 sono soglie congelate di sviluppo e qui vengono usate solo per verificare la pipeline.",
            "Un flag=True in questo smoke test NON è una dichiarazione di piena prevista.",
            "I valori mancanti sono gestiti tecnicamente da HistGradientBoosting ma restano una limitazione scientifica.",
            "Nessun modello, calibratore o threshold è stato modificato.",
            "",
            f"Predictions CSV : {predictions_p}",
            f"Predictions JSON: {json_p}",
            f"HTML            : {html_p}",
        ]
    )

    txt_p.write_text(
        "\n".join(
            txt_lines
        ) + "\n",
        encoding="utf-8",
    )

    # Self-contained lightweight HTML.
    rows_html = []

    for _, r in display.iterrows():
        rows_html.append(
            "<tr>"
            + f"<td>{html.escape(str(r['receptor_id']))}</td>"
            + f"<td>{r['calibrated_probability_24h']:.2f}%</td>"
            + f"<td>{r['calibrated_probability_48h']:.2f}%</td>"
            + f"<td>{r['calibrated_probability_72h']:.2f}%</td>"
            + f"<td>{html.escape(str(r['flag_max_csi_24h']))}</td>"
            + f"<td>{html.escape(str(r['flag_max_csi_48h']))}</td>"
            + f"<td>{html.escape(str(r['flag_max_csi_72h']))}</td>"
            + f"<td>{int(r['missing_dynamic_features'])}</td>"
            + f"<td>{int(r['missing_p1_features'])}</td>"
            + "</tr>"
        )

    warnings_html = "".join(
        f"<p><strong>{html.escape(x)}</strong></p>"
        for x in warning_lines
    )

    html_text = f"""<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<title>NW FloodWatch — Technical Smoke Test</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
    max-width: 1400px;
    margin: 30px auto;
    padding: 0 24px;
    line-height: 1.4;
}}
.warning {{
    border: 3px solid #111;
    padding: 16px 20px;
    margin-bottom: 24px;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 14px;
}}
th, td {{
    border: 1px solid #aaa;
    padding: 7px 8px;
    text-align: right;
}}
th:first-child, td:first-child {{
    text-align: left;
}}
.small {{
    font-size: 13px;
}}
</style>
</head>
<body>
<h1>NW FloodWatch — Technical Smoke Test v1.0</h1>
<div class="warning">
{warnings_html}
</div>

<p><strong>Run:</strong> {html.escape(run_id)}<br>
<strong>Issue date:</strong> {html.escape(str(prediction['issue_date'].iloc[0]))}<br>
<strong>Mode:</strong> {html.escape(bulletin_mode)}<br>
<strong>Scientific interpretation allowed:</strong> {scientific_allowed}</p>

<h2>Output tecnico dei modelli congelati</h2>
<table>
<thead>
<tr>
<th>Recettore</th>
<th>P 24 h</th>
<th>P 48 h</th>
<th>P 72 h</th>
<th>MAX_CSI 24</th>
<th>MAX_CSI 48</th>
<th>MAX_CSI 72</th>
<th>Dynamic NaN</th>
<th>P1 NaN</th>
</tr>
</thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>

<p class="small">
Le probabilità sono mostrate esclusivamente per il collaudo tecnico della
catena IFS/CMEMS → feature engine → modello → Platt → reporting.
In modalità NON_SCIENTIFIC non devono essere interpretate come previsione
di piena o allerta. MAX_CSI/RECALL80 sono soglie congelate di sviluppo.
</p>
</body>
</html>
"""

    html_p.write_text(
        html_text,
        encoding="utf-8",
    )

    progress(
        "PHASE 4/5",
        1,
        1,
        start,
        (
            f"TXT/HTML/CSV/JSON written | "
            f"mode={bulletin_mode}"
        ),
    )

    # ------------------------------------------------------------------
    # PHASE 5/5 — freeze audit
    # ------------------------------------------------------------------
    print(
        "\nPHASE 5/5 — freeze technical smoke audit"
    )
    start = time.time()

    any_max_csi = {
        str(h): int(
            prediction[
                f"flag_max_csi_{h}h"
            ].sum()
        )
        for h in HORIZONS
    }

    any_recall80 = {
        str(h): int(
            prediction[
                f"flag_recall80_{h}h"
            ].sum()
        )
        for h in HORIZONS
    }

    audit = {
        "version": "1.0",
        "overall_status": (
            "PASS_TECHNICAL_SMOKE_INFERENCE__NON_SCIENTIFIC"
            if not scientific_allowed
            else "PASS_SCIENTIFIC_BETA_INFERENCE"
        ),
        "run_id": run_id,
        "issue_date": str(
            prediction[
                "issue_date"
            ].iloc[0]
        ),
        "bulletin_mode": bulletin_mode,
        "compatibility_status":
            compatibility.get(
                "overall_status",
                "",
            ),
        "technical_smoke_inference_allowed":
            technical_allowed,
        "scientific_interpretation_allowed":
            scientific_allowed,
        "frozen_checksums_verified":
            frozen_ok,
        "receptors":
            int(
                len(
                    prediction
                )
            ),
        "predictors":
            int(
                len(
                    predictor_order
                )
            ),
        "input_missing_cells":
            missing_cells,
        "model_or_protocol_modified":
            False,
        "calibration_method":
            "FROZEN_PLATT_ON_LOGIT_RAW_PROBABILITY",
        "threshold_flags_max_csi_count":
            any_max_csi,
        "threshold_flags_recall80_count":
            any_recall80,
        "highest_calibrated_probability_by_horizon":
            {
                str(h): {
                    "receptor_id": rid,
                    "probability": p,
                }
                for h, rid, p
                in max_rows
            },
        "interpretation":
            (
                "Software-path verification only. "
                "Do not interpret as flood probability when scientific gate is false."
                if not scientific_allowed
                else "Prospective scientific beta output."
            ),
        "next_step":
            (
                "Use this smoke run to verify end-to-end mechanics only. "
                "Continue daily cache accumulation. From Sep 1 rebuild all current "
                "features including canonical MedSea anomalies, rerun compatibility "
                "gate, and start prospective beta if the scientific gate permits."
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

    progress(
        "PHASE 5/5",
        1,
        1,
        start,
        (
            f"status={audit['overall_status']} "
            f"| max_csi flags={any_max_csi}"
        ),
    )

    print(
        "\n"
        + "=" * 220
    )
    print(
        f"OVERALL STATUS : {audit['overall_status']}"
    )
    print(
        f"Run ID         : {run_id}"
    )
    print(
        f"Bulletin mode  : {bulletin_mode}"
    )
    print(
        f"Scientific use : {scientific_allowed}"
    )
    print(
        f"Input NaN cells: {missing_cells}"
    )
    print()
    print(
        "HIGHEST CALIBRATED NUMERIC OUTPUT — SOFTWARE DIAGNOSTIC ONLY"
    )

    for h, rid, p in max_rows:
        print(
            f"{h:>2} h | {rid:<24} | {100*p:6.2f}%"
        )

    print()
    print(
        "MAX_CSI FLAGS BY HORIZON : "
        + str(
            any_max_csi
        )
    )
    print(
        "RECALL80 FLAGS BY HORIZON: "
        + str(
            any_recall80
        )
    )
    print()
    print(
        "IMPORTANT: if Scientific use=False, none of the above is a flood forecast."
    )
    print(
        f"TXT  : {txt_p}"
    )
    print(
        f"HTML : {html_p}"
    )
    print(
        f"CSV  : {predictions_p}"
    )
    print(
        f"JSON : {json_p}"
    )
    print(
        f"Audit: {audit_json_p}"
    )
    print(
        "=" * 220
    )


if __name__ == "__main__":
    main()
