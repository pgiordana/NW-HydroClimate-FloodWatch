#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
freeze_nw_observation_time_conventions_v1_0.py

Congela le convenzioni temporali canoniche delle osservazioni NW senza
riscrivere i ~27.8 milioni di valori già standardizzati.

Prerequisiti:
- nw_observations_standardized_v1_3/
    observation_series_registry_v1_3.csv
    station_receptor_relations_v1_3.csv
    variable_dictionary_v1_3.csv
    observation_registry_audit_v1_3.json
- nw_observations_values_v1_0/
    standardized_series_manifest_v1_2.csv
    standardized_values_audit_v1_2.json

Decisioni canoniche:
ARPA Piemonte
    daily source date; nessuna inferenza subgiornaliera.

Centro Funzionale RAVDA
    UTC DOCUMENTATO dal portale ufficiale Dataview:
    "Nello scarico dei dati ... (orario UTC)".
    Quindi, per i file standardizzati già prodotti:
        effective_timestamp_utc = timestamp_source + 'Z'
    senza necessità di riscrivere tutti i CSV.gz.

ARPAL Liguria
    UTC dichiarato dal portale e già preservato nel builder.

Output:
nw_observations_standardized_v1_4/
  observation_series_registry_v1_4.csv
  station_receptor_relations_v1_4.csv
  variable_dictionary_v1_4.csv
  observation_time_conventions_v1_0.csv
  values_manifest_time_overlay_v1_0.csv
  observation_time_audit_v1_0.json
  observation_time_audit_v1_0.txt

NON modifica dati sorgente.
NON riscrive i file valori standardizzati.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


EXPECTED_REGISTRY = 1709
EXPECTED_DATA_OK = 1312

VDA_OFFICIAL_URL = "https://presidi2.regione.vda.it/str_dataview_download"

TIME_POLICIES = {
    "ARPA_PIEMONTE": {
        "time_basis": "DAILY_SOURCE_DATE",
        "effective_timezone": "DATE_ONLY",
        "timestamp_utc_rule": "",
        "documentation_status": "SOURCE_DAILY_DATE",
        "source_url": "",
        "note": (
            "Serie giornaliere: la data sorgente è conservata senza "
            "inferire un timestamp subgiornaliero."
        ),
    },
    "CENTRO_FUNZIONALE_RAVDA": {
        "time_basis": "UTC",
        "effective_timezone": "UTC",
        "timestamp_utc_rule": "timestamp_source interpreted as UTC",
        "documentation_status": "DOCUMENTED_OFFICIAL_DATAVIEW",
        "source_url": VDA_OFFICIAL_URL,
        "note": (
            "Il portale ufficiale Dataview dichiara esplicitamente che "
            "nello scarico dei dati i parametri sono riportati in orario UTC."
        ),
    },
    "ARPAL": {
        "time_basis": "UTC",
        "effective_timezone": "UTC",
        "timestamp_utc_rule": "timestamp_utc already materialized",
        "documentation_status": "PORTAL_DECLARED_UTC",
        "source_url": "",
        "note": (
            "Timestamp UTC già materializzato nei file standardizzati "
            "Liguria; start/end intervallo preservati."
        ),
    },
}


def main():
    root = Path(__file__).resolve().parent

    src = root / "nw_observations_standardized_v1_3"
    values_root = root / "nw_observations_values_v1_0"
    out = root / "nw_observations_standardized_v1_4"
    out.mkdir(parents=True, exist_ok=True)

    reg_p = src / "observation_series_registry_v1_3.csv"
    rel_p = src / "station_receptor_relations_v1_3.csv"
    dict_p = src / "variable_dictionary_v1_3.csv"
    reg_audit_p = src / "observation_registry_audit_v1_3.json"

    man_p = values_root / "standardized_series_manifest_v1_2.csv"
    values_audit_p = values_root / "standardized_values_audit_v1_2.json"

    print("=" * 132)
    print("NW OBSERVATIONS — FREEZE TIME CONVENTIONS v1.0")
    print("=" * 132)

    for p in [
        reg_p,
        rel_p,
        dict_p,
        reg_audit_p,
        man_p,
        values_audit_p,
    ]:
        if not p.exists():
            raise SystemExit(f"Manca: {p}")

    reg_audit = json.loads(reg_audit_p.read_text(encoding="utf-8"))
    values_audit = json.loads(values_audit_p.read_text(encoding="utf-8"))

    if reg_audit.get("overall_status") != "PASS":
        raise SystemExit("Registry v1.3 non PASS.")
    if values_audit.get("overall_status") != "PASS":
        raise SystemExit("Standardized values v1.2 non PASS.")

    reg = pd.read_csv(reg_p, low_memory=False)
    rel = pd.read_csv(rel_p, low_memory=False)
    vardict = pd.read_csv(dict_p, low_memory=False)
    man = pd.read_csv(man_p, low_memory=False)

    reasons = []

    if len(reg) != EXPECTED_REGISTRY:
        reasons.append(
            f"REGISTRY_ROWS={len(reg)} expected={EXPECTED_REGISTRY}"
        )

    if len(man) != EXPECTED_DATA_OK:
        reasons.append(
            f"MANIFEST_ROWS={len(man)} expected={EXPECTED_DATA_OK}"
        )

    # Add effective/canonical time fields to the registry.
    reg["time_basis_canonical"] = ""
    reg["effective_timezone"] = ""
    reg["timestamp_utc_rule"] = ""
    reg["time_documentation_status"] = ""
    reg["time_documentation_url"] = ""

    for provider, policy in TIME_POLICIES.items():
        m = reg["provider"].eq(provider)
        if not m.any():
            reasons.append(f"PROVIDER_NOT_FOUND:{provider}")
            continue

        reg.loc[m, "time_basis_canonical"] = policy["time_basis"]
        reg.loc[m, "effective_timezone"] = policy["effective_timezone"]
        reg.loc[m, "timestamp_utc_rule"] = policy["timestamp_utc_rule"]
        reg.loc[m, "time_documentation_status"] = policy[
            "documentation_status"
        ]
        reg.loc[m, "time_documentation_url"] = policy["source_url"]

        if provider == "CENTRO_FUNZIONALE_RAVDA":
            reg.loc[m, "timezone_status"] = (
                "DOCUMENTED_UTC_OFFICIAL_DATAVIEW"
            )

    unresolved_data_ok = int(
        (
            reg["scientific_status"].astype(str).str.upper().eq("DATA_OK")
            & reg["effective_timezone"].astype(str).str.strip().eq("")
        ).sum()
    )
    if unresolved_data_ok:
        reasons.append(
            f"DATA_OK_WITH_UNRESOLVED_EFFECTIVE_TIMEZONE={unresolved_data_ok}"
        )

    # Manifest overlay: no rewrite of values.
    overlay = man[
        [
            "provider",
            "source_series_id",
            "output_path",
            "timezone_status",
        ]
    ].copy()

    overlay["effective_timezone"] = overlay["provider"].map(
        lambda p: TIME_POLICIES[str(p)]["effective_timezone"]
    )
    overlay["timestamp_utc_rule"] = overlay["provider"].map(
        lambda p: TIME_POLICIES[str(p)]["timestamp_utc_rule"]
    )
    overlay["time_documentation_status"] = overlay["provider"].map(
        lambda p: TIME_POLICIES[str(p)]["documentation_status"]
    )
    overlay["time_documentation_url"] = overlay["provider"].map(
        lambda p: TIME_POLICIES[str(p)]["source_url"]
    )

    # Explicit invariant for VdA.
    vda_overlay = overlay[
        overlay["provider"].eq("CENTRO_FUNZIONALE_RAVDA")
    ]
    if len(vda_overlay) != 504:
        reasons.append(
            f"VDA_OVERLAY_ROWS={len(vda_overlay)} expected=504"
        )

    time_rows = []
    for provider, policy in TIME_POLICIES.items():
        time_rows.append({
            "provider": provider,
            **policy,
        })
    time_df = pd.DataFrame(time_rows)

    reg_out = out / "observation_series_registry_v1_4.csv"
    rel_out = out / "station_receptor_relations_v1_4.csv"
    dict_out = out / "variable_dictionary_v1_4.csv"
    time_out = out / "observation_time_conventions_v1_0.csv"
    overlay_out = out / "values_manifest_time_overlay_v1_0.csv"

    reg.to_csv(reg_out, index=False)
    rel.to_csv(rel_out, index=False)
    vardict.to_csv(dict_out, index=False)
    time_df.to_csv(time_out, index=False)
    overlay.to_csv(overlay_out, index=False)

    overall = "PASS" if not reasons else "REVIEW"

    report = {
        "version": "1.0",
        "overall_status": overall,
        "registry_rows": int(len(reg)),
        "data_ok_manifest_rows": int(len(man)),
        "unresolved_data_ok_effective_timezone": unresolved_data_ok,
        "policies": TIME_POLICIES,
        "vda_documentation": {
            "official_url": VDA_OFFICIAL_URL,
            "documented_statement": (
                "Nello scarico dei dati ... parametri (orario UTC)."
            ),
            "canonical_interpretation": (
                "timestamp_source in the standardized VdA files is UTC."
            ),
            "materialization_policy": (
                "Do not rewrite existing ~27.8M-row standardized value store; "
                "downstream code must materialize timestamp_utc from "
                "timestamp_source for VdA using this overlay."
            ),
        },
        "raw_modified": False,
        "values_rewritten": False,
        "reasons": reasons,
        "next_step": (
            "Build canonical daily station/target observation layer using "
            "the effective time rules frozen here."
        ),
    }

    (out / "observation_time_audit_v1_0.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "=" * 132,
        "NW OBSERVATIONS — FREEZE TIME CONVENTIONS v1.0",
        "=" * 132,
        f"OVERALL STATUS                         : {overall}",
        f"Registry rows                          : {len(reg)} / {EXPECTED_REGISTRY}",
        f"DATA_OK manifest rows                  : {len(man)} / {EXPECTED_DATA_OK}",
        f"DATA_OK unresolved effective timezone  : {unresolved_data_ok}",
        "",
        "CANONICAL TIME POLICIES",
        "ARPA Piemonte : daily source date",
        "RAVDA         : UTC DOCUMENTED by official Dataview download page",
        "ARPAL         : UTC portal-declared; already materialized",
        "",
        "VdA policy",
        "timestamp_source is interpreted as UTC.",
        "Existing value files are NOT rewritten; downstream uses the overlay.",
        "",
        f"Registry : {reg_out}",
        f"Time     : {time_out}",
        f"Overlay  : {overlay_out}",
    ]

    (out / "observation_time_audit_v1_0.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines[3:]))
    print("\n" + "=" * 132)
    print(f"OVERALL STATUS : {overall}")
    print(f"Output         : {out}")

    if reasons:
        print("REASONS:")
        for r in reasons:
            print(f"  - {r}")

    print("=" * 132)

    if overall != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
