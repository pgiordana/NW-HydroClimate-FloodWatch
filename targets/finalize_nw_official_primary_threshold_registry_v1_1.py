#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
finalize_nw_official_primary_threshold_registry_v1_1.py

Completa il registro ufficiale delle soglie PRIMARIE aggiungendo le quattro
sezioni liguri ancora pending nella v1.0.

FONTI ARPAL
-----------
Valori letti dalle linee orizzontali ufficiali dei grafici idrometrici ARPAL:
- linea arancione = LIVELLO DI GUARDIA
- linea rossa    = LIVELLO DI ESONDAZIONE

Sezioni:
- Bisagno a Genova - Firpo       : guardia 3.0 m, esondazione 4.5 m
- Neva a Cisano sul Neva         : guardia 3.0 m, esondazione 4.0 m
- Magra a Fornola                : guardia 3.5 m, esondazione 5.0 m
- Polcevera a Genova-Pontedecimo : guardia 2.5 m, esondazione 4.0 m

NOTA CENTA
----------
LIG_CENTA resta rappresentato da Cisano sul Neva come PRIMARY_PROXY_TRIBUTARY:
la soglia è ufficiale per il NEVA a Cisano, NON per il Centa integrato.

IMPORTANTE
----------
- Non usa candidati numerici estratti dai dati osservati.
- Non crea soglie statistiche.
- Non converte stage<->flow.
- Non crea ancora event labels.
- Conserva tutte le soglie ufficiali disponibili.

INPUT
-----
nw_official_primary_threshold_registry_v1_0/
  official_primary_threshold_registry_v1_0.csv
  hydrological_primary_target_network_v1_1.csv

OUTPUT
------
nw_official_primary_threshold_registry_v1_1/
  hydrological_primary_target_network_v1_1.csv
  official_primary_threshold_registry_v1_1.csv
  official_primary_threshold_registry_verified_v1_1.csv
  official_primary_threshold_registry_audit_v1_1.json
  official_primary_threshold_registry_audit_v1_1.txt
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pandas as pd


ARPAL_2024_EVENT_URL = (
    "https://www.arpal.liguria.it/contenuti_statici/pubblicazioni/"
    "rapporti_eventi/2024/REM_20241016_arancioneABCDE.pdf"
)

ARPAL_2024_MULTI_EVENT_URL = (
    "https://www.arpal.liguria.it/contenuti_statici/pubblicazioni/"
    "rapporti_eventi/2024/REM_SPEDITIVO_20241007-1028_completo.pdf"
)

ARPAL_2024_POLCEVERA_URL = (
    "https://www.arpal.liguria.it/contenuti_statici/pubblicazioni/"
    "rapporti_eventi/2024/REM_20241008_rossaC_vers20241125.pdf"
)


LIGURIA_THRESHOLDS = {
    "LIG_BISAGNO": {
        "station": "bisagno_firpo_level",
        "guardia_m": 3.0,
        "esondazione_m": 4.5,
        "source_url": ARPAL_2024_EVENT_URL,
        "source_page": 14,  # PDF page number as printed; zero-index not used here
        "source_figure": "Figura 45",
        "source_caption": "Livello idrometrico (Bisagno a P.lla Firpo)",
    },
    "LIG_CENTA": {
        "station": "centa_cisano_neva_level",
        "guardia_m": 3.0,
        "esondazione_m": 4.0,
        "source_url": ARPAL_2024_EVENT_URL,
        "source_page": 13,
        "source_figure": "Figura 37",
        "source_caption": "Livello idrometrico (Neva a Cisano sul Neva)",
    },
    "LIG_MAGRA": {
        "station": "magra_fornola_level",
        "guardia_m": 3.5,
        "esondazione_m": 5.0,
        "source_url": ARPAL_2024_MULTI_EVENT_URL,
        "source_page": 11,
        "source_figure": "Figura 25",
        "source_caption": "Livello idrometrico (Magra a Fornola)",
    },
    "LIG_POLCEVERA": {
        "station": "polcevera_pontedecimo_level",
        "guardia_m": 2.5,
        "esondazione_m": 4.0,
        "source_url": ARPAL_2024_POLCEVERA_URL,
        "source_page": 12,
        "source_figure": "Figura 35",
        "source_caption": "Livello idrometrico (Genova a Pontedecimo - B)",
    },
}


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
        msg += f" | {current[:95]}"
    print(msg.ljust(215), end="", flush=True)
    if done >= total:
        print(flush=True)


def main():
    root = Path(__file__).resolve().parent

    src_root = root / "nw_official_primary_threshold_registry_v1_0"
    registry_p = src_root / "official_primary_threshold_registry_v1_0.csv"
    primary_p = src_root / "hydrological_primary_target_network_v1_1.csv"

    out_root = root / "nw_official_primary_threshold_registry_v1_1"
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 188)
    print("NW HYDROLOGY — FINALIZE OFFICIAL PRIMARY THRESHOLD REGISTRY v1.1")
    print("=" * 188)

    for p in (registry_p, primary_p):
        if not p.exists():
            raise SystemExit(f"Manca: {p}")

    registry = pd.read_csv(registry_p, low_memory=False)
    primary = pd.read_csv(primary_p, low_memory=False)

    if len(registry) != 20:
        raise SystemExit(
            f"Registro v1.0 atteso 20 righe, ottenute {len(registry)}."
        )

    # ------------------------------------------------------------------
    # PHASE 1/2 — patch exact ARPAL graph thresholds
    # ------------------------------------------------------------------
    print("\nPHASE 1/2 — insert exact ARPAL guard/esondazione levels")
    start1 = time.time()

    patched = registry.copy()
    total = len(LIGURIA_THRESHOLDS)

    for i, (receptor, cfg) in enumerate(LIGURIA_THRESHOLDS.items(), 1):
        mask = patched["receptor_id"].astype(str).eq(receptor)

        if int(mask.sum()) != 1:
            raise SystemExit(
                f"{receptor}: attesa 1 riga nel registro, trovate {int(mask.sum())}."
            )

        station = str(patched.loc[mask, "station_name"].iloc[0])
        if station != cfg["station"]:
            raise SystemExit(
                f"{receptor}: station mismatch '{station}' != '{cfg['station']}'."
            )

        patched.loc[mask, "threshold_registry_status"] = (
            "VERIFIED_OFFICIAL_GRAPH_SOURCE"
        )
        patched.loc[mask, "threshold_family"] = "STAGE"
        patched.loc[mask, "threshold_unit"] = "m"

        patched.loc[mask, "threshold_1_name"] = ""
        patched.loc[mask, "threshold_1_value"] = None

        patched.loc[mask, "threshold_2_name"] = "GUARDIA"
        patched.loc[mask, "threshold_2_value"] = cfg["guardia_m"]

        patched.loc[mask, "threshold_3_name"] = "ESONDAZIONE"
        patched.loc[mask, "threshold_3_value"] = cfg["esondazione_m"]

        patched.loc[mask, "source_entity"] = "ARPAL_CFMI_PC"
        patched.loc[mask, "source_product"] = "RAPPORTO_EVENTO_IDROGRAMMA"
        patched.loc[mask, "source_url"] = cfg["source_url"]
        patched.loc[mask, "source_snapshot_note"] = (
            f"{cfg['source_figure']} — {cfg['source_caption']}; "
            f"pagina {cfg['source_page']} del rapporto ufficiale ARPAL. "
            "Valori ricavati dalle linee orizzontali allineate alla scala "
            "numerica del grafico."
        )
        patched.loc[mask, "source_value_semantics"] = (
            "Linea arancione = LIVELLO DI GUARDIA; "
            "linea rossa = LIVELLO DI ESONDAZIONE."
        )
        patched.loc[mask, "use_same_family_as_canonical_observation"] = True

        if receptor == "LIG_CENTA":
            patched.loc[mask, "scientific_note"] = (
                "Official thresholds refer to Neva at Cisano sul Neva. "
                "This is a tributary proxy for LIG_CENTA, not an integrated "
                "Centa-basin closure."
            )
        else:
            patched.loc[mask, "scientific_note"] = (
                "Official ARPAL stage thresholds retained exactly as guard "
                "and overflow levels. No statistical threshold and no "
                "stage-flow conversion used."
            )

        progress(
            "PHASE 1/2",
            i,
            total,
            start1,
            f"{receptor} | {station}",
        )

    # ------------------------------------------------------------------
    # PHASE 2/2 — audit and canonical outputs
    # ------------------------------------------------------------------
    print("\nPHASE 2/2 — audit complete registry")
    start2 = time.time()

    verified_mask = patched["threshold_registry_status"].astype(str).isin(
        {
            "VERIFIED_OFFICIAL_SOURCE",
            "VERIFIED_OFFICIAL_GRAPH_SOURCE",
        }
    )

    verified = patched[verified_mask].copy()
    pending = patched[~verified_mask].copy()

    if len(verified) != 20 or len(pending) != 0:
        raise SystemExit(
            f"Registro non completo: verified={len(verified)}, pending={len(pending)}."
        )

    if patched["receptor_id"].nunique() != 20:
        raise SystemExit("Duplicati/mancanze nei receptor_id.")

    # Monotonicity checks where thresholds exist.
    monotonic_errors = []

    for _, row in patched.iterrows():
        vals = []
        for c in (
            "threshold_1_value",
            "threshold_2_value",
            "threshold_3_value",
        ):
            v = pd.to_numeric(row.get(c), errors="coerce")
            if pd.notna(v):
                vals.append(float(v))

        if len(vals) >= 2:
            if any(b <= a for a, b in zip(vals, vals[1:])):
                monotonic_errors.append(
                    f"{row['receptor_id']} {row['station_name']}: {vals}"
                )

    if monotonic_errors:
        raise SystemExit(
            "Soglie non monotone:\n" + "\n".join(monotonic_errors)
        )

    primary_out = out_root / "hydrological_primary_target_network_v1_1.csv"
    registry_out = out_root / "official_primary_threshold_registry_v1_1.csv"
    verified_out = out_root / "official_primary_threshold_registry_verified_v1_1.csv"
    audit_json = out_root / "official_primary_threshold_registry_audit_v1_1.json"
    audit_txt = out_root / "official_primary_threshold_registry_audit_v1_1.txt"

    primary.to_csv(primary_out, index=False)
    patched.to_csv(registry_out, index=False)
    verified.to_csv(verified_out, index=False)

    flow_n = int(patched["threshold_family"].eq("FLOW").sum())
    stage_n = int(patched["threshold_family"].eq("STAGE").sum())
    proxy_n = int(
        patched["primary_role"].astype(str).eq("PRIMARY_PROXY_TRIBUTARY").sum()
    )

    report = {
        "version": "1.1",
        "overall_status": "PASS_WITH_CENTA_PROXY",
        "primary_controls": int(len(primary)),
        "official_threshold_rows_verified": int(len(verified)),
        "flow_threshold_rows": flow_n,
        "stage_threshold_rows": stage_n,
        "pending_threshold_rows": 0,
        "primary_proxy_tributary_rows": proxy_n,
        "statistical_thresholds_created": False,
        "local_observation_candidate_values_used": False,
        "stage_flow_conversion_performed": False,
        "event_labels_created": False,
        "known_gap": (
            "LIG_CENTA is represented by Neva at Cisano sul Neva. "
            "An integrated Centa/Arroscia control remains desirable."
        ),
        "next_step": (
            "Resolve the observed series matching each threshold family and "
            "build station-day threshold exceedance states without using "
            "future information."
        ),
    }

    audit_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    compact_cols = [
        "receptor_id",
        "station_name",
        "primary_role",
        "threshold_registry_status",
        "threshold_family",
        "threshold_1_name",
        "threshold_1_value",
        "threshold_2_name",
        "threshold_2_value",
        "threshold_3_name",
        "threshold_3_value",
    ]

    lines = [
        "=" * 188,
        "NW HYDROLOGY — OFFICIAL PRIMARY THRESHOLD REGISTRY v1.1",
        "=" * 188,
        "OVERALL STATUS                              : PASS_WITH_CENTA_PROXY",
        f"Primary controls                            : {len(primary)}",
        f"Official threshold rows verified            : {len(verified)}",
        f"  FLOW rows                                 : {flow_n}",
        f"  STAGE rows                                : {stage_n}",
        "Pending threshold rows                      : 0",
        f"Primary proxy tributary rows                 : {proxy_n}",
        "",
        "REGISTRY",
        patched[compact_cols].to_string(index=False),
        "",
        "IMPORTANT",
        "All 20 primary rows now have source-grounded official thresholds.",
        "LIG_CENTA thresholds refer to Neva at Cisano, not the integrated Centa.",
        "No statistical threshold and no event label are created here.",
        "",
        f"Primary v1.1 : {primary_out}",
        f"Registry     : {registry_out}",
        f"Verified     : {verified_out}",
    ]

    audit_txt.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    progress(
        "PHASE 2/2",
        1,
        1,
        start2,
        "complete official threshold registry written",
    )

    print("\n" + "=" * 188)
    print("\n".join(lines[3:]))
    print("=" * 188)
    print("OVERALL STATUS : PASS_WITH_CENTA_PROXY")
    print(f"Output         : {out_root}")
    print("=" * 188)


if __name__ == "__main__":
    main()
