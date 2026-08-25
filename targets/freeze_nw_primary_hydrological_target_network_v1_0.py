#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
freeze_nw_primary_hydrological_target_network_v1_0.py

Congela la RETE PRIMARIA dei controlli idrologici per i 20 recettori che
dispongono attualmente di osservazioni idrologiche.

SCELTA METODOLOGICA
-------------------
Dopo la preselezione automatica dei 105 controlli e il review pack v1.1,
la rete primaria viene ridotta a una sola sezione di riferimento per recettore,
privilegiando:
- sezione idrologicamente integrativa / più a valle quando appropriato;
- uso operativo documentato nelle reti regionali di piena;
- appartenenza all'asta principale;
- permanenza nel set CORE/EXTENDED già validato.

ECCEZIONE CENTA
---------------
L'attuale dataset osservato dispone, tra i controlli raccomandati, di
Cisano sul Neva. Questa stazione appartiene al bacino del Centa ma misura
il tributario Neva, non l'intero Centa. Viene quindi congelata soltanto come
PRIMARY_PROXY_TRIBUTARY e NON come sezione integrativa dell'intero recettore.
Il recettore LIG_CENTA resta esplicitamente da completare con una sezione
sull'Arroscia e/o sul Centa a valle della confluenza quando disponibile.

IMPORTANTE
----------
- Non assegna soglie ufficiali.
- Non crea etichette di piena.
- Non elimina le stazioni secondarie/ausiliarie.
- Congela esclusivamente la scelta della rete PRIMARIA di riferimento.
- Le soglie verranno verificate solo per questa rete ridotta, con l'eccezione
  Centa che rimane proxy finché non viene completato.

INPUT
-----
nw_hydrological_target_control_preselection_v1_0/
  hydrological_target_control_preselection_v1_0.csv

nw_hydrological_threshold_dual_family_review_v1_4/
  threshold_review_control_summary_v1_4.csv   [opzionale ma atteso]

OUTPUT
------
nw_hydrological_primary_target_network_v1_0/
  hydrological_primary_target_network_v1_0.csv
  hydrological_primary_target_network_audit_v1_0.json
  hydrological_primary_target_network_audit_v1_0.txt
  hydrological_nonprimary_controls_retained_v1_0.csv
"""

from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd


PRIMARY_PLAN = {
    "LIG_BISAGNO": {
        "choices": ["bisagno_firpo_level"],
        "role": "PRIMARY_TARGET",
        "basis": "DOWNSTREAM_OPERATIONAL_CONTROL",
        "note": "Passerella Firpo is the principal downstream Bisagno control before the covered reach."
    },
    "LIG_CENTA": {
        "choices": ["centa_cisano_neva_level"],
        "role": "PRIMARY_PROXY_TRIBUTARY",
        "basis": "CURRENT_DATASET_PROXY_ONLY",
        "note": "Cisano measures the Neva tributary. It is not an integrated Centa closure."
    },
    "LIG_MAGRA": {
        "choices": ["magra_fornola_level"],
        "role": "PRIMARY_TARGET",
        "basis": "DOWNSTREAM_AFTER_MAJOR_CONFLUENCE",
        "note": "Fornola is downstream of the Magra-Vara confluence and is preferred to Nasceto."
    },
    "LIG_POLCEVERA": {
        "choices": ["polcevera_pontedecimo_level"],
        "role": "PRIMARY_TARGET",
        "basis": "ARPAL_FIDUCIARY_CONTROL",
        "note": "Pontedecimo is an ARPAL principal hydrometric control on the Polcevera."
    },
    "NW_BORMIDA": {
        "choices": ["CASSINE BORMIDA"],
        "role": "PRIMARY_TARGET",
        "basis": "DOWNSTREAM_OPERATIONAL_FLOOD_SECTION",
        "note": "Cassine is preferred to upstream branch controls for basin-integrated flood response."
    },
    "NW_CHISONE": {
        "choices": ["SAN MARTINO CHISONE"],
        "role": "PRIMARY_TARGET",
        "basis": "BEST_MAINSTEM_CONTROL_IN_VALIDATED_SET",
        "note": "Retained from the review pack; stronger margin over upstream alternative."
    },
    "NW_DORA_BALTEA": {
        "choices": ["VEROLENGO DORA BALTEA", "TAVAGNASCO DORA BALTEA"],
        "role": "PRIMARY_TARGET",
        "basis": "DOWNSTREAM_MAINSTEM_PRIORITY",
        "note": "Prefer Verolengo for the full receptor; Tavagnasco is fallback at the Valle d'Aosta outlet."
    },
    "NW_DORA_RIPARIA": {
        "choices": ["TORINO DORA RIPARIA"],
        "role": "PRIMARY_TARGET",
        "basis": "DOWNSTREAM_OPERATIONAL_FLOOD_SECTION",
        "note": "Torino is the lower-basin integrating control."
    },
    "NW_MAIRA": {
        "choices": ["RACCONIGI MAIRA", "BUSCA MAIRA"],
        "role": "PRIMARY_TARGET",
        "basis": "DOWNSTREAM_OPERATIONAL_FLOOD_SECTION_PRIORITY",
        "note": "Prefer Racconigi for lower-basin integration; Busca is fallback if unavailable."
    },
    "NW_ORBA": {
        "choices": ["BASALUZZO ORBA"],
        "role": "PRIMARY_TARGET",
        "basis": "OPERATIONAL_FLOOD_SECTION",
        "note": "Basaluzzo is retained as the operational Orba control."
    },
    "NW_ORCO": {
        "choices": ["SAN BENIGNO ORCO"],
        "role": "PRIMARY_TARGET",
        "basis": "DOWNSTREAM_OPERATIONAL_FLOOD_SECTION",
        "note": "San Benigno is preferred over the upstream Spineto control for basin integration."
    },
    "NW_PELLICE": {
        "choices": ["VILLAFRANCA PELLICE"],
        "role": "PRIMARY_TARGET",
        "basis": "DOWNSTREAM_OPERATIONAL_FLOOD_SECTION",
        "note": "Villafranca is retained as lower-basin Pellice control."
    },
    "NW_SCRIVIA": {
        "choices": ["GUAZZORA SCRIVIA"],
        "role": "PRIMARY_TARGET",
        "basis": "DOWNSTREAM_OPERATIONAL_FLOOD_SECTION",
        "note": "Guazzora is preferred to Serravalle for lower-basin integration."
    },
    "NW_SESIA": {
        "choices": ["PALESTRO SESIA"],
        "role": "PRIMARY_TARGET",
        "basis": "DOWNSTREAM_OPERATIONAL_FLOOD_SECTION",
        "note": "Palestro is the lower Sesia operational control."
    },
    "NW_STURA_DEMONTE": {
        "choices": ["FOSSANO STURA DI DEMONTE", "GAIOLA STURA DI DEMONTE"],
        "role": "PRIMARY_TARGET",
        "basis": "DOWNSTREAM_OPERATIONAL_FLOOD_SECTION_PRIORITY",
        "note": "Prefer Fossano, an operational and basin-representative section; Gaiola is fallback."
    },
    "NW_STURA_LANZO": {
        "choices": ["TORINO STURA DI LANZO", "LANZO STURA DI LANZO"],
        "role": "PRIMARY_TARGET",
        "basis": "DOWNSTREAM_OPERATIONAL_FLOOD_SECTION_PRIORITY",
        "note": "Prefer Torino for lower-basin integration; Lanzo is fallback."
    },
    "NW_TANARO_ALTO": {
        "choices": ["FARIGLIANO TANARO", "PIANTORRE TANARO"],
        "role": "PRIMARY_TARGET",
        "basis": "DOWNSTREAM_UPPER_TANARO_OPERATIONAL_SECTION_PRIORITY",
        "note": "Prefer Farigliano as the downstream integrating control of the upper Tanaro receptor; Piantorre is fallback."
    },
    "NW_TANARO_MEDIO_BASSO": {
        "choices": ["MONTECASTELLO TANARO"],
        "role": "PRIMARY_TARGET",
        "basis": "DOWNSTREAM_OPERATIONAL_FLOOD_SECTION",
        "note": "Montecastello is retained as lower Tanaro integrating control."
    },
    "NW_TOCE": {
        "choices": ["CANDOGLIA TOCE"],
        "role": "PRIMARY_TARGET",
        "basis": "DOWNSTREAM_OPERATIONAL_FLOOD_SECTION",
        "note": "Candoglia is retained as the lower Toce operational control."
    },
    "NW_VARAITA": {
        "choices": ["POLONGHERA VARAITA"],
        "role": "PRIMARY_TARGET",
        "basis": "DOWNSTREAM_OPERATIONAL_FLOOD_SECTION",
        "note": "Polonghera is retained as lower Varaita control."
    },
}


def norm(value) -> str:
    value = "" if value is None else str(value)
    value = unicodedata.normalize("NFKD", value)
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = value.upper()
    value = re.sub(r"[^A-Z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


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
        msg += f" | {current[:90]}"
    print(msg.ljust(205), end="", flush=True)
    if done >= total:
        print(flush=True)


def threshold_lookup(root: Path):
    p = (
        root
        / "nw_hydrological_threshold_dual_family_review_v1_4"
        / "threshold_review_control_summary_v1_4.csv"
    )
    if not p.exists():
        return {}, False
    df = pd.read_csv(p, low_memory=False)
    out = {}
    for _, r in df.iterrows():
        cid = str(r.get("candidate_id", ""))
        out[cid] = {
            "threshold_review_status_v1_4": str(r.get("review_status", "")),
            "stage_candidate_count_v1_4": int(pd.to_numeric(r.get("stage_candidate_count", 0), errors="coerce") or 0),
            "flow_candidate_count_v1_4": int(pd.to_numeric(r.get("flow_candidate_count", 0), errors="coerce") or 0),
        }
    return out, True


def choose_station(sub: pd.DataFrame, choices: list[str]):
    station_norm = sub["station_name"].map(norm)

    for priority, choice in enumerate(choices, 1):
        target = norm(choice)
        hit = sub[station_norm.eq(target)].copy()
        if len(hit):
            # If duplicate station rows exist, take highest preselection score.
            if "preselection_score_v1_0" in hit.columns:
                hit["_score"] = pd.to_numeric(
                    hit["preselection_score_v1_0"],
                    errors="coerce",
                ).fillna(-1e99)
                hit = hit.sort_values("_score", ascending=False)
            return hit.iloc[0], priority

    return None, None


def main():
    root = Path(__file__).resolve().parent

    pre_p = (
        root
        / "nw_hydrological_target_control_preselection_v1_0"
        / "hydrological_target_control_preselection_v1_0.csv"
    )

    out_root = root / "nw_hydrological_primary_target_network_v1_0"
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 180)
    print("NW HYDROLOGY — FREEZE PRIMARY TARGET NETWORK v1.0")
    print("=" * 180)

    if not pre_p.exists():
        raise SystemExit(f"Manca: {pre_p}")

    df = pd.read_csv(pre_p, low_memory=False)
    threshold_map, threshold_available = threshold_lookup(root)

    receptors = sorted(df["receptor_id"].astype(str).unique())
    planned = sorted(PRIMARY_PLAN)

    missing_plan = sorted(set(receptors) - set(planned))
    extra_plan = sorted(set(planned) - set(receptors))

    if missing_plan:
        raise SystemExit(
            "Recettori senza piano PRIMARY: " + ", ".join(missing_plan)
        )

    if extra_plan:
        print("WARNING — planned receptors absent from input:", ", ".join(extra_plan))

    print(f"Input recommended controls : {len(df)}")
    print(f"Receptors                  : {len(receptors)}")
    print(f"Threshold review v1.4      : {'AVAILABLE' if threshold_available else 'NOT FOUND'}")

    # ------------------------------------------------------------------
    # PHASE 1/2 — explicit primary selection
    # ------------------------------------------------------------------
    print("\nPHASE 1/2 — explicit basin-wise PRIMARY selection")
    start1 = time.time()

    selected_rows = []
    errors = []
    total = len(receptors)

    for i, receptor in enumerate(receptors, 1):
        sub = df[df["receptor_id"].astype(str).eq(receptor)].copy()
        plan = PRIMARY_PLAN[receptor]

        chosen, priority = choose_station(sub, plan["choices"])

        if chosen is None:
            errors.append(
                f"{receptor}: none of planned choices found: "
                + " | ".join(plan["choices"])
            )
        else:
            rec = chosen.to_dict()
            rec.update(
                {
                    "frozen_primary_role_v1_0": plan["role"],
                    "primary_choice_priority_used": priority,
                    "primary_choice_candidates_in_order":
                        "|".join(plan["choices"]),
                    "primary_selection_basis_v1_0": plan["basis"],
                    "primary_selection_note_v1_0": plan["note"],
                    "primary_network_version": "v1.0",
                    "primary_role_frozen": (
                        plan["role"] == "PRIMARY_TARGET"
                    ),
                    "requires_receptor_completion": (
                        plan["role"] == "PRIMARY_PROXY_TRIBUTARY"
                    ),
                }
            )

            tinfo = threshold_map.get(
                str(chosen["candidate_id"]),
                {
                    "threshold_review_status_v1_4": "NOT_AVAILABLE",
                    "stage_candidate_count_v1_4": 0,
                    "flow_candidate_count_v1_4": 0,
                },
            )
            rec.update(tinfo)
            selected_rows.append(rec)

        progress(
            "PHASE 1/2",
            i,
            total,
            start1,
            receptor,
        )

    if errors:
        print("\nERRORS")
        for e in errors:
            print(" -", e)
        raise SystemExit("PRIMARY network freeze aborted: missing planned station(s).")

    primary = pd.DataFrame(selected_rows)

    # ------------------------------------------------------------------
    # PHASE 2/2 — audit and retained non-primary controls
    # ------------------------------------------------------------------
    print("\nPHASE 2/2 — audit and retained non-primary controls")
    start2 = time.time()

    primary_ids = set(primary["candidate_id"].astype(str))

    retained = df[
        ~df["candidate_id"].astype(str).isin(primary_ids)
    ].copy()

    primary_targets = int(
        primary["frozen_primary_role_v1_0"].eq("PRIMARY_TARGET").sum()
    )
    primary_proxies = int(
        primary["frozen_primary_role_v1_0"].eq("PRIMARY_PROXY_TRIBUTARY").sum()
    )

    with_threshold_candidate = int(
        (
            pd.to_numeric(
                primary["stage_candidate_count_v1_4"],
                errors="coerce",
            ).fillna(0)
            +
            pd.to_numeric(
                primary["flow_candidate_count_v1_4"],
                errors="coerce",
            ).fillna(0)
        ).gt(0).sum()
    )

    primary_out = out_root / "hydrological_primary_target_network_v1_0.csv"
    retained_out = out_root / "hydrological_nonprimary_controls_retained_v1_0.csv"
    audit_json = out_root / "hydrological_primary_target_network_audit_v1_0.json"
    audit_txt = out_root / "hydrological_primary_target_network_audit_v1_0.txt"

    primary.to_csv(primary_out, index=False)
    retained.to_csv(retained_out, index=False)

    overall = (
        "PASS_WITH_CENTA_PROXY"
        if primary_proxies == 1
        else "PASS"
    )

    report = {
        "version": "1.0",
        "overall_status": overall,
        "input_recommended_controls": int(len(df)),
        "receptors": int(len(receptors)),
        "primary_rows": int(len(primary)),
        "frozen_primary_targets": primary_targets,
        "primary_proxy_tributary": primary_proxies,
        "nonprimary_controls_retained": int(len(retained)),
        "primary_with_local_numeric_threshold_candidate":
            with_threshold_candidate,
        "thresholds_assigned": False,
        "event_labels_created": False,
        "secondary_network_frozen": False,
        "known_gap": (
            "LIG_CENTA currently represented only by Neva at Cisano; "
            "integrated Centa/Arroscia control still required."
        ),
        "next_step": (
            "Verify official hydrometric thresholds only for the frozen "
            "PRIMARY_TARGET network; separately complete LIG_CENTA."
        ),
    }

    audit_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    display_cols = [
        "receptor_id",
        "station_name",
        "frozen_primary_role_v1_0",
        "suitability_tier",
        "variable_code",
        "threshold_review_status_v1_4",
        "stage_candidate_count_v1_4",
        "primary_selection_basis_v1_0",
    ]

    compact = primary[display_cols].copy()

    lines = [
        "=" * 180,
        "NW HYDROLOGY — PRIMARY TARGET NETWORK v1.0",
        "=" * 180,
        f"OVERALL STATUS                                  : {overall}",
        f"Input recommended controls                      : {len(df)}",
        f"Receptors                                       : {len(receptors)}",
        f"Frozen PRIMARY_TARGET                           : {primary_targets}",
        f"PRIMARY_PROXY_TRIBUTARY                         : {primary_proxies}",
        f"Non-primary controls retained                   : {len(retained)}",
        f"Primary controls with local numeric candidate   : {with_threshold_candidate}",
        "",
        "PRIMARY NETWORK",
        compact.to_string(index=False),
        "",
        "IMPORTANT",
        "Secondary/auxiliary controls are retained but not frozen for threshold verification.",
        "No threshold is assigned and no event label is created.",
        "LIG_CENTA remains a proxy-only receptor until an Arroscia/Centa integrated control is added.",
        "",
        f"Primary network : {primary_out}",
        f"Retained others : {retained_out}",
    ]

    audit_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    progress("PHASE 2/2", 1, 1, start2, "outputs written")

    print("\n" + "=" * 180)
    print("\n".join(lines[3:]))
    print("=" * 180)
    print(f"OVERALL STATUS : {overall}")
    print(f"Output         : {out_root}")
    print("=" * 180)


if __name__ == "__main__":
    main()
