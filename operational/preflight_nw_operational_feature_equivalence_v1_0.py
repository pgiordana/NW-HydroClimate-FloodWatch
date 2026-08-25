#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preflight_nw_operational_feature_equivalence_v1_0.py

FASE 12 — PREFLIGHT DI EQUIVALENZA OPERATIVA DELLE 83 FEATURE DINAMICHE.

OBIETTIVO
---------
Costruire il "contratto tecnico" tra il modello CORE congelato e la futura
pipeline giornaliera autonoma.

Lo script NON scarica ancora dati meteo/oceanografici e NON esegue previsioni.
Per ciascuna feature dinamica canonica stabilisce:

- sorgente storica usata nel training;
- sorgente operativa candidata;
- parametro/derivazione operativa;
- unità/semantica;
- disponibilità al mattino;
- necessità di cache antecedente;
- eventuale mismatch di risoluzione, modello o issue-time;
- stato di equivalenza;
- priorità, usando l'importanza già calcolata su F3 VALIDATION.

PRINCIPIO METODOLOGICO
----------------------
Il CORE è stato addestrato su ERA5 + MedSea con convenzione end-of-day t.
Una esecuzione al mattino NON può fingere che le statistiche dell'intero giorno
corrente siano già osservate.

Perciò distinguiamo:

1) equivalenza esatta/locale;
2) ricostruzione da cache di giorni completati;
3) proxy operativo disponibile ma da validare;
4) gap che impedisce la riproduzione esatta.

La prima beta può usare proxy operativi SOLO se marcata sperimentale e se
ogni sostituzione è registrata. Non cambia il modello CORE e non è un nuovo
addestramento.

SORGENTI OPERATIVE CANDIDATE
----------------------------
ATMOSFERA:
ECMWF Open Data — IFS real-time, 0.25°, GRIB2.
Parametri utili disponibili includono q/u/v su pressure levels, tcwv, msl,
precipitazione, CAPE/mucape, snow depth e volumetric soil water.
Accesso: HTTP / pacchetto Python ecmwf-opendata.

MARE:
Copernicus Marine
MEDSEA_ANALYSISFORECAST_PHY_006_013
temperature daily:
cmems_mod_med_phy-tem_anfc_4.2km_P1D-m
Accesso: Copernicus Marine Toolbox Python API.

LIMITI GIÀ NOTI
---------------
- ERA5 storico IVT: integrazione su griglia verticale più ricca; ECMWF Open
  Data ha un sottoinsieme dei pressure levels. IVT operativo è quindi proxy
  finché non ne verifichiamo l'equivalenza.
- precip_max_1h ERA5 non è riproducibile esattamente con output open IFS a
  cadenza 3 h: serve proxy/derivazione alternativa.
- statistiche "today" del training erano end-of-day; al mattino devono essere
  stimate da analisi + forecast dello stesso giorno oppure differite.
- Copernicus Marine NRT ha propria latenza: al mattino può essere disponibile
  l'ultimo giorno pubblicato, non necessariamente il giorno corrente.
- nessun NaN metodologico MedSea deve essere sostituito con zero.

OUTPUT
------
nw_operational_feature_equivalence_preflight_v1_0/
  operational_feature_equivalence_v1_0.csv
  operational_feature_blockers_v1_0.csv
  operational_provider_registry_v1_0.csv
  operational_beta_policy_v1_0.csv
  operational_priority_features_v1_0.csv
  operational_equivalence_audit_v1_0.json
  operational_equivalence_audit_v1_0.txt
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_DYNAMIC_FEATURES = 83


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
        msg += f" | {str(current)[:135]}"

    print(msg.ljust(285), end="", flush=True)
    if done >= total:
        print(flush=True)


def find_dynamic_whitelist(root: Path) -> Path:
    candidates = [
        root
        / "nw_hydroclimate_core_release_v1_0"
        / "metadata"
        / "primary_dynamic_feature_whitelist_canonical_v1_3.csv",
        root
        / "nw_dynamic_causal_feature_whitelist_canonical_v1_3"
        / "primary_dynamic_feature_whitelist_canonical_v1_3.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise SystemExit(
        "Whitelist dinamica canonica v1.3 non trovata.\n"
        + "\n".join(str(p) for p in candidates)
    )


def find_importance(root: Path) -> Path | None:
    p = (
        root
        / "nw_hydroclimate_core_release_v1_0"
        / "validation_interpretation"
        / "feature_permutation_importance_f3_validation_v1_0.csv"
    )
    return p if p.exists() else None


def is_support_companion(name: str) -> bool:
    tokens = [
        "support_weight",
        "medsea_support_baseline",
        "medsea_support_angle_wide",
        "medsea_support_robust_core",
    ]
    return any(t in name for t in tokens)


def is_pure_lag(name: str) -> bool:
    return any(
        token in name
        for token in [
            "_prev1d",
            "_prev3d",
            "_prev7d",
            "_prev14d",
        ]
    ) and "incl_today" not in name and "change_24h" not in name


def era5_parameter_mapping(name: str):
    """
    Returns:
      provider_parameter
      transformation
      temporal_resolution_note
      vertical_note
      equivalence_class
      morning_status
      requires_cache
      blocker
    """

    # Calendar feature is exact and local.
    if name.endswith("season_day"):
        return {
            "operational_provider": "LOCAL_CALENDAR",
            "provider_parameter": "issue_date",
            "transformation":
                "Compute season-day exactly with the same Sep-Dec convention as training.",
            "temporal_resolution_note": "EXACT_LOCAL",
            "vertical_resolution_note": "NOT_APPLICABLE",
            "equivalence_class": "EXACT_LOCAL",
            "morning_status": "READY",
            "requires_local_cache": False,
            "blocker": "",
        }

    # Base parameter family.
    provider_parameter = ""
    transformation = ""
    vertical_note = "NOT_APPLICABLE"
    resolution_note = "IFS_OPEN_DATA_0P25_DEG__3H_TYPICAL_CONTROL_STEPS"
    blocker = ""
    equivalence = "PROXY_REQUIRES_VALIDATION"
    morning = "AVAILABLE_AS_ANALYSIS_FORECAST_PROXY"
    cache = False

    if "ivt_" in name and "dir_" not in name:
        provider_parameter = "q,u,v pressure-level fields"
        transformation = (
            "Recompute IVT/transport metrics over the receptor using the frozen "
            "integration equations; preserve units kg m-1 s-1."
        )
        vertical_note = (
            "MISMATCH: training ERA5 IVT used denser pressure-level integration; "
            "ECMWF Open Data exposes a reduced pressure-level set."
        )
        blocker = "IVT_VERTICAL_LEVEL_EQUIVALENCE_NOT_YET_VALIDATED"

    elif "ivt_dir_sin" in name or "ivt_dir_cos" in name:
        provider_parameter = "derived from operational IVT east/north components"
        transformation = (
            "Compute IVT direction then sine/cosine exactly as frozen feature builder."
        )
        vertical_note = "INHERITS_OPERATIONAL_IVT_VERTICAL_LEVEL_MISMATCH"
        blocker = "IVT_DIRECTION_DEPENDS_ON_UNVALIDATED_OPERATIONAL_IVT_PROXY"

    elif "tcwv" in name:
        provider_parameter = "tcwv"
        transformation = (
            "Aggregate ECMWF total column water vapour to receptor/day using "
            "the same spatial and temporal statistic encoded by the feature."
        )

    elif "cape" in name:
        provider_parameter = "mucape/cape candidate"
        transformation = (
            "Construct daily mean/max CAPE analogue from available IFS fields; "
            "must validate variable-definition compatibility against ERA5 CAPE."
        )
        blocker = "CAPE_DEFINITION_AND_TEMPORAL_SAMPLING_EQUIVALENCE_NOT_YET_VALIDATED"

    elif "mslp" in name:
        provider_parameter = "msl"
        transformation = (
            "Construct receptor daily mean/min MSLP in Pa with frozen aggregation."
        )

    elif any(x in name for x in ["u925", "v925", "q925", "t925", "wind925"]):
        provider_parameter = "u,v,q,t @ 925 hPa"
        transformation = (
            "Aggregate pressure-level fields at 925 hPa; derive wind magnitude when required."
        )
        vertical_note = "925_HPA_AVAILABLE"

    elif any(x in name for x in ["u850", "v850", "q850", "t850", "wind850"]):
        provider_parameter = "u,v,q,t @ 850 hPa"
        transformation = (
            "Aggregate pressure-level fields at 850 hPa; derive wind magnitude when required."
        )
        vertical_note = "850_HPA_AVAILABLE"

    elif any(x in name for x in ["u700", "v700", "q700", "t700", "wind700"]):
        provider_parameter = "u,v,q,t @ 700 hPa"
        transformation = (
            "Aggregate pressure-level fields at 700 hPa; derive wind magnitude when required."
        )
        vertical_note = "700_HPA_AVAILABLE"

    elif "qwind925_proxy" in name:
        provider_parameter = "q,u,v @ 925 hPa"
        transformation = "Recompute q × wind proxy with frozen formula."
        vertical_note = "925_HPA_AVAILABLE"

    elif "qwind850_proxy" in name:
        provider_parameter = "q,u,v @ 850 hPa"
        transformation = "Recompute q × wind proxy with frozen formula."
        vertical_note = "850_HPA_AVAILABLE"

    elif "qwind700_proxy" in name:
        provider_parameter = "q,u,v @ 700 hPa"
        transformation = "Recompute q × wind proxy with frozen formula."
        vertical_note = "700_HPA_AVAILABLE"

    elif "precip_max_1h" in name or "precip_max1h" in name:
        provider_parameter = "tp/tprate"
        transformation = (
            "Approximate high-intensity precipitation from available operational "
            "forecast increments/rates; DO NOT call it exact 1-hour maximum."
        )
        resolution_note = (
            "BLOCKING_MISMATCH: open IFS control forecast output cadence does not "
            "reproduce ERA5 hourly maximum exactly."
        )
        blocker = "PRECIP_1H_MAX_NOT_EXACTLY_REPRODUCIBLE_FROM_OPEN_IFS_CADENCE"
        equivalence = "NON_EXACT_PROXY"
        morning = "PROXY_ONLY"

    elif "precip" in name:
        provider_parameter = "tp"
        transformation = (
            "Derive precipitation accumulation using forecast increments and frozen "
            "daily/rolling-window definitions; convert to mm."
        )

    elif "soil_water_l1" in name:
        provider_parameter = "vsw level=1"
        transformation = (
            "Use volumetric soil water layer 1; validate IFS-vs-ERA5 distribution before beta interpretation."
        )
        blocker = "MODEL_SYSTEM_DISTRIBUTION_SHIFT_NOT_YET_VALIDATED"

    elif "soil_water_l2" in name:
        provider_parameter = "vsw level=2"
        transformation = (
            "Use volumetric soil water layer 2; validate IFS-vs-ERA5 distribution before beta interpretation."
        )
        blocker = "MODEL_SYSTEM_DISTRIBUTION_SHIFT_NOT_YET_VALIDATED"

    elif "soil_water_l3" in name:
        provider_parameter = "vsw level=3 (IFS Open Data)"
        transformation = (
            "Use volumetric soil water layer 3; validate layer definition/distribution against ERA5."
        )
        blocker = "MODEL_SYSTEM_DISTRIBUTION_SHIFT_NOT_YET_VALIDATED"

    elif "soil_profile_mean" in name:
        provider_parameter = "vsw levels 1-3"
        transformation = (
            "Recompute frozen mean soil-profile feature from operational layers 1-3."
        )
        blocker = "MODEL_SYSTEM_DISTRIBUTION_SHIFT_NOT_YET_VALIDATED"

    elif "snow_depth_mwe" in name:
        provider_parameter = "sd"
        transformation = (
            "Use snow depth water equivalent and frozen receptor aggregation."
        )

    else:
        provider_parameter = "UNMAPPED"
        transformation = "Manual mapping required."
        equivalence = "UNMAPPED"
        morning = "BLOCKED"
        blocker = "UNMAPPED_ERA5_FEATURE"

    # Derived lag/current semantics.
    if is_pure_lag(name):
        cache = True
        morning = (
            "READY_AFTER_OPERATIONAL_DAILY_CACHE"
            if equivalence != "UNMAPPED"
            else "BLOCKED"
        )
        transformation += (
            " Feature is antecedent-only: derive from completed operational daily cache."
        )

    if "change_24h" in name:
        cache = True
        morning = "CURRENT_PROXY_PLUS_PREVIOUS_DAY_CACHE"
        transformation += (
            " Requires current operational proxy and previous completed-day cache."
        )
        if not blocker:
            blocker = "CURRENT_DAY_PROXY_NOT_IDENTICAL_TO_END_OF_DAY_ERA5_TRAINING_STATE"

    if "incl_today" in name:
        cache = True
        morning = "CURRENT_DAY_ANALYSIS_FORECAST_FILL_REQUIRED"
        transformation += (
            " Includes the current day; at a morning run the incomplete day must be "
            "estimated from analysis + forecast or the issue convention must change."
        )
        if not blocker:
            blocker = "MORNING_RUN_CANNOT_OBSERVE_FULL_TRAINING_DAY_T"

    # Same-day/raw daily features were end-of-day during training.
    current_markers = [
        "_mean",
        "_max",
        "_min",
        "_sum",
        "soil_water",
        "snow_depth",
        "qwind",
    ]

    if (
        any(m in name for m in current_markers)
        and not is_pure_lag(name)
        and "prev" not in name
        and "incl_today" not in name
        and "change_24h" not in name
    ):
        if morning == "AVAILABLE_AS_ANALYSIS_FORECAST_PROXY":
            morning = "CURRENT_DAY_ANALYSIS_FORECAST_FILL_REQUIRED"
        if not blocker:
            blocker = "MORNING_RUN_CANNOT_OBSERVE_FULL_TRAINING_DAY_T"

    return {
        "operational_provider": "ECMWF_OPEN_DATA_IFS",
        "provider_parameter": provider_parameter,
        "transformation": transformation,
        "temporal_resolution_note": resolution_note,
        "vertical_resolution_note": vertical_note,
        "equivalence_class": equivalence,
        "morning_status": morning,
        "requires_local_cache": cache,
        "blocker": blocker,
    }


def medsea_mapping(name: str):
    if is_support_companion(name):
        return {
            "operational_provider": "LOCAL_DERIVED_FROM_OPERATIONAL_IVT",
            "provider_parameter": "operational IVT + frozen MedSea geometry/support algorithm",
            "transformation": (
                "Recompute frozen corridor/support weights and robustness flags. "
                "Preserve no-support semantics; never replace no-support with zero "
                "for marine physical values."
            ),
            "temporal_resolution_note": "DERIVED_PER_ISSUE_DAY",
            "vertical_resolution_note": "INHERITS_OPERATIONAL_IVT_LIMITATIONS",
            "equivalence_class": "DERIVED_PROXY_REQUIRES_IVT_VALIDATION",
            "morning_status": "AVAILABLE_IF_OPERATIONAL_IVT_AVAILABLE",
            "requires_local_cache": False,
            "blocker": "INHERITS_OPERATIONAL_IVT_EQUIVALENCE_GAP",
        }

    if "x_ivt_proxy" in name:
        return {
            "operational_provider": "COPERNICUS_MARINE_PLUS_ECMWF_OPEN_DATA_IFS",
            "provider_parameter": "MedSea temperature/OHC field × operational IVT",
            "transformation": (
                "Recompute frozen SST/OHC anomaly × IVT interaction after the "
                "marine corridor and IVT proxy have been constructed."
            ),
            "temporal_resolution_note":
                "CMEMS_DAILY_NRT_PLUS_OPERATIONAL_IVT",
            "vertical_resolution_note":
                "MARINE_141_LEVEL_PRODUCT__ATMOSPHERIC_IVT_PROXY_LIMITATION",
            "equivalence_class": "PROXY_REQUIRES_VALIDATION",
            "morning_status": "LATEST_AVAILABLE_MARINE_DAY_PLUS_OPERATIONAL_IVT",
            "requires_local_cache": True,
            "blocker":
                "CMEMS_PUBLICATION_LATENCY_AND_OPERATIONAL_IVT_EQUIVALENCE_NOT_YET_VALIDATED",
        }

    if "sst_anom" in name:
        parameter = "thetao near-surface from daily temperature dataset"
        transform = (
            "Extract near-surface temperature, compute frozen corridor statistic "
            "and anomaly against the frozen 1991-2020 climatology."
        )
    elif "ohc" in name:
        parameter = "thetao profile 0-100 m"
        transform = (
            "Integrate 0-100 m heat content using the same physical convention "
            "as the historical builder; compute corridor/anomaly if required."
        )
    elif "tmean" in name:
        parameter = "thetao profile 0-100 m"
        transform = (
            "Compute 0-100 m mean temperature with frozen depth/corridor convention; "
            "derive anomaly against frozen climatology when required."
        )
    else:
        return {
            "operational_provider": "UNMAPPED",
            "provider_parameter": "UNMAPPED",
            "transformation": "Manual mapping required.",
            "temporal_resolution_note": "UNKNOWN",
            "vertical_resolution_note": "UNKNOWN",
            "equivalence_class": "UNMAPPED",
            "morning_status": "BLOCKED",
            "requires_local_cache": False,
            "blocker": "UNMAPPED_MEDSEA_FEATURE",
        }

    return {
        "operational_provider": "COPERNICUS_MARINE_MEDSEA_ANALYSISFORECAST",
        "provider_parameter": parameter,
        "transformation": transform,
        "temporal_resolution_note":
            "DAILY_PRODUCT__PUBLICATION_LATENCY_MUST_BE_LOGGED",
        "vertical_resolution_note":
            "MEDSEA_ANALYSISFORECAST_PHY_006_013__141_LEVELS",
        "equivalence_class": "PROXY_REQUIRES_VALIDATION",
        "morning_status": "LATEST_PUBLISHED_DAY_NOT_GUARANTEED_CURRENT_DAY",
        "requires_local_cache": True,
        "blocker": "CMEMS_PUBLICATION_LATENCY_AND_PRODUCT_SHIFT_VS_HISTORICAL_SERIES",
    }


def main():
    root = Path(__file__).resolve().parent

    whitelist_p = find_dynamic_whitelist(root)
    importance_p = find_importance(root)

    out = (
        root
        / "nw_operational_feature_equivalence_preflight_v1_0"
    )
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 220)
    print("NW HYDROCLIMATE — OPERATIONAL FEATURE EQUIVALENCE PREFLIGHT v1.0")
    print("=" * 220)

    # ------------------------------------------------------------------
    # PHASE 1/4
    # ------------------------------------------------------------------
    print("\nPHASE 1/4 — load canonical 83-feature whitelist")
    start = time.time()

    w = pd.read_csv(
        whitelist_p,
        low_memory=False,
    )

    if len(w) != EXPECTED_DYNAMIC_FEATURES:
        raise SystemExit(
            f"Dynamic whitelist rows={len(w)}, expected={EXPECTED_DYNAMIC_FEATURES}"
        )

    required_cols = {
        "canonical_feature_name",
        "source",
        "feature_column",
        "model_role",
    }

    missing = sorted(required_cols - set(w.columns))
    if missing:
        raise SystemExit(
            "Whitelist missing columns: " + ", ".join(missing)
        )

    importance = None

    if importance_p is not None:
        importance = pd.read_csv(
            importance_p,
            low_memory=False,
        )

    progress(
        "PHASE 1/4",
        1,
        1,
        start,
        f"features={len(w)} | importance={'YES' if importance is not None else 'NO'}",
    )

    # ------------------------------------------------------------------
    # PHASE 2/4
    # ------------------------------------------------------------------
    print("\nPHASE 2/4 — map every dynamic feature to an operational path")
    start = time.time()

    rows = []

    for i, (_, r) in enumerate(
        w.sort_values(
            ["source", "canonical_feature_name"]
        ).iterrows(),
        1,
    ):
        canonical = str(
            r["canonical_feature_name"]
        )
        source = str(
            r["source"]
        )

        if source == "era5":
            mapping = era5_parameter_mapping(
                canonical
            )
        elif source == "medsea_ivt":
            mapping = medsea_mapping(
                canonical
            )
        else:
            mapping = {
                "operational_provider": "UNMAPPED",
                "provider_parameter": "UNMAPPED",
                "transformation": "Manual mapping required.",
                "temporal_resolution_note": "UNKNOWN",
                "vertical_resolution_note": "UNKNOWN",
                "equivalence_class": "UNMAPPED",
                "morning_status": "BLOCKED",
                "requires_local_cache": False,
                "blocker": f"UNSUPPORTED_SOURCE:{source}",
            }

        row = {
            "canonical_feature_name": canonical,
            "training_source": source,
            "training_feature_column":
                str(r["feature_column"]),
            "model_role":
                str(r["model_role"]),
            "training_missingness_semantics":
                str(r.get("missingness_semantics", "")),
            "training_issue_time_semantics":
                str(r.get("issue_time_semantics", "")),
            **mapping,
        }

        rows.append(row)

        progress(
            "PHASE 2/4",
            i,
            len(w),
            start,
            f"{source} | {canonical} | {mapping['equivalence_class']}",
        )

    eq = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # PHASE 3/4
    # ------------------------------------------------------------------
    print("\nPHASE 3/4 — attach model importance and identify operational blockers")
    start = time.time()

    if importance is not None:
        imp = (
            importance.groupby(
                "predictor",
                as_index=False,
            )
            .agg(
                max_validation_ap_drop=(
                    "mean_ap_drop",
                    "max",
                ),
                mean_validation_ap_drop=(
                    "mean_ap_drop",
                    "mean",
                ),
                best_importance_rank=(
                    "importance_rank",
                    "min",
                ),
            )
            .rename(
                columns={
                    "predictor":
                        "canonical_feature_name"
                }
            )
        )

        eq = eq.merge(
            imp,
            on="canonical_feature_name",
            how="left",
            validate="one_to_one",
        )
    else:
        eq["max_validation_ap_drop"] = np.nan
        eq["mean_validation_ap_drop"] = np.nan
        eq["best_importance_rank"] = np.nan

    eq["operational_priority"] = np.select(
        [
            pd.to_numeric(
                eq["best_importance_rank"],
                errors="coerce",
            ).le(10),
            pd.to_numeric(
                eq["best_importance_rank"],
                errors="coerce",
            ).le(25),
        ],
        [
            "P1_TOP10_ANY_HORIZON",
            "P2_TOP25_ANY_HORIZON",
        ],
        default="P3_OTHER_CORE",
    )

    eq["exact_morning_reproduction"] = (
        eq["equivalence_class"].eq(
            "EXACT_LOCAL"
        )
        & eq["morning_status"].eq(
            "READY"
        )
    )

    eq["has_operational_path"] = (
        ~eq["equivalence_class"].eq(
            "UNMAPPED"
        )
        & ~eq["operational_provider"].eq(
            "UNMAPPED"
        )
    )

    blockers = eq[
        eq["blocker"].astype(str).str.len().gt(0)
        | eq["equivalence_class"].eq("UNMAPPED")
    ].copy()

    blockers = blockers.sort_values(
        [
            "operational_priority",
            "max_validation_ap_drop",
        ],
        ascending=[True, False],
    )

    progress(
        "PHASE 3/4",
        1,
        1,
        start,
        (
            f"mapped={int(eq['has_operational_path'].sum())}/{len(eq)} "
            f"| exact_morning={int(eq['exact_morning_reproduction'].sum())} "
            f"| blockers={len(blockers)}"
        ),
    )

    # ------------------------------------------------------------------
    # PHASE 4/4
    # ------------------------------------------------------------------
    print("\nPHASE 4/4 — write provider registry, beta policy and audit")
    start = time.time()

    providers = pd.DataFrame(
        [
            {
                "provider_id":
                    "ECMWF_OPEN_DATA_IFS",
                "purpose":
                    "Operational atmospheric state and short forecast proxy",
                "product":
                    "IFS real-time Open Data subset",
                "spatial_resolution":
                    "0.25 degree",
                "format":
                    "GRIB2",
                "update_cycle":
                    "00/06/12/18 UTC streams; operational availability depends on dissemination",
                "python_access":
                    "ecmwf-opendata / HTTP",
                "authentication":
                    "No user API key normally required for public open-data HTTP access",
                "operational_role":
                    "CANDIDATE_CORE_ATMOSPHERIC_PROXY",
            },
            {
                "provider_id":
                    "COPERNICUS_MARINE_MEDSEA_ANALYSISFORECAST",
                "purpose":
                    "Operational Mediterranean temperature/SST/OHC proxy",
                "product":
                    "MEDSEA_ANALYSISFORECAST_PHY_006_013",
                "dataset":
                    "cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
                "spatial_resolution":
                    "~0.042 degree (~4 km)",
                "format":
                    "NetCDF / lazy xarray via Toolbox",
                "update_cycle":
                    "Daily product; publication latency must be recorded at every run",
                "python_access":
                    "copernicusmarine Python API / Toolbox",
                "authentication":
                    "Copernicus Marine account/token environment configuration may be required",
                "operational_role":
                    "CANDIDATE_CORE_MEDSEA_PROXY",
            },
            {
                "provider_id":
                    "LOCAL_OPERATIONAL_CACHE",
                "purpose":
                    "Store previous completed operational feature snapshots",
                "product":
                    "Local Parquet/CSV cache",
                "spatial_resolution":
                    "receptor-day",
                "format":
                    "Parquet",
                "update_cycle":
                    "One snapshot per daily run",
                "python_access":
                    "local filesystem",
                "authentication":
                    "None",
                "operational_role":
                    "REQUIRED_FOR_1D_3D_7D_14D_ANTECEDENTS",
            },
        ]
    )

    beta_policy = pd.DataFrame(
        [
            {
                "policy_id": "OP1",
                "rule":
                    "The frozen CORE model, calibrators and thresholds are never changed during beta.",
            },
            {
                "policy_id": "OP2",
                "rule":
                    "Every run records source product, model cycle, valid time, retrieval time and file checksum.",
            },
            {
                "policy_id": "OP3",
                "rule":
                    "No missing physical or methodological value is silently replaced by zero.",
            },
            {
                "policy_id": "OP4",
                "rule":
                    "MedSea no-support states preserve the frozen conditional-support semantics.",
            },
            {
                "policy_id": "OP5",
                "rule":
                    "Current-day training features estimated using analysis+forecast are explicitly marked OPERATIONAL_PROXY.",
            },
            {
                "policy_id": "OP6",
                "rule":
                    "The beta bulletin is experimental probabilistic information and is not an official civil-protection alert.",
            },
            {
                "policy_id": "OP7",
                "rule":
                    "If a P1 feature has no valid operational value/path, the run must be DEGRADED or ABORTED; never fabricate a value.",
            },
            {
                "policy_id": "OP8",
                "rule":
                    "Direct future-weather variables that were not predictors during training cannot be added to CORE v1.0.",
            },
            {
                "policy_id": "OP9",
                "rule":
                    "The beta must keep a daily immutable archive so prospective verification is possible.",
            },
            {
                "policy_id": "OP10",
                "rule":
                    "A morning-run proxy is not claimed equivalent to ERA5 until quantitative equivalence tests are completed.",
            },
        ]
    )

    equivalence_p = (
        out
        / "operational_feature_equivalence_v1_0.csv"
    )
    blockers_p = (
        out
        / "operational_feature_blockers_v1_0.csv"
    )
    providers_p = (
        out
        / "operational_provider_registry_v1_0.csv"
    )
    policy_p = (
        out
        / "operational_beta_policy_v1_0.csv"
    )
    priority_p = (
        out
        / "operational_priority_features_v1_0.csv"
    )
    audit_json_p = (
        out
        / "operational_equivalence_audit_v1_0.json"
    )
    audit_txt_p = (
        out
        / "operational_equivalence_audit_v1_0.txt"
    )

    eq.to_csv(
        equivalence_p,
        index=False,
    )
    blockers.to_csv(
        blockers_p,
        index=False,
    )
    providers.to_csv(
        providers_p,
        index=False,
    )
    beta_policy.to_csv(
        policy_p,
        index=False,
    )

    priority = eq[
        eq["operational_priority"].isin(
            [
                "P1_TOP10_ANY_HORIZON",
                "P2_TOP25_ANY_HORIZON",
            ]
        )
    ].sort_values(
        [
            "operational_priority",
            "max_validation_ap_drop",
        ],
        ascending=[True, False],
    )

    priority.to_csv(
        priority_p,
        index=False,
    )

    unmapped = int(
        (~eq["has_operational_path"]).sum()
    )

    exact_morning = int(
        eq["exact_morning_reproduction"].sum()
    )

    proxy_count = int(
        eq["equivalence_class"]
        .astype(str)
        .str.contains(
            "PROXY",
            regex=False,
        )
        .sum()
    )

    p1 = eq[
        eq["operational_priority"].eq(
            "P1_TOP10_ANY_HORIZON"
        )
    ]

    p1_with_blocker = int(
        p1["blocker"].astype(str).str.len().gt(0).sum()
    )

    if unmapped:
        overall = "FAIL_UNMAPPED_DYNAMIC_FEATURES"
    else:
        overall = (
            "PASS_REGISTRY_BUILT__BETA_PROXY_VALIDATION_REQUIRED"
        )

    audit = {
        "version": "1.0",
        "overall_status": overall,
        "dynamic_features": int(len(eq)),
        "operational_paths_mapped":
            int(eq["has_operational_path"].sum()),
        "unmapped_features": unmapped,
        "exact_morning_reproduction_features":
            exact_morning,
        "proxy_equivalence_features":
            proxy_count,
        "features_with_documented_blocker":
            int(len(blockers)),
        "p1_top_features":
            int(len(p1)),
        "p1_features_with_blocker":
            p1_with_blocker,
        "exact_operational_core_ready":
            False,
        "experimental_beta_possible":
            unmapped == 0,
        "model_retraining_required_for_first_proxy_beta":
            False,
        "critical_methodological_fact":
            (
                "Historical CORE issue-time was end-of-day t; a morning runner "
                "requires explicit analysis+forecast proxy construction for "
                "same-day features or a changed issue-time semantics."
            ),
        "next_step":
            (
                "Probe real provider access and retrieve a minimal current sample "
                "from ECMWF Open Data and Copernicus Marine. Then compare raw "
                "provider variables against the operational equivalence registry."
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

    class_counts = (
        eq["equivalence_class"]
        .value_counts()
        .rename_axis("equivalence_class")
        .reset_index(name="feature_count")
    )

    morning_counts = (
        eq["morning_status"]
        .value_counts()
        .rename_axis("morning_status")
        .reset_index(name="feature_count")
    )

    priority_display = priority[
        [
            "canonical_feature_name",
            "training_source",
            "operational_priority",
            "operational_provider",
            "provider_parameter",
            "equivalence_class",
            "morning_status",
            "blocker",
            "max_validation_ap_drop",
        ]
    ].head(40)

    lines = [
        "=" * 220,
        "NW HYDROCLIMATE — OPERATIONAL FEATURE EQUIVALENCE PREFLIGHT v1.0",
        "=" * 220,
        f"OVERALL STATUS                    : {overall}",
        f"Dynamic features                  : {len(eq)}",
        f"Operational paths mapped          : {int(eq['has_operational_path'].sum())}",
        f"Unmapped features                 : {unmapped}",
        f"Exact morning reproduction        : {exact_morning}",
        f"Proxy-equivalence features        : {proxy_count}",
        f"Features with documented blocker  : {len(blockers)}",
        f"P1 features with blocker          : {p1_with_blocker}/{len(p1)}",
        "Exact operational CORE ready      : False",
        f"Experimental beta possible        : {unmapped == 0}",
        "",
        "EQUIVALENCE CLASS COUNTS",
        class_counts.to_string(index=False),
        "",
        "MORNING STATUS COUNTS",
        morning_counts.to_string(index=False),
        "",
        "HIGH-PRIORITY OPERATIONAL FEATURES",
        priority_display.to_string(index=False),
        "",
        "IMPORTANT",
        "This registry does not claim that IFS is numerically identical to ERA5.",
        "Morning same-day features were end-of-day features in training and therefore require an explicit proxy policy.",
        "IVT requires quantitative equivalence validation because the open pressure-level set differs from the historical ERA5 integration.",
        "ERA5 1-hour precipitation maximum is not exactly reproducible from the open IFS temporal cadence.",
        "CMEMS data vintage/latency must be recorded at every run.",
        "No NaN is filled with zero.",
        "",
        "NEXT STEP",
        "Run provider-access probes for ECMWF Open Data and Copernicus Marine using current data.",
        "Do not build the final autonomous bulletin runner until those probes pass.",
        "",
        f"Equivalence : {equivalence_p}",
        f"Blockers    : {blockers_p}",
        f"Providers   : {providers_p}",
        f"Policy      : {policy_p}",
        f"Priority    : {priority_p}",
        f"Output      : {out}",
    ]

    audit_txt_p.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    progress(
        "PHASE 4/4",
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
