#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
freeze_nw_dynamic_causal_feature_whitelist_v1_3.py

CONGELA LA WHITELIST DINAMICA CAUSALE DOPO IL PREFLIGHT v1.2.

ATTENZIONE
----------
Questa whitelist riguarda SOLO le feature dinamiche.
NON è ancora il master dataset finale: prima del master definitivo dovranno
essere aggiunti/auditati i descrittori statici dei recettori (area, morfometria,
quota/pendenza da DEM, ecc.).

DECISIONI CANONICHE
-------------------
PRIMARY DYNAMIC CORE
- ERA5 / MedSea-IVT fisiche classificate:
    CORE_CAUSAL
    CORE_CAUSAL_STRUCTURAL_LAG_GAP
    CORE_CAUSAL_CONDITIONAL_SUPPORT
- support-state MedSea mantenuti quando servono a interpretare i valori
  condizionali;
- `season_day` ERA5 come unico indice stagionale grezzo ammesso;
- esclusi:
    era5.label
    season_year
    angoli grezzi già rappresentati/diagnostici
    duplicati cross-source
    QA/network metadata
    `marine_sector_interp_fraction` (metadato numerico di interpolazione)
    `medsea_support_angle_narrow` perché è alias esatto di
      `medsea_support_robust_core` nel prodotto canonico v1.2.

MEDSEA SUPPORT
--------------
Il prodotto canonico MedSea v1.2 ha congelato ±45° come baseline e conserva
la robustezza angolare ±30°/±45°/±60°.
Per il modello primario manteniamo:
- i 5 corridor support weights;
- medsea_support_robust_core   (equivale al supporto narrow ±30°);
- medsea_support_baseline      (±45°);
- medsea_support_angle_wide    (±60°).

Non si zero-imputano i valori SST/OHC quando il supporto manca.

OBSERVED PRECIPITATION
----------------------
Le 7 statistiche osservate di precipitazione con copertura pooled elevata NON
entrano nel PRIMARY CORE, perché il preflight v1.2 mostra che solo l'85% delle
coppie receptor×fold raggiunge >=80% di disponibilità e il minimo scende a
~42%.

Vengono congelate in una whitelist separata:
    OPTIONAL_OBS_PRECIP_AUGMENTATION
e viene prodotto un registro receptor×fold di eleggibilità:
    eligible = tutte le 7 feature >=80% nel FIT del fold.

Nessuna coppia non eleggibile deve essere trattata come "precipitazione zero".

CALENDARIO
----------
- `season_day` ERA5: incluso.
- `month`, `day_of_year`: esclusi dalla whitelist primaria perché ridondanti
  con season_day.
- `season_year`: escluso per evitare un indice grezzo di trend temporale.

INPUT
-----
nw_causal_feature_whitelist_preflight_v1_2/
  causal_feature_inventory_v1_2.csv
  causal_feature_availability_by_fold_receptor_v1_2.csv
  primary_core_feature_candidates_v1_2.csv
  observed_meteo_high_coverage_candidates_v1_2.csv
  cross_source_exact_duplicate_audit_v1_2.csv
  excluded_leakage_registry_v1_2.csv
  source_integrity_audit_v1_2.csv

OUTPUT
------
nw_dynamic_causal_feature_whitelist_canonical_v1_3/
  primary_dynamic_feature_whitelist_canonical_v1_3.csv
  optional_observed_precip_whitelist_canonical_v1_3.csv
  optional_observed_precip_receptor_fold_eligibility_v1_3.csv
  excluded_dynamic_feature_registry_canonical_v1_3.csv
  dynamic_feature_policy_canonical_v1_3.csv
  checksums_sha256_canonical_v1_3.csv
  dynamic_feature_whitelist_audit_v1_3.json
  dynamic_feature_whitelist_audit_v1_3.txt
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


OBS_ELIGIBILITY_MIN = 0.80
EXPECTED_RECEPTOR_FOLD_PAIRS = 20 * 3

PRIMARY_CLASSES = {
    "CORE_CAUSAL",
    "CORE_CAUSAL_STRUCTURAL_LAG_GAP",
    "CORE_CAUSAL_CONDITIONAL_SUPPORT",
}

KEEP_SUPPORT_COMPANIONS = {
    "sst_corridor_support_weight",
    "ohc_corridor_support_weight",
    "ohc_abs_corridor_support_weight",
    "tmean_corridor_support_weight",
    "tmean_anom_corridor_support_weight",
    "medsea_support_robust_core",
    "medsea_support_baseline",
    "medsea_support_angle_wide",
}

EXPLICIT_PRIMARY_CALENDAR = {
    ("era5", "season_day"),
}

EXPLICIT_PRIMARY_EXCLUSIONS = {
    ("medsea_ivt", "marine_sector_interp_fraction"):
        "NUMERICAL_SECTOR_INTERPOLATION_METADATA",
    ("medsea_ivt", "medsea_support_angle_narrow"):
        "ALIAS_OF_MEDSEA_SUPPORT_ROBUST_CORE",
    ("era5", "month"):
        "REDUNDANT_WITH_SEASON_DAY",
    ("era5", "day_of_year"):
        "REDUNDANT_WITH_SEASON_DAY",
}

EXPECTED_OBS_FEATURES = {
    "obs_precip_mm_mean",
    "obs_precip_mm_median",
    "obs_precip_mm_min",
    "obs_precip_mm_max",
    "obs_precip_mm_p90",
    "obs_precip_mm_std",
    "obs_precip_mm_coverage_weighted_mean",
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
        msg += f" | {str(current)[:120]}"

    print(msg.ljust(255), end="", flush=True)
    if done >= total:
        print(flush=True)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parent

    src = root / "nw_causal_feature_whitelist_preflight_v1_2"

    inventory_p = src / "causal_feature_inventory_v1_2.csv"
    availability_p = (
        src / "causal_feature_availability_by_fold_receptor_v1_2.csv"
    )
    core_p = src / "primary_core_feature_candidates_v1_2.csv"
    obs_high_p = src / "observed_meteo_high_coverage_candidates_v1_2.csv"
    duplicates_p = src / "cross_source_exact_duplicate_audit_v1_2.csv"
    leakage_p = src / "excluded_leakage_registry_v1_2.csv"
    integrity_p = src / "source_integrity_audit_v1_2.csv"

    required = [
        inventory_p,
        availability_p,
        core_p,
        obs_high_p,
        duplicates_p,
        leakage_p,
        integrity_p,
    ]

    out = root / "nw_dynamic_causal_feature_whitelist_canonical_v1_3"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 212)
    print("NW HYDROCLIMATE — FREEZE DYNAMIC CAUSAL FEATURE WHITELIST v1.3")
    print("=" * 212)

    for p in required:
        if not p.exists():
            raise SystemExit(f"Manca: {p}")

    # ------------------------------------------------------------------
    # PHASE 1/4 — validate preflight
    # ------------------------------------------------------------------
    print("\nPHASE 1/4 — validate v1.2 preflight before freeze")
    start1 = time.time()

    inv = pd.read_csv(inventory_p, low_memory=False)
    avail = pd.read_csv(availability_p, low_memory=False)
    obs_high = pd.read_csv(obs_high_p, low_memory=False)
    leakage = pd.read_csv(leakage_p, low_memory=False)
    integrity = pd.read_csv(integrity_p, low_memory=False)

    errors = []
    warnings = []

    if (
        (integrity["duplicates"] > 0).any()
        or (integrity["bad_dates"] > 0).any()
        or (~integrity["reference_shape_ok"].astype(bool)).any()
    ):
        errors.append("SOURCE_INTEGRITY_NOT_PASS")

    era5_label = leakage[
        leakage["source"].astype(str).eq("era5")
        & leakage["feature_column"]
        .astype(str).str.strip().str.lower().eq("label")
    ]

    if len(era5_label) != 1:
        errors.append(
            f"ERA5_LABEL_EXCLUSION_ROWS={len(era5_label)}"
        )

    unexplained = inv[
        inv["feature_class"].astype(str).eq(
            "REVIEW_UNEXPLAINED_CORE_MISSINGNESS"
        )
    ]
    if len(unexplained):
        errors.append(
            f"UNEXPLAINED_CORE_MISSINGNESS={len(unexplained)}"
        )

    exact_dup_lower = inv[
        inv["exact_duplicate_lower_priority"].astype(str)
        .str.lower()
        .isin({"true", "1"})
    ]
    if len(exact_dup_lower) != 4:
        warnings.append(
            f"Expected 4 exact lower-priority duplicates, found {len(exact_dup_lower)}"
        )

    obs_names = set(
        obs_high["feature_column"].astype(str)
    )
    if obs_names != EXPECTED_OBS_FEATURES:
        errors.append(
            "OBS_HIGH_FEATURE_SET_DIFFERS_FROM_EXPECTED: "
            f"{sorted(obs_names)}"
        )

    progress(
        "PHASE 1/4",
        1,
        1,
        start1,
        f"errors={len(errors)} warnings={len(warnings)}",
    )

    if errors:
        print("\nFREEZE ABORTED")
        for e in errors:
            print(f" - {e}")
        raise SystemExit(2)

    # ------------------------------------------------------------------
    # PHASE 2/4 — construct primary dynamic whitelist
    # ------------------------------------------------------------------
    print("\nPHASE 2/4 — construct primary dynamic whitelist")
    start2 = time.time()

    primary_rows = []
    exclusion_rows = []

    for _, r in inv.iterrows():
        source = str(r["source"])
        col = str(r["feature_column"])
        cls = str(r["feature_class"])
        key = (source, col)

        include = False
        role = ""
        reason = ""

        if key in EXPLICIT_PRIMARY_EXCLUSIONS:
            include = False
            reason = EXPLICIT_PRIMARY_EXCLUSIONS[key]

        elif key in EXPLICIT_PRIMARY_CALENDAR:
            include = True
            role = "PRIMARY_CALENDAR_KNOWN"
            reason = "CANONICAL_SEASON_PROGRESS_INDEX"

        elif cls in PRIMARY_CLASSES:
            include = True
            role = "PRIMARY_DYNAMIC_CORE"
            reason = cls

        elif (
            cls == "SUPPORT_STATE_COMPANION"
            and col in KEEP_SUPPORT_COMPANIONS
        ):
            include = True
            role = "PRIMARY_SUPPORT_COMPANION"
            reason = "REQUIRED_TO_INTERPRET_CONDITIONAL_MEDSEA_FEATURES"

        elif cls == "SUPPORT_STATE_COMPANION":
            include = False
            reason = "SUPPORT_COMPANION_NOT_SELECTED_FOR_PRIMARY_CORE"

        else:
            include = False
            reason = f"FEATURE_CLASS_{cls}"

        if include:
            primary_rows.append(
                {
                    "feature_order": len(primary_rows) + 1,
                    "source": source,
                    "feature_column": col,
                    "canonical_feature_name":
                        f"{source}__{col}",
                    "model_role": role,
                    "source_feature_class": cls,
                    "missingness_semantics":
                        (
                            "STRUCTURAL_SEASON_EDGE_NAN"
                            if cls == "CORE_CAUSAL_STRUCTURAL_LAG_GAP"
                            else
                            "CONDITIONAL_MEDSEA_SUPPORT__NO_ZERO_IMPUTATION"
                            if cls == "CORE_CAUSAL_CONDITIONAL_SUPPORT"
                            else
                            "SUPPORT_STATE"
                            if role == "PRIMARY_SUPPORT_COMPANION"
                            else
                            "STANDARD"
                        ),
                    "issue_time_semantics":
                        r.get("issue_time_class", ""),
                    "selection_reason": reason,
                }
            )
        else:
            exclusion_rows.append(
                {
                    "source": source,
                    "feature_column": col,
                    "source_feature_class": cls,
                    "exclusion_reason": reason,
                }
            )

    primary = pd.DataFrame(primary_rows)

    # Defensive audits.
    if primary["canonical_feature_name"].duplicated().any():
        errors.append("DUPLICATE_PRIMARY_CANONICAL_FEATURE_NAMES")

    if (
        primary["feature_column"].astype(str).str.lower().eq("label")
    ).any():
        errors.append("LABEL_PRESENT_IN_PRIMARY_WHITELIST")

    if (
        primary["feature_column"].astype(str).eq("season_year")
    ).any():
        errors.append("SEASON_YEAR_PRESENT_IN_PRIMARY_WHITELIST")

    required_support = KEEP_SUPPORT_COMPANIONS
    kept_support = set(
        primary.loc[
            primary["model_role"].eq("PRIMARY_SUPPORT_COMPANION"),
            "feature_column",
        ].astype(str)
    )

    if kept_support != required_support:
        errors.append(
            "MEDSEA_SUPPORT_COMPANION_SET_MISMATCH: "
            f"kept={sorted(kept_support)}"
        )

    if errors:
        print("\nFREEZE ABORTED")
        for e in errors:
            print(f" - {e}")
        raise SystemExit(2)

    progress(
        "PHASE 2/4",
        1,
        1,
        start2,
        f"primary features={len(primary)}",
    )

    # ------------------------------------------------------------------
    # PHASE 3/4 — optional observed precipitation + eligibility matrix
    # ------------------------------------------------------------------
    print("\nPHASE 3/4 — freeze optional observed-precipitation augmentation")
    start3 = time.time()

    optional_obs = obs_high[
        obs_high["source"].astype(str).eq("obs_meteo")
    ].copy()

    optional_obs = optional_obs[
        optional_obs["feature_column"].astype(str).isin(EXPECTED_OBS_FEATURES)
    ].copy()

    optional_obs = optional_obs.sort_values("feature_column").reset_index(drop=True)

    optional_obs_out_rows = []

    for i, r in optional_obs.iterrows():
        optional_obs_out_rows.append(
            {
                "feature_order": i + 1,
                "source": "obs_meteo",
                "feature_column": str(r["feature_column"]),
                "canonical_feature_name":
                    f"obs_meteo__{r['feature_column']}",
                "model_role": "OPTIONAL_OBS_PRECIP_AUGMENTATION",
                "issue_time_semantics":
                    "END_OF_DAY_T_OBSERVED_ONLY",
                "network_semantics":
                    "STATION_NETWORK_SUMMARY_NOT_AREA_INTERPOLATION",
                "missingness_policy":
                    "NO_ZERO_IMPUTATION__USE_ONLY_IN_ELIGIBLE_RECEPTOR_FOLD_PAIRS",
            }
        )

    optional_obs_out = pd.DataFrame(optional_obs_out_rows)

    a = avail[
        avail["source"].astype(str).eq("obs_meteo")
        & avail["feature_column"].astype(str).isin(EXPECTED_OBS_FEATURES)
    ].copy()

    if not len(a):
        raise SystemExit("Observed precipitation availability rows missing.")

    pair = (
        a.groupby(
            ["fold_id", "receptor_id"],
            as_index=False,
        )
        .agg(
            feature_count=("feature_column", "nunique"),
            minimum_feature_availability=(
                "availability_fraction",
                "min",
            ),
            mean_feature_availability=(
                "availability_fraction",
                "mean",
            ),
        )
    )

    pair["all_7_features_present"] = (
        pair["feature_count"].eq(len(EXPECTED_OBS_FEATURES))
    )

    pair["obs_precip_augmentation_eligible"] = (
        pair["all_7_features_present"]
        & pair["minimum_feature_availability"].ge(
            OBS_ELIGIBILITY_MIN
        )
    )

    pair["eligibility_reason"] = np.where(
        pair["obs_precip_augmentation_eligible"],
        "ALL_7_OBS_PRECIP_FEATURES_GE_80PCT_IN_FIT",
        "NOT_ALL_7_OBS_PRECIP_FEATURES_GE_80PCT_IN_FIT",
    )

    if len(pair) != EXPECTED_RECEPTOR_FOLD_PAIRS:
        raise SystemExit(
            f"Receptor-fold pairs={len(pair)}, "
            f"expected={EXPECTED_RECEPTOR_FOLD_PAIRS}"
        )

    eligible_pairs = int(
        pair["obs_precip_augmentation_eligible"].sum()
    )

    progress(
        "PHASE 3/4",
        1,
        1,
        start3,
        f"optional obs features={len(optional_obs_out)} eligible pairs={eligible_pairs}/60",
    )

    # ------------------------------------------------------------------
    # PHASE 4/4 — write canonical artifacts
    # ------------------------------------------------------------------
    print("\nPHASE 4/4 — write canonical whitelist, policy, checksums")
    start4 = time.time()

    primary_p = (
        out / "primary_dynamic_feature_whitelist_canonical_v1_3.csv"
    )
    obs_p = (
        out / "optional_observed_precip_whitelist_canonical_v1_3.csv"
    )
    elig_p = (
        out / "optional_observed_precip_receptor_fold_eligibility_v1_3.csv"
    )
    exclusion_p = (
        out / "excluded_dynamic_feature_registry_canonical_v1_3.csv"
    )
    policy_p = (
        out / "dynamic_feature_policy_canonical_v1_3.csv"
    )
    audit_json = (
        out / "dynamic_feature_whitelist_audit_v1_3.json"
    )
    audit_txt = (
        out / "dynamic_feature_whitelist_audit_v1_3.txt"
    )

    primary.to_csv(primary_p, index=False)
    optional_obs_out.to_csv(obs_p, index=False)
    pair.to_csv(elig_p, index=False)
    pd.DataFrame(exclusion_rows).to_csv(exclusion_p, index=False)

    policy = pd.DataFrame(
        [
            {
                "policy_id": "F1",
                "rule":
                    "Primary model uses only the frozen dynamic CORE whitelist "
                    "plus canonical static receptor descriptors to be audited separately.",
            },
            {
                "policy_id": "F2",
                "rule":
                    "ERA5 label/target/future/threshold fields are excluded from predictors.",
            },
            {
                "policy_id": "F3",
                "rule":
                    "season_day is the only raw seasonal progress index kept; "
                    "season_year is excluded.",
            },
            {
                "policy_id": "F4",
                "rule":
                    "ERA5 7/14-day lag NaNs at September season edges are structural "
                    "and are not interpreted as zero.",
            },
            {
                "policy_id": "F5",
                "rule":
                    "MedSea conditional SST/OHC/proxy NaNs mean no supported marine "
                    "source under the coupling method and are never zero-imputed.",
            },
            {
                "policy_id": "F6",
                "rule":
                    "MedSea baseline ±45° remains canonical; support robustness "
                    "is represented by robust-core/baseline/wide support flags.",
            },
            {
                "policy_id": "F7",
                "rule":
                    "Observed precipitation is not part of the primary regional CORE "
                    "because receptor-fold coverage is non-universal.",
            },
            {
                "policy_id": "F8",
                "rule":
                    "Observed-precip augmentation may be evaluated only on receptor-fold "
                    "pairs passing the frozen >=80% FIT availability criterion.",
            },
            {
                "policy_id": "F9",
                "rule":
                    "Observed precipitation remains a station-network summary, not an "
                    "area-interpolated basin rainfall field.",
            },
            {
                "policy_id": "F10",
                "rule":
                    "Same-day dynamic features assume an end-of-day-t issue convention; "
                    "intraday deployment requires an availability-specific variant.",
            },
            {
                "policy_id": "F11",
                "rule":
                    "No Validation/Test target or model skill is used to choose this whitelist.",
            },
            {
                "policy_id": "F12",
                "rule":
                    "No feature imputation is performed at whitelist-freeze stage.",
            },
        ]
    )

    policy.to_csv(policy_p, index=False)

    files_for_hash = [
        primary_p,
        obs_p,
        elig_p,
        exclusion_p,
        policy_p,
    ]

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
        out / "checksums_sha256_canonical_v1_3.csv"
    )
    checksums.to_csv(checksums_p, index=False)

    role_counts = (
        primary["model_role"]
        .value_counts()
        .rename_axis("model_role")
        .reset_index(name="feature_count")
    )

    audit = {
        "version": "1.3",
        "overall_status":
            "PASS_WITH_OPTIONAL_OBS_PRECIP_PARTIAL_NETWORK",
        "primary_dynamic_features": int(len(primary)),
        "primary_role_counts": {
            str(r["model_role"]): int(r["feature_count"])
            for _, r in role_counts.iterrows()
        },
        "optional_observed_precip_features":
            int(len(optional_obs_out)),
        "observed_precip_receptor_fold_pairs":
            int(len(pair)),
        "observed_precip_eligible_receptor_fold_pairs":
            eligible_pairs,
        "observed_precip_ineligible_receptor_fold_pairs":
            int(len(pair) - eligible_pairs),
        "observed_precip_pair_eligibility_fraction":
            float(eligible_pairs / len(pair)),
        "era5_label_in_primary": False,
        "season_year_in_primary": False,
        "validation_test_used_for_selection": False,
        "target_skill_used_for_selection": False,
        "feature_imputation_performed": False,
        "medsea_no_support_zero_imputation": False,
        "dynamic_whitelist_state":
            "CLOSED_CANONICAL_V1_3",
        "master_dataset_state":
            "NOT_YET_FINAL__STATIC_RECEPTOR_DESCRIPTORS_STILL_REQUIRED",
        "next_step":
            "Build/audit static receptor descriptors from basin geometries and "
            "Copernicus DEM before constructing the definitive foldwise master matrix.",
        "warnings": warnings,
    }

    audit_json.write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "=" * 212,
        "NW HYDROCLIMATE — CANONICAL DYNAMIC CAUSAL FEATURE WHITELIST v1.3",
        "=" * 212,
        "OVERALL STATUS                         : PASS_WITH_OPTIONAL_OBS_PRECIP_PARTIAL_NETWORK",
        f"Primary dynamic features                : {len(primary)}",
        f"Optional observed-precip features       : {len(optional_obs_out)}",
        f"Observed precip eligible pairs          : {eligible_pairs}/{len(pair)}",
        f"Observed precip ineligible pairs        : {len(pair)-eligible_pairs}/{len(pair)}",
        "ERA5 label in primary                  : False",
        "season_year in primary                 : False",
        "Validation/Test used for selection     : False",
        "Feature imputation                     : False",
        "",
        "PRIMARY ROLE COUNTS",
        role_counts.to_string(index=False),
        "",
        "OPTIONAL OBS-PRECIP ELIGIBILITY",
        pair.to_string(index=False),
        "",
        "IMPORTANT",
        "Dynamic whitelist is now frozen, but the master dataset is NOT final.",
        "Static receptor descriptors still need a separate terrain/morphometry preflight.",
        "Observed precipitation is optional augmentation only; ineligible receptor-fold pairs are never zero-filled.",
        "",
        f"Primary whitelist : {primary_p}",
        f"Optional obs      : {obs_p}",
        f"Eligibility       : {elig_p}",
        f"Policy            : {policy_p}",
        f"Output            : {out}",
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
        "dynamic whitelist frozen",
    )

    print("\n" + "=" * 212)
    print("\n".join(lines[3:]))
    print("=" * 212)
    print("OVERALL STATUS : PASS_WITH_OPTIONAL_OBS_PRECIP_PARTIAL_NETWORK")
    print(f"Output         : {out}")
    print("=" * 212)


if __name__ == "__main__":
    main()
