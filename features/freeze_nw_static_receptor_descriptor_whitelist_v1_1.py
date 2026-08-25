#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
freeze_nw_static_receptor_descriptor_whitelist_v1_1.py

CONGELA LA WHITELIST STATICA DEI 21 RECETTORI.

INPUT
-----
nw_static_receptor_descriptors_preflight_v1_0/
  static_receptor_descriptors_v1_0.csv
  static_receptor_geometry_audit_v1_0.csv
  static_receptor_dem_tile_usage_v1_0.csv

nw_foldwise_q95_modeling_labels_canonical_v1_3/
  foldwise_q95_modeling_labels_canonical_v1_3.csv

OBIETTIVO
---------
Il preflight statico v1.0 è PASS:
- 21 recettori;
- 0 duplicati;
- 0 missing nei descrittori;
- 0 recettori con copertura DEM < 0.98.

Questa fase congela un insieme PARSIMONIOSO di descrittori statici fisicamente
interpretabili, evitando ridondanze deterministiche o quasi-deterministiche.

WHITELIST PRIMARIA
------------------
Geometria:
- area_km2
- circularity_4piA_P2
- convexity_area_ratio

Orografia:
- elevation_m_mean
- elevation_m_p90
- relief_m
- hypsometric_integral_proxy
- elevation_fraction_lt500m
- elevation_fraction_500_1000m
- elevation_fraction_1000_1500m

Pendenza:
- slope_deg_mean
- slope_deg_p90
- slope_fraction_ge15deg
- slope_fraction_ge30deg

ESCLUSIONI VOLUTE
-----------------
- perimeter_km: ridondante con area + circularity;
- equivalent_diameter_km: deterministico da area;
- centroid_lon/lat: metadata geografici, non predittori fisici primari;
- elevation min/max/std/median/p10: esclusi dal CORE parsimonioso;
- elevation_fraction_ge1500m: complementare alle altre tre frazioni;
- slope min/max/std/median/p10: esclusi dal CORE parsimonioso;
- dem_* / semantics strings: metadata/QC.

NOTA METODOLOGICA
-----------------
I 21 poligoni sono MODEL RECEPTORS, non sono automaticamente bacini di chiusura
idrometrica perfetti. I descrittori statici servono a caratterizzare il contesto
spaziale nel modello pooled/hierarchical.

Il DEM Copernicus GLO-30 è un DSM regionale e NON è adatto a progetto
idraulico di dettaglio.

LIG_ENTELLA viene mantenuto nel registro statico dei 21 recettori ma, finché
non esiste un target idrologico primario canonico, non entra nelle righe
supervisionate del modello.

OUTPUT
------
nw_static_receptor_descriptor_whitelist_canonical_v1_1/
  static_receptor_descriptor_whitelist_canonical_v1_1.csv
  static_receptor_descriptor_values_canonical_v1_1.csv
  excluded_static_descriptor_registry_v1_1.csv
  receptor_static_model_scope_registry_v1_1.csv
  static_descriptor_policy_canonical_v1_1.csv
  checksums_sha256_canonical_v1_1.csv
  static_descriptor_whitelist_audit_v1_1.json
  static_descriptor_whitelist_audit_v1_1.txt
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_RECEPTORS = 21
MIN_DEM_VALID_FRACTION = 0.98

PRIMARY_STATIC_FEATURES = [
    ("area_km2", "GEOMETRY_SCALE"),
    ("circularity_4piA_P2", "GEOMETRY_SHAPE"),
    ("convexity_area_ratio", "GEOMETRY_SHAPE"),
    ("elevation_m_mean", "OROGRAPHY"),
    ("elevation_m_p90", "OROGRAPHY"),
    ("relief_m", "OROGRAPHY_RELIEF"),
    ("hypsometric_integral_proxy", "OROGRAPHY_HYPSOMETRY"),
    ("elevation_fraction_lt500m", "ELEVATION_BAND"),
    ("elevation_fraction_500_1000m", "ELEVATION_BAND"),
    ("elevation_fraction_1000_1500m", "ELEVATION_BAND"),
    ("slope_deg_mean", "SLOPE"),
    ("slope_deg_p90", "SLOPE"),
    ("slope_fraction_ge15deg", "SLOPE_THRESHOLD_FRACTION"),
    ("slope_fraction_ge30deg", "SLOPE_THRESHOLD_FRACTION"),
]

EXCLUSION_REASONS = {
    "perimeter_km":
        "REDUNDANT_WITH_AREA_AND_CIRCULARITY",
    "centroid_lon":
        "GEOGRAPHIC_LOCATION_METADATA_NOT_PRIMARY_PHYSICAL_PREDICTOR",
    "centroid_lat":
        "GEOGRAPHIC_LOCATION_METADATA_NOT_PRIMARY_PHYSICAL_PREDICTOR",
    "equivalent_diameter_km":
        "DETERMINISTIC_FUNCTION_OF_AREA",
    "elevation_m_min":
        "PARSIMONIOUS_CORE_EXCLUDES_EXTRA_DISTRIBUTION_STATISTIC",
    "elevation_m_p10":
        "PARSIMONIOUS_CORE_EXCLUDES_EXTRA_DISTRIBUTION_STATISTIC",
    "elevation_m_median":
        "PARSIMONIOUS_CORE_EXCLUDES_EXTRA_DISTRIBUTION_STATISTIC",
    "elevation_m_max":
        "PARSIMONIOUS_CORE_USES_RELIEF_AND_P90",
    "elevation_m_std":
        "PARSIMONIOUS_CORE_EXCLUDES_EXTRA_DISTRIBUTION_STATISTIC",
    "elevation_fraction_ge1500m":
        "EXACT_COMPLEMENT_OF_OTHER_ELEVATION_BAND_FRACTIONS",
    "slope_deg_min":
        "PARSIMONIOUS_CORE_EXCLUDES_EXTRA_DISTRIBUTION_STATISTIC",
    "slope_deg_p10":
        "PARSIMONIOUS_CORE_EXCLUDES_EXTRA_DISTRIBUTION_STATISTIC",
    "slope_deg_median":
        "PARSIMONIOUS_CORE_EXCLUDES_EXTRA_DISTRIBUTION_STATISTIC",
    "slope_deg_max":
        "PARSIMONIOUS_CORE_EXCLUDES_EXTRA_DISTRIBUTION_STATISTIC",
    "slope_deg_std":
        "PARSIMONIOUS_CORE_EXCLUDES_EXTRA_DISTRIBUTION_STATISTIC",
    "dem_tiles_used":
        "DEM_PROCESSING_METADATA",
    "dem_target_resolution_m":
        "DEM_PROCESSING_METADATA",
    "dem_valid_pixels":
        "DEM_QC_METADATA",
    "dem_polygon_pixels":
        "DEM_QC_METADATA",
    "dem_pixel_valid_fraction":
        "DEM_QC_METADATA",
    "dem_target_crs":
        "DEM_PROCESSING_METADATA",
    "static_descriptor_semantics":
        "SEMANTIC_METADATA",
    "dem_source":
        "SOURCE_METADATA",
    "dem_use_scope":
        "SEMANTIC_METADATA",
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parent

    src = root / "nw_static_receptor_descriptors_preflight_v1_0"
    descriptors_p = src / "static_receptor_descriptors_v1_0.csv"
    geometry_p = src / "static_receptor_geometry_audit_v1_0.csv"
    tile_usage_p = src / "static_receptor_dem_tile_usage_v1_0.csv"

    labels_p = (
        root
        / "nw_foldwise_q95_modeling_labels_canonical_v1_3"
        / "foldwise_q95_modeling_labels_canonical_v1_3.csv"
    )

    out = root / "nw_static_receptor_descriptor_whitelist_canonical_v1_1"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 210)
    print("NW HYDROCLIMATE — FREEZE STATIC RECEPTOR DESCRIPTOR WHITELIST v1.1")
    print("=" * 210)

    for p in (descriptors_p, geometry_p, tile_usage_p, labels_p):
        if not p.exists():
            raise SystemExit(f"Manca: {p}")

    # ------------------------------------------------------------------
    # PHASE 1/3 — validate static preflight
    # ------------------------------------------------------------------
    print("\nPHASE 1/3 — validate static descriptor preflight")
    start1 = time.time()

    d = pd.read_csv(descriptors_p, low_memory=False)
    g = pd.read_csv(geometry_p, low_memory=False)
    labels = pd.read_csv(labels_p, low_memory=False)

    errors = []
    warnings = []

    if len(d) != EXPECTED_RECEPTORS:
        errors.append(f"DESCRIPTOR_ROWS={len(d)}")

    if d["receptor_id"].duplicated().any():
        errors.append("DUPLICATE_RECEPTOR_ID")

    if d["receptor_id"].isna().any():
        errors.append("MISSING_RECEPTOR_ID")

    if len(g) != EXPECTED_RECEPTORS:
        errors.append(f"GEOMETRY_AUDIT_ROWS={len(g)}")

    if "dem_pixel_valid_fraction" not in d.columns:
        errors.append("MISSING_DEM_VALID_FRACTION")
    else:
        low = d[
            pd.to_numeric(
                d["dem_pixel_valid_fraction"],
                errors="coerce",
            ) < MIN_DEM_VALID_FRACTION
        ]
        if len(low):
            errors.append(
                "LOW_DEM_VALID_FRACTION_RECEPTORS="
                + ",".join(low["receptor_id"].astype(str))
            )

    required_features = [x[0] for x in PRIMARY_STATIC_FEATURES]
    missing_features = sorted(set(required_features) - set(d.columns))
    if missing_features:
        errors.append(
            "MISSING_PRIMARY_STATIC_FEATURES="
            + ",".join(missing_features)
        )

    for col in required_features:
        x = pd.to_numeric(d[col], errors="coerce")
        if x.isna().any():
            errors.append(
                f"PRIMARY_FEATURE_MISSING_VALUES:{col}={int(x.isna().sum())}"
            )

    if errors:
        progress(
            "PHASE 1/3",
            1,
            1,
            start1,
            f"errors={len(errors)}",
        )
        print("\nFREEZE ABORTED")
        for e in errors:
            print(f" - {e}")
        raise SystemExit(2)

    progress(
        "PHASE 1/3",
        1,
        1,
        start1,
        "static preflight integrity PASS",
    )

    # ------------------------------------------------------------------
    # PHASE 2/3 — freeze whitelist + values + model-scope registry
    # ------------------------------------------------------------------
    print("\nPHASE 2/3 — freeze parsimonious static whitelist")
    start2 = time.time()

    whitelist_rows = []

    for order, (col, family) in enumerate(PRIMARY_STATIC_FEATURES, 1):
        whitelist_rows.append(
            {
                "feature_order": order,
                "source": "static_receptor",
                "feature_column": col,
                "canonical_feature_name":
                    f"static__{col}",
                "feature_family": family,
                "model_role": "PRIMARY_STATIC_DESCRIPTOR",
                "value_semantics":
                    "MODEL_RECEPTOR_DESCRIPTOR",
                "scaling_policy":
                    "FIT_ONLY_IF_MODEL_REQUIRES_SCALING",
            }
        )

    whitelist = pd.DataFrame(whitelist_rows)

    value_cols = ["receptor_id"] + required_features
    values = d[value_cols].copy()

    excluded_rows = []

    for col in d.columns:
        if col == "receptor_id":
            continue
        if col in required_features:
            continue

        excluded_rows.append(
            {
                "feature_column": col,
                "exclusion_reason":
                    EXCLUSION_REASONS.get(
                        col,
                        "NOT_SELECTED_FOR_PARSIMONIOUS_PRIMARY_STATIC_CORE",
                    ),
            }
        )

    excluded = pd.DataFrame(excluded_rows)

    target_receptors = sorted(
        labels["receptor_id"].astype(str).unique()
    )

    scope = pd.DataFrame(
        {
            "receptor_id": sorted(
                d["receptor_id"].astype(str).unique()
            )
        }
    )

    scope["has_canonical_modeling_target_v1_3"] = (
        scope["receptor_id"].isin(target_receptors)
    )

    scope["supervised_model_scope"] = np.where(
        scope["has_canonical_modeling_target_v1_3"],
        "IN_PRIMARY_SUPERVISED_MODEL",
        "STATIC_RECEPTOR_ONLY__NO_PRIMARY_HYDRO_TARGET",
    )

    no_target = scope[
        ~scope["has_canonical_modeling_target_v1_3"]
    ]["receptor_id"].astype(str).tolist()

    if len(no_target) != 1:
        warnings.append(
            f"Expected 1 receptor without modeling target, found {len(no_target)}: {no_target}"
        )

    progress(
        "PHASE 2/3",
        1,
        1,
        start2,
        (
            f"primary static features={len(whitelist)} "
            f"| target receptors={len(target_receptors)} "
            f"| no-target={no_target}"
        ),
    )

    # ------------------------------------------------------------------
    # PHASE 3/3 — policy, outputs, checksums
    # ------------------------------------------------------------------
    print("\nPHASE 3/3 — write canonical static artifacts")
    start3 = time.time()

    whitelist_out = (
        out / "static_receptor_descriptor_whitelist_canonical_v1_1.csv"
    )
    values_out = (
        out / "static_receptor_descriptor_values_canonical_v1_1.csv"
    )
    excluded_out = (
        out / "excluded_static_descriptor_registry_v1_1.csv"
    )
    scope_out = (
        out / "receptor_static_model_scope_registry_v1_1.csv"
    )
    policy_out = (
        out / "static_descriptor_policy_canonical_v1_1.csv"
    )
    audit_json = (
        out / "static_descriptor_whitelist_audit_v1_1.json"
    )
    audit_txt = (
        out / "static_descriptor_whitelist_audit_v1_1.txt"
    )

    whitelist.to_csv(whitelist_out, index=False)
    values.to_csv(values_out, index=False)
    excluded.to_csv(excluded_out, index=False)
    scope.to_csv(scope_out, index=False)

    policy = pd.DataFrame(
        [
            {
                "policy_id": "S1",
                "rule":
                    "Static descriptors characterize model receptors and are "
                    "not asserted to be exact gauged catchment closures.",
            },
            {
                "policy_id": "S2",
                "rule":
                    "Copernicus GLO-30 is used for regional orography/morphometry "
                    "only, not for detailed hydraulic design.",
            },
            {
                "policy_id": "S3",
                "rule":
                    "Primary static CORE is parsimonious; deterministic and strongly "
                    "redundant descriptors are excluded before modeling.",
            },
            {
                "policy_id": "S4",
                "rule":
                    "Centroid longitude/latitude are retained only as metadata and "
                    "not used as primary physical predictors.",
            },
            {
                "policy_id": "S5",
                "rule":
                    "Static feature scaling, if required by a model, must be fitted "
                    "inside the training fold only.",
            },
            {
                "policy_id": "S6",
                "rule":
                    "LIG_ENTELLA remains in the 21-receptor static registry but is "
                    "not in supervised modeling until a canonical primary hydrology "
                    "target exists.",
            },
        ]
    )

    policy.to_csv(policy_out, index=False)

    hash_targets = [
        whitelist_out,
        values_out,
        excluded_out,
        scope_out,
        policy_out,
    ]

    checksums = pd.DataFrame(
        [
            {
                "file": p.name,
                "sha256": sha256(p),
                "bytes": p.stat().st_size,
            }
            for p in hash_targets
        ]
    )

    checksums_out = (
        out / "checksums_sha256_canonical_v1_1.csv"
    )
    checksums.to_csv(checksums_out, index=False)

    family_counts = (
        whitelist["feature_family"]
        .value_counts()
        .rename_axis("feature_family")
        .reset_index(name="feature_count")
    )

    audit = {
        "version": "1.1",
        "overall_status": "PASS",
        "receptors": int(len(values)),
        "supervised_target_receptors": int(len(target_receptors)),
        "receptors_without_canonical_target": no_target,
        "primary_static_features": int(len(whitelist)),
        "primary_static_missing_values": 0,
        "dem_min_valid_fraction_required": MIN_DEM_VALID_FRACTION,
        "centroid_coordinates_in_primary": False,
        "equivalent_diameter_in_primary": False,
        "perimeter_in_primary": False,
        "scaling_fitted_globally": False,
        "static_whitelist_state": "CLOSED_CANONICAL_V1_1",
        "next_step":
            "Build definitive foldwise master matrix from canonical dynamic "
            "whitelist v1.3 + canonical static whitelist v1.1 + canonical "
            "modeling labels v1.3, followed by a final leakage/missingness audit.",
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
        "=" * 210,
        "NW HYDROCLIMATE — CANONICAL STATIC RECEPTOR DESCRIPTOR WHITELIST v1.1",
        "=" * 210,
        "OVERALL STATUS                       : PASS",
        f"Receptors                            : {len(values)}",
        f"Supervised target receptors          : {len(target_receptors)}",
        f"Receptors without canonical target   : {no_target}",
        f"Primary static features              : {len(whitelist)}",
        "Primary static missing values        : 0",
        "Centroid coordinates in primary      : False",
        "Perimeter/equiv. diameter in primary : False",
        "",
        "STATIC FEATURE FAMILY COUNTS",
        family_counts.to_string(index=False),
        "",
        "PRIMARY STATIC WHITELIST",
        whitelist[
            [
                "feature_order",
                "canonical_feature_name",
                "feature_family",
            ]
        ].to_string(index=False),
        "",
        "MODEL SCOPE",
        scope.to_string(index=False),
        "",
        "IMPORTANT",
        "This freezes the static descriptor whitelist; it does not yet build the final master matrix.",
        "The static values describe model receptors, not guaranteed exact gauged catchments.",
        "Any scaling/normalization must be fitted inside FIT only.",
        "",
        f"Whitelist : {whitelist_out}",
        f"Values    : {values_out}",
        f"Scope     : {scope_out}",
        f"Policy    : {policy_out}",
        f"Output    : {out}",
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
        "static whitelist frozen",
    )

    print("\n" + "=" * 210)
    print("\n".join(lines[3:]))
    print("=" * 210)
    print("OVERALL STATUS : PASS")
    print(f"Output         : {out}")
    print("=" * 210)


if __name__ == "__main__":
    main()
