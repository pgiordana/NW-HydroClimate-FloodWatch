#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
finalize_tanaro_arroscia_control_basins_v1_0.py

Consolida i risultati di validazione già PASS nei tre prodotti canonici
del progetto Tanaro–Arroscia:

  tanaro_arroscia/basins/tanaro_upper_catchment.geojson
  tanaro_arroscia/basins/arroscia_upper_catchment.geojson
  tanaro_arroscia/basins/control_sections.geojson

Usa esclusivamente lo scenario canonico full-fill, mantenendo nei metadati
l'esito della sensitivity max_depth=25 m.

Sezioni di controllo idrometrico:
- GARESSIO TANARO
- PIEVE DI TECO (IDRO) / T. ARROSCIA

IMPORTANTE:
queste sono sezioni idrologiche di controllo/validazione e NON coincidono
necessariamente con le future opere di presa/restituzione del collegamento.

Input attesi:
- tanaro_garessio_validation_v1_1/tanaro_garessio_dem_basins.geojson
- tanaro_garessio_validation_v1_1/tanaro_garessio_snap_points.geojson
- tanaro_garessio_validation_v1_1/tanaro_garessio_validation_report_v1_1.json
- arroscia_pieve_teco_validation_v1_0/arroscia_pieve_teco_dem_basins.geojson
- arroscia_pieve_teco_validation_v1_0/arroscia_pieve_teco_snap_points.geojson
- arroscia_pieve_teco_validation_v1_0/arroscia_pieve_teco_validation_report_v1_0.json

NON modifica i risultati di validazione.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd


VERSION = "1.0"
WGS84 = "EPSG:4326"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(path: Path):
    if not path.exists():
        raise SystemExit(f"Input mancante: {path}")


def pick_full_basin(path: Path):
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84, allow_override=True)
    if "scenario" not in gdf.columns:
        raise SystemExit(f"Campo scenario mancante in {path}")
    sub = gdf[gdf["scenario"].astype(str) == "full_fill"].copy()
    if len(sub) != 1:
        raise SystemExit(
            f"Atteso 1 record full_fill in {path}, trovati {len(sub)}"
        )
    return sub.to_crs(WGS84)


def pick_selected_outlet(path: Path):
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84, allow_override=True)
    if "kind" not in gdf.columns:
        raise SystemExit(f"Campo kind mancante in {path}")
    sub = gdf[gdf["kind"].astype(str) == "selected_dem_outlet"].copy()
    if len(sub) != 1:
        raise SystemExit(
            f"Atteso 1 selected_dem_outlet in {path}, trovati {len(sub)}"
        )
    return sub.to_crs(WGS84)


def main():
    root = Path(__file__).resolve().parent
    basins = root / "tanaro_arroscia" / "basins"

    tan_dir = basins / "tanaro_garessio_validation_v1_1"
    arr_dir = basins / "arroscia_pieve_teco_validation_v1_0"

    tan_basin_p = tan_dir / "tanaro_garessio_dem_basins.geojson"
    tan_pts_p = tan_dir / "tanaro_garessio_snap_points.geojson"
    tan_rep_p = tan_dir / "tanaro_garessio_validation_report_v1_1.json"

    arr_basin_p = arr_dir / "arroscia_pieve_teco_dem_basins.geojson"
    arr_pts_p = arr_dir / "arroscia_pieve_teco_snap_points.geojson"
    arr_rep_p = arr_dir / "arroscia_pieve_teco_validation_report_v1_0.json"

    for p in (
        tan_basin_p, tan_pts_p, tan_rep_p,
        arr_basin_p, arr_pts_p, arr_rep_p
    ):
        require(p)

    tan_rep = read_json(tan_rep_p)
    arr_rep = read_json(arr_rep_p)

    tan_basin = pick_full_basin(tan_basin_p)
    arr_basin = pick_full_basin(arr_basin_p)

    tan_out = pick_selected_outlet(tan_pts_p)
    arr_out = pick_selected_outlet(arr_pts_p)

    # --- canonical catchments ---
    tan_canon = tan_basin[["geometry"]].copy()
    tan_canon["basin_id"] = "TANARO_GARESSIO"
    tan_canon["name"] = "Tanaro upper catchment at Garessio"
    tan_canon["watercourse"] = "Tanaro"
    tan_canon["control_section"] = "GARESSIO TANARO"
    tan_canon["scenario"] = "full_fill"
    tan_canon["dem_area_km2"] = float(
        tan_rep["full_fill"]["delineated_area_km2"]
    )
    tan_canon["official_area_km2"] = float(
        tan_rep["official_reference"]["basin_area_km2"]
    )
    tan_canon["area_error_pct"] = float(
        tan_rep["full_fill"]["error_vs_official_pct"]
    )
    tan_canon["sensitivity_xor_km2"] = float(
        tan_rep["scenario_comparison"]["xor_area_km2"]
    )
    tan_canon["validation"] = "PASS"
    tan_canon["validation_version"] = "tanaro_garessio_v1_1"

    arr_canon = arr_basin[["geometry"]].copy()
    arr_canon["basin_id"] = "ARROSCIA_PIEVE_TECO"
    arr_canon["name"] = "Arroscia upper catchment at Pieve di Teco"
    arr_canon["watercourse"] = "Arroscia"
    arr_canon["control_section"] = "PIEVE DI TECO (IDRO)"
    arr_canon["scenario"] = "full_fill"
    arr_canon["dem_area_km2"] = float(
        arr_rep["full_fill"]["delineated_area_km2"]
    )
    arr_canon["official_area_km2"] = float(
        arr_rep["official_reference"]["basin_area_km2"]
    )
    arr_canon["area_error_pct"] = float(
        arr_rep["full_fill"]["error_vs_official_pct"]
    )
    arr_canon["sensitivity_xor_km2"] = float(
        arr_rep["scenario_comparison"]["xor_area_km2"]
    )
    arr_canon["validation"] = "PASS"
    arr_canon["validation_version"] = "arroscia_pieve_teco_v1_0"

    # Ensure geometry last is not required by GeoJSON, but keep explicit CRS.
    tan_path = basins / "tanaro_upper_catchment.geojson"
    arr_path = basins / "arroscia_upper_catchment.geojson"

    tan_canon.to_file(tan_path, driver="GeoJSON")
    arr_canon.to_file(arr_path, driver="GeoJSON")

    # --- canonical control sections ---
    tan_geom = tan_out.geometry.iloc[0]
    arr_geom = arr_out.geometry.iloc[0]

    control = gpd.GeoDataFrame(
        [
            {
                "section_id": "TANARO_GARESSIO",
                "name": "GARESSIO TANARO",
                "watercourse": "Tanaro",
                "role": "hydrologic_control_section",
                "official_lon": float(
                    tan_rep["official_reference"]["longitude"]
                ),
                "official_lat": float(
                    tan_rep["official_reference"]["latitude"]
                ),
                "official_area_km2": float(
                    tan_rep["official_reference"]["basin_area_km2"]
                ),
                "dem_area_km2": float(
                    tan_rep["full_fill"]["delineated_area_km2"]
                ),
                "area_error_pct": float(
                    tan_rep["full_fill"]["error_vs_official_pct"]
                ),
                "snap_distance_station_m": float(
                    tan_rep["selected_outlet"][
                        "distance_to_original_station_m"
                    ]
                ),
                "snap_distance_network_m": float(
                    tan_rep["selected_outlet"][
                        "distance_to_official_line_m"
                    ]
                ),
                "sensitivity_xor_km2": float(
                    tan_rep["scenario_comparison"]["xor_area_km2"]
                ),
                "validation": "PASS",
                "note": (
                    "Sezione idrometrica di controllo; non implica "
                    "localizzazione dell'opera di presa/restituzione."
                ),
                "geometry": tan_geom,
            },
            {
                "section_id": "ARROSCIA_PIEVE_TECO",
                "name": "PIEVE DI TECO (IDRO)",
                "watercourse": "Arroscia",
                "role": "hydrologic_control_section",
                "official_lon": float(
                    arr_rep["official_reference"]["longitude"]
                ),
                "official_lat": float(
                    arr_rep["official_reference"]["latitude"]
                ),
                "official_area_km2": float(
                    arr_rep["official_reference"]["basin_area_km2"]
                ),
                "dem_area_km2": float(
                    arr_rep["full_fill"]["delineated_area_km2"]
                ),
                "area_error_pct": float(
                    arr_rep["full_fill"]["error_vs_official_pct"]
                ),
                "snap_distance_station_m": float(
                    arr_rep["selected_outlet"][
                        "distance_to_original_station_m"
                    ]
                ),
                "snap_distance_network_m": float(
                    arr_rep["selected_outlet"][
                        "distance_to_official_line_m"
                    ]
                ),
                "sensitivity_xor_km2": float(
                    arr_rep["scenario_comparison"]["xor_area_km2"]
                ),
                "validation": "PASS",
                "note": (
                    "Sezione idrometrica di controllo; non implica "
                    "localizzazione dell'opera di presa/restituzione."
                ),
                "geometry": arr_geom,
            },
        ],
        geometry="geometry",
        crs=WGS84,
    )

    control_path = basins / "control_sections.geojson"
    control.to_file(control_path, driver="GeoJSON")

    # --- catalog/report ---
    catalog = {
        "version": VERSION,
        "status": "CLOSED_PASS",
        "canonical_routing": "full_fill",
        "sensitivity_routing": "max_depth_25m",
        "reason_for_canonical_choice": (
            "Both official control sections are unchanged between full-fill "
            "and max_depth=25m sensitivity (XOR basin area = 0 km2)."
        ),
        "tanaro": {
            "control_section": "GARESSIO TANARO",
            "official_area_km2": float(
                tan_rep["official_reference"]["basin_area_km2"]
            ),
            "dem_area_km2": float(
                tan_rep["full_fill"]["delineated_area_km2"]
            ),
            "error_pct": float(
                tan_rep["full_fill"]["error_vs_official_pct"]
            ),
            "sensitivity_xor_km2": float(
                tan_rep["scenario_comparison"]["xor_area_km2"]
            ),
            "source_validation": str(tan_rep_p),
        },
        "arroscia": {
            "control_section": "PIEVE DI TECO (IDRO)",
            "official_area_km2": float(
                arr_rep["official_reference"]["basin_area_km2"]
            ),
            "dem_area_km2": float(
                arr_rep["full_fill"]["delineated_area_km2"]
            ),
            "error_pct": float(
                arr_rep["full_fill"]["error_vs_official_pct"]
            ),
            "sensitivity_xor_km2": float(
                arr_rep["scenario_comparison"]["xor_area_km2"]
            ),
            "source_validation": str(arr_rep_p),
        },
        "canonical_outputs": {
            "tanaro_upper_catchment": str(tan_path),
            "arroscia_upper_catchment": str(arr_path),
            "control_sections": str(control_path),
        },
        "interpretation": (
            "These catchments and points are hydrologic control basins/sections "
            "for joint hydrograph and flood-timing analysis. They must not be "
            "treated as predetermined hydraulic transfer intake/outfall sites."
        ),
    }

    cat_path = basins / "canonical_basins_catalog_v1_0.json"
    cat_path.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    txt_path = basins / "canonical_basins_report_v1_0.txt"
    lines = [
        "TANARO–ARROSCIA | CANONICAL CONTROL BASINS v1.0",
        "=" * 88,
        "STATUS                    : CLOSED_PASS",
        "Canonical routing         : full_fill",
        "Sensitivity routing       : max_depth_25m",
        "",
        "TANARO — GARESSIO TANARO",
        f"  Official area [km2]     : {catalog['tanaro']['official_area_km2']:.3f}",
        f"  DEM area [km2]          : {catalog['tanaro']['dem_area_km2']:.6f}",
        f"  Error [%]               : {catalog['tanaro']['error_pct']:.3f}",
        f"  Sensitivity XOR [km2]   : {catalog['tanaro']['sensitivity_xor_km2']:.6f}",
        "",
        "ARROSCIA — PIEVE DI TECO (IDRO)",
        f"  Official area [km2]     : {catalog['arroscia']['official_area_km2']:.3f}",
        f"  DEM area [km2]          : {catalog['arroscia']['dem_area_km2']:.6f}",
        f"  Error [%]               : {catalog['arroscia']['error_pct']:.3f}",
        f"  Sensitivity XOR [km2]   : {catalog['arroscia']['sensitivity_xor_km2']:.6f}",
        "",
        "CANONICAL OUTPUTS",
        f"  {tan_path}",
        f"  {arr_path}",
        f"  {control_path}",
        "",
        "NOTA",
        "  Le sezioni sono sezioni idrologiche di controllo.",
        "  Non costituiscono scelta dell'opera di presa o di restituzione.",
        "  La localizzazione delle opere verrà valutata dopo l'analisi congiunta",
        "  degli idrogrammi, dell'asincronia dei picchi e della capacità residua.",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nCatalogo: {cat_path}")
    print(f"Report  : {txt_path}")


if __name__ == "__main__":
    main()
