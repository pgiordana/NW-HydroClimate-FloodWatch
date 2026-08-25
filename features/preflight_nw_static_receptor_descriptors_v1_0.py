#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preflight_nw_static_receptor_descriptors_v1_0.py

PREFLIGHT DEI DESCRITTORI STATICI DEI 21 RECETTORI REGIONALI.

SCOPO
-----
Costruire e auditare i descrittori statici necessari al modello pooled/
hierarchical regionale PRIMA del master dataset definitivo.

INPUT CANONICI
--------------
1) Geometrie:
   basins_final/nw_receptors_final.geojson

2) Copernicus DEM GLO-30:
   regional_inputs/terrain/copernicus_dem_glo30/tiles/*.tif
   regional_inputs/terrain/copernicus_dem_glo30/tile_index.csv

Il downloader canonico regionale ha salvato il DEM in questa struttura.

SEMANTICA
---------
Le geometrie sono i 21 recettori spaziali del modello. I descrittori derivati
non devono essere interpretati automaticamente come morfometria di una sezione
idrologica di chiusura perfetta se il receptor è stato definito come unità
modellistica aggregata.

DESCRITTORI
-----------
Geometria:
- area_km2
- perimeter_km
- centroid_lon / centroid_lat
- equivalent_diameter_km
- circularity_4piA_P2
- convexity_area_ratio

DEM:
- elevation min/p10/median/mean/p90/max/std
- relief
- hypsometric_integral_proxy = (mean-min)/(max-min)
- fractions elevation <500, 500-1000, 1000-1500, >=1500 m
- slope mean/median/p90/max/std
- fractions slope >=15° and >=30°

METODO DEM
----------
Per ogni receptor:
1) seleziona solo i tile GLO-30 che intersecano la bbox;
2) crea un mosaico limitato alla bbox del receptor;
3) riproietta il mosaico in EPSG:32632 a 30 m;
4) calcola pendenza in coordinate metriche;
5) rasterizza la geometria del receptor sul raster UTM;
6) estrae le statistiche solo dai pixel interni validi.

Non viene creato un DTM di dettaglio e non si usa questo prodotto per sezioni
idrauliche o progetto di galleria. Copernicus GLO-30 è un DSM regionale.

OUTPUT
------
nw_static_receptor_descriptors_preflight_v1_0/
  static_receptor_descriptors_v1_0.csv
  static_receptor_dem_tile_usage_v1_0.csv
  static_receptor_geometry_audit_v1_0.csv
  static_receptor_descriptor_audit_v1_0.json
  static_receptor_descriptor_audit_v1_0.txt
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.merge import merge
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, Resampling
    from shapely.geometry import box
    from shapely.ops import transform as shp_transform
    from pyproj import Transformer
except Exception as exc:
    raise SystemExit(
        "\nMancano i pacchetti GIS necessari.\n"
        "Esegui questo script con l'ambiente terrain, per esempio:\n"
        "  ../.venv_terrain/bin/python "
        "preflight_nw_static_receptor_descriptors_v1_0.py\n"
        f"\nErrore import: {exc}"
    )


EXPECTED_RECEPTORS = 21
TARGET_CRS = "EPSG:32632"
TARGET_RES_M = 30.0
NODATA = np.nan


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
        f"| rate {rate:7.3f}/s | ETA {fmt_seconds(eta)}"
    )
    if current:
        msg += f" | {str(current)[:120]}"

    print(msg.ljust(260), end="", flush=True)
    if done >= total:
        print(flush=True)


def q(arr, p):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return np.nan
    return float(np.nanquantile(arr, p))


def finite_stats(arr, prefix):
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]

    if not x.size:
        return {
            f"{prefix}_min": np.nan,
            f"{prefix}_p10": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_std": np.nan,
        }

    return {
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_p10": q(x, 0.10),
        f"{prefix}_median": q(x, 0.50),
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_p90": q(x, 0.90),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_std": float(np.std(x)),
    }


def safe_fraction(mask):
    arr = np.asarray(mask)
    if arr.size == 0:
        return np.nan
    return float(np.mean(arr))


def load_tiles(tile_dir: Path, tile_index: Path):
    """
    Return list of {path, bounds_wgs84, crs, index_status}.
    """
    rows = []

    indexed = {}

    if tile_index.exists():
        idx = pd.read_csv(tile_index, low_memory=False)
        for _, r in idx.iterrows():
            raw = str(r.get("path", "")).strip()
            if raw and raw.lower() not in {"nan", "none"}:
                p = Path(raw).expanduser()
                if not p.is_absolute():
                    p = (tile_index.parent / p).resolve()
                indexed[p.name] = str(r.get("status", ""))

    files = sorted(tile_dir.glob("*.tif"))

    if not files:
        raise SystemExit(f"Nessun DEM tile trovato in {tile_dir}")

    to_wgs84_cache = {}

    for p in files:
        try:
            with rasterio.open(p) as ds:
                if ds.crs is None:
                    raise RuntimeError("CRS missing")

                key = str(ds.crs)

                if key not in to_wgs84_cache:
                    to_wgs84_cache[key] = Transformer.from_crs(
                        ds.crs,
                        "EPSG:4326",
                        always_xy=True,
                    ).transform

                geom = box(*ds.bounds)
                geom_wgs = shp_transform(
                    to_wgs84_cache[key],
                    geom,
                )

                rows.append(
                    {
                        "path": p,
                        "name": p.name,
                        "bounds_wgs84": geom_wgs.bounds,
                        "crs": str(ds.crs),
                        "width": int(ds.width),
                        "height": int(ds.height),
                        "index_status": indexed.get(p.name, ""),
                    }
                )

        except Exception as exc:
            raise SystemExit(
                f"Tile DEM non leggibile: {p}\n{exc}"
            )

    return rows


def overlapping_tiles(geom_wgs, tiles):
    gb = box(*geom_wgs.bounds)

    return [
        t
        for t in tiles
        if gb.intersects(box(*t["bounds_wgs84"]))
    ]


def mosaic_bbox_wgs84(geom_wgs, selected_tiles):
    """
    Merge selected source tiles within geometry bbox.
    Assumes GLO-30 tiles share compatible CRS/grid.
    """
    srcs = [rasterio.open(t["path"]) for t in selected_tiles]

    try:
        crs_set = {str(s.crs) for s in srcs}

        if len(crs_set) != 1:
            raise RuntimeError(
                f"DEM tile CRS non uniformi: {sorted(crs_set)}"
            )

        source_crs = srcs[0].crs

        # GeoJSON receptor is WGS84. Convert bbox to source CRS if necessary.
        if str(source_crs).upper() in {
            "EPSG:4326",
            "OGC:CRS84",
        }:
            src_bounds = geom_wgs.bounds
        else:
            tfm = Transformer.from_crs(
                "EPSG:4326",
                source_crs,
                always_xy=True,
            ).transform
            gsrc = shp_transform(tfm, geom_wgs)
            src_bounds = gsrc.bounds

        arr, transform = merge(
            srcs,
            bounds=src_bounds,
            nodata=np.nan,
            dtype="float32",
            masked=False,
        )

        return arr[0].astype(np.float32), transform, source_crs

    finally:
        for s in srcs:
            s.close()


def reproject_bbox_to_utm(
    src_arr,
    src_transform,
    src_crs,
    geom_utm,
):
    minx, miny, maxx, maxy = geom_utm.bounds

    width = max(
        1,
        int(math.ceil((maxx - minx) / TARGET_RES_M)),
    )
    height = max(
        1,
        int(math.ceil((maxy - miny) / TARGET_RES_M)),
    )

    dst_transform = from_origin(
        minx,
        maxy,
        TARGET_RES_M,
        TARGET_RES_M,
    )

    dst = np.full(
        (height, width),
        np.nan,
        dtype=np.float32,
    )

    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=TARGET_CRS,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
        num_threads=2,
    )

    return dst, dst_transform


def terrain_stats(dem, transform, geom_utm):
    inside = geometry_mask(
        [geom_utm.__geo_interface__],
        out_shape=dem.shape,
        transform=transform,
        invert=True,
        all_touched=False,
    )

    valid = inside & np.isfinite(dem)
    z = dem[valid].astype(float)

    if not z.size:
        raise RuntimeError("Nessun pixel DEM valido nel receptor.")

    # Gradient is computed on the full bbox, before polygon masking, to avoid
    # artificial edge slopes generated by polygon nodata boundaries.
    gy, gx = np.gradient(
        dem.astype(np.float64),
        TARGET_RES_M,
        TARGET_RES_M,
    )

    slope = np.degrees(
        np.arctan(
            np.sqrt(
                np.square(gx)
                + np.square(gy)
            )
        )
    )

    slope_valid = slope[valid]
    slope_valid = slope_valid[
        np.isfinite(slope_valid)
    ]

    elev = finite_stats(z, "elevation_m")
    slp = finite_stats(slope_valid, "slope_deg")

    zmin = elev["elevation_m_min"]
    zmax = elev["elevation_m_max"]
    zmean = elev["elevation_m_mean"]

    relief = zmax - zmin

    hyps = (
        (zmean - zmin) / relief
        if np.isfinite(relief) and relief > 0
        else np.nan
    )

    result = {
        **elev,
        "relief_m": relief,
        "hypsometric_integral_proxy": hyps,
        "elevation_fraction_lt500m":
            safe_fraction(z < 500.0),
        "elevation_fraction_500_1000m":
            safe_fraction(
                (z >= 500.0) & (z < 1000.0)
            ),
        "elevation_fraction_1000_1500m":
            safe_fraction(
                (z >= 1000.0) & (z < 1500.0)
            ),
        "elevation_fraction_ge1500m":
            safe_fraction(z >= 1500.0),
        **slp,
        "slope_fraction_ge15deg":
            safe_fraction(slope_valid >= 15.0),
        "slope_fraction_ge30deg":
            safe_fraction(slope_valid >= 30.0),
        "dem_valid_pixels": int(z.size),
        "dem_polygon_pixels": int(inside.sum()),
        "dem_pixel_valid_fraction":
            float(z.size / inside.sum())
            if inside.sum()
            else np.nan,
    }

    return result


def main():
    root = Path(__file__).resolve().parent

    receptors_p = (
        root
        / "basins_final"
        / "nw_receptors_final.geojson"
    )

    terrain_root = (
        root
        / "regional_inputs"
        / "terrain"
        / "copernicus_dem_glo30"
    )

    tile_dir = terrain_root / "tiles"
    tile_index = terrain_root / "tile_index.csv"

    out = (
        root
        / "nw_static_receptor_descriptors_preflight_v1_0"
    )
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 212)
    print("NW HYDROCLIMATE — STATIC RECEPTOR DESCRIPTORS PREFLIGHT v1.0")
    print("=" * 212)

    if not receptors_p.exists():
        raise SystemExit(f"Manca: {receptors_p}")

    # ------------------------------------------------------------------
    # PHASE 1/4 — geometries and DEM inventory
    # ------------------------------------------------------------------
    print("\nPHASE 1/4 — audit receptor geometries and DEM tiles")
    start1 = time.time()

    gdf = gpd.read_file(receptors_p)

    if "receptor_id" not in gdf.columns:
        raise SystemExit(
            "nw_receptors_final.geojson non contiene receptor_id."
        )

    if gdf.crs is None:
        # GeoJSON RFC 7946 default.
        gdf = gdf.set_crs(
            "EPSG:4326",
            allow_override=True,
        )

    gdf_wgs = gdf.to_crs("EPSG:4326")
    gdf_utm = gdf.to_crs(TARGET_CRS)

    geometry_audit_rows = []

    for idx, row in gdf_wgs.iterrows():
        rid = str(row["receptor_id"])
        geom = row.geometry

        geometry_audit_rows.append(
            {
                "receptor_id": rid,
                "geometry_type": geom.geom_type,
                "is_empty": bool(geom.is_empty),
                "is_valid": bool(geom.is_valid),
                "wgs84_minx": float(geom.bounds[0]),
                "wgs84_miny": float(geom.bounds[1]),
                "wgs84_maxx": float(geom.bounds[2]),
                "wgs84_maxy": float(geom.bounds[3]),
            }
        )

    geometry_audit = pd.DataFrame(
        geometry_audit_rows
    )

    if len(gdf_wgs) != EXPECTED_RECEPTORS:
        raise SystemExit(
            f"Recettori={len(gdf_wgs)}, attesi={EXPECTED_RECEPTORS}"
        )

    if gdf_wgs["receptor_id"].duplicated().any():
        raise SystemExit("receptor_id duplicati.")

    bad_geom = geometry_audit[
        geometry_audit["is_empty"]
        | ~geometry_audit["is_valid"]
    ]

    if len(bad_geom):
        raise SystemExit(
            f"Geometrie vuote/non valide={len(bad_geom)}"
        )

    tiles = load_tiles(
        tile_dir,
        tile_index,
    )

    progress(
        "PHASE 1/4",
        1,
        1,
        start1,
        f"receptors={len(gdf_wgs)} DEM tiles={len(tiles)}",
    )

    # ------------------------------------------------------------------
    # PHASE 2/4 — geometry descriptors
    # ------------------------------------------------------------------
    print("\nPHASE 2/4 — compute metric geometry descriptors")
    start2 = time.time()

    geom_rows = []

    total = len(gdf_wgs)

    for i, (_, row_wgs) in enumerate(
        gdf_wgs.sort_values("receptor_id").iterrows(),
        1,
    ):
        rid = str(row_wgs["receptor_id"])

        geom_wgs = row_wgs.geometry

        geom_utm = gdf_utm.loc[
            gdf_wgs["receptor_id"].astype(str).eq(rid)
        ].iloc[0].geometry

        area_m2 = float(geom_utm.area)
        perimeter_m = float(geom_utm.length)

        centroid = geom_wgs.centroid

        eq_diam_m = (
            2.0 * math.sqrt(area_m2 / math.pi)
            if area_m2 > 0
            else np.nan
        )

        circularity = (
            4.0 * math.pi * area_m2
            / (perimeter_m * perimeter_m)
            if perimeter_m > 0
            else np.nan
        )

        hull_area = float(geom_utm.convex_hull.area)

        convexity = (
            area_m2 / hull_area
            if hull_area > 0
            else np.nan
        )

        geom_rows.append(
            {
                "receptor_id": rid,
                "area_km2": area_m2 / 1e6,
                "perimeter_km": perimeter_m / 1000.0,
                "centroid_lon": float(centroid.x),
                "centroid_lat": float(centroid.y),
                "equivalent_diameter_km":
                    eq_diam_m / 1000.0,
                "circularity_4piA_P2":
                    circularity,
                "convexity_area_ratio":
                    convexity,
            }
        )

        progress(
            "PHASE 2/4",
            i,
            total,
            start2,
            f"{rid} | area={area_m2/1e6:.1f} km2",
        )

    geom_df = pd.DataFrame(geom_rows)

    # ------------------------------------------------------------------
    # PHASE 3/4 — DEM descriptors
    # ------------------------------------------------------------------
    print("\nPHASE 3/4 — compute DEM elevation/slope descriptors")
    start3 = time.time()

    terrain_rows = []
    tile_usage_rows = []

    sorted_wgs = gdf_wgs.sort_values(
        "receptor_id"
    )

    total = len(sorted_wgs)

    for i, (_, row_wgs) in enumerate(
        sorted_wgs.iterrows(),
        1,
    ):
        rid = str(row_wgs["receptor_id"])
        geom_wgs = row_wgs.geometry

        geom_utm = gdf_utm.loc[
            gdf_wgs["receptor_id"].astype(str).eq(rid)
        ].iloc[0].geometry

        selected = overlapping_tiles(
            geom_wgs,
            tiles,
        )

        if not selected:
            raise SystemExit(
                f"{rid}: nessun tile DEM intersecante."
            )

        for t in selected:
            tile_usage_rows.append(
                {
                    "receptor_id": rid,
                    "tile_name": t["name"],
                    "tile_path": str(t["path"]),
                    "tile_crs": t["crs"],
                    "tile_index_status":
                        t["index_status"],
                }
            )

        try:
            src_arr, src_transform, src_crs = (
                mosaic_bbox_wgs84(
                    geom_wgs,
                    selected,
                )
            )

            dem_utm, dst_transform = (
                reproject_bbox_to_utm(
                    src_arr,
                    src_transform,
                    src_crs,
                    geom_utm,
                )
            )

            stats = terrain_stats(
                dem_utm,
                dst_transform,
                geom_utm,
            )

        except Exception as exc:
            raise SystemExit(
                f"{rid}: errore terrain processing: {exc}"
            )

        terrain_rows.append(
            {
                "receptor_id": rid,
                "dem_tiles_used": int(len(selected)),
                "dem_target_crs": TARGET_CRS,
                "dem_target_resolution_m":
                    TARGET_RES_M,
                **stats,
            }
        )

        progress(
            "PHASE 3/4",
            i,
            total,
            start3,
            (
                f"{rid} | tiles={len(selected)} "
                f"| valid_px={stats['dem_valid_pixels']} "
                f"| elev_mean={stats['elevation_m_mean']:.1f}m"
            ),
        )

    terrain_df = pd.DataFrame(terrain_rows)
    tile_usage = pd.DataFrame(tile_usage_rows)

    # ------------------------------------------------------------------
    # PHASE 4/4 — combine and audit
    # ------------------------------------------------------------------
    print("\nPHASE 4/4 — combine descriptors and final audit")
    start4 = time.time()

    descriptors = geom_df.merge(
        terrain_df,
        on="receptor_id",
        how="outer",
        validate="one_to_one",
    )

    descriptors["static_descriptor_semantics"] = (
        "MODEL_RECEPTOR_GEOMETRY__NOT_ASSUMED_EXACT_GAUGED_CATCHMENT"
    )

    descriptors["dem_source"] = (
        "Copernicus DEM GLO-30 DSM"
    )

    descriptors["dem_use_scope"] = (
        "REGIONAL_OROGRAPHY_AND_MORPHOMETRY_ONLY"
    )

    duplicates = int(
        descriptors["receptor_id"].duplicated().sum()
    )

    missing_receptors = int(
        descriptors["receptor_id"].isna().sum()
    )

    numeric_cols = [
        c
        for c in descriptors.columns
        if c not in {
            "receptor_id",
            "dem_target_crs",
            "static_descriptor_semantics",
            "dem_source",
            "dem_use_scope",
        }
    ]

    numeric_missing = {}

    for c in numeric_cols:
        x = pd.to_numeric(
            descriptors[c],
            errors="coerce",
        )

        m = int(x.isna().sum())

        if m:
            numeric_missing[c] = m

    low_dem_coverage = descriptors[
        pd.to_numeric(
            descriptors["dem_pixel_valid_fraction"],
            errors="coerce",
        ) < 0.98
    ].copy()

    implausible_area = descriptors[
        pd.to_numeric(
            descriptors["area_km2"],
            errors="coerce",
        ) <= 0
    ].copy()

    implausible_relief = descriptors[
        pd.to_numeric(
            descriptors["relief_m"],
            errors="coerce",
        ) < 0
    ].copy()

    if (
        len(descriptors) != EXPECTED_RECEPTORS
        or duplicates
        or missing_receptors
        or len(implausible_area)
        or len(implausible_relief)
    ):
        overall = "FAIL"
    elif numeric_missing:
        overall = "PASS_WITH_DESCRIPTOR_MISSINGNESS_REVIEW"
    elif len(low_dem_coverage):
        overall = "PASS_WITH_LOW_DEM_COVERAGE_REVIEW"
    else:
        overall = "PASS"

    descriptors_out = (
        out / "static_receptor_descriptors_v1_0.csv"
    )
    tile_usage_out = (
        out / "static_receptor_dem_tile_usage_v1_0.csv"
    )
    geometry_audit_out = (
        out / "static_receptor_geometry_audit_v1_0.csv"
    )
    audit_json = (
        out / "static_receptor_descriptor_audit_v1_0.json"
    )
    audit_txt = (
        out / "static_receptor_descriptor_audit_v1_0.txt"
    )

    descriptors.to_csv(
        descriptors_out,
        index=False,
    )

    tile_usage.to_csv(
        tile_usage_out,
        index=False,
    )

    geometry_audit.to_csv(
        geometry_audit_out,
        index=False,
    )

    report = {
        "version": "1.0",
        "overall_status": overall,
        "receptors": int(len(descriptors)),
        "expected_receptors": EXPECTED_RECEPTORS,
        "duplicate_receptor_ids": duplicates,
        "dem_tiles_available": int(len(tiles)),
        "target_crs": TARGET_CRS,
        "target_resolution_m": TARGET_RES_M,
        "descriptor_numeric_missing_columns":
            numeric_missing,
        "receptors_dem_valid_fraction_lt_0_98":
            low_dem_coverage[
                "receptor_id"
            ].astype(str).tolist(),
        "dem_source":
            "Copernicus DEM GLO-30 DSM",
        "dem_suitable_for_detailed_hydraulic_design":
            False,
        "receptor_geometry_semantics":
            "MODEL_RECEPTOR_NOT_ASSUMED_EXACT_GAUGED_CATCHMENT",
        "imputation_performed": False,
        "next_step": (
            "If PASS, review/freeze the static descriptor whitelist, "
            "then join canonical dynamic features + static descriptors + "
            "canonical v1.3 modeling labels into the definitive foldwise "
            "master matrix."
        ),
    }

    audit_json.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    compact_cols = [
        "receptor_id",
        "area_km2",
        "elevation_m_mean",
        "elevation_m_p90",
        "relief_m",
        "slope_deg_mean",
        "slope_deg_p90",
        "dem_pixel_valid_fraction",
        "dem_tiles_used",
    ]

    lines = [
        "=" * 212,
        "NW HYDROCLIMATE — STATIC RECEPTOR DESCRIPTORS PREFLIGHT v1.0",
        "=" * 212,
        f"OVERALL STATUS                         : {overall}",
        f"Receptors                              : {len(descriptors)}",
        f"DEM tiles available                    : {len(tiles)}",
        f"Duplicate receptor ids                 : {duplicates}",
        f"Descriptor columns with missing values : {len(numeric_missing)}",
        f"Receptors DEM valid fraction < 0.98    : {len(low_dem_coverage)}",
        "",
        "RECEPTOR STATIC SUMMARY",
        descriptors[compact_cols].to_string(index=False),
        "",
        "IMPORTANT",
        "Copernicus GLO-30 is used as regional DSM, not as detailed hydraulic DTM.",
        "The 21 polygons are model receptors; descriptor values do not prove an exact gauge catchment closure.",
        "No imputation is performed.",
        "",
        f"Descriptors : {descriptors_out}",
        f"Tile usage  : {tile_usage_out}",
        f"Geometry QC : {geometry_audit_out}",
        f"Output      : {out}",
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
        f"status={overall}",
    )

    print("\n" + "=" * 212)
    print("\n".join(lines[3:]))
    print("=" * 212)
    print(f"OVERALL STATUS : {overall}")
    print(f"Output         : {out}")
    print("=" * 212)


if __name__ == "__main__":
    main()
