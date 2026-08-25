#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_tanaro_arroscia_terrain_v1_1.py

Preparazione topografica canonica Tanaro–Arroscia da TINITALY 1.1 (10 m).\n\nCorrezione v1.1: compatibilità Numba/pyflwdir per il calcolo slope;\nla trasformazione rasterio.Affine viene convertita in tupla numerica.

FASE PREPARE
------------
1. individua i 4 tasselli TINITALY
2. verifica CRS, risoluzione, nodata, allineamento
3. controlla quantitativamente le fasce di sovrapposizione
4. costruisce il mosaico completo 10 m
5. ritaglia una work area attorno al dominio Tanaro–Arroscia, con buffer metrico
6. produce report QA JSON/TXT

FASE HYDROLOGY
--------------
7. depression filling (Wang & Liu) con pyflwdir
8. D8 flow direction
9. slope
10. flow accumulation in celle
11. upstream area in km²
12. report QA idrologico JSON/TXT

NON modifica mai i TIF raw.

Default domain WGS84:
    west=7.68, south=43.95, east=8.18, north=44.28
Default buffer:
    10 km

Output:
    tanaro_arroscia/terrain/
        dem_tinitaly_mosaic_10m.tif
        dem_tinitaly_workarea_10m.tif
        dem_hydroconditioned_10m.tif
        slope.tif
        flow_direction_d8.tif
        flow_accumulation_cells.tif
        flow_accumulation_km2.tif
        terrain_prepare_report_v1_1.json
        terrain_prepare_report_v1_1.txt
        terrain_hydrology_report_v1_1.json
        terrain_hydrology_report_v1_1.txt

Uso consigliato:
    python build_tanaro_arroscia_terrain_v1_1.py --prepare-only

Dopo aver controllato il report:
    python build_tanaro_arroscia_terrain_v1_1.py --hydrology-only

Oppure, dopo il collaudo:
    python build_tanaro_arroscia_terrain_v1_1.py

Ambiente:
    .venv_terrain
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.windows import from_bounds, bounds as window_bounds
from rasterio.warp import transform_bounds


VERSION = "1.1"
EXPECTED_CRS_EPSG = 32632
EXPECTED_RES = 10.0
EXPECTED_NODATA = -9999.0
EXPECTED_DTYPE = "float32"
EXPECTED_TILE_COUNT = 4

DEFAULT_WGS84_BBOX = (7.68, 43.95, 8.18, 44.28)  # west, south, east, north
DEFAULT_BUFFER_KM = 10.0

D8_NODATA = 247
FLOAT_NODATA = -9999.0


def die(msg: str, code: int = 3) -> None:
    print(f"\nERRORE: {msg}", file=sys.stderr)
    raise SystemExit(code)


def tif_tiles(root: Path) -> List[Path]:
    return sorted(root.glob("**/*.tif"))


def aligned_to_grid(value: float, origin: float, res: float, tol: float = 1e-6) -> bool:
    q = (value - origin) / res
    return abs(q - round(q)) <= tol


def inspect_tiles(paths: List[Path]) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    warnings: List[str] = []

    if len(paths) != EXPECTED_TILE_COUNT:
        errors.append(f"Numero tasselli: {len(paths)}; attesi {EXPECTED_TILE_COUNT}.")

    ref = None
    for p in paths:
        with rasterio.open(p) as ds:
            row = {
                "file": str(p),
                "name": p.name,
                "crs": str(ds.crs),
                "epsg": ds.crs.to_epsg() if ds.crs else None,
                "width": ds.width,
                "height": ds.height,
                "res": [float(ds.res[0]), float(ds.res[1])],
                "bounds": {
                    "left": ds.bounds.left,
                    "bottom": ds.bounds.bottom,
                    "right": ds.bounds.right,
                    "top": ds.bounds.top,
                },
                "nodata": ds.nodata,
                "dtype": ds.dtypes[0],
                "count": ds.count,
                "transform": tuple(ds.transform),
            }
            rows.append(row)

            if row["epsg"] != EXPECTED_CRS_EPSG:
                errors.append(f"{p.name}: EPSG={row['epsg']}, atteso {EXPECTED_CRS_EPSG}.")
            if abs(ds.res[0] - EXPECTED_RES) > 1e-9 or abs(ds.res[1] - EXPECTED_RES) > 1e-9:
                errors.append(f"{p.name}: risoluzione={ds.res}, attesa 10 x 10 m.")
            if ds.nodata != EXPECTED_NODATA:
                errors.append(f"{p.name}: nodata={ds.nodata}, atteso {EXPECTED_NODATA}.")
            if ds.dtypes[0] != EXPECTED_DTYPE:
                errors.append(f"{p.name}: dtype={ds.dtypes[0]}, atteso {EXPECTED_DTYPE}.")
            if ds.count != 1:
                errors.append(f"{p.name}: band count={ds.count}, atteso 1.")
            if abs(ds.transform.b) > 1e-12 or abs(ds.transform.d) > 1e-12:
                errors.append(f"{p.name}: raster ruotato/skewed; non ammesso.")

            if ref is None:
                ref = row
            else:
                if row["crs"] != ref["crs"]:
                    errors.append(f"{p.name}: CRS diverso dal riferimento {ref['name']}.")
                if row["dtype"] != ref["dtype"]:
                    errors.append(f"{p.name}: dtype diverso dal riferimento {ref['name']}.")

    if rows:
        # Tutti i bordi devono ricadere sulla stessa griglia a 10 m.
        origin_x = min(r["bounds"]["left"] for r in rows)
        origin_y = max(r["bounds"]["top"] for r in rows)
        for r in rows:
            b = r["bounds"]
            for key in ("left", "right"):
                if not aligned_to_grid(float(b[key]), origin_x, EXPECTED_RES):
                    errors.append(f"{r['name']}: {key} non allineato alla griglia comune.")
            for key in ("top", "bottom"):
                if not aligned_to_grid(float(b[key]), origin_y, EXPECTED_RES):
                    errors.append(f"{r['name']}: {key} non allineato alla griglia comune.")

    return rows, errors, warnings


def intersect_bounds(a, b):
    left = max(a.left, b.left)
    bottom = max(a.bottom, b.bottom)
    right = min(a.right, b.right)
    top = min(a.top, b.top)
    if right <= left or top <= bottom:
        return None
    return left, bottom, right, top


def overlap_audit(paths: List[Path]) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    results: List[Dict[str, Any]] = []
    errors: List[str] = []
    warnings: List[str] = []

    for pa, pb in combinations(paths, 2):
        with rasterio.open(pa) as a, rasterio.open(pb) as b:
            ib = intersect_bounds(a.bounds, b.bounds)
            if ib is None:
                continue

            wa = from_bounds(*ib, transform=a.transform).round_offsets().round_lengths()
            wb = from_bounds(*ib, transform=b.transform).round_offsets().round_lengths()

            aa = a.read(1, window=wa, masked=True)
            bb = b.read(1, window=wb, masked=True)

            if aa.shape != bb.shape:
                errors.append(
                    f"Overlap {pa.name}/{pb.name}: shape diverse {aa.shape} vs {bb.shape}."
                )
                continue

            av = np.asarray(aa.filled(np.nan), dtype=np.float64)
            bv = np.asarray(bb.filled(np.nan), dtype=np.float64)
            valid = np.isfinite(av) & np.isfinite(bv)

            nvalid = int(valid.sum())
            if nvalid:
                diff = np.abs(av[valid] - bv[valid])
                mean_abs = float(diff.mean())
                p95 = float(np.percentile(diff, 95))
                max_abs = float(diff.max())
                identical = int(np.sum(diff == 0.0))
                identical_fraction = identical / nvalid
            else:
                mean_abs = p95 = max_abs = None
                identical_fraction = None

            rec = {
                "tile_a": pa.name,
                "tile_b": pb.name,
                "bounds": list(map(float, ib)),
                "shape": list(aa.shape),
                "valid_pairs": nvalid,
                "mean_abs_diff_m": mean_abs,
                "p95_abs_diff_m": p95,
                "max_abs_diff_m": max_abs,
                "identical_fraction": identical_fraction,
            }
            results.append(rec)

            if nvalid == 0:
                warnings.append(f"Overlap {pa.name}/{pb.name}: nessuna coppia valida.")
            elif max_abs is not None:
                if max_abs > 10.0:
                    errors.append(
                        f"Overlap {pa.name}/{pb.name}: max differenza {max_abs:.3f} m > 10 m."
                    )
                elif max_abs > 0.50:
                    warnings.append(
                        f"Overlap {pa.name}/{pb.name}: max differenza {max_abs:.3f} m > 0.5 m."
                    )

    if not results:
        warnings.append("Nessuna sovrapposizione geometrica trovata tra i tasselli.")

    return results, errors, warnings


def creation_options() -> Dict[str, Any]:
    return {
        "driver": "GTiff",
        "compress": "DEFLATE",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "IF_SAFER",
    }


def build_mosaic(paths: List[Path], out: Path, overwrite: bool) -> None:
    if out.exists() and not overwrite:
        print(f"SKIP mosaico già presente: {out}")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    print(f"\nCostruzione mosaico: {out}")
    print("Merge policy: FIRST valid pixel; sorgenti in ordine alfabetico.")
    print("Memoria merge: 256 MB.")

    srcs = [rasterio.open(p) for p in paths]
    try:
        merge(
            srcs,
            nodata=EXPECTED_NODATA,
            dtype=EXPECTED_DTYPE,
            method="first",
            target_aligned_pixels=True,
            mem_limit=256,
            dst_path=out,
            dst_kwds=creation_options(),
        )
    finally:
        for src in srcs:
            src.close()

    if not out.exists() or out.stat().st_size == 0:
        die("Il mosaico non è stato creato correttamente.")


def compute_workarea_bounds(mosaic_path: Path, bbox_wgs84, buffer_km: float):
    with rasterio.open(mosaic_path) as ds:
        if not ds.crs:
            die("Mosaico senza CRS.")
        west, south, east, north = bbox_wgs84
        projected = transform_bounds(
            "EPSG:4326",
            ds.crs,
            west,
            south,
            east,
            north,
            densify_pts=21,
        )
        buf = buffer_km * 1000.0
        req = (
            projected[0] - buf,
            projected[1] - buf,
            projected[2] + buf,
            projected[3] + buf,
        )

        # Clamping alla copertura effettiva del mosaico.
        clamped = (
            max(req[0], ds.bounds.left),
            max(req[1], ds.bounds.bottom),
            min(req[2], ds.bounds.right),
            min(req[3], ds.bounds.top),
        )
        if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
            die("Work area non interseca il mosaico.")

        win = from_bounds(*clamped, transform=ds.transform)
        win = win.round_offsets().round_lengths()

        # Garantisce finestra interna al raster.
        full = rasterio.windows.Window(0, 0, ds.width, ds.height)
        win = win.intersection(full)

        actual_bounds = window_bounds(win, ds.transform)
        transform = ds.window_transform(win)

        return win, actual_bounds, transform, ds.crs


def build_workarea(
    mosaic_path: Path,
    out: Path,
    bbox_wgs84,
    buffer_km: float,
    overwrite: bool,
) -> Dict[str, Any]:
    win, actual_bounds, out_transform, crs = compute_workarea_bounds(
        mosaic_path, bbox_wgs84, buffer_km
    )

    if out.exists() and not overwrite:
        print(f"SKIP work area già presente: {out}")
    else:
        if out.exists():
            out.unlink()
        print(f"\nRitaglio work area: {out}")

        with rasterio.open(mosaic_path) as src:
            profile = src.profile.copy()
            profile.update(
                width=int(win.width),
                height=int(win.height),
                transform=out_transform,
                nodata=EXPECTED_NODATA,
                dtype=EXPECTED_DTYPE,
                **creation_options(),
            )

            with rasterio.open(out, "w", **profile) as dst:
                # Lettura/scrittura a blocchi della destinazione per limitare memoria.
                for _, dw in dst.block_windows(1):
                    # dw è relativo alla work area; trasformalo in finestra del mosaico.
                    sw = rasterio.windows.Window(
                        col_off=win.col_off + dw.col_off,
                        row_off=win.row_off + dw.row_off,
                        width=dw.width,
                        height=dw.height,
                    )
                    arr = src.read(1, window=sw)
                    dst.write(arr, 1, window=dw)

    with rasterio.open(out) as ds:
        valid_count = 0
        nodata_count = 0
        vmin = math.inf
        vmax = -math.inf

        for _, w in ds.block_windows(1):
            a = ds.read(1, window=w)
            valid = np.isfinite(a) & (a != ds.nodata)
            nv = int(valid.sum())
            valid_count += nv
            nodata_count += int(a.size - nv)
            if nv:
                vals = a[valid]
                vmin = min(vmin, float(vals.min()))
                vmax = max(vmax, float(vals.max()))

        return {
            "path": str(out),
            "crs": str(ds.crs),
            "epsg": ds.crs.to_epsg() if ds.crs else None,
            "width": ds.width,
            "height": ds.height,
            "res": list(map(float, ds.res)),
            "bounds": {
                "left": ds.bounds.left,
                "bottom": ds.bounds.bottom,
                "right": ds.bounds.right,
                "top": ds.bounds.top,
            },
            "buffer_km": buffer_km,
            "bbox_wgs84": list(map(float, bbox_wgs84)),
            "valid_cells": valid_count,
            "nodata_cells": nodata_count,
            "valid_fraction": valid_count / (valid_count + nodata_count),
            "elevation_min_m": None if vmin == math.inf else vmin,
            "elevation_max_m": None if vmax == -math.inf else vmax,
        }


def write_prepare_report(
    terrain_dir: Path,
    tiles,
    overlaps,
    errors,
    warnings,
    mosaic_path: Path,
    workarea_info,
):
    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")

    with rasterio.open(mosaic_path) as ds:
        mosaic_info = {
            "path": str(mosaic_path),
            "size_bytes": mosaic_path.stat().st_size,
            "crs": str(ds.crs),
            "epsg": ds.crs.to_epsg() if ds.crs else None,
            "width": ds.width,
            "height": ds.height,
            "res": list(map(float, ds.res)),
            "bounds": {
                "left": ds.bounds.left,
                "bottom": ds.bounds.bottom,
                "right": ds.bounds.right,
                "top": ds.bounds.top,
            },
            "nodata": ds.nodata,
            "dtype": ds.dtypes[0],
        }

    obj = {
        "version": VERSION,
        "stage": "PREPARE",
        "status": status,
        "tiles": tiles,
        "overlaps": overlaps,
        "mosaic": mosaic_info,
        "workarea": workarea_info,
        "errors": errors,
        "warnings": warnings,
        "merge_policy": "first valid pixel, alphabetical source order",
        "raw_modified": False,
    }

    jp = terrain_dir / "terrain_prepare_report_v1_1.json"
    tp = terrain_dir / "terrain_prepare_report_v1_1.txt"
    jp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "TANARO–ARROSCIA | TERRAIN PREPARE v1.1",
        "=" * 80,
        f"STATUS       : {status}",
        f"Tasselli     : {len(tiles)}",
        f"Overlap test : {len(overlaps)}",
        f"ERROR        : {len(errors)}",
        f"WARN         : {len(warnings)}",
        "",
        "MOSAIC:",
        f"  {mosaic_path}",
        f"  size raster: {mosaic_info['width']} x {mosaic_info['height']}",
        f"  CRS: {mosaic_info['crs']}",
        "",
        "WORK AREA:",
        f"  {workarea_info['path']}",
        f"  size raster: {workarea_info['width']} x {workarea_info['height']}",
        f"  valid_fraction: {workarea_info['valid_fraction']:.6f}",
        f"  elevazione: {workarea_info['elevation_min_m']} .. {workarea_info['elevation_max_m']} m",
        "",
    ]
    if errors:
        lines.append("ERRORI:")
        lines.extend(f"  - {x}" for x in errors)
        lines.append("")
    if warnings:
        lines.append("WARNING:")
        lines.extend(f"  - {x}" for x in warnings)
        lines.append("")

    tp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return status


def write_float_raster(path: Path, data: np.ndarray, profile: Dict[str, Any], nodata: float = FLOAT_NODATA):
    p = profile.copy()
    p.update(
        dtype="float32",
        count=1,
        nodata=nodata,
        **creation_options(),
    )
    arr = np.asarray(data, dtype=np.float32)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr, 1)


def run_hydrology(workarea: Path, terrain_dir: Path, overwrite: bool) -> str:
    try:
        import pyflwdir
    except Exception as exc:
        die(f"Impossibile importare pyflwdir: {exc}")

    outputs = {
        "filled": terrain_dir / "dem_hydroconditioned_10m.tif",
        "slope": terrain_dir / "slope.tif",
        "d8": terrain_dir / "flow_direction_d8.tif",
        "acc_cells": terrain_dir / "flow_accumulation_cells.tif",
        "acc_km2": terrain_dir / "flow_accumulation_km2.tif",
    }

    if not overwrite and all(p.exists() for p in outputs.values()):
        print("SKIP hydrology: tutti gli output esistono già.")
        return "SKIPPED_EXISTING"

    print("\nCaricamento work area in memoria...")
    with rasterio.open(workarea) as src:
        elev = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    if crs is None:
        die("Work area senza CRS.")
    if crs.is_geographic:
        die("La fase idrologica richiede CRS metrico; trovato CRS geografico.")

    valid = np.isfinite(elev) & (elev != nodata)
    if not np.any(valid):
        die("Work area senza celle DEM valide.")

    # In pyflwdir il default max_depth<0 riempie tutte le depressioni.
    print("Depression filling + D8...")
    filled, d8 = pyflwdir.dem.fill_depressions(
        elevtn=elev,
        outlets="edge",
        nodata=nodata,
        max_depth=-1.0,
        connectivity=8,
    )

    # Assicura nodata coerente nelle celle esterne.
    filled = np.asarray(filled, dtype=np.float32)
    d8 = np.asarray(d8, dtype=np.uint8)
    filled[~valid] = FLOAT_NODATA
    d8[~valid] = D8_NODATA

    print("Costruzione oggetto flow direction...")
    flw = pyflwdir.from_array(
        d8,
        ftype="d8",
        mask=valid,
        transform=transform,
        latlon=False,
        cache=True,
    )

    print("Calcolo slope...")
    # pyflwdir.dem.slope è compilata con Numba: con pyflwdir 0.5.12 +
    # Numba 0.61.2 su Python 3.13, rasterio.Affine non è tipizzabile in
    # nopython mode. La funzione accetta la trasformazione affine anche
    # come sequenza numerica; passiamo quindi una tupla di 9 float,
    # equivalente all'Affine di rasterio ma pienamente compatibile con Numba.
    transform_numba = tuple(float(x) for x in transform)
    slope = pyflwdir.dem.slope(
        filled,
        nodata=FLOAT_NODATA,
        latlon=False,
        transform=transform_numba,
    )
    slope = np.asarray(slope, dtype=np.float32)
    slope[~valid] = FLOAT_NODATA

    print("Calcolo flow accumulation [cells]...")
    acc_cells = np.asarray(flw.upstream_area(unit="cell"), dtype=np.float32)
    acc_cells[~valid] = FLOAT_NODATA

    print("Calcolo upstream area [km2]...")
    acc_km2 = np.asarray(flw.upstream_area(unit="km2"), dtype=np.float32)
    acc_km2[~valid] = FLOAT_NODATA

    # Scrittura output.
    for p in outputs.values():
        if p.exists() and overwrite:
            p.unlink()

    write_float_raster(outputs["filled"], filled, profile)
    write_float_raster(outputs["slope"], slope, profile)
    write_float_raster(outputs["acc_cells"], acc_cells, profile)
    write_float_raster(outputs["acc_km2"], acc_km2, profile)

    d8_profile = profile.copy()
    d8_profile.update(
        dtype="uint8",
        count=1,
        nodata=D8_NODATA,
        **creation_options(),
    )
    # predictor 3 non è appropriato a uint8: rimuovilo.
    d8_profile["predictor"] = 2
    with rasterio.open(outputs["d8"], "w", **d8_profile) as dst:
        dst.write(d8, 1)

    valid_filled = filled[valid]
    change = valid_filled - elev[valid]
    changed = np.abs(change) > 1e-6

    max_uparea = float(np.max(acc_km2[valid]))
    max_acc_cells = float(np.max(acc_cells[valid]))
    pits = len(flw.idxs_pit)

    report = {
        "version": VERSION,
        "stage": "HYDROLOGY",
        "status": "PASS",
        "workarea": str(workarea),
        "crs": str(crs),
        "shape": list(elev.shape),
        "valid_cells": int(valid.sum()),
        "conditioning": {
            "method": "pyflwdir.dem.fill_depressions",
            "outlets": "edge",
            "connectivity": 8,
            "max_depth": -1.0,
            "changed_cells": int(changed.sum()),
            "changed_fraction": float(changed.sum() / valid.sum()),
            "max_fill_depth_m": float(np.max(change[changed])) if np.any(changed) else 0.0,
            "mean_fill_depth_changed_m": float(np.mean(change[changed])) if np.any(changed) else 0.0,
        },
        "flow": {
            "ftype": "d8",
            "d8_nodata": D8_NODATA,
            "pit_or_outlet_count": pits,
            "max_accumulation_cells": max_acc_cells,
            "max_upstream_area_km2": max_uparea,
        },
        "outputs": {k: str(v) for k, v in outputs.items()},
        "raw_modified": False,
    }

    jp = terrain_dir / "terrain_hydrology_report_v1_1.json"
    tp = terrain_dir / "terrain_hydrology_report_v1_1.txt"
    jp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "TANARO–ARROSCIA | TERRAIN HYDROLOGY v1.1",
        "=" * 80,
        "STATUS                 : PASS",
        f"Shape                  : {elev.shape[1]} x {elev.shape[0]}",
        f"Valid cells            : {int(valid.sum())}",
        f"Conditioned cells      : {int(changed.sum())}",
        f"Conditioned fraction   : {changed.sum() / valid.sum():.8f}",
        f"Max fill depth [m]     : {report['conditioning']['max_fill_depth_m']:.6f}",
        f"Pit/outlet count       : {pits}",
        f"Max accumulation cells : {max_acc_cells:.0f}",
        f"Max upstream area km2  : {max_uparea:.6f}",
        "",
        "OUTPUT:",
    ] + [f"  {k}: {v}" for k, v in outputs.items()]

    tp.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nFASE HYDROLOGY COMPLETATA.")
    print(f"Report: {tp}")
    return "PASS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--hydrology-only", action="store_true")
    ap.add_argument("--buffer-km", type=float, default=DEFAULT_BUFFER_KM)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = args.project_root.expanduser().resolve()
    raw_dir = root / "tanaro_arroscia" / "static_geo" / "dem_tinitaly_10m"
    terrain_dir = root / "tanaro_arroscia" / "terrain"
    terrain_dir.mkdir(parents=True, exist_ok=True)

    mosaic = terrain_dir / "dem_tinitaly_mosaic_10m.tif"
    workarea = terrain_dir / "dem_tinitaly_workarea_10m.tif"

    do_prepare = not args.hydrology_only
    do_hydro = not args.prepare_only

    print("=" * 100)
    print("TANARO–ARROSCIA | TERRAIN BUILDER v1.1")
    print(f"Project root : {root}")
    print(f"Raw DEM      : {raw_dir}")
    print(f"Output       : {terrain_dir}")
    print(f"Buffer       : {args.buffer_km:.1f} km")
    print(f"Mode         : {'PREPARE ONLY' if args.prepare_only else ('HYDROLOGY ONLY' if args.hydrology_only else 'ALL')}")
    print("=" * 100)

    if do_prepare:
        paths = tif_tiles(raw_dir)
        print(f"\nTasselli trovati: {len(paths)}")
        for p in paths:
            print(f"  - {p}")

        tiles, errors, warnings = inspect_tiles(paths)
        overlaps, ov_errors, ov_warnings = overlap_audit(paths)
        errors.extend(ov_errors)
        warnings.extend(ov_warnings)

        print(f"\nQA preliminare: ERROR={len(errors)} WARN={len(warnings)}")
        for x in errors:
            print(f"ERROR: {x}")
        for x in warnings:
            print(f"WARN : {x}")

        if errors:
            # Scrive almeno una diagnostica essenziale e NON crea il mosaico.
            report = {
                "version": VERSION,
                "stage": "PREPARE_PRECHECK",
                "status": "FAIL",
                "tiles": tiles,
                "overlaps": overlaps,
                "errors": errors,
                "warnings": warnings,
            }
            p = terrain_dir / "terrain_prepare_report_v1_1.json"
            p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            die("Precheck DEM fallito: non costruisco il mosaico.", 2)

        build_mosaic(paths, mosaic, args.overwrite)

        workarea_info = build_workarea(
            mosaic,
            workarea,
            DEFAULT_WGS84_BBOX,
            args.buffer_km,
            args.overwrite,
        )

        status = write_prepare_report(
            terrain_dir,
            tiles,
            overlaps,
            errors,
            warnings,
            mosaic,
            workarea_info,
        )

        print("\n" + "=" * 100)
        print(f"PREPARE STATUS : {status}")
        print(f"Mosaico        : {mosaic}")
        print(f"Work area      : {workarea}")
        print(f"Report         : {terrain_dir / 'terrain_prepare_report_v1_1.txt'}")
        print("=" * 100)

        if status == "FAIL":
            return 2

    if do_hydro:
        if not workarea.exists():
            die(
                "Work area non presente. Eseguire prima --prepare-only oppure il programma senza flag.",
                2,
            )
        status = run_hydrology(workarea, terrain_dir, args.overwrite)
        return 0 if status in ("PASS", "SKIPPED_EXISTING") else 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
