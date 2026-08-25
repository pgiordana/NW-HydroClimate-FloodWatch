#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_tanaro_arroscia_hydrology_maxdepth_v1_0.py

Sensitivity run for Tanaro–Arroscia hydrologic conditioning.

Purpose
-------
The canonical full-fill run (max_depth < 0) produced a deep internal fill
(~72.85 m) in a compact high-altitude depression. This script DOES NOT
replace or overwrite that run. It produces a parallel D8 routing scenario
where depressions deeper than a chosen threshold are preserved as pits.

Default threshold: 25 m.

According to pyflwdir.dem.fill_depressions, depressions whose pour-point
depth exceeds max_depth are set as pits rather than completely filled.

Inputs
------
tanaro_arroscia/terrain/dem_tinitaly_workarea_10m.tif

Optional comparison inputs from canonical full-fill run:
tanaro_arroscia/terrain/flow_direction_d8.tif
tanaro_arroscia/terrain/flow_accumulation_km2.tif

Outputs (for default 25 m)
--------------------------
tanaro_arroscia/terrain/sensitivity_maxdepth25m/
    dem_hydroconditioned_maxdepth25m.tif
    flow_direction_d8_maxdepth25m.tif
    flow_accumulation_cells_maxdepth25m.tif
    flow_accumulation_km2_maxdepth25m.tif
    hydrology_maxdepth25m_report_v1_0.json
    hydrology_maxdepth25m_report_v1_0.txt

Raw/source DEM files are never modified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
import pyflwdir


VERSION = "1.0"
D8_NODATA = 247
FLOAT_NODATA = -9999.0


def die(msg: str, code: int = 2):
    print(f"ERRORE: {msg}", file=sys.stderr)
    raise SystemExit(code)


def tag_depth(depth: float) -> str:
    if float(depth).is_integer():
        return f"{int(depth)}m"
    return f"{depth:g}m".replace(".", "p")


def creation_float():
    return dict(
        driver="GTiff",
        compress="DEFLATE",
        predictor=3,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER",
    )


def creation_uint8():
    return dict(
        driver="GTiff",
        compress="DEFLATE",
        predictor=2,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER",
    )


def write_float(path: Path, arr: np.ndarray, profile: dict):
    p = profile.copy()
    p.update(dtype="float32", count=1, nodata=FLOAT_NODATA, **creation_float())
    with rasterio.open(path, "w", **p) as dst:
        dst.write(np.asarray(arr, dtype=np.float32), 1)


def write_d8(path: Path, arr: np.ndarray, profile: dict):
    p = profile.copy()
    p.update(dtype="uint8", count=1, nodata=D8_NODATA, **creation_uint8())
    with rasterio.open(path, "w", **p) as dst:
        dst.write(np.asarray(arr, dtype=np.uint8), 1)


def compare_raster_same_grid(path: Path, arr: np.ndarray, valid: np.ndarray, nodata=None):
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        if src.shape != arr.shape:
            return {"error": f"shape mismatch {src.shape} vs {arr.shape}"}
        ref = src.read(1)
        ref_valid = np.isfinite(ref)
        if src.nodata is not None:
            ref_valid &= ref != src.nodata
        m = valid & ref_valid
        if not np.any(m):
            return {"error": "no common valid cells"}
        d = np.asarray(arr[m], dtype=np.float64) - np.asarray(ref[m], dtype=np.float64)
        return {
            "common_valid_cells": int(m.sum()),
            "changed_cells_abs_gt_1e-6": int(np.sum(np.abs(d) > 1e-6)),
            "changed_fraction": float(np.sum(np.abs(d) > 1e-6) / m.sum()),
            "mean_difference": float(np.mean(d)),
            "mean_absolute_difference": float(np.mean(np.abs(d))),
            "max_absolute_difference": float(np.max(np.abs(d))),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-depth", type=float, default=25.0)
    ap.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.max_depth <= 0:
        die("--max-depth deve essere > 0 per questa sensitivity run.")

    root = args.project_root.expanduser().resolve()
    terrain = root / "tanaro_arroscia" / "terrain"
    work = terrain / "dem_tinitaly_workarea_10m.tif"

    if not work.exists():
        die(f"DEM work area mancante: {work}")

    tag = tag_depth(args.max_depth)
    out = terrain / f"sensitivity_maxdepth{tag}"
    out.mkdir(parents=True, exist_ok=True)

    filled_p = out / f"dem_hydroconditioned_maxdepth{tag}.tif"
    d8_p = out / f"flow_direction_d8_maxdepth{tag}.tif"
    acc_cells_p = out / f"flow_accumulation_cells_maxdepth{tag}.tif"
    acc_km2_p = out / f"flow_accumulation_km2_maxdepth{tag}.tif"
    json_p = out / f"hydrology_maxdepth{tag}_report_v1_0.json"
    txt_p = out / f"hydrology_maxdepth{tag}_report_v1_0.txt"

    outputs = [filled_p, d8_p, acc_cells_p, acc_km2_p]
    if all(p.exists() for p in outputs) and not args.overwrite:
        print("Tutti gli output esistono già. Usa --overwrite per rigenerarli.")
        print(f"Report atteso: {txt_p}")
        return 0

    print("=" * 96)
    print("TANARO–ARROSCIA | LIMITED-DEPTH HYDROLOGY SENSITIVITY v1.0")
    print(f"Project root : {root}")
    print(f"Work DEM     : {work}")
    print(f"max_depth    : {args.max_depth:.3f} m")
    print(f"Output       : {out}")
    print("=" * 96)

    print("\nCaricamento DEM...")
    with rasterio.open(work) as src:
        elev = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    valid = np.isfinite(elev)
    if nodata is not None:
        valid &= elev != nodata
    if not np.any(valid):
        die("Nessuna cella DEM valida.")
    if crs is None or crs.is_geographic:
        die("Serve un CRS metrico valido.")

    print("Conditioning limitato + D8...")
    filled, d8 = pyflwdir.dem.fill_depressions(
        elevtn=elev,
        outlets="edge",
        nodata=nodata,
        max_depth=float(args.max_depth),
        connectivity=8,
    )
    filled = np.asarray(filled, dtype=np.float32)
    d8 = np.asarray(d8, dtype=np.uint8)
    filled[~valid] = FLOAT_NODATA
    d8[~valid] = D8_NODATA

    print("Costruzione flow-direction object...")
    flw = pyflwdir.from_array(
        d8,
        ftype="d8",
        mask=valid,
        transform=transform,
        latlon=False,
        cache=True,
    )

    print("Accumulo [cells]...")
    acc_cells = np.asarray(flw.upstream_area(unit="cell"), dtype=np.float32)
    acc_cells[~valid] = FLOAT_NODATA

    print("Accumulo [km2]...")
    acc_km2 = np.asarray(flw.upstream_area(unit="km2"), dtype=np.float32)
    acc_km2[~valid] = FLOAT_NODATA

    delta_fill = filled[valid] - elev[valid]
    changed = delta_fill > 1e-6
    pits = int(len(flw.idxs_pit))

    print("Confronto con scenario full-fill esistente...")
    full_d8_p = terrain / "flow_direction_d8.tif"
    full_acc_p = terrain / "flow_accumulation_km2.tif"

    # D8 comparison: exact categorical difference, computed separately.
    d8_compare = None
    if full_d8_p.exists():
        with rasterio.open(full_d8_p) as src:
            ref = src.read(1)
            m = valid.copy()
            if src.nodata is not None:
                m &= ref != src.nodata
            d8_diff = m & (ref != d8)
            d8_compare = {
                "common_valid_cells": int(m.sum()),
                "changed_d8_cells": int(d8_diff.sum()),
                "changed_d8_fraction": float(d8_diff.sum() / m.sum()),
            }

    acc_compare = compare_raster_same_grid(full_acc_p, acc_km2, valid)

    for p in outputs:
        if p.exists() and args.overwrite:
            p.unlink()

    print("Scrittura raster...")
    write_float(filled_p, filled, profile)
    write_d8(d8_p, d8, profile)
    write_float(acc_cells_p, acc_cells, profile)
    write_float(acc_km2_p, acc_km2, profile)

    report = {
        "version": VERSION,
        "status": "PASS",
        "scenario": {
            "max_depth_m": float(args.max_depth),
            "outlets": "edge",
            "connectivity": 8,
            "interpretation": (
                "depressions deeper than max_depth are retained as pits "
                "instead of being fully filled"
            ),
        },
        "grid": {
            "crs": str(crs),
            "shape": [int(elev.shape[0]), int(elev.shape[1])],
            "valid_cells": int(valid.sum()),
            "cell_area_m2": float(abs(transform.a * transform.e)),
        },
        "conditioning": {
            "changed_cells": int(changed.sum()),
            "changed_fraction": float(changed.sum() / valid.sum()),
            "max_actual_fill_m": float(np.max(delta_fill[changed])) if np.any(changed) else 0.0,
            "mean_actual_fill_changed_m": float(np.mean(delta_fill[changed])) if np.any(changed) else 0.0,
            "pit_or_outlet_count": pits,
        },
        "routing": {
            "max_accumulation_cells": float(np.max(acc_cells[valid])),
            "max_upstream_area_km2": float(np.max(acc_km2[valid])),
        },
        "comparison_to_fullfill": {
            "fullfill_d8_path": str(full_d8_p) if full_d8_p.exists() else None,
            "fullfill_acc_km2_path": str(full_acc_p) if full_acc_p.exists() else None,
            "d8": d8_compare,
            "accumulation_km2": acc_compare,
        },
        "outputs": {
            "filled": str(filled_p),
            "d8": str(d8_p),
            "acc_cells": str(acc_cells_p),
            "acc_km2": str(acc_km2_p),
        },
        "raw_modified": False,
    }

    json_p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "TANARO–ARROSCIA | LIMITED-DEPTH HYDROLOGY SENSITIVITY v1.0",
        "=" * 96,
        "STATUS                      : PASS",
        f"max_depth [m]               : {args.max_depth:.3f}",
        f"Valid cells                 : {int(valid.sum())}",
        f"Conditioned cells           : {int(changed.sum())}",
        f"Conditioned fraction        : {changed.sum()/valid.sum():.8f}",
        f"Max actual fill [m]         : {report['conditioning']['max_actual_fill_m']:.6f}",
        f"Pit/outlet count            : {pits}",
        f"Max accumulation cells      : {report['routing']['max_accumulation_cells']:.0f}",
        f"Max upstream area [km2]     : {report['routing']['max_upstream_area_km2']:.6f}",
        "",
        "CONFRONTO CON FULL-FILL:",
    ]
    if d8_compare is not None:
        lines += [
            f"  D8 changed cells          : {d8_compare['changed_d8_cells']}",
            f"  D8 changed fraction       : {d8_compare['changed_d8_fraction']:.8f}",
        ]
    else:
        lines.append("  D8 comparison             : non disponibile")

    if acc_compare is not None and "error" not in acc_compare:
        lines += [
            f"  Acc changed cells         : {acc_compare['changed_cells_abs_gt_1e-6']}",
            f"  Acc changed fraction      : {acc_compare['changed_fraction']:.8f}",
            f"  Acc mean abs diff [km2]   : {acc_compare['mean_absolute_difference']:.6f}",
            f"  Acc max abs diff [km2]    : {acc_compare['max_absolute_difference']:.6f}",
        ]
    else:
        lines.append(f"  Acc comparison            : {acc_compare}")

    lines += [
        "",
        "OUTPUT:",
        f"  {filled_p}",
        f"  {d8_p}",
        f"  {acc_cells_p}",
        f"  {acc_km2_p}",
        f"  {json_p}",
    ]

    txt_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nReport: {txt_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
