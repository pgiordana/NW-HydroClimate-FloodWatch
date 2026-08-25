#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_tanaro_arroscia_hydroconditioning_v1_0.py

Diagnostica mirata del conditioning idrologico TINITALY Tanaro–Arroscia.

Obiettivo:
- verificare che il max fill depth (~72.8 m) sia un'anomalia locale e non
  un artefatto esteso;
- quantificare celle e cluster sopra soglie 1, 5, 10, 25, 50 m;
- distinguere cluster vicini al bordo/nodata da cluster interni;
- produrre CSV/JSON/TXT senza modificare alcun raster.

Input attesi:
tanaro_arroscia/terrain/dem_tinitaly_workarea_10m.tif
tanaro_arroscia/terrain/dem_hydroconditioned_10m.tif

Output:
tanaro_arroscia/terrain/audit_hydroconditioning_v1_0/
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage


THRESHOLDS = [0.01, 1.0, 5.0, 10.0, 25.0, 50.0]
CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)
EDGE_BUFFER_CELLS = 10  # 100 m a 10 m
NODATA_BUFFER_CELLS = 10


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    root = Path(__file__).resolve().parent
    terrain = root / "tanaro_arroscia" / "terrain"
    orig_p = terrain / "dem_tinitaly_workarea_10m.tif"
    fill_p = terrain / "dem_hydroconditioned_10m.tif"
    out = terrain / "audit_hydroconditioning_v1_0"
    out.mkdir(parents=True, exist_ok=True)

    if not orig_p.exists() or not fill_p.exists():
        raise SystemExit("Input mancanti: servono workarea e dem_hydroconditioned.")

    with rasterio.open(orig_p) as a, rasterio.open(fill_p) as b:
        if a.shape != b.shape or a.transform != b.transform or a.crs != b.crs:
            raise SystemExit("Raster non allineati.")

        orig = a.read(1).astype(np.float32)
        filled = b.read(1).astype(np.float32)
        transform = a.transform
        nodata_a = a.nodata
        nodata_b = b.nodata
        crs = a.crs

    valid = np.isfinite(orig) & np.isfinite(filled)
    if nodata_a is not None:
        valid &= orig != nodata_a
    if nodata_b is not None:
        valid &= filled != nodata_b

    depth = np.full(orig.shape, np.nan, dtype=np.float32)
    depth[valid] = filled[valid] - orig[valid]

    # Tolleriamo rumore numerico molto piccolo.
    neg = valid & (depth < -1e-4)
    positive = valid & (depth > 1e-6)

    rows, cols = orig.shape

    edge_mask = np.zeros(orig.shape, dtype=bool)
    n = EDGE_BUFFER_CELLS
    edge_mask[:n, :] = True
    edge_mask[-n:, :] = True
    edge_mask[:, :n] = True
    edge_mask[:, -n:] = True

    nodata = ~valid
    if np.any(nodata):
        nodata_near = ndimage.binary_dilation(nodata, iterations=NODATA_BUFFER_CELLS)
    else:
        nodata_near = np.zeros_like(valid)

    summary = {
        "crs": str(crs),
        "shape": [rows, cols],
        "valid_cells": int(valid.sum()),
        "negative_fill_cells": int(neg.sum()),
        "positive_fill_cells": int(positive.sum()),
        "positive_fill_fraction": float(positive.sum() / valid.sum()),
        "max_fill_depth_m": float(np.nanmax(depth)),
        "mean_positive_fill_m": float(np.nanmean(depth[positive])) if np.any(positive) else 0.0,
        "median_positive_fill_m": float(np.nanmedian(depth[positive])) if np.any(positive) else 0.0,
        "p95_positive_fill_m": float(np.nanpercentile(depth[positive], 95)) if np.any(positive) else 0.0,
        "p99_positive_fill_m": float(np.nanpercentile(depth[positive], 99)) if np.any(positive) else 0.0,
        "thresholds": {},
    }

    cluster_rows = []

    for th in THRESHOLDS:
        mask = valid & (depth >= th)
        labels, nlab = ndimage.label(mask, structure=CONNECTIVITY)

        objs = ndimage.find_objects(labels)
        clusters = []
        for lab, slc in enumerate(objs, start=1):
            if slc is None:
                continue
            local = labels[slc] == lab
            count = int(local.sum())
            if count == 0:
                continue

            rr, cc = np.where(labels == lab)
            dvals = depth[rr, cc]
            r0, r1 = int(rr.min()), int(rr.max())
            c0, c1 = int(cc.min()), int(cc.max())

            x0, y0 = transform * (c0, r1 + 1)
            x1, y1 = transform * (c1 + 1, r0)

            near_edge = bool(np.any(edge_mask[rr, cc]))
            near_nodata = bool(np.any(nodata_near[rr, cc]))

            rec = {
                "threshold_m": th,
                "cluster_id": lab,
                "cells": count,
                "area_km2": count * 100.0 / 1e6,
                "max_fill_m": float(np.max(dvals)),
                "mean_fill_m": float(np.mean(dvals)),
                "row_min": r0,
                "row_max": r1,
                "col_min": c0,
                "col_max": c1,
                "bbox_xmin": float(min(x0, x1)),
                "bbox_ymin": float(min(y0, y1)),
                "bbox_xmax": float(max(x0, x1)),
                "bbox_ymax": float(max(y0, y1)),
                "near_edge_100m": near_edge,
                "near_nodata_100m": near_nodata,
            }
            clusters.append(rec)
            cluster_rows.append(rec)

        clusters_sorted = sorted(clusters, key=lambda r: (-r["max_fill_m"], -r["cells"]))
        summary["thresholds"][str(th)] = {
            "cells": int(mask.sum()),
            "fraction_valid": float(mask.sum() / valid.sum()),
            "clusters": int(nlab),
            "largest_cluster_cells": max((r["cells"] for r in clusters), default=0),
            "max_fill_m": max((r["max_fill_m"] for r in clusters), default=0.0),
            "top5_clusters": clusters_sorted[:5],
        }

    # Top 50 singole celle per fill depth.
    valid_idx = np.flatnonzero(valid)
    dflat = depth.ravel()
    order = valid_idx[np.argsort(dflat[valid_idx])[-50:]][::-1]
    top_cells = []
    for k, idx in enumerate(order, 1):
        r, c = np.unravel_index(idx, depth.shape)
        x, y = transform * (c + 0.5, r + 0.5)
        top_cells.append({
            "rank": k,
            "row": int(r),
            "col": int(c),
            "x": float(x),
            "y": float(y),
            "fill_depth_m": float(depth[r, c]),
            "near_edge_100m": bool(edge_mask[r, c]),
            "near_nodata_100m": bool(nodata_near[r, c]),
            "original_elev_m": float(orig[r, c]),
            "filled_elev_m": float(filled[r, c]),
        })

    write_csv(
        out / "fill_clusters_v1_0.csv",
        sorted(cluster_rows, key=lambda r: (r["threshold_m"], -r["max_fill_m"], -r["cells"])),
        [
            "threshold_m","cluster_id","cells","area_km2","max_fill_m","mean_fill_m",
            "row_min","row_max","col_min","col_max",
            "bbox_xmin","bbox_ymin","bbox_xmax","bbox_ymax",
            "near_edge_100m","near_nodata_100m"
        ]
    )
    write_csv(
        out / "top_fill_cells_v1_0.csv",
        top_cells,
        [
            "rank","row","col","x","y","fill_depth_m",
            "near_edge_100m","near_nodata_100m",
            "original_elev_m","filled_elev_m"
        ]
    )

    (out / "hydroconditioning_audit_v1_0.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Valutazione prudente.
    reasons = []
    if summary["negative_fill_cells"] > 0:
        reasons.append("Sono presenti celle con abbassamento > 1e-4 m.")
    n25 = summary["thresholds"]["25.0"]["cells"]
    n50 = summary["thresholds"]["50.0"]["cells"]
    if n25 > 5000:
        reasons.append(f"Molte celle con fill >=25 m: {n25}.")
    if n50 > 1000:
        reasons.append(f"Molte celle con fill >=50 m: {n50}.")

    status = "PASS_FOR_REVIEW" if not reasons else "REVIEW_REQUIRED"
    summary["status"] = status
    summary["review_reasons"] = reasons

    lines = [
        "TANARO–ARROSCIA | HYDROCONDITIONING AUDIT v1.0",
        "=" * 84,
        f"STATUS                  : {status}",
        f"Valid cells             : {summary['valid_cells']}",
        f"Positive fill cells     : {summary['positive_fill_cells']}",
        f"Positive fill fraction  : {summary['positive_fill_fraction']:.8f}",
        f"Negative fill cells     : {summary['negative_fill_cells']}",
        f"Max fill depth [m]      : {summary['max_fill_depth_m']:.6f}",
        f"Mean positive fill [m]  : {summary['mean_positive_fill_m']:.6f}",
        f"Median positive fill [m]: {summary['median_positive_fill_m']:.6f}",
        f"P95 positive fill [m]   : {summary['p95_positive_fill_m']:.6f}",
        f"P99 positive fill [m]   : {summary['p99_positive_fill_m']:.6f}",
        "",
        "SOGLIE:",
    ]
    for th in THRESHOLDS:
        s = summary["thresholds"][str(th)]
        lines.append(
            f"  >= {th:5.2f} m : cells={s['cells']:8d} | clusters={s['clusters']:6d} | "
            f"largest={s['largest_cluster_cells']:7d} | max={s['max_fill_m']:.3f} m"
        )

    lines += ["", "TOP 10 CELLE:"]
    for r in top_cells[:10]:
        lines.append(
            f"  #{r['rank']:02d} fill={r['fill_depth_m']:.3f} m "
            f"x={r['x']:.1f} y={r['y']:.1f} "
            f"edge={r['near_edge_100m']} nodata={r['near_nodata_100m']}"
        )

    if reasons:
        lines += ["", "REVIEW REASONS:"] + [f"  - {x}" for x in reasons]

    (out / "hydroconditioning_audit_v1_0.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    # riscrive JSON con status finale
    (out / "hydroconditioning_audit_v1_0.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n".join(lines))
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()
