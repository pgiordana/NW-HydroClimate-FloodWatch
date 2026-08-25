#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
repair_nw_operational_raw_cache_surface_v1_1.py

FASE 14B — CORREZIONE MIRATA DEL RAW CACHE OPERATIVO.

Motivo:
nel run v1.0 ECMWF ha scritto:
    No index entries for param=cape
    Did you mean 'mucape' instead of 'cape'?
    No index entries for param=swvl1/swvl2/swvl3

Il file era comunque non vuoto perché conteneva gli altri parametri richiesti.
Quindi il semplice controllo "file non vuoto" aveva prodotto un PASS troppo
permissivo per CURRENT_DAY_SURFACE_STATE.

Correzione conforme all'attuale ECMWF Open Data:
- CAPE operativo candidato: mucape
  (Most-unstable CAPE; NON identico alla CAPE ERA5 del training)
- soil water operativo: vsw con livelli 1,2,3
  (Volumetric soil water; NON usare gli shortName ERA5 swvl1/swvl2/swvl3)

Lo script:
1) individua l'ultimo raw-cache run;
2) NON riscarica i ~277 MB già validi;
3) scarica separatamente:
   - msl, tcwv, sd
   - mucape
   - vsw level 1
   - vsw level 2
   - vsw level 3
4) produce manifest/audit v1.1;
5) conserva esplicitamente i mismatch semantici da validare.

NON costruisce feature e NON esegue il modello.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from ecmwf.opendata import Client


ECMWF_SOURCES = ["ecmwf", "aws", "google"]
DAY_STEPS = list(range(0, 25, 3))


def fmt_seconds(seconds):
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
        msg += f" | {str(current)[:140]}"
    print(msg.ljust(290), end="", flush=True)
    if done >= total:
        print(flush=True)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def short_error(exc):
    return f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')[:600]}"


def make_client(source):
    return Client(
        source=source,
        model="ifs",
        resol="0p25",
    )


def retrieve(target, issue_date, param, step, levelist=None):
    errors = []

    for source in ECMWF_SOURCES:
        try:
            client = make_client(source)

            kwargs = dict(
                stream="oper",
                type="fc",
                date=int(issue_date.strftime("%Y%m%d")),
                time=0,
                step=step,
                param=param,
                target=str(target),
            )

            if levelist is not None:
                kwargs["levelist"] = levelist

            result = client.retrieve(**kwargs)

            if not target.exists() or target.stat().st_size <= 0:
                raise RuntimeError("Output file missing/empty.")

            return {
                "status": "PASS",
                "source": source,
                "result_datetime": str(getattr(result, "datetime", "")),
                "error": "",
            }

        except Exception as exc:
            errors.append(f"{source}: {short_error(exc)}")
            if target.exists() and target.stat().st_size == 0:
                target.unlink(missing_ok=True)

    return {
        "status": "FAIL",
        "source": "",
        "result_datetime": "",
        "error": " | ".join(errors),
    }


def latest_raw_cache(root):
    cache_root = root / "nw_operational_raw_cache"

    if not cache_root.exists():
        raise SystemExit(f"Manca: {cache_root}")

    runs = sorted(
        [
            p for p in cache_root.iterdir()
            if p.is_dir() and p.name.endswith("T00Z")
        ],
        key=lambda p: p.name,
    )

    if not runs:
        raise SystemExit("Nessun raw-cache run trovato.")

    return runs[-1]


def parse_run_id(run_dir):
    return datetime.strptime(
        run_dir.name,
        "%Y%m%dT00Z",
    ).replace(tzinfo=timezone.utc)


def row(role, status, source, path, params, levels, semantics, error=""):
    ok = path.exists() and path.stat().st_size > 0
    return {
        "role": role,
        "status": status,
        "source": source,
        "parameters": params,
        "levels": levels,
        "steps_hours": ",".join(str(x) for x in DAY_STEPS),
        "semantics": semantics,
        "file": str(path),
        "bytes": path.stat().st_size if ok else 0,
        "sha256": sha256(path) if ok else "",
        "error": error,
    }


def main():
    root = Path(__file__).resolve().parent
    run_dir = latest_raw_cache(root)
    issue_cycle = parse_run_id(run_dir)
    issue_date = issue_cycle.date()

    ecmwf_dir = run_dir / "ecmwf"
    ecmwf_dir.mkdir(parents=True, exist_ok=True)

    audit_v1_p = run_dir / "raw_cache_audit_v1_0.json"

    if not audit_v1_p.exists():
        raise SystemExit(f"Manca audit v1.0: {audit_v1_p}")

    audit_v1 = json.loads(
        audit_v1_p.read_text(encoding="utf-8")
    )

    if not str(audit_v1.get("overall_status", "")).startswith("PASS_RAW_CACHE_CURRENT"):
        raise SystemExit(
            f"Raw cache v1.0 non in stato utilizzabile: {audit_v1.get('overall_status')}"
        )

    print("=" * 220)
    print("NW HYDROCLIMATE — RAW CACHE SURFACE REPAIR v1.1")
    print("=" * 220)
    print(f"Run da correggere : {run_dir.name}")
    print(f"Issue cycle       : {issue_cycle.isoformat()}")

    tasks = [
        {
            "role": "SURFACE_SAFE_MSL_TCWV_SD",
            "filename": f"ifs_{run_dir.name}_msl_tcwv_sd_steps0_24_v1_1.grib2",
            "param": ["msl", "tcwv", "sd"],
            "levelist": None,
            "params_text": "msl,tcwv,sd",
            "levels_text": "",
            "semantics": (
                "Operational IFS current-day proxy fields. "
                "These shortNames are directly available in Open Data."
            ),
        },
        {
            "role": "SURFACE_MUCAPE_PROXY",
            "filename": f"ifs_{run_dir.name}_mucape_steps0_24_v1_1.grib2",
            "param": ["mucape"],
            "levelist": None,
            "params_text": "mucape",
            "levels_text": "",
            "semantics": (
                "Most-unstable CAPE operational proxy. "
                "NOT semantically identical to ERA5 CAPE used in training; "
                "quantitative compatibility validation required."
            ),
        },
        {
            "role": "SOIL_VSW_LAYER1_PROXY",
            "filename": f"ifs_{run_dir.name}_vsw_layer1_steps0_24_v1_1.grib2",
            "param": ["vsw"],
            "levelist": [1],
            "params_text": "vsw",
            "levels_text": "1",
            "semantics": (
                "IFS volumetric soil water layer 1 proxy for training soil-water layer 1. "
                "Distribution/layer-definition compatibility must be validated."
            ),
        },
        {
            "role": "SOIL_VSW_LAYER2_PROXY",
            "filename": f"ifs_{run_dir.name}_vsw_layer2_steps0_24_v1_1.grib2",
            "param": ["vsw"],
            "levelist": [2],
            "params_text": "vsw",
            "levels_text": "2",
            "semantics": (
                "IFS volumetric soil water layer 2 proxy for training soil-water layer 2. "
                "Distribution/layer-definition compatibility must be validated."
            ),
        },
        {
            "role": "SOIL_VSW_LAYER3_PROXY",
            "filename": f"ifs_{run_dir.name}_vsw_layer3_steps0_24_v1_1.grib2",
            "param": ["vsw"],
            "levelist": [3],
            "params_text": "vsw",
            "levels_text": "3",
            "semantics": (
                "IFS volumetric soil water layer 3 proxy for training soil-water layer 3. "
                "Distribution/layer-definition compatibility must be validated."
            ),
        },
    ]

    print("\nPHASE 1/3 — download corrected field families separately")
    start = time.time()

    rows = []

    for i, t in enumerate(tasks, 1):
        target = ecmwf_dir / t["filename"]

        rr = retrieve(
            target=target,
            issue_date=issue_date,
            param=t["param"],
            step=DAY_STEPS,
            levelist=t["levelist"],
        )

        rows.append(
            row(
                role=t["role"],
                status=rr["status"],
                source=rr["source"],
                path=target,
                params=t["params_text"],
                levels=t["levels_text"],
                semantics=t["semantics"],
                error=rr["error"],
            )
        )

        progress(
            "PHASE 1/3",
            i,
            len(tasks),
            start,
            f"{t['role']} | {rr['status']}",
        )

    repair = pd.DataFrame(rows)

    print("\nPHASE 2/3 — strict family-level completeness audit")
    start = time.time()

    required_roles = {x["role"] for x in tasks}
    passing_roles = set(
        repair.loc[repair["status"].eq("PASS"), "role"].astype(str)
    )

    missing_roles = sorted(required_roles - passing_roles)

    strict_pass = len(missing_roles) == 0

    progress(
        "PHASE 2/3",
        1,
        1,
        start,
        (
            f"strict_pass={strict_pass} | "
            f"families={len(passing_roles)}/{len(required_roles)}"
        ),
    )

    print("\nPHASE 3/3 — write corrected manifest and methodological audit")
    start = time.time()

    old_manifest_p = run_dir / "raw_cache_manifest_v1_0.csv"
    old_manifest = pd.read_csv(old_manifest_p, low_memory=False)

    # We keep the old manifest for provenance but explicitly supersede
    # its permissive CURRENT_DAY_SURFACE_STATE interpretation.
    old_manifest["manifest_version"] = "v1.0_PROVENANCE"
    old_manifest["superseded_note"] = ""

    mask_old_surface = old_manifest["role"].astype(str).eq(
        "CURRENT_DAY_SURFACE_STATE"
    )

    old_manifest.loc[
        mask_old_surface,
        "superseded_note",
    ] = (
        "SUPERSEDED_FOR_CAPE_SOIL_AUDIT: request was partially fulfilled; "
        "cape/swvl1/swvl2/swvl3 were not actually retrieved."
    )

    repair_out = repair.copy()
    repair_out["provider"] = "ECMWF_OPEN_DATA_IFS"
    repair_out["issue_cycle_utc"] = issue_cycle.isoformat()
    repair_out["valid_date_utc"] = issue_date.isoformat()
    repair_out["manifest_version"] = "v1.1_CORRECTION"
    repair_out["superseded_note"] = ""

    # Align columns.
    all_cols = list(dict.fromkeys(
        list(old_manifest.columns) + list(repair_out.columns)
    ))

    corrected_manifest = pd.concat(
        [
            old_manifest.reindex(columns=all_cols),
            repair_out.reindex(columns=all_cols),
        ],
        ignore_index=True,
    )

    manifest_p = run_dir / "raw_cache_manifest_v1_1.csv"
    repair_registry_p = run_dir / "raw_cache_surface_repair_registry_v1_1.csv"
    audit_json_p = run_dir / "raw_cache_audit_v1_1.json"
    audit_txt_p = run_dir / "raw_cache_audit_v1_1.txt"

    corrected_manifest.to_csv(manifest_p, index=False)
    repair.to_csv(repair_registry_p, index=False)

    if strict_pass:
        overall = "PASS_RAW_CACHE_SURFACE_REPAIRED_V1_1__FEATURE_ENGINE_READY"
    else:
        overall = "FAIL_RAW_CACHE_SURFACE_REPAIR_INCOMPLETE"

    audit = {
        "version": "1.1",
        "overall_status": overall,
        "run_id": run_dir.name,
        "issue_cycle_utc": issue_cycle.isoformat(),
        "v1_0_surface_pass_was_too_permissive": True,
        "reason": (
            "Grouped ECMWF retrieve produced a non-empty file while silently "
            "omitting unsupported shortNames cape/swvl1/swvl2/swvl3."
        ),
        "corrected_cape_parameter": "mucape",
        "corrected_soil_parameter": "vsw",
        "corrected_soil_levels": [1, 2, 3],
        "strict_family_pass": strict_pass,
        "missing_corrected_families": missing_roles,
        "model_prediction_performed": False,
        "feature_engine_performed": False,
        "semantic_equivalence_claimed": False,
        "next_step": (
            "Build receptor-level feature engine from v1.1 raw cache. "
            "Treat mucape and vsw features as operational proxies requiring "
            "distribution compatibility assessment versus historical ERA5."
        ),
    }

    audit_json_p.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "=" * 220,
        "NW HYDROCLIMATE — RAW CACHE SURFACE REPAIR v1.1",
        "=" * 220,
        f"OVERALL STATUS               : {overall}",
        f"Run ID                       : {run_dir.name}",
        f"Strict corrected families    : {len(passing_roles)}/{len(required_roles)}",
        f"Missing corrected families   : {missing_roles}",
        "Old grouped surface PASS     : SUPERSEDED FOR CAPE/SOIL",
        "Correct CAPE candidate       : mucape",
        "Correct soil-water parameter : vsw levels 1,2,3",
        "Semantic equivalence claimed : False",
        "",
        "CORRECTED FIELD FAMILIES",
        repair[
            [
                "role",
                "status",
                "source",
                "parameters",
                "levels",
                "steps_hours",
                "bytes",
                "error",
            ]
        ].to_string(index=False),
        "",
        "IMPORTANT",
        "mucape is not the same variable as the ERA5 CAPE predictor used in training.",
        "vsw layers are operational IFS soil-water proxies and must be distribution-checked against ERA5 training features.",
        "The old v1.0 raw files remain untouched for provenance.",
        "No model prediction has been made.",
        "",
        "NEXT STEP",
        "Proceed to receptor-level feature engine only if the status is PASS.",
        "",
        f"Corrected manifest : {manifest_p}",
        f"Repair registry    : {repair_registry_p}",
        f"Audit              : {audit_json_p}",
    ]

    audit_txt_p.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    progress(
        "PHASE 3/3",
        1,
        1,
        start,
        f"status={overall}",
    )

    print("\n" + "=" * 220)
    print("\n".join(lines[3:]))
    print("=" * 220)
    print(f"OVERALL STATUS : {overall}")
    print(f"Output         : {run_dir}")
    print("=" * 220)


if __name__ == "__main__":
    main()
