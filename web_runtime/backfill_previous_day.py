#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

EXPECTED_RECEPTORS = 20
CORE_MONTHS = {9, 10, 11, 12}
CACHE_DIRNAME = "nw_operational_daily_feature_cache_v1_0"
CACHE_FILENAME = "operational_dynamic_daily_cache_v1_0.parquet"

FOLLOWUP_COMPONENTS = [
    "repair_nw_operational_raw_cache_surface_v1_1.py",
    "build_nw_operational_receptor_features_current_v1_1.py",
    "build_nw_operational_medsea_corridor_current_v1_1.py",
    "update_nw_operational_antecedent_cache_current_v1_0.py",
]


def cached_rows(cache_path: Path, target_date) -> int:
    if not cache_path.exists():
        return 0
    cache = pd.read_parquet(cache_path, columns=["receptor_id", "issue_date"])
    dates = pd.to_datetime(cache["issue_date"], errors="coerce").dt.date
    rows = cache.loc[dates.eq(target_date), "receptor_id"].astype(str)
    return int(rows.nunique())


def run_script(script: Path, runtime_root: Path, env: dict[str, str]) -> None:
    print(f"$ {sys.executable} {script}", flush=True)
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(runtime_root),
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Backfill component failed with return code {proc.returncode}: {script.name}"
        )


def run_raw_builder_for_date(runtime_root: Path, target_date) -> None:
    script = runtime_root / "build_nw_operational_raw_cache_current_v1_1.py"
    if not script.exists():
        raise RuntimeError(f"Raw-cache builder missing: {script}")

    spec = importlib.util.spec_from_file_location("nwfloodwatch_backfill_raw_builder", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import raw-cache builder: {script}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    issue_cycle = datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        0,
        0,
        tzinfo=timezone.utc,
    )

    # The frozen builder normally locks the latest 00Z cycle.  For a missing
    # immediately preceding day we override only that discovery function; all
    # retrieval, feature construction and cache formulas remain unchanged.
    module.find_latest_00z = lambda: ("ecmwf", issue_cycle)

    print(
        "BACKFILL RAW CACHE | "
        f"forcing exact historical issue cycle {issue_cycle.isoformat()} "
        "through the unchanged v1.0-rc1 retrieval/build logic",
        flush=True,
    )
    module.main()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Repair a missing immediately previous NW FloodWatch receptor-day "
            "before the current production run."
        )
    )
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument(
        "--target-date",
        help="Optional YYYY-MM-DD override; default is previous UTC day.",
    )
    args = parser.parse_args()

    runtime_root = Path(args.runtime_root).resolve()
    if not runtime_root.exists():
        raise RuntimeError(f"Runtime root missing: {runtime_root}")

    if args.target_date:
        target_date = datetime.strptime(args.target_date, "%Y-%m-%d").date()
    else:
        target_date = datetime.now(timezone.utc).date() - timedelta(days=1)

    if target_date.month not in CORE_MONTHS:
        print(
            f"ANTECEDENT BACKFILL SKIP | {target_date} is outside Sep-Dec CORE season.",
            flush=True,
        )
        return 0

    cache_path = runtime_root / CACHE_DIRNAME / CACHE_FILENAME
    present = cached_rows(cache_path, target_date)
    if present == EXPECTED_RECEPTORS:
        print(
            f"ANTECEDENT BACKFILL SKIP | {target_date} already has "
            f"{EXPECTED_RECEPTORS}/{EXPECTED_RECEPTORS} receptor rows.",
            flush=True,
        )
        return 0

    print(
        f"ANTECEDENT BACKFILL REQUIRED | {target_date} has "
        f"{present}/{EXPECTED_RECEPTORS} receptor rows.",
        flush=True,
    )
    print(
        "Policy: regenerate the missing day from its exact ECMWF 00Z issue cycle "
        "and issue-day CMEMS field; no interpolation, zero-fill or bridging is allowed.",
        flush=True,
    )

    run_raw_builder_for_date(runtime_root, target_date)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    for idx, name in enumerate(FOLLOWUP_COMPONENTS, start=1):
        script = runtime_root / name
        if not script.exists():
            raise RuntimeError(f"Backfill component missing: {script}")
        print(
            f"BACKFILL PHASE {idx}/{len(FOLLOWUP_COMPONENTS)} | {name}",
            flush=True,
        )
        run_script(script, runtime_root, env)

    repaired = cached_rows(cache_path, target_date)
    if repaired != EXPECTED_RECEPTORS:
        raise RuntimeError(
            f"Backfill verification failed for {target_date}: "
            f"{repaired}/{EXPECTED_RECEPTORS} receptor rows"
        )

    print(
        f"ANTECEDENT BACKFILL PASS | {target_date} restored with "
        f"{repaired}/{EXPECTED_RECEPTORS} receptor rows.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ANTECEDENT BACKFILL FAIL: {exc}", file=sys.stderr, flush=True)
        raise
