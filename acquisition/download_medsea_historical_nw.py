#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, shutil, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "medsea_historical_nw"
DAILY_DIR = OUT / "daily_sst"
MONTHLY_DIR = OUT / "monthly_temp_0_110m"
STATIC_DIR = OUT / "static"
MANIFEST = OUT / "download_manifest.jsonl"

START_YEAR, END_YEAR = 1987, 2025

DATASET_DAILY_TEMP = "cmems_mod_med_phy-temp_my_4.2km_P1D-m"
DATASET_MONTHLY_TEMP = "cmems_mod_med_phy-temp_my_4.2km_P1M-m"
DATASET_STATIC = "cmems_mod_med_phy_my_4.2km_static"

WEST, EAST, SOUTH, NORTH = -4.0, 13.0, 36.0, 45.97
SST_MIN_DEPTH, SST_MAX_DEPTH = 0.0, 2.0
OHC_MIN_DEPTH, OHC_MAX_DEPTH = 0.0, 110.0

MIN_VALID_BYTES = 10_000
MIN_FREE_GB = 10.0
MAX_ATTEMPTS = 5

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=START_YEAR)
    p.add_argument("--end-year", type=int, default=END_YEAR)
    p.add_argument("--test", action="store_true",
                   help="Scarica solo ottobre 2000, più gli statici se mancanti.")
    p.add_argument("--only-family",
                   choices=["daily_sst","monthly_3d","static"])
    return p.parse_args()

def ensure_tool():
    exe = shutil.which("copernicusmarine")
    if not exe:
        raise RuntimeError(
            "Comando 'copernicusmarine' non trovato. Attiva lo stesso .venv del progetto."
        )
    return exe

def check_disk():
    OUT.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(OUT).free / 1024**3
    print(f"Spazio libero sul disco: {free_gb:.1f} GB")
    if free_gb < MIN_FREE_GB:
        raise RuntimeError(f"Spazio libero insufficiente: {free_gb:.1f} GB")

def file_ok(path):
    return path.exists() and path.stat().st_size >= MIN_VALID_BYTES

def manifest(**rec):
    rec["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    with MANIFEST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def run_cmd(cmd, target, family, label):
    target.parent.mkdir(parents=True, exist_ok=True)
    if file_ok(target):
        print(f"  SKIP {family}: {target.name} ({target.stat().st_size/1e6:.1f} MB)")
        return
    if target.exists():
        target.unlink()

    last = None
    for attempt in range(1, MAX_ATTEMPTS+1):
        t0 = time.time()
        try:
            print(f"  {family}: tentativo {attempt}/{MAX_ATTEMPTS} -> {target.name}")
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                raise RuntimeError(f"copernicusmarine exit code {proc.returncode}")
            if not file_ok(target):
                raise RuntimeError(f"File assente o troppo piccolo: {target}")
            elapsed = time.time()-t0
            print(f"  OK {family}: {target.stat().st_size/1e6:.1f} MB in {elapsed/60:.1f} min")
            manifest(status="ok", family=family, label=label,
                     path=str(target), bytes=target.stat().st_size,
                     seconds=round(elapsed,1))
            return
        except KeyboardInterrupt:
            print("\nInterrotto dall'utente. I file già completati restano validi.")
            raise
        except Exception as exc:
            last = exc
            print(f"  ERRORE {family}: {exc}")
            manifest(status="error", family=family, label=label,
                     path=str(target), error=str(exc))
            if target.exists() and target.stat().st_size < MIN_VALID_BYTES:
                target.unlink()
            if attempt < MAX_ATTEMPTS:
                wait = min(60 * 2**(attempt-1), 300)
                print(f"  Attendo {wait} s e riprovo...")
                time.sleep(wait)
    raise RuntimeError(f"Fallito {family} {label}: {last}")

def common_subset(exe, dataset, out_dir, out_name):
    return [
        exe, "subset",
        "--dataset-id", dataset,
        "--variable", "thetao",
        "--minimum-longitude", str(WEST),
        "--maximum-longitude", str(EAST),
        "--minimum-latitude", str(SOUTH),
        "--maximum-latitude", str(NORTH),
        "--output-directory", str(out_dir),
        "--output-filename", out_name,
    ]

def download_static(exe):
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    bathy = STATIC_DIR / "medsea_my_bathy_mask_source_domain.nc"
    cmd_bathy = [
        exe, "subset",
        "--dataset-id", DATASET_STATIC,
        "--dataset-part", "bathy",
        "--variable", "deptho",
        "--variable", "deptho_lev",
        "--variable", "mask",
        "--minimum-longitude", str(WEST),
        "--maximum-longitude", str(EAST),
        "--minimum-latitude", str(SOUTH),
        "--maximum-latitude", str(NORTH),
        "--output-directory", str(STATIC_DIR),
        "--output-filename", bathy.name,
    ]
    run_cmd(cmd_bathy, bathy, "static_bathy", "multiyear grid")

    metrics = STATIC_DIR / "medsea_my_grid_metrics_0_110m_source_domain.nc"
    cmd_metrics = [
        exe, "subset",
        "--dataset-id", DATASET_STATIC,
        "--dataset-part", "coords",
        "--variable", "e1t",
        "--variable", "e2t",
        "--variable", "e3t",
        "--minimum-longitude", str(WEST),
        "--maximum-longitude", str(EAST),
        "--minimum-latitude", str(SOUTH),
        "--maximum-latitude", str(NORTH),
        "--minimum-depth", str(OHC_MIN_DEPTH),
        "--maximum-depth", str(OHC_MAX_DEPTH),
        "--output-directory", str(STATIC_DIR),
        "--output-filename", metrics.name,
    ]
    run_cmd(cmd_metrics, metrics, "static_metrics", "multiyear grid")

def download_daily_sst(exe, year, test):
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    if test:
        start, end, tag = f"{year}-10-01T00:00:00", f"{year}-10-31T23:59:59", f"{year}10_test"
    else:
        start, end, tag = f"{year}-09-01T00:00:00", f"{year}-12-31T23:59:59", f"{year}_SepDec"
    target = DAILY_DIR / f"medsea_daily_sst_{tag}.nc"
    cmd = common_subset(exe, DATASET_DAILY_TEMP, DAILY_DIR, target.name)
    cmd += [
        "--minimum-depth", str(SST_MIN_DEPTH),
        "--maximum-depth", str(SST_MAX_DEPTH),
        "--start-datetime", start,
        "--end-datetime", end,
    ]
    run_cmd(cmd, target, "daily_sst", tag)

def download_monthly_3d(exe, year, test):
    MONTHLY_DIR.mkdir(parents=True, exist_ok=True)
    if test:
        start, end, tag = f"{year}-10-01T00:00:00", f"{year}-10-31T23:59:59", f"{year}10_test"
    else:
        start, end, tag = f"{year}-09-01T00:00:00", f"{year}-12-31T23:59:59", f"{year}_SepDec"
    target = MONTHLY_DIR / f"medsea_monthly_temp_0_110m_{tag}.nc"
    cmd = common_subset(exe, DATASET_MONTHLY_TEMP, MONTHLY_DIR, target.name)
    cmd += [
        "--minimum-depth", str(OHC_MIN_DEPTH),
        "--maximum-depth", str(OHC_MAX_DEPTH),
        "--start-datetime", start,
        "--end-datetime", end,
    ]
    run_cmd(cmd, target, "monthly_3d", tag)

def print_plan(args):
    if args.test:
        years, label = [2000], "TEST: ottobre 2000"
    else:
        years = list(range(args.start_year, args.end_year+1))
        label = f"Sep-Dic {args.start_year}-{args.end_year}"
    fam = [args.only_family] if args.only_family else ["static","daily_sst","monthly_3d"]
    print("COPERNICUS MARINE HISTORICAL NW — DOWNLOAD PLAN")
    print(f"Periodo: {label}")
    print(f"Anni: {len(years)}")
    print(f"Famiglie: {', '.join(fam)}")
    print(f"Dominio: lon {WEST}..{EAST}; lat {SOUTH}..{NORTH}")
    print("SST: giornaliera, solo primo livello (~1 m)")
    print("OHC input: temperatura mensile 0-110 m")
    print("Climatologia: verrà calcolata localmente sul 1991-2020")
    print("Resume: automatico; file completati vengono saltati.")
    print()

def main():
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("--start-year deve essere <= --end-year")

    exe = ensure_tool()
    check_disk()
    print_plan(args)

    if args.only_family in (None, "static"):
        print("="*88)
        print("STATICI MULTIYEAR")
        download_static(exe)
        check_disk()

    if args.only_family == "static":
        print("\nDOWNLOAD STATICI COMPLETATO.")
        return

    years = [2000] if args.test else list(range(args.start_year, args.end_year+1))
    for i, year in enumerate(years, 1):
        print("="*88)
        print(f"ANNO {year} ({i}/{len(years)})")
        if args.only_family in (None, "daily_sst"):
            download_daily_sst(exe, year, args.test)
        if args.only_family in (None, "monthly_3d"):
            download_monthly_3d(exe, year, args.test)
        check_disk()
        print(f"PROGRESSO ANNI: {i}/{len(years)}")

    print("\nCOPERNICUS MARINE HISTORICAL DOWNLOAD: COMPLETE")
    print(f"Output: {OUT}")
    print(f"Manifest: {MANIFEST}")
    print("Il prossimo script calcolerà climatologia 1991-2020, SST anomaly e OHC anomaly 0-100 m.")

if __name__ == "__main__":
    main()
