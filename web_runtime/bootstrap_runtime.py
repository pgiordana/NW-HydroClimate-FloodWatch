#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

RELEASE_URL = (
    "https://github.com/pgiordana/NW-HydroClimate-FloodWatch/"
    "releases/download/v1.0-rc1/NW_FloodWatch_Mac_Windows.zip"
)
RELEASE_SHA256 = "c5199726e08db1fecbcf0e71ab147f2e42a754460ef051888e5c7259554694c2"
ARCHIVE_NAME = "NW_FloodWatch_Mac_Windows.zip"
RUNTIME_DIRNAME = "NW_FloodWatch_Definitivo"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    print(f"[PHASE 1/3] Download release asset -> {target}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
        total = response.headers.get("Content-Length")
        total_n = int(total) if total and total.isdigit() else None
        done = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total_n:
                pct = 100.0 * done / total_n
                print(
                    f"\r[PHASE 1/3] {done/1024/1024:7.1f}/{total_n/1024/1024:7.1f} MiB "
                    f"({pct:5.1f}%)",
                    end="",
                    flush=True,
                )
        if total_n:
            print(flush=True)
    tmp.replace(target)


def extract(archive: Path, root: Path) -> Path:
    runtime = root / RUNTIME_DIRNAME
    print(f"[PHASE 3/3] Extract -> {runtime}", flush=True)
    if runtime.exists():
        shutil.rmtree(runtime)
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)
    if not runtime.exists():
        raise RuntimeError(f"Archivio estratto ma runtime non trovato: {runtime}")
    return runtime


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and verify the exact NW FloodWatch v1.0-rc1 runtime release."
    )
    parser.add_argument(
        "--work-dir",
        default=".web_runtime",
        help="Directory used for downloaded/extracted runtime (default: .web_runtime)",
    )
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    work = Path(args.work_dir).resolve()
    downloads = work / "downloads"
    archive = downloads / ARCHIVE_NAME
    extracted = work / "runtime"

    if args.force_download and archive.exists():
        archive.unlink()

    if archive.exists() and sha256(archive) != RELEASE_SHA256:
        print("Cached archive checksum mismatch: re-downloading.", flush=True)
        archive.unlink()

    if not archive.exists():
        download(RELEASE_URL, archive)
    else:
        print(f"[PHASE 1/3] Reuse verified local archive candidate: {archive}", flush=True)

    print("[PHASE 2/3] Verify SHA256", flush=True)
    digest = sha256(archive)
    if digest != RELEASE_SHA256:
        raise RuntimeError(
            f"Release SHA256 mismatch: expected {RELEASE_SHA256}, got {digest}"
        )
    print(f"[PHASE 2/3] PASS {digest}", flush=True)

    runtime = extracted / RUNTIME_DIRNAME
    marker = runtime / "DATABASE_RUNTIME_READY.txt"
    if runtime.exists() and marker.exists() and not args.force_download:
        print(f"[PHASE 3/3] Existing extracted runtime reused: {runtime}", flush=True)
    else:
        runtime = extract(archive, extracted)

    print(f"RUNTIME_ROOT={runtime}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
