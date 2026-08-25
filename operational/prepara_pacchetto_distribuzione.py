#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "NW_FloodWatch_Mac_Windows.zip"
PREFIX = "NW_FloodWatch_Definitivo"

EXCLUDE_DIRS = {
    ".venv",
    "__pycache__",
    "nw_floodwatch_output",
    "nw_operational_raw_cache",
    "nw_operational_feature_snapshot",
}
EXCLUDE_FILES = {
    ".DS_Store",
    "assembla_database_da_progetto.py",
    "NW_FloodWatch_Bollettino_DEMO.pdf",
}
MUTABLE_PREFIXES = (
    "nw_operational_daily_feature_cache_v1_0/",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def excluded(p: Path) -> bool:
    rel = p.relative_to(ROOT)
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        return True
    if p.name in EXCLUDE_FILES or p.suffix.lower() == ".pyc":
        return True
    return False


def is_mutable_rel(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    return any(rel.startswith(prefix) for prefix in MUTABLE_PREFIXES)


def build_distribution_manifest() -> bytes:
    """Refresh hashes at packaging time and explicitly label mutable cache files."""
    src = ROOT / "DATABASE_RUNTIME_MANIFEST_SHA256.csv"
    if not src.exists():
        raise RuntimeError("Manca DATABASE_RUNTIME_MANIFEST_SHA256.csv")
    with src.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    out = io.StringIO(newline="")
    writer = csv.DictWriter(
        out,
        fieldnames=["relative_path", "size_bytes", "sha256", "integrity_policy"],
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        rel = (row.get("relative_path") or "").strip()
        if not rel:
            continue
        p = ROOT / rel
        if not p.exists():
            # A missing mutable cache is allowed by the verifier, but if it is in the
            # source manifest we keep packaging strict: the current distribution should
            # be self-contained at the time it is built.
            raise RuntimeError(f"File manifest mancante durante packaging: {rel}")
        writer.writerow({
            "relative_path": rel.replace("\\", "/"),
            "size_bytes": p.stat().st_size,
            "sha256": sha256(p),
            "integrity_policy": "MUTABLE_RUNTIME" if is_mutable_rel(rel) else "IMMUTABLE",
        })
    return out.getvalue().encode("utf-8")


def clean_ready_text(manifest_bytes: bytes) -> bytes:
    count = max(manifest_bytes.decode("utf-8").count("\n") - 1, 0)
    text = (
        "NW FloodWatch runtime database READY\n"
        "Source: DISTRIBUTION_PACKAGE\n"
        f"Files: {count}\n"
        "Manifest: DATABASE_RUNTIME_MANIFEST_SHA256.csv\n"
        "Status: PASS_RUNTIME_DATABASE_ASSEMBLED\n"
        "Note: entries marked MUTABLE_RUNTIME are expected to change after each operational run.\n"
    )
    return text.encode("utf-8")


def zipinfo_for(path: Path, arcname: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo.from_file(path, arcname=arcname)
    info.compress_type = zipfile.ZIP_DEFLATED
    info._compresslevel = 6
    if path.suffix == ".command":
        info.external_attr = (0o100755 & 0xFFFF) << 16
    return info


def main() -> int:
    verifier = ROOT / "verifica_pacchetto_multiplatform.py"
    if not verifier.exists():
        raise SystemExit("Manca verifica_pacchetto_multiplatform.py")
    rc = subprocess.run([sys.executable, str(verifier)], cwd=ROOT).returncode
    if rc != 0:
        raise SystemExit("Verifica pacchetto fallita: ZIP non creato.")

    manifest_bytes = build_distribution_manifest()
    ready_bytes = clean_ready_text(manifest_bytes)

    files = [p for p in ROOT.rglob("*") if p.is_file() and not excluded(p)]
    # The two generated files are written from memory so their distributed content is clean/current.
    generated_names = {"DATABASE_RUNTIME_MANIFEST_SHA256.csv", "DATABASE_RUNTIME_READY.txt"}
    normal_files = [p for p in files if p.name not in generated_names]

    total = sum(p.stat().st_size for p in normal_files) + len(manifest_bytes) + len(ready_bytes)
    done = 0

    if OUT.exists():
        OUT.unlink()

    print(f"Creazione: {OUT}")
    print(f"File     : {len(normal_files) + 2}")
    print(f"Bytes    : {total}")

    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Write refreshed manifest and sanitized READY marker first.
        zf.writestr(f"{PREFIX}/DATABASE_RUNTIME_MANIFEST_SHA256.csv", manifest_bytes)
        done += len(manifest_bytes)
        zf.writestr(f"{PREFIX}/DATABASE_RUNTIME_READY.txt", ready_bytes)
        done += len(ready_bytes)

        for i, p in enumerate(normal_files, 1):
            rel = p.relative_to(ROOT)
            arc = str(Path(PREFIX) / rel).replace(os.sep, "/")
            info = zipinfo_for(p, arc)
            with p.open("rb") as src, zf.open(info, "w", force_zip64=True) as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
                    done += len(chunk)
            pct = 100.0 * done / total if total else 100.0
            if i == 1 or i == len(normal_files) or i % 5 == 0:
                print(f"ZIP {i:3d}/{len(normal_files):3d} | {pct:6.2f}% | {rel}", flush=True)

    # Lightweight post-build integrity check: ensure the ZIP contains launchers and refreshed manifest.
    required_in_zip = {
        f"{PREFIX}/nw_flood_watch.py",
        f"{PREFIX}/setup_nw_floodwatch.command",
        f"{PREFIX}/avvia_nw_floodwatch.command",
        f"{PREFIX}/setup_nw_floodwatch_windows.bat",
        f"{PREFIX}/avvia_nw_floodwatch_windows.bat",
        f"{PREFIX}/DATABASE_RUNTIME_MANIFEST_SHA256.csv",
        f"{PREFIX}/DATABASE_RUNTIME_READY.txt",
    }
    with zipfile.ZipFile(OUT, "r") as zf:
        names = set(zf.namelist())
        missing = sorted(required_in_zip - names)
        if missing:
            OUT.unlink(missing_ok=True)
            raise SystemExit("ZIP incompleto, mancano: " + ", ".join(missing))

    print("=" * 78)
    print("STATUS: PASS_DISTRIBUTION_ZIP_CREATED")
    print(f"ZIP   : {OUT}")
    print("Manifest runtime aggiornato al momento del packaging.")
    print("Cache antecedente marcata MUTABLE_RUNTIME: le sue modifiche future sono attese.")
    print("Esclusi: .venv, output/snapshot/raw cache di collaudo, file di sviluppo.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
