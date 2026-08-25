#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import platform
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED = [
    "DATABASE_RUNTIME_MANIFEST_SHA256.csv",
    "DATABASE_RUNTIME_READY.txt",
    "nw_flood_watch.py",
    "build_nw_operational_raw_cache_current_v1_1.py",
    "repair_nw_operational_raw_cache_surface_v1_1.py",
    "build_nw_operational_receptor_features_current_v1_1.py",
    "build_nw_operational_medsea_corridor_current_v1_1.py",
    "update_nw_operational_antecedent_cache_current_v1_0.py",
    "audit_nw_operational_beta_gate_current_v1_0.py",
    "requirements.txt",
    "config/bulletin_config.json",
    "data/operational_assets/reservoir_registry.csv",
    "nw_flood_models_frozen_development_v1_2",
    "nw_hydroclimate_core_release_v1_0",
    "nw_hydroclimate_foldwise_master_core_canonical_v1_0",
    "nw_operational_feature_equivalence_preflight_v1_0",
    "medsea_historical_analysis",
    "medsea_historical_nw",
]

# The antecedent cache is intentionally mutable: every operational run updates it.
MUTABLE_PREFIXES = (
    "nw_operational_daily_feature_cache_v1_0/",
)

# Build strings at runtime so the verifier does not flag its own source code.
FORBIDDEN_RUNTIME_MARKERS = [
    "/Users/" + "giordana/",
    "MedSea_" + "Copernicus_" + "Pipeline_v1",
]

STRICT_SCAN_SUFFIXES = {
    ".py", ".json", ".command", ".bat", ".ps1", ".toml", ".yaml", ".yml", ".ini"
}
PROVENANCE_SCAN_SUFFIXES = {".txt", ".md", ".csv"}
EXCLUDED_DIRS = {
    ".venv", "nw_floodwatch_output", "nw_operational_raw_cache",
    "nw_operational_feature_snapshot", "__pycache__"
}
SKIP_SCAN_FILES = {
    "assembla_database_da_progetto.py",
    "verifica_pacchetto_multiplatform.py",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_mutable(rel: str, row: dict[str, str]) -> bool:
    policy = (row.get("integrity_policy") or "").strip().upper()
    if policy == "MUTABLE_RUNTIME":
        return True
    return any(rel.replace("\\", "/").startswith(prefix) for prefix in MUTABLE_PREFIXES)


def main() -> int:
    print("=" * 78)
    print("NW FLOODWATCH - VERIFICA PACCHETTO MULTIPIATTAFORMA v2")
    print("=" * 78)
    print(f"Root      : {ROOT}")
    print(f"OS        : {platform.system()} {platform.release()}")
    print(f"Python    : {platform.python_version()}")
    print(f"Arch      : {struct.calcsize('P') * 8} bit")

    problems: list[str] = []
    warnings: list[str] = []
    infos: list[str] = []

    for rel in REQUIRED:
        if not (ROOT / rel).exists():
            problems.append(f"MISSING: {rel}")

    manifest = ROOT / "DATABASE_RUNTIME_MANIFEST_SHA256.csv"
    immutable_checked = 0
    mutable_seen = 0
    if manifest.exists():
        with manifest.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        print(f"Manifest  : {len(rows)} file")
        for i, row in enumerate(rows, 1):
            rel = (row.get("relative_path") or "").strip()
            if not rel:
                problems.append(f"MANIFEST ROW {i}: relative_path vuoto")
                continue
            p = ROOT / rel
            mutable = is_mutable(rel, row)
            if mutable:
                mutable_seen += 1
            else:
                immutable_checked += 1

            if not p.exists():
                if mutable:
                    warnings.append(f"MUTABLE CACHE MISSING (ricreabile): {rel}")
                else:
                    problems.append(f"MANIFEST MISSING: {rel}")
                continue

            expected_size = int(row.get("size_bytes") or 0)
            observed_size = p.stat().st_size
            expected_hash = (row.get("sha256") or "").strip().lower()
            observed_hash = sha256(p).lower() if expected_hash else ""

            size_mismatch = expected_size and observed_size != expected_size
            hash_mismatch = expected_hash and observed_hash != expected_hash

            if mutable:
                if size_mismatch or hash_mismatch:
                    warnings.append(
                        f"MUTABLE CACHE CHANGED (atteso dopo un run): {rel}"
                    )
            else:
                if size_mismatch:
                    problems.append(f"SIZE MISMATCH: {rel}")
                    continue
                if hash_mismatch:
                    problems.append(f"SHA256 MISMATCH: {rel}")

            if i == 1 or i == len(rows) or i % 5 == 0:
                print(f"HASH      : {i}/{len(rows)}", end="\r", flush=True)
        print(" " * 50, end="\r")
        print(f"Immutable : {immutable_checked} verificati")
        print(f"Mutable   : {mutable_seen} file cache (esistenza controllata; hash non bloccante)")

    # Runtime code/config must not contain author-machine paths.
    # Provenance/audit text may legitimately preserve historical source paths;
    # those are reported as warnings because they are not executable dependencies.
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in EXCLUDED_DIRS for part in p.parts):
            continue
        if p.name in SKIP_SCAN_FILES:
            continue
        suffix = p.suffix.lower()
        if suffix not in STRICT_SCAN_SUFFIXES and suffix not in PROVENANCE_SCAN_SUFFIXES:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for marker in FORBIDDEN_RUNTIME_MARKERS:
            if marker not in text:
                continue
            rel = p.relative_to(ROOT)
            if suffix in STRICT_SCAN_SUFFIXES:
                problems.append(f"ABSOLUTE/DEV PATH IN RUNTIME FILE: {rel} -> {marker}")
            else:
                warnings.append(f"PROVENANCE PATH ONLY: {rel} -> {marker}")

    if sys.version_info[:2] not in {(3, 12), (3, 13)}:
        warnings.append("Python consigliato: 3.12 o 3.13.")
    if struct.calcsize("P") * 8 != 64:
        problems.append("Python non e a 64 bit.")
    if (ROOT / ".venv").exists():
        warnings.append(".venv presente: corretta per il test locale, sara esclusa dallo ZIP.")

    # De-duplicate while preserving order.
    warnings = list(dict.fromkeys(warnings))
    problems = list(dict.fromkeys(problems))

    print(f"Problemi  : {len(problems)}")
    print(f"Warning   : {len(warnings)}")
    for w in warnings:
        print(f"WARNING   : {w}")
    for p in problems:
        print(f"FAIL      : {p}")

    if problems:
        print("STATUS    : FAIL_PACKAGE_CHECK")
        return 1

    print("STATUS    : PASS_PACKAGE_CHECK")
    print("Nota      : le modifiche della cache antecedente sono attese e non sono")
    print("            corruzione del database immutabile. I percorsi presenti solo")
    print("            nei file di audit/provenienza non sono dipendenze esecutive.")
    print("            PASS non sostituisce un test end-to-end reale su Windows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
