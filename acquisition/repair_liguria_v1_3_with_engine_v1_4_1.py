#!/usr/bin/env python3
"""
REPAIR ARPAL LIGURIA v1.4.1 -> dataset v1.3
============================================

Scopo:
- NON riscarica tutto.
- Riusa il motore robusto di download_liguria_groundtruth_v1_4_1.py.
- Ripara IN PLACE il dataset principale:
    observations_nw/liguria_groundtruth_v1_3/hourly

Seleziona automaticamente:
1) tutte le combinazioni target-anno classificate MISSING o WRONG_YEAR
   dal QC scientifico v1.1;
2) tutti i file di precipitazione già presenti il cui header "Parametro"
   NON contiene "PRECIPITAZIONE CUMULATA" (es. Varese Ligure / manto nevoso);
3) per sicurezza, tutti gli anni 1987-2025 di magra_varese_ligure_precip.

NON ritenta i 343 NO_DATES già confermati come "Nessun dato presente",
salvo quelli di Varese Ligure, che vanno rigenerati perché il parametro
originario era sbagliato.

Requisito:
  nella stessa cartella deve esserci:
    download_liguria_groundtruth_v1_4_1.py

Uso:
  python repair_liguria_v1_3_with_engine_v1_4_1.py --plan
  caffeinate -i python repair_liguria_v1_3_with_engine_v1_4_1.py
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

from playwright.sync_api import sync_playwright

import download_liguria_groundtruth_v1_4_1 as eng


ROOT = Path(__file__).resolve().parent

BASE_V13 = ROOT / "observations_nw" / "liguria_groundtruth_v1_3"
HOURLY_V13 = BASE_V13 / "hourly"

MATRIX = (
    BASE_V13
    / "qc_scientific_v1_1"
    / "target_year_matrix.csv"
)

REPAIR_DIAG = BASE_V13 / "_repair_diagnostics_v1_4_1"
REPAIR_MANIFEST = BASE_V13 / "arpal_repair_manifest_v1_4_1.jsonl"

START_YEAR = 1987
END_YEAR = 2025


def decode(raw: bytes) -> str:
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig", errors="replace")
    try:
        return raw.decode("utf-8")
    except Exception:
        try:
            return raw.decode("cp1252")
        except Exception:
            return raw.decode("latin-1", errors="replace")


def read_matrix_repairs():
    repairs = set()

    if not MATRIX.exists():
        raise FileNotFoundError(
            f"Non trovo la matrice QC v1.1:\n{MATRIX}"
        )

    with MATRIX.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            status = (row.get("final_status") or "").strip()
            if status in {"MISSING", "WRONG_YEAR"}:
                slug = (row.get("target") or "").strip()
                try:
                    year = int(row["year"])
                except Exception:
                    continue
                repairs.add((slug, year, status))

    return repairs


def scan_wrong_precip_parameters():
    """
    Cerca file di precipitazione già presenti con parametro sbagliato.
    Un file NO_DATA è comunque informativo perché contiene:
      "Parametro",...
    """
    repairs = set()

    for path in sorted(HOURLY_V13.rglob("*_precip/*_09-12.csv")):
        text = decode(path.read_bytes())
        slug = path.parent.name

        m_year = re.search(r"_(\d{4})_09-12\.csv$", path.name, re.I)
        if not m_year:
            continue
        year = int(m_year.group(1))

        # Header ARPAL tipico:
        # "Parametro",PRECIPITAZIONE - PRECIPITAZIONE CUMULATA (mm)
        m = re.search(
            r'["\']?Parametro["\']?\s*[,;]\s*(.+)',
            text,
            flags=re.I,
        )
        if not m:
            # Non aggiungiamo automaticamente: il file potrebbe essere
            # buono ma con struttura differente. Il QC lo gestisce altrove.
            continue

        param_line = m.group(1).strip().strip('"').strip("'")
        norm = re.sub(r"\s+", " ", param_line).upper()

        if "PRECIPITAZIONE CUMULATA" not in norm:
            repairs.add((slug, year, f"WRONG_PARAMETER:{param_line[:100]}"))

    return repairs


def build_plan():
    reasons = {}

    def add(slug, year, reason):
        reasons.setdefault((slug, year), set()).add(reason)

    for slug, year, reason in read_matrix_repairs():
        add(slug, year, reason)

    for slug, year, reason in scan_wrong_precip_parameters():
        add(slug, year, reason)

    # Varese Ligure: rigeneriamo tutta la serie 1987-2025 perché la v1.3
    # poteva scegliere "ALTEZZA DEL MANTO NEVOSO".
    for year in range(START_YEAR, END_YEAR + 1):
        add(
            "magra_varese_ligure_precip",
            year,
            "FORCE_VARESE_CORRECT_PRECIP",
        )

    return reasons


def all_cfg_by_slug():
    return {cfg["slug"]: cfg for cfg in eng.TARGETS}


def print_plan(plan):
    print("=" * 110)
    print("REPAIR PLAN ARPAL LIGURIA")
    print("=" * 110)
    print("Combinazioni da riparare:", len(plan))

    by_slug = Counter(slug for slug, _ in plan)
    for slug, count in sorted(by_slug.items()):
        years = sorted(y for s, y in plan if s == slug)
        print(
            f"{slug:<42s} {count:3d} | "
            f"{min(years)}-{max(years)}"
        )

    reason_counts = Counter(
        reason
        for rs in plan.values()
        for reason in rs
    )

    print("\nMotivi:")
    for reason, count in reason_counts.most_common():
        print(f"  {reason:<70s} {count:4d}")

    print("=" * 110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--plan",
        action="store_true",
        help="Mostra soltanto il piano; nessun download.",
    )
    ap.add_argument(
        "--headed",
        action="store_true",
        help="Browser visibile.",
    )
    args = ap.parse_args()

    plan = build_plan()
    print_plan(plan)

    if args.plan:
        return

    cfg_map = all_cfg_by_slug()

    unknown = sorted(
        {slug for slug, _ in plan if slug not in cfg_map}
    )
    if unknown:
        raise RuntimeError(
            "Target del QC non presenti nel motore v1.4.1: "
            + ", ".join(unknown)
        )

    # Reindirizza il motore v1.4.1 sul dataset principale v1.3.
    eng.OUT = HOURLY_V13
    eng.DIAG = REPAIR_DIAG
    eng.MANIFEST = REPAIR_MANIFEST

    eng.OUT.mkdir(parents=True, exist_ok=True)
    eng.DIAG.mkdir(parents=True, exist_ok=True)
    eng.MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        context = browser.new_context(
            accept_downloads=True,
            locale="it-IT",
        )

        catalog = eng.prepare_catalog(context)

        resolved = {}
        for slug in sorted({slug for slug, _ in plan}):
            cfg = cfg_map[slug]
            rec = eng.resolve_station(catalog, cfg)
            if rec is None:
                print(f"NON TROVATA: {slug}")
                stats["station_not_found"] += 1
            else:
                resolved[slug] = rec
                print(
                    f"RISOLTA {slug:<42s} -> "
                    f"{rec['code']} {rec['name']}"
                )

        for slug, year in sorted(plan):
            cfg = cfg_map[slug]
            rec = resolved.get(slug)
            if rec is None:
                continue

            reasons = ", ".join(sorted(plan[(slug, year)]))
            print("\n" + "#" * 110)
            print(
                f"REPAIR {slug} | {year} | {reasons}"
            )
            print("#" * 110)

            result = eng.download_one(
                context,
                cfg,
                rec,
                year,
                force=True,
            )
            stats[result] += 1

        context.close()
        browser.close()

    print("\n" + "=" * 110)
    print("REPAIR COMPLETATO")
    print("=" * 110)
    for k, v in sorted(stats.items()):
        print(f"{k:<30s} {v:5d}")
    print("Manifest repair:", REPAIR_MANIFEST)
    print("=" * 110)


if __name__ == "__main__":
    main()
