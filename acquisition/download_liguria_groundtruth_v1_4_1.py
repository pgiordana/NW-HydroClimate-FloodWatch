#!/usr/bin/env python3
"""
LIGURIA REGIONAL GROUND-TRUTH DOWNLOADER v1.4.1
=============================================

Versione costruita sul MOTORE ARPAL v5.4 che ha già funzionato davvero
nel ramo Tanaro-Arroscia.

Differenza rispetto alle v1.0/v1.1 fallite:
- NON usa la modalità BACINO;
- NON reinventa la lettura dei frame;
- usa esattamente il flusso verificato:
    TipoTema=STAZIONE
    frame Punto -> select Ubic + Frequenza
    frame Scheda -> Param + TipoOutput
    Frequenza HH
    output XLS = "File .csv (MS Excel)"
    periodo settembre-dicembre, anno per anno
    click su images/accediAiDati.gif
- usa, se presente, il catalogo ARPAL già salvato dal downloader
  Tanaro-Arroscia v5.4 (274 stazioni), evitando una discovery inutile.

Scopo scientifico
-----------------
Completare la ground truth osservativa dei 5 recettori liguri del documento
principale. Per evitare duplicazioni, CENTA viene escluso di default perché
Colle di Nava / Pieve di Teco / Ranzo / Pieve di Teco IDRO / Pogli sono già
nel ramo Tanaro-Arroscia. Si può includere con --include-centa.

Target regionali minimi:
BISAGNO
  - Davagna precip
  - Creto precip
  - Genova-Geirato precip
  - La Presa livello
  - Genova-Firpo livello

POLCEVERA
  - Mignanego precip
  - Isoverde precip
  - Genova-Bolzaneto precip
  - Genova-Pontedecimo livello
  - Genova-Rivarolo livello

ENTELLA
  - Neirone precip
  - Giacopiane-Lago precip
  - Cichero precip
  - Panesi livello
  - Carasco livello

MAGRA
  - Varese Ligure precip
  - Tavarone precip
  - Brugnato precip
  - Nasceto livello
  - Fornola livello

CENTA opzionale
  - Colle di Nava precip
  - Pieve di Teco precip
  - Ranzo precip
  - Pieve di Teco IDRO livello
  - Pogli d'Ortovero livello
  - Cisano sul Neva livello

Uso
---
1) SOLO RISOLUZIONE TARGET, nessun download:
   python download_liguria_groundtruth_v1_4_1.py --discover

2) TEST 2020 di una stazione:
   python download_liguria_groundtruth_v1_4_1.py \
       --test --only bisagno_davagna_precip

3) COMPLETO, headless:
   caffeinate -i python download_liguria_groundtruth_v1_4_1.py

4) Solo un bacino:
   caffeinate -i python download_liguria_groundtruth_v1_4_1.py --basin BISAGNO

5) Audit:
   python download_liguria_groundtruth_v1_4_1.py --audit

Output
------
observations_nw/liguria_groundtruth_v1_4/
  catalog/
  hourly/<target>/<target>_YYYY_09-12.csv
  _diagnostics/
  arpal_regional_manifest_v1_4.jsonl
  audit_liguria_v1_4.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout


# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path(__file__).resolve().parent

BASE = ROOT / "observations_nw" / "liguria_groundtruth_v1_4"
OUT = BASE / "hourly"
DIAG = BASE / "_diagnostics"
CATALOG_DIR = BASE / "catalog"
MANIFEST = BASE / "arpal_regional_manifest_v1_4.jsonl"
AUDIT_FILE = BASE / "audit_liguria_v1_4.txt"

EXISTING_CATALOG = (
    ROOT / "tanaro_arroscia" / "observations" / "arpal_omirl"
    / "catalog" / "arpal_station_catalog.csv"
)

PORTAL_URL = (
    "https://ambientepub.regione.liguria.it/"
    "SiraQualMeteo/script/PubAccessoDatiMeteo.asp"
)

START_YEAR = 1987
END_YEAR = 2025

MIN_BYTES = 50
NAV_TIMEOUT = 90_000
DOWNLOAD_TIMEOUT = 120_000


def T(
    slug,
    basin,
    station_terms,
    parameter_terms,
    kind,
    station_code=None,
):
    return {
        "slug": slug,
        "basin": basin,
        "station_code": station_code,
        "station_terms": station_terms,
        "parameter_terms": parameter_terms,
        "kind": kind,
    }


LEVEL_TERMS = [
    "LIVELLO IDROMETRICO",
    "ALTEZZA IDROMETRICA",
    "IDROMETR",
    "LIVELLO",
]

PRECIP_TERMS = [
    # IMPORTANTE: NON usare il termine generico "PRECIPITAZIONE".
    # Nel portale ARPAL anche "PRECIPITAZIONE - ALTEZZA DEL MANTO NEVOSO"
    # contiene quella parola e nella v1.3 Varese Ligure è stato quindi
    # selezionato erroneamente come neve.
    "PRECIPITAZIONE CUMULATA",
]

TARGETS = [
    # BISAGNO
    T(
        "bisagno_davagna_precip", "BISAGNO",
        ["DAVAGNA"], PRECIP_TERMS, "precipitation",
        station_code="ME00196",
    ),
    T(
        "bisagno_creto_precip", "BISAGNO",
        ["CRETO"], PRECIP_TERMS, "precipitation",
        station_code="ME00060",
    ),
    T(
        "bisagno_geirato_precip", "BISAGNO",
        ["GENOVA - GEIRATO", "GEIRATO"], PRECIP_TERMS, "precipitation",
        station_code="ME00302",
    ),
    T(
        "bisagno_lapresa_level", "BISAGNO",
        ["LA PRESA"], LEVEL_TERMS, "water_level",
        station_code="ME00048",
    ),
    T(
        "bisagno_firpo_level", "BISAGNO",
        ["GENOVA - FIRPO", "FIRPO"], LEVEL_TERMS, "water_level",
        station_code="ME00013",
    ),

    # POLCEVERA
    T(
        "polcevera_mignanego_precip", "POLCEVERA",
        ["MIGNANEGO"], PRECIP_TERMS, "precipitation",
        station_code="ME00050",
    ),
    T(
        "polcevera_isoverde_precip", "POLCEVERA",
        ["ISOVERDE"], PRECIP_TERMS, "precipitation",
    ),
    T(
        "polcevera_bolzaneto_precip", "POLCEVERA",
        ["GENOVA - BOLZANETO", "BOLZANETO"], PRECIP_TERMS, "precipitation",
        station_code="ME00046",
    ),
    T(
        "polcevera_pontedecimo_level", "POLCEVERA",
        ["GENOVA - PONTEDECIMO", "PONTEDECIMO"], LEVEL_TERMS, "water_level",
        station_code="ME00039",
    ),
    T(
        "polcevera_rivarolo_level", "POLCEVERA",
        ["GENOVA - RIVAROLO", "RIVAROLO"], LEVEL_TERMS, "water_level",
        station_code="ME00294",
    ),

    # ENTELLA
    T(
        "entella_neirone_precip", "ENTELLA",
        ["NEIRONE"], PRECIP_TERMS, "precipitation",
    ),
    T(
        "entella_giacopiane_precip", "ENTELLA",
        ["GIACOPIANE - LAGO", "GIACOPIANE"], PRECIP_TERMS, "precipitation",
        station_code="ME00070",
    ),
    T(
        "entella_cichero_precip", "ENTELLA",
        ["CICHERO"], PRECIP_TERMS, "precipitation",
    ),
    T(
        "entella_panesi_level", "ENTELLA",
        ["PANESI"], LEVEL_TERMS, "water_level",
    ),
    T(
        "entella_carasco_level", "ENTELLA",
        ["CARASCO"], LEVEL_TERMS, "water_level",
    ),

    # MAGRA / VARA
    T(
        "magra_varese_ligure_precip", "MAGRA",
        ["VARESE LIGURE"], PRECIP_TERMS, "precipitation",
    ),
    T(
        "magra_tavarone_precip", "MAGRA",
        ["TAVARONE"], PRECIP_TERMS, "precipitation",
    ),
    T(
        "magra_brugnato_precip", "MAGRA",
        ["BRUGNATO"], PRECIP_TERMS, "precipitation",
    ),
    T(
        "magra_nasceto_level", "MAGRA",
        ["NASCETO"], LEVEL_TERMS, "water_level",
    ),
    T(
        "magra_fornola_level", "MAGRA",
        ["FORNOLA"], LEVEL_TERMS, "water_level",
    ),

    # CENTA - opzionale; in parte già presente nel ramo Tanaro-Arroscia
    T(
        "centa_colle_nava_precip", "CENTA",
        ["COLLE DI NAVA"], PRECIP_TERMS, "precipitation",
        station_code="ME00025",
    ),
    T(
        "centa_pieve_teco_precip", "CENTA",
        ["PIEVE DI TECO"], PRECIP_TERMS, "precipitation",
        station_code="ME00090",
    ),
    T(
        "centa_ranzo_precip", "CENTA",
        ["RANZO"], PRECIP_TERMS, "precipitation",
        station_code="ME00026",
    ),
    T(
        "centa_pieve_teco_level", "CENTA",
        ["PIEVE DI TECO (IDRO)", "PIEVE DI TECO"], LEVEL_TERMS, "water_level",
        station_code="ME00342",
    ),
    T(
        "centa_pogli_level", "CENTA",
        ["POGLI D'ORTOVERO", "POGLI", "ORTOVERO"], LEVEL_TERMS, "water_level",
        station_code="ME00023",
    ),
    T(
        "centa_cisano_neva_level", "CENTA",
        ["CISANO SUL NEVA", "CISANO", "NEVA"], LEVEL_TERMS, "water_level",
    ),
]


# =============================================================================
# UTILS
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=START_YEAR)
    p.add_argument("--end-year", type=int, default=END_YEAR)
    p.add_argument("--test", action="store_true")
    p.add_argument("--discover", action="store_true")
    p.add_argument("--headed", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--only", default=None)
    p.add_argument(
        "--basin",
        choices=["BISAGNO", "POLCEVERA", "ENTELLA", "MAGRA", "CENTA"],
        default=None,
    )
    p.add_argument(
        "--include-centa",
        action="store_true",
        help="Include anche CENTA, già in larga parte acquisito nel ramo Tanaro-Arroscia.",
    )
    p.add_argument("--audit", action="store_true")
    return p.parse_args()


def setup():
    for p in (BASE, OUT, DIAG, CATALOG_DIR):
        p.mkdir(parents=True, exist_ok=True)


def norm(x: Any) -> str:
    s = "" if x is None else str(x)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.upper().replace("\xa0", " ").strip()
    return re.sub(r"\s+", " ", s)


def safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")


def file_ok(path: Path):
    return path.exists() and path.is_file() and path.stat().st_size >= MIN_BYTES


def log_manifest(**rec):
    rec["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    with MANIFEST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def options(sel):
    return sel.locator("option").evaluate_all(
        """els => els.map(o => ({
            text: (o.textContent || '').trim(),
            value: o.value || ''
        }))"""
    )


def all_frames(page):
    return list(page.frames)


def wait_until(fn, timeout_s=20, interval=0.4):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            result = fn()
            if result:
                return result
        except Exception:
            pass
        time.sleep(interval)
    return None


# =============================================================================
# FRAME / DIAGNOSTICS
# =============================================================================

def frame_by_name(page, name):
    for fr in all_frames(page):
        if fr.name == name:
            return fr
    return None


def frame_snapshot(page):
    out = []
    for idx, fr in enumerate(all_frames(page)):
        item = {
            "index": idx,
            "name": fr.name,
            "url": fr.url,
            "selects": [],
            "inputs": [],
        }

        try:
            sels = fr.locator("select")
            for i in range(sels.count()):
                sel = sels.nth(i)
                try:
                    opts = options(sel)
                except Exception:
                    opts = []
                item["selects"].append({
                    "index": i,
                    "name": sel.get_attribute("name"),
                    "onchange": sel.get_attribute("onchange"),
                    "options": opts,
                })
        except Exception:
            pass

        try:
            ins = fr.locator("input")
            for i in range(ins.count()):
                el = ins.nth(i)
                typ = norm(el.get_attribute("type") or "text")
                if typ in {
                    "RADIO", "SUBMIT", "IMAGE",
                    "BUTTON", "HIDDEN", "TEXT"
                }:
                    item["inputs"].append({
                        "index": i,
                        "type": typ,
                        "name": el.get_attribute("name"),
                        "value": el.get_attribute("value"),
                        "onclick": el.get_attribute("onclick"),
                    })
        except Exception:
            pass

        out.append(item)
    return out


def save_diag(page, tag, extra=None):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = DIAG / f"{stamp}_{safe(tag)}"
    try:
        page.screenshot(path=str(base.with_suffix(".png")), full_page=True)
    except Exception:
        pass

    obj = {
        "url": page.url,
        "frames": frame_snapshot(page),
        "extra": extra,
    }
    base.with_suffix(".json").write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return base


# =============================================================================
# EXACT WORKING v5.4 ENGINE
# =============================================================================

def enter_station_mode(page):
    page.goto(PORTAL_URL, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
    page.wait_for_timeout(1200)

    def find_radio():
        for fr in all_frames(page):
            loc = fr.locator(
                'input[type="radio"][name="TipoTema"][value="STAZIONE"]'
            )
            if loc.count():
                return fr, loc.first
        return None

    found = wait_until(find_radio, timeout_s=20)
    if not found:
        raise RuntimeError("Non trovo TipoTema=STAZIONE nel portale completo.")

    _, radio = found
    radio.click()

    def punto_ready():
        punto = frame_by_name(page, "Punto")
        if punto is None:
            return None
        ubic = punto.locator('select[name="Ubic"]')
        freq = punto.locator('select[name="Frequenza"]')
        if ubic.count() and freq.count():
            return punto
        return None

    punto = wait_until(punto_ready, timeout_s=20)
    if punto is None:
        raise RuntimeError(
            "Il click STAZIONE non ha prodotto Ubic/Frequenza nel frame Punto."
        )

    scheda = wait_until(lambda: frame_by_name(page, "Scheda"), timeout_s=10)

    print(
        f"  STAZIONE mode OK | Punto={punto.url} | "
        f"Scheda={(scheda.url if scheda else 'non ancora disponibile')}"
    )
    return punto


def get_station_catalog(page):
    punto = frame_by_name(page, "Punto")
    if punto is None:
        raise RuntimeError("Frame Punto assente.")

    ubic = punto.locator('select[name="Ubic"]')
    if ubic.count() == 0:
        raise RuntimeError("select Ubic assente.")

    catalog = []
    for opt in options(ubic):
        if not opt["value"]:
            continue
        text = opt["text"]
        province = None
        m = re.search(r"\(([^()]*)\)\s*$", text)
        if m:
            province = m.group(1).strip()
        catalog.append({
            "code": opt["value"],
            "name": text,
            "province": province,
        })
    return catalog


def save_catalog(catalog):
    jpath = CATALOG_DIR / "arpal_station_catalog.json"
    cpath = CATALOG_DIR / "arpal_station_catalog.csv"

    jpath.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with cpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["code", "name", "province"])
        w.writeheader()
        w.writerows(catalog)

    print(f"  Catalogo stazioni: {len(catalog)} voci")
    print(f"  Salvato: {cpath}")


def load_existing_catalog():
    if not EXISTING_CATALOG.exists():
        return None
    try:
        with EXISTING_CATALOG.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rows = [
            {
                "code": r.get("code", ""),
                "name": r.get("name", ""),
                "province": r.get("province", ""),
            }
            for r in rows
            if r.get("code")
        ]
        if len(rows) >= 200:
            print(
                f"  REUSE catalogo già verificato dal Tanaro-Arroscia: "
                f"{len(rows)} stazioni"
            )
            save_catalog(rows)
            return rows
    except Exception:
        pass
    return None


def resolve_station(catalog, cfg):
    code = cfg.get("station_code")
    if code:
        for rec in catalog:
            if rec["code"] == code:
                return rec

    terms = [norm(x) for x in cfg["station_terms"]]

    for rec in catalog:
        name = norm(re.sub(r"\s*\([^()]*\)\s*$", "", rec["name"]))
        if any(name == t for t in terms):
            return rec

    for rec in catalog:
        name = norm(rec["name"])
        if any(t in name for t in terms):
            return rec
    return None


def select_station(page, station_rec):
    punto = frame_by_name(page, "Punto")
    ubic = punto.locator('select[name="Ubic"]')

    ubic.select_option(value=station_rec["code"])
    print(f"  Stazione: {station_rec['code']} | {station_rec['name']}")

    def punto_station_ready():
        fr = frame_by_name(page, "Punto")
        if fr is None:
            return None
        try:
            u = fr.locator('select[name="Ubic"]')
            f = fr.locator('select[name="Frequenza"]')
            if u.count() == 0 or f.count() == 0:
                return None
            selected = u.input_value()
            fvals = {o["value"]: o["text"] for o in options(f)}
            if selected == station_rec["code"] and "HH" in fvals:
                return fr
        except Exception:
            return None
        return None

    punto = wait_until(punto_station_ready, timeout_s=20, interval=0.35)
    if punto is None:
        raise RuntimeError(
            f"Dopo la scelta {station_rec['code']} il frame Punto non è "
            "tornato stabile con Frequenza disponibile."
        )

    def scheda_param_ready():
        fr = frame_by_name(page, "Scheda")
        if fr is None:
            return None
        try:
            p = fr.locator('select[name="Param"]')
            if p.count() and len(options(p)) > 0:
                return fr
        except Exception:
            return None
        return None

    scheda = wait_until(scheda_param_ready, timeout_s=20, interval=0.35)
    if scheda is None:
        raise RuntimeError(
            f"La scelta {station_rec['code']} non ha prodotto Param nel frame Scheda."
        )
    return scheda


def select_frequency_hourly(page):
    def frequency_ready():
        punto = frame_by_name(page, "Punto")
        if punto is None:
            return None
        freq = punto.locator('select[name="Frequenza"]')
        try:
            if freq.count() == 0:
                return None
            vals = {o["value"]: o["text"] for o in options(freq)}
        except Exception:
            return None
        if "HH" in vals:
            return punto, freq, vals
        return None

    found = wait_until(frequency_ready, timeout_s=30, interval=0.35)
    if not found:
        raise RuntimeError(
            "Il frame Punto non ha ripristinato Frequenza=HH entro 30 secondi."
        )

    _, freq, _ = found
    freq.select_option(value="HH")
    print("  Frequenza: HH | Orario")

    # IMPORTANTE: la selezione HH ricarica dinamicamente Scheda.
    # v1.4 non controllava il valore di ritorno di wait_until e poteva
    # proseguire mentre Param era temporaneamente assente.
    def scheda_ready():
        scheda = frame_by_name(page, "Scheda")
        if scheda is None:
            return None
        try:
            param = scheda.locator('select[name="Param"]')
            if param.count() and len(options(param)) > 0:
                return scheda
        except Exception:
            return None
        return None

    scheda = wait_until(scheda_ready, timeout_s=30, interval=0.35)
    if scheda is None:
        raise RuntimeError(
            "Dopo Frequenza=HH il frame Scheda non ha ripristinato Param entro 30 secondi."
        )

    page.wait_for_timeout(500)
    return scheda


def select_parameter(page, terms):
    """
    Selezione parametro ARPAL v1.4.

    Regola fondamentale:
    - per pioggia accetta SOLO opzioni che contengono
      "PRECIPITAZIONE CUMULATA";
    - per livello preferisce "LIVELLO MEDIO DEL TORRENTE",
      poi le altre formulazioni idrometriche ammesse.

    Questo evita il bug v1.3 in cui il termine generico "PRECIPITAZIONE"
    poteva selezionare "ALTEZZA DEL MANTO NEVOSO".
    """
    def param_ready():
        scheda = frame_by_name(page, "Scheda")
        if scheda is None:
            return None
        try:
            param = scheda.locator('select[name="Param"]')
            if param.count() and len(options(param)) > 0:
                return scheda, param
        except Exception:
            return None
        return None

    found = wait_until(param_ready, timeout_s=30, interval=0.35)
    if not found:
        raise RuntimeError("Param assente/non pronto nel frame Scheda dopo 30 secondi.")

    scheda, param = found
    opts = options(param)
    nterms = [norm(t) for t in terms]

    # Classifica il tipo richiesto.
    wants_rain = any("PRECIPITAZIONE CUMULATA" in t for t in nterms)
    wants_level = any(
        x in " ".join(nterms)
        for x in (
            "LIVELLO MEDIO DEL TORRENTE",
            "LIVELLO IDROMETRICO",
            "ALTEZZA IDROMETRICA",
            "IDROMETR",
            "LIVELLO",
        )
    )

    candidates = []
    for o in opts:
        if not o["value"]:
            continue
        txt = norm(o["text"])

        if wants_rain:
            if "PRECIPITAZIONE CUMULATA" in txt:
                candidates.append((100, o))
            continue

        if wants_level:
            if "LIVELLO MEDIO DEL TORRENTE" in txt:
                candidates.append((100, o))
            elif "LIVELLO IDROMETRICO" in txt:
                candidates.append((90, o))
            elif "ALTEZZA IDROMETRICA" in txt:
                candidates.append((80, o))
            elif "IDROMETR" in txt:
                candidates.append((70, o))
            elif "LIVELLO" in txt:
                candidates.append((60, o))
            continue

        # fallback solo per eventuali usi futuri non rain/non level
        score = 0
        for t in nterms:
            if t == txt:
                score = max(score, 100)
            elif t in txt:
                score = max(score, 50)
        if score:
            candidates.append((score, o))

    if not candidates:
        avail = [o["text"] for o in opts]
        raise RuntimeError(
            f"Parametro scientificamente corretto {terms} non disponibile. "
            f"Disponibili: {avail}"
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    chosen = candidates[0][1]

    # Guard rail esplicito.
    chosen_txt = norm(chosen["text"])
    if wants_rain and "PRECIPITAZIONE CUMULATA" not in chosen_txt:
        raise RuntimeError(
            f"GUARD-RAIL: parametro pioggia errato selezionato: {chosen['text']}"
        )
    if wants_rain and "MANTO NEVOSO" in chosen_txt:
        raise RuntimeError(
            f"GUARD-RAIL: parametro neve rifiutato: {chosen['text']}"
        )

    param.select_option(value=chosen["value"])
    print(f"  Parametro: {chosen['value']} | {chosen['text']}")
    page.wait_for_timeout(400)
    return chosen


def select_csv(page):
    scheda = frame_by_name(page, "Scheda")
    output = scheda.locator('select[name="TipoOutput"]')
    if output.count() == 0:
        raise RuntimeError("TipoOutput assente.")

    vals = {o["value"]: o["text"] for o in options(output)}
    if "XLS" in vals:
        output.select_option(value="XLS")
        chosen = {"value": "XLS", "text": vals["XLS"]}
    elif "ASCII" in vals:
        output.select_option(value="ASCII")
        chosen = {"value": "ASCII", "text": vals["ASCII"]}
    else:
        raise RuntimeError(f"Output CSV/ASCII non disponibile: {vals}")

    print(f"  Output: {chosen['value']} | {chosen['text']}")
    page.wait_for_timeout(300)

    sep = scheda.locator('select[name="Separatore"]')
    if sep.count():
        try:
            visible = sep.is_visible()
            enabled = sep.is_enabled()
        except Exception:
            visible = enabled = False
        if visible and enabled:
            sep_vals = {o["value"]: o["text"] for o in options(sep)}
            if ";" in sep_vals:
                sep.select_option(value=";")
        else:
            print(
                "  Separatore: non richiesto per output XLS/CSV "
                "(controllo nascosto)"
            )
    return chosen


def fill_period(page, start_date, end_date):
    scheda = frame_by_name(page, "Scheda")
    if scheda is None:
        raise RuntimeError("Scheda assente.")

    start_ok = False
    end_ok = False
    rows = scheda.locator("tr")

    for i in range(rows.count()):
        row = rows.nth(i)
        try:
            txt = norm(row.inner_text())
        except Exception:
            continue
        inputs = row.locator('input[type="text"], input:not([type])')

        if "INIZIO PERIODO" in txt and inputs.count():
            inputs.nth(0).fill(start_date)
            if inputs.count() > 1:
                inputs.nth(1).fill("00:00")
            start_ok = True

        if "FINE PERIODO" in txt and inputs.count():
            inputs.nth(0).fill(end_date)
            if inputs.count() > 1:
                inputs.nth(1).fill("23:59")
            end_ok = True

    if not (start_ok and end_ok):
        raise RuntimeError("Campi periodo non riconosciuti.")

    print(f"  Periodo: {start_date} 00:00 -> {end_date} 23:59")


def find_access_button(page):
    scheda = frame_by_name(page, "Scheda")
    if scheda is None:
        return None

    imgs = scheda.locator("img")
    for i in range(imgs.count()):
        img = imgs.nth(i)
        try:
            blob = " ".join([
                norm(img.get_attribute("src") or ""),
                norm(img.get_attribute("alt") or ""),
                norm(img.get_attribute("title") or ""),
            ])
            compact = re.sub(r"[^A-Z0-9]", "", blob)
            if "ACCEDIAIDATI" in compact:
                parent_link = img.locator("xpath=ancestor::a[1]")
                if parent_link.count():
                    print(
                        "  Comando finale: link contenente "
                        "images/accediAiDati.gif"
                    )
                    return parent_link.first
                return img
        except Exception:
            continue

    loc = scheda.locator(
        'input[type="submit"], input[type="image"], button, a'
    )
    for i in range(loc.count()):
        el = loc.nth(i)
        try:
            blob = " ".join([
                norm(el.inner_text() or ""),
                norm(el.get_attribute("value") or ""),
                norm(el.get_attribute("alt") or ""),
                norm(el.get_attribute("title") or ""),
                norm(el.get_attribute("src") or ""),
            ])
            compact = re.sub(r"[^A-Z0-9]", "", blob)
            if "ACCEDI AI DATI" in blob or "ACCEDIAIDATI" in compact:
                return el
        except Exception:
            continue
    return None


def rendered_data_to_file(page, target):
    for fr in all_frames(page):
        try:
            body = fr.locator("body").inner_text()
        except Exception:
            continue
        lines = [x for x in body.splitlines() if x.strip()]
        sample = "\n".join(lines[:10])
        if len(lines) >= 3 and any(sep in sample for sep in (";", "\t", ",")):
            target.write_text(body, encoding="utf-8")
            return True
    return False


def download_current(page, target):
    """
    Flusso ARPAL corretto (v1.3).

    Il click su "Accedi ai dati" NON genera un download diretto:
      javascript:invia('ESTRAZ', <id>, document.forms[0])
    apre un popup PubAccessoDatiMeteoPost.asp che contiene un link assoluto:
      https://ambientepub.regione.liguria.it/SiraQualMeteo/report/<id>.csv

    La v1.2 attendeva un evento download sulla pagina principale e, in fallback,
    salvava per errore il testo del frame catalogo stazioni. La v1.3:
      1) attende il popup;
      2) estrae il vero href .csv (o il hidden UrlFileDat);
      3) scarica i byte con la request API dello stesso browser context;
      4) salva solo la risposta del vero report CSV;
      5) NON usa più rendered_data_to_file().
    """
    button = find_access_button(page)
    if button is None:
        raise RuntimeError("Comando finale 'Accedi ai dati' non trovato.")

    popup = None

    try:
        with page.expect_popup(timeout=30_000) as popup_info:
            button.click()
        popup = popup_info.value
    except Exception as exc:
        # Il portale può aver aperto comunque una nuova pagina prima che
        # expect_popup la intercetti: cerchiamola nel context.
        page.wait_for_timeout(1500)
        candidates = [
            p for p in page.context.pages
            if p is not page and "PubAccessoDatiMeteoPost.asp" in p.url
        ]
        if candidates:
            popup = candidates[-1]
        else:
            raise RuntimeError(
                "Il click 'Accedi ai dati' non ha aperto il popup ARPAL. "
                f"Dettaglio: {exc}"
            )

    try:
        popup.wait_for_load_state("domcontentloaded", timeout=NAV_TIMEOUT)
    except Exception:
        pass

    print(f"  Popup ARPAL: {popup.url}")

    def extract_csv_url():
        # 1) link esplicito "Scaricare qui il file..."
        try:
            links = popup.locator('a[href*="/SiraQualMeteo/report/"][href$=".csv"]')
            if links.count():
                return links.first.get_attribute("href")
        except Exception:
            pass

        # 2) qualunque href .csv
        try:
            links = popup.locator('a[href$=".csv"]')
            if links.count():
                return links.first.get_attribute("href")
        except Exception:
            pass

        # 3) hidden UrlFileDat
        try:
            hidden = popup.locator('input[name="UrlFileDat"]')
            if hidden.count():
                val = hidden.first.get_attribute("value")
                if val and ".csv" in val.lower():
                    return val
        except Exception:
            pass

        return None

    csv_url = wait_until(extract_csv_url, timeout_s=30, interval=0.5)

    if not csv_url:
        body = ""
        try:
            body = norm(popup.locator("body").inner_text(timeout=5000))
        except Exception:
            pass

        if any(x in body for x in (
            "NESSUN DATO",
            "DATI NON DISPONIBILI",
            "NON SONO PRESENTI DATI",
            "NESSUN VALORE",
        )):
            try:
                popup.close()
            except Exception:
                pass
            return "no_data"

        diag = save_diag(
            popup,
            "popup_without_csv_link",
            {"popup_url": popup.url},
        )
        raise RuntimeError(
            "Popup ARPAL aperto ma nessun link .csv trovato. "
            f"Diagnostica: {diag}"
        )

    print(f"  CSV reale: {csv_url}")

    # Scarica il report generato mantenendo cookie/sessione del browser context.
    resp = page.context.request.get(
        csv_url,
        timeout=DOWNLOAD_TIMEOUT,
        fail_on_status_code=False,
    )

    status = resp.status
    ctype = resp.headers.get("content-type", "")
    body = resp.body()

    print(
        f"  GET report: HTTP {status} | {ctype} | "
        f"{len(body)/1024:.1f} kB"
    )

    if status != 200:
        raise RuntimeError(
            f"Download del report CSV fallito: HTTP {status} | {csv_url}"
        )

    if not body or len(body) < MIN_BYTES:
        raise RuntimeError(
            f"Report CSV vuoto/troppo piccolo: {len(body)} byte | {csv_url}"
        )

    # Evita di salvare HTML di errore con estensione csv.
    head = body[:5000].decode("latin-1", errors="ignore").lower()
    if "<html" in head or "<!doctype html" in head:
        raise RuntimeError(
            "Il link report ha restituito HTML invece del CSV."
        )

    target.write_bytes(body)

    try:
        popup.close()
    except Exception:
        pass

    return "report_csv"


# =============================================================================
# FILE QC
# =============================================================================

def inspect_download(path):
    """
    Controllo del file ARPAL scaricato.

    REGOLE:
    - rifiuta HTML;
    - rifiuta il catalogo stazioni (il falso CSV da ~6.4 kB visto nelle v1.2/v5.4);
    - cerca date nel formato gg/mm/aaaa o aaaa-mm-gg;
    - non richiede un numero minimo fisso di righe, perché alcune serie possono
      essere sparse o iniziare tardi.
    """
    raw = path.read_bytes()
    sample = raw[:500000]

    text = None
    enc = None
    for e in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = sample.decode(e)
            enc = e
            break
        except Exception:
            pass

    if text is None:
        text = sample.decode("latin-1", errors="replace")
        enc = "latin-1"

    lower = text.lower()
    is_html = "<html" in lower or "<!doctype html" in lower

    lines = [x.strip() for x in text.splitlines() if x.strip()]
    first_lines = [norm(x) for x in lines[:80]]

    # Il falso file osservato aveva:
    # Stazione
    # ALPE GORRETO (Genova)
    # ...
    # DAVAGNA (Genova)
    catalog_like = False
    if first_lines:
        if first_lines[0] == "STAZIONE":
            station_name_hits = sum(
                1 for x in first_lines[1:]
                if "(" in x and ")" in x
            )
            # norm() rimuove poco, quindi usiamo anche l'originale.
            station_name_hits = max(
                station_name_hits,
                sum(1 for x in lines[1:80] if "(" in x and ")" in x)
            )
            if station_name_hits >= 5:
                catalog_like = True

    date_patterns = [
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
    ]
    date_hits = sum(len(re.findall(p, text)) for p in date_patterns)

    suspicious = bool(is_html or catalog_like or date_hits == 0)

    return {
        "bytes": path.stat().st_size,
        "encoding_guess": enc,
        "html_like": is_html,
        "catalog_like": catalog_like,
        "date_hits_sample": date_hits,
        "line_count_sample": len(lines),
        "suspicious": suspicious,
    }


# =============================================================================
# PORTAL / DOWNLOAD
# =============================================================================

def prepare_catalog(context, force_online=False):
    if not force_online:
        cat = load_existing_catalog()
        if cat:
            return cat

    page = context.new_page()
    try:
        enter_station_mode(page)
        catalog = get_station_catalog(page)
        save_catalog(catalog)
        return catalog
    finally:
        page.close()


def parameter_available_probe(context, cfg, station_rec):
    page = context.new_page()
    try:
        enter_station_mode(page)
        select_station(page, station_rec)
        select_frequency_hourly(page)
        try:
            chosen = select_parameter(page, cfg["parameter_terms"])
            return True, chosen["text"]
        except Exception as exc:
            return False, str(exc)
    finally:
        page.close()


def download_one(context, cfg, station_rec, year, force=False):
    folder = OUT / cfg["slug"]
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{cfg['slug']}_{year}_09-12.csv"

    if file_ok(target) and not force:
        qc = inspect_download(target)
        print(
            f"SKIP {cfg['slug']} {year}: già presente "
            f"({qc['bytes']/1024:.1f} kB, "
            f"date_hits_sample={qc['date_hits_sample']})"
        )
        return "skip"

    last_exc = None

    # Il portale ASP usa frame che si ricaricano in modo asincrono.
    # Un singolo "Param assente" dopo che la stessa stazione era stata
    # verificata correttamente è un race/transiente, non prova di assenza
    # del parametro. Rifacciamo l'intera preparazione fino a 3 volte.
    for attempt in range(1, 4):
        page = context.new_page()
        try:
            print(f"\n{year} | {cfg['slug']} | tentativo {attempt}/3")
            enter_station_mode(page)
            select_station(page, station_rec)
            select_frequency_hourly(page)
            param = select_parameter(page, cfg["parameter_terms"])
            output = select_csv(page)
            fill_period(page, f"01/09/{year}", f"31/12/{year}")

            result = download_current(page, target)

            if result == "no_data":
                print(f"  NO DATA {year}")
                if target.exists():
                    target.unlink()
                log_manifest(
                    status="no_data",
                    basin=cfg["basin"],
                    slug=cfg["slug"],
                    year=year,
                    station_code=station_rec["code"],
                    station_name=station_rec["name"],
                )
                return "no_data"

            if not file_ok(target):
                raise RuntimeError(
                    f"Output prodotto ma file assente/troppo piccolo: {target}"
                )

            qc = inspect_download(target)

            print(
                f"  OK {target.name}: {target.stat().st_size/1024:.1f} kB | "
                f"date_hits_sample={qc['date_hits_sample']} | "
                f"suspicious={qc['suspicious']}"
            )

            log_manifest(
                status="ok_suspicious" if qc["suspicious"] else "ok",
                basin=cfg["basin"],
                slug=cfg["slug"],
                kind=cfg["kind"],
                year=year,
                station_code=station_rec["code"],
                station_name=station_rec["name"],
                parameter=param["text"],
                output=output["text"],
                path=str(target),
                bytes=target.stat().st_size,
                mode=result,
                file_qc=qc,
            )
            return "ok_suspicious" if qc["suspicious"] else "ok"

        except Exception as exc:
            last_exc = exc
            print(f"  Tentativo {attempt}/3 fallito: {exc}")

            if attempt == 3:
                diag = save_diag(
                    page,
                    f"{cfg['slug']}_{year}",
                    {
                        "error": str(exc),
                        "target": cfg,
                        "station": station_rec,
                        "year": year,
                        "attempt": attempt,
                    },
                )
                print(f"  ERRORE definitivo {year}: {exc}")
                print(f"  Diagnostica: {diag}.*")

                log_manifest(
                    status="error",
                    basin=cfg["basin"],
                    slug=cfg["slug"],
                    year=year,
                    station_code=station_rec["code"],
                    station_name=station_rec["name"],
                    error=str(exc),
                    diagnostics=str(diag),
                )
                return "error"

            # breve pausa prima di ricreare pagina e frame da zero
            time.sleep(2.0)

        finally:
            try:
                page.close()
            except Exception:
                pass

    raise RuntimeError(f"Errore inatteso dopo retry: {last_exc}")


# =============================================================================
# AUDIT
# =============================================================================

def audit():
    setup()

    files = list(OUT.rglob("*.csv"))
    stats = defaultdict(lambda: {
        "files": 0,
        "suspicious": 0,
        "bytes": 0,
    })

    for p in files:
        slug = p.parent.name
        q = inspect_download(p)
        stats[slug]["files"] += 1
        stats[slug]["bytes"] += q["bytes"]
        if q["suspicious"]:
            stats[slug]["suspicious"] += 1

    lines = [
        "=" * 100,
        "LIGURIA REGIONAL GROUND TRUTH v1.4 - AUDIT",
        "=" * 100,
        f"CSV totali: {len(files)}",
        "",
        "target                                        files suspicious size_MB",
    ]

    for slug in sorted(stats):
        s = stats[slug]
        lines.append(
            f"{slug:45s} {s['files']:5d} {s['suspicious']:10d} "
            f"{s['bytes']/1024/1024:7.2f}"
        )

    if MANIFEST.exists():
        rows = []
        for line in MANIFEST.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
        counts = defaultdict(int)
        for r in rows:
            counts[r.get("status", "unknown")] += 1
        lines += ["", "Manifest status:"]
        for k in sorted(counts):
            lines.append(f"  {k:20s} {counts[k]:5d}")

    lines += [
        "",
        "NOTA:",
        "- suspicious non significa automaticamente corrotto; indica che il",
        "  controllo leggero non ha trovato timestamp nel campione o ha visto HTML.",
        "- Dopo il download completo faremo un parser specifico ARPAL per validare",
        "  righe, date, valori, copertura temporale e duplicati.",
        "=" * 100,
    ]

    AUDIT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    setup()

    if args.audit:
        audit()
        return

    if args.start_year > args.end_year:
        raise ValueError("--start-year deve essere <= --end-year")

    targets = TARGETS[:]

    # CENTA è già in larga parte coperto dal ramo Tanaro-Arroscia.
    if not args.include_centa and args.basin != "CENTA":
        targets = [t for t in targets if t["basin"] != "CENTA"]

    if args.basin:
        targets = [t for t in targets if t["basin"] == args.basin]

    if args.only:
        targets = [t for t in targets if t["slug"] == args.only]
        if not targets:
            raise ValueError(
                f"Target {args.only!r} non trovato nel set corrente."
            )

    print("=" * 100)
    print("LIGURIA REGIONAL GROUND-TRUTH DOWNLOADER v1.4.1")
    print("Motore ARPAL v1.4.1: parametro stretto + attese/retry robusti sui frame dinamici.")
    print(f"Output : {OUT}")
    print(f"Target : {len(targets)}")
    print(f"Periodo: {args.start_year}-{args.end_year}, settembre-dicembre")
    print("Browser: HEADLESS" if not args.headed else "Browser: VISIBILE")
    print("=" * 100)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        context = browser.new_context(
            accept_downloads=True,
            locale="it-IT",
        )

        catalog = prepare_catalog(context)

        print("\nRISOLUZIONE TARGET")
        resolved = []
        for cfg in targets:
            rec = resolve_station(catalog, cfg)
            resolved.append((cfg, rec))
            if rec:
                print(
                    f"  {cfg['slug']:<38s} -> "
                    f"{rec['code']:<10s} {rec['name']}"
                )
            else:
                print(
                    f"  {cfg['slug']:<38s} -> NON TROVATO "
                    f"{cfg['station_terms']}"
                )
                log_manifest(
                    status="station_not_found",
                    basin=cfg["basin"],
                    slug=cfg["slug"],
                    terms=cfg["station_terms"],
                )

        if args.discover:
            print("\nModalità --discover: nessuna serie scaricata.")
            context.close()
            browser.close()
            return

        years = (
            [2020]
            if args.test
            else list(range(args.start_year, args.end_year + 1))
        )

        for cfg, station_rec in resolved:
            if station_rec is None:
                continue

            print()
            print("#" * 100)
            print(f"TARGET: {cfg['slug']} | {cfg['basin']} | {cfg['kind']}")
            print("#" * 100)

            # v1.4.1 non esegue più un probe separato del parametro:
            # raddoppiava inutilmente la navigazione e poteva lasciare il
            # vecchio portale in uno stato transiente. Ogni anno viene
            # preparato e validato direttamente con retry robusto.
            for year in years:
                download_one(
                    context,
                    cfg,
                    station_rec,
                    year,
                    force=args.force,
                )
                time.sleep(0.8)

        context.close()
        browser.close()

    audit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrotto. I file già completati restano validi.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nFATAL: {exc}", file=sys.stderr)
        sys.exit(1)
