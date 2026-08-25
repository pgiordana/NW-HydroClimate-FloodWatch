#!/usr/bin/env python3
"""
Tanaro–Arroscia | ARPAL / OMIRL historical downloader v5.4
===========================================================

Questa versione nasce dalla struttura REALE osservata nel portale:

FRAME Punto
-----------
select name="Ubic"
    es. ME00025 = COLLE DI NAVA (Imperia)
    onchange="inviaTema(document.forms[0],'STAZIONE',false);"

select name="Frequenza"
    HH = Orario
    GG = Giornaliero
    ...
    onchange="inviaTema(document.forms[0],'STAZIONE',true)"

FRAME Scheda
------------
select name="Param"
    es. PRECPBIWC1 = PRECIPITAZIONE - Precipitazione Cumulata

select name="TipoOutput"
    HTML  = Tabella html
    XLS   = File .csv (MS Excel)
    ASCII = File ascii

select name="Separatore"
    TAB, ;, ,, @, AMP

La v5:
1. apre il frameset ufficiale;
2. entra in modalità STAZIONE;
3. salva l'INTERO catalogo stazioni ARPAL in JSON/CSV;
4. risolve i target usando prima il codice esatto, poi il nome;
5. seleziona stazione -> Frequenza HH -> parametro -> XLS/CSV;
6. compila il periodo;
7. scarica settembre-dicembre anno per anno;
8. non presume che 1987 abbia dati: un anno vuoto viene saltato e il ciclo continua;
9. è restart-safe;
10. salva diagnostica completa dei frame in caso di errore strutturale.

Target iniziali
---------------
- Colle di Nava          ME00025 : precipitazione oraria
- Pieve di Teco          ME00090 : precipitazione oraria
- Pieve di Teco (IDRO)   ME00342 : livello idrometrico (se disponibile)
- Ranzo                            : precipitazione oraria (risolto dal catalogo)
- Pogli / Ortovero                : livello idrometrico (risolto dal catalogo)

Nota:
Mendatica non compare tra le prime opzioni osservate del catalogo ARPAL;
Colle di Nava è il target montano certo e pertinente già verificato.

Prerequisiti:
    pip install -U playwright
    python -m playwright install chromium

SCOPERTA catalogo, senza scaricare serie:
    python download_arpal_tanaro_arroscia_v5.py --discover

TEST 2020 di tutti i target:
    python download_arpal_tanaro_arroscia_v5.py --test

TEST singolo:
    python download_arpal_tanaro_arroscia_v5.py --test --only colle_nava_precip

Completo:
    python download_arpal_tanaro_arroscia_v5.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout


# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "tanaro_arroscia" / "observations" / "arpal_omirl"
OUT = BASE / "hourly"
DIAG = BASE / "_diagnostics"
CATALOG_DIR = BASE / "catalog"
MANIFEST = BASE / "arpal_download_manifest_v5_4.jsonl"

PORTAL_URL = (
    "https://ambientepub.regione.liguria.it/"
    "SiraQualMeteo/script/PubAccessoDatiMeteo.asp"
)

START_YEAR = 1987
END_YEAR = 2025

TARGETS = [
    {
        "slug": "colle_nava_precip",
        "station_code": "ME00025",
        "station_terms": ["COLLE DI NAVA"],
        "parameter_terms": ["PRECIPITAZIONE"],
        "kind": "precipitation",
    },
    {
        "slug": "pieve_teco_precip",
        "station_code": "ME00090",
        "station_terms": ["PIEVE DI TECO"],
        "parameter_terms": ["PRECIPITAZIONE"],
        "kind": "precipitation",
    },
    {
        "slug": "pieve_teco_level",
        "station_code": "ME00342",
        "station_terms": ["PIEVE DI TECO (IDRO)", "PIEVE DI TECO"],
        "parameter_terms": [
            "LIVELLO IDROMETRICO",
            "ALTEZZA IDROMETRICA",
            "IDROMETR",
            "LIVELLO",
        ],
        "kind": "water_level",
    },
    {
        "slug": "ranzo_precip",
        "station_code": None,
        "station_terms": ["RANZO"],
        "parameter_terms": ["PRECIPITAZIONE"],
        "kind": "precipitation",
    },
    {
        "slug": "pogli_level",
        "station_code": None,
        "station_terms": ["POGLI", "ORTOVERO"],
        "parameter_terms": [
            "LIVELLO IDROMETRICO",
            "ALTEZZA IDROMETRICA",
            "IDROMETR",
            "LIVELLO",
        ],
        "kind": "water_level",
    },
]

MIN_BYTES = 50
NAV_TIMEOUT = 90_000
DOWNLOAD_TIMEOUT = 120_000


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
                    "options": opts,  # NIENTE TRONCAMENTO in v5
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
# ENTER STATION MODE
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

    tipo_frame, radio = found
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


# =============================================================================
# STATION CATALOG
# =============================================================================

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

    return cpath


def resolve_station(catalog, cfg):
    code = cfg.get("station_code")
    if code:
        for rec in catalog:
            if rec["code"] == code:
                return rec

    terms = [norm(x) for x in cfg["station_terms"]]

    # match esatto sul nome senza provincia
    for rec in catalog:
        name = norm(re.sub(r"\s*\([^()]*\)\s*$", "", rec["name"]))
        if any(name == t for t in terms):
            return rec

    # contenimento
    for rec in catalog:
        name = norm(rec["name"])
        if any(t in name for t in terms):
            return rec

    return None


def print_target_resolution(catalog):
    print("\nRISOLUZIONE TARGET")
    for cfg in TARGETS:
        rec = resolve_station(catalog, cfg)
        if rec:
            print(
                f"  {cfg['slug']:<24} -> "
                f"{rec['code']:<10} {rec['name']}"
            )
        else:
            print(
                f"  {cfg['slug']:<24} -> NON TROVATO "
                f"{cfg['station_terms']}"
            )


# =============================================================================
# SELECTION FLOW
# =============================================================================

def select_by_value_with_change(sel, value):
    """
    select_option genera normalmente input/change; il vecchio sito usa
    onchange=inviaTema(...), quindi lasciamo che venga eseguito nativamente.
    """
    sel.select_option(value=value)


def select_station(page, station_rec):
    punto = frame_by_name(page, "Punto")
    ubic = punto.locator('select[name="Ubic"]')
    before = frame_by_name(page, "Scheda")
    before_url = before.url if before else ""

    select_by_value_with_change(ubic, station_rec["code"])
    print(f"  Stazione: {station_rec['code']} | {station_rec['name']}")

    # La scelta della stazione ricarica i frame. Prima aspettiamo che Punto
    # torni con la stazione ancora selezionata e con Frequenza disponibile.
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

    # Poi aspettiamo il nuovo Scheda con almeno un parametro.
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
    """
    Dopo la scelta di Ubic il vecchio portale ARPAL ricarica il frame Punto.
    Non possiamo quindi cercare Frequenza immediatamente: aspettiamo che il
    nuovo frame sia tornato stabile e che select[name="Frequenza"] contenga HH.
    """
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

    found = wait_until(frequency_ready, timeout_s=20, interval=0.35)
    if not found:
        raise RuntimeError(
            "Il frame Punto non ha ripristinato Frequenza=HH "
            "entro 20 secondi dopo la scelta della stazione."
        )

    punto, freq, vals = found
    freq.select_option(value="HH")
    print("  Frequenza: HH | Orario")

    # Anche la frequenza ha onchange=inviaTema(...,true) e può ricaricare
    # Punto/Scheda: attendiamo che il frame Scheda torni disponibile.
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

    wait_until(scheda_ready, timeout_s=20, interval=0.35)
    page.wait_for_timeout(300)


def select_parameter(page, terms):
    scheda = frame_by_name(page, "Scheda")
    param = scheda.locator('select[name="Param"]')
    if param.count() == 0:
        raise RuntimeError("Param assente nel frame Scheda.")

    nt = [norm(t) for t in terms]
    opts = options(param)

    chosen = None
    for o in opts:
        txt = norm(o["text"])
        if o["value"] and any(t in txt for t in nt):
            chosen = o
            break

    if chosen is None:
        avail = [o["text"] for o in opts]
        raise RuntimeError(
            f"Parametro {terms} non disponibile. Disponibili: {avail}"
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

    # Dal diagnostico reale: XLS = "File .csv (MS Excel)"
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
        # Nel portale ARPAL il select Separatore resta nel DOM anche quando
        # TipoOutput=XLS (CSV Excel), ma in quel caso è nascosto e NON va
        # selezionato. Playwright, correttamente, rifiuta di interagire con
        # controlli invisibili. Lo usiamo solo quando il portale lo mostra.
        try:
            visible = sep.is_visible()
            enabled = sep.is_enabled()
        except Exception:
            visible = False
            enabled = False

        if visible and enabled:
            sep_vals = {o["value"]: o["text"] for o in options(sep)}
            if ";" in sep_vals:
                sep.select_option(value=";")
                print("  Separatore: ; | Punto e virgola")
        else:
            print("  Separatore: non richiesto per output XLS/CSV (controllo nascosto)")

    return chosen


# =============================================================================
# PERIOD FIELDS
# =============================================================================

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

    # fallback per name/id che contengono Data/Ora
    if not (start_ok and end_ok):
        text_inputs = scheda.locator('input[type="text"], input:not([type])')
        names = []
        for i in range(text_inputs.count()):
            el = text_inputs.nth(i)
            names.append((
                i,
                el.get_attribute("name"),
                el.get_attribute("id"),
                el.get_attribute("value"),
            ))

        raise RuntimeError(
            "Campi periodo non riconosciuti. "
            f"Input disponibili: {names}"
        )

    print(f"  Periodo: {start_date} 00:00 -> {end_date} 23:59")


# =============================================================================
# FINAL DOWNLOAD
# =============================================================================

def find_access_button(page):
    """
    Il portale ARPAL non usa un input[type=image] per "Accedi ai dati":
    la pagina reale contiene un elemento IMG con src tipo
        /SiraQualMeteo/images/accediAiDati.gif
    eventualmente racchiuso in un link.

    Restituiamo quindi, in ordine:
    1) il link antenato dell'immagine, se presente;
    2) l'immagine stessa;
    3) fallback sui vecchi submit/button.
    """
    scheda = frame_by_name(page, "Scheda")
    if scheda is None:
        return None

    # Caso reale verificato sul portale.
    imgs = scheda.locator('img')
    for i in range(imgs.count()):
        img = imgs.nth(i)
        try:
            src = norm(img.get_attribute("src") or "")
            alt = norm(img.get_attribute("alt") or "")
            title = norm(img.get_attribute("title") or "")
            blob = " ".join([src, alt, title])
            compact = re.sub(r"[^A-Z0-9]", "", blob)

            if "ACCEDIAIDATI" in compact:
                # Se l'IMG è dentro <a>, clicchiamo il link: è più fedele
                # all'HTML originale e preserva href/onclick del sito.
                parent_link = img.locator("xpath=ancestor::a[1]")
                if parent_link.count():
                    print(
                        "  Comando finale: link contenente "
                        "images/accediAiDati.gif"
                    )
                    return parent_link.first

                print(
                    "  Comando finale: immagine "
                    "images/accediAiDati.gif"
                )
                return img
        except Exception:
            continue

    # Fallback per eventuali varianti future.
    loc = scheda.locator(
        'input[type="submit"], input[type="image"], button, a'
    )

    for i in range(loc.count()):
        el = loc.nth(i)
        try:
            typ = norm(el.get_attribute("type") or "")
            src = norm(el.get_attribute("src") or "")
            alt = norm(el.get_attribute("alt") or "")
            title = norm(el.get_attribute("title") or "")
            value = norm(el.get_attribute("value") or "")
            text = norm(el.inner_text() or "")

            blob = " ".join([text, value, alt, title, src])
            compact = re.sub(r"[^A-Z0-9]", "", blob)

            if (
                "ACCEDI AI DATI" in blob
                or "ACCEDIAIDATI" in compact
            ):
                return el

        except Exception:
            continue

    return None

def rendered_data_to_file(page, target):
    # Se il server apre il contenuto in un frame/pagina invece di download.
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
    button = find_access_button(page)
    if button is None:
        raise RuntimeError(
            "Comando finale 'Accedi ai dati' non trovato "
            "(né IMG né link/submit)."
        )

    # Caso preferito: il server risponde con Content-Disposition/download.
    try:
        with page.expect_download(timeout=DOWNLOAD_TIMEOUT) as info:
            button.click()
        dl = info.value
        dl.save_as(str(target))
        return "download"
    except PWTimeout:
        # Il click è comunque avvenuto: alcuni vecchi ASP rispondono
        # navigando/rendendo il contenuto invece di generare un download.
        pass

    page.wait_for_timeout(1800)

    if rendered_data_to_file(page, target):
        return "rendered"

    # rileva messaggi di assenza dati
    scheda = frame_by_name(page, "Scheda")
    txt = ""
    if scheda:
        try:
            txt = norm(scheda.locator("body").inner_text())
        except Exception:
            pass

    if any(x in txt for x in (
        "NESSUN DATO",
        "DATI NON DISPONIBILI",
        "NON SONO PRESENTI DATI",
        "NESSUN VALORE",
    )):
        return "no_data"

    raise RuntimeError(
        "Nessun download/tabella e nessun messaggio 'no data' riconosciuto."
    )


# =============================================================================
# ONE YEAR / TARGET
# =============================================================================

def prepare_portal(context):
    page = context.new_page()
    enter_station_mode(page)

    catalog = get_station_catalog(page)
    save_catalog(catalog)

    return page, catalog


def download_one(context, cfg, station_rec, year, force=False):
    folder = OUT / cfg["slug"]
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{cfg['slug']}_{year}_09-12.csv"

    if file_ok(target) and not force:
        print(f"SKIP {cfg['slug']} {year}: già presente")
        return "skip"

    page = context.new_page()

    try:
        print(f"\n{year} | {cfg['slug']}")
        enter_station_mode(page)

        select_station(page, station_rec)
        select_frequency_hourly(page)
        param = select_parameter(page, cfg["parameter_terms"])
        output = select_csv(page)

        fill_period(page, f"01/09/{year}", f"31/12/{year}")

        result = download_current(page, target)

        if result == "no_data":
            print(f"  NO DATA {year}: la stazione non ha dati nel periodo.")
            log_manifest(
                status="no_data",
                slug=cfg["slug"],
                year=year,
                station_code=station_rec["code"],
                station_name=station_rec["name"],
            )
            if target.exists():
                target.unlink()
            return "no_data"

        if not file_ok(target):
            raise RuntimeError(
                f"Output prodotto ma file assente/troppo piccolo: {target}"
            )

        print(
            f"  OK {target.name}: "
            f"{target.stat().st_size/1024:.1f} kB"
        )

        log_manifest(
            status="ok",
            slug=cfg["slug"],
            year=year,
            station_code=station_rec["code"],
            station_name=station_rec["name"],
            parameter=param["text"],
            output=output["text"],
            path=str(target),
            bytes=target.stat().st_size,
            mode=result,
        )

        return "ok"

    except Exception as exc:
        diag = save_diag(
            page,
            f"{cfg['slug']}_{year}",
            {
                "error": str(exc),
                "target": cfg,
                "station": station_rec,
                "year": year,
            },
        )

        print(f"  ERRORE {year}: {exc}")
        print(f"  Diagnostica: {diag}.*")

        log_manifest(
            status="error",
            slug=cfg["slug"],
            year=year,
            station_code=station_rec["code"],
            station_name=station_rec["name"],
            error=str(exc),
            diagnostics=str(diag),
        )

        return "error"

    finally:
        page.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    setup()

    if args.start_year > args.end_year:
        raise ValueError("--start-year deve essere <= --end-year")

    targets = TARGETS
    if args.only:
        targets = [x for x in TARGETS if x["slug"] == args.only]
        if not targets:
            raise ValueError(
                f"Target {args.only!r} sconosciuto. "
                f"Disponibili: {[x['slug'] for x in TARGETS]}"
            )

    print("=" * 96)
    print("TANARO–ARROSCIA | ARPAL/OMIRL DOWNLOADER v5.4")
    print("Struttura del form basata sul diagnostico reale ARPAL.")
    print(f"Portale: {PORTAL_URL}")
    print(f"Output : {OUT}")
    print("=" * 96)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        context = browser.new_context(
            accept_downloads=True,
            locale="it-IT",
        )

        # DISCOVERY INIZIALE
        page, catalog = prepare_portal(context)
        print_target_resolution(catalog)
        page.close()

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

        for cfg in targets:
            station_rec = resolve_station(catalog, cfg)

            print()
            print("#" * 96)
            print(f"TARGET: {cfg['slug']}")
            print("#" * 96)

            if station_rec is None:
                print(
                    f"NON TROVATO nel catalogo ARPAL: "
                    f"{cfg['station_terms']}. Target saltato."
                )
                log_manifest(
                    status="station_not_found",
                    slug=cfg["slug"],
                    terms=cfg["station_terms"],
                )
                continue

            print(
                f"RISOLTO -> {station_rec['code']} | "
                f"{station_rec['name']}"
            )

            errors_in_a_row = 0

            for idx, year in enumerate(years, start=1):
                status = download_one(
                    context,
                    cfg,
                    station_rec,
                    year,
                    force=args.force,
                )

                if status == "error":
                    errors_in_a_row += 1
                else:
                    errors_in_a_row = 0

                # Un singolo anno può essere privo di dati: NON interrompere.
                # Tre errori strutturali consecutivi invece indicano che il
                # form/parametro non è gestito correttamente.
                if errors_in_a_row >= 3:
                    print(
                        "  Tre errori consecutivi: interrompo questo target "
                        "per evitare richieste inutili."
                    )
                    break

                time.sleep(0.8)

        context.close()
        browser.close()

    print()
    print("=" * 96)
    print("FINE ARPAL v5.4")
    print(f"Catalogo: {CATALOG_DIR / 'arpal_station_catalog.csv'}")
    print(f"Manifest: {MANIFEST}")
    print("=" * 96)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrotto. I file già completati restano validi.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nFATAL: {exc}", file=sys.stderr)
        sys.exit(1)
