#!/usr/bin/env python3
"""
Tanaro–Arroscia | ARPA Piemonte historical downloader v1.1
===========================================================

QUARTO TERMINALE DELLA PIPELINE

Scopo
-----
Scaricare dalla Banca Dati Storica REST di ARPA Piemonte le osservazioni
giornaliere utili per il versante Tanaro/Tanarello:

1) RETE IDROLOGICA
   - livello idrometrico medio
   - portata media
   - metadata stazione / corso d'acqua / bacino
   - priorità: Ponte di Nava / Ormea / Garessio / alto Tanaro

2) RETE METEOROLOGICA
   - precipitazione totale giornaliera (ptot)
   - altre variabili giornaliere disponibili vengono conservate nel raw
   - priorità geografica: alto bacino Tanaro/Tanarello
   - target noti: Ponte di Nava Tanaro, Piaggia, Colle San Bernardo,
     Monte Berlino e altre stazioni nel box geografico

Periodo standard
----------------
1987–2025, settembre–dicembre.
L'API viene interrogata sul periodo richiesto e il filtro autunnale viene
applicato nuovamente in locale per sicurezza.

Fonti/API
---------
ARPA Piemonte Banca Dati Storica:
    https://utility.arpa.piemonte.it/docs/

Hydro:
    /meteoidro/stazione_idrologica/
    /meteoidro/dati_giornalieri_idro/

Meteo:
    /meteoidro/stazione_meteorologica/   (probe automatico)
    /meteoidro/dati_giornalieri_meteo/

Il programma NON usa e NON aggira i moduli con AntiSpam per dati
orari/suborari.

Installazione
-------------
    pip install -U requests pandas

Discovery soltanto
------------------
    python download_arpa_piemonte_tanaro.py --discover

Test 2020
---------
    python download_arpa_piemonte_tanaro.py --test

Completo
--------
    python download_arpa_piemonte_tanaro.py

Solo idrologia
--------------
    python download_arpa_piemonte_tanaro.py --hydro-only

Solo meteo
----------
    python download_arpa_piemonte_tanaro.py --meteo-only

Resume
------
- CSV validi già presenti vengono saltati.
- JSON raw vengono conservati.
- retry automatico su 429/500/502/503/504 e timeout.
- un errore su una stazione NON ferma le altre.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

try:
    import pandas as pd
except ImportError:
    pd = None


# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path(__file__).resolve().parent

BASE = ROOT / "tanaro_arroscia" / "observations" / "arpa_piemonte"
HYDRO_DIR = BASE / "daily_hydro"
METEO_DIR = BASE / "daily_meteo"
CATALOG_DIR = BASE / "catalog"
DIAG_DIR = BASE / "_diagnostics"
MANIFEST = BASE / "arpa_download_manifest_v1_1.jsonl"

API_ROOT = "https://utility.arpa.piemonte.it/meteoidro"

HYDRO_STATIONS_URL = f"{API_ROOT}/stazione_idrologica/"
HYDRO_DATA_URL = f"{API_ROOT}/dati_giornalieri_idro/"

# La nomenclatura MeteoWeb normalmente usa stazione_meteorologica.
# Manteniamo fallback automatici per eventuali alias.
METEO_STATION_CANDIDATES = [
    f"{API_ROOT}/stazione_meteorologica/",
    f"{API_ROOT}/stazione_meteo/",
]
METEO_DATA_URL = f"{API_ROOT}/dati_giornalieri_meteo/"

START_YEAR = 1987
END_YEAR = 2025

# Alto Tanaro/Tanarello, con margine prudente.
# Serve per discovery, non per definire il bacino idrologico finale.
BBOX = {
    "south": 43.92,
    "north": 44.35,
    "west": 7.38,
    "east": 8.18,
}

# Stazioni note / altamente pertinenti per il nostro studio.
PRIORITY_TERMS = [
    "PONTE DI NAVA",
    "PONTE DI NAVA TANARO",
    "PIAGGIA",
    "ORMEA",
    "GARESSIO",
    "COLLE SAN BERNARDO",
    "MONTE BERLINO",
    "TANARELLO",
]

# Per l'idrologia includiamo anche stazioni esplicitamente sul Tanaro.
HYDRO_RIVER_TERMS = [
    "TANARO",
    "TANARELLO",
]

REQUEST_TIMEOUT = 90
MAX_RETRIES = 8
MIN_FILE_BYTES = 60

TRANSIENT_CODES = {429, 500, 502, 503, 504}


# =============================================================================
# TARGET CURATI PER LO STUDIO TANARO–ARROSCIA
# =============================================================================
#
# Dopo il test reale del 21/08/2026, il filtro geografico generico risultava
# troppo ampio: includeva Pesio, Gesso, Vermenagna, Stura di Demonte e Bormida.
# Per il confronto interbacino ci concentriamo invece sull'alto Tanaro /
# Tanarello e sulle stazioni immediatamente utili alla propagazione verso
# Ormea/Garessio.

HYDRO_TARGET_CODES = {
    # punto chiave immediatamente a valle della confluenza dell'alto Tanaro
    "PIE-004155-901": "PONTE DI NAVA TANARO",
    # ramo sorgentizio / controllo a monte
    "PIE-004155-902": "PORNASSINO NEGRONE",
    # controllo di propagazione a valle
    "PIE-004095-900": "GARESSIO TANARO",
}

METEO_TARGET_CODES = {
    # alta valle / sorgenti Tanarello-Negrone
    "PIE-004031-900": "UPEGA",
    "PIE-004031-901": "PIAGGIA",
    # alta valle presso Viozene / Tanaro
    "PIE-004155-900": "VIOZENE",
    # stazione chiave presso l'idrometro
    "PIE-004155-901": "PONTE DI NAVA TANARO",
    # controllo orografico e propagazione verso Garessio
    "PIE-004095-901": "COLLE SAN BERNARDO",
    "PIE-004095-902": "MONTE BERLINO",
    # utile come sensore locale nell'area di Ormea anche se ptot può non
    # essere disponibile in tutti gli anni
    "PIE-004155-904": "ORMEA - STANTI",
}


# =============================================================================
# CLI / UTILS
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Scarica ARPA Piemonte storico per alto Tanaro/Tanarello."
    )
    p.add_argument("--start-year", type=int, default=START_YEAR)
    p.add_argument("--end-year", type=int, default=END_YEAR)
    p.add_argument("--test", action="store_true", help="Limita i dati al 2020.")
    p.add_argument("--discover", action="store_true")
    p.add_argument("--hydro-only", action="store_true")
    p.add_argument("--meteo-only", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def setup_dirs():
    for p in [
        BASE,
        HYDRO_DIR,
        METEO_DIR,
        CATALOG_DIR,
        DIAG_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def norm(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.upper().replace("\xa0", " ").strip()
    return re.sub(r"\s+", " ", s)


def safe_name(x: Any) -> str:
    s = norm(x)
    s = re.sub(r"[^A-Z0-9._-]+", "_", s)
    return s.strip("_")[:90] or "UNKNOWN"


def file_ok(path: Path) -> bool:
    return (
        path.exists()
        and path.is_file()
        and path.stat().st_size >= MIN_FILE_BYTES
    )


def log_manifest(**record):
    record["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    with MANIFEST.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(record, ensure_ascii=False, default=str) + "\n"
        )


def write_json(path: Path, data: Any):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def write_csv_records(path: Path, records: list[dict]):
    if not records:
        # scrive comunque un file esplicativo piccolo
        path.write_text("no_records\n", encoding="utf-8")
        return

    if pd is not None:
        pd.DataFrame(records).to_csv(path, index=False)
        return

    cols = []
    seen = set()
    for rec in records:
        for key in rec:
            if key not in seen:
                seen.add(key)
                cols.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)


# =============================================================================
# HTTP ROBUSTO
# =============================================================================

def session_factory():
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Tanaro-Arroscia hydrology feasibility study "
            "(ARPA Piemonte public historical API)"
        ),
        "Accept": "application/json,text/plain,*/*",
    })
    return s


def request_json(session, url, params=None, label="request"):
    """
    Retry automatico su errori transitori / timeout.
    Gli errori 4xx non transitori vengono restituiti subito.
    """
    last_exc = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            if r.status_code in TRANSIENT_CODES:
                raise requests.HTTPError(
                    f"{r.status_code} temporary server response",
                    response=r,
                )

            r.raise_for_status()

            try:
                return r.json()
            except Exception as exc:
                sample = r.text[:500]
                raise RuntimeError(
                    f"{label}: risposta non JSON. "
                    f"HTTP {r.status_code}; sample={sample!r}"
                ) from exc

        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.HTTPError,
        ) as exc:
            last_exc = exc

            status = None
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                status = exc.response.status_code

            # 4xx non transitorio: non serve riprovare.
            if status is not None and status not in TRANSIENT_CODES:
                raise

            if attempt >= MAX_RETRIES:
                break

            wait_s = min(20 * (2 ** (attempt - 1)), 300)
            print(
                f"  {label}: problema temporaneo "
                f"(tentativo {attempt}/{MAX_RETRIES}): {exc}"
            )
            print(f"  Attendo {wait_s} s e riprovo...")
            time.sleep(wait_s)

    raise RuntimeError(
        f"{label}: fallito dopo {MAX_RETRIES} tentativi. "
        f"Ultimo errore: {last_exc}"
    )


def paginated_get(session, url, params=None, label="API"):
    """
    Gestisce Django REST Framework:
        {count, next, previous, results}
    ma accetta anche liste JSON dirette.
    """
    params = dict(params or {})
    out = []
    page = 1

    while True:
        q = dict(params)
        q["page"] = page

        data = request_json(
            session,
            url,
            q,
            label=f"{label} page {page}",
        )

        if isinstance(data, list):
            out.extend(data)
            break

        if not isinstance(data, dict):
            raise RuntimeError(
                f"{label}: JSON inatteso: {type(data).__name__}"
            )

        results = data.get("results", [])
        if isinstance(results, list):
            out.extend(results)
        else:
            raise RuntimeError(
                f"{label}: campo results non è una lista."
            )

        if not data.get("next"):
            break

        page += 1
        if page > 10000:
            raise RuntimeError(f"{label}: paginazione anomala >10000 pagine.")

        time.sleep(0.10)

    return out


# =============================================================================
# METADATA HELPERS
# =============================================================================

def pick_first(rec, keys):
    for k in keys:
        if k in rec and rec[k] not in (None, ""):
            return rec[k]
    return None


def station_id(rec):
    # L'ID API affidabile è spesso il segmento finale di "url":
    # .../stazione_idrologica/PIE-006012-700/
    candidates = [
        rec.get("id"),
        rec.get("pk"),
        rec.get("fk_id_punto_misura_idro"),
        rec.get("fk_id_punto_misura_meteo"),
        rec.get("url"),
    ]

    for value in candidates:
        if value is None:
            continue
        s = str(value).strip().rstrip("/")
        if "/" in s:
            s = s.split("/")[-1]
        if s:
            return s

    return None


def station_name(rec):
    return pick_first(
        rec,
        ["denominazione", "nome", "station_name", "localita"],
    ) or "UNKNOWN"


def station_lat_lon(rec):
    lat = pick_first(
        rec,
        [
            "latitudine_n_wgs84_d",
            "latitudine",
            "latitude",
            "lat",
        ],
    )
    lon = pick_first(
        rec,
        [
            "longitudine_e_wgs84_d",
            "longitudine",
            "longitude",
            "lon",
        ],
    )

    try:
        return float(lat), float(lon)
    except (TypeError, ValueError):
        return None, None


def inside_bbox(rec):
    lat, lon = station_lat_lon(rec)
    if lat is None or lon is None:
        return False

    return (
        BBOX["south"] <= lat <= BBOX["north"]
        and BBOX["west"] <= lon <= BBOX["east"]
    )


def metadata_text(rec):
    keys = [
        "denominazione",
        "nome",
        "localita",
        "comune",
        "corso_acqua",
        "bacino_idrografico",
        "sigla_prov",
        "provincia",
    ]
    return " | ".join(norm(rec.get(k)) for k in keys)


def priority_name_match(rec):
    text = metadata_text(rec)
    return any(term in text for term in PRIORITY_TERMS)


def select_hydro_stations(records):
    """
    Selezione CURATA, non più geografica generica.
    Manteniamo solo le sezioni idrologiche direttamente utili:
    Ponte di Nava, Pornassino-Negrone, Garessio-Tanaro.
    """
    selected = []
    for rec in records:
        sid = station_id(rec)
        if sid in HYDRO_TARGET_CODES:
            selected.append(rec)

    return dedupe_stations(selected)


def select_meteo_stations(records):
    """
    Selezione CURATA delle stazioni meteorologiche dell'alto Tanaro.
    Evita di scaricare stazioni del Pesio, Gesso, Vermenagna, Stura,
    Bormida e altri affluenti non pertinenti al confronto Arroscia.
    """
    selected = []
    for rec in records:
        sid = station_id(rec)
        if sid in METEO_TARGET_CODES:
            selected.append(rec)

    return dedupe_stations(selected)


def dedupe_stations(records):
    seen = set()
    out = []

    for rec in records:
        key = station_id(rec) or (
            norm(station_name(rec)),
            station_lat_lon(rec),
        )

        if str(key) in seen:
            continue

        seen.add(str(key))
        out.append(rec)

    return out


def save_catalog(kind, all_records, selected_records):
    write_json(
        CATALOG_DIR / f"arpa_{kind}_stations_all.json",
        all_records,
    )
    write_csv_records(
        CATALOG_DIR / f"arpa_{kind}_stations_all.csv",
        all_records,
    )

    write_json(
        CATALOG_DIR / f"arpa_{kind}_stations_selected.json",
        selected_records,
    )
    write_csv_records(
        CATALOG_DIR / f"arpa_{kind}_stations_selected.csv",
        selected_records,
    )


def print_selected(kind, selected):
    print()
    print(f"{kind.upper()} | STAZIONI SELEZIONATE: {len(selected)}")

    for rec in selected:
        sid = station_id(rec)
        name = station_name(rec)
        river = rec.get("corso_acqua", "")
        basin = rec.get("bacino_idrografico", "")
        lat, lon = station_lat_lon(rec)

        print(
            f"  {sid or '?':<20} | {name:<34} | "
            f"river={river!s:<16} | basin={basin!s:<12} | "
            f"lat={lat} lon={lon}"
        )


# =============================================================================
# DATA FILTER
# =============================================================================

def parse_date(value):
    if value in (None, ""):
        return None

    s = str(value).strip()

    # ISO standard
    try:
        return datetime.fromisoformat(s[:10])
    except Exception:
        pass

    if pd is not None:
        try:
            x = pd.to_datetime(value, errors="raise", dayfirst=False)
            return x.to_pydatetime()
        except Exception:
            pass

    return None


def date_from_record(rec):
    return parse_date(
        pick_first(rec, ["data", "date", "giorno", "DATA"])
    )


def autumn_filter(records, start_year, end_year):
    out = []

    for rec in records:
        dt = date_from_record(rec)

        if dt is None:
            # Non buttiamo record solo perché il campo data ha un nome
            # inatteso: lo conserviamo e lo segnaleremo in QA.
            out.append(rec)
            continue

        if start_year <= dt.year <= end_year and 9 <= dt.month <= 12:
            out.append(rec)

    return out


def variable_summary(records):
    if not records:
        return {}

    keys = set()
    for rec in records[:500]:
        keys.update(rec.keys())

    summary = {}

    interesting = [
        "ptot",
        "precipitazione",
        "livellomedio",
        "portatamedia",
        "livello",
        "portata",
    ]

    for key in sorted(keys):
        nk = norm(key)
        if any(norm(x) in nk for x in interesting):
            non_null = sum(
                1
                for r in records
                if r.get(key) not in (None, "", "nan", "NaN")
            )
            summary[key] = non_null

    return summary


# =============================================================================
# ENDPOINT DISCOVERY
# =============================================================================

def discover_meteo_station_endpoint(session):
    errors = []

    for url in METEO_STATION_CANDIDATES:
        try:
            data = request_json(
                session,
                url,
                {"page": 1},
                label=f"probe {url}",
            )

            if isinstance(data, dict) and "results" in data:
                print(f"METEO metadata endpoint OK: {url}")
                return url

            if isinstance(data, list):
                print(f"METEO metadata endpoint OK: {url}")
                return url

            errors.append((url, "JSON senza results/list"))

        except Exception as exc:
            errors.append((url, str(exc)))

    detail = "\n".join(f"  - {u}: {e}" for u, e in errors)

    raise RuntimeError(
        "Non riesco a individuare l'endpoint anagrafico meteo ARPA.\n"
        + detail
    )


# =============================================================================
# DOWNLOAD STATION DATA
# =============================================================================

def download_station(
    session,
    *,
    kind,
    rec,
    data_url,
    fk_param,
    out_dir,
    start_year,
    end_year,
    force=False,
):
    sid = station_id(rec)
    name = station_name(rec)

    if not sid:
        print(f"  SKIP {name}: id API non ricavabile.")
        return "skip"

    slug = f"{sid}_{safe_name(name)}"
    raw_json = out_dir / f"{slug}_raw.json"
    autumn_csv = (
        out_dir
        / f"{slug}_autumn_{start_year}_{end_year}.csv"
    )
    meta_json = out_dir / f"{slug}_metadata.json"

    if file_ok(autumn_csv) and not force:
        print(f"  SKIP {sid} | {name}: CSV già presente.")
        return "skip"

    # L'API documentata accetta data_min / data_max.
    params = {
        fk_param: sid,
        "data_min": f"{start_year}-01-01",
        "data_max": f"{end_year}-12-31",
    }

    print(f"  DOWNLOAD {kind}: {sid} | {name}")

    t0 = time.time()

    try:
        records = paginated_get(
            session,
            data_url,
            params=params,
            label=f"{kind} {sid}",
        )
    except Exception as exc:
        print(f"    ERRORE: {exc}")
        log_manifest(
            status="error",
            source="arpa_piemonte",
            kind=kind,
            station_id=sid,
            station_name=name,
            error=str(exc),
        )
        return "error"

    write_json(raw_json, records)
    write_json(meta_json, rec)

    filtered = autumn_filter(records, start_year, end_year)
    write_csv_records(autumn_csv, filtered)

    elapsed = time.time() - t0
    vars_found = variable_summary(filtered)

    print(
        f"    OK: {len(records)} record API; "
        f"{len(filtered)} record set-dic; "
        f"{elapsed:.1f} s"
    )

    if vars_found:
        print(f"    Variabili chiave: {vars_found}")

    log_manifest(
        status="ok",
        source="arpa_piemonte",
        kind=kind,
        station_id=sid,
        station_name=name,
        total_rows=len(records),
        autumn_rows=len(filtered),
        variables=vars_found,
        raw_json=str(raw_json),
        autumn_csv=str(autumn_csv),
        seconds=round(elapsed, 1),
    )

    return "ok"


# =============================================================================
# HYDRO
# =============================================================================

def discover_hydro(session):
    print()
    print("=" * 96)
    print("ARPA PIEMONTE | DISCOVERY RETE IDROLOGICA")
    print("=" * 96)

    records = paginated_get(
        session,
        HYDRO_STATIONS_URL,
        label="hydro station catalog",
    )

    selected = select_hydro_stations(records)
    save_catalog("hydro", records, selected)

    print(f"Catalogo idrologico totale: {len(records)} stazioni/record")
    print_selected("hydro", selected)
    found_codes = {station_id(r) for r in selected}
    missing = [c for c in HYDRO_TARGET_CODES if c not in found_codes]
    if missing:
        print(f"ATTENZIONE: target idro non trovati nel catalogo: {missing}")

    return selected


def run_hydro(session, selected, start_year, end_year, force=False):
    print()
    print("=" * 96)
    print("ARPA PIEMONTE | DOWNLOAD GIORNALIERO IDROLOGICO")
    print("=" * 96)

    for idx, rec in enumerate(selected, 1):
        print(f"\n[{idx}/{len(selected)}]")
        download_station(
            session,
            kind="daily_hydro",
            rec=rec,
            data_url=HYDRO_DATA_URL,
            fk_param="fk_id_punto_misura_idro",
            out_dir=HYDRO_DIR,
            start_year=start_year,
            end_year=end_year,
            force=force,
        )


# =============================================================================
# METEO
# =============================================================================

def discover_meteo(session):
    print()
    print("=" * 96)
    print("ARPA PIEMONTE | DISCOVERY RETE METEOROLOGICA")
    print("=" * 96)

    station_url = discover_meteo_station_endpoint(session)

    records = paginated_get(
        session,
        station_url,
        label="meteo station catalog",
    )

    selected = select_meteo_stations(records)
    save_catalog("meteo", records, selected)

    print(f"Catalogo meteorologico totale: {len(records)} stazioni/record")
    print_selected("meteo", selected)
    found_codes = {station_id(r) for r in selected}
    missing = [c for c in METEO_TARGET_CODES if c not in found_codes]
    if missing:
        print(f"ATTENZIONE: target meteo non trovati nel catalogo: {missing}")

    return selected, station_url


def run_meteo(session, selected, start_year, end_year, force=False):
    print()
    print("=" * 96)
    print("ARPA PIEMONTE | DOWNLOAD GIORNALIERO METEOROLOGICO")
    print("=" * 96)

    for idx, rec in enumerate(selected, 1):
        print(f"\n[{idx}/{len(selected)}]")
        download_station(
            session,
            kind="daily_meteo",
            rec=rec,
            data_url=METEO_DATA_URL,
            fk_param="fk_id_punto_misura_meteo",
            out_dir=METEO_DIR,
            start_year=start_year,
            end_year=end_year,
            force=force,
        )


# =============================================================================
# README
# =============================================================================

def write_readme():
    p = BASE / "README_ARPA.md"
    if p.exists():
        return

    txt = """# ARPA Piemonte — Tanaro/Tanarello

Questa cartella contiene dati scaricati dalla Banca Dati Storica
ufficiale ARPA Piemonte.

## daily_hydro
Serie giornaliere idrologiche:
- livello idrometrico medio
- portata media, quando disponibile

## daily_meteo
Serie giornaliere meteorologiche:
- precipitazione totale (`ptot`) quando disponibile
- altre variabili presenti nell'API sono conservate nei JSON raw

## catalog
Anagrafica completa e sottoinsieme di stazioni selezionate.

## Nota temporale
Il downloader conserva settembre–dicembre 1987–2025 per coerenza con
ERA5/GloFAS e con il catalogo degli eventi autunnali.

## Nota sui dati orari/suborari
Questo script NON interagisce con i moduli ARPA dotati di AntiSpam.
I dati orari/suborari saranno richiesti successivamente per un insieme
mirato di eventi critici individuati dallo screening statistico.
"""
    p.write_text(txt, encoding="utf-8")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    setup_dirs()
    write_readme()

    if args.start_year > args.end_year:
        raise ValueError("--start-year deve essere <= --end-year")

    if args.hydro_only and args.meteo_only:
        raise ValueError(
            "Usa al massimo uno tra --hydro-only e --meteo-only."
        )

    if args.test:
        start_year = end_year = 2020
    else:
        start_year = args.start_year
        end_year = args.end_year

    session = session_factory()

    print("=" * 96)
    print("TANARO–ARROSCIA | ARPA PIEMONTE DOWNLOADER v1.1 — TARGET CURATI")
    print(f"Periodo: {start_year}–{end_year}, filtro settembre–dicembre")
    print(f"Output : {BASE}")
    print("API    : Banca Dati Storica ARPA Piemonte")
    print("=" * 96)

    hydro_selected = []
    meteo_selected = []

    if not args.meteo_only:
        try:
            hydro_selected = discover_hydro(session)
        except Exception as exc:
            print(f"\nHYDRO DISCOVERY ERROR: {exc}")
            log_manifest(
                status="discovery_error",
                kind="hydro",
                error=str(exc),
            )

    if not args.hydro_only:
        try:
            meteo_selected, _ = discover_meteo(session)
        except Exception as exc:
            print(f"\nMETEO DISCOVERY ERROR: {exc}")
            log_manifest(
                status="discovery_error",
                kind="meteo",
                error=str(exc),
            )

    if args.discover:
        print()
        print("=" * 96)
        print("DISCOVERY COMPLETATA — nessuna serie scaricata")
        print("=" * 96)
        return

    if hydro_selected:
        run_hydro(
            session,
            hydro_selected,
            start_year,
            end_year,
            force=args.force,
        )

    if meteo_selected:
        run_meteo(
            session,
            meteo_selected,
            start_year,
            end_year,
            force=args.force,
        )

    print()
    print("=" * 96)
    print("FINE ARPA PIEMONTE DOWNLOADER")
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
