#!/usr/bin/env python3
"""
NW OBSERVATIONS COLLECTOR v1.1
==============================

Scopo
-----
Raccogliere la base osservativa storica necessaria per costruire le etichette
Y_rain e Y_flood del modello regionale sui 21 recettori idrografici NW.

Periodo standard: settembre-dicembre 1987-2025.

Fonti:
- ARPA Piemonte, Banca Dati Storica REST:
    https://utility.arpa.piemonte.it/meteoidro/
- ARPAL/OMIRL, archivio storico:
    https://ambientepub.regione.liguria.it/SiraQualMeteo/script/PubAccessoDatiMeteo.asp
- Centro Funzionale Regione Autonoma Valle d'Aosta, Dataview:
    https://presidi2.regione.vda.it/str_dataview_download

Input locale richiesto:
    basins_final/nw_receptors_final.geojson

Output:
    observations_nw/
        station_basin_map.csv
        download_manifest.jsonl
        piemonte/
            catalog/
            daily_meteo/
            daily_hydro/
        liguria/
            catalog/
            hourly/
            _diagnostics/
        vda/
            downloads/
            _diagnostics/
        _diagnostics/
        README_OBSERVATIONS_NW.md

Principi
--------
1. I dati raw vengono conservati; non vengono sovrascritti senza --force.
2. ARPA Piemonte: usa SOLO la Banca Dati Storica giornaliera pubblica.
   Non aggira il modulo AntiSpam per dati orari/suborari.
3. ARPAL: usa il portale storico ufficiale, tramite Playwright.
4. Valle d'Aosta: il portale ufficiale richiede dati personali/consensi.
   Questa v1.1 fornisce una modalità assistita con browser visibile:
       --provider vda --vda-assisted --headed
   I download effettuati dall'utente nel form ufficiale vengono intercettati
   e salvati localmente. Nessun dato personale viene scritto nel manifest.
5. Le stazioni piemontesi vengono associate automaticamente ai recettori
   tramite coordinate e geometrie di nw_receptors_final.geojson.
6. La v1.1 aggiunge QC automatico ARPA Piemonte: periodo effettivo,
   conteggio ptot/livello/portata e copertura anno×bacino.
7. Per Liguria si usa la modalità BACINO del portale OMIRL per ricavare la
   lista di stazioni e poi la modalità STAZIONE per scaricare precipitazione
   e livello idrometrico orari quando disponibili.

Esempi
-------
Discovery generale, nessun download di serie:
    python download_observations_nw_v1_1.py --discover

Test Piemonte 2020:
    python download_observations_nw_v1_1.py --provider piemonte --test

Test Liguria, browser visibile:
    python download_observations_nw_v1_1.py --provider liguria --test --headed

Valle d'Aosta, modalità assistita:
    python download_observations_nw_v1_1.py --provider vda --vda-assisted --headed

Completo Piemonte + Liguria:
    caffeinate -i python download_observations_nw_v1_1.py --provider piemonte
    caffeinate -i python download_observations_nw_v1_1.py --provider liguria

NOTA:
Non avviare il ramo Liguria mentre un altro downloader ARPAL/OMIRL è già
in esecuzione sullo stesso Mac. Terminare prima quel download.
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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from shapely.geometry import Point, shape
except Exception:
    Point = None
    shape = None

try:
    from playwright.sync_api import (
        sync_playwright,
        TimeoutError as PWTimeout,
    )
except Exception:
    sync_playwright = None
    PWTimeout = Exception


# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path(__file__).resolve().parent
BASINS_FILE = ROOT / "basins_final" / "nw_receptors_final.geojson"

OUT = ROOT / "observations_nw"
PIE = OUT / "piemonte"
PIE_CATALOG = PIE / "catalog"
PIE_METEO = PIE / "daily_meteo"
PIE_HYDRO = PIE / "daily_hydro"
PIE_QC = PIE / "qc"

LIG = OUT / "liguria"
LIG_CATALOG = LIG / "catalog"
LIG_HOURLY = LIG / "hourly"
LIG_DIAG = LIG / "_diagnostics"

VDA = OUT / "vda"
VDA_DOWNLOADS = VDA / "downloads"
VDA_DIAG = VDA / "_diagnostics"

DIAG = OUT / "_diagnostics"
MANIFEST = OUT / "download_manifest.jsonl"
STATION_BASIN_MAP = OUT / "station_basin_map.csv"
README = OUT / "README_OBSERVATIONS_NW.md"

START_YEAR = 1987
END_YEAR = 2025
AUTUMN_MONTHS = {9, 10, 11, 12}

REQUEST_TIMEOUT = 90
MAX_RETRIES = 8
TRANSIENT = {429, 500, 502, 503, 504}
MIN_FILE_BYTES = 60

ARPA_ROOT = "https://utility.arpa.piemonte.it/meteoidro"
ARPA_HYDRO_STATIONS = f"{ARPA_ROOT}/stazione_idrologica/"
ARPA_HYDRO_DATA = f"{ARPA_ROOT}/dati_giornalieri_idro/"
ARPA_METEO_STATION_CANDIDATES = [
    f"{ARPA_ROOT}/stazione_meteorologica/",
    f"{ARPA_ROOT}/stazione_meteo/",
]
ARPA_METEO_DATA = f"{ARPA_ROOT}/dati_giornalieri_meteo/"

ARPAL_PORTAL = (
    "https://ambientepub.regione.liguria.it/"
    "SiraQualMeteo/script/PubAccessoDatiMeteo.asp"
)

VDA_PORTAL = "https://presidi2.regione.vda.it/str_dataview_download"

# Questi recettori sono i target marittimi principali ARPAL.
# Per i bacini transregionali vengono inoltre letti i nomi liguri
# direttamente dalle proprietà del GeoJSON finale.
LIG_PRIMARY = {
    "LIG_BISAGNO",
    "LIG_POLCEVERA",
    "LIG_ENTELLA",
    "LIG_MAGRA",
    "LIG_CENTA",
}

# Parole chiave di fallback se il nome "liguria_names" non coincide
# esattamente con il catalogo BACINI del portale storico.
LIG_BASIN_FALLBACKS = {
    "LIG_BISAGNO": ["BISAGNO"],
    "LIG_POLCEVERA": ["POLCEVERA"],
    "LIG_ENTELLA": ["ENTELLA"],
    "LIG_MAGRA": ["MAGRA"],
    "LIG_CENTA": ["CENTA"],
    "NW_TANARO_ALTO": ["TANARO"],
    "NW_BORMIDA": ["BORMIDA"],
    "NW_ORBA": ["ORBA"],
    "NW_SCRIVIA": ["SCRIVIA"],
}

PRECIP_TERMS = ["PRECIPITAZIONE", "PRECIP", "PIOGGIA"]
LEVEL_TERMS = [
    "LIVELLO IDROMETRICO",
    "ALTEZZA IDROMETRICA",
    "IDROMETR",
    "LIVELLO",
]

NAV_TIMEOUT = 90_000
DOWNLOAD_TIMEOUT = 120_000


# =============================================================================
# CLI / BASIC UTILS
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Scarica osservazioni storiche per i 21 recettori NW."
    )
    p.add_argument(
        "--provider",
        choices=["all", "piemonte", "liguria", "vda"],
        default="all",
    )
    p.add_argument("--start-year", type=int, default=START_YEAR)
    p.add_argument("--end-year", type=int, default=END_YEAR)
    p.add_argument(
        "--buffer-km",
        type=float,
        default=5.0,
        help="Buffer prudente per assegnare stazioni ai bacini (default 5 km).",
    )
    p.add_argument("--discover", action="store_true")
    p.add_argument("--test", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--headed", action="store_true")
    p.add_argument(
        "--max-stations-per-basin",
        type=int,
        default=0,
        help="0=tutte; in --test viene comunque limitato a 1.",
    )
    p.add_argument(
        "--vda-assisted",
        action="store_true",
        help="Apre il form ufficiale VdA e intercetta i download fatti dall'utente.",
    )
    p.add_argument(
        "--audit",
        action="store_true",
        help="Mostra un inventario locale senza fare richieste di rete.",
    )
    return p.parse_args()


def setup_dirs():
    for p in [
        OUT, PIE, PIE_CATALOG, PIE_METEO, PIE_HYDRO, PIE_QC,
        LIG, LIG_CATALOG, LIG_HOURLY, LIG_DIAG,
        VDA, VDA_DOWNLOADS, VDA_DIAG, DIAG,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def norm(x: Any) -> str:
    s = "" if x is None else str(x)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.upper().replace("\xa0", " ").strip()
    return re.sub(r"\s+", " ", s)


def safe(x: Any) -> str:
    s = norm(x)
    s = re.sub(r"[^A-Z0-9._-]+", "_", s)
    return s.strip("_")[:120] or "UNKNOWN"


def file_ok(path: Path, min_bytes: int = MIN_FILE_BYTES):
    return path.exists() and path.is_file() and path.stat().st_size >= min_bytes


def write_json(path: Path, obj: Any):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("no_records\n", encoding="utf-8")
        return
    if pd is not None:
        pd.DataFrame(rows).to_csv(path, index=False)
        return
    fields = []
    seen = set()
    for rec in rows:
        for k in rec:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def log_manifest(**rec):
    rec["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    with MANIFEST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def write_readme():
    text = """# Observations NW

Base osservativa storica per il modello regionale mare-atmosfera-bacino.

## Fonti
- ARPA Piemonte: Banca Dati Storica giornaliera REST.
- ARPAL/OMIRL: archivio storico, dati orari quando disponibili.
- Centro Funzionale RAVDA: portale Dataview.

## Periodo
Settembre-dicembre 1987-2025, limitatamente al periodo realmente disponibile
per ciascun sensore.

## Uso scientifico
I dati osservati servono a costruire:
- Y_rain: evento pluviometrico estremo / quantità osservata;
- Y_flood: piena critica / livello / portata, dove disponibile.

ERA5 non sostituisce queste osservazioni come ground truth.

## Limite Piemonte
La Banca Dati Storica automatizzabile è giornaliera. I dati ARPA Piemonte
orari/suborari sono gestiti da un form con AntiSpam e limiti di richiesta:
questo script NON lo aggira. Dopo lo screening degli eventi critici verranno
richiesti in modo mirato i periodi orari necessari per rifinire 1/3/6/12 h.

## Raw
Non cancellare JSON/CSV raw, cataloghi, diagnostica e manifest.
"""
    README.write_text(text, encoding="utf-8")


# =============================================================================
# BASINS
# =============================================================================

def load_basins():
    if shape is None or Point is None:
        raise RuntimeError(
            "Manca shapely. Usa l'ambiente esistente oppure installa shapely "
            "solo quando non ci sono download lunghi in esecuzione."
        )
    if not BASINS_FILE.exists():
        raise FileNotFoundError(f"Manca {BASINS_FILE}")

    obj = json.loads(BASINS_FILE.read_text(encoding="utf-8"))
    out = []
    for feat in obj.get("features", []):
        props = feat.get("properties") or {}
        rid = props.get("receptor_id")
        if not rid:
            continue
        geom = shape(feat.get("geometry"))
        if not geom.is_valid:
            geom = geom.buffer(0)
        out.append({
            "receptor_id": rid,
            "label": props.get("label", rid),
            "region": props.get("region", ""),
            "liguria_names": props.get("liguria_names", ""),
            "geometry": geom,
        })
    if len(out) != 21:
        print(f"ATTENZIONE: recettori caricati {len(out)} invece di 21.")
    return out


def basin_matches_point(basin, lon, lat, buffer_km):
    p = Point(float(lon), float(lat))
    g = basin["geometry"]
    if g.covers(p):
        return True, 0.0

    # Per una selezione di candidate stations è sufficiente un buffer
    # prudente in gradi. NON viene usato per calcoli scientifici di area.
    buffer_deg = max(buffer_km, 0.0) / 111.0
    if buffer_deg > 0 and g.buffer(buffer_deg).covers(p):
        # Distanza approssimata in km.
        d_deg = g.distance(p)
        return True, d_deg * 111.0
    return False, None


def basin_assignments_for_point(basins, lon, lat, buffer_km):
    hits = []
    for b in basins:
        ok, d = basin_matches_point(b, lon, lat, buffer_km)
        if ok:
            hits.append((b["receptor_id"], d or 0.0))
    hits.sort(key=lambda x: x[1])
    return hits


# =============================================================================
# ROBUST HTTP
# =============================================================================

def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "NW flood-probability research / official public historical data"
        ),
        "Accept": "application/json,text/plain,*/*",
    })
    return s


def request_json(session, url, params=None, label="request"):
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code in TRANSIENT:
                raise requests.HTTPError(
                    f"HTTP {r.status_code}",
                    response=r,
                )
            r.raise_for_status()
            return r.json()
        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.HTTPError,
        ) as exc:
            last = exc
            code = None
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                code = exc.response.status_code
            if code is not None and code not in TRANSIENT:
                raise
            if attempt >= MAX_RETRIES:
                break
            wait = min(10 * (2 ** (attempt - 1)), 180)
            print(f"  {label}: retry {attempt}/{MAX_RETRIES}: {exc}")
            print(f"  attesa {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"{label}: fallito dopo {MAX_RETRIES} tentativi: {last}")


def paginated_get(session, url, params=None, label="API"):
    out = []
    page = 1
    params = dict(params or {})
    while True:
        q = dict(params)
        q["page"] = page
        data = request_json(session, url, q, f"{label} page {page}")
        if isinstance(data, list):
            out.extend(data)
            break
        if not isinstance(data, dict):
            raise RuntimeError(f"{label}: JSON inatteso {type(data).__name__}")
        rows = data.get("results", [])
        if not isinstance(rows, list):
            raise RuntimeError(f"{label}: results non-lista")
        out.extend(rows)
        if not data.get("next"):
            break
        page += 1
        if page > 10000:
            raise RuntimeError(f"{label}: paginazione > 10000")
        time.sleep(0.08)
    return out


# =============================================================================
# ARPA PIEMONTE
# =============================================================================

def pick_first(rec, keys):
    for k in keys:
        if rec.get(k) not in (None, ""):
            return rec[k]
    return None


def arpa_station_id(rec):
    vals = [
        rec.get("id"), rec.get("pk"),
        rec.get("fk_id_punto_misura_idro"),
        rec.get("fk_id_punto_misura_meteo"),
        rec.get("url"),
    ]
    for v in vals:
        if v is None:
            continue
        s = str(v).strip().rstrip("/")
        if "/" in s:
            s = s.split("/")[-1]
        if s:
            return s
    return None


def arpa_station_name(rec):
    return pick_first(
        rec,
        ["denominazione", "nome", "station_name", "localita"],
    ) or "UNKNOWN"


def arpa_lat_lon(rec):
    lat = pick_first(
        rec,
        ["latitudine_n_wgs84_d", "latitudine", "latitude", "lat"],
    )
    lon = pick_first(
        rec,
        ["longitudine_e_wgs84_d", "longitudine", "longitude", "lon"],
    )
    try:
        return float(lat), float(lon)
    except Exception:
        return None, None


def discover_arpa_meteo_endpoint(session):
    errors = []
    for url in ARPA_METEO_STATION_CANDIDATES:
        try:
            rows = paginated_get(session, url, label="probe meteo station")
            if rows:
                return url, rows
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Nessun endpoint meteo valido: " + " | ".join(errors))


def select_arpa_stations(records, basins, buffer_km, kind):
    selected = []
    map_rows = []

    for rec in records:
        sid = arpa_station_id(rec)
        lat, lon = arpa_lat_lon(rec)
        if not sid or lat is None or lon is None:
            continue

        hits = basin_assignments_for_point(basins, lon, lat, buffer_km)
        if not hits:
            continue

        receptor_ids = [rid for rid, _ in hits]
        selected.append(rec)

        map_rows.append({
            "provider": "ARPA_PIEMONTE",
            "kind": kind,
            "station_id": sid,
            "station_name": arpa_station_name(rec),
            "latitude": lat,
            "longitude": lon,
            "receptor_ids": " | ".join(receptor_ids),
            "nearest_receptor": receptor_ids[0],
            "distance_to_nearest_basin_km": round(hits[0][1], 3),
        })

    # dedupe station ids
    dedup = {}
    for rec in selected:
        dedup[arpa_station_id(rec)] = rec

    return list(dedup.values()), map_rows


def parse_date(v):
    if v in (None, ""):
        return None
    s = str(v).strip()
    try:
        return datetime.fromisoformat(s[:10])
    except Exception:
        pass
    if pd is not None:
        try:
            return pd.to_datetime(v, errors="raise").to_pydatetime()
        except Exception:
            pass
    return None


def filter_autumn(records, start_year, end_year):
    out = []
    for rec in records:
        dt = parse_date(
            pick_first(rec, ["data", "date", "giorno", "DATA"])
        )
        if dt is None:
            # Conserviamo i record con schema inatteso: il QC li segnalerà.
            out.append(rec)
            continue
        if start_year <= dt.year <= end_year and dt.month in AUTUMN_MONTHS:
            out.append(rec)
    return out


def arpa_variable_summary(records):
    """Elenca le variabili meteo-idrologiche rilevanti realmente presenti."""
    keys = set()
    for r in records[:1000]:
        keys.update(r.keys())
    interesting = [
        "PTOT", "PRECIPITAZIONE",
        "LIVELLOMEDIO", "LIVELLO",
        "PORTATAMEDIA", "PORTATA",
    ]
    return sorted(
        k for k in keys
        if any(x in norm(k) for x in interesting)
    )


def value_is_valid(v):
    """True anche per 0.0; False per None, stringa vuota e NaN."""
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    s = str(v).strip()
    if not s or norm(s) in {"NAN", "NONE", "NULL", "N/A"}:
        return False
    return True


def metric_columns(records, metric):
    """
    Restituisce le colonne che possono rappresentare una metrica.
    Per i conteggi si usa ANY tra le colonne candidate, così restano
    utilizzabili anche eventuali varianti storiche dell'API.
    """
    keys = set()
    for r in records[:2000]:
        keys.update(r.keys())

    nk = {k: norm(k) for k in keys}

    if metric == "ptot":
        preferred = ["PTOT", "PRECIPITAZIONE"]
        return sorted(
            k for k, n in nk.items()
            if n == "PTOT" or "PRECIPITAZIONE" in n
        )

    if metric == "level":
        return sorted(
            k for k, n in nk.items()
            if n.startswith("LIVELLOMEDIO")
            or n.startswith("LIVFREAMEDIO")
            or ("LIVELLO" in n and "CLASSE" not in n)
        )

    if metric == "discharge":
        return sorted(
            k for k, n in nk.items()
            if n.startswith("PORTATAMEDIA")
            or ("PORTATA" in n and "CLASSE" not in n)
        )

    return []


def row_has_metric(rec, columns):
    return any(value_is_valid(rec.get(c)) for c in columns)


def records_from_csv(path):
    """Rilegge un CSV già presente per poter produrre QC anche in caso di SKIP."""
    if not path.exists() or path.stat().st_size < 5:
        return []
    try:
        if pd is not None:
            df = pd.read_csv(path)
            if list(df.columns) == ["no_records"]:
                return []
            return df.to_dict("records")
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            if rows and set(rows[0]) == {"no_records"}:
                return []
            return rows
    except Exception:
        return []


def station_qc_from_records(
    *,
    kind,
    station_id,
    station_name,
    records,
    requested_start_year,
    requested_end_year,
    source_status,
):
    """
    Produce QC stazione + QC stazione/anno.
    I record passati sono quelli settembre-dicembre.
    """
    dated = []
    for r in records:
        dt = parse_date(pick_first(r, ["data", "date", "giorno", "DATA"]))
        if dt is not None:
            dated.append((dt, r))

    dated.sort(key=lambda x: x[0])

    cols_ptot = metric_columns(records, "ptot")
    cols_level = metric_columns(records, "level")
    cols_discharge = metric_columns(records, "discharge")

    valid_ptot = sum(row_has_metric(r, cols_ptot) for _, r in dated)
    valid_level = sum(row_has_metric(r, cols_level) for _, r in dated)
    valid_discharge = sum(row_has_metric(r, cols_discharge) for _, r in dated)

    years = sorted({dt.year for dt, _ in dated})
    first_date = dated[0][0].date().isoformat() if dated else ""
    last_date = dated[-1][0].date().isoformat() if dated else ""

    station_row = {
        "kind": kind,
        "station_id": station_id,
        "station_name": station_name,
        "requested_start_year": requested_start_year,
        "requested_end_year": requested_end_year,
        "status": source_status,
        "autumn_rows": len(dated),
        "first_autumn_date": first_date,
        "last_autumn_date": last_date,
        "first_year_with_autumn_data": years[0] if years else "",
        "last_year_with_autumn_data": years[-1] if years else "",
        "years_with_autumn_data": len(years),
        "valid_ptot_days": valid_ptot,
        "valid_level_days": valid_level,
        "valid_discharge_days": valid_discharge,
        "ptot_columns": " | ".join(cols_ptot),
        "level_columns": " | ".join(cols_level),
        "discharge_columns": " | ".join(cols_discharge),
        "has_any_data": bool(dated),
        "has_ptot": valid_ptot > 0,
        "has_level": valid_level > 0,
        "has_discharge": valid_discharge > 0,
    }

    by_year = defaultdict(list)
    for dt, r in dated:
        by_year[dt.year].append(r)

    year_rows = []
    for year in range(requested_start_year, requested_end_year + 1):
        rr = by_year.get(year, [])
        year_rows.append({
            "kind": kind,
            "station_id": station_id,
            "station_name": station_name,
            "year": year,
            "autumn_rows": len(rr),
            "valid_ptot_days": sum(row_has_metric(r, cols_ptot) for r in rr),
            "valid_level_days": sum(row_has_metric(r, cols_level) for r in rr),
            "valid_discharge_days": sum(row_has_metric(r, cols_discharge) for r in rr),
        })

    return station_row, year_rows


def write_piemonte_qc(
    *,
    basins,
    station_rows,
    station_year_rows,
    hydro_map,
    meteo_map,
    start_year,
    end_year,
):
    """
    Report finale:
      station_coverage.csv
      station_year_coverage.csv
      basin_year_coverage.csv
      basin_summary.csv
      piemonte_observations_qc.txt
    """
    PIE_QC.mkdir(parents=True, exist_ok=True)

    write_csv(PIE_QC / "station_coverage.csv", station_rows)
    write_csv(PIE_QC / "station_year_coverage.csv", station_year_rows)

    # mapping (kind, station_id) -> recettori
    mapping = defaultdict(set)
    for r in hydro_map + meteo_map:
        key = (str(r.get("kind", "")), str(r.get("station_id", "")))
        for rid in str(r.get("receptor_ids", "")).split("|"):
            rid = rid.strip()
            if rid:
                mapping[key].add(rid)

    # inizializza tutti i 21 recettori x tutti gli anni
    agg = {}
    for b in basins:
        rid = b["receptor_id"]
        for year in range(start_year, end_year + 1):
            agg[(rid, year)] = {
                "receptor_id": rid,
                "year": year,
                "meteo_any": set(),
                "meteo_ptot": set(),
                "hydro_any": set(),
                "hydro_level": set(),
                "hydro_discharge": set(),
                "ptot_valid_station_days": 0,
                "level_valid_station_days": 0,
                "discharge_valid_station_days": 0,
            }

    for r in station_year_rows:
        kind = r["kind"]
        sid = str(r["station_id"])
        year = int(r["year"])
        map_kind = "meteo" if kind == "daily_meteo" else "hydro"
        receptors = mapping.get((map_kind, sid), set())

        for rid in receptors:
            key = (rid, year)
            if key not in agg:
                continue
            a = agg[key]

            if kind == "daily_meteo":
                if int(r["autumn_rows"]) > 0:
                    a["meteo_any"].add(sid)
                if int(r["valid_ptot_days"]) > 0:
                    a["meteo_ptot"].add(sid)
                a["ptot_valid_station_days"] += int(r["valid_ptot_days"])

            elif kind == "daily_hydro":
                if int(r["autumn_rows"]) > 0:
                    a["hydro_any"].add(sid)
                if int(r["valid_level_days"]) > 0:
                    a["hydro_level"].add(sid)
                if int(r["valid_discharge_days"]) > 0:
                    a["hydro_discharge"].add(sid)
                a["level_valid_station_days"] += int(r["valid_level_days"])
                a["discharge_valid_station_days"] += int(r["valid_discharge_days"])

    basin_year_rows = []
    for (rid, year), a in sorted(agg.items()):
        basin_year_rows.append({
            "receptor_id": rid,
            "year": year,
            "meteo_stations_with_any_data": len(a["meteo_any"]),
            "meteo_stations_with_valid_ptot": len(a["meteo_ptot"]),
            "hydro_stations_with_any_data": len(a["hydro_any"]),
            "hydro_stations_with_valid_level": len(a["hydro_level"]),
            "hydro_stations_with_valid_discharge": len(a["hydro_discharge"]),
            "ptot_valid_station_days": a["ptot_valid_station_days"],
            "level_valid_station_days": a["level_valid_station_days"],
            "discharge_valid_station_days": a["discharge_valid_station_days"],
        })

    write_csv(PIE_QC / "basin_year_coverage.csv", basin_year_rows)

    # Sintesi per bacino
    def med(vals):
        vals = sorted(vals)
        if not vals:
            return 0.0
        n = len(vals)
        if n % 2:
            return float(vals[n // 2])
        return (vals[n//2 - 1] + vals[n//2]) / 2.0

    basin_summary = []
    for b in basins:
        rid = b["receptor_id"]
        rr = [x for x in basin_year_rows if x["receptor_id"] == rid]
        pt = [x["meteo_stations_with_valid_ptot"] for x in rr]
        lv = [x["hydro_stations_with_valid_level"] for x in rr]
        qd = [x["hydro_stations_with_valid_discharge"] for x in rr]

        basin_summary.append({
            "receptor_id": rid,
            "years_total": len(rr),
            "years_with_ptot": sum(v > 0 for v in pt),
            "years_with_level": sum(v > 0 for v in lv),
            "years_with_discharge": sum(v > 0 for v in qd),
            "min_ptot_stations_per_year": min(pt) if pt else 0,
            "median_ptot_stations_per_year": med(pt),
            "max_ptot_stations_per_year": max(pt) if pt else 0,
            "min_level_stations_per_year": min(lv) if lv else 0,
            "median_level_stations_per_year": med(lv),
            "max_level_stations_per_year": max(lv) if lv else 0,
            "min_discharge_stations_per_year": min(qd) if qd else 0,
            "median_discharge_stations_per_year": med(qd),
            "max_discharge_stations_per_year": max(qd) if qd else 0,
        })

    write_csv(PIE_QC / "basin_summary.csv", basin_summary)

    # Report testuale
    lines = []
    lines.append("=" * 100)
    lines.append("ARPA PIEMONTE — OBSERVATIONS NW | QC REPORT v1.1")
    lines.append("=" * 100)
    lines.append(f"Periodo richiesto: settembre-dicembre {start_year}-{end_year}")
    lines.append(f"Stazioni/serie analizzate: {len(station_rows)}")
    lines.append("")

    met = [r for r in station_rows if r["kind"] == "daily_meteo"]
    hyd = [r for r in station_rows if r["kind"] == "daily_hydro"]

    lines.append(
        f"METEO: {len(met)} serie | con dati={sum(bool(r['has_any_data']) for r in met)} "
        f"| con ptot={sum(bool(r['has_ptot']) for r in met)}"
    )
    lines.append(
        f"HYDRO: {len(hyd)} serie | con dati={sum(bool(r['has_any_data']) for r in hyd)} "
        f"| con livello={sum(bool(r['has_level']) for r in hyd)} "
        f"| con portata={sum(bool(r['has_discharge']) for r in hyd)}"
    )
    lines.append("")
    lines.append("COPERTURA PER BACINO")
    lines.append(
        "receptor_id                    years_ptot years_level years_discharge"
    )
    for r in basin_summary:
        lines.append(
            f"{r['receptor_id']:<30s} "
            f"{r['years_with_ptot']:>10d} "
            f"{r['years_with_level']:>11d} "
            f"{r['years_with_discharge']:>15d}"
        )

    lines.append("")
    lines.append("Interpretazione:")
    lines.append(
        "- ptot giornaliera osservata -> ground truth principale per Y_rain a 24/48 h."
    )
    lines.append(
        "- livellomedio/portatamedia -> ground truth idrologica giornaliera per screening Y_flood."
    )
    lines.append(
        "- i massimi rapidi possono essere attenuati dalle medie giornaliere; "
        "per gli eventi selezionati serviranno dati orari/suborari ufficiali."
    )
    lines.append("=" * 100)

    (PIE_QC / "piemonte_observations_qc.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("QC PIEMONTE SCRITTO")
    print("=" * 100)
    print(PIE_QC / "station_coverage.csv")
    print(PIE_QC / "station_year_coverage.csv")
    print(PIE_QC / "basin_year_coverage.csv")
    print(PIE_QC / "basin_summary.csv")
    print(PIE_QC / "piemonte_observations_qc.txt")
    print("=" * 100)


def download_arpa_station(
    session,
    *,
    kind,
    rec,
    data_url,
    fk_param,
    out_dir,
    start_year,
    end_year,
    force,
):
    sid = arpa_station_id(rec)
    name = arpa_station_name(rec)
    stem = f"{sid}_{safe(name)}"

    raw = out_dir / f"{stem}_raw.json"
    meta = out_dir / f"{stem}_metadata.json"
    csv_out = out_dir / f"{stem}_autumn_{start_year}_{end_year}.csv"

    if file_ok(csv_out) and not force:
        filtered = records_from_csv(csv_out)
        station_qc, year_qc = station_qc_from_records(
            kind=kind,
            station_id=sid,
            station_name=name,
            records=filtered,
            requested_start_year=start_year,
            requested_end_year=end_year,
            source_status="reuse",
        )
        print(
            f"  REUSE {sid} | {name} | set-dic={len(filtered)} "
            f"vars={arpa_variable_summary(filtered)}"
        )
        return station_qc, year_qc

    params = {
        fk_param: sid,
        "data_min": f"{start_year}-01-01",
        "data_max": f"{end_year}-12-31",
    }

    print(f"  DOWNLOAD {kind}: {sid} | {name}")
    t0 = time.time()
    try:
        rows = paginated_get(
            session, data_url, params, f"{kind} {sid}"
        )
    except Exception as exc:
        print(f"    ERROR: {exc}")
        log_manifest(
            source="ARPA_PIEMONTE",
            kind=kind,
            station_id=sid,
            station_name=name,
            status="error",
            error=str(exc),
        )
        station_qc, year_qc = station_qc_from_records(
            kind=kind,
            station_id=sid,
            station_name=name,
            records=[],
            requested_start_year=start_year,
            requested_end_year=end_year,
            source_status="error",
        )
        return station_qc, year_qc

    filtered = filter_autumn(rows, start_year, end_year)
    write_json(raw, rows)
    write_json(meta, rec)
    write_csv(csv_out, filtered)

    station_qc, year_qc = station_qc_from_records(
        kind=kind,
        station_id=sid,
        station_name=name,
        records=filtered,
        requested_start_year=start_year,
        requested_end_year=end_year,
        source_status="ok" if filtered else "no_data_in_requested_period",
    )

    print(
        f"    OK total={len(rows)} set-dic={len(filtered)} "
        f"vars={arpa_variable_summary(filtered)} "
        f"ptot={station_qc['valid_ptot_days']} "
        f"level={station_qc['valid_level_days']} "
        f"Q={station_qc['valid_discharge_days']} "
        f"{time.time()-t0:.1f}s"
    )

    log_manifest(
        source="ARPA_PIEMONTE",
        kind=kind,
        station_id=sid,
        station_name=name,
        status=station_qc["status"],
        total_rows=len(rows),
        autumn_rows=len(filtered),
        first_autumn_date=station_qc["first_autumn_date"],
        last_autumn_date=station_qc["last_autumn_date"],
        valid_ptot_days=station_qc["valid_ptot_days"],
        valid_level_days=station_qc["valid_level_days"],
        valid_discharge_days=station_qc["valid_discharge_days"],
        raw=str(raw),
        csv=str(csv_out),
    )
    return station_qc, year_qc


def limit_selected_by_basin(selected, map_rows, max_per_basin):
    if max_per_basin <= 0:
        return selected, map_rows

    # Un limite prudente per test/uso controllato:
    # conserva fino a N stazioni per nearest_receptor.
    by_id = {arpa_station_id(r): r for r in selected}
    keep_ids = set()
    counts = defaultdict(int)

    rows_sorted = sorted(
        map_rows,
        key=lambda x: (
            x["nearest_receptor"],
            x["distance_to_nearest_basin_km"],
            x["station_id"],
        ),
    )
    for row in rows_sorted:
        rid = row["nearest_receptor"]
        if counts[rid] < max_per_basin:
            keep_ids.add(row["station_id"])
            counts[rid] += 1

    selected2 = [by_id[sid] for sid in keep_ids if sid in by_id]
    map2 = [r for r in map_rows if r["station_id"] in keep_ids]
    return selected2, map2


def run_piemonte(args, basins):
    print("\n" + "=" * 100)
    print("ARPA PIEMONTE | OSSERVAZIONI REGIONALI")
    print("=" * 100)

    s = make_session()

    hydro_all = paginated_get(
        s, ARPA_HYDRO_STATIONS, label="ARPA hydro catalog"
    )
    meteo_url, meteo_all = discover_arpa_meteo_endpoint(s)

    write_json(PIE_CATALOG / "hydro_stations_all.json", hydro_all)
    write_csv(PIE_CATALOG / "hydro_stations_all.csv", hydro_all)
    write_json(PIE_CATALOG / "meteo_stations_all.json", meteo_all)
    write_csv(PIE_CATALOG / "meteo_stations_all.csv", meteo_all)

    hydro_sel, hydro_map = select_arpa_stations(
        hydro_all, basins, args.buffer_km, "hydro"
    )
    meteo_sel, meteo_map = select_arpa_stations(
        meteo_all, basins, args.buffer_km, "meteo"
    )

    limit = args.max_stations_per_basin
    if args.test:
        limit = 1
    hydro_sel, hydro_map = limit_selected_by_basin(
        hydro_sel, hydro_map, limit
    )
    meteo_sel, meteo_map = limit_selected_by_basin(
        meteo_sel, meteo_map, limit
    )

    write_json(PIE_CATALOG / "hydro_stations_selected.json", hydro_sel)
    write_csv(PIE_CATALOG / "hydro_station_basin_map.csv", hydro_map)
    write_json(PIE_CATALOG / "meteo_stations_selected.json", meteo_sel)
    write_csv(PIE_CATALOG / "meteo_station_basin_map.csv", meteo_map)

    append_station_map(hydro_map + meteo_map)

    print(f"Catalogo idro totale: {len(hydro_all)}")
    print(f"Idro selezionate    : {len(hydro_sel)}")
    print(f"Catalogo meteo totale: {len(meteo_all)}")
    print(f"Meteo selezionate    : {len(meteo_sel)}")
    print(f"Endpoint meteo: {meteo_url}")

    if args.discover:
        print("DISCOVERY Piemonte completata: nessuna serie scaricata.")
        return

    sy, ey = (2020, 2020) if args.test else (args.start_year, args.end_year)

    station_qc_rows = []
    station_year_qc_rows = []

    for i, rec in enumerate(hydro_sel, 1):
        print(f"\nHYDRO [{i}/{len(hydro_sel)}]")
        sqc, yqc = download_arpa_station(
            s,
            kind="daily_hydro",
            rec=rec,
            data_url=ARPA_HYDRO_DATA,
            fk_param="fk_id_punto_misura_idro",
            out_dir=PIE_HYDRO,
            start_year=sy,
            end_year=ey,
            force=args.force,
        )
        station_qc_rows.append(sqc)
        station_year_qc_rows.extend(yqc)

    for i, rec in enumerate(meteo_sel, 1):
        print(f"\nMETEO [{i}/{len(meteo_sel)}]")
        sqc, yqc = download_arpa_station(
            s,
            kind="daily_meteo",
            rec=rec,
            data_url=ARPA_METEO_DATA,
            fk_param="fk_id_punto_misura_meteo",
            out_dir=PIE_METEO,
            start_year=sy,
            end_year=ey,
            force=args.force,
        )
        station_qc_rows.append(sqc)
        station_year_qc_rows.extend(yqc)

    write_piemonte_qc(
        basins=basins,
        station_rows=station_qc_rows,
        station_year_rows=station_year_qc_rows,
        hydro_map=hydro_map,
        meteo_map=meteo_map,
        start_year=sy,
        end_year=ey,
    )


# =============================================================================
# STATION MAP
# =============================================================================

def append_station_map(rows):
    if not rows:
        return

    existing = []
    if STATION_BASIN_MAP.exists() and STATION_BASIN_MAP.stat().st_size > 20:
        try:
            if pd is not None:
                existing = pd.read_csv(STATION_BASIN_MAP).to_dict("records")
            else:
                with STATION_BASIN_MAP.open(encoding="utf-8") as f:
                    existing = list(csv.DictReader(f))
        except Exception:
            existing = []

    all_rows = existing + rows
    dedup = {}
    for r in all_rows:
        key = (
            str(r.get("provider")),
            str(r.get("kind")),
            str(r.get("station_id")),
            str(r.get("receptor_ids")),
        )
        dedup[key] = r

    write_csv(STATION_BASIN_MAP, list(dedup.values()))


# =============================================================================
# ARPAL / OMIRL — PLAYWRIGHT
# =============================================================================

def require_playwright():
    if sync_playwright is None:
        raise RuntimeError(
            "Manca playwright. Non installarlo mentre altri download lunghi "
            "sono attivi. Quando possibile: pip install playwright && "
            "python -m playwright install chromium"
        )


def options(sel):
    return sel.locator("option").evaluate_all(
        """els => els.map(o => ({
            text: (o.textContent || '').trim(),
            value: o.value || ''
        }))"""
    )


def all_frames(page):
    return list(page.frames)


def frame_by_name(page, name):
    for fr in all_frames(page):
        if fr.name == name:
            return fr
    return None


def wait_until(fn, timeout_s=20, interval=0.35):
    end = time.time() + timeout_s
    while time.time() < end:
        try:
            x = fn()
            if x:
                return x
        except Exception:
            pass
        time.sleep(interval)
    return None


def arpal_snapshot(page):
    out = []
    for idx, fr in enumerate(all_frames(page)):
        item = {
            "index": idx,
            "name": fr.name,
            "url": fr.url,
            "selects": [],
            "inputs": [],
            "links_buttons": [],
        }
        try:
            sels = fr.locator("select")
            for i in range(sels.count()):
                sel = sels.nth(i)
                item["selects"].append({
                    "name": sel.get_attribute("name"),
                    "options": options(sel),
                })
        except Exception:
            pass
        try:
            ins = fr.locator("input")
            for i in range(ins.count()):
                el = ins.nth(i)
                item["inputs"].append({
                    "type": el.get_attribute("type"),
                    "name": el.get_attribute("name"),
                    "value": el.get_attribute("value"),
                })
        except Exception:
            pass
        try:
            els = fr.locator("a,button,input[type=button],input[type=submit],input[type=image]")
            for i in range(min(els.count(), 100)):
                el = els.nth(i)
                try:
                    blob = " | ".join([
                        el.inner_text() or "",
                        el.get_attribute("value") or "",
                        el.get_attribute("alt") or "",
                        el.get_attribute("title") or "",
                    ])
                    if blob.strip(" |"):
                        item["links_buttons"].append(blob)
                except Exception:
                    pass
        except Exception:
            pass
        out.append(item)
    return out


def save_arpal_diag(page, tag, extra=None):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = LIG_DIAG / f"{stamp}_{safe(tag)}"
    try:
        page.screenshot(path=str(base.with_suffix(".png")), full_page=True)
    except Exception:
        pass
    write_json(
        base.with_suffix(".json"),
        {"url": page.url, "frames": arpal_snapshot(page), "extra": extra},
    )
    return base


def enter_arpal_mode(page, mode):
    mode = mode.upper()
    page.goto(ARPAL_PORTAL, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
    page.wait_for_timeout(1000)

    def find_radio():
        for fr in all_frames(page):
            loc = fr.locator(
                f'input[type="radio"][name="TipoTema"][value="{mode}"]'
            )
            if loc.count():
                return fr, loc.first
        return None

    found = wait_until(find_radio, timeout_s=20)
    if not found:
        raise RuntimeError(f"Non trovo TipoTema={mode}")
    _, radio = found
    radio.click()

    def punto_ready():
        fr = frame_by_name(page, "Punto")
        if fr is None:
            return None
        loc = fr.locator('select[name="Ubic"]')
        if loc.count():
            return fr
        return None

    punto = wait_until(punto_ready, timeout_s=20)
    if punto is None:
        raise RuntimeError(f"Modalità {mode}: Ubic non disponibile.")
    return punto


def arpal_catalog(page, mode):
    punto = enter_arpal_mode(page, mode)
    ubic = punto.locator('select[name="Ubic"]')
    rows = []
    for o in options(ubic):
        if not o["value"]:
            continue
        rows.append({"code": o["value"], "name": o["text"]})
    return rows


def choose_catalog_item(catalog, terms):
    terms_n = [norm(x) for x in terms if norm(x)]
    if not terms_n:
        return None

    # exact normalized
    for r in catalog:
        n = norm(r["name"])
        if any(n == t for t in terms_n):
            return r

    # contains all words of candidate term, then any
    for t in terms_n:
        words = [w for w in t.split() if len(w) >= 3]
        for r in catalog:
            n = norm(r["name"])
            if words and all(w in n for w in words):
                return r

    for r in catalog:
        n = norm(r["name"])
        if any(t in n or n in t for t in terms_n):
            return r
    return None


def find_list_stations_control(page):
    for fr in all_frames(page):
        els = fr.locator(
            "a,button,input[type=button],input[type=submit],input[type=image]"
        )
        for i in range(els.count()):
            el = els.nth(i)
            try:
                blob = norm(" ".join([
                    el.inner_text() or "",
                    el.get_attribute("value") or "",
                    el.get_attribute("alt") or "",
                    el.get_attribute("title") or "",
                ]))
            except Exception:
                continue
            if "LISTA STAZION" in blob:
                return fr, el
    return None


def extract_station_names_from_text(text, station_catalog):
    ntext = norm(text)
    hits = []
    # confronto contro il catalogo ufficiale globale: molto più robusto
    # che tentare di interpretare la tabella/popup.
    for r in station_catalog:
        bare = re.sub(r"\s*\([^()]*\)\s*$", "", r["name"])
        if len(norm(bare)) >= 4 and norm(bare) in ntext:
            hits.append(r)
    dedup = {}
    for r in hits:
        dedup[r["code"]] = r
    return list(dedup.values())


def arpal_stations_for_basin(context, basin_rec, station_catalog):
    page = context.new_page()
    try:
        punto = enter_arpal_mode(page, "BACINO")
        basin_catalog = []
        ubic = punto.locator('select[name="Ubic"]')
        for o in options(ubic):
            if o["value"]:
                basin_catalog.append({"code": o["value"], "name": o["text"]})

        target = choose_catalog_item(basin_catalog, basin_rec["terms"])
        if target is None:
            raise RuntimeError(
                f"Bacino non trovato per {basin_rec['receptor_id']}: "
                f"{basin_rec['terms']}"
            )

        ubic.select_option(value=target["code"])
        page.wait_for_timeout(700)

        control = wait_until(lambda: find_list_stations_control(page), 10)
        text = ""

        if control:
            _, el = control
            popup = None
            try:
                with page.expect_popup(timeout=5000) as pi:
                    el.click()
                popup = pi.value
                popup.wait_for_load_state("domcontentloaded", timeout=15000)
                text = popup.locator("body").inner_text(timeout=10000)
                popup.close()
            except Exception:
                # Il click può aggiornare un frame invece di aprire popup.
                try:
                    el.click()
                except Exception:
                    pass
                page.wait_for_timeout(1200)
                chunks = []
                for fr in all_frames(page):
                    try:
                        chunks.append(fr.locator("body").inner_text(timeout=3000))
                    except Exception:
                        pass
                text = "\n".join(chunks)

        # fallback: testo dei frame dopo la selezione
        if not text:
            chunks = []
            for fr in all_frames(page):
                try:
                    chunks.append(fr.locator("body").inner_text(timeout=3000))
                except Exception:
                    pass
            text = "\n".join(chunks)

        hits = extract_station_names_from_text(text, station_catalog)

        if not hits:
            diag = save_arpal_diag(
                page,
                f"basin_station_list_{basin_rec['receptor_id']}",
                {
                    "target_basin": target,
                    "terms": basin_rec["terms"],
                },
            )
            raise RuntimeError(
                f"Nessuna stazione estratta per {target['name']}. "
                f"Diagnostica: {diag}"
            )

        return target, hits

    finally:
        page.close()


def arpal_select_station(page, station):
    punto = enter_arpal_mode(page, "STAZIONE")
    ubic = punto.locator('select[name="Ubic"]')
    ubic.select_option(value=station["code"])
    page.wait_for_timeout(500)

    def freq_ready():
        fr = frame_by_name(page, "Punto")
        if fr is None:
            return None
        f = fr.locator('select[name="Frequenza"]')
        if not f.count():
            return None
        vals = {o["value"]: o["text"] for o in options(f)}
        if "HH" in vals:
            return f
        return None

    freq = wait_until(freq_ready, timeout_s=20)
    if freq is None:
        raise RuntimeError("Frequenza HH non disponibile.")
    freq.select_option(value="HH")
    page.wait_for_timeout(500)

    def param_ready():
        fr = frame_by_name(page, "Scheda")
        if fr is None:
            return None
        p = fr.locator('select[name="Param"]')
        if p.count() and len(options(p)) > 0:
            return p
        return None

    param = wait_until(param_ready, timeout_s=20)
    if param is None:
        raise RuntimeError("Parametro non disponibile nel frame Scheda.")
    return param


def choose_option_terms(sel, terms):
    terms_n = [norm(t) for t in terms]
    opts = options(sel)
    for o in opts:
        txt = norm(o["text"])
        if o["value"] and any(t in txt for t in terms_n):
            sel.select_option(value=o["value"])
            return o
    return None


def arpal_select_output_csv(page):
    scheda = frame_by_name(page, "Scheda")
    out = scheda.locator('select[name="TipoOutput"]')
    vals = {o["value"]: o["text"] for o in options(out)}
    if "XLS" in vals:
        out.select_option(value="XLS")
        return {"value": "XLS", "text": vals["XLS"]}
    if "ASCII" in vals:
        out.select_option(value="ASCII")
        return {"value": "ASCII", "text": vals["ASCII"]}
    raise RuntimeError(f"Nessun output CSV/ASCII: {vals}")


def arpal_fill_period(page, year):
    scheda = frame_by_name(page, "Scheda")
    start_ok = end_ok = False
    rows = scheda.locator("tr")

    for i in range(rows.count()):
        row = rows.nth(i)
        try:
            txt = norm(row.inner_text())
        except Exception:
            continue
        inputs = row.locator('input[type="text"], input:not([type])')
        if "INIZIO PERIODO" in txt and inputs.count():
            inputs.nth(0).fill(f"01/09/{year}")
            if inputs.count() > 1:
                inputs.nth(1).fill("00:00")
            start_ok = True
        if "FINE PERIODO" in txt and inputs.count():
            inputs.nth(0).fill(f"31/12/{year}")
            if inputs.count() > 1:
                inputs.nth(1).fill("23:59")
            end_ok = True

    if not (start_ok and end_ok):
        raise RuntimeError("Campi periodo ARPAL non trovati.")


def arpal_find_access_control(page):
    for fr in all_frames(page):
        els = fr.locator(
            "a,button,input[type=submit],input[type=image],input[type=button]"
        )
        for i in range(els.count()):
            el = els.nth(i)
            try:
                blob = norm(" ".join([
                    el.inner_text() or "",
                    el.get_attribute("value") or "",
                    el.get_attribute("alt") or "",
                    el.get_attribute("title") or "",
                    el.get_attribute("src") or "",
                ]))
            except Exception:
                continue
            if "ACCEDI AI DATI" in blob or "ACCEDIAIDATI" in blob:
                return el
    return None


def arpal_download_current(page, target):
    control = arpal_find_access_control(page)
    if control is None:
        raise RuntimeError("Comando Accedi ai dati non trovato.")

    try:
        with page.expect_download(timeout=DOWNLOAD_TIMEOUT) as di:
            control.click()
        d = di.value
        d.save_as(str(target))
        return "download"
    except PWTimeout:
        page.wait_for_timeout(1500)

    # fallback testo delimitato
    for fr in all_frames(page):
        try:
            text = fr.locator("body").inner_text(timeout=3000)
        except Exception:
            continue
        lines = [x for x in text.splitlines() if x.strip()]
        if len(lines) >= 2 and any(sep in lines[0] for sep in [";", "\t", ","]):
            target.write_text(text, encoding="utf-8")
            return "text"

    # Il portale può mostrare messaggio "nessun dato".
    all_text = ""
    for fr in all_frames(page):
        try:
            all_text += "\n" + fr.locator("body").inner_text(timeout=2000)
        except Exception:
            pass
    nt = norm(all_text)
    if "NESSUN DATO" in nt or "DATI NON DISPONIBILI" in nt:
        return "no_data"

    raise RuntimeError("Nessun download riconoscibile dopo Accedi ai dati.")


def arpal_download_one(
    context,
    station,
    kind,
    year,
    receptor_ids,
    force=False,
):
    slug = f"{station['code']}_{safe(station['name'])}_{kind}"
    folder = LIG_HOURLY / slug
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{slug}_{year}_09-12.csv"

    if file_ok(target, 50) and not force:
        print(f"  SKIP {slug} {year}")
        return "skip"

    page = context.new_page()
    try:
        param = arpal_select_station(page, station)
        terms = PRECIP_TERMS if kind == "precipitation" else LEVEL_TERMS
        chosen = choose_option_terms(param, terms)
        if chosen is None:
            return "parameter_unavailable"

        arpal_select_output_csv(page)
        arpal_fill_period(page, year)
        result = arpal_download_current(page, target)

        if result == "no_data":
            if target.exists():
                target.unlink()
            log_manifest(
                source="ARPAL_OMIRL",
                station_id=station["code"],
                station_name=station["name"],
                kind=kind,
                year=year,
                receptor_ids=receptor_ids,
                status="no_data",
            )
            return "no_data"

        if not file_ok(target, 50):
            raise RuntimeError(f"File assente/troppo piccolo: {target}")

        print(
            f"  OK {slug} {year}: {target.stat().st_size/1024:.1f} kB"
        )
        log_manifest(
            source="ARPAL_OMIRL",
            station_id=station["code"],
            station_name=station["name"],
            kind=kind,
            year=year,
            receptor_ids=receptor_ids,
            status="ok",
            path=str(target),
            bytes=target.stat().st_size,
        )
        return "ok"

    except Exception as exc:
        diag = save_arpal_diag(
            page,
            f"{slug}_{year}",
            {"station": station, "kind": kind, "year": year, "error": str(exc)},
        )
        print(f"  ERROR {slug} {year}: {exc}")
        print(f"  diagnostica: {diag}")
        log_manifest(
            source="ARPAL_OMIRL",
            station_id=station["code"],
            station_name=station["name"],
            kind=kind,
            year=year,
            receptor_ids=receptor_ids,
            status="error",
            error=str(exc),
            diagnostics=str(diag),
        )
        return "error"
    finally:
        page.close()


def arpal_basin_specs(basins):
    specs = []
    for b in basins:
        rid = b["receptor_id"]
        terms = []

        lig_names = b.get("liguria_names") or ""
        for x in str(lig_names).split("|"):
            x = x.strip()
            if x:
                terms.append(x)

        terms.extend(LIG_BASIN_FALLBACKS.get(rid, []))

        # usa ARPAL per i cinque bacini liguri e per i compositi con parte ligure
        if rid in LIG_PRIMARY or terms:
            dedup = []
            seen = set()
            for t in terms:
                nt = norm(t)
                if nt and nt not in seen:
                    seen.add(nt)
                    dedup.append(t)
            if dedup:
                specs.append({"receptor_id": rid, "terms": dedup})
    return specs


def run_liguria(args, basins):
    require_playwright()

    print("\n" + "=" * 100)
    print("ARPAL / OMIRL | OSSERVAZIONI LIGURIA")
    print("=" * 100)

    specs = arpal_basin_specs(basins)
    print("Recettori con parte ligure:", ", ".join(x["receptor_id"] for x in specs))

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        context = browser.new_context(
            accept_downloads=True,
            locale="it-IT",
        )

        page = context.new_page()
        try:
            station_catalog = arpal_catalog(page, "STAZIONE")
            basin_catalog = arpal_catalog(page, "BACINO")
        finally:
            page.close()

        write_json(LIG_CATALOG / "station_catalog.json", station_catalog)
        write_csv(LIG_CATALOG / "station_catalog.csv", station_catalog)
        write_json(LIG_CATALOG / "basin_catalog.json", basin_catalog)
        write_csv(LIG_CATALOG / "basin_catalog.csv", basin_catalog)

        if args.discover:
            print(f"Catalogo stazioni: {len(station_catalog)}")
            print(f"Catalogo bacini  : {len(basin_catalog)}")
            for spec in specs:
                try:
                    target, stations = arpal_stations_for_basin(
                        context, spec, station_catalog
                    )
                    print(
                        f"{spec['receptor_id']}: {target['name']} -> "
                        f"{len(stations)} stazioni"
                    )
                    write_csv(
                        LIG_CATALOG / f"{spec['receptor_id']}_stations.csv",
                        stations,
                    )
                except Exception as exc:
                    print(f"{spec['receptor_id']}: ERROR {exc}")
            browser.close()
            return

        # Costruisce un'unica mappa stazione -> recettori.
        station_to_receptors = defaultdict(set)
        station_by_code = {}

        specs_to_run = specs
        if args.test:
            specs_to_run = [
                x for x in specs
                if x["receptor_id"] == "LIG_BISAGNO"
            ][:1] or specs[:1]

        for spec in specs_to_run:
            target, stations = arpal_stations_for_basin(
                context, spec, station_catalog
            )
            print(
                f"\n{spec['receptor_id']} | {target['name']} | "
                f"{len(stations)} stazioni"
            )
            write_csv(
                LIG_CATALOG / f"{spec['receptor_id']}_stations.csv",
                stations,
            )

            maxn = args.max_stations_per_basin
            if args.test:
                maxn = 1
            if maxn > 0:
                stations = stations[:maxn]

            for st in stations:
                station_by_code[st["code"]] = st
                station_to_receptors[st["code"]].add(spec["receptor_id"])

        map_rows = []
        for code, receptors in sorted(station_to_receptors.items()):
            st = station_by_code[code]
            map_rows.append({
                "provider": "ARPAL_OMIRL",
                "kind": "candidate_precip_or_hydro",
                "station_id": code,
                "station_name": st["name"],
                "latitude": "",
                "longitude": "",
                "receptor_ids": " | ".join(sorted(receptors)),
                "nearest_receptor": sorted(receptors)[0],
                "distance_to_nearest_basin_km": "",
            })
        append_station_map(map_rows)
        write_csv(LIG_CATALOG / "station_basin_map.csv", map_rows)

        years = [2020] if args.test else list(range(args.start_year, args.end_year + 1))

        # Per ogni stazione proviamo precipitation e water_level.
        for idx, (code, receptors) in enumerate(sorted(station_to_receptors.items()), 1):
            st = station_by_code[code]
            rids = sorted(receptors)
            print(
                f"\n[{idx}/{len(station_to_receptors)}] "
                f"{st['code']} | {st['name']} | {rids}"
            )

            # Probe rapido dei parametri; poi i singoli anni gestiscono
            # eventuali indisponibilità temporali.
            for kind in ["precipitation", "water_level"]:
                errors = 0
                for year in years:
                    status = arpal_download_one(
                        context,
                        st,
                        kind,
                        year,
                        rids,
                        force=args.force,
                    )
                    if status == "parameter_unavailable":
                        print(f"  {kind}: parametro non disponibile -> stop")
                        break
                    if status == "error":
                        errors += 1
                    else:
                        errors = 0
                    if errors >= 3:
                        print(
                            f"  {kind}: 3 errori consecutivi -> stop prudenziale"
                        )
                        break
                    time.sleep(0.5)

        browser.close()


# =============================================================================
# VALLE D'AOSTA — ASSISTED OFFICIAL PORTAL
# =============================================================================

def vda_snapshot(page):
    obj = {
        "url": page.url,
        "selects": [],
        "inputs": [],
        "buttons_links": [],
    }
    try:
        sels = page.locator("select")
        for i in range(sels.count()):
            sel = sels.nth(i)
            try:
                opts = sel.locator("option").evaluate_all(
                    """els => els.map(o => ({
                        text:(o.textContent||'').trim(),
                        value:o.value||''
                    }))"""
                )
            except Exception:
                opts = []
            obj["selects"].append({
                "index": i,
                "name": sel.get_attribute("name"),
                "id": sel.get_attribute("id"),
                "options": opts,
            })
    except Exception:
        pass

    try:
        ins = page.locator("input")
        for i in range(ins.count()):
            el = ins.nth(i)
            obj["inputs"].append({
                "index": i,
                "type": el.get_attribute("type"),
                "name": el.get_attribute("name"),
                "id": el.get_attribute("id"),
                "placeholder": el.get_attribute("placeholder"),
                "value": el.get_attribute("value"),
            })
    except Exception:
        pass

    try:
        els = page.locator("button,a,input[type=submit],input[type=button]")
        for i in range(min(els.count(), 200)):
            el = els.nth(i)
            try:
                blob = " | ".join([
                    el.inner_text() or "",
                    el.get_attribute("value") or "",
                    el.get_attribute("title") or "",
                ])
                if blob.strip(" |"):
                    obj["buttons_links"].append(blob)
            except Exception:
                pass
    except Exception:
        pass
    return obj


def save_vda_diag(page, tag):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = VDA_DIAG / f"{stamp}_{safe(tag)}"
    try:
        page.screenshot(path=str(base.with_suffix(".png")), full_page=True)
    except Exception:
        pass
    try:
        base.with_suffix(".html").write_text(
            page.content(), encoding="utf-8"
        )
    except Exception:
        pass
    write_json(base.with_suffix(".json"), vda_snapshot(page))
    return base


def unique_download_path(suggested):
    name = safe(Path(suggested).stem) + Path(suggested).suffix
    if not Path(name).suffix:
        name += ".bin"
    target = VDA_DOWNLOADS / name
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    for i in range(2, 10000):
        p = target.with_name(f"{stem}_{i}{suffix}")
        if not p.exists():
            return p
    raise RuntimeError("Troppi file omonimi VdA.")


def run_vda(args):
    require_playwright()

    print("\n" + "=" * 100)
    print("CENTRO FUNZIONALE VALLE D'AOSTA | DATAVIEW")
    print("=" * 100)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        context = browser.new_context(
            accept_downloads=True,
            locale="it-IT",
        )
        page = context.new_page()
        page.goto(VDA_PORTAL, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
        page.wait_for_timeout(2000)

        diag = save_vda_diag(page, "initial_form")
        print(f"Diagnostica iniziale: {diag}")

        if args.discover:
            print(
                "DISCOVERY VdA completata. Nessun dato personale richiesto "
                "e nessun download eseguito."
            )
            browser.close()
            return

        if not args.vda_assisted:
            print(
                "\nVdA: il portale ufficiale richiede campi personali e consensi. "
                "Per rispetto del flusso ufficiale questa v1.1 non li inventa "
                "e non li memorizza."
            )
            print(
                "Rilancia con:\n"
                "  python download_observations_nw_v1_1.py "
                "--provider vda --vda-assisted --headed"
            )
            browser.close()
            return

        if not args.headed:
            raise ValueError("--vda-assisted richiede --headed")

        saved = []

        def on_download(download):
            try:
                target = unique_download_path(download.suggested_filename)
                download.save_as(str(target))
                saved.append(target)
                print(f"\n[VdA] DOWNLOAD salvato -> {target}")
                log_manifest(
                    source="RAVDA_DATAVIEW",
                    status="ok",
                    path=str(target),
                    suggested_filename=download.suggested_filename,
                )
            except Exception as exc:
                print(f"\n[VdA] errore salvataggio download: {exc}")

        page.on("download", on_download)

        print(
            "\nIl browser ufficiale VdA è aperto.\n"
            "Nel form seleziona i dati della Dora Baltea/rete regionale utili "
            "al modello:\n"
            "  - periodo settembre-dicembre per gli anni disponibili;\n"
            "  - passo ORARIO quando disponibile;\n"
            "  - PRECIPITAZIONE;\n"
            "  - ALTEZZA IDROMETRICA;\n"
            "  - tutte le stazioni pertinenti alla Dora Baltea.\n"
            "Compila personalmente i campi obbligatori e i consensi del sito.\n"
            "I file scaricati saranno intercettati e salvati in observations_nw/vda/downloads/.\n"
            "Questo script NON registra nome, cognome, email o altri dati personali."
        )
        input(
            "\nQuando hai terminato tutti i download nel browser, "
            "torna qui e premi INVIO..."
        )

        page.wait_for_timeout(1500)
        save_vda_diag(page, "final_form_state")
        browser.close()

        print(f"File VdA intercettati in questa sessione: {len(saved)}")
        for p in saved:
            print("  ", p)


# =============================================================================
# LOCAL AUDIT
# =============================================================================

def count_files(path, pattern="*"):
    return sum(1 for p in path.rglob(pattern) if p.is_file()) if path.exists() else 0


def local_audit():
    print("=" * 100)
    print("OBSERVATIONS NW | INVENTARIO LOCALE")
    print("=" * 100)
    print("Piemonte daily meteo CSV :", count_files(PIE_METEO, "*.csv"))
    print("Piemonte daily hydro CSV :", count_files(PIE_HYDRO, "*.csv"))
    print("Liguria hourly CSV        :", count_files(LIG_HOURLY, "*.csv"))
    print("VdA download file         :", count_files(VDA_DOWNLOADS, "*"))
    print("Station-basin map         :", STATION_BASIN_MAP if STATION_BASIN_MAP.exists() else "MANCANTE")
    print("Manifest                  :", MANIFEST if MANIFEST.exists() else "MANCANTE")
    print("QC Piemonte               :", (PIE_QC / "piemonte_observations_qc.txt") if (PIE_QC / "piemonte_observations_qc.txt").exists() else "MANCANTE")
    print("=" * 100)


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    setup_dirs()
    write_readme()

    if args.start_year > args.end_year:
        raise ValueError("--start-year deve essere <= --end-year")

    if args.audit:
        local_audit()
        return

    basins = load_basins()

    print("=" * 100)
    print("NW OBSERVATIONS COLLECTOR v1.1")
    print(f"Recettori: {len(basins)}")
    print(f"Periodo: {args.start_year}-{args.end_year}, settembre-dicembre")
    print(f"Provider: {args.provider}")
    print(f"Output: {OUT}")
    print("=" * 100)

    providers = (
        ["piemonte", "liguria", "vda"]
        if args.provider == "all"
        else [args.provider]
    )

    for provider in providers:
        if provider == "piemonte":
            run_piemonte(args, basins)
        elif provider == "liguria":
            run_liguria(args, basins)
        elif provider == "vda":
            run_vda(args)

    print("\n" + "=" * 100)
    print("FINE NW OBSERVATIONS COLLECTOR")
    print(f"Output: {OUT}")
    print(f"Manifest: {MANIFEST}")
    print(f"Mappa stazioni-bacini: {STATION_BASIN_MAP}")
    print("=" * 100)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrotto. I file già completati restano validi.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nFATAL: {exc}", file=sys.stderr)
        sys.exit(1)
