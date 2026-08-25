#!/usr/bin/env python3
"""
Tanaro–Arroscia | Static geodata downloader v1.1
=================================================

Versione corretta dopo il probe reale.

Problemi emersi nel probe v1.0
------------------------------
- Geoportale Piemonte: metadata API / WCS / WFS diretti -> HTTP 401 da script.
- Liguria M2056: workspace raggiungibile, ma nessuna coverage WCS utile esposta.
- Liguria WFS vettoriali: OK.

Strategia v1.1
--------------
1. DEM UNICO DI LAVORO: TINITALY 1.1, DTM bare-earth 10 m, INGV, CC BY 4.0.
   Vengono scaricati i 4 tasselli che coprono interamente il bbox di studio:
     w48535_s10
     w48540_s10
     w49035_s10
     w49040_s10

2. PIEMONTE:
   - reticolo idrografico dettagliato BDTRE tramite ArcGIS REST ARPA Piemonte
   - corpi idrici WFD
   - bacini dei corpi idrici WFD
   - GeoPiemonte: faglie/contatti tettonici, Quaternario, Substrato
   - SIFraP: frane puntuali/lineari/superficiali/areali
   - PAI fasce fluviali: tentativo download ZIP ufficiale regionale

3. LIGURIA:
   - Reticolo e bacini DGR 1280/2023
   - Fasce fluviali
   - Suscettività al dissesto
   - Rischio geomorfologico/idrogeologico
   - Rischio idraulico/idrogeologico
   - IFFI
   - Litologia
   - Uso del suolo 2019
   Tutto via WFS GeoServer, ritagliato sul bbox.

4. DTM regionali 5 m:
   Lo script salva i link ufficiali per il download manuale successivo.
   Li useremo come upgrade nella fase di tracciamento dettagliato della galleria.

Area di studio WGS84
--------------------
W = 7.68
S = 43.95
E = 8.18
N = 44.28

Uso
---
Probe rapido:
    python download_tanaro_arroscia_static_geo_v1_1.py --probe

Download completo:
    python download_tanaro_arroscia_static_geo_v1_1.py

Senza DEM:
    python download_tanaro_arroscia_static_geo_v1_1.py --no-dem

Forza riscarico:
    python download_tanaro_arroscia_static_geo_v1_1.py --force

Requisiti
---------
    pip install -U requests
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

import requests


# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "tanaro_arroscia" / "static_geo"

DEM_DIR = BASE / "dem_tinitaly_10m"
PIE = BASE / "piemonte"
LIG = BASE / "liguria"
META = BASE / "metadata"
DIAG = BASE / "_diagnostics"
MANIFEST = BASE / "static_geo_manifest_v1_1.jsonl"

BBOX = {
    "west": 7.68,
    "south": 43.95,
    "east": 8.18,
    "north": 44.28,
}

MIN_BYTES = 100
TIMEOUT = 120
STREAM_TIMEOUT = 600
RETRIES = 6
TRANSIENT = {429, 500, 502, 503, 504}

TINITALY_TILES = [
    "w48535_s10",
    "w48540_s10",
    "w49035_s10",
    "w49040_s10",
]

TINITALY_BASE = "https://tinitaly.pi.ingv.it/data"

# ---------------------------------------------------------------------------
# Piemonte - ArcGIS REST pubblici
# ---------------------------------------------------------------------------

PIE_ARCGIS_LAYERS = [
    {
        "group": "reticolo",
        "slug": "bdtre_elemento_idrico",
        "title": "BDTRE - Elemento idrico (reticolo dettagliato)",
        "url": (
            "https://webgis.arpa.piemonte.it/server/rest/services/test/"
            "Idrografia_Elementi_Lineari_WFS_Regione/MapServer/0/query"
        ),
    },
    {
        "group": "reticolo",
        "slug": "wfd_corpi_idrici_fiumi",
        "title": "Corpi idrici Fiumi WFD 2000/60/CE",
        "url": (
            "https://webgis.arpa.piemonte.it/ags/rest/services/acqua/"
            "Reticolo_idrografico_WFD_2000_60_CE/FeatureServer/1/query"
        ),
    },
    {
        "group": "bacini",
        "slug": "wfd_bacini_corpi_idrici_fiumi",
        "title": "Bacini dei corpi idrici fiumi WFD",
        "url": (
            "https://webgis.arpa.piemonte.it/ags/rest/services/acqua/"
            "Reticolo_idrografico_WFD_2000_60_CE/FeatureServer/4/query"
        ),
    },
    {
        "group": "geologia",
        "slug": "geopiemonte_faglie_contatti_tettonici",
        "title": "GeoPiemonte - Contatti tettonici e faglie",
        "url": (
            "https://webgis.arpa.piemonte.it/ags/rest/services/geologia/"
            "Geo_Piemonte_250k/FeatureServer/4/query"
        ),
    },
    {
        "group": "geologia",
        "slug": "geopiemonte_quaternario",
        "title": "GeoPiemonte - Quaternario",
        "url": (
            "https://webgis.arpa.piemonte.it/ags/rest/services/geologia/"
            "Geo_Piemonte_250k/FeatureServer/6/query"
        ),
    },
    {
        "group": "geologia",
        "slug": "geopiemonte_substrato",
        "title": "GeoPiemonte - Substrato",
        "url": (
            "https://webgis.arpa.piemonte.it/ags/rest/services/geologia/"
            "Geo_Piemonte_250k/FeatureServer/7/query"
        ),
    },
    {
        "group": "frane",
        "slug": "sifrap_frane_puntuali",
        "title": "SIFraP - Frane puntuali",
        "url": (
            "https://webgis.arpa.piemonte.it/ags/rest/services/"
            "rischi_naturali/SIFraP_SI_Frane_Piemonte/FeatureServer/0/query"
        ),
    },
    {
        "group": "frane",
        "slug": "sifrap_frane_lineari",
        "title": "SIFraP - Frane lineari",
        "url": (
            "https://webgis.arpa.piemonte.it/ags/rest/services/"
            "rischi_naturali/SIFraP_SI_Frane_Piemonte/FeatureServer/1/query"
        ),
    },
    {
        "group": "frane",
        "slug": "sifrap_frane_superficiali",
        "title": "SIFraP - Frane superficiali poligonali",
        "url": (
            "https://webgis.arpa.piemonte.it/ags/rest/services/"
            "rischi_naturali/SIFraP_SI_Frane_Piemonte/FeatureServer/7/query"
        ),
    },
    {
        "group": "frane",
        "slug": "sifrap_frane_areali",
        "title": "SIFraP - Frane areali",
        "url": (
            "https://webgis.arpa.piemonte.it/ags/rest/services/"
            "rischi_naturali/SIFraP_SI_Frane_Piemonte/FeatureServer/8/query"
        ),
    },
]

PIE_PAI_ZIP = (
    "https://www.datigeo-piem-download.it/direct/Geoportale/"
    "RegionePiemonte/PAI/FASCE_FLUVIALI_VIGENTI.zip"
)

# alternativa ARPA, utile anche come archivio completo
PIE_WFD_ZIP = (
    "https://webgis.arpa.piemonte.it/w-metadoc/Download/"
    "WFD2000_60_CE_BDTRE.zip"
)

# ---------------------------------------------------------------------------
# Liguria GeoServer
# ---------------------------------------------------------------------------

LIG_GEOSERVER = "https://geoservizi.regione.liguria.it/geoserver"

LIG_MAPS = [
    (2542, "reticolo_bacini_dgr1280_2023",
     "Reticolo Idrografico e Bacini DGR 1280/2023"),
    (447, "pdb_fasce_fluviali",
     "P.d.B. Fasce fluviali"),
    (448, "pdb_suscettivita_dissesto",
     "P.d.B. Suscettività al dissesto"),
    (449, "pdb_rischio_geomorfologico",
     "P.d.B. Rischio geomorfologico/idrogeologico"),
    (450, "pdb_rischio_idraulico",
     "P.d.B. Rischio idraulico/idrogeologico"),
    (492, "iffi_frane",
     "Inventario Fenomeni Franosi IFFI"),
    (1907, "litologia",
     "Litologia"),
    (2090, "uso_suolo_2019",
     "Uso del suolo 2019"),
]

REGIONAL_DTM_LINKS = {
    "Piemonte DTM 5m ICE 2009-2011": (
        "https://www.geoportale.piemonte.it/visregpigo/"
        "?action-type=dwl&layer=scDTM5ICE&request=getCapabilities"
        "&title=Scarico-Ripresa+Aerea+ICE+2009-2011+-DTM5"
        "&url=https%3A%2F%2Fgeomap.reteunitaria.piemonte.it%2Fws%2F"
        "taims%2Frp-01%2Ftaimsscaricogp%2Fwms_scaricogp"
        "%3Fservice%3DWMS&version=1.3"
    ),
    "Liguria DTM 5m ed. 2023": (
        "https://srvcarto.regione.liguria.it/geoservices/apps/"
        "viewer/pages/apps/download/index.html?id=2056"
    ),
}


# =============================================================================
# CLI / FS
# =============================================================================

def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", action="store_true")
    p.add_argument("--no-dem", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def setup():
    for p in [
        BASE, DEM_DIR, PIE, LIG, META, DIAG,
        PIE / "reticolo",
        PIE / "bacini",
        PIE / "geologia",
        PIE / "frane",
        PIE / "pai",
        LIG / "vectors",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def file_ok(path: Path, min_bytes=MIN_BYTES):
    return path.exists() and path.is_file() and path.stat().st_size >= min_bytes


def log(**rec):
    rec["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    with MANIFEST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def safe(x):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x)).strip("_")


def write_json(path: Path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def write_manual_links():
    p = BASE / "REGIONAL_DTM_5M_UPGRADE.txt"
    lines = [
        "DTM REGIONALI 5 m — UPGRADE SUCCESSIVO",
        "",
        "La pipeline v1.1 usa TINITALY 1.1 a 10 m come DEM unico e coerente.",
        "Per il tracciato dettagliato della galleria sostituiremo/affiancheremo:",
        "",
    ]
    for name, url in REGIONAL_DTM_LINKS.items():
        lines += [name, url, ""]
    p.write_text("\n".join(lines), encoding="utf-8")


def write_readme():
    p = BASE / "README_STATIC_GEO_V1_1.md"
    txt = f"""# Tanaro–Arroscia — Static geodata v1.1

BBox WGS84:
- W {BBOX['west']}
- S {BBOX['south']}
- E {BBOX['east']}
- N {BBOX['north']}

## DEM di lavoro
TINITALY 1.1, INGV, 10 m, bare-earth, EPSG:32632, CC BY 4.0.

Tasselli:
{chr(10).join('- ' + x for x in TINITALY_TILES)}

## DTM regionali 5 m
Non vengono automatizzati in questa versione perché i servizi di download
regionali risultano interattivi/protetti da automazione. I link sono salvati
in REGIONAL_DTM_5M_UPGRADE.txt.

## Piemonte
- BDTRE elemento idrico
- WFD corpi idrici e bacini
- GeoPiemonte (faglie, Quaternario, Substrato)
- SIFraP
- PAI fasce fluviali (ZIP ufficiale, best effort)

## Liguria
WFS regionali ritagliati sul bbox:
- reticolo e bacini
- fasce fluviali
- suscettività
- rischio geomorfologico e idraulico
- IFFI
- litologia
- uso del suolo
"""
    p.write_text(txt, encoding="utf-8")


# =============================================================================
# HTTP
# =============================================================================

def session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 Chrome/151 Safari/537.36 "
            "Tanaro-Arroscia-study"
        ),
        "Accept": "*/*",
    })
    return s


def req(s, url, *, params=None, stream=False, timeout=None, label="request"):
    last = None
    timeout = timeout or (STREAM_TIMEOUT if stream else TIMEOUT)

    for attempt in range(1, RETRIES + 1):
        try:
            r = s.get(
                url,
                params=params,
                stream=stream,
                timeout=timeout,
                allow_redirects=True,
            )

            if r.status_code in TRANSIENT:
                raise requests.HTTPError(
                    f"HTTP {r.status_code}",
                    response=r,
                )

            r.raise_for_status()
            return r

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

            if attempt == RETRIES:
                break

            wait = min(5 * 2 ** (attempt - 1), 60)
            print(f"      {label}: retry {attempt}/{RETRIES}: {exc}")
            time.sleep(wait)

    raise RuntimeError(f"{label}: fallito: {last}")


def probe_url(s, url, label):
    r = req(s, url, stream=True, label=label)
    size = r.headers.get("content-length", "?")
    ctype = r.headers.get("content-type", "?")
    print(f"    OK {label}: HTTP {r.status_code} | {ctype} | size={size}")
    r.close()


def download(s, url, target: Path, *, force=False, label=None):
    if file_ok(target) and not force:
        print(f"    SKIP {target.name}")
        return target

    label = label or target.name
    tmp = target.with_suffix(target.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    r = req(s, url, stream=True, label=label)
    total = int(r.headers.get("content-length", "0") or 0)
    done = 0

    with tmp.open("wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            done += len(chunk)
            if total and done % (25 * 1024 * 1024) < len(chunk):
                print(
                    f"      {done/1024/1024:.0f}/"
                    f"{total/1024/1024:.0f} MB "
                    f"({100*done/total:.0f}%)"
                )

    if done < MIN_BYTES:
        raise RuntimeError(f"{label}: file troppo piccolo ({done} byte)")

    tmp.replace(target)
    print(f"    OK {target.name}: {target.stat().st_size/1024/1024:.1f} MB")
    return target


def extract_zip(path: Path, dest: Path):
    marker = dest / ".extracted_ok"
    if marker.exists():
        print(f"    SKIP estrazione {path.name}")
        return

    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as z:
        z.extractall(dest)

    marker.write_text(
        datetime.now(timezone.utc).isoformat(),
        encoding="utf-8",
    )
    print(f"    Estratto -> {dest}")


# =============================================================================
# DEM TINITALY
# =============================================================================

def run_dem(s, probe=False, force=False):
    print("\n" + "=" * 96)
    print("DEM UNICO | TINITALY 1.1 — 10 m")
    print("=" * 96)

    for tile in TINITALY_TILES:
        url = f"{TINITALY_BASE}/{tile}/{tile}.zip"
        target = DEM_DIR / f"{tile}.zip"

        if probe:
            try:
                probe_url(s, url, tile)
            except Exception as exc:
                print(f"    ERROR {tile}: {exc}")
            continue

        try:
            download(s, url, target, force=force, label=tile)
            extract_zip(target, DEM_DIR / tile)
            log(status="ok", kind="dem_tinitaly", tile=tile, path=str(target))
        except Exception as exc:
            print(f"    WARN {tile}: {exc}")
            log(status="error", kind="dem_tinitaly", tile=tile, error=str(exc))


# =============================================================================
# ARCGIS REST
# =============================================================================

def arcgis_base_params():
    return {
        "where": "1=1",
        "geometry": (
            f"{BBOX['west']},{BBOX['south']},"
            f"{BBOX['east']},{BBOX['north']}"
        ),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
    }


def arcgis_count(s, query_url):
    p = arcgis_base_params()
    p.update({
        "returnCountOnly": "true",
        "f": "json",
    })
    r = req(s, query_url, params=p, label="ArcGIS count")
    obj = r.json()
    if "error" in obj:
        raise RuntimeError(obj["error"])
    return int(obj.get("count", 0))


def arcgis_download_geojson(
    s,
    query_url,
    target,
    *,
    force=False,
):
    if file_ok(target) and not force:
        print(f"      SKIP {target.name}")
        return

    count = arcgis_count(s, query_url)
    print(f"      features bbox: {count}")

    features = []
    batch = 1000

    for offset in range(0, count, batch):
        p = arcgis_base_params()
        p.update({
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": "4326",
            "resultOffset": str(offset),
            "resultRecordCount": str(batch),
            "f": "geojson",
        })

        r = req(
            s,
            query_url,
            params=p,
            label=f"ArcGIS offset {offset}",
        )

        obj = r.json()
        if "error" in obj:
            raise RuntimeError(obj["error"])

        part = obj.get("features", [])
        features.extend(part)

    fc = {
        "type": "FeatureCollection",
        "features": features,
    }
    write_json(target, fc)
    print(f"      OK {target.name}: {len(features)} features")


def run_piemonte_arcgis(s, probe=False, force=False):
    print("\n" + "=" * 96)
    print("PIEMONTE | ARCGIS REST PUBBLICI")
    print("=" * 96)

    for item in PIE_ARCGIS_LAYERS:
        folder = PIE / item["group"]
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{item['slug']}.geojson"

        print(f"\n  {item['title']}")

        try:
            if probe:
                n = arcgis_count(s, item["url"])
                print(f"      OK count bbox = {n}")
            else:
                arcgis_download_geojson(
                    s,
                    item["url"],
                    target,
                    force=force,
                )
                log(
                    status="ok",
                    kind="piemonte_arcgis",
                    slug=item["slug"],
                    path=str(target),
                )
        except Exception as exc:
            print(f"      WARN: {exc}")
            log(
                status="error",
                kind="piemonte_arcgis",
                slug=item["slug"],
                error=str(exc),
            )


def run_piemonte_archives(s, probe=False, force=False):
    print("\n" + "=" * 96)
    print("PIEMONTE | ARCHIVI UFFICIALI")
    print("=" * 96)

    items = [
        (
            "wfd_bdtre_completo",
            PIE_WFD_ZIP,
            PIE / "reticolo" / "WFD2000_60_CE_BDTRE.zip",
        ),
        (
            "pai_fasce_fluviali_vigenti",
            PIE_PAI_ZIP,
            PIE / "pai" / "FASCE_FLUVIALI_VIGENTI.zip",
        ),
    ]

    for slug, url, target in items:
        print(f"\n  {slug}")
        try:
            if probe:
                probe_url(s, url, slug)
            else:
                download(s, url, target, force=force, label=slug)
                extract_zip(target, target.with_suffix(""))
                log(
                    status="ok",
                    kind="piemonte_archive",
                    slug=slug,
                    path=str(target),
                )
        except Exception as exc:
            print(f"    WARN: {exc}")
            log(
                status="error",
                kind="piemonte_archive",
                slug=slug,
                error=str(exc),
            )


# =============================================================================
# LIGURIA WFS
# =============================================================================

def localname(tag):
    return tag.split("}", 1)[-1] if "}" in tag else tag


def lig_url(mid):
    return f"{LIG_GEOSERVER}/M{mid}/ows"


def wfs_feature_types(xml_bytes):
    root = ET.fromstring(xml_bytes)
    out = []

    for ft in root.iter():
        if localname(ft.tag) != "FeatureType":
            continue

        name = None
        title = None

        for ch in ft:
            ln = localname(ch.tag)
            if ln == "Name" and ch.text:
                name = ch.text.strip()
            elif ln == "Title" and ch.text:
                title = ch.text.strip()

        if name:
            out.append((name, title or name))

    return out


def lig_capabilities(s, mid):
    p = {
        "service": "WFS",
        "request": "GetCapabilities",
        "version": "2.0.0",
    }
    r = req(
        s,
        lig_url(mid),
        params=p,
        label=f"Liguria M{mid} capabilities",
    )
    return r.content


def lig_download_layer(s, mid, typename, target, force=False):
    if file_ok(target) and not force:
        print(f"      SKIP {target.name}")
        return

    p = {
        "service": "WFS",
        "request": "GetFeature",
        "version": "2.0.0",
        "typeNames": typename,
        "srsName": "EPSG:4326",
        "bbox": (
            f"{BBOX['west']},{BBOX['south']},"
            f"{BBOX['east']},{BBOX['north']},EPSG:4326"
        ),
        "outputFormat": "application/json",
    }

    r = req(
        s,
        lig_url(mid),
        params=p,
        label=f"Liguria {typename}",
    )

    # GeoServer può rispondere JSON o XML di eccezione.
    if not r.content.lstrip().startswith(b"{"):
        diag = DIAG / f"liguria_{mid}_{safe(typename)}.bin"
        diag.write_bytes(r.content)
        raise RuntimeError(
            f"risposta non GeoJSON; diagnostica={diag}"
        )

    obj = r.json()
    write_json(target, obj)
    print(
        f"      OK {target.name}: "
        f"{len(obj.get('features', []))} features"
    )


def run_liguria(s, probe=False, force=False):
    print("\n" + "=" * 96)
    print("LIGURIA | WFS REGIONALI")
    print("=" * 96)

    for mid, slug, title in LIG_MAPS:
        print(f"\n  M{mid} | {title}")

        try:
            caps = lig_capabilities(s, mid)
            caps_path = META / f"liguria_M{mid}_wfs_capabilities.xml"
            caps_path.write_bytes(caps)

            types = wfs_feature_types(caps)
            print(f"    feature type: {len(types)}")

            for typename, typename_title in types:
                print(f"      - {typename} | {typename_title}")

            if probe:
                continue

            out = LIG / "vectors" / f"M{mid}_{slug}"
            out.mkdir(parents=True, exist_ok=True)

            for typename, typename_title in types:
                target = out / f"{safe(typename)}.geojson"
                try:
                    lig_download_layer(
                        s,
                        mid,
                        typename,
                        target,
                        force=force,
                    )
                    log(
                        status="ok",
                        kind="liguria_wfs",
                        map_id=mid,
                        typename=typename,
                        path=str(target),
                    )
                except Exception as exc:
                    print(f"      WARN {typename}: {exc}")
                    log(
                        status="error",
                        kind="liguria_wfs",
                        map_id=mid,
                        typename=typename,
                        error=str(exc),
                    )

        except Exception as exc:
            print(f"    WARN M{mid}: {exc}")
            log(
                status="error",
                kind="liguria_map",
                map_id=mid,
                error=str(exc),
            )


# =============================================================================
# SUMMARY
# =============================================================================

def count_files(path):
    return sum(1 for x in path.rglob("*") if x.is_file()) if path.exists() else 0


def main():
    args = args_parser()
    setup()
    write_readme()
    write_manual_links()

    s = session()

    print("=" * 96)
    print("TANARO–ARROSCIA | STATIC GEODATA DOWNLOADER v1.1")
    print(
        "BBOX WGS84: "
        f"{BBOX['west']}, {BBOX['south']}, "
        f"{BBOX['east']}, {BBOX['north']}"
    )
    print(f"Output: {BASE}")
    print("=" * 96)

    if args.probe:
        print("MODALITÀ PROBE — nessun download pesante.")

    if not args.no_dem:
        run_dem(s, probe=args.probe, force=args.force)

    run_piemonte_arcgis(
        s,
        probe=args.probe,
        force=args.force,
    )

    run_piemonte_archives(
        s,
        probe=args.probe,
        force=args.force,
    )

    run_liguria(
        s,
        probe=args.probe,
        force=args.force,
    )

    print("\n" + "=" * 96)
    print("RIEPILOGO")
    print("=" * 96)
    print(f"DEM TINITALY : {count_files(DEM_DIR)} file")
    print(f"Piemonte     : {count_files(PIE)} file")
    print(f"Liguria      : {count_files(LIG)} file")
    print(f"Metadata     : {count_files(META)} file")
    print(f"Diagnostica  : {count_files(DIAG)} file")
    print(f"Manifest     : {MANIFEST}")
    print()
    print(
        "DTM regionali 5 m: link salvati in "
        f"{BASE / 'REGIONAL_DTM_5M_UPGRADE.txt'}"
    )
    print(
        "Nessuna elaborazione GIS/idraulica è stata eseguita."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrotto. Il programma è restart-safe.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nFATAL: {exc}", file=sys.stderr)
        sys.exit(1)
