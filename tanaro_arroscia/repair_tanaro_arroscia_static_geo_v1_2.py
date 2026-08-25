#!/usr/bin/env python3
"""
Tanaro–Arroscia | Static geodata verifier / repair v1.2
========================================================

Scopo
-----
Verifica e ripara SOLO i layer vettoriali scaricati dalla v1.1.

Perché serve
------------
Nel download v1.1 alcuni layer Liguria hanno restituito esattamente
5000 feature, indizio forte del limite di risposta WFS.

Inoltre alcuni layer Piemonte hanno mostrato:
    count probe > feature GeoJSON scaricate

Questa v1.2:

LIGURIA
-------
- interroga WFS 2.0 con resultType=hits per ottenere numberMatched;
- confronta numberMatched con il GeoJSON locale;
- se manca qualcosa, usa paginazione:
      startIndex=0,2000,4000,...
      count=2000
- unisce tutte le pagine in un unico GeoJSON;
- verifica che il conteggio finale coincida con numberMatched.

PIEMONTE
--------
- usa returnIdsOnly=true con lo stesso filtro spaziale;
- ottiene l'elenco completo degli ObjectID;
- scarica gli oggetti a blocchi tramite objectIds;
- ricostruisce il GeoJSON completo;
- verifica che numero feature == numero ObjectID.

Non riscarica:
- DEM TINITALY
- archivi PAI/WFD
- metadata
- layer già verificati completi, salvo --force.

Uso
---
    python repair_tanaro_arroscia_static_geo_v1_2.py

Forza riscrittura di tutti i layer:
    python repair_tanaro_arroscia_static_geo_v1_2.py --force
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import requests


# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "tanaro_arroscia" / "static_geo"

PIE = BASE / "piemonte"
LIG = BASE / "liguria"
DIAG = BASE / "_diagnostics"
REPORT = BASE / "static_geo_integrity_report_v1_2.json"
MANIFEST = BASE / "static_geo_repair_manifest_v1_2.jsonl"

BBOX = {
    "west": 7.68,
    "south": 43.95,
    "east": 8.18,
    "north": 44.28,
}

TIMEOUT = 120
RETRIES = 6
TRANSIENT = {429, 500, 502, 503, 504}

# GeoServer page size deliberately below common maxFeatures limits.
WFS_PAGE_SIZE = 2000

# ArcGIS objectIds are requested in modest chunks.
ARCGIS_ID_CHUNK = 200


PIE_ARCGIS_LAYERS = [
    {
        "group": "reticolo",
        "slug": "bdtre_elemento_idrico",
        "title": "BDTRE - Elemento idrico",
        "url": (
            "https://webgis.arpa.piemonte.it/server/rest/services/test/"
            "Idrografia_Elementi_Lineari_WFS_Regione/MapServer/0/query"
        ),
    },
    {
        "group": "reticolo",
        "slug": "wfd_corpi_idrici_fiumi",
        "title": "Corpi idrici Fiumi WFD",
        "url": (
            "https://webgis.arpa.piemonte.it/ags/rest/services/acqua/"
            "Reticolo_idrografico_WFD_2000_60_CE/FeatureServer/1/query"
        ),
    },
    {
        "group": "bacini",
        "slug": "wfd_bacini_corpi_idrici_fiumi",
        "title": "Bacini dei corpi idrici Fiumi WFD",
        "url": (
            "https://webgis.arpa.piemonte.it/ags/rest/services/acqua/"
            "Reticolo_idrografico_WFD_2000_60_CE/FeatureServer/4/query"
        ),
    },
    {
        "group": "geologia",
        "slug": "geopiemonte_faglie_contatti_tettonici",
        "title": "GeoPiemonte - Faglie e contatti tettonici",
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
     "IFFI"),
    (1907, "litologia",
     "Litologia"),
    (2090, "uso_suolo_2019",
     "Uso del suolo 2019"),
]

LIG_GEOSERVER = "https://geoservizi.regione.liguria.it/geoserver"


# =============================================================================
# BASIC UTILS
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def setup():
    DIAG.mkdir(parents=True, exist_ok=True)


def safe(x):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x)).strip("_")


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def load_geojson_count(path: Path):
    if not path.exists():
        return None

    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        features = obj.get("features")
        if isinstance(features, list):
            return len(features)
    except Exception:
        pass

    return None


def append_manifest(**rec):
    rec["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    with MANIFEST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


# =============================================================================
# HTTP
# =============================================================================

def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 Tanaro-Arroscia static geodata integrity checker"
        ),
        "Accept": "*/*",
    })
    return s


def get(s, url, params=None, label="request"):
    last = None

    for attempt in range(1, RETRIES + 1):
        try:
            r = s.get(
                url,
                params=params,
                timeout=TIMEOUT,
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

            if attempt >= RETRIES:
                break

            wait = min(3 * (2 ** (attempt - 1)), 30)
            print(f"      retry {attempt}/{RETRIES}: {exc}")
            time.sleep(wait)

    raise RuntimeError(f"{label}: fallito: {last}")


# =============================================================================
# PIEMONTE ArcGIS — IDs FIRST
# =============================================================================

def arcgis_spatial_params():
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


def arcgis_get_ids(s, url):
    p = arcgis_spatial_params()
    p.update({
        "returnIdsOnly": "true",
        "f": "json",
    })

    obj = get(s, url, p, "ArcGIS returnIdsOnly").json()

    if "error" in obj:
        raise RuntimeError(obj["error"])

    ids = obj.get("objectIds") or []
    return sorted(set(ids))


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


def arcgis_get_features_by_ids(s, url, ids):
    features = []

    for idx, batch in enumerate(chunks(ids, ARCGIS_ID_CHUNK), 1):
        p = {
            "objectIds": ",".join(str(x) for x in batch),
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
        }

        obj = get(
            s,
            url,
            p,
            f"ArcGIS objectIds batch {idx}",
        ).json()

        if "error" in obj:
            raise RuntimeError(obj["error"])

        part = obj.get("features") or []
        features.extend(part)

    return features


def repair_piemonte(s, force=False):
    print("\n" + "=" * 100)
    print("PIEMONTE | VERIFICA OBJECT IDs")
    print("=" * 100)

    results = []

    for item in PIE_ARCGIS_LAYERS:
        target = PIE / item["group"] / f"{item['slug']}.geojson"
        local_count = load_geojson_count(target)

        print(f"\n  {item['title']}")
        print(f"    locale: {local_count}")

        try:
            ids = arcgis_get_ids(s, item["url"])
            expected = len(ids)
            print(f"    objectIds servizio: {expected}")

            if not force and local_count == expected:
                print("    OK già completo")
                status = "complete"
                final_count = local_count

            else:
                print("    RISCRIVO tramite ObjectID...")
                features = arcgis_get_features_by_ids(
                    s,
                    item["url"],
                    ids,
                )

                final_count = len(features)
                write_json(
                    target,
                    {
                        "type": "FeatureCollection",
                        "features": features,
                    },
                )

                if final_count == expected:
                    print(f"    OK riparato: {final_count}/{expected}")
                    status = "repaired"
                else:
                    print(
                        f"    ATTENZIONE: {final_count}/{expected}; "
                        "salvo elenco IDs per diagnosi"
                    )
                    status = "mismatch_after_repair"

                    # Try to discover returned object-id field from properties.
                    write_json(
                        DIAG / f"piemonte_{item['slug']}_expected_ids.json",
                        ids,
                    )

            rec = {
                "source": "Piemonte",
                "layer": item["slug"],
                "local_before": local_count,
                "expected": expected,
                "final": final_count,
                "status": status,
            }
            results.append(rec)
            append_manifest(**rec)

        except Exception as exc:
            print(f"    ERROR: {exc}")
            rec = {
                "source": "Piemonte",
                "layer": item["slug"],
                "local_before": local_count,
                "status": "error",
                "error": str(exc),
            }
            results.append(rec)
            append_manifest(**rec)

    return results


# =============================================================================
# LIGURIA WFS — HITS + PAGINATION
# =============================================================================

def lig_url(mid):
    return f"{LIG_GEOSERVER}/M{mid}/ows"


def localname(tag):
    return tag.split("}", 1)[-1] if "}" in tag else tag


def wfs_feature_types(s, mid):
    p = {
        "service": "WFS",
        "request": "GetCapabilities",
        "version": "2.0.0",
    }
    data = get(s, lig_url(mid), p, f"M{mid} capabilities").content
    root = ET.fromstring(data)

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


def wfs_bbox_value():
    return (
        f"{BBOX['west']},{BBOX['south']},"
        f"{BBOX['east']},{BBOX['north']},EPSG:4326"
    )


def wfs_number_matched(s, mid, typename):
    p = {
        "service": "WFS",
        "request": "GetFeature",
        "version": "2.0.0",
        "typeNames": typename,
        "srsName": "EPSG:4326",
        "bbox": wfs_bbox_value(),
        "resultType": "hits",
    }

    content = get(
        s,
        lig_url(mid),
        p,
        f"WFS hits {typename}",
    ).content

    root = ET.fromstring(content)

    # WFS 2.0 standard attribute.
    value = root.attrib.get("numberMatched")

    if value is not None and str(value).isdigit():
        return int(value)

    # Some GeoServer configurations expose numberOfFeatures.
    value = root.attrib.get("numberOfFeatures")

    if value is not None and str(value).isdigit():
        return int(value)

    diag = DIAG / f"liguria_{mid}_{safe(typename)}_hits.xml"
    diag.write_bytes(content)

    raise RuntimeError(
        "Impossibile leggere numberMatched/numberOfFeatures "
        f"da resultType=hits; vedi {diag}"
    )


def wfs_page(s, mid, typename, start_index, count):
    p = {
        "service": "WFS",
        "request": "GetFeature",
        "version": "2.0.0",
        "typeNames": typename,
        "srsName": "EPSG:4326",
        "bbox": wfs_bbox_value(),
        "outputFormat": "application/json",
        "startIndex": str(start_index),
        "count": str(count),
    }

    r = get(
        s,
        lig_url(mid),
        p,
        f"WFS page {typename} {start_index}",
    )

    if not r.content.lstrip().startswith(b"{"):
        diag = (
            DIAG
            / f"liguria_{mid}_{safe(typename)}_page_{start_index}.bin"
        )
        diag.write_bytes(r.content)
        raise RuntimeError(
            f"pagina non GeoJSON; vedi {diag}"
        )

    obj = r.json()
    return obj.get("features") or []


def wfs_all_features(s, mid, typename, expected):
    features = []
    start = 0

    while start < expected:
        page = wfs_page(
            s,
            mid,
            typename,
            start,
            WFS_PAGE_SIZE,
        )

        if not page:
            break

        features.extend(page)

        print(
            f"        pagina start={start}: "
            f"{len(page)} | totale={len(features)}/{expected}"
        )

        start += len(page)

        # Avoid infinite loops if server ignores startIndex.
        if len(features) > expected + WFS_PAGE_SIZE:
            raise RuntimeError(
                "WFS pagination anomala: più feature del previsto."
            )

    return features


def repair_liguria(s, force=False):
    print("\n" + "=" * 100)
    print("LIGURIA | VERIFICA WFS numberMatched + PAGINAZIONE")
    print("=" * 100)

    results = []

    for mid, slug, title in LIG_MAPS:
        print(f"\n  M{mid} | {title}")

        try:
            types = wfs_feature_types(s, mid)
        except Exception as exc:
            print(f"    ERROR capabilities: {exc}")
            continue

        outdir = LIG / "vectors" / f"M{mid}_{slug}"

        for typename, typename_title in types:
            target = outdir / f"{safe(typename)}.geojson"
            local_count = load_geojson_count(target)

            print(f"\n    {typename} | {typename_title}")
            print(f"      locale: {local_count}")

            try:
                expected = wfs_number_matched(
                    s,
                    mid,
                    typename,
                )
                print(f"      numberMatched: {expected}")

                if not force and local_count == expected:
                    print("      OK già completo")
                    final_count = local_count
                    status = "complete"

                else:
                    print("      RISCRIVO con paginazione WFS 2.0...")
                    features = wfs_all_features(
                        s,
                        mid,
                        typename,
                        expected,
                    )
                    final_count = len(features)

                    write_json(
                        target,
                        {
                            "type": "FeatureCollection",
                            "features": features,
                        },
                    )

                    if final_count == expected:
                        print(f"      OK riparato: {final_count}/{expected}")
                        status = "repaired"
                    else:
                        print(
                            f"      ATTENZIONE finale: "
                            f"{final_count}/{expected}"
                        )
                        status = "mismatch_after_repair"

                rec = {
                    "source": "Liguria",
                    "map_id": mid,
                    "layer": typename,
                    "local_before": local_count,
                    "expected": expected,
                    "final": final_count,
                    "status": status,
                }
                results.append(rec)
                append_manifest(**rec)

            except Exception as exc:
                print(f"      ERROR: {exc}")
                rec = {
                    "source": "Liguria",
                    "map_id": mid,
                    "layer": typename,
                    "local_before": local_count,
                    "status": "error",
                    "error": str(exc),
                }
                results.append(rec)
                append_manifest(**rec)

    return results


# =============================================================================
# REPORT
# =============================================================================

def summarize(results):
    counts = {}
    for rec in results:
        status = rec.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def main():
    args = parse_args()
    setup()
    s = make_session()

    print("=" * 100)
    print("TANARO–ARROSCIA | STATIC GEODATA VERIFY / REPAIR v1.2")
    print(f"Base: {BASE}")
    print("=" * 100)

    pie = repair_piemonte(s, force=args.force)
    lig = repair_liguria(s, force=args.force)

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "bbox_wgs84": BBOX,
        "piemonte": pie,
        "liguria": lig,
        "summary": summarize(pie + lig),
    }

    write_json(REPORT, report)

    print("\n" + "=" * 100)
    print("RIEPILOGO INTEGRITÀ")
    print("=" * 100)

    for status, n in sorted(report["summary"].items()):
        print(f"{status:<24} {n}")

    mismatches = [
        x
        for x in pie + lig
        if x.get("status") in {
            "error",
            "mismatch_after_repair",
        }
    ]

    print(f"\nReport: {REPORT}")

    if mismatches:
        print(
            f"ATTENZIONE: restano {len(mismatches)} layer "
            "da controllare."
        )
        sys.exit(2)

    print("TUTTI I LAYER VERIFICATI / RIPARATI CORRETTAMENTE.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrotto. Puoi rilanciare lo script.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nFATAL: {exc}", file=sys.stderr)
        sys.exit(1)
