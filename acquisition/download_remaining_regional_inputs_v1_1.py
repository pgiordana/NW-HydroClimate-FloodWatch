#!/usr/bin/env python3
# DOWNLOAD REMAINING REGIONAL INPUTS v1.1
#
# Completa i dati ancora mancanti del LAVORO PRINCIPALE regionale:
#   1) Copernicus DEM GLO-30 sui 21 recettori;
#   2) snapshot/documenti ufficiali delle soglie Piemonte;
#   3) soglie/procedure Dora Baltea - Valle d'Aosta;
#   4) Libro Blu + appendice soglie meteoidrologiche Liguria.
#
# Per NON interferire con l'eventuale downloader ARPAL Tanaro-Arroscia,
# il comando predefinito NON interroga OMIRL.
#
# Più avanti:
#   --run-liguria-observations --headed
#       richiama download_observations_nw_v1_1.py --provider liguria
#   --run-vda-assisted --headed
#       apre il Dataview VdA tramite lo script osservazioni v1.1
#
# Comandi:
#   python download_remaining_regional_inputs_v1_1.py --probe
#   caffeinate -i python download_remaining_regional_inputs_v1_1.py
#   python download_remaining_regional_inputs_v1_1.py --audit
#
# v1.1 corregge il controllo di dimensione dei tile COG: tile costieri o
# quasi interamente marini possono essere validi anche sotto 1 MB; la
# validazione usa ora firma TIFF + dimensione minima prudenziale 10 kB.
#
# Nota: Copernicus GLO-30 e' un DSM (~30 m), adatto alle feature
# orografiche regionali ma NON sostituisce DTM 5 m/sezioni reali per
# l'eventuale progetto idraulico Tanaro-Arroscia.

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from shapely.geometry import shape
except Exception:
    shape = None


ROOT = Path(__file__).resolve().parent
BASINS = ROOT / "basins_final" / "nw_receptors_final.geojson"

OUT = ROOT / "regional_inputs"
TERRAIN = OUT / "terrain" / "copernicus_dem_glo30"
TILES = TERRAIN / "tiles"
META = TERRAIN / "source_metadata"

THRESH = OUT / "thresholds"
PIE = THRESH / "piemonte"
LIG = THRESH / "liguria"
VDA = THRESH / "vda"

MANIFEST_DIR = OUT / "manifests"
MANIFEST = MANIFEST_DIR / "remaining_inputs_manifest.jsonl"
QC_DIR = OUT / "qc"
QC = QC_DIR / "remaining_inputs_qc.txt"

OBS_SCRIPT = ROOT / "download_observations_nw_v1_1.py"

COP30 = "https://copernicus-dem-30m.s3.amazonaws.com"
TIMEOUT = 120
MAX_RETRIES = 8
TRANSIENT = {429, 500, 502, 503, 504}
CHUNK = 1024 * 1024
USER_AGENT = "NW hydro-meteorological university research - public data downloader"

PIEMONTE_FILES = {
    "piemonte_soglie_idrometriche_snapshot.pdf":
        "https://www.arpa.piemonte.it/rischi_naturali/boll/tabelle_idro.pdf",
    "piemonte_soglie_pluviometriche_snapshot.pdf":
        "https://www.arpa.piemonte.it/rischi_naturali/boll/tabelle_pluvio.pdf",
    "piemonte_previsione_piene_thresholds.html":
        "https://www.arpa.piemonte.it/rischi_naturali/snippets_arpa/piene/",
    "piemonte_livelli_idro_page.html":
        "https://www.arpa.piemonte.it/rischi_naturali/snippets_arpa_graphs/tabella_livelli_idro/",
    "piemonte_livelli_pluv_page.html":
        "https://www.arpa.piemonte.it/rischi_naturali/snippets_arpa_graphs/tabella_livelli_pluv/",
}

VDA_FILES = {
    "vda_dora_baltea_livelli_e_soglie_snapshot.pdf":
        "https://cf.regione.vda.it/allegati/bollettini/dettaglio/livelli_Dora_Baltea.pdf",
    "vda_procedure_sistema_allertamento_dgr_1565_2022.pdf":
        "https://cf.regione.vda.it/uploads/page/10/procedure-sistema-allertamento-dgr-1565-2022.pdf",
}

LIG_SYSTEM_PAGE = (
    "https://www.regione.liguria.it/homepage-protezione-civile/cosa-cerchi/"
    "procedure-di-allerta-per-rischio-meteo-idrogeologico-idraulico-nivologico.html"
)
LIG_BOOK_PAGE = (
    "https://www.regione.liguria.it/component/publiccompetitions/document/"
    "39694:libro-blu-2020.html"
)
LIG_THRESH_PAGE = (
    "https://www.regione.liguria.it/component/publiccompetitions/document/"
    "39696:appendice-elenco-soglie-meteoidrologiche-2020.html"
)


def args_parse():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", action="store_true")
    p.add_argument("--terrain-only", action="store_true")
    p.add_argument("--thresholds-only", action="store_true")
    p.add_argument("--audit", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--buffer-deg", type=float, default=0.10)
    p.add_argument("--run-liguria-observations", action="store_true")
    p.add_argument("--run-vda-assisted", action="store_true")
    p.add_argument("--headed", action="store_true")
    return p.parse_args()


def setup():
    for d in [OUT, TERRAIN, TILES, META, THRESH, PIE, LIG, VDA,
              MANIFEST_DIR, QC_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def now():
    return datetime.now(timezone.utc).isoformat()


def log(**rec):
    rec["timestamp_utc"] = now()
    with MANIFEST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def hb(n):
    x = float(n)
    for u in ["B", "kB", "MB", "GB", "TB"]:
        if x < 1024 or u == "TB":
            return f"{x:.1f} {u}"
        x /= 1024


def request_retry(s, method, url, *, stream=False):
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = s.request(method, url, timeout=TIMEOUT, stream=stream)
            if r.status_code in TRANSIENT:
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            return r
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            last = exc
            code = getattr(getattr(exc, "response", None), "status_code", None)
            if code is not None and code not in TRANSIENT:
                raise
            if attempt == MAX_RETRIES:
                break
            wait = min(5 * 2 ** (attempt - 1), 120)
            print(f"    retry {attempt}/{MAX_RETRIES}: {exc}; attesa {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"fallito {url}: {last}")


def download(s, url, target, *, force=False, min_bytes=100, group="download"):
    if target.exists() and target.stat().st_size >= min_bytes and not force:
        print(f"  REUSE {target.name} ({hb(target.stat().st_size)})")
        return True

    part = target.with_suffix(target.suffix + ".part")
    if part.exists():
        part.unlink()

    print(f"  GET {target.name}")
    t0 = time.time()
    try:
        r = request_retry(s, "GET", url, stream=True)
        ctype = r.headers.get("content-type", "")
        with part.open("wb") as f:
            for chunk in r.iter_content(CHUNK):
                if chunk:
                    f.write(chunk)
        if part.stat().st_size < min_bytes:
            raise RuntimeError(f"file troppo piccolo: {part.stat().st_size} bytes")
        part.replace(target)
        print(f"    OK {hb(target.stat().st_size)} in {time.time()-t0:.1f}s")
        log(group=group, url=url, path=str(target), status="ok",
            bytes=target.stat().st_size, content_type=ctype)
        return True
    except Exception as exc:
        if part.exists():
            part.unlink()
        print(f"    ERROR {exc}")
        log(group=group, url=url, path=str(target), status="error", error=str(exc))
        return False


def save_html(s, url, target, *, force=False):
    if target.exists() and target.stat().st_size > 100 and not force:
        print(f"  REUSE {target.name}")
        return target.read_text(encoding="utf-8", errors="replace")

    print(f"  GET {target.name}")
    r = request_retry(s, "GET", url)
    target.write_text(r.text, encoding="utf-8")
    log(group="official_thresholds", url=url, path=str(target),
        status="ok", bytes=target.stat().st_size,
        content_type=r.headers.get("content-type", ""))
    return r.text


def basin_bounds(buffer_deg):
    if shape is None:
        raise RuntimeError("shapely non disponibile")
    if not BASINS.exists():
        raise FileNotFoundError(f"Manca {BASINS}")

    obj = json.loads(BASINS.read_text(encoding="utf-8"))
    bs = []
    n = 0
    for feat in obj.get("features", []):
        g = shape(feat["geometry"])
        if not g.is_valid:
            g = g.buffer(0)
        bs.append(g.bounds)
        n += 1
    if not bs:
        raise RuntimeError("GeoJSON senza geometrie")

    return n, (
        min(b[0] for b in bs) - buffer_deg,
        min(b[1] for b in bs) - buffer_deg,
        max(b[2] for b in bs) + buffer_deg,
        max(b[3] for b in bs) + buffer_deg,
    )


def tile_name(lat, lon):
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    token = f"{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00"
    stem = f"Copernicus_DSM_COG_10_{token}_DEM"
    url = f"{COP30}/{stem}/{stem}.tif"
    return stem, url


def tiles_for_bbox(bbox):
    minx, miny, maxx, maxy = bbox
    x0, x1 = math.floor(minx), math.floor(maxx - 1e-10)
    y0, y1 = math.floor(miny), math.floor(maxy - 1e-10)
    rows = []
    for lat in range(y0, y1 + 1):
        for lon in range(x0, x1 + 1):
            stem, url = tile_name(lat, lon)
            rows.append({
                "lat_sw": lat,
                "lon_sw": lon,
                "stem": stem,
                "url": url,
                "path": str(TILES / f"{stem}.tif"),
            })
    return rows


def valid_tif(p):
    if not p.exists() or p.stat().st_size < 10_000:
        return False
    with p.open("rb") as f:
        magic = f.read(4)
    return magic in (b"II*\x00", b"MM\x00*")


def dem_one(t, force):
    target = Path(t["path"])
    if valid_tif(target) and not force:
        return {**t, "status": "reuse", "bytes": target.stat().st_size}
    ok = download(make_session(), t["url"], target, force=force,
                  min_bytes=10_000, group="copernicus_dem_glo30")
    return {**t, "status": "ok" if ok and valid_tif(target) else "error",
            "bytes": target.stat().st_size if target.exists() else 0}


def download_dem(a):
    print("\n" + "=" * 100)
    print("COPERNICUS DEM GLO-30 | OROGRAFIA REGIONALE")
    print("=" * 100)

    n, bbox = basin_bounds(a.buffer_deg)
    rows = tiles_for_bbox(bbox)
    print(f"Recettori: {n}")
    print(f"BBOX (+{a.buffer_deg:.2f}°): {bbox}")
    print(f"Tile candidati: {len(rows)}")

    if a.probe:
        cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
        rows = sorted(
            rows,
            key=lambda t: abs(t["lon_sw"]+0.5-cx) + abs(t["lat_sw"]+0.5-cy)
        )[:1]
        print(f"PROBE: solo {rows[0]['stem']}")

    results = []
    workers = max(1, min(a.workers, 6))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(dem_one, t, a.force): t for t in rows}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
            except Exception as exc:
                t = futs[fut]
                r = {**t, "status": "error", "bytes": 0, "error": str(exc)}
            results.append(r)
            print(f"  DEM [{i}/{len(rows)}] {r['stem']} -> "
                  f"{r['status']} {hb(r.get('bytes',0))}")

    results.sort(key=lambda r: (r["lat_sw"], r["lon_sw"]))
    index = TERRAIN / "tile_index.csv"
    with index.open("w", newline="", encoding="utf-8") as f:
        fields = ["lat_sw", "lon_sw", "stem", "status", "bytes", "url", "path"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fields})

    metadata = {
        "dataset": "Copernicus DEM GLO-30 Public",
        "type": "Digital Surface Model (DSM)",
        "nominal_resolution": "~30 m / 1 arc-second",
        "format": "Cloud Optimized GeoTIFF",
        "bbox": bbox,
        "buffer_deg": a.buffer_deg,
        "registry": "https://registry.opendata.aws/copernicus-dem/",
        "source_bucket": "s3://copernicus-dem-30m/",
        "copernicus_collection":
            "https://dataspace.copernicus.eu/explore-data/data-collections/"
            "copernicus-contributing-missions/collections-description/COP-DEM",
        "note": "Per feature orografiche regionali; non per progetto idraulico di dettaglio.",
        "generated_utc": now(),
    }
    (META / "copernicus_dem_glo30_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )

    good = sum(r["status"] in {"ok", "reuse"} for r in results)
    print(f"DEM validi: {good}/{len(results)}")
    return good == len(results)


def hrefs(html, base):
    links = re.findall(r'href\s*=\s*["\']([^"\']+)["\']', html, flags=re.I)
    return [urljoin(base, x.replace("&amp;", "&")) for x in links]


def follow_doc_page(s, page_url, page_file, pdf_file, keywords, force):
    html = save_html(s, page_url, page_file, force=force)
    candidates = []
    for u in hrefs(html, page_url):
        lu = u.lower()
        score = 0
        if ".pdf" in lu:
            score += 6
        if "download.php" in lu:
            score += 5
        for k in keywords:
            if k.lower() in lu:
                score += 2
        if score:
            candidates.append((score, u))
    candidates.sort(reverse=True)

    for _, u in candidates[:10]:
        if download(s, u, pdf_file, force=force, min_bytes=5000,
                    group="official_thresholds"):
            try:
                with pdf_file.open("rb") as f:
                    if f.read(4) == b"%PDF":
                        return True
            except Exception:
                pass
    print(f"  WARN: PDF non individuato automaticamente da {page_url}")
    return False


def parse_piemonte_piene():
    html = PIE / "piemonte_previsione_piene_thresholds.html"
    if pd is None or not html.exists():
        return False
    try:
        tables = pd.read_html(str(html))
    except Exception as exc:
        print(f"  WARN estrazione tabella piene: {exc}")
        return False

    for df in tables:
        cols = " ".join(map(str, df.columns)).upper()
        if "SOGLIA" in cols and ("STAZIONE" in cols or "CORSO" in cols):
            target = PIE / "piemonte_piene_soglie_portata_snapshot.csv"
            df.to_csv(target, index=False)
            print(f"  OK tabella soglie portata -> {target.name} ({len(df)} righe)")
            log(group="official_thresholds", path=str(target),
                status="parsed_ok", rows=len(df), derived_from=str(html))
            return True
    print("  WARN nessuna tabella piene riconosciuta.")
    return False


def download_thresholds(a):
    print("\n" + "=" * 100)
    print("SOGLIE UFFICIALI / DOCUMENTI DI RIFERIMENTO")
    print("=" * 100)
    s = make_session()

    print("\nPIEMONTE")
    for name, url in PIEMONTE_FILES.items():
        target = PIE / name
        if name.endswith(".html"):
            try:
                save_html(s, url, target, force=a.force)
            except Exception as exc:
                print(f"  ERROR {name}: {exc}")
        else:
            download(s, url, target, force=a.force, min_bytes=5000,
                     group="official_thresholds")
    parse_piemonte_piene()

    print("\nVALLE D'AOSTA")
    for name, url in VDA_FILES.items():
        download(s, url, VDA / name, force=a.force, min_bytes=5000,
                 group="official_thresholds")

    print("\nLIGURIA")
    try:
        save_html(s, LIG_SYSTEM_PAGE,
                  LIG / "liguria_sistema_allertamento_page.html",
                  force=a.force)
    except Exception as exc:
        print(f"  ERROR pagina sistema Liguria: {exc}")

    follow_doc_page(
        s, LIG_BOOK_PAGE,
        LIG / "liguria_libro_blu_2020_page.html",
        LIG / "liguria_libro_blu_2020.pdf",
        ["libro", "blu", "2020"], a.force
    )

    follow_doc_page(
        s, LIG_THRESH_PAGE,
        LIG / "liguria_elenco_soglie_meteoidrologiche_2020_page.html",
        LIG / "liguria_elenco_soglie_meteoidrologiche_2020.pdf",
        ["soglie", "meteoidrologiche", "2020"], a.force
    )

    note = {
        "generated_utc": now(),
        "piemonte": "Snapshot correnti ARPA: soglie idrometriche, pluviometriche e soglie di portata del bollettino piene.",
        "vda": "Il PDF Dora Baltea riporta H1/H2/H3; le procedure DGR 1565/2022 ne descrivono il significato.",
        "liguria": "Libro Blu 2020 e appendice ufficiale soglie sono riferimenti; nel dataset finale le soglie per stazione saranno ricontrollate contro OMIRL/Allerta Liguria per possibili aggiornamenti.",
    }
    (THRESH / "README_THRESHOLDS.json").write_text(
        json.dumps(note, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )


def other_arpal_running():
    try:
        r = subprocess.run(
            ["pgrep", "-fl", "download_arpal_tanaro_arroscia"],
            capture_output=True, text=True, timeout=5
        )
        return r.returncode == 0 and bool(r.stdout.strip()), r.stdout.strip()
    except Exception:
        return False, ""


def run_liguria(a):
    print("\n" + "=" * 100)
    print("OSSERVAZIONI REGIONALI LIGURIA")
    print("=" * 100)

    if not OBS_SCRIPT.exists():
        raise FileNotFoundError(f"Manca {OBS_SCRIPT}")

    running, detail = other_arpal_running()
    if running:
        print("NON AVVIO: e' ancora attivo il downloader ARPAL Tanaro-Arroscia:")
        print(detail)
        print("Rilancia questo comando quando quello avra' terminato.")
        return False

    cmd = [sys.executable, str(OBS_SCRIPT), "--provider", "liguria"]
    if a.headed:
        cmd.append("--headed")
    print("Eseguo:", " ".join(cmd))
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    log(group="liguria_regional_observations", command=" ".join(cmd),
        returncode=rc, status="ok" if rc == 0 else "error")
    return rc == 0


def run_vda(a):
    print("\n" + "=" * 100)
    print("OSSERVAZIONI VALLE D'AOSTA - DATAVIEW ASSISTITO")
    print("=" * 100)

    if not OBS_SCRIPT.exists():
        raise FileNotFoundError(f"Manca {OBS_SCRIPT}")
    if not a.headed:
        print("Serve --headed per il form ufficiale VdA.")
        return False

    cmd = [
        sys.executable, str(OBS_SCRIPT),
        "--provider", "vda", "--vda-assisted", "--headed"
    ]
    print("Eseguo:", " ".join(cmd))
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    log(group="vda_assisted_observations", command=" ".join(cmd),
        returncode=rc, status="ok" if rc == 0 else "error")
    return rc == 0


def write_qc():
    n, bbox = basin_bounds(0.10)
    expected = tiles_for_bbox(bbox)
    valid = [p for p in TILES.glob("*.tif") if valid_tif(p)]

    refs = [
        PIE / "piemonte_soglie_idrometriche_snapshot.pdf",
        PIE / "piemonte_soglie_pluviometriche_snapshot.pdf",
        PIE / "piemonte_previsione_piene_thresholds.html",
        VDA / "vda_dora_baltea_livelli_e_soglie_snapshot.pdf",
        VDA / "vda_procedure_sistema_allertamento_dgr_1565_2022.pdf",
        LIG / "liguria_sistema_allertamento_page.html",
        LIG / "liguria_libro_blu_2020.pdf",
        LIG / "liguria_elenco_soglie_meteoidrologiche_2020.pdf",
    ]

    lines = [
        "=" * 100,
        "REMAINING REGIONAL INPUTS - QC v1.1",
        "=" * 100,
        f"Recettori: {n}",
        f"BBOX (+0.10 deg): {bbox}",
        f"DEM attesi sulla BBOX: {len(expected)}",
        f"DEM TIFF validi presenti: {len(valid)}",
        "",
        "DOCUMENTI/SNAPSHOT:",
    ]
    for p in refs:
        lines.append(f"{'OK' if p.exists() else 'MISSING':8s} {p}")

    lines += [
        "",
        "ANCORA DA FARE DOPO QUESTO BLOCCO:",
        "- ARPAL/OMIRL regionale per Liguria (--run-liguria-observations);",
        "- storico Valle d'Aosta tramite Dataview (--run-vda-assisted --headed);",
        "- normalizzazione soglie per stazione/bacino;",
        "- catalogo eventi + Y_rain + Y_flood;",
        "- ARPA Piemonte orario/suborario solo sugli eventi selezionati.",
        "=" * 100,
    ]
    QC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit():
    n, bbox = basin_bounds(0.10)
    expected = len(tiles_for_bbox(bbox))
    valid = sum(valid_tif(p) for p in TILES.glob("*.tif"))
    print("=" * 100)
    print("AUDIT REMAINING REGIONAL INPUTS")
    print("=" * 100)
    print("Recettori                  :", n)
    print("BBOX                       :", bbox)
    print("DEM validi / attesi        :", valid, "/", expected)
    print("PDF soglie/documenti       :", len(list(THRESH.rglob("*.pdf"))))
    print("HTML snapshot              :", len(list(THRESH.rglob("*.html"))))
    print("CSV soglie                 :", len(list(THRESH.rglob("*.csv"))))
    print("Manifest                   :", MANIFEST if MANIFEST.exists() else "MANCANTE")
    print("QC                         :", QC if QC.exists() else "MANCANTE")
    print("=" * 100)


def main():
    a = args_parse()
    setup()

    if a.audit:
        audit()
        return
    if a.terrain_only and a.thresholds_only:
        raise ValueError("Usa solo uno tra --terrain-only e --thresholds-only")

    print("=" * 100)
    print("DOWNLOAD REMAINING REGIONAL INPUTS v1.1")
    print("Root  :", ROOT)
    print("Output:", OUT)
    print("=" * 100)

    if a.probe:
        print("PROBE: 1 tile DEM + documenti/snapshot soglie.")

    if not a.thresholds_only:
        download_dem(a)
    if not a.terrain_only:
        download_thresholds(a)

    if a.run_liguria_observations:
        run_liguria(a)
    if a.run_vda_assisted:
        run_vda(a)

    write_qc()

    print("\n" + "=" * 100)
    print("FINE")
    print("QC      :", QC)
    print("Manifest:", MANIFEST)
    print("=" * 100)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrotto. I file gia' completati restano validi.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nFATAL: {exc}", file=sys.stderr)
        sys.exit(1)
