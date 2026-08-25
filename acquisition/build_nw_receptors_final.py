#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import csv
import json
import sys

try:
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union
except Exception:
    print("ERRORE: manca shapely.")
    print("Installa con: pip install shapely")
    sys.exit(2)

ROOT = Path(__file__).resolve().parent
BASINS = ROOT / "basins"
OUT = ROOT / "basins_final"
OUT.mkdir(exist_ok=True)

PIE2 = BASINS / "piemonte_bacini_secondo_livello.geojson"
LIG = BASINS / "liguria_01_M2542_L9861.geojson"

OUT_GEOJSON = OUT / "nw_receptors_final.geojson"
OUT_CSV = OUT / "nw_receptors_final.csv"
OUT_REPORT = OUT / "nw_receptors_final_report.txt"

# Per il Piemonte usiamo il secondo livello come geometria primaria.
# Per i bacini transregionali uniamo, quando disponibile, il ramo ligure nominato.
# Nessuna imputazione artificiale dei tributari valdostani: Dora Baltea resta il recettore valdostano iniziale.
SPECS = [
    # id, label, region, priority, piem_names, lig_names, notes
    ("NW_DORA_RIPARIA", "Dora Riparia / Valle di Susa", "Piemonte", 1,
     ["DORA RIPARIA"], [],
     "Recettore principale della Valle di Susa."),
    ("NW_STURA_LANZO", "Stura di Lanzo / Valli di Lanzo", "Piemonte", 1,
     ["STURA DI LANZO"], [],
     "Recettore principale delle Valli di Lanzo; dettaglio Viù/Valgrande disponibile nel livello 1."),
    ("NW_STURA_DEMONTE", "Stura di Demonte", "Piemonte", 1,
     ["STURA DI DEMONTE"], [],
     "Recettore alpino occidentale prioritario."),
    ("NW_TANARO_ALTO", "Alto Tanaro", "Piemonte/Liguria", 1,
     ["ALTO TANARO"], ["T. TANARO"],
     "Composito transregionale; escluso esplicitamente il falso match VALLE DI CANEVA."),
    ("NW_TANARO_MEDIO_BASSO", "Tanaro medio-basso", "Piemonte", 1,
     ["TANARO"], [],
     "Separato dall'ALTO TANARO per modellare risposte diverse lungo l'asta."),
    ("NW_BORMIDA", "Bormida", "Piemonte/Liguria", 1,
     ["BORMIDA"], ["BORMIDA DI SPIGNO", "F. BORMIDA DI MILLESIMO"],
     "Composito transregionale."),
    ("NW_ORBA", "Orba", "Piemonte/Liguria", 1,
     ["ORBA"], ["T. ORBA"],
     "Composito transregionale."),
    ("NW_SCRIVIA", "Scrivia", "Piemonte/Liguria", 1,
     ["SCRIVIA"], ["T. SCRIVIA"],
     "Composito transregionale."),
    ("NW_DORA_BALTEA", "Dora Baltea", "Valle d'Aosta/Piemonte", 1,
     ["DORA BALTEA"], [],
     "Recettore iniziale per Valle d'Aosta; tributari valdostani da aggiungere con fonte dedicata."),
    ("NW_ORCO", "Orco", "Piemonte", 2,
     ["ORCO"], [],
     "Recettore secondario."),
    ("NW_PELLICE", "Pellice", "Piemonte", 2,
     ["PELLICE"], [],
     "Tenuto separato dal Chisone."),
    ("NW_CHISONE", "Chisone", "Piemonte", 2,
     ["CHISONE"], [],
     "Tenuto separato dal Pellice."),
    ("NW_MAIRA", "Maira", "Piemonte", 2,
     ["MAIRA"], [],
     "Recettore secondario."),
    ("NW_VARAITA", "Varaita", "Piemonte", 2,
     ["VARAITA"], [],
     "Recettore secondario."),
    ("NW_SESIA", "Sesia", "Piemonte", 2,
     ["SESIA", "ALTO SESIA"], [],
     "Unione di SESIA e ALTO SESIA per il recettore complessivo."),
    ("NW_TOCE", "Toce", "Piemonte", 2,
     ["TOCE"], [],
     "Recettore secondario."),

    # Liguria marittima
    ("LIG_BISAGNO", "Bisagno", "Liguria", 1,
     [], ["T. BISAGNO"],
     "Unione di tutti i poligoni esattamente nominati T. BISAGNO."),
    ("LIG_POLCEVERA", "Polcevera", "Liguria", 1,
     [], ["T. POLCEVERA"],
     "Bacino marittimo genovese prioritario."),
    ("LIG_ENTELLA", "Entella", "Liguria", 1,
     [], ["FIUME ENTELLA"],
     "Il layer nominato restituisce Entella; Lavagna non viene aggiunto artificialmente."),
    ("LIG_MAGRA", "Magra", "Liguria", 1,
     [], ["F. MAGRA"],
     "Il layer nominato restituisce Magra; il Vara verrà raffinato in una fase successiva se necessario."),
    ("LIG_CENTA", "Centa", "Liguria", 2,
     [], ["F. CENTA"],
     "Recettore marittimo di Ponente."),
]

def load(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Manca {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def exact_features(fc, field, names):
    wanted = {str(x).strip().upper() for x in names}
    out = []
    for feat in fc.get("features", []):
        value = (feat.get("properties") or {}).get(field)
        if value is None:
            continue
        if str(value).strip().upper() in wanted:
            out.append(feat)
    return out

def union_features(features):
    geoms = [shape(f["geometry"]) for f in features if f.get("geometry")]
    if not geoms:
        return None
    g = unary_union(geoms)
    if not g.is_valid:
        g = g.buffer(0)
    return g

def main():
    pie = load(PIE2)
    lig = load(LIG)

    out_features = []
    rows = []
    report = [
        "NW RECEPTORS — FINAL SELECTION REPORT",
        "",
        "Primary rule:",
        "- Piemonte: exact-name selection from official second-level basin layer.",
        "- Transregional basins: union with exact-name Liguria basin polygons where explicitly available.",
        "- Liguria maritime basins: exact-name selection from L9861.",
        "- Valle d'Aosta: Dora Baltea only at this stage; no invented local tributary geometry.",
        "",
    ]

    failures = []

    for rid, label, region, priority, pie_names, lig_names, notes in SPECS:
        pie_feats = exact_features(pie, "nome", pie_names) if pie_names else []
        lig_feats = exact_features(lig, "nome_bacino", lig_names) if lig_names else []
        all_feats = pie_feats + lig_feats

        geom = union_features(all_feats)
        if geom is None or geom.is_empty:
            failures.append(rid)
            report.extend([
                "=" * 90,
                f"{rid} | {label}",
                "*** NO GEOMETRY FOUND ***",
                f"Piemonte exact names: {pie_names}",
                f"Liguria exact names: {lig_names}",
                "",
            ])
            continue

        minx, miny, maxx, maxy = geom.bounds
        centroid = geom.centroid

        props = {
            "receptor_id": rid,
            "label": label,
            "region": region,
            "priority": priority,
            "piemonte_names": " | ".join(pie_names),
            "liguria_names": " | ".join(lig_names),
            "piemonte_parts": len(pie_feats),
            "liguria_parts": len(lig_feats),
            "source_method": "exact-name + union",
            "notes": notes,
        }

        out_features.append({
            "type": "Feature",
            "properties": props,
            "geometry": mapping(geom),
        })

        rows.append({
            **props,
            "centroid_lon": centroid.x,
            "centroid_lat": centroid.y,
            "bbox_w": minx,
            "bbox_s": miny,
            "bbox_e": maxx,
            "bbox_n": maxy,
            "geometry_type": geom.geom_type,
        })

        report.extend([
            "=" * 90,
            f"{rid} | {label} | priority {priority}",
            f"Region: {region}",
            f"Piemonte exact names: {pie_names or '-'}",
            f"Liguria exact names: {lig_names or '-'}",
            f"Parts: Piemonte={len(pie_feats)}, Liguria={len(lig_feats)}",
            f"Geometry type: {geom.geom_type}",
            f"Centroid: {centroid.y:.5f} N, {centroid.x:.5f} E",
            f"BBox: {minx:.5f}, {miny:.5f}, {maxx:.5f}, {maxy:.5f}",
            f"Notes: {notes}",
            "",
        ])

    OUT_GEOJSON.write_text(
        json.dumps({"type": "FeatureCollection", "features": out_features}, ensure_ascii=False),
        encoding="utf-8"
    )

    fieldnames = [
        "receptor_id","label","region","priority",
        "piemonte_names","liguria_names","piemonte_parts","liguria_parts",
        "source_method","notes","centroid_lon","centroid_lat",
        "bbox_w","bbox_s","bbox_e","bbox_n","geometry_type"
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    report.extend([
        "=" * 90,
        f"Final receptors created: {len(out_features)} / {len(SPECS)}",
        f"Failures: {', '.join(failures) if failures else 'none'}",
        "",
        "IMPORTANT MODEL NOTE",
        "These polygons are atmospheric/hydrometeorological receptor geometries.",
        "They do not by themselves determine flood probability.",
        "Later stages must add rainfall, atmospheric moisture transport, antecedent wetness and hydrological response.",
        "",
        f"GeoJSON: {OUT_GEOJSON}",
        f"CSV: {OUT_CSV}",
    ])

    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")

    print(f"Creati {len(out_features)} recettori finali su {len(SPECS)}.")
    if failures:
        print("ATTENZIONE - recettori mancanti:", ", ".join(failures))
    print(OUT_GEOJSON)
    print(OUT_CSV)
    print(OUT_REPORT)

if __name__ == "__main__":
    main()
