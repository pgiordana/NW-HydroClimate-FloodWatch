#!/usr/bin/env python3
"""
AUDIT FINALE ARPAL LIGURIA v1.1
===============================

Correzione rispetto alla v1.0:
- i "duplicati timestamp" della v1.0 erano un artefatto del parser:
  ogni riga ARPAL contiene spesso inizio e fine intervallo, e l'audit
  contava entrambe le date. Qui si usa SOLO il primo timestamp di ogni riga
  dati per il controllo dei duplicati.
- stampa esplicitamente gli eventuali BAD_* con target, anno e parametro.

NON modifica né scarica nulla.
"""

from __future__ import annotations
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "observations_nw" / "liguria_groundtruth_v1_3"
HOURLY = BASE / "hourly"
OUT = BASE / "qc_final_v1_1"
OUT.mkdir(parents=True, exist_ok=True)

YEARS = list(range(1987, 2026))

DATE_PATTERNS = [
    re.compile(r"\b(?P<d>\d{1,2})/(?P<m>\d{1,2})/(?P<y>\d{4})(?:[ T]+(?P<h>\d{1,2}):(?P<mi>\d{2}))?\b"),
    re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})(?:[ T]+(?P<h>\d{1,2}):(?P<mi>\d{2}))?\b"),
]

NO_DATA_PATTERNS = [
    r"nessun dato presente", r"nessun dato", r"dati non disponibili",
    r"non sono presenti dati", r"nessun valore", r"non risultano dati",
    r"assenza di dati", r"non esistono dati", r"nessun record",
]

BAD_STATUSES = {
    "MISSING","BAD_HTML","BAD_WRONG_PARAMETER","BAD_EMPTY",
    "BAD_NO_DATES","BAD_WRONG_YEAR","BAD_WRONG_SEASON",
}

def decode(raw):
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig", errors="replace"), "utf-8-sig"
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        try:
            return raw.decode("cp1252"), "cp1252"
        except UnicodeDecodeError:
            return raw.decode("latin-1", errors="replace"), "latin-1"

def first_timestamp_per_line(text):
    out=[]
    for line in text.splitlines():
        found=None
        for pat in DATE_PATTERNS:
            m=pat.search(line)
            if m:
                try:
                    found=(
                        int(m.group("y")),
                        int(m.group("m")),
                        int(m.group("d")),
                        int(m.group("h") or 0),
                        int(m.group("mi") or 0),
                    )
                except Exception:
                    found=None
                break
        if found:
            out.append(found)
    return out

def extract_parameter(text):
    m=re.search(r'["\']?Parametro["\']?\s*[,;]\s*(.+)', text, flags=re.I)
    return m.group(1).strip().strip('"').strip("'") if m else ""

def parameter_ok(slug,param):
    p=re.sub(r"\s+"," ",param).upper()
    if slug.endswith("_precip"):
        return "PRECIPITAZIONE CUMULATA" in p
    if slug.endswith("_level"):
        return "LIVELLO" in p or "IDROMETR" in p
    return False

def has_no_data(text):
    low=text.lower()
    return any(re.search(p,low) for p in NO_DATA_PATTERNS)

def inspect(path,slug,year):
    raw=path.read_bytes()
    text,enc=decode(raw)
    lower=text[:20000].lower()
    lines=[x for x in text.splitlines() if x.strip()]
    param=extract_parameter(text)
    pok=parameter_ok(slug,param)
    ts=first_timestamp_per_line(text)
    unique=set(ts)
    dup=len(ts)-len(unique)

    wrong_year=0
    wrong_season=0
    for y,mo,d,h,mi in ts:
        legit=(y==year+1 and mo==1 and d==1)
        if y!=year and not legit:
            wrong_year+=1
        if mo not in {9,10,11,12} and not legit:
            wrong_season+=1

    if "<html" in lower or "<!doctype html" in lower:
        status="BAD_HTML"
    elif not pok:
        status="BAD_WRONG_PARAMETER"
    elif has_no_data(text):
        status="NO_DATA_CONFIRMED"
    elif len(lines)<=2:
        status="BAD_EMPTY"
    elif not ts:
        status="BAD_NO_DATES"
    elif wrong_year:
        status="BAD_WRONG_YEAR"
    elif wrong_season:
        status="BAD_WRONG_SEASON"
    else:
        status="DATA_OK"

    return {
        "target":slug,"year":year,"status":status,
        "parameter_ok":pok,"parameter":param,"encoding":enc,
        "bytes":len(raw),"observation_rows":len(ts),
        "unique_observation_timestamps":len(unique),
        "duplicate_observation_timestamps":dup,
        "wrong_year_hits":wrong_year,
        "wrong_season_hits":wrong_season,
        "path":str(path),
    }

def write_csv(path,rows):
    rows=list(rows)
    if not rows:
        path.write_text("",encoding="utf-8")
        return
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def main():
    targets=sorted(p.name for p in HOURLY.iterdir() if p.is_dir())
    rows=[]

    for slug in targets:
        for year in YEARS:
            p=HOURLY/slug/f"{slug}_{year}_09-12.csv"
            if p.exists():
                rows.append(inspect(p,slug,year))
            else:
                rows.append({
                    "target":slug,"year":year,"status":"MISSING",
                    "parameter_ok":"","parameter":"","encoding":"",
                    "bytes":0,"observation_rows":0,
                    "unique_observation_timestamps":0,
                    "duplicate_observation_timestamps":0,
                    "wrong_year_hits":0,"wrong_season_hits":0,
                    "path":str(p),
                })

    rows.sort(key=lambda r:(r["target"],r["year"]))
    write_csv(OUT/"file_qc_final_v1_1.csv",rows)

    bad=[r for r in rows if r["status"] in BAD_STATUSES]
    write_csv(OUT/"bad_cases_v1_1.csv",bad)

    c=Counter(r["status"] for r in rows)
    param_bad=sum(1 for r in rows if r["status"]!="MISSING" and r["parameter_ok"] is False)
    dup_files=sum(1 for r in rows if int(r["duplicate_observation_timestamps"] or 0)>0)
    dup_total=sum(int(r["duplicate_observation_timestamps"] or 0) for r in rows)

    expected=len(targets)*len(YEARS)
    acceptable=c["DATA_OK"]+c["NO_DATA_CONFIRMED"]
    passed=(acceptable==expected and not bad and param_bad==0)

    lines=[
        "="*108,
        "AUDIT FINALE ARPAL LIGURIA v1.1",
        "="*108,
        f"Target rilevati                    : {len(targets)}",
        f"Combinazioni attese                : {expected}",
        f"Combinazioni accettabili           : {acceptable}",
        "",
        "Classificazione:",
    ]
    for k,v in sorted(c.items()):
        lines.append(f"  {k:34s} {v:5d}")

    lines += [
        "",
        f"Parametri scientificamente errati  : {param_bad}",
        f"File con veri timestamp duplicati   : {dup_files}",
        f"Veri timestamp duplicati complessivi: {dup_total}",
        "",
    ]

    if bad:
        lines.append("CASI BAD DA RISOLVERE:")
        for r in bad:
            lines.append(
                f"  {r['status']}: {r['target']} | {r['year']} | "
                f"parametro={r['parameter']!r}"
            )
        lines.append("")

    lines += [
        f"ESITO AUTOMATICO: {'PASS' if passed else 'FAIL'}",
        "",
        "Output:",
        f"  {OUT/'file_qc_final_v1_1.csv'}",
        f"  {OUT/'bad_cases_v1_1.csv'}",
        "="*108,
    ]

    report="\n".join(lines)+"\n"
    (OUT/"audit_finale_arpal_liguria_v1_1.txt").write_text(report,encoding="utf-8")
    print(report)

if __name__=="__main__":
    main()
