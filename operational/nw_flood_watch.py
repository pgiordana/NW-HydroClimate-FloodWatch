#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NW FloodWatch - definitive local runner v1.0-rc1

Daily use:
    python nw_flood_watch.py

Useful options:
    --force-full      re-download/rebuild even if today's run exists
    --report-only     generate bulletin from latest completed snapshot only
    --open            open the produced PDF
    --allow-degraded-smoke
                      allow non-scientific smoke inference if an in-season
                      operational gate blocks the beta
    --pdf-demo        create a layout demo PDF without running the model

This program orchestrates the frozen operational pipeline and creates a PDF
bulletin. It is not an official warning system.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import shutil
import subprocess
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


VERSION = "1.0-rc1"
HORIZONS = [24, 48, 72]
EXPECTED_RECEPTORS = 20
EXPECTED_PREDICTORS = 97
MAX_CSI = "MAX_CSI"
RECALL80 = "RECALL80_MAX_PRECISION"

COMPONENTS = [
    ("1/6 DATI ECMWF/CMEMS", "build_nw_operational_raw_cache_current_v1_1.py"),
    ("2/6 CONTROLLO SURFACE", "repair_nw_operational_raw_cache_surface_v1_1.py"),
    ("3/6 FEATURE BACINI", "build_nw_operational_receptor_features_current_v1_1.py"),
    ("4/6 MEDSEA x IVT", "build_nw_operational_medsea_corridor_current_v1_1.py"),
    ("5/6 CACHE ANTECEDENTI", "update_nw_operational_antecedent_cache_current_v1_0.py"),
    ("6/6 BETA GATE", "audit_nw_operational_beta_gate_current_v1_0.py"),
]

SEVERITY = {
    "GRAY": 0,
    "GREEN": 1,
    "YELLOW": 2,
    "RED": 3,
}


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_config(root: Path) -> dict:
    p = root / "config" / "bulletin_config.json"
    if not p.exists():
        raise RuntimeError(f"Configurazione bollettino mancante: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p)).reshape(-1, 1)


def apply_platt(calibrator, raw_prob):
    return calibrator.predict_proba(logit(raw_prob))[:, 1]


def human_label(receptor_id: str) -> str:
    x = str(receptor_id)
    for prefix in ("NW_", "LIG_"):
        if x.startswith(prefix):
            x = x[len(prefix):]
    return x.replace("_", " ").title()


def latest_snapshot_with_gate(root: Path) -> Path | None:
    base = root / "nw_operational_feature_snapshot"
    if not base.exists():
        return None
    candidates = sorted(
        [
            p for p in base.iterdir()
            if p.is_dir()
            and (p / "operational_beta_gate_audit_v1_0.json").exists()
            and (p / "operational_full_97_predictors_v1_3.parquet").exists()
        ],
        key=lambda p: p.name,
    )
    return candidates[-1] if candidates else None


def gate_issue_date(snapshot: Path) -> str | None:
    try:
        obj = json.loads(
            (snapshot / "operational_beta_gate_audit_v1_0.json").read_text(
                encoding="utf-8"
            )
        )
        return obj.get("issue_date")
    except Exception:
        return None


def run_component(root: Path, label: str, filename: str, log_handle):
    script = root / filename
    if not script.exists():
        raise RuntimeError(f"Componente mancante: {script}")

    started = time.time()
    print(f"[{label}] AVVIO", flush=True)
    log_handle.write(
        f"\n{'='*120}\n{label} | {filename}\nSTART {now_text()}\n{'='*120}\n"
    )
    log_handle.flush()

    proc = subprocess.Popen(
        [sys.executable, str(script)],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    last_compact = ""

    for line in proc.stdout:
        log_handle.write(line)
        log_handle.flush()

        # Keep terminal compact while preserving full detail in the log.
        clean = line.replace("\r", "").strip()
        if not clean:
            continue

        interesting = (
            "OVERALL STATUS" in clean
            or "100.00%" in clean
            or clean.startswith("Top-up CMEMS")
            or clean.startswith("WARNING")
            or clean.startswith("ERROR")
            or "Traceback" in clean
        )

        if interesting and clean != last_compact:
            print(f"  {clean[:220]}", flush=True)
            last_compact = clean

    code = proc.wait()
    elapsed = time.time() - started
    log_handle.write(f"END {now_text()} | returncode={code} | elapsed={elapsed:.1f}s\n")
    log_handle.flush()

    if code != 0:
        print(f"[{label}] FAIL dopo {elapsed:.1f}s", flush=True)
        raise RuntimeError(f"{label} fallito con return code {code}")

    print(f"[{label}] PASS | elapsed {elapsed:.1f}s", flush=True)


def verify_frozen_checksums(frozen_root: Path):
    registry_p = frozen_root / "checksums_sha256_v1_2.csv"
    if not registry_p.exists():
        raise RuntimeError(f"Manca checksum registry: {registry_p}")

    registry = pd.read_csv(registry_p, low_memory=False)
    failures = []

    for _, r in registry.iterrows():
        p = frozen_root / Path(str(r["file"]))
        if not p.exists():
            failures.append(f"MISSING:{r['file']}")
            continue
        if sha256(p) != str(r["sha256"]):
            failures.append(f"HASH_MISMATCH:{r['file']}")

    if failures:
        raise RuntimeError(
            "Integrita artefatti congelati non valida:\n" + "\n".join(failures)
        )
    return registry_p


def threshold_map(threshold_registry: pd.DataFrame) -> dict:
    out = {}
    for h in HORIZONS:
        subset = threshold_registry[
            pd.to_numeric(threshold_registry["horizon_hours"], errors="coerce").eq(h)
        ]
        for policy in (MAX_CSI, RECALL80):
            x = subset[subset["threshold_policy"].astype(str).eq(policy)]
            if len(x) != 1:
                raise RuntimeError(f"Soglia {h}h/{policy}: rows={len(x)}")
            out[(h, policy)] = float(x.iloc[0]["threshold"])
    return out


def model_schema_ok(model, predictor_order):
    if hasattr(model, "feature_names_in_"):
        return list(map(str, model.feature_names_in_)) == predictor_order
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_) == len(predictor_order)
    return True


def horizon_status(prob: float, h: int, thresholds: dict, scientific_mode: bool) -> str:
    if not scientific_mode or not np.isfinite(prob):
        return "GRAY"
    if prob >= thresholds[(h, MAX_CSI)]:
        return "RED"
    if prob >= thresholds[(h, RECALL80)]:
        return "YELLOW"
    return "GREEN"


def overall_status(row: pd.Series, thresholds: dict, scientific_mode: bool) -> str:
    if not scientific_mode:
        return "GRAY"
    states = [
        horizon_status(float(row[f"calibrated_probability_{h}h"]), h, thresholds, True)
        for h in HORIZONS
    ]
    return max(states, key=lambda s: SEVERITY[s])


def status_label(status: str) -> str:
    return {
        "GREEN": "VERDE - nessun segnale sperimentale",
        "YELLOW": "GIALLO - da approfondire",
        "RED": "ROSSO - segnale sperimentale elevato",
        "GRAY": "GRIGIO - non interpretabile",
    }[status]


def load_verified_assets(root: Path) -> pd.DataFrame:
    p = root / "data" / "operational_assets" / "reservoir_registry.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    if len(df) == 0:
        return df

    for c in ("verified", "action_enabled"):
        if c not in df.columns:
            df[c] = "NO"

    yes = {"YES", "Y", "TRUE", "1", "SI", "SÌ"}
    mask = (
        df["verified"].astype(str).str.upper().isin(yes)
        & df["action_enabled"].astype(str).str.upper().isin(yes)
    )
    return df[mask].copy()


def action_note_for_basin(
    receptor_id: str,
    status: str,
    assets: pd.DataFrame,
    scientific_mode: bool,
) -> tuple[str, str]:
    if not scientific_mode:
        return (
            "Nessuna interpretazione predittiva scientifica fuori dal periodo di validita.",
            "",
        )

    if status == "GREEN":
        return (
            "Monitoraggio ordinario; confrontare sempre con bollettini e dati ufficiali.",
            "",
        )

    local = pd.DataFrame()
    if len(assets) and "receptor_id" in assets.columns:
        local = assets[assets["receptor_id"].astype(str).eq(str(receptor_id))]

    base = (
        "Approfondire con fonti ufficiali: precipitazione prevista/nowcasting, "
        "stazioni idrometriche, stato di suolo/neve e criticita locali."
    )

    if len(local) == 0:
        return base, ""

    names = ", ".join(local["asset_name"].astype(str).tolist())
    short = (
        f"Asset associati: {names}. Verificare con i gestori autorizzati livello invaso, "
        "volume residuo e capacita di laminazione."
    )
    detail = (
        f"{human_label(receptor_id)} - asset associati: {names}. Verificare con il/i gestore/i "
        "autorizzato/i il livello dell'invaso, il volume residuo disponibile, la capacita degli "
        "organi di scarico, i vincoli regolatori e la sicurezza a valle. Eventuali manovre o la "
        "creazione preventiva di capacita di laminazione devono essere valutate esclusivamente "
        "dai soggetti competenti secondo le regole approvate e non sono mai ordinate da NW FloodWatch."
    )
    return short, detail


def build_pdf_bulletin(
    root: Path,
    output_pdf: Path,
    result: pd.DataFrame,
    gate: dict,
    thresholds: dict,
    bulletin_mode: str,
    scientific_mode: bool,
    summary_lines: list[str],
    config: dict,
    demo: bool = False,
):
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            PageBreak,
            KeepTogether,
        )
    except Exception as exc:
        raise RuntimeError(
            "ReportLab non installato. Eseguire: python -m pip install reportlab"
        ) from exc

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    assets = load_verified_assets(root)

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleNW",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=23,
        alignment=TA_CENTER,
        spaceAfter=5 * mm,
    )
    subtitle = ParagraphStyle(
        "SubtitleNW",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        alignment=TA_CENTER,
        spaceAfter=4 * mm,
    )
    h2 = ParagraphStyle(
        "H2NW",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        spaceBefore=3 * mm,
        spaceAfter=2 * mm,
    )
    body = ParagraphStyle(
        "BodyNW",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        alignment=TA_LEFT,
        spaceAfter=2 * mm,
    )
    small = ParagraphStyle(
        "SmallNW",
        parent=body,
        fontSize=7.4,
        leading=9.2,
    )
    warning_style = ParagraphStyle(
        "WarningNW",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=8.2,
        leading=10.5,
    )

    page_w, page_h = landscape(A4)
    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=14 * mm,
        title=f"{config['product_name']} - {result['issue_date'].iloc[0]}",
        author=config["author_name"],
    )

    def footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#444444"))
        footer_text = (
            f"{config['author_name']} - {config['author_email']} | "
            f"NW FloodWatch {VERSION} | pagina {doc_obj.page}"
        )
        canvas.drawCentredString(page_w / 2.0, 6 * mm, footer_text)
        canvas.restoreState()

    story = []
    story.append(Paragraph(config["product_name"], title))
    story.append(Paragraph(config["product_subtitle"], subtitle))

    if demo:
        demo_box = Table(
            [[Paragraph("DEMO DI IMPAGINAZIONE - DATI FITTIZI, NON USARE", warning_style)]],
            colWidths=[page_w - 30 * mm],
        )
        demo_box.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF0F0")),
                ("BOX", (0, 0), (-1, -1), 1.5, colors.red),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ])
        )
        story.extend([demo_box, Spacer(1, 3 * mm)])

    warning_content = [
        [Paragraph("AVVERTENZA IMPORTANTE - DOCUMENTO NON UFFICIALE", warning_style)],
        [Paragraph(config["disclaimer"], body)],
        [Paragraph(config["season_warning"], warning_style)],
        [Paragraph(config["target_warning"], body)],
    ]
    warning_table = Table(warning_content, colWidths=[page_w - 30 * mm])
    warning_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F4CCCC")),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#FFF7F7")),
            ("BOX", (0, 0), (-1, -1), 1.2, colors.HexColor("#990000")),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D9A3A3")),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    story.extend([warning_table, Spacer(1, 3 * mm)])

    issue = str(result["issue_date"].iloc[0])
    run_id = str(result["run_id"].iloc[0])
    gate_status = str(gate.get("overall_status", ""))
    meta = [
        ["Data del bollettino", issue, "Run/ciclo", run_id],
        ["Modalita", bulletin_mode, "Gate", gate_status],
        ["Periodo CORE", "1 settembre - 31 dicembre", "Interpretazione scientifica", "SI" if scientific_mode else "NO"],
    ]
    meta_table = Table(meta, colWidths=[32*mm, 68*mm, 45*mm, 110*mm])
    meta_table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F3F3F3")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#888888")),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#BBBBBB")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.extend([meta_table, Spacer(1, 3 * mm)])
    story.append(Paragraph(config["data_sources"], small))

    story.append(Paragraph("Sintesi del run", h2))
    for x in summary_lines:
        story.append(Paragraph(f"- {x}", body))

    # Legend with actual frozen thresholds.
    if scientific_mode:
        legend_text = (
            "Semaforo sperimentale (NON allerta ufficiale): "
            "VERDE = probabilita sotto la soglia sensibile congelata; "
            "GIALLO = >= soglia sensibile RECALL80 e < MAX_CSI; "
            "ROSSO = >= MAX_CSI. "
            f"Soglie 24h: {100*thresholds[(24,RECALL80)]:.1f}% / {100*thresholds[(24,MAX_CSI)]:.1f}%; "
            f"48h: {100*thresholds[(48,RECALL80)]:.1f}% / {100*thresholds[(48,MAX_CSI)]:.1f}%; "
            f"72h: {100*thresholds[(72,RECALL80)]:.1f}% / {100*thresholds[(72,MAX_CSI)]:.1f}%."
        )
    else:
        legend_text = (
            "Semaforo disattivato: il run non e scientificamente interpretabile. "
            "Le celle sono mostrate in grigio anche se il modello produce valori numerici."
        )
    story.append(Paragraph(legend_text, small))

    story.append(Paragraph("Probabilita per sottozona", h2))

    table_data = [[
        "Sottozona",
        "P(Q95) 24 h",
        "P(Q95) 48 h",
        "P(Q95) 72 h",
        "Semaforo sperimentale",
        "Nota / approfondimento",
    ]]
    cell_statuses = []
    detailed_actions = []

    for _, r in result.sort_values("receptor_id").iterrows():
        statuses = {}
        for h in HORIZONS:
            p = float(r.get(f"calibrated_probability_{h}h", np.nan))
            statuses[h] = horizon_status(p, h, thresholds, scientific_mode)

        overall = overall_status(r, thresholds, scientific_mode)
        short_note, detailed = action_note_for_basin(
            str(r["receptor_id"]), overall, assets, scientific_mode
        )
        if detailed and overall in {"YELLOW", "RED"}:
            detailed_actions.append((overall, detailed))

        row = [
            human_label(r["receptor_id"]),
            f"{100*float(r['calibrated_probability_24h']):.2f}%" if np.isfinite(r.get("calibrated_probability_24h", np.nan)) else "n.d.",
            f"{100*float(r['calibrated_probability_48h']):.2f}%" if np.isfinite(r.get("calibrated_probability_48h", np.nan)) else "n.d.",
            f"{100*float(r['calibrated_probability_72h']):.2f}%" if np.isfinite(r.get("calibrated_probability_72h", np.nan)) else "n.d.",
            status_label(overall),
            Paragraph(short_note, small),
        ]
        table_data.append(row)
        cell_statuses.append((statuses, overall))

    col_widths = [38*mm, 21*mm, 21*mm, 21*mm, 48*mm, 112*mm]
    main_table = Table(
        table_data,
        colWidths=col_widths,
        repeatRows=1,
        hAlign="LEFT",
    )

    base_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, 0), 7.4),
        ("FONTSIZE", (0, 1), (-1, -1), 7.0),
        ("ALIGN", (1, 1), (4, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#AAAAAA")),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]

    bg = {
        "GREEN": colors.HexColor("#D9EAD3"),
        "YELLOW": colors.HexColor("#FFF2CC"),
        "RED": colors.HexColor("#F4CCCC"),
        "GRAY": colors.HexColor("#E7E6E6"),
    }
    fg = {
        "GREEN": colors.HexColor("#274E13"),
        "YELLOW": colors.HexColor("#7F6000"),
        "RED": colors.HexColor("#990000"),
        "GRAY": colors.HexColor("#555555"),
    }

    for i, (statuses, overall) in enumerate(cell_statuses, start=1):
        for col, h in zip((1, 2, 3), HORIZONS):
            s = statuses[h]
            base_style.append(("BACKGROUND", (col, i), (col, i), bg[s]))
            base_style.append(("TEXTCOLOR", (col, i), (col, i), fg[s]))
            base_style.append(("FONTNAME", (col, i), (col, i), "Helvetica-Bold"))
        base_style.append(("BACKGROUND", (4, i), (4, i), bg[overall]))
        base_style.append(("TEXTCOLOR", (4, i), (4, i), fg[overall]))
        base_style.append(("FONTNAME", (4, i), (4, i), "Helvetica-Bold"))

    main_table.setStyle(TableStyle(base_style))
    story.append(main_table)

    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Azioni di approfondimento (non prescrittive)", h2))

    if not scientific_mode:
        story.append(
            Paragraph(
                "Nessuna azione operativa viene proposta: il run e fuori dal dominio di validita scientifica o e stato bloccato dal gate.",
                body,
            )
        )
    elif detailed_actions:
        for sev, text in detailed_actions:
            story.append(Paragraph(f"- {text}", body))
    else:
        flagged_present = any(
            overall_status(r, thresholds, scientific_mode) in {"YELLOW", "RED"}
            for _, r in result.iterrows()
        )
        if flagged_present:
            story.append(
                Paragraph(
                    "Sono presenti segnali sperimentali gialli/rossi, ma nel registro runtime non risultano asset idraulici verificati e abilitati per una nota specifica. Approfondire quindi con le fonti ufficiali e, se pertinente, con i gestori delle opere presenti nel bacino.",
                    body,
                )
            )
        else:
            story.append(
                Paragraph(
                    "Nessun recettore richiede, in questo run, un approfondimento asset-specifico sulla base delle soglie sperimentali congelate.",
                    body,
                )
            )

    story.append(
        Paragraph(
            "Sicurezza delle dighe e degli invasi: NW FloodWatch non ordina mai svuotamenti, rilasci o altre manovre. "
            "Quando un segnale interessa un bacino con un asset verificato nel registro, il bollettino puo solo suggerire al gestore competente di verificare il margine di laminazione e valutare, secondo regole approvate, capacita degli scarichi e sicurezza a valle, se sia possibile creare ulteriore capacita. Qualunque manovra resta una decisione esclusiva dei soggetti autorizzati.",
            warning_style,
        )
    )

    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("Istruzioni per il giorno successivo", h2))
    next_steps = [
        "Assicurarsi che il computer disponga di connessione Internet.",
        "Aprire la cartella NW_FloodWatch_Definitivo.",
        "Su macOS fare doppio clic su avvia_nw_floodwatch.command; in alternativa, dal Terminale attivare .venv e lanciare: python nw_flood_watch.py --open.",
        "Il programma individua il nuovo ciclo ECMWF 00Z, aggiorna Copernicus Marine, ricostruisce le feature, aggiorna la cache antecedente, applica il gate e genera un nuovo PDF.",
        "Usare --force-full soltanto se si desidera ricostruire da zero il run della stessa data.",
        "Confrontare sempre il risultato con i bollettini e le allerte ufficiali prima di qualunque valutazione operativa.",
    ]
    for i, x in enumerate(next_steps, start=1):
        story.append(Paragraph(f"{i}. {x}", body))

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Contatti", h2))
    story.append(
        Paragraph(
            f"Autore: <b>{config['author_name']}</b> - {config['author_role']}<br/>"
            f"Email: <b>{config['author_email']}</b><br/>"
            "Segnalazioni, osservazioni metodologiche e confronti sui risultati sono benvenuti.",
            body,
        )
    )

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Tracciabilita", h2))
    story.append(
        Paragraph(
            f"Versione software: {VERSION}. Run: {run_id}. Gate: {gate_status}. "
            "Il dettaglio macchina, i checksum dei modelli, i dati strutturati e l'audit del run sono conservati nella cartella di output associata.",
            small,
        )
    )

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output_pdf


def render_outputs(
    root: Path,
    snapshot: Path,
    gate: dict,
    allow_degraded_smoke: bool,
    config: dict,
):
    input_p = snapshot / "operational_full_97_predictors_v1_3.parquet"
    dictionary_p = (
        root
        / "nw_hydroclimate_core_release_v1_0"
        / "metadata"
        / "nw_hydroclimate_master_predictor_dictionary_v1_0.csv"
    )
    if not dictionary_p.exists():
        dictionary_p = (
            root
            / "nw_hydroclimate_foldwise_master_core_canonical_v1_0"
            / "nw_hydroclimate_master_predictor_dictionary_v1_0.csv"
        )

    frozen_root = root / "nw_flood_models_frozen_development_v1_2"
    threshold_p = frozen_root / "development_threshold_registry_v1_2.csv"

    for p in (input_p, dictionary_p, threshold_p):
        if not p.exists():
            raise RuntimeError(f"Manca artefatto per inferenza: {p}")

    beta_allowed = bool(gate.get("prospective_beta_allowed", False))
    in_season = bool(gate.get("in_core_season_sep_dec", False))
    unexpected_missing = int(gate.get("unexpected_operational_missing_cells", 0))

    if beta_allowed:
        bulletin_mode = "EXPERIMENTAL_PROSPECTIVE_BETA"
        scientific_mode = True
        perform_inference = True
    elif not in_season:
        bulletin_mode = "TECHNICAL_SMOKE_TEST__OUT_OF_SEASON__NON_SCIENTIFIC"
        scientific_mode = False
        perform_inference = True
    elif allow_degraded_smoke:
        bulletin_mode = "DEGRADED_TECHNICAL_SMOKE__NON_SCIENTIFIC"
        scientific_mode = False
        perform_inference = True
    else:
        bulletin_mode = "BETA_BLOCKED__NO_INFERENCE"
        scientific_mode = False
        perform_inference = False

    operational = pd.read_parquet(input_p)
    dictionary = pd.read_csv(dictionary_p, low_memory=False)
    predictor_order = dictionary["predictor"].astype(str).tolist()

    if len(operational) != EXPECTED_RECEPTORS:
        raise RuntimeError(f"Operational rows={len(operational)}, expected=20")
    if len(predictor_order) != EXPECTED_PREDICTORS:
        raise RuntimeError(f"Predictors={len(predictor_order)}, expected=97")

    actual = [
        c for c in operational.columns
        if c not in {"receptor_id", "issue_date", "run_id"}
    ]
    if actual != predictor_order:
        raise RuntimeError("Predictor order differs from frozen dictionary.")

    output_root = root / "nw_floodwatch_output" / snapshot.name
    output_root.mkdir(parents=True, exist_ok=True)

    result = operational[["receptor_id", "issue_date", "run_id"]].copy()
    result["basin_label"] = result["receptor_id"].astype(str).map(human_label)

    dynamic_cols = [p for p in predictor_order if not p.startswith("static__")]
    X = operational[predictor_order].apply(pd.to_numeric, errors="coerce")
    result["missing_dynamic_features"] = (
        X[dynamic_cols].isna().sum(axis=1).astype(int).to_numpy()
    )

    threshold_registry = pd.read_csv(threshold_p, low_memory=False)
    thresholds = threshold_map(threshold_registry)
    model_registry_rows = []

    if perform_inference:
        checksum_registry_p = verify_frozen_checksums(frozen_root)
        for h in HORIZONS:
            model_p = frozen_root / "models" / f"horizon_{h}h_base_model.joblib"
            calibrator_p = frozen_root / "models" / f"horizon_{h}h_platt_calibrator.joblib"
            model = joblib.load(model_p)
            calibrator = joblib.load(calibrator_p)
            if not model_schema_ok(model, predictor_order):
                raise RuntimeError(f"{h}h model schema mismatch.")

            raw = model.predict_proba(X)[:, 1]
            calibrated = apply_platt(calibrator, raw)
            t_max = thresholds[(h, MAX_CSI)]
            t_r80 = thresholds[(h, RECALL80)]

            result[f"raw_probability_{h}h"] = raw
            result[f"calibrated_probability_{h}h"] = calibrated
            result[f"flag_max_csi_{h}h"] = calibrated >= t_max
            result[f"flag_recall80_{h}h"] = calibrated >= t_r80
            result[f"semaphore_{h}h"] = [
                horizon_status(float(p), h, thresholds, scientific_mode)
                for p in calibrated
            ]

            model_registry_rows.append({
                "horizon_hours": h,
                "model_file": str(model_p),
                "calibrator_file": str(calibrator_p),
                "threshold_max_csi": t_max,
                "threshold_recall80": t_r80,
                "calibrated_min": float(np.min(calibrated)),
                "calibrated_max": float(np.max(calibrated)),
                "checksum_registry": str(checksum_registry_p),
            })
    else:
        for h in HORIZONS:
            result[f"raw_probability_{h}h"] = np.nan
            result[f"calibrated_probability_{h}h"] = np.nan
            result[f"flag_max_csi_{h}h"] = False
            result[f"flag_recall80_{h}h"] = False
            result[f"semaphore_{h}h"] = "GRAY"

    result["bulletin_mode"] = bulletin_mode
    result["scientific_beta_interpretation"] = scientific_mode
    result = result.sort_values("receptor_id").reset_index(drop=True)

    result["overall_semaphore"] = [
        overall_status(r, thresholds, scientific_mode)
        for _, r in result.iterrows()
    ]

    assets = load_verified_assets(root)
    notes = []
    for _, r in result.iterrows():
        short, _ = action_note_for_basin(
            str(r["receptor_id"]),
            str(r["overall_semaphore"]),
            assets,
            scientific_mode,
        )
        notes.append(short)
    result["action_note"] = notes

    summary_lines = []
    if scientific_mode:
        red = result[result["overall_semaphore"].eq("RED")]
        yellow = result[result["overall_semaphore"].eq("YELLOW")]
        if len(red) == 0 and len(yellow) == 0:
            summary_lines.append(
                "Nessuna sottozona supera le soglie sperimentali congelate a 24, 48 o 72 ore."
            )
        if len(red):
            summary_lines.append(
                "Segnale sperimentale elevato (rosso) in: "
                + ", ".join(red["basin_label"].astype(str))
                + ". Approfondire immediatamente con fonti ufficiali."
            )
        if len(yellow):
            summary_lines.append(
                "Segnale da approfondire (giallo) in: "
                + ", ".join(yellow["basin_label"].astype(str))
                + "."
            )
    elif perform_inference:
        summary_lines.append(
            "La catena software ha prodotto l'output numerico dei tre modelli, ma il gate non ne autorizza l'interpretazione come previsione di piena."
        )
    else:
        summary_lines.append(
            "Il gate operativo ha bloccato l'inferenza; nessuna probabilita viene emessa."
        )

    result_csv_p = output_root / "NW_FloodWatch_predictions.csv"
    result_json_p = output_root / "NW_FloodWatch_predictions.json"
    model_registry_p = output_root / "NW_FloodWatch_model_registry.csv"
    audit_p = output_root / "NW_FloodWatch_run_audit.json"
    txt_p = output_root / "NW_FloodWatch_Bollettino.txt"
    html_p = output_root / "NW_FloodWatch_Bollettino.html"
    pdf_p = output_root / config.get("pdf_filename", "NW_FloodWatch_Bollettino.pdf")

    result.to_csv(result_csv_p, index=False)
    result_json_p.write_text(
        json.dumps(result.to_dict(orient="records"), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    pd.DataFrame(model_registry_rows).to_csv(model_registry_p, index=False)

    audit = {
        "nw_floodwatch_version": VERSION,
        "run_id": snapshot.name,
        "issue_date": str(result["issue_date"].iloc[0]),
        "generated_at_local": now_text(),
        "bulletin_mode": bulletin_mode,
        "prospective_beta_allowed": beta_allowed,
        "scientific_beta_interpretation": scientific_mode,
        "in_core_season": in_season,
        "unexpected_operational_missing_cells": unexpected_missing,
        "distribution_equivalence_confirmed": bool(
            gate.get("distribution_equivalence_confirmed", False)
        ),
        "official_warning_use_allowed": False,
        "model_inference_performed": perform_inference,
        "model_or_threshold_modified": False,
        "summary": summary_lines,
        "source_beta_gate": str(snapshot / "operational_beta_gate_audit_v1_0.json"),
        "pdf": str(pdf_p),
    }
    audit_p.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    # Plain text companion.
    lines = [
        "NW FLOODWATCH - BOLLETTINO SPERIMENTALE",
        "",
        config["disclaimer"],
        "",
        config["season_warning"],
        "",
        config["target_warning"],
        "",
        f"Run: {snapshot.name}",
        f"Data: {result['issue_date'].iloc[0]}",
        f"Modalita: {bulletin_mode}",
        "",
        "SINTESI",
        *[f"- {x}" for x in summary_lines],
        "",
    ]
    if perform_inference:
        t = result[[
            "receptor_id",
            "calibrated_probability_24h",
            "calibrated_probability_48h",
            "calibrated_probability_72h",
            "overall_semaphore",
            "action_note",
        ]].copy()
        for h in HORIZONS:
            t[f"calibrated_probability_{h}h"] = (
                100 * t[f"calibrated_probability_{h}h"]
            ).round(2)
        lines.append(t.to_string(index=False))
    lines.extend([
        "",
        f"Autore: {config['author_name']} - {config['author_email']}",
        "Aggiornamento domani: eseguire python nw_flood_watch.py --open",
    ])
    txt_p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Lightweight HTML companion.
    html_rows = []
    for _, r in result.iterrows():
        html_rows.append(
            "<tr>"
            f"<td>{html.escape(str(r['basin_label']))}</td>"
            f"<td>{100*r['calibrated_probability_24h']:.2f}%</td>"
            f"<td>{100*r['calibrated_probability_48h']:.2f}%</td>"
            f"<td>{100*r['calibrated_probability_72h']:.2f}%</td>"
            f"<td>{html.escape(status_label(str(r['overall_semaphore'])))}</td>"
            f"<td>{html.escape(str(r['action_note']))}</td>"
            "</tr>"
        )
    html_text = f"""<!doctype html><html lang='it'><head><meta charset='utf-8'>
<title>NW FloodWatch</title><style>
body{{font-family:Arial,sans-serif;max-width:1500px;margin:30px auto;padding:0 25px;line-height:1.45}}
.warning{{border:3px solid #900;padding:14px;background:#fff5f5}}table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border:1px solid #aaa;padding:6px}}th{{background:#1F4E78;color:white}}
</style></head><body><h1>NW FloodWatch</h1><div class='warning'><b>DOCUMENTO NON UFFICIALE</b><p>{html.escape(config['disclaimer'])}</p><p><b>{html.escape(config['season_warning'])}</b></p></div>
<h2>{html.escape(bulletin_mode)}</h2><ul>{''.join(f'<li>{html.escape(x)}</li>' for x in summary_lines)}</ul>
<table><thead><tr><th>Sottozona</th><th>24h</th><th>48h</th><th>72h</th><th>Semaforo sperimentale</th><th>Nota</th></tr></thead><tbody>{''.join(html_rows)}</tbody></table>
<p><b>{html.escape(config['author_name'])}</b> - {html.escape(config['author_email'])}</p></body></html>"""
    html_p.write_text(html_text, encoding="utf-8")

    build_pdf_bulletin(
        root=root,
        output_pdf=pdf_p,
        result=result,
        gate=gate,
        thresholds=thresholds,
        bulletin_mode=bulletin_mode,
        scientific_mode=scientific_mode,
        summary_lines=summary_lines,
        config=config,
    )

    latest_dir = root / "nw_floodwatch_output"
    shutil.copy2(pdf_p, latest_dir / "LATEST_NW_FloodWatch_Bollettino.pdf")
    shutil.copy2(txt_p, latest_dir / "LATEST_NW_FloodWatch_Bollettino.txt")
    shutil.copy2(html_p, latest_dir / "LATEST_NW_FloodWatch_Bollettino.html")

    return {
        "mode": bulletin_mode,
        "scientific_mode": scientific_mode,
        "perform_inference": perform_inference,
        "output_dir": output_root,
        "pdf": pdf_p,
        "html": html_p,
        "txt": txt_p,
        "csv": result_csv_p,
        "json": result_json_p,
        "audit": audit_p,
        "summary": summary_lines,
        "result": result,
    }


def build_demo(root: Path, config: dict) -> Path:
    # Synthetic layout-only data. Deliberately not based on current conditions.
    ids = [
        "LIG_BISAGNO", "LIG_CENTA", "LIG_MAGRA", "LIG_POLCEVERA",
        "NW_BORMIDA", "NW_CHISONE", "NW_DORA_BALTEA", "NW_DORA_RIPARIA",
        "NW_MAIRA", "NW_ORBA", "NW_ORCO", "NW_PELLICE", "NW_SCRIVIA",
        "NW_SESIA", "NW_STURA_DEMONTE", "NW_STURA_LANZO", "NW_TANARO_ALTO",
        "NW_TANARO_MEDIO_BASSO", "NW_TOCE", "NW_VARAITA",
    ]
    rng = np.random.default_rng(42)
    result = pd.DataFrame({
        "receptor_id": ids,
        "issue_date": ["2026-09-15"] * len(ids),
        "run_id": ["DEMO_LAYOUT"] * len(ids),
    })
    result["calibrated_probability_24h"] = rng.uniform(0.01, 0.42, len(ids))
    result["calibrated_probability_48h"] = rng.uniform(0.02, 0.45, len(ids))
    result["calibrated_probability_72h"] = rng.uniform(0.03, 0.48, len(ids))

    thresholds = {
        (24, RECALL80): 0.132, (24, MAX_CSI): 0.321,
        (48, RECALL80): 0.095, (48, MAX_CSI): 0.355,
        (72, RECALL80): 0.078, (72, MAX_CSI): 0.340,
    }
    gate = {
        "overall_status": "DEMO_ONLY",
        "prospective_beta_allowed": True,
        "in_core_season_sep_dec": True,
    }
    out = root / "NW_FloodWatch_Bollettino_DEMO.pdf"
    build_pdf_bulletin(
        root=root,
        output_pdf=out,
        result=result,
        gate=gate,
        thresholds=thresholds,
        bulletin_mode="DEMO_LAYOUT_ONLY",
        scientific_mode=True,
        summary_lines=[
            "Esempio grafico: dati fittizi esclusivamente per verificare l'impaginazione.",
        ],
        config=config,
        demo=True,
    )
    return out


def print_final_summary(rendered):
    print("\n" + "=" * 120)
    print("NW FLOODWATCH - RUN COMPLETATO")
    print("=" * 120)
    print(f"Modalita    : {rendered['mode']}")
    print(f"Scientifico : {rendered['scientific_mode']}")
    print(f"Inferenza   : {rendered['perform_inference']}")
    for line in rendered["summary"]:
        print(f"- {line}")

    if rendered["perform_inference"]:
        result = rendered["result"]
        print("\nMassimo output calibrato per orizzonte:")
        for h in HORIZONS:
            idx = result[f"calibrated_probability_{h}h"].idxmax()
            r = result.loc[idx]
            print(
                f"{h:>2} h | {r['receptor_id']:<24} | "
                f"{100*r[f'calibrated_probability_{h}h']:6.2f}% | "
                f"{status_label(str(r['overall_semaphore']))}"
            )

    print(f"\nPDF  : {rendered['pdf']}")
    print(f"HTML : {rendered['html']}")
    print(f"CSV  : {rendered['csv']}")
    print(f"JSON : {rendered['json']}")
    print(f"Audit: {rendered['audit']}")
    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(description="NW FloodWatch definitive local runner")
    parser.add_argument("--force-full", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--allow-degraded-smoke", action="store_true")
    parser.add_argument("--open", action="store_true", help="Apre il PDF al termine")
    parser.add_argument("--pdf-demo", action="store_true", help="Genera solo un PDF demo con dati fittizi")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config = load_config(root)

    if args.pdf_demo:
        p = build_demo(root, config)
        print(f"PDF demo creato: {p}")
        if args.open:
            webbrowser.open(p.resolve().as_uri())
        return

    print("=" * 120)
    print(f"NW FLOODWATCH v{VERSION}")
    print("=" * 120)
    print(f"Cartella     : {root}")
    print(f"Python       : {sys.executable}")
    print(f"Avvio        : {now_text()}")

    latest = latest_snapshot_with_gate(root)
    today = datetime.now().astimezone().date().isoformat()
    use_existing = False

    if args.report_only:
        if latest is None:
            raise SystemExit("--report-only richiesto ma non esiste uno snapshot completato.")
        use_existing = True
    elif not args.force_full and latest is not None and gate_issue_date(latest) == today:
        print("Snapshot della data odierna gia presente: rigenero il bollettino senza riscaricare.")
        print("Per forzare l'intera catena usare --force-full.")
        use_existing = True

    if not use_existing:
        outroot = root / "nw_floodwatch_output"
        outroot.mkdir(parents=True, exist_ok=True)
        log_p = outroot / "NW_FloodWatch_pipeline_latest.log"
        with log_p.open("w", encoding="utf-8") as log_handle:
            for label, filename in COMPONENTS:
                run_component(root, label, filename, log_handle)

        latest = latest_snapshot_with_gate(root)
        if latest is None:
            raise RuntimeError("Pipeline terminata ma nessuno snapshot con beta gate e stato trovato.")

    assert latest is not None
    gate = json.loads(
        (latest / "operational_beta_gate_audit_v1_0.json").read_text(encoding="utf-8")
    )

    rendered = render_outputs(
        root=root,
        snapshot=latest,
        gate=gate,
        allow_degraded_smoke=args.allow_degraded_smoke,
        config=config,
    )
    print_final_summary(rendered)

    if args.open:
        webbrowser.open(rendered["pdf"].resolve().as_uri())


if __name__ == "__main__":
    main()
