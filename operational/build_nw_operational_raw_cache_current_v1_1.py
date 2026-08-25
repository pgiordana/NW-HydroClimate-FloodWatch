#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_nw_operational_raw_cache_current_v1_1.py

FASE 14 — COSTRUZIONE DEL PRIMO RAW CACHE OPERATIVO NW HYDROCLIMATE.

Prerequisiti runtime:
    ecmwf-opendata e copernicusmarine installati e accesso Internet.
    La disponibilita dei provider viene verificata direttamente a ogni run.

COSA FA
-------
1) individua il più recente ciclo IFS 00 UTC realmente disponibile;
2) scarica i campi necessari per ricostruire il giorno di emissione t:
   - msl, tcwv, cape, sd, swvl1, swvl2, swvl3 ai passi 0..24 h;
   - tp ai passi 3..24 h;
   - q/u/v/t a 925/850/700 hPa ai passi 0..24 h;
   - q/u/v su un set verticale più ampio per il proxy IVT;
3) prova a recuperare il totale tp delle due giornate precedenti
   per inizializzare il rolling di 3 giorni;
4) scarica thetao Copernicus Marine per IL GIORNO DI EMISSIONE, non per
   la data massima del catalogo (che comprende la previsione futura);
5) salva manifest, checksum, ciclo, valid times e stato del cache.

IMPORTANTE
----------
- Questo è RAW CACHE: non costruisce ancora le 83 feature.
- Non esegue il modello.
- Il giorno corrente IFS è un forecast-filled day: è un PROXY operativo
  del giorno t completo del training ERA5.
- L'IVT usa un set verticale IFS Open Data, non ancora dichiarato
  numericamente equivalente all'IVT ERA5 storico.
- Il campo CMEMS del giorno t può essere analisi o forecast NRT a seconda
  del ciclo di produzione disponibile; viene quindi marcato come proxy.
- I file ECMWF sono globali GRIB2 ma filtrati per parametro/step.
- Il programma usa fallback ecmwf -> aws -> google se necessario.

OUTPUT
------
nw_operational_raw_cache/
  YYYYMMDDT00Z/
    ecmwf/
    copernicus_marine/
    raw_cache_manifest_v1_0.csv
    raw_cache_coverage_v1_0.csv
    raw_cache_audit_v1_0.json
    raw_cache_audit_v1_0.txt
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


ECMWF_SOURCES = ["ecmwf", "aws", "google"]

STATE_PARAMS = [
    "msl",
    "tcwv",
    "cape",
    "sd",
    "swvl1",
    "swvl2",
    "swvl3",
]

TP_PARAM = ["tp"]

LOW_PL_PARAMS = ["q", "u", "v", "t"]
LOW_LEVELS = [925, 850, 700]

IVT_PARAMS = ["q", "u", "v"]
IVT_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300]

DAY_STEPS = list(range(0, 25, 3))
ACCUM_STEPS = list(range(3, 25, 3))

CMEMS_DATASET_ID = "cmems_mod_med_phy-tem_anfc_4.2km_P1D-m"
CMEMS_VARIABLE = "thetao"

# Broad Western/Central Mediterranean domain.
# Chosen to cover source corridors relevant to NW Italy while avoiding
# downloading the entire Mediterranean basin.
CMEMS_BBOX = {
    "minimum_longitude": -6.0,
    "maximum_longitude": 20.0,
    "minimum_latitude": 30.2,
    "maximum_latitude": 46.0,
    "minimum_depth": 1.0183,
    "maximum_depth": 100.0,
}


def fmt_seconds(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def progress(phase, done, total, started, current=""):
    elapsed = max(time.time() - started, 1e-9)
    pct = 100.0 * done / total if total else 100.0
    rate = done / elapsed if done else 0.0
    eta = (total - done) / rate if rate > 0 else float("inf")

    msg = (
        f"\r{phase} | {done}/{total} | {pct:6.2f}% "
        f"| elapsed {fmt_seconds(elapsed)} "
        f"| rate {rate:7.3f}/s | ETA {fmt_seconds(eta)}"
    )
    if current:
        msg += f" | {str(current)[:135]}"

    print(msg.ljust(285), end="", flush=True)
    if done >= total:
        print(flush=True)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def short_error(exc):
    return f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')[:600]}"


def module_available(name):
    return importlib.util.find_spec(name) is not None


def normalize_latest(value):
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def make_client(source):
    from ecmwf.opendata import Client

    return Client(
        source=source,
        model="ifs",
        resol="0p25",
    )


def find_latest_00z():
    errors = []

    for source in ECMWF_SOURCES:
        try:
            client = make_client(source)

            latest = client.latest(
                stream="oper",
                type="fc",
                time=0,
                step=3,
                param="msl",
            )

            return source, normalize_latest(latest)

        except Exception as exc:
            errors.append(
                f"{source}: {short_error(exc)}"
            )

    raise RuntimeError(
        "Nessuna sorgente ECMWF ha restituito il latest 00Z.\n"
        + "\n".join(errors)
    )


def retrieve_ecmwf(
    target,
    *,
    date,
    time_utc,
    step,
    param,
    levelist=None,
):
    """
    Retrieve with source fallback.
    """
    errors = []

    for source in ECMWF_SOURCES:
        try:
            client = make_client(source)

            kwargs = dict(
                stream="oper",
                type="fc",
                date=int(date.strftime("%Y%m%d")),
                time=int(time_utc),
                step=step,
                param=param,
                target=str(target),
            )

            if levelist is not None:
                kwargs["levelist"] = levelist

            result = client.retrieve(**kwargs)

            if (
                not target.exists()
                or target.stat().st_size <= 0
            ):
                raise RuntimeError(
                    "Retrieval completed but target is empty."
                )

            return {
                "status": "PASS",
                "source": source,
                "result_datetime": str(
                    getattr(result, "datetime", "")
                ),
                "error": "",
            }

        except Exception as exc:
            errors.append(
                f"{source}: {short_error(exc)}"
            )

            if target.exists() and target.stat().st_size == 0:
                target.unlink(missing_ok=True)

    return {
        "status": "FAIL",
        "source": "",
        "result_datetime": "",
        "error": " | ".join(errors),
    }


def manifest_row(
    provider,
    role,
    path,
    status,
    source="",
    issue_cycle="",
    valid_date="",
    params="",
    levels="",
    steps="",
    semantics="",
    error="",
):
    exists = Path(path).exists()
    nonempty = exists and Path(path).stat().st_size > 0

    return {
        "provider": provider,
        "role": role,
        "status": status,
        "source": source,
        "issue_cycle_utc": issue_cycle,
        "valid_date_utc": valid_date,
        "parameters": params,
        "levels": levels,
        "steps_hours": steps,
        "semantics": semantics,
        "file": str(path),
        "bytes": Path(path).stat().st_size if nonempty else 0,
        "sha256": sha256(path) if nonempty else "",
        "error": error,
    }


def download_cmems_issue_day(out_dir, issue_date):
    if not module_available("copernicusmarine"):
        return {
            "status": "FAIL_PACKAGE_MISSING",
            "file": "",
            "error": (
                "copernicusmarine non installato."
            ),
        }

    import copernicusmarine

    filename = (
        f"cmems_thetao_wmed_0_100m_{issue_date.isoformat()}.nc"
    )
    target = out_dir / filename

    try:
        result = copernicusmarine.subset(
            dataset_id=CMEMS_DATASET_ID,
            variables=[CMEMS_VARIABLE],
            start_datetime=f"{issue_date.isoformat()}T00:00:00",
            end_datetime=f"{issue_date.isoformat()}T23:59:59",
            minimum_longitude=CMEMS_BBOX["minimum_longitude"],
            maximum_longitude=CMEMS_BBOX["maximum_longitude"],
            minimum_latitude=CMEMS_BBOX["minimum_latitude"],
            maximum_latitude=CMEMS_BBOX["maximum_latitude"],
            minimum_depth=CMEMS_BBOX["minimum_depth"],
            maximum_depth=CMEMS_BBOX["maximum_depth"],
            output_directory=str(out_dir),
            output_filename=filename,
            overwrite=True,
            disable_progress_bar=False,
        )

        possible = [target]

        result_path = getattr(result, "file_path", None)
        if result_path:
            possible.insert(0, Path(str(result_path)))

        selected = next(
            (
                p
                for p in possible
                if p.exists()
                and p.stat().st_size > 0
            ),
            None,
        )

        if selected is None:
            raise RuntimeError(
                "subset() completed but no output file found."
            )

        return {
            "status": "PASS",
            "file": str(selected),
            "error": "",
        }

    except Exception as exc:
        return {
            "status": "FAIL",
            "file": str(target),
            "error": short_error(exc),
        }


def main():
    root = Path(__file__).resolve().parent

    # Definitive runner v1.1: no stale provider-probe file is required.
    # Provider availability is checked live below via ECMWF latest/retrieval
    # and Copernicus Marine subset calls.
    if not module_available("ecmwf.opendata"):
        raise SystemExit(
            "Manca ecmwf-opendata."
        )

    print("=" * 220)
    print("NW HYDROCLIMATE — OPERATIONAL RAW CACHE CURRENT v1.1")
    print("=" * 220)

    # ------------------------------------------------------------------
    # PHASE 1/6 — lock issue cycle
    # ------------------------------------------------------------------
    print("\nPHASE 1/6 — lock latest available ECMWF 00Z issue cycle")
    start = time.time()

    latest_source, issue_cycle = find_latest_00z()
    issue_date = issue_cycle.date()

    run_id = issue_cycle.strftime("%Y%m%dT00Z")

    out = (
        root
        / "nw_operational_raw_cache"
        / run_id
    )
    ecmwf_dir = out / "ecmwf"
    cmems_dir = out / "copernicus_marine"

    ecmwf_dir.mkdir(parents=True, exist_ok=True)
    cmems_dir.mkdir(parents=True, exist_ok=True)

    progress(
        "PHASE 1/6",
        1,
        1,
        start,
        f"issue_cycle={issue_cycle.isoformat()} | source={latest_source}",
    )

    manifest = []

    # ------------------------------------------------------------------
    # PHASE 2/6 — current-day surface fields
    # ------------------------------------------------------------------
    print("\nPHASE 2/6 — download current-day IFS surface/state fields")
    start = time.time()

    surface_p = (
        ecmwf_dir
        / f"ifs_{run_id}_surface_state_steps0_24.grib2"
    )

    r1 = retrieve_ecmwf(
        surface_p,
        date=issue_date,
        time_utc=0,
        step=DAY_STEPS,
        param=STATE_PARAMS,
    )

    manifest.append(
        manifest_row(
            "ECMWF_OPEN_DATA_IFS",
            "CURRENT_DAY_SURFACE_STATE",
            surface_p,
            r1["status"],
            source=r1["source"],
            issue_cycle=issue_cycle.isoformat(),
            valid_date=issue_date.isoformat(),
            params=",".join(STATE_PARAMS),
            steps=",".join(str(x) for x in DAY_STEPS),
            semantics=(
                "Forecast-filled current day proxy. "
                "Instantaneous/state fields sampled every 3 h."
            ),
            error=r1["error"],
        )
    )

    tp_p = (
        ecmwf_dir
        / f"ifs_{run_id}_tp_steps3_24.grib2"
    )

    r2 = retrieve_ecmwf(
        tp_p,
        date=issue_date,
        time_utc=0,
        step=ACCUM_STEPS,
        param=TP_PARAM,
    )

    manifest.append(
        manifest_row(
            "ECMWF_OPEN_DATA_IFS",
            "CURRENT_DAY_PRECIP_ACCUM",
            tp_p,
            r2["status"],
            source=r2["source"],
            issue_cycle=issue_cycle.isoformat(),
            valid_date=issue_date.isoformat(),
            params="tp",
            steps=",".join(str(x) for x in ACCUM_STEPS),
            semantics=(
                "Forecast accumulation from 00Z initialization; "
                "step 24 is candidate daily total proxy."
            ),
            error=r2["error"],
        )
    )

    progress(
        "PHASE 2/6",
        2,
        2,
        start,
        f"surface={r1['status']} | tp={r2['status']}",
    )

    # ------------------------------------------------------------------
    # PHASE 3/6 — pressure levels + IVT levels
    # ------------------------------------------------------------------
    print("\nPHASE 3/6 — download pressure-level fields and operational IVT inputs")
    start = time.time()

    low_p = (
        ecmwf_dir
        / f"ifs_{run_id}_q_u_v_t_925_850_700_steps0_24.grib2"
    )

    r3 = retrieve_ecmwf(
        low_p,
        date=issue_date,
        time_utc=0,
        step=DAY_STEPS,
        param=LOW_PL_PARAMS,
        levelist=LOW_LEVELS,
    )

    manifest.append(
        manifest_row(
            "ECMWF_OPEN_DATA_IFS",
            "CURRENT_DAY_LOW_PRESSURE_LEVELS",
            low_p,
            r3["status"],
            source=r3["source"],
            issue_cycle=issue_cycle.isoformat(),
            valid_date=issue_date.isoformat(),
            params=",".join(LOW_PL_PARAMS),
            levels=",".join(str(x) for x in LOW_LEVELS),
            steps=",".join(str(x) for x in DAY_STEPS),
            semantics=(
                "Forecast-filled current-day q/u/v/t sampled every 3 h."
            ),
            error=r3["error"],
        )
    )

    ivt_p = (
        ecmwf_dir
        / f"ifs_{run_id}_q_u_v_ivt_levels_steps0_24.grib2"
    )

    r4 = retrieve_ecmwf(
        ivt_p,
        date=issue_date,
        time_utc=0,
        step=DAY_STEPS,
        param=IVT_PARAMS,
        levelist=IVT_LEVELS,
    )

    manifest.append(
        manifest_row(
            "ECMWF_OPEN_DATA_IFS",
            "CURRENT_DAY_IVT_INPUTS",
            ivt_p,
            r4["status"],
            source=r4["source"],
            issue_cycle=issue_cycle.isoformat(),
            valid_date=issue_date.isoformat(),
            params=",".join(IVT_PARAMS),
            levels=",".join(str(x) for x in IVT_LEVELS),
            steps=",".join(str(x) for x in DAY_STEPS),
            semantics=(
                "Operational IVT proxy input. Vertical level set differs "
                "from historical ERA5 implementation; validation required."
            ),
            error=r4["error"],
        )
    )

    progress(
        "PHASE 3/6",
        2,
        2,
        start,
        f"low_levels={r3['status']} | ivt_inputs={r4['status']}",
    )

    # ------------------------------------------------------------------
    # PHASE 4/6 — bootstrap 3-day precipitation history
    # ------------------------------------------------------------------
    print("\nPHASE 4/6 — try retained prior 00Z cycles for 3-day precipitation bootstrap")
    start = time.time()

    prior_pass = 0

    for idx, lag_days in enumerate([1, 2], 1):
        d = issue_date - timedelta(days=lag_days)
        prior_id = d.strftime("%Y%m%dT00Z")

        p = (
            ecmwf_dir
            / f"ifs_{prior_id}_tp_step24_bootstrap.grib2"
        )

        rr = retrieve_ecmwf(
            p,
            date=d,
            time_utc=0,
            step=24,
            param=["tp"],
        )

        if rr["status"] == "PASS":
            prior_pass += 1

        manifest.append(
            manifest_row(
                "ECMWF_OPEN_DATA_IFS",
                f"PRIOR_DAY_TP_BOOTSTRAP_LAG{lag_days}D",
                p,
                rr["status"],
                source=rr["source"],
                issue_cycle=f"{d.isoformat()}T00:00:00+00:00",
                valid_date=d.isoformat(),
                params="tp",
                steps="24",
                semantics=(
                    "One-time operational bootstrap from retained historical "
                    "forecast cycle, NOT ERA5 observation/reanalysis."
                ),
                error=rr["error"],
            )
        )

        progress(
            "PHASE 4/6",
            idx,
            2,
            start,
            f"lag={lag_days}d | {rr['status']}",
        )

    # ------------------------------------------------------------------
    # PHASE 5/6 — CMEMS issue-day marine state proxy
    # ------------------------------------------------------------------
    print("\nPHASE 5/6 — download issue-day Copernicus Marine thetao 0-100 m")
    start = time.time()

    marine = download_cmems_issue_day(
        cmems_dir,
        issue_date,
    )

    marine_path = Path(
        marine["file"]
        if marine["file"]
        else cmems_dir / "missing.nc"
    )

    manifest.append(
        manifest_row(
            "COPERNICUS_MARINE_MEDSEA",
            "ISSUE_DAY_THETAO_0_100M_PROXY",
            marine_path,
            marine["status"],
            source=CMEMS_DATASET_ID,
            issue_cycle=issue_cycle.isoformat(),
            valid_date=issue_date.isoformat(),
            params="thetao",
            levels="1.0183-100m",
            semantics=(
                "Issue-day marine temperature field from analysis/forecast "
                "product. Never uses catalogue maximum future date as current state."
            ),
            error=marine["error"],
        )
    )

    progress(
        "PHASE 5/6",
        1,
        1,
        start,
        f"CMEMS={marine['status']}",
    )

    # ------------------------------------------------------------------
    # PHASE 6/6 — manifest/coverage/audit
    # ------------------------------------------------------------------
    print("\nPHASE 6/6 — freeze raw-cache manifest, coverage and checksums")
    start = time.time()

    manifest_df = pd.DataFrame(
        manifest
    )

    critical_roles = [
        "CURRENT_DAY_SURFACE_STATE",
        "CURRENT_DAY_PRECIP_ACCUM",
        "CURRENT_DAY_LOW_PRESSURE_LEVELS",
        "CURRENT_DAY_IVT_INPUTS",
        "ISSUE_DAY_THETAO_0_100M_PROXY",
    ]

    critical = manifest_df[
        manifest_df["role"].isin(
            critical_roles
        )
    ].copy()

    critical_pass = bool(
        len(critical) == len(critical_roles)
        and critical["status"].eq("PASS").all()
    )

    coverage = pd.DataFrame(
        [
            {
                "coverage_item": "CURRENT_DAY_CRITICAL_RAW_FIELDS",
                "available_units": int(
                    critical["status"].eq("PASS").sum()
                ),
                "required_units": len(critical_roles),
                "coverage_fraction": float(
                    critical["status"].eq("PASS").mean()
                ),
                "status": (
                    "PASS"
                    if critical_pass
                    else "FAIL"
                ),
            },
            {
                "coverage_item": "PRECIP_3D_BOOTSTRAP_PRIOR_DAYS",
                "available_units": prior_pass,
                "required_units": 2,
                "coverage_fraction": prior_pass / 2.0,
                "status": (
                    "PASS"
                    if prior_pass == 2
                    else "PARTIAL"
                ),
            },
            {
                "coverage_item": "PRECIP_7D_ROLLING_CACHE",
                "available_units": min(1 + prior_pass, 7),
                "required_units": 7,
                "coverage_fraction": min(1 + prior_pass, 7) / 7.0,
                "status": "WARMUP_REQUIRED",
            },
            {
                "coverage_item": "PRECIP_14D_ROLLING_CACHE",
                "available_units": min(1 + prior_pass, 14),
                "required_units": 14,
                "coverage_fraction": min(1 + prior_pass, 14) / 14.0,
                "status": "WARMUP_REQUIRED",
            },
            {
                "coverage_item": "FULL_OPERATIONAL_ANTECEDENT_CACHE",
                "available_units": 1,
                "required_units": 14,
                "coverage_fraction": 1 / 14.0,
                "status": "WARMUP_REQUIRED",
            },
        ]
    )

    manifest_p = (
        out
        / "raw_cache_manifest_v1_0.csv"
    )
    coverage_p = (
        out
        / "raw_cache_coverage_v1_0.csv"
    )
    audit_json_p = (
        out
        / "raw_cache_audit_v1_0.json"
    )
    audit_txt_p = (
        out
        / "raw_cache_audit_v1_0.txt"
    )

    manifest_df.to_csv(
        manifest_p,
        index=False,
    )
    coverage.to_csv(
        coverage_p,
        index=False,
    )

    if critical_pass:
        if prior_pass == 2:
            overall = (
                "PASS_RAW_CACHE_CURRENT__3D_PRECIP_BOOTSTRAP_READY"
            )
        else:
            overall = (
                "PASS_RAW_CACHE_CURRENT__ANTECEDENT_WARMUP_PARTIAL"
            )
    else:
        overall = (
            "FAIL_CRITICAL_CURRENT_RAW_FIELDS"
        )

    total_bytes = int(
        pd.to_numeric(
            manifest_df["bytes"],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    audit = {
        "version": "1.0",
        "overall_status": overall,
        "issue_cycle_utc":
            issue_cycle.isoformat(),
        "issue_date_utc":
            issue_date.isoformat(),
        "latest_00z_discovery_source":
            latest_source,
        "critical_current_raw_fields_pass":
            critical_pass,
        "prior_precip_bootstrap_days_pass":
            prior_pass,
        "raw_cache_total_bytes":
            total_bytes,
        "model_prediction_performed":
            False,
        "feature_engine_performed":
            False,
        "operational_semantics":
            "FORECAST_FILLED_DAY_T_PROXY",
        "cmems_semantics":
            "ISSUE_DAY_ANALYSIS_FORECAST_PROXY",
        "ivt_equivalence_validated":
            False,
        "full_7d_14d_cache_ready":
            False,
        "next_step":
            (
                "Parse/crop raw fields to the 20 target receptors, build the "
                "current receptor-day operational feature snapshot, and compare "
                "all constructible P1/P2 features against the frozen feature "
                "definitions. Missing warm-up lags remain NaN and must be logged."
            ),
    }

    audit_json_p.write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    display_cols = [
        "provider",
        "role",
        "status",
        "source",
        "valid_date_utc",
        "parameters",
        "levels",
        "steps_hours",
        "bytes",
        "error",
    ]

    lines = [
        "=" * 220,
        "NW HYDROCLIMATE — OPERATIONAL RAW CACHE CURRENT v1.1",
        "=" * 220,
        f"OVERALL STATUS                 : {overall}",
        f"Issue cycle UTC                : {issue_cycle.isoformat()}",
        f"Latest-00Z discovery source    : {latest_source}",
        f"Critical current raw fields    : {critical_pass}",
        f"Prior precip bootstrap days    : {prior_pass}/2",
        f"Raw cache total bytes          : {total_bytes}",
        "Full 7d/14d antecedent cache   : False — warm-up required",
        "Model prediction performed     : False",
        "",
        "RAW CACHE MANIFEST",
        manifest_df[display_cols].to_string(index=False),
        "",
        "CACHE COVERAGE",
        coverage.to_string(index=False),
        "",
        "IMPORTANT",
        "The current day is represented by IFS 00Z analysis/forecast samples through +24 h.",
        "This is an operational proxy for the end-of-day ERA5 state used during training.",
        "CMEMS uses the issue date, not the catalogue maximum date; the latter includes future forecast days.",
        "The IVT input level set is operational and still requires ERA5-vs-IFS compatibility validation.",
        "Any unavailable rolling antecedent feature remains NaN during warm-up; it is never filled with zero.",
        "",
        "NEXT STEP",
        "Build receptor-level feature-engine v1.0 from this raw cache.",
        "",
        f"Manifest : {manifest_p}",
        f"Coverage : {coverage_p}",
        f"Audit    : {audit_json_p}",
        f"Output   : {out}",
    ]

    audit_txt_p.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    progress(
        "PHASE 6/6",
        1,
        1,
        start,
        f"status={overall}",
    )

    print("\n" + "=" * 220)
    print("\n".join(lines[3:]))
    print("=" * 220)
    print(f"OVERALL STATUS : {overall}")
    print(f"Output         : {out}")
    print("=" * 220)


if __name__ == "__main__":
    main()
