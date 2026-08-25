#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_nw_observation_daily_station_layer_v1_0.py

Costruisce il layer giornaliero canonico PER SERIE/STAZIONE a partire dai
27.8 milioni di valori standardizzati già chiusi PASS.

Prerequisiti:
- nw_observations_values_v1_0/
    standardized_series_manifest_v1_2.csv
    standardized_values_audit_v1_2.json
- nw_observations_standardized_v1_4/
    observation_series_registry_v1_4.csv
    observation_time_audit_v1_0.json
    values_manifest_time_overlay_v1_0.csv

Principi:
- NON aggrega ancora tra stazioni e NON media livelli/portate tra sezioni.
- Mantiene separata ogni `source_column`.
- Piemonte daily: preserva la data sorgente.
- VdA: `timestamp_source` interpretato UTC secondo freeze v1.4.
- ARPAL: usa `timestamp_utc` già materializzato.
- Nessuna imputazione.
- I giorni incompleti NON vengono cancellati: sono marcati con copertura.
- Il valore giornaliero primario dipende dalla variabile:
    PRECIP_MM, SUNSHINE_DURATION_MIN, LEAF_WETNESS_DURATION_S -> SUM
    DISCHARGE_MIN_M3_S -> MIN
    DISCHARGE_MAX_M3_S -> MAX
    WIND_DIR_DEG -> CIRCULAR_MEAN
    tutte le altre -> MEAN
- Per tutte le variabili scalari vengono anche conservati min/max/mean/sum.
- La copertura è calcolata rispetto alla cadenza nominale inferita
  per ciascuna serie/source_column.

Output:
nw_observations_daily_v1_0/
  series/<provider>/<source_series_id>.daily.csv.gz
  metadata/<provider>/<source_series_id>.daily.meta.json
  daily_series_manifest_v1_0.csv
  daily_value_summary_v1_0.csv
  daily_station_layer_audit_v1_0.json
  daily_station_layer_audit_v1_0.txt

Restart-safe.
NON modifica i valori standardizzati né i dati sorgente.
"""

from __future__ import annotations

import json
import math
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_SERIES = 1312
TARGET_MONTHS = {9, 10, 11, 12}

SUM_CODES = {
    "PRECIP_MM",
    "SUNSHINE_DURATION_MIN",
    "LEAF_WETNESS_DURATION_S",
}

MIN_PRIMARY_CODES = {
    "DISCHARGE_MIN_M3_S",
}

MAX_PRIMARY_CODES = {
    "DISCHARGE_MAX_M3_S",
}

CIRCULAR_CODES = {
    "WIND_DIR_DEG",
}


def safe_name(s):
    s = str(s or "")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    return s or "series"


def atomic_write_csv_gz(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, compression="gzip")
    tmp.replace(path)


def atomic_write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def fingerprint(path: Path):
    st = path.stat()
    return {
        "source_size_bytes": int(st.st_size),
        "source_mtime_ns": int(st.st_mtime_ns),
    }


def parse_boolish_na(x):
    try:
        return pd.isna(x)
    except Exception:
        return False


def circular_mean_deg(values):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan

    rad = np.deg2rad(np.mod(arr, 360.0))
    s = np.sin(rad).mean()
    c = np.cos(rad).mean()

    strength = float(np.hypot(s, c))

    if strength < 1e-12:
        return np.nan, strength

    ang = math.degrees(math.atan2(s, c))
    if ang < 0:
        ang += 360.0

    return float(ang), strength


def infer_nominal_cadence_seconds(df_sub, provider, time_resolution):
    # Piemonte daily: semanticamente 1 valore/giorno per source_column.
    if str(time_resolution) == "daily" or provider == "ARPA_PIEMONTE":
        return 86400, 1, "DAILY_SOURCE"

    # ARPAL: intervallo esplicito.
    if "interval_seconds" in df_sub.columns:
        ints = pd.to_numeric(
            df_sub["interval_seconds"],
            errors="coerce",
        ).dropna()

        ints = ints[ints > 0]

        if len(ints):
            med = float(ints.median())
            expected = round(86400 / med)

            if (
                expected >= 1
                and abs(expected * med - 86400) <= max(1.0, med * 0.01)
            ):
                return int(round(med)), int(expected), "INTERVAL_SECONDS"

    # VdA o eventuali altre serie subgiornaliere:
    # cadenza nominale dalla mediana dei delta intra-stagionali.
    ts = pd.to_datetime(
        df_sub["_timestamp_effective"],
        errors="coerce",
    ).dropna().sort_values()

    if len(ts) >= 2:
        delta = (
            ts.diff()
            .dt.total_seconds()
            .dropna()
        )

        # Escludi salti tra giorni/stagioni/anni molto grandi.
        delta = delta[
            (delta > 0)
            & (delta <= 6 * 3600)
        ]

        if len(delta):
            med = float(delta.median())
            expected = round(86400 / med)

            if (
                expected >= 1
                and expected <= 288
                and abs(expected * med - 86400) <= max(1.0, med * 0.02)
            ):
                return int(round(med)), int(expected), "MEDIAN_TIMESTAMP_DELTA"

    return None, None, "UNRESOLVED"


def effective_timestamp(df, provider):
    if provider == "ARPA_PIEMONTE":
        # I file giornalieri hanno date_source.
        return pd.to_datetime(
            df["date_source"],
            errors="coerce",
        )

    if provider == "CENTRO_FUNZIONALE_RAVDA":
        # Freeze v1.4: timestamp_source è UTC.
        return pd.to_datetime(
            df["timestamp_source"],
            errors="coerce",
        )

    if provider == "ARPAL":
        # timestamp_utc già materializzato, ma fallback prudente.
        if "timestamp_utc" in df.columns:
            ts = pd.to_datetime(
                df["timestamp_utc"],
                errors="coerce",
                utc=True,
            )
            # Rimuoviamo il tz object solo per raggruppare per giorno UTC.
            return ts.dt.tz_convert("UTC").dt.tz_localize(None)

        return pd.to_datetime(
            df["timestamp_source"],
            errors="coerce",
        )

    raise ValueError(f"Provider inatteso: {provider}")


def primary_policy(code):
    if code in SUM_CODES:
        return "SUM"
    if code in MIN_PRIMARY_CODES:
        return "MIN"
    if code in MAX_PRIMARY_CODES:
        return "MAX"
    if code in CIRCULAR_CODES:
        return "CIRCULAR_MEAN"
    return "MEAN"


def build_daily_for_group(
    g,
    *,
    provider,
    source_series_id,
    station_id,
    target,
    receptor_ids_source,
    variable_code,
    source_column,
    unit_source,
    unit_canonical,
    time_resolution,
    time_basis_canonical,
    effective_timezone,
):
    g = g.copy()

    cadence_s, expected_count, cadence_method = infer_nominal_cadence_seconds(
        g,
        provider,
        time_resolution,
    )

    if expected_count is None:
        raise ValueError(
            f"Cadence unresolved: {source_series_id} / "
            f"{variable_code} / {source_column}"
        )

    vals = pd.to_numeric(
        g["value_numeric"],
        errors="coerce",
    )

    mask = (
        g["_timestamp_effective"].notna()
        & vals.notna()
    )

    g = g.loc[mask].copy()
    g["_value"] = vals.loc[mask].astype(float)

    if g.empty:
        raise ValueError(
            f"Nessun valore numerico: {source_series_id} / {source_column}"
        )

    g["_date_model"] = (
        g["_timestamp_effective"]
        .dt.strftime("%Y-%m-%d")
    )

    rows = []
    policy = primary_policy(variable_code)

    for date_model, d in g.groupby("_date_model", sort=True):
        arr = d["_value"].to_numpy(dtype=float)
        count = int(len(arr))

        coverage = count / expected_count
        if count > expected_count:
            completeness = "OVERCOMPLETE"
        elif count == expected_count:
            completeness = "COMPLETE"
        else:
            completeness = "PARTIAL"

        mean_v = float(np.mean(arr))
        min_v = float(np.min(arr))
        max_v = float(np.max(arr))
        sum_v = float(np.sum(arr))

        circ_strength = np.nan

        if policy == "SUM":
            primary = sum_v
        elif policy == "MIN":
            primary = min_v
        elif policy == "MAX":
            primary = max_v
        elif policy == "CIRCULAR_MEAN":
            primary, circ_strength = circular_mean_deg(arr)
            # Min/max lineari non sono semanticamente utili per direzione.
            min_v = np.nan
            max_v = np.nan
            mean_v = np.nan
            sum_v = np.nan
        else:
            primary = mean_v

        year = int(date_model[:4])
        month = int(date_model[5:7])

        rows.append({
            "source_series_id": source_series_id,
            "provider": provider,
            "station_id": station_id,
            "target": target,
            "receptor_ids_source": receptor_ids_source,
            "variable_code": variable_code,
            "source_column": source_column,
            "unit_source": unit_source,
            "unit_canonical": unit_canonical,
            "date_model": date_model,
            "year": year,
            "month": month,
            "time_basis_canonical": time_basis_canonical,
            "effective_timezone": effective_timezone,
            "time_resolution_source": time_resolution,
            "nominal_cadence_seconds": cadence_s,
            "expected_count_per_day": expected_count,
            "observed_count": count,
            "coverage_fraction": float(coverage),
            "day_completeness": completeness,
            "daily_primary_statistic": policy,
            "daily_value": primary,
            "daily_mean": mean_v,
            "daily_min": min_v,
            "daily_max": max_v,
            "daily_sum": sum_v,
            "circular_resultant_strength": circ_strength,
            "cadence_method": cadence_method,
        })

    return pd.DataFrame(rows)


def main():
    root = Path(__file__).resolve().parent

    values_root = root / "nw_observations_values_v1_0"
    std_root = root / "nw_observations_standardized_v1_4"

    manifest_p = (
        values_root / "standardized_series_manifest_v1_2.csv"
    )
    values_audit_p = (
        values_root / "standardized_values_audit_v1_2.json"
    )
    registry_p = (
        std_root / "observation_series_registry_v1_4.csv"
    )
    time_audit_p = (
        std_root / "observation_time_audit_v1_0.json"
    )
    overlay_p = (
        std_root / "values_manifest_time_overlay_v1_0.csv"
    )

    out_root = root / "nw_observations_daily_v1_0"
    series_root = out_root / "series"
    meta_root = out_root / "metadata"
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 140)
    print("NW OBSERVATIONS — DAILY STATION LAYER BUILDER v1.0")
    print("=" * 140)

    for p in [
        manifest_p,
        values_audit_p,
        registry_p,
        time_audit_p,
        overlay_p,
    ]:
        if not p.exists():
            raise SystemExit(f"Manca: {p}")

    values_audit = json.loads(
        values_audit_p.read_text(encoding="utf-8")
    )
    time_audit = json.loads(
        time_audit_p.read_text(encoding="utf-8")
    )

    if values_audit.get("overall_status") != "PASS":
        raise SystemExit("Standardized values v1.2 non PASS.")
    if time_audit.get("overall_status") != "PASS":
        raise SystemExit("Time freeze v1.0 non PASS.")

    manifest = pd.read_csv(
        manifest_p,
        low_memory=False,
    )
    registry = pd.read_csv(
        registry_p,
        low_memory=False,
    )
    overlay = pd.read_csv(
        overlay_p,
        low_memory=False,
    )

    if len(manifest) != EXPECTED_SERIES:
        raise SystemExit(
            f"Manifest rows={len(manifest)}, atteso={EXPECTED_SERIES}"
        )

    if not manifest["status"].astype(str).str.upper().eq("PASS").all():
        raise SystemExit("Manifest v1.2 contiene serie non PASS.")

    reg_lookup = registry.set_index("source_series_id", drop=False)
    overlay_lookup = overlay.set_index("source_series_id", drop=False)

    records = []
    processed = 0
    skipped = 0
    errors = 0

    for seq, (_, m) in enumerate(manifest.iterrows(), 1):
        sid = str(m["source_series_id"])
        provider = str(m["provider"])

        if sid not in reg_lookup.index:
            raise SystemExit(f"Registry v1.4: manca {sid}")
        if sid not in overlay_lookup.index:
            raise SystemExit(f"Time overlay: manca {sid}")

        rr = reg_lookup.loc[sid]
        if isinstance(rr, pd.DataFrame):
            rr = rr.iloc[0]

        oo = overlay_lookup.loc[sid]
        if isinstance(oo, pd.DataFrame):
            oo = oo.iloc[0]

        source_path = Path(str(m["output_path"]))
        fp = fingerprint(source_path)

        out_name = safe_name(sid) + ".daily.csv.gz"
        out_path = series_root / safe_name(provider) / out_name
        meta_path = (
            meta_root
            / safe_name(provider)
            / (out_name + ".meta.json")
        )

        # Restart-safe.
        if out_path.exists() and meta_path.exists():
            try:
                old = json.loads(meta_path.read_text(encoding="utf-8"))
                if (
                    old.get("status") == "PASS"
                    and old.get("source_size_bytes") == fp["source_size_bytes"]
                    and old.get("source_mtime_ns") == fp["source_mtime_ns"]
                    and old.get("time_freeze_version") == "v1.0"
                ):
                    records.append(old)
                    skipped += 1

                    if seq % 100 == 0:
                        print(
                            f"{seq}/{EXPECTED_SERIES} | "
                            f"processed={processed} skipped={skipped} errors={errors}"
                        )
                    continue
            except Exception:
                pass

        try:
            df = pd.read_csv(
                source_path,
                low_memory=False,
            )

            required = {
                "variable_code",
                "source_column",
                "value_numeric",
                "unit_source",
                "unit_canonical",
                "time_resolution",
            }
            missing = sorted(required - set(df.columns))
            if missing:
                raise ValueError(
                    f"Colonne standardizzate mancanti: {missing}"
                )

            df["_timestamp_effective"] = effective_timestamp(
                df,
                provider,
            )

            bad_ts = int(df["_timestamp_effective"].isna().sum())
            if bad_ts:
                raise ValueError(
                    f"Timestamp effective non parsabili: {bad_ts}"
                )

            # Sep-Dec only invariant.
            if not df["_timestamp_effective"].dt.month.isin(TARGET_MONTHS).all():
                raise ValueError("Righe fuori settembre-dicembre")

            parts = []

            group_cols = [
                "variable_code",
                "source_column",
                "unit_source",
                "unit_canonical",
                "time_resolution",
            ]

            for keys, g in df.groupby(
                group_cols,
                dropna=False,
                sort=False,
            ):
                (
                    variable_code,
                    source_column,
                    unit_source,
                    unit_canonical,
                    time_resolution,
                ) = keys

                part = build_daily_for_group(
                    g,
                    provider=provider,
                    source_series_id=sid,
                    station_id=str(rr["station_id"]),
                    target=str(rr.get("target", "") or ""),
                    receptor_ids_source=str(
                        rr.get("receptor_ids_source", "") or ""
                    ),
                    variable_code=str(variable_code),
                    source_column=str(source_column),
                    unit_source=str(unit_source),
                    unit_canonical=str(unit_canonical),
                    time_resolution=str(time_resolution),
                    time_basis_canonical=str(
                        rr["time_basis_canonical"]
                    ),
                    effective_timezone=str(
                        rr["effective_timezone"]
                    ),
                )

                parts.append(part)

            daily = pd.concat(
                parts,
                ignore_index=True,
            )

            # Key uniqueness.
            key_cols = [
                "source_series_id",
                "variable_code",
                "source_column",
                "date_model",
            ]
            dup = int(daily.duplicated(key_cols).sum())
            if dup:
                raise ValueError(
                    f"Duplicate daily keys={dup}"
                )

            if daily["daily_value"].isna().any():
                # Circular mean can be undefined only for exact resultant zero.
                bad = daily[
                    daily["daily_value"].isna()
                    & ~daily["variable_code"].eq("WIND_DIR_DEG")
                ]
                if len(bad):
                    raise ValueError(
                        f"NaN daily_value non-circular={len(bad)}"
                    )

            atomic_write_csv_gz(
                daily,
                out_path,
            )

            comp_counts = {
                str(k): int(v)
                for k, v in daily["day_completeness"]
                .value_counts()
                .to_dict()
                .items()
            }

            variable_counts = {
                str(k): int(v)
                for k, v in daily["variable_code"]
                .value_counts()
                .to_dict()
                .items()
            }

            cadence_values = sorted(
                {
                    int(x)
                    for x in pd.to_numeric(
                        daily["nominal_cadence_seconds"],
                        errors="coerce",
                    ).dropna().unique()
                }
            )

            meta = {
                "status": "PASS",
                "provider": provider,
                "source_series_id": sid,
                "station_id": str(rr["station_id"]),
                "target": str(rr.get("target", "") or ""),
                "source_standardized_path": str(source_path),
                "output_path": str(out_path),
                "daily_rows": int(len(daily)),
                "date_min": str(daily["date_model"].min()),
                "date_max": str(daily["date_model"].max()),
                "variable_counts": variable_counts,
                "completeness_counts": comp_counts,
                "nominal_cadence_seconds": cadence_values,
                "time_basis_canonical": str(
                    rr["time_basis_canonical"]
                ),
                "effective_timezone": str(
                    rr["effective_timezone"]
                ),
                "source_size_bytes": fp["source_size_bytes"],
                "source_mtime_ns": fp["source_mtime_ns"],
                "time_freeze_version": "v1.0",
                "builder_version": "1.0",
            }

            atomic_write_json(
                meta,
                meta_path,
            )

            records.append(meta)
            processed += 1

        except Exception as exc:
            errors += 1

            meta = {
                "status": "ERROR",
                "provider": provider,
                "source_series_id": sid,
                "station_id": str(rr["station_id"]),
                "target": str(rr.get("target", "") or ""),
                "source_standardized_path": str(source_path),
                "output_path": str(out_path),
                "daily_rows": 0,
                "date_min": "",
                "date_max": "",
                "variable_counts": {},
                "completeness_counts": {},
                "nominal_cadence_seconds": [],
                "time_basis_canonical": str(
                    rr["time_basis_canonical"]
                ),
                "effective_timezone": str(
                    rr["effective_timezone"]
                ),
                "source_size_bytes": fp["source_size_bytes"],
                "source_mtime_ns": fp["source_mtime_ns"],
                "time_freeze_version": "v1.0",
                "builder_version": "1.0",
                "error": repr(exc),
            }

            atomic_write_json(
                meta,
                meta_path,
            )
            records.append(meta)

        if seq % 100 == 0:
            print(
                f"{seq}/{EXPECTED_SERIES} | "
                f"processed={processed} skipped={skipped} errors={errors}"
            )

    man = pd.DataFrame(records)

    pass_df = man[man["status"].eq("PASS")].copy()
    err_df = man[~man["status"].eq("PASS")].copy()

    reasons = []

    if len(man) != EXPECTED_SERIES:
        reasons.append(
            f"MANIFEST_ROWS={len(man)} expected={EXPECTED_SERIES}"
        )

    if len(pass_df) != EXPECTED_SERIES:
        reasons.append(
            f"PASS_SERIES={len(pass_df)} expected={EXPECTED_SERIES}"
        )

    if len(err_df):
        reasons.append(
            f"ERROR_SERIES={len(err_df)}"
        )

    provider_counts = (
        pass_df["provider"].value_counts().to_dict()
    )

    expected_provider = {
        "ARPA_PIEMONTE": 397,
        "CENTRO_FUNZIONALE_RAVDA": 504,
        "ARPAL": 411,
    }

    for p, expected in expected_provider.items():
        got = int(provider_counts.get(p, 0))
        if got != expected:
            reasons.append(
                f"{p}_PASS={got} expected={expected}"
            )

    missing_outputs = int(
        (
            ~pass_df["output_path"]
            .map(lambda s: Path(str(s)).exists())
        ).sum()
    )
    if missing_outputs:
        reasons.append(
            f"MISSING_OUTPUT_FILES={missing_outputs}"
        )

    # Build value summary reading only the daily outputs (small compared with raw).
    summary_parts = []

    for _, r in pass_df.iterrows():
        d = pd.read_csv(
            Path(str(r["output_path"])),
            usecols=[
                "provider",
                "variable_code",
                "day_completeness",
                "coverage_fraction",
                "date_model",
            ],
            low_memory=False,
        )

        s = (
            d.groupby(
                ["provider", "variable_code", "day_completeness"],
                dropna=False,
            )
            .agg(
                daily_rows=("date_model", "size"),
                mean_coverage=("coverage_fraction", "mean"),
            )
            .reset_index()
        )

        s["series_contribution"] = 1
        summary_parts.append(s)

    summary_raw = pd.concat(
        summary_parts,
        ignore_index=True,
    )

    summary = (
        summary_raw.groupby(
            ["provider", "variable_code", "day_completeness"],
            dropna=False,
        )
        .agg(
            daily_rows=("daily_rows", "sum"),
            mean_coverage=("mean_coverage", "mean"),
        )
        .reset_index()
        .sort_values(
            ["provider", "variable_code", "day_completeness"]
        )
    )

    manifest_out = (
        out_root / "daily_series_manifest_v1_0.csv"
    )
    summary_out = (
        out_root / "daily_value_summary_v1_0.csv"
    )

    man.to_csv(
        manifest_out,
        index=False,
    )
    summary.to_csv(
        summary_out,
        index=False,
    )

    total_daily_rows = int(
        pd.to_numeric(
            pass_df["daily_rows"],
            errors="coerce",
        ).fillna(0).sum()
    )

    overall = "PASS" if not reasons else "REVIEW"

    report = {
        "version": "1.0",
        "overall_status": overall,
        "expected_series": EXPECTED_SERIES,
        "manifest_rows": int(len(man)),
        "pass_series": int(len(pass_df)),
        "error_series": int(len(err_df)),
        "processed_this_run": processed,
        "restart_safe_skipped": skipped,
        "total_daily_rows": total_daily_rows,
        "provider_pass_counts": {
            str(k): int(v)
            for k, v in provider_counts.items()
        },
        "missing_output_files": missing_outputs,
        "aggregation_policy": {
            "SUM": sorted(SUM_CODES),
            "MIN": sorted(MIN_PRIMARY_CODES),
            "MAX": sorted(MAX_PRIMARY_CODES),
            "CIRCULAR_MEAN": sorted(CIRCULAR_CODES),
            "MEAN": "all remaining scalar variables",
        },
        "important_policy": (
            "Hydrological stations/sections are NOT spatially averaged here. "
            "Each source_series_id/source_column remains separate."
        ),
        "incomplete_days_dropped": False,
        "imputation": False,
        "raw_modified": False,
        "reasons": reasons,
        "next_step": (
            "Use daily layer plus station_receptor_relations_v1_4 to build "
            "meteorological basin predictors and separately define/select "
            "hydrological target/control series per receptor."
        ),
    }

    atomic_write_json(
        report,
        out_root / "daily_station_layer_audit_v1_0.json",
    )

    lines = [
        "=" * 140,
        "NW OBSERVATIONS — DAILY STATION LAYER BUILDER v1.0",
        "=" * 140,
        f"OVERALL STATUS          : {overall}",
        f"Expected series         : {EXPECTED_SERIES}",
        f"Manifest rows           : {len(man)}",
        f"PASS series             : {len(pass_df)}",
        f"ERROR series            : {len(err_df)}",
        f"Processed this run      : {processed}",
        f"Restart-safe skipped    : {skipped}",
        f"Total daily rows        : {total_daily_rows}",
        f"Missing output files    : {missing_outputs}",
        "",
        "PASS SERIES BY PROVIDER",
        f"ARPA_PIEMONTE           : {int(provider_counts.get('ARPA_PIEMONTE', 0))}",
        f"CENTRO_FUNZIONALE_RAVDA: {int(provider_counts.get('CENTRO_FUNZIONALE_RAVDA', 0))}",
        f"ARPAL                   : {int(provider_counts.get('ARPAL', 0))}",
        "",
        "DAILY VALUE SUMMARY",
        summary.to_string(index=False),
        "",
        "POLICY",
        "Incomplete days are preserved and flagged.",
        "No spatial averaging of river stage/discharge is performed.",
        "VdA and ARPAL subdaily values are grouped by UTC day.",
        "Piemonte daily values preserve source date.",
        "",
        f"Manifest: {manifest_out}",
        f"Summary : {summary_out}",
    ]

    (
        out_root / "daily_station_layer_audit_v1_0.txt"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 140)
    print("\n".join(lines[3:]))
    print("\n" + "=" * 140)
    print(f"OVERALL STATUS : {overall}")
    print(f"Output         : {out_root}")

    if reasons:
        print("REASONS:")
        for r in reasons:
            print(f"  - {r}")

    print("=" * 140)

    if overall != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
