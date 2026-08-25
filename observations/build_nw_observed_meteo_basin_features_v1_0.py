#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_nw_observed_meteo_basin_features_v1_0.py

Costruisce le feature meteorologiche OSSERVATE giornaliere per i 21 recettori
NW, a partire dal daily station layer v1.0 e dal mapping canonico v1.3.

IMPORTANTE
----------
- Usa TUTTE le relazioni fisiche eleggibili stazione->recettore, non soltanto
  `canonical_is_primary`. Questo preserva correttamente le appartenenze
  multiple/nidificate.
- `canonical_is_primary` viene mantenuto solo come diagnostica.
- NON aggrega livello/portata: le variabili idrologiche non entrano qui.
- NON imputa dati mancanti.
- NON estende VdA prima del 1996.
- I giorni parziali sono mantenuti, con metriche di copertura.
- Le precipitazioni sono riassunti di RETE (mean/median/max/p90), NON una
  stima areale interpolata del bacino.
- Queste sono feature derivate da osservazioni: non vanno usate in un modello
  previsionale a lead positivo oltre il tempo di emissione senza opportune
  regole causali/lag.

Output:
nw_observed_meteo_basin_features_v1_0/
  observed_meteo_basin_daily_1987_2025_v1_0.csv
  observed_meteo_basin_coverage_summary_v1_0.csv
  observed_meteo_basin_feature_dictionary_v1_0.csv
  observed_meteo_basin_audit_v1_0.json
  observed_meteo_basin_audit_v1_0.txt

Griglia canonica:
21 recettori × tutti i giorni Sep-Dec 1987-2025 = 99,918 righe.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_RECEPTORS = [
    "NW_DORA_RIPARIA",
    "NW_STURA_LANZO",
    "NW_STURA_DEMONTE",
    "NW_TANARO_ALTO",
    "NW_TANARO_MEDIO_BASSO",
    "NW_BORMIDA",
    "NW_ORBA",
    "NW_SCRIVIA",
    "NW_DORA_BALTEA",
    "NW_ORCO",
    "NW_PELLICE",
    "NW_CHISONE",
    "NW_MAIRA",
    "NW_VARAITA",
    "NW_SESIA",
    "NW_TOCE",
    "LIG_BISAGNO",
    "LIG_POLCEVERA",
    "LIG_ENTELLA",
    "LIG_MAGRA",
    "LIG_CENTA",
]

VAR_PREFIX = {
    "PRECIP_MM": "precip_mm",
    "AIR_TEMP_C": "air_temp_c",
    "REL_HUMIDITY_PCT": "rel_humidity_pct",
    "WIND_SPEED_M_S": "wind_speed_m_s",
    "WIND_DIR_DEG": "wind_dir_deg",
    "AIR_PRESSURE_HPA": "air_pressure_hpa",
    "SNOW_DEPTH_CM": "snow_depth_cm",
    "SOLAR_RAD_W_M2": "solar_rad_w_m2",
    "SUNSHINE_DURATION_MIN": "sunshine_duration_min",
}

EXPECTED_ROWS = 99918


def safe_str(v):
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def normalize_bool(v):
    if isinstance(v, bool):
        return v
    return safe_str(v).lower() in {
        "true", "1", "yes", "y", "si", "sì"
    }


def station_key(row):
    provider = safe_str(row.get("provider"))
    station_id = safe_str(row.get("station_id"))
    target = safe_str(row.get("target"))

    # ARPAL registry is year-segmented by target; target is the stable
    # physical/parameter identity used in the mapping.
    if provider == "ARPAL" and target:
        return f"{provider}::{target}"

    if station_id:
        return f"{provider}::{station_id}"

    if target:
        return f"{provider}::{target}"

    return f"{provider}::{safe_str(row.get('source_series_id'))}"


def all_sepdec_dates():
    dates = []
    for year in range(1987, 2026):
        dates.extend(
            pd.date_range(
                f"{year}-09-01",
                f"{year}-12-31",
                freq="D",
            )
        )
    return pd.DatetimeIndex(dates)


def circular_mean_deg(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan

    rad = np.deg2rad(np.mod(arr, 360.0))
    s = float(np.mean(np.sin(rad)))
    c = float(np.mean(np.cos(rad)))
    r = float(np.hypot(s, c))

    if r < 1e-12:
        return np.nan, r

    angle = math.degrees(math.atan2(s, c))
    if angle < 0:
        angle += 360.0

    return float(angle), r


def circular_weighted_mean_deg(values, weights):
    arr = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)

    mask = np.isfinite(arr) & np.isfinite(w) & (w > 0)
    arr = arr[mask]
    w = w[mask]

    if len(arr) == 0 or float(w.sum()) <= 0:
        return np.nan, np.nan

    rad = np.deg2rad(np.mod(arr, 360.0))
    s = float(np.average(np.sin(rad), weights=w))
    c = float(np.average(np.cos(rad), weights=w))
    r = float(np.hypot(s, c))

    if r < 1e-12:
        return np.nan, r

    angle = math.degrees(math.atan2(s, c))
    if angle < 0:
        angle += 360.0

    return float(angle), r


def load_candidate_daily(candidate):
    path = Path(safe_str(candidate["daily_output_path"]))
    if not path.exists():
        raise FileNotFoundError(path)

    usecols = [
        "variable_code",
        "source_column",
        "date_model",
        "daily_value",
        "coverage_fraction",
        "day_completeness",
        "circular_resultant_strength",
    ]

    df = pd.read_csv(
        path,
        usecols=usecols,
        low_memory=False,
    )

    code = safe_str(candidate["variable_code"])
    df = df[df["variable_code"].astype(str).eq(code)].copy()

    if df.empty:
        return df

    df["date_model"] = pd.to_datetime(
        df["date_model"],
        errors="coerce",
    )
    df["daily_value"] = pd.to_numeric(
        df["daily_value"],
        errors="coerce",
    )
    df["coverage_fraction"] = pd.to_numeric(
        df["coverage_fraction"],
        errors="coerce",
    )
    df["circular_resultant_strength"] = pd.to_numeric(
        df["circular_resultant_strength"],
        errors="coerce",
    )

    df = df[
        df["date_model"].notna()
        & df["daily_value"].notna()
    ].copy()

    return df


def dynamic_eligible_counts(candidates, date_index):
    """
    Per date, conta le stazioni/target con almeno una serie DATA_OK la cui
    finestra date_min-date_max include quel giorno.

    Per ARPAL, i segmenti annuali DATA_OK rendono eleggibile il target solo
    negli anni effettivamente presenti; i NO_DATA_CONFIRMED non vengono
    trasformati in falsi zeri.
    """
    intervals = []

    for _, r in candidates.iterrows():
        d0 = pd.to_datetime(
            safe_str(r.get("date_min")),
            errors="coerce",
        )
        d1 = pd.to_datetime(
            safe_str(r.get("date_max")),
            errors="coerce",
        )

        if pd.isna(d0) or pd.isna(d1):
            continue

        intervals.append(
            (
                station_key(r),
                pd.Timestamp(d0).normalize(),
                pd.Timestamp(d1).normalize(),
            )
        )

    # station -> merged boolean mask on canonical dates
    station_masks = {}

    for skey, d0, d1 in intervals:
        mask = (
            (date_index >= d0)
            & (date_index <= d1)
        )

        if skey in station_masks:
            station_masks[skey] |= mask
        else:
            station_masks[skey] = mask.copy()

    if not station_masks:
        return pd.Series(
            np.zeros(len(date_index), dtype=int),
            index=date_index,
        )

    mat = np.vstack(
        [m.astype(np.uint8) for m in station_masks.values()]
    )

    return pd.Series(
        mat.sum(axis=0).astype(int),
        index=date_index,
    )


def aggregate_variable_for_receptor(
    candidates,
    receptor_id,
    variable_code,
    date_index,
):
    pieces = []
    overcomplete_rows = 0

    for _, cand in candidates.iterrows():
        d = load_candidate_daily(cand)

        if d.empty:
            continue

        d["station_key"] = station_key(cand)
        d["source_series_id"] = safe_str(
            cand["source_series_id"]
        )
        d["provider"] = safe_str(cand["provider"])
        d["canonical_is_primary"] = normalize_bool(
            cand["canonical_is_primary"]
        )

        overcomplete_rows += int(
            (
                d["coverage_fraction"] > 1.000001
            ).sum()
        )

        pieces.append(d)

    prefix = VAR_PREFIX[variable_code]
    idx = pd.DataFrame(
        {"date": date_index}
    )

    eligible = dynamic_eligible_counts(
        candidates,
        date_index,
    )

    if not pieces:
        idx[f"obs_{prefix}_eligible_station_count"] = (
            eligible.values
        )
        return idx, {
            "receptor_id": receptor_id,
            "variable_code": variable_code,
            "candidate_series": int(len(candidates)),
            "candidate_stations": int(
                len(
                    {
                        station_key(r)
                        for _, r in candidates.iterrows()
                    }
                )
            ),
            "days_with_values": 0,
            "overcomplete_station_days": overcomplete_rows,
        }

    x = pd.concat(
        pieces,
        ignore_index=True,
    )

    # Duplicate station-date-source_column is not expected. Different
    # source columns are allowed, but same station/date/column twice is not.
    dup = int(
        x.duplicated(
            [
                "station_key",
                "date_model",
                "source_column",
            ]
        ).sum()
    )

    if dup:
        raise ValueError(
            f"{receptor_id}/{variable_code}: "
            f"duplicate station-date-source_column={dup}"
        )

    rows = []

    for date, g in x.groupby(
        "date_model",
        sort=True,
    ):
        values = g["daily_value"].to_numpy(dtype=float)
        cov = (
            pd.to_numeric(
                g["coverage_fraction"],
                errors="coerce",
            )
            .fillna(0.0)
            .clip(lower=0.0, upper=1.0)
            .to_numpy(dtype=float)
        )

        station_count = int(
            g["station_key"].nunique()
        )
        primary_station_count = int(
            g.loc[
                g["canonical_is_primary"],
                "station_key",
            ].nunique()
        )

        complete_fraction = float(
            g["day_completeness"]
            .astype(str)
            .eq("COMPLETE")
            .mean()
        )

        coverage_mean = float(
            np.nanmean(
                pd.to_numeric(
                    g["coverage_fraction"],
                    errors="coerce",
                )
            )
        )
        coverage_min = float(
            np.nanmin(
                pd.to_numeric(
                    g["coverage_fraction"],
                    errors="coerce",
                )
            )
        )

        base = {
            "date": pd.Timestamp(date).normalize(),
            f"obs_{prefix}_station_count": station_count,
            f"obs_{prefix}_primary_station_count":
                primary_station_count,
            f"obs_{prefix}_coverage_mean":
                coverage_mean,
            f"obs_{prefix}_coverage_min":
                coverage_min,
            f"obs_{prefix}_complete_fraction":
                complete_fraction,
        }

        if variable_code == "WIND_DIR_DEG":
            cmean, strength = circular_mean_deg(
                values
            )
            wcmean, wstrength = (
                circular_weighted_mean_deg(
                    values,
                    cov,
                )
            )

            base.update({
                f"obs_{prefix}_cmean":
                    cmean,
                f"obs_{prefix}_resultant_strength":
                    strength,
                f"obs_{prefix}_coverage_weighted_cmean":
                    wcmean,
                f"obs_{prefix}_coverage_weighted_resultant_strength":
                    wstrength,
            })
        else:
            finite = values[np.isfinite(values)]

            if len(finite):
                base.update({
                    f"obs_{prefix}_mean":
                        float(np.mean(finite)),
                    f"obs_{prefix}_median":
                        float(np.median(finite)),
                    f"obs_{prefix}_min":
                        float(np.min(finite)),
                    f"obs_{prefix}_max":
                        float(np.max(finite)),
                    f"obs_{prefix}_p90":
                        float(np.quantile(finite, 0.90)),
                    f"obs_{prefix}_std":
                        float(np.std(finite, ddof=0)),
                })

                wmask = (
                    np.isfinite(values)
                    & np.isfinite(cov)
                    & (cov > 0)
                )

                if wmask.any():
                    base[
                        f"obs_{prefix}_coverage_weighted_mean"
                    ] = float(
                        np.average(
                            values[wmask],
                            weights=cov[wmask],
                        )
                    )
                else:
                    base[
                        f"obs_{prefix}_coverage_weighted_mean"
                    ] = np.nan

        rows.append(base)

    agg = pd.DataFrame(rows)

    idx = idx.merge(
        agg,
        on="date",
        how="left",
    )

    idx[f"obs_{prefix}_eligible_station_count"] = (
        eligible.reindex(date_index).values
    )

    station_col = f"obs_{prefix}_station_count"
    eligible_col = (
        f"obs_{prefix}_eligible_station_count"
    )

    if station_col not in idx.columns:
        idx[station_col] = np.nan

    denom = pd.to_numeric(
        idx[eligible_col],
        errors="coerce",
    )

    numer = pd.to_numeric(
        idx[station_col],
        errors="coerce",
    )

    idx[f"obs_{prefix}_station_fraction"] = np.where(
        denom > 0,
        numer.fillna(0) / denom,
        np.nan,
    )

    diag = {
        "receptor_id": receptor_id,
        "variable_code": variable_code,
        "candidate_series": int(
            candidates["source_series_id"].nunique()
        ),
        "candidate_stations": int(
            len(
                {
                    station_key(r)
                    for _, r in candidates.iterrows()
                }
            )
        ),
        "days_with_values": int(
            x["date_model"].nunique()
        ),
        "date_min": str(
            x["date_model"].min().date()
        ),
        "date_max": str(
            x["date_model"].max().date()
        ),
        "overcomplete_station_days":
            overcomplete_rows,
    }

    return idx, diag


def build_feature_dictionary(columns):
    rows = []

    for c in columns:
        if c in {
            "receptor_id",
            "date",
            "season_year",
            "month",
            "day_of_year",
        }:
            continue

        if not c.startswith("obs_"):
            continue

        if c.endswith("_eligible_station_count"):
            stat = "eligible_station_count"
        elif c.endswith("_primary_station_count"):
            stat = "primary_station_count"
        elif c.endswith("_station_count"):
            stat = "station_count"
        elif c.endswith("_station_fraction"):
            stat = "station_fraction"
        elif c.endswith("_coverage_mean"):
            stat = "coverage_mean"
        elif c.endswith("_coverage_min"):
            stat = "coverage_min"
        elif c.endswith("_complete_fraction"):
            stat = "complete_fraction"
        elif c.endswith("_coverage_weighted_mean"):
            stat = "coverage_weighted_mean"
        elif c.endswith("_coverage_weighted_cmean"):
            stat = "coverage_weighted_circular_mean"
        elif c.endswith("_coverage_weighted_resultant_strength"):
            stat = "coverage_weighted_resultant_strength"
        elif c.endswith("_resultant_strength"):
            stat = "circular_resultant_strength"
        elif c.endswith("_cmean"):
            stat = "circular_mean"
        elif c.endswith("_mean"):
            stat = "mean"
        elif c.endswith("_median"):
            stat = "median"
        elif c.endswith("_min"):
            stat = "min"
        elif c.endswith("_max"):
            stat = "max"
        elif c.endswith("_p90"):
            stat = "p90"
        elif c.endswith("_std"):
            stat = "std"
        else:
            stat = "other"

        rows.append({
            "feature": c,
            "statistic": stat,
            "source": "observed_station_network",
            "spatial_semantics": (
                "network_summary_not_area_interpolation"
            ),
            "causal_use_note": (
                "Observation-derived; enforce issue-time/lag rules "
                "before predictive use."
            ),
        })

    return pd.DataFrame(rows)


def main():
    root = Path(__file__).resolve().parent

    mapping_root = (
        root / "nw_observations_basin_mapping_v1_3"
    )
    daily_root = (
        root / "nw_observations_daily_v1_0"
    )

    audit_mapping_p = (
        mapping_root
        / "basin_observation_mapping_audit_v1_3.json"
    )
    candidates_p = (
        mapping_root
        / "meteorological_candidate_series_v1_3.csv"
    )
    daily_audit_p = (
        daily_root
        / "daily_station_layer_audit_v1_0.json"
    )

    out_root = (
        root / "nw_observed_meteo_basin_features_v1_0"
    )
    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 146)
    print(
        "NW OBSERVATIONS — OBSERVED METEO BASIN FEATURES v1.0"
    )
    print("=" * 146)

    for p in [
        audit_mapping_p,
        candidates_p,
        daily_audit_p,
    ]:
        if not p.exists():
            raise SystemExit(f"Manca: {p}")

    mapping_audit = json.loads(
        audit_mapping_p.read_text(
            encoding="utf-8"
        )
    )
    daily_audit = json.loads(
        daily_audit_p.read_text(
            encoding="utf-8"
        )
    )

    if mapping_audit.get(
        "overall_status"
    ) != "PASS":
        raise SystemExit(
            "Basin mapping v1.3 non PASS."
        )

    if daily_audit.get(
        "overall_status"
    ) != "PASS":
        raise SystemExit(
            "Daily station layer v1.0 non PASS."
        )

    cand = pd.read_csv(
        candidates_p,
        low_memory=False,
    )

    if cand.empty:
        raise SystemExit(
            "Meteorological candidate table vuota."
        )

    unexpected_codes = sorted(
        set(
            cand["variable_code"]
            .dropna()
            .astype(str)
        )
        - set(VAR_PREFIX)
    )

    if unexpected_codes:
        raise SystemExit(
            "Variable codes meteo non gestiti: "
            + "|".join(unexpected_codes)
        )

    date_index = all_sepdec_dates()

    if len(date_index) != 4758:
        raise SystemExit(
            f"Canonical days={len(date_index)}, atteso=4758"
        )

    receptor_frames = []
    diagnostics = []

    for i, receptor_id in enumerate(
        EXPECTED_RECEPTORS,
        1,
    ):
        base = pd.DataFrame({
            "date": date_index
        })

        rec = cand[
            cand["receptor_id"]
            .astype(str)
            .eq(receptor_id)
        ].copy()

        for variable_code in VAR_PREFIX:
            sub = rec[
                rec["variable_code"]
                .astype(str)
                .eq(variable_code)
            ].copy()

            feat, diag = (
                aggregate_variable_for_receptor(
                    sub,
                    receptor_id,
                    variable_code,
                    date_index,
                )
            )

            diagnostics.append(diag)

            base = base.merge(
                feat,
                on="date",
                how="left",
                validate="one_to_one",
            )

        base.insert(
            0,
            "receptor_id",
            receptor_id,
        )
        base["season_year"] = (
            base["date"].dt.year
        )
        base["month"] = (
            base["date"].dt.month
        )
        base["day_of_year"] = (
            base["date"].dt.dayofyear
        )
        base["date"] = (
            base["date"]
            .dt.strftime("%Y-%m-%d")
        )

        receptor_frames.append(base)

        print(
            f"{i:02d}/21 | {receptor_id} | "
            f"candidate_rows={len(rec)}"
        )

    out = pd.concat(
        receptor_frames,
        ignore_index=True,
    )

    reasons = []

    if len(out) != EXPECTED_ROWS:
        reasons.append(
            f"ROWS={len(out)} expected={EXPECTED_ROWS}"
        )

    if out[
        ["receptor_id", "date"]
    ].duplicated().any():
        reasons.append(
            "DUPLICATE_RECEPTOR_DATE_KEYS"
        )

    receptors_seen = sorted(
        out["receptor_id"].unique()
    )

    if receptors_seen != sorted(
        EXPECTED_RECEPTORS
    ):
        reasons.append(
            "RECEPTOR_SET_MISMATCH"
        )

    # Every receptor must have at least one observed precipitation day.
    precip_mean_col = (
        "obs_precip_mm_mean"
    )

    precip_days = (
        out.groupby("receptor_id")[
            precip_mean_col
        ]
        .apply(
            lambda s:
            int(s.notna().sum())
        )
        .to_dict()
    )

    no_precip = [
        r
        for r in EXPECTED_RECEPTORS
        if int(
            precip_days.get(r, 0)
        ) == 0
    ]

    if no_precip:
        reasons.append(
            "RECEPTORS_WITH_ZERO_OBS_PRECIP_DAYS="
            + "|".join(no_precip)
        )

    diag = pd.DataFrame(
        diagnostics
    )

    # Coverage summary.
    coverage_rows = []

    for _, d in diag.iterrows():
        rec = d["receptor_id"]
        code = d["variable_code"]
        prefix = VAR_PREFIX[code]

        sub = out[
            out["receptor_id"].eq(rec)
        ]

        value_col = (
            f"obs_{prefix}_cmean"
            if code == "WIND_DIR_DEG"
            else f"obs_{prefix}_mean"
        )

        if value_col in sub.columns:
            valid_days = int(
                sub[value_col]
                .notna()
                .sum()
            )
        else:
            valid_days = 0

        station_fraction_col = (
            f"obs_{prefix}_station_fraction"
        )

        if (
            station_fraction_col
            in sub.columns
        ):
            mean_station_fraction = float(
                pd.to_numeric(
                    sub[
                        station_fraction_col
                    ],
                    errors="coerce",
                )
                .mean()
            )
        else:
            mean_station_fraction = np.nan

        coverage_rows.append({
            **d.to_dict(),
            "valid_days_on_canonical_grid":
                valid_days,
            "mean_station_fraction_when_defined":
                mean_station_fraction,
        })

    coverage = pd.DataFrame(
        coverage_rows
    )

    overcomplete_total = int(
        pd.to_numeric(
            coverage[
                "overcomplete_station_days"
            ],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    # Overcomplete does not fail automatically: it is surfaced for QC.
    # The upstream daily layer already preserved and flagged these days.

    feature_dict = (
        build_feature_dictionary(
            out.columns
        )
    )

    data_out = (
        out_root
        / "observed_meteo_basin_daily_1987_2025_v1_0.csv"
    )
    coverage_out = (
        out_root
        / "observed_meteo_basin_coverage_summary_v1_0.csv"
    )
    dict_out = (
        out_root
        / "observed_meteo_basin_feature_dictionary_v1_0.csv"
    )

    out.to_csv(
        data_out,
        index=False,
    )
    coverage.to_csv(
        coverage_out,
        index=False,
    )
    feature_dict.to_csv(
        dict_out,
        index=False,
    )

    overall = (
        "PASS"
        if not reasons
        else "REVIEW"
    )

    report = {
        "version": "1.0",
        "overall_status": overall,
        "rows": int(len(out)),
        "expected_rows": EXPECTED_ROWS,
        "receptors": int(
            out["receptor_id"].nunique()
        ),
        "canonical_days_per_receptor":
            4758,
        "date_start": "1987-09-01",
        "date_end": "2025-12-31",
        "months": [9, 10, 11, 12],
        "precip_days_by_receptor": {
            str(k): int(v)
            for k, v
            in precip_days.items()
        },
        "receptors_with_zero_observed_precip_days":
            no_precip,
        "overcomplete_station_days":
            overcomplete_total,
        "spatial_policy": (
            "All eligible physical station-receptor memberships retained. "
            "Statistics are station-network summaries, not area-weighted "
            "interpolation."
        ),
        "hydrology_included": False,
        "imputation": False,
        "missing_days_fabricated": False,
        "causal_use_warning": (
            "Observation-derived features must obey issue-time/lead-time "
            "constraints in predictive modeling. No future observations "
            "may enter predictors."
        ),
        "raw_modified": False,
        "reasons": reasons,
        "next_step": (
            "Freeze/select hydrological target/control series separately, "
            "then define forecast lead/event labels before model training."
        ),
    }

    audit_json = (
        out_root
        / "observed_meteo_basin_audit_v1_0.json"
    )
    audit_txt = (
        out_root
        / "observed_meteo_basin_audit_v1_0.txt"
    )

    audit_json.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    compact = coverage[
        coverage[
            "variable_code"
        ].eq("PRECIP_MM")
    ][
        [
            "receptor_id",
            "candidate_stations",
            "valid_days_on_canonical_grid",
            "mean_station_fraction_when_defined",
        ]
    ].copy()

    lines = [
        "=" * 146,
        "NW OBSERVATIONS — OBSERVED METEO BASIN FEATURES v1.0",
        "=" * 146,
        f"OVERALL STATUS                   : {overall}",
        f"Rows                             : {len(out)} / {EXPECTED_ROWS}",
        f"Receptors                        : {out['receptor_id'].nunique()} / 21",
        f"Days per receptor                : 4758",
        f"Duplicate receptor/date keys     : {int(out[['receptor_id','date']].duplicated().sum())}",
        f"Receptors with zero precip days  : {len(no_precip)}",
        f"Overcomplete station-days        : {overcomplete_total}",
        "",
        "PRECIPITATION NETWORK COVERAGE",
        compact.to_string(index=False),
        "",
        "POLICY",
        "All physical station-receptor memberships retained.",
        "Precipitation features are network summaries, not area interpolation.",
        "No hydrological stage/discharge aggregation is performed.",
        "No imputation.",
        "Observed features require causal lag/issue-time rules before predictive use.",
        "",
        f"Data      : {data_out}",
        f"Coverage  : {coverage_out}",
        f"Dictionary: {dict_out}",
    ]

    audit_txt.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 146)
    print("\n".join(lines[3:]))
    print("\n" + "=" * 146)
    print(
        f"OVERALL STATUS : {overall}"
    )
    print(
        f"Output         : {out_root}"
    )

    if reasons:
        print("REASONS:")
        for r in reasons:
            print(f"  - {r}")

    print("=" * 146)

    if overall != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
