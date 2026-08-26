#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from datetime import datetime, timezone

SOURCES = ("ecmwf", "aws", "google")


def fmt_seconds(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def normalize_dt(value) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def latest_00z():
    from ecmwf.opendata import Client

    errors = []
    for source in SOURCES:
        try:
            client = Client(source=source, model="ifs", resol="0p25")
            latest = client.latest(
                stream="oper",
                type="fc",
                time=0,
                step=3,
                param="msl",
            )
            return source, normalize_dt(latest), errors
        except Exception as exc:
            errors.append(f"{source}: {type(exc).__name__}: {str(exc)[:240]}")
    return None, None, errors


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Wait until the current UTC date ECMWF IFS 00Z cycle is really available. "
            "Fails instead of silently using yesterday's cycle."
        )
    )
    ap.add_argument("--interval-minutes", type=int, default=10)
    ap.add_argument("--max-wait-minutes", type=int, default=90)
    args = ap.parse_args()

    interval = max(1, args.interval_minutes) * 60
    max_wait = max(0, args.max_wait_minutes) * 60
    target_date = datetime.now(timezone.utc).date()
    started = time.time()
    attempt = 0

    print("=" * 100, flush=True)
    print("ECMWF 00Z FRESHNESS GATE", flush=True)
    print(f"Required UTC issue date: {target_date.isoformat()} 00Z", flush=True)
    print(
        f"Retry interval: {args.interval_minutes} min | max wait: {args.max_wait_minutes} min",
        flush=True,
    )
    print("=" * 100, flush=True)

    while True:
        attempt += 1
        elapsed = time.time() - started
        remaining = max(0.0, max_wait - elapsed)
        source, latest, errors = latest_00z()

        if latest is not None:
            print(
                f"Attempt {attempt} | elapsed {fmt_seconds(elapsed)} | "
                f"latest={latest.isoformat()} | source={source}",
                flush=True,
            )
            if latest.date() == target_date and latest.hour == 0:
                print(
                    f"ECMWF_FRESHNESS_PASS | current-day 00Z available via {source}",
                    flush=True,
                )
                return 0
        else:
            print(
                f"Attempt {attempt} | elapsed {fmt_seconds(elapsed)} | no source returned latest 00Z",
                flush=True,
            )
            for err in errors:
                print(f"  {err}", flush=True)

        if elapsed >= max_wait:
            raise RuntimeError(
                "ECMWF_FRESHNESS_FAIL: current-day 00Z not available within the allowed wait window. "
                "Daily production aborted; yesterday's cycle will NOT be published as today's run."
            )

        sleep_for = min(interval, remaining)
        print(
            f"WAIT | next check in {sleep_for/60:.0f} min | "
            f"elapsed {fmt_seconds(elapsed)} | remaining {fmt_seconds(remaining)}",
            flush=True,
        )
        time.sleep(sleep_for)


if __name__ == "__main__":
    raise SystemExit(main())
