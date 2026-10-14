#!/usr/bin/env python3
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _worker(kwargs: dict[str, Any], result_queue: Any) -> None:
    """Execute one Copernicus Marine subset call in an isolated process."""
    try:
        import copernicusmarine

        result = copernicusmarine.subset(**kwargs)
        result_path = getattr(result, "file_path", None)
        result_queue.put(
            {
                "ok": True,
                "result_path": str(result_path) if result_path else "",
            }
        )
    except BaseException:
        result_queue.put(
            {
                "ok": False,
                "error": traceback.format_exc()[-6000:],
            }
        )


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(
            f"CMEMS_RESILIENT_CONFIG_WARNING | {name}={raw!r} non valido; uso {default}.",
            flush=True,
        )
        return default
    return max(1, value)


def _candidate_paths(kwargs: dict[str, Any], child_result_path: str) -> list[Path]:
    paths: list[Path] = []
    if child_result_path:
        paths.append(Path(child_result_path))

    output_directory = kwargs.get("output_directory")
    output_filename = kwargs.get("output_filename")
    if output_directory and output_filename:
        target = Path(str(output_directory)) / str(output_filename)
        if target not in paths:
            paths.append(target)
    return paths


def _remove_partial_target(kwargs: dict[str, Any]) -> None:
    output_directory = kwargs.get("output_directory")
    output_filename = kwargs.get("output_filename")
    if not output_directory or not output_filename:
        return

    target = Path(str(output_directory)) / str(output_filename)
    try:
        if target.is_file():
            target.unlink()
    except OSError as exc:
        print(
            f"CMEMS_RESILIENT_CLEANUP_WARNING | target={target} | "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )


def robust_subset(**kwargs: Any) -> SimpleNamespace:
    """
    Run the exact requested Copernicus Marine subset with bounded retries.

    Scientific semantics are unchanged: the dataset, variable, date, domain,
    depths and output name are forwarded untouched. No stale-day fallback,
    interpolation or zero filling is permitted.
    """
    attempts = _positive_int_env("NWF_CMEMS_MAX_ATTEMPTS", 4)
    timeout_seconds = _positive_int_env(
        "NWF_CMEMS_ATTEMPT_TIMEOUT_SECONDS",
        300,
    )
    retry_delays = [30, 90, 180]

    output_directory = kwargs.get("output_directory")
    output_filename = kwargs.get("output_filename")
    target = (
        Path(str(output_directory)) / str(output_filename)
        if output_directory and output_filename
        else None
    )

    errors: list[str] = []

    for attempt in range(1, attempts + 1):
        _remove_partial_target(kwargs)
        print(
            f"CMEMS_RESILIENT_ATTEMPT | {attempt}/{attempts} | "
            f"timeout={timeout_seconds}s | target={target or 'unspecified'}",
            flush=True,
        )

        try:
            context = mp.get_context("fork")
        except ValueError:
            context = mp.get_context("spawn")

        result_queue = context.Queue(maxsize=1)
        process = context.Process(
            target=_worker,
            args=(dict(kwargs), result_queue),
            daemon=False,
        )
        started = time.monotonic()
        process.start()
        process.join(timeout_seconds)
        elapsed = time.monotonic() - started

        if process.is_alive():
            process.terminate()
            process.join(10)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(5)
            error = (
                f"TimeoutError: copernicusmarine.subset exceeded "
                f"{timeout_seconds}s"
            )
            child_payload: dict[str, Any] = {"ok": False, "error": error}
        else:
            try:
                child_payload = result_queue.get(timeout=3)
            except queue.Empty:
                child_payload = {
                    "ok": False,
                    "error": (
                        "RuntimeError: Copernicus Marine worker exited "
                        f"without a result (exitcode={process.exitcode})"
                    ),
                }

        result_queue.close()
        result_queue.join_thread()

        child_result_path = str(child_payload.get("result_path", ""))
        selected = next(
            (
                path
                for path in _candidate_paths(kwargs, child_result_path)
                if path.is_file() and path.stat().st_size > 0
            ),
            None,
        )

        if child_payload.get("ok") and selected is not None:
            print(
                f"CMEMS_RESILIENT_PASS | attempt={attempt}/{attempts} | "
                f"elapsed={elapsed:.1f}s | bytes={selected.stat().st_size} | "
                f"file={selected}",
                flush=True,
            )
            return SimpleNamespace(file_path=str(selected))

        error_text = str(
            child_payload.get(
                "error",
                "RuntimeError: subset completed without a non-empty output file",
            )
        ).replace("\x00", "")
        compact = " ".join(error_text.split())[-1600:]
        errors.append(f"attempt {attempt}: {compact}")
        print(
            f"CMEMS_RESILIENT_FAIL | attempt={attempt}/{attempts} | "
            f"elapsed={elapsed:.1f}s | {compact}",
            flush=True,
        )

        auth_failure = any(
            marker in compact
            for marker in (
                "InvalidUsernameOrPassword",
                "401 Unauthorized",
                "403 Forbidden",
            )
        )
        if auth_failure:
            print(
                "CMEMS_RESILIENT_ABORT | errore di autenticazione non transitorio.",
                flush=True,
            )
            break

        if attempt < attempts:
            delay = retry_delays[min(attempt - 1, len(retry_delays) - 1)]
            print(
                f"CMEMS_RESILIENT_WAIT | {delay}s prima del nuovo tentativo.",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError(
        "Copernicus Marine subset failed after bounded retries. "
        "No alternative date or synthetic value was used. "
        + " | ".join(errors)
    )
