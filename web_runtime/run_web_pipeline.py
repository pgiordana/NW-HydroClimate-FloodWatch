#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CACHE_DIRNAME = "nw_operational_daily_feature_cache_v1_0"


def elapsed_text(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}h {minutes:02d}m {sec:02d}s" if hours else f"{minutes:d}m {sec:02d}s"


def copy_tree_overlay(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        q = dst / rel
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
        elif p.is_file():
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)


def phase(label: str, idx: int, total: int, started: float) -> None:
    pct = 100.0 * idx / total
    print(
        f"[WEB PHASE {idx}/{total}] {label} | {pct:5.1f}% | elapsed {elapsed_text(time.time()-started)}",
        flush=True,
    )


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with return code {code}: {' '.join(cmd)}")


def require_full_run_secrets(env: dict[str, str]) -> None:
    missing = [
        name
        for name in (
            "COPERNICUSMARINE_SERVICE_USERNAME",
            "COPERNICUSMARINE_SERVICE_PASSWORD",
        )
        if not env.get(name)
    ]
    if missing:
        raise RuntimeError(
            "Full operational run requires GitHub/host secrets: " + ", ".join(missing)
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Headless Linux/web wrapper around the unchanged NW FloodWatch v1.0-rc1 runtime."
    )
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--site-dir", required=True)
    parser.add_argument("--persistent-cache-dir")
    parser.add_argument("--mode", choices=("demo", "full", "report-only"), default="demo")
    parser.add_argument("--allow-degraded-smoke", action="store_true")
    args = parser.parse_args()

    runtime_root = Path(args.runtime_root).resolve()
    site_dir = Path(args.site_dir).resolve()
    persistent_cache = (
        Path(args.persistent_cache_dir).resolve() if args.persistent_cache_dir else None
    )
    project_root = Path(__file__).resolve().parent
    started = time.time()
    total = 5

    runner = runtime_root / "nw_flood_watch.py"
    if not runner.exists():
        raise RuntimeError(f"NW FloodWatch runtime runner not found: {runner}")

    phase("Runtime integrity/preflight", 1, total, started)
    ready = runtime_root / "DATABASE_RUNTIME_READY.txt"
    if not ready.exists():
        raise RuntimeError(f"Runtime ready marker missing: {ready}")
    print(ready.read_text(encoding="utf-8", errors="replace").strip(), flush=True)

    phase("Restore persistent antecedent cache", 2, total, started)
    runtime_cache = runtime_root / CACHE_DIRNAME
    if persistent_cache and persistent_cache.exists():
        print(f"Overlay persistent cache {persistent_cache} -> {runtime_cache}", flush=True)
        copy_tree_overlay(persistent_cache, runtime_cache)
    else:
        print("No previous external cache found; using cache bundled with v1.0-rc1.", flush=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    phase("Execute unchanged scientific runtime", 3, total, started)
    if args.mode == "demo":
        run([sys.executable, str(runner), "--pdf-demo"], runtime_root, env)
    else:
        require_full_run_secrets(env)
        cmd = [sys.executable, str(runner)]
        if args.mode == "full":
            cmd.append("--force-full")
        elif args.mode == "report-only":
            cmd.append("--report-only")
        if args.allow_degraded_smoke:
            cmd.append("--allow-degraded-smoke")
        run(cmd, runtime_root, env)

    phase("Export static-web payload", 4, total, started)
    export_cmd = [
        sys.executable,
        str(project_root / "export_web_payload.py"),
        "--runtime-root",
        str(runtime_root),
        "--site-dir",
        str(site_dir),
    ]
    if args.mode == "demo":
        export_cmd.append("--demo")
    run(export_cmd, project_root, env)

    phase("Persist mutable antecedent cache", 5, total, started)
    if persistent_cache:
        if persistent_cache.exists():
            shutil.rmtree(persistent_cache)
        if runtime_cache.exists():
            shutil.copytree(runtime_cache, persistent_cache)
            print(f"Persistent cache saved: {persistent_cache}", flush=True)
        else:
            print("Runtime cache not found after run; nothing persisted.", flush=True)

    print(
        f"WEB PIPELINE PASS | mode={args.mode} | total elapsed {elapsed_text(time.time()-started)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"WEB PIPELINE FAIL: {exc}", file=sys.stderr, flush=True)
        raise
