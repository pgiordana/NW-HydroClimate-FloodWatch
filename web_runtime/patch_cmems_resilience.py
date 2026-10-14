#!/usr/bin/env python3
from __future__ import annotations

import argparse
import py_compile
import shutil
from pathlib import Path


TARGETS = (
    "build_nw_operational_raw_cache_current_v1_1.py",
    "build_nw_operational_medsea_corridor_current_v1_1.py",
)
IMPORT_LINE = (
    "from cmems_resilient_subset import robust_subset as _nwf_cmems_subset\n"
)
ORIGINAL_CALL = "copernicusmarine.subset("
PATCHED_CALL = "_nwf_cmems_subset("


def patch_runtime_script(path: Path) -> int:
    if not path.is_file():
        raise RuntimeError(f"Runtime script missing: {path}")

    source = path.read_text(encoding="utf-8")
    original_count = source.count(ORIGINAL_CALL)
    patched_count = source.count(PATCHED_CALL)

    if IMPORT_LINE.strip() not in source:
        anchor = "from __future__ import annotations\n"
        if anchor not in source:
            raise RuntimeError(f"Future-import anchor missing in {path.name}")
        source = source.replace(anchor, anchor + "\n" + IMPORT_LINE, 1)

    if original_count:
        source = source.replace(ORIGINAL_CALL, PATCHED_CALL)
    elif patched_count == 0:
        raise RuntimeError(
            f"No Copernicus Marine subset call found in {path.name}"
        )

    path.write_text(source, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)

    total_patched = source.count(PATCHED_CALL)
    print(
        f"CMEMS_RUNTIME_PATCH | file={path.name} | "
        f"replaced={original_count} | patched_calls={total_patched} | COMPILE_PASS",
        flush=True,
    )
    return total_patched


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inject bounded retry/timeout handling around the unchanged "
            "Copernicus Marine subset requests in the extracted runtime."
        )
    )
    parser.add_argument("--runtime-root", required=True)
    args = parser.parse_args()

    runtime_root = Path(args.runtime_root).resolve()
    if not runtime_root.is_dir():
        raise RuntimeError(f"Runtime root missing: {runtime_root}")

    helper_source = Path(__file__).with_name("cmems_resilient_subset.py")
    if not helper_source.is_file():
        raise RuntimeError(f"CMEMS helper missing: {helper_source}")

    helper_target = runtime_root / helper_source.name
    shutil.copy2(helper_source, helper_target)
    py_compile.compile(str(helper_target), doraise=True)
    print(
        f"CMEMS_RUNTIME_HELPER | copied={helper_target} | COMPILE_PASS",
        flush=True,
    )

    patched_calls = sum(
        patch_runtime_script(runtime_root / filename)
        for filename in TARGETS
    )
    if patched_calls < len(TARGETS):
        raise RuntimeError(
            f"Expected at least {len(TARGETS)} patched calls, got {patched_calls}"
        )

    print(
        "CMEMS_RUNTIME_RESILIENCE_PASS | "
        f"scripts={len(TARGETS)} | calls={patched_calls}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
