#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

OLD = "  $('seasonNotice').hidden = Boolean(data.in_core_season && interpretable);"
NEW = """  const seasonNotice = $('seasonNotice');
  if (!data.in_core_season) {
    seasonNotice.hidden = false;
    seasonNotice.innerHTML = '<strong>Fuori dal periodo di validità sperimentale.</strong> I colori e i valori numerici di questo run non devono essere interpretati come previsione di piena.';
  } else if (!interpretable) {
    seasonNotice.hidden = false;
    seasonNotice.innerHTML = '<strong>Run nel periodo di validità, ma bloccato dal gate operativo.</strong> Mancano uno o più requisiti di qualità o completezza dei dati: questo run non deve essere interpretato come previsione di piena.';
  } else {
    seasonNotice.hidden = true;
  }"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-dir", required=True)
    args = parser.parse_args()

    app = Path(args.site_dir).resolve() / "app.js"
    text = app.read_text(encoding="utf-8")

    if NEW in text:
        print("SITE NOTICE PATCH: already applied", flush=True)
        return 0
    if OLD not in text:
        raise RuntimeError("SITE NOTICE PATCH: expected app.js statement not found")

    app.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print("SITE NOTICE PATCH: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
