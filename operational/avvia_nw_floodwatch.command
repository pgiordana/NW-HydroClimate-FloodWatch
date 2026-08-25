#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  echo "Ambiente Python non trovato. Eseguire prima setup_nw_floodwatch.command"
  read -n 1 -s -r -p "Premi un tasto per chiudere..."
  exit 1
fi
source .venv/bin/activate
python nw_flood_watch.py --open
echo
read -n 1 -s -r -p "Premi un tasto per chiudere..."
echo
