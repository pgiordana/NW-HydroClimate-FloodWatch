#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo
echo "Installazione completata."
echo "Per avviare: doppio clic su avvia_nw_floodwatch.command"
read -n 1 -s -r -p "Premi un tasto per chiudere..."
echo
