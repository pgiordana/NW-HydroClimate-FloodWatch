# NW FloodWatch Web Runtime — `web-runtime-v1`

Questo strato è **additivo** e non modifica i file scientifici già presenti su `main`.

## Principio

La sorgente scientifica resta la release pubblica immutabile `v1.0-rc1`:

- asset: `NW_FloodWatch_Mac_Windows.zip`
- SHA256: `c5199726e08db1fecbcf0e71ab147f2e42a754460ef051888e5c7259554694c2`

Il runtime web scarica esattamente quell'asset, verifica l'hash, esegue `nw_flood_watch.py` senza modificarlo e converte solo gli output finali in un payload per un sito statico.

## Componenti

- `bootstrap_runtime.py`: download, verifica SHA256 ed estrazione della release.
- `run_web_pipeline.py`: wrapper Linux/headless, ripristino e salvataggio della cache antecedente.
- `export_web_payload.py`: genera `site/data/latest.json`, copia il PDF e la geometria dei recettori.
- `site/`: frontend statico destinato a Cloudflare Pages.

## Test GitHub Actions

Il workflow `.github/workflows/web-runtime-smoke.yml` è volutamente **manuale**. Non pubblica nulla e non modifica DNS.

Modalità `demo`: verifica download della release, installazione Linux, PDF demo, payload JSON, mappa e artefatto statico.

Modalità `full`: richiede due GitHub Actions secrets:

- `COPERNICUSMARINE_SERVICE_USERNAME`
- `COPERNICUSMARINE_SERVICE_PASSWORD`

## Sequenza prevista

1. smoke test demo su GitHub Actions;
2. configurazione secrets Copernicus Marine;
3. primo run operativo Linux completo;
4. collegamento del sito statico a Cloudflare;
5. attivazione schedulazione giornaliera solo dopo i test;
6. DNS di `nwfloodwatch.it` esclusivamente alla fine.
