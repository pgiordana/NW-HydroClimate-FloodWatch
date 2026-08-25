# NW HydroClimate / NW FloodWatch

Experimental probabilistic flood forecasting research pipeline for North-Western Italy
(Piemonte, Valle d'Aosta and Liguria), integrating Mediterranean thermal state,
ERA5/ECMWF atmospheric fields, regional hydrometeorological observations and
basin-scale descriptors.

## Important notice

This is a **personal, experimental university-student project by Paolo Giordana**.
It is **not** an official warning system, is not certified for institutional or
safety-critical use, and does not replace ARPA, regional Functional Centres,
Civil Protection authorities, dam operators or other competent bodies.

NW FloodWatch probabilities refer to the experimental model target: exceedance
of a receptor-specific statistical hydrological Q95 threshold within 24, 48 or
72 hours. Experimental colours/signals are not official alert levels.

The experimental predictive scope is **1 September–31 December**. Outside this
season, software runs are technical only and must not be interpreted as
scientifically valid flood predictions.

## Scientific status

- Historical period: September–December 1987–2025.
- 21 geographical receptors; 20 supervised hydrological targets.
- CORE master: 263,520 rows, 97 predictors.
- Frozen models: HistGradientBoosting, 24/48/72 h.
- Final independent holdout: 2023–2025.
- No further tuning is allowed on the 2023–2025 final holdout while retaining
  its status as an independent test.
- Operational IFS/CMEMS substitutions remain experimental and require
  prospective shadow verification.

## Repository structure

- `acquisition/` — canonical data acquisition and source-specific audit scripts.
- `observations/` — interregional observation standardisation and mapping.
- `features/` — ERA5, MedSea×IVT and static/dynamic feature engineering.
- `targets/` — hydrological target network, Q95 labels and thresholds.
- `modeling/` — CORE master, validation-only benchmark, freeze/calibration and final holdout.
- `operational/` — NW FloodWatch release-candidate runtime, macOS/Windows launchers and gate.
- `tanaro_arroscia/` — separate Tanaro–Arroscia research branch.
- `docs/` — technical and methodological documentation.
- `metadata/` — small frozen-model and predictor metadata.

Only the canonical/final scripts are intentionally published. Prototype,
superseded and failed diagnostic versions are excluded.

## Downloadable application

The complete portable runtime package is distributed as a GitHub Release asset:

`NW_FloodWatch_Mac_Windows.zip`

The large historical runtime database is not duplicated in normal Git history.

## Installation

See the manual in `docs/` and the README contained in the portable package.

## Author

**Paolo Giordana**  
University student / independent experimental research  
Email: **paolo@giordana.me**

## Disclaimer

Outputs are complementary research information only. They must not be used as
the sole basis for operational, civil-protection or hydraulic-infrastructure
decisions. Decisions remain the responsibility of competent authorised actors
using official data, procedures and independent professional assessment.
