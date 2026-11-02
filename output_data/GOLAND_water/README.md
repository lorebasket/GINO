# GOLAND_water study

**Objective:** find a wing whose flutter speed is strongly sensitive to relative
submergence h/c, so that the BEM (free-surface) contribution carries scientific
weight. The target metric is **dV_flutter between depth cases**.

## Layout

| Path | What it holds |
|---|---|
| **`REPORT.md`** | **Leggere questo per primo: report completo di fine sessione 2026-09-30.** |
| `_scripts/` | script di lancio e di analisi riutilizzabili |
| `bem_depth_probe/` | **The one validated result.** A(omega; h) from Capytaine, h/c = 0.25 … 4. |
| `sweeps/` | Multi-run families (depth sweeps, parameter scans). |
| `secondary/` | Individual one-off runs kept for provenance. |
| `secondary/_loose_root_files/` | Stray csv/png/log that old default-named runs dropped at the study root. |

## Status

Vedere **`REPORT.md`** — sostituisce lo stato precedente.

In breve: trovato un caso realizzabile (acciaio, tf=0.40, xea=0.15/xcm=0.75) con
**dV_flutter = 26.9%** fra h/c 0.25 e 2, in `sweeps/tf04_depth_B_soft/`.
Il BEM conta solo se **h/c < 1** E **k_w*h ~ 1**; il flutter richiede **mu\* >~ 2**.
