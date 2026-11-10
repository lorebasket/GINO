# GOLAND_water study

## Stato attuale: report rev H (10/11/2026)

`report/GOLAND_water_sommergenza_LV_revH.pdf` è il report di riferimento, scritto per un lettore esterno. Il flutter di un'ala sommersa vicino alla superficie libera è calcolato con la superficie anche nelle forze circolatorie: un'ala immagine nell'UVLM, cioè il limite φ = 0 ad alto Froude.

**Casi:**

| Caso | Cos'è | Risultato |
|---|---|---|
| **E3** | Caso realizzabile: acciaio, corda 0.294 m, semi-apertura 1.029 m (AR 7), t/c 12 %, EI/GJ 24 % / 3.0 % del pieno, ψ 0.4, xea 0.15, xcm 0.75; f1 = 7.28 Hz | V_f 42.03 / 42.59 / 44.72 / 45.84 kn a h/c 2 / 1 / 0.5 / 0.4, cioè +9.1 % (+0.8 % con la superficie solo nel BEM) |
| **Caso B** | Esempio numerico leggero: corda 0.294 m, semi-apertura 0.882 m, t/c 20 %, EI/GJ 0.087 % / 0.0217 %; f1 = 1.08 Hz | V_f 8.16 / 7.27 / 5.23 / 4.76 kn. Con Fn_h ≈ 2 servono le onde: risultato non affidabile |

**Cartelle:**

| Percorso | Contenuto |
|---|---|
| `report/revH/` | Sorgenti del report: `deck_revH.js` (pptxgenjs), `figure_revH.py`, `vgvf_revH.py`, figure e dati |
| `runs/GOLAND_water_c0p294_s1p029_*eifrac0p24_*delta*_hc*/` | E3, UVLM + BEM (metodo delta) senza immagine, h/c 2 / 1 / 0.5 / 0.4 |
| `casoB_numericSteel_revE/` | Caso B, UVLM + BEM senza immagine, h/c 0.15 … 2 |
| `testO/` | Reticolo di vortici 2D/3D con immagine (`fattore_immagine.py`) e sonda con fattore stazionario |
| `testP/` | Ala immagine in SHARPy: correzioni Q(h) su griglia M16 e scansioni per E3 (radice) e caso B (`casoB/`) |
| `testP/naca1232/` | Replica di NACA TR 1232 e confronto SHARPy con le misure |
| `testP/vanwalree/` | Lettura della tesi di van Walree (1999) e replica del limite φ = 0 (Fig. 3.9a) |
| `_scripts/` | Script di lancio e scansione (`gwscan.py`, `casoD.sh`, `casoB_revE.sh`, …) |

**Codice:** `FLUID/sharpy/gino_bem/image_surface.py` e `precompute_quvlm_goland.py --image-depth h`.

Le tabelle Q(k) fini (`FLUID/PanelAero/Qjj/Quvlm_GOLAND_water_*fineM32_krylov.npz`) non sono nel repository. Si rigenerano con `precompute_quvlm_goland.py`; i parametri sono negli script di `testP/`.

---

## Storico (fino a ottobre 2026)


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
