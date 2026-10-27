# utl — strumenti generici di calcolo, post-processing e controllo

Script validi per **qualsiasi caso** in `examples/<CASE>/config_<CASE>.py`: il caso si
passa come argomento, niente percorsi o parametri cablati. I percorsi di modi, tabella
GAF, BEM e uscite vengono da `paths(FSI)` del config del caso, se la definisce, altrimenti
dalle opzioni `--dry --gaf --bem --anc --base`.

Ambiente: `conda run -n GINO python utl/<script>.py ...` (le GAF UVLM si calcolano
nell'ambiente `sharpy`, con gli script in `FLUID/sharpy/gino_bem/`).

| script | cosa fa |
|---|---|
| `run_case.py` | lancia la pipeline (`main.run_flutter_analysis_single`) con override del config (`--set`, `--json`, `--env`) e sweep di un parametro alla volta (`--sweep`) |
| `run_chain.sh` | catena completa: modi secchi → contenuto dei modi → GAF UVLM → BEM → A_UVLM_nc → scansioni uvlm e delta (→ p-k con `PK=1`) |
| `export_modes.py` | modi secchi → `dry_modes.npz` (pos, sub, Phi, f, M̂, K̂, Ĉ) per GAF e BEM |
| `classify_modes.py` | contenuto dei modi per sotto-modello e gdl, simmetria port/stbd |
| `uvlm_nc_mass.py` | massa non circolatoria dell'UVLM dalla tabella GAF (fit di Re Q in k²) |
| `scan.py` | scansione p-k auto-consistente (tutte le radici, nessun tracker), stadi `uvlm`, `delta`, `bem` |
| `flutter_from_scan.py` | primo flutter persistente dai file `scan_*.txt` |
| `plot_vgvf_scan.py` | V-g / V-f dalle scansioni (rami ricostruiti per continuità, o radici grezze con `--raw`) |
| `branches.py` | riassunto dei rami di un run p-k: convergenza, salti, g massimo, attraversamenti g = 0 |
| `plot_vgvf_compare.py` | sovrapposizione V-g / V-f di più run p-k (+ SHARPy, + punto sperimentale) |
| `summarize_runs.py` | tabella V_f / f_f su più run, grafico a barre (matrice di metodi) o tornado (sensibilità) |

Ogni script ha l'help completo: `python utl/<script>.py -h`.

Gli strumenti di controllo dell'accoppiamento (sovrapposizione trave / UVLM / BEM, massa
d'acqua nel p-k, massa aggiunta UVLM vs BEM) sono in `COUPLING/debug/`.

## Da dove vengono

Sostituiscono gli script per singolo caso che stavano negli esempi (rimasti in locale,
fuori dal repository):

| vecchi script | ora |
|---|---|
| `Model2t/run_baseline_*`, `run_bem_*`, `run_rfa_*`, `run_pk_*`, `run_method_matrix`, `run_input_sensitivity`, `run_00_baseline_sensitivity`; `GOLAND/run_goland_*`; `GOLAND_water/run_*_sweep`, `run_underwater_flutter_tune` | `run_case.py` (`--set`, `--sweep`) |
| `TFOIL_SIMPLE/run_chain.sh`, `chain_nm20.sh`, `screen.sh` | `run_chain.sh` |
| `TFOIL_SIMPLE/export_modes.py`, `classify_modes.py`, `uvlm_nc_mass.py`, `scan.py`, `flutter_from_scan.py`, `branches.py` | stesso nome qui |
| `TFOIL_SIMPLE/plot_vgvf_scan.py`, `plot_vgvf_roots.py`, `plot_sommergenza.py`, `plot_sweep_V1V2.py` | `plot_vgvf_scan.py`, `flutter_from_scan.py` |
| `*/plot_*_comparison.py`, `plot_*_vgvf.py`, `plot_input_sensitivity_*` | `plot_vgvf_compare.py` |
| `Model2t/summarize_method_matrix.py`, `plot_input_sensitivity_tornado.py` | `summarize_runs.py` |
| `TFOIL_SIMPLE/plot_overlay.py`, `Model2t/plot_aero_beam_front_xz*.py` | `COUPLING/debug/plot_model_overlay.py` |
| `TFOIL_SIMPLE/check_hydro_mass_in_pk.py`, `verifica_iniezione.py` | `COUPLING/debug/check_hydro_mass_in_pk.py`, `scan.py --a-source uvlm` (test di identità) |
| `TFOIL_SIMPLE/added_mass_checks.py`, `Model2t/plot_uvlm_vs_capy_added_mass_comparison.py` | `COUPLING/debug/compare_added_mass.py` |

Non portati, perché legati a un singolo studio già chiuso: figure del paper e dei report
(`build_*.js`, `plot_paper_*`, `report_data_*`), calibrazioni di Model2t
(`calibrate_*_wet_1t.py`), diagnostiche di parità GOLAND–SHARPy (`diag_*`,
`run_goland_sharpy_parity.py`, `run_goland_pk_nm4_eta_cal.py`).
