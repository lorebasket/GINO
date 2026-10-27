# TFOIL_SIMPLE — T-foil semplificato

**Tipo: caso semplificativo, di puro calcolo.**

T-foil (timone + elevatore) ricavato da AC38 con geometria e materiali
semplificati: anima in carbonio + schiuma e pelle sottile. Sostituisce AC38 per
provare la catena su una geometria a T senza i dettagli della pala reale. Serve a
confrontare la sola UVLM (massa apparente dell'UVLM, nessuna superficie libera)
con UVLM + BEM (Capytaine, superficie libera), sempre con lo stadio *delta*.

- `config_TFOIL_SIMPLE.py`: configurazione GINO e `paths()` (modi, GAF, BEM, uscite)
- `tfoil_simple_geom.py`: geometria, materiale e sezioni (unica fonte, variabili
  `TFS_*`). Lo importano anche la trave FEA, il caso SHARPy
  (`FLUID/sharpy/gino_bem/tfoil_simple.py`) e il BEM
  (`FLUID/capytaine/run_tfoil_simple_radiation.py`), per questo è versionato
  insieme al config.

Configurazione di riferimento:
`TFS_FK=0.5 TFS_XEA=0.35 TFS_TCK=0.18 TFS_TSKIN=0.0006 TFS_EDX=-0.20 TFS_NM=20 TFS_GRID=n40c20e16`

Catena completa (modi → GAF UVLM → BEM → scansioni p-k uvlm e delta):

```bash
export TFS_FK=0.5 TFS_XEA=0.35 TFS_TCK=0.18 TFS_TSKIN=0.0006 TFS_EDX=-0.20 TFS_NM=20 TFS_GRID=n40c20e16
CASE=TFOIL_SIMPLE \
GAF_CMD='FLUID/sharpy/gino_bem/precompute_quvlm_tfoil_simple.py --gino {dry} --nm {nm} --grid n40c20e16 --out {gaf}' \
BEM_CMD='FLUID/capytaine/run_tfoil_simple_radiation.py --modes {dry} --nm {nm} --out {bem}' \
bash utl/run_chain.sh
```
