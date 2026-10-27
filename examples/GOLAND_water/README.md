# GOLAND_water — ala tipo Goland immersa in acqua

**Tipo: caso puramente numerico (nessun dato sperimentale).**

Pianta dell'ala di Goland, completamente immersa, ridimensionata per andare in
flutter in acqua. Lo scopo è mostrare quando conta il BEM (massa aggiunta e
radiazione con superficie libera, Capytaine) rispetto alla sola UVLM: si cerca una
velocità di flutter che dipenda in modo marcato dalla sommergenza relativa h/c.

Si tengono due casi, salvati in `output_data/GOLAND_water/`:

| caso | t/c | accoppiamento flesso-torsionale ψ | cartella (rev D) |
|---|---|---|---|
| A | 40 % | nessuno | `casoA_numeric_revD/` |
| B | 20 % | +0.40 | `casoB_numericSteel_revD/` |

Ogni caso contiene lo sweep di sommergenza h/c = 0.15 … 2 (UVLM + BEM *delta*) e il
riferimento solo UVLM. Valori e confronto in `report/GOLAND_water_sommergenza_LV_revD.pdf`.

Entrambi in acciaio, corda 294 mm × apertura 882 mm, x_EA = 0.15c, x_CM = 0.75c.

- Configurazione: `config_GOLAND_water.py`. Variabili d'ambiente `GOLAND_WATER_*`
  (rigidezze, assi, corda, apertura, griglia in ω): vedi l'intestazione del file.
- Esecuzione: `python main.py GOLAND_water`; sweep: `python utl/run_case.py GOLAND_water --sweep ...`
