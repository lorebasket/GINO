# Model2t — Besch & Liu, Model 2T (configurazione G)

**Tipo: caso di letteratura con dati sperimentali, usato per validazione in acqua.**

Montante idrodinamico con pod di estremità provato in vasca da Besch & Liu (1973,
NSRDC Report 4012), configurazione G: pod di 37.7 in in totale, 38.4 lb, baricentro
circa 7.8 in a poppa dell'asse elastico. Flutter sperimentale: **18.1 kn, 6.4 Hz**.

Valida la catena in acqua: trave FEM + pod concentrato, aerodinamica UVLM (GAF di
SHARPy proiettate sui modi di GINO) e massa aggiunta / radiazione da BEM Capytaine
(montante + pod), con p-k. La rigidezza torsionale GJ è calibrata sulla frequenza
bagnata 1T del rapporto.

- Configurazione: `config_Model2T.py`
- Esecuzione: `python main.py Model2t`
- Varianti di metodo (DLM / UVLM, strip / Capytaine, rimozione B2, RFA):
  `python utl/run_case.py Model2t --set ...`; confronto con
  `utl/plot_vgvf_compare.py ... --exp 18.1,6.4`
- Risultati di riferimento: `output_data/Model2t/`
