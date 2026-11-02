# GOLAND — ala di Goland in aria

**Tipo: caso di letteratura, usato per validazione.**

Ala rettangolare a sbalzo di Goland (1945): apertura 6.096 m, corda 1.829 m, asse
elastico al 33 % e baricentro al 43 % della corda. È il benchmark classico del
flutter in aria; qui serve a validare la catena di GINO (trave FEM + DLM o UVLM +
p-k) contro la letteratura e contro l'esempio Goland ufficiale di SHARPy
(stessa analisi in velocità, 120–180 m/s).

- Configurazione: `config_GOLAND.py`
- Esecuzione: `python main.py GOLAND`
- Varianti (DLM / UVLM, numero di modi, smorzamento): `python utl/run_case.py GOLAND --set ...`
- Tabella GAF UVLM: `FLUID/sharpy/gino_bem/precompute_quvlm_goland.py`
- Risultati di riferimento: `output_data/GOLAND/`
