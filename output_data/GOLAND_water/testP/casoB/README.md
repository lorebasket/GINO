# Test P, caso B: UVLM con ala immagine (φ = 0), 06/11/2026

Stessa procedura di E3 (`../README.md`). Il caso B è quello di rev E/F:
- span 0.882 m, t/c 20 %, acciaio;
- EI/GJ 0.087 % / 0.0217 %, ψ 0.4, α 0.5°;
- tabella fine `Quvlm_GOLAND_water_casoB_psiSHARPy_fineM32_krylov.npz`.

## Calcolo

La correzione immagine è calcolata su M16 con riferimento all'immagine lontana (h = 14.7 m) e sommata alla tabella fine:
- `catena_casoB.sh`: tabelle e scansioni 7.5–12 kn;
- `scan_basse.sh`: scansioni 2–7.6 kn;
- `verifiche.sh`: riferimento senza immagine e griglia M8.

Costo: 3.5–4.2 GB e 6–8 minuti per tabella M16; una scansione circa 3.5 minuti.

## Controlli

| Controllo | Esito |
|---|---|
| Scansione di rev E rifatta con gwscan (h/c 0.4, senza immagine) | 8.36 kn a 4.27 Hz, come rev F |
| Portanza statica con / senza immagine (h/c 2, 1, 0.5, 0.4) | 0.959, 0.907, 0.824, 0.792 |
| \|Q\| nella banda del flutter (k ≈ 6.5 1/m) | −0.3 % … −12.8 % |
| Massa non circolatoria a h/c 0.4 | −6…−9 % |
| Griglia della correzione a h/c 0.4, M8 contro M16 | V_f 4.71 contro 4.76 kn |

## Risultati

Inviluppo del ramo instabile fra 2.5 e 6 Hz. V_f in kn; rev F è la superficie libera solo nel BEM.

| h/c | V_f rev F | V_f con immagine | f_f [Hz] | Fn_h al flutter |
|---|---|---|---|---|
| 2.0 | 8.26 | 8.16 | 4.26 | 1.7 |
| 1.0 | 8.28 | 7.27 | 4.79 | 2.2 |
| 0.5 | 8.34 | 5.23 | 5.38 | 2.2 |
| 0.4 | 8.36 | 4.76 | 5.50 | 2.3 |

- L'immagine **abbassa** V_f: −12 % a h/c 1, −43 % a h/c 0.4. È l'opposto di E3, dove V_f saliva del 9 %.
- Il ramo che diventa instabile non è quello di flutter di rev E (4.3 Hz) ma un ramo a circa 5.5 Hz.
- Senza immagine quel ramo resta smorzato fra 2 e 7.6 kn, con σ/ω da −0.03 a −0.045.
- Con l'immagine perde lo smorzamento circolatorio e attraversa a 4.8 kn. L'attraversamento è piatto (σ/ω ≈ −0.01 già a 3–4 kn).
- **Validità:** al flutter Fn_h ≈ 2.2 e il termine di gravità trascurato pesa 1/Fn_h² ≈ 0.2. Per le parti oscillanti τ ≈ 9 e ω²h/g ≈ 14.
- φ = 0 è al limite: siamo nella zona 1 < Fn_h < 5, dove la superficie reale pesa di più. Il numero va confermato con la funzione di Green ondosa (van Walree) prima di usarlo.
