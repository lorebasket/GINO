# Test O (T1.7) su E3 — sonda circolatoria f(h/c), 05/11/2026

Parte circolatoria di Q (Q − (2/ρ)·A_nc·k²) moltiplicata per f(h/c) = C_Lα(h)/C_Lα(∞) sotto superficie a
pressione nulla (φ = 0, immagine concorde a +h). A_nc e il delta col BEM restano quelli di rev E/F.

- `fattore_immagine.py`: f(h/c) da reticolo di vortici con immagine, 2D (lastra) e 3D (pianta E3, ala intera AR 7).
  Il 2D tende a −(c/h)²/16 per h grande (0.9851 contro 0.9844 a h/c 2). Il documento teorico (§6) scrive /32: è
  sottostimato di un fattore 2 (il downwash dell'immagine concentrata a c/4 va letto a 3/4 della corda).
- `run_testO.sh`: scansioni auto-consistenti (`_scripts/gwscan.py --circ-factor`), 1 thread, dV 0.05 m/s.
- `tabella_testO.py`: attraversamenti con l'inviluppo 20–40 Hz di rev E; `risultati_testO.json`, `testO_sigma.png`.

| h/c | f 2D | f 3D | V_f UVLM+BEM | + immagine 2D | + immagine 3D | Fn_h |
|---|---|---|---|---|---|---|
| 2.0 | 0.985 | 0.953 | 42.36 | 42.48 | 42.81 | 9.1 |
| 1.0 | 0.947 | 0.896 | 42.46 | 42.96 | 43.69 | 12.9 |
| 0.5 | 0.855 | 0.805 | 42.65 | 44.49 | 45.54 | 18.3 |
| 0.4 | 0.814 | 0.770 | 42.71 | 45.29 | 46.35 | 20.5 |

UVLM-only 42.71 kn. Controllo d'identità (f = 1.000001, h/c 2): 42.352 kn contro 42.358 di rev E.
Variazione h/c 2 → 0.4: solo BEM +0.8 %; con immagine +6.6 % (2D) / +8.3 % (3D).

Limiti della sonda: f scalare uguale per tutti i k e tutte le voci modali; f stazionario (l'immagine della scia
instazionaria a k ≈ 1.2 non è rappresentata); h misurata al piano della corda. Lo risolve solo il clone (T1.8).
