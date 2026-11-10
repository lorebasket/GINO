# Test P (T1.8) su E3: UVLM con l'ala immagine della superficie libera, 05–06/11/2026

Superficie libera a pressione nulla (φ = 0, valida per Fn_h ≫ 1 e ω²h/g ≫ 1; E3: Fn_h 9–21, ω²h/g ≥ 400).
La condizione si soddisfa con una copia dell'ala traslata a +2h, con la stessa cinematica (stessa traslazione
verticale, rotazione e incidenza). Al primo ordine la sua circolazione è concorde con quella dell'ala vera.

## Implementazione (`FLUID/sharpy/gino_bem/image_surface.py`, `precompute_quvlm_goland.py --image-depth h`)

- Ogni elemento di trave è duplicato sugli stessi nodi, con massa e rigidezza dimezzate. La copia porta la
  superficie immagine, con la linea media spostata di 2h/c.
- Il guadagno trave → griglia dell'immagine è quello dell'ala vera, quindi niente traslazione spuria 2h·θ.
- Le forze dell'immagine non arrivano alla trave: escluse dalla statica, dal guadagno griglia → trave e da Kss.
- Il calcolo è fatto con modi GINO di h/c 2, identici (anche nel segno) a quelli delle cartelle h/c 1, 0.5 e 0.4.

## Memoria e griglia

- Con l'immagine, la griglia fine M32N32Ms10 supera i 10 GB e va in MemoryError.
- L'effetto immagine è quindi calcolato su M16 e sommato alla tabella fine:
  `Q(h) = Q_fine + [Q_M16,img(h) − Q_M16,img(h = 14.7 m)]`.
- Il riferimento con l'immagine lontana (2h = 100 c) cancella il residuo numerico della riduzione Krylov su un
  sistema di dimensione doppia: rispetto alla tabella senza immagine è 0.3 % a k = 8.5 e 1 % a k = 24.
- Script: `somma_delta.py`, `run_scan.sh`.

## Controlli

| Controllo | Esito |
|---|---|
| Tabella fine rigenerata senza immagine | identica a quella di rev E (scarto 0.0000) |
| Immagine a 2h = 29.4 m | portanza statica dell'ala vera invariata (9.592 N); ΔQ = 0 a k = 0 |
| Portanza statica con immagine / senza, h/c 2, 1, 0.5, 0.4 | 0.950, 0.893, 0.802, 0.767 contro 0.953, 0.896, 0.805, 0.770 del reticolo 3D del test O |
| Massa non circolatoria h/c 2 → 0.4 | UVLM con immagine −7…−9 %, BEM −8…−11 % (stesso verso: nel delta la superficie non è contata due volte) |
| Griglia della ΔQ, M8 contro M16 | 7–8 % della ΔQ a h/c 0.4, quindi su V_f circa ±0.2 kn |

## Risultati (inviluppo 20–40 Hz di rev E, `tabella_testP.py` → `risultati_testP.json`, `testP_sigma.png`)

V_f in kn, UVLM-only 42.71 kn.

| h/c | V_f UVLM+BEM (rev E) | V_f UVLM con immagine + BEM | f_f [Hz] | sonda test O 3D |
|---|---|---|---|---|
| 2.0 | 42.36 | 42.03 | 28.2 | 42.81 |
| 1.0 | 42.46 | 42.59 | 28.3 | 43.69 |
| 0.5 | 42.65 | 44.72 | 28.4 | 45.54 |
| 0.4 | 42.71 | 45.84 | 28.4 | 46.35 |

- Da h/c 2 a 0.4: +0.8 % col solo BEM, +9.1 % con l'immagine (sonda del test O: +8.3 %).
- A h/c ≥ 1 l'effetto è sotto l'1 %.
- Nella banda del flutter (k ≈ 8.5) |Q| cala dell'8–12 % a h/c 0.4, contro il 23 % stazionario. La sonda del test
  O, con un fattore stazionario, sovrastima V_f di circa 0.5–1 kn a h/c ≤ 0.5.
