# Piano validazione ABRAMSON1965 — overnight

**Obiettivo:** decidere se il framework è **coerente e validabile** sul benchmark Abramson,
oppure se va dichiarato **limitato** (accuratezza bounded, non matching esatto).

**Riferimento sperimentale:** V_f ≈ 48.1 kn, f_accoppiata ≈ 17.5 Hz (Abramson & Ransleben 1965).

---

## Livello 1 — Coerenza interna (necessario, non sufficiente)

| ID | Test | Criterio pass | Cosa invalida |
|----|------|---------------|---------------|
| L1a | Modi secchi vs paper | f₁≈10 Hz, f₂≈20.6 Hz (±5 %) | Modello strutturale errato |
| L1b | Cross-method V_f | Spread < 15 kn tra PK/RFA strip sulla stessa Qjj | Coupling o convenzioni inconsistenti |
| L1c | Trend xea | V_f ↓ verso prua, ↑ verso poppa (strip già ok) | Accoppiamento SC–DLM incoerente |
| L1d | Segno δ / g | g=0 crossing coerente con δ Abramson (conv. codice) | Convenzione autovalori |
| L1e | Roger fit health | ‖Blag‖/‖B0‖ < 50×, no flip B1[0,0] | RFA inaffidabile → V–g a scatti |

## Livello 2 — Confronto fisico (validazione benchmark)

| ID | Test | Criterio “buono” | Criterio “limitato” |
|----|------|------------------|---------------------|
| L2a | V_f vs 48.1 kn | 40–55 kn con measured-Q o tuning noto | Sempre < 35 kn (DLM troppo stabile) o > 60 kn |
| L2b | V–δ modo 1 vs Fig.5 | Forma qualitativa: δ sale verso 0 tra 30–45 kn | Trend invertito o salti > 5 s⁻¹ |
| L2c | Im(Q) vs curve 3 | Fattore scale stabile 0.3–0.7 su banda k flutter | Fattore >>2 o segno errato |
| L2d | f al flutter | 15–20 Hz (branch accoppiata) | < 10 Hz o > 25 Hz senza spiegazione |

## Livello 3 — Robustezza numerica

| ID | Test | Azione |
|----|------|--------|
| L3a | Tracking Roger | `rfa_predict_and_select` off vs on → curve più smooth? |
| L3b | Capytaine mesh | warning λ/8 → mesh convergence study |
| L3c | Grid DLM | nspan×nchord 32×16 vs 40×20: ΔV_f < 10 % |
| L3d | aero_im_Q_scale | sweep 0.4–1.0 documentare sensibilità V_f |

## Verdetto atteso (letteratura)

Il paper Abramson dice esplicitamente che **2D Theodorsen e 3D Reissner–Stevens non danno V_f
nel ordine di grandezza corretto**; solo i **coefficienti misurati 3D** danno predizione conservativa.

Quindi: **matching esatto a 48 kn con DLM puro non è criterio di fallimento del framework** —
è evidenza che serve Im(Q) calibrato (measured Q, scale, o Capytaine ben validato).

### Categorie di esito

1. **COERENTE** — L1 tutto pass; L2a entro banda con almeno un metodo tunato; trend fisici ok.
2. **COERENTE CON LIMITAZIONI** — L1 pass; V_f sotto 48 kn sistematicamente; Im(Q) DLM sovrastima;
   validazione = trend + ordine di grandezza, non punto.
3. **NON VALIDABILE** — L1 fallisce; metodi divergono; trend xea invertito; RFA instabile.

---

## Esecuzione automatica

```bash
/home/lorenzo/anaconda3/envs/GINO/bin/python \
  examples/ABRAMSON1965/run_abramson_validation_overnight.py
```

Output: `output_data/ABRAMSON1965/validation/validation_report.md`
