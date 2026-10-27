# Roger RFA: convenzione `paper_zurich` (report conversazione)

**Data:** 5 giugno 2026  
**Contesto:** allineamento Roger fit / EOM al paper (Kier, eq. 16–21) con `Qjj` PanelAero precomputate.

---

## 1. Problema

Nel codice FSI coesistono due convenzioni di frequenza ridotta:

| Simbolo | Definizione | Uso |
|---------|-------------|-----|
| `k_pa` | ω/V | Indice su disco PanelAero, PK, interpolazione `Qjj` |
| `k*` | ωb/V, b = c/2 | Paper (Theodorsen / Kier): ŝ = sb/V, eq. 16–21 |

Ponte: **k* = k_pa · b**.

La variabile di Laplace **ŝ = sb/V** è fisica e indipendente da PanelAero. PanelAero usa `k_pa` in input ma incorpora la geometria locale (`aerogrid['l']`) nei kernel DLM.

---

## 2. Decisione architetturale (due livelli)

### Livello A — Caricamento `Qjj` e downwash (PanelAero)

**Non cambiare** quando si interpolano le matrici precomputate:

```text
k_pa = ω/V
Qjj  = interp(qjj_dir, k_pa, Ma)
w_vel = i·k_pa · Z · Φ
w_slope = Z_qs · Φ
Q_modal = Z_force^T · A · (Qjj @ (w_vel + w_slope))
```

Il downwash resta `i·(ω/V)·Z·Φ` perché è la coppia coerente con `Qjj(k_pa)` sul disco.

Equivalente algebrico paper: `i·k*·(Z/b)·Φ = i·k_pa·Z·Φ`.

### Livello B — Roger fit + EOM (paper)

Tutta l'identificazione e l'assemblaggio usano **k*** e **ŝ = sb/V**:

```text
Q(ik*) ≈ A0 + j·k*·A1 − k*²·A2 + Σ_l A_lag,l · (k*² + j·k*·β_l)/(k*² + β_l²)
```

Equazioni effettive (eq. 16, ramo idro senza A₂ DLM):

```text
M_eff = M_s + A_∞          (Capytaine; rfa_apply_b2_mass=False)
K_eff = K_s − ½ρV² A0
C_eff = C_s − ½ρbV A1 + B_∞ + C_emp
ẋa,l = q̇ − (V/b)·β_l·xa,l   →   blag_l = β_l / b  [1/m]
```

---

## 3. Cosa era sbagliato prima

Il fit Roger usava la base razionale direttamente in **k_pa** (griglia PanelAero), identificando coefficienti `B1 ≈ ∂Im(Q)/∂k_pa` invece di **A1 = ∂Im(Q)/∂k***.

Poi `C_eff` moltiplicava per `chord/2` (= b): corretto **solo se** `A1` proviene da fit in **k***. Con fit in `k_pa`, si rischia un fattore **b** di disallineamento.

---

## 4. Flag `rfa_roger_convention`

| Valore | Roger RFA (`roger_fit`, `RFA_PK_method`) | PK (`pk_method`) |
|--------|------------------------------------------|------------------|
| `'panelaero_legacy'` | Fit/reconstruct in `k_pa`; `C_eff` con `B1·b` | `C_aero = ½ρV/k_pa·Im(Q)` |
| `'paper_zurich'` | Fit/reconstruct in `k*`; `C_eff = ½ρbV·A1` | `C_aero = ½ρbV/k*·Im(Q)`, `k*=k_pa·b` |

**Downwash, load `Qjj`, iterazione PK su `k`:** sempre `k_pa = ω/V` in entrambe le modalità.

---

## 5. Configurazione consigliata (Abramson / idro)

```python
rfa_roger_convention = 'paper_zurich'
rfa_apply_b2_mass = False          # AM da Capytaine/strip in M_hat
rfa_blag_dimensionless = True      # forzato internamente da paper_zurich
blag = np.array([0.5, 1.0, 1.5, 2.0])  # β_l adimensionali
rfa_blag_ref_length = 'semichord'
```

---

## 6. Checklist validazione

1. Errore fit Roger `‖Q_fit − Q_ref‖/‖Q_ref‖` < 1–5% su griglia `k_pa`.
2. Confronto PK vs RFA su curve V–g / V–f (stesso caso, stesso `aero_im_Q_scale`).
3. Confronto Abramson: `U = V/(bω) = 1/(k*·b)`; non confondere con `k_pa`.
4. Con RFA+Capytaine: verificare assenza doppio conteggio inerzia (`B2` DLM + `A_∞`).

---

## 7. Riferimenti nel codice

- `COUPLING/flutter_solver.py`: `_fit_roger_to_Q_modal`, `_reconstruct_Q_modal`, `_build_A_aug`, `_roger_lag_poles`
- `COUPLING/hydroelastic_utl/aero_system.py`: PK `Ac` assembly + downwash (`p_over_V = iω/V`)
- `docs/.../formulazioni_sistemi_idroelastici_aumentati.txt`: EOM RFA+strip/Capytaine

---

## 8. Capytaine su modi fisici (RFA+Capytaine)

Capytaine è precomputato sui modi secchi dell'analisi modale iniziale
(`num_modes_egv`, es. 4). Il flutter può tracciare un sottoinsieme
(`modes_to_analyze`, es. [0, 1]), ma **Capytaine si applica sempre a tutti i
modi strutturali fisici (dry)**.

**Flag principale** — `capytaine_structural_pool_embedding` (default `True`):

- `True` → nuova implementazione (blocco strutturale ri-posizionato a ogni passo V).
- `False` → legacy: matrice Capytaine costante su tutti i modi dry `0..num_modes_egv-1`.

**Embedding dinamico** (`capytaine_dynamic_physical_mask=True`, solo se pool embedding attivo):

- Matrice `num_modes_egv × num_modes_egv`, inizialmente **tutta zero**.
- Si copia solo il blocco `num_modes_flutter_egv × num_modes_flutter_egv` dalla Capytaine
  (indici sorgente `0 .. num_modes_flutter_egv-1`).
- **V step 1:** bootstrap (autovalori senza Capytaine) → mappa strutturale→pool come STEP F init
  (es. modi strutturali 0,1 → DOF 1,2 se il pool ha una radice spuria a indice 0).
- **V step ≥ 2:** mappa dal tracking MAC del passo precedente
  (es. modi strutturali 0,1 → DOF 2,3).

Disattivare: `capytaine_dynamic_physical_mask=False` → blocco su `modes_to_analyze`.

---

## 9. Prossimi passi

- Validare su ABRAMSON1965 con `paper_zurich` + embedding Capytaine.
- Se stabile, rendere `paper_zurich` l'unica convenzione e rimuovere il flag legacy.
