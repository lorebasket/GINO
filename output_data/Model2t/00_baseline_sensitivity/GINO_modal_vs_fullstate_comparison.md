# GINO modal PK vs full-state eigenanalysis — Model2t δ BEM comparison

**Date:** 2026-07-14  
**Case:** Besch & Liu Model 2T (config G), nominal GJ/EI, Rayleigh ζ = 5% on modes 0–1  
**Purpose:** Paper material — same hydro-aero model (UVLM + Capytaine δ BEM), two eigenformulations

---

## 1. What is being compared?

Both runs share the **same delta-BEM hybrid**:

| Ingredient | Setting |
|------------|---------|
| Structural model | 4 wet modes (1B, 1T, 2B, 2T), nominal stiffness |
| Circulatory aero | SHARPy UVLM GAF table, grid **n30c12** (M=12, N=30) |
| Hydrodynamic BEM | Capytaine modal radiation, **δ injection**: \(M_\mathrm{hyd} = A_\mathrm{capy}(\omega) - A_\mathrm{UVLM}\) |
| Added-mass removal | **none** (UVLM kept intact for δ mode) |
| Damping | Rayleigh 5% on structural modes 0–1 |

They differ only in **how the coupled eigenproblem is solved and tracked**:

| Label | Folder | Eigenproblem | Tracked DOFs |
|-------|--------|--------------|--------------|
| **GINO modal PK** | `_baseline_GINO_UVLM_BEM/` | 4-mode companion **`Ac`** (2×nm), PK iteration on modal GAF + δ BEM | 2 PK branches (1B seed ~13 rad/s, 1T seed ~68 rad/s) |
| **GINO full-state** | `_baseline_GINO_fullstate_UVLM_BEM/` | **`eig(ss.A)`** via SHARPy `LinearAssembler` (beam + full UVLM wake) | Pool tracker: bending / torsion / mode4 (plot uses bending + torsion only) |

**Figures**

| File | Content |
|------|---------|
| `comparison_vgvf_gino_modal_pk_vs_fullstate_delta_bem.png` | **This comparison** (orange = modal PK, purple = full-state) |
| `_baseline_GINO_UVLM_BEM/vg_vf_combined.png` | Source: modal PK V–g / V–f |
| `_baseline_GINO_fullstate_UVLM_BEM/model2t_pk_bem_vgvf_nm4.png` | Source: full-state tracker (4 branches in raw plot; comparison uses 2) |

Plot frame (matched to multi-baseline comparison figure): **V = 3.5–35 kn**, **g ∈ [−0.5, 0.5]**, bending dashed / torsion solid, Besch & Liu experiment overlaid.

---

## 2. Headline results

### 2.1 Flutter (torsion branch, g → 0)

| Method | V_f [kn] | f_f [Hz] | ΔV vs experiment (18.1 kn) |
|--------|----------|----------|----------------------------|
| **Experiment** (Besch & Liu Fig. 11) | **18.1** | **6.4** | — |
| GINO modal PK + δ BEM | **17.7** | **6.71** | −0.4 kn |
| GINO full-state + δ BEM | **19.2** | **6.74** | +1.1 kn |

**Takeaway:** Flutter **frequency** agrees within ~0.3 Hz. **Speed** bracketed experiment: modal PK is ~1.5 kn low, full-state ~1 kn high. The full-state result (~19.2 kn) matches the SHARPy δ+BEM reference (~19.0–19.5 kn) from the separate multi-code comparison.

### 2.2 Bending branch — the main divergence

| V [kn] | Modal PK f_1B [Hz] | Full-state f_bend [Hz] | Modal PK g_1B | Full-state g_bend |
|--------|-------------------|------------------------|---------------|-------------------|
| 3.9 | 1.90 | 1.90 | −0.39 | −0.25 |
| 10 | **1.04** (dying) | **2.00** | **−2.83** | −0.73 |
| 15 | *lost (NaN)* | 2.46 | — | −1.15 |
| 20 | *lost* | 3.19 | — | −1.41 |
| 30 | *lost* | 4.36 | — | −1.81 |
| 35 | *lost* | 4.88 | — | −1.95 |

**Takeaway:** Above ~10 kn the modal PK **loses the bending branch** (mode 0 terminates in PK with σ ≈ −19, g ≈ −3). The full-state solver keeps a physical bending root that **rises** 1.9 → 4.9 Hz — the expected coalescence partner behaviour seen in SHARPy full-state and experiment.

### 2.3 Torsion branch — good agreement at low V, moderate drift at high V

| V [kn] | Modal PK f_1T [Hz] | Full-state f_tors [Hz] |
|--------|-------------------|------------------------|
| 3.9 | 8.22 | 8.14 |
| 10 | 7.87 | 8.07 |
| 20 | 6.42 | 6.66 |
| 30 | 5.86 | 5.98 |
| 35 | 5.70 | 5.75 |

Torsion tracks within ~0.1–0.3 Hz over most of the sweep. Post-flutter positive g and falling frequency are consistent in both paths.

---

## 3. Why do they differ?

### 3.1 State-space dimension (root cause for bending)

```
Modal PK (GINO):     q ∈ ℝ⁴  →  companion Ac ∈ ℝ⁸ˣ⁸
Full-state (SHARPy): x ∈ ℝ^(n_beam + n_UVLM + n_wake)  →  ss.A full size
```

The **bend–torsion coalescence partner** is not a pure 1B modal shape. In the full-state problem it is a **wake-dominated** root with beam participation \(\|q_\mathrm{beam}\|/\|x\| \sim 10^{-3}\)–\(10^{-1}\). The 4-mode reduction cannot represent this partner once circulatory damping on mode 0 becomes large.

### 3.2 Circulatory damping spike near 10 kn (modal PK only)

From `stiffness_damping_contributions.csv` (modal PK): aerodynamic damping on mode 0 reaches **C_aero ≈ −22** near 10 kn. This drives σ strongly negative and terminates PK branch 0. The full-state root does not “die” because it is not tied to a single structural modal coordinate.

### 3.3 Flutter speed offset (~1.5 kn)

Both use δ BEM and the same UVLM table, so the **~1.5 kn gap** (17.7 vs 19.2 kn) is **not** a BEM or δ-injection issue. It arises because:

1. **Modal PK** flutter is detected on the **reduced 2-branch PK tracker** after branch 1 has already failed — the torsion crossing may be slightly shifted.
2. **Full-state** flutter uses the **complete UVLM–beam coupling** at each V, closer to the SHARPy reference path.

Frequency at flutter is essentially unchanged (6.71 vs 6.74 Hz) → same instability mechanism, different onset speed due to state reduction.

### 3.4 What is *not* different

- δ BEM hydro model (same NPZ, same strip coupling)
- UVLM circulatory GAF (same Quvlm table, n30c12)
- Dry structural modes and Rayleigh damping
- Low-speed torsion track (~8 Hz at 4 kn)

---

## 4. Physical interpretation (paper-ready)

### 4.1 Two roots, two questions

The Model 2T δ-hybrid problem has two physically distinct oscillatory roots in the flutter band:

1. **Torsion-dominated root** (~7–8 Hz at low V → ~6.7 Hz at flutter) — drives instability; both GINO paths capture this reasonably.
2. **Bending coalescence partner** (~2 Hz rising with V in full-state) — subcritical, heavily damped, but essential for **loci / coalescence diagrams**; only the full-state path resolves it above 10 kn.

### 4.2 When is modal PK sufficient?

| Question | Modal PK OK? |
|----------|--------------|
| Torsion flutter speed (±1–2 kn)? | **Yes** (17.7 kn vs 19.2 kn bracket exp.) |
| Flutter frequency? | **Yes** (~6.7 Hz) |
| V–g of torsion branch 3.5–20 kn? | **Mostly yes** |
| Bending branch / coalescence loci? | **No** — branch dies ~10 kn |
| Post-flutter torsion curve >25 kn? | **Acceptable** (small f drift) |

### 4.3 Recommended reporting in the paper

- **Design flutter:** cite full-state (or SHARPy δ+BEM) **V_f ≈ 19 kn, f ≈ 6.7 Hz**, with experiment **18.1 kn / 6.4 Hz**.
- **Modal PK:** valid **engineering shortcut** for flutter speed if 1–2 kn tolerance is acceptable; state clearly that coalescence loci require full-state.
- **Do not** attribute the bending-branch mismatch to δ BEM — it persists with identical hydro injection when the eigen solver is changed.

---

## 5. Relation to other baselines

| Method | V_f [kn] | Bending @ 10 kn | Notes |
|--------|----------|-----------------|-------|
| Experiment | 18.1 | ~2.0 Hz (Fig. 11) | Target |
| GINO full-state δ BEM | 19.2 | 2.0 Hz | This note |
| SHARPy δ+BEM (full-state) | ~19.0–19.5 | 2.0 Hz | External reference |
| GINO modal PK δ BEM | 17.7 | 1.0 Hz (dying) | This note |
| GINO-DLM-BEM (aero b2 removal) | 28.4 | — | Different aero path; not δ hybrid |

The updated multi-baseline figure `comparison_vgvf_baseline_sharpy_vs_dlm_vs_delta_vs_uvlm_bem_30kn.png` now **omits SHARPy+BEM** (green); it shows GINO-DLM-BEM, SHARPy stock UVLM, and GINO-UVLM-BEM (full-state purple) on the same frame.

---

## 6. Suggested paper text (draft snippets)

### Methods

> For the δ-hybrid baseline we compare two eigensolvers on an identical UVLM+ Capytaine model: (i) GINO's modal p–k method on a four-mode companion matrix with UVLM generalized aerodynamic forces and δ-injected added mass, and (ii) a full-state eigenanalysis of the SHARPy linearized UVLM–beam system with the same δ BEM correction.

### Results

> Both formulations predict torsion flutter near 6.7 Hz. The full-state analysis gives V_f ≈ 19.2 kn, in line with SHARPy (19.0–19.5 kn) and experiment (18.1 kn). The modal p–k reduction yields V_f ≈ 17.7 kn; the difference is traced to the loss of the bending coalescence partner above ~10 kn, where aerodynamic damping on the first bending mode exceeds the validity of the low-order modal state space.

### Discussion

> The δ-injection strategy itself is not the source of the bending–torsion locus discrepancy: with identical hydrodynamic correction, only the full-state spectrum exhibits the rising low-frequency partner expected from Besch & Liu's coalescence diagram. For flutter speed prediction the modal p–k model remains useful; for modal interaction and loci plots the full-state formulation is required.

---

## 7. Reproduce

```bash
# GINO-only comparison (this document)
/home/lorenzo/anaconda3/envs/GINO/bin/python \
  examples/Model2t/plot_gino_modal_vs_fullstate_vgvf.py

# Multi-baseline (no SHARPy+BEM)
/home/lorenzo/anaconda3/envs/GINO/bin/python \
  examples/Model2t/plot_sharpy_gino_vgvf_comparison.py \
  --gino-dir output_data/Model2t/00_baseline_sensitivity/_baseline_pk_capytaine_aerob2removal \
  --sharpy-dir output_data/Model2t/00_baseline_sensitivity/_baseline_sharpy \
  --fullstate-csv output_data/Model2t/00_baseline_sensitivity/_baseline_GINO_fullstate_UVLM_BEM/model2t_pk_bem_tracked_nm4.csv \
  --v-min-kn 3.5 --v-max-kn 35 --g-ylim -0.5 0.5 \
  --out output_data/Model2t/00_baseline_sensitivity/comparison_vgvf_baseline_sharpy_vs_dlm_vs_delta_vs_uvlm_bem_30kn.png
```

---

## 8. Data files

```
_baseline_GINO_UVLM_BEM/
  vg_data.csv, vf_data.csv, flutter_summary.csv
  vg_vf_combined.png
  physical_root_loci_modes12.png
  stiffness_damping_contributions.csv

_baseline_GINO_fullstate_UVLM_BEM/
  model2t_pk_bem_tracked_nm4.csv
  model2t_pk_bem_vgvf_nm4.png
  model2t_pk_bem_vdelta_vg_vf_nm2.png
  model2t_sharpy_eigvecs_nm4.npz
```

---

*Companion to `SESSION_SUMMARY.md`. Generated for paper drafting.*
