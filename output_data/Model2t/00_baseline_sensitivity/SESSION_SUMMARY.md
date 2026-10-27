# Model2t baseline session summary

**Date:** 2026-07-14  
**Scope:** GINO vs SHARPy coalescence loci, delta BEM, modal PK vs full-state eigenanalysis  
**Status:** Full-state sweep completed; modal PK post-processing updated; MAC-seeded plots still need refinement

---

## 1. Goals of this session

1. Make GINO `physical_root_loci_modes12.png` comparable to SHARPy delta-BEM coalescence tracking.
2. Start velocity sweeps at **2 m/s** (not 0 m/s / 0.5 m/s).
3. Understand why GINO **Branch 1 (1B)** diverges from SHARPy above ~10 kn.
4. Run a **full-state eigenanalysis** (UVLM + beam `ss.A`) and save results in a **new folder**.

---

## 2. Executive summary

| Topic | Conclusion |
|-------|------------|
| **Flutter speed** | GINO modal PK and SHARPy full-state agree reasonably: **~17.7 kn @ 6.7 Hz** (GINO PK) vs **~19.2 kn @ 6.7 Hz** (SHARPy / full-state torsion). Delta BEM is **not** the main cause of the Branch-1 mismatch. |
| **Bending branch (1B)** | GINO 4-modal PK **dies near 10 kn** (huge circulatory damping on mode 0). Post-processor then re-locks onto the torsion partner. SHARPy / full-state bending **rises** 1.9 → 5.5 Hz over 2–40 kn. |
| **Root cause** | GINO solves an **8×8 modal companion `Ac`** (4 modes). SHARPy solves the **full state-space** (beam + UVLM wake). The bending coalescence partner is wake-dominated (beam participation ~10⁻³). |
| **Full-state run** | Completed via SHARPy `eig(ss.A)` bridge + GINO post-processing. Bending track now matches SHARPy qualitatively. |
| **Open issue** | GINO MAC-seeded `physical_root_loci_modes12.png` in the full-state folder can **swap partners** when two roots sit near ~2 Hz and ~8 Hz; prefer `model2t_pk_bem_tracked_nm4.csv` / `model2t_pk_bem_vgvf_nm4.png` for clean tracks. |

---

## 3. Output folders (current standings)

All paths relative to repo root: `output_data/Model2t/00_baseline_sensitivity/`

### 3.1 GINO modal PK + delta BEM (main baseline)

**Folder:** `_baseline_GINO_UVLM_BEM/`

| Artifact | Role |
|----------|------|
| `vf_data.csv`, `vg_data.csv`, `eigenvalue_data.csv` | PK flutter sweep (may still start at 0.5 m/s unless re-run with `--force`) |
| `physical_root_loci_modes12.png` | SHARPy-style MAC loci from modal `Ac` |
| `physical_root_loci.csv` | Branch 1/2 tracked roots |
| `stiffness_damping_contributions.csv` | Per-mode M, K, C breakdown (shows large `C_aero` on mode 0 near 10 kn) |
| `flutter_summary.csv` | Branch 2 flutter: **17.67 kn @ 6.71 Hz**; Branch 1 no crossing |

**Re-run PK baseline from 2 m/s:**

```bash
PYTHONPATH="$(pwd):STRUCTURE:STRUCTURE/FEA/fea_utl:FLUID/PanelAero:FLUID:FLUID/capytaine" \
  /home/lorenzo/anaconda3/envs/GINO/bin/python \
  examples/Model2t/run_baseline_GINO_UVLM_BEM.py --force
```

**Re-run physical-root post-processing only:**

```bash
PYTHONPATH="$(pwd):STRUCTURE:STRUCTURE/FEA/fea_utl:FLUID/PanelAero:FLUID:FLUID/capytaine" \
  /home/lorenzo/anaconda3/envs/GINO/bin/python \
  examples/Model2t/sweep_gino_physical_roots.py \
  --outdir output_data/Model2t/00_baseline_sensitivity/_baseline_GINO_UVLM_BEM
```

---

### 3.2 SHARPy delta-BEM reference

**Folder:** `_baseline_sharpy_deltainjection_plusbem/`

| Artifact | Role |
|----------|------|
| `model2t_pk_bem_tracked_nm4.csv` | Pool tracker: bending / torsion / mode4 |
| `physical_root_loci_modes12.png` | Reference coalescence loci |
| `root_pool_all.csv` | Per-velocity root pool with modal content |

**Reference flutter (torsion):** g crosses 0 near **19.0–19.5 kn**, f ≈ **6.7 Hz**.

**Reference bending @ 10 kn:** f ≈ **2.00 Hz** (rising with V).

---

### 3.3 GINO full-state (new — completed 2026-07-14)

**Folder:** `_baseline_GINO_fullstate_UVLM_BEM/`

| Artifact | Role |
|----------|------|
| `model2t_sharpy_eigvecs_nm4.npz` | Full `eig(ss.A)` at each V (~28 MB) |
| `model2t_pk_bem_tracked_nm4.csv` | **Best bending/torsion tracks** (SHARPy pool logic) |
| `model2t_pk_bem_vgvf_nm4.png` | V–g / V–f from tracker |
| `root_pool_all.csv` | 218 pooled roots over 73 velocity steps |
| `physical_root_loci_modes12.png` | GINO MAC-seeded post-proc (can swap partners) |
| `physical_root_loci.csv` | Seeded-branch CSV |
| `gino_physical_roots_nm4.npz` | Compact tracked roots |

**Sweep settings:**

- V: **2 m/s → 40 kn**, step **0.5 kn** (73 points)
- BEM: **delta**, grid **n30c12**, **nm = 4**
- Const ω: **7.45 Hz** (ω\* = 46.81 rad/s)
- Seeds for MAC post-proc: **1.86 Hz + 7.45 Hz**

**Re-run:**

```bash
# Full sweep + post-process (needs sharpy env, ~25 min)
SHARPY_PYTHON=/home/lorenzo/anaconda3/envs/sharpy/bin/python \
  PYTHONPATH="$(pwd):STRUCTURE:STRUCTURE/FEA/fea_utl:FLUID/PanelAero:FLUID:FLUID/capytaine" \
  /home/lorenzo/anaconda3/envs/GINO/bin/python \
  examples/Model2t/sweep_gino_fullstate_eigen.py

# Post-process only (if NPZ exists)
... sweep_gino_fullstate_eigen.py --skip-sharpy-run
```

---

## 4. Code changes made in this session

### 4.1 SHARPy-comparable modal post-processing

**Files:** `COUPLING/hydroelastic_utl/pk_physical_spectrum.py`, `examples/Model2t/sweep_gino_physical_roots.py`

| Change | Detail |
|--------|--------|
| k-policy default | `pk_converged`: use PK-converged `k_b(V)=ω_PK/V` from `eigenvalue_data.csv`; fallback to MAC-chained `ω_branch/V` when PK branch 0 drops |
| Seeds | Explicit **1.86 + 7.45 Hz** (not `{1B,1T}` name filter) |
| Participation | Displacement-only modal energy; bottom panel plots **`part_bend`** (1B share of q) |
| V range | **2 m/s – 40 kn** via `--v-min-m-s 2.0` |
| MAC tracking | Frequency-nearest fallback; no hard terminate at MAC < 0.35 |
| `fullstate_sharpy` policy | Added for full-state plots: title + beam `part` (not `part_bend`) |

### 4.2 Velocity floor at 2 m/s

**Files:** `examples/Model2t/config_Model2T.py`, `examples/Model2t/run_baseline_GINO_UVLM_BEM.py`, `sweep_gino_physical_roots.py`

- `V_list = linspace(2.0, 20.58, …)` in config and runner
- `vgvf_plot_V_min_m_s = 2.0`

**Note:** Existing PK CSVs in `_baseline_GINO_UVLM_BEM/` may pre-date this unless `run_baseline_GINO_UVLM_BEM.py --force` was re-run.

### 4.3 Delta BEM (earlier in session)

**Files:** `sharpy/gino_bem/precompute_bem_delta.py`, `COUPLING/hydroelastic_utl/bem_delta.py`, `aero_system`, `flutter_solver`, Model2t runner

- `pk_bem_mode=delta` wired through GINO flutter path
- Forces `added_mass_removal='none'` so UVLM stays intact for delta injection

### 4.4 Full-state bridge (new)

**Files:**

- `COUPLING/hydroelastic_utl/sharpy_fullstate_spectrum.py` — SHARPy subprocess + NPZ MAC-chain
- `examples/Model2t/sweep_gino_fullstate_eigen.py` — entry point

GINO has **no native full UVLM state-space**; only modal `Quvlm` + `Ac`. Full-state requires SHARPy `LinearAssembler` → `eig(ss.A)`.

---

## 5. Key numerical comparison

### 5.1 Bending frequency vs velocity

| V [kn] | GINO modal PK Branch1 | SHARPy bending | Full-state bending |
|--------|----------------------|----------------|-------------------|
| 3.9 | 1.90 Hz | 1.91 Hz | 1.90 Hz |
| 10 | **1.04–1.23 Hz** (branch dying / jump) | **2.00 Hz** | **2.02 Hz** |
| 20 | re-locked ~7–8 Hz partner | (tracker ends ~14 kn in ref CSV) | **3.26 Hz** |
| 40 | re-locked torsion partner | — | **5.53 Hz** |

### 5.2 Torsion / flutter

| Source | V_f [kn] | f_f [Hz] |
|--------|----------|----------|
| GINO modal PK (`flutter_summary.csv`) | 17.67 | 6.71 |
| SHARPy delta BEM (torsion g→0) | ~19.0–19.5 | ~6.7 |
| Full-state (`model2t_pk_bem_tracked_nm4.csv`) | ~19.4 (g crosses 0 between 18.9 and 19.9 kn) | ~6.72 |

### 5.3 Participation metric

| Path | Definition | Typical value (bending) |
|------|------------|---------------------------|
| GINO modal PK post-proc | `part_bend` = 1B share of **modal q** | ~1.0 (misleading for coalescence) |
| SHARPy / full-state tracker | `part` = ‖beam‖ / ‖full state‖ | ~0.002–0.17 (wake-dominated) |

---

## 6. Why GINO Branch 1 behaves differently

1. **State reduction:** GINO PK uses 4 structural modes in companion form; SHARPy retains full UVLM wake states. The physical bending root is not a pure 1B modal shape.

2. **Circulatory damping spike:** Near 10 kn, mode 0 gets very negative aerodynamic damping (`C_aero` ≈ −22 in `stiffness_damping_contributions.csv`). PK branch 0 terminates (σ ≈ −18.6, g ≈ −3).

3. **Post-processor re-locking:** After 1B dies in PK, `physical_root_loci` MAC chain jumps to the torsion partner (~7–8 Hz) for Branch 1 — this is a **tracking artifact**, not a new physical root.

4. **Full-state confirms:** With `eig(ss.A)`, bending rises smoothly like SHARPy; flutter speed stays within ~1–2 kn of SHARPy.

---

## 7. Known issues and next steps

### Done

- [x] SHARPy-comparable modal post-processing (`pk_converged`, seeds, `part_bend`, V ≥ 2 m/s)
- [x] Delta BEM integration in GINO flutter path
- [x] Full-state sweep + new output folder
- [x] `fullstate_sharpy` policy in `write_physical_root_artifacts()`

### Open / suggested

- [ ] **Fix MAC-seeded post-processor** for full-state NPZ so `physical_root_loci_modes12.png` matches `model2t_pk_bem_tracked_nm4.csv` (partner swap near 5–6 kn when bending ~1.9 Hz and torsion ~8 Hz coexist)
- [ ] **Overlay plot:** full-state vs `_baseline_sharpy_deltainjection_plusbem/` vs modal GINO PK on one figure
- [ ] **Re-run** `run_baseline_GINO_UVLM_BEM.py --force` so PK CSVs also start at 2 m/s
- [ ] **Paper figures:** use `model2t_pk_bem_vgvf_nm4.png` (full-state) and SHARPy reference for coalescence; use modal PK only with caveat about 4-mode reduction

---

## 8. Quick file map

```
examples/Model2t/
  run_baseline_GINO_UVLM_BEM.py      # PK flutter baseline
  sweep_gino_physical_roots.py       # Modal Ac MAC loci
  sweep_gino_fullstate_eigen.py      # Full-state SHARPy + GINO post-proc
  config_Model2T.py                  # V_list from 2 m/s

COUPLING/hydroelastic_utl/
  pk_physical_spectrum.py            # MAC loci, pk_converged, write_physical_root_artifacts
  sharpy_fullstate_spectrum.py       # SHARPy eig(ss.A) bridge
  bem_delta.py                       # Delta BEM loader

sharpy/gino_bem/
  run_model2t_pk_bem.py              # SHARPy const-ω sweep (used by full-state script)
  precompute_bem_delta.py            # Delta offset precompute

output_data/Model2t/00_baseline_sensitivity/
  _baseline_GINO_UVLM_BEM/           # Modal PK + delta BEM
  _baseline_sharpy_deltainjection_plusbem/   # SHARPy reference
  _baseline_GINO_fullstate_UVLM_BEM/ # Full-state (new)
  comparison_vgvf_baseline_sharpy_vs_pk_capytaine_vs_delta.png
```

---

## 9. Default parameters (reference)

| Parameter | Value |
|-----------|-------|
| Seed frequencies | 1.86 Hz, 7.45 Hz |
| Const ω (Hz) | 7.45 |
| V min | 2.0 m/s (≈ 3.89 kn) |
| V max | 40 kn |
| V step (full-state) | 0.5 kn |
| Grid | n30c12 (M=12, N=30) |
| Structural modes | 4 |
| BEM mode | delta |
| k-policy (modal post-proc) | `pk_converged` |

---

## 10. How to read the plots

| Plot | Trust for… |
|------|------------|
| `_baseline_GINO_UVLM_BEM/physical_root_loci_modes12.png` | Modal PK coalescence attempt; Branch 1 wrong above ~10 kn |
| `_baseline_sharpy_deltainjection_plusbem/physical_root_loci_modes12.png` | Reference full-state coalescence |
| `_baseline_GINO_fullstate_UVLM_BEM/model2t_pk_bem_vgvf_nm4.png` | **Best full-state V–g / V–f** (bending + torsion) |
| `_baseline_GINO_fullstate_UVLM_BEM/physical_root_loci_modes12.png` | Same physics as tracker but MAC seeds can swap branches |
| `comparison_vgvf_*.png` | High-level PK vs SHARPy flutter curves |

---

*Generated from Cursor agent session on Model2t baseline / coalescence analysis. No git commits were made during this session.*
