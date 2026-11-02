# UVLM basis / Q diagnostic (Goland)

Generated for the SHARPy UVLM vs GINO-UVLM discrete p-k bending-branch investigation.

## Outputs

| File | Content |
|------|---------|
| `dry_mode_shapes_gino_vs_sharpy.png` | Plunge + twist profiles (tip-/scale-matched) |
| `dry_mode_mac_parity_v2.csv` | MAC table — current `gino_uvlm` run vs SHARPy |
| `dry_mode_mac_quvlm_source_v2.csv` | MAC table — Quvlm build CSV vs SHARPy |
| `quvlm_vs_live_sharpy_q.png` / `.csv` | Element-wise Q: stored Quvlm vs live SHARPy⊕C |
| `quvlm_q_gino_vs_recovered_sh.png` | Quvlm GINO-basis vs C-recovered SHARPy basis |
| `analysis_ready.json` | Machine-readable summary |
| `sharpy_dry_modes.npz` | SHARPy half-wing W/TH from API |

## Dry mode shapes (parity run)

| Mode | Δf % | MAC_w (plunge) | MAC_th (twist) | MAC full | c_w (w_lsq) |
|------|------|----------------|----------------|----------|-------------|
| 1 bend | **+0.93%** | **1.000** | **0.341** | 0.933 | +1.41 |
| 2 tors | +0.09% | **1.000** | **1.000** | 0.998 | −1.42 |
| 3 | +0.48% | 1.000 | 0.981 | 0.962 | −1.39 |
| 4 | +0.82% | 1.000 | 0.894 | 0.922 | −1.42 |

**Finding:** Plunge shapes match perfectly. The **bending mode twist content does not** (MAC_th ≈ 0.34). Torsion is essentially identical. `w_lsq` projection only scales on plunge, so this twist mismatch is **not** corrected when building Quvlm.

Also: Quvlm was built from `output_data/GOLAND/GOLAND_dry_egv_*.csv` (bend **7.85 Hz**, +2.6% vs SHARPy), while the current parity flutter run uses bend **7.72 Hz** (+0.9%). Mild basis drift between Quvlm build and current structure.

## Element-wise Quvlm vs live SHARPy GAF

Same C / surface_factor=0.5 transform as `precompute_quvlm_goland.py`:

| Entry | mean rel. err | max rel. err | @ k≈0.42 |
|-------|---------------|--------------|----------|
| Q00 bend–bend | **1.8%** | 5.4% | Quvlm ≈ live⊕C |
| Q01 bend–tors | 0.9% | 2.2% | match |
| Q11 tors–tors | 0.4% | 1.0% | match |

**Finding:** Stored Quvlm is consistent with live SHARPy freqresp projected by C. The pipeline is not “broken”; residual Q00 error is small vs the V–g bend gap.

## Implication for bending V–g gap

1. Not a gross Uz shape error (MAC_w = 1).
2. Not a Quvlm bookkeeping bug (live Q matches).
3. **Bend-mode twist mismatch** (MAC_th ≈ 0.34) + **w_lsq ignoring twist** is the structural-basis smoking gun for bend-specific aero (Q₀₁-dominated).
4. Secondary: p-k vs Ass, Quvlm build vs current dry CSV drift.

## Next steps (when back)

1. Quantify tip twist on SHARPy vs GINO bend mode (why MAC_th is low).
2. Try `full_lsq` or twist-aware congruence; or regenerate Quvlm from **current** parity dry modes.
3. Optional: GINO p-k with `quvlm_sharpy_structural_basis` once identity path is solid.
