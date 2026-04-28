# PanelAero — Aerodynamic Influence Coefficients Precomputation

This module builds the aerodynamic panel grid and precomputes the **Qjj matrices** (generalised aerodynamic force matrices) using the **Doublet Lattice Method (DLM)** or the **Vortex Lattice Method (VLM)**. These matrices are stored on disk and loaded at runtime by `main.py` during the flutter analysis.

---

## Folder Structure

```
PanelAero/
├── Qjj/
│   ├── executer.py                  ← Single-configuration precomputation
│   ├── executer_multi_aerogrid.py   ← Multi-configuration batch precomputation
│   ├── precompute_qjj.py            ← DLM core (called by executer)
│   ├── precompute_qjj_vlm.py        ← VLM core (called by executer)
│   └── qjj_precomputed/             ← (not uploaded) Output folder
│
└── panelaero_utl/
    ├── DLM.py                       ← Doublet Lattice Method implementation
    ├── build_aeromodel.py           ← CAERO1-based flat panel grid builder
    ├── build_aeromodel_crvs.py      ← Curve-based aerogrid builder (for complex shapes)
    ├── CAERO1_cards/                ← Generated CAERO1 panel definition files
    ├── pk_method_utl/               ← P-K flutter solver utilities
    └── ...
```

---

## Qjj Precomputation

The Qjj matrices encode the aerodynamic response of the panel grid as a function of **reduced frequency** `k` and **Mach number** `Ma`. They must be precomputed once for each geometry/fluid/grid configuration before running the flutter analysis.

### Output naming convention

Each precomputed case is saved in a subfolder of `Qjj/qjj_precomputed/` named:

```
{blade_name}_{fluid}_alpha{alpha}_nspan{nspan}_nchord{nchord}_klist_{dlm_method}/
```

For example:
```
GOLAND_air_alpha0.0_nspan20_nchord10_klist_quartic/
```

Each subfolder contains:
- `k_{ik:03d}__Ma_{iMa:04d}_real.npy` / `_imag.npy` — Qjj matrices at each (k, Ma) point
- `aerogrid.npz` — the aerodynamic grid used, saved for reuse in `main.py`

---

## Using `executer.py` — single configuration

`executer.py` computes Qjj for **one geometry and one set of parameters** at a time.

### 1. Configure the run

Open `Qjj/executer.py` and set the parameters at the top of `main()`:

```python
blade_name  = 'GOLAND'      # geometry name (see list below)
fluid       = 'air'          # 'air' or 'water'
nspan       = [20]           # list of spanwise panel counts to compute
AR          = [0.5]          # list of aspect ratios (nchord = nspan * AR)
DLM         = True           # compute DLM Qjj matrices
VLM         = False          # compute VLM Vjj matrices
DLM_method  = 'quartic'      # 'quartic' (Rodden 1998) or 'parabolic' (Rodden 1971/72)
attack_angle = [0.0]         # angle of attack in degrees
```

Available `blade_name` values (method 1 — flat CAERO1 panel):

| Name | Description |
|---|---|
| `GOLAND` | Goland benchmark wing (air) |
| `hollowell` | Hollowell 1982 hydrofoil |
| `ABRAMSON1965` | Abramson 1965 hydrofoil |
| `grid_conv` | Grid convergence study geometry |
| `NACA0003` | NACA0003 thin hydrofoil |
| `1x1grid` | Unit square panel (debugging) |

For complex 3D geometries (e.g. ETNZ foils), set `method = 2` and provide LE/TE curve point files via `blade_name`.

### 2. Set the reduced frequency and velocity ranges

```python
k_list  = np.round(np.concatenate([
    np.linspace(0.001, 1, 10),
    np.linspace(1, 4, 10),
    np.linspace(4, higher_k, 100)
]), 3)

V_list  = np.linspace(5, 55, 50)        # velocity range [m/s]
Ma_list = V_list / c_sound[fluid]        # converted to Mach numbers
```

### 3. Run

```bash
cd PanelAero/Qjj
python executer.py
```

The script will:
1. Build the aerodynamic grid (flat panel or curve-based)
2. Rotate it by the angle of attack
3. Compute Qjj (DLM) and/or Vjj (VLM) for all (k, Ma) combinations
4. Save results in `qjj_precomputed/`
5. Save the aerogrid as `aerogrid.npz` in the same folder
6. Print a diagnostics summary and save a CSV report (`executer_diagnostics.csv`)

---

## Using `executer_multi_aerogrid.py` — batch computation

`executer_multi_aerogrid.py` loops over **multiple span/chord combinations** automatically, useful for grid convergence studies or parameter sweeps.

Configure and run it in the same way as `executer.py`:

```bash
cd PanelAero/Qjj
python executer_multi_aerogrid.py
```

It follows the same output naming convention and produces one subfolder per (nspan, nchord) combination.

---

## Resuming an interrupted computation

Both executors support **resume mode** — if a run is interrupted, re-running will skip already-computed (k, Ma) files and continue from where it stopped:

```python
precompute_qjj_grid(..., resume=True, verify_existing=True, ...)
```

---

## Notes

- The `qjj_precomputed/` output folder is **not uploaded** to the repository (excluded via `.gitignore`) due to its large size. It must be recomputed locally.
- The DLM implementation follows Rodden's formulation. The `quartic` method (Rodden 1998) is recommended for production runs; `parabolic` (Rodden 1971/72) is faster but less accurate at high reduced frequencies.
- CAERO1 card files generated during aerogrid construction are saved in `panelaero_utl/CAERO1_cards/`.
