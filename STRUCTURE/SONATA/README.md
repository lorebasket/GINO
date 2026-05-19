# SONATA — Cross-Sectional Beam Property Computation

This folder contains the blade/foil geometry definitions and scripts used to compute cross-sectional structural properties (stiffness and mass matrices) via the **SONATA** and **ANBA4** frameworks.

The outputs of these scripts (CSV files in each example's `csv_export/` subfolder) feed directly into the hydroelastic flutter analysis in `main.py`. For cases that also use Capytaine BEM radiation, these structural properties are used first to compute the dry beam modes; those dry modes are required before running `FLUID/capytaine/run_modal_radiation.py`.

---

## Dependencies

These scripts require two external tools that must be cloned/installed separately and are **not included** in this repository:

| Dependency | Purpose | Clone into |
|---|---|---|
| [SONATA](https://github.com/WISDEM/SONATA) | Blade cross-section meshing and analysis | `SONATA/9_gitclone/` |
| [ANBA4](https://github.com/ANBA4/anba4) | Cross-sectional stiffness solver (FEniCS-based) | `FSI/anba4/` |

ANBA4 also requires **FEniCS** (`dolfin`). Installation is recommended via a dedicated conda environment.

---

## Folder Structure

```
SONATA/
├── NACA0012/              ← NACA0012 symmetric hydrofoil
│   ├── NACA0012.py        ← Run this to compute beam properties
│   ├── NACA0012.yaml      ← Geometry and material definition
│   └── csv_export/        ← Output: stiffness & mass matrices (CSV)
│
├── AGARD445.6/            ← AGARD 445.6 benchmark wing (air, flutter)
│   ├── AGARD445.6.py
│   ├── AGARD445.6.yaml
│   └── csv_export/
│
├── NACA0015/              ← NACA0015 hydrofoil (thicker variant)
│   ├── NACA0015.py
│   ├── NACA0015.yaml
│   └── csv_export/
│
├── hollowell-1982/        ← Hollowell 1982 benchmark hydrofoil
│   ├── ...
│   └── csv_export/
│
├── ETNZ/                  ← ETNZ foil geometry (multi-body cases)
│   ├── tnz_arm/
│   ├── tnz_foil_sx/
│   ├── tnz_foil_dx/
│   └── ...
│
└── 9_gitclone/            ← (not uploaded) Cloned SONATA source
```

---

## Workflow

Each example follows the same steps:

### 1. Set paths

At the top of each `.py` file, update the paths to point to your local SONATA and ANBA4 installations:

```python
anba4_path  = '/your/path/to/anba4'
sonata_path = '/your/path/to/SONATA/9_gitclone'
```

### 2. Run the script

From inside the example folder, run:

```bash
cd SONATA/NACA0012
python NACA0012.py
```

This will:
1. Load the geometry from the `.yaml` file
2. Mesh the cross-sections using SONATA
3. Solve the cross-sectional stiffness and mass problem with ANBA4
4. Export the results to `csv_export/`

### 3. Check the outputs

The `csv_export/` folder will contain:

| File | Content |
|---|---|
| `*_anbax_beam_properties_general.csv` | EI, GJ, EA, mass per unit length, etc. |
| `*_anbax_beam_properties_stiff_matrices.csv` | Full 6×6 stiffness matrix per section |
| `*_anbax_beam_properties_mass_matrices.csv` | Full 6×6 mass matrix per section |

These CSV files are read by `main.py` when `aero_source = 'sonata'` is set in the case configuration.

---

## Key flags in each script

| Flag | Description |
|---|---|
| `flag_run_anbax` | Run ANBA4 stiffness solver (must be `True` to compute properties) |
| `flag_csv_export` | Export results to CSV (must be `True` to feed into `main.py`) |
| `flag_wf` | Plot wire-frame of the cross-section |
| `flag_lft` | Plot lofted 3D surface of the blade |
| `flag_topo` | Plot mesh topology |
| `radial_stations` | List of normalized span positions `[0.0, ..., 1.0]` to analyze |

---

## Notes

- Units in the `.yaml` files are in **mm** for better meshing convergence. The scripts handle conversion to meters internally.
- The `9_gitclone/` folder (not uploaded) also contains additional SONATA example blades (IEA 15MW, IEA 10MW, etc.) for reference.
- For cases that do **not** require SONATA outputs (e.g. GOLAND, ABRAMSON1965), the structural properties are defined analytically in the case config file and these scripts do not need to be run.
- The Capytaine BEM implementation and its validation are under development; current Abramson benchmark notes are in [`../../docs/resources/benchmarks/abramson1965_report/abramson1965_current_report.pdf`](../../docs/resources/benchmarks/abramson1965_report/abramson1965_current_report.pdf).
