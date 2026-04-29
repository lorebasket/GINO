# FLUTTER-FSI

An Opensource Framework for the hydroelastic analysis of composite beams.  
Developed as part of a Master's thesis at Politecnico di Torino, in collaboration with:
Pietro Casalone (Politecnico di Torino), Davide Tagliapietra (Shickler Tagliapietra Yacht Enngineering), Luca Valsecchi (Toolspole) and Paolo Motta (Toolspole).

The framework couples a **finite element beam model** (FEA), an **unsteady aerodynamic panel method** (DLM/VLM), a **boundary element method** (BEM), and a **flutter solver** (P-K method or Roger RFA) to predict the flutter onset speed and frequency of hydrofoils, wings, and multi-body structures.

---

## Repository Structure

```
FSI/
├── main.py                          ← Entry point: run a flutter analysis
├── examples/                        ← Per-case configuration files
│   ├── config.py                    ← Global defaults and config loader
│   ├── GOLAND/                      ← Goland benchmark (analytic, air)
│   ├── AGARD445.6/                  ← AGARD 445.6 benchmark (analytic, air)
│   ├── hollowell/                   ← Hollowell 1982 hydrofoil
│   ├── ABRAMSON1965/                ← Abramson 1965 hydrofoil
│   ├── GOLAND_sonata/               ← Goland wing (SONATA-derived properties)
│   ├── tnz_multibody/               ← ETNZ multi-body foil (SONATA required)
│   └── ...
│
├── SONATA/                          ← Beam cross-section analysis (SONATA + ANBA4)
│   └── README.md                    ← ← See this for SONATA workflow
│
├── PanelAero/                       ← Aerodynamic panel model and Qjj precomputation
│   └── README.md                    ← ← See this for Qjj precomputation workflow
│
├── FEA/
│   └── fea_utl/                     ← FE beam model utilities
│
├── Hydroelastic_analysis_workflow/  ← Analysis pipeline modules
│   ├── structural_model.py
│   ├── structural_analysis.py
│   ├── aerodynamic_model.py
│   ├── aero_structural_coupling.py
│   ├── flutter_solver.py
│   └── post_processing.py
│
└── docs/                            ← Thesis and bibliography
```

---

## Prerequisites

Install the required Python packages (a conda environment is recommended):

```bash
pip install numpy scipy matplotlib
```

For cases using SONATA-derived structural properties, additional dependencies are required — see [`SONATA/README.md`](SONATA/README.md).

---

## Running a Flutter Analysis

### Quick start

```bash
python main.py <case_name>
```

For example:

```bash
python main.py GOLAND
python main.py AGARD445.6
python main.py hollowell
```

### Command-line options

| Option | Description |
|---|---|
| `case_name` | Name of the analysis case (default: `GOLAND`) |

Output (plots and logs) is saved to `output_plots/`.

---

## Analysis Pipeline

`main.py` executes the following steps for each case:

```
1. Load configuration          examples/<case_name>/config_<case_name>.py
        ↓
2. Build structural model      FEA beam model from config or SONATA CSV outputs
        ↓
3. Structural analysis         Dry natural frequencies and mode shapes
        ↓
4. (optional) Wet modes        Fluid-at-rest frequency shift (non-circulatory added mass & damping)
        ↓
5. Build aerodynamic model     Load precomputed aerogrid and Qjj matrices
        ↓
6. Aero-structural coupling    Z matrix coupling (beam ↔ aerogrid)
        ↓
7. Flutter solver              P-K method or Roger RFA eigenvalue sweep
        ↓
8. Post-processing             V-g / V-f plots, eigenvalue trajectories
```

---

## Case Types

### Cases that do NOT require SONATA

Structural properties are defined analytically in the case config file. No preprocessing needed — just make sure the Qjj matrices are precomputed (see below).

| Case | Description | Fluid |
|---|---|---|
| `GOLAND` | Goland benchmark wing | air |
| `AGARD445.6` | AGARD 445.6 swept wing | air |
| `hollowell` | Hollowell 1982 cantilevered hydrofoil | water |
| `ABRAMSON1965` | Abramson 1965 hydrofoil | water |
| `NACA0003` | Thin NACA0003 hydrofoil | water |
| `grid_conv` | Grid convergence study | water |

### Cases that require SONATA outputs

Structural properties are read from CSV files exported by the SONATA scripts. These must be computed first before running `main.py`.

| Case | SONATA example to run first |
|---|---|
| `GOLAND_sonata` | `SONATA/GOLAND/GOLAND.py` (if present) |
| `tnz_multibody` | `SONATA/ETNZ/` scripts |
| `wing01` | `SONATA/wing01/` scripts |

See [`SONATA/README.md`](SONATA/README.md) for the full workflow.

---

## Precomputing Qjj Matrices

All cases require precomputed aerodynamic influence coefficient matrices (Qjj). These are **not included** in the repository due to their size and must be computed locally before running `main.py`.

See [`PanelAero/README.md`](PanelAero/README.md) for the full workflow.

In short:

```bash
cd PanelAero/Qjj
python executer.py          # single configuration
# or
python executer_multi_aerogrid.py   # multiple configurations (grid convergence)
```

The case config file (`examples/<case_name>/config_<case_name>.py`) specifies which precomputed folder to load via the `qjj_dir` and `aerogrid_path` parameters.

---

## Adding a New Case

1. Create a folder `examples/<your_case>/`
2. Add `config_<your_case>.py` — define geometry, stiffness, fluid, and Qjj path  
   (copy an existing case as template, e.g. `examples/GOLAND/config_GOLAND.py`)
3. Precompute the Qjj matrices for the new geometry using `PanelAero/Qjj/executer.py`
4. If structural properties come from SONATA, run the corresponding SONATA script first
5. Run: `python main.py <your_case>`

---

## Output

Results are saved in `output_plots/`:

- `flutter_analysis_<case>_<timestamp>.log` — full console log
- V-g and V-f flutter diagrams (PNG)
- Eigenvalue trajectory plots
- Mode shape visualisations (if enabled in config)
- Displacement/force CSVs (if enabled in config)

---

## Reference

For the theoretical background, see the thesis document in `docs/`.
