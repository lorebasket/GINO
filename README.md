# GINO - An Opensource Framework for the hydroelastic analysis of composite hydrofoils.

[![License: Dual%20(GPLv3%20or%20Commercial)](https://img.shields.io/badge/License-Dual%20GPLv3%20or%20Commercial-blue.svg)](LICENSE)

Developed as part of a Master's thesis at Politecnico di Torino, in collaboration with:
Pietro Casalone (Politecnico di Torino), Davide Tagliapietra (Shickler Tagliapietra Yacht Enngineering), Luca Valsecchi (Toolspole) and Paolo Motta (Toolspole).

The increasing integration of composite materials into marine lifting surfaces, such as hydrofoils
and turbine blades, is unlocking new opportunities for weight reduction and structural efficiency.
However, this trend amplifies the need for reliable prediction of unsteady hydrodynamic behaviour
and coupled hydro-elastic instabilities, with flutter remaining one of the most critical challenges. Most
available numerical tools are either accurate but computationally expensive or rely on oversimplified
modelling that fails to capture the complex coupling between hydrodynamics, structural anisotropy,
and dynamic stability. This work introduces a comprehensive framework that directly addresses these
limitations.

Copyright (C) 2025 Lorenzo Baraltech — Dual-licensed under [GPL-3.0-or-later](LICENSE) or a separate [commercial license](COMMERCIAL_LICENSE.md).

The framework couples a **finite element beam model** (FEA), an **aerodynamic panel method** (DLM), a **boundary element method** (BEM), and a **flutter solver** (P-K method or Roger RFA) to predict the flutter onset speed and frequency of hydrofoils, wings, and multi-body structures.

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
├── STRUCTURE/
│   ├── FEA/                         ← Structural model + FE utilities
│   └── SONATA/                      ← Beam cross-section analysis (SONATA + ANBA4)
│       └── README.md                ← ← See this for SONATA workflow
│
├── FLUID/
│   ├── aerodynamic_model.py         ← Aerodynamic model assembly
│   ├── PanelAero/                   ← Qjj precomputation utilities
│   │   └── README.md                ← ← See this for Qjj precomputation workflow
│   └── capytaine/                   ← BEM mesh and modal radiation precomputation
│       └── README.md                ← ← See this for Capytaine BEM workflow
│
├── COUPLING/                        ← Aero-structural coupling + flutter solvers
│   ├── aero_structural_coupling.py
│   ├── flutter_solver.py
│   └── hydroelastic_utl/
│
└── docs/                            ← Thesis and bibliography
```

---

## Prerequisites

A **conda environment** is strongly recommended (Python 3.9–3.11).

---

### 1. SONATA — cross-sectional beam analysis

SONATA computes the 6×6 cross-sectional stiffness and mass matrices from blade geometry YAML files.  
It is required only for cases listed under *"Cases that require SONATA outputs"* (e.g. `GOLAND_sonata`, `tnz_multibody`, `wing01`).

Clone it into `STRUCTURE/SONATA/9_gitclone/`:

```bash
cd STRUCTURE/SONATA
git clone https://github.com/WISDEM/SONATA 9_gitclone
```

> See [`STRUCTURE/SONATA/README.md`](STRUCTURE/SONATA/README.md) for the full workflow and per-script path configuration.

---

### 2. PanelAero — aerodynamic influence coefficients (DLM)

PanelAero provides the Doublet Lattice Method and Vortex Lattice Method solvers used to precompute the Qjj matrices.  
Install via pip:

```bash
pip install PanelAero
```

Repository: [github.com/DLR-AE/PanelAero](https://github.com/DLR-AE/PanelAero)

> See [`FLUID/PanelAero/README.md`](FLUID/PanelAero/README.md) for the Qjj precomputation workflow.

---

### 3. Capytaine — BEM added mass and radiation damping

Capytaine provides the Boundary Element Method (BEM) solver used to compute frequency-dependent modal added-mass and radiation-damping matrices for hydroelastic correction.

Install the BEM-specific packages with:

```bash
pip install capytaine gmsh meshio
```

The Capytaine radiation workflow requires an exported hydrodynamic mesh and, for modal runs, previously computed dry structural modes of the beam.

> See [`FLUID/capytaine/README.md`](FLUID/capytaine/README.md) for the mesh generation and modal radiation workflow.

---

### 4. ANBA4 — cross-sectional stiffness solver (FEniCS-based)

ANBA4 is the finite-element solver called by SONATA. It requires **FEniCS** (`dolfin`).  
Install FEniCS first via conda (Linux / WSL only):

```bash
conda create -n sonata-env -c conda-forge fenics python=3.10
conda activate sonata-env
```

Then clone ANBA4 into `FSI/anba4/`:

```bash
git clone https://github.com/ANBA4/anba4 anba4
```

> ANBA4 and FEniCS are only needed when running SONATA scripts. The main flutter analysis (`main.py`) does **not** require them.

---

### 5. Python packages

Install all remaining dependencies with:

```bash
pip install -r requirements.txt
```

This covers the core stack and PanelAero: `numpy`, `scipy`, `matplotlib`, `pandas`, `pyyaml`, and `PanelAero`.
Install `capytaine`, `gmsh`, and `meshio` separately when using the BEM workflow.

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
4. (optional) Capytaine BEM    Load precomputed modal added mass & radiation damping
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
| `GOLAND_sonata` | `STRUCTURE/SONATA/GOLAND/GOLAND.py` (if present) |
| `tnz_multibody` | `STRUCTURE/SONATA/ETNZ/` scripts |
| `wing01` | `STRUCTURE/SONATA/wing01/` scripts |

See [`STRUCTURE/SONATA/README.md`](STRUCTURE/SONATA/README.md) for the full workflow.

---

## Precomputing Qjj Matrices

All cases require precomputed aerodynamic influence coefficient matrices (Qjj). These are **not included** in the repository due to their size and must be computed locally before running `main.py`.

See [`FLUID/PanelAero/README.md`](FLUID/PanelAero/README.md) for the full workflow.

In short:

```bash
cd FLUID/PanelAero/Qjj
python executer.py          # single configuration
# or
python executer_multi_aerogrid.py   # multiple configurations (grid convergence)
```

The case config file (`examples/<case_name>/config_<case_name>.py`) specifies which precomputed folder to load via the `qjj_dir` and `aerogrid_path` parameters.

---

## Precomputing Capytaine BEM Matrices

Water cases can also use Capytaine BEM results for non-circulatory added mass and radiation damping. This is a separate preprocessing workflow from PanelAero.

In short:

```bash
python FLUID/capytaine/build_capytaine_mesh.py --case_name ABRAMSON1965
python main.py ABRAMSON1965  # generate dry beam modal CSV files
python FLUID/capytaine/run_modal_radiation.py --case-name ABRAMSON1965
```

The `run_modal_radiation.py` step needs the dry beam modal CSV files already produced in `output_data/<case_name>/`. If the case uses SONATA structural properties, run the SONATA preprocessing before the dry modal step. The case config then points the flutter solver to `capytaine_results_dir`, `mesh_path`, `depth`, and `depth_index`.

The validation of the Capytaine BEM implementation is under development. Current Abramson benchmark notes are collected in [`docs/resources/benchmarks/abramson1965_report/abramson1965_current_report.pdf`](docs/resources/benchmarks/abramson1965_report/abramson1965_current_report.pdf).

See [`FLUID/capytaine/README.md`](FLUID/capytaine/README.md) for the full workflow.

---

## Adding a New Case

1. Create a folder `examples/<your_case>/`
2. Add `config_<your_case>.py` — define geometry, stiffness, fluid, and Qjj path  
   (copy an existing case as template, e.g. `examples/GOLAND/config_GOLAND.py`)
3. Precompute the Qjj matrices for the new geometry using `FLUID/PanelAero/Qjj/executer.py`
4. If structural properties come from SONATA, run the corresponding SONATA script first
5. For Capytaine BEM correction, build the mesh and run `FLUID/capytaine/run_modal_radiation.py` after dry beam modes exist
6. Run or rerun: `python main.py <your_case>`

---

## Output

Results are saved in `output_plots/`:

- `flutter_analysis_<case>_<timestamp>.log` — full console log
- V-g and V-f flutter diagrams (PNG)
- Eigenvalue trajectory plots
- Mode shape visualisations (if enabled in config)
- Displacement/force CSVs (if enabled in config)

---

## Third-Party Software & Licenses

This framework is built on top of external open-source libraries that are dynamically linked (called at runtime) but are **not** distributed as part of this repository. Their source code, copyright, and license terms remain entirely with their respective authors.

---

### PanelAero — BSD 3-Clause License

**Repository:** [github.com/DLR-AE/PanelAero](https://github.com/DLR-AE/PanelAero)  
**Author:** Arne Voß, Deutsches Zentrum für Luft- und Raumfahrt e.V. (DLR)  
**License:** [BSD 3-Clause "New" or "Revised" License](https://github.com/DLR-AE/PanelAero/blob/master/LICENSE)

PanelAero is used in this project to compute aerodynamic influence coefficient matrices (Qjj) via the Doublet Lattice Method (DLM) and Vortex Lattice Method (VLM). It is installed as a standard Python package (`pip install PanelAero`) and called through the scripts in `FLUID/PanelAero/Qjj/`.

**Compliance statement:** The BSD 3-Clause License permits free use, modification, and integration into other projects, provided that the copyright notice and license text are retained. This framework does not redistribute PanelAero source code or binaries. The copyright notice is reproduced here in accordance with clause 1 of the license:

> Copyright (c) 2020–2022, Deutsches Zentrum für Luft- und Raumfahrt e.V.  
> All rights reserved.

The name of DLR is not used to endorse or promote this project, in compliance with clause 3 of the license.

---

### SONATA — GNU Lesser General Public License v3 (LGPL-3.0)

**Repository:** [github.com/WISDEM/SONATA](https://github.com/WISDEM/SONATA) / [github.com/NLRWindSystems/SONATA](https://github.com/NLRWindSystems/SONATA)  
**License:** [GNU Lesser General Public License v3.0](https://github.com/NLRWindSystems/SONATA/blob/master/license.txt)

SONATA is used in this project to compute the 6×6 cross-sectional stiffness and mass matrices of composite beam sections from YAML geometry files. It is cloned separately by the user into `STRUCTURE/SONATA/9_gitclone/` and is only required for specific analysis cases (e.g. `GOLAND_sonata`, `tnz_multibody`, `wing01`). The main flutter pipeline (`main.py`) does **not** depend on SONATA at runtime.

**Compliance statement:** The GNU LGPLv3 permits use of the library in a larger application without imposing the LGPL on the application itself, provided that:
1. The user can replace the LGPL-covered library with a modified version (satisfied: SONATA is cloned independently by the user, not bundled here).
2. The use of SONATA is prominently disclosed and the license is referenced (satisfied: see above).

This framework does **not** distribute a modified copy of SONATA. No SONATA source code is included in this repository. The LGPL license terms therefore do **not** extend to the original code of this framework.

---

### Capytaine — GNU General Public License v3 (GPL-3.0)

**Repository:** [github.com/capytaine/capytaine](https://github.com/capytaine/capytaine)  
**License:** [GNU General Public License v3.0](https://github.com/capytaine/capytaine/blob/master/LICENSE)

Capytaine is used in this project as a Python BEM solver for linear potential-flow radiation problems. It is installed separately by the user and called through the scripts in `FLUID/capytaine/`.

**Compliance statement:** This framework does **not** redistribute Capytaine source code or binaries. Capytaine remains an optional external dependency for BEM preprocessing, and its license terms remain with the upstream project.

---

### Summary

| Software | License | Usage mode | Code included in this repo |
|---|---|---|---|
| **This framework (GINO)** | **Dual: GPL-3.0-or-later OR Commercial** | — | Yes |
| PanelAero | BSD 3-Clause | `pip install` + called at runtime | No |
| SONATA | GNU LGPL v3 | Cloned separately + called at runtime | No |
| Capytaine | GNU GPL v3 | `pip install` + called at runtime | No |

These libraries are used in compliance with their respective licenses. Licensing of this framework itself is defined by the project dual-license terms in [`LICENSE`](LICENSE) and [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md).

---

## Reference

For the theoretical background, see the thesis document in `docs/`.
